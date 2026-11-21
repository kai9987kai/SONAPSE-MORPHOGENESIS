"""Oscillatory binding units (frontier module ``kuramoto_binding``).

Inspired by *Artificial Kuramoto Oscillatory Neurons* (AKOrN) -- Takeru Miyato, Sindy Löwe, Andreas
Geiger and Max Welling, ICLR 2025 (oral), April 2025; arXiv:2410.13821, https://arxiv.org/abs/2410.13821.
This module is a synthetic computational abstraction. It implements the paper's discrete oscillator
update and uses it in three small settings; it does not reproduce the paper's networks (no learned
convolutional or attention couplings, no readout blocks, no training) or any of its results.
Full, verified references: ``docs/frontier/modules/kuramoto_binding.md``.

Update (paper Sec. 3 Eq. 2 and Sec. 4 Eqs. 4-5, here with scalar couplings ``J_ij * I_N``)::

    dx_i = Omega_i x_i + Proj_{x_i}(c_i + sum_j J_ij x_j),     Proj_x(y) = y - <y, x> x
    x_i <- Pi(x_i + gamma * dx_i),                              Pi(v) = v / ||v||

Each ``x_i`` is a unit vector in ``R^N`` (``N`` even), ``Omega_i`` is an antisymmetric natural-frequency
matrix (block rotations), ``c_i`` the conditional stimulus, and ``K`` iterations run per call. The global
order parameter is ``R = ||mean_i x_i||`` (Kuramoto's ``|<exp(i theta)>|`` when ``N = 2``).

Settings
--------
* :class:`KuramotoExtension` -- NEUROWEAVE engine extension: one oscillator per region (or per sampled
  unit), couplings aggregated from the engine's synapse table at ``bind``, stimuli from the group's mean
  output. It only adds readout features; it never changes the drive.
* :class:`KuramotoObserver` -- read-only live observer: oscillators driven by subsampled rates with fixed
  sparse coupling plus a slow, bounded Hebbian co-activity estimate; reports order-parameter statistics,
  phase-cluster counts and the metastability and chimera indices of Shanahan (Chaos 20, 013108, 2010).
* :func:`run_binding_benchmark` -- a synthetic binding task scored with the adjusted Rand index (Hubert &
  Arabie, 1985), against a shuffled-coupling and a zero-coupling control (plus a non-oscillatory reference).

All randomness comes from ``numpy.random.default_rng(seed + offset)``.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..neuroweave.spec import REGIONS
from . import harness
from .extension import EngineExtension
from .spec import EVIDENCE_TIER, FrontierSpecV1

__all__ = ['MODULE_ID', 'BENCHMARK_ID', 'SOURCE', 'EXPERIMENTS', 'OBSERVER_STATE_SCHEMA', 'KuramotoExtension',
           'KuramotoObserver',
           'project_tangent', 'natural_frequency_matrices', 'akorn_step', 'akorn_run', 'kuramoto_energy',
           'order_parameter', 'group_coherence', 'phases', 'phase_locking_value', 'count_phase_clusters',
           'shanahan_indices', 'adjusted_rand_index', 'kmeans_labels', 'make_binding_task', 'correlation_coupling',
           'shuffle_coupling', 'run_experiment', 'run_binding_benchmark']

MODULE_ID = 'kuramoto_binding'
BENCHMARK_ID = MODULE_ID + '_benchmark'
SOURCE = ('Miyato, Löwe, Geiger & Welling, "Artificial Kuramoto Oscillatory Neurons", ICLR 2025 (oral), '
          'April 2025, arXiv:2410.13821, https://arxiv.org/abs/2410.13821')
OBSERVER_STATE_SCHEMA = 'KuramotoObserverStateV1'

_EXT_OFFSET = 61_001
_UNIT_OFFSET = 61_007
_OBS_OFFSET = 61_013
_TASK_OFFSET = 61_027
_INIT_OFFSET = 61_031
_KMEANS_OFFSET = 61_043
_SHUFFLE_OFFSET = 61_051
_EPS = 1e-12
_MAX_COUPLING = 1e6
_STATS_RATE = 0.05          # observer: EWMA rate of the running mean / variance used to standardise rates
_Z_CLIP = 3.0               # observer: standardised co-activity is clipped to +/-3 sd, then scaled to [-1, 1]
# FrontierSpecV1 observer parameters meant for other observers; accepted and ignored here.
_SPEC_OBSERVER_PARAMS = frozenset({'sync_pairs', 'sae_features', 'sae_k', 'fc_window', 'max_observed_units',
                                   'observe_every'})


# ====================================================================== validation helpers
def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _num(name: str, value: Any, low: float, high: float, *, open_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if not math.isfinite(v) or v < low or v > high or (open_low and v <= low):
        raise ValueError(f'{name} must be a finite number in {"(" if open_low else "["}{low}, {high}]')
    return v


def _even_dim(value: Any) -> int:
    d = _int('dim', value, 2, 64)
    if d % 2:
        raise ValueError('dim must be even (oscillators rotate in 2-d planes)')
    return d


def _normalize(v: np.ndarray) -> np.ndarray:
    """Rows scaled to unit norm (a zero row, which the update cannot produce, maps to e_0)."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.where(norm > _EPS, norm, 1.0)
    zero = norm[..., 0] <= _EPS
    if np.any(zero):
        out[zero] = 0.0
        out[zero, 0] = 1.0
    return out


def _unit_rows(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    return _normalize(rng.standard_normal((count, dim)))


def _as_states(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2 or a.shape[1] % 2 or not np.all(np.isfinite(a)):
        raise ValueError('oscillator states must be a finite (C, N) array with N even and >= 2')
    return a


def _check_coupling(coupling: Any, c: int, n: int) -> np.ndarray | None:
    if coupling is None:
        return None
    j = np.asarray(coupling, dtype=np.float64)
    if j.shape not in ((c, c), (c * n, c * n)) or not np.all(np.isfinite(j)) or np.any(np.abs(j) > _MAX_COUPLING):
        raise ValueError(f'coupling must be a finite ({c}, {c}) or ({c * n}, {c * n}) array with |J| <= 1e6')
    return j


def _check_stimulus(stimulus: Any, c: int, n: int) -> np.ndarray | None:
    if stimulus is None:
        return None
    s = np.asarray(stimulus, dtype=np.float64)
    if s.shape != (c, n) or not np.all(np.isfinite(s)) or np.any(np.abs(s) > _MAX_COUPLING):
        raise ValueError(f'stimulus must be a finite ({c}, {n}) array with |c| <= 1e6')
    return s


def _check_omega(omega: Any, c: int, n: int) -> np.ndarray | None:
    if omega is None:
        return None
    om = np.asarray(omega, dtype=np.float64)
    if om.shape == (n, n):
        om = np.broadcast_to(om, (c, n, n))
    if om.shape != (c, n, n) or not np.all(np.isfinite(om)) or np.any(np.abs(om) > _MAX_COUPLING):
        raise ValueError(f'omega must be a finite ({c}, {n}, {n}) or ({n}, {n}) array')
    if np.max(np.abs(om + np.swapaxes(om, 1, 2))) > 1e-9 * max(1.0, float(np.max(np.abs(om)))):
        raise ValueError('omega must be antisymmetric (Omega^T = -Omega)')
    return om


# ====================================================================== AKOrN core
def project_tangent(x: Any, y: Any) -> np.ndarray:
    """``Proj_x(y) = y - <y, x> x`` row-wise (the tangent-space projection for unit ``x``)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.ndim < 1 or not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        raise ValueError('x and y must be finite arrays of the same shape')
    return ya - np.sum(ya * xa, axis=-1, keepdims=True) * xa


def natural_frequency_matrices(frequencies: Any, dim: int) -> np.ndarray:
    """Antisymmetric ``Omega_i`` built from per-plane angular frequencies.

    ``frequencies`` is ``(C,)`` (same frequency in every plane) or ``(C, dim // 2)``. Plane ``p`` uses the
    block ``[[0, -w], [w, 0]]`` on coordinates ``(2p, 2p + 1)``, so ``dtheta/dt = w`` (counter-clockwise)."""
    d = _even_dim(dim)
    half = d // 2
    f = np.asarray(frequencies, dtype=np.float64)
    if f.ndim == 1:
        f = np.repeat(f[:, None], half, axis=1)
    if f.ndim != 2 or f.shape[0] < 1 or f.shape[1] != half or not np.all(np.isfinite(f)) or np.any(np.abs(f) > 1e6):
        raise ValueError(f'frequencies must be finite with shape (C,) or (C, {half})')
    om = np.zeros((f.shape[0], d, d))
    for p in range(half):
        om[:, 2 * p, 2 * p + 1] = -f[:, p]
        om[:, 2 * p + 1, 2 * p] = f[:, p]
    return om


def _step(x: np.ndarray, coupling: np.ndarray | None, stimulus: np.ndarray | None, omega: np.ndarray | None,
          gamma: float) -> np.ndarray:
    """One unchecked AKOrN update of unit-norm ``x`` (shapes already validated)."""
    c, n = x.shape
    y = np.zeros_like(x) if stimulus is None else stimulus.copy()
    if coupling is not None:
        y += coupling @ x if coupling.shape == (c, c) else (coupling @ x.reshape(-1)).reshape(c, n)
    dx = y - np.sum(y * x, axis=1, keepdims=True) * x
    if omega is not None:
        dx += np.einsum('cij,cj->ci', omega, x)
    return _normalize(x + gamma * dx)


def akorn_step(x: Any, coupling: Any = None, stimulus: Any = None, omega: Any = None, *,
               gamma: float = 0.1) -> np.ndarray:
    """One discrete AKOrN update ``x <- Pi(x + gamma (Omega x + Proj_x(c + J x)))``.

    ``x`` is ``(C, N)`` and is first put on the sphere; ``coupling`` is ``(C, C)`` scalar couplings or a
    ``(C*N, C*N)`` block matrix; ``stimulus`` is ``(C, N)``; ``omega`` is ``(C, N, N)`` or ``(N, N)``
    antisymmetric. ``None`` means zero."""
    xa = _as_states(x)
    c, n = xa.shape
    return _step(_normalize(xa), _check_coupling(coupling, c, n), _check_stimulus(stimulus, c, n),
                 _check_omega(omega, c, n), _num('gamma', gamma, 0.0, 10.0, open_low=True))


def akorn_run(x: Any, coupling: Any = None, stimulus: Any = None, omega: Any = None, *, gamma: float = 0.1,
              iterations: int = 16) -> np.ndarray:
    """``iterations`` AKOrN updates with a constant stimulus (validated once)."""
    xa = _as_states(x)
    c, n = xa.shape
    j, s, om = _check_coupling(coupling, c, n), _check_stimulus(stimulus, c, n), _check_omega(omega, c, n)
    g = _num('gamma', gamma, 0.0, 10.0, open_low=True)
    k = _int('iterations', iterations, 0, 100_000)
    out = _normalize(xa)
    for _ in range(k):
        out = _step(out, j, s, om, g)
    return out


def kuramoto_energy(x: Any, coupling: Any = None, stimulus: Any = None) -> float:
    """``E = -1/2 sum_ij x_i^T J_ij x_j - sum_i c_i^T x_i`` (paper Eq. 3). It is a Lyapunov function of the
    continuous flow only for symmetric scalar couplings, shared ``Omega`` and ``Omega c_i = 0``."""
    xa = _as_states(x)
    c, n = xa.shape
    j, s = _check_coupling(coupling, c, n), _check_stimulus(stimulus, c, n)
    e = 0.0
    if j is not None:
        e -= 0.5 * float(np.sum(j * (xa @ xa.T))) if j.shape == (c, c) else 0.5 * float(xa.reshape(-1) @ j @ xa.reshape(-1))
    if s is not None:
        e -= float(np.sum(s * xa))
    return e


def order_parameter(x: Any) -> float:
    """``R = ||mean_i x_i||`` in ``[0, 1]`` for unit-norm rows (clipped at 1 against round-off)."""
    return min(1.0, float(np.linalg.norm(_as_states(x).mean(axis=0))))


def group_coherence(x: Any, labels: Any, n_groups: int | None = None) -> np.ndarray:
    """Per-group order parameter ``||mean_{i in g} x_i||`` (0 for an empty group; clipped at 1)."""
    xa = _as_states(x)
    lab = np.asarray(labels)
    if lab.shape != (xa.shape[0],) or not np.issubdtype(lab.dtype, np.integer) or np.any(lab < 0):
        raise ValueError('labels must be non-negative ints, one per oscillator')
    g = int(lab.max()) + 1 if n_groups is None else _int('n_groups', n_groups, int(lab.max()) + 1, 1_000_000)
    sums = np.zeros((g, xa.shape[1]))
    np.add.at(sums, lab, xa)
    cnt = np.bincount(lab, minlength=g)
    return np.minimum(1.0, np.linalg.norm(sums, axis=1) / np.maximum(cnt, 1))


def phases(x: Any) -> np.ndarray:
    """Phase of each oscillator in its first rotation plane, ``atan2(x_1, x_0)`` (exact for ``N = 2``)."""
    xa = _as_states(x)
    return np.arctan2(xa[:, 1], xa[:, 0])


def phase_locking_value(phase_a: Any, phase_b: Any) -> float:
    """``|mean_t exp(i (phi_a(t) - phi_b(t)))|`` (Lachaux et al., 1999), in ``[0, 1]``."""
    a = np.asarray(phase_a, dtype=np.float64)
    b = np.asarray(phase_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError('phase series must be finite, 1-D, non-empty and of equal length')
    return float(np.abs(np.mean(np.exp(1j * (a - b)))))


def count_phase_clusters(phase: Any, *, bins: int = 36, min_fraction: float = 0.1) -> int:
    """Number of phase clusters from an angle histogram.

    A bin is *dense* when it holds at least ``max(1, ceil(min_fraction * M))`` of the ``M`` oscillators;
    a cluster is a circular run of dense bins bounded by sparse bins. A spread-out (incoherent) population
    gives 0, and so does a histogram with no sparse bin at all (nothing separates one cluster from
    another). A cluster smaller than about ``2 * min_fraction`` of the population can be missed when it
    straddles a bin edge."""
    p = np.asarray(phase, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError('phase must be a finite 1-D array')
    b = _int('bins', bins, 2, 100_000)
    frac = _num('min_fraction', min_fraction, 0.0, 1.0, open_low=True)
    if p.size == 0:
        return 0
    idx = np.minimum(b - 1, np.floor(np.mod(p, 2.0 * np.pi) / (2.0 * np.pi / b)).astype(np.int64))
    dense = np.bincount(idx, minlength=b) >= max(1, math.ceil(frac * p.size - 1e-9))
    if dense.all():
        return 0
    return int(np.sum(dense & ~np.roll(dense, 1)))


def shanahan_indices(community_order: Any) -> dict[str, float]:
    """Metastability and chimera indices of Shanahan (2010) from community synchrony ``phi`` of shape ``(T, M)``.

    As defined in the paper: ``metastability_index`` (lambda) is the mean over the ``M`` communities of the
    variance over time ``sigma_met(c) = 1/(T-1) sum_t (phi_c(t) - <phi_c>_T)^2``; ``chimera_index`` (chi)
    is the mean over the ``T`` samples of the variance across communities
    ``sigma_chi(t) = 1/(M-1) sum_c (phi_c(t) - <phi(t)>_C)^2``. A variance over a single sample (``T = 1``
    or ``M = 1``) is reported as 0."""
    phi = np.asarray(community_order, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[0] < 1 or phi.shape[1] < 1 or not np.all(np.isfinite(phi)):
        raise ValueError('community_order must be a finite (T, M) array')
    t, m = phi.shape
    met = float(np.mean(np.var(phi, axis=0, ddof=1))) if t > 1 else 0.0
    chi = float(np.mean(np.var(phi, axis=1, ddof=1))) if m > 1 else 0.0
    return {'metastability_index': met, 'chimera_index': chi}


# ====================================================================== clustering scores
def adjusted_rand_index(labels_true: Any, labels_pred: Any) -> float:
    """Adjusted Rand index (Hubert & Arabie, 1985): ``(index - expected) / (max_index - expected)``.

    1 for identical partitions (up to relabelling), about 0 for chance agreement, negative below chance.
    When the denominator vanishes (both partitions are one cluster, or both all singletons) the
    partitions are identical and the score is 1."""
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError('label arrays must be 1-D and of equal length')
    n = a.size
    if n < 2:
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(table, (ai.ravel(), bi.ravel()), 1)

    def pairs(v: np.ndarray) -> int:
        v = v.astype(np.int64)
        return int(np.sum(v * (v - 1) // 2))

    index, sum_a, sum_b = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    total = n * (n - 1) // 2
    expected = sum_a * sum_b / total
    denom = 0.5 * (sum_a + sum_b) - expected
    if denom == 0.0:
        return 1.0
    return float((index - expected) / denom)


def kmeans_labels(points: Any, k: int, *, seed: int = 0, restarts: int = 8, max_iter: int = 100) -> np.ndarray:
    """Seeded Lloyd k-means with k-means++ seeding (Arthur & Vassilvitskii, SODA 2007); the restart with the
    lowest inertia wins. An empty cluster is re-seeded at the worst-fitting point.

    On unit vectors Euclidean k-means is equivalent to clustering by cosine similarity."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError('points must be a finite (M, D) array')
    m = x.shape[0]
    k = _int('k', k, 1, m)
    rng = np.random.default_rng(_int('seed', seed, 0, 2**62))
    restarts = _int('restarts', restarts, 1, 1000)
    max_iter = _int('max_iter', max_iter, 1, 100_000)
    best: tuple[float, np.ndarray] | None = None
    rows = np.arange(m)
    for _ in range(restarts):
        centers = [x[int(rng.integers(m))]]
        d2 = np.sum((x - centers[0]) ** 2, axis=1)
        for _ in range(1, k):
            total = float(d2.sum())
            idx = int(rng.integers(m)) if total <= 0.0 else int(rng.choice(m, p=d2 / total))
            centers.append(x[idx])
            d2 = np.minimum(d2, np.sum((x - x[idx]) ** 2, axis=1))
        cen = np.array(centers)
        for _ in range(max_iter):
            dist = np.sum((x[:, None, :] - cen[None, :, :]) ** 2, axis=2)
            lab = dist.argmin(axis=1)
            new = cen.copy()
            for j in range(k):
                members = lab == j
                if members.any():
                    new[j] = x[members].mean(axis=0)
                else:                                   # empty cluster: re-seed at the worst-fit point
                    new[j] = x[int(dist[rows, lab].argmax())]
            done = np.allclose(new, cen, rtol=0.0, atol=1e-12)
            cen = new
            if done:
                break
        dist = np.sum((x[:, None, :] - cen[None, :, :]) ** 2, axis=2)
        lab = dist.argmin(axis=1)
        inertia = float(dist[rows, lab].sum())
        if best is None or inertia < best[0] - 1e-12:
            best = (inertia, lab)
    assert best is not None
    return best[1].astype(np.int64)


# ====================================================================== engine extension
class KuramotoExtension(EngineExtension):
    """AKOrN oscillators attached to a NEUROWEAVE engine (features only; the drive is never changed).

    ``mode='region'``: one oscillator per NEUROWEAVE region (10). ``mode='units'``: one oscillator per
    sampled unit (``units`` of them, sampled at ``bind`` with the extension seed).

    * Coupling ``J`` (copied at ``bind``; the engine is not kept): synapse weights summed over
      ``src -> dst`` pairs into ``J[group(dst), group(src)]``, divided by the target group's size (mean
      input per target unit), diagonal zeroed (self-coupling vanishes under the projection), then scaled
      so the largest absolute row sum equals ``|coupling_scale|`` (a negative scale flips every sign).
      Inhibitory synapses give negative (anti-phase) couplings. Later plasticity is not tracked: ``J`` is
      a snapshot.
    * Stimulus ``c_g = stimulus_scale * m_g * u_g``: the group's current mean output ``m_g`` times a fixed
      seeded unit vector ``u_g``.
    * ``iterations`` AKOrN updates per engine step with natural frequencies
      ``omega_mean + omega_spread * N(0, 1)`` per oscillator and plane.

    Features (``feature_dim = C + 1 + top_pairs``; 15 in region mode with the defaults):
    ``C`` cosines of each oscillator with the global mean-field direction (0 when ``R`` is ~0), the global
    order parameter ``R``, and the ``top_pairs`` largest pairwise phase-locking values (first rotation
    plane), sorted in descending order. PLVs are exponentially weighted over steps (rate ``plv_rate``) and
    bias-corrected. All features lie in ``[-1, 1]``. The extension has no trainable parameters, so it is
    trivially frozen when ``ctx['learn']`` is false."""

    id = MODULE_ID

    def __init__(self, seed: int = 0, *, mode: str = 'region', units: int = 16, dim: int = 2, iterations: int = 4,
                 gamma: float = 0.1, coupling_scale: float = 1.0, stimulus_scale: float = 1.0, omega_mean: float = 1.0,
                 omega_spread: float = 0.5, top_pairs: int = 4, plv_rate: float = 0.1) -> None:
        self.seed = _int('seed', seed, 0, 2**31 - 1)
        if mode not in ('region', 'units'):
            raise ValueError("mode must be 'region' or 'units'")
        self.mode = mode
        self.units = _int('units', units, 2, 4096)
        self.dim = _even_dim(dim)
        self.iterations = _int('iterations', iterations, 1, 256)
        self.gamma = _num('gamma', gamma, 0.0, 10.0, open_low=True)
        self.coupling_scale = _num('coupling_scale', coupling_scale, -100.0, 100.0)
        self.stimulus_scale = _num('stimulus_scale', stimulus_scale, -100.0, 100.0)
        self.omega_mean = _num('omega_mean', omega_mean, -100.0, 100.0)
        self.omega_spread = _num('omega_spread', omega_spread, 0.0, 100.0)
        self.plv_rate = _num('plv_rate', plv_rate, 0.0, 1.0, open_low=True)
        self.oscillators = len(REGIONS) if mode == 'region' else self.units
        c = self.oscillators
        self.top_pairs = _int('top_pairs', top_pairs, 0, c * (c - 1) // 2)
        self.feature_dim = c + 1 + self.top_pairs
        rng = np.random.default_rng(self.seed + _EXT_OFFSET)
        self._x0 = _unit_rows(rng, c, self.dim)
        self._freq = self.omega_mean + self.omega_spread * rng.standard_normal((c, self.dim // 2))
        self._omega = natural_frequency_matrices(self._freq, self.dim)
        self._stim_dir = _unit_rows(rng, c, self.dim)
        self._iu = np.triu_indices(c, 1)
        # filled by bind()
        self._n = 0
        self._group = np.zeros(0, dtype=np.int64)          # oscillator index per unit, -1 = not represented
        self._count = np.zeros(c)
        self._coupling = np.zeros((c, c))
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def config(self) -> dict[str, Any]:
        return {'seed': self.seed, 'mode': self.mode, 'units': self.units, 'dim': self.dim,
                'iterations': self.iterations, 'gamma': self.gamma, 'coupling_scale': self.coupling_scale,
                'stimulus_scale': self.stimulus_scale, 'omega_mean': self.omega_mean,
                'omega_spread': self.omega_spread, 'top_pairs': self.top_pairs, 'plv_rate': self.plv_rate,
                'oscillators': self.oscillators, 'feature_dim': self.feature_dim}

    def bind(self, engine: Any) -> None:
        try:
            n = int(engine.n)
            brain = engine.brain
            region = np.asarray(brain.cells.region, dtype=np.int64)[:n].copy()
            n_regions = len(brain.regions)
            syn = brain.synapses
            src = np.asarray(syn.src, dtype=np.int64).copy()
            dst = np.asarray(syn.dst, dtype=np.int64).copy()
            w = np.asarray(syn.w, dtype=np.float64).copy()
        except AttributeError as exc:  # pragma: no cover - only non-NEUROWEAVE engines
            raise ValueError('kuramoto_binding needs a NEUROWEAVE engine (brain.cells.region, brain.synapses)') from exc
        c = self.oscillators
        if region.size != n or src.shape != dst.shape or src.shape != w.shape or not np.all(np.isfinite(w)):
            raise ValueError('engine synapse table is inconsistent')
        if self.mode == 'region':
            if n_regions != c:
                raise ValueError(f'region mode expects {c} regions, engine has {n_regions}')
            group = region
        else:
            if n < c:
                raise ValueError(f'units ({c}) must not exceed engine.n ({n})')
            chosen = np.sort(np.random.default_rng(self.seed + _UNIT_OFFSET).choice(n, size=c, replace=False))
            group = np.full(n, -1, dtype=np.int64)
            group[chosen] = np.arange(c)
        count = np.bincount(group[group >= 0], minlength=c).astype(np.float64)
        ok = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        s, d, ww = src[ok], dst[ok], w[ok]
        keep = (group[s] >= 0) & (group[d] >= 0)
        j = np.zeros((c, c))
        np.add.at(j, (group[d[keep]], group[s[keep]]), ww[keep])
        j /= np.maximum(count, 1.0)[:, None]
        np.fill_diagonal(j, 0.0)
        row = float(np.max(np.abs(j).sum(axis=1))) if c else 0.0
        self._coupling = self.coupling_scale * j / row if row > _EPS else np.zeros((c, c))
        self._n, self._group, self._count = n, group, count
        self.reset()

    def reset(self) -> None:
        self._x = self._x0.copy()
        self._plv = np.zeros(self._iu[0].size, dtype=np.complex128)
        self._steps = 0
        self._last_mean = np.zeros(self.oscillators)

    # ------------------------------------------------------------------ hooks
    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any]) -> None:
        if self._n == 0:
            raise ValueError('kuramoto_binding is not bound to an engine')
        out = np.asarray(output, dtype=np.float64).ravel()
        if not np.all(np.isfinite(out)):
            raise ValueError('output must be finite')
        o = np.zeros(self._n)
        m = min(self._n, out.size)               # neurons born after bind are ignored (fixed shape)
        o[:m] = np.clip(out[:m], -10.0, 10.0)
        rep = self._group >= 0
        mean = np.bincount(self._group[rep], weights=o[rep], minlength=self.oscillators) / np.maximum(self._count, 1.0)
        stim = self.stimulus_scale * mean[:, None] * self._stim_dir
        x = self._x
        for _ in range(self.iterations):
            x = _step(x, self._coupling, stim, self._omega, self.gamma)
        self._x = x
        if self._plv.size:
            th = np.arctan2(x[:, 1], x[:, 0])
            a = self.plv_rate
            self._plv = (1.0 - a) * self._plv + a * np.exp(1j * (th[self._iu[0]] - th[self._iu[1]]))
        self._steps += 1
        self._last_mean = mean

    def coupling_matrix(self) -> np.ndarray:
        """The oscillator coupling ``J`` copied from the engine at ``bind`` (``J[target, source]``); a copy."""
        return self._coupling.copy()

    def states(self) -> np.ndarray:
        """Current oscillator states ``(C, N)``, unit rows; a copy."""
        return self._x.copy()

    def _plv_values(self) -> np.ndarray:
        if self._steps == 0 or not self._plv.size:
            return np.zeros(self._plv.size)
        weight = 1.0 - (1.0 - self.plv_rate) ** self._steps      # bias correction of the EWMA
        return np.clip(np.abs(self._plv) / max(weight, _EPS), 0.0, 1.0)

    def features(self) -> np.ndarray:
        x = self._x
        mf = x.mean(axis=0)
        r = float(np.linalg.norm(mf))
        coh = x @ (mf / r) if r > 1e-9 else np.zeros(self.oscillators)
        top = np.zeros(self.top_pairs)
        if self.top_pairs:
            plv = np.sort(self._plv_values())[::-1]
            top[:min(self.top_pairs, plv.size)] = plv[:self.top_pairs]
        return np.clip(np.concatenate([coh, [r], top]), -1.0, 1.0)     # every block is a cosine or a modulus

    # ------------------------------------------------------------------ accounting / persistence
    def parameter_count(self) -> int:
        return 0

    def state_numbers(self) -> int:
        return int(self._coupling.size + self._omega.size + self._stim_dir.size + self._x.size + self._x0.size
                   + 2 * self._plv.size + self._group.size + self._count.size + self._last_mean.size + 1)

    def telemetry(self) -> dict[str, Any]:
        plv = self._plv_values()
        off = self._coupling[~np.eye(self.oscillators, dtype=bool)]
        nonzero = off[off != 0.0]
        return {'id': self.id, 'mode': self.mode, 'oscillators': self.oscillators, 'dim': self.dim,
                'steps': int(self._steps), 'order_parameter': order_parameter(self._x),
                'mean_plv': float(plv.mean()) if plv.size else 0.0, 'max_plv': float(plv.max()) if plv.size else 0.0,
                'coupling_max_abs_row_sum': float(np.max(np.abs(self._coupling).sum(axis=1))),
                'coupling_nonzero': int(nonzero.size),
                'coupling_negative_fraction': float(np.mean(nonzero < 0.0)) if nonzero.size else 0.0}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id, 'config': self.config(), 'n': int(self._n), 'steps': int(self._steps),
                'x': self._x.tolist(), 'plv_re': self._plv.real.tolist(), 'plv_im': self._plv.imag.tolist(),
                'last_mean': self._last_mean.tolist(), 'coupling': self._coupling.tolist()}

    def import_state(self, state: Mapping[str, Any]) -> None:
        super().import_state(state)
        c, p = self.oscillators, self._iu[0].size
        try:
            cfg = dict(state['config'])
            n, steps = state['n'], state['steps']
            x = np.asarray(state['x'], dtype=np.float64)
            plv = np.asarray(state['plv_re'], dtype=np.float64) + 1j * np.asarray(state['plv_im'], dtype=np.float64)
            last = np.asarray(state['last_mean'], dtype=np.float64)
            coupling = np.asarray(state['coupling'], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('kuramoto_binding state is incomplete or malformed') from exc
        if cfg != self.config() or n != self._n or type(steps) is not int or steps < 0:
            raise ValueError('kuramoto_binding state does not match this extension (config, n or steps)')
        if (x.shape != (c, self.dim) or plv.shape != (p,) or last.shape != (c,) or coupling.shape != (c, c)
                or not all(np.all(np.isfinite(a)) for a in (x, plv, last, coupling))):
            raise ValueError('kuramoto_binding state arrays have the wrong shape or non-finite values')
        if np.max(np.abs(np.linalg.norm(x, axis=1) - 1.0)) > 1e-6 or np.any(np.abs(plv) > 1.0 + 1e-9):
            raise ValueError('kuramoto_binding state is out of bounds (unit norm, |plv| <= 1)')
        if np.any(np.abs(coupling) > _MAX_COUPLING) or np.any(np.diag(coupling) != 0.0):
            raise ValueError('kuramoto_binding coupling is out of bounds')
        # stored as given (unit norm checked above): renormalising would perturb the last bits and break the
        # bit-exact continuation of an exported run
        self._x, self._plv, self._last_mean = x.copy(), plv, last
        self._coupling, self._steps = coupling, int(steps)


# ====================================================================== live observer
class KuramotoObserver:
    """Read-only AKOrN observer of live unit activity.

    ``M = min(oscillators, n_units)`` oscillators, each driven by one subsampled unit (seeded, sorted
    indices): stimulus ``c_i = stimulus_scale * tanh(rate_i) * u_i`` with a fixed seeded unit vector ``u_i``.
    Coupling ``J = J0 + hebbian_scale * H / sqrt(M - 1)``:

    * ``J0`` -- fixed seeded sparse coupling, ``fan_in`` random partners per oscillator with weight
      ``coupling / fan_in``;
    * ``H`` -- slow Hebbian co-activity estimate: rates are standardised with running EWMA mean/variance
      (rate 0.05) and clipped to +/-3 sd (``z``); ``S <- (1 - eta) S + eta z z^T`` starts at the identity
      (a shrinkage prior), and ``H_ij = S_ij / sqrt(S_ii S_jj)`` off the diagonal, so ``|H| <= 1`` and early
      estimates stay near 0. ``eta = hebbian_rate``.

    ``iterations`` AKOrN updates run per observation. Communities (for Shanahan's indices) are contiguous
    blocks of the sampled unit order (NEUROWEAVE regions are contiguous unit blocks) or the given ``groups``
    labels (one per unit). The observer copies what it reads and never writes to its input. Unknown keyword
    parameters raise ``ValueError``; FrontierSpecV1 parameters meant for other observers (``sync_pairs``,
    ``sae_features``, ``sae_k``, ``fc_window``, ``max_observed_units``, ``observe_every``) are ignored."""

    module_id = MODULE_ID
    schema = OBSERVER_STATE_SCHEMA

    def __init__(self, n_units: int, *, seed: int = 7, history: int = 256, oscillators: int = 64, dim: int = 2,
                 iterations: int = 2, gamma: float = 0.1, omega_mean: float = 1.0, omega_spread: float = 0.3,
                 fan_in: int = 4, coupling: float = 0.5, hebbian_rate: float = 0.01, hebbian_scale: float = 2.0,
                 stimulus_scale: float = 0.5, communities: int = 8, groups: Sequence[int] | None = None,
                 bins: int = 36, min_cluster_fraction: float = 0.1, **params: Any) -> None:
        unknown = sorted(set(params) - _SPEC_OBSERVER_PARAMS)
        if unknown:
            raise ValueError(f'unknown KuramotoObserver parameters: {unknown}')
        self.n_units = _int('n_units', n_units, 1, 1_000_000)
        self.seed = _int('seed', seed, 0, 2**31 - 1)
        self.history = _int('history', history, 2, 8192)
        self.oscillators = _int('oscillators', oscillators, 1, 4096)
        self.dim = _even_dim(dim)
        self.iterations = _int('iterations', iterations, 1, 64)
        self.gamma = _num('gamma', gamma, 0.0, 10.0, open_low=True)
        self.omega_mean = _num('omega_mean', omega_mean, -100.0, 100.0)
        self.omega_spread = _num('omega_spread', omega_spread, 0.0, 100.0)
        self.fan_in = _int('fan_in', fan_in, 0, 4096)
        self.coupling = _num('coupling', coupling, -100.0, 100.0)
        self.hebbian_rate = _num('hebbian_rate', hebbian_rate, 0.0, 1.0)
        self.hebbian_scale = _num('hebbian_scale', hebbian_scale, -100.0, 100.0)
        self.stimulus_scale = _num('stimulus_scale', stimulus_scale, -100.0, 100.0)
        self.communities = _int('communities', communities, 1, 4096)
        self.bins = _int('bins', bins, 2, 3600)
        self.min_cluster_fraction = _num('min_cluster_fraction', min_cluster_fraction, 0.0, 1.0, open_low=True)
        m = self.m = min(self.oscillators, self.n_units)
        rng = np.random.default_rng(self.seed + _OBS_OFFSET)
        self._units = (np.arange(self.n_units) if m == self.n_units
                       else np.sort(rng.choice(self.n_units, size=m, replace=False)))
        if groups is None:
            self.groups = None
            self.n_communities = min(self.communities, m)
            self._community = (np.arange(m) * self.n_communities) // m
        else:
            g = np.asarray(groups)
            if g.shape != (self.n_units,) or not np.issubdtype(g.dtype, np.integer) or np.any(g < 0):
                raise ValueError('groups must be one non-negative int label per unit')
            self.groups = [int(v) for v in g]
            _, inverse = np.unique(g[self._units], return_inverse=True)
            self._community = inverse.astype(np.int64).ravel()
            self.n_communities = int(self._community.max()) + 1
        self._x0 = _unit_rows(rng, m, self.dim)
        self._omega = natural_frequency_matrices(
            self.omega_mean + self.omega_spread * rng.standard_normal((m, self.dim // 2)), self.dim)
        self._stim_dir = _unit_rows(rng, m, self.dim)
        k = min(self.fan_in, m - 1)
        self._j0 = np.zeros((m, m))
        if k:
            for i in range(m):
                others = rng.choice(m - 1, size=k, replace=False)
                self._j0[i, others + (others >= i)] = self.coupling / k
        self.reset()

    @classmethod
    def from_spec(cls, n_units: int, spec: Any, **params: Any) -> 'KuramotoObserver':
        """Observer sized from a :class:`~synapse.frontier.spec.FrontierSpecV1` (``seed``, ``history``);
        ``params`` are the observer's own keyword parameters (``oscillators``, ``dim``, ...)."""
        if not isinstance(spec, FrontierSpecV1):
            raise ValueError('spec must be a FrontierSpecV1')
        return cls(n_units, seed=spec.seed, history=spec.history, **params)

    @property
    def observed_steps(self) -> int:
        return int(self._steps)

    # ------------------------------------------------------------------ state
    def config(self) -> dict[str, Any]:
        return {'n_units': self.n_units, 'seed': self.seed, 'history': self.history, 'oscillators': self.oscillators,
                'dim': self.dim, 'iterations': self.iterations, 'gamma': self.gamma, 'omega_mean': self.omega_mean,
                'omega_spread': self.omega_spread, 'fan_in': self.fan_in, 'coupling': self.coupling,
                'hebbian_rate': self.hebbian_rate, 'hebbian_scale': self.hebbian_scale,
                'stimulus_scale': self.stimulus_scale, 'communities': self.communities, 'groups': self.groups,
                'bins': self.bins, 'min_cluster_fraction': self.min_cluster_fraction}

    def reset(self) -> None:
        m = self.m
        self._x = self._x0.copy()
        self._coact = np.eye(m)
        self._mu = np.zeros(m)
        self._var = np.ones(m)
        self._r_hist = np.zeros(self.history)
        self._comm_hist = np.zeros((self.history, self.n_communities))
        self._cluster_hist = np.zeros(self.history, dtype=np.int64)
        self._pos = 0
        self._filled = 0
        self._steps = 0

    def hebbian(self) -> np.ndarray:
        """Hebbian co-activity estimate ``H`` (running correlation, zero diagonal, ``|H| <= 1``); a copy."""
        d = np.sqrt(np.clip(np.diag(self._coact), 0.0, None))
        denom = np.outer(d, d)
        h = np.where(denom > _EPS, self._coact / np.where(denom > _EPS, denom, 1.0), 0.0)
        np.fill_diagonal(h, 0.0)
        return np.clip(h, -1.0, 1.0)

    def coupling_matrix(self) -> np.ndarray:
        """Current effective coupling ``J0 + hebbian_scale * H / sqrt(M - 1)``; a copy."""
        return self._j0 + self.hebbian_scale * self.hebbian() / math.sqrt(max(1, self.m - 1))

    # ------------------------------------------------------------------ observe
    def observe(self, x: np.ndarray) -> None:
        v = np.asarray(x, dtype=np.float64)
        if v.ndim != 1 or v.size != self.n_units or not np.all(np.isfinite(v)):
            raise ValueError(f'observe expects a finite 1-D array of length {self.n_units}')
        rate = np.clip(v[self._units], -10.0, 10.0)        # fancy indexing copies; the input is never written
        if self._steps == 0:
            self._mu = rate.copy()
            z = np.zeros(self.m)
        else:
            d = rate - self._mu
            self._mu = self._mu + _STATS_RATE * d
            self._var = (1.0 - _STATS_RATE) * (self._var + _STATS_RATE * d * d)
            z = np.clip(d / np.sqrt(self._var + 1e-6), -_Z_CLIP, _Z_CLIP)
        if self.hebbian_rate > 0.0:
            self._coact = (1.0 - self.hebbian_rate) * self._coact + self.hebbian_rate * np.outer(z, z)
        coupling = self.coupling_matrix()
        stim = self.stimulus_scale * np.tanh(rate)[:, None] * self._stim_dir
        xs = self._x
        for _ in range(self.iterations):
            xs = _step(xs, coupling, stim, self._omega, self.gamma)
        self._x = xs
        p = self._pos
        self._r_hist[p] = min(1.0, float(np.linalg.norm(xs.mean(axis=0))))
        self._comm_hist[p] = np.minimum(1.0, group_coherence(xs, self._community, self.n_communities))
        self._cluster_hist[p] = count_phase_clusters(np.arctan2(xs[:, 1], xs[:, 0]), bins=self.bins,
                                                     min_fraction=self.min_cluster_fraction)
        self._pos = (p + 1) % self.history
        self._filled = min(self.history, self._filled + 1)
        self._steps += 1

    def summary(self) -> dict[str, Any]:
        """JSON-safe statistics over the bounded history (``None`` before the first observation).

        ``R_std`` is the metastability of the global order parameter (its std over the history);
        ``metastability_index`` and ``chimera_index`` follow Shanahan (2010) over the communities;
        ``phase_clusters`` counts dense runs of the phase histogram (first rotation plane)."""
        h = self._filled
        base: dict[str, Any] = {'module': MODULE_ID, 'evidence_tier': EVIDENCE_TIER, 'observed_steps': int(self._steps),
                                'history_len': int(h), 'history': self.history, 'oscillators': self.m,
                                'dim': self.dim, 'communities': self.n_communities}
        hebb = np.abs(self.hebbian())
        base['hebbian_mean_abs'] = float(hebb.sum() / max(1, self.m * (self.m - 1)))
        base['hebbian_max_abs'] = float(hebb.max()) if hebb.size else 0.0
        if h == 0:
            base.update({k: None for k in ('R_last', 'R_mean', 'R_std', 'metastability_index', 'chimera_index',
                                           'community_R_mean', 'phase_clusters', 'phase_clusters_mean')})
            return base
        last = (self._pos - 1) % self.history
        r, comm, cl = self._r_hist[:h], self._comm_hist[:h], self._cluster_hist[:h]   # order is irrelevant here
        idx = shanahan_indices(comm)
        base.update({'R_last': float(self._r_hist[last]), 'R_mean': float(r.mean()), 'R_std': float(r.std()),
                     'metastability_index': idx['metastability_index'], 'chimera_index': idx['chimera_index'],
                     'community_R_mean': float(comm.mean()), 'phase_clusters': int(self._cluster_hist[last]),
                     'phase_clusters_mean': float(cl.mean())})
        return base

    def export_state(self) -> dict[str, Any]:
        return {'module': MODULE_ID, 'schema': self.schema, 'config': self.config(), 'x': self._x.tolist(),
                'coactivity': self._coact.tolist(), 'mean': self._mu.tolist(), 'var': self._var.tolist(),
                'r_history': self._r_hist.tolist(), 'community_history': self._comm_hist.tolist(),
                'cluster_history': self._cluster_hist.tolist(), 'pos': int(self._pos), 'filled': int(self._filled),
                'steps': int(self._steps)}

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('module') != MODULE_ID or state.get('schema') != self.schema:
            raise ValueError('state does not belong to a KuramotoObserver')
        m, hlen = self.m, self.history
        try:
            cfg = dict(state['config'])
            x = np.asarray(state['x'], dtype=np.float64)
            coact = np.asarray(state['coactivity'], dtype=np.float64)
            mu = np.asarray(state['mean'], dtype=np.float64)
            var = np.asarray(state['var'], dtype=np.float64)
            r_hist = np.asarray(state['r_history'], dtype=np.float64)
            comm = np.asarray(state['community_history'], dtype=np.float64)
            cl_raw = np.asarray(state['cluster_history'])
            pos, filled, steps = state['pos'], state['filled'], state['steps']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('KuramotoObserver state is incomplete or malformed') from exc
        if cfg != self.config():
            raise ValueError('KuramotoObserver state was made with a different configuration')
        shapes = ((x, (m, self.dim)), (coact, (m, m)), (mu, (m,)), (var, (m,)), (r_hist, (hlen,)),
                  (comm, (hlen, self.n_communities)))
        if any(a.shape != s or not np.all(np.isfinite(a)) for a, s in shapes):
            raise ValueError('KuramotoObserver state arrays have the wrong shape or non-finite values')
        if cl_raw.shape != (hlen,) or not np.issubdtype(cl_raw.dtype, np.integer) or np.any(cl_raw < 0):
            raise ValueError('cluster_history must hold non-negative ints')
        if not all(type(v) is int for v in (pos, filled, steps)) or not (0 <= pos < hlen and 0 <= filled <= hlen
                                                                         and steps >= filled):
            raise ValueError('KuramotoObserver counters are inconsistent')
        if filled < hlen and pos != filled:
            raise ValueError('KuramotoObserver ring-buffer position is inconsistent')
        bound = _Z_CLIP ** 2 + 1.0
        if (np.max(np.abs(np.linalg.norm(x, axis=1) - 1.0)) > 1e-6 or not np.allclose(coact, coact.T, atol=1e-12)
                or np.any(np.abs(coact) > bound) or np.any(np.diag(coact) < 0.0) or np.any(var < 0.0)
                or np.any(np.abs(mu) > 10.0 + 1e-9) or np.any((r_hist < 0.0) | (r_hist > 1.0))
                or np.any((comm < 0.0) | (comm > 1.0))):
            raise ValueError('KuramotoObserver state is out of bounds')
        self._x, self._coact, self._mu, self._var = x.copy(), coact.copy(), mu.copy(), var.copy()   # bit-exact resume
        self._r_hist, self._comm_hist, self._cluster_hist = r_hist.copy(), comm.copy(), cl_raw.astype(np.int64)
        self._pos, self._filled, self._steps = pos, filled, steps


# ====================================================================== synthetic binding benchmark
def make_binding_task(seed: int, *, groups: int = 4, per_group: int = 8, samples: int = 64,
                      noise: float = 1.0) -> dict[str, Any]:
    """``groups`` latent signals ``s_g(t) ~ N(0, 1)``; feature ``i`` of group ``g`` is
    ``a_i s_g(t) + noise * eps_i(t)`` with loading ``a_i ~ U(0.5, 1.5)``. Returns ``features`` (C, T),
    ground-truth ``labels`` (C,) and a digest of the features."""
    seed = _int('seed', seed, 0, 2**31 - 1)
    g = _int('groups', groups, 2, 64)
    f = _int('per_group', per_group, 2, 256)
    t = _int('samples', samples, 4, 100_000)
    sigma = _num('noise', noise, 0.0, 100.0)
    rng = np.random.default_rng(seed + _TASK_OFFSET)
    latent = rng.standard_normal((g, t))
    labels = np.repeat(np.arange(g), f)
    loading = rng.uniform(0.5, 1.5, labels.size)
    feats = loading[:, None] * latent[labels] + sigma * rng.standard_normal((labels.size, t))
    digest = hashlib.sha256(json.dumps(np.round(feats, 12).tolist()).encode('utf-8')).hexdigest()
    return {'features': feats, 'labels': labels, 'digest': digest}


def correlation_coupling(features: Any, *, scale: float = 2.0, center: bool = True) -> np.ndarray:
    """Symmetric couplings from input correlations: Pearson correlation of the feature rows, minus the mean
    off-diagonal correlation when ``center`` (a uniform global inhibition, so weakly related features repel),
    zero diagonal, scaled so the largest absolute row sum equals ``scale``."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)):
        raise ValueError('features must be a finite (C, T) array with C, T >= 2')
    s = _num('scale', scale, -100.0, 100.0)
    xc = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.sum(xc * xc, axis=1))
    unit = xc / np.where(sd > _EPS, sd, 1.0)[:, None]
    corr = unit @ unit.T                                 # a constant feature has zero correlation with all
    np.fill_diagonal(corr, 0.0)
    if center:
        c = corr.shape[0]
        corr = corr - corr.sum() / (c * (c - 1))
        np.fill_diagonal(corr, 0.0)
    row = float(np.max(np.abs(corr).sum(axis=1)))
    return s * corr / row if row > _EPS else np.zeros_like(corr)


def shuffle_coupling(coupling: Any, seed: int) -> np.ndarray:
    """Control coupling: the upper-triangle values of a symmetric coupling matrix permuted (seeded) and
    mirrored, so the multiset of coupling values is kept and its alignment with the inputs is destroyed."""
    j = np.asarray(coupling, dtype=np.float64)
    if j.ndim != 2 or j.shape[0] != j.shape[1] or not np.all(np.isfinite(j)) or not np.allclose(j, j.T):
        raise ValueError('coupling must be a finite symmetric square matrix')
    iu = np.triu_indices(j.shape[0], 1)
    vals = j[iu][np.random.default_rng(_int('seed', seed, 0, 2**62)).permutation(iu[0].size)]
    out = np.zeros_like(j)
    out[iu] = vals
    return out + out.T


def _seeds(seeds: Any) -> tuple[int, ...]:
    try:
        out = tuple(_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


_BENCH_DEFAULTS: dict[str, Any] = {'groups': 4, 'per_group': 8, 'samples': 256, 'noise': 1.0, 'dim': 2,
                                   'iterations': 120, 'gamma': 0.1, 'coupling_scale': 2.0, 'omega_mean': 1.0,
                                   'omega_spread': 0.05, 'kmeans_restarts': 8}
_BENCH_QUICK: dict[str, Any] = {'groups': 3, 'per_group': 6, 'samples': 128}


def _max_group_cosine(x: np.ndarray, labels: np.ndarray, groups: int) -> float:
    """Largest cosine between the mean-field directions of two different ground-truth groups.

    Near 1 means two groups ended up (nearly) in phase, which no clustering of the final states can undo."""
    sums = np.zeros((groups, x.shape[1]))
    np.add.at(sums, labels, x)
    norm = np.linalg.norm(sums, axis=1)
    unit = sums / np.where(norm > _EPS, norm, 1.0)[:, None]
    cos = unit @ unit.T
    return float(np.clip(cos[np.triu_indices(groups, 1)].max(), -1.0, 1.0))


def _binding_arm(seed: int, x0: np.ndarray, coupling: np.ndarray | None, omega: np.ndarray, labels: np.ndarray,
                 cfg: Mapping[str, Any], checkpoints: Sequence[int]) -> dict[str, Any]:
    x = x0.copy()
    g = int(cfg['groups'])
    curve = []
    for t in range(int(cfg['iterations']) + 1):
        if t in checkpoints:
            pred = kmeans_labels(x, g, seed=seed + _KMEANS_OFFSET, restarts=int(cfg['kmeans_restarts']))
            curve.append(adjusted_rand_index(labels, pred))
        if t < cfg['iterations']:
            x = _step(x, coupling, None, omega, float(cfg['gamma']))
    return {'seed': int(seed), 'ari': curve[-1], 'ari_by_iteration': curve,
            'within_group_R': float(group_coherence(x, labels, g).mean()), 'global_R': order_parameter(x),
            'max_group_cosine': _max_group_cosine(x, labels, g)}


def _arm_summary(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {'ari_mean': float(np.mean([a['ari'] for a in per_seed])),
                           'ari_min': float(np.min([a['ari'] for a in per_seed]))}
    for key in ('within_group_R', 'global_R', 'max_group_cosine'):
        if key in per_seed[0]:
            out[f'{key}_mean'] = float(np.mean([a[key] for a in per_seed]))
    if 'ari_by_iteration' in per_seed[0]:
        out['ari_by_iteration_mean'] = np.mean([a['ari_by_iteration'] for a in per_seed], axis=0).tolist()
    return out


def run_binding_benchmark(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Synthetic binding: do oscillators coupled by input correlations synchronise by latent group?

    Arms share each seed's task, initial states and natural frequencies: ``candidate`` (correlation
    coupling), ``shuffled_control`` (same coupling values, shuffled), ``zero_control`` (no coupling) and a
    non-oscillatory ``reference_correlation_kmeans`` (k-means on correlation rows). Final oscillator states
    are clustered with seeded k-means (``k = groups``) and scored by the adjusted Rand index. The verdict
    is ``harness.compare`` of candidate vs shuffled-control ARI over seeds."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    unknown = sorted(set(overrides) - set(_BENCH_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown benchmark parameters: {unknown}')
    cfg = {**_BENCH_DEFAULTS, **(_BENCH_QUICK if quick else {}), **overrides}
    groups, per_group = _int('groups', cfg['groups'], 2, 64), _int('per_group', cfg['per_group'], 2, 256)
    _int('samples', cfg['samples'], 4, 100_000)
    _num('noise', cfg['noise'], 0.0, 100.0)
    dim = _even_dim(cfg['dim'])
    iterations = _int('iterations', cfg['iterations'], 1, 100_000)
    _num('gamma', cfg['gamma'], 0.0, 10.0, open_low=True)
    _num('coupling_scale', cfg['coupling_scale'], -100.0, 100.0)
    _num('omega_mean', cfg['omega_mean'], -100.0, 100.0)
    _num('omega_spread', cfg['omega_spread'], 0.0, 100.0)
    _int('kmeans_restarts', cfg['kmeans_restarts'], 1, 1000)
    c = groups * per_group
    checkpoints = sorted({0, iterations // 8, iterations // 4, iterations // 2, iterations})
    arms: dict[str, list[dict[str, Any]]] = {'candidate': [], 'shuffled_control': [], 'zero_control': [],
                                             'reference_correlation_kmeans': []}
    digests = []
    t0 = time.perf_counter()
    for s in seed_list:
        task = make_binding_task(s, groups=groups, per_group=per_group, samples=cfg['samples'], noise=cfg['noise'])
        labels = task['labels']
        digests.append(task['digest'])
        coupling = correlation_coupling(task['features'], scale=cfg['coupling_scale'])
        rng = np.random.default_rng(s + _INIT_OFFSET)
        x0 = _unit_rows(rng, c, dim)
        omega = natural_frequency_matrices(
            cfg['omega_mean'] + cfg['omega_spread'] * rng.standard_normal((c, dim // 2)), dim)
        for name, j in (('candidate', coupling), ('shuffled_control', shuffle_coupling(coupling, s + _SHUFFLE_OFFSET)),
                        ('zero_control', None)):
            arms[name].append(_binding_arm(s, x0, j, omega, labels, cfg, checkpoints))
        corr_rows = correlation_coupling(task['features'], scale=1.0, center=False)
        ref = kmeans_labels(corr_rows, groups, seed=s + _KMEANS_OFFSET, restarts=int(cfg['kmeans_restarts']))
        arms['reference_correlation_kmeans'].append({'seed': int(s), 'ari': adjusted_rand_index(labels, ref)})
    wall = time.perf_counter() - t0
    ari = {name: [a['ari'] for a in per_seed] for name, per_seed in arms.items()}
    comparisons = {
        'candidate_vs_shuffled_control': harness.compare(ari['candidate'], ari['shuffled_control']),
        'candidate_vs_zero_control': harness.compare(ari['candidate'], ari['zero_control']),
        'candidate_vs_reference_correlation_kmeans': harness.compare(ari['candidate'],
                                                                     ari['reference_correlation_kmeans']),
    }
    flops_iter = 2 * c * c + 12 * c * dim + 4 * c * dim * dim
    receipt = harness.base_receipt(BENCHMARK_ID, 'binding_benchmark', {
        **cfg, 'seeds': list(seed_list), 'quick': quick, 'oscillators': c, 'checkpoints': checkpoints,
        'stimulus': 'none (c_i = 0; binding comes from the coupling alone)',
        'coupling': 'pearson correlation minus mean off-diagonal correlation, zero diagonal, max |row sum| = coupling_scale',
        'clustering': 'seeded k-means++ (k = groups) on final unit vectors; adjusted Rand index vs ground truth',
        'diagnostics': 'within_group_R = mean ground-truth group order parameter; max_group_cosine = largest cosine '
                       'between two group mean-field directions (near 1: two groups in phase)'})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['task_digests'] = digests
    receipt['arms'] = {name: {'per_seed': per_seed, 'summary': _arm_summary(per_seed)} for name, per_seed in arms.items()}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['candidate_vs_shuffled_control']['verdict']
    receipt['verdict_basis'] = 'candidate_vs_shuffled_coupling_control'
    receipt['budget'] = {'oscillators': c, 'dim': dim, 'state_numbers_per_arm': c * dim + c * dim * dim + c * c,
                         'trainable_parameters': 0,
                         'capacity_note': 'candidate and shuffled control use the same multiset of coupling values'}
    receipt['estimated_cost'] = {'flops_per_iteration': flops_iter,
                                 'oscillator_flops_total': flops_iter * iterations * 3 * len(seed_list)}
    receipt['measured_wallclock_s'] = wall
    receipt['notes'] = ('Synthetic binding task with a known partition. Oscillator updates follow the AKOrN rule '
                        'with fixed (not learned) correlation couplings and no stimulus; nothing here reproduces '
                        'AKOrN results. The task is easy for a direct clustering of the correlation matrix too (see '
                        'the reference arm), so this tests that the oscillator dynamics implement binding, not that '
                        'they beat conventional clustering. Limitation: with symmetric scalar couplings the '
                        'between-group repulsion only constrains the resultant of the group mean fields, so for four '
                        'or more groups (and for three groups whose latents correlate by chance in a finite sample) '
                        'two groups can settle nearly in phase; max_group_cosine near 1 flags this, and within_group_R '
                        'shows whether each group synchronised internally.')
    return harness.seal(receipt)


# ====================================================================== NEUROWEAVE extension experiment
_EXT_PARAM_KEYS = ('mode', 'units', 'dim', 'iterations', 'gamma', 'coupling_scale', 'stimulus_scale', 'omega_mean',
                   'omega_spread', 'top_pairs', 'plv_rate')
_RUN_KEYS = ('task', 'heldout_seeds', 'steps', 'heldout_steps', 'neurons', 'topology', 'mechanisms', 'metric')


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Baseline vs ``KuramotoExtension`` vs random-trace control on NEUROWEAVE ``context_switch``.

    ``quick=True`` uses the harness quick sizes (24 neurons, 24 train steps, one 12-step held-out stream);
    ``seeds`` are always used as given. Extension parameters and harness run parameters can be overridden."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    unknown = sorted(set(overrides) - set(_EXT_PARAM_KEYS) - set(_RUN_KEYS))
    if unknown:
        raise ValueError(f'unknown run_experiment parameters: {unknown}')
    ext_params = {k: overrides[k] for k in _EXT_PARAM_KEYS if k in overrides}
    probe = KuramotoExtension(0, **ext_params)          # validates the extension parameters up front
    run: dict[str, Any] = {'task': 'context_switch', 'heldout_seeds': (100, 101), 'steps': 300, 'heldout_steps': None,
                           'neurons': 48, 'topology': 'clustered', 'mechanisms': ()}
    if quick:
        run.update({k: v for k, v in harness.QUICK_ARGS.items() if k != 'seeds'})
    run.update({k: overrides[k] for k in _RUN_KEYS if k in overrides})
    notes = (f'Candidate: AKOrN-style oscillators ({probe.mode} mode, {probe.oscillators} oscillators, N={probe.dim}) '
             'coupled by region-aggregated synapse weights copied at bind and driven by group mean outputs; they add '
             f'{probe.feature_dim} readout features and never change the drive. Control: random leaky traces with the '
             'same feature count. Synthetic temporal task; not a reproduction of AKOrN results.')
    t0 = time.perf_counter()
    receipt = dict(harness.run_extension_experiment(MODULE_ID, lambda s: KuramotoExtension(int(s), **ext_params),
                                                    seeds=seed_list, control='random_trace', notes=notes, **run))
    wall = time.perf_counter() - t0
    receipt.pop('digest', None)
    cfg = probe.config()
    cfg.pop('seed')
    receipt['config'] = {**receipt['config'], 'quick': quick, 'extension': cfg}
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['measured_wallclock_s_total'] = wall
    return harness.seal(receipt)


EXPERIMENTS = {MODULE_ID: run_experiment, BENCHMARK_ID: run_binding_benchmark}
