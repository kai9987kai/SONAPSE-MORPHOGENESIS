"""Structural, functional and effective connectivity (frontier module ``connectomics``).

Inspired by the strictly separated structural / functional / effective connectomics of SONAPSE-CONNECTOME
(kai9987kai; ``sonapse_connectome/connectome/functional.py`` and ``effective.py``). This module re-implements
those ideas for NEUROWEAVE brains with known-answer validation. It is a synthetic computational abstraction: the
"structure" is a toy synapse table and the "function" is the activity of abstract units, not imaging data.

What is computed
----------------
* **Functional connectivity (FC).** Pearson and Spearman correlation matrices, and partial correlation from a
  ridge-regularised precision matrix: ``Omega = (R + ridge * I)^-1`` on the correlation matrix ``R`` and
  ``P_ij = -Omega_ij / sqrt(Omega_ii Omega_jj)``. Zero-variance units get zero rows and a zero diagonal.
* **Dynamic FC (dFC).** Sliding-window Pearson FC, k-means clustering of the windowed upper triangles into FC
  states, the number of states chosen by the average silhouette width, and dwell times (run lengths of the state
  sequence). In the spirit of Allen et al. (2014); see the doc for what differs.
* **Effective connectivity (EC).** A ridge VAR(1) (lagged regression, Granger-style predictive influence, not an
  interventional effect): ``x_{t+1} = A x_t + e``, ``A[target, source]``. The significance mask comes from a
  per-target conditional block-permutation null (each target keeps its own past; the other sources are
  block-permuted jointly in time), a two-sided normal p-value from the null mean and spread, and
  Benjamini-Hochberg FDR control over the off-diagonal edges.
* **Structure-function coupling.** Spearman correlation between symmetric structural weights and FC over node
  pairs (all pairs and structurally connected pairs), and the same for weighted communicability
  ``exp(D^-1/2 S D^-1/2)`` as an analytic predictor of FC.
* **FC modularity proxy.** Newman-Girvan modularity of the positive part of FC under a leading-eigenvector
  partition (recursive bisection, no vertex-moving refinement), so it is a lower bound on the maximum modularity.

References (verified 2026-11-19; full list and notes in ``docs/frontier/modules/connectomics.md``)
---------------------------------------------------------------------------------------------------
* C. J. Honey, O. Sporns, L. Cammoun, X. Gigandet, J. P. Thiran, R. Meuli, P. Hagmann, "Predicting human
  resting-state functional connectivity from structural connectivity", PNAS 106(6):2035-2040, February 2009,
  https://doi.org/10.1073/pnas.0811168106
* E. A. Allen, E. Damaraju, S. M. Plis, E. B. Erhardt, T. Eichele, V. D. Calhoun, "Tracking whole-brain
  connectivity dynamics in the resting state", Cerebral Cortex 24(3):663-676, March 2014,
  https://doi.org/10.1093/cercor/bhs352
* E. Estrada, N. Hatano, "Communicability in complex networks", Phys. Rev. E 77:036111, March 2008,
  https://doi.org/10.1103/PhysRevE.77.036111
* J. J. Crofts, D. J. Higham, "A weighted communicability measure applied to complex brain networks",
  J. R. Soc. Interface 6(33):411-414, April 2009, https://doi.org/10.1098/rsif.2008.0484 (eq. 2.2)
* M. E. J. Newman, "Modularity and community structure in networks", PNAS 103(23):8577-8582, June 2006,
  https://doi.org/10.1073/pnas.0601602103
* C. W. J. Granger, "Investigating causal relations by econometric models and cross-spectral methods",
  Econometrica 37(3):424-438, August 1969, https://doi.org/10.2307/1912791
* Y. Benjamini, Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple
  testing", J. R. Stat. Soc. B 57(1):289-300, 1995, https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
* P. J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis",
  J. Comput. Appl. Math. 20:53-65, November 1987, https://doi.org/10.1016/0377-0427(87)90125-7
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ['MODULE_ID', 'ConnectivityObserver', 'pearson_fc', 'spearman_fc', 'partial_correlation',
           'partial_from_covariance', 'precision_to_partial', 'sliding_window_fc', 'kmeans', 'silhouette_score',
           'dwell_statistics', 'dynamic_fc_states', 'var1_fit', 'block_permutation', 'var1_permutation_null',
           'benjamini_hochberg', 'effective_connectivity', 'auroc', 'structural_matrix', 'weighted_communicability',
           'structure_function_coupling', 'modularity', 'fc_communities', 'run_experiment', 'EXPERIMENTS']

MODULE_ID = 'connectomics'
OBSERVER_STATE_SCHEMA = 'ConnectivityObserverStateV1'
_VAR_OFFSET = 63_001
_CHAIN_OFFSET = 63_011
_SWITCH_OFFSET = 63_021
_KMEANS_OFFSET = 63_031
_PERM_OFFSET = 63_041
_OBS_UNITS_OFFSET = 63_051
_OBS_PERM_OFFSET = 63_061
_STATIONARY_OFFSET = 63_071
_TOP_EDGES = 5
_EPS = 1e-12
_SPEC_OBSERVER_PARAMS = frozenset({'sync_pairs', 'sae_features', 'sae_k', 'fc_window', 'max_observed_units',
                                   'observe_every'})


# ====================================================================== validation helpers
def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _check_float(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number in [{low}, {high}]')
    v = float(value)
    if not (math.isfinite(v) and low <= v <= high):
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return v


def _check_seeds(seeds: Any) -> tuple[int, ...]:
    try:
        out = tuple(_check_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


def _series(name: str, value: Any, *, min_rows: int = 3, min_cols: int = 2) -> np.ndarray:
    """A private float64 copy of a finite ``(T, n)`` array with ``T >= min_rows`` and ``n >= min_cols``."""
    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] < min_cols:
        raise ValueError(f'{name} must be a 2-D array with at least {min_rows} rows and {min_cols} columns')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    return arr


def _square(name: str, value: Any, n: int | None = None) -> np.ndarray:
    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1 or (n is not None and arr.shape[0] != n):
        raise ValueError(f'{name} must be a square matrix' + ('' if n is None else f' of size {n}'))
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    return arr


# ====================================================================== functional connectivity
def _zscore_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column z-scores (population sd) and the mask of non-constant columns (constant columns become 0)."""
    # A power-of-two rescaling keeps centering and squaring in range even
    # when different units are recorded at very different amplitudes.
    _, exponents = np.frexp(np.max(np.abs(X), axis=0))
    Z = np.ldexp(X, -exponents)
    Z -= Z[0].copy()
    Z -= Z.mean(axis=0)
    sd = np.sqrt((Z * Z).mean(axis=0))
    ok = sd > 0.0
    out = np.zeros_like(Z)
    out[:, ok] = Z[:, ok] / sd[ok]
    return out, ok


def _pearson(X: np.ndarray) -> np.ndarray:
    Z, ok = _zscore_columns(X)
    C = np.clip((Z.T @ Z) / X.shape[0], -1.0, 1.0)
    np.fill_diagonal(C, ok.astype(np.float64))
    return C


def pearson_fc(X: Any) -> np.ndarray:
    """Pearson FC of a ``(T, n)`` series. Constant units get zero rows/columns and a zero diagonal entry."""
    return _pearson(_series('X', X))


def spearman_fc(X: Any) -> np.ndarray:
    """Spearman FC: Pearson correlation of per-unit ranks (average ranks for ties)."""
    from scipy.stats import rankdata
    return _pearson(rankdata(_series('X', X), axis=0))


def precision_to_partial(precision: Any) -> np.ndarray:
    """Partial correlations ``-Omega_ij / sqrt(Omega_ii Omega_jj)`` (unit diagonal) from a precision matrix."""
    Om = _square('precision', precision)
    d = np.diag(Om)
    if np.any(d <= 0.0):
        raise ValueError('precision matrix must have a positive diagonal')
    s = np.sqrt(d)
    P = -Om / np.outer(s, s)
    np.fill_diagonal(P, 1.0)
    return np.clip(0.5 * (P + P.T), -1.0, 1.0)


def partial_from_covariance(cov: Any, *, ridge: float = 0.0) -> np.ndarray:
    """Partial correlations from a covariance matrix through the ridge-regularised precision ``(R + ridge I)^-1``
    of its correlation matrix ``R``. Units with zero variance get zero rows and a zero diagonal entry. With
    ``ridge = 0`` the result is exact; a singular ``R`` then raises ``ValueError``."""
    C = _square('cov', cov)
    ridge = _check_float('ridge', ridge, 0.0, 1e6)
    if not np.allclose(C, C.T, rtol=1e-9, atol=1e-12):
        raise ValueError('cov must be symmetric')
    C = 0.5 * (C + C.T)
    d = np.diag(C).copy()
    if np.any(d < 0):
        raise ValueError('cov must have non-negative variances')
    ok = d > 0
    if np.any(C[~ok] != 0):
        raise ValueError('zero-variance units must have zero covariance')
    n = C.shape[0]
    R = np.eye(n)
    s = np.sqrt(d[ok])
    R[np.ix_(ok, ok)] = C[np.ix_(ok, ok)] / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    S = R + ridge * np.eye(n)
    w = np.linalg.eigvalsh(S)
    if not w[0] > 1e-10 * max(1.0, float(w[-1])):
        raise ValueError('correlation matrix is singular or not positive definite; use ridge > 0')
    P = precision_to_partial(np.linalg.inv(S))
    P[~ok, :] = 0.0
    P[:, ~ok] = 0.0
    np.fill_diagonal(P, ok.astype(np.float64))
    return P


def partial_correlation(X: Any, *, ridge: float = 1e-2) -> np.ndarray:
    """Partial correlations of a ``(T, n)`` series from the ridge-regularised precision of its correlation matrix."""
    X = _series('X', X)
    return partial_from_covariance(_pearson(X), ridge=ridge)


# ====================================================================== dynamic FC
def sliding_window_fc(X: Any, window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Window start indices and Pearson FC per window, shape ``(W, n, n)``."""
    X = _series('X', X)
    T = X.shape[0]
    window = _check_int('window', window, 3, T)
    step = _check_int('step', step, 1, T)
    starts = np.arange(0, T - window + 1, step, dtype=np.int64)
    mats = np.stack([_pearson(X[s:s + window]) for s in starts])
    return starts, mats


def _sq_dists(V: np.ndarray, C: np.ndarray) -> np.ndarray:
    d = (V * V).sum(1)[:, None] - 2.0 * (V @ C.T) + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _canonical_labels(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel clusters in order of first appearance so labels do not depend on initialisation order."""
    order: list[int] = []
    for v in labels.tolist():
        if v not in order:
            order.append(v)
    order += [j for j in range(centers.shape[0]) if j not in order]
    remap = np.empty(centers.shape[0], dtype=np.int64)
    remap[np.asarray(order, dtype=np.int64)] = np.arange(centers.shape[0])
    return remap[labels], centers[np.asarray(order, dtype=np.int64)]


def kmeans(data: Any, k: int, *, seed: int = 0, n_init: int = 8, max_iter: int = 100) -> dict[str, Any]:
    """Deterministic k-means (Lloyd iterations, D^2-weighted seeding, ``n_init`` restarts, best inertia kept).

    Seeding and restarts use ``numpy.random.default_rng(seed + offset)``. Empty clusters are re-seeded with the
    point farthest from its centroid among clusters that keep at least one other member, so degenerate data
    (repeated points) never leaves a cluster empty. Labels are canonicalised to order of first appearance."""
    V = _series('data', data, min_rows=1, min_cols=1)
    W = V.shape[0]
    k = _check_int('k', k, 1, W)
    seed = _check_int('seed', seed, 0, 2**31 - 1)
    n_init = _check_int('n_init', n_init, 1, 1000)
    max_iter = _check_int('max_iter', max_iter, 1, 100_000)
    rng = np.random.default_rng(seed + _KMEANS_OFFSET)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(n_init):
        idx = [int(rng.integers(W))]
        d2 = _sq_dists(V, V[idx])[:, 0]
        for _j in range(1, k):
            total = float(d2.sum())
            nxt = int(rng.integers(W)) if total <= 0.0 else int(rng.choice(W, p=d2 / total))
            idx.append(nxt)
            d2 = np.minimum(d2, _sq_dists(V, V[[nxt]])[:, 0])
        C = V[idx].copy()
        labels = np.full(W, -1, dtype=np.int64)
        for _it in range(max_iter):
            D = _sq_dists(V, C)
            new = D.argmin(axis=1).astype(np.int64)
            for j in range(k):                      # re-seed empty clusters deterministically
                if not np.any(new == j):
                    # the point farthest from its centroid, taken only from a cluster with >= 2 members so
                    # that no other cluster is emptied (k <= W guarantees one exists); ties -> lowest index
                    sizes = np.bincount(new, minlength=k)
                    dist = np.where(sizes[new] > 1, D[np.arange(W), new], -np.inf)
                    far = int(dist.argmax())
                    new[far] = j
                    D[far] = np.inf
                    D[far, j] = 0.0
            if np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                C[j] = V[labels == j].mean(axis=0)
        inertia = float(_sq_dists(V, C)[np.arange(W), labels].sum())
        if best is None or inertia < best[2] - 1e-12 * max(1.0, best[2]):
            best = (labels.copy(), C.copy(), inertia)
    assert best is not None
    labels, C = _canonical_labels(best[0], best[1])
    return {'labels': labels, 'centroids': C, 'inertia': best[2], 'k': k}


def silhouette_score(data: Any, labels: Any) -> float | None:
    """Mean silhouette width (Rousseeuw, 1987) with Euclidean distances; ``None`` unless 2 <= clusters < points.
    Points alone in their cluster score 0."""
    V = _series('data', data, min_rows=1, min_cols=1)
    lab = np.asarray(labels)
    if lab.shape != (V.shape[0],) or lab.dtype.kind not in 'iu':
        raise ValueError('labels must be one integer per row of data')
    uniq, inv = np.unique(lab, return_inverse=True)
    W, k = V.shape[0], uniq.size
    if k < 2 or k >= W:
        return None
    D = np.sqrt(_sq_dists(V, V))
    onehot = np.zeros((W, k))
    onehot[np.arange(W), inv] = 1.0
    sums = D @ onehot
    sizes = onehot.sum(0)
    own_size = sizes[inv]
    a = np.where(own_size > 1, sums[np.arange(W), inv] / np.maximum(own_size - 1.0, 1.0), 0.0)
    mean_other = sums / sizes[None, :]
    mean_other[np.arange(W), inv] = np.inf
    b = mean_other.min(axis=1)
    denom = np.maximum(a, b)
    s = np.where((own_size > 1) & (denom > 0), (b - a) / np.where(denom > 0, denom, 1.0), 0.0)
    return float(s.mean())


def dwell_statistics(labels: Any, n_states: int | None = None) -> dict[str, Any]:
    """Run lengths of a state sequence: per-state mean dwell, overall mean dwell, occupancy and transitions."""
    lab = np.asarray(labels)
    if lab.ndim != 1 or lab.size == 0 or lab.dtype.kind not in 'iu' or lab.min() < 0:
        raise ValueError('labels must be a non-empty 1-D array of non-negative integers')
    k = int(lab.max()) + 1 if n_states is None else _check_int('n_states', n_states, int(lab.max()) + 1, 1_000_000)
    change = np.flatnonzero(np.diff(lab) != 0)
    starts = np.r_[0, change + 1]
    lengths = np.diff(np.r_[starts, lab.size])
    run_state = lab[starts]
    per_state = []
    for s in range(k):
        r = lengths[run_state == s]
        per_state.append(float(r.mean()) if r.size else None)
    return {'n_states': k, 'runs': int(lengths.size), 'transitions': int(change.size),
            'mean_dwell': float(lengths.mean()), 'mean_dwell_per_state': per_state,
            'occupancy': (np.bincount(lab, minlength=k) / lab.size).tolist()}


def dynamic_fc_states(X: Any, *, window: int, step: int | None = None, k_max: int = 4,
                      min_silhouette: float = 0.25, seed: int = 0, n_init: int = 8) -> dict[str, Any]:
    """Sliding-window FC states.

    Windowed FC upper triangles are clustered with :func:`kmeans` for ``k = 2 .. k_max``; the k with the largest
    mean silhouette wins. When no k reaches ``min_silhouette`` (a heuristic default of 0.25) or there are fewer
    than three windows, the series is reported as a single FC state. Dwell times are run lengths of the window
    state sequence, in windows and in samples (``x step``)."""
    X = _series('X', X)
    T, n = X.shape
    window = _check_int('window', window, 3, T)
    step = max(1, window // 4) if step is None else _check_int('step', step, 1, T)
    k_max = _check_int('k_max', k_max, 2, 16)
    min_silhouette = _check_float('min_silhouette', min_silhouette, -1.0, 1.0)
    seed = _check_int('seed', seed, 0, 2**31 - 1)
    starts, mats = sliding_window_fc(X, window, step)
    iu = np.triu_indices(n, 1)
    V = mats[:, iu[0], iu[1]]
    W = int(starts.size)
    scores: dict[int, float | None] = {}
    best_k, best_s, labels = 1, None, np.zeros(W, dtype=np.int64)
    if W >= 3:
        for k in range(2, min(k_max, W - 1) + 1):
            # modulo keeps the derived seed valid for seeds near 2**31 - 1 (unchanged for all smaller seeds)
            km = kmeans(V, k, seed=(seed + 101 * k) % 2**31, n_init=n_init)
            s = silhouette_score(V, km['labels'])
            scores[k] = s
            if s is not None and (best_s is None or s > best_s + 1e-12):
                best_k, best_s, labels = k, s, km['labels']
    if best_s is None or best_s < min_silhouette:
        best_k, labels = 1, np.zeros(W, dtype=np.int64)
    dw = dwell_statistics(labels, best_k)
    centroids = np.stack([mats[labels == j].mean(axis=0) for j in range(best_k)])
    return {'n_states': best_k, 'labels': labels, 'starts': starts, 'window': window, 'step': step, 'windows': W,
            'silhouette': {int(k): v for k, v in scores.items()}, 'best_silhouette': best_s,
            'mean_dwell_windows': dw['mean_dwell'], 'mean_dwell_samples': dw['mean_dwell'] * step,
            'mean_dwell_per_state_windows': dw['mean_dwell_per_state'], 'occupancy': dw['occupancy'],
            'transitions': dw['transitions'], 'state_fc': centroids}


# ====================================================================== effective connectivity
def _var_design(X: Any, ridge: float, standardize: bool) -> tuple[np.ndarray, np.ndarray, float]:
    X = _series('X', X, min_rows=4)
    ridge = _check_float('ridge', ridge, 0.0, 1e6)
    if type(standardize) is not bool:
        raise ValueError('standardize must be a boolean')
    if standardize:
        X, _ok = _zscore_columns(X)
    P = X[:-1] - X[:-1].mean(axis=0)
    F = X[1:] - X[1:].mean(axis=0)
    return P, F, ridge * P.shape[0]


def _solve(G: np.ndarray, R: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(G, R)
    except np.linalg.LinAlgError as exc:
        raise ValueError('lagged design is singular; use ridge > 0') from exc


def var1_fit(X: Any, *, ridge: float = 1e-2, standardize: bool = True) -> np.ndarray:
    """Ridge VAR(1): ``A = argmin sum_t ||x_{t+1} - A x_t||^2 + lambda ||A||_F^2`` with ``lambda = ridge * (T-1)``
    on (optionally z-scored) series with intercepts. Returns ``A[target, source]``."""
    P, F, lam = _var_design(X, ridge, standardize)
    n = P.shape[1]
    return _solve(P.T @ P + lam * np.eye(n), P.T @ F).T


def block_permutation(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of ``range(length)`` that reorders consecutive blocks of ``block`` indices (the last block
    may be shorter) and keeps the order inside each block."""
    length = _check_int('length', length, 1, 2**40)
    block = _check_int('block', block, 1, length)
    nb = -(-length // block)
    order = rng.permutation(nb)
    idx = (order[:, None] * block + np.arange(block)[None, :]).ravel()
    return idx[idx < length]


def _default_block(T: int) -> int:
    return max(2, int(round(T ** (1.0 / 3.0)))) if T >= 8 else 1


def _null_coefficients(P: np.ndarray, F: np.ndarray, lam: float, draws: int, block: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Per-target conditional block-permutation null for the ridge VAR(1).

    For draw b the rows of the source matrix are block-permuted (one permutation for all sources), except that
    target i keeps its own unpermuted past. Each target's ridge system is assembled from shared cross-products and
    all targets are solved in one batched call. Returns ``(draws, n, n)`` coefficients ``[target, source]``."""
    Tm, n = P.shape
    ar = np.arange(n)
    self_g = (P * P).sum(axis=0)
    self_r = (P * F).sum(axis=0)
    ridge_eye = lam * np.eye(n)
    out = np.empty((draws, n, n))
    for b in range(draws):
        Pp = P[block_permutation(Tm, block, rng)]
        cpp = Pp.T @ Pp
        cross = P.T @ Pp                    # cross[i, j] = P_i . Pp_j
        rf = Pp.T @ F                       # rf[j, i] = Pp_j . F_i
        G = np.repeat(cpp[None, :, :], n, axis=0)
        G[ar, ar, :] = cross                # row i of target i's Gram
        G[ar, :, ar] = cross                # column i of target i's Gram
        G[ar, ar, ar] = self_g
        r = rf.T.copy()
        r[ar, ar] = self_r
        out[b] = _solve(G + ridge_eye, r[..., None])[..., 0]
    return out


def var1_permutation_null(X: Any, *, ridge: float = 1e-2, standardize: bool = True, draws: int = 99,
                          block: int | None = None, seed: int = 0) -> np.ndarray:
    """Null coefficients ``(draws, n, n)`` of the per-target conditional block-permutation null (see
    :func:`effective_connectivity`)."""
    P, F, lam = _var_design(X, ridge, standardize)
    draws = _check_int('draws', draws, 1, 100_000)
    block = _default_block(P.shape[0]) if block is None else _check_int('block', block, 1, P.shape[0])
    rng = np.random.default_rng(_check_int('seed', seed, 0, 2**31 - 1) + _PERM_OFFSET)
    return _null_coefficients(P, F, lam, draws, block, rng)


def benjamini_hochberg(p_values: Any, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up rejections at FDR level ``alpha`` (boolean mask, same shape as the input)."""
    p = np.asarray(p_values, dtype=np.float64)
    alpha = _check_float('alpha', alpha, 0.0, 1.0)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError('p-values must be finite and in [0, 1]')
    flat = p.ravel()
    m = flat.size
    reject = np.zeros(m, dtype=bool)
    if m:
        order = np.argsort(flat, kind='stable')
        below = flat[order] <= alpha * np.arange(1, m + 1) / m
        if below.any():
            reject[order[:int(np.flatnonzero(below).max()) + 1]] = True
    return reject.reshape(p.shape)


def _ec_from_design(P: np.ndarray, F: np.ndarray, lam: float, draws: int, block: int, alpha: float,
                    rng: np.random.Generator) -> dict[str, Any]:
    n = P.shape[1]
    A = _solve(P.T @ P + lam * np.eye(n), P.T @ F).T
    null = _null_coefficients(P, F, lam, draws, block, rng)
    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1) if draws > 1 else np.zeros((n, n))
    dev = A - mu
    z = np.divide(dev, sd, out=np.zeros((n, n)), where=sd > _EPS)
    p_norm = np.vectorize(lambda v: math.erfc(abs(v) / math.sqrt(2.0)))(z) if n else np.zeros((0, 0))
    p_rank = (1.0 + (np.abs(null - mu[None]) >= np.abs(dev)[None] - 1e-15).sum(axis=0)) / (1.0 + draws)
    off = ~np.eye(n, dtype=bool)
    mask = np.zeros((n, n), dtype=bool)
    mask[off] = benjamini_hochberg(p_norm[off], alpha)
    mask &= sd > _EPS
    pairs = n * (n - 1)
    return {'A': A, 'z': z, 'p': p_norm, 'p_rank': p_rank, 'mask': mask, 'null_mean': mu, 'null_sd': sd,
            'density': float(mask.sum() / pairs) if pairs else 0.0, 'significant': int(mask.sum()),
            'draws': int(draws), 'block': int(block), 'alpha': float(alpha)}


def effective_connectivity(X: Any, *, ridge: float = 1e-2, null_draws: int = 99, block: int | None = None,
                           alpha: float = 0.05, seed: int = 0, standardize: bool = True) -> dict[str, Any]:
    """Ridge VAR(1) effective connectivity with a block-permutation significance mask.

    ``A[target, source]`` comes from :func:`var1_fit`. For every target the null refits its ridge regression with
    its own past intact and the other sources block-permuted jointly in time (``null_draws`` draws, block length
    ``round((T-1)^(1/3))`` by default), which keeps each source's autocorrelation inside blocks but breaks lagged
    coupling. Each edge gets ``z = (A - mean_null) / sd_null`` and a two-sided normal p-value; the mask is
    Benjamini-Hochberg at FDR ``alpha`` over the off-diagonal edges. Rank p-values ``(1 + #{|null| >= |A|}) /
    (1 + draws)`` are reported too. Lagged regression measures predictive (Granger-style) influence only."""
    P, F, lam = _var_design(X, ridge, standardize)
    draws = _check_int('null_draws', null_draws, 2, 100_000)
    block = _default_block(P.shape[0]) if block is None else _check_int('block', block, 1, P.shape[0])
    alpha = _check_float('alpha', alpha, 0.0, 1.0)
    rng = np.random.default_rng(_check_int('seed', seed, 0, 2**31 - 1) + _PERM_OFFSET)
    out = _ec_from_design(P, F, lam, draws, block, alpha, rng)
    out['ridge'] = float(ridge)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based average ranks with tie averaging (matches scipy.stats.rankdata)."""
    n = values.size
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    order = np.argsort(values, kind='mergesort')
    sorted_v = values[order]
    change = np.empty(n + 1, dtype=bool)
    change[0] = True
    change[1:n] = sorted_v[1:] != sorted_v[:-1]
    change[n] = True
    bounds = np.flatnonzero(change)
    if bounds.size == n + 1:
        ranks[order] = np.arange(1.0, n + 1.0, dtype=np.float64)
        return ranks
    avg_sorted = np.empty(n, dtype=np.float64)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        avg_sorted[start:stop] = 0.5 * (start + stop + 1)
    ranks[order] = avg_sorted
    return ranks


def auroc(scores: Any, labels: Any) -> float | None:
    """Area under the ROC curve via the Mann-Whitney statistic (average ranks for ties); ``None`` if a class is empty."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.shape != y.shape or not np.all(np.isfinite(s)):
        raise ValueError('scores must be finite and match labels')
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None
    r = _average_ranks(s)
    return float((r[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


# ====================================================================== structure and coupling
def structural_matrix(n: int, src: Any, dst: Any, weight: Any, *, absolute: bool = True,
                      symmetric: bool = False) -> np.ndarray:
    """Dense ``(n, n)`` structural matrix ``W[src, dst]`` (duplicate edges summed, diagonal zeroed).
    ``absolute`` takes ``|w|``; ``symmetric`` returns ``W + W.T``."""
    n = _check_int('n', n, 1, 100_000)
    s = np.asarray(src)
    d = np.asarray(dst)
    w = np.asarray(weight, dtype=np.float64)
    if s.ndim != 1 or s.shape != d.shape or s.shape != w.shape or (s.size and (s.dtype.kind not in 'iu' or d.dtype.kind not in 'iu')):
        raise ValueError('src, dst and weight must be 1-D arrays of equal length (integer endpoints)')
    if s.size and (s.min() < 0 or s.max() >= n or d.min() < 0 or d.max() >= n):
        raise ValueError('edge endpoint out of range')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must be finite')
    W = np.zeros((n, n))
    np.add.at(W, (s.astype(np.int64), d.astype(np.int64)), np.abs(w) if absolute else w)
    np.fill_diagonal(W, 0.0)
    return W + W.T if symmetric else W


def weighted_communicability(S: Any) -> np.ndarray:
    """Weighted communicability ``exp(D^-1/2 S D^-1/2)`` (Crofts & Higham, 2009, eq. 2.2) of a symmetric
    non-negative matrix, ``D = diag(sum_k S_ik)``; isolated nodes keep their (zero) rows."""
    S = _square('S', S)
    if not np.allclose(S, S.T, rtol=1e-9, atol=1e-12) or np.any(S < 0.0):
        raise ValueError('S must be symmetric and non-negative')
    S = 0.5 * (S + S.T)
    d = S.sum(axis=1)
    inv = np.where(d > _EPS, 1.0 / np.sqrt(np.where(d > _EPS, d, 1.0)), 0.0)
    N = inv[:, None] * S * inv[None, :]
    vals, vecs = np.linalg.eigh(N)
    return (vecs * np.exp(vals)[None, :]) @ vecs.T


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    if a.size < 3 or np.ptp(a) <= 0.0 or np.ptp(b) <= 0.0:
        return None
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float(np.clip((ra * rb).sum() / den, -1.0, 1.0)) if den > 0 else None


def structure_function_coupling(structure: Any, fc: Any) -> dict[str, Any]:
    """Spearman coupling between symmetric structural weights and FC over node pairs ``i < j``.

    ``structure`` is any non-negative ``(n, n)`` matrix (it is symmetrised as ``(S + S.T) / 2``). Reports the
    coupling over all pairs and over structurally connected pairs, against signed and absolute FC, and the
    same for weighted communicability (an analytic predictor that also credits indirect walks)."""
    S = _square('structure', structure)
    F = _square('fc', fc, S.shape[0])
    if np.any(S < 0.0):
        raise ValueError('structure must be non-negative')
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    s, f = S[iu], F[iu]
    conn = s > 0.0
    G = weighted_communicability(S)
    g = G[iu]
    return {'pairs': int(s.size), 'connected_pairs': int(conn.sum()),
            'sc_fc_spearman': _spearman(s, f), 'sc_absfc_spearman': _spearman(s, np.abs(f)),
            'sc_fc_spearman_connected': _spearman(s[conn], f[conn]) if conn.sum() >= 3 else None,
            'communicability_fc_spearman': _spearman(g, f),
            'communicability_absfc_spearman': _spearman(g, np.abs(f)),
            'mean_fc_connected': float(f[conn].mean()) if conn.any() else None,
            'mean_fc_unconnected': float(f[~conn].mean()) if (~conn).any() else None}


def modularity(W: Any, labels: Any) -> float:
    """Newman-Girvan modularity ``Q = (1/2m) sum_ij (W_ij - k_i k_j / 2m) [c_i == c_j]`` of a symmetric
    non-negative weight matrix (0 for an empty graph)."""
    A = _square('W', W)
    if np.any(A < 0.0):
        raise ValueError('W must be non-negative')
    A = 0.5 * (A + A.T)
    lab = np.asarray(labels)
    if lab.shape != (A.shape[0],):
        raise ValueError('labels must have one entry per node')
    two_m = float(A.sum())
    if two_m <= _EPS:
        return 0.0
    k = A.sum(axis=1)
    same = lab[:, None] == lab[None, :]
    return float(((A - np.outer(k, k) / two_m) * same).sum() / two_m)


def fc_communities(fc: Any, *, max_communities: int = 8) -> dict[str, Any]:
    """Leading-eigenvector communities (Newman, 2006) of the positive part of an FC matrix.

    Recursive bisection: a group g splits by the signs of the leading eigenvector of its generalised modularity
    matrix ``B^(g)_ij = B_ij - delta_ij sum_{l in g} B_il`` while the leading eigenvalue and the gain
    ``s^T B^(g) s / 4m`` are positive and fewer than ``max_communities`` groups exist. No vertex-moving
    refinement, so the reported modularity is a proxy (a lower bound on the maximum)."""
    F = _square('fc', fc)
    max_communities = _check_int('max_communities', max_communities, 1, 4096)
    W = np.clip(0.5 * (F + F.T), 0.0, None)
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    two_m = float(W.sum())
    labels = np.zeros(n, dtype=np.int64)
    if two_m <= _EPS or n < 2:
        return {'labels': labels, 'communities': 1, 'modularity': 0.0}
    k = W.sum(axis=1)
    B = W - np.outer(k, k) / two_m
    pending, final = [np.arange(n)], []
    while pending:
        g = pending.pop(0)
        if g.size < 2 or len(final) + len(pending) + 1 >= max_communities:
            final.append(g)
            continue
        Bg = B[np.ix_(g, g)].copy()
        Bg[np.diag_indices(g.size)] -= Bg.sum(axis=1)
        vals, vecs = np.linalg.eigh(Bg)
        lead = vecs[:, -1]
        s = np.where(lead >= 0.0, 1.0, -1.0)
        gain = float(s @ Bg @ s) / (2.0 * two_m)
        tol = 1e-10 * (1.0 + float(np.abs(vals).max()))
        if vals[-1] <= tol or gain <= 1e-12 or np.all(s == s[0]):
            final.append(g)
            continue
        pending += [g[s > 0], g[s < 0]]
    final.sort(key=lambda grp: int(grp.min()))
    for c, grp in enumerate(final):
        labels[grp] = c
    return {'labels': labels, 'communities': len(final), 'modularity': modularity(W, labels)}


# ====================================================================== live observer
_METRIC_INT = ('computed_at', 'rows', 'constant_units')
_METRIC_OPT_INT = ('fc_communities', 'dfc_window', 'dfc_windows', 'dfc_states', 'dfc_transitions', 'ec_significant')
_METRIC_OPT_FLOAT = ('mean_abs_fc', 'fc_modularity', 'dfc_silhouette', 'dfc_mean_dwell_windows',
                     'dfc_mean_dwell_steps', 'ec_density')
_METRIC_KEYS = _METRIC_INT + _METRIC_OPT_INT + _METRIC_OPT_FLOAT + ('ec_top_edges',)
_EDGE_KEYS = ('source', 'target', 'weight', 'z', 'significant')


class ConnectivityObserver:
    """Read-only live observer: FC, dynamic FC states and ridge-VAR(1) EC over a bounded activity history.

    * A deterministic subset of at most ``max_units`` observed units (seeded) is kept in a ring buffer of
      ``history`` rows. ``observe`` copies the values; it never writes to its input.
    * Metrics are recomputed at most every ``max(8, window // 4)`` observations, once ``window`` rows exist:
      static FC and EC on the last ``window`` rows, dynamic FC on the whole buffer (window ``max(8, window // 4)``,
      step half of it). EC uses ``ec_null_draws`` per-target block-permutation draws seeded by
      ``(seed, observed steps)``, so results do not depend on how often ``summary`` is called.
    * ``summary()`` returns the cached metrics: mean |FC|, the FC modularity proxy and community count, the
      number of dFC states with silhouette and mean dwell, EC density with the top-5 directed edges by |z|, and
      the observed step count. Unit indices refer to positions in the observed vector.
    * ``window`` defaults to 128; the FrontierSpecV1 name ``fc_window`` is accepted as the same parameter
      (clipped to ``history``), and the spec parameters of other observers are accepted and ignored."""

    module_id = MODULE_ID

    def __init__(self, n_units: int, *, seed: int = 7, history: int = 256, window: int | None = None,
                 max_units: int = 32, k_max: int = 4, ec_null_draws: int = 31, ridge: float = 0.05,
                 alpha: float = 0.05, min_silhouette: float = 0.25, **params: Any) -> None:
        unknown = sorted(set(params) - _SPEC_OBSERVER_PARAMS)
        if unknown:
            raise ValueError(f'unknown ConnectivityObserver parameters: {unknown}')
        self.n_units = _check_int('n_units', n_units, 2, 1_000_000)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.history = _check_int('history', history, 16, 8192)
        fc_window = params.get('fc_window')            # FrontierSpecV1 name of the static FC / EC window
        if fc_window is not None:
            fc_window = min(_check_int('fc_window', fc_window, 8, 1_000_000), self.history)
            if window is not None and window != fc_window:
                raise ValueError('window and fc_window disagree')
        if window is None:
            window = 128 if fc_window is None else fc_window
        self.window = _check_int('window', window, 8, self.history)
        self.max_units = _check_int('max_units', max_units, 2, 256)
        self.k_max = _check_int('k_max', k_max, 2, 8)
        self.ec_null_draws = _check_int('ec_null_draws', ec_null_draws, 2, 999)
        self.ridge = _check_float('ridge', ridge, 1e-6, 1e3)
        self.alpha = _check_float('alpha', alpha, 1e-6, 0.5)
        self.min_silhouette = _check_float('min_silhouette', min_silhouette, 0.0, 1.0)
        self.recompute_every = max(8, self.window // 4)
        self.dfc_window = max(8, self.window // 4)
        m = min(self.n_units, self.max_units)
        if m == self.n_units:
            units = np.arange(m, dtype=np.int64)
        else:
            units = np.sort(np.random.default_rng(self.seed + _OBS_UNITS_OFFSET).choice(self.n_units, m, replace=False))
        self._units = units.astype(np.int64)
        self.reset()

    @classmethod
    def from_spec(cls, n_units: int, spec: Any) -> 'ConnectivityObserver':
        """Build from a ``FrontierSpecV1`` (``seed``, ``history``, ``fc_window`` clipped to ``history``)."""
        from .spec import FrontierSpecV1
        if not isinstance(spec, FrontierSpecV1):
            raise ValueError('spec must be a FrontierSpecV1')
        return cls(n_units, seed=int(spec.seed), history=int(spec.history),
                   window=min(int(spec.fc_window), int(spec.history)))

    def reset(self) -> None:
        self._buf = np.zeros((self.history, self._units.size))
        self._steps = 0
        self._last = -1
        self._metrics: dict[str, Any] | None = None

    @property
    def observed_steps(self) -> int:
        return self._steps

    @property
    def units(self) -> np.ndarray:
        return self._units.copy()

    def _rows(self) -> np.ndarray:
        count = min(self._steps, self.history)
        if self._steps <= self.history:
            return self._buf[:count].copy()
        pos = self._steps % self.history
        return np.concatenate([self._buf[pos:], self._buf[:pos]])

    def observe(self, x: np.ndarray) -> None:
        try:
            v = np.array(x, dtype=np.float64)            # private copy: the caller's array is never touched
        except (TypeError, ValueError) as exc:
            raise ValueError(f'observation must be a finite 1-D array of length {self.n_units}') from exc
        if v.shape != (self.n_units,) or not np.all(np.isfinite(v)):
            raise ValueError(f'observation must be a finite 1-D array of length {self.n_units}')
        self._buf[self._steps % self.history] = v[self._units]
        self._steps += 1
        if self._steps >= self.window and (self._last < 0 or self._steps - self._last >= self.recompute_every):
            self._metrics = self._compute()
            self._last = self._steps

    def _compute(self) -> dict[str, Any]:
        rows = self._rows()
        static = rows[-self.window:]
        fc = _pearson(static)
        ok = np.diag(fc) > 0.5
        idx = np.flatnonzero(ok)
        m: dict[str, Any] = {k: None for k in _METRIC_KEYS}
        m.update(computed_at=int(self._steps), rows=int(rows.shape[0]), constant_units=int((~ok).sum()),
                 ec_top_edges=[])
        if idx.size < 2:
            return m
        sub = fc[np.ix_(idx, idx)]
        iu = np.triu_indices(idx.size, 1)
        m['mean_abs_fc'] = float(np.abs(sub[iu]).mean())
        com = fc_communities(sub)
        m['fc_modularity'] = float(com['modularity'])
        m['fc_communities'] = int(com['communities'])
        data = rows[:, idx]
        w = min(self.dfc_window, data.shape[0])
        step = max(1, w // 2)
        n_win = (data.shape[0] - w) // step + 1
        if n_win >= 3:
            d = dynamic_fc_states(data, window=w, step=step, k_max=self.k_max, min_silhouette=self.min_silhouette,
                                  seed=self.seed, n_init=4)
            m.update(dfc_window=int(w), dfc_windows=int(d['windows']), dfc_states=int(d['n_states']),
                     dfc_silhouette=None if d['best_silhouette'] is None else float(d['best_silhouette']),
                     dfc_mean_dwell_windows=float(d['mean_dwell_windows']),
                     dfc_mean_dwell_steps=float(d['mean_dwell_samples']), dfc_transitions=int(d['transitions']))
        Z, _ok = _zscore_columns(static[:, idx])
        P = Z[:-1] - Z[:-1].mean(axis=0)
        F = Z[1:] - Z[1:].mean(axis=0)
        rng = np.random.default_rng([self.seed + _OBS_PERM_OFFSET, self._steps])
        ec = _ec_from_design(P, F, self.ridge * P.shape[0], self.ec_null_draws, _default_block(P.shape[0]),
                             self.alpha, rng)
        m['ec_density'] = float(ec['density'])
        m['ec_significant'] = int(ec['significant'])
        off = ~np.eye(idx.size, dtype=bool)
        tgt, src = np.nonzero(off)
        mag = np.abs(ec['z'][tgt, src])
        order = np.lexsort((src, tgt, -mag))[:_TOP_EDGES]
        units = self._units[idx]
        m['ec_top_edges'] = [{'source': int(units[src[e]]), 'target': int(units[tgt[e]]),
                              'weight': float(ec['A'][tgt[e], src[e]]), 'z': float(ec['z'][tgt[e], src[e]]),
                              'significant': bool(ec['mask'][tgt[e], src[e]])} for e in order]
        return m

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {'module': MODULE_ID, 'evidence_tier': 'synthetic_abstraction',
                               'n_units': self.n_units, 'analysed_units': int(self._units.size),
                               'unit_sample': self._units.tolist(), 'history': self.history, 'window': self.window,
                               'recompute_every': self.recompute_every, 'observed_steps': self._steps,
                               'buffered': min(self._steps, self.history), 'ready': self._metrics is not None}
        metrics = self._metrics if self._metrics is not None else {k: None for k in _METRIC_KEYS}
        for key in _METRIC_KEYS:
            value = metrics[key]
            out[key] = [dict(e) for e in value] if key == 'ec_top_edges' and value is not None else value
        if out['ec_top_edges'] is None:
            out['ec_top_edges'] = []
        return out

    def _config(self) -> dict[str, Any]:
        return {'n_units': self.n_units, 'seed': self.seed, 'history': self.history, 'window': self.window,
                'max_units': self.max_units, 'k_max': self.k_max, 'ec_null_draws': self.ec_null_draws,
                'ridge': self.ridge, 'alpha': self.alpha, 'min_silhouette': self.min_silhouette}

    def export_state(self) -> dict[str, Any]:
        metrics = None if self._metrics is None else {
            k: ([dict(e) for e in v] if k == 'ec_top_edges' else v) for k, v in self._metrics.items()}
        return {'module': MODULE_ID, 'schema': OBSERVER_STATE_SCHEMA, **self._config(), 'units': self._units.tolist(),
                'buffer': self._buf.tolist(), 'steps': self._steps, 'last': self._last, 'metrics': metrics}

    def _validate_metrics(self, metrics: Any, steps: int, last: int) -> dict[str, Any] | None:
        if last < 0:
            if metrics is not None:
                raise ValueError('connectivity observer state has metrics but no recompute')
            return None
        if not isinstance(metrics, Mapping) or set(metrics) != set(_METRIC_KEYS):
            raise ValueError('connectivity observer metrics have the wrong keys')
        out: dict[str, Any] = {}
        for key in _METRIC_INT:
            out[key] = _check_int(f'metrics.{key}', metrics[key], 0, 2**62)
        for key in _METRIC_OPT_INT:
            out[key] = None if metrics[key] is None else _check_int(f'metrics.{key}', metrics[key], 0, 2**62)
        for key in _METRIC_OPT_FLOAT:
            out[key] = None if metrics[key] is None else _check_float(f'metrics.{key}', metrics[key], -1.0, 1e12)
        if out['computed_at'] != last or out['rows'] != min(last, self.history) or out['constant_units'] > self._units.size:
            raise ValueError('connectivity observer metrics do not match the recompute step')
        for key in ('mean_abs_fc', 'ec_density'):
            if out[key] is not None and not 0.0 <= out[key] <= 1.0:
                raise ValueError(f'metrics.{key} must lie in [0, 1]')
        edges = metrics['ec_top_edges']
        if not isinstance(edges, list) or len(edges) > _TOP_EDGES:
            raise ValueError('metrics.ec_top_edges must be a list of at most 5 edges')
        clean = []
        allowed = set(self._units.tolist())
        for e in edges:
            if not isinstance(e, Mapping) or set(e) != set(_EDGE_KEYS):
                raise ValueError('ec_top_edges entries have the wrong keys')
            s = _check_int('edge.source', e['source'], 0, self.n_units - 1)
            t = _check_int('edge.target', e['target'], 0, self.n_units - 1)
            if s == t or s not in allowed or t not in allowed or type(e['significant']) is not bool:
                raise ValueError('ec_top_edges entry is invalid')
            clean.append({'source': s, 'target': t, 'weight': _check_float('edge.weight', e['weight'], -1e12, 1e12),
                          'z': _check_float('edge.z', e['z'], -1e12, 1e12), 'significant': e['significant']})
        out['ec_top_edges'] = clean
        return out

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate the whole state first; nothing changes unless every check passes."""
        if not isinstance(state, Mapping) or state.get('module') != MODULE_ID or state.get('schema') != OBSERVER_STATE_SCHEMA:
            raise ValueError('state is not a connectivity observer state')
        for key, value in self._config().items():
            if state.get(key) != value:
                raise ValueError(f'connectivity observer state does not match this observer ({key})')
        if state.get('units') != self._units.tolist():
            raise ValueError('connectivity observer state units do not match this observer')
        try:
            buf = np.array(state.get('buffer'), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError('buffer must be numeric') from exc
        if buf.shape != self._buf.shape or not np.all(np.isfinite(buf)):
            raise ValueError('connectivity observer buffer has the wrong shape or non-finite values')
        steps = _check_int('steps', state.get('steps'), 0, 2**62)
        last = state.get('last')
        if last != -1:
            last = _check_int('last', last, self.window, steps)
        if steps >= self.window:
            if last == -1 or steps - last >= self.recompute_every:
                raise ValueError('connectivity observer state skipped a scheduled recompute')
            if (last - self.window) % self.recompute_every:
                raise ValueError('connectivity observer state recomputed off the schedule')
        elif last != -1:
            raise ValueError('connectivity observer state recomputed before the window filled')
        if steps < self.history and np.any(buf[steps:] != 0.0):
            raise ValueError('connectivity observer buffer has values beyond the observed steps')
        metrics = self._validate_metrics(state.get('metrics'), steps, int(last))
        self._buf, self._steps, self._last, self._metrics = buf, steps, int(last), metrics


# ====================================================================== lab experiment
_DEFAULTS: dict[str, Any] = {
    'var_units': 8, 'var_steps': 3000, 'var_edge_prob': 0.15, 'var_spectral_radius': 0.8,
    'ec_ridge': 1e-2, 'ec_null_draws': 199, 'ec_alpha': 0.05,
    'chain_steps': 4000, 'chain_coupling': 0.8, 'pc_ridge': 1e-3,
    'switch_units': 6, 'switch_block': 150, 'switch_blocks': 8, 'switch_corr': 0.8,
    'dfc_window': 40, 'dfc_step': 8, 'dfc_k_max': 4, 'min_silhouette': 0.25,
    'topologies': ('clustered', 'ring_lattice', 'hub_skewed', 'multi_region'), 'neurons': 32, 'nw_steps': 600,
    'nw_burn_in': 50, 'mechanisms': (), 'nw_null_draws': 99, 'nw_pc_ridge': 0.1,
}
_QUICK: dict[str, Any] = {'var_steps': 1500, 'ec_null_draws': 99, 'chain_steps': 2000, 'switch_blocks': 6,
                          'topologies': ('clustered', 'ring_lattice', 'hub_skewed'), 'neurons': 24, 'nw_steps': 240,
                          'nw_burn_in': 40, 'nw_null_draws': 49}


def _validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    from ..neuroweave.spec import MECHANISM_IDS, TOPOLOGIES
    c = dict(cfg)
    c['var_units'] = _check_int('var_units', c['var_units'], 3, 64)
    c['var_steps'] = _check_int('var_steps', c['var_steps'], 100, 1_000_000)
    c['var_edge_prob'] = _check_float('var_edge_prob', c['var_edge_prob'], 0.01, 0.9)
    c['var_spectral_radius'] = _check_float('var_spectral_radius', c['var_spectral_radius'], 0.05, 0.99)
    c['ec_ridge'] = _check_float('ec_ridge', c['ec_ridge'], 0.0, 1e3)
    c['ec_null_draws'] = _check_int('ec_null_draws', c['ec_null_draws'], 2, 100_000)
    c['ec_alpha'] = _check_float('ec_alpha', c['ec_alpha'], 1e-6, 0.5)
    c['chain_steps'] = _check_int('chain_steps', c['chain_steps'], 100, 10_000_000)
    c['chain_coupling'] = _check_float('chain_coupling', c['chain_coupling'], 0.1, 5.0)
    c['pc_ridge'] = _check_float('pc_ridge', c['pc_ridge'], 0.0, 1.0)
    c['switch_units'] = _check_int('switch_units', c['switch_units'], 4, 64)
    c['switch_block'] = _check_int('switch_block', c['switch_block'], 20, 1_000_000)
    c['switch_blocks'] = _check_int('switch_blocks', c['switch_blocks'], 4, 10_000)
    c['switch_corr'] = _check_float('switch_corr', c['switch_corr'], 0.1, 0.95)
    c['dfc_window'] = _check_int('dfc_window', c['dfc_window'], 8, c['switch_block'])
    c['dfc_step'] = _check_int('dfc_step', c['dfc_step'], 1, c['dfc_window'])
    c['dfc_k_max'] = _check_int('dfc_k_max', c['dfc_k_max'], 2, 8)
    c['min_silhouette'] = _check_float('min_silhouette', c['min_silhouette'], 0.0, 1.0)
    try:
        topologies = tuple(c['topologies'])
        mechanisms = tuple(c['mechanisms'])
    except TypeError as exc:
        raise ValueError('topologies and mechanisms must be sequences of names') from exc
    if not topologies or len(set(topologies)) != len(topologies) or any(t not in TOPOLOGIES for t in topologies):
        raise ValueError(f'topologies must be distinct names from {TOPOLOGIES}')
    if any(m not in MECHANISM_IDS for m in mechanisms):
        raise ValueError('mechanisms must be NEUROWEAVE mechanism ids')
    c['topologies'], c['mechanisms'] = list(topologies), list(mechanisms)
    c['neurons'] = _check_int('neurons', c['neurons'], 8, 512)
    c['nw_steps'] = _check_int('nw_steps', c['nw_steps'], 40, 100_000)
    c['nw_burn_in'] = _check_int('nw_burn_in', c['nw_burn_in'], 0, c['nw_steps'] - 30)
    if c['nw_steps'] - c['nw_burn_in'] < c['dfc_window']:
        raise ValueError('nw_steps - nw_burn_in must be at least dfc_window (the NEUROWEAVE dFC needs one window)')
    c['nw_null_draws'] = _check_int('nw_null_draws', c['nw_null_draws'], 2, 100_000)
    c['nw_pc_ridge'] = _check_float('nw_pc_ridge', c['nw_pc_ridge'], 1e-6, 10.0)
    return c


def _simulate_sparse_var(rng: np.random.Generator, n: int, steps: int, edge_prob: float,
                         radius: float) -> tuple[np.ndarray, np.ndarray]:
    off = ~np.eye(n, dtype=bool)
    support = (rng.random((n, n)) < edge_prob) & off
    if not support.any():
        i, j = np.argwhere(off)[int(rng.integers(n * (n - 1)))]
        support[i, j] = True
    A = np.zeros((n, n))
    k = int(support.sum())
    A[support] = rng.uniform(0.25, 0.5, k) * rng.choice(np.array([-1.0, 1.0]), k)
    A[np.diag_indices(n)] = rng.uniform(0.2, 0.5, n)
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho > radius:
        A *= radius / rho
    burn = 100
    noise = rng.standard_normal((steps + burn, n))
    X = np.empty((steps + burn, n))
    x = np.zeros(n)
    for t in range(steps + burn):
        x = A @ x + noise[t]
        X[t] = x
    return A, X[burn:]


def _validate_var(seed: int, c: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(seed + _VAR_OFFSET)
    A, X = _simulate_sparse_var(rng, c['var_units'], c['var_steps'], c['var_edge_prob'], c['var_spectral_radius'])
    ec = effective_connectivity(X, ridge=c['ec_ridge'], null_draws=c['ec_null_draws'], alpha=c['ec_alpha'],
                                seed=seed, standardize=False)
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    truth = (A != 0.0)[off]
    est = ec['A'][off]
    mask = ec['mask'][off]
    tp, fp = int((mask & truth).sum()), int((mask & ~truth).sum())
    return {'seed': seed, 'units': n, 'steps': c['var_steps'], 'true_edges': int(truth.sum()),
            'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(A)))),
            'support_auroc': auroc(np.abs(est), truth),
            'sign_accuracy': float((np.sign(est[truth]) == np.sign(A[off][truth])).mean()),
            'max_abs_coefficient_error': float(np.max(np.abs(ec['A'] - A))),
            'mask_true_positives': tp, 'mask_false_positives': fp,
            'mask_precision': float(tp / (tp + fp)) if tp + fp else None,
            'mask_recall': float(tp / max(1, int(truth.sum()))), 'ec_density': ec['density'], 'block': ec['block']}


def _validate_chain(seed: int, c: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(seed + _CHAIN_OFFSET)
    T, a = c['chain_steps'], c['chain_coupling']
    x = rng.standard_normal(T)
    y = a * x + rng.standard_normal(T)
    z = a * y + rng.standard_normal(T)
    X = np.column_stack([x, y, z])
    fc = pearson_fc(X)
    pc = partial_correlation(X, ridge=c['pc_ridge'])
    exact = {'xy': a / (1.0 + a * a), 'yz': a / math.sqrt(1.0 + a * a), 'xz': 0.0,
             'fc_xz': a * a / math.sqrt(a * a * (a * a + 1.0) + 1.0)}
    tol = 4.0 / math.sqrt(T)
    est = {'xy': float(pc[0, 1]), 'yz': float(pc[1, 2]), 'xz': float(pc[0, 2])}
    return {'seed': seed, 'steps': T, 'tolerance': tol, 'partial': est, 'exact_partial': exact,
            'fc_xz': float(fc[0, 2]), 'spearman_xz': float(spearman_fc(X)[0, 2]),
            'max_partial_error': float(max(abs(est[k] - exact[k]) for k in ('xy', 'yz', 'xz')))}


def _two_state_cov(n: int, r: float) -> tuple[np.ndarray, np.ndarray]:
    h = n // 2
    A, B = np.eye(n), np.eye(n)
    A[:h, :h] = r
    B[h:, h:] = r
    np.fill_diagonal(A, 1.0)
    np.fill_diagonal(B, 1.0)
    return A, B


def _validate_switching(seed: int, c: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(seed + _SWITCH_OFFSET)
    n, L, nb = c['switch_units'], c['switch_block'], c['switch_blocks']
    CA, CB = _two_state_cov(n, c['switch_corr'])
    chol = [np.linalg.cholesky(CA), np.linalg.cholesky(CB)]
    X = np.vstack([rng.standard_normal((L, n)) @ chol[b % 2].T for b in range(nb)])
    w, step = c['dfc_window'], c['dfc_step']
    kw = dict(window=w, step=step, k_max=c['dfc_k_max'], min_silhouette=c['min_silhouette'], seed=seed)
    d = dynamic_fc_states(X, **kw)
    starts = d['starts']
    pure = (starts // L) == ((starts + w - 1) // L)
    truth = (starts // L) % 2
    accuracy = None
    if d['n_states'] == 2 and pure.any():
        lab = d['labels'][pure]
        accuracy = float(max((lab == truth[pure]).mean(), (lab != truth[pure]).mean()))
    rng2 = np.random.default_rng(seed + _STATIONARY_OFFSET)
    Xs = rng2.standard_normal((L * nb, n)) @ np.linalg.cholesky(0.5 * (CA + CB)).T
    ds = dynamic_fc_states(Xs, **kw)
    return {'seed': seed, 'units': n, 'block': L, 'blocks': nb, 'windows': d['windows'], 'n_states': d['n_states'],
            'silhouette': d['silhouette'], 'pure_windows': int(pure.sum()), 'pure_window_accuracy': accuracy,
            'mean_dwell_samples': d['mean_dwell_samples'], 'expected_dwell_samples_upper': float(L),
            'stationary_control_states': ds['n_states'], 'stationary_control_silhouette': ds['silhouette']}


def _neuroweave_run(topology: str, seed: int, c: Mapping[str, Any]) -> dict[str, Any]:
    from ..neuroweave.engine import NeuroWeaveEngine
    from ..neuroweave.experiments import engine_topology, make_spec
    from ..neuroweave.tasks import make_stream
    from ..neuroweave.topology import topology_metrics
    steps = c['nw_steps']
    engine = NeuroWeaveEngine(make_spec(c['mechanisms'], seed=seed, neurons=c['neurons'], topology=topology,
                                        steps=steps))
    stream = make_stream(seed, steps)
    outs = []
    for i in range(steps):          # frozen dynamics: the structure that generates the activity stays fixed
        outs.append(np.array(engine.step(stream.inputs[i], stream.sensory[i], reward=0.0, learn=False)['output']))
    X = np.stack(outs)[c['nw_burn_in']:]
    topo = engine_topology(engine)
    n = int(topo.n)
    W = structural_matrix(n, topo.src, topo.dst, topo.weight, absolute=False)
    S = np.abs(W) + np.abs(W).T
    fc = pearson_fc(X)
    pc = partial_correlation(X, ridge=c['nw_pc_ridge'])
    ok = np.diag(fc) > 0.5
    iu = np.triu_indices(n, 1)
    both = ok[iu[0]] & ok[iu[1]]
    coupling = structure_function_coupling(S, fc)
    ec = effective_connectivity(X, ridge=c['ec_ridge'], null_draws=c['nw_null_draws'], alpha=c['ec_alpha'],
                                seed=seed)
    off = ~np.eye(n, dtype=bool)
    present = (W.T != 0.0)                     # present[target, source]: a synapse source -> target exists
    sig = ec['mask'] & off
    sign_ok = np.sign(ec['A'][sig & present]) == np.sign(W.T[sig & present])
    d = dynamic_fc_states(X, window=c['dfc_window'], step=c['dfc_step'], k_max=c['dfc_k_max'],
                          min_silhouette=c['min_silhouette'], seed=seed)
    com = fc_communities(fc[np.ix_(ok, ok)]) if ok.sum() >= 2 else {'modularity': 0.0, 'communities': 1}
    tm = topology_metrics(topo)
    return {
        'seed': seed, 'neurons': n, 'samples': int(X.shape[0]), 'synapses': int(topo.edges),
        'constant_units': int((~ok).sum()),
        'structure': {k: tm[k] for k in ('density', 'clustering_mean', 'modularity', 'mean_shortest_path',
                                         'global_efficiency', 'inter_region_edge_fraction')},
        'mean_abs_fc': float(np.abs(fc[iu][both]).mean()) if both.any() else None,
        'mean_abs_partial': float(np.abs(pc[iu][both]).mean()) if both.any() else None,
        'fc_modularity_proxy': float(com['modularity']), 'fc_communities': int(com['communities']),
        'coupling': coupling,
        'ec': {'density': ec['density'], 'significant': ec['significant'], 'block': ec['block'],
               'structure_auroc': auroc(np.abs(ec['A'][off]), present[off]),
               'significant_on_synapses_fraction': float((sig & present).sum() / sig.sum()) if sig.any() else None,
               'sign_agreement_on_synapses': float(sign_ok.mean()) if sign_ok.size else None},
        'dfc': {'n_states': d['n_states'], 'best_silhouette': d['best_silhouette'], 'windows': d['windows'],
                'mean_dwell_samples': d['mean_dwell_samples']},
    }


def _finite_tree(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return all(_finite_tree(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_finite_tree(v) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return math.isfinite(float(obj))
    return True


def _mean_of(values: Sequence[Any]) -> float | None:
    v = [float(x) for x in values if x is not None]
    return float(np.mean(v)) if v else None


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Known-answer validation of the connectivity estimators, then a descriptive NEUROWEAVE report.

    Validation (per seed, checks must hold for every seed): (1) a sparse stable VAR(1) with known ``A``: EC support
    AUROC > 0.9, sign accuracy on true edges >= 0.9 and pooled mask precision >= 0.8; (2) a Gaussian chain
    X -> Y -> Z: partial correlations within ``4 / sqrt(T)`` of their exact values (X-Z exactly 0) while
    FC(X, Z) stays clearly non-zero; (3) a two-state switching process: dFC finds 2 states with pure-window
    accuracy >= 0.9, and a stationary control with the averaged covariance gives 1 state. Then (4) NEUROWEAVE: for
    each topology the frozen engine runs on the task stream and the module reports structure-function coupling,
    communicability coupling, EC/structure agreement and dFC/FC summaries. Part (4) is descriptive: no verdict is
    drawn from it beyond finiteness. ``kind='validation'``, ``verdict`` in ``('pass', 'fail')``."""
    from . import harness
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = _check_seeds(seeds)
    cfg = dict(_DEFAULTS)
    if quick:
        cfg.update(_QUICK)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_experiment overrides: {unknown}')
    cfg.update(overrides)
    c = _validate_config(cfg)
    t0 = time.perf_counter()
    var = [_validate_var(s, c) for s in seeds]
    chain = [_validate_chain(s, c) for s in seeds]
    switch = [_validate_switching(s, c) for s in seeds]
    t_val = time.perf_counter() - t0
    t1 = time.perf_counter()
    nw: dict[str, Any] = {}
    for topo in c['topologies']:
        runs = [_neuroweave_run(topo, s, c) for s in seeds]
        nw[topo] = {'per_seed': runs, 'mean': {
            'sc_fc_spearman': _mean_of([r['coupling']['sc_fc_spearman'] for r in runs]),
            'sc_fc_spearman_connected': _mean_of([r['coupling']['sc_fc_spearman_connected'] for r in runs]),
            'communicability_fc_spearman': _mean_of([r['coupling']['communicability_fc_spearman'] for r in runs]),
            'communicability_absfc_spearman': _mean_of([r['coupling']['communicability_absfc_spearman'] for r in runs]),
            'mean_abs_fc': _mean_of([r['mean_abs_fc'] for r in runs]),
            'ec_structure_auroc': _mean_of([r['ec']['structure_auroc'] for r in runs]),
            'ec_density': _mean_of([r['ec']['density'] for r in runs]),
            'dfc_states': _mean_of([r['dfc']['n_states'] for r in runs])}}
    t_nw = time.perf_counter() - t1
    tp = sum(v['mask_true_positives'] for v in var)
    fp = sum(v['mask_false_positives'] for v in var)
    pooled_precision = tp / (tp + fp) if tp + fp else 0.0
    checks = {
        'var_support_auroc_above_0.9': all(v['support_auroc'] is not None and v['support_auroc'] > 0.9 for v in var),
        'var_sign_accuracy_at_least_0.9': all(v['sign_accuracy'] >= 0.9 for v in var),
        'var_mask_pooled_precision_at_least_0.8': bool(pooled_precision >= 0.8),
        'chain_partials_match_exact_values': all(ch['max_partial_error'] <= ch['tolerance'] for ch in chain),
        'chain_fc_xz_clearly_nonzero': all(abs(ch['fc_xz']) > 0.3 for ch in chain),
        'switching_two_states_recovered': all(s['n_states'] == 2 for s in switch),
        'switching_pure_window_accuracy_at_least_0.9': all(
            s['pure_window_accuracy'] is not None and s['pure_window_accuracy'] >= 0.9 for s in switch),
        'stationary_control_single_state': all(s['stationary_control_states'] == 1 for s in switch),
        'neuroweave_values_finite': _finite_tree(nw),
    }
    receipt = harness.base_receipt(MODULE_ID, 'validation', {**c, 'quick': quick, 'seeds': list(seeds)})
    receipt['validations'] = {'var_recovery': var, 'var_pooled_mask_precision': pooled_precision,
                              'chain_partial_correlation': chain, 'dfc_switching': switch}
    receipt['neuroweave'] = {'per_topology': nw, 'mechanisms': list(c['mechanisms']),
                             'note': 'descriptive only: values are reported, no verdict is drawn from them'}
    receipt['checks'] = checks
    receipt['verdict'] = 'pass' if all(checks.values()) else 'fail'
    receipt['verdict_basis'] = 'all_known_answer_checks' if receipt['verdict'] == 'pass' else 'failed_checks: ' + ', '.join(
        k for k, v in checks.items() if not v)
    nets = len(c['topologies']) * len(seeds)
    receipt['estimated_cost'] = {
        'engine_steps': int(nets * c['nw_steps']),
        'ec_null_ridge_solves': int(len(seeds) * c['ec_null_draws'] * c['var_units']
                                    + nets * c['nw_null_draws'] * c['neurons']),
        'fc_windows': int(sum(s['windows'] for s in switch) * 2)}
    receipt['measured_wallclock_s'] = {'validation': t_val, 'neuroweave': t_nw, 'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'Synthetic validation of FC, partial correlation, dFC states and ridge VAR(1) EC on processes with known '
        'answers, then a descriptive structure-function report on frozen NEUROWEAVE brains (the synapse table '
        'that generated the activity). EC is lagged predictive influence (Granger-style), not an interventional '
        'effect; the dFC state count uses a silhouette heuristic (0.25 by default); the FC modularity is a '
        'leading-eigenvector proxy. Nothing here is imaging data or evidence of biological fidelity.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
