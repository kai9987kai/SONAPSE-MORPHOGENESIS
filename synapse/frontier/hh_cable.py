"""Hodgkin-Huxley cable reference (frontier module ``hh_cable``).

Inspired by NeuroPulse-CableLab (kai9987kai), a browser multi-compartment Hodgkin-Huxley workbench, and built
on the membrane equations of A. L. Hodgkin and A. F. Huxley, "A quantitative description of membrane current
and its application to conduction and excitation in nerve", J. Physiol. 117(4):500-544, August 1952,
doi:10.1113/jphysiol.1952.sp004764. This is a synthetic computational reference: it integrates the 1952
equations on a discretised, uniform, sealed-end cable. It does not reproduce the paper's numerical procedure
(HH solved a travelling-wave ODE by shooting) and it is not a validated neuron simulator. Full, verified
references: ``docs/frontier/modules/hh_cable.md``.

Model (``v`` in mV, rest near -65 mV; HH's own variable is the displacement ``V = -(v + 65)``, depolarisation
negative)::

    Cm dv_i/dt = I_stim,i + g_ax (v_{i-1} - 2 v_i + v_{i+1}) - gNa m^3 h (v - ENa) - gK n^4 (v - EK) - gL (v - EL)
    dx/dt     = phi (alpha_x(v) (1 - x) - beta_x(v) x),          x in {m, h, n},  phi = 3 ** ((T - 6.3) / 10)
    g_ax      = a / (2 Ra dx^2)                                    (HH eqn 28, central differences)

Sealed ends: no axial current leaves the first or the last compartment. The rate functions are HH eqns 12, 13,
20, 21, 23 and 24 (p. 519) rewritten for ``v``; Table 3 (p. 520) gives the constants (ENa = 50, EK = -77 and
EL = -54.387 mV here). The stimulus is a current density (uA/cm^2) on the membrane of compartment 0.

Integrators: ``'rk4'`` (classical Runge-Kutta; ``__init__`` rejects time steps beyond a conservative linear
stability bound) and ``'exp_euler'`` (gates integrated exactly for frozen voltage, as in Rush-Larsen; the voltage
takes an exponential step with frozen conductances and neighbour voltages; first order, stable for any step).

Spike times are upward crossings of 0 mV (re-armed below -30 mV) located inside the step with cubic Hermite
interpolation, so they keep the integrator's order. All randomness-free; nothing here touches global state.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ['MODULE_ID', 'SOURCE', 'EXPERIMENTS', 'HH_REST_MV', 'HH1952_REFERENCE_AXON', 'METHODS', 'HHCable',
           'vtrap', 'hh_rates', 'steady_state_gates', 'temperature_factor', 'neuroweave_fi_curve',
           'neuroweave_threshold', 'adaptation_index', 'average_ranks', 'spearman_rho', 'compare_fi_curves',
           'run_experiment']

MODULE_ID = 'hh_cable'
SOURCE = ('Hodgkin & Huxley, "A quantitative description of membrane current and its application to conduction and '
          'excitation in nerve", J. Physiol. 117(4):500-544, August 1952, doi:10.1113/jphysiol.1952.sp004764; '
          'NeuroPulse-CableLab (kai9987kai)')
METHODS = ('rk4', 'exp_euler')
HH_REST_MV = -65.0                 # rest of the shifted convention (HH chose V_l so the ionic current is 0 at V = 0)
REFERENCE_TEMPERATURE_C = 6.3      # HH rate constants are for 6.3 C and scale with a Q10 of 3 (p. 519)
Q10 = 3.0
# HH 1952, p. 528: K = 10.47 /ms at 18.5 C, a = 238 um, R2 = 35.4 ohm cm, Cm = 1 uF/cm^2 -> 18.8 m/s (observed 21.2).
HH1952_REFERENCE_AXON: dict[str, float] = {'diameter_um': 476.0, 'Ra_ohm_cm': 35.4, 'Cm_uF_cm2': 1.0,
                                           'temperature_c': 18.5, 'computed_velocity_m_s': 18.8,
                                           'observed_velocity_m_s': 21.2}

SPIKE_THRESHOLD_MV = 0.0
REARM_MV = -30.0
_RK4_STABILITY = 2.5               # dt * lambda_max must stay below this (RK4 real-axis limit is ~2.785)
_MAX_STEPS = 2_000_000
_MAX_BATCH = 4096
_MAX_TRACE = 200_000
_FINITE_CHECK_EVERY = 256
_E = math.e

# fused-kernel constants in u = v + 65 (HH's V = -u); rates are stacked as rows (alpha_m, alpha_h, alpha_n,
# beta_m, beta_h, beta_n) so the three gate equations are one vectorised expression
_AH_K, _AH_C = -1.0 / 20.0, math.log(0.07)                                         # alpha_h = 0.07 exp(-u/20)
_BETA_K = np.array([-1.0 / 18.0, -0.1, -1.0 / 80.0])[:, None, None]               # beta_m, beta_h argument, beta_n
_BETA_C = np.array([math.log(4.0), 3.0, math.log(0.125)])[:, None, None]          # 4 e^(-u/18), e^(-(u-30)/10), ...
_TRAP_SHIFT = np.array([25.0, 10.0])[:, None, None]                              # alpha_m: u - 25, alpha_n: u - 10
_TRAP_SCALE = np.array([-0.1, -0.01])[:, None, None]


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


def _vector(name: str, value: Any, *, low: float = -math.inf, high: float = math.inf, min_size: int = 1,
            max_size: int = _MAX_BATCH) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.ndim != 1 or not min_size <= arr.size <= max_size:
        raise ValueError(f'{name} must be a 1-D sequence of {min_size}..{max_size} numbers')
    if not np.all(np.isfinite(arr)) or np.any(arr < low) or np.any(arr > high):
        raise ValueError(f'{name} must be finite and within [{low}, {high}]')
    return arr


# ====================================================================== HH rate functions (6.3 C)
def vtrap(x: Any, y: float) -> np.ndarray:
    """``x / (1 - exp(-x / y))`` evaluated stably; equals ``y + x / 2`` to first order near ``x = 0``."""
    x = np.asarray(x, dtype=np.float64)
    y = float(y)
    den = -np.expm1(-x / y)
    small = np.abs(x) < 1e-7
    with np.errstate(divide='ignore', invalid='ignore'):
        out = x / np.where(small, 1.0, den)
    return np.where(small, y + 0.5 * x, out)


def hh_rates(v: Any) -> tuple[np.ndarray, ...]:
    """``(alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n)`` in 1/ms at 6.3 C for membrane potential ``v`` (mV).

    HH eqns 12, 13, 20, 21, 23, 24 with ``V = -(v + 65)``."""
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError('v must be finite')
    u = v + 65.0
    return (0.1 * vtrap(v + 40.0, 10.0), 4.0 * np.exp(-u / 18.0), 0.07 * np.exp(-u / 20.0),
            1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0)), 0.01 * vtrap(v + 55.0, 10.0), 0.125 * np.exp(-u / 80.0))


def steady_state_gates(v: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(m_inf, h_inf, n_inf)`` at voltage ``v`` (temperature independent: ``phi`` scales alpha and beta alike)."""
    am, bm, ah, bh, an, bn = hh_rates(v)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def temperature_factor(temperature_c: float) -> float:
    """``phi = 3 ** ((T - 6.3) / 10)`` (HH p. 519: rate constants are for 6.3 C and scale with a Q10 of 3)."""
    t = _num('temperature_c', temperature_c, -5.0, 45.0)
    return float(Q10 ** ((t - REFERENCE_TEMPERATURE_C) / 10.0))


def _gate_rate_bound(v_low: float, v_high: float) -> float:
    """max over v in [v_low, v_high] of alpha_x + beta_x for the three gates (6.3 C)."""
    v = np.linspace(v_low, v_high, 2001)
    am, bm, ah, bh, an, bn = hh_rates(v)
    return float(max(np.max(am + bm), np.max(ah + bh), np.max(an + bn)))


def _hermite_crossing(v0: float, v1: float, d0: float, d1: float, dt: float, level: float) -> float:
    """Fraction ``s`` in [0, 1] of the step at which the cubic Hermite interpolant reaches ``level``."""
    if v1 == v0:
        return 1.0
    def p(s: float) -> float:
        s2, s3 = s * s, s * s * s
        return ((2 * s3 - 3 * s2 + 1) * v0 + (s3 - 2 * s2 + s) * dt * d0 + (-2 * s3 + 3 * s2) * v1
                + (s3 - s2) * dt * d1)
    lo, hi = 0.0, 1.0
    if p(lo) >= level:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p(mid) >= level:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _first_success(flags: Sequence[bool]) -> int | None:
    for i, f in enumerate(flags):
        if f:
            return i
    return None


def _monotone(flags: Sequence[bool]) -> bool:
    """True when every evaluation after the first success also succeeded."""
    i = _first_success(flags)
    return i is None or all(flags[i:])


def _batched_bisection(fires: Callable[[np.ndarray], list[bool]], lo: float, hi: float, *, rounds: int, batch: int,
                       integer: bool = False) -> dict[str, Any]:
    """Smallest value in ``[lo, hi]`` for which ``fires`` holds, assuming a monotone response.

    Round 1 evaluates ``batch`` points including both ends (one batched simulation); every further round evaluates
    ``batch`` interior points of the current bracket. ``bracket = [a, b]``: ``a`` is the largest value seen silent and
    ``b`` the smallest seen firing. ``status`` is ``'resolved'``, ``'lower_fires'`` or ``'upper_silent'``;
    ``lower_fires``/``upper_fires`` are the first grid's two ends and ``monotone`` reports whether every evaluated
    grid agreed with the monotonicity assumption."""
    grid = np.linspace(lo, hi, batch)
    if integer:
        grid = np.unique(np.rint(grid)).astype(np.int64)
    ok = fires(grid)
    monotone = _monotone(ok)
    evaluations = int(grid.size)
    ends = {'lower_fires': bool(ok[0]), 'upper_fires': bool(ok[-1])}
    if ok[0]:
        return {'status': 'lower_fires', 'bracket': [grid[0].item(), grid[0].item()], 'monotone': monotone,
                'evaluations': evaluations, **ends}
    i = _first_success(ok)
    if i is None:
        return {'status': 'upper_silent', 'bracket': [grid[-1].item(), None], 'monotone': monotone,
                'evaluations': evaluations, **ends}
    a, b = grid[i - 1].item(), grid[i].item()
    for _ in range(rounds - 1):
        inner = np.linspace(a, b, batch + 2)[1:-1]
        if integer:
            inner = np.unique(np.rint(inner)).astype(np.int64)
            inner = inner[(inner > a) & (inner < b)]
            if inner.size == 0:
                break
        ok = fires(inner)
        monotone = monotone and _monotone(ok)
        evaluations += int(inner.size)
        i = _first_success(ok)
        if i is None:
            a = inner[-1].item()
        else:
            b = inner[i].item()
            a = inner[i - 1].item() if i > 0 else a
    return {'status': 'resolved', 'bracket': [a, b], 'monotone': monotone, 'evaluations': evaluations, **ends}


# ====================================================================== cable
class HHCable:
    """Uniform multi-compartment Hodgkin-Huxley cable with sealed ends and a step stimulus at compartment 0.

    Units: lengths in um, ``Ra`` in ohm cm, ``Cm`` in uF/cm^2, conductances in mS/cm^2, potentials in mV,
    ``dt_ms`` in ms, stimulus current densities in uA/cm^2. Every simulation starts from the resting state."""

    def __init__(self, compartments: int = 1, length_um: float = 100.0, diameter_um: float = 476.0,
                 Ra: float = 35.4, Cm: float = 1.0, gNa: float = 120.0, gK: float = 36.0, gL: float = 0.3,
                 ENa: float = 50.0, EK: float = -77.0, EL: float = -54.387, dt_ms: float = 0.01,
                 method: str = 'rk4', temperature_c: float = REFERENCE_TEMPERATURE_C) -> None:
        self.compartments = _int('compartments', compartments, 1, 4000)
        self.length_um = _num('length_um', length_um, 1.0, 1e6)
        self.diameter_um = _num('diameter_um', diameter_um, 0.05, 5000.0)
        self.Ra = _num('Ra', Ra, 1.0, 1e5)
        self.Cm = _num('Cm', Cm, 0.01, 100.0)
        self.gNa = _num('gNa', gNa, 0.0, 1e4)
        self.gK = _num('gK', gK, 0.0, 1e4)
        self.gL = _num('gL', gL, 0.0, 1e4)
        self.ENa = _num('ENa', ENa, -200.0, 200.0)
        self.EK = _num('EK', EK, -200.0, 200.0)
        self.EL = _num('EL', EL, -200.0, 200.0)
        self.dt_ms = _num('dt_ms', dt_ms, 0.0, 0.1, open_low=True)
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        self.method = method
        self.temperature_c = _num('temperature_c', temperature_c, -5.0, 45.0)
        self.phi = temperature_factor(self.temperature_c)
        if self.gNa + self.gK + self.gL <= 0.0:
            raise ValueError('at least one membrane conductance must be positive')
        self._gax = self.axial_conductance(self.diameter_um)
        bound = self.stability_bound(self._gax)
        if self.method == 'rk4' and self.dt_ms * bound > _RK4_STABILITY:
            raise ValueError(f'dt_ms={self.dt_ms} exceeds the conservative RK4 stability limit '
                             f'{_RK4_STABILITY / bound:.6g} ms for this cable; reduce dt_ms or use method="exp_euler"')
        self._rest: float | None = None

    # ------------------------------------------------------------------ derived quantities
    @property
    def dx_um(self) -> float:
        return self.length_um / self.compartments

    def axial_conductance(self, diameter_um: float) -> float:
        """``a / (2 Ra dx^2)`` between neighbouring compartments, in mS/cm^2 (0 for a single compartment)."""
        if self.compartments == 1:
            return 0.0
        a_cm = 0.5 * float(diameter_um) * 1e-4
        dx_cm = self.dx_um * 1e-4
        return 1e3 * a_cm / (2.0 * self.Ra * dx_cm * dx_cm)

    def stability_bound(self, gax: float) -> float:
        """Conservative bound (1/ms) on the stiffness: voltage ``(4 g_ax + gNa + gK + gL) / Cm`` (all channels
        open, worst Laplacian mode) and the fastest gate rate ``phi (alpha + beta)`` over a wide voltage range."""
        lap = 0.0 if self.compartments == 1 else (2.0 if self.compartments == 2 else 4.0) * gax
        v_lo = min(self.EK, self.EL, self.ENa) - 30.0
        v_hi = max(self.EK, self.EL, self.ENa) + 30.0
        return max((lap + self.gNa + self.gK + self.gL) / self.Cm, self.phi * _gate_rate_bound(v_lo, v_hi))

    def config(self) -> dict[str, Any]:
        return {'compartments': self.compartments, 'length_um': self.length_um, 'diameter_um': self.diameter_um,
                'Ra': self.Ra, 'Cm': self.Cm, 'gNa': self.gNa, 'gK': self.gK, 'gL': self.gL, 'ENa': self.ENa,
                'EK': self.EK, 'EL': self.EL, 'dt_ms': self.dt_ms, 'method': self.method,
                'temperature_c': self.temperature_c}

    def replace(self, **changes: Any) -> 'HHCable':
        """A new, fully validated cable with some parameters changed."""
        unknown = sorted(set(changes) - set(self.config()))
        if unknown:
            raise ValueError(f'unknown HHCable parameters: {unknown}')
        return HHCable(**{**self.config(), **changes})

    # ------------------------------------------------------------------ resting state
    def _ionic_ss(self, v: np.ndarray) -> np.ndarray:
        m, h, n = steady_state_gates(v)
        return self.gNa * m ** 3 * h * (v - self.ENa) + self.gK * n ** 4 * (v - self.EK) + self.gL * (v - self.EL)

    def resting_potential(self) -> float:
        """Most negative stable zero of the steady-state ionic current (bisection to 1e-10 mV)."""
        if self._rest is None:
            grid = np.linspace(-150.0, 100.0, 2501)
            f = self._ionic_ss(grid)
            idx = np.flatnonzero((f[:-1] < 0.0) & (f[1:] >= 0.0))
            if idx.size == 0:
                raise ValueError('no stable resting potential in [-150, 100] mV for these parameters')
            lo, hi = float(grid[idx[0]]), float(grid[idx[0] + 1])
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if float(self._ionic_ss(np.array(mid))) < 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-10:
                    break
            self._rest = 0.5 * (lo + hi)
        return self._rest

    # ------------------------------------------------------------------ integrator
    def _rhs(self, S: np.ndarray, stim0: np.ndarray, gcol: np.ndarray) -> np.ndarray:
        """Time derivative of the stacked state ``S = [v, m, h, n]`` of shape (4, B, N)."""
        v = S[0]
        u = v + 65.0
        R = np.empty((6,) + v.shape)                          # alpha_m, alpha_h, alpha_n, beta_m, beta_h, beta_n
        np.exp(_BETA_K * u + _BETA_C, out=R[3:])              # 4 e^(-u/18), e^(-(u-30)/10), 0.125 e^(-u/80)
        R[4] += 1.0
        np.reciprocal(R[4], out=R[4])                         # beta_h = 1 / (1 + e^(-(u-30)/10))
        np.exp(_AH_K * u + _AH_C, out=R[1])                   # alpha_h = 0.07 e^(-u/20)
        x = u - _TRAP_SHIFT                                   # alpha_m and alpha_n arguments
        with np.errstate(divide='ignore', invalid='ignore'):
            trap = (_TRAP_SCALE * x) / np.expm1(-0.1 * x)     # 0.1 x / (1 - e^(-x/10)), 0.01 x / (1 - e^(-x/10))
        if not math.isfinite(trap.sum()):                     # x == 0 exactly (0/0): use the limit y + x/2
            trap = np.where(np.isfinite(trap), trap, -_TRAP_SCALE * (10.0 + 0.5 * x))
        R[0:3:2] = trap
        D = np.empty_like(S)
        np.multiply(self.phi, R[:3] - (R[:3] + R[3:]) * S[1:], out=D[1:])   # phi (alpha (1 - x) - beta x)
        m, h, n = S[1], S[2], S[3]
        n2 = n * n
        dv = -(self.gNa * (m * m * m) * h * (v - self.ENa) + self.gK * (n2 * n2) * (v - self.EK)
               + self.gL * (v - self.EL))
        if v.shape[1] > 1:
            d = gcol * (v[:, 1:] - v[:, :-1])
            dv[:, :-1] += d
            dv[:, 1:] -= d
        dv[:, 0] += stim0
        D[0] = dv / self.Cm
        return D

    def _step_rk4(self, S: np.ndarray, stim0: np.ndarray, gcol: np.ndarray) -> np.ndarray:
        dt = self.dt_ms
        k1 = self._rhs(S, stim0, gcol)
        k2 = self._rhs(S + (0.5 * dt) * k1, stim0, gcol)
        k3 = self._rhs(S + (0.5 * dt) * k2, stim0, gcol)
        k4 = self._rhs(S + dt * k3, stim0, gcol)
        return S + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)

    def _step_exp_euler(self, S: np.ndarray, stim0: np.ndarray, gcol: np.ndarray, deg: np.ndarray) -> np.ndarray:
        dt, phi = self.dt_ms, self.phi
        v = S[0]
        am, bm, ah, bh, an, bn = hh_rates(v)
        out = np.empty_like(S)
        for i, (a, b) in enumerate(((am, bm), (ah, bh), (an, bn)), start=1):
            s = a + b
            inf = a / s
            out[i] = inf + (S[i] - inf) * np.exp(-dt * phi * s)
        m, h, n = out[1], out[2], out[3]
        gna = self.gNa * m * m * m * h
        n2 = n * n
        gk = self.gK * n2 * n2
        drive = gna * self.ENa + gk * self.EK + self.gL * self.EL
        if v.shape[1] > 1:
            nb = np.zeros_like(v)
            nb[:, 1:] += v[:, :-1]
            nb[:, :-1] += v[:, 1:]
            drive = drive + gcol * nb
        drive[:, 0] += stim0
        G = gna + gk + self.gL + gcol * deg
        vinf = drive / G
        out[0] = vinf + (v - vinf) * np.exp(-dt * G / self.Cm)
        return out

    def _simulate(self, on: np.ndarray, end: np.ndarray, amp: np.ndarray, n_steps: int, rec: np.ndarray, *,
                  gax: np.ndarray | None = None, trace_site: int | None = None,
                  trace_every: int = 1) -> dict[str, Any]:
        """Integrate a batch. ``on``/``end``/``amp`` have shape (B, P): pulse ``p`` of element ``b`` injects
        ``amp`` into compartment 0 during steps ``on <= k < end``. Returns spike times (ms) per element and
        recorded compartment, peak voltages and (optionally) a decimated voltage trace."""
        B = amp.shape[0]
        N = self.compartments
        g = np.full(B, self._gax) if gax is None else np.asarray(gax, dtype=np.float64)
        gcol = g[:, None]
        deg = np.full(N, 2.0)
        if N == 1:
            deg[:] = 0.0
        else:
            deg[0] = deg[-1] = 1.0
        v0 = self.resting_potential()
        m0, h0, n0 = (float(x) for x in steady_state_gates(np.array(v0)))
        S = np.empty((4, B, N))
        S[0], S[1], S[2], S[3] = v0, m0, h0, n0
        changes = np.unique(np.concatenate([on.ravel(), end.ravel()]))
        changes = [int(c) for c in changes if 0 < c < n_steps]
        stim0 = (amp * ((on <= 0) & (end > 0))).sum(axis=1)
        prev = S[0][:, rec].copy()
        armed = prev < REARM_MV
        peak = prev.copy()
        times: list[list[list[float]]] = [[[] for _ in range(rec.size)] for _ in range(B)]
        trace = None
        if trace_site is not None:
            samples = n_steps // trace_every + 1
            trace = np.empty((B, samples))
            trace[:, 0] = S[0][:, trace_site]
        dt = self.dt_ms
        ci = 0
        for k in range(n_steps):
            if ci < len(changes) and k == changes[ci]:
                stim0 = (amp * ((on <= k) & (end > k))).sum(axis=1)
                ci += 1
            if self.method == 'rk4':
                S_new = self._step_rk4(S, stim0, gcol)
            else:
                S_new = self._step_exp_euler(S, stim0, gcol, deg)
            cur = S_new[0][:, rec]
            up = armed & (cur >= SPIKE_THRESHOLD_MV)
            if up.any():
                d0 = self._rhs(S, stim0, gcol)[0][:, rec]
                d1 = self._rhs(S_new, stim0, gcol)[0][:, rec]
                for b, s in zip(*np.nonzero(up)):
                    frac = _hermite_crossing(float(prev[b, s]), float(cur[b, s]), float(d0[b, s]), float(d1[b, s]),
                                             dt, SPIKE_THRESHOLD_MV)
                    times[int(b)][int(s)].append((k + frac) * dt)
                armed &= ~up
            armed |= cur < REARM_MV
            np.maximum(peak, cur, out=peak)
            if trace is not None and (k + 1) % trace_every == 0:
                trace[:, (k + 1) // trace_every] = S_new[0][:, trace_site]
            if (k + 1) % _FINITE_CHECK_EVERY == 0 and not np.all(np.isfinite(S_new)):
                raise FloatingPointError('HH cable integration diverged (non-finite state)')
            prev = cur
            S = S_new
        if not np.all(np.isfinite(S)):
            raise FloatingPointError('HH cable integration diverged (non-finite state)')
        out: dict[str, Any] = {'spike_times_ms': times, 'peak_mv': peak, 'final_v_mv': S[0].copy()}
        if trace is not None:
            out['trace_mv'] = trace
            out['trace_dt_ms'] = dt * trace_every
        return out

    def _pulses_to_steps(self, pulses: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            arr = np.asarray(pulses, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError('pulses must be numeric (onset_ms, duration_ms, amplitude) rows') from exc
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[2] != 3 or not 1 <= arr.shape[0] <= _MAX_BATCH or not 1 <= arr.shape[1] <= 64:
            raise ValueError('pulses must have shape (P, 3) or (B, P, 3) with 1 <= B <= 4096 and 1 <= P <= 64')
        if not np.all(np.isfinite(arr)):
            raise ValueError('pulses must be finite')
        if np.any(arr[..., 0] < 0.0) or np.any(arr[..., 1] < 0.0) or np.any(np.abs(arr[..., 2]) > 1e7):
            raise ValueError('pulse onsets and durations must be >= 0 and |amplitude| <= 1e7 uA/cm^2')
        on = np.rint(arr[..., 0] / self.dt_ms).astype(np.int64)
        end = on + np.rint(arr[..., 1] / self.dt_ms).astype(np.int64)
        return on, end, arr[..., 2].copy()

    def _steps(self, t_ms: float) -> int:
        t = _num('t_stop_ms', t_ms, 0.0, 1e5, open_low=True)
        n = int(round(t / self.dt_ms))
        if not 1 <= n <= _MAX_STEPS:
            raise ValueError(f't_stop_ms / dt_ms must give 1..{_MAX_STEPS} steps')
        return n

    def _sites(self, record: Any) -> np.ndarray:
        if record is None:
            return np.arange(self.compartments, dtype=np.int64)
        idx = np.asarray(record)
        if idx.ndim != 1 or idx.size == 0 or idx.dtype.kind not in 'iu':
            raise ValueError('record must be a non-empty 1-D sequence of compartment indices')
        if idx.min() < 0 or idx.max() >= self.compartments:
            raise ValueError('record index out of range')
        return idx.astype(np.int64)

    def run(self, pulses: Any, *, t_stop_ms: float, record: Any = None, trace_site: int | None = None,
            trace_every: int = 1) -> dict[str, Any]:
        """Simulate from rest. ``pulses`` rows are ``(onset_ms, duration_ms, amplitude_uA_cm2)`` applied to
        compartment 0 (shape (P, 3) for one run or (B, P, 3) for a batch); onsets and durations are rounded to
        the time grid. Returns ``spike_times_ms[b][site]``, ``peak_mv`` (B, sites), ``final_v_mv`` and, with
        ``trace_site``, ``trace_mv`` (B, samples) every ``trace_every`` steps."""
        on, end, amp = self._pulses_to_steps(pulses)
        n = self._steps(t_stop_ms)
        rec = self._sites(record)
        if trace_site is not None:
            trace_site = _int('trace_site', trace_site, 0, self.compartments - 1)
            trace_every = _int('trace_every', trace_every, 1, _MAX_STEPS)
            if (n // trace_every + 1) * amp.shape[0] > _MAX_TRACE:
                raise ValueError(f'trace would exceed {_MAX_TRACE} samples; increase trace_every')
        out = self._simulate(on, end, amp, n, rec, trace_site=trace_site, trace_every=trace_every)
        out['record'] = rec.tolist()
        out['pulse_steps'] = {'onset': on.tolist(), 'end': end.tolist()}
        return out

    # ------------------------------------------------------------------ measurements
    def _step_batch(self, amplitudes: np.ndarray, onset_ms: float, duration_ms: float) -> np.ndarray:
        rows = np.zeros((amplitudes.size, 1, 3))
        rows[:, 0, 0] = onset_ms
        rows[:, 0, 1] = duration_ms
        rows[:, 0, 2] = amplitudes
        return rows

    def _default_site(self) -> int:
        return self.compartments - 1

    def threshold_current(self, duration_ms: float, *, onset_ms: float = 1.0, window_ms: float | None = None,
                          site: int | None = None, bracket: tuple[float, float] = (0.0, 50.0), rounds: int = 2,
                          batch: int = 32) -> dict[str, Any]:
        """Smallest step amplitude (uA/cm^2) giving at least one spike at ``site`` (default: the last compartment,
        i.e. a spike that reached the far end) within ``window_ms`` of the onset.

        Batched search with ``rounds`` batched simulations of ``batch`` amplitudes each (the first includes both
        ends of ``bracket``). Returns the final bracket, its midpoint as ``threshold`` and ``resolved``."""
        duration_ms = _num('duration_ms', duration_ms, 0.0, 1e5, open_low=True)
        onset_ms = _num('onset_ms', onset_ms, 0.0, 1e5)
        window_ms = duration_ms + 10.0 if window_ms is None else _num('window_ms', window_ms, 0.0, 1e5, open_low=True)
        rounds = _int('rounds', rounds, 1, 12)
        batch = _int('batch', batch, 3, 1024)
        site = self._default_site() if site is None else _int('site', site, 0, self.compartments - 1)
        lo, hi = (_num('bracket', b, 0.0, 1e7) for b in bracket)
        if not hi > lo:
            raise ValueError('bracket must be (low, high) with high > low')
        n = self._steps(onset_ms + window_ms)
        rec = np.array([site], dtype=np.int64)

        def fires(amps: np.ndarray) -> list[bool]:
            on, end, amp = self._pulses_to_steps(self._step_batch(amps, onset_ms, duration_ms))
            res = self._simulate(on, end, amp, n, rec)
            return [len(res['spike_times_ms'][b][0]) > 0 for b in range(amps.size)]

        res = _batched_bisection(fires, lo, hi, rounds=rounds, batch=batch)
        a, b = res['bracket']
        resolved = res['status'] == 'resolved'
        return {'threshold': 0.5 * (a + b) if resolved else None, 'bracket': [a, b], 'resolved': resolved,
                'reason': None if resolved else res['status'], 'monotone': res['monotone'],
                'evaluations': res['evaluations']}

    def fi_curve(self, amplitudes: Any, *, window_ms: float = 200.0, transient_ms: float = 50.0,
                 onset_ms: float = 1.0, site: int | None = None) -> dict[str, Any]:
        """Steady firing rate for constant steps of each amplitude (one batched run).

        Spikes after ``onset + transient`` count; the rate is ``(k - 1) / (t_last - t_first)`` (Hz) for ``k >= 2``
        such spikes and 0 otherwise, so a lone onset spike does not count as repetitive firing."""
        amps = _vector('amplitudes', amplitudes, low=-1e6, high=1e6)
        window_ms = _num('window_ms', window_ms, 0.0, 1e5, open_low=True)
        transient_ms = _num('transient_ms', transient_ms, 0.0, window_ms)
        onset_ms = _num('onset_ms', onset_ms, 0.0, 1e5)
        site = self._default_site() if site is None else _int('site', site, 0, self.compartments - 1)
        n = self._steps(onset_ms + window_ms)
        on, end, amp = self._pulses_to_steps(self._step_batch(amps, onset_ms, window_ms))
        res = self._simulate(on, end, amp, n, np.array([site], dtype=np.int64))
        t0 = onset_ms + transient_ms
        rates, counts, isis = [], [], []
        for b in range(amps.size):
            st = res['spike_times_ms'][b][0]
            steady = [t for t in st if t >= t0]
            counts.append(len(st))
            rates.append((len(steady) - 1) * 1000.0 / (steady[-1] - steady[0]) if len(steady) >= 2 else 0.0)
            isis.append(np.diff(st).tolist())
        return {'amplitudes': amps.tolist(), 'rates_hz': rates, 'spike_counts': counts, 'isis_ms': isis,
                'peak_mv': res['peak_mv'][:, 0].tolist(), 'final_v_mv': res['final_v_mv'][:, site].tolist(),
                'window_ms': window_ms, 'transient_ms': transient_ms}

    def conduction_velocity(self, *, amplitude: float = 3000.0, duration_ms: float = 0.2, t_stop_ms: float = 2.0,
                            sites: tuple[int, int] | None = None,
                            diameters_um: Sequence[float] | None = None) -> dict[str, Any]:
        """Conduction velocity (m/s) from first-spike arrival times at two compartments (default at 1/3 and 2/3
        of the cable). The stimulated end and the sealed far end both bias the local velocity over a few
        millimetres for squid-sized axons (the wave speeds up as it approaches the sealed end), so the cable
        should be long compared with that range; ``run_experiment`` uses 40 mm. With ``diameters_um`` the run
        is batched over diameters (each one checked against the RK4 stability bound)."""
        if self.compartments < 4:
            raise ValueError('conduction velocity needs at least 4 compartments')
        amplitude = _num('amplitude', amplitude, 0.0, 1e7, open_low=True)
        duration_ms = _num('duration_ms', duration_ms, 0.0, 1e5, open_low=True)
        if sites is None:
            sites = (self.compartments // 3, (2 * self.compartments) // 3)
        i1, i2 = (_int('site', s, 0, self.compartments - 1) for s in sites)
        if not i2 > i1:
            raise ValueError('sites must be increasing')
        diam = (np.array([self.diameter_um]) if diameters_um is None
                else _vector('diameters_um', diameters_um, low=0.05, high=5000.0, max_size=64))
        gax = np.array([self.axial_conductance(d) for d in diam])
        if self.method == 'rk4' and self.dt_ms * self.stability_bound(float(gax.max())) > _RK4_STABILITY:
            raise ValueError('dt_ms exceeds the RK4 stability limit for the largest diameter')
        n = self._steps(t_stop_ms)
        on, end, amp = self._pulses_to_steps(self._step_batch(np.full(diam.size, amplitude), 0.0, duration_ms))
        mid = self.compartments // 2
        res = self._simulate(on, end, amp, n, np.array([i1, mid, i2], dtype=np.int64), gax=gax)
        dist_um = (i2 - i1) * self.dx_um
        rows = []
        for b, d in enumerate(diam):
            st = res['spike_times_ms'][b]
            t1 = st[0][0] if st[0] else None
            t2 = st[2][0] if st[2] else None
            ok = t1 is not None and t2 is not None and t2 > t1
            rows.append({'diameter_um': float(d), 'axial_conductance_mS_cm2': float(gax[b]),
                         'arrival_ms': [t1, t2], 'velocity_m_s': dist_um / (t2 - t1) * 1e-3 if ok else None,
                         'mid_peak_mv': float(res['peak_mv'][b, 1])})
        return {'sites': [i1, i2], 'distance_um': dist_um, 'rows': rows}

    def refractory_period(self, *, pulse_ms: float = 0.5, threshold: float | None = None,
                          conditioning_multiple: float = 2.0, test_multiple: float = 4.0, max_interval_ms: float = 20.0,
                          onset_ms: float = 1.0, rounds: int = 2, batch: int = 32, site: int | None = None,
                          threshold_bracket: tuple[float, float] = (0.0, 500.0), threshold_rounds: int = 2,
                          tail_ms: float = 10.0) -> dict[str, Any]:
        """Paired-pulse refractory period: the smallest onset-to-onset interval at which a test pulse of
        ``test_multiple`` x the single-pulse threshold evokes a second spike after a conditioning spike.

        Batched search over intervals on the time grid (``rounds`` batched simulations of ``batch`` intervals;
        the first includes the shortest interval, one pulse width, and ``max_interval_ms``); each run lasts until
        ``tail_ms`` after the latest test pulse. With a strong test pulse this is an operational estimate of the
        absolute refractory period (recovery assumed monotone near the transition; ``monotone`` reports whether
        every evaluated grid agreed). Without ``threshold`` the single-pulse threshold is searched first
        (``threshold_rounds`` batched simulations) and the upper end of its bracket is used."""
        pulse_ms = _num('pulse_ms', pulse_ms, 0.0, 100.0, open_low=True)
        conditioning_multiple = _num('conditioning_multiple', conditioning_multiple, 1.0, 1e3, open_low=True)
        test_multiple = _num('test_multiple', test_multiple, 1.0, 1e3, open_low=True)
        max_interval_ms = _num('max_interval_ms', max_interval_ms, pulse_ms, 1e4, open_low=True)
        onset_ms = _num('onset_ms', onset_ms, 0.0, 1e4)
        tail_ms = _num('tail_ms', tail_ms, 1.0, 1e4)
        rounds = _int('rounds', rounds, 1, 12)
        threshold_rounds = _int('threshold_rounds', threshold_rounds, 1, 12)
        batch = _int('batch', batch, 3, 1024)
        site = self._default_site() if site is None else _int('site', site, 0, self.compartments - 1)
        thr_info = None
        if threshold is None:
            thr_info = self.threshold_current(pulse_ms, onset_ms=onset_ms, site=site, bracket=threshold_bracket,
                                              rounds=threshold_rounds, batch=batch)
            if not thr_info['resolved']:
                raise ValueError(f'single-pulse threshold not found in {threshold_bracket}')
            threshold = thr_info['bracket'][1]          # upper end: certainly suprathreshold
        threshold = _num('threshold', threshold, 0.0, 1e7, open_low=True)
        dt = self.dt_ms
        pulse_steps = max(1, int(round(pulse_ms / dt)))
        lo, hi = pulse_steps, int(round(max_interval_ms / dt))
        if not hi > lo:
            raise ValueError('max_interval_ms must exceed the pulse width by at least one time step')
        n = self._steps(onset_ms + max_interval_ms + pulse_ms + tail_ms)
        rec = np.array([site], dtype=np.int64)
        on0 = int(round(onset_ms / dt))

        def second_spike(intervals: np.ndarray) -> list[bool]:
            B = intervals.size
            on = np.empty((B, 2), dtype=np.int64)
            on[:, 0] = on0
            on[:, 1] = on0 + intervals
            end = on + pulse_steps
            amp = np.empty((B, 2))
            amp[:, 0] = conditioning_multiple * threshold
            amp[:, 1] = test_multiple * threshold
            res = self._simulate(on, end, amp, n, rec)
            out = []
            for b in range(B):
                st = res['spike_times_ms'][b][0]
                t_test = (on0 + intervals[b]) * dt
                out.append(len(st) >= 2 and st[0] < t_test and st[-1] >= t_test)
            return out

        res = _batched_bisection(second_spike, lo, hi, rounds=rounds, batch=batch, integer=True)
        info: dict[str, Any] = {'pulse_ms': pulse_steps * dt, 'threshold_uA_cm2': threshold,
                                'threshold_search': thr_info, 'test_amplitude': test_multiple * threshold,
                                'conditioning_amplitude': conditioning_multiple * threshold,
                                'fails_at_min_interval': not res['lower_fires'],
                                'succeeds_at_max_interval': res['upper_fires'],
                                'evaluations': res['evaluations']}
        if res['status'] != 'resolved':
            info.update({'refractory_ms': None, 'bracket_ms': None, 'monotone': res['monotone']})
            return info
        a, b = res['bracket']
        info.update({'refractory_ms': b * dt, 'bracket_ms': [a * dt, b * dt], 'monotone': res['monotone']})
        return info


# ====================================================================== NEUROWEAVE cell (abstract) reference
def neuroweave_fi_curve(drives: Any, *, steps: int = 400, transient: int = 100, seed: int = 0,
                        flags: Sequence[str] = ('membrane_dynamics',)) -> dict[str, Any]:
    """Spike counts and steady rates of isolated NEUROWEAVE cells (``CellPopulation``) under constant external drive.

    One cell per drive value, no recurrent input, neutral controls. The rate is ``(k - 1) / (t_last - t_first)``
    spikes per step over the spikes after ``transient`` (0 for fewer than two), as for the HH curve."""
    from ..neuroweave.cells import NEURON_FLAGS, CellPopulation, CompartmentDrive
    u = _vector('drives', drives, low=-100.0, high=100.0)
    steps = _int('steps', steps, 2, 1_000_000)
    transient = _int('transient', transient, 0, steps - 1)
    seed = _int('seed', seed, 0, 2**31 - 1)
    flags = frozenset(flags)
    if not flags <= set(NEURON_FLAGS):
        raise ValueError(f'flags must be NEUROWEAVE neuron flags {NEURON_FLAGS}')
    k = u.size
    cells = CellPopulation(k, seed=seed, regions=np.zeros(k, dtype=np.int64), signs=np.ones(k, dtype=np.int64))
    zero = np.zeros(k)
    spikes = np.zeros((steps, k), dtype=bool)
    for t in range(steps):
        cells.step(CompartmentDrive(zero.copy(), zero.copy(), zero.copy(), zero.copy(), u.copy()), flags)
        spikes[t] = cells.spike > 0
    rates, counts, isis = [], [], []
    for j in range(k):
        st = np.flatnonzero(spikes[:, j])
        steady = st[st >= transient]
        counts.append(int(st.size))
        rates.append(float((steady.size - 1) / (steady[-1] - steady[0])) if steady.size >= 2 else 0.0)
        isis.append(np.diff(st).astype(float).tolist())
    return {'drives': u.tolist(), 'rates_per_step': rates, 'spike_counts': counts, 'isis_steps': isis,
            'steps': steps, 'transient': transient, 'flags': sorted(flags)}


def neuroweave_threshold(*, steps: int = 400, seed: int = 0, bracket: tuple[float, float] = (0.0, 2.0),
                         rounds: int = 2, batch: int = 64, flags: Sequence[str] = ('membrane_dynamics',)) -> dict[str, Any]:
    """Smallest constant external drive that makes an isolated NEUROWEAVE cell spike within ``steps`` steps.

    Same batched search as :meth:`HHCable.threshold_current`: ``rounds`` populations of ``batch`` cells, the
    first including both ends of ``bracket``."""
    lo, hi = (_num('bracket', b, -100.0, 100.0) for b in bracket)
    if not hi > lo:
        raise ValueError('bracket must be (low, high) with high > low')
    rounds = _int('rounds', rounds, 1, 12)
    batch = _int('batch', batch, 3, 4096)

    def fires(values: np.ndarray) -> list[bool]:
        res = neuroweave_fi_curve(values, steps=steps, transient=0, seed=seed, flags=flags)
        return [c > 0 for c in res['spike_counts']]

    res = _batched_bisection(fires, lo, hi, rounds=rounds, batch=batch)
    a, b = res['bracket']
    resolved = res['status'] == 'resolved'
    return {'threshold': 0.5 * (a + b) if resolved else None, 'bracket': [a, b], 'resolved': resolved,
            'reason': None if resolved else res['status'], 'monotone': res['monotone'],
            'evaluations': res['evaluations']}


def adaptation_index(isis: Sequence[float]) -> float | None:
    """Mean of ``(ISI_{k+1} - ISI_k) / (ISI_{k+1} + ISI_k)``: 0 for regular firing, > 0 when firing slows down.
    ``None`` with fewer than two intervals."""
    x = np.asarray(list(isis), dtype=np.float64)
    if x.size < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0.0):
        return None
    return float(np.mean((x[1:] - x[:-1]) / (x[1:] + x[:-1])))


def average_ranks(x: Any) -> np.ndarray:
    """Ranks 1..n of a 1-D sequence with tied values sharing their average rank."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError('average_ranks needs a finite 1-D sequence')
    order = np.argsort(a, kind='mergesort')
    s = a[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])          # first index of each run of equal values
    ends = np.r_[starts[1:], s.size]
    ranks = np.empty(a.size)
    ranks[order] = np.repeat(0.5 * (starts + ends - 1) + 1.0, ends - starts)
    return ranks


def spearman_rho(a: Any, b: Any) -> float | None:
    """Spearman rank correlation (Pearson correlation of average ranks); ``None`` when either side is constant."""
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.size != rb.size or ra.size < 2:
        raise ValueError('spearman_rho needs two sequences of equal length >= 2')
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return None if den == 0.0 else float(ra @ rb) / den


def compare_fi_curves(levels: Sequence[float], hh_rates: Sequence[float],
                      nw_rates: Sequence[float]) -> dict[str, Any]:
    """Max-normalised curves on shared threshold-normalised drive levels and their Spearman rank correlation
    (over the suprathreshold levels, ``level >= 1``). Descriptive only."""
    lv = np.asarray(levels, dtype=np.float64)
    a = np.asarray(hh_rates, dtype=np.float64)
    b = np.asarray(nw_rates, dtype=np.float64)
    if not (lv.shape == a.shape == b.shape) or lv.ndim != 1:
        raise ValueError('levels and rates must be 1-D sequences of equal length')
    na = a / a.max() if a.max() > 0 else np.zeros_like(a)
    nb = b / b.max() if b.max() > 0 else np.zeros_like(b)
    sel = lv >= 1.0
    rho = None
    if sel.sum() >= 3 and np.ptp(na[sel]) > 0 and np.ptp(nb[sel]) > 0:
        rho = spearman_rho(na[sel], nb[sel])

    def onset(x: np.ndarray) -> float | None:
        i = _first_success(list(x > 0))
        return None if i is None else float(lv[i])

    return {'levels': lv.tolist(), 'hh_normalised': na.tolist(), 'neuroweave_normalised': nb.tolist(),
            'spearman_suprathreshold': rho, 'hh_repetitive_onset_level': onset(a),
            'neuroweave_repetitive_onset_level': onset(b),
            'hh_first_nonzero_normalised_rate': None if onset(a) is None else float(na[a > 0][0]),
            'neuroweave_first_nonzero_normalised_rate': None if onset(b) is None else float(nb[b > 0][0]),
            'mean_abs_difference': float(np.mean(np.abs(na - nb)))}


# ====================================================================== validation experiment
_DEFAULTS: dict[str, Any] = {
    'patch_dt_ms': 0.0125, 'patch_temperature_c': 6.3, 'fi_window_ms': 200.0, 'fi_transient_ms': 50.0,
    'fi_levels': (0.0, 0.5, 0.9, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0),
    'threshold_bracket': (0.0, 20.0), 'threshold_window_ms': 40.0, 'threshold_rounds': 2, 'search_batch': 32,
    'pulse_ms': 0.5, 'pulse_bracket': (0.0, 100.0), 'refractory_test_multiple': 4.0, 'refractory_rounds': 2,
    'refractory_max_interval_ms': 20.0, 'refractory_tail_ms': 10.0, 'convergence_amplitude': 20.0,
    'convergence_dts': (0.0125, 0.00625, 0.003125), 'convergence_t_stop_ms': 4.0, 'convergence_floor_ms': 1e-6,
    # 40 mm cable, velocity between 1/3 and 2/3 of its length: both ends bias the local speed over several mm
    'cable_temperature_c': 18.5, 'cable_length_um': 40000.0, 'cable_compartments': 200, 'cable_dt_ms': 0.0006,
    'cable_diameters_um': (119.0, 238.0, 476.0), 'cable_Ra': 35.4, 'cable_amplitude': 3000.0,
    'cable_pulse_ms': 0.2, 'cable_t_stop_ms': 3.4, 'reference_tolerance': 0.02, 'scaling_tolerance': 0.03,
    'neuroweave_steps': 600, 'neuroweave_transient': 150, 'adaptation_level': 4.0,
}
_QUICK: dict[str, Any] = {'fi_window_ms': 40.0, 'fi_transient_ms': 10.0, 'threshold_window_ms': 15.0,
                          'threshold_rounds': 1, 'refractory_rounds': 1, 'search_batch': 64,
                          'threshold_bracket': (0.0, 5.0), 'pulse_bracket': (0.0, 40.0),
                          'fi_levels': (0.0, 0.5, 0.9, 1.1, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0),
                          'refractory_max_interval_ms': 12.0, 'refractory_tail_ms': 4.0, 'convergence_t_stop_ms': 2.5,
                          'cable_compartments': 60, 'cable_dt_ms': 0.005,
                          'neuroweave_steps': 150, 'neuroweave_transient': 40}


def _seeds(seeds: Any) -> tuple[int, ...]:
    try:
        out = tuple(_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


def _validate_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    c = dict(cfg)
    for key in ('patch_dt_ms', 'cable_dt_ms'):
        c[key] = _num(key, c[key], 0.0, 0.1, open_low=True)
    for key in ('patch_temperature_c', 'cable_temperature_c'):
        c[key] = _num(key, c[key], -5.0, 45.0)
    c['fi_window_ms'] = _num('fi_window_ms', c['fi_window_ms'], 10.0, 5000.0)
    c['fi_transient_ms'] = _num('fi_transient_ms', c['fi_transient_ms'], 0.0, c['fi_window_ms'] / 2)
    c['fi_levels'] = tuple(float(x) for x in _vector('fi_levels', c['fi_levels'], low=0.0, high=1000.0, min_size=3,
                                                     max_size=128))
    if sorted(set(c['fi_levels'])) != list(c['fi_levels']) or sum(1 for x in c['fi_levels'] if x >= 1.0) < 3:
        raise ValueError('fi_levels must be increasing with at least three levels >= 1')
    if c['fi_levels'][-1] < 1.1:
        raise ValueError('fi_levels must include a suprathreshold level >= 1.1')
    for key in ('threshold_bracket', 'pulse_bracket'):
        b = tuple(float(x) for x in _vector(key, c[key], low=0.0, high=1e7, min_size=2, max_size=2))
        if not b[1] > b[0]:
            raise ValueError(f'{key} must be increasing')
        c[key] = b
    c['threshold_window_ms'] = _num('threshold_window_ms', c['threshold_window_ms'], 5.0, c['fi_window_ms'])
    c['threshold_rounds'] = _int('threshold_rounds', c['threshold_rounds'], 1, 12)
    c['search_batch'] = _int('search_batch', c['search_batch'], 4, 256)
    c['pulse_ms'] = _num('pulse_ms', c['pulse_ms'], 0.0, 10.0, open_low=True)
    c['refractory_test_multiple'] = _num('refractory_test_multiple', c['refractory_test_multiple'], 1.0, 100.0,
                                         open_low=True)
    c['refractory_rounds'] = _int('refractory_rounds', c['refractory_rounds'], 1, 12)
    c['refractory_max_interval_ms'] = _num('refractory_max_interval_ms', c['refractory_max_interval_ms'],
                                           c['pulse_ms'], 500.0, open_low=True)
    c['refractory_tail_ms'] = _num('refractory_tail_ms', c['refractory_tail_ms'], 2.0, 100.0)
    c['convergence_amplitude'] = _num('convergence_amplitude', c['convergence_amplitude'], 0.0, 1e5, open_low=True)
    dts = tuple(float(x) for x in _vector('convergence_dts', c['convergence_dts'], low=1e-6, high=0.1,
                                          min_size=3, max_size=3))
    if not all(math.isclose(coarse, 2.0 * fine) for coarse, fine in zip(dts, dts[1:])):
        raise ValueError('convergence_dts must halve at each refinement')
    c['convergence_dts'] = dts
    c['convergence_t_stop_ms'] = _num('convergence_t_stop_ms', c['convergence_t_stop_ms'], 2.0, 100.0)
    c['convergence_floor_ms'] = _num('convergence_floor_ms', c['convergence_floor_ms'], 0.0, 1e-3, open_low=True)
    c['cable_length_um'] = _num('cable_length_um', c['cable_length_um'], 100.0, 1e6)
    c['cable_compartments'] = _int('cable_compartments', c['cable_compartments'], 8, 4000)
    diam = tuple(float(x) for x in _vector('cable_diameters_um', c['cable_diameters_um'], low=0.05, high=5000.0,
                                           min_size=2, max_size=16))
    if sorted(set(diam)) != list(diam):
        raise ValueError('cable_diameters_um must be strictly increasing')
    c['cable_diameters_um'] = diam
    c['cable_Ra'] = _num('cable_Ra', c['cable_Ra'], 1.0, 1e5)
    c['cable_amplitude'] = _num('cable_amplitude', c['cable_amplitude'], 0.0, 1e7, open_low=True)
    c['cable_pulse_ms'] = _num('cable_pulse_ms', c['cable_pulse_ms'], 0.0, 10.0, open_low=True)
    c['cable_t_stop_ms'] = _num('cable_t_stop_ms', c['cable_t_stop_ms'], 0.1, 100.0)
    c['reference_tolerance'] = _num('reference_tolerance', c['reference_tolerance'], 0.0, 1.0, open_low=True)
    c['scaling_tolerance'] = _num('scaling_tolerance', c['scaling_tolerance'], 0.0, 0.5, open_low=True)
    c['neuroweave_steps'] = _int('neuroweave_steps', c['neuroweave_steps'], 20, 100_000)
    c['neuroweave_transient'] = _int('neuroweave_transient', c['neuroweave_transient'], 0, c['neuroweave_steps'] // 2)
    c['adaptation_level'] = _num('adaptation_level', c['adaptation_level'], 1.0, 1000.0)
    return c


def _convergence(c: Mapping[str, Any]) -> dict[str, Any]:
    """First-spike time of the patch for a strong 1 ms step at three time steps (RK4 and exponential Euler)."""
    out: dict[str, Any] = {}
    for method in METHODS:
        times = []
        for dt in c['convergence_dts']:
            cable = HHCable(dt_ms=dt, method=method, temperature_c=c['patch_temperature_c'])
            res = cable.run([[0.5, 1.0, c['convergence_amplitude']]], t_stop_ms=c['convergence_t_stop_ms'], record=[0])
            st = res['spike_times_ms'][0][0]
            times.append(st[0] if st else None)
        if any(t is None for t in times):
            out[method] = {'spike_times_ms': times, 'error_coarse': None, 'error_fine': None, 'observed_order': None}
            continue
        e1, e2 = abs(times[0] - times[1]), abs(times[1] - times[2])
        ratio = c['convergence_dts'][0] / c['convergence_dts'][1]
        order = math.log(e1 / e2) / math.log(ratio) if e1 > 0 and e2 > 0 else None
        out[method] = {'spike_times_ms': times, 'error_coarse': e1, 'error_fine': e2, 'observed_order': order}
    return out


def _estimated_cost(c: Mapping[str, Any], n_seeds: int, refr: Mapping[str, Any]) -> dict[str, int]:
    """Sequential integration steps (each batched over its amplitudes/intervals/diameters) and cell updates."""
    dt = c['patch_dt_ms']
    steps = lambda t_ms, step=dt: int(round(t_ms / step))
    pulse_rounds = c['threshold_rounds'] if refr.get('threshold_search') else 0
    patch = (c['threshold_rounds'] * steps(1.0 + c['threshold_window_ms']) + steps(1.0 + c['fi_window_ms'])
             + pulse_rounds * steps(1.0 + c['pulse_ms'] + 10.0)
             + c['refractory_rounds'] * steps(1.0 + c['refractory_max_interval_ms'] + c['pulse_ms']
                                              + c['refractory_tail_ms']))
    convergence = 2 * sum(steps(c['convergence_t_stop_ms'], h) for h in c['convergence_dts'])
    cable = 3 * steps(c['cable_t_stop_ms'], c['cable_dt_ms'])               # dt and dt/2 runs
    nw_sequential = 2 * c['neuroweave_steps'] + n_seeds * c['neuroweave_steps']
    nw_cells = 2 * 64 * c['neuroweave_steps'] + n_seeds * len(c['fi_levels']) * c['neuroweave_steps']
    return {'patch_rk4_steps': int(patch), 'convergence_steps_both_methods': int(convergence),
            'cable_rk4_steps': int(cable),
            'cable_compartment_updates': int(cable * len(c['cable_diameters_um']) * c['cable_compartments']),
            'neuroweave_sequential_steps': int(nw_sequential), 'neuroweave_cell_updates': int(nw_cells)}


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Known-answer validation of the HH cable, then a descriptive f-I comparison with one NEUROWEAVE cell.

    Checks (``kind='validation'``, ``verdict`` in ``('pass', 'fail')``): resting potential within 0.5 mV of the
    documented -65 mV and stationary without input; a sub-threshold step stays silent and a supra-threshold one
    fires with a peak above 0 mV; conduction velocity positive, increasing with diameter, with a log-log slope
    within ``scaling_tolerance`` of the 1/2 of HH eqn 34 (``theta = sqrt(K a / (2 R2 CM))``) and within
    ``reference_tolerance`` of HH's computed 18.8 m/s for their axon (18.5 C, d = 476 um, Ra = 35.4 ohm cm),
    measured between 1/3 and 2/3 of a 40 mm cable; spike timing converging when dt is halved (patch, RK4) and
    the velocity changing little at dt/2 (and exponential Euler converging at first order); paired-pulse
    refractoriness (second pulse fails at the shortest interval, succeeds at the longest).
    The HH model is deterministic: ``seeds`` only seed the NEUROWEAVE ``CellPopulation`` constructors (whose random
    state does not enter ``membrane_dynamics``), so per-seed NEUROWEAVE curves are reported and expected equal."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    cfg = dict(_DEFAULTS)
    if quick:
        cfg.update(_QUICK)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_experiment overrides: {unknown}')
    cfg.update(overrides)
    c = _validate_config(cfg)
    from . import harness
    t0 = time.perf_counter()
    wall: dict[str, float] = {}

    # ---------------------------------------------------------------- patch (single compartment)
    t = time.perf_counter()
    patch = HHCable(dt_ms=c['patch_dt_ms'], temperature_c=c['patch_temperature_c'])
    rest = patch.resting_potential()
    thr = patch.threshold_current(c['fi_window_ms'], window_ms=c['threshold_window_ms'],
                                  bracket=c['threshold_bracket'], rounds=c['threshold_rounds'],
                                  batch=c['search_batch'])
    if not thr['resolved']:
        raise RuntimeError(f'HH long-step threshold not bracketed by {c["threshold_bracket"]}: {thr["reason"]}')
    i_th = float(thr['threshold'])
    levels = np.array(c['fi_levels'])
    fi = patch.fi_curve(levels * i_th, window_ms=c['fi_window_ms'], transient_ms=c['fi_transient_ms'])
    wall['patch_threshold_and_fi'] = time.perf_counter() - t
    t = time.perf_counter()
    refr = patch.refractory_period(pulse_ms=c['pulse_ms'], test_multiple=c['refractory_test_multiple'],
                                   max_interval_ms=c['refractory_max_interval_ms'],
                                   threshold_bracket=c['pulse_bracket'], threshold_rounds=c['threshold_rounds'],
                                   rounds=c['refractory_rounds'], batch=c['search_batch'],
                                   tail_ms=c['refractory_tail_ms'])
    wall['patch_refractory'] = time.perf_counter() - t
    t = time.perf_counter()
    conv = _convergence(c)
    wall['patch_convergence'] = time.perf_counter() - t

    zero_row = int(np.flatnonzero(levels == 0.0)[0]) if np.any(levels == 0.0) else None
    sub = [i for i, x in enumerate(levels) if 0.0 < x <= 0.5]
    supra = [i for i, x in enumerate(levels) if x >= 1.1]
    peak_patch = max(fi['peak_mv'][i] for i in supra)

    # ---------------------------------------------------------------- cable (HH 1952 reference axon family)
    t = time.perf_counter()
    cable = HHCable(compartments=c['cable_compartments'], length_um=c['cable_length_um'],
                    diameter_um=HH1952_REFERENCE_AXON['diameter_um'], Ra=c['cable_Ra'], dt_ms=c['cable_dt_ms'],
                    temperature_c=c['cable_temperature_c'])
    vel = cable.conduction_velocity(amplitude=c['cable_amplitude'], duration_ms=c['cable_pulse_ms'],
                                    t_stop_ms=c['cable_t_stop_ms'], diameters_um=c['cable_diameters_um'])
    vel_half = cable.replace(dt_ms=c['cable_dt_ms'] / 2).conduction_velocity(
        amplitude=c['cable_amplitude'], duration_ms=c['cable_pulse_ms'], t_stop_ms=c['cable_t_stop_ms'],
        diameters_um=c['cable_diameters_um'])
    wall['cable'] = time.perf_counter() - t
    v = [r['velocity_m_s'] for r in vel['rows']]
    v_half = [r['velocity_m_s'] for r in vel_half['rows']]
    v_ok = all(x is not None and x > 0 for x in v)
    slope = None
    if v_ok:
        slope = float(np.polyfit(np.log(c['cable_diameters_um']), np.log(v), 1)[0])
    ref_idx = (c['cable_diameters_um'].index(HH1952_REFERENCE_AXON['diameter_um'])
               if HH1952_REFERENCE_AXON['diameter_um'] in c['cable_diameters_um'] else None)
    ref_v = v[ref_idx] if ref_idx is not None else None
    reference_applies = (ref_idx is not None and c['cable_temperature_c'] == HH1952_REFERENCE_AXON['temperature_c']
                         and c['cable_Ra'] == HH1952_REFERENCE_AXON['Ra_ohm_cm'])
    ref_rel = (abs(ref_v - HH1952_REFERENCE_AXON['computed_velocity_m_s']) / HH1952_REFERENCE_AXON['computed_velocity_m_s']
               if ref_v is not None else None)
    dt_rel = [abs(a - b) / b if a is not None and b else None for a, b in zip(v, v_half)]

    # ---------------------------------------------------------------- NEUROWEAVE cell (descriptive)
    t = time.perf_counter()
    nth = neuroweave_threshold(steps=c['neuroweave_steps'], seed=seed_list[0], rounds=2)
    if not nth['resolved']:
        raise RuntimeError('NEUROWEAVE cell threshold not bracketed')
    nw_runs = []
    for s in seed_list:        # same drive levels for every seed: the curves must agree if the seed is irrelevant
        nfi = neuroweave_fi_curve(levels * nth['threshold'], steps=c['neuroweave_steps'],
                                  transient=c['neuroweave_transient'], seed=s)
        nw_runs.append({'seed': s, 'fi': nfi})
    wall['neuroweave'] = time.perf_counter() - t
    nw = nw_runs[0]
    cmp = compare_fi_curves(levels, fi['rates_hz'], nw['fi']['rates_per_step'])
    a_level = c['adaptation_level']
    a_idx = int(np.argmin(np.abs(levels - a_level)))
    adaptation = {'level': float(levels[a_idx]),
                  'hh': adaptation_index(fi['isis_ms'][a_idx]),
                  'neuroweave': adaptation_index(nw['fi']['isis_steps'][a_idx])}

    checks = {
        'resting_within_0.5mV_of_documented_-65mV': bool(abs(rest - HH_REST_MV) <= 0.5),
        'rest_stationary_without_input': bool(zero_row is not None and fi['spike_counts'][zero_row] == 0
                                              and abs(fi['final_v_mv'][zero_row] - rest) < 1e-6),
        'subthreshold_steps_silent': bool(sub) and all(fi['spike_counts'][i] == 0 for i in sub),
        'suprathreshold_steps_fire': all(fi['spike_counts'][i] >= 1 for i in supra),
        'spike_peak_above_0mV': bool(peak_patch > 0.0) and all(r['mid_peak_mv'] > 0.0 for r in vel['rows']),
        'velocity_positive': v_ok,
        'velocity_increases_with_diameter': v_ok and all(b > a for a, b in zip(v, v[1:])),
        'velocity_scaling_exponent_within_tolerance_of_0.5': (slope is not None
                                                              and abs(slope - 0.5) <= c['scaling_tolerance']),
        # RK4 at these steps is already at the interpolation/rounding floor, so "converges" means: the change at
        # dt/2 at least halves, or both changes are below ``convergence_floor_ms``
        'spike_time_converges_under_dt_halving': bool(
            conv['rk4']['error_fine'] is not None and conv['rk4']['error_fine'] < 1e-3
            and conv['rk4']['error_fine'] <= max(0.5 * conv['rk4']['error_coarse'], c['convergence_floor_ms'])),
        'exp_euler_first_order_convergence': bool(
            conv['exp_euler']['observed_order'] is not None and 0.7 <= conv['exp_euler']['observed_order'] <= 1.5),
        'velocity_changes_below_1pct_at_dt_half': all(x is not None and x < 0.01 for x in dt_rel),
        'refractory_second_pulse_fails_then_recovers': bool(refr['fails_at_min_interval']
                                                            and refr['succeeds_at_max_interval']
                                                            and refr['refractory_ms'] is not None),
    }
    if reference_applies:
        checks['reference_velocity_within_tolerance_of_hh1952_computed'] = bool(
            ref_rel is not None and ref_rel <= c['reference_tolerance'])

    receipt = harness.base_receipt(MODULE_ID, 'validation', {**c, 'quick': quick, 'seeds': list(seed_list),
                                                             'spike_threshold_mv': SPIKE_THRESHOLD_MV,
                                                             'rearm_mv': REARM_MV})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['patch'] = {
        'config': patch.config(), 'resting_potential_mv': rest, 'documented_rest_mv': HH_REST_MV,
        'threshold_long_step': {**thr, 'step_ms': c['fi_window_ms']},
        'spike_peak_mv': peak_patch,
        'fi_curve': {**{k: fi[k] for k in ('amplitudes', 'rates_hz', 'spike_counts', 'peak_mv')},
                     'levels': list(c['fi_levels']), 'window_ms': fi['window_ms'], 'transient_ms': fi['transient_ms']},
        'refractory': refr, 'convergence': conv}
    receipt['cable'] = {
        'config': cable.config(), 'distance_um': vel['distance_um'], 'sites': vel['sites'],
        'rows': vel['rows'], 'velocities_dt_half_m_s': v_half, 'dt_half_relative_change': dt_rel,
        'log_log_slope': slope, 'expected_slope': 0.5,
        'reference': {**HH1952_REFERENCE_AXON, 'model_velocity_m_s': ref_v, 'relative_difference': ref_rel,
                      'applies': reference_applies}}
    receipt['neuroweave_reference'] = {
        'cell': 'synapse.neuroweave.cells.CellPopulation, flags {membrane_dynamics}, constant external drive, '
                'no recurrent input, neutral controls; time unit = one engine step',
        'threshold_drive': nth['threshold'], 'threshold_bracket': nth['bracket'],
        'threshold_seed': seed_list[0],
        'per_seed': [{'seed': r['seed'], 'rates_per_step': r['fi']['rates_per_step'],
                      'spike_counts': r['fi']['spike_counts']} for r in nw_runs],
        'seed_invariant': all(r['fi']['rates_per_step'] == nw['fi']['rates_per_step']
                              and r['fi']['spike_counts'] == nw['fi']['spike_counts'] for r in nw_runs),
        'comparison': cmp, 'adaptation_index': adaptation,
        'note': 'descriptive only: the abstract cell is compared in shape (threshold- and max-normalised), '
                'never validated against the HH reference'}
    receipt['checks'] = checks
    receipt['verdict'] = 'pass' if all(checks.values()) else 'fail'
    receipt['verdict_basis'] = ('all_known_answer_checks' if receipt['verdict'] == 'pass'
                                else 'failed_checks: ' + ', '.join(k for k, ok in checks.items() if not ok))
    receipt['estimated_cost'] = _estimated_cost(c, len(seed_list), refr)
    receipt['measured_wallclock_s'] = {**wall, 'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'Hodgkin-Huxley (1952) membrane on a uniform sealed-end cable, central differences with g_ax = a/(2 Ra dx^2), '
        'RK4 unless stated; checks use cases with known answers (documented rest, all-or-none response, '
        'sqrt(diameter) velocity scaling of a uniform cable, HH\'s own computed velocity of 18.8 m/s for their '
        'axon, time-step convergence, refractoriness). The NEUROWEAVE comparison is descriptive: an abstract '
        'discrete-time unit is compared in normalised shape only. Not a validated neuron simulator; no claim of '
        'biological fidelity for the NEUROWEAVE cell.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
