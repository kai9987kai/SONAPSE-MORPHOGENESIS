"""Anytime-valid inference (frontier module ``anytime_inference``).

Inspired by Ian Waudby-Smith and Aaditya Ramdas, *Estimating means of bounded random variables by
betting*, J. R. Stat. Soc. B 86(1):1-27 (February 2024), doi:10.1093/jrsssb/qkad009, and by Aaditya
Ramdas, Peter Grünwald, Vladimir Vovk and Glenn Shafer, *Game-theoretic statistics and safe
anytime-valid inference*, Statistical Science 38(4):576-601 (November 2023), doi:10.1214/23-STS894.
This module is a synthetic computational abstraction: it implements two constructions from the first
paper and checks their stated guarantees in seeded simulations; it reproduces none of the papers'
experiments. Full, verified references: ``docs/frontier/modules/anytime_inference.md``.

Betting confidence sequence (paper Sec. 4.4, Eqs. 24-26 and Theorem 3; section, equation and appendix
numbers follow arXiv:2010.09686v7, the version checked). Observations ``x_t`` in
``[lo, hi]`` are rescaled to ``X_t = (x_t - lo) / (hi - lo)`` in ``[0, 1]``; for each candidate mean ``m``
on a grid::

    K+_t(m) = prod_{i<=t} (1 + lam+_i(m) (X_i - m)),     K-_t(m) = prod_{i<=t} (1 - lam-_i(m) (X_i - m))
    K±_t(m) = max{theta K+_t(m), (1 - theta) K-_t(m)}
    lam+_t(m) = min(lam_t, c / m),   lam-_t(m) = min(lam_t, c / (1 - m))                       (Eq. 25)
    lam_t = sqrt(2 log(2/alpha) / (sigma2_{t-1} t log(1 + t)))                                (Eq. 26)
    mu_t = (1/2 + sum_{i<=t} X_i) / (t + 1),   sigma2_t = (1/4 + sum_{i<=t} (X_i - mu_i)^2) / (t + 1)
    CS_t = intersection_{i<=t} {m : K±_i(m) < 1/alpha}

The set is an interval (Theorem 3), so the grid search returns a conservative superset: each bound is
the first *excluded* grid point outside the included run (when no grid point is included, the excluded
neighbours of the grid minimisers of ``K±``, which bracket the set because ``K±`` is quasiconvex in ``m``).

Paired e-process for ``H0: E[D_t | D_1..D_{t-1}] = 0`` with ``|D_t| <= B``: with ``X_t = (1 + D_t/B)/2``
and ``m = 1/2``, ``E_t = (K+_t + K-_t) / 2`` (the paper's mixture ``M±`` with ``theta = 1/2``). Both
one-sided processes bet with the approximate-GRAPA rule (paper App. B.3) computed from the Eq. 26
estimators, split by sign: ``lam_t = clip((mu_{t-1} - m) / (sigma2_{t-1} + (mu_{t-1} - m)^2), -c/(1-m), c/m)``,
``lam+_t = max(lam_t, 0)``, ``lam-_t = max(-lam_t, 0)``. Every factor is at least ``1 - c > 0`` and the
bets are predictable, so under ``H0`` each one-sided process and their average are test martingales;
Ville's inequality gives ``P(exists t: E_t >= 1/alpha) <= alpha`` -- valid under optional stopping
(Ramdas et al. 2023).

All randomness comes from ``numpy.random.default_rng(seed + offset)``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import harness

__all__ = ['MODULE_ID', 'SOURCE', 'EXPERIMENTS', 'betting_confidence_sequence', 'paired_e_process',
           'anytime_compare', 'naive_peeking_t_test', 'run_experiment']

MODULE_ID = 'anytime_inference'
SOURCE = ('Waudby-Smith & Ramdas, "Estimating means of bounded random variables by betting", J. R. Stat. Soc. B '
          '86(1):1-27, February 2024, doi:10.1093/jrsssb/qkad009; Ramdas, Grünwald, Vovk & Shafer, '
          '"Game-theoretic statistics and safe anytime-valid inference", Statistical Science 38(4):576-601, '
          'November 2023, doi:10.1214/23-STS894')

_NULL_BERNOULLI_OFFSET = 83_003
_NULL_SKEWED_OFFSET = 83_011
_EFFECT_OFFSET = 83_021
_COVER_BETA_OFFSET = 83_029
_COVER_BERNOULLI_OFFSET = 83_041
_LOG_CAP = 690.0                 # reported e-values are capped at exp(690) ~ 1e299, so every value stays finite
_MAX_N = 1_000_000
_MAX_CS_WORK = 200_000_000       # observations x grid points per confidence-sequence call
_MAX_SIM_CELLS = 4_000_000       # replicates x horizon per simulated scenario (about 20 such float64 arrays at peak)
_SKEW_A, _SKEW_B = 0.5, 4.0      # skewed null: D = Beta(0.5, 4) - 1/9  (mean exactly 0, values in [-1/9, 8/9])


# ====================================================================== validation helpers
def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _num(name: str, value: Any, low: float, high: float, *, open_low: bool = False, open_high: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if (not math.isfinite(v) or v < low or v > high or (open_low and v <= low) or (open_high and v >= high)):
        raise ValueError(f'{name} must be a finite number in {"(" if open_low else "["}{low}, {high}{")" if open_high else "]"}')
    return v


def _series(name: str, x: Any) -> np.ndarray:
    if isinstance(x, (str, bytes)) or isinstance(x, Mapping):
        raise ValueError(f'{name} must be a 1-D sequence of finite numbers')
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a 1-D sequence of finite numbers') from exc
    if a.ndim != 1 or a.size == 0 or a.size > _MAX_N or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a non-empty 1-D sequence of at most {_MAX_N} finite numbers')
    return a.copy()


def _alpha(alpha: Any) -> float:
    return _num('alpha', alpha, 0.0, 1.0, open_low=True, open_high=True)


def _c(c: Any) -> float:
    return _num('c', c, 0.0, 1.0, open_low=True, open_high=True)


# ====================================================================== core (batched over rows)
def _plugin_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predictable ``mu_{t-1}`` and ``sigma2_{t-1}`` (Eq. 26 estimators, priors 1/2 and 1/4) along the last axis."""
    n = x.shape[-1]
    t = np.arange(1, n + 1, dtype=np.float64)
    mu = (0.5 + np.cumsum(x, axis=-1)) / (t + 1.0)
    sig = (0.25 + np.cumsum((x - mu) ** 2, axis=-1)) / (t + 1.0)
    lead = x.shape[:-1] + (1,)
    mu_prev = np.concatenate([np.full(lead, 0.5), mu[..., :-1]], axis=-1)
    sig_prev = np.concatenate([np.full(lead, 0.25), sig[..., :-1]], axis=-1)
    return mu_prev, sig_prev


def _hedged_bounds(x: np.ndarray, alpha: float, grid_size: int, c: float,
                   theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-time conservative bounds of ``{m : K±_t(m) < 1/alpha}`` for rows of ``x`` in [0, 1] (shape (R, n))."""
    rows, n = x.shape
    grid = np.linspace(0.0, 1.0, grid_size + 1)
    with np.errstate(divide='ignore'):
        cap_plus = np.where(grid > 0.0, c / np.where(grid > 0.0, grid, 1.0), np.inf)
        cap_minus = np.where(grid < 1.0, c / np.where(grid < 1.0, 1.0 - grid, 1.0), np.inf)
    _, sig_prev = _plugin_moments(x)
    t = np.arange(1, n + 1, dtype=np.float64)
    lam = np.sqrt(2.0 * math.log(2.0 / alpha) / (sig_prev * t * np.log1p(t)))
    log_kp = np.zeros((rows, grid.size))
    log_km = np.zeros((rows, grid.size))
    thresh = math.log(1.0 / alpha)
    log_theta, log_rest = math.log(theta), math.log(1.0 - theta)
    lower = np.empty((rows, n))
    upper = np.empty((rows, n))
    last = grid.size - 1
    rev = np.arange(grid.size)[::-1]
    for i in range(n):
        dev = x[:, i, None] - grid[None, :]
        log_kp += np.log1p(np.minimum(lam[:, i, None], cap_plus) * dev)
        log_km += np.log1p(-np.minimum(lam[:, i, None], cap_minus) * dev)
        log_k = np.maximum(log_theta + log_kp, log_rest + log_km)
        inside = log_k < thresh
        any_in = inside.any(axis=1)
        first = np.argmax(inside, axis=1)
        final = last - np.argmax(inside[:, rev], axis=1)
        # no grid point inside: bracket every grid minimiser (K± is quasiconvex in m, so the set lies there)
        best_first = np.argmin(log_k, axis=1)
        best_last = last - np.argmin(log_k[:, rev], axis=1)
        lo_idx = np.where(any_in, first, best_first) - 1
        hi_idx = np.where(any_in, final, best_last) + 1
        lower[:, i] = grid[np.clip(lo_idx, 0, last)]
        upper[:, i] = grid[np.clip(hi_idx, 0, last)]
    return lower, upper


def _paired_log_capital(y: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log K+, log K- and log E for rows of ``y = D / B`` in [-1, 1] (shape (R, n)); null mean m = 1/2 on X."""
    x = 0.5 * (1.0 + y)
    m = 0.5
    mu_prev, sig_prev = _plugin_moments(x)
    lam = np.clip((mu_prev - m) / (sig_prev + (mu_prev - m) ** 2), -c / (1.0 - m), c / m)
    lam_plus = np.maximum(lam, 0.0)
    lam_minus = np.maximum(-lam, 0.0)
    log_kp = np.cumsum(np.log1p(lam_plus * (x - m)), axis=-1)
    log_km = np.cumsum(np.log1p(-lam_minus * (x - m)), axis=-1)
    log_e = np.logaddexp(log_kp, log_km) - math.log(2.0)
    return log_kp, log_km, log_e


def _cap_exp(log_v: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(log_v, _LOG_CAP))


# ====================================================================== public API
def betting_confidence_sequence(x: Any, lo: float, hi: float, alpha: float = 0.05, *, grid_size: int = 1000,
                                c: float = 0.5, theta: float = 0.5) -> dict[str, Any]:
    """Hedged-capital betting confidence sequence for the mean of ``x`` (values in ``[lo, hi]``).

    Returns ``lower``/``upper`` (running intersection, length ``n``), ``lower_raw``/``upper_raw`` (the set at
    each time before intersecting), ``final`` = ``(lower[-1], upper[-1])`` and ``empty`` (the running
    intersection became empty, which happens with probability at most ``alpha``). Bounds are conservative to
    the grid resolution ``(hi - lo) / grid_size``."""
    a = _series('x', x)
    lo_v = _num('lo', lo, -1e12, 1e12)
    hi_v = _num('hi', hi, -1e12, 1e12)
    if not hi_v > lo_v:
        raise ValueError('hi must be greater than lo')
    if np.any(a < lo_v) or np.any(a > hi_v):
        raise ValueError('every observation must lie in [lo, hi]')
    al = _alpha(alpha)
    g = _int('grid_size', grid_size, 8, 100_000)
    cc = _c(c)
    th = _num('theta', theta, 0.0, 1.0, open_low=True, open_high=True)
    if a.size * (g + 1) > _MAX_CS_WORK:
        raise ValueError(f'len(x) * (grid_size + 1) must be at most {_MAX_CS_WORK}')
    width = hi_v - lo_v
    unit = np.clip((a - lo_v) / width, 0.0, 1.0)
    lower_raw, upper_raw = _hedged_bounds(unit[None, :], al, g, cc, th)
    lower = np.maximum.accumulate(lower_raw[0])
    upper = np.minimum.accumulate(upper_raw[0])
    scale = lambda v: lo_v + width * v  # noqa: E731
    return {'n': int(a.size), 'alpha': al, 'lo': lo_v, 'hi': hi_v, 'grid_size': g, 'c': cc, 'theta': th,
            'lower': scale(lower), 'upper': scale(upper), 'lower_raw': scale(lower_raw[0]),
            'upper_raw': scale(upper_raw[0]), 'final': (float(scale(lower[-1])), float(scale(upper[-1]))),
            'empty': bool(lower[-1] > upper[-1]), 'method': 'hedged_capital_predictable_plugin'}


def paired_e_process(differences: Any, bound: float, *, c: float = 0.5) -> dict[str, Any]:
    """Two-sided betting e-process for ``H0: E[D_t | past] = 0`` from paired differences ``|D_t| <= bound``.

    ``e_process[t]`` is ``(K+ + K-) / 2`` after ``t + 1`` observations; ``e_plus`` / ``e_minus`` are the one-sided
    processes (evidence for a positive / negative mean). Values are capped at ``exp(690)`` to stay finite."""
    d = _series('differences', differences)
    b = _num('bound', bound, 0.0, 1e12, open_low=True)
    if np.any(np.abs(d) > b):
        raise ValueError('every |difference| must be at most bound')
    cc = _c(c)
    log_kp, log_km, log_e = _paired_log_capital(np.clip(d / b, -1.0, 1.0)[None, :], cc)
    return {'n': int(d.size), 'bound': b, 'c': cc, 'e_process': _cap_exp(log_e[0]), 'log_e_process': log_e[0],
            'e_plus': _cap_exp(log_kp[0]), 'e_minus': _cap_exp(log_km[0]),
            'method': 'two_sided_mixture_of_one_sided_agrapa_bets'}


def anytime_compare(candidate: Any, reference: Any, *, bound: float, alpha: float = 0.05,
                    higher_is_better: bool = True, c: float = 0.5, grid_size: int = 1000) -> dict[str, Any]:
    """Anytime-valid paired comparison of two aligned score streams (``|candidate - reference| <= bound``).

    The e-process is monitored after every observation; ``rejected`` is true when it reaches ``1/alpha``
    (``stopped_at`` is that 1-based time and ``e_value`` the stopped value), so peeking never inflates the
    type-I error beyond ``alpha`` provided ``E[D_t | past] = 0`` under the null. The direction comes from the
    one-sided process that is at least ``1/alpha`` at the stopping time. ``cs_lower``/``cs_upper`` bound the
    mean difference ``candidate - reference`` at the last observation (betting CS, level ``1 - alpha``)."""
    a = _series('candidate', candidate)
    r = _series('reference', reference)
    if a.shape != r.shape:
        raise ValueError('candidate and reference must have the same length')
    if type(higher_is_better) is not bool:
        raise ValueError('higher_is_better must be a boolean')
    b = _num('bound', bound, 0.0, 1e12, open_low=True)
    al = _alpha(alpha)
    d = a - r
    if np.any(np.abs(d) > b):
        raise ValueError('every |candidate - reference| must be at most bound')
    ep = paired_e_process(d, b, c=c)
    cs = betting_confidence_sequence(d, -b, b, al, grid_size=grid_size, c=c)
    log_e = ep['log_e_process']
    crossed = np.flatnonzero(log_e >= math.log(1.0 / al))
    rejected = bool(crossed.size)
    stop = int(crossed[0]) if rejected else d.size - 1
    direction = None
    if rejected:
        direction = 'positive' if ep['e_plus'][stop] >= ep['e_minus'][stop] else 'negative'
    if not rejected:
        verdict = 'no_detectable_difference'
    else:
        verdict = 'improved' if (direction == 'positive') == higher_is_better else 'worse'
    return {'n': int(d.size), 'alpha': al, 'bound': b, 'higher_is_better': higher_is_better,
            'mean_difference': float(d.mean()), 'e_value': float(ep['e_process'][stop]),
            'e_value_final': float(ep['e_process'][-1]), 'rejected': rejected,
            'stopped_at': stop + 1 if rejected else None, 'direction': direction,
            'cs_lower': float(cs['lower'][-1]), 'cs_upper': float(cs['upper'][-1]),
            'cs_excludes_zero': bool(cs['lower'][-1] > 0.0 or cs['upper'][-1] < 0.0), 'cs_empty': cs['empty'],
            'verdict': verdict, 'method': 'betting_e_process_and_hedged_confidence_sequence'}


def naive_peeking_t_test(differences: Any, alpha: float = 0.05, *, min_n: int = 2) -> dict[str, Any]:
    """The invalid baseline: a two-sided one-sample t-test of mean 0 repeated after every observation.

    Returns whether any look at ``n >= min_n`` gave ``p < alpha`` and whether the single final look did."""
    d = _series('differences', differences)
    al = _alpha(alpha)
    mn = _int('min_n', min_n, 2, _MAX_N)
    any_look, final_look = _naive_t_rejections(d[None, :], al, mn)
    return {'n': int(d.size), 'alpha': al, 'min_n': mn, 'rejected_any_look': bool(any_look[0]),
            'rejected_final_look': bool(final_look[0])}


def _naive_t_rejections(d: np.ndarray, alpha: float, min_n: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import stdtrit                       # Student-t quantile (lighter import than scipy.stats)
    rows, n = d.shape
    if n < 2:
        return np.zeros(rows, dtype=bool), np.zeros(rows, dtype=bool)
    t = np.arange(1, n + 1, dtype=np.float64)
    s1 = np.cumsum(d, axis=1)
    s2 = np.cumsum(d * d, axis=1)
    mean = s1 / t
    with np.errstate(divide='ignore', invalid='ignore'):
        var = np.maximum(s2 - t * mean * mean, 0.0) / np.maximum(t - 1.0, 1.0)
        se = np.sqrt(var / t)
        stat = np.where(se > 0.0, np.abs(mean) / np.where(se > 0.0, se, 1.0), 0.0)
    crit = np.full(n, np.inf)
    df = np.arange(1, n)                                   # look at t = 2..n has df = t - 1
    crit[1:] = stdtrit(df.astype(np.float64), 1.0 - alpha / 2.0)
    looks = np.arange(1, n + 1) >= max(2, min_n)
    reject = (stat > crit[None, :]) & looks[None, :]
    return reject.any(axis=1), reject[:, -1]


# ====================================================================== validation experiment
_DEFAULTS: dict[str, Any] = {'alpha': 0.05, 'replicates': 400, 'horizon': 500, 'grid_size': 400, 'c': 0.5,
                             'theta': 0.5, 'naive_min_n': 10, 'null_p': 0.6, 'effect_p': 0.75, 'effect_reference_p': 0.5,
                             'cover_beta': (2.0, 5.0), 'cover_bernoulli_p': 0.3, 'power_min': 0.8, 'mc_z': 3.0}
_QUICK: dict[str, Any] = {'replicates': 100, 'horizon': 200, 'grid_size': 200}


def _seeds(seeds: Any) -> tuple[int, ...]:
    if isinstance(seeds, (str, bytes, Mapping)):
        raise ValueError('seeds must be an iterable of ints')
    try:
        out = tuple(_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


def _config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown run_experiment parameters: {unknown}')
    cfg = {**_DEFAULTS, **(_QUICK if quick else {}), **overrides}
    cfg['alpha'] = _alpha(cfg['alpha'])
    cfg['replicates'] = _int('replicates', cfg['replicates'], 10, 100_000)
    cfg['horizon'] = _int('horizon', cfg['horizon'], 20, 100_000)
    cfg['grid_size'] = _int('grid_size', cfg['grid_size'], 8, 100_000)
    cfg['c'] = _c(cfg['c'])
    cfg['theta'] = _num('theta', cfg['theta'], 0.0, 1.0, open_low=True, open_high=True)
    cfg['naive_min_n'] = _int('naive_min_n', cfg['naive_min_n'], 2, cfg['horizon'])
    for key in ('null_p', 'effect_p', 'effect_reference_p', 'cover_bernoulli_p'):
        cfg[key] = _num(key, cfg[key], 0.0, 1.0, open_low=True, open_high=True)
    if not cfg['effect_p'] > cfg['effect_reference_p']:
        raise ValueError('effect_p must exceed effect_reference_p (the effect scenario is a positive difference)')
    beta = cfg['cover_beta']
    if isinstance(beta, (str, bytes)) or len(tuple(beta)) != 2:
        raise ValueError('cover_beta must be a pair of positive numbers')
    cfg['cover_beta'] = tuple(_num('cover_beta', v, 0.0, 1000.0, open_low=True) for v in beta)
    cfg['power_min'] = _num('power_min', cfg['power_min'], 0.0, 1.0)
    cfg['mc_z'] = _num('mc_z', cfg['mc_z'], 0.0, 10.0)
    if cfg['replicates'] * (cfg['grid_size'] + 1) > 20_000_000:
        raise ValueError('replicates * (grid_size + 1) must be at most 20,000,000 (memory bound)')
    if cfg['replicates'] * cfg['horizon'] > _MAX_SIM_CELLS:
        raise ValueError(f'replicates * horizon must be at most {_MAX_SIM_CELLS:,} (memory bound)')
    return cfg


def _paired_bernoulli(rng: np.random.Generator, shape: tuple[int, int], p_cand: float, p_ref: float) -> np.ndarray:
    cand = (rng.random(shape) < p_cand).astype(np.float64)
    ref = (rng.random(shape) < p_ref).astype(np.float64)
    return cand - ref


def _seed_block(seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    r, n, al, cc = cfg['replicates'], cfg['horizon'], cfg['alpha'], cfg['c']
    thresh = math.log(1.0 / al)
    out: dict[str, Any] = {'seed': int(seed)}
    # ---- nulls: e-process under optional stopping vs repeated t-tests
    nulls = {
        'paired_bernoulli': _paired_bernoulli(np.random.default_rng(seed + _NULL_BERNOULLI_OFFSET), (r, n),
                                              cfg['null_p'], cfg['null_p']),
        'skewed_continuous': (np.random.default_rng(seed + _NULL_SKEWED_OFFSET).beta(_SKEW_A, _SKEW_B, (r, n))
                              - _SKEW_A / (_SKEW_A + _SKEW_B)),
    }
    for name, d in nulls.items():
        _, _, log_e = _paired_log_capital(d, cc)
        naive_any, naive_final = _naive_t_rejections(d, al, cfg['naive_min_n'])
        final_e = _cap_exp(log_e[:, -1])
        out[f'null_{name}'] = {
            'e_process_rejections': int(np.sum(log_e.max(axis=1) >= thresh)),
            'naive_peeking_rejections': int(naive_any.sum()), 'fixed_n_t_rejections': int(naive_final.sum()),
            'final_e_sum': float(final_e.sum()), 'final_e_sq_sum': float(np.sum(final_e ** 2)),
            'final_e_max': float(final_e.max()), 'replicates': r}
    # ---- power on a clear positive effect
    d = _paired_bernoulli(np.random.default_rng(seed + _EFFECT_OFFSET), (r, n), cfg['effect_p'],
                          cfg['effect_reference_p'])
    log_kp, log_km, log_e = _paired_log_capital(d, cc)
    hit = log_e >= thresh
    rejected = hit.any(axis=1)
    stop = np.where(rejected, np.argmax(hit, axis=1), n - 1)
    rows = np.arange(r)
    positive = log_kp[rows, stop] >= log_km[rows, stop]
    naive_any, _ = _naive_t_rejections(d, al, cfg['naive_min_n'])
    out['effect'] = {'replicates': r, 'rejections': int(rejected.sum()),
                     'wrong_direction_rejections': int(np.sum(rejected & ~positive)),
                     'stopping_times': (stop[rejected] + 1).tolist(),
                     'naive_peeking_rejections': int(naive_any.sum())}
    # ---- uniform-in-time coverage of the betting CS
    a_beta, b_beta = cfg['cover_beta']
    cover = {
        'beta': (np.random.default_rng(seed + _COVER_BETA_OFFSET).beta(a_beta, b_beta, (r, n)), a_beta / (a_beta + b_beta)),
        'bernoulli': ((np.random.default_rng(seed + _COVER_BERNOULLI_OFFSET).random((r, n))
                       < cfg['cover_bernoulli_p']).astype(np.float64), cfg['cover_bernoulli_p']),
    }
    for name, (x, mu) in cover.items():
        lower_raw, upper_raw = _hedged_bounds(x, al, cfg['grid_size'], cc, cfg['theta'])
        lower = np.maximum.accumulate(lower_raw, axis=1)
        upper = np.minimum.accumulate(upper_raw, axis=1)
        covered_all = np.all((lower <= mu) & (mu <= upper), axis=1)
        early = min(n - 1, 19)
        out[f'cover_{name}'] = {'replicates': r, 'true_mean': float(mu), 'covered_all_times': int(covered_all.sum()),
                                'width_t20_sum': float(np.sum(upper[:, early] - lower[:, early])),
                                'width_final_sum': float(np.sum(upper[:, -1] - lower[:, -1])),
                                'empty_final': int(np.sum(lower[:, -1] > upper[:, -1]))}
    return out


def _rate(k: int, n: int) -> dict[str, float]:
    p = k / n
    return {'rate': p, 'mc_se': math.sqrt(max(p * (1.0 - p), 1e-12) / n), 'count': int(k), 'n': int(n)}


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Seeded validation of the e-process and the betting CS under continuous monitoring.

    For every seed: ``replicates`` sequences of length ``horizon`` per scenario, with a look after every
    observation. Checks (pooled over seeds; ``mc_z`` Monte-Carlo standard errors of slack):

    * type-I error of the e-process <= ``alpha`` under two nulls (paired Bernoulli, skewed continuous);
    * the naive repeated two-sided t-test (looks from ``naive_min_n``) exceeds ``alpha`` clearly;
    * the mean final e-value under each null is <= 1 + slack (it is a test martingale with expectation exactly 1;
      the terminal value is heavy-tailed, so the sample mean usually falls below 1 and the check is one-sided);
    * the running-intersection CS covers the true mean at every time with frequency >= ``1 - alpha``
      (Beta and Bernoulli data);
    * power on a clear effect (paired Bernoulli ``effect_p`` vs ``effect_reference_p``) >= ``power_min`` and
      wrong-direction rejections <= ``alpha``."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    cfg = _config(quick, overrides)
    t0 = time.perf_counter()
    blocks = [_seed_block(s, cfg) for s in seed_list]
    wall = time.perf_counter() - t0
    al, z = cfg['alpha'], cfg['mc_z']
    total = cfg['replicates'] * len(seed_list)
    results: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for name in ('paired_bernoulli', 'skewed_continuous'):
        key = f'null_{name}'
        e_rej = sum(b[key]['e_process_rejections'] for b in blocks)
        naive = sum(b[key]['naive_peeking_rejections'] for b in blocks)
        fixed = sum(b[key]['fixed_n_t_rejections'] for b in blocks)
        e_sum = sum(b[key]['final_e_sum'] for b in blocks)
        e_sq = sum(b[key]['final_e_sq_sum'] for b in blocks)
        e_mean = e_sum / total
        e_sd = math.sqrt(max(e_sq / total - e_mean ** 2, 0.0) * total / max(total - 1, 1))
        alpha_se = math.sqrt(al * (1.0 - al) / total)
        results[key] = {'e_process_type1': _rate(e_rej, total), 'naive_peeking_t_type1': _rate(naive, total),
                        'fixed_n_t_type1': _rate(fixed, total),
                        'final_e_value': {'mean': e_mean, 'sd': e_sd, 'mc_se': e_sd / math.sqrt(total),
                                          'max': max(b[key]['final_e_max'] for b in blocks)}}
        checks[f'e_process_type1_le_alpha_{name}'] = bool(e_rej / total <= al + z * alpha_se)
        naive_rate = naive / total
        naive_se = math.sqrt(max(naive_rate * (1.0 - naive_rate), 1e-12) / total)
        checks[f'naive_peeking_t_exceeds_alpha_{name}'] = bool(naive_rate - z * naive_se > al)
        checks[f'mean_final_e_value_le_1_{name}'] = bool(e_mean <= 1.0 + z * e_sd / math.sqrt(total))
    rej = sum(b['effect']['rejections'] for b in blocks)
    wrong = sum(b['effect']['wrong_direction_rejections'] for b in blocks)
    times = sorted(t for b in blocks for t in b['effect']['stopping_times'])
    results['effect'] = {'power': _rate(rej, total), 'wrong_direction': _rate(wrong, total),
                         'naive_peeking_t_rejections': _rate(sum(b['effect']['naive_peeking_rejections'] for b in blocks), total),
                         'median_stopping_time': float(np.median(times)) if times else None,
                         'mean_difference': cfg['effect_p'] - cfg['effect_reference_p']}
    checks['power_on_clear_effect'] = bool(rej / total >= cfg['power_min'])
    checks['wrong_direction_rate_le_alpha'] = bool(wrong / total <= al)
    for name in ('beta', 'bernoulli'):
        key = f'cover_{name}'
        cov = sum(b[key]['covered_all_times'] for b in blocks)
        results[key] = {'uniform_coverage': _rate(cov, total), 'true_mean': blocks[0][key]['true_mean'],
                        'mean_width_t20': sum(b[key]['width_t20_sum'] for b in blocks) / total,
                        'mean_width_final': sum(b[key]['width_final_sum'] for b in blocks) / total,
                        'empty_final': int(sum(b[key]['empty_final'] for b in blocks))}
        checks[f'cs_uniform_coverage_ge_1_minus_alpha_{name}'] = bool(
            cov / total >= 1.0 - al - z * math.sqrt(al * (1.0 - al) / total))
        checks[f'cs_shrinks_{name}'] = bool(results[key]['mean_width_final'] < results[key]['mean_width_t20'])
    n, g = cfg['horizon'], cfg['grid_size']
    receipt = harness.base_receipt(MODULE_ID, 'validation', {
        **cfg, 'seeds': list(seed_list), 'quick': quick, 'looks': 'after every observation',
        'nulls': {'paired_bernoulli': f'D = Bern({cfg["null_p"]}) - Bern({cfg["null_p"]}), independent',
                  'skewed_continuous': f'D = Beta({_SKEW_A}, {_SKEW_B}) - {_SKEW_A / (_SKEW_A + _SKEW_B):.6f}'},
        'effect': f'D = Bern({cfg["effect_p"]}) - Bern({cfg["effect_reference_p"]}), independent, bound 1',
        'coverage': {'beta': f'X ~ Beta{tuple(cfg["cover_beta"])} on [0, 1]',
                     'bernoulli': f'X ~ Bern({cfg["cover_bernoulli_p"]})'},
        'e_process': 'mean of one-sided aGRAPA betting processes on X = (1 + D)/2 at m = 1/2; reject when E_t >= 1/alpha',
        'confidence_sequence': 'hedged capital (theta, c) with predictable plug-in bets; grid search; running intersection',
        'naive_baseline': 'two-sided one-sample t-test at every look n >= naive_min_n'})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['per_seed'] = [{k: v for k, v in b.items() if k != 'effect'} | {'effect': {
        kk: vv for kk, vv in b['effect'].items() if kk != 'stopping_times'}} for b in blocks]
    receipt['results'] = results
    receipt['checks'] = checks
    receipt['verdict'] = 'pass' if all(checks.values()) else 'fail'
    receipt['verdict_basis'] = 'all_checks_pass' if receipt['verdict'] == 'pass' else 'failed_checks: ' + ', '.join(
        k for k, v in checks.items() if not v)
    receipt['estimated_cost'] = {'e_process_updates': 3 * total * n, 't_test_looks': 3 * total * n,
                                 'cs_capital_updates': 2 * 2 * total * n * (g + 1)}
    receipt['measured_wallclock_s'] = wall
    receipt['notes'] = ('Synthetic Monte-Carlo validation of two constructions from Waudby-Smith & Ramdas (2024): the '
                        'hedged-capital confidence sequence with predictable plug-in bets (grid search, conservative to '
                        'one grid step) and a two-sided e-process (mean of two one-sided aGRAPA betting processes). '
                        'Guarantees hold when E[D_t | past] = 0 (for the e-process) and observations are bounded with a '
                        'constant conditional mean (for the CS); serially dependent streams whose conditional mean '
                        'drifts are not covered. Rates are Monte-Carlo estimates with stated standard errors. The '
                        'terminal e-value under a null is heavy-tailed (rare large values carry its expectation of '
                        '1), so its sample mean usually falls below 1 and its naive standard error understates the '
                        'spread; the exact expectation is checked by path enumeration in the tests. Nothing here '
                        'reproduces the papers\' experiments.')
    return harness.seal(receipt)


EXPERIMENTS = {MODULE_ID: run_experiment}
