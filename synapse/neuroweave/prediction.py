"""Hierarchical multi-horizon prediction and an action-grounded efference self-model.

``PredictiveHierarchy`` keeps, for every horizon ``h`` (default 1, 4, 16, 64
steps), an ensemble of bootstrap recursive-least-squares (RLS) linear
predictors mapping the current state (plus an optional action) to the state
``h`` steps ahead. Learning is strictly *prequential*: a prediction made at
time ``t`` is stored and only scored (and only used as a training pair) when
the observation at ``t + h`` arrives, so no future information leaks into a
prediction.

``EfferenceSelfModel`` compares intended, issued, predicted and observed
action outcomes with an online linear forward model. Both are synthetic
engineering abstractions: they make no claim of biological fidelity.
"""
from __future__ import annotations

import base64
from collections import deque
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_HORIZONS: tuple[int, ...] = (1, 4, 16, 64)
PREDICTION_SEED_OFFSET = 4101
EFFERENCE_SEED_OFFSET = 4202

GATE_MIN = 0.58
GATE_MAX = 1.0
W_BOUND = 25.0
P_TRACE_MAX = 1.0e5
SURPRISE_MAX = 10.0
VOLATILITY_MAX = 10.0
STATE_BOUND = 1.0e6
ERROR_CLIP = 10.0

# Trace rates for the normalised prediction error.
FAST_RATE = 0.3
MEDIUM_RATE = 0.05
SLOW_RATE = 0.005
SURPRISE_RATE = 0.3
DISAGREEMENT_RATE = 0.1
STAT_RATE = 0.02


def _as_vector(name: str, value: Any, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f'{name} must be a 1-D vector of length {dim}, got shape {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    if np.any(np.abs(arr) > STATE_BOUND):
        raise ValueError(f'{name} exceeds the magnitude bound {STATE_BOUND}')
    return arr


# Arrays with more elements than this are exported as base64 little-endian float64
# (still pure JSON data) so a dim=512 hierarchy exports in well under a second.
PACKED_MIN_SIZE = 16384


def _pack(arr: np.ndarray) -> Any:
    """JSON-safe array encoding: nested lists when small, base64 '<f8' bytes when large."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size < PACKED_MIN_SIZE:
        return a.tolist()
    return {'dtype': '<f8', 'shape': list(a.shape),
            'b64': base64.b64encode(np.ascontiguousarray(a, dtype='<f8').tobytes()).decode('ascii')}


def _unpack(name: str, value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    try:
        if value.get('dtype') != '<f8' or set(value) != {'dtype', 'shape', 'b64'}:
            raise ValueError('packed array must be {dtype: "<f8", shape, b64}')
        shape = tuple(int(v) for v in value['shape'])
        raw = base64.b64decode(value['b64'].encode('ascii'), validate=True)
        n = int(np.prod(shape)) if shape else 1
        if len(raw) != 8 * n:
            raise ValueError('packed array byte length does not match shape')
        return np.frombuffer(raw, dtype='<f8').reshape(shape).astype(np.float64)
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f'{name}: invalid packed array: {exc}') from exc


def _matrix(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        arr = np.asarray(_unpack(name, value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not numeric: {exc}') from exc
    if arr.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    return arr


def _finite_float(name: str, value: Any, low: float = -math.inf, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if not math.isfinite(v) or v < low or v > high:
        raise ValueError(f'{name}={value!r} outside [{low}, {high}]')
    return v


def _check_P(name: str, P: np.ndarray) -> None:
    """RLS inverse-covariance sanity: symmetric, non-negative diagonal, bounded trace."""
    if np.any(np.diag(P) < 0) or float(np.trace(P)) > P_TRACE_MAX * (1 + 1e-9):
        raise ValueError(f'{name} must have a non-negative diagonal and trace <= {P_TRACE_MAX}')
    if not np.allclose(P, P.T, rtol=1e-9, atol=1e-9 * max(1.0, float(np.abs(P).max()))):
        raise ValueError(f'{name} must be symmetric')


def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or value > high:
        raise ValueError(f'{name}={value!r} must be an int in [{low}, {high}]')
    return int(value)


def _rls_update(W: np.ndarray, P: np.ndarray, phi: np.ndarray, y: np.ndarray,
                weight: float, forgetting: float) -> None:
    """In-place weighted exponentially-forgetting RLS step with bounds."""
    Pphi = P @ phi
    denom = forgetting / weight + float(phi @ Pphi)
    K = Pphi / denom
    err = y - W @ phi
    W += np.outer(err, K)
    P -= np.outer(K, Pphi)
    P /= forgetting
    P[...] = 0.5 * (P + P.T)
    tr = float(np.trace(P))
    if not math.isfinite(tr) or tr > P_TRACE_MAX:
        P[...] = np.eye(P.shape[0]) * (P_TRACE_MAX / P.shape[0]) if not math.isfinite(tr) else P * (P_TRACE_MAX / tr)
    np.clip(W, -W_BOUND, W_BOUND, out=W)


class PredictiveHierarchy:
    """Ensembles of bootstrap RLS predictors over several prediction horizons."""

    def __init__(self, dim: int, horizons: Sequence[int] = DEFAULT_HORIZONS, ensemble: int = 3,
                 seed: int = 7, *, action_dim: int = 0, forgetting: float = 0.995,
                 p_init: float = 100.0, init_scale: float = 0.05) -> None:
        self.dim = _int('dim', dim, 1, 4096)
        hs = tuple(int(h) for h in horizons)
        if not hs or any(h < 1 or h > 4096 for h in hs) or list(hs) != sorted(set(hs)):
            raise ValueError('horizons must be strictly increasing positive ints')
        self.horizons = hs
        self.ensemble = _int('ensemble', ensemble, 1, 32)
        self.seed = _int('seed', seed, 0, 2**31 - 1)
        self.action_dim = _int('action_dim', action_dim, 0, 4096)
        self.forgetting = _finite_float('forgetting', forgetting, 0.5, 1.0)
        self.p_init = _finite_float('p_init', p_init, 1e-6, 1e6)
        self.init_scale = _finite_float('init_scale', init_scale, 0.0, 10.0)
        self.feature_dim = self.dim + self.action_dim + 1
        self._rng = np.random.default_rng(self.seed + PREDICTION_SEED_OFFSET)
        F = self.feature_dim
        # Distinct small random initial weights per member give a non-degenerate ensemble.
        self.W = {h: [self._rng.normal(0.0, self.init_scale, size=(self.dim, F)) for _ in range(self.ensemble)]
                  for h in hs}
        self.P = {h: [np.eye(F) * self.p_init for _ in range(self.ensemble)] for h in hs}
        self.step_count = 0
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max(hs))
        self.state_mean = np.zeros(self.dim)
        self.state_var = np.zeros(self.dim)
        self.state_count = 0
        zeros = {h: 0.0 for h in hs}
        self.error_fast = dict(zeros)
        self.error_medium = dict(zeros)
        self.error_slow = dict(zeros)
        self.error_stat_mean = dict(zeros)
        self.error_stat_var = dict(zeros)
        self.surprise = dict(zeros)
        self.surprise_trace = dict(zeros)
        self.disagreement = dict(zeros)
        self.last_error = dict(zeros)
        self.scored = {h: 0 for h in hs}

    # ------------------------------------------------------------------ helpers
    def _features(self, state: Any, action: Any = None) -> np.ndarray:
        x = _as_vector('state', state, self.dim)
        if self.action_dim:
            a = np.zeros(self.action_dim) if action is None else _as_vector('action', action, self.action_dim)
        else:
            if action is not None and np.asarray(action).size:
                raise ValueError('action supplied but action_dim == 0')
            a = np.zeros(0)
        return np.concatenate([x, a, [1.0]])

    def _scale(self) -> float:
        return max(float(np.sqrt(np.mean(self.state_var))), 1e-3)

    def _ensemble_predict(self, h: int, phi: np.ndarray) -> np.ndarray:
        return np.stack([W @ phi for W in self.W[h]])

    def _score(self, h: int, pred: np.ndarray, target: np.ndarray) -> float:
        err = float(np.sqrt(np.mean((pred - target) ** 2)))
        norm = min(err / self._scale(), ERROR_CLIP)
        m, v = self.error_stat_mean[h], self.error_stat_var[h]
        if self.scored[h] >= 2:
            z = (norm - m) / (math.sqrt(v) + 1e-6)
            s = float(np.clip(z, 0.0, SURPRISE_MAX))
        else:
            s = 0.0
        self.surprise[h] = s
        self.surprise_trace[h] += SURPRISE_RATE * (s - self.surprise_trace[h])
        rate = max(STAT_RATE, 1.0 / (self.scored[h] + 1))
        d = norm - m
        self.error_stat_mean[h] = m + rate * d
        self.error_stat_var[h] = min((1 - rate) * (v + rate * d * d), 1e6)
        n_inv = 1.0 / (self.scored[h] + 1)  # count-based warm-up removes initial-value bias
        self.error_fast[h] += max(FAST_RATE, n_inv) * (norm - self.error_fast[h])
        self.error_medium[h] += max(MEDIUM_RATE, n_inv) * (norm - self.error_medium[h])
        self.error_slow[h] += max(SLOW_RATE, n_inv) * (norm - self.error_slow[h])
        self.last_error[h] = norm
        self.scored[h] += 1
        return norm

    def _learn(self, h: int, phi: np.ndarray, target: np.ndarray) -> None:
        # Online bootstrap: each member sees the sample Poisson(1) times (capped).
        counts = np.minimum(self._rng.poisson(1.0, size=self.ensemble), 4)
        for m in range(self.ensemble):
            if counts[m] > 0:
                _rls_update(self.W[h][m], self.P[h][m], phi, target, float(counts[m]), self.forgetting)

    def _update_state_stats(self, x: np.ndarray) -> None:
        self.state_count += 1
        rate = max(1.0 / self.state_count, STAT_RATE)
        d = x - self.state_mean
        self.state_mean = self.state_mean + rate * d
        self.state_var = np.minimum((1 - rate) * (self.state_var + rate * d * d), 1e12)

    # ------------------------------------------------------------------ public
    def forecast(self, state: Any, action: Any = None) -> dict[int, dict[str, Any]]:
        """Pure forecast: per horizon ensemble mean and disagreement (std)."""
        phi = self._features(state, action)
        out: dict[int, dict[str, Any]] = {}
        for h in self.horizons:
            preds = self._ensemble_predict(h, phi)
            std = preds.std(axis=0)
            out[h] = {'mean': preds.mean(axis=0), 'disagreement': float(std.mean()),
                      'disagreement_vector': std, 'normalised_disagreement': float(std.mean()) / self._scale()}
        return out

    def observe(self, state: Any, action: Any = None, *, learn: bool = True) -> dict[str, Any]:
        """Stream one observation: score/learn matured predictions, then predict ahead."""
        phi = self._features(state, action)
        x = phi[:self.dim]
        self._update_state_stats(x)
        n = len(self._buffer)
        scored: dict[int, float] = {}
        for h in self.horizons:
            if n >= h:
                entry = self._buffer[n - h]
                if entry['step'] != self.step_count - h:  # pragma: no cover - structural guard
                    raise RuntimeError('prediction buffer out of sync')
                scored[h] = self._score(h, entry['preds'][h], x)
                if learn:
                    self._learn(h, entry['phi'], x)
        preds = {}
        for h in self.horizons:
            ens = self._ensemble_predict(h, phi)
            preds[h] = ens.mean(axis=0)
            dn = float(ens.std(axis=0).mean()) / self._scale()
            self.disagreement[h] += DISAGREEMENT_RATE * (min(dn, 1e3) - self.disagreement[h])
        self._buffer.append({'step': self.step_count, 'phi': phi, 'preds': preds})
        self.step_count += 1
        return {'step': self.step_count - 1, 'scored': scored, 'surprise': dict(self.surprise),
                'horizon': self.adaptive_horizon(), 'gate': self.uncertainty_gate()}

    def update(self, state: Any, target: Any, *, horizon: int = 1, action: Any = None) -> float:
        """Direct supervised step for one horizon; returns the normalised pre-update error."""
        if horizon not in self.W:
            raise ValueError(f'unknown horizon {horizon}')
        phi = self._features(state, action)
        y = _as_vector('target', target, self.dim)
        # Targets are observations: they drive the running state scale used to
        # normalise errors (otherwise update()-only use saturates at ERROR_CLIP).
        self._update_state_stats(y)
        ens = self._ensemble_predict(horizon, phi)
        dn = float(ens.std(axis=0).mean()) / self._scale()
        self.disagreement[horizon] += DISAGREEMENT_RATE * (min(dn, 1e3) - self.disagreement[horizon])
        err = self._score(horizon, ens.mean(axis=0), y)
        self._learn(horizon, phi, y)
        return err

    def counterfactual(self, state: Any, actions: Iterable[Any]) -> list[dict[str, Any]]:
        """Forecast outcomes of alternative actions without mutating any state."""
        if not self.action_dim:
            raise ValueError('counterfactual prediction requires action_dim > 0')
        results = []
        for i, a in enumerate(actions):
            fc = self.forecast(state, a)
            results.append({'index': i, 'action': np.asarray(a, dtype=np.float64).tolist(),
                            'forecast': {h: {'mean': v['mean'].tolist(), 'disagreement': v['disagreement']}
                                         for h, v in fc.items()}})
        return results

    def _horizon(self, horizon: int | None) -> int:
        h = self.horizons[0] if horizon is None else horizon
        if h not in self.W:
            raise ValueError(f'unknown horizon {horizon!r}')
        return h

    def volatility(self, horizon: int | None = None) -> float:
        h = self._horizon(horizon)
        slow = self.error_slow[h]
        return float(np.clip(self.error_fast[h] / (slow + 1e-6) - 1.0, 0.0, VOLATILITY_MAX))

    def adaptive_horizon(self, threshold: float = 0.35, *, surprise_threshold: float = 2.0,
                         volatility_threshold: float = 0.75, min_samples: int = 8) -> int:
        """Longest horizon whose error + disagreement is under ``threshold``,
        stepped down under high surprise or error volatility."""
        idx = 0
        for i, h in enumerate(self.horizons):
            if self.scored[h] < min_samples:
                continue
            u = max(self.error_medium[h], self.error_fast[h]) + self.disagreement[h]
            if u <= threshold:
                idx = i
        h0 = self.horizons[0]
        steps_down = int(self.surprise_trace[h0] > surprise_threshold) + int(self.volatility(h0) > volatility_threshold)
        return self.horizons[max(0, idx - steps_down)]

    def uncertainty_gate(self, horizon: int | None = None) -> float:
        """Gate in [0.58, 1]: 1 with no disagreement, lower as the ensemble disagrees."""
        h = self._horizon(horizon)
        return float(np.clip(GATE_MAX - (GATE_MAX - GATE_MIN) * min(self.disagreement[h], 1.0), GATE_MIN, GATE_MAX))

    def metrics(self) -> dict[str, Any]:
        return {str(h): {'error_ema': self.error_medium[h], 'fast': self.error_fast[h], 'medium': self.error_medium[h],
                         'slow': self.error_slow[h], 'surprise': self.surprise[h], 'surprise_trace': self.surprise_trace[h],
                         'disagreement': self.disagreement[h], 'scored': self.scored[h]} for h in self.horizons}

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        per = {}
        for h in self.horizons:
            per[str(h)] = {
                'W': [_pack(w) for w in self.W[h]], 'P': [_pack(p) for p in self.P[h]],
                'error_fast': self.error_fast[h], 'error_medium': self.error_medium[h], 'error_slow': self.error_slow[h],
                'error_stat_mean': self.error_stat_mean[h], 'error_stat_var': self.error_stat_var[h],
                'surprise': self.surprise[h], 'surprise_trace': self.surprise_trace[h],
                'disagreement': self.disagreement[h], 'last_error': self.last_error[h], 'scored': self.scored[h],
            }
        return {
            'schema': 'NeuroWeavePredictiveHierarchyV1',
            'config': {'dim': self.dim, 'horizons': list(self.horizons), 'ensemble': self.ensemble, 'seed': self.seed,
                       'action_dim': self.action_dim, 'forgetting': self.forgetting, 'p_init': self.p_init,
                       'init_scale': self.init_scale},
            'step': self.step_count,
            'state_stats': {'mean': self.state_mean.tolist(), 'var': self.state_var.tolist(), 'count': self.state_count},
            'horizons': per,
            'buffer': [{'step': e['step'], 'phi': _pack(e['phi']),
                        'preds': {str(h): e['preds'][h].tolist() for h in self.horizons}} for e in self._buffer],
            'rng': _rng_state(self._rng),
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('schema') != 'NeuroWeavePredictiveHierarchyV1':
            raise ValueError('not a NeuroWeavePredictiveHierarchyV1 state')
        cfg = state.get('config')
        expected = {'dim': self.dim, 'horizons': list(self.horizons), 'ensemble': self.ensemble,
                    'action_dim': self.action_dim}
        if not isinstance(cfg, Mapping) or any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError('state config does not match this hierarchy')
        seed = _int('seed', cfg.get('seed'), 0, 2**31 - 1)
        forgetting = _finite_float('forgetting', cfg.get('forgetting'), 0.5, 1.0)
        p_init = _finite_float('p_init', cfg.get('p_init'), 1e-6, 1e6)
        init_scale = _finite_float('init_scale', cfg.get('init_scale'), 0.0, 10.0)
        step = _int('step', state.get('step'), 0, 2**62)
        ss = state.get('state_stats')
        if not isinstance(ss, Mapping):
            raise ValueError('state_stats missing')
        s_mean = _matrix('state_stats.mean', ss.get('mean'), (self.dim,))
        s_var = _matrix('state_stats.var', ss.get('var'), (self.dim,))
        if np.any(s_var < 0) or np.any(s_var > 1e12) or np.any(np.abs(s_mean) > STATE_BOUND):
            raise ValueError('state_stats out of bounds')
        s_count = _int('state_stats.count', ss.get('count'), 0, 2**62)
        F = self.feature_dim
        per = state.get('horizons')
        if not isinstance(per, Mapping):
            raise ValueError('horizons missing')
        new: dict[int, dict[str, Any]] = {}
        for h in self.horizons:
            d = per.get(str(h))
            if not isinstance(d, Mapping):
                raise ValueError(f'horizon {h} missing')
            Ws, Ps = d.get('W'), d.get('P')
            if not isinstance(Ws, list) or not isinstance(Ps, list) or len(Ws) != self.ensemble or len(Ps) != self.ensemble:
                raise ValueError(f'horizon {h}: ensemble size mismatch')
            rec = {'W': [_matrix(f'W[{h}]', w, (self.dim, F)) for w in Ws],
                   'P': [_matrix(f'P[{h}]', p, (F, F)) for p in Ps]}
            if any(np.any(np.abs(w) > W_BOUND) for w in rec['W']):
                raise ValueError('W outside bounds')
            for i, Pm in enumerate(rec['P']):
                _check_P(f'P[{h}][{i}]', Pm)
            for key, low, high in (('error_fast', 0, ERROR_CLIP), ('error_medium', 0, ERROR_CLIP),
                                   ('error_slow', 0, ERROR_CLIP), ('error_stat_mean', 0, ERROR_CLIP), ('error_stat_var', 0, 1e6),
                                   ('surprise', 0, SURPRISE_MAX), ('surprise_trace', 0, SURPRISE_MAX),
                                   ('disagreement', 0, 1e3), ('last_error', 0, ERROR_CLIP)):
                rec[key] = _finite_float(f'{key}[{h}]', d.get(key), low, high)
            rec['scored'] = _int(f'scored[{h}]', d.get('scored'), 0, 2**62)
            new[h] = rec
        buf = state.get('buffer')
        if not isinstance(buf, list) or len(buf) > max(self.horizons):
            raise ValueError('buffer malformed')
        entries = []
        for i, e in enumerate(buf):
            if not isinstance(e, Mapping):
                raise ValueError('buffer entry malformed')
            es = _int('buffer.step', e.get('step'), 0, 2**62)
            if es != step - len(buf) + i:
                raise ValueError('buffer steps are not contiguous with step')
            phi = _matrix('buffer.phi', e.get('phi'), (F,))
            if phi[-1] != 1.0 or np.any(np.abs(phi) > STATE_BOUND):
                raise ValueError('buffer.phi must be bounded with a trailing bias of 1')
            preds_raw = e.get('preds')
            if not isinstance(preds_raw, Mapping):
                raise ValueError('buffer.preds malformed')
            preds = {h: _matrix('buffer.preds', preds_raw.get(str(h)), (self.dim,)) for h in self.horizons}
            if any(np.any(np.abs(v) > 1e12) for v in preds.values()):
                raise ValueError('buffer.preds out of bounds')
            entries.append({'step': es, 'phi': phi, 'preds': preds})
        rng = _restore_rng(state.get('rng'))
        # ---- all validated: assign
        self.seed, self.forgetting, self.p_init, self.init_scale = seed, forgetting, p_init, init_scale
        self.step_count = step
        self.state_mean, self.state_var, self.state_count = s_mean, s_var, s_count
        for h, rec in new.items():
            self.W[h], self.P[h] = rec['W'], rec['P']
            self.error_fast[h], self.error_medium[h], self.error_slow[h] = rec['error_fast'], rec['error_medium'], rec['error_slow']
            self.error_stat_mean[h], self.error_stat_var[h] = rec['error_stat_mean'], rec['error_stat_var']
            self.surprise[h], self.surprise_trace[h] = rec['surprise'], rec['surprise_trace']
            self.disagreement[h], self.last_error[h], self.scored[h] = rec['disagreement'], rec['last_error'], rec['scored']
        self._buffer = deque(entries, maxlen=max(self.horizons))
        self._rng = rng


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    st = rng.bit_generator.state
    return {'bit_generator': st['bit_generator'],
            'state': {'state': str(st['state']['state']), 'inc': str(st['state']['inc'])},
            'has_uint32': int(st['has_uint32']), 'uinteger': int(st['uinteger'])}


def _restore_rng(data: Any) -> np.random.Generator:
    try:
        if not isinstance(data, Mapping) or data.get('bit_generator') != 'PCG64':
            raise ValueError('rng must be a PCG64 state')
        st = {'bit_generator': 'PCG64',
              'state': {'state': int(data['state']['state']), 'inc': int(data['state']['inc'])},
              'has_uint32': int(data['has_uint32']), 'uinteger': int(data['uinteger'])}
        rng = np.random.default_rng(0)
        rng.bit_generator.state = st
        return rng
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'invalid rng state: {exc}') from exc


class EfferenceSelfModel:
    """Compares intended / issued / predicted / observed action outcomes.

    A linear forward model ``outcome ~ G @ issued + b`` (initialised to the
    identity) is learned online by RLS. ``record`` reports:

    * ``authorship`` in [0, 1]: agreement of the observed outcome with the
      outcome predicted from the issued command.
    * ``disturbance``: RMS of the residual component orthogonal to the issued
      command (not explainable by it).
    * ``actuator_slip``: RMS(intended - issued) plus the gain mismatch between
      observed and expected gain along the issued direction.
    * ``prediction_error``: RMS(observed - externally supplied prediction).
    * ``control_reliability``: EMA of authorship discounted by slip.
    """

    def __init__(self, dim: int, *, seed: int = 7, forgetting: float = 0.99, error_scale: float = 0.25,
                 reliability_rate: float = 0.1, p_init: float = 10.0) -> None:
        self.dim = _int('dim', dim, 1, 4096)
        self.seed = _int('seed', seed, 0, 2**31 - 1)
        self.forgetting = _finite_float('forgetting', forgetting, 0.5, 1.0)
        self.error_scale = _finite_float('error_scale', error_scale, 1e-6, 1e6)
        self.reliability_rate = _finite_float('reliability_rate', reliability_rate, 1e-4, 1.0)
        self.p_init = _finite_float('p_init', p_init, 1e-6, 1e6)
        # Seeded tiny jitter on the bias keeps the RNG contract without biasing the identity prior.
        rng = np.random.default_rng(self.seed + EFFERENCE_SEED_OFFSET)
        self.G = np.hstack([np.eye(self.dim), rng.normal(0.0, 1e-6, size=(self.dim, 1))])
        self.P = np.eye(self.dim + 1) * self.p_init
        self.control_reliability = 1.0
        self.authorship_ema = 1.0
        self.disturbance_ema = 0.0
        self.slip_ema = 0.0
        self.records = 0
        self.last: dict[str, Any] = {}

    def forward(self, issued: Any) -> np.ndarray:
        u = _as_vector('issued', issued, self.dim)
        return self.G @ np.concatenate([u, [1.0]])

    def record(self, intended: Any, issued: Any, predicted_outcome: Any, observed_outcome: Any,
               *, learn: bool = True) -> dict[str, Any]:
        it = _as_vector('intended', intended, self.dim)
        u = _as_vector('issued', issued, self.dim)
        pred_ext = _as_vector('predicted_outcome', predicted_outcome, self.dim)
        obs = _as_vector('observed_outcome', observed_outcome, self.dim)
        phi = np.concatenate([u, [1.0]])
        expected = self.G @ phi
        residual = obs - expected
        rms = lambda v: float(np.sqrt(np.mean(v * v)))  # noqa: E731
        forward_error = rms(residual)
        authorship = float(np.clip(math.exp(-forward_error / self.error_scale), 0.0, 1.0))
        uu = float(u @ u)
        if uu > 1e-12:
            par = (float(residual @ u) / uu) * u
            g_obs = float(obs @ u) / uu
            g_exp = float(expected @ u) / uu
            gain_mismatch = abs(g_obs - g_exp)
        else:
            par = np.zeros(self.dim)
            gain_mismatch = 0.0
        perp = residual - par
        disturbance = rms(perp)
        command_slip = rms(it - u)
        actuator_slip = command_slip + gain_mismatch
        prediction_error = rms(obs - pred_ext)
        slip_penalty = min(actuator_slip / self.error_scale, 1.0)
        target_rel = authorship * (1.0 - 0.5 * slip_penalty)
        r = self.reliability_rate
        self.control_reliability = float(np.clip(self.control_reliability + r * (target_rel - self.control_reliability), 0, 1))
        self.authorship_ema = float(np.clip(self.authorship_ema + r * (authorship - self.authorship_ema), 0, 1))
        self.disturbance_ema = float(min(self.disturbance_ema + r * (disturbance - self.disturbance_ema), 1e6))
        self.slip_ema = float(min(self.slip_ema + r * (actuator_slip - self.slip_ema), 1e6))
        if learn:
            W = self.G.copy()
            P = self.P.copy()
            _rls_update(W, P, phi, obs, 1.0, self.forgetting)
            self.G, self.P = W, P
        self.records += 1
        self.last = {'authorship': authorship, 'disturbance': disturbance,
                     'disturbance_vector': perp.tolist(), 'actuator_slip': actuator_slip,
                     'command_slip': command_slip, 'gain_mismatch': gain_mismatch,
                     'forward_error': forward_error, 'prediction_error': prediction_error,
                     'control_reliability': self.control_reliability}
        return dict(self.last)

    def export_state(self) -> dict[str, Any]:
        return {'schema': 'NeuroWeaveEfferenceSelfModelV1',
                'config': {'dim': self.dim, 'seed': self.seed, 'forgetting': self.forgetting,
                           'error_scale': self.error_scale, 'reliability_rate': self.reliability_rate,
                           'p_init': self.p_init},
                'G': _pack(self.G), 'P': _pack(self.P),
                'control_reliability': self.control_reliability, 'authorship_ema': self.authorship_ema,
                'disturbance_ema': self.disturbance_ema, 'slip_ema': self.slip_ema, 'records': self.records}

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('schema') != 'NeuroWeaveEfferenceSelfModelV1':
            raise ValueError('not a NeuroWeaveEfferenceSelfModelV1 state')
        cfg = state.get('config')
        if not isinstance(cfg, Mapping) or cfg.get('dim') != self.dim:
            raise ValueError('dim mismatch')
        G = _matrix('G', state.get('G'), (self.dim, self.dim + 1))
        if np.any(np.abs(G) > W_BOUND):
            raise ValueError('G outside bounds')
        P = _matrix('P', state.get('P'), (self.dim + 1, self.dim + 1))
        _check_P('P', P)
        rel = _finite_float('control_reliability', state.get('control_reliability'), 0.0, 1.0)
        auth = _finite_float('authorship_ema', state.get('authorship_ema'), 0.0, 1.0)
        dist = _finite_float('disturbance_ema', state.get('disturbance_ema'), 0.0, 1e6)
        slip = _finite_float('slip_ema', state.get('slip_ema'), 0.0, 1e6)
        records = _int('records', state.get('records'), 0, 2**62)
        forgetting = _finite_float('forgetting', cfg.get('forgetting'), 0.5, 1.0)
        scale = _finite_float('error_scale', cfg.get('error_scale'), 1e-6, 1e6)
        rate = _finite_float('reliability_rate', cfg.get('reliability_rate'), 1e-4, 1.0)
        seed = _int('seed', cfg.get('seed'), 0, 2**31 - 1)
        p_init = _finite_float('p_init', cfg.get('p_init'), 1e-6, 1e6)
        self.G, self.P, self.seed, self.p_init = G, P, seed, p_init
        self.control_reliability, self.authorship_ema, self.disturbance_ema, self.slip_ema = rel, auth, dist, slip
        self.records = records
        self.forgetting, self.error_scale, self.reliability_rate = forgetting, scale, rate


__all__ = ['PredictiveHierarchy', 'EfferenceSelfModel', 'DEFAULT_HORIZONS', 'GATE_MIN', 'GATE_MAX']
