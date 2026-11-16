"""Neuromodulator field, appraisal-driven affect state and transmitter inference.

This is a *synthetic abstraction* of neuromodulation: six bounded scalar control
fields (dopamine-, serotonin-, acetylcholine-, noradrenaline-, octopamine- and
endocannabinoid-like) driven by task appraisal signals, plus an appraisal
``AffectState`` used as a set of control variables. Nothing here claims
biological fidelity, emotions or consciousness; the names only identify the
functional role each field plays in the control law.

Neutrality: when neither ``neuromodulator_field`` nor ``affective_modulation``
is present, ``update`` is a no-op and ``controls`` returns exact neutral values
(scales 1.0, shifts 0.0, ``sensory_weight`` all ones). Even with the flags on,
controls are computed from deviations relative to the resting baselines, so a
freshly constructed system is also neutral.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

import numpy as np

from .spec import AFFECT_VARIABLES, NEUROMODULATORS, SENSORY_CHANNELS, SENSORY_DIM

SCHEMA = 'NeuroWeaveNeuromodV1'
RNG_OFFSET = 1100

DA, SHT, ACH, NE, OA, ECB = range(6)
BASELINES = np.array([0.30, 0.50, 0.30, 0.20, 0.20, 0.10], dtype=np.float64)
# per-step relaxation rates toward the instantaneous target
RATES = np.array([0.20, 0.05, 0.15, 0.20, 0.10, 0.04], dtype=np.float64)
TRACE_RATE = 0.12
TRACE_HIGH = 1.0
RECEPTOR_LOW, RECEPTOR_HIGH = 0.5, 1.5
SCALE_LOW, SCALE_HIGH = 0.25, 2.0
SHIFT_LOW, SHIFT_HIGH = -0.5, 0.5
GAIN_LOW, GAIN_HIGH = 0.5, 1.5

AFFECT_BASELINE = {
    'valence': 0.0, 'arousal': 0.15, 'curiosity': 0.2, 'stress': 0.0,
    'motivation': 0.5, 'novelty': 0.0, 'reward_prediction_error': 0.0,
}
_AFFECT_BOUNDS = {name: ((-1.0, 1.0) if name in ('valence', 'reward_prediction_error') else (0.0, 1.0))
                  for name in AFFECT_VARIABLES}

CONTROL_KEYS = (
    'learning_rate_scale', 'plasticity_scale', 'attention', 'exploration',
    'consolidation_scale', 'prediction_precision', 'sensory_weight',
    'decision_temperature', 'persistence', 'threshold_shift', 'reward_scale', 'gain',
)

# Per-channel sensory-weight profiles (exteroceptive channels gain precision
# under ACh; threat-relevant channels gain weight under NE / stress).
_EXTERO = np.array([1.0 if mod in ('visual', 'auditory', 'chemical') or cid in ('thermal', 'mechanosensation')
                    else -0.25 for cid, mod in SENSORY_CHANNELS], dtype=np.float64)
_THREAT = np.array([1.0 if cid in ('visual_motion', 'visual_edge_density', 'mechanosensation', 'thermal', 'audio_high')
                    else 0.0 for cid, _ in SENSORY_CHANNELS], dtype=np.float64)
_INTERO = np.array([1.0 if cid in ('interoception', 'gustatory', 'olfactory') else 0.0
                    for cid, _ in SENSORY_CHANNELS], dtype=np.float64)

TRANSMITTER_LABELS = ('glutamate', 'gaba', 'acetylcholine', 'dopamine', 'serotonin', 'octopamine')


def _flagset(flags: Iterable[str] | None) -> frozenset[str]:
    if flags is None:
        return frozenset()
    if isinstance(flags, str):
        return frozenset({flags})
    return frozenset(str(f) for f in flags)


def _finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


@dataclass
class AffectState:
    """Appraisal control variables. valence/rpe in [-1, 1]; others in [0, 1]."""

    valence: float = AFFECT_BASELINE['valence']
    arousal: float = AFFECT_BASELINE['arousal']
    curiosity: float = AFFECT_BASELINE['curiosity']
    stress: float = AFFECT_BASELINE['stress']
    motivation: float = AFFECT_BASELINE['motivation']
    novelty: float = AFFECT_BASELINE['novelty']
    reward_prediction_error: float = AFFECT_BASELINE['reward_prediction_error']

    @property
    def rpe(self) -> float:
        return self.reward_prediction_error

    def clamp(self) -> None:
        for name, (lo, hi) in _AFFECT_BOUNDS.items():
            setattr(self, name, _clip(getattr(self, name), lo, hi))

    def to_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in AFFECT_VARIABLES], dtype=np.float64)

    @classmethod
    def validate_dict(cls, data: Any) -> dict[str, float]:
        if not isinstance(data, Mapping) or set(data) != set(AFFECT_VARIABLES):
            raise ValueError(f'affect must be a mapping with keys {AFFECT_VARIABLES}')
        out = {}
        for name in AFFECT_VARIABLES:
            v = _finite_scalar(f'affect.{name}', data[name])
            lo, hi = _AFFECT_BOUNDS[name]
            if not lo <= v <= hi:
                raise ValueError(f'affect.{name} must be in [{lo}, {hi}]')
            out[name] = v
        return out


def neutral_controls() -> dict[str, Any]:
    return {
        'learning_rate_scale': 1.0, 'plasticity_scale': 1.0, 'attention': 1.0, 'exploration': 1.0,
        'consolidation_scale': 1.0, 'prediction_precision': 1.0, 'sensory_weight': np.ones(SENSORY_DIM),
        'decision_temperature': 1.0, 'persistence': 1.0, 'threshold_shift': 0.0, 'reward_scale': 1.0,
        'gain': 1.0,
    }


class NeuromodulatorSystem:
    """Six bounded modulator levels, per-unit receptors and local traces, affect."""

    def __init__(self, n: int, *, seed: int, signs: np.ndarray | None = None,
                 src: np.ndarray | None = None, dst: np.ndarray | None = None) -> None:
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
            raise ValueError('n must be a positive int')
        self.n = int(n)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed + RNG_OFFSET)
        self.signs = (np.ones(self.n, dtype=np.int8) if signs is None
                      else np.asarray(signs, dtype=np.int8).copy())
        if self.signs.shape != (self.n,):
            raise ValueError('signs must have shape (n,)')
        self.src = np.zeros(0, dtype=np.int64) if src is None else np.asarray(src, dtype=np.int64).copy()
        self.dst = np.zeros(0, dtype=np.int64) if dst is None else np.asarray(dst, dtype=np.int64).copy()
        if self.src.shape != self.dst.shape or self.src.ndim != 1:
            raise ValueError('src and dst must be 1-D arrays of the same shape')
        if self.src.size and (self.src.min() < 0 or self.dst.min() < 0
                              or self.src.max() >= self.n or self.dst.max() >= self.n):
            raise ValueError('src/dst endpoints must lie in [0, n)')
        # Drawn once; only *used* when receptor_heterogeneity is on.
        self.receptor_profile = rng.uniform(RECEPTOR_LOW, RECEPTOR_HIGH, size=(self.n, 6))
        self.levels = BASELINES.copy()
        self.traces = np.tile(BASELINES, (self.n, 1))
        self.affect = AffectState()
        self.reward_expectation = 0.0
        self.surprise_ema = 0.0
        self.sustained_activity = 0.0
        self.steps = 0

    # ------------------------------------------------------------- receptors
    def receptor_matrix(self, flags: Iterable[str] | None = None) -> np.ndarray:
        if 'receptor_heterogeneity' in _flagset(flags):
            return self.receptor_profile.copy()
        return np.ones((self.n, 6), dtype=np.float64)

    @property
    def level_dict(self) -> dict[str, float]:
        return {name: float(self.levels[k]) for k, name in enumerate(NEUROMODULATORS)}

    # ---------------------------------------------------------------- update
    def update(self, reward: float, prediction_error: float, novelty: float, threat: float,
               energy: float, flags: Iterable[str] | None = None, *,
               unit_activity: np.ndarray | None = None) -> None:
        """Appraise one step. ``unit_activity`` (n,) optionally drives local eCB traces."""
        flags = _flagset(flags)
        reward = _clip(_finite_scalar('reward', reward), -1.0, 1.0)
        pe = _clip(abs(_finite_scalar('prediction_error', prediction_error)), 0.0, 1.0)
        novelty = _clip(_finite_scalar('novelty', novelty), 0.0, 1.0)
        threat = _clip(_finite_scalar('threat', threat), 0.0, 1.0)
        energy = _clip(_finite_scalar('energy', energy), 0.0, 1.0)
        act = None
        if unit_activity is not None:
            act = np.asarray(unit_activity, dtype=np.float64)
            if act.shape != (self.n,) or not np.all(np.isfinite(act)):
                raise ValueError('unit_activity must be a finite array of shape (n,)')
            act = np.clip(np.abs(act), 0.0, 1.0)
        mod_on = 'neuromodulator_field' in flags
        aff_on = 'affective_modulation' in flags
        if not (mod_on or aff_on):
            return  # neutral: no state change

        # ---- appraisal (always computed internally when any flag is on)
        rpe = _clip(reward - self.reward_expectation, -1.0, 1.0)
        self.reward_expectation = _clip(0.9 * self.reward_expectation + 0.1 * reward, -1.0, 1.0)
        self.surprise_ema = _clip(0.8 * self.surprise_ema + 0.2 * pe, 0.0, 1.0)
        deficit = 1.0 - energy
        arousal_drive = _clip(0.45 * pe + 0.3 * threat + 0.15 * abs(reward) + 0.1 * novelty, 0.0, 1.0)
        self.sustained_activity = _clip(0.97 * self.sustained_activity + 0.03 * arousal_drive, 0.0, 1.0)

        if mod_on:
            target = BASELINES.copy()
            target[DA] += 0.5 * rpe + 0.15 * reward
            calm = (1.0 - self.surprise_ema) * (1.0 - threat)
            # stability (calm, positive expectation) raises 5-HT; stress lowers it
            target[SHT] += 0.2 * max(0.0, self.reward_expectation) * calm - 0.35 * (1.0 - calm)
            target[ACH] += 0.45 * pe + 0.25 * novelty
            target[NE] += 0.4 * threat + 0.3 * pe + 0.2 * arousal_drive
            target[OA] += 0.35 * arousal_drive + 0.25 * deficit
            target[ECB] += 0.8 * max(0.0, self.sustained_activity - 0.15)
            target = np.clip(target, 0.0, 1.0)
            self.levels = np.clip(self.levels + RATES * (target - self.levels), 0.0, 1.0)
            # Local per-unit traces. Each unit sees the global level, optionally with
            # an activity-dependent local (retrograde, eCB-like) component, and its
            # receptor density scales the *deviation from baseline* it experiences.
            # At rest (levels == baselines, no activity input) every trace equals the
            # baseline, so receptor heterogeneity is neutral until modulation moves.
            local = np.tile(self.levels, (self.n, 1))
            if act is not None:
                local[:, ECB] = np.clip(self.levels[ECB] * (0.5 + act), 0.0, 1.0)
            receptors = self.receptor_matrix(flags)
            drive = np.clip(BASELINES[None, :] + receptors * (local - BASELINES[None, :]), 0.0, 1.0)
            self.traces = np.clip(self.traces + TRACE_RATE * (drive - self.traces), 0.0, TRACE_HIGH)

        if aff_on:
            a = self.affect
            buffer = self.levels[ECB] if mod_on else BASELINES[ECB]
            da_level = self.levels[DA] if mod_on else BASELINES[DA]
            a.reward_prediction_error = rpe
            a.valence = 0.95 * a.valence + 0.05 * reward
            a.arousal = 0.9 * a.arousal + 0.1 * _clip(0.15 + 0.85 * arousal_drive, 0.0, 1.0)
            a.novelty = 0.85 * a.novelty + 0.15 * novelty
            a.curiosity = 0.9 * a.curiosity + 0.1 * (a.novelty * (1.0 - threat))
            stress_drive = _clip(0.45 * threat + 0.3 * deficit + 0.25 * pe, 0.0, 1.0)
            stress_drive *= 1.0 - 0.6 * max(0.0, buffer - BASELINES[ECB])  # eCB-like buffering
            a.stress = 0.92 * a.stress + 0.08 * stress_drive
            a.motivation = 0.92 * a.motivation + 0.08 * _clip(deficit * (0.5 + da_level) + 0.25 * max(0.0, a.valence), 0.0, 1.0)
            a.clamp()
        self.steps += 1

    # -------------------------------------------------------------- controls
    def controls(self, flags: Iterable[str] | None = None) -> dict[str, Any]:
        flags = _flagset(flags)
        out = neutral_controls()
        mod_on = 'neuromodulator_field' in flags
        aff_on = 'affective_modulation' in flags
        if not (mod_on or aff_on):
            return out
        sw = np.ones(SENSORY_DIM)
        lr = plast = cons = prec = temp = pers = att = expl = rscale = gain = 1.0
        thr = 0.0
        if mod_on:
            d = self.levels - BASELINES
            lr *= 1.0 + 1.2 * d[DA]
            plast *= 1.0 + 1.0 * d[DA] - 0.6 * d[ECB]
            cons *= 1.0 + 1.2 * d[SHT]
            prec *= 1.0 + 1.5 * d[ACH]
            att *= 1.0 + 0.6 * d[ACH] + 0.3 * d[NE]
            expl *= 1.0 + 0.6 * d[NE] - 0.4 * d[SHT]
            pers *= 1.0 + 0.6 * d[SHT]
            gain *= 1.0 + 0.4 * d[NE] + 0.3 * d[OA] - 0.3 * d[ECB]
            temp *= 1.0 + 0.5 * d[NE] - 0.3 * d[SHT]
            thr += -0.2 * d[NE] + 0.15 * d[ECB]
            sw = sw * (1.0 + 1.2 * d[ACH] * _EXTERO + 0.8 * d[NE] * _THREAT + 0.6 * d[OA] * _INTERO)
        if aff_on:
            a = self.affect
            expl *= 1.0 + 1.0 * (a.curiosity - AFFECT_BASELINE['curiosity'])
            att *= 1.0 + 0.6 * a.stress
            thr += 0.3 * a.stress
            pers *= 1.0 + 1.0 * (a.motivation - AFFECT_BASELINE['motivation'])
            rscale *= 1.0 + 0.6 * a.valence
            gain *= 1.0 + 0.5 * (a.arousal - AFFECT_BASELINE['arousal'])
            temp *= 1.0 - 0.6 * (a.arousal - AFFECT_BASELINE['arousal'])
            sw = sw * (1.0 + 0.5 * a.stress * _THREAT)

        def s(x: float) -> float:
            return _clip(x, SCALE_LOW, SCALE_HIGH)

        out.update(
            learning_rate_scale=s(lr), plasticity_scale=s(plast), attention=s(att), exploration=s(expl),
            consolidation_scale=s(cons), prediction_precision=s(prec),
            sensory_weight=np.clip(sw, SCALE_LOW, SCALE_HIGH),
            decision_temperature=s(temp), persistence=s(pers),
            threshold_shift=_clip(thr, SHIFT_LOW, SHIFT_HIGH), reward_scale=s(rscale), gain=s(gain),
        )
        return out

    def per_unit_gain(self, flags: Iterable[str] | None = None) -> np.ndarray:
        """Per-unit excitability multiplier in [0.5, 1.5] from receptor-weighted local traces.

        Traces already carry each unit's receptor scaling (see ``update``), so the
        gain is the weighted deviation of the traces from the resting baselines:
        exactly 1.0 for a fresh system, with or without receptor heterogeneity.
        """
        flags = _flagset(flags)
        if 'neuromodulator_field' not in flags:
            return np.ones(self.n, dtype=np.float64)
        dev = self.traces - BASELINES[None, :]
        coef = np.array([0.25, -0.15, 0.10, 0.30, 0.20, -0.25])
        g = 1.0 + dev @ coef
        return np.clip(g, GAIN_LOW, GAIN_HIGH)

    # -------------------------------------------------------- transmitters
    def infer_transmitters(self, src: np.ndarray | None = None, dst: np.ndarray | None = None,
                           w: np.ndarray | None = None, sign: np.ndarray | None = None) -> dict[str, Any]:
        src = self.src if src is None else src
        dst = self.dst if dst is None else dst
        if w is None:
            w = np.ones(np.asarray(src).shape[0])
        return infer_transmitters(src, dst, w, self.signs if sign is None else sign, n=self.n)

    # --------------------------------------------------------- persistence
    def export_state(self) -> dict[str, Any]:
        return {
            'schema': SCHEMA,
            'n': self.n,
            'seed': self.seed,
            'levels': self.level_dict,
            'baselines': {name: float(BASELINES[k]) for k, name in enumerate(NEUROMODULATORS)},
            'receptor_profile': self.receptor_profile.tolist(),
            'traces': self.traces.tolist(),
            'affect': self.affect.to_dict(),
            'reward_expectation': float(self.reward_expectation),
            'surprise_ema': float(self.surprise_ema),
            'sustained_activity': float(self.sustained_activity),
            'steps': int(self.steps),
        }

    def import_state(self, state: dict) -> None:
        if not isinstance(state, Mapping) or state.get('schema') != SCHEMA:
            raise ValueError('neuromod state must be a mapping with schema ' + SCHEMA)
        n_state = state.get('n')
        if isinstance(n_state, bool) or not isinstance(n_state, int) or n_state != self.n:
            raise ValueError('neuromod state n does not match this system')
        levels = state.get('levels')
        if not isinstance(levels, Mapping) or set(levels) != set(NEUROMODULATORS):
            raise ValueError('levels must map the six modulators')
        lv = np.array([_finite_scalar(f'levels.{k}', levels[k]) for k in NEUROMODULATORS])
        if np.any(lv < 0) or np.any(lv > 1):
            raise ValueError('levels must be in [0, 1]')

        def mat(name: str, lo: float, hi: float) -> np.ndarray:
            try:
                arr = np.asarray(state.get(name), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{name} must be a numeric matrix') from exc
            if arr.shape != (self.n, 6) or not np.all(np.isfinite(arr)):
                raise ValueError(f'{name} must be a finite (n, 6) matrix')
            if np.any(arr < lo) or np.any(arr > hi):
                raise ValueError(f'{name} out of bounds [{lo}, {hi}]')
            return arr

        rp = mat('receptor_profile', RECEPTOR_LOW, RECEPTOR_HIGH)
        tr = mat('traces', 0.0, TRACE_HIGH)
        affect = AffectState.validate_dict(state.get('affect'))
        rexp = _finite_scalar('reward_expectation', state.get('reward_expectation'))
        sur = _finite_scalar('surprise_ema', state.get('surprise_ema'))
        sus = _finite_scalar('sustained_activity', state.get('sustained_activity'))
        if not -1 <= rexp <= 1 or not 0 <= sur <= 1 or not 0 <= sus <= 1:
            raise ValueError('appraisal scalars out of bounds')
        steps = state.get('steps')
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a non-negative int')
        # ---- commit
        self.levels = lv
        self.receptor_profile = rp
        self.traces = tr
        self.affect = AffectState(**affect)
        self.reward_expectation, self.surprise_ema, self.sustained_activity = rexp, sur, sus
        self.steps = steps


# ---------------------------------------------------------------------------
def _csr(src: np.ndarray, dst: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compressed out-neighbour lists for a (deduplicated) edge list."""
    order = np.lexsort((dst, src))
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
    return ptr, dst[order]


def _gather(ptr: np.ndarray, idx: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Concatenated neighbour lists of ``nodes`` (vectorised CSR gather)."""
    starts = ptr[nodes]
    lens = ptr[nodes + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(total)
    return idx[offs]


def _structural_features(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Per-neuron connectivity fingerprint in [0, 1]: indeg, outdeg, in/out weight,
    reciprocity, two-hop reach, neighbour degree, triangle density.

    Sparse (adjacency-list) computation: memory O(n + edges), time roughly
    O(sum of two-hop path counts), so it stays usable at the population ceiling
    instead of materialising dense n x n matrix products.
    """
    feats = np.zeros((n, 8), dtype=np.float64)
    if src.size == 0:
        return feats
    aw = np.abs(w)
    indeg = np.bincount(dst, minlength=n).astype(np.float64)
    outdeg = np.bincount(src, minlength=n).astype(np.float64)
    inw = np.bincount(dst, weights=aw, minlength=n)
    outw = np.bincount(src, weights=aw, minlength=n)
    # unique directed edges (duplicates collapse, as in a 0/1 adjacency)
    codes = np.unique(src * n + dst)
    us, ud = codes // n, codes % n
    recip = np.isin(ud * n + us, codes)
    reciprocity = np.bincount(us, weights=recip.astype(np.float64), minlength=n) / np.maximum(1.0, outdeg)
    # undirected simple graph (no self loops) for neighbour degree / clustering
    ucodes = np.unique(np.concatenate([us * n + ud, ud * n + us]))
    a, b = ucodes // n, ucodes % n
    keep = a != b
    a, b = a[keep], b[keep]
    deg_u = np.bincount(a, minlength=n).astype(np.float64)
    neighbour = np.bincount(a, weights=deg_u[b], minlength=n) / np.maximum(1.0, deg_u)
    dptr, didx = _csr(us, ud, n)
    uptr, uidx = _csr(a, b, n)
    two_hop = np.zeros(n, dtype=np.float64)
    tri = np.zeros(n, dtype=np.float64)
    mark = np.zeros(n, dtype=bool)
    for u in range(n):
        out_u = didx[dptr[u]:dptr[u + 1]]
        if out_u.size:
            two_hop[u] = np.unique(_gather(dptr, didx, out_u)).size
        nb = uidx[uptr[u]:uptr[u + 1]]
        if nb.size > 1:
            mark[nb] = True
            # closed wedges u-v-x with x also a neighbour of u: each triangle twice
            tri[u] = mark[_gather(uptr, uidx, nb)].sum() / 2.0
            mark[nb] = False
    possible = deg_u * (deg_u - 1) / 2.0
    clustering = np.where(possible > 0, tri / np.maximum(possible, 1.0), 0.0)

    def norm(x: np.ndarray) -> np.ndarray:
        m = float(x.max())
        return x / m if m > 0 else x

    feats[:, 0] = norm(indeg)
    feats[:, 1] = norm(outdeg)
    feats[:, 2] = norm(inw)
    feats[:, 3] = norm(outw)
    feats[:, 4] = np.clip(reciprocity, 0, 1)
    feats[:, 5] = norm(two_hop)
    feats[:, 6] = norm(neighbour)
    feats[:, 7] = np.clip(clustering, 0, 1)
    return feats


def infer_transmitters(src: np.ndarray, dst: np.ndarray, w: np.ndarray, sign: np.ndarray,
                       *, n: int | None = None) -> dict[str, Any]:
    """Connectivity-only six-class transmitter *hypothesis* (no measured labels).

    Uses graph structure only; ``sign`` is read for length/consistency reporting
    and is never modified. Returns ``labels``, ``predictions`` (list[str]),
    ``probabilities`` (n, 6), ``confidence`` (n,), ``transmitter_gain`` (n,) in
    [0.94, 1.06] and ``sign_consistent`` (n,) bool (glutamate<->+1, gaba<->-1,
    other classes always consistent).
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    w = np.asarray(w, dtype=np.float64)
    sign = np.asarray(sign)
    if n is None:
        n = int(sign.shape[0])
    if sign.shape != (n,):
        raise ValueError('sign must have shape (n,)')
    if src.shape != dst.shape or src.shape != w.shape or src.ndim != 1:
        raise ValueError('src, dst and w must be 1-D arrays of equal length')
    if src.size and (src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n):
        raise ValueError('edge endpoints out of range')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must be finite')
    f = _structural_features(src.astype(np.int64), dst.astype(np.int64), w, n)
    indeg, outdeg, inw, outw, rec, two, nb, tri = (f[:, k] for k in range(8))
    raw = np.stack([
        0.18 + 0.62 * outdeg + 0.12 * outw + 0.08 * rec,               # glutamate
        0.18 + 0.62 * indeg + 0.12 * two + 0.08 * (1 - rec),            # gaba
        0.18 + 0.34 * rec + 0.28 * tri + 0.20 * two + 0.08 * nb,        # acetylcholine
        0.18 + 0.35 * outdeg + 0.28 * nb + 0.16 * rec + 0.08 * outw,    # dopamine
        0.18 + 0.38 * indeg + 0.28 * two + 0.16 * nb + 0.08 * inw,      # serotonin
        0.18 + 0.35 * outdeg + 0.25 * tri + 0.20 * outw + 0.08 * (1 - rec),  # octopamine
    ], axis=1)
    z = (raw - raw.max(axis=1, keepdims=True)) * 3.0
    ex = np.exp(z)
    prob = ex / ex.sum(axis=1, keepdims=True)
    best = prob.argmax(axis=1)
    conf = prob[np.arange(n), best]
    predictions = [TRANSMITTER_LABELS[b] for b in best]
    consistent = np.array([not ((p == 'glutamate' and s < 0) or (p == 'gaba' and s > 0))
                           for p, s in zip(predictions, sign)], dtype=bool)
    return {
        'labels': list(TRANSMITTER_LABELS),
        'predictions': predictions,
        'probabilities': prob,
        'confidence': conf,
        'transmitter_gain': np.clip(0.94 + 0.12 * conf, 0.94, 1.06),
        'sign_consistent': consistent,
        'evidence': 'connectivity_only_hypothesis',
    }
