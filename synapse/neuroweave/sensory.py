"""Sixteen-channel sensory hub: feature extraction, prediction and reliability fusion.

Privacy: raw camera frames and audio buffers are never stored. Feature
extractors are pure functions that return derived scalars only; the hub keeps
at most a down-sampled (<=16x16) luminance map for motion estimation, which is
never exported. ``manifest()['raw_media_stored']`` is always ``False``.

All channel values are bounded to [-1, 1]. This is a synthetic abstraction of
multimodal integration, not a model of any biological sensory system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .spec import EVIDENCE_TIER, SENSORY_CHANNELS, SENSORY_CHANNEL_IDS, SENSORY_DIM, SENSORY_MODES, NeuroWeaveSpecV1

SCHEMA = 'NeuroWeaveSensoryV1'
RNG_OFFSET = 1300
MAP_MAX = 16
STALE_TAU = 8.0          # reliability e-folding age (hub steps)
MAX_AGE = 64             # samples older than this are treated as missing
RELIABILITY_SCALE = 0.02  # residual variance at which reliability halves
ASSOC_LR = 0.05
ASSOC_BOUND = 2.0
MAX_OBJECT_REGIONS = 16

CHANNEL_INDEX = {cid: i for i, cid in enumerate(SENSORY_CHANNEL_IDS)}
VISUAL_CHANNELS = {
    'visual_luminance': 'luminance', 'visual_red_green': 'red_green',
    'visual_blue_yellow': 'blue_yellow', 'visual_edge_density': 'edge_density',
    'visual_motion': 'motion',
}
AUDIO_CHANNELS = {'audio_low': 'low', 'audio_mid': 'mid', 'audio_high': 'high'}
VISUAL_FEATURE_KEYS = ('luminance', 'red_green', 'blue_yellow', 'edge_density', 'motion', 'change',
                       'centroid_x', 'centroid_y', 'spatial_spread', 'object_regions', 'temporal_pattern')
AUDIO_FEATURE_KEYS = ('amplitude', 'low', 'mid', 'high', 'change', 'onset', 'pattern')
AUDIO_BANDS = {'low': (20.0, 300.0), 'mid': (300.0, 2000.0), 'high': (2000.0, 16000.0)}


def _flagset(flags: Iterable[str] | None) -> frozenset[str]:
    if flags is None:
        return frozenset()
    if isinstance(flags, str):
        return frozenset({flags})
    return frozenset(str(f) for f in flags)


def _bound(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _finite_float(name: str, value: Any) -> float:
    """Strict scalar coercion: rejects bools, non-numeric types and non-finite values."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(out):
        raise ValueError(f'{name} must be finite')
    return out


# ---------------------------------------------------------------------- vision
def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError('frame must be an HxWx3 array')
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError('frame must be numeric')
    out = arr.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError('frame must be finite')
    if out.max(initial=0.0) > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def _downsample(lum: np.ndarray) -> np.ndarray:
    h, w = lum.shape
    rows = np.linspace(0, h, min(MAP_MAX, h) + 1).astype(int)
    cols = np.linspace(0, w, min(MAP_MAX, w) + 1).astype(int)
    sums = np.add.reduceat(np.add.reduceat(lum, rows[:-1], axis=0), cols[:-1], axis=1)
    counts = np.outer(np.diff(rows), np.diff(cols))
    return sums / counts


def _count_regions(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                count += 1
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
    return min(count, MAX_OBJECT_REGIONS)


def _visual(frame: Any, previous: Any = None) -> tuple[dict[str, float], np.ndarray]:
    img = _as_frame(frame)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    small = _downsample(lum)
    prev_map = None
    if previous is not None:
        prev = np.asarray(previous)
        if prev.ndim == 3:
            prev_map = _downsample(np.tensordot(_as_frame(prev), [0.299, 0.587, 0.114], axes=([2], [0])))
        elif prev.ndim == 2 and np.all(np.isfinite(prev)):
            prev_map = prev.astype(np.float64)
        if prev_map is not None and prev_map.shape != small.shape:
            prev_map = None
    gx = np.abs(np.diff(lum, axis=1)) if lum.shape[1] > 1 else np.zeros((lum.shape[0], 0))
    gy = np.abs(np.diff(lum, axis=0)) if lum.shape[0] > 1 else np.zeros((0, lum.shape[1]))
    edge_px = (gx > 0.1).sum() + (gy > 0.1).sum()
    edge_total = max(1, gx.size + gy.size)
    total = float(lum.sum())
    h, w = lum.shape
    if total > 1e-9:
        ys, xs = np.mgrid[0:h, 0:w]
        xn = (xs + 0.5) / w
        yn = (ys + 0.5) / h
        cx = float((lum * xn).sum() / total)
        cy = float((lum * yn).sum() / total)
        spread = float(np.sqrt((lum * ((xn - cx) ** 2 + (yn - cy) ** 2)).sum() / total) / np.sqrt(1 / 6))
    else:
        cx = cy = 0.5
        spread = 0.0
    bright = small > max(0.5, float(small.mean() + 0.5 * small.std()))
    if prev_map is not None:
        diff = np.abs(small - prev_map)
        motion = float(diff.mean())
        change = float(abs(small.mean() - prev_map.mean()))
        temporal = float((diff > 0.05).mean())
    else:
        motion = change = temporal = 0.0
    feats = {
        'luminance': _bound(lum.mean()),
        'red_green': _bound((r - g).mean()),
        'blue_yellow': _bound((b - 0.5 * (r + g)).mean()),
        'edge_density': _bound(edge_px / edge_total),
        'motion': _bound(motion),
        'change': _bound(change),
        'centroid_x': _bound(cx),
        'centroid_y': _bound(cy),
        'spatial_spread': _bound(spread),
        'object_regions': int(_count_regions(bright)),
        'temporal_pattern': _bound(temporal),
    }
    return feats, small


def visual_features(frame: np.ndarray, previous: np.ndarray | None = None) -> dict[str, float]:
    """Derived scalar features of an HxWx3 frame. ``previous`` may be a frame or a
    <=16x16 luminance map. The frame is not retained."""
    feats, _ = _visual(frame, previous)
    return feats


# ----------------------------------------------------------------------- audio
def audio_features(samples: np.ndarray, sample_rate: float, previous: Mapping[str, float] | None = None) -> dict[str, float]:
    """Derived scalars from an audio buffer via a Hann-windowed rFFT. Not retained."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError('samples must be a finite 1-D buffer with >= 8 samples')
    sample_rate = _finite_float('sample_rate', sample_rate)
    if sample_rate <= 0:
        raise ValueError('sample_rate must be positive')
    x = np.clip(x, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(x ** 2)))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    energies = {name: float(spec[(freqs >= lo) & (freqs < hi)].sum()) for name, (lo, hi) in AUDIO_BANDS.items()}
    tot = sum(energies.values())
    if tot > 1e-12 and rms > 1e-6:
        bands = {k: v / tot for k, v in energies.items()}
    else:
        bands = {k: 0.0 for k in energies}
    # onset / rhythmic pattern from a short-time envelope
    frames = np.array_split(x, 16)
    env = np.array([np.sqrt(np.mean(f ** 2)) if f.size else 0.0 for f in frames])
    rises = np.diff(env)
    onset = float(np.clip(rises.max(initial=0.0) / (env.max() + 1e-9), 0.0, 1.0)) if env.max() > 1e-6 else 0.0
    ec = env - env.mean()
    denom = float((ec ** 2).sum())
    if denom > 1e-12:
        ac = np.array([(ec[:-lag] * ec[lag:]).sum() / denom for lag in range(2, 8)])
        pattern = float(np.clip(ac.max(), 0.0, 1.0))
    else:
        pattern = 0.0
    amplitude = float(np.clip(rms * 2.0, 0.0, 1.0))
    prev_amp = float(previous.get('amplitude', amplitude)) if previous else amplitude
    return {
        'amplitude': amplitude,
        'low': bands['low'], 'mid': bands['mid'], 'high': bands['high'],
        'change': float(np.clip(abs(amplitude - prev_amp), 0.0, 1.0)),
        'onset': onset,
        'pattern': pattern,
    }


# ----------------------------------------------------------------------- frame
@dataclass
class SensoryFrame:
    value: np.ndarray
    reliability: np.ndarray
    missing: np.ndarray
    predicted: np.ndarray
    prediction_error: np.ndarray
    uncertainty: float
    imputations: dict[str, float] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    attention: np.ndarray = field(default_factory=lambda: np.ones(SENSORY_DIM))
    weighted: np.ndarray = field(default_factory=lambda: np.zeros(SENSORY_DIM))
    age: np.ndarray = field(default_factory=lambda: np.zeros(SENSORY_DIM))
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'value': self.value.tolist(), 'reliability': self.reliability.tolist(),
            'missing': self.missing.tolist(), 'predicted': self.predicted.tolist(),
            'prediction_error': self.prediction_error.tolist(), 'uncertainty': float(self.uncertainty),
            'imputations': dict(self.imputations), 'provenance': list(self.provenance),
            'attention': self.attention.tolist(), 'weighted': self.weighted.tolist(),
            'age': self.age.tolist(), 'step': int(self.step),
        }


# ------------------------------------------------------------------------- hub
class SensoryHub:
    """Per-channel mode routing, temporal prediction, reliability and cross-modal fusion."""

    def __init__(self, spec: NeuroWeaveSpecV1 | None = None) -> None:
        self.spec = spec if spec is not None else NeuroWeaveSpecV1()
        self.seed = int(self.spec.seed)
        self.modes: dict[str, str] = {c: self.spec.sensory_mode(c) for c in SENSORY_CHANNEL_IDS}
        self.step = 0
        # latest samples per source: channel -> (value, hub_step)
        self._simulated: dict[str, tuple[float, int]] = {}
        self._external: dict[str, tuple[float, int]] = {}
        self._live: dict[str, tuple[float, int]] = {}
        # AR(1) predictor statistics per channel
        self.pred_mean = np.zeros(SENSORY_DIM)
        self.pred_var = np.full(SENSORY_DIM, 0.05)
        self.pred_cov = np.zeros(SENSORY_DIM)
        self.last_value = np.zeros(SENSORY_DIM)
        self.has_last = np.zeros(SENSORY_DIM, dtype=bool)
        self.resid_var = np.full(SENSORY_DIM, 0.05)
        # cross-modal association
        self.assoc = np.zeros((SENSORY_DIM, SENSORY_DIM))
        self.assoc_err_var = np.full(SENSORY_DIM, 0.25)
        self.assoc_updates = 0
        # privacy-safe live context
        self._lum_map: np.ndarray | None = None   # <=16x16, never exported
        self.last_audio: dict[str, float] = {}
        self.last_visual: dict[str, float] = {}

    # ---------------------------------------------------------------- modes
    def set_mode(self, channel: str, mode: str) -> None:
        if channel not in CHANNEL_INDEX:
            raise ValueError(f'unknown sensory channel {channel!r}')
        if mode not in SENSORY_MODES:
            raise ValueError(f'mode must be one of {SENSORY_MODES}')
        self.modes[channel] = mode

    def set_external(self, channel: str, value: float) -> None:
        if channel not in CHANNEL_INDEX:
            raise ValueError(f'unknown sensory channel {channel!r}')
        self._external[channel] = (_bound(_finite_float('external value', value)), self.step)

    # ----------------------------------------------------------- live input
    def visual_features(self, frame: np.ndarray, previous: np.ndarray | None = None) -> dict[str, float]:
        """Pure feature extraction; uses the stored small map as default previous."""
        feats, _ = _visual(frame, self._lum_map if previous is None else previous)
        return feats

    def push_visual(self, frame: np.ndarray) -> dict[str, float]:
        feats, small = _visual(frame, self._lum_map)
        self._lum_map = small.copy()
        self.last_visual = dict(feats)
        for cid, key in VISUAL_CHANNELS.items():
            self._live[cid] = (_bound(feats[key]), self.step)
        return feats

    def audio_features(self, samples: np.ndarray, sample_rate: float) -> dict[str, float]:
        return audio_features(samples, sample_rate, self.last_audio or None)

    def push_audio(self, samples: np.ndarray, sample_rate: float) -> dict[str, float]:
        feats = audio_features(samples, sample_rate, self.last_audio or None)
        self.last_audio = dict(feats)
        for cid, key in AUDIO_CHANNELS.items():
            self._live[cid] = (_bound(feats[key]), self.step)
        return feats

    def push_live(self, channel: str, value: float) -> None:
        """Live scalar for non-camera/microphone channels (e.g. a device sensor)."""
        if channel not in CHANNEL_INDEX:
            raise ValueError(f'unknown sensory channel {channel!r}')
        self._live[channel] = (_bound(_finite_float('live value', value)), self.step)

    # ------------------------------------------------------------ simulation
    def simulate(self, ecology_obs: Mapping[str, Any] | None = None, step: int | None = None) -> np.ndarray:
        """Deterministic 16-d simulated sensation; cross-modally correlated latents."""
        step = self.step if step is None else int(step)
        if ecology_obs is not None and 'sensory' in ecology_obs:
            vec = np.asarray(ecology_obs['sensory'], dtype=np.float64)
            if vec.shape != (SENSORY_DIM,) or not np.all(np.isfinite(vec)):
                raise ValueError('ecology_obs["sensory"] must be a finite 16-vector')
            vec = np.clip(vec, -1.0, 1.0)
        else:
            rng = np.random.default_rng([self.seed + RNG_OFFSET, max(0, step)])
            t = float(step)
            z = np.array([np.sin(2 * np.pi * t / 37.0), np.sin(2 * np.pi * t / 23.0 + 1.0),
                          np.cos(2 * np.pi * t / 53.0)])
            load = np.array([
                [0.40, 0.05, 0.00], [0.30, -0.10, 0.00], [-0.20, 0.10, 0.00], [0.25, 0.20, 0.00],
                [0.05, 0.40, 0.00], [0.30, 0.00, 0.10], [0.00, 0.30, 0.10], [0.00, 0.35, -0.10],
                [0.00, 0.00, 0.40], [0.00, 0.05, 0.35], [0.10, 0.00, 0.30], [0.00, 0.10, -0.35],
                [0.00, 0.35, 0.00], [0.05, 0.30, 0.05], [0.20, 0.00, 0.20], [0.00, 0.00, 0.10],
            ])
            vec = 0.5 + load @ z + rng.normal(0.0, 0.02, SENSORY_DIM)
            if ecology_obs:
                for key, cid in (('energy', 'interoception'), ('temperature', 'thermal'),
                                 ('speed', 'proprioception'), ('collision', 'mechanosensation'),
                                 ('food_proximity', 'olfactory'), ('obstacle_proximity', 'environment_state'),
                                 ('task_context', 'task_context')):
                    if key in ecology_obs and ecology_obs[key] is not None:
                        vec[CHANNEL_INDEX[cid]] = _finite_float(f'ecology_obs[{key!r}]', ecology_obs[key])
            vec = np.clip(vec, -1.0, 1.0)
        for cid in SENSORY_CHANNEL_IDS:
            self._simulated[cid] = (float(vec[CHANNEL_INDEX[cid]]), self.step)
        return vec

    # ------------------------------------------------------------- fusion
    def _sample(self, cid: str, raw: Mapping[str, Any] | None) -> tuple[float | None, int, str]:
        mode = self.modes[cid]
        if mode == 'disabled':
            return None, 0, 'disabled'
        if raw is not None and cid in raw:
            v = raw[cid]
            if v is None:
                return None, 0, 'missing'
            return _bound(_finite_float(f'raw value for {cid}', v)), 0, {'simulated': 'simulated', 'external': 'external', 'live': 'sensory'}[mode]
        store = {'simulated': self._simulated, 'external': self._external, 'live': self._live}[mode]
        if cid not in store:
            return None, 0, 'missing'
        value, t = store[cid]
        age = max(0, self.step - t)
        if age > MAX_AGE:
            return None, age, 'missing'
        return value, age, {'simulated': 'simulated', 'external': 'external', 'live': 'sensory'}[mode]

    def predict(self) -> np.ndarray:
        coef = np.clip(self.pred_cov / np.maximum(self.pred_var, 1e-6), -0.95, 0.95)
        base = np.where(self.has_last, self.last_value, self.pred_mean)
        return np.clip(self.pred_mean + coef * (base - self.pred_mean), -1.0, 1.0)

    def fuse(self, raw: Mapping[str, Any] | None = None, flags: Iterable[str] | None = None, *,
             sensory_weight: np.ndarray | None = None, learn: bool = True) -> SensoryFrame:
        flags = _flagset(flags)
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError('raw must be a mapping channel -> float | None')
        if raw is not None:
            unknown = set(raw) - set(CHANNEL_INDEX)
            if unknown:
                raise ValueError(f'unknown sensory channels {sorted(unknown)}')
        if sensory_weight is None:
            attention = np.ones(SENSORY_DIM)
        else:
            attention = np.asarray(sensory_weight, dtype=np.float64)
            if attention.shape != (SENSORY_DIM,) or not np.all(np.isfinite(attention)):
                raise ValueError('sensory_weight must be a finite 16-vector')
            attention = np.clip(attention, 0.0, 2.0)

        predicted = self.predict()
        value = np.zeros(SENSORY_DIM)
        age = np.zeros(SENSORY_DIM)
        missing = np.zeros(SENSORY_DIM, dtype=bool)
        disabled = np.zeros(SENSORY_DIM, dtype=bool)
        provenance: list[str] = []
        for cid in SENSORY_CHANNEL_IDS:
            i = CHANNEL_INDEX[cid]
            v, a, prov = self._sample(cid, raw)
            provenance.append(prov)
            age[i] = a
            if prov == 'disabled':
                disabled[i] = True
            elif v is None:
                missing[i] = True
            else:
                value[i] = v
        observed = ~(missing | disabled)
        base_rel = 1.0 / (1.0 + self.resid_var / RELIABILITY_SCALE)
        reliability = np.where(observed, base_rel * np.exp(-age / STALE_TAU), 0.0)
        pe = np.where(observed, np.abs(value - predicted), 0.0)

        imputations: dict[str, float] = {}
        use_xmodal = 'cross_modal_memory' in flags
        centred = np.where(observed, value - self.pred_mean, 0.0)
        if use_xmodal and missing.any() and observed.any():
            for i in np.flatnonzero(missing):
                est = float(np.clip(self.pred_mean[i] + self.assoc[i] @ centred, -1.0, 1.0))
                value[i] = est
                quality = 1.0 / (1.0 + self.assoc_err_var[i] / RELIABILITY_SCALE)
                reliability[i] = 0.5 * quality * float(base_rel[observed].mean())
                provenance[i] = 'predicted'
                imputations[SENSORY_CHANNEL_IDS[i]] = est

        if learn:
            self._learn(value, observed, predicted, centred, use_xmodal)
        active = ~disabled
        uncertainty = float(1.0 - reliability[active].mean()) if active.any() else 1.0
        weighted = value * reliability * attention
        frame = SensoryFrame(
            value=value, reliability=np.clip(reliability, 0.0, 1.0), missing=missing,
            predicted=predicted, prediction_error=pe, uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
            imputations=imputations, provenance=provenance, attention=attention,
            weighted=weighted, age=age, step=self.step,
        )
        self.step += 1
        return frame

    def _learn(self, value: np.ndarray, observed: np.ndarray, predicted: np.ndarray,
               centred: np.ndarray, use_xmodal: bool) -> None:
        o = observed
        resid = value - predicted
        self.resid_var[o] = np.clip(0.9 * self.resid_var[o] + 0.1 * resid[o] ** 2, 1e-6, 4.0)
        prev_c = self.last_value - self.pred_mean
        cur_c = value - self.pred_mean
        both = o & self.has_last
        self.pred_cov[both] = np.clip(0.95 * self.pred_cov[both] + 0.05 * prev_c[both] * cur_c[both], -4.0, 4.0)
        self.pred_var[o] = np.clip(0.95 * self.pred_var[o] + 0.05 * cur_c[o] ** 2, 1e-6, 4.0)
        self.pred_mean[o] = np.clip(0.95 * self.pred_mean[o] + 0.05 * value[o], -1.0, 1.0)
        self.last_value[o] = value[o]
        self.has_last |= o
        if use_xmodal and o.sum() >= 2:
            idx = np.flatnonzero(o)
            sub = centred[idx]
            W = self.assoc[np.ix_(idx, idx)]
            est = W @ sub
            err = sub - est
            self.assoc_err_var[idx] = np.clip(0.95 * self.assoc_err_var[idx] + 0.05 * err ** 2, 1e-6, 4.0)
            # LMS step, normalised when the input is large so that
            # lr * ||x||^2 <= 1 (the stability condition of the LMS update).
            lr = ASSOC_LR * min(1.0, 1.0 / max(1e-12, ASSOC_LR * float(sub @ sub)))
            W = W + lr * np.outer(err, sub)
            np.fill_diagonal(W, 0.0)
            self.assoc[np.ix_(idx, idx)] = np.clip(W, -ASSOC_BOUND, ASSOC_BOUND)
            self.assoc_updates += 1

    # ------------------------------------------------------------ manifest
    def manifest(self) -> dict[str, Any]:
        return {
            'schema': SCHEMA,
            'channels': [{'id': cid, 'modality': mod, 'mode': self.modes[cid],
                          'live_source': ('camera_derived_scalar' if cid in VISUAL_CHANNELS else
                                          'microphone_derived_scalar' if cid in AUDIO_CHANNELS else 'scalar')}
                         for cid, mod in SENSORY_CHANNELS],
            'raw_media_stored': False,
            'raw_media_exported': False,
            'retained_live_data': 'derived_scalars_only',
            'motion_map_max_shape': [MAP_MAX, MAP_MAX],
            'motion_map_exported': False,
            'evidence_tier': EVIDENCE_TIER,
        }

    # --------------------------------------------------------- persistence
    @staticmethod
    def _store_to_json(store: dict[str, tuple[float, int]]) -> dict[str, list]:
        return {k: [float(v), int(t)] for k, (v, t) in store.items()}

    def export_state(self) -> dict[str, Any]:
        return {
            'schema': SCHEMA,
            'step': int(self.step),
            'modes': dict(self.modes),
            'simulated': self._store_to_json(self._simulated),
            'external': self._store_to_json(self._external),
            'live': self._store_to_json(self._live),
            'pred_mean': self.pred_mean.tolist(),
            'pred_var': self.pred_var.tolist(),
            'pred_cov': self.pred_cov.tolist(),
            'last_value': self.last_value.tolist(),
            'has_last': self.has_last.tolist(),
            'resid_var': self.resid_var.tolist(),
            'assoc': self.assoc.tolist(),
            'assoc_err_var': self.assoc_err_var.tolist(),
            'assoc_updates': int(self.assoc_updates),
            'last_visual': {k: (int(v) if k == 'object_regions' else float(v)) for k, v in self.last_visual.items()},
            'last_audio': {k: float(v) for k, v in self.last_audio.items()},
            'raw_media_stored': False,
        }

    def import_state(self, state: dict) -> None:
        if not isinstance(state, Mapping) or state.get('schema') != SCHEMA:
            raise ValueError('sensory state must be a mapping with schema ' + SCHEMA)
        step = state.get('step')
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('step must be a non-negative int')
        modes = state.get('modes')
        if not isinstance(modes, Mapping) or set(modes) != set(SENSORY_CHANNEL_IDS) or \
                any(m not in SENSORY_MODES for m in modes.values()):
            raise ValueError('modes must map every channel to a valid mode')

        def store(name: str) -> dict[str, tuple[float, int]]:
            data = state.get(name)
            if not isinstance(data, Mapping):
                raise ValueError(f'{name} must be a mapping')
            out = {}
            for k, item in data.items():
                if k not in CHANNEL_INDEX or not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(f'bad {name} entry {k!r}')
                v, t = item
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not -1 <= v <= 1:
                    raise ValueError(f'{name}.{k} value must be finite in [-1, 1]')
                if isinstance(t, bool) or not isinstance(t, int) or t < 0:
                    raise ValueError(f'{name}.{k} step must be a non-negative int')
                out[k] = (float(v), t)
            return out

        def vec(name: str, lo: float, hi: float, shape=(SENSORY_DIM,)) -> np.ndarray:
            try:
                arr = np.asarray(state.get(name), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{name} must be numeric') from exc
            if arr.shape != shape or not np.all(np.isfinite(arr)) or np.any(arr < lo) or np.any(arr > hi):
                raise ValueError(f'{name} must be finite with shape {shape} in [{lo}, {hi}]')
            return arr

        sim, ext, live = store('simulated'), store('external'), store('live')
        pm = vec('pred_mean', -1, 1)
        pv = vec('pred_var', 1e-6, 4)
        pc = vec('pred_cov', -4, 4)
        lv = vec('last_value', -1, 1)
        hl = state.get('has_last')
        if not isinstance(hl, list) or len(hl) != SENSORY_DIM or not all(type(x) is bool for x in hl):
            raise ValueError('has_last must be a list of 16 booleans')
        rv = vec('resid_var', 1e-6, 4)
        A = vec('assoc', -ASSOC_BOUND, ASSOC_BOUND, (SENSORY_DIM, SENSORY_DIM))
        if np.any(np.diag(A) != 0):
            raise ValueError('assoc diagonal must be zero')
        aev = vec('assoc_err_var', 1e-6, 4)
        au = state.get('assoc_updates')
        if isinstance(au, bool) or not isinstance(au, int) or au < 0:
            raise ValueError('assoc_updates must be a non-negative int')
        lvis, laud = state.get('last_visual', {}), state.get('last_audio', {})
        for name, d, allowed in (('last_visual', lvis, VISUAL_FEATURE_KEYS), ('last_audio', laud, AUDIO_FEATURE_KEYS)):
            if not isinstance(d, Mapping) or not set(d) <= set(allowed) or not all(
                    not isinstance(v, bool) and isinstance(v, (int, float)) and np.isfinite(v) and abs(v) <= MAX_OBJECT_REGIONS
                    for v in d.values()):
                raise ValueError(f'{name} must map known feature names to bounded finite scalars')
        # ---- commit
        self.step = step
        self.modes = dict(modes)
        self._simulated, self._external, self._live = sim, ext, live
        self.pred_mean, self.pred_var, self.pred_cov, self.last_value = pm, pv, pc, lv
        self.has_last = np.array(hl, dtype=bool)
        self.resid_var, self.assoc, self.assoc_err_var = rv, A, aev
        self.assoc_updates = au
        self.last_visual, self.last_audio = dict(lvis), dict(laud)
        self._lum_map = None  # never persisted
