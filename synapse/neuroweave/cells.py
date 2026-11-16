"""Multi-compartment synthetic unit population (NumPy re-implementation).

With no neuron mechanism flags the update is the NEUROWEAVE v0.1 baseline::

    fast    = 0.58 fast + u + 0.52 I
    slow    = 0.94 slow + 0.14 u + 0.18 I
    context = 0.69 context + I
    output  = tanh(0.84 fast + 0.16 I)

where ``u`` is the feed-forward (external) drive and ``I`` the summed recurrent
synaptic drive. Each mechanism flag replaces or extends a well-defined part of
that update. All units are synthetic computational abstractions: the soma
"roles" and dendrite names label state channels, they are not claims of
biological fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = ['CompartmentDrive', 'CellPopulation', 'NEURON_FLAGS', 'STATE_FIELDS', 'q88', 'neutral_controls',
           'validate_controls']

NEURON_FLAGS: tuple[str, ...] = (
    'multi_soma', 'extended_somas', 'dendritic_compartments', 'membrane_dynamics', 'axonal_branches',
    'astrocyte_regulation', 'homeostatic_governor', 'predictive_coding', 'hierarchical_temporal_memory',
    'q88_state', 'sparse_code',
)

DENDRITE_FIELDS = ('proximal', 'distal', 'apical', 'basal')
SOMA_FIELDS = ('primary', 'secondary', 'fast', 'slow', 'context', 'integrator', 'burst', 'gating')
OTHER_FIELDS = (
    'membrane', 'threshold', 'refractory', 'spike', 'calcium', 'adaptation', 'trace', 'hillock', 'branch_a',
    'branch_b', 'terminal', 'release', 'homeo_gain', 'astrocyte', 'myelin', 'utility', 'consolidation',
    'pred_estimate', 'pred_error', 'surprise', 'tm_fast', 'tm_mid', 'tm_slow', 'output',
)
STATE_FIELDS: tuple[str, ...] = DENDRITE_FIELDS + SOMA_FIELDS + OTHER_FIELDS

STATE_LIMIT = 100.0
BASE_THRESHOLD = 0.53
HOMEO_TARGET = 0.2
_CELL_OFFSET = 2_003

# Bounds per field (lo, hi) checked on import and enforced after each step.
_BOUNDS: dict[str, tuple[float, float]] = {name: (-STATE_LIMIT, STATE_LIMIT) for name in STATE_FIELDS}
_BOUNDS.update({
    'output': (-1.0, 1.0), 'spike': (0.0, 1.0), 'refractory': (0.0, 2.0), 'calcium': (0.0, 4.0),
    'adaptation': (0.0, 1.0), 'trace': (0.0, 1.0), 'threshold': (0.05, 1.6), 'gating': (0.65, 1.35),
    'homeo_gain': (0.5, 1.5), 'astrocyte': (0.3, 1.2), 'myelin': (0.2, 0.98), 'utility': (0.0, STATE_LIMIT),
    'consolidation': (0.0, 2.0), 'surprise': (0.0, 2.0), 'tm_fast': (0.0, 1.0), 'tm_mid': (0.0, 1.0),
    'tm_slow': (0.0, 1.0), 'release': (-1.0, 1.0), 'secondary': (-1.0, 1.0), 'burst': (0.0, STATE_LIMIT),
})


def q88(x: np.ndarray) -> np.ndarray:
    """Round to the signed Q8.8 fixed-point grid (16-bit, 1/256 resolution)."""
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 256.0) / 256.0, -128.0, 32767.0 / 256.0)


def neutral_controls() -> dict[str, float]:
    return {'gain': 1.0, 'threshold_shift': 0.0, 'attention': 1.0, 'plasticity_scale': 1.0}


def validate_controls(controls: Mapping[str, Any] | None, extra: Iterable[str] = ()) -> dict[str, float]:
    """Return the neutral-filled numeric controls; raise ValueError on non-numeric / non-finite values."""
    c = neutral_controls()
    for key in extra:
        c.setdefault(key, 1.0)
    if controls is not None and not isinstance(controls, Mapping):
        raise ValueError('controls must be a mapping')
    if controls:
        for key in c:
            if key in controls and controls[key] is not None:
                value = controls[key]
                if isinstance(value, bool):
                    raise ValueError(f'control {key} must be a number')
                try:
                    v = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'control {key} must be a number') from exc
                if not np.isfinite(v):
                    raise ValueError(f'control {key} must be finite')
                c[key] = v
    return c


def _controls(controls: Mapping[str, Any] | None) -> tuple[float, float, float, float]:
    c = validate_controls(controls)
    return (float(np.clip(c['gain'], 0.0, 4.0)), float(np.clip(c['threshold_shift'], -1.0, 1.0)),
            float(np.clip(c['attention'], 0.0, 4.0)), float(np.clip(c['plasticity_scale'], 0.0, 4.0)))


@dataclass
class CompartmentDrive:
    """Per-compartment recurrent drive plus feed-forward external drive (length n each)."""

    proximal: np.ndarray
    distal: np.ndarray
    apical: np.ndarray
    basal: np.ndarray
    external: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> 'CompartmentDrive':
        return cls(*(np.zeros(n) for _ in range(5)))

    @property
    def total(self) -> np.ndarray:
        """Summed recurrent synaptic drive (the v0.1 ``incoming``)."""
        return self.proximal + self.distal + self.apical + self.basal

    def validate(self, n: int) -> None:
        for name in ('proximal', 'distal', 'apical', 'basal', 'external'):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.shape != (n,):
                raise ValueError(f'drive.{name} must be an array of shape ({n},)')
            if not np.all(np.isfinite(arr)):
                raise ValueError(f'drive.{name} must be finite')


class CellPopulation:
    """Vectorised population of multi-compartment synthetic units."""

    def __init__(self, n: int, *, seed: int, regions: Iterable[int] | np.ndarray, signs: Iterable[int] | np.ndarray):
        if type(n) is not int or n < 1:
            raise ValueError('n must be a positive int')
        region = np.asarray(regions, dtype=np.int64)
        sign = np.asarray(signs, dtype=np.int8)
        if region.shape != (n,) or sign.shape != (n,):
            raise ValueError('regions and signs must have length n')
        if not np.all(np.isin(sign, (-1, 1))):
            raise ValueError('signs must be +1/-1')
        self.n = n
        self.seed = int(seed)
        self.region = region.copy()
        self.sign = sign.copy()
        rng = np.random.default_rng(self.seed + _CELL_OFFSET)
        for name in STATE_FIELDS:
            setattr(self, name, self._default(name, n))
        self.myelin = 0.55 + 0.35 * rng.random(n)
        self.steps = 0

    # ------------------------------------------------------------------ defaults
    @staticmethod
    def _default(name: str, k: int) -> np.ndarray:
        if name == 'threshold':
            return np.full(k, BASE_THRESHOLD)
        if name in ('gating', 'homeo_gain', 'astrocyte'):
            return np.ones(k)
        if name == 'myelin':
            return np.full(k, 0.55)
        return np.zeros(k)

    def grow(self, k: int, region: int, signs: Iterable[int] | None = None) -> np.ndarray:
        """Append k neutral units in `region`; returns their indices."""
        if type(k) is not int or k < 0:
            raise ValueError('k must be a non-negative int')
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        new_sign = np.ones(k, dtype=np.int8) if signs is None else np.asarray(list(signs), dtype=np.int8)
        if new_sign.shape != (k,) or not np.all(np.isin(new_sign, (-1, 1))):
            raise ValueError('signs must be k values of +1/-1')
        for name in STATE_FIELDS:
            setattr(self, name, np.concatenate([getattr(self, name), self._default(name, k)]))
        self.region = np.concatenate([self.region, np.full(k, int(region), dtype=np.int64)])
        self.sign = np.concatenate([self.sign, new_sign])
        idx = np.arange(self.n, self.n + k, dtype=np.int64)
        self.n += k
        return idx

    def reset_units(self, idx) -> None:
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= self.n):
            raise ValueError('unit index out of range')
        for name in STATE_FIELDS:
            arr = getattr(self, name)
            arr[idx] = self._default(name, idx.size)

    # ------------------------------------------------------------------ update
    def step(self, drive: CompartmentDrive, flags: Iterable[str] = frozenset(), controls: Mapping[str, Any] | None = None,
             *, sparse_k: int | None = None) -> np.ndarray:
        flags = frozenset(flags)
        n = self.n
        drive.validate(n)
        gain, tshift, attention, pscale = _controls(controls)
        u = drive.external
        inc = drive.total
        prev_out = self.output.copy()
        apical_in = attention * drive.apical

        # Dendrites: leaky integrators with distinct time constants (passive state when flag off).
        self.proximal = 0.58 * self.proximal + 0.42 * (drive.proximal + 0.34 * u)
        self.basal = 0.70 * self.basal + 0.30 * (drive.basal + 0.20 * u)
        self.distal = 0.94 * self.distal + 0.06 * drive.distal
        self.apical = 0.86 * self.apical + 0.14 * apical_in
        if 'dendritic_compartments' in flags:
            apical_gate = 1.0 + 0.5 * np.tanh(2.0 * self.apical)          # context-dependent [0.5, 1.5]
            eff = self.proximal + apical_gate * self.basal + self.distal + 0.25 * self.apical
        else:
            eff = inc

        # v0.1 soma channels.
        self.fast = 0.58 * self.fast + u + 0.52 * eff
        self.slow = 0.94 * self.slow + 0.14 * u + 0.18 * eff
        self.context = 0.69 * self.context + attention * eff

        if 'multi_soma' in flags:
            fused = (0.35 * self.fast + 0.18 * self.slow + 0.19 * self.context + 0.13 * self.basal
                     + 0.08 * self.apical + 0.07 * self.trace)
            self.integrator = 0.91 * self.integrator + 0.09 * fused
            pre = fused + 0.1 * self.integrator
        else:
            pre = 0.84 * self.fast + 0.16 * eff

        if 'extended_somas' in flags:
            self.primary = 0.5 * self.primary + 0.5 * (u + eff)             # immediate integration
            self.secondary = 0.97 * self.secondary + 0.03 * np.tanh(self.primary)  # slow persistent context
            self.burst = 0.72 * self.burst + 0.28 * np.maximum(0.0, self.primary - (self.threshold + tshift))
            self.gating = np.clip(0.8 * self.gating + 0.2 * (1.0 + 0.35 * np.tanh(self.secondary)
                                                             - 0.1 * np.tanh(self.burst)), 0.65, 1.35)
            pre = self.gating * (pre + 0.25 * self.secondary + 0.2 * self.burst)

        if 'hierarchical_temporal_memory' in flags:
            pre = pre + 0.045 * (self.tm_fast - self.tm_mid) + 0.035 * (self.tm_mid - self.tm_slow)

        if 'predictive_coding' in flags:
            prediction = 0.58 * self.pred_estimate + 0.27 * self.fast + 0.15 * self.slow
            err = pre - prediction
            self.pred_error = err
            self.surprise = np.clip(0.94 * self.surprise + 0.06 * np.abs(err), 0.0, 2.0)
            pre = pre + np.clip(err, -1.5, 1.5) * 0.055 / (1.0 + self.surprise)
            self.pred_estimate = 0.91 * prediction + 0.09 * pre

        membrane = 'membrane_dynamics' in flags
        spike = np.zeros(n)
        if membrane:
            self.adaptation = np.clip(0.995 * self.adaptation + 0.005 * np.abs(prev_out), 0.0, 1.0)
            self.threshold = np.clip(np.clip(BASE_THRESHOLD + 0.18 * self.adaptation, 0.42, 0.78) + tshift, 0.05, 1.6)
            self.membrane = 0.78 * self.membrane + 0.22 * pre - 0.06 * self.adaptation
            self.refractory = np.maximum(0.0, self.refractory - 1.0)
            fired = (self.membrane > self.threshold) & (self.refractory == 0)
            self.membrane = np.where(fired, 0.26 * self.membrane, self.membrane)
            refractory_hold = self.refractory > 0
            self.refractory = np.where(fired, 2.0, self.refractory)
            spike = fired.astype(np.float64)
            self.calcium = np.clip(0.965 * self.calcium + 0.075 * spike + 0.008 * np.abs(pre), 0.0, 4.0)
            pre = pre + 0.15 * spike - 0.08 * self.calcium
            pre = np.where(refractory_hold, 0.7 * pre, pre)
        else:
            self.threshold = np.full(n, float(np.clip(BASE_THRESHOLD + tshift, 0.05, 1.6)))

        # Axon hillock is always tracked; branches shape the output only when enabled.
        self.hillock = 0.66 * self.hillock + 0.34 * np.where(spike > 0, 1.0, np.tanh(pre))
        if 'axonal_branches' in flags:
            self.branch_a = 0.72 * self.branch_a + 0.28 * (self.hillock + 0.2 * self.apical)
            self.branch_b = 0.84 * self.branch_b + 0.16 * (self.hillock - 0.15 * self.distal)
            self.terminal = 0.6 * self.branch_a + 0.4 * self.branch_b
            # myelin proxy sets axonal conduction efficacy of terminal release (0.6 .. ~0.99)
            self.release = np.tanh(self.terminal) * (0.4 + 0.6 * self.myelin)
            pre = pre + 0.12 * self.release
            self.myelin = np.clip(self.myelin + pscale * (0.0005 * np.abs(self.hillock) - 0.00008), 0.2, 0.98)

        a = np.tanh(gain * pre - tshift)
        if membrane:
            a = np.where(spike > 0, np.clip(a + 0.12 * self.hillock, -1.0, 1.0), a)

        if 'sparse_code' in flags and sparse_k is not None and 0 < int(sparse_k) < n:
            winners = np.argsort(-np.abs(a), kind='stable')[: int(sparse_k)]
            mask = np.zeros(n, dtype=bool)
            mask[winners] = True
            a = np.where(mask, a, 0.08 * a)

        if 'homeostatic_governor' in flags:
            self.homeo_gain = np.clip(self.homeo_gain + 0.02 * pscale * (HOMEO_TARGET - np.abs(prev_out)), 0.5, 1.5)
            a = a * self.homeo_gain
        if 'astrocyte_regulation' in flags:
            self.astrocyte = np.clip(0.97 * self.astrocyte + 0.03 / (1.0 + 1.6 * np.abs(prev_out)), 0.3, 1.2)
            a = a * np.clip(self.astrocyte, 0.58, 1.12)
        a = np.clip(a, -1.0, 1.0)

        if 'q88_state' in flags:
            a = q88(a)
            for name in ('fast', 'slow', 'context', 'membrane', 'integrator', 'proximal', 'distal', 'apical', 'basal'):
                setattr(self, name, q88(getattr(self, name)))

        if not membrane:
            spike = (a > self.threshold).astype(np.float64)
        self.spike = spike
        self.output = a

        rate = 0.008 * min(pscale, 4.0)
        self.trace = np.clip((1.0 - rate) * self.trace + rate * np.abs(a), 0.0, 1.0)
        if 'hierarchical_temporal_memory' in flags:
            self.tm_fast = 0.62 * self.tm_fast + 0.38 * np.abs(a)
            self.tm_mid = 0.94 * self.tm_mid + 0.06 * self.tm_fast
            self.tm_slow = 0.994 * self.tm_slow + 0.006 * self.tm_mid
        self.utility = 0.985 * self.utility + 0.015 * np.abs(a * np.tanh(self.fast))
        self._enforce_bounds()
        self.steps += 1
        return self.output

    def _enforce_bounds(self) -> None:
        for name in STATE_FIELDS:
            lo, hi = _BOUNDS[name]
            arr = getattr(self, name)
            np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi, out=arr)
            setattr(self, name, arr)

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {name: getattr(self, name).tolist() for name in STATE_FIELDS}
        state['n'] = int(self.n)
        state['seed'] = int(self.seed)
        state['steps'] = int(self.steps)
        state['region'] = self.region.tolist()
        state['sign'] = self.sign.astype(int).tolist()
        return state

    @staticmethod
    def validate_state(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(state, Mapping):
            raise ValueError('cell state must be a mapping')
        n = state.get('n')
        if type(n) is not int or n < 1:
            raise ValueError('cell state n must be a positive int')
        for key in ('seed', 'steps'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f'cell state {key} must be a non-negative int')
        out: dict[str, np.ndarray] = {}
        for name in STATE_FIELDS:
            if name not in state:
                raise ValueError(f'cell state missing {name}')
            try:
                arr = np.asarray(state[name], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'cell state {name} is not numeric') from exc
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise ValueError(f'cell state {name} must be {n} finite values')
            lo, hi = _BOUNDS[name]
            if arr.size and (arr.min() < lo - 1e-9 or arr.max() > hi + 1e-9):
                raise ValueError(f'cell state {name} out of bounds [{lo}, {hi}]')
            out[name] = arr
        try:
            region = np.asarray(state.get('region'), dtype=np.int64)
            sign = np.asarray(state.get('sign'), dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError('cell region/sign must be integer lists') from exc
        if region.shape != (n,) or sign.shape != (n,) or not np.all(np.isin(sign, (-1, 1))) or (region.size and region.min() < 0):
            raise ValueError('cell region/sign malformed')
        out['region'] = region
        out['sign'] = sign.astype(np.int8)
        return out

    def import_state(self, state: Mapping[str, Any]) -> None:
        arrays = self.validate_state(state)
        self.n = int(state['n'])
        self.seed = int(state['seed'])
        self.steps = int(state['steps'])
        for name, arr in arrays.items():
            setattr(self, name, arr.copy())
