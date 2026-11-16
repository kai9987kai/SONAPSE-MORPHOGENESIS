"""Sonapse graft-bus adapter for the NEUROWEAVE engine (the only torch module in this package).

``NeuroWeaveBusChannel`` follows the ``DSAGBusChannel`` contract: it proposes a
bounded residual current over the Sonapse brain's ``n`` neurons and leaves the
final global clipping to ``NeuralGraftCommunicationBus``.

* Disabled: returns ``torch.zeros_like(r)`` and never touches the engine.
* ``observe_only``: the engine steps (telemetry, learning) but the returned
  current is exactly zero.
* Otherwise the current is ``g_eff * P_out(engine.output)`` where ``P_out`` is
  a fixed seeded sparse projection and ``g_eff`` ramps linearly from 0 to
  ``spec.gain`` over ``spec.ramp_steps`` engine steps, clipped to
  ``spec.max_gain``.

The engine advances once per brain forward pass (microstep 0 with
``advance=True``); later microsteps reuse the cached current and
``advance=False`` calls are read-only.
"""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from time import perf_counter_ns
from typing import Any, Mapping

import numpy as np
import torch

from ..graft_bus import NeuralGraftProtocol
from .engine import NeuroWeaveEngine
from .spec import EVIDENCE_TIER, SENSORY_CHANNEL_IDS, TASK_INPUT_DIM, NeuroWeaveSpecV1

__all__ = ['NeuroWeaveBusChannel', 'CHECKPOINT_SCHEMA', 'obs_to_sensory']

CHECKPOINT_SCHEMA = 'NeuroWeaveCheckpointV1'
BRIDGE_OFFSET = 9_001
OUT_FAN = 4
_VISUAL_FROM_CAMERA = {'visual_luminance': 'luminance', 'visual_red_green': 'red_green',
                       'visual_blue_yellow': 'blue_yellow', 'visual_edge_density': 'edge_density',
                       'visual_motion': 'motion'}


def _finite_number(value: Any, name: str) -> float:
    """A checkpoint scalar: a real int/float (not bool, not a string such as 'Infinity') and finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def obs_to_sensory(obs: Any, previous_depth: np.ndarray | None = None) -> tuple[dict[str, float | None], np.ndarray]:
    """Derive the 16 NEUROWEAVE channels from a Sonapse 155-d observation.

    Layout: depth[0:33], red[33:66], green[66:99], blue[99:132], proprioception[132:140]
    (speed 136, turn 137), task one-hot[140:149], cue[149:153] (target lateral/forward,
    energy, flag), last reward 153, time 154. Channels without a Sonapse source are
    ``None`` (missing) so the hub can flag or cross-modally impute them.
    Returns ``(channels, depth)``; ``depth`` (33 derived floats) is the motion reference.
    """
    o = np.asarray(obs, dtype=np.float64).ravel()
    if o.shape != (155,) or not np.all(np.isfinite(o)):
        raise ValueError('observation must be finite shape (155,)')
    depth = np.clip(o[:33], 0.0, 1.0)
    r, g, b = o[33:66], o[66:99], o[99:132]
    w = depth / max(1e-6, float(depth.sum())) if depth.sum() > 0 else np.full(33, 1.0 / 33)
    motion = 0.0 if previous_depth is None else float(np.mean(np.abs(depth - previous_depth))) * 4.0
    task = o[140:149]
    ch: dict[str, float | None] = {
        'visual_luminance': float(np.mean(depth * (r + g + b) / 3.0)) * 2.0 - 0.5,
        'visual_red_green': float(w @ (r - g)),
        'visual_blue_yellow': float(w @ (b - 0.5 * (r + g))),
        'visual_edge_density': float(np.mean(np.abs(np.diff(depth)))) * 4.0,
        'visual_motion': motion,
        'audio_low': None,
        'audio_mid': float(abs(o[136])),
        'audio_high': None,
        'olfactory': float(o[150]) * 2.0 - 1.0,
        'gustatory': math.tanh(float(o[153])),
        'thermal': None,
        'interoception': float(o[151]) * 2.0 - 1.0,
        'proprioception': 0.7 * float(o[136]) + 0.3 * float(o[137]),
        'mechanosensation': float(depth.max()),
        'environment_state': float(depth.mean()) * 2.0 - 1.0,
        'task_context': (float(np.argmax(task)) / 8.0 * 2.0 - 1.0) if task.any() else 0.0,
    }
    out = {k: (None if v is None else float(np.clip(v, -1.0, 1.0))) for k, v in ch.items()}
    assert tuple(out) == SENSORY_CHANNEL_IDS
    return out, depth.copy()


class NeuroWeaveBusChannel(NeuralGraftProtocol):
    """Residual NEUROWEAVE current for the Sonapse graft bus (zero-gain neutral)."""

    def __init__(self, spec: NeuroWeaveSpecV1, input_dim: int, name: str = 'neuroweave'):
        if not isinstance(spec, NeuroWeaveSpecV1):
            raise TypeError('spec must be a NeuroWeaveSpecV1')
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError('input_dim must be a positive int')
        self.spec = spec
        self.input_dim = input_dim
        self._name = name
        self.engine = NeuroWeaveEngine(spec)
        rng = np.random.default_rng(spec.seed + BRIDGE_OFFSET)
        self.in_proj = rng.standard_normal((TASK_INPUT_DIM, input_dim)) * (2.0 / math.sqrt(input_dim))
        self.out_idx = rng.integers(0, spec.neurons, size=(input_dim, OUT_FAN))
        self.out_w = rng.standard_normal((input_dim, OUT_FAN)) / math.sqrt(OUT_FAN)
        self._enabled = spec.enabled
        self._lineage_id = hashlib.sha256((spec.spec_digest + str(input_dim)).encode()).hexdigest()[:16]
        self._reset_runtime()
        self._last_latency_ms = 0.0

    def _reset_runtime(self) -> None:
        self.engine_steps = 0
        self._cached: np.ndarray | None = None
        self._pending_sensory: dict[str, float | None] | None = None
        self._pending_reward = 0.0
        self._pending_done = False
        self._prev_depth: np.ndarray | None = None
        self._camera_frames = 0
        self._camera_rejected = 0
        self._outcomes = 0
        self._last_current_inf_norm = 0.0
        self._last_step: dict[str, Any] = {}

    # ------------------------------------------------------------------ protocol
    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self.spec.enabled)

    def set_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError('enabled must be a boolean')
        self._enabled = value

    def reset(self, batch_size: int = 1) -> None:
        """Episode boundary: drop the cached current and cross-step engine transients (learning is kept)."""
        self._cached = None
        self._prev_depth = None
        self.engine.end_episode()

    def effective_gain(self) -> float:
        """0 before the first engine step; ramps to ``spec.gain`` over ``ramp_steps``; clipped to ``max_gain``."""
        if self.engine_steps <= 0:
            return 0.0
        g = self.spec.gain
        if self.spec.ramp_steps > 0:
            g *= min(1.0, (self.engine_steps - 1) / self.spec.ramp_steps)
        return float(min(g, self.spec.max_gain))

    def _advance(self, r: torch.Tensor, adapt: bool) -> None:
        rv = r.detach().to('cpu', torch.float64).flatten().numpy()
        x = np.zeros(self.input_dim)
        m = min(self.input_dim, rv.size)
        x[:m] = rv[:m]
        task = np.tanh(self.in_proj @ (x - x.mean()))
        res = self.engine.step(task, self._pending_sensory, reward=self._pending_reward, learn=bool(adapt))
        self._pending_sensory = None
        self._pending_reward = 0.0
        out = res['output']
        o = np.zeros(self.spec.neurons)
        k = min(o.size, out.size)
        o[:k] = out[:k]
        self._cached = (self.out_w * o[self.out_idx]).sum(axis=1)
        self.engine_steps += 1
        self._last_step = {'step': int(res['step']), 'readout_probability': float(res['readout_probability']),
                           'sensory_uncertainty': float(res['sensory']['uncertainty']),
                           'prediction_surprise': float(res['prediction']['surprise']),
                           'uncertainty_gate': float(res['prediction']['gate']), 'events': int(res['events'])}

    def compute_current(self, r: torch.Tensor, microstep: int = 0, advance: bool = True, adapt: bool = True,
                        phase_angles: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(r)
        if advance and (int(microstep) == 0 or self._cached is None):
            started = perf_counter_ns()
            self._advance(r, adapt)
            self._last_latency_ms = (perf_counter_ns() - started) / 1_000_000.0
        if self.spec.observe_only or self._cached is None:
            return torch.zeros_like(r)
        gain = self.effective_gain()
        cur = gain * self._cached
        n = r.numel()
        full = np.zeros(n)
        m = min(n, cur.size)
        full[:m] = cur[:m]
        self._last_current_inf_norm = float(np.abs(full).max()) if n else 0.0
        return torch.as_tensor(full, dtype=r.dtype, device=r.device).reshape(r.shape)

    # ------------------------------------------------------------------ Sonapse hooks
    def observe_sensory(self, obs155: Any, camera: Any = None) -> dict[str, float | None]:
        """Queue the 16 derived channels for the next engine step. Camera frames yield scalars only."""
        channels, depth = obs_to_sensory(obs155, self._prev_depth)
        self._prev_depth = depth
        if camera is not None:
            try:
                frame = np.asarray(camera, dtype=np.float64)
                if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
                    frame = np.moveaxis(frame, 0, -1)
                if frame.ndim == 2:
                    frame = np.repeat(frame[:, :, None], 3, axis=2)
                if frame.ndim != 3 or frame.shape[-1] not in (1, 3) or not np.all(np.isfinite(frame)):
                    raise ValueError('unsupported camera layout')
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=2)
                if frame.max() > 1.0:
                    frame = frame / 255.0
                feats = self.engine.sensory.push_visual(np.clip(frame, 0.0, 1.0))  # keeps derived scalars only
                for cid, key in _VISUAL_FROM_CAMERA.items():
                    channels[cid] = float(np.clip(feats[key], -1.0, 1.0))
                self._camera_frames += 1
            except (ValueError, TypeError):
                self._camera_rejected += 1
        self._pending_sensory = channels
        return dict(channels)

    def observe_outcome(self, reward: float, done: bool = False) -> None:
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError('reward must be finite')
        self._pending_reward = float(np.clip(reward, -10.0, 10.0))
        self._pending_done = bool(done)
        self._outcomes += 1
        if done:
            self.engine.end_episode()
            self._prev_depth = None

    # ------------------------------------------------------------------ telemetry
    def get_telemetry_dict(self, include_timing: bool = False) -> dict[str, Any]:
        if type(include_timing) is not bool:
            raise ValueError('include_timing must be a boolean')
        values = {
            'schema': 'NeuroWeaveChannelTelemetryV1', 'enabled': self.enabled,
            'observe_only': self.spec.observe_only, 'gain_configured': self.spec.gain,
            'max_gain': self.spec.max_gain, 'effective_gain': self.effective_gain(),
            'engine_steps': int(self.engine_steps), 'input_dim': self.input_dim,
            'proposed_current_inf_norm': self._last_current_inf_norm,
            'camera_frames': self._camera_frames, 'camera_rejected': self._camera_rejected,
            'outcomes': self._outcomes, 'mechanisms': list(self.spec.mechanisms), 'lineage_id': self._lineage_id,
            'spec_digest': self.spec.spec_digest, 'last_step': dict(self._last_step),
            'engine': self.engine.telemetry(include_timing=False),
            'raw_media_stored': False, 'evidence_tier': EVIDENCE_TIER,
            'telemetry_namespace': f'neuroweave/{self._name}',
        }
        if include_timing:
            values['latency_ms'] = self._last_latency_ms
        return values

    # ------------------------------------------------------------------ persistence
    def _runtime_state(self) -> dict[str, Any]:
        return {
            'engine_steps': int(self.engine_steps),
            'cached': None if self._cached is None else self._cached.tolist(),
            'pending_sensory': None if self._pending_sensory is None else dict(self._pending_sensory),
            'pending_reward': float(self._pending_reward), 'pending_done': bool(self._pending_done),
            'prev_depth': None if self._prev_depth is None else self._prev_depth.tolist(),
            'camera_frames': int(self._camera_frames), 'camera_rejected': int(self._camera_rejected),
            'outcomes': int(self._outcomes), 'last_current_inf_norm': float(self._last_current_inf_norm),
            'last_step': deepcopy(self._last_step), 'enabled_override': bool(self._enabled),
        }

    def export_state(self) -> dict[str, Any]:
        payload = {
            'schema': CHECKPOINT_SCHEMA, 'schema_version': 1, 'evidence_tier': EVIDENCE_TIER,
            'spec': self.spec.to_dict(), 'spec_digest': self.spec.spec_digest, 'input_dim': self.input_dim,
            'engine': self.engine.export_state(), 'runtime': self._runtime_state(),
            'raw_media_stored': False,
        }
        payload['digest'] = _canonical_digest(payload)
        return payload

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('schema') != CHECKPOINT_SCHEMA or state.get('schema_version') != 1:
            raise ValueError('not a NeuroWeaveCheckpointV1 state')
        body = {k: v for k, v in state.items() if k != 'digest'}
        try:
            digest = _canonical_digest(body)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'NEUROWEAVE checkpoint is not canonical JSON: {exc}') from exc
        if state.get('digest') != digest:
            raise ValueError('NEUROWEAVE checkpoint digest mismatch')
        if state.get('spec_digest') != self.spec.spec_digest:
            raise ValueError('NEUROWEAVE spec digest mismatch')
        if state.get('input_dim') != self.input_dim:
            raise ValueError('NEUROWEAVE checkpoint host dimension mismatch')
        rt = state.get('runtime')
        if not isinstance(rt, Mapping):
            raise ValueError('runtime must be a mapping')
        try:
            steps = rt['engine_steps']
            if type(steps) is not int or steps < 0:
                raise ValueError('engine_steps invalid')
            cached = rt['cached']
            if cached is not None:
                cached = np.asarray(cached, dtype=np.float64)
                if cached.shape != (self.input_dim,) or not np.all(np.isfinite(cached)):
                    raise ValueError('cached current malformed')
            ps = rt['pending_sensory']
            if ps is not None:
                if not isinstance(ps, Mapping) or set(ps) - set(SENSORY_CHANNEL_IDS):
                    raise ValueError('pending sensory malformed')
                ps = {k: (None if v is None else _finite_number(v, 'pending sensory')) for k, v in ps.items()}
            reward = _finite_number(rt['pending_reward'], 'pending reward')
            pdepth = rt['prev_depth']
            if pdepth is not None:
                pdepth = np.asarray(pdepth, dtype=np.float64)
                if pdepth.shape != (33,) or not np.all(np.isfinite(pdepth)):
                    raise ValueError('prev_depth malformed')
            ints = {k: rt[k] for k in ('camera_frames', 'camera_rejected', 'outcomes')}
            if any(type(v) is not int or v < 0 for v in ints.values()):
                raise ValueError('runtime counters malformed')
            last_norm = _finite_number(rt['last_current_inf_norm'], 'last_current_inf_norm')
            last_step = rt['last_step']
            if not isinstance(last_step, Mapping):
                raise ValueError('last_step malformed')
            for k, v in last_step.items():
                _finite_number(v, f'last_step.{k}')
            enabled = rt['enabled_override']
            if type(enabled) is not bool or type(rt['pending_done']) is not bool:
                raise ValueError('runtime booleans malformed')
        except (KeyError, TypeError) as exc:
            raise ValueError(f'malformed NEUROWEAVE runtime: {exc}') from exc
        self.engine.import_state(state['engine'])   # validates fully before mutating
        self.engine_steps = steps
        self._cached = cached
        self._pending_sensory = ps
        self._pending_reward = reward
        self._pending_done = rt['pending_done']
        self._prev_depth = pdepth
        self._camera_frames, self._camera_rejected, self._outcomes = (ints['camera_frames'], ints['camera_rejected'],
                                                                       ints['outcomes'])
        self._last_current_inf_norm = last_norm
        self._last_step = dict(last_step)
        self._enabled = enabled
