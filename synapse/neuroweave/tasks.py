"""Deterministic seeded temporal benchmark tasks for the NEUROWEAVE laboratory.

Modelled on NEUROWEAVE-LAB 0.4.0 ``makeStimulus`` (echo-2, echo-6, XOR, parity
and a phase-switching rule) and extended with a rhythm/periodicity task. Every
task is a binary target over one shared 8-d input stream, so several tasks can
be scored on the same run (the lesion assay uses this for a memory probe).

Deviation from NEUROWEAVE (deliberate): NEUROWEAVE copies the two previous
bits into the input, which turns XOR/parity into memoryless functions of the
current input. Here the past is *not* given; the system must hold it in its
recurrent state. Inputs (all in [-1, 1]):

====  ===============================================================
0     current random bit (+1 / -1)
1     sin(0.035 t)         fast clock
2     cos(0.021 t)         second clock
3     sin(0.006 t)         slow clock
4     context cue: +/-1 for the first ``cue_steps`` of each context
      block, 0 afterwards (the rule must be remembered)
5     rhythm pulse: +1 on pulse steps, -1 otherwise
6     uniform noise in [-0.5, 0.5]
7     constant bias 1.0
====  ===============================================================

A 16-channel sensory vector (``sensory``) carries noisy, partly missing
(``None``) redundant copies of the task signals; it is consumed only when an
engine enables ``multimodal_sensory``.

Streams are pure functions of ``(seed, steps, params)``; train and held-out
streams differ only by seed. Everything is a synthetic benchmark
(``evidence_tier='synthetic_abstraction'``).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .spec import EVIDENCE_TIER, SENSORY_CHANNEL_IDS, SENSORY_DIM, TASK_INPUT_DIM

__all__ = ['TASKS', 'TASK_DESCRIPTIONS', 'TaskStream', 'make_stream', 'task_targets', 'chance_accuracy',
           'TASK_SEED_OFFSET']

TASK_SEED_OFFSET = 717          # NEUROWEAVE uses seed + 717 for its stimulus RNG
TASK_DESCRIPTIONS: dict[str, str] = {
    'delayed_recall': 'target = bit two steps ago (NEUROWEAVE echo2)',
    'delayed_recall_6': 'target = bit six steps ago (NEUROWEAVE echo6)',
    'xor': 'target = bit(t) XOR bit(t-1)',
    'parity': 'target = bit(t) XOR bit(t-1) XOR bit(t-2)',
    'rhythm': 'target = 1 when the next step carries a rhythm pulse (period P, seeded phase)',
    'context_switch': 'target = bit(t) XOR bit(t-1) in context A, bit(t) XOR bit(t-2) in context B; '
                      'the context is cued only at the start of each block',
}
TASKS: tuple[str, ...] = tuple(TASK_DESCRIPTIONS)
_CHANNEL = {c: i for i, c in enumerate(SENSORY_CHANNEL_IDS)}


@dataclass(frozen=True)
class TaskStream:
    """One deterministic stream. ``inputs`` (T, 8), ``targets`` task -> (T,) int8, ``sensory`` list of dicts."""

    seed: int
    steps: int
    inputs: np.ndarray
    targets: dict[str, np.ndarray]
    sensory: tuple[dict[str, float | None], ...]
    params: dict[str, Any]

    def target(self, task: str) -> np.ndarray:
        if task not in self.targets:
            raise ValueError(f'unknown task {task!r}; expected one of {TASKS}')
        return self.targets[task]

    @property
    def digest(self) -> str:
        payload = {'inputs': self.inputs.round(12).tolist(),
                   'targets': {k: v.tolist() for k, v in sorted(self.targets.items())},
                   'params': self.params, 'seed': self.seed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {'seed': self.seed, 'steps': self.steps, 'params': dict(self.params), 'digest': self.digest,
                'positive_rate': {k: float(v.mean()) if v.size else 0.0 for k, v in sorted(self.targets.items())},
                'evidence_tier': EVIDENCE_TIER}


def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


def task_targets(bits: np.ndarray, t0: int = 0, *, period: int = 5, phase: int = 0, block: int = 40) -> dict[str, np.ndarray]:
    """Targets for every task from a bit sequence (pure). ``t0`` is the absolute time of ``bits[0]``."""
    b = np.asarray(bits, dtype=np.int64)
    T = b.size
    lag = lambda k: np.concatenate([np.zeros(min(k, T), np.int64), b[:max(0, T - k)]])  # noqa: E731
    p1, p2, p6 = lag(1), lag(2), lag(6)
    t = t0 + np.arange(T)
    ctx = (t // block) % 2
    return {
        'delayed_recall': p2.astype(np.int8),
        'delayed_recall_6': p6.astype(np.int8),
        'xor': (b ^ p1).astype(np.int8),
        'parity': (b ^ p1 ^ p2).astype(np.int8),
        'rhythm': (((t + 1 + phase) % period) == 0).astype(np.int8),
        'context_switch': np.where(ctx == 0, b ^ p1, b ^ p2).astype(np.int8),
    }


def make_stream(seed: int, steps: int, *, period: int = 5, block: int = 40, cue_steps: int = 3,
                sensory_noise: float = 0.15, missing_rate: float = 0.2) -> TaskStream:
    """Build a deterministic stream from ``numpy.random.default_rng(seed + 717)``."""
    _check_int('seed', seed, 0, 2**31 - 1)
    _check_int('steps', steps, 1, 1_000_000)
    _check_int('period', period, 2, 1000)
    _check_int('block', block, 2, 1_000_000)
    _check_int('cue_steps', cue_steps, 1, block)
    for name, v in (('sensory_noise', sensory_noise), ('missing_rate', missing_rate)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')
    rng = np.random.default_rng(seed + TASK_SEED_OFFSET)
    phase = int(rng.integers(0, period))
    bits = (rng.random(steps) >= 0.5).astype(np.int64)
    noise = rng.random(steps) - 0.5
    t = np.arange(steps, dtype=np.float64)
    ctx = ((np.arange(steps) // block) % 2).astype(np.float64)
    cue = np.where((np.arange(steps) % block) < cue_steps, np.where(ctx == 0, 1.0, -1.0), 0.0)
    pulse = np.where(((np.arange(steps) + phase) % period) == 0, 1.0, -1.0)
    x = np.zeros((steps, TASK_INPUT_DIM))
    x[:, 0] = np.where(bits == 1, 1.0, -1.0)
    x[:, 1] = np.sin(t * 0.035)
    x[:, 2] = np.cos(t * 0.021)
    x[:, 3] = np.sin(t * 0.006)
    x[:, 4] = cue
    x[:, 5] = pulse
    x[:, 6] = noise
    x[:, 7] = 1.0
    targets = task_targets(bits, 0, period=period, phase=phase, block=block)
    # sensory: noisy redundant copies; independent stream so inputs do not depend on sensory params
    srng = np.random.default_rng(seed + TASK_SEED_OFFSET + 1)
    sn = float(sensory_noise)
    eps = srng.normal(0.0, sn, (steps, SENSORY_DIM))
    miss = srng.random((steps, SENSORY_DIM)) < float(missing_rate)
    sources = {
        'visual_luminance': x[:, 0], 'audio_mid': x[:, 0], 'mechanosensation': 0.5 * x[:, 0],
        'audio_low': x[:, 5], 'visual_motion': 0.5 * x[:, 5],
        'task_context': x[:, 4], 'environment_state': x[:, 3], 'proprioception': x[:, 1],
    }
    sensory: list[dict[str, float | None]] = []
    for i in range(steps):
        row: dict[str, float | None] = {}
        for cid in SENSORY_CHANNEL_IDS:
            j = _CHANNEL[cid]
            base = sources[cid][i] if cid in sources else 0.0
            row[cid] = None if miss[i, j] else float(np.clip(base + eps[i, j], -1.0, 1.0))
        sensory.append(row)
    params = {'period': period, 'phase': phase, 'block': block, 'cue_steps': cue_steps,
              'sensory_noise': sn, 'missing_rate': float(missing_rate)}
    return TaskStream(seed=seed, steps=steps, inputs=x, targets=targets, sensory=tuple(sensory), params=params)


def chance_accuracy(targets: np.ndarray) -> float:
    """Accuracy of the best constant predictor (majority class)."""
    t = np.asarray(targets, dtype=np.float64)
    if t.size == 0:
        return 0.0
    m = float(t.mean())
    return max(m, 1.0 - m)

