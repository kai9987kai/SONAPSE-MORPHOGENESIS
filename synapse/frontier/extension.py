"""Engine-extension interface (v1) between the frontier pack and the NEUROWEAVE engine.

``NeuroWeaveEngine.attach_extension`` accepts any object with this shape. Every
hook on the base class is neutral, so a subclass only overrides what it uses.

Step context
------------
``ctx`` is a mapping the engine builds once per step:

``step`` (int), ``learn`` (bool), ``reward`` (float; the effective reward given
to the brain this step), ``raw_reward`` (float), ``task_input``
(``np.ndarray | None``), ``sensory`` (``np.ndarray`` of 16 fused channels),
``controls`` (the neuromodulatory controls mapping) and ``flags``
(``frozenset`` of enabled NEUROWEAVE mechanisms).

Rules
-----
* ``transform_drive`` may change the *values* of the external drive but must
  return the same shape (and ``None`` when it receives ``None``). The engine
  rejects anything else.
* An extension must not keep a reference to the engine: ``bind`` copies what it
  needs. ``copy.deepcopy(extension)`` must give an independent clone.
* All state is finite and bounded; randomness comes from a
  ``numpy.random.default_rng`` seeded in ``bind``/``__init__``.
* ``features()`` returns exactly ``feature_dim`` finite values in ``[-10, 10]``.
* With ``ctx['learn']`` false an extension must not adapt (frozen evaluation).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

EXTENSION_INTERFACE_VERSION = 1


class EngineExtension:
    """Neutral base class for NEUROWEAVE engine extensions."""

    id: str = 'extension'
    interface_version: int = EXTENSION_INTERFACE_VERSION
    feature_dim: int = 0

    # ------------------------------------------------------------------ lifecycle
    def bind(self, engine: Any) -> None:
        """Size the extension from the engine (``engine.n``, regions, seed). Must not keep ``engine``."""

    def reset(self) -> None:
        """Forget cross-step transients (episode boundary)."""

    # ------------------------------------------------------------------ hooks
    def transform_drive(self, external: np.ndarray | None, ctx: Mapping[str, Any]) -> np.ndarray | None:
        return external

    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any]) -> None:
        """Called after the brain step with the brain output (a copy)."""

    def features(self) -> np.ndarray:
        """Extra readout features (length ``feature_dim``)."""
        return np.zeros(self.feature_dim)

    # ------------------------------------------------------------------ accounting / persistence
    def parameter_count(self) -> int:
        """Trainable parameters, for budget matching."""
        return 0

    def state_numbers(self) -> int:
        """Total stored numbers (parameters + state), for budget matching."""
        return 0

    def telemetry(self) -> dict[str, Any]:
        return {'id': self.id}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id}

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('id') != self.id:
            raise ValueError(f'state does not belong to extension {self.id!r}')


class RandomTraceControl(EngineExtension):
    """Capacity-matched control: random leaky traces of the brain output.

    It gives the readout the same number of extra features (and comparable
    temporal memory) as a candidate extension, without the candidate's
    mechanism. A candidate is only credited when it beats this control.
    """

    id = 'random_trace_control'

    def __init__(self, feature_dim: int, seed: int = 0, *, decay_range: tuple[float, float] = (0.5, 0.98)) -> None:
        if type(feature_dim) is not int or not 1 <= feature_dim <= 4096:
            raise ValueError('feature_dim must be an int in [1, 4096]')
        lo, hi = float(decay_range[0]), float(decay_range[1])
        if not 0.0 <= lo <= hi < 1.0:
            raise ValueError('decay_range must satisfy 0 <= lo <= hi < 1')
        self.feature_dim = feature_dim
        self.seed = int(seed)
        self.decay_range = (lo, hi)
        self._n = 0
        self._proj = np.zeros((feature_dim, 0))
        self._decay = np.linspace(lo, hi, feature_dim)
        self._trace = np.zeros(feature_dim)

    def bind(self, engine: Any) -> None:
        self._n = int(engine.n)
        rng = np.random.default_rng(self.seed + 40_009)
        self._proj = rng.standard_normal((self.feature_dim, self._n)) / np.sqrt(max(1, self._n))
        self._trace = np.zeros(self.feature_dim)

    def reset(self) -> None:
        self._trace = np.zeros(self.feature_dim)

    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any]) -> None:
        o = np.zeros(self._n)
        m = min(self._n, output.size)        # neurons born later are ignored (fixed shape)
        o[:m] = output[:m]
        self._trace = self._decay * self._trace + (1.0 - self._decay) * np.tanh(self._proj @ o)

    def features(self) -> np.ndarray:
        return np.clip(self._trace, -10.0, 10.0)

    def state_numbers(self) -> int:
        return int(self._proj.size + self._decay.size + self._trace.size)

    def telemetry(self) -> dict[str, Any]:
        return {'id': self.id, 'feature_dim': self.feature_dim, 'trace_norm': float(np.linalg.norm(self._trace))}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id, 'feature_dim': self.feature_dim, 'seed': self.seed, 'n': self._n,
                'decay_range': list(self.decay_range), 'trace': self._trace.tolist()}

    def import_state(self, state: Mapping[str, Any]) -> None:
        super().import_state(state)
        trace = np.asarray(state.get('trace'), dtype=np.float64)
        if (state.get('feature_dim') != self.feature_dim or state.get('n') != self._n
                or state.get('seed') != self.seed or state.get('decay_range') != list(self.decay_range)
                or trace.shape != (self.feature_dim,) or not np.all(np.isfinite(trace))
                or np.any(np.abs(trace) > 1.0)):
            raise ValueError('random_trace_control state does not match this extension')
        self._trace = trace.copy()
