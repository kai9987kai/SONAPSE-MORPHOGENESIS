"""Live step-latency telemetry and deterministic scaling/Pareto benchmark lab.

Contract & Scientific Integrity:
- ``StepPerfTracker`` records real measured wall-clock per-stage timings on the live
  ``LabRuntime`` using a bounded ring buffer. It never mutates agent weights, world
  state, RNG streams, or checkpoint payloads.
- ``run_scaling_profile`` and ``run_subsystem_pareto_sweep`` always execute on fresh
  isolated ``LabRuntime`` instances (never touching a live runtime) and seal receipts
  with a wall-clock-independent SHA-256 digest covering configuration and deterministic
  trajectory fingerprints while keeping measured wall-clock and CPU timings explicit
  and unfabricated.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from .config import SynapseConfig
from .stateio import digest as state_digest

__all__ = [
    'PERF_TELEMETRY_SCHEMA',
    'PERF_RECEIPT_SCHEMA',
    'PARETO_ARM_IDS',
    'StepPerfTracker',
    'perf_receipt_digest',
    'seal_perf_receipt',
    'verify_perf_receipt',
    'run_scaling_profile',
    'run_subsystem_pareto_sweep',
]

PERF_TELEMETRY_SCHEMA = 'StepPerfTelemetryV1'
PERF_RECEIPT_SCHEMA = 'PerfLabReceiptV1'
PARETO_ARM_IDS: tuple[str, ...] = (
    'baseline',
    'experience_planning',
    'dsag',
    'neuroweave',
    'frontier_observers',
    'dendritic_oscillatory',
    'neural_graft',
    'morphogenesis',
)
_WALLCLOCK_KEYS = frozenset({
    'measured_wallclock_ms',
    'measured_cpu_ms',
    'init_ms',
    'step_wall_ms',
    'step_cpu_ms',
    'mean_step_ms',
    'p50_step_ms',
    'p95_step_ms',
    'min_step_ms',
    'max_step_ms',
    'snapshot_ms',
    'measured_steps_per_sec',
    'measured_neuron_steps_per_sec',
    'overhead_ms_vs_baseline',
    'overhead_ratio_vs_baseline',
    'stage_mean_ms',
    'bottleneck_stage',
    'pareto_efficient',
    'scaling_efficiency',
    'digest',
})


def _strip_wallclock(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {
            str(k): _strip_wallclock(v)
            for k, v in obj.items()
            if str(k) not in _WALLCLOCK_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_strip_wallclock(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def perf_receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Wall-clock-independent SHA-256 digest over deterministic config & trajectories."""
    canonical = json.dumps(_strip_wallclock(receipt), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def seal_perf_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out['schema'] = PERF_RECEIPT_SCHEMA
    out['schema_version'] = 1
    out['evidence_tier'] = 'measured_runtime_wallclock'
    out['digest'] = perf_receipt_digest(out)
    return out


def verify_perf_receipt(receipt: Mapping[str, Any]) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    if receipt.get('schema') != PERF_RECEIPT_SCHEMA or receipt.get('schema_version') != 1:
        return False
    recorded = receipt.get('digest')
    if not isinstance(recorded, str) or len(recorded) != 64:
        return False
    return recorded == perf_receipt_digest(receipt)


class StepPerfTracker:
    """Bounded O(1) rolling per-stage wall-clock latency tracker for ``LabRuntime``."""

    __slots__ = ('window', '_samples', 'total_recorded', 'last_snapshot_ms')

    def __init__(self, window: int = 128) -> None:
        if type(window) is not int or not 4 <= window <= 4096:
            raise ValueError('window must be an int in [4, 4096]')
        self.window = window
        self._samples: deque[tuple[float, float, float, float, float]] = deque(maxlen=window)
        self.total_recorded = 0
        self.last_snapshot_ms: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self.total_recorded = 0
        self.last_snapshot_ms = None

    def record(
        self,
        total_ms: float,
        act_ms: float,
        uncertainty_ms: float,
        world_ms: float,
        learn_ms: float,
    ) -> None:
        self._samples.append((total_ms, act_ms, uncertainty_ms, world_ms, learn_ms))
        self.total_recorded += 1

    def record_snapshot(self, snapshot_ms: float) -> None:
        self.last_snapshot_ms = float(snapshot_ms)

    def summary(
        self,
        *,
        neurons: int = 0,
        active_subsystems: Sequence[str] = (),
    ) -> dict[str, Any]:
        count = len(self._samples)
        if count == 0:
            return {
                'schema': PERF_TELEMETRY_SCHEMA,
                'schema_version': 1,
                'evidence_tier': 'measured_runtime_wallclock',
                'neurons': int(neurons),
                'recorded_steps': int(self.total_recorded),
                'window_samples': 0,
                'window_capacity': int(self.window),
                'total_ms': {'mean': None, 'p50': None, 'p95': None, 'min': None, 'max': None, 'last': None},
                'stage_mean_ms': {'act': None, 'uncertainty': None, 'world': None, 'learn': None, 'overhead': None},
                'last_snapshot_ms': None if self.last_snapshot_ms is None else round(self.last_snapshot_ms, 3),
                'measured_steps_per_sec': None,
                'measured_neuron_steps_per_sec': None,
                'bottleneck_stage': None,
                'active_subsystems': list(active_subsystems),
            }

        arr = np.asarray(self._samples, dtype=np.float64)
        totals = arr[:, 0]
        mean_total = float(totals.mean())
        p50 = float(np.percentile(totals, 50))
        p95 = float(np.percentile(totals, 95))
        act_m = float(arr[:, 1].mean())
        unc_m = float(arr[:, 2].mean())
        world_m = float(arr[:, 3].mean())
        learn_m = float(arr[:, 4].mean())
        overhead_m = max(0.0, mean_total - (act_m + unc_m + world_m + learn_m))
        stages = {
            'act': round(act_m, 4),
            'uncertainty': round(unc_m, 4),
            'world': round(world_m, 4),
            'learn': round(learn_m, 4),
            'overhead': round(overhead_m, 4),
        }
        bottleneck = max(stages, key=stages.get)
        sps = (1000.0 / mean_total) if mean_total > 1e-9 else 0.0
        return {
            'schema': PERF_TELEMETRY_SCHEMA,
            'schema_version': 1,
            'evidence_tier': 'measured_runtime_wallclock',
            'neurons': int(neurons),
            'recorded_steps': int(self.total_recorded),
            'window_samples': count,
            'window_capacity': int(self.window),
            'total_ms': {
                'mean': round(mean_total, 4),
                'p50': round(p50, 4),
                'p95': round(p95, 4),
                'min': round(float(totals.min()), 4),
                'max': round(float(totals.max()), 4),
                'last': round(float(totals[-1]), 4),
            },
            'stage_mean_ms': stages,
            'last_snapshot_ms': None if self.last_snapshot_ms is None else round(self.last_snapshot_ms, 3),
            'measured_steps_per_sec': round(sps, 2),
            'measured_neuron_steps_per_sec': round(sps * float(neurons), 1),
            'bottleneck_stage': bottleneck,
            'active_subsystems': list(active_subsystems),
        }


def active_subsystem_list(config: SynapseConfig) -> list[str]:
    """Return a deterministic list of enabled opt-in subsystems for a ``SynapseConfig``."""
    names: list[str] = []
    flag_names = (
        'experience_planning', 'surprise_memory', 'experimental_cognition',
        'continual_backprop', 'fast_weight_memory', 'latent_reasoning',
        'value_guided_planning', 'self_model', 'neural_graft', 'active_training',
        'programmed_connectome', 'dendritic_ltc', 'oscillatory_coherence',
        'verifiable_dpo', 'dvgem_memory', 'tether_coordinator',
        'criticality_homeostat', 'holographic_vsa', 'cross_connectome_graft',
        'aether_graft_bridge', 'graft_bus_enabled', 'neural_comm_enabled',
        'neurowave_x2_enabled', 'neuroendocrine', 'sleep_replay',
    )
    for flag in flag_names:
        if getattr(config, flag, False):
            names.append(flag)
    for sub in ('dsag', 'neuroweave', 'frontier', 'morphogenesis'):
        spec = getattr(config, sub, None)
        if spec is not None and getattr(spec, 'enabled', False):
            names.append(sub)
    return names


def _build_pareto_arm_config(
    arm_id: str,
    *,
    nodes_per_source: int,
    seed: int,
    task: str,
    compute_threads: int,
) -> SynapseConfig:
    base = dict(nodes_per_source=nodes_per_source, seed=seed, task=task, compute_threads=compute_threads)
    if arm_id == 'baseline':
        return SynapseConfig(**base)
    if arm_id == 'experience_planning':
        return SynapseConfig(**base, experience_planning=True, surprise_memory=True, experience_horizon=3)
    if arm_id == 'dsag':
        from .dsag.config import DSAGSpecV1
        return SynapseConfig(**base, dsag=DSAGSpecV1(enabled=True, seed=seed))
    if arm_id == 'neuroweave':
        from .neuroweave.spec import NeuroWeaveSpecV1
        return SynapseConfig(**base, neuroweave=NeuroWeaveSpecV1(enabled=True, seed=seed))
    if arm_id == 'frontier_observers':
        from .frontier.spec import FrontierSpecV1, OBSERVER_MODULE_IDS
        return SynapseConfig(**base, frontier=FrontierSpecV1(enabled=True, seed=seed, modules=OBSERVER_MODULE_IDS))
    if arm_id == 'dendritic_oscillatory':
        return SynapseConfig(**base, dendritic_ltc=True, oscillatory_coherence=True)
    if arm_id == 'neural_graft':
        return SynapseConfig(**base, neural_graft=True, active_training=True)
    if arm_id == 'morphogenesis':
        from sonapse_morphogenesis.core.bridge import MorphogenesisSpecV1
        return SynapseConfig(**base, morphogenesis=MorphogenesisSpecV1.max_preset(seed=seed, neurons=16))
    raise ValueError(f'unknown pareto arm: {arm_id!r}; supported: {PARETO_ARM_IDS}')


def _profile_single_runtime(config: SynapseConfig, *, steps: int, learn: bool) -> dict[str, Any]:
    from .runtime import LabRuntime

    t_init0 = time.perf_counter()
    rt = LabRuntime(config)
    init_ms = (time.perf_counter() - t_init0) * 1000.0

    # Warm 1 step outside timing only if steps > 2? Keep exact deterministic count:
    # run the exact requested `steps` so trajectory digest matches `LabRuntime.step(steps)`.
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    rt.step(steps=steps, learn=learn, include_snapshot=False)
    step_wall_ms = (time.perf_counter() - wall0) * 1000.0
    step_cpu_ms = (time.process_time() - cpu0) * 1000.0

    snap0 = time.perf_counter()
    rt.snapshot()
    snapshot_ms = (time.perf_counter() - snap0) * 1000.0

    perf = rt.perf_telemetry()
    rewards = [float(row['reward']) for row in rt.recent[-steps:]]
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    traj_digest = state_digest({
        'steps': int(rt.total_steps),
        'last_action': rt.last_action,
        'obs': rt.obs,
        'rewards': rewards,
    })
    neurons = int(rt.agent.brain.n)
    synapses = int(len(rt.agent.graph.base_pre) + len(rt.agent.graph.graft_pre))
    sps = (steps * 1000.0 / step_wall_ms) if step_wall_ms > 1e-9 else 0.0
    return {
        'neurons': neurons,
        'synapses': synapses,
        'steps': int(steps),
        'learn': bool(learn),
        'active_subsystems': active_subsystem_list(config),
        'active_subsystem_count': len(active_subsystem_list(config)),
        'init_ms': round(init_ms, 3),
        'step_wall_ms': round(step_wall_ms, 3),
        'step_cpu_ms': round(step_cpu_ms, 3),
        'mean_step_ms': perf['total_ms']['mean'],
        'p50_step_ms': perf['total_ms']['p50'],
        'p95_step_ms': perf['total_ms']['p95'],
        'snapshot_ms': round(snapshot_ms, 3),
        'measured_steps_per_sec': round(sps, 2),
        'measured_neuron_steps_per_sec': round(sps * neurons, 1),
        'stage_mean_ms': perf['stage_mean_ms'],
        'bottleneck_stage': perf['bottleneck_stage'],
        'mean_reward': round(mean_reward, 6),
        'trajectory_digest': traj_digest,
    }


def run_scaling_profile(
    *,
    neuron_counts: Sequence[int] = (48, 192, 999),
    steps: int = 12,
    learn: bool = True,
    seed: int = 7,
    task: str = 'forage',
    compute_threads: int = 1,
) -> dict[str, Any]:
    """Measure real wall-clock, CPU, snapshot, and throughput scaling across neuron counts."""
    counts = [int(n) for n in neuron_counts]
    if not 1 <= len(counts) <= 8 or any(not 48 <= n <= 12000 for n in counts):
        raise ValueError('neuron_counts must contain 1..8 integers in [48, 12000]')
    if type(steps) is not int or not 1 <= steps <= 256:
        raise ValueError('steps must be an int in [1, 256]')
    if type(learn) is not bool:
        raise ValueError('learn must be boolean')
    if type(compute_threads) is not int or not 1 <= compute_threads <= 32:
        raise ValueError('compute_threads must be an int in [1, 32]')

    rows: list[dict[str, Any]] = []
    for n in counts:
        cfg = SynapseConfig(total_neurons=n, seed=int(seed), task=task, compute_threads=compute_threads)
        rows.append(_profile_single_runtime(cfg, steps=steps, learn=learn))

    scaling_efficiency: dict[str, Any] = {}
    if len(rows) >= 2 and rows[0]['mean_step_ms'] and rows[-1]['mean_step_ms']:
        n_ratio = rows[-1]['neurons'] / max(1, rows[0]['neurons'])
        t_ratio = rows[-1]['mean_step_ms'] / max(1e-9, rows[0]['mean_step_ms'])
        scaling_efficiency = {
            'neuron_scale_factor': round(n_ratio, 3),
            'latency_scale_factor': round(t_ratio, 3),
            'sublinear_ratio': round(t_ratio / max(1e-9, n_ratio), 4),
        }

    return seal_perf_receipt({
        'kind': 'scaling_profile',
        'config': {
            'neuron_counts': counts,
            'steps': steps,
            'learn': learn,
            'seed': int(seed),
            'task': task,
            'compute_threads': compute_threads,
        },
        'profiles': rows,
        'scaling_efficiency': scaling_efficiency,
        'non_claim': 'Synthetic short-horizon CPU latency and throughput profile; does not measure held-out task capability.',
    })


def run_subsystem_pareto_sweep(
    *,
    nodes_per_source: int = 16,
    steps: int = 12,
    learn: bool = True,
    seed: int = 7,
    task: str = 'forage',
    compute_threads: int = 1,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare isolated subsystem configurations for honest latency overhead & Pareto efficiency."""
    if type(nodes_per_source) is not int or not 16 <= nodes_per_source <= 2000:
        raise ValueError('nodes_per_source must be an int in [16, 2000]')
    if type(steps) is not int or not 1 <= steps <= 256:
        raise ValueError('steps must be an int in [1, 256]')
    if type(learn) is not bool:
        raise ValueError('learn must be boolean')
    selected = tuple(PARETO_ARM_IDS) if arms is None else tuple(arms)
    if not selected or len(selected) > len(PARETO_ARM_IDS):
        raise ValueError('arms must be a non-empty subset of PARETO_ARM_IDS')
    if 'baseline' not in selected:
        selected = ('baseline',) + selected

    results: list[dict[str, Any]] = []
    for arm_id in selected:
        cfg = _build_pareto_arm_config(
            arm_id,
            nodes_per_source=nodes_per_source,
            seed=int(seed),
            task=task,
            compute_threads=compute_threads,
        )
        row = _profile_single_runtime(cfg, steps=steps, learn=learn)
        row['arm'] = arm_id
        results.append(row)

    base_ms = max(1e-9, float(results[0]['mean_step_ms'] or 1e-9))
    for row in results:
        ms = float(row['mean_step_ms'] or 0.0)
        row['overhead_ms_vs_baseline'] = round(ms - base_ms, 4)
        row['overhead_ratio_vs_baseline'] = round(ms / base_ms, 4)

    # Non-dominated frontier in (minimize mean_step_ms, maximize active_subsystem_count)
    for i, a in enumerate(results):
        a_ms = float(a['mean_step_ms'] or 0.0)
        a_cap = int(a['active_subsystem_count'])
        dominated = False
        for j, b in enumerate(results):
            if i == j:
                continue
            b_ms = float(b['mean_step_ms'] or 0.0)
            b_cap = int(b['active_subsystem_count'])
            if b_ms <= a_ms and b_cap >= a_cap and (b_ms < a_ms or b_cap > a_cap):
                dominated = True
                break
        a['pareto_efficient'] = not dominated

    return seal_perf_receipt({
        'kind': 'subsystem_pareto_sweep',
        'config': {
            'nodes_per_source': nodes_per_source,
            'steps': steps,
            'learn': learn,
            'seed': int(seed),
            'task': task,
            'compute_threads': compute_threads,
            'arms': list(selected),
        },
        'arms': results,
        'non_claim': 'Pareto frontier reflects measured short-horizon CPU latency vs. enabled subsystem count, not empirical task superiority.',
    })
