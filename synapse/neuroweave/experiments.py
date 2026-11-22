"""NEUROWEAVE experimental laboratory: matched, ablated, seed-replicated experiments.

Every experiment in :data:`spec.EXPERIMENTS` runs through :func:`run_experiment`
and returns a JSON-serialisable receipt with

* a **baseline** arm (no mechanisms, except documented ``support_mechanisms``
  that every arm shares so the comparison is meaningful), a **candidate** arm
  (the experiment's mechanisms) and one **ablation** arm per mechanism
  (candidate minus that mechanism);
* prequential (predict-then-learn) accuracy / log-loss / Brier score on the
  training stream, and a **frozen** evaluation (``learn=False`` on a clone) on
  held-out seed streams that were never trained on;
* prediction metrics, estimated compute (ops / bytes, deterministic) kept
  separate from measured wall-clock (``*wallclock*`` keys), and parameter /
  state counts for budget matching;
* paired per-seed differences with mean, sd and a two-sided 95 % t interval;
* a verdict in ``{'improved', 'no_detectable_difference', 'worse'}`` decided
  only by whether the held-out CI excludes zero (never by complexity);
* a SHA-256 digest over the receipt with wall-clock fields removed, so the
  same arguments give the same digest, and a full config echo.

Kinds: ``temporal_task`` (seeded tasks from :mod:`tasks`), ``micro``
(Micro-Collective vs persistence vs a parameter-matched dense predictor),
``ecology`` (engine as controller, or as a Sonapse residual), ``topology``
(all 12 topologies at matched neuron and edge budgets), ``lesion`` (delegates
to :mod:`lesion`) and ``graft`` (specialist graft coupled vs uncoupled, with
exact-removal checks).

Everything is a synthetic computational benchmark
(``evidence_tier='synthetic_abstraction'``); nothing here is evidence of
biological fidelity.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .engine import NeuroWeaveEngine
from .micro_collective import MicroCollective, SpecialistGraft
from .spec import (EVIDENCE_TIER, EXPERIMENT_IDS, EXPERIMENTS, MECHANISM_IDS, SENSORY_DIM, TOPOLOGIES,
                   NeuroWeaveSpecV1)
from .tasks import TASKS, TaskStream, chance_accuracy, make_stream
from .topology import Topology, build_topology, topology_metrics

__all__ = [
    'LAB_SCHEMA', 'VERDICTS', 'EXPERIMENT_TASK', 'SUPPORT_MECHANISMS', 'ReadoutHead', 'DenseMLP',
    'EXPERIMENT_SUPPORT', 'clone_engine', 'make_spec', 'run_task_phase', 'paired_stats', 'verdict_from_stats', 'receipt_digest',
    'verify_receipt', 'run_experiment', 'run_all_experiments', 'results_to_csv', 'engine_topology',
    'QUICK_ARGS', 'SUPPLEMENTARY_EXPERIMENTS', 'SUPPLEMENTARY_IDS', 'supplementary_catalog',
    'run_supplementary_experiment',
]

LAB_SCHEMA = 'NeuroWeaveLabReceiptV1'
VERDICTS = ('improved', 'no_detectable_difference', 'worse')
_EXP = {e.id: e for e in EXPERIMENTS}
_LOG_EPS = 1e-6
_HEAD_OFFSET = 9_173
_MICRO_DATA_OFFSET = 9_311
_TRIM_OFFSET = 9_419
_ECO_NOISE_OFFSET = 9_521

# Default task per temporal experiment (overridable with ``task=``).
EXPERIMENT_TASK: dict[str, str] = {
    'multimodal_integration': 'xor', 'affective_neuromodulation': 'context_switch', 'meta_plasticity': 'xor',
    'predictive_replay': 'rhythm', 'context_attention': 'context_switch', 'flyhash_projection': 'parity',
    'stdp': 'delayed_recall', 'adaptive_conduction': 'delayed_recall', 'quantised_synapses': 'xor',
    'hierarchical_temporal_memory': 'delayed_recall_6', 'uncertainty_gating': 'rhythm',
    'predictive_coding': 'rhythm', 'structural_plasticity': 'xor', 'associative_memory': 'delayed_recall',
    'multi_soma_integration': 'parity', 'multi_region_routing': 'context_switch', 'cross_modal_learning': 'xor',
    'neural_grafting': 'xor', 'lesion_robustness': 'xor', 'topology_optimisation': 'delayed_recall',
    'micro_collective_learning': 'xor', 'expert_routing': 'xor', 'specialist_evolution': 'xor',
    'action_self_model': 'xor',
}
# Mechanisms shared by *every* arm of a kind (never ablated) so the comparison is meaningful.
SUPPORT_MECHANISMS: dict[str, tuple[str, ...]] = {
    'ecology': ('multimodal_sensory',),               # otherwise the controller receives no input at all
    'graft': ('multimodal_sensory', 'micro_collective'),   # the graft only acts through the Micro-Collective
}
# Per-experiment support (also never ablated): the consumer each named mechanism needs to act at
# all in the engine, so its ablation can measure something.
EXPERIMENT_SUPPORT: dict[str, tuple[str, ...]] = {
    'predictive_replay': ('memory_ecology', 'neuromodulator_field'),   # sleep needs memory; forecasts reach the
                                                                       # brain through the neuromodulator appraisal
    'predictive_coding': ('neuromodulator_field',),                    # hierarchical surprise -> appraisal
    'action_self_model': ('reward_plasticity', 'neuromodulator_field'),  # efference scales a consumed reward;
                                                                         # counterfactual forecasts -> appraisal
}
QUICK_ARGS: dict[str, Any] = {'seeds': (0, 1), 'heldout_seeds': (100,), 'steps': 24, 'neurons': 24,
                              'heldout_steps': 12}

_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
         2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
         2.045, 2.042)


# ====================================================================== JSON / digest helpers
def _jsonable(obj: Any) -> Any:
    # Common telemetry leaves already have their final JSON representation.
    if type(obj) in (int, bool, str) or obj is None:
        return obj
    if type(obj) is float:
        return obj if math.isfinite(obj) else None
    if type(obj) is np.ndarray and obj.ndim and (
        obj.dtype.kind in 'biu' or
        (obj.dtype.type in (np.float16, np.float32, np.float64) and np.isfinite(obj).all())
    ):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f'not JSON-serialisable: {type(obj).__name__}')


def _strip_wallclock(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _strip_wallclock(v) for k, v in obj.items() if 'wallclock' not in str(k) and k != 'digest'}
    if isinstance(obj, list):
        return [_strip_wallclock(v) for v in obj]
    return obj


def receipt_digest(receipt: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the receipt without ``digest`` and any ``*wallclock*`` key."""
    payload = _strip_wallclock(_jsonable(receipt))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def verify_receipt(receipt: Mapping[str, Any]) -> bool:
    """True when the receipt's stored digest matches its content (tamper check)."""
    return isinstance(receipt, Mapping) and receipt.get('digest') == receipt_digest(receipt)


def _finalise(receipt: dict[str, Any]) -> dict[str, Any]:
    out = _jsonable(receipt)
    out['digest'] = receipt_digest(out)
    return out


# ====================================================================== statistics
def _t975(df: int) -> float:
    return _T975[df - 1] if 1 <= df <= len(_T975) else 1.96


def paired_stats(candidate: Sequence[float], reference: Sequence[float]) -> dict[str, Any]:
    """Paired differences ``candidate - reference`` with mean, sd and a two-sided 95 % t interval."""
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError('paired samples must have equal length')
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = int(d.size)
    mean = float(d.mean()) if n else None
    sd = float(d.std(ddof=1)) if n >= 2 else None
    if n >= 2:
        half = _t975(n - 1) * sd / math.sqrt(n)
        ci = [mean - half, mean + half]
    else:
        ci = None
    return {'n': n, 'differences': d.tolist(), 'mean': mean, 'sd': sd, 'ci95': ci, 'method': 'paired_t_95'}


def verdict_from_stats(stats: Mapping[str, Any], *, higher_is_better: bool = True) -> dict[str, Any]:
    """Verdict from a paired CI only. 'improved' requires the whole 95 % CI on the better side of 0."""
    ci = stats.get('ci95')
    if ci is None or stats.get('n', 0) < 2:
        return {'verdict': 'no_detectable_difference', 'reason': 'insufficient_seeds_for_ci'}
    sd = stats.get('sd')
    if sd is None or not sd > 0.0:
        # identical paired differences collapse the interval to a point: no variance estimate
        return {'verdict': 'no_detectable_difference', 'reason': 'degenerate_ci'}
    lo, hi = (ci[0], ci[1]) if higher_is_better else (-ci[1], -ci[0])
    if lo > 0.0:
        return {'verdict': 'improved', 'reason': 'ci95_excludes_zero'}
    if hi < 0.0:
        return {'verdict': 'worse', 'reason': 'ci95_excludes_zero'}
    return {'verdict': 'no_detectable_difference', 'reason': 'ci95_contains_zero'}


def _bonferroni_verdict(stats: Mapping[str, Any], family: int, *, higher_is_better: bool = True) -> dict[str, Any]:
    """Verdict from a Bonferroni-adjusted paired t-interval (family-wise 95 %) over ``family`` comparisons."""
    base = verdict_from_stats(stats, higher_is_better=higher_is_better)
    if base['reason'] in ('insufficient_seeds_for_ci', 'degenerate_ci'):
        return base
    from scipy.stats import t as _t
    n, mean, sd = int(stats['n']), float(stats['mean']), float(stats['sd'])
    half = float(_t.ppf(1.0 - 0.05 / (2.0 * max(1, family)), n - 1)) * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    out = {'ci95_family_adjusted': [lo, hi], 'family_size': int(family)}
    if not higher_is_better:
        lo, hi = -hi, -lo
    if lo > 0.0:
        return {'verdict': 'improved', 'reason': 'adjusted_ci_excludes_zero', **out}
    if hi < 0.0:
        return {'verdict': 'worse', 'reason': 'adjusted_ci_excludes_zero', **out}
    return {'verdict': 'no_detectable_difference', 'reason': 'adjusted_ci_contains_zero', **out}


def _mean(xs: Iterable[Any]) -> float | None:
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(v)) if v else None


# ====================================================================== readouts / predictors
class ReadoutHead:
    """Experiment-owned logistic readout over brain features (shared by every arm, so it is budget-neutral).

    Features are the brain's 28 readout features plus the first ``n0`` unit outputs
    (``n0`` = initial population; neurons born later are ignored so the head shape is fixed)."""

    def __init__(self, n0: int, seed: int, lr: float = 0.05) -> None:
        self.n0 = int(n0)
        self.dim = 28 + self.n0
        rng = np.random.default_rng(int(seed) + _HEAD_OFFSET)
        self.w = (rng.random(self.dim) - 0.5) * 0.02
        self.b = 0.0
        self.lr = float(lr)
        self.updates = 0

    def features(self, engine: NeuroWeaveEngine, out: np.ndarray) -> np.ndarray:
        rf = engine.brain.readout_features()
        o = np.zeros(self.n0)
        m = min(self.n0, out.size)
        o[:m] = out[:m]
        x = np.concatenate([rf[:28], np.zeros(max(0, 28 - rf.size)), o])
        return np.clip(x, -10.0, 10.0)

    def predict(self, x: np.ndarray) -> float:
        z = float(np.clip(self.b + self.w @ x, -40.0, 40.0))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, x: np.ndarray, y: float) -> None:
        e = float(y) - self.predict(x)
        self.w = np.clip(self.w + self.lr * e * x, -8.0, 8.0)
        self.b = float(np.clip(self.b + self.lr * e, -8.0, 8.0))
        self.updates += 1

    @property
    def parameter_count(self) -> int:
        return self.dim + 1

    def copy(self) -> 'ReadoutHead':
        return copy.deepcopy(self)


class DenseMLP:
    """Dense tanh MLP next-vector predictor (budget-matched comparator for the Micro-Collective).

    Uses the same protocol as the Micro-Collective: every 5th sample is held out of training,
    online SGD with a uniform replay batch of 8 from a 256-item FIFO, global gradient clip 1."""

    def __init__(self, dim: int, hidden: int, seed: int, lr: float = 0.1, replay_batch: int = 8,
                 replay_capacity: int = 256, validation_every: int = 5) -> None:
        rng = np.random.default_rng(int(seed) + _MICRO_DATA_OFFSET + 1)
        self.dim, self.hidden = int(dim), int(hidden)
        self.w1 = rng.normal(0.0, 1.0 / math.sqrt(dim), (hidden, dim))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, 0.5 / math.sqrt(hidden), (dim, hidden))
        self.b2 = np.zeros(dim)
        self.lr, self.replay_batch, self.replay_capacity = float(lr), int(replay_batch), int(replay_capacity)
        self.validation_every = int(validation_every)
        self.replay: list[tuple[np.ndarray, np.ndarray]] = []
        self._rng = np.random.default_rng(int(seed) + _MICRO_DATA_OFFSET + 2)
        self.samples = 0
        self.sgd_steps = 0

    @property
    def parameter_count(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.w2 @ np.tanh(self.w1 @ x + self.b1) + self.b2, -10.0, 10.0)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> None:
        self.samples += 1
        if self.samples % self.validation_every == 0:
            return
        self.replay.append((x.copy(), y.copy()))
        if len(self.replay) > self.replay_capacity:
            self.replay.pop(0)
        idx = []
        if len(self.replay) > 1 and self.replay_batch:
            idx = self._rng.integers(0, len(self.replay) - 1, min(self.replay_batch, len(self.replay) - 1)).tolist()
        X = np.stack([x] + [self.replay[i][0] for i in idx])
        T = np.stack([y] + [self.replay[i][1] for i in idx])
        H = np.tanh(X @ self.w1.T + self.b1)
        Y = H @ self.w2.T + self.b2
        E = (Y - T) / X.shape[0]
        dw2 = E.T @ H
        db2 = E.sum(0)
        dH = (E @ self.w2) * (1.0 - H * H)
        dw1 = dH.T @ X
        db1 = dH.sum(0)
        grads = [dw1, db1, dw2, db2]
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = 1.0 / norm if norm > 1.0 else 1.0
        for w, g in zip((self.w1, self.b1, self.w2, self.b2), grads):
            w -= self.lr * scale * g
            np.clip(w, -5.0, 5.0, out=w)
        self.sgd_steps += 1


# ====================================================================== engine helpers
def make_spec(mechanisms: Iterable[str], *, seed: int, neurons: int, topology: str, steps: int) -> NeuroWeaveSpecV1:
    """Lab spec: the structural / sleep schedules scale with the run length so they actually fire."""
    return NeuroWeaveSpecV1(seed=int(seed), neurons=int(neurons), topology=topology, mechanisms=tuple(mechanisms),
                            max_neurons=max(int(neurons), min(4096, 2 * int(neurons))),
                            structural_interval=max(5, int(steps) // 6), sleep_interval=max(10, int(steps) // 3))


def clone_engine(engine: NeuroWeaveEngine) -> NeuroWeaveEngine:
    """Independent deep copy through the validated export/import path."""
    other = NeuroWeaveEngine(engine.spec)
    other.import_state(engine.export_state())
    return other


def _state_digest(engine: NeuroWeaveEngine) -> str:
    return hashlib.sha256(json.dumps(_jsonable(engine.export_state()), sort_keys=True).encode('utf-8')).hexdigest()


def _count_numbers(obj: Any) -> int:
    t = type(obj)
    if t is float or t is int:
        return 1
    if t is list or t is tuple:
        total = 0
        for v in obj:
            tv = type(v)
            if tv is float or tv is int:
                total += 1
            elif tv is list or tv is tuple or tv is dict:
                total += _count_numbers(v)
            elif tv is not str and tv is not bool and v is not None:
                total += _count_numbers(v)
        return total
    if t is dict:
        total = 0
        for v in obj.values():
            tv = type(v)
            if tv is float or tv is int:
                total += 1
            elif tv is list or tv is tuple or tv is dict:
                total += _count_numbers(v)
            elif tv is not str and tv is not bool and v is not None:
                total += _count_numbers(v)
        return total
    if t is str or t is bool or obj is None:
        return 0
    if isinstance(obj, Mapping):
        return sum(_count_numbers(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_numbers(v) for v in obj)
    return 1 if isinstance(obj, (int, float)) and not isinstance(obj, bool) else 0


def _budget(engine: NeuroWeaveEngine, head: ReadoutHead | None) -> dict[str, int]:
    f = engine.flags
    syn = engine.brain.synapses
    micro = engine.micro.parameter_count if 'micro_collective' in f else 0
    graft = 0
    if engine.graft is not None:
        graft = int(sum(w.size for w in engine.graft._weights()))
    trainable = int(syn.size) + (head.parameter_count if head is not None else 0) + int(micro) + graft
    return {'neurons': int(engine.n), 'synapses': int(syn.size), 'trainable_parameters': trainable,
            'micro_parameters': int(micro), 'graft_parameters': graft,
            'state_numbers': _count_numbers(engine.export_state())}


def _estimated_cost(engine: NeuroWeaveEngine, head_steps: int, head_dim: int) -> dict[str, int]:
    b = engine.brain
    micro_ops = 6 * engine.micro.parameter_count * 9 * engine.micro.sgd_steps if 'micro_collective' in engine.flags else 0
    return {'brain_ops': int(b.estimated['ops_total']), 'micro_ops': int(micro_ops),
            'readout_ops': int(4 * head_dim * head_steps),
            'total_ops': int(b.estimated['ops_total'] + micro_ops + 4 * head_dim * head_steps),
            'state_bytes': int(b.estimated['bytes'])}


def engine_topology(engine: NeuroWeaveEngine) -> Topology:
    """Current connectome of an engine as a :class:`Topology` (for :func:`topology_metrics`)."""
    syn, c = engine.brain.synapses, engine.brain.cells
    n = int(engine.n)
    return Topology(n=n, src=syn.src.astype(np.int64).copy(), dst=syn.dst.astype(np.int64).copy(),
                    weight=np.asarray(syn.w, dtype=np.float64).copy(), delay=np.asarray(syn.delay, dtype=np.int64).copy(),
                    sign=np.asarray(c.sign[:n]).astype(np.int8), region=np.asarray(c.region[:n], dtype=np.int64),
                    name=engine.spec.topology, seed=int(engine.spec.seed))


def run_task_phase(engine: NeuroWeaveEngine, heads: Mapping[str, ReadoutHead], stream: TaskStream, *,
                   learn: bool, start: int = 0, stop: int | None = None, silence: np.ndarray | None = None,
                   use_sensory: bool = True) -> dict[str, Any]:
    """Prequential run over ``stream[start:stop]``.

    For each step: engine step -> (silenced units clamped to 0) -> every head predicts, is
    scored, and (if ``learn``) is updated. The first head's task supplies the
    NEUROWEAVE-style signed reward (``+/-min(1, |y - p| + 0.2)``) given to the engine on the
    *next* step. With ``learn=False`` engine plasticity and head updates are disabled."""
    stop = stream.steps if stop is None else stop
    tasks = list(heads)
    correct = {t: [] for t in tasks}
    losses = {t: [] for t in tasks}
    briers = {t: [] for t in tasks}
    pred_err, micro_err, surprise = [], [], []
    reward = 0.0
    sil = None if silence is None or len(silence) == 0 else np.asarray(silence, dtype=np.int64)
    for i in range(start, stop):
        res = engine.step(stream.inputs[i], stream.sensory[i] if use_sensory else None,
                          reward=reward if learn else 0.0, learn=learn)
        out = res['output']
        if sil is not None:
            idx = sil[sil < engine.brain.cells.output.size]
            engine.brain.cells.output[idx] = 0.0
            out = out.copy()
            out[sil[sil < out.size]] = 0.0
        x = None
        for k, t in enumerate(tasks):
            head = heads[t]
            if x is None or head.n0 != heads[tasks[0]].n0:
                x = head.features(engine, out)
            p = head.predict(x)
            y = int(stream.target(t)[i])
            correct[t].append(1.0 if (p >= 0.5) == (y == 1) else 0.0)
            pc = min(max(p, _LOG_EPS), 1.0 - _LOG_EPS)
            losses[t].append(-(y * math.log(pc) + (1 - y) * math.log(1.0 - pc)))
            briers[t].append((p - y) ** 2)
            if learn:
                head.update(x, y)
                if k == 0:
                    mag = min(1.0, abs(y - p) + 0.2)
                    reward = mag if correct[t][-1] else -mag
        if res['prediction']['active']:
            pred_err.append(float(res['prediction']['error']))
            surprise.append(float(res['prediction']['surprise']))
        if res['micro']['active']:
            micro_err.append(float(res['micro'].get('error', 0.0)))
    out_metrics: dict[str, Any] = {'steps': stop - start, 'tasks': {}}
    for t in tasks:
        tg = stream.target(t)[start:stop]
        out_metrics['tasks'][t] = {'accuracy': _mean(correct[t]), 'log_loss': _mean(losses[t]),
                                   'brier': _mean(briers[t]), 'chance_accuracy': chance_accuracy(tg),
                                   'correct': correct[t]}
    out_metrics['prediction'] = {'hierarchy_error': _mean(pred_err), 'surprise': _mean(surprise),
                                 'micro_forecast_error': _mean(micro_err)}
    return out_metrics


# ====================================================================== temporal-task arms
def _task_arm(mechs: Sequence[str], seed: int, *, task: str, steps: int, heldout_seeds: Sequence[int],
              heldout_steps: int, neurons: int, topology: str, coupling: float | None = None,
              detach_check: bool = False) -> dict[str, Any]:
    spec = make_spec(mechs, seed=seed, neurons=neurons, topology=topology, steps=steps)
    engine = NeuroWeaveEngine(spec)
    if coupling is not None and engine.graft is not None:
        engine.set_specialist_coupling(coupling)
    head = ReadoutHead(engine.n, seed)
    stream = make_stream(seed, steps)
    t0 = time.perf_counter()
    train = run_task_phase(engine, {task: head}, stream, learn=True)
    train_wall = time.perf_counter() - t0
    held = []
    t1 = time.perf_counter()
    for hs in heldout_seeds:
        e2 = clone_engine(engine)
        h2 = head.copy()
        r = run_task_phase(e2, {task: h2}, make_stream(hs, heldout_steps), learn=False)
        held.append(r)
    held_wall = time.perf_counter() - t1
    tm = train['tasks'][task]
    arm = {
        'seed': seed, 'mechanisms': list(spec.mechanisms), 'train_stream_digest': stream.digest,
        'train': {k: tm[k] for k in ('accuracy', 'log_loss', 'brier', 'chance_accuracy')},
        'train_prediction': train['prediction'],
        'heldout': {'accuracy': _mean(h['tasks'][task]['accuracy'] for h in held),
                    'log_loss': _mean(h['tasks'][task]['log_loss'] for h in held),
                    'brier': _mean(h['tasks'][task]['brier'] for h in held),
                    'chance_accuracy': _mean(h['tasks'][task]['chance_accuracy'] for h in held),
                    'micro_forecast_error': _mean(h['prediction']['micro_forecast_error'] for h in held),
                    'hierarchy_error': _mean(h['prediction']['hierarchy_error'] for h in held)},
        'estimated_cost': _estimated_cost(engine, steps, head.dim),
        'budget': _budget(engine, head),
        'measured_wallclock_s': {'train': train_wall, 'heldout': held_wall},
    }
    if detach_check:
        arm['removal'] = _graft_removal_checks(engine, stream, seed, steps)
    return arm


def _graft_removal_checks(engine: NeuroWeaveEngine, stream: TaskStream, seed: int, steps: int) -> dict[str, Any]:
    """(1) detaching == setting coupling 0 from then on (no residue); (2) an uncoupled graft is
    output-identical to no graft at all (zero-initialised contribution)."""
    k = max(4, min(16, steps // 2))
    cont = make_stream(seed + 50_000, k)
    a, b = clone_engine(engine), clone_engine(engine)
    a.detach_specialist()
    b.set_specialist_coupling(0.0)
    diff1 = 0.0
    for i in range(k):
        oa = a.step(cont.inputs[i], cont.sensory[i], learn=True)['output']
        ob = b.step(cont.inputs[i], cont.sensory[i], learn=True)['output']
        diff1 = max(diff1, float(np.max(np.abs(oa - ob))))
    mechs = tuple(m for m in engine.spec.mechanisms)
    no_graft = tuple(m for m in mechs if m != 'specialist_graft')
    spec_g = make_spec(mechs, seed=seed, neurons=engine.spec.neurons, topology=engine.spec.topology, steps=steps)
    spec_n = make_spec(no_graft, seed=seed, neurons=engine.spec.neurons, topology=engine.spec.topology, steps=steps)
    g, n = NeuroWeaveEngine(spec_g), NeuroWeaveEngine(spec_n)
    diff2 = 0.0
    for i in range(k):
        og = g.step(stream.inputs[i], stream.sensory[i], learn=True)['output']
        on = n.step(stream.inputs[i], stream.sensory[i], learn=True)['output']
        diff2 = max(diff2, float(np.max(np.abs(og - on))))
    return {'steps': k, 'detach_vs_zero_coupling_max_abs_diff': diff1, 'detach_restores_uncoupled_outputs': diff1 == 0.0,
            'uncoupled_vs_no_graft_max_abs_diff': diff2, 'uncoupled_equals_no_graft': diff2 == 0.0}


# ====================================================================== micro arms
def _sensory_frames(seed: int, steps: int, config: str = 'default') -> np.ndarray:
    """Deterministic ecology sensory stream under smooth random actions (steps + 1 frames)."""
    from .ecology import NeuroWeaveEcology
    eco = NeuroWeaveEcology(config, seed=seed)
    rng = np.random.default_rng(seed + _MICRO_DATA_OFFSET)
    z = np.zeros(2)
    frames = [eco.sensory_vector(0)]
    for _ in range(steps):
        z = 0.85 * z + 0.15 * rng.normal(0.0, 1.0, 2)
        act = np.array([np.clip(0.5 + 0.5 * z[0], -1, 1), np.clip(z[1], -1, 1)])
        _, _, done, _ = eco.step(act[None, :])
        frames.append(eco.sensory_vector(0))
        if done:
            eco.reset()
    return np.clip(np.asarray(frames, dtype=np.float64), -1.0, 1.0)


def _visible(x: np.ndarray, multimodal: bool) -> np.ndarray:
    if multimodal:
        return x
    v = np.zeros_like(x)
    v[:5] = x[:5]              # visual channels only
    return v


class _Persistence:
    parameter_count = 0
    sgd_steps = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.copy()

    def train_step(self, x: np.ndarray, y: np.ndarray) -> None:
        return None


def _micro_model(mechs: Sequence[str], seed: int, neurons: int, topology: str, steps: int):
    spec = make_spec(mechs, seed=seed, neurons=neurons, topology=topology, steps=steps)
    if 'micro_collective' not in spec.mechanisms:
        return spec, _Persistence(), None
    micro = MicroCollective.from_spec(spec)
    graft = None
    if 'specialist_graft' in spec.mechanisms:
        graft = SpecialistGraft(SENSORY_DIM, SENSORY_DIM, seed=seed)
        micro.attach_specialist(graft)
    return spec, micro, graft


def _micro_predict(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, MicroCollective):
        return np.asarray(model.predict(x)[0])
    return model.predict(x)


def _micro_run(model: Any, graft: SpecialistGraft | None, frames: np.ndarray, *, learn: bool, multimodal: bool,
               coupling_after: int | None = None, coupling: float = 0.1) -> dict[str, float | None]:
    errs, pers = [], []
    for i in range(frames.shape[0] - 1):
        x, y = _visible(frames[i], multimodal), frames[i + 1]
        if graft is not None and coupling_after is not None and learn and i == coupling_after:
            graft.set_coupling(coupling)
        if graft is not None and not learn:
            graft.step(x)                 # advance the graft's recurrent state (inference only)
        p = _micro_predict(model, x)
        errs.append(float(np.mean((p - y) ** 2)))
        pers.append(float(np.mean((frames[i] - y) ** 2)))
        if learn:
            model.train_step(x, y)
    return {'mse': _mean(errs), 'persistence_mse': _mean(pers)}


def _micro_arm(mechs: Sequence[str], seed: int, *, steps: int, heldout_seeds: Sequence[int], heldout_steps: int,
               neurons: int, topology: str, dense: bool = False) -> dict[str, Any]:
    frames = _sensory_frames(seed, steps)
    multimodal = 'multimodal_sensory' in mechs or dense
    if dense:
        ref = MicroCollective.from_spec(make_spec(('micro_collective',), seed=seed, neurons=neurons,
                                                  topology=topology, steps=steps))
        hidden = max(1, int(round((ref.parameter_count - SENSORY_DIM) / (2 * SENSORY_DIM + 1))))
        model, graft, spec_mechs = DenseMLP(SENSORY_DIM, hidden, seed), None, ['dense_matched_mlp']
    else:
        spec, model, graft = _micro_model(mechs, seed, neurons, topology, steps)
        spec_mechs = list(spec.mechanisms)
    t0 = time.perf_counter()
    train = _micro_run(model, graft, frames, learn=True, multimodal=multimodal,
                       coupling_after=max(1, steps // 4) if graft is not None else None)
    wall = time.perf_counter() - t0
    held = []
    for hs in heldout_seeds:
        m2 = copy.deepcopy(model)
        g2 = m2._specialist if isinstance(m2, MicroCollective) else None
        held.append(_micro_run(m2, g2, _sensory_frames(hs, heldout_steps), learn=False, multimodal=multimodal))
    params = int(getattr(model, 'parameter_count', 0)) + (int(sum(w.size for w in graft._weights())) if graft else 0)
    return {'seed': seed, 'mechanisms': spec_mechs, 'predictor': type(model).__name__.lstrip('_'),
            'train': {'mse': train['mse'], 'persistence_mse': train['persistence_mse']},
            'heldout': {'mse': _mean(h['mse'] for h in held), 'persistence_mse': _mean(h['persistence_mse'] for h in held)},
            'estimated_cost': {'total_ops': int(6 * params * 9 * int(getattr(model, 'sgd_steps', 0)) + 2 * params * steps)},
            'budget': {'trainable_parameters': params},
            'measured_wallclock_s': {'train': wall}}


# ====================================================================== ecology arms
def _ecology_loop(engine: NeuroWeaveEngine, eco: Any, steps: int, *, learn: bool, rng: np.random.Generator) -> dict[str, Any]:
    reward, total = 0.0, 0.0
    intended = issued = observed = None
    prev = np.zeros(2)
    auth, rel = [], []
    episodes = 0
    for _ in range(steps):
        res = engine.step(None, eco.sensory_vector(0), reward=reward if learn else 0.0, learn=learn,
                          intended_action=intended, issued_action=issued, observed_outcome=observed)
        c = res['controls']
        if res['efference'] is not None:
            auth.append(float(res['efference']['authorship']))
            rel.append(float(res['efference']['control_reliability']))
        intended = engine.action_from_output(res['output'])
        keep = float(np.clip(0.3 * c['persistence'], 0.0, 0.9))
        noise = rng.normal(0.0, 0.1 * c['exploration'], 2) if learn else np.zeros(2)
        issued = np.clip(keep * prev + (1.0 - keep) * intended + noise, -1.0, 1.0)
        prev = issued
        obs, rewards, done, _ = eco.step(issued[None, :])
        observed = np.array([obs[0]['speed'], obs[0]['turn']])
        reward = float(rewards[0])
        total += reward
        if done:
            episodes += 1
            eco.reset()
            engine.end_episode()
            intended = issued = observed = None
            prev = np.zeros(2)
    s = eco.summary()
    return {'total_reward': total, 'food_eaten': float(s['food_eaten'][0]), 'collisions': float(s['collisions'][0]),
            'mean_authorship': _mean(auth), 'mean_control_reliability': _mean(rel), 'episodes': episodes}


def _ecology_arm(mechs: Sequence[str], seed: int, *, steps: int, heldout_seeds: Sequence[int], heldout_steps: int,
                 neurons: int, topology: str, config: str = 'default') -> dict[str, Any]:
    from .ecology import NeuroWeaveEcology
    spec = make_spec(mechs, seed=seed, neurons=neurons, topology=topology, steps=steps)
    engine = NeuroWeaveEngine(spec)
    t0 = time.perf_counter()
    train = _ecology_loop(engine, NeuroWeaveEcology(config, seed=seed), steps, learn=True,
                          rng=np.random.default_rng(seed + _ECO_NOISE_OFFSET))
    wall = time.perf_counter() - t0
    held = [_ecology_loop(clone_engine(engine), NeuroWeaveEcology(config, seed=hs), heldout_steps, learn=False,
                          rng=np.random.default_rng(hs + _ECO_NOISE_OFFSET)) for hs in heldout_seeds]
    keys = ('total_reward', 'food_eaten', 'collisions', 'mean_authorship', 'mean_control_reliability')
    return {'seed': seed, 'mechanisms': list(spec.mechanisms), 'controller': 'engine', 'train': train,
            'heldout': {k: _mean(h[k] for h in held) for k in keys},
            'estimated_cost': _estimated_cost(engine, 0, 0), 'budget': _budget(engine, None),
            'measured_wallclock_s': {'train': wall}}


def _sonapse_arm(mechs: Sequence[str] | None, seed: int, *, steps: int, heldout_seeds: Sequence[int],
                 heldout_steps: int, neurons: int, topology: str, config: str = 'default',
                 gain: float = 0.05) -> dict[str, Any]:
    """Sonapse IntegratedAgent with (``mechs`` not None) or without the NEUROWEAVE residual channel."""
    from synapse.agent import IntegratedAgent
    from synapse.config import SynapseConfig
    from .ecology import NeuroWeaveEcology
    nw = None
    if mechs is not None:
        nw = NeuroWeaveSpecV1(enabled=True, gain=gain, max_gain=max(gain, 0.1), ramp_steps=max(1, steps // 4),
                              seed=seed, neurons=neurons, topology=topology, mechanisms=tuple(mechs),
                              max_neurons=max(neurons, min(4096, 2 * neurons)))
    agent = IntegratedAgent(SynapseConfig(nodes_per_source=16, seed=seed, neuroweave=nw))
    t0 = time.perf_counter()
    train = NeuroWeaveEcology(config, seed=seed).run_synapse_agent(agent, steps, learn=True)
    wall = time.perf_counter() - t0
    channel = getattr(agent, 'neuroweave_channel', None)
    authorship = float(channel.engine.authorship) if channel is not None else None
    held = []
    for hs in heldout_seeds:
        try:
            a2 = copy.deepcopy(agent)
        except Exception:  # pragma: no cover - fall back to evaluating in place (recorded)
            a2 = agent
        held.append(NeuroWeaveEcology(config, seed=hs).run_synapse_agent(a2, heldout_steps, learn=False))
    keys = ('total_reward', 'food_eaten', 'collisions')
    return {'seed': seed, 'mechanisms': [] if mechs is None else list(nw.mechanisms),
            'controller': 'sonapse_residual' if mechs is not None else 'sonapse_baseline',
            'train': {**{k: float(train[k]) for k in keys}, 'mean_authorship': authorship},
            'heldout': {**{k: _mean(h[k] for h in held) for k in keys}, 'mean_authorship': None},
            'estimated_cost': {'total_ops': None}, 'budget': {'neuroweave_neurons': neurons if mechs is not None else 0},
            'measured_wallclock_s': {'train': wall}}


# ====================================================================== topology assay
def _trim_engine(engine: NeuroWeaveEngine, budget: int, seed: int, salt: int) -> int:
    syn = engine.brain.synapses
    extra = int(syn.size) - int(budget)
    if extra <= 0:
        return 0
    rng = np.random.default_rng(seed + _TRIM_OFFSET + salt)
    mask = np.zeros(syn.size, dtype=bool)
    mask[rng.choice(syn.size, size=extra, replace=False)] = True
    engine.brain.synapses = syn.lesion(mask)
    return extra


def _topology_run(cfg: Mapping[str, Any]) -> dict[str, Any]:
    seeds, heldout_seeds = cfg['seeds'], cfg['heldout_seeds']
    steps, heldout_steps, neurons, task = cfg['steps'], cfg['heldout_steps'], cfg['neurons'], cfg['task']
    mechs = cfg['candidate_mechanisms']
    per_topology: dict[str, list[dict[str, Any]]] = {t: [] for t in TOPOLOGIES}
    budgets = {}
    for seed in seeds:
        probe = make_spec(mechs, seed=seed, neurons=neurons, topology='clustered', steps=steps)
        budget = min(build_topology(t, neurons, seed=seed, fan_in=probe.fan_in,
                                    inhibitory_fraction=probe.inhibitory_fraction).edges for t in TOPOLOGIES)
        budgets[str(seed)] = budget
        for ti, topo in enumerate(TOPOLOGIES):
            spec = make_spec(mechs, seed=seed, neurons=neurons, topology=topo, steps=steps)
            engine = NeuroWeaveEngine(spec)
            removed = _trim_engine(engine, budget, seed, ti)
            metrics = topology_metrics(engine_topology(engine))
            head = ReadoutHead(engine.n, seed)
            stream = make_stream(seed, steps)
            t0 = time.perf_counter()
            train = run_task_phase(engine, {task: head}, stream, learn=True)
            wall = time.perf_counter() - t0
            held, lesioned = [], []
            for hs in heldout_seeds:
                hstream = make_stream(hs, heldout_steps)
                held.append(run_task_phase(clone_engine(engine), {task: head.copy()}, hstream, learn=False))
                le = clone_engine(engine)
                rng = np.random.default_rng(seed + _TRIM_OFFSET + 977 + ti)
                m = np.zeros(le.brain.synapses.size, dtype=bool)
                k = int(round(0.2 * m.size))
                m[rng.choice(m.size, size=k, replace=False)] = True
                le.brain.synapses = le.brain.synapses.lesion(m)
                lesioned.append(run_task_phase(le, {task: head.copy()}, hstream, learn=False))
            acc = _mean(h['tasks'][task]['accuracy'] for h in held)
            lacc = _mean(h['tasks'][task]['accuracy'] for h in lesioned)
            per_topology[topo].append({
                'seed': seed, 'edges_initial': int(metrics['edges'] + removed), 'edges_trimmed': int(removed),
                'metrics': {k: metrics[k] for k in ('n', 'edges', 'density', 'clustering_mean', 'modularity',
                                                    'mean_shortest_path', 'global_efficiency',
                                                    'reachable_pair_fraction', 'hub_top10_in_share',
                                                    'in_degree_gini', 'regional_communication_efficiency',
                                                    'inter_region_edge_fraction')},
                'train': {k: train['tasks'][task][k] for k in ('accuracy', 'log_loss')},
                'heldout': {'accuracy': acc, 'log_loss': _mean(h['tasks'][task]['log_loss'] for h in held)},
                'lesion_20pct_random': {'heldout_accuracy': lacc,
                                        'robustness': (lacc / acc) if (acc and lacc is not None) else None},
                'estimated_cost': _estimated_cost(engine, steps, head.dim), 'budget': _budget(engine, head),
                'measured_wallclock_s': {'train': wall},
            })
    reference = 'clustered'
    family = len(TOPOLOGIES) - 1     # every topology is compared with the reference; best-of selection
    ref_acc = [r['heldout']['accuracy'] for r in per_topology[reference]]
    ranking, comparisons = [], {}
    for topo in TOPOLOGIES:
        accs = [r['heldout']['accuracy'] for r in per_topology[topo]]
        summary = {'heldout_accuracy': _mean(accs),
                   'robustness': _mean(r['lesion_20pct_random']['robustness'] for r in per_topology[topo]),
                   'global_efficiency': _mean(r['metrics']['global_efficiency'] for r in per_topology[topo]),
                   'total_ops': _mean(r['estimated_cost']['total_ops'] for r in per_topology[topo])}
        ranking.append({'topology': topo, **summary})
        if topo != reference:
            st = paired_stats(accs, ref_acc)
            comparisons[f'{topo}_vs_{reference}'] = {'heldout_accuracy': st, **_bonferroni_verdict(st, family)}
    ranking.sort(key=lambda r: (-(r['heldout_accuracy'] if r['heldout_accuracy'] is not None else -1.0), r['topology']))
    best = ranking[0]['topology']
    overall = comparisons.get(f'{best}_vs_{reference}', {'verdict': 'no_detectable_difference',
                                                          'reason': 'reference_is_best'})
    return {'arms': {t: {'per_seed': per_topology[t]} for t in TOPOLOGIES}, 'edge_budget_per_seed': budgets,
            'ranking': ranking, 'comparisons': comparisons, 'best_topology': best, 'reference_topology': reference,
            'verdict': overall['verdict'], 'verdict_basis': {'comparison': f'{best}_vs_{reference}',
                                                             'metric': 'heldout_accuracy',
                                                             'multiple_comparisons': f'bonferroni_{family}',
                                                             'reason': overall.get('reason')}}


# ====================================================================== orchestration
def _check_seq(name: str, value: Any, *, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f'{name} must be a sequence of ints')
    out = tuple(value)
    if (not out and not allow_empty) or any(type(s) is not int or not 0 <= s < 2**31 - 60_000 for s in out) \
            or len(set(out)) != len(out):
        raise ValueError(f'{name} must be distinct non-negative ints')
    return out


def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


def _summarise(per_seed: list[dict[str, Any]], section: str) -> dict[str, Any]:
    keys = sorted({k for r in per_seed for k, v in r.get(section, {}).items() if not isinstance(v, (dict, list))})
    return {k: _mean(r[section].get(k) for r in per_seed) for k in keys}


def _metric(per_seed: list[dict[str, Any]], section: str, key: str) -> list[float]:
    return [float('nan') if r[section].get(key) is None else float(r[section][key]) for r in per_seed]


def run_experiment(experiment_id: str, *, seeds: Sequence[int] = (0, 1, 2), heldout_seeds: Sequence[int] = (100, 101),
                   steps: int = 300, neurons: int = 48, topology: str = 'clustered',
                   extra_mechanisms: Sequence[str] = (), ablations: bool = True, heldout_steps: int | None = None,
                   task: str | None = None, controller: str = 'engine', ecology_config: str = 'default',
                   lesion_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run one registered experiment and return its receipt (see module docstring)."""
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f'unknown experiment {experiment_id!r}')
    info = _EXP[experiment_id]
    seeds = _check_seq('seeds', seeds)
    heldout_seeds = _check_seq('heldout_seeds', heldout_seeds)
    if set(seeds) & set(heldout_seeds):
        raise ValueError('held-out seeds must not overlap training seeds')
    _check_int('steps', steps, 4, 100_000)
    _check_int('neurons', neurons, 8, 4096)
    heldout_steps = max(8, steps // 2) if heldout_steps is None else _check_int('heldout_steps', heldout_steps, 2, 100_000)
    if topology not in TOPOLOGIES:
        raise ValueError(f'topology must be one of {TOPOLOGIES}')
    extra = tuple(extra_mechanisms)
    bad = [m for m in extra if m not in MECHANISM_IDS]
    if bad:
        raise ValueError(f'unknown extra mechanisms: {bad}')
    if type(ablations) is not bool:
        raise ValueError('ablations must be a boolean')
    task = EXPERIMENT_TASK.get(experiment_id, 'xor') if task is None else task
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}')
    if controller not in ('engine', 'sonapse_residual'):
        raise ValueError("controller must be 'engine' or 'sonapse_residual'")
    kind = info.kind
    support = SUPPORT_MECHANISMS.get(kind, ()) + EXPERIMENT_SUPPORT.get(experiment_id, ())
    canon = lambda ms: tuple(m for m in MECHANISM_IDS if m in set(ms))  # noqa: E731
    exp_mechs = canon(info.mechanisms + extra)
    candidate = canon(exp_mechs + support)
    baseline = canon(support)
    config = {'experiment_id': experiment_id, 'kind': kind, 'seeds': list(seeds), 'heldout_seeds': list(heldout_seeds),
              'steps': steps, 'heldout_steps': heldout_steps, 'neurons': neurons, 'topology': topology,
              'extra_mechanisms': list(extra), 'ablations': ablations, 'task': task, 'controller': controller,
              'ecology_config': ecology_config, 'lesion_kwargs': dict(lesion_kwargs or {}),
              'candidate_mechanisms': list(candidate), 'baseline_mechanisms': list(baseline),
              'support_mechanisms': list(support)}
    receipt: dict[str, Any] = {
        'schema': LAB_SCHEMA, 'experiment_id': experiment_id, 'title': info.title, 'kind': kind, 'task': task,
        'mechanisms': list(exp_mechs), 'candidate_mechanisms': list(candidate), 'baseline_mechanisms': list(baseline),
        'support_mechanisms': list(support), 'config': config, 'evidence_tier': EVIDENCE_TIER,
        'verdict_policy': 'paired 95% t-interval on held-out metric must exclude 0; complexity is never rewarded',
    }
    started = time.perf_counter()
    if kind == 'topology':
        receipt.update(_topology_run({**config, 'seeds': seeds, 'heldout_seeds': heldout_seeds,
                                      'candidate_mechanisms': candidate}))
    elif kind == 'lesion':
        from .lesion import run_lesion_assay
        kw = {'seeds': seeds, 'neurons': neurons, 'topology': topology, 'steps': steps,
              'recovery_steps': max(4, steps // 2), 'heldout_steps': heldout_steps, 'task': task,
              'mechanisms': candidate, **dict(lesion_kwargs or {})}
        assay = run_lesion_assay(**kw)
        receipt['lesion_assay'] = assay
        receipt['comparisons'] = assay['comparisons']
        receipt['verdict'] = assay['verdict']
        receipt['verdict_basis'] = assay['verdict_basis']
    else:
        receipt.update(_arms_run(kind, info.id, exp_mechs, candidate, baseline, seeds, heldout_seeds, steps,
                                 heldout_steps, neurons, topology, task, ablations, controller, ecology_config))
    receipt['measured_wallclock_s_total'] = time.perf_counter() - started
    return _finalise(receipt)


def _arms_run(kind: str, exp_id: str, exp_mechs: tuple[str, ...], candidate: tuple[str, ...], baseline: tuple[str, ...],
              seeds: tuple[int, ...], heldout_seeds: tuple[int, ...], steps: int, heldout_steps: int, neurons: int,
              topology: str, task: str, ablations: bool, controller: str, eco_config: str) -> dict[str, Any]:
    common = dict(steps=steps, heldout_seeds=heldout_seeds, heldout_steps=heldout_steps, neurons=neurons,
                  topology=topology)
    arms_spec: dict[str, tuple[str, ...] | None] = {'baseline': baseline, 'candidate': candidate}
    if ablations and not (kind == 'ecology' and controller == 'sonapse_residual'):
        for m in exp_mechs:
            arms_spec[f'minus_{m}'] = tuple(x for x in candidate if x != m)
    notes: list[str] = []
    if kind == 'temporal_task':
        def run(name, mechs, seed):
            return _task_arm(mechs, seed, task=task, **common)
        primary, higher, guard, guard_higher = ('heldout', 'accuracy'), True, ('heldout', 'log_loss'), False
    elif kind == 'graft':
        def run(name, mechs, seed):
            coupled = name != 'baseline' and 'specialist_graft' in mechs
            return _task_arm(mechs, seed, task=task, coupling=0.2 if coupled else None,
                             detach_check=(name == 'candidate'), **common)
        arms_spec['baseline'] = candidate          # uncoupled graft (coupling stays at its zero init)
        notes.append('baseline = candidate mechanisms with the specialist graft uncoupled (coupling 0); '
                     'candidate couples it at 0.2')
        primary, higher, guard, guard_higher = ('heldout', 'micro_forecast_error'), False, ('heldout', 'accuracy'), True
    elif kind == 'micro':
        def run(name, mechs, seed):
            if name == 'baseline':
                return _micro_arm(candidate, seed, dense=True, **common)
            return _micro_arm(mechs, seed, **common)
        notes.append('baseline = parameter-matched dense tanh MLP with the same training protocol; '
                     'persistence (y_next = x) is reported in every arm; an arm without micro_collective '
                     'falls back to persistence; without multimodal_sensory only visual channels are observed')
        primary, higher, guard, guard_higher = ('heldout', 'mse'), False, None, True
    elif kind == 'ecology':
        if controller == 'sonapse_residual':
            def run(name, mechs, seed):
                return _sonapse_arm(None if name == 'baseline' else mechs, seed, config=eco_config, **common)
            notes.append('Sonapse IntegratedAgent with vs without the NEUROWEAVE residual graft-bus channel; '
                         'ablations are skipped for this controller (cost)')
        else:
            def run(name, mechs, seed):
                return _ecology_arm(mechs, seed, config=eco_config, **common)
        primary, higher, guard, guard_higher = ('heldout', 'total_reward'), True, None, True
    else:  # pragma: no cover - registry kinds are closed
        raise ValueError(f'unsupported kind {kind!r}')

    arms: dict[str, Any] = {}
    for name, mechs in arms_spec.items():
        per_seed = [run(name, mechs, s) for s in seeds]
        arms[name] = {'mechanisms': per_seed[0]['mechanisms'], 'per_seed': per_seed,
                      'summary': {'train': _summarise(per_seed, 'train'), 'heldout': _summarise(per_seed, 'heldout'),
                                  'estimated_cost': _summarise(per_seed, 'estimated_cost'),
                                  'budget': _summarise(per_seed, 'budget')}}

    def compare(a: str, b: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        pa, pb = arms[a]['per_seed'], arms[b]['per_seed']
        st = paired_stats(_metric(pa, *primary), _metric(pb, *primary))
        out['primary'] = {'section': primary[0], 'metric': primary[1], 'higher_is_better': higher, **st}
        gst = None
        if guard is not None:
            gst = paired_stats(_metric(pa, *guard), _metric(pb, *guard))
            out['guard'] = {'section': guard[0], 'metric': guard[1], 'higher_is_better': guard_higher, **gst}
        if kind in ('temporal_task', 'graft'):
            out['train_accuracy'] = paired_stats(_metric(pa, 'train', 'accuracy'), _metric(pb, 'train', 'accuracy'))
        pv = verdict_from_stats(st, higher_is_better=higher)
        if gst is not None and pv['verdict'] == 'improved':
            if verdict_from_stats(gst, higher_is_better=guard_higher)['verdict'] == 'worse':
                pv = {'verdict': 'no_detectable_difference', 'reason': 'guard_metric_significantly_worse'}
        out.update(pv)
        return out

    def trajectory(name: str) -> Any:
        return [_strip_wallclock(_jsonable({k: r.get(k) for k in ('train', 'heldout')})) for r in arms[name]['per_seed']]

    comparisons = {'candidate_vs_baseline': compare('candidate', 'baseline')}
    ablation_verdicts = {}
    inert: list[str] = []
    cand_traj = trajectory('candidate')
    for name in arms:
        if name.startswith('minus_'):
            c = compare('candidate', name)
            mech = name[len('minus_'):]
            if trajectory(name) == cand_traj:
                # removing the mechanism reproduced the candidate exactly: it does not act in this
                # configuration, so no ablation verdict about it can be drawn
                c.update(verdict='no_detectable_difference', reason='inert_in_this_configuration', inert=True)
                ablation_verdicts[mech] = 'inert_in_this_configuration'
                inert.append(mech)
            else:
                ablation_verdicts[mech] = c['verdict']
            comparisons[f'candidate_vs_{name}'] = c
    if inert:
        notes.append('inert in this configuration (minus arm bit-identical to candidate on train and held-out '
                      'metrics): ' + ', '.join(inert))
    if kind == 'micro':
        st = paired_stats(_metric(arms['candidate']['per_seed'], 'heldout', 'mse'),
                          _metric(arms['candidate']['per_seed'], 'heldout', 'persistence_mse'))
        comparisons['candidate_vs_persistence'] = {'primary': {'metric': 'mse', 'higher_is_better': False, **st},
                                                   **verdict_from_stats(st, higher_is_better=False)}
    cb, bb = arms['candidate']['summary']['budget'], arms['baseline']['summary']['budget']
    cp, bp = cb.get('trainable_parameters'), bb.get('trainable_parameters')
    budget = {'candidate_trainable_parameters': cp, 'baseline_trainable_parameters': bp,
              'parameter_ratio': (cp / bp) if (cp is not None and bp) else None,
              'budget_matched_within_10pct': (cp is not None and bp is not None and bp > 0 and abs(cp / bp - 1.0) <= 0.10)}
    result = {'arms': arms, 'comparisons': comparisons, 'ablation_verdicts': ablation_verdicts, 'budget': budget,
              'inert_mechanisms': inert,
              'verdict': comparisons['candidate_vs_baseline']['verdict'],
              'verdict_basis': {'comparison': 'candidate_vs_baseline', 'section': primary[0], 'metric': primary[1],
                                'higher_is_better': higher, 'reason': comparisons['candidate_vs_baseline']['reason']},
              'notes': notes}
    if kind == 'graft':
        result['removal_checks'] = [r['removal'] for r in arms['candidate']['per_seed']]
    return result


def run_all_experiments(quick: bool = True, *, experiment_ids: Sequence[str] | None = None,
                        **overrides: Any) -> list[dict[str, Any]]:
    """Run every registered experiment (``quick`` = tiny smoke settings)."""
    ids = EXPERIMENT_IDS if experiment_ids is None else tuple(experiment_ids)
    out = []
    for eid in ids:
        kw: dict[str, Any] = dict(QUICK_ARGS) if quick else {}
        if quick and _EXP[eid].kind == 'lesion':
            kw['lesion_kwargs'] = {'fractions': (0.2,), 'recovery_steps': 8}
        kw.update(overrides)
        out.append(run_experiment(eid, **kw))
    return out


# ====================================================================== supplementary experiments
# Additive registry: these are *not* part of spec.EXPERIMENTS (which stays at 24 entries) and are not run by
# run_all_experiments. Each entry maps an id to (title, kind, module, function, description).
SUPPLEMENTARY_EXPERIMENTS: dict[str, dict[str, str]] = {
    'sequence_search': {
        'id': 'sequence_search', 'title': 'Budget-matched synthetic sequence-variant search',
        'kind': 'sequence_search', 'module': 'sequence_search', 'function': 'run_sequence_search_assay',
        'description': ('Fly-Protein-inspired sparse random-projection guidance vs evolutionary vs random search '
                        'at an equal query budget on synthetic landscapes; not protein modelling.'),
    },
}
SUPPLEMENTARY_IDS: tuple[str, ...] = tuple(SUPPLEMENTARY_EXPERIMENTS)


def supplementary_catalog() -> list[dict[str, str]]:
    return [{k: v for k, v in e.items() if k not in ('module', 'function')} for e in SUPPLEMENTARY_EXPERIMENTS.values()]


def run_supplementary_experiment(experiment_id: str, **kwargs: Any) -> dict[str, Any]:
    """Run one supplementary experiment (lazy import of its module) and return its sealed receipt."""
    if experiment_id not in SUPPLEMENTARY_EXPERIMENTS:
        raise ValueError(f'unknown supplementary experiment {experiment_id!r}; choose from {list(SUPPLEMENTARY_IDS)}')
    import importlib
    info = SUPPLEMENTARY_EXPERIMENTS[experiment_id]
    fn = getattr(importlib.import_module(f'{__package__}.{info["module"]}'), info['function'])
    return fn(**kwargs)


_CSV_FIELDS = ('experiment_id', 'kind', 'task', 'arm', 'mechanisms', 'seeds', 'train_metric', 'train_value',
               'heldout_metric', 'heldout_value', 'verdict', 'ci95_low', 'ci95_high', 'evidence_tier', 'digest')


def results_to_csv(receipts: Iterable[Mapping[str, Any]]) -> str:
    """One CSV row per (receipt, arm); the verdict/CI columns refer to that arm's comparison with the candidate."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, lineterminator='\n')
    w.writeheader()
    for r in receipts:
        basis = r.get('verdict_basis', {}) or {}
        metric = basis.get('metric', 'accuracy')
        section = basis.get('section', 'heldout')
        comps = r.get('comparisons', {}) or {}
        arms = r.get('arms', {}) or {}
        if not arms:
            ci = (comps.get(basis.get('comparison', ''), {}) or {}).get('primary', {}).get('ci95') or [None, None]
            w.writerow({'experiment_id': r['experiment_id'], 'kind': r['kind'], 'task': r.get('task'), 'arm': 'assay',
                        'mechanisms': ' '.join(r.get('candidate_mechanisms', [])),
                        'seeds': ' '.join(map(str, r['config']['seeds'])), 'verdict': r.get('verdict'),
                        'ci95_low': ci[0], 'ci95_high': ci[1], 'heldout_metric': metric,
                        'evidence_tier': r.get('evidence_tier'), 'digest': r.get('digest')})
            continue
        for name, arm in arms.items():
            summ = arm.get('summary') or {}
            if r.get('kind') == 'topology':
                comp = comps.get(f"{name}_vs_{r.get('reference_topology')}", {})
            elif name == 'candidate':
                comp = comps.get('candidate_vs_baseline', {})
            else:
                comp = comps.get(f'candidate_vs_{name}', {})
            ci = (comp.get('primary') or comp.get('heldout_accuracy') or {}).get('ci95') or [None, None]
            if summ:
                tv, hv = summ.get('train', {}).get(metric), summ.get(section, {}).get(metric)
            else:  # topology arms
                tv = _mean(p['train']['accuracy'] for p in arm['per_seed'])
                hv = _mean(p['heldout']['accuracy'] for p in arm['per_seed'])
            w.writerow({'experiment_id': r['experiment_id'], 'kind': r['kind'], 'task': r.get('task'), 'arm': name,
                        'mechanisms': ' '.join(arm.get('mechanisms', []) or []),
                        'seeds': ' '.join(map(str, r['config']['seeds'])), 'train_metric': metric, 'train_value': tv,
                        'heldout_metric': metric, 'heldout_value': hv, 'verdict': comp.get('verdict', ''),
                        'ci95_low': ci[0], 'ci95_high': ci[1], 'evidence_tier': r.get('evidence_tier'),
                        'digest': r.get('digest')})
    return buf.getvalue()
