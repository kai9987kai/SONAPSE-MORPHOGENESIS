"""Quality-diversity architecture search over NEUROWEAVE configurations (frontier module ``quality_diversity``).

Inspired by MAP-Elites -- Jean-Baptiste Mouret and Jeff Clune, "Illuminating search spaces by mapping elites",
arXiv:1504.04909, April 2015 -- by the parent selection of the Darwin Gödel Machine (DGM) -- Jenny Zhang,
Shengran Hu, Cong Lu, Robert Lange and Jeff Clune, arXiv:2505.22954, May 2025 (ICLR 2026 poster) -- and by the
sibling ``kai9987kai`` projects SONAPSE-ORIGIN (a 2-D MAP-Elites archive with mutation and crossover, an NSGA-II
Pareto front) and sonapse-evo (named structural operators with preconditions, a failure archive with reasons,
verify-before-commit transactions). This is a synthetic computational abstraction: it searches a small declared
space of NEUROWEAVE switches on tiny temporal tasks and reproduces none of those papers' systems or results.
Full, verified references: ``docs/frontier/modules/quality_diversity.md``.

Search loop (MAP-Elites, paper Fig. 2, with a DGM-style parent distribution)::

    for t = 1 .. budget:
        if t <= G or the archive is empty:  x' <- random genome
        else:  a, b ~ Categorical(p) (two draws, with replacement);  x' <- mutate(crossover(a, b) or a)
        receipt <- evaluate(x')            # NEUROWEAVE train run + frozen held-out runs, SHA-256 sealed
        reasons <- verify(x', receipt)     # + re-evaluation digest check on every k-th candidate
        if reasons:  log the rejection     # the archive is unchanged
        else:  b' <- (bin of log2 estimated ops/step, bin of activity sparseness);  q' <- (accuracy, -log loss)
               if cell b' is empty or q' > q(elite[b']):  elite[b'] <- x'

    s_i = 1 / (1 + exp(-lambda (f_i - f_0)))     h_i = 1 / (1 + n_i)     p_i = s_i h_i / sum_j s_j h_j

``f_i`` is an elite's held-out accuracy and ``n_i`` its number of children that passed verification (DGM counts
children that kept codebase-editing functionality); as in DGM (arXiv v3, App. C.2) only elites with ``f_i < 1`` are
eligible, and -- our addition, which DGM does not define -- all elites are when every one is perfect. ``lambda = 10``
and ``f_0 = 0.5`` are the DGM values. DGM samples ``k`` parents with replacement and each produces its own child;
here each iteration draws two parents with replacement and makes one child (the second parent is only a crossover
partner). Activity sparseness is Hoyer's (2004) measure ``(sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)`` of each
held-out output vector, averaged over steps. QD-score (Pugh, Soros, Szerlip & Stanley, GECCO 2015; Pugh, Soros &
Stanley, 2016) is the total fitness over filled cells; reliability, precision and coverage follow Mouret & Clune
(2015, Sec. 5 and 9.4). The summary also lists the elites on the Pareto front of (fitness, estimated cost), the
sibling projects' multi-objective view of the same archive.

All randomness comes from ``numpy.random.default_rng(seed + offset)``; the NEUROWEAVE runs are seeded by the
engine itself. Nothing here touches global NumPy/PyTorch random or thread state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..neuroweave.engine import NeuroWeaveEngine
from ..neuroweave.experiments import (ReadoutHead, _estimated_cost, _jsonable, _mean, _strip_wallclock, clone_engine,
                                      make_spec, run_task_phase)
from ..neuroweave.spec import MECHANISM_IDS, TOPOLOGIES, NeuroWeaveSpecV1
from ..neuroweave.tasks import TASKS as NEUROWEAVE_TASKS, make_stream
from . import harness
from .extension import EngineExtension
from .spec import EVIDENCE_TIER, LAB_SCHEMA

__all__ = ['MODULE_ID', 'SOURCE', 'EXPERIMENTS', 'CANDIDATE_RECEIPT_ID', 'ARCHIVE_SCHEMA', 'SEARCH_MECHANISMS',
           'NEURON_CHOICES', 'SPARSITY_CHOICES', 'MUTATION_OPERATORS', 'REJECTION_REASONS', 'Genome', 'GenomeSpace',
           'ActivityRecorder', 'NeuroWeaveEvaluator', 'LandscapeEvaluator', 'EliteArchive', 'QualityDiversitySearch',
           'hoyer_sparseness', 'mutate', 'crossover', 'applicable_operators', 'dgm_parent_probabilities',
           'candidate_metrics', 'verify_candidate', 'illumination_metrics', 'pareto_front', 'run_experiment']

MODULE_ID = 'quality_diversity'
CANDIDATE_RECEIPT_ID = MODULE_ID + '_candidate'     # experiment id inside per-candidate receipts (not registered)
ARCHIVE_SCHEMA = 'QualityDiversityArchiveV1'
SOURCE = ('SONAPSE-ORIGIN and sonapse-evo (kai9987kai); Mouret & Clune, "Illuminating search spaces by mapping '
          'elites", arXiv:1504.04909, April 2015, https://arxiv.org/abs/1504.04909; Zhang, Hu, Lu, Lange & Clune, '
          '"Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents", arXiv:2505.22954, May 2025 '
          '(ICLR 2026), https://arxiv.org/abs/2505.22954')

# A declared search set of ten NEUROWEAVE switches (canonical MECHANISM_IDS order). Five of them also change the
# engine's estimated operation count, so the compute-cost descriptor is not only a function of size and topology.
SEARCH_MECHANISMS: tuple[str, ...] = tuple(m for m in MECHANISM_IDS if m in {
    'structural_encoder', 'sparse_code', 'reward_plasticity', 'synaptic_delays', 'homeostatic_governor',
    'context_attention', 'flyhash_projection', 'adaptive_routing', 'hierarchical_temporal_memory',
    'membrane_dynamics'})
NEURON_CHOICES: tuple[int, ...] = (16, 24, 32)
SPARSITY_CHOICES: tuple[float, ...] = (0.05, 0.15, 0.3, 0.5)    # k-winners fraction; expressed with 'sparse_code'
MUTATION_OPERATORS = ('toggle_mechanism', 'swap_mechanism', 'change_topology', 'step_neurons', 'step_sparsity')
REJECTION_REASONS = ('evaluation_error', 'receipt_not_a_mapping', 'receipt_digest_mismatch', 'receipt_schema_mismatch',
                     'genome_outside_space', 'genome_mismatch', 'protocol_mismatch', 'invalid_spec', 'spec_mismatch',
                     'invalid_metrics', 'inconsistent_metrics', 'spot_check_error', 'spot_check_digest_mismatch')
ORIGINS = ('random', 'mutation', 'crossover')
STATUSES = ('elite', 'not_elite', 'rejected')
SELECTIONS = ('dgm', 'uniform')
METHODS = ('qd', 'random')
DESCRIPTOR_KEYS = ('log2_ops_per_step', 'activity_sparseness')
MAX_BUDGET = 10_000
MAX_ARCHIVE_CHARS = 64 * 1024 * 1024
_MAX_OPERATORS = 8
_MAX_TEXT = 200

_INIT_OFFSET = 67_001        # random-genome stream (shared by both arms of a seed: common random numbers)
_VAR_OFFSET = 67_013         # parent selection / variation stream (QD arm only)

_NODE_KEYS = frozenset({'id', 'genome', 'origin', 'parents', 'operators', 'status', 'reasons', 'error', 'cell',
                        'fitness', 'log_loss', 'descriptors', 'digest', 'spot_checked', 'displaced_by'})
_ELITE_KEYS = frozenset({'cell', 'id', 'genome', 'fitness', 'log_loss', 'descriptors', 'digest', 'receipt'})
_REJECTION_KEYS = frozenset({'id', 'genome', 'reasons', 'error', 'digest'})


# ====================================================================== validation helpers
def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _num(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return v


def _seq(name: str, value: Any) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f'{name} must be a list or tuple')
    return tuple(value)


def _seed_tuple(name: str, value: Any) -> tuple[int, ...]:
    out = tuple(_int(name, s, 0, 2**31 - 1) for s in _seq(name, value))
    if not out or len(set(out)) != len(out):
        raise ValueError(f'{name} must be non-empty and unique')
    return out


def _rng(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator')
    return rng


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode('utf-8')).hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


# ====================================================================== genome and search space
@dataclass(frozen=True)
class Genome:
    """One NEUROWEAVE configuration: a mechanism subset, a topology, a population size and a k-WTA fraction."""

    mechanisms: tuple[str, ...]
    topology: str
    neurons: int
    sparsity: float

    def __post_init__(self) -> None:
        mechs = _seq('mechanisms', self.mechanisms)
        if any(not isinstance(m, str) or m not in MECHANISM_IDS for m in mechs) or len(set(mechs)) != len(mechs):
            raise ValueError('mechanisms must be unique NEUROWEAVE mechanism ids')
        object.__setattr__(self, 'mechanisms', tuple(m for m in MECHANISM_IDS if m in set(mechs)))
        if not isinstance(self.topology, str) or self.topology not in TOPOLOGIES:
            raise ValueError(f'topology must be one of {TOPOLOGIES}')
        object.__setattr__(self, 'neurons', _int('neurons', self.neurons, 8, 4096))
        object.__setattr__(self, 'sparsity', _num('sparsity', self.sparsity, 0.01, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {'mechanisms': list(self.mechanisms), 'topology': self.topology, 'neurons': self.neurons,
                'sparsity': self.sparsity}

    @classmethod
    def from_dict(cls, data: Any) -> 'Genome':
        if not isinstance(data, Mapping) or set(data) != {'mechanisms', 'topology', 'neurons', 'sparsity'}:
            raise ValueError('genome must be a mapping with mechanisms, topology, neurons and sparsity')
        return cls(tuple(_seq('mechanisms', data['mechanisms'])), data['topology'], data['neurons'], data['sparsity'])

    @property
    def key(self) -> str:
        """Canonical JSON identity (used for duplicate detection and lineage)."""
        return _canonical(self.to_dict())

    def to_spec(self, *, seed: int, steps: int) -> NeuroWeaveSpecV1:
        """The lab spec (``experiments.make_spec``) with this genome's k-WTA fraction."""
        spec = make_spec(self.mechanisms, seed=int(seed), neurons=self.neurons, topology=self.topology, steps=int(steps))
        return replace(spec, sparsity=self.sparsity)


@dataclass(frozen=True)
class GenomeSpace:
    """The declared search space. Choice lists are canonicalised (NEUROWEAVE order, ascending numbers)."""

    mechanisms: tuple[str, ...] = SEARCH_MECHANISMS
    topologies: tuple[str, ...] = TOPOLOGIES
    neurons: tuple[int, ...] = NEURON_CHOICES
    sparsity: tuple[float, ...] = SPARSITY_CHOICES

    def __post_init__(self) -> None:
        mechs = _seq('mechanisms', self.mechanisms)
        if any(not isinstance(m, str) or m not in MECHANISM_IDS for m in mechs) or len(set(mechs)) != len(mechs):
            raise ValueError('mechanisms must be unique NEUROWEAVE mechanism ids')
        topos = _seq('topologies', self.topologies)
        if not topos or any(not isinstance(t, str) or t not in TOPOLOGIES for t in topos) or len(set(topos)) != len(topos):
            raise ValueError(f'topologies must be a non-empty unique subset of {TOPOLOGIES}')
        neurons = tuple(_int('neurons', n, 8, 4096) for n in _seq('neurons', self.neurons))
        sparsity = tuple(_num('sparsity', s, 0.01, 1.0) for s in _seq('sparsity', self.sparsity))
        for name, vals in (('neurons', neurons), ('sparsity', sparsity)):
            if not 1 <= len(vals) <= 64 or len(set(vals)) != len(vals):
                raise ValueError(f'{name} must hold 1 to 64 unique values')
        object.__setattr__(self, 'mechanisms', tuple(m for m in MECHANISM_IDS if m in set(mechs)))
        object.__setattr__(self, 'topologies', tuple(t for t in TOPOLOGIES if t in set(topos)))
        object.__setattr__(self, 'neurons', tuple(sorted(neurons)))
        object.__setattr__(self, 'sparsity', tuple(sorted(sparsity)))
        if self.size < 2:
            raise ValueError('the search space must contain at least two genomes')

    @property
    def size(self) -> int:
        return 2 ** len(self.mechanisms) * len(self.topologies) * len(self.neurons) * len(self.sparsity)

    def contains(self, genome: Any) -> bool:
        return (isinstance(genome, Genome) and set(genome.mechanisms) <= set(self.mechanisms)
                and genome.topology in self.topologies and genome.neurons in self.neurons
                and genome.sparsity in self.sparsity)

    def random_genome(self, rng: np.random.Generator) -> Genome:
        """Uniform over the space: each mechanism on with probability 1/2, every other gene uniform."""
        rng = _rng(rng)
        on = rng.random(len(self.mechanisms)) < 0.5
        return Genome(tuple(m for m, b in zip(self.mechanisms, on) if b),
                      self.topologies[int(rng.integers(len(self.topologies)))],
                      self.neurons[int(rng.integers(len(self.neurons)))],
                      self.sparsity[int(rng.integers(len(self.sparsity)))])

    def to_dict(self) -> dict[str, Any]:
        return {'mechanisms': list(self.mechanisms), 'topologies': list(self.topologies),
                'neurons': list(self.neurons), 'sparsity': list(self.sparsity)}

    @classmethod
    def from_dict(cls, data: Any) -> 'GenomeSpace':
        if not isinstance(data, Mapping) or set(data) != {'mechanisms', 'topologies', 'neurons', 'sparsity'}:
            raise ValueError('genome space must be a mapping with mechanisms, topologies, neurons and sparsity')
        return cls(*(tuple(_seq(k, data[k])) for k in ('mechanisms', 'topologies', 'neurons', 'sparsity')))


# ====================================================================== variation operators (sonapse-evo style)
def _check_in_space(genome: Any, space: Any) -> None:
    if not isinstance(space, GenomeSpace):
        raise ValueError('space must be a GenomeSpace')
    if not space.contains(genome):
        raise ValueError('genome must be a Genome inside the search space')


def applicable_operators(genome: Genome, space: GenomeSpace) -> tuple[str, ...]:
    """Operators whose preconditions hold; every listed operator always changes the genome."""
    _check_in_space(genome, space)
    on = len(genome.mechanisms)
    ok = {'toggle_mechanism': len(space.mechanisms) >= 1,
          'swap_mechanism': 1 <= on < len(space.mechanisms),
          'change_topology': len(space.topologies) >= 2,
          'step_neurons': len(space.neurons) >= 2,
          'step_sparsity': len(space.sparsity) >= 2}
    return tuple(op for op in MUTATION_OPERATORS if ok[op])


def _step(values: tuple, current: Any, rng: np.random.Generator) -> Any:
    i = values.index(current)
    moves = [j for j in (i - 1, i + 1) if 0 <= j < len(values)]
    return values[moves[int(rng.integers(len(moves)))]]


def _apply(op: str, g: Genome, space: GenomeSpace, rng: np.random.Generator) -> tuple[Genome, str]:
    if op == 'toggle_mechanism':
        m = space.mechanisms[int(rng.integers(len(space.mechanisms)))]
        if m in g.mechanisms:
            return replace(g, mechanisms=tuple(x for x in g.mechanisms if x != m)), f'toggle_mechanism:-{m}'
        return replace(g, mechanisms=g.mechanisms + (m,)), f'toggle_mechanism:+{m}'
    if op == 'swap_mechanism':
        off = tuple(m for m in space.mechanisms if m not in g.mechanisms)
        out_m = g.mechanisms[int(rng.integers(len(g.mechanisms)))]
        in_m = off[int(rng.integers(len(off)))]
        return (replace(g, mechanisms=tuple(x for x in g.mechanisms if x != out_m) + (in_m,)),
                f'swap_mechanism:-{out_m}+{in_m}')
    if op == 'change_topology':
        others = tuple(t for t in space.topologies if t != g.topology)
        t = others[int(rng.integers(len(others)))]
        return replace(g, topology=t), f'change_topology:{g.topology}->{t}'
    if op == 'step_neurons':
        n = _step(space.neurons, g.neurons, rng)
        return replace(g, neurons=n), f'step_neurons:{g.neurons}->{n}'
    s = _step(space.sparsity, g.sparsity, rng)
    return replace(g, sparsity=s), f'step_sparsity:{g.sparsity}->{s}'


def mutate(genome: Genome, space: GenomeSpace, rng: np.random.Generator, *, extra_prob: float = 0.25,
           max_operators: int = 3) -> tuple[Genome, tuple[str, ...]]:
    """Apply one uniformly chosen applicable operator, then each further one with probability ``extra_prob``
    (at most ``max_operators``). The child always differs from ``genome``; the audit trail is returned."""
    _check_in_space(genome, space)
    rng = _rng(rng)
    extra_prob = _num('extra_prob', extra_prob, 0.0, 0.95)
    max_operators = _int('max_operators', max_operators, 1, _MAX_OPERATORS - 2)
    count = 1
    while count < max_operators and rng.random() < extra_prob:
        count += 1
    g, trail = genome, []
    for _ in range(count):
        ops = applicable_operators(g, space)
        g, detail = _apply(ops[int(rng.integers(len(ops)))], g, space, rng)
        trail.append(detail)
    if g == genome:                         # e.g. the same mechanism toggled twice: one more change
        ops = applicable_operators(g, space)
        g, detail = _apply(ops[int(rng.integers(len(ops)))], g, space, rng)
        trail.append(detail)
    return g, tuple(trail)


def crossover(a: Genome, b: Genome, space: GenomeSpace, rng: np.random.Generator) -> Genome:
    """Uniform crossover: each mechanism's membership and each other gene comes from either parent (p = 1/2)."""
    _check_in_space(a, space)
    _check_in_space(b, space)
    rng = _rng(rng)
    pick = rng.random(len(space.mechanisms) + 3) < 0.5
    mechs = tuple(m for i, m in enumerate(space.mechanisms) if m in (a.mechanisms if pick[i] else b.mechanisms))
    k = len(space.mechanisms)
    return Genome(mechs, a.topology if pick[k] else b.topology, a.neurons if pick[k + 1] else b.neurons,
                  a.sparsity if pick[k + 2] else b.sparsity)


def dgm_parent_probabilities(fitness: Sequence[float], children: Sequence[int], *, sharpness: float = 10.0,
                             midpoint: float = 0.5) -> np.ndarray:
    """DGM parent distribution (Zhang et al., 2025, arXiv v3 App. C.2): ``p_i ∝ sigmoid(sharpness (f_i - midpoint))
    / (1 + n_i)`` over elites with ``f_i < 1`` (our fallback: all elites when every one has ``f_i = 1``)."""
    f = np.asarray(fitness, dtype=np.float64)
    n = np.asarray(children)
    if f.ndim != 1 or f.size == 0 or n.shape != f.shape:
        raise ValueError('fitness and children must be equal-length non-empty 1-D sequences')
    if not np.all(np.isfinite(f)) or np.any(f < 0.0) or np.any(f > 1.0):
        raise ValueError('fitness values must be finite and in [0, 1]')
    if n.dtype.kind not in 'iu' or np.any(n < 0):
        raise ValueError('children must be non-negative integers')
    sharpness = _num('sharpness', sharpness, 0.0, 100.0)
    midpoint = _num('midpoint', midpoint, 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-sharpness * (f - midpoint)))      # |exponent| <= 100: no overflow, s > 0
    w = s / (1.0 + n.astype(np.float64))
    eligible = f < 1.0
    if not eligible.any():
        eligible[:] = True
    w = np.where(eligible, w, 0.0)
    return w / w.sum()


# ====================================================================== activity sparseness
def hoyer_sparseness(x: Any) -> float | None:
    """Hoyer (2004) sparseness of a vector, in [0, 1]: 1 for one non-zero entry, 0 when all magnitudes are equal.
    Returns ``None`` for the all-zero vector, where the measure is undefined."""
    a = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    if a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError('x must hold at least two finite values')
    peak = float(a.max())
    if peak == 0.0:
        return None
    b = a / peak                                           # scale first: no underflow in the squares
    n = a.size
    v = (math.sqrt(n) - float(b.sum()) / math.sqrt(float(b @ b))) / (math.sqrt(n) - 1.0)
    return float(min(1.0, max(0.0, v)))


class ActivityRecorder(EngineExtension):
    """Read-only engine extension: running mean of the Hoyer sparseness of the engine output.

    It adds no readout features and leaves the drive untouched (the neutral base hooks), so an engine's
    trajectory is identical with and without it (tested). All-zero (silent) steps are counted separately.
    It keeps counting during frozen (``learn=False``) runs, where it is used: these are measurements that never
    feed back into the engine or the readout, so frozen evaluation is unaffected. ``reset()`` keeps the counts."""

    id = 'qd_activity_recorder'
    feature_dim = 0

    def __init__(self) -> None:
        self._sum = 0.0
        self._scored = 0
        self._silent = 0

    def after_step(self, output: np.ndarray, ctx: Mapping[str, Any]) -> None:
        h = hoyer_sparseness(output)
        if h is None:
            self._silent += 1
        else:
            self._sum += h
            self._scored += 1

    def state_numbers(self) -> int:
        return 3

    def stats(self) -> dict[str, Any]:
        return {'hoyer_sum': self._sum, 'scored_steps': self._scored, 'silent_steps': self._silent}

    @staticmethod
    def combine(recorders: Iterable['ActivityRecorder']) -> dict[str, Any]:
        """Pooled mean over every scored step; a run whose steps were all silent counts as fully sparse (1.0)."""
        total, scored, silent = 0.0, 0, 0
        for r in recorders:
            total += r._sum
            scored += r._scored
            silent += r._silent
        return {'hoyer_sparseness': float(total / scored) if scored else 1.0, 'scored_steps': scored,
                'silent_steps': silent}

    def telemetry(self) -> dict[str, Any]:
        return {'id': self.id, **self.stats()}

    def export_state(self) -> dict[str, Any]:
        return {'id': self.id, **self.stats()}

    def import_state(self, state: Mapping[str, Any]) -> None:
        super().import_state(state)
        if set(state) != {'id', 'hoyer_sum', 'scored_steps', 'silent_steps'}:
            raise ValueError('activity recorder state has unexpected keys')
        scored = _int('scored_steps', state['scored_steps'], 0, 2**62)
        silent = _int('silent_steps', state['silent_steps'], 0, 2**62)
        total = _num('hoyer_sum', state['hoyer_sum'], 0.0, float(scored))
        self._sum, self._scored, self._silent = total, scored, silent


# ====================================================================== candidate receipts and verification
def _candidate_receipt(genome: Genome, spec: NeuroWeaveSpecV1, *, seed: int, protocol: Mapping[str, Any],
                       train: Mapping[str, Any], heldout: Mapping[str, Any], estimated_cost: Mapping[str, Any],
                       activity: Mapping[str, Any], budget: Mapping[str, Any] | None,
                       wallclock: Mapping[str, float]) -> dict[str, Any]:
    steps = int(protocol['steps'])
    ops = float(estimated_cost['total_ops'])
    if not ops > 0.0:
        raise ValueError('estimated total_ops must be positive')
    receipt = harness.base_receipt(CANDIDATE_RECEIPT_ID, 'candidate_evaluation', {
        'genome': genome.to_dict(), 'seed': int(seed), 'spec': spec.to_dict(), 'protocol': dict(protocol)})
    receipt['module'] = MODULE_ID
    receipt['train'] = dict(train)
    receipt['heldout'] = dict(heldout)
    receipt['activity'] = dict(activity)
    receipt['estimated_cost'] = dict(estimated_cost)
    receipt['budget'] = None if budget is None else dict(budget)
    receipt['fitness'] = heldout['accuracy']
    receipt['descriptors'] = {'log2_ops_per_step': math.log2(ops / steps),
                              'activity_sparseness': activity['hoyer_sparseness']}
    receipt['measured_wallclock_s'] = dict(wallclock)
    return harness.seal(receipt)


def candidate_metrics(receipt: Mapping[str, Any]) -> dict[str, float]:
    """Fitness, tie-break and descriptors recomputed from a candidate receipt's raw fields (``ValueError`` if
    a field is missing or out of range). Fitness is held-out accuracy; lower held-out log loss breaks ties."""
    try:
        steps = receipt['config']['protocol']['steps']
        acc = receipt['heldout']['accuracy']
        loss = receipt['heldout']['log_loss']
        ops = receipt['estimated_cost']['total_ops']
        sparse = receipt['activity']['hoyer_sparseness']
    except (KeyError, TypeError) as exc:
        raise ValueError('candidate receipt is missing metric fields') from exc
    steps = _int('protocol steps', steps, 1, 10_000_000)
    acc = _num('heldout accuracy', acc, 0.0, 1.0)
    loss = _num('heldout log_loss', loss, 0.0, 1e6)
    ops = _num('estimated total_ops', ops, 0.0, 1e18)
    sparse = _num('activity sparseness', sparse, 0.0, 1.0)
    if not ops > 0.0:
        raise ValueError('estimated total_ops must be positive')
    return {'fitness': acc, 'log_loss': loss, 'log2_ops_per_step': math.log2(ops / steps), 'activity_sparseness': sparse}


def verify_candidate(genome: Genome, receipt: Any, *, seed: int, protocol: Mapping[str, Any],
                     space: GenomeSpace) -> list[str]:
    """Static admission checks; returns the failed reason codes (empty = verified). It never raises for a bad
    receipt: the seal, schema, genome, protocol, spec (valid *and* equal to the spec the genome implies) and
    metric consistency are all checked against what the search asked for, not against the receipt itself.
    Bad *arguments* (``seed``, ``protocol`` without an int ``steps``, ``space``) raise ``ValueError``."""
    seed = _int('seed', seed, 0, 2**31 - 1)
    if not isinstance(protocol, Mapping) or type(protocol.get('steps')) is not int:
        raise ValueError('protocol must be a mapping with an int steps')
    if not isinstance(space, GenomeSpace):
        raise ValueError('space must be a GenomeSpace')
    if not isinstance(receipt, Mapping):
        return ['receipt_not_a_mapping']
    reasons: list[str] = []
    if not harness.verify_receipt(receipt):
        reasons.append('receipt_digest_mismatch')
    if (receipt.get('schema') != LAB_SCHEMA or receipt.get('experiment_id') != CANDIDATE_RECEIPT_ID
            or receipt.get('kind') != 'candidate_evaluation' or receipt.get('evidence_tier') != EVIDENCE_TIER
            or receipt.get('module') != MODULE_ID):
        reasons.append('receipt_schema_mismatch')
    if not space.contains(genome):
        reasons.append('genome_outside_space')
    cfg = receipt.get('config')
    cfg = cfg if isinstance(cfg, Mapping) else {}
    if not isinstance(genome, Genome) or cfg.get('genome') != genome.to_dict():
        reasons.append('genome_mismatch')
    if cfg.get('protocol') != _jsonable(dict(protocol)) or cfg.get('seed') != seed:
        reasons.append('protocol_mismatch')
    try:
        spec = NeuroWeaveSpecV1.from_dict(cfg.get('spec'))
    except (ValueError, TypeError):
        reasons.append('invalid_spec')
    else:
        if not isinstance(genome, Genome) or spec != genome.to_spec(seed=seed, steps=int(protocol['steps'])):
            reasons.append('spec_mismatch')
    try:
        m = candidate_metrics(receipt)
    except ValueError:
        reasons.append('invalid_metrics')
    else:
        if (receipt.get('fitness') != m['fitness'] or receipt.get('descriptors') !=
                {'log2_ops_per_step': m['log2_ops_per_step'], 'activity_sparseness': m['activity_sparseness']}):
            reasons.append('inconsistent_metrics')
    return reasons


# ====================================================================== evaluators
def _engine_budget(engine: NeuroWeaveEngine, head: ReadoutHead) -> dict[str, int]:
    """``experiments._budget`` without its ``state_numbers`` count (a full state export per candidate)."""
    f = engine.flags
    syn = engine.brain.synapses
    micro = int(engine.micro.parameter_count) if 'micro_collective' in f else 0
    graft = int(sum(w.size for w in engine.graft._weights())) if engine.graft is not None else 0
    return {'neurons': int(engine.n), 'synapses': int(syn.size),
            'trainable_parameters': int(syn.size) + int(head.parameter_count) + micro + graft,
            'micro_parameters': micro, 'graft_parameters': graft}


class NeuroWeaveEvaluator:
    """Fitness of a genome on a NEUROWEAVE temporal task, following ``experiments._task_arm``'s protocol.

    Train: a fresh engine from ``genome.to_spec(seed, steps)`` and a fresh ``ReadoutHead`` run prequentially on
    ``make_stream(seed, steps)``. Held out: for each held-out seed, the trained engine (a validated clone for all
    but the last stream) and a copy of the head run frozen (``learn=False``) on ``make_stream(h, heldout_steps)``,
    with an :class:`ActivityRecorder` attached. Fitness = mean held-out accuracy. With ``sparsity = 0.15`` (the
    spec default) the metrics and estimated cost equal ``_task_arm``'s exactly (tested); the recorder does not
    change the trajectory (tested). The k-WTA ``sparsity`` gene only acts when ``sparse_code`` is on."""

    def __init__(self, *, task: str = 'delayed_recall', steps: int = 48, heldout_seeds: Sequence[int] = (100, 101),
                 heldout_steps: int = 24) -> None:
        if task not in NEUROWEAVE_TASKS:
            raise ValueError(f'task must be one of {NEUROWEAVE_TASKS}')
        self.task = task
        self.steps = _int('steps', steps, 4, 100_000)
        self.heldout_seeds = _seed_tuple('heldout_seeds', heldout_seeds)
        self.heldout_steps = _int('heldout_steps', heldout_steps, 1, 100_000)

    def protocol(self, heldout_seeds: Sequence[int] | None = None) -> dict[str, Any]:
        held = self.heldout_seeds if heldout_seeds is None else heldout_seeds
        return {'kind': 'neuroweave_temporal_task', 'task': self.task, 'steps': self.steps,
                'heldout_seeds': list(held), 'heldout_steps': self.heldout_steps,
                'readout': 'experiments.ReadoutHead (logistic, lr 0.05), frozen held-out evaluation on clones',
                'activity': 'Hoyer sparseness of held-out outputs (ActivityRecorder)'}

    def __call__(self, genome: Genome, seed: int) -> dict[str, Any]:
        return self.evaluate(genome, seed)

    def evaluate(self, genome: Genome, seed: int, *, heldout_seeds: Sequence[int] | None = None) -> dict[str, Any]:
        if not isinstance(genome, Genome):
            raise ValueError('genome must be a Genome')
        seed = _int('seed', seed, 0, 2**31 - 1)
        held = self.heldout_seeds if heldout_seeds is None else _seed_tuple('heldout_seeds', heldout_seeds)
        if seed in held:
            raise ValueError('the training seed must not be a held-out seed')
        spec = genome.to_spec(seed=seed, steps=self.steps)
        engine = NeuroWeaveEngine(spec)
        head = ReadoutHead(engine.n, seed)
        stream = make_stream(seed, self.steps)
        t0 = time.perf_counter()
        train = run_task_phase(engine, {self.task: head}, stream, learn=True)
        cost = _estimated_cost(engine, self.steps, head.dim)       # training steps only (before any held-out step)
        budget = _engine_budget(engine, head)
        t1 = time.perf_counter()
        runs, recorders = [], []
        for k, h in enumerate(held):
            # Every held-out stream starts from the trained state: earlier streams run on validated clones, the
            # last one on the trained engine itself (it is not used again; equal to a clone's run, tested).
            target = engine if k == len(held) - 1 else clone_engine(engine)
            recorder = ActivityRecorder()
            target.attach_extension(recorder)
            runs.append(run_task_phase(target, {self.task: head.copy()}, make_stream(h, self.heldout_steps),
                                       learn=False))
            recorders.append(recorder)
        t2 = time.perf_counter()
        keys = ('accuracy', 'log_loss', 'brier', 'chance_accuracy')
        tm = train['tasks'][self.task]
        heldout = {k: _mean(r['tasks'][self.task][k] for r in runs) for k in keys}
        return _candidate_receipt(
            genome, spec, seed=seed, protocol=self.protocol(held), train={k: tm[k] for k in keys}, heldout=heldout,
            estimated_cost=cost, activity=ActivityRecorder.combine(recorders), budget=budget,
            wallclock={'train': t1 - t0, 'heldout': t2 - t1})


class LandscapeEvaluator:
    """Deterministic synthetic evaluator with a known answer (no NEUROWEAVE run), for tests and dry runs.

    Fitness is the fraction of genes equal to a hidden ``target`` genome (each search-set mechanism's membership,
    topology, neurons, sparsity), so ``target`` is the unique optimum with fitness 1. Analytic descriptors:
    ``ops/step = 16 * neurons * (4 + #mechanisms)``; activity sparseness ``1 - sparsity`` with ``sparse_code``
    and ``0.05`` without. Log loss ``0.75 - 0.5 * fitness + 0.01 * u(genome)`` (``u`` from the genome hash) gives
    deterministic tie-breaks."""

    def __init__(self, target: Genome, space: GenomeSpace, *, steps: int = 16) -> None:
        if not isinstance(space, GenomeSpace) or not space.contains(target):
            raise ValueError('target must be a Genome inside the GenomeSpace')
        self.target, self.space = target, space
        self.steps = _int('steps', steps, 4, 100_000)

    def protocol(self) -> dict[str, Any]:
        return {'kind': 'synthetic_landscape', 'target': self.target.to_dict(), 'space': self.space.to_dict(),
                'steps': self.steps}

    def fitness(self, genome: Genome) -> float:
        _check_in_space(genome, self.space)
        t = self.target
        matches = sum((m in genome.mechanisms) == (m in t.mechanisms) for m in self.space.mechanisms)
        matches += (genome.topology == t.topology) + (genome.neurons == t.neurons) + (genome.sparsity == t.sparsity)
        return matches / (len(self.space.mechanisms) + 3)

    def __call__(self, genome: Genome, seed: int) -> dict[str, Any]:
        seed = _int('seed', seed, 0, 2**31 - 1)
        f = self.fitness(genome)
        u = int(hashlib.sha256(genome.key.encode('utf-8')).hexdigest()[:8], 16) / 2**32
        sparse = 1.0 - genome.sparsity if 'sparse_code' in genome.mechanisms else 0.05
        ops = 16 * genome.neurons * (4 + len(genome.mechanisms)) * self.steps
        heldout = {'accuracy': f, 'log_loss': 0.75 - 0.5 * f + 0.01 * u, 'brier': None, 'chance_accuracy': 0.5}
        return _candidate_receipt(genome, genome.to_spec(seed=seed, steps=self.steps), seed=seed,
                                  protocol=self.protocol(), train=dict(heldout), heldout=heldout,
                                  estimated_cost={'total_ops': ops},
                                  activity={'hoyer_sparseness': sparse, 'scored_steps': self.steps, 'silent_steps': 0},
                                  budget=None, wallclock={'total': 0.0})


# ====================================================================== the archive (grid + lineage + failures)
def _quality(entry: Mapping[str, Any]) -> tuple[float, float]:
    return (float(entry['fitness']), -float(entry['log_loss']))


def _bin(value: float, lo: float, hi: float, bins: int) -> int:
    return int(min(bins - 1, max(0, math.floor((value - lo) / (hi - lo) * bins))))


def pareto_front(points: Any) -> list[int]:
    """Indices of the non-dominated rows of ``points`` (every column maximised), in input order.

    Row ``a`` dominates row ``b`` when ``a >= b`` everywhere and ``a > b`` somewhere, so equal rows never
    dominate each other and all stay on the front (the sonapse-evo / NSGA-II first front, without ranking)."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0 or not np.all(np.isfinite(p)):
        raise ValueError('points must be a non-empty finite 2-D array')
    front = []
    for i in range(p.shape[0]):
        dominated = np.any(np.all(p >= p[i], axis=1) & np.any(p > p[i], axis=1))
        if not dominated:
            front.append(i)
    return front


class EliteArchive:
    """MAP-Elites grid over (log2 estimated ops per training step, activity sparseness), plus the lineage DAG of
    every evaluated candidate and a log of rejected candidates with reasons.

    Invariant (checked again on import by replaying the lineage): each cell holds the verified candidate with the
    highest ``(fitness, -log_loss)`` among all verified candidates that mapped to it; on an exact tie the earlier
    one stays. Buffers are bounded by ``max_nodes`` (the evaluation budget)."""

    def __init__(self, *, cost_bins: int = 4, cost_range: Sequence[float] = (9.75, 11.75), activity_bins: int = 4,
                 activity_range: Sequence[float] = (0.0, 0.8), max_nodes: int = MAX_BUDGET) -> None:
        self.cost_bins = _int('cost_bins', cost_bins, 1, 64)
        self.activity_bins = _int('activity_bins', activity_bins, 1, 64)
        self.cost_range = self._range('cost_range', cost_range, 0.0, 64.0)
        self.activity_range = self._range('activity_range', activity_range, 0.0, 1.0)
        self.max_nodes = _int('max_nodes', max_nodes, 1, MAX_BUDGET)
        self._grid: dict[tuple[int, int], dict[str, Any]] = {}
        self._lineage: list[dict[str, Any]] = []
        self._rejections: list[dict[str, Any]] = []
        self._children: dict[int, int] = {}

    @staticmethod
    def _range(name: str, value: Any, low: float, high: float) -> tuple[float, float]:
        v = _seq(name, value)
        if len(v) != 2:
            raise ValueError(f'{name} must be (lo, hi)')
        lo, hi = _num(name, v[0], low, high), _num(name, v[1], low, high)
        if not lo < hi:
            raise ValueError(f'{name} must satisfy lo < hi')
        return lo, hi

    # ------------------------------------------------------------------ geometry and metrics
    def grid_config(self) -> dict[str, Any]:
        return {'cost_bins': self.cost_bins, 'cost_range': list(self.cost_range), 'activity_bins': self.activity_bins,
                'activity_range': list(self.activity_range), 'max_nodes': self.max_nodes}

    @property
    def total_cells(self) -> int:
        return self.cost_bins * self.activity_bins

    def cell_of(self, descriptors: Mapping[str, Any]) -> tuple[int, int]:
        """Grid cell of ``{'log2_ops_per_step', 'activity_sparseness'}`` (out-of-range values join the edge bins)."""
        if not isinstance(descriptors, Mapping) or set(descriptors) != set(DESCRIPTOR_KEYS):
            raise ValueError(f'descriptors must be a mapping with keys {DESCRIPTOR_KEYS}')
        c = _num('log2_ops_per_step', descriptors['log2_ops_per_step'], -1e9, 1e9)
        a = _num('activity_sparseness', descriptors['activity_sparseness'], 0.0, 1.0)
        return (_bin(c, *self.cost_range, self.cost_bins), _bin(a, *self.activity_range, self.activity_bins))

    @property
    def evaluations(self) -> int:
        return len(self._lineage)

    def __len__(self) -> int:
        return len(self._grid)

    def coverage(self) -> float:
        return len(self._grid) / self.total_cells

    def qd_score(self) -> float:
        """Pugh, Soros & Stanley (2016): sum of elite fitness over filled cells (empty cells count 0)."""
        return float(sum(e['fitness'] for e in self._grid.values()))

    def elites(self) -> list[dict[str, Any]]:
        """Current elites sorted by cell (copies, without the stored receipts)."""
        return [{k: _copy(v) for k, v in self._grid[c].items() if k != 'receipt'} for c in sorted(self._grid)]

    def elite_receipt(self, cell: Sequence[int]) -> dict[str, Any]:
        key = tuple(int(x) for x in cell)
        if key not in self._grid:
            raise KeyError(key)
        return _copy(self._grid[key]['receipt'])

    def best(self) -> dict[str, Any] | None:
        """Highest ``(fitness, -log_loss)`` elite; ties go to the earliest candidate."""
        if not self._grid:
            return None
        e = max(self._grid.values(), key=lambda x: (_quality(x), -x['id']))
        return {k: _copy(v) for k, v in e.items() if k != 'receipt'}

    def pareto_front(self) -> list[dict[str, Any]]:
        """Elites not dominated in (higher fitness, lower log2 estimated ops per step), cheapest first."""
        elites = self.elites()
        if not elites:
            return []
        idx = pareto_front([[e['fitness'], -e['descriptors']['log2_ops_per_step']] for e in elites])
        rows = [{'id': elites[i]['id'], 'cell': elites[i]['cell'], 'fitness': elites[i]['fitness'],
                 'log2_ops_per_step': elites[i]['descriptors']['log2_ops_per_step']} for i in idx]
        return sorted(rows, key=lambda r: (r['log2_ops_per_step'], r['id']))

    def children_count(self, node_id: int) -> int:
        return self._children.get(int(node_id), 0)

    def lineage(self) -> list[dict[str, Any]]:
        return _copy(self._lineage)

    def rejections(self) -> list[dict[str, Any]]:
        return _copy(self._rejections)

    # ------------------------------------------------------------------ admission transaction
    def record(self, *, genome: Genome, origin: str, parents: Sequence[int], operators: Sequence[str],
               receipt: Any, reasons: Sequence[str], spot_checked: bool, error: str | None = None) -> dict[str, Any]:
        """Log one evaluated candidate and admit it when it passed verification and beats its cell's elite.

        Every argument is checked before anything changes; a candidate with ``reasons`` is logged as rejected
        and leaves the grid untouched. Returns the new lineage node."""
        if len(self._lineage) >= self.max_nodes:
            raise ValueError('the archive is full (max_nodes candidates recorded)')
        if not isinstance(genome, Genome):
            raise ValueError('genome must be a Genome')
        if origin not in ORIGINS:
            raise ValueError(f'origin must be one of {ORIGINS}')
        parents = tuple(_int('parent', p, 0, MAX_BUDGET) for p in _seq('parents', parents))
        expected = {'random': 0, 'mutation': 1, 'crossover': 2}[origin]
        if len(parents) != expected or len(set(parents)) != len(parents):
            raise ValueError(f'{origin} candidates need {expected} distinct parents')
        current = {e['id'] for e in self._grid.values()}
        if any(p not in current for p in parents):
            raise ValueError('parents must be current elites')
        operators = tuple(_seq('operators', operators))
        if len(operators) > _MAX_OPERATORS or any(not isinstance(o, str) or not 0 < len(o) <= 96 for o in operators):
            raise ValueError(f'operators must be at most {_MAX_OPERATORS} short strings')
        reasons = list(dict.fromkeys(_seq('reasons', reasons)))
        if any(r not in REJECTION_REASONS for r in reasons):
            raise ValueError(f'reasons must come from {REJECTION_REASONS}')
        if type(spot_checked) is not bool:
            raise ValueError('spot_checked must be a boolean')
        if error is not None and not isinstance(error, str):
            raise ValueError('error must be a string or None')
        node_id = len(self._lineage)
        digest = receipt.get('digest') if isinstance(receipt, Mapping) else None
        node: dict[str, Any] = {
            'id': node_id, 'genome': genome.to_dict(), 'origin': origin, 'parents': list(parents),
            'operators': list(operators), 'status': 'rejected', 'reasons': reasons,
            'error': None if error is None else error[:_MAX_TEXT], 'cell': None, 'fitness': None, 'log_loss': None,
            'descriptors': None, 'digest': digest if _is_hex64(digest) else None, 'spot_checked': spot_checked,
            'displaced_by': None}
        if reasons:
            self._lineage.append(node)
            self._rejections.append({k: _copy(node[k]) for k in ('id', 'genome', 'reasons', 'error', 'digest')})
            return _copy(node)
        if not harness.verify_receipt(receipt):
            raise ValueError('a candidate without rejection reasons must carry a verifying receipt')
        m = candidate_metrics(receipt)
        descriptors = {k: m[k] for k in DESCRIPTOR_KEYS}
        cell = self.cell_of(descriptors)
        incumbent = self._grid.get(cell)
        better = incumbent is None or _quality(m) > _quality(incumbent)
        stored = _strip_wallclock(receipt)          # wall-clock is outside the digest: keeps exports deterministic
        stored['digest'] = receipt['digest']
        node.update({'status': 'elite' if better else 'not_elite', 'cell': list(cell), 'fitness': m['fitness'],
                     'log_loss': m['log_loss'], 'descriptors': descriptors})
        # ---- commit (nothing above has changed the archive)
        self._lineage.append(node)
        for p in parents:
            self._children[p] = self._children.get(p, 0) + 1
        if better:
            if incumbent is not None:
                self._lineage[incumbent['id']]['displaced_by'] = node_id
            self._grid[cell] = {'cell': list(cell), 'id': node_id, 'genome': genome.to_dict(), 'fitness': m['fitness'],
                                'log_loss': m['log_loss'], 'descriptors': dict(descriptors), 'digest': node['digest'],
                                'receipt': stored}
        return _copy(node)

    # ------------------------------------------------------------------ persistence
    def _payload(self) -> dict[str, Any]:
        return {'grid': self.grid_config(),
                'elites': [_copy(self._grid[c]) for c in sorted(self._grid)],
                'lineage': _copy(self._lineage), 'rejections': _copy(self._rejections)}

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical archive payload (the integrity hash written by :meth:`export_json`)."""
        return _sha256(self._payload())

    def export_json(self) -> str:
        payload = self._payload()
        return _canonical({'schema': ARCHIVE_SCHEMA, 'schema_version': 1, 'sha256': _sha256(payload),
                           'payload': payload})

    @classmethod
    def from_json(cls, text: str) -> 'EliteArchive':
        grid = _parse_archive(text)['grid']
        archive = cls(cost_bins=grid['cost_bins'], cost_range=grid['cost_range'], activity_bins=grid['activity_bins'],
                      activity_range=grid['activity_range'], max_nodes=grid['max_nodes'])
        archive.import_json(text)
        return archive

    def import_json(self, text: str) -> None:
        """Replace this archive's contents with a validated export. Everything is checked first -- the SHA-256
        integrity hash, the schema, every elite receipt's seal and metrics, and a full replay of the lineage that
        must reproduce each node's status, the grid and the rejection log -- and nothing changes on failure."""
        payload = _parse_archive(text)
        if payload['grid'] != self.grid_config():
            raise ValueError('archive grid configuration does not match this archive')
        lineage = payload['lineage']
        if not isinstance(lineage, list) or len(lineage) > self.max_nodes:
            raise ValueError('lineage must be a list of at most max_nodes nodes')
        grid: dict[tuple[int, int], dict[str, Any]] = {}       # replayed: cell -> lineage node
        children: dict[int, int] = {}
        displaced: dict[int, int] = {}
        for i, node in enumerate(lineage):
            self._check_node(node, i, {e['id'] for e in grid.values()})
            for p in node['parents']:
                if node['status'] != 'rejected':
                    children[p] = children.get(p, 0) + 1
            if node['status'] == 'rejected':
                continue
            cell = tuple(node['cell'])
            incumbent = grid.get(cell)
            better = incumbent is None or _quality(node) > _quality(incumbent)
            if better != (node['status'] == 'elite'):
                raise ValueError(f'lineage node {i} status contradicts the best-per-cell replay')
            if better:
                if incumbent is not None:
                    displaced[incumbent['id']] = i
                grid[cell] = node
        for i, node in enumerate(lineage):
            if node['displaced_by'] != displaced.get(i):
                raise ValueError(f'lineage node {i} displaced_by contradicts the replay')
        elites = payload['elites']
        if not isinstance(elites, list) or len(elites) != len(grid):
            raise ValueError('elites do not match the replayed grid')
        new_grid: dict[tuple[int, int], dict[str, Any]] = {}
        for e in elites:
            if not isinstance(e, Mapping) or set(e) != _ELITE_KEYS:
                raise ValueError('malformed elite entry')
            raw_cell = e['cell']
            cell = (tuple(raw_cell) if isinstance(raw_cell, list) and all(type(x) is int for x in raw_cell)
                    else None)
            node = grid.get(cell)
            if node is None or cell in new_grid or e['id'] != node['id']:
                raise ValueError('elite entry does not match the replayed grid')
            for k in ('genome', 'fitness', 'log_loss', 'descriptors', 'digest'):
                if e[k] != node[k]:
                    raise ValueError(f'elite {e["id"]} {k} differs from its lineage node')
            receipt = e['receipt']
            if not harness.verify_receipt(receipt) or receipt.get('digest') != e['digest']:
                raise ValueError(f'elite {e["id"]} receipt does not verify')
            m = candidate_metrics(receipt)
            if (m['fitness'] != e['fitness'] or m['log_loss'] != e['log_loss']
                    or {k: m[k] for k in DESCRIPTOR_KEYS} != e['descriptors']
                    or receipt.get('config', {}).get('genome') != e['genome']):
                raise ValueError(f'elite {e["id"]} entry differs from its receipt')
            new_grid[cell] = _copy(dict(e))
        expected_rejections = [{k: node[k] for k in ('id', 'genome', 'reasons', 'error', 'digest')}
                               for node in lineage if node['status'] == 'rejected']
        if payload['rejections'] != expected_rejections:
            raise ValueError('rejection log does not match the lineage')
        # ---- commit
        self._grid = new_grid
        self._lineage = _copy(lineage)
        self._rejections = _copy(expected_rejections)
        self._children = children

    def _check_node(self, node: Any, i: int, current_elites: set[int]) -> None:
        if not isinstance(node, Mapping) or set(node) != _NODE_KEYS or type(node['id']) is not int or node['id'] != i:
            raise ValueError(f'malformed lineage node {i}')
        Genome.from_dict(node['genome'])
        if node['origin'] not in ORIGINS or node['status'] not in STATUSES or type(node['spot_checked']) is not bool:
            raise ValueError(f'lineage node {i} has an invalid origin, status or spot_checked flag')
        parents = node['parents']
        expected = {'random': 0, 'mutation': 1, 'crossover': 2}[node['origin']]
        if (not isinstance(parents, list) or len(parents) != expected or len(set(parents)) != len(parents)
                or any(type(p) is not int or p not in current_elites for p in parents)):
            raise ValueError(f'lineage node {i} parents must be elites at the time it was created')
        ops = node['operators']
        if (not isinstance(ops, list) or len(ops) > _MAX_OPERATORS
                or any(not isinstance(o, str) or not 0 < len(o) <= 96 for o in ops)):
            raise ValueError(f'lineage node {i} has invalid operators')
        if node['error'] is not None and (not isinstance(node['error'], str) or len(node['error']) > _MAX_TEXT):
            raise ValueError(f'lineage node {i} has an invalid error message')
        if node['displaced_by'] is not None and (type(node['displaced_by']) is not int or node['displaced_by'] <= i):
            raise ValueError(f'lineage node {i} has an invalid displaced_by')
        reasons = node['reasons']
        if not isinstance(reasons, list) or any(r not in REJECTION_REASONS for r in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError(f'lineage node {i} has invalid reasons')
        if node['status'] == 'rejected':
            if not reasons or any(node[k] is not None for k in ('cell', 'fitness', 'log_loss', 'descriptors')):
                raise ValueError(f'rejected lineage node {i} must have reasons and no metrics')
            if node['digest'] is not None and not _is_hex64(node['digest']):
                raise ValueError(f'lineage node {i} has an invalid digest')
            return
        if reasons or not _is_hex64(node['digest']):
            raise ValueError(f'verified lineage node {i} must have a digest and no reasons')
        _num('fitness', node['fitness'], 0.0, 1.0)
        _num('log_loss', node['log_loss'], 0.0, 1e6)
        cell = node['cell']
        if (not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell)
                or tuple(cell) != self.cell_of(node['descriptors'])):
            raise ValueError(f'lineage node {i} cell does not match its descriptors')


def _copy(obj: Any) -> Any:
    """Deep copy of JSON-like data (dicts, lists, scalars)."""
    if isinstance(obj, dict):
        return {k: _copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy(v) for v in obj]
    return obj


def _parse_archive(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or len(text) > MAX_ARCHIVE_CHARS:
        raise ValueError(f'archive must be a JSON string of at most {MAX_ARCHIVE_CHARS} characters')
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise ValueError('archive is not valid JSON') from exc
    if not isinstance(doc, dict) or set(doc) != {'schema', 'schema_version', 'sha256', 'payload'}:
        raise ValueError('archive must have schema, schema_version, sha256 and payload')
    if doc['schema'] != ARCHIVE_SCHEMA or doc['schema_version'] != 1:
        raise ValueError('unsupported archive schema')
    payload = doc['payload']
    if not isinstance(payload, dict) or doc['sha256'] != _sha256(payload):
        raise ValueError('archive integrity hash mismatch')
    if set(payload) != {'grid', 'elites', 'lineage', 'rejections'} or not isinstance(payload['grid'], dict):
        raise ValueError('archive payload has unexpected keys')
    if set(payload['grid']) != {'cost_bins', 'cost_range', 'activity_bins', 'activity_range', 'max_nodes'}:
        raise ValueError('archive grid configuration has unexpected keys')
    return payload


# ====================================================================== the search
class QualityDiversitySearch:
    """Verifier-gated MAP-Elites (``method='qd'``) or random search (``method='random'``) at a fixed budget.

    Both methods insert into the same kind of archive through the same gate, so they differ only in how
    candidates are proposed. The first ``initial_random`` genomes of both come from the same seeded stream.
    ``spot_check_every = k`` re-evaluates candidates number k, 2k, 3k, ... (evaluation order) when they passed the
    static checks and requires the same receipt digest (``0`` disables spot checks; ablation only). Evaluator
    exceptions become rejections.

    ``cache`` (optional dict, shared between searches) memoises first evaluations by (protocol, seed, genome),
    so two arms that propose the same candidate run it once; every candidate still costs one unit of budget.
    Spot checks and :meth:`audit` always call the evaluator itself, never the cache."""

    def __init__(self, evaluator: Callable[[Genome, int], Mapping[str, Any]], *, seed: int = 0, budget: int = 32,
                 method: str = 'qd', space: GenomeSpace | None = None, initial_random: int = 8,
                 selection: str = 'dgm', crossover_rate: float = 0.25, extra_mutation_prob: float = 0.25,
                 spot_check_every: int = 4, selection_sharpness: float = 10.0, selection_midpoint: float = 0.5,
                 max_resample: int = 16, cost_bins: int = 4, cost_range: Sequence[float] = (9.75, 11.75),
                 activity_bins: int = 4, activity_range: Sequence[float] = (0.0, 0.8),
                 cache: dict | None = None) -> None:
        if not callable(evaluator) or not callable(getattr(evaluator, 'protocol', None)):
            raise ValueError('evaluator must be callable as evaluator(genome, seed) and provide protocol()')
        protocol = evaluator.protocol()
        if not isinstance(protocol, Mapping) or type(protocol.get('steps')) is not int:
            raise ValueError('evaluator.protocol() must be a mapping with an int steps')
        if cache is not None and not isinstance(cache, dict):
            raise ValueError('cache must be a dict or None')
        self.evaluator = evaluator
        self.protocol = _jsonable(dict(protocol))
        self._cache = cache
        self._cache_prefix = _sha256(self.protocol)
        self.cache_hits = 0
        self.evaluator_calls = 0
        self.space = GenomeSpace() if space is None else space
        if not isinstance(self.space, GenomeSpace):
            raise ValueError('space must be a GenomeSpace')
        self.seed = _int('seed', seed, 0, 2**31 - 1)
        self.budget = _int('budget', budget, 1, MAX_BUDGET)
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        if selection not in SELECTIONS:
            raise ValueError(f'selection must be one of {SELECTIONS}')
        self.method, self.selection = method, selection
        self.initial_random = _int('initial_random', initial_random, 1, self.budget)
        self.crossover_rate = _num('crossover_rate', crossover_rate, 0.0, 1.0)
        self.extra_mutation_prob = _num('extra_mutation_prob', extra_mutation_prob, 0.0, 0.95)
        self.spot_check_every = _int('spot_check_every', spot_check_every, 0, self.budget)
        self.selection_sharpness = _num('selection_sharpness', selection_sharpness, 0.0, 100.0)
        self.selection_midpoint = _num('selection_midpoint', selection_midpoint, 0.0, 1.0)
        self.max_resample = _int('max_resample', max_resample, 1, 1000)
        self.archive = EliteArchive(cost_bins=cost_bins, cost_range=cost_range, activity_bins=activity_bins,
                                    activity_range=activity_range, max_nodes=self.budget)
        self._init_rng = np.random.default_rng(self.seed + _INIT_OFFSET)
        self._var_rng = np.random.default_rng(self.seed + _VAR_OFFSET)
        self._seen: set[str] = set()
        self.spot_checks = 0
        self.estimated_ops_verified = 0.0
        self.measured_wallclock_s = 0.0
        self._qd_trajectory: list[float] = []
        self._coverage_trajectory: list[float] = []

    def config(self) -> dict[str, Any]:
        return {'method': self.method, 'seed': self.seed, 'budget': self.budget, 'initial_random': self.initial_random,
                'selection': self.selection, 'crossover_rate': self.crossover_rate,
                'extra_mutation_prob': self.extra_mutation_prob, 'spot_check_every': self.spot_check_every,
                'selection_sharpness': self.selection_sharpness, 'selection_midpoint': self.selection_midpoint,
                'max_resample': self.max_resample, 'grid': self.archive.grid_config(), 'space': self.space.to_dict()}

    @property
    def done(self) -> bool:
        return self.archive.evaluations >= self.budget

    # ------------------------------------------------------------------ proposal
    def _random_candidate(self) -> tuple[Genome, str, tuple[int, ...], tuple[str, ...]]:
        g = self.space.random_genome(self._init_rng)
        for _ in range(self.max_resample - 1):
            if g.key not in self._seen:
                break
            g = self.space.random_genome(self._init_rng)
        return g, 'random', (), ()

    def parent_probabilities(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Current elites (by cell) and the parent distribution the next variation step would use."""
        elites = self.archive.elites()
        if not elites:
            raise ValueError('the archive is empty')
        if self.selection == 'uniform':
            return elites, np.full(len(elites), 1.0 / len(elites))
        return elites, dgm_parent_probabilities([e['fitness'] for e in elites],
                                                [self.archive.children_count(e['id']) for e in elites],
                                                sharpness=self.selection_sharpness, midpoint=self.selection_midpoint)

    def _varied_candidate(self) -> tuple[Genome, str, tuple[int, ...], tuple[str, ...]]:
        elites, p = self.parent_probabilities()
        rng = self._var_rng
        for _ in range(self.max_resample):
            i, j = (int(v) for v in rng.choice(len(elites), size=2, replace=True, p=p))
            a, b = elites[i], elites[j]
            ga = Genome.from_dict(a['genome'])
            if rng.random() < self.crossover_rate and a['id'] != b['id']:
                child = crossover(ga, Genome.from_dict(b['genome']), self.space, rng)
                child, ops = mutate(child, self.space, rng, extra_prob=self.extra_mutation_prob)
                cand = (child, 'crossover', (a['id'], b['id']), ('crossover',) + ops)
            else:
                child, ops = mutate(ga, self.space, rng, extra_prob=self.extra_mutation_prob)
                cand = (child, 'mutation', (a['id'],), ops)
            if child.key not in self._seen:
                break
        return cand

    # ------------------------------------------------------------------ one gated evaluation
    def _evaluate(self, genome: Genome) -> Any:
        """First evaluation of a candidate, through the shared cache when one is given (copies in and out)."""
        key = (self._cache_prefix, self.seed, genome.key)
        if self._cache is not None and key in self._cache:
            self.cache_hits += 1
            return _copy(self._cache[key])
        self.evaluator_calls += 1
        receipt = self.evaluator(genome, self.seed)
        if self._cache is not None and isinstance(receipt, dict) and len(self._cache) < MAX_BUDGET:
            self._cache[key] = _copy(receipt)
        return receipt

    def step(self) -> dict[str, Any]:
        """Propose, evaluate, verify (with a spot check when due) and record one candidate."""
        if self.done:
            raise ValueError('the evaluation budget is exhausted')
        index = self.archive.evaluations
        if self.method == 'random' or index < self.initial_random or len(self.archive) == 0:
            genome, origin, parents, ops = self._random_candidate()
        else:
            genome, origin, parents, ops = self._varied_candidate()
        self._seen.add(genome.key)
        t0 = time.perf_counter()
        error = None
        try:
            receipt: Any = self._evaluate(genome)
        except Exception as exc:                      # an evaluator failure is a rejection, never a crash
            receipt, reasons, error = None, ['evaluation_error'], f'{type(exc).__name__}: {exc}'
        else:
            reasons = verify_candidate(genome, receipt, seed=self.seed, protocol=self.protocol, space=self.space)
        spot = False
        if not reasons and self.spot_check_every and (index + 1) % self.spot_check_every == 0:
            spot = True
            self.spot_checks += 1
            self.evaluator_calls += 1
            try:
                again = self.evaluator(genome, self.seed)        # never the cache: an independent re-run
            except Exception as exc:
                reasons, error = ['spot_check_error'], f'{type(exc).__name__}: {exc}'
            else:
                if (not isinstance(again, Mapping) or again.get('digest') != receipt.get('digest')
                        or not harness.verify_receipt(again)):
                    reasons = ['spot_check_digest_mismatch']
        self.measured_wallclock_s += time.perf_counter() - t0
        node = self.archive.record(genome=genome, origin=origin, parents=parents, operators=ops, receipt=receipt,
                                   reasons=reasons, spot_checked=spot, error=error)
        if node['status'] != 'rejected':
            self.estimated_ops_verified += float(receipt['estimated_cost']['total_ops'])
        self._qd_trajectory.append(self.archive.qd_score())
        self._coverage_trajectory.append(self.archive.coverage())
        return node

    def run(self) -> dict[str, Any]:
        while not self.done:
            self.step()
        return self.summary()

    def audit(self) -> dict[str, Any]:
        """Re-evaluate every current elite and check that its receipt digest reproduces (read-only)."""
        rows = []
        for e in self.archive.elites():
            try:
                again = self.evaluator(Genome.from_dict(e['genome']), self.seed)
                ok = isinstance(again, Mapping) and again.get('digest') == e['digest'] and harness.verify_receipt(again)
            except Exception:
                ok = False
            rows.append({'id': e['id'], 'cell': e['cell'], 'reproduced': bool(ok)})
        return {'checked': len(rows), 'reproduced': sum(r['reproduced'] for r in rows),
                'mismatched_ids': [r['id'] for r in rows if not r['reproduced']]}

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict[str, Any]:
        lineage = self.archive.lineage()
        depth: list[int] = []
        for node in lineage:
            depth.append(1 + max(depth[p] for p in node['parents']) if node['parents'] else 0)
        reasons = Counter(r for node in lineage for r in node['reasons'])
        status = Counter(node['status'] for node in lineage)
        best = self.archive.best()
        if best is not None:            # 'receipt_digest', not 'digest': receipt seals skip keys named 'digest'
            best['receipt_digest'] = best.pop('digest')
        fits = [node['fitness'] for node in lineage if node['status'] != 'rejected']
        return {
            'method': self.method, 'seed': self.seed, 'evaluations': len(lineage), 'spot_checks': self.spot_checks,
            'evaluator_calls': self.evaluator_calls, 'cache_hits': self.cache_hits,
            'admitted_as_elite': status['elite'], 'verified_not_elite': status['not_elite'],
            'rejected': status['rejected'], 'displaced': sum(n['displaced_by'] is not None for n in lineage),
            'rejections_by_reason': dict(sorted(reasons.items())),
            'qd_score': self.archive.qd_score(), 'coverage': self.archive.coverage(), 'filled_cells': len(self.archive),
            'total_cells': self.archive.total_cells, 'best_fitness': None if best is None else best['fitness'],
            'verified_fitness': None if not fits else {'min': min(fits), 'max': max(fits),
                                                       'mean': float(np.mean(fits)), 'distinct_values': len(set(fits))},
            'best': best, 'elites': [{**{k: e[k] for k in ('cell', 'id', 'fitness', 'log_loss', 'genome')},
                                      'receipt_digest': e['digest']} for e in self.archive.elites()],
            'pareto_front': self.archive.pareto_front(),
            'qd_score_trajectory': list(self._qd_trajectory), 'coverage_trajectory': list(self._coverage_trajectory),
            'lineage': {'nodes': len(lineage), 'max_depth': max(depth) if depth else 0,
                        'origins': {o: sum(n['origin'] == o for n in lineage) for o in ORIGINS}},
            'archive_sha256': self.archive.sha256, 'estimated_ops_verified_candidates': self.estimated_ops_verified,
            'measured_wallclock_s': self.measured_wallclock_s}


# ====================================================================== illumination metrics
def illumination_metrics(maps: Mapping[str, Mapping[Any, float]]) -> dict[str, dict[str, float | None]]:
    """Mouret & Clune (2015, Sec. 9.4) metrics of several runs' maps (cell -> best fitness, fitness >= 0) over the
    cells any of them filled: global reliability (mean over those cells of own / best, 0 where unfilled),
    precision (the same mean over own filled cells) and coverage (own filled / union). Where the best value
    is 0 an own value of 0 counts as ratio 1."""
    if not isinstance(maps, Mapping) or not maps:
        raise ValueError('maps must be a non-empty mapping of run name -> {cell: fitness}')
    clean: dict[str, dict[tuple, float]] = {}
    for name, m in maps.items():
        if not isinstance(m, Mapping):
            raise ValueError('each map must be a mapping of cell -> fitness')
        clean[str(name)] = {tuple(c) if isinstance(c, (list, tuple)) else (c,): _num('fitness', v, 0.0, 1e12)
                            for c, v in m.items()}
    union = set().union(*(m.keys() for m in clean.values()))
    best = {c: max(m.get(c, 0.0) for m in clean.values()) for c in union}
    out: dict[str, dict[str, float | None]] = {}
    for name, m in clean.items():
        ratios = [m[c] / best[c] if best[c] > 0.0 else 1.0 for c in m]
        out[name] = {'reliability': float(sum(ratios) / len(union)) if union else None,
                     'precision': float(np.mean(ratios)) if ratios else None,
                     'coverage_of_union': float(len(m) / len(union)) if union else None}
    return out


# ====================================================================== lab experiment
# delayed_recall is the default task: in a 2026-11-20 scan of 10 random genomes of the default space (genomes from
# default_rng(0), seed 0, 48 training steps, held-out seeds 100 and 101 x 24 steps) it gave the most distinct held-out
# accuracies (6 values, sd 0.075; xor 4, parity 3, delayed_recall_6 2) while rhythm and context_switch gave every
# genome the same accuracy. Its best values were still close to the majority-class rate (0.604 vs 0.583).
_DEFAULTS: dict[str, Any] = {
    'task': 'delayed_recall', 'steps': 48, 'heldout_steps': 24, 'heldout_seeds': (100, 101),
    'test_seeds': (200, 201), 'budget': 16, 'initial_random': 6, 'selection': 'dgm', 'crossover_rate': 0.25,
    'extra_mutation_prob': 0.25, 'spot_check_every': 4, 'selection_sharpness': 10.0, 'selection_midpoint': 0.5,
    'max_resample': 16, 'mechanisms': SEARCH_MECHANISMS, 'topologies': TOPOLOGIES, 'neurons': NEURON_CHOICES,
    'sparsity': SPARSITY_CHOICES, 'cost_bins': 4, 'cost_range': (10.0, 12.0), 'activity_bins': 4,
    'activity_range': (0.0, 0.8)}
_QUICK: dict[str, Any] = {'steps': 16, 'heldout_steps': 8, 'heldout_seeds': (100,), 'test_seeds': (200,),
                          'budget': 5, 'initial_random': 3, 'spot_check_every': 2, 'neurons': (16,),
                          'cost_range': (10.1, 10.8)}
_SEARCH_KEYS = ('budget', 'initial_random', 'selection', 'crossover_rate', 'extra_mutation_prob', 'spot_check_every',
                'selection_sharpness', 'selection_midpoint', 'max_resample', 'cost_bins', 'cost_range',
                'activity_bins', 'activity_range')
_ARMS = (('quality_diversity', 'qd'), ('random_search', 'random'))
_METRICS = ('qd_score', 'coverage', 'best_fitness', 'best_test_accuracy', 'reliability', 'precision',
            'coverage_of_union')


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Verifier-gated MAP-Elites with DGM-style parent selection vs random search at an equal evaluation budget.

    Per seed both arms search the same space with the same evaluator (NEUROWEAVE ``task``, training stream
    ``seed``, held-out streams ``heldout_seeds``), the same grid, the same admission gate and the same spot-check
    schedule; their first ``initial_random`` candidates are identical (common random numbers, evaluated once via
    a per-seed cache). Each arm's best elite is re-evaluated on fresh ``test_seeds`` streams. The verdict is
    ``harness.compare`` of QD-score paired over seeds. ``quick=True`` shrinks the run (16 neurons, 16 + 8 steps,
    5 candidates per arm, every 2nd one spot-checked); ``seeds`` are always used as given."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seed_tuple('seeds', seeds)
    unknown = sorted(set(overrides) - set(_DEFAULTS))
    if unknown:
        raise ValueError(f'unknown run_experiment parameters: {unknown}')
    cfg = {**_DEFAULTS, **(_QUICK if quick else {}), **overrides}
    space = GenomeSpace(mechanisms=cfg['mechanisms'], topologies=cfg['topologies'], neurons=cfg['neurons'],
                        sparsity=cfg['sparsity'])
    evaluator = NeuroWeaveEvaluator(task=cfg['task'], steps=cfg['steps'], heldout_seeds=cfg['heldout_seeds'],
                                    heldout_steps=cfg['heldout_steps'])
    test_seeds = _seed_tuple('test_seeds', cfg['test_seeds'])
    held = evaluator.heldout_seeds
    if set(seed_list) & set(held) or set(seed_list) & set(test_seeds) or set(held) & set(test_seeds):
        raise ValueError('seeds, heldout_seeds and test_seeds must be pairwise disjoint')
    search_kw = {k: cfg[k] for k in _SEARCH_KEYS}
    QualityDiversitySearch(evaluator, seed=seed_list[0], space=space, **search_kw)     # validate before running
    arms: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _ARMS}
    t0 = time.perf_counter()
    for s in seed_list:
        cache: dict = {}                     # per seed: identical candidates of the two arms are evaluated once
        tests: dict[str, dict[str, Any]] = {}
        maps = {}
        for name, method in _ARMS:
            search = QualityDiversitySearch(evaluator, seed=s, method=method, space=space, cache=cache, **search_kw)
            row = search.run()
            best = search.archive.best()
            if best is None:
                row['best_test'] = None
            else:
                key = _canonical(best['genome'])
                if key not in tests:
                    test = evaluator.evaluate(Genome.from_dict(best['genome']), s, heldout_seeds=test_seeds)
                    tests[key] = {**{k: test['heldout'][k] for k in ('accuracy', 'log_loss', 'chance_accuracy')},
                                  'receipt_digest': test['digest']}
                row['best_test'] = dict(tests[key])
            row['best_test_accuracy'] = None if best is None else row['best_test']['accuracy']
            maps[name] = {tuple(e['cell']): e['fitness'] for e in search.archive.elites()}
            arms[name].append(row)
        illum = illumination_metrics(maps)
        for name, _ in _ARMS:
            arms[name][-1].update(illum[name])
    wall = time.perf_counter() - t0

    vals = {name: {m: [row[m] for row in rows] for m in _METRICS} for name, rows in arms.items()}
    comparisons = {m: harness.compare(vals['quality_diversity'][m], vals['random_search'][m]) for m in _METRICS}
    receipt = harness.base_receipt(MODULE_ID, 'search_comparison', {
        **{k: cfg[k] for k in ('task', 'steps', 'heldout_steps')}, 'seeds': list(seed_list),
        'heldout_seeds': list(held), 'test_seeds': list(test_seeds), 'quick': quick, **search_kw,
        'space': space.to_dict(), 'space_size': space.size, 'evaluator': evaluator.protocol(),
        'arms': {name: method for name, method in _ARMS},
        'fitness': 'mean held-out accuracy (tie-break: lower held-out log loss)',
        'descriptors': 'log2(estimated total_ops / training steps), mean Hoyer sparseness of held-out outputs',
        'admission': 'receipt seal + schema + genome + protocol + spec validity/equality + metric consistency; '
                     'every spot_check_every-th verified candidate is re-evaluated and must reproduce its digest',
        'parent_selection': 'DGM: p_i ~ sigmoid(selection_sharpness (f_i - selection_midpoint)) / (1 + n_i) over '
                            'elites with f_i < 1; two draws with replacement (crossover when distinct)'})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['arms'] = {name: {'per_seed': rows, 'summary': {m: _mean(vals[name][m]) for m in _METRICS}}
                       for name, rows in arms.items()}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['qd_score']['verdict']
    receipt['verdict_basis'] = 'qd_score_quality_diversity_vs_random_search_equal_candidate_evaluation_budget'
    receipt['budget'] = {
        'candidate_evaluations_per_arm_per_seed': int(cfg['budget']),
        'spot_check_evaluations': {name: [row['spot_checks'] for row in rows] for name, rows in arms.items()},
        'evaluator_calls': {name: [row['evaluator_calls'] for row in rows] for name, rows in arms.items()},
        'cache_hits': {name: [row['cache_hits'] for row in rows] for name, rows in arms.items()},
        'test_evaluations_per_arm_per_seed': 1, 'search_space_size': space.size,
        'grid_cells': int(cfg['cost_bins']) * int(cfg['activity_bins']), 'trainable_parameters_searched': 0,
        'capacity_note': 'both arms evaluate the same number of candidates under the same gate and spot-check '
                         'schedule; the shared cache only avoids re-running identical candidates and does not '
                         'change any result; neither arm adds parameters to the evaluated engines'}
    receipt['estimated_cost'] = {name: {'ops_verified_candidates': [row['estimated_ops_verified_candidates']
                                                                    for row in rows]}
                                 for name, rows in arms.items()}
    receipt['measured_wallclock_s'] = wall
    receipt['notes'] = (
        'Synthetic search over NEUROWEAVE switches at tiny sizes. Fitness is held-out accuracy on one short '
        'temporal stream: it is coarse (1 / heldout_steps), often near chance and noisy, so the archive records '
        'coverage plus selection on noise as much as real quality (see verified_fitness per arm). best_fitness '
        "is selected on the held-out stream and is optimistic, which is why each arm's best elite is re-scored on "
        'fresh test streams (best_test). Descriptor ranges are fixed in the config; values outside them join the '
        'edge cells. Receipts and archive exports are SHA-256 sealed, which detects accidental or naive tampering '
        'but is not a signature: only the periodic re-evaluation spot check tests that a result reproduces. '
        'MAP-Elites and the DGM parent rule are reused as published ideas; nothing here reproduces either '
        "paper's systems or results.")
    return harness.seal(receipt)


EXPERIMENTS = {MODULE_ID: run_experiment}
