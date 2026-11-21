"""Frontier module ``quality_diversity``: verifier-gated MAP-Elites over NEUROWEAVE configurations."""
import copy
import hashlib
import itertools
import json
import math
import random
import sys

import numpy as np
import pytest

from synapse.frontier import harness
from synapse.frontier import quality_diversity as Q
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, LAB_SCHEMA, MODULE_IDS, OBSERVER_MODULE_IDS
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import _state_digest, _task_arm, make_spec
from synapse.neuroweave.spec import MECHANISM_IDS, TOPOLOGIES
from synapse.neuroweave.tasks import make_stream

SMALL = Q.GenomeSpace(mechanisms=('sparse_code', 'reward_plasticity', 'synaptic_delays', 'membrane_dynamics',
                                  'context_attention', 'flyhash_projection'),
                      topologies=('clustered', 'ring_lattice', 'erdos_renyi'), neurons=(16, 32), sparsity=(0.05, 0.3))
TARGET = Q.Genome(('reward_plasticity', 'membrane_dynamics', 'flyhash_projection'), 'ring_lattice', 32, 0.3)
TINY_NW = {'task': 'delayed_recall', 'steps': 16, 'heldout_seeds': (100, 101), 'heldout_steps': 8}


def _landscape(target=TARGET, space=SMALL):
    return Q.LandscapeEvaluator(target, space)


def _search(evaluator=None, **kw):
    args = {'seed': 0, 'budget': 40, 'space': SMALL, 'initial_random': 8, 'spot_check_every': 5}
    args.update(kw)
    return Q.QualityDiversitySearch(_landscape() if evaluator is None else evaluator, **args)


def _global_state():
    torch = sys.modules.get('torch')
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return (kind, keys.tobytes(), pos, has_gauss, cached, random.getstate(),
            None if torch is None else (torch.get_rng_state().numpy().tobytes(), torch.get_num_threads()))


def _reseal(receipt, edit):
    r = copy.deepcopy(receipt)
    edit(r)
    return harness.seal(r)


class _Wrapped:
    """Evaluator wrapper: ``edit(receipt, genome, n)`` (``n`` = this genome's call number) may alter, replace or
    refuse the inner evaluator's result."""

    def __init__(self, edit, inner=None):
        self.inner = _landscape() if inner is None else inner
        self.edit = edit
        self.calls = 0
        self.per_genome = {}

    def protocol(self):
        return self.inner.protocol()

    def __call__(self, genome, seed):
        self.calls += 1
        n = self.per_genome[genome.key] = self.per_genome.get(genome.key, 0) + 1
        return self.edit(self.inner(genome, seed), genome, n)


def _inflate(r):
    r['heldout']['accuracy'] = r['fitness'] = 1.0


def _replay(genome, trail):
    """Re-apply a logged mutation trail (toggle / swap / topology / neurons / sparsity) to ``genome``."""
    g = genome
    for entry in trail:
        op, _, arg = entry.partition(':')
        mechs = set(g.mechanisms)
        if op == 'toggle_mechanism':
            assert (arg[1:] in mechs) == (arg[0] == '-')
            g = Q.Genome(tuple(mechs ^ {arg[1:]}), g.topology, g.neurons, g.sparsity)
        elif op == 'swap_mechanism':
            out_m, in_m = arg[1:].split('+')
            assert out_m in mechs and in_m not in mechs
            g = Q.Genome(tuple((mechs - {out_m}) | {in_m}), g.topology, g.neurons, g.sparsity)
        elif op == 'change_topology':
            old, new = arg.split('->')
            assert old == g.topology != new
            g = Q.Genome(g.mechanisms, new, g.neurons, g.sparsity)
        elif op == 'step_neurons':
            old, new = (int(v) for v in arg.split('->'))
            assert old == g.neurons
            g = Q.Genome(g.mechanisms, g.topology, new, g.sparsity)
        else:
            assert op == 'step_sparsity'
            old, new = (float(v) for v in arg.split('->'))
            assert old == g.sparsity
            g = Q.Genome(g.mechanisms, g.topology, g.neurons, new)
    return g


def _assert_best_per_cell(archive):
    """Each cell's elite is the verified candidate with the highest (fitness, -log_loss); earliest on ties."""
    best = {}
    for node in archive.lineage():
        if node['status'] == 'rejected':
            continue
        cell, q = tuple(node['cell']), (node['fitness'], -node['log_loss'])
        if cell not in best or q > best[cell][0]:
            best[cell] = (q, node['id'])
    elites = archive.elites()
    assert {tuple(e['cell']): e['id'] for e in elites} == {c: v[1] for c, v in best.items()}
    for e in elites:
        assert tuple(e['cell']) == archive.cell_of(e['descriptors'])
        receipt = archive.elite_receipt(e['cell'])
        assert verify_receipt(receipt) and receipt['digest'] == e['digest']
        assert Q.candidate_metrics(receipt)['fitness'] == e['fitness']
    assert archive.qd_score() == pytest.approx(sum(e['fitness'] for e in elites))
    assert archive.coverage() == len(elites) / archive.total_cells


# ====================================================================== registry, genome and search space
def test_module_is_registered_with_one_experiment_and_a_ten_mechanism_search_set():
    assert Q.MODULE_ID == 'quality_diversity' and Q.MODULE_ID in MODULE_IDS and Q.MODULE_ID not in OBSERVER_MODULE_IDS
    assert list(Q.EXPERIMENTS) == ['quality_diversity'] and Q.EXPERIMENTS['quality_diversity'] is Q.run_experiment
    assert len(Q.SEARCH_MECHANISMS) == 10 and set(Q.SEARCH_MECHANISMS) <= set(MECHANISM_IDS)
    space = Q.GenomeSpace()
    assert space.topologies == TOPOLOGIES and space.neurons == (16, 24, 32) and space.sparsity == (0.05, 0.15, 0.3, 0.5)
    assert space.size == 2 ** 10 * len(TOPOLOGIES) * 3 * 4
    assert 'arXiv:1504.04909' in Q.SOURCE and 'arXiv:2505.22954' in Q.SOURCE


def test_genome_is_canonical_round_trips_and_builds_the_lab_spec():
    g = Q.Genome(('membrane_dynamics', 'sparse_code'), 'hybrid', 24, 0.3)
    assert g.mechanisms == ('sparse_code', 'membrane_dynamics')            # NEUROWEAVE canonical order
    assert Q.Genome.from_dict(json.loads(json.dumps(g.to_dict()))) == g
    assert g.key == Q.Genome(('sparse_code', 'membrane_dynamics'), 'hybrid', 24, 0.3).key
    spec = g.to_spec(seed=5, steps=48)
    lab = make_spec(g.mechanisms, seed=5, neurons=24, topology='hybrid', steps=48)
    assert spec.sparsity == 0.3 and lab.sparsity == 0.15
    assert {k: v for k, v in spec.to_dict().items() if k != 'sparsity'} == \
        {k: v for k, v in lab.to_dict().items() if k != 'sparsity'}


@pytest.mark.parametrize('args', [
    (('not_a_mechanism',), 'clustered', 16, 0.15), (('sparse_code', 'sparse_code'), 'clustered', 16, 0.15),
    ('sparse_code', 'clustered', 16, 0.15), ((), 'torus', 16, 0.15), ((), 'clustered', 4, 0.15),
    ((), 'clustered', 16.0, 0.15), ((), 'clustered', True, 0.15), ((), 'clustered', 16, 0.0),
    ((), 'clustered', 16, float('nan')), ((), 'clustered', 16, '0.1')])
def test_genome_rejects_invalid_genes(args):
    with pytest.raises(ValueError):
        Q.Genome(*args)


@pytest.mark.parametrize('kwargs', [
    {'mechanisms': ('nope',)}, {'topologies': ()}, {'topologies': ('clustered', 'clustered')},
    {'neurons': ()}, {'neurons': (16, 16)}, {'sparsity': (0.0,)}, {'neurons': 16},
    {'mechanisms': (), 'topologies': ('clustered',), 'neurons': (16,), 'sparsity': (0.15,)}])
def test_genome_space_validation(kwargs):
    with pytest.raises(ValueError):
        Q.GenomeSpace(**kwargs)
    with pytest.raises(ValueError):
        Q.Genome.from_dict({'mechanisms': [], 'topology': 'clustered', 'neurons': 16})


def test_random_genomes_are_seeded_uniform_and_inside_the_space():
    space = Q.GenomeSpace()
    a = [space.random_genome(np.random.default_rng(9)) for _ in range(3)]
    assert a[0] == a[1] == a[2]
    rng = np.random.default_rng(1)
    draws = [space.random_genome(rng) for _ in range(400)]
    assert all(space.contains(g) for g in draws)
    assert {g.topology for g in draws} == set(TOPOLOGIES) and {g.neurons for g in draws} == {16, 24, 32}
    assert 4.0 < np.mean([len(g.mechanisms) for g in draws]) < 6.0          # each switch on with p = 1/2
    assert Q.GenomeSpace.from_dict(json.loads(json.dumps(space.to_dict()))) == space
    assert not space.contains(Q.Genome((), 'clustered', 64, 0.15)) and not space.contains(TARGET.to_dict())
    with pytest.raises(ValueError):
        space.random_genome(np.random.RandomState(0))


# ====================================================================== variation operators
def test_mutation_respects_preconditions_and_its_audit_trail_explains_the_child():
    full = Q.Genome(SMALL.mechanisms, 'clustered', 16, 0.05)
    empty = Q.Genome((), 'clustered', 16, 0.05)
    assert 'swap_mechanism' not in Q.applicable_operators(full, SMALL)
    assert 'swap_mechanism' not in Q.applicable_operators(empty, SMALL)
    assert Q.applicable_operators(TARGET, SMALL) == Q.MUTATION_OPERATORS
    narrow = Q.GenomeSpace(mechanisms=('sparse_code',), topologies=('clustered',), neurons=(16,), sparsity=(0.05, 0.3))
    assert Q.applicable_operators(Q.Genome((), 'clustered', 16, 0.05), narrow) == ('toggle_mechanism', 'step_sparsity')
    rng = np.random.default_rng(3)
    used = set()
    for parent in (full, empty, TARGET):
        for _ in range(150):
            child, trail = Q.mutate(parent, SMALL, rng)
            assert child != parent and SMALL.contains(child) and 1 <= len(trail) <= 4
            assert _replay(parent, trail) == child
            used.update(t.partition(':')[0] for t in trail)
            single, one = Q.mutate(parent, SMALL, rng, extra_prob=0.0)
            assert len(one) == 1 and single != parent and _replay(parent, one) == single
    assert used == set(Q.MUTATION_OPERATORS)
    for bad in ({'extra_prob': 1.0}, {'max_operators': 0}, {'max_operators': 7}):
        with pytest.raises(ValueError):
            Q.mutate(TARGET, SMALL, rng, **bad)
    with pytest.raises(ValueError):
        Q.mutate(Q.Genome((), 'hybrid', 16, 0.05), SMALL, rng)                 # outside the space


def test_crossover_takes_every_gene_from_one_of_the_parents():
    a = Q.Genome(('sparse_code', 'reward_plasticity'), 'clustered', 16, 0.05)
    b = Q.Genome(('membrane_dynamics', 'context_attention', 'flyhash_projection'), 'erdos_renyi', 32, 0.3)
    rng = np.random.default_rng(5)
    children = [Q.crossover(a, b, SMALL, rng) for _ in range(300)]
    for c in children:
        assert all((m in c.mechanisms) in {m in a.mechanisms, m in b.mechanisms} for m in SMALL.mechanisms)
        assert c.topology in (a.topology, b.topology) and SMALL.contains(c)
    assert len(set(children)) > 50 and 0.4 < np.mean([c.topology == a.topology for c in children]) < 0.6
    assert all(Q.crossover(a, a, SMALL, rng) == a for _ in range(10))


def test_dgm_parent_probabilities_follow_the_published_rule():
    f, n = [0.2, 0.5, 0.8, 0.8], [0, 1, 0, 3]
    w = [1.0 / (1.0 + math.exp(-10.0 * (x - 0.5))) / (1 + k) for x, k in zip(f, n)]
    p = Q.dgm_parent_probabilities(f, n)
    assert np.allclose(p, [x / sum(w) for x in w], rtol=0.0, atol=1e-12) and p.sum() == pytest.approx(1.0)
    assert p[2] > p[1] > p[0] and p[2] == pytest.approx(4.0 * p[3])        # fitter and less-explored first
    assert np.allclose(Q.dgm_parent_probabilities([1.0, 0.6, 1.0], [0, 5, 0]), [0.0, 1.0, 0.0])   # f = 1 not eligible
    assert np.allclose(Q.dgm_parent_probabilities([1.0, 1.0], [0, 1]), [2 / 3, 1 / 3])           # all perfect: fallback
    assert np.allclose(Q.dgm_parent_probabilities([0.3, 0.9], [2, 0], sharpness=0.0), [0.25, 0.75])
    for args, kw in [(([], []), {}), (([0.5], [1, 2]), {}), (([1.5], [0]), {}), (([math.nan], [0]), {}),
                     (([0.5], [-1]), {}), (([0.5], [0.5]), {}), (([0.5], [0]), {'sharpness': -1.0}),
                     (([0.5], [0]), {'midpoint': 2.0})]:
        with pytest.raises(ValueError):
            Q.dgm_parent_probabilities(*args, **kw)


# ====================================================================== activity sparseness descriptor
def test_hoyer_sparseness_known_values():
    assert Q.hoyer_sparseness([0.0, 0.0, 3.0, 0.0]) == pytest.approx(1.0)
    assert Q.hoyer_sparseness([-2.0, 2.0, 2.0, -2.0]) == pytest.approx(0.0, abs=1e-12)
    assert Q.hoyer_sparseness([1.0, 1.0, 0.0, 0.0]) == pytest.approx(2.0 - math.sqrt(2.0))
    x = np.random.default_rng(0).standard_normal(32)
    assert Q.hoyer_sparseness(x) == pytest.approx(Q.hoyer_sparseness(-7.5 * x))
    assert Q.hoyer_sparseness([1e-300, 0.0, 0.0]) == pytest.approx(1.0)                # no underflow
    assert Q.hoyer_sparseness(np.zeros(5)) is None
    for bad in ([1.0], [], [0.0, math.nan], [math.inf, 1.0]):
        with pytest.raises(ValueError):
            Q.hoyer_sparseness(bad)


def test_activity_recorder_never_changes_the_engine_trajectory():
    spec = make_spec(('sparse_code', 'membrane_dynamics'), seed=4, neurons=16, topology='clustered', steps=16)
    plain, metered = NeuroWeaveEngine(spec), NeuroWeaveEngine(spec)
    rec = Q.ActivityRecorder()
    metered.attach_extension(rec)
    stream = make_stream(4, 14)
    scores = []
    for i in range(14):
        kw = {'reward': 0.3 if i % 2 else -0.2, 'learn': i < 9}
        a = plain.step(stream.inputs[i], stream.sensory[i], **kw)['output']
        b = metered.step(stream.inputs[i], stream.sensory[i], **kw)['output']
        assert np.array_equal(a, b)
        scores.append(Q.hoyer_sparseness(b))
    assert _state_digest(plain) == _state_digest(metered)
    st = rec.stats()
    assert st['scored_steps'] + st['silent_steps'] == 14 and st['silent_steps'] == scores.count(None)
    assert st['hoyer_sum'] == pytest.approx(sum(s for s in scores if s is not None))
    assert rec.feature_dim == 0 and rec.parameter_count() == 0 and rec.features().size == 0


def test_activity_recorder_round_trip_copy_and_validation():
    rec = Q.ActivityRecorder()
    for v in ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]):
        rec.after_step(np.asarray(v), {})
    state = rec.export_state()
    assert state['scored_steps'] == 2 and state['silent_steps'] == 1 and state['hoyer_sum'] == pytest.approx(1.0)
    twin = copy.deepcopy(rec)
    twin.after_step(np.asarray([0.0, 2.0]), {})
    assert rec.export_state() == state and twin.stats()['scored_steps'] == 3
    other = Q.ActivityRecorder()
    other.import_state(json.loads(json.dumps(state)))
    assert other.export_state() == state
    pooled = Q.ActivityRecorder.combine([rec, twin])
    assert pooled == {'hoyer_sparseness': pytest.approx(3.0 / 5.0), 'scored_steps': 5, 'silent_steps': 2}
    assert Q.ActivityRecorder.combine([Q.ActivityRecorder()])['hoyer_sparseness'] == 1.0      # all silent: fully sparse
    for bad in ({**state, 'id': 'other'}, {**state, 'extra': 1}, {**state, 'scored_steps': -1},
                {**state, 'hoyer_sum': 3.5}, {**state, 'silent_steps': 1.5}, {**state, 'hoyer_sum': math.nan}, []):
        fresh = Q.ActivityRecorder()
        with pytest.raises(ValueError):
            fresh.import_state(bad)
        assert fresh.stats() == {'hoyer_sum': 0.0, 'scored_steps': 0, 'silent_steps': 0}


# ====================================================================== NEUROWEAVE evaluator
def test_neuroweave_evaluator_reproduces_the_lab_task_arm_and_seals_a_verifiable_receipt():
    g = Q.Genome(('sparse_code', 'reward_plasticity', 'synaptic_delays', 'membrane_dynamics'), 'clustered', 16, 0.15)
    ev = Q.NeuroWeaveEvaluator(**TINY_NW)
    r = ev(g, 3)
    arm = _task_arm(g.mechanisms, 3, task='delayed_recall', steps=16, heldout_seeds=(100, 101), heldout_steps=8,
                    neurons=16, topology='clustered')
    assert r['train'] == arm['train']
    assert r['heldout'] == {k: arm['heldout'][k] for k in ('accuracy', 'log_loss', 'brier', 'chance_accuracy')}
    assert r['estimated_cost'] == arm['estimated_cost']
    assert r['budget'] == {k: arm['budget'][k] for k in r['budget']}
    assert verify_receipt(r) and r['evidence_tier'] == EVIDENCE_TIER and r['schema'] == LAB_SCHEMA
    assert r['fitness'] == r['heldout']['accuracy']
    assert r['descriptors']['log2_ops_per_step'] == math.log2(r['estimated_cost']['total_ops'] / 16)
    assert r['activity']['scored_steps'] + r['activity']['silent_steps'] == 2 * 8
    assert Q.verify_candidate(g, r, seed=3, protocol=ev.protocol(), space=Q.GenomeSpace()) == []


def test_descriptors_are_deterministic_and_respond_to_the_genes_that_drive_them():
    ev = Q.NeuroWeaveEvaluator(task='delayed_recall', steps=16, heldout_seeds=(100,), heldout_steps=8)
    g = Q.Genome(('sparse_code', 'context_attention'), 'ring_lattice', 16, 0.05)
    a, b = ev(g, 0), ev(g, 0)
    assert a['digest'] == b['digest'] and a['descriptors'] == b['descriptors']
    archive = Q.EliteArchive()
    assert archive.cell_of(a['descriptors']) == archive.cell_of(b['descriptors'])
    denser = ev(Q.Genome(g.mechanisms, g.topology, 16, 0.5), 0)                 # k-WTA keeps 8 of 16 units, not 1
    assert denser['descriptors']['activity_sparseness'] < a['descriptors']['activity_sparseness']
    cheaper = ev(Q.Genome(('sparse_code',), g.topology, 16, 0.05), 0)           # without context attention's ops
    assert cheaper['descriptors']['log2_ops_per_step'] < a['descriptors']['log2_ops_per_step']
    bigger = ev(Q.Genome(g.mechanisms, g.topology, 32, 0.05), 0)
    assert bigger['descriptors']['log2_ops_per_step'] > a['descriptors']['log2_ops_per_step']
    off_a, off_b = ev(Q.Genome((), g.topology, 16, 0.05), 0), ev(Q.Genome((), g.topology, 16, 0.5), 0)
    assert off_a['heldout'] == off_b['heldout'] and off_a['descriptors'] == off_b['descriptors']   # neutral gene
    assert off_a['digest'] != off_b['digest']                                                     # different spec


def test_neuroweave_search_reproduces_on_spot_checks_audit_and_archive_round_trip():
    ev = Q.NeuroWeaveEvaluator(task='delayed_recall', steps=16, heldout_seeds=(100,), heldout_steps=8)
    s = Q.QualityDiversitySearch(ev, seed=0, budget=4, space=Q.GenomeSpace(neurons=(16,)), initial_random=2,
                                 spot_check_every=2, cost_range=(10.1, 10.8))
    out = s.run()
    assert out['rejected'] == 0 and out['spot_checks'] == 2 and out['evaluator_calls'] == 4 + 2
    assert s.audit() == {'checked': len(s.archive), 'reproduced': len(s.archive), 'mismatched_ids': []}
    for e in s.archive.elites():
        receipt = s.archive.elite_receipt(e['cell'])
        g = Q.Genome.from_dict(e['genome'])
        assert Q.verify_candidate(g, receipt, seed=0, protocol=s.protocol, space=s.space) == []
        assert 'measured_wallclock_s' not in receipt                    # stored without wall-clock: deterministic
    text = s.archive.export_json()
    assert Q.EliteArchive.from_json(text).export_json() == text        # real float metrics survive JSON exactly
    _assert_best_per_cell(s.archive)


def test_neuroweave_evaluator_validation():
    for kwargs in ({'task': 'nope'}, {'steps': 2}, {'heldout_seeds': ()}, {'heldout_seeds': (5, 5)},
                   {'heldout_steps': 0}):
        with pytest.raises(ValueError):
            Q.NeuroWeaveEvaluator(**{**TINY_NW, **kwargs})
    ev = Q.NeuroWeaveEvaluator(**TINY_NW)
    g = Q.Genome((), 'clustered', 16, 0.15)
    for args in ((g, 100), (g.to_dict(), 0), (g, -1), (g, 1.5)):
        with pytest.raises(ValueError):
            ev(*args)
    with pytest.raises(ValueError):
        Q.LandscapeEvaluator(Q.Genome((), 'hybrid', 16, 0.05), SMALL)


# ====================================================================== verification gate
def test_verify_candidate_accepts_a_genuine_receipt_and_names_each_tampering():
    ev = _landscape()
    g = Q.Genome(('sparse_code', 'synaptic_delays'), 'clustered', 16, 0.05)
    r = ev(g, 0)
    proto = ev.protocol()

    def check(receipt, genome=g, seed=0, protocol=proto):
        return set(Q.verify_candidate(genome, receipt, seed=seed, protocol=protocol, space=SMALL))

    assert r['fitness'] < 1.0 and check(r) == set()
    unsealed = copy.deepcopy(r)
    unsealed['fitness'] = 1.0
    assert check(unsealed) == {'receipt_digest_mismatch', 'inconsistent_metrics'}
    assert check(_reseal(r, lambda x: x.update(fitness=1.0))) == {'inconsistent_metrics'}
    assert check(_reseal(r, lambda x: x['descriptors'].update(activity_sparseness=0.0))) == {'inconsistent_metrics'}
    # A consistently re-sealed lie passes every static check: only the re-evaluation spot check can catch it.
    assert check(_reseal(r, _inflate)) == set()
    assert check(r, genome=TARGET) == {'genome_mismatch', 'spec_mismatch'}
    assert check(r, seed=1) == {'protocol_mismatch', 'spec_mismatch'}
    # The lab spec's structural / sleep schedules are max(5, steps // 6) and max(10, steps // 3): equal at 16 and
    # 17 steps (only the protocol differs) but not at 60.
    assert check(r, protocol={**proto, 'steps': 17}) == {'protocol_mismatch'}
    assert check(r, protocol={**proto, 'steps': 60}) == {'protocol_mismatch', 'spec_mismatch'}
    assert check(_reseal(r, lambda x: x['config']['spec'].update(neurons=-5))) == {'invalid_spec'}
    assert check(_reseal(r, lambda x: x['config']['spec'].update(sparsity=0.5))) == {'spec_mismatch'}
    assert check(_reseal(r, lambda x: x.update(experiment_id='quality_diversity'))) == {'receipt_schema_mismatch'}
    assert check(_reseal(r, lambda x: x.update(evidence_tier='measured'))) == {'receipt_schema_mismatch'}
    assert check(_reseal(r, lambda x: x.pop('activity'))) == {'invalid_metrics'}
    assert check(_reseal(r, lambda x: x['heldout'].update(accuracy=1.5))) == {'invalid_metrics'}
    outside = Q.Genome(g.mechanisms, g.topology, 24, 0.05)
    assert 'genome_outside_space' in check(r, genome=outside)
    assert check(['not', 'a', 'receipt']) == {'receipt_not_a_mapping'}
    for bad in ({'seed': -1}, {'seed': True}, {'protocol': {'kind': 'x'}}, {'protocol': None}):
        with pytest.raises(ValueError):
            check(r, **bad)
    with pytest.raises(ValueError):
        Q.verify_candidate(g, r, seed=0, protocol=proto, space=SMALL.to_dict())


def test_tampered_and_failed_candidates_are_rejected_logged_and_never_admitted():
    def hostile(r, g, n):
        if 'context_attention' in g.mechanisms:
            raise RuntimeError('boom')
        if 'flyhash_projection' in g.mechanisms:
            return ['not', 'a', 'receipt']
        if 'sparse_code' in g.mechanisms:
            r['fitness'] = 1.0                                     # edited after sealing
        return r

    s = _search(_Wrapped(hostile), budget=60, spot_check_every=0, crossover_rate=0.5)
    out = s.run()
    lineage = s.archive.lineage()
    expected = {}
    for node in lineage:
        mechs = node['genome']['mechanisms']
        if 'context_attention' in mechs:
            reasons = ['evaluation_error']
            assert node['error'] == 'RuntimeError: boom' and node['digest'] is None
        elif 'flyhash_projection' in mechs:
            reasons = ['receipt_not_a_mapping']
        elif 'sparse_code' in mechs:
            reasons = ['receipt_digest_mismatch', 'inconsistent_metrics']
        else:
            reasons = []
        assert node['reasons'] == reasons and (node['status'] == 'rejected') == bool(reasons)
        if reasons:
            expected[node['id']] = reasons
            assert node['cell'] is None and node['fitness'] is None and node['displaced_by'] is None
    assert len(expected) > 10 and len(lineage) - len(expected) > 5
    assert {r['id']: r['reasons'] for r in s.archive.rejections()} == expected
    assert sum(out['rejections_by_reason'].values()) == sum(len(v) for v in expected.values())
    assert out['rejected'] == len(expected) and out['evaluations'] == 60
    elite_ids = {e['id'] for e in s.archive.elites()}
    assert not elite_ids & set(expected) and not any(set(n['parents']) & set(expected) for n in lineage)
    assert all(e['fitness'] < 1.0 for e in s.archive.elites())
    _assert_best_per_cell(s.archive)


def test_failed_spot_checks_are_rejected_and_logged():
    # A run that does not reproduce: each call re-seals a slightly different log loss.
    def flaky(r, g, n):
        return _reseal(r, lambda x: x['heldout'].update(log_loss=x['heldout']['log_loss'] + 1e-3 * n))

    s = _search(_Wrapped(flaky), budget=24, spot_check_every=3)
    s.run()
    for node in s.archive.lineage():
        due = (node['id'] + 1) % 3 == 0
        assert node['spot_checked'] is due
        assert node['reasons'] == (['spot_check_digest_mismatch'] if due else [])
    assert s.spot_checks == 8 and len(s.archive.rejections()) == 8
    assert s.audit()['reproduced'] == 0                                  # nothing admitted reproduces either
    unchecked = _search(_Wrapped(flaky), budget=24, spot_check_every=0)
    unchecked.run()
    assert not unchecked.archive.rejections() and unchecked.spot_checks == 0      # toggle: no spot checks, no catch

    # A cheater that inflates its first answer for sparse_code genomes (consistently re-sealed) is caught when
    # every candidate is spot-checked, and slips through when nothing is.
    def cheat(r, g, n):
        return _reseal(r, _inflate) if n == 1 and 'sparse_code' in g.mechanisms else r

    gated = _search(_Wrapped(cheat), budget=30, spot_check_every=1)
    gated.run()
    for node in gated.archive.lineage():
        cheated = 'sparse_code' in node['genome']['mechanisms']
        assert node['reasons'] == (['spot_check_digest_mismatch'] if cheated else [])
    truth = _landscape()
    for e in gated.archive.elites():                 # every admitted elite carries its true (honest) fitness
        assert 'sparse_code' not in e['genome']['mechanisms']
        assert e['fitness'] == truth.fitness(Q.Genome.from_dict(e['genome']))
    assert gated.audit()['mismatched_ids'] == []
    open_door = _search(_Wrapped(cheat), budget=30, spot_check_every=0)
    open_door.run()
    assert not open_door.archive.rejections()
    lies = [e for e in open_door.archive.elites() if 'sparse_code' in e['genome']['mechanisms']]
    assert lies and all(e['fitness'] == 1.0 for e in lies)        # inflated answers admitted unchecked
    assert {e['id'] for e in lies} <= set(open_door.audit()['mismatched_ids'])   # a later audit exposes them

    def broken_rerun(r, g, n):
        if n > 1:
            raise OSError('disk gone')
        return r

    s = _search(_Wrapped(broken_rerun), budget=10, spot_check_every=2)
    s.run()
    for node in s.archive.lineage():
        if node['spot_checked']:
            assert node['reasons'] == ['spot_check_error'] and node['error'] == 'OSError: disk gone'


def test_the_shared_cache_cannot_hide_a_result_that_does_not_reproduce():
    def cheat(r, g, n):
        return _reseal(r, _inflate) if n == 1 else r

    evaluator = _Wrapped(cheat)
    cache = {}
    first = _search(evaluator, budget=6, initial_random=6, spot_check_every=0, cache=cache)
    first.run()
    assert all(n['fitness'] == 1.0 for n in first.archive.lineage())     # unchecked: the lies are admitted
    second = _search(evaluator, budget=6, initial_random=6, method='random', spot_check_every=1, cache=cache)
    second.run()
    assert second.cache_hits == 6 and second.evaluator_calls == 6           # cached answers, fresh spot checks
    assert [n['genome'] for n in second.archive.lineage()] == [n['genome'] for n in first.archive.lineage()]
    assert all(n['reasons'] == ['spot_check_digest_mismatch'] for n in second.archive.lineage())


# ====================================================================== archive: best per cell, lineage, persistence
def test_archive_keeps_the_best_candidate_per_cell_with_log_loss_and_age_tie_breaks():
    ev = _landscape()
    g = Q.Genome(('reward_plasticity',), 'clustered', 16, 0.05)
    base = ev(g, 0)

    def receipt(fitness, log_loss):
        return _reseal(base, lambda x: (_set(x, fitness, log_loss)))

    def _set(x, fitness, log_loss):
        x['heldout']['accuracy'] = x['fitness'] = fitness
        x['heldout']['log_loss'] = log_loss

    archive = Q.EliteArchive()
    plan = [(0.5, 0.7, 'elite', None), (0.7, 0.6, 'elite', 4), (0.6, 0.1, 'not_elite', None),
            (0.7, 0.6, 'not_elite', None), (0.7, 0.5, 'elite', None)]
    for fitness, loss, _, _ in plan:
        archive.record(genome=g, origin='random', parents=(), operators=(), receipt=receipt(fitness, loss), reasons=(),
                       spot_checked=False)
    lineage = archive.lineage()
    assert [n['status'] for n in lineage] == [p[2] for p in plan]
    assert [n['displaced_by'] for n in lineage] == [1, 4, None, None, None]
    assert len(archive) == 1 and archive.best()['id'] == 4 and archive.qd_score() == pytest.approx(0.7)
    _assert_best_per_cell(archive)
    snapshot = archive.export_json()
    bad_calls = [
        {'receipt': receipt(0.9, 0.1), 'reasons': (), 'origin': 'mutation', 'parents': (0,)},       # 0 is no elite
        {'receipt': receipt(0.9, 0.1), 'reasons': (), 'origin': 'mutation', 'parents': ()},
        {'receipt': receipt(0.9, 0.1), 'reasons': (), 'origin': 'crossover', 'parents': (4, 4)},
        {'receipt': receipt(0.9, 0.1), 'reasons': ('made_up',), 'origin': 'random', 'parents': ()},
        {'receipt': receipt(0.9, 0.1), 'reasons': (), 'origin': 'cloned', 'parents': ()},
        {'receipt': {**receipt(0.9, 0.1), 'fitness': 0.95}, 'reasons': (), 'origin': 'random', 'parents': ()},
        {'receipt': _reseal(base, lambda x: x['heldout'].update(accuracy=None)), 'reasons': (), 'origin': 'random',
         'parents': ()}]
    for kw in bad_calls:
        with pytest.raises(ValueError):
            archive.record(genome=g, operators=(), spot_checked=False, **kw)
        assert archive.export_json() == snapshot                                # validated before any change
    with pytest.raises(ValueError):
        archive.record(genome=g, origin='random', parents=(), operators=(), receipt=None, reasons=(),
                       spot_checked=1)
    full = Q.EliteArchive(max_nodes=1)
    full.record(genome=g, origin='random', parents=(), operators=(), receipt=None, reasons=['evaluation_error'],
                spot_checked=False)
    with pytest.raises(ValueError):
        full.record(genome=g, origin='random', parents=(), operators=(), receipt=None, reasons=['evaluation_error'],
                    spot_checked=False)
    for kw in ({'cost_bins': 0}, {'cost_range': (2.0, 1.0)}, {'activity_range': (0.0, 1.5)}, {'max_nodes': 0}):
        with pytest.raises(ValueError):
            Q.EliteArchive(**kw)


def test_landscape_archive_matches_a_brute_force_best_per_cell_map():
    space = Q.GenomeSpace(mechanisms=('sparse_code', 'reward_plasticity'), topologies=('clustered', 'ring_lattice'),
                          neurons=(16, 32), sparsity=(0.05, 0.3))
    target = Q.Genome(('reward_plasticity',), 'ring_lattice', 32, 0.3)
    ev = _landscape(target, space)
    s = Q.QualityDiversitySearch(ev, seed=2, budget=space.size, method='random', space=space, initial_random=1,
                                 max_resample=1000, spot_check_every=0)
    s.run()
    assert len({json.dumps(n['genome'], sort_keys=True) for n in s.archive.lineage()}) == space.size == 32
    brute = {}
    for mechs, topo, neurons, sparsity in itertools.product(
            [m for k in range(3) for m in itertools.combinations(space.mechanisms, k)], space.topologies,
            space.neurons, space.sparsity):
        g = Q.Genome(mechs, topo, neurons, sparsity)
        m = Q.candidate_metrics(ev(g, 2))
        cell = s.archive.cell_of({k: m[k] for k in Q.DESCRIPTOR_KEYS})
        if cell not in brute or (m['fitness'], -m['log_loss']) > brute[cell][0]:
            brute[cell] = ((m['fitness'], -m['log_loss']), g.to_dict())
    assert {tuple(e['cell']): e['genome'] for e in s.archive.elites()} == {c: v[1] for c, v in brute.items()}
    assert s.archive.best()['genome'] == target.to_dict() and s.archive.best()['fitness'] == 1.0
    _assert_best_per_cell(s.archive)


def test_lineage_is_a_consistent_dag_and_selection_uses_its_child_counts():
    s = _search(budget=60, crossover_rate=0.5)
    out = s.run()
    lineage = s.archive.lineage()
    current, by_cell, depth = set(), {}, []
    for node in lineage:
        assert node['id'] == len(depth) and set(node['parents']) <= current      # parents were elites back then
        assert len(node['parents']) == {'random': 0, 'mutation': 1, 'crossover': 2}[node['origin']]
        ops = node['operators']
        if node['origin'] == 'random':
            assert ops == []
        elif node['origin'] == 'mutation':
            assert ops and 'crossover' not in ops
            parent = Q.Genome.from_dict(lineage[node['parents'][0]]['genome'])
            assert _replay(parent, ops) == Q.Genome.from_dict(node['genome'])
        else:
            assert ops[0] == 'crossover' and len(ops) >= 2
        depth.append(1 + max(depth[p] for p in node['parents']) if node['parents'] else 0)
        if node['status'] == 'elite':
            current.discard(by_cell.get(tuple(node['cell'])))
            by_cell[tuple(node['cell'])] = node['id']
            current.add(node['id'])
    assert current == {e['id'] for e in s.archive.elites()}
    assert [n['origin'] for n in lineage[:8]] == ['random'] * 8
    assert {'mutation', 'crossover'} <= {n['origin'] for n in lineage}
    for node in lineage:
        kids = sum(node['id'] in n['parents'] for n in lineage if n['status'] != 'rejected')
        assert s.archive.children_count(node['id']) == kids
    assert out['lineage'] == {'nodes': 60, 'max_depth': max(depth),
                              'origins': {o: sum(n['origin'] == o for n in lineage) for o in Q.ORIGINS}}
    elites, p = s.parent_probabilities()
    expected = Q.dgm_parent_probabilities([e['fitness'] for e in elites],
                                          [s.archive.children_count(e['id']) for e in elites])
    assert np.array_equal(p, expected) and [e['cell'] for e in elites] == sorted(e['cell'] for e in elites)
    u = _search(budget=10, selection='uniform', initial_random=10)
    u.run()
    _, pu = u.parent_probabilities()
    assert np.allclose(pu, 1.0 / len(pu))
    _assert_best_per_cell(s.archive)


def test_archive_export_import_round_trip_and_integrity_hash():
    def hostile(r, g, n):
        if 'context_attention' in g.mechanisms:
            raise RuntimeError('boom')
        return r

    s = _search(_Wrapped(hostile), budget=40, crossover_rate=0.5)
    s.run()
    text = s.archive.export_json()
    doc = json.loads(text)
    assert doc['schema'] == Q.ARCHIVE_SCHEMA and doc['sha256'] == s.archive.sha256
    canonical = json.dumps(doc['payload'], sort_keys=True, separators=(',', ':'))
    assert doc['sha256'] == hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    assert s.archive.rejections() and len(s.archive) > 2
    copy_ = Q.EliteArchive.from_json(text)
    assert copy_.export_json() == text and copy_.sha256 == s.archive.sha256
    assert copy_.lineage() == s.archive.lineage() and copy_.rejections() == s.archive.rejections()
    assert copy_.elites() == s.archive.elites() and copy_.qd_score() == s.archive.qd_score()
    assert all(copy_.children_count(i) == s.archive.children_count(i) for i in range(40))
    _assert_best_per_cell(copy_)
    again = _search(_Wrapped(hostile), budget=40, crossover_rate=0.5)
    again.run()
    assert again.archive.export_json() == text                                 # deterministic export


def _rehashed(doc):
    doc = copy.deepcopy(doc)
    doc['sha256'] = hashlib.sha256(json.dumps(doc['payload'], sort_keys=True, separators=(',', ':'))
                                   .encode('utf-8')).hexdigest()
    return json.dumps(doc)


def test_archive_import_detects_tampering_and_changes_nothing():
    def hostile(r, g, n):
        if 'context_attention' in g.mechanisms:
            raise RuntimeError('boom')
        return r

    s = _search(_Wrapped(hostile), budget=40, crossover_rate=0.5)
    s.run()
    text = s.archive.export_json()
    doc = json.loads(text)
    pay = doc['payload']
    elite = pay['elites'][0]
    eid = elite['id']
    not_elite = next(n['id'] for n in pay['lineage'] if n['status'] == 'not_elite')
    child = next(n['id'] for n in pay['lineage'] if n['origin'] == 'mutation')
    target = _search(budget=40)                      # same grid configuration (incl. max_nodes) as the export
    for _ in range(5):
        target.step()
    before = target.archive.export_json()

    def edited(fn, rehash=True):
        d = copy.deepcopy(doc)
        fn(d['payload'])
        return _rehashed(d) if rehash else json.dumps(d)

    def set_receipt_fitness(p):
        r = p['elites'][0]['receipt']
        r['heldout']['accuracy'] = r['fitness'] = 1.0

    def resealed_receipt(p):
        set_receipt_fitness(p)
        p['elites'][0]['receipt'] = harness.seal(p['elites'][0]['receipt'])
        p['elites'][0]['digest'] = p['lineage'][eid]['digest'] = p['elites'][0]['receipt']['digest']

    cases = [
        (edited(lambda p: p['elites'][0].update(fitness=1.0), rehash=False), 'integrity hash'),
        (edited(lambda p: p['elites'][0].update(fitness=1.0)), 'differs from its lineage node'),
        (edited(set_receipt_fitness), 'does not verify'),
        (edited(resealed_receipt), 'differs from its receipt'),
        (edited(lambda p: p['lineage'][not_elite].update(status='elite')), 'contradicts'),
        (edited(lambda p: p['lineage'][eid].update(fitness=0.0)), 'contradicts|differs from its lineage node'),
        (edited(lambda p: p['lineage'][child].update(parents=[not_elite])), 'parents'),
        (edited(lambda p: p['rejections'].pop()), 'rejection log'),
        (edited(lambda p: p['lineage'][0].update(displaced_by=None)) if pay['lineage'][0]['displaced_by'] is not None
         else edited(lambda p: p['lineage'][0].update(displaced_by=39)), 'displaced_by'),
        (edited(lambda p: p['elites'].pop()), 'elites'),
        (edited(lambda p: p['elites'][0].update(cell=[[0], 1])), 'does not match the replayed grid'),
        (edited(lambda p: p['lineage'][1].update(id=True)), 'malformed lineage node 1'),
        (edited(lambda p: p['lineage'][eid].update(cell=[float(c) for c in p['lineage'][eid]['cell']])),
         'cell does not match'),
        (edited(lambda p: p['grid'].update(cost_bins=8)), 'grid'),
        (text.replace(Q.ARCHIVE_SCHEMA, 'OtherArchiveV1'), 'schema'),
        (text[:-2], 'JSON'),
        (json.dumps({**doc, 'extra': 1}), 'schema'),
        (42, 'JSON string'),
    ]
    for bad, message in cases:
        with pytest.raises(ValueError, match=message):
            target.archive.import_json(bad)
        assert target.archive.export_json() == before                                # validated before mutating
    with pytest.raises(ValueError):
        Q.EliteArchive(cost_bins=3).import_json(text)                              # grid configuration differs
    target.archive.import_json(text)
    assert target.archive.export_json() == text


# ====================================================================== the search
def test_qd_climbs_a_known_landscape_that_random_search_does_not_solve_at_equal_budget():
    hits = {'qd': 0, 'random': 0}
    best, qd_score, mean_fit = ({'qd': [], 'random': []} for _ in range(3))
    for method in hits:
        for seed in range(5):
            s = _search(seed=seed, budget=80, method=method)
            out = s.run()
            hits[method] += out['best_fitness'] == 1.0
            best[method].append(out['best_fitness'])
            qd_score[method].append(out['qd_score'])
            mean_fit[method].append(out['verified_fitness']['mean'])
            if out['best_fitness'] == 1.0:
                assert s.archive.best()['genome'] == TARGET.to_dict()
    assert hits['qd'] >= 3 and hits['qd'] > hits['random']
    assert np.mean(best['qd']) > np.mean(best['random'])
    # Illumination on the same landscape: higher QD-score (better elites in the cells) at an equal budget.
    assert sum(q > r for q, r in zip(qd_score['qd'], qd_score['random'])) >= 4
    assert np.mean(qd_score['qd']) > np.mean(qd_score['random']) and np.mean(mean_fit['qd']) > np.mean(mean_fit['random'])


def test_search_is_deterministic_toggles_change_it_and_global_state_is_untouched():
    before = _global_state()
    a, b = _search(budget=30, crossover_rate=0.5), _search(budget=30, crossover_rate=0.5)
    sa, sb = a.run(), b.run()
    assert _global_state() == before
    strip = lambda d: {k: v for k, v in d.items() if k != 'measured_wallclock_s'}
    assert strip(sa) == strip(sb) and a.archive.export_json() == b.archive.export_json()
    variants = {
        'seed': _search(budget=30, crossover_rate=0.5, seed=1), 'uniform': _search(budget=30, crossover_rate=0.5,
                                                                                   selection='uniform'),
        'no_crossover': _search(budget=30, crossover_rate=0.0), 'random': _search(budget=30, method='random'),
        'more_spot_checks': _search(budget=30, crossover_rate=0.5, spot_check_every=1),
        'flat_sigmoid': _search(budget=30, crossover_rate=0.5, selection_sharpness=0.0),
        'single_operator': _search(budget=30, crossover_rate=0.5, extra_mutation_prob=0.0)}
    for name, other in variants.items():
        other.run()
        assert other.archive.sha256 != a.archive.sha256, name
    assert all(n['origin'] != 'crossover' for n in variants['no_crossover'].archive.lineage())
    assert all(n['origin'] == 'random' for n in variants['random'].archive.lineage())
    assert variants['more_spot_checks'].spot_checks == 30 and a.spot_checks == 6
    flat = variants['flat_sigmoid']                 # sharpness 0: only the child-count term h_i is left
    elites, p = flat.parent_probabilities()
    w = np.array([0.0 if e['fitness'] >= 1.0 else 1.0 / (1 + flat.archive.children_count(e['id'])) for e in elites])
    assert np.allclose(p, w / w.sum(), rtol=0.0, atol=1e-12)
    assert all(len(n['operators']) == {'random': 0, 'mutation': 1, 'crossover': 2}[n['origin']]
               for n in variants['single_operator'].archive.lineage())
    # common random numbers: both methods start from the same initial candidates
    first = [n['genome'] for n in a.archive.lineage()[:8]]
    assert first == [n['genome'] for n in variants['random'].archive.lineage()[:8]]


def test_budget_cache_and_counters():
    counting = _Wrapped(lambda r, g, n: r)
    s = _search(counting, budget=12, spot_check_every=4)
    s.run()
    assert s.done and s.archive.evaluations == 12 and s.spot_checks == 3
    assert counting.calls == s.evaluator_calls == 12 + 3 and s.cache_hits == 0
    with pytest.raises(ValueError):
        s.step()
    cache = {}
    one = _search(counting, budget=12, cache=cache)
    one.run()
    assert one.cache_hits == 0 and one.evaluator_calls == 12 + 2 and len(cache) == 12
    two = _search(counting, budget=12, cache=cache)
    two.run()
    assert two.cache_hits == 12 and two.evaluator_calls == 2                  # only the spot checks re-ran
    assert two.archive.export_json() == one.archive.export_json()
    assert counting.calls == 15 + 14 + 2


@pytest.mark.parametrize('kwargs', [
    {'method': 'grid'}, {'selection': 'tournament'}, {'budget': 0}, {'budget': Q.MAX_BUDGET + 1},
    {'initial_random': 0}, {'initial_random': 41}, {'spot_check_every': 41}, {'crossover_rate': 1.5},
    {'extra_mutation_prob': 1.0}, {'selection_sharpness': -1.0}, {'max_resample': 0}, {'cache': []},
    {'cost_bins': 0}, {'activity_range': (0.5, 0.5)}, {'space': SMALL.to_dict()}, {'seed': -1}])
def test_search_validation(kwargs):
    with pytest.raises(ValueError):
        _search(**kwargs)


def test_search_requires_an_evaluator_with_a_protocol():
    with pytest.raises(ValueError):
        Q.QualityDiversitySearch(lambda g, s: {}, budget=4)
    with pytest.raises(ValueError):
        Q.QualityDiversitySearch(_Wrapped(lambda r, g, n: r, inner=type('P', (), {'protocol': lambda self: {}})()))


# ====================================================================== metrics
def test_illumination_metrics_and_pareto_front_known_answers():
    out = Q.illumination_metrics({'a': {(0, 0): 0.5, (0, 1): 1.0}, 'b': {(0, 0): 1.0, (1, 1): 0.5}})
    assert out['a'] == pytest.approx({'reliability': 0.5, 'precision': 0.75, 'coverage_of_union': 2 / 3})
    assert out['b'] == pytest.approx({'reliability': 2 / 3, 'precision': 1.0, 'coverage_of_union': 2 / 3})
    zero = Q.illumination_metrics({'a': {(0, 0): 0.0}, 'b': {}})
    assert zero['a'] == {'reliability': 1.0, 'precision': 1.0, 'coverage_of_union': 1.0}
    assert zero['b'] == {'reliability': 0.0, 'precision': None, 'coverage_of_union': 0.0}
    assert Q.pareto_front([[1, 1], [2, 0], [0, 2], [0.5, 0.5], [1, 1]]) == [0, 1, 2, 4]
    assert Q.pareto_front([[3.0]]) == [0]
    for bad in ([], [[1.0, math.nan]], [1.0, 2.0]):
        with pytest.raises(ValueError):
            Q.pareto_front(bad)
    for bad in ({}, {'a': [0.5]}, {'a': {(0, 0): -1.0}}, []):
        with pytest.raises(ValueError):
            Q.illumination_metrics(bad)
    s = _search(budget=40)
    s.run()
    front = s.archive.pareto_front()
    elites = s.archive.elites()
    for row in front:
        assert not any(e['fitness'] >= row['fitness'] and e['descriptors']['log2_ops_per_step'] <=
                       row['log2_ops_per_step'] and (e['fitness'], -e['descriptors']['log2_ops_per_step']) !=
                       (row['fitness'], -row['log2_ops_per_step']) for e in elites)
    assert [r['log2_ops_per_step'] for r in front] == sorted(r['log2_ops_per_step'] for r in front)


# ====================================================================== lab experiment
def test_run_experiment_quick_receipt_is_sealed_deterministic_and_budget_matched():
    before = _global_state()
    first = Q.run_experiment(quick=True)
    assert _global_state() == before
    assert verify_receipt(first) and first['verdict'] in VERDICTS and first['evidence_tier'] == EVIDENCE_TIER
    assert first['schema'] == LAB_SCHEMA and first['experiment_id'] == 'quality_diversity'
    assert first['kind'] == 'search_comparison' and first['verdict'] == first['comparisons']['qd_score']['verdict']
    cfg = first['config']
    assert cfg['quick'] is True and cfg['seeds'] == [0, 1, 2] and cfg['budget'] == 5 and cfg['steps'] == 16
    assert cfg['task'] == 'delayed_recall' and cfg['space']['neurons'] == [16] and cfg['spot_check_every'] == 2
    assert set(first['arms']) == {'quality_diversity', 'random_search'}
    for name, rows in ((n, a['per_seed']) for n, a in first['arms'].items()):
        assert [r['seed'] for r in rows] == [0, 1, 2]
        for r in rows:
            assert r['evaluations'] == 5 and r['spot_checks'] == 2 and r['rejected'] == 0
            assert r['qd_score'] == pytest.approx(sum(e['fitness'] for e in r['elites']))
            assert r['best_test_accuracy'] == r['best_test']['accuracy'] and 0.0 < r['coverage'] <= 1.0
    assert all(r['lineage']['origins']['random'] == 5 for r in first['arms']['random_search']['per_seed'])
    assert all(r['cache_hits'] == 3 for r in first['arms']['random_search']['per_seed'])   # shared first 3
    assert len(json.dumps(first)) < 64_000

    def nested_digest_keys(obj, top=True):          # receipt seals skip keys named 'digest' at any depth
        if isinstance(obj, dict):
            return sum((k == 'digest' and not top) + nested_digest_keys(v, False) for k, v in obj.items())
        return sum(nested_digest_keys(v, False) for v in obj) if isinstance(obj, list) else 0

    assert nested_digest_keys(first) == 0                           # so every candidate digest is sealed
    rows = first['arms']['quality_diversity']['per_seed']
    assert all(len(e['receipt_digest']) == 64 for r in rows for e in r['elites'])
    assert all(len(r['best_test']['receipt_digest']) == 64 for r in rows)
    assert Q.run_experiment(quick=True)['digest'] == first['digest']
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'improved' if first['verdict'] != 'improved' else 'worse'
    assert not verify_receipt(tampered)


@pytest.mark.parametrize('kwargs', [
    {'quick': 1}, {'surprise': 1}, {'seeds': ()}, {'seeds': (0, 0)}, {'seeds': (100,)}, {'test_seeds': (100,)},
    {'budget': 0}, {'task': 'nope'}, {'initial_random': 9}, {'neurons': ()}])
def test_run_experiment_validates_arguments_before_running(kwargs):
    with pytest.raises(ValueError):
        Q.run_experiment(**{'quick': True, **kwargs})
