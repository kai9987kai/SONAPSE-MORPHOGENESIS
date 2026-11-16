"""NEUROWEAVE laboratory: tasks, experiments, lesion assay and promotion ledger (tiny, fast settings)."""
import copy
import json
import os

import numpy as np
import pytest

from synapse.neuroweave import experiments as ex
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import (VERDICTS, clone_engine, make_spec, paired_stats, receipt_digest,
                                            results_to_csv, run_all_experiments, run_experiment, verdict_from_stats,
                                            verify_receipt)
from synapse.neuroweave.lesion import apply_lesion, lesion_order, run_lesion_assay
from synapse.neuroweave.promotion import STABLE_PRESET, PromotionLedger
from synapse.neuroweave.spec import EVIDENCE_TIER, EXPERIMENT_IDS, TOPOLOGIES
from synapse.neuroweave.tasks import TASKS, make_stream, task_targets

TINY = dict(steps=12, heldout_steps=6, neurons=16)


@pytest.fixture(scope='module')
def receipts():
    return run_all_experiments(True, **TINY)


# ---------------------------------------------------------------- tasks
def test_task_streams_are_deterministic_and_heldout_streams_differ():
    a, b, c = make_stream(3, 40), make_stream(3, 40), make_stream(103, 40)
    assert a.digest == b.digest and np.array_equal(a.inputs, b.inputs)
    assert a.digest != c.digest
    assert a.inputs.shape == (40, 8) and len(a.sensory) == 40 and set(a.targets) == set(TASKS)
    bits = (a.inputs[:, 0] > 0).astype(int)
    t = task_targets(bits, period=a.params['period'], phase=a.params['phase'], block=a.params['block'])
    assert np.array_equal(t['delayed_recall'][2:], bits[:-2])
    assert np.array_equal(t['xor'][1:], bits[1:] ^ bits[:-1])
    assert np.array_equal(t['parity'][2:], bits[2:] ^ bits[1:-1] ^ bits[:-2])
    assert np.all(np.abs(a.inputs) <= 1.0)
    # the past is not leaked into the input: only the current bit is present
    prev = np.where(bits[:-1] == 1, 1.0, -1.0)
    assert not any(np.array_equal(a.inputs[1:, col], prev) for col in range(8))
    with pytest.raises(ValueError):
        make_stream(-1, 10)


# ---------------------------------------------------------------- experiments
def test_every_experiment_runs_in_quick_mode_with_a_valid_receipt(receipts):
    assert [r['experiment_id'] for r in receipts] == list(EXPERIMENT_IDS)
    for r in receipts:
        assert verify_receipt(r), r['experiment_id']
        assert r['evidence_tier'] == EVIDENCE_TIER and r['schema'] == ex.LAB_SCHEMA
        assert r['verdict'] in VERDICTS
        assert r['config']['seeds'] == [0, 1] and r['config']['steps'] == 12
        assert len(r['digest']) == 64
        json.dumps(r)
        if r['kind'] in ('temporal_task', 'micro', 'ecology', 'graft'):
            assert {'baseline', 'candidate'} <= set(r['arms'])
            assert set(r['ablation_verdicts']) == set(r['mechanisms'])
            cmp = r['comparisons']['candidate_vs_baseline']['primary']
            assert cmp['n'] == 2 and cmp['ci95'] is not None
            for arm in r['arms'].values():
                for rec in arm['per_seed']:
                    assert 'measured_wallclock_s' in rec and 'estimated_cost' in rec and 'budget' in rec
    by = {r['experiment_id']: r for r in receipts}
    topo = by['topology_optimisation']
    assert set(topo['arms']) == set(TOPOLOGIES)
    for seed, budget in topo['edge_budget_per_seed'].items():
        edges = {t: [p['metrics']['edges'] for p in topo['arms'][t]['per_seed'] if str(p['seed']) == seed][0]
                 for t in TOPOLOGIES}
        assert set(edges.values()) == {budget}            # matched edge budget across all 12 topologies
    micro = by['micro_collective_learning']
    assert micro['arms']['baseline']['per_seed'][0]['predictor'] == 'DenseMLP'
    assert 'candidate_vs_persistence' in micro['comparisons']
    assert micro['budget']['budget_matched_within_10pct']
    eco = by['action_self_model']['arms']['candidate']['per_seed'][0]
    assert eco['train']['mean_authorship'] is not None and 'collisions' in eco['heldout']
    lesion = by['lesion_robustness']['lesion_assay']
    assert lesion['originals_untouched'] is True
    csv = results_to_csv(receipts)
    assert csv.splitlines()[0].startswith('experiment_id,kind,task,arm')
    assert len(csv.splitlines()) > len(receipts)


def test_graft_removal_restores_exact_uncoupled_outputs(receipts):
    graft = {r['experiment_id']: r for r in receipts}['neural_grafting']
    for chk in graft['removal_checks']:
        assert chk['detach_restores_uncoupled_outputs'] and chk['detach_vs_zero_coupling_max_abs_diff'] == 0.0
        assert chk['uncoupled_equals_no_graft'] and chk['uncoupled_vs_no_graft_max_abs_diff'] == 0.0


def test_experiment_is_deterministic_excluding_wallclock():
    kw = dict(seeds=(0, 1), heldout_seeds=(100,), **TINY)
    a = run_experiment('context_attention', **kw)
    b = run_experiment('context_attention', **kw)
    assert a['digest'] == b['digest']
    c = run_experiment('context_attention', **{**kw, 'seeds': (0, 2)})
    assert c['digest'] != a['digest']
    tampered = copy.deepcopy(a)
    tampered['verdict'] = 'improved' if a['verdict'] != 'improved' else 'worse'
    assert not verify_receipt(tampered)
    with pytest.raises(ValueError):
        run_experiment('context_attention', seeds=(0,), heldout_seeds=(0,), **TINY)
    with pytest.raises(ValueError):
        run_experiment('not_an_experiment')


def test_sonapse_residual_controller_runs_the_real_agent():
    r = run_experiment('action_self_model', seeds=(0, 1), heldout_seeds=(100,), steps=4, heldout_steps=2,
                       neurons=16, controller='sonapse_residual')
    assert verify_receipt(r) and set(r['arms']) == {'baseline', 'candidate'}
    assert r['arms']['candidate']['per_seed'][0]['controller'] == 'sonapse_residual'
    assert r['arms']['baseline']['per_seed'][0]['controller'] == 'sonapse_baseline'


def test_verdict_logic_uses_only_the_confidence_interval():
    up = paired_stats([0.70, 0.72, 0.71], [0.60, 0.61, 0.60])
    assert verdict_from_stats(up)['verdict'] == 'improved' and up['ci95'][0] > 0
    down = paired_stats([0.50, 0.52, 0.49], [0.60, 0.61, 0.60])   # non-identical differences (sd > 0)
    assert verdict_from_stats(down)['verdict'] == 'worse'
    flat = paired_stats([0.50, 0.51, 0.50], [0.60, 0.61, 0.60])     # identical differences: zero-width CI
    assert verdict_from_stats(flat) == {'verdict': 'no_detectable_difference', 'reason': 'degenerate_ci'}
    noisy = paired_stats([0.70, 0.50, 0.65], [0.60, 0.61, 0.60])
    assert verdict_from_stats(noisy)['verdict'] == 'no_detectable_difference'
    single = paired_stats([0.9], [0.1])
    assert verdict_from_stats(single) == {'verdict': 'no_detectable_difference', 'reason': 'insufficient_seeds_for_ci'}
    lower = paired_stats([0.10, 0.11, 0.10], [0.20, 0.21, 0.20])     # e.g. MSE: lower is better
    assert verdict_from_stats(lower, higher_is_better=False)['verdict'] == 'improved'
    assert verdict_from_stats(lower, higher_is_better=True)['verdict'] == 'worse'
    same = paired_stats([0.5, 0.5], [0.5, 0.5])
    assert verdict_from_stats(same)['verdict'] == 'no_detectable_difference'


# ---------------------------------------------------------------- lesions
def test_lesions_touch_clones_only_and_progressive_damage_is_monotone():
    eng = NeuroWeaveEngine(make_spec((), seed=0, neurons=24, topology='multi_region', steps=12))
    for _ in range(3):
        eng.step(np.ones(8) * 0.2)
    before = json.dumps(ex._jsonable(eng.export_state()), sort_keys=True)
    clone = clone_engine(eng)
    unit, order = lesion_order(clone, 'hub', np.random.default_rng(0))
    info = apply_lesion(clone, unit, order[:5])
    assert info['neurons_silenced'] == 5 and info['edges_removed'] > 0
    assert clone.brain.synapses.size < eng.brain.synapses.size
    assert json.dumps(ex._jsonable(eng.export_state()), sort_keys=True) == before

    r = run_lesion_assay(kinds=('intact', 'random', 'hub', 'region', 'memory_pathway', 'cross_region'),
                         fractions=(0.6, 0.1, 0.3), seeds=(0, 1), neurons=24, steps=10, recovery_steps=5,
                         heldout_steps=5)
    assert verify_receipt(r) and r['originals_untouched'] is True and r['progressive_monotone_damage'] is True
    assert r['config']['fractions'] == [0.1, 0.3, 0.6]
    for seed in (0, 1):
        for kind in ('random', 'hub', 'region', 'memory_pathway', 'cross_region'):
            rs = sorted((x for x in r['records'] if x['seed'] == seed and x['kind'] == kind), key=lambda x: x['fraction'])
            removed = [x['edges_removed'] for x in rs]
            eff = [x['connectivity']['global_efficiency'] for x in rs]
            assert removed == sorted(removed) and removed[-1] > 0
            assert all(b <= a + 1e-12 for a, b in zip(eff, eff[1:]))
    intact = [x for x in r['records'] if x['kind'] == 'intact']
    assert all(x['edges_removed'] == 0 and x['accuracy'] == x['intact_accuracy'] for x in intact)
    for x in r['records']:
        assert x['recovery_time'] is None or 1 <= x['recovery_time'] <= 5
        assert x['functional_redundancy'] is None or 0.0 <= x['functional_redundancy'] <= 1.0
    with pytest.raises(ValueError):
        run_lesion_assay(kinds=('bogus',), seeds=(0,), steps=4, recovery_steps=2)


# ---------------------------------------------------------------- promotion
def _improved(receipt):
    """Test fixture: a receipt edited to show a held-out improvement, then re-sealed with a valid digest."""
    r = copy.deepcopy(receipt)
    r['verdict'] = 'improved'
    r['comparisons']['candidate_vs_baseline']['verdict'] = 'improved'
    r['comparisons']['candidate_vs_baseline']['primary']['ci95'] = [0.01, 0.05]
    r['digest'] = receipt_digest(r)
    return r


def test_promotion_gate_atomic_write_rollback_and_archive(receipts, tmp_path, monkeypatch):
    by = {r['experiment_id']: r for r in receipts}
    path = tmp_path / 'ledger.json'
    led = PromotionLedger(path)

    # real quick-mode evidence is not an improvement -> failed and archived, never deleted
    weak = by['context_attention']
    assert weak['verdict'] != 'improved'
    cand = led.propose(['context_attention'], [weak])
    ev = led.evaluate(cand['id'], compatibility_tests_passed=True)
    assert not ev['passed'] and any('not \'improved\'' in m for m in ev['reasons'])
    assert led.archive[-1]['id'] == cand['id'] and led.archive[-1]['receipts'][0]['digest'] == weak['digest']
    with pytest.raises(KeyError):
        led.candidate(cand['id'])

    good = _improved(by['context_attention'])
    # compatibility flag is mandatory
    c0 = led.propose(['context_attention'], [good])
    assert not led.evaluate(c0['id'], compatibility_tests_passed=False)['passed']
    # a tampered receipt fails the digest check
    bad = copy.deepcopy(good)
    bad['comparisons']['candidate_vs_baseline']['primary']['ci95'] = [0.5, 0.9]
    cb = led.propose(['context_attention'], [bad])
    assert any('digest' in m for m in led.evaluate(cb['id'], compatibility_tests_passed=True)['reasons'])
    # a receipt testing mechanisms outside the candidate set fails
    co = led.propose(['stdp'], [good])
    assert any('outside' in m for m in led.evaluate(co['id'], compatibility_tests_passed=True)['reasons'])
    assert len(led.archive) == 4

    c1 = led.propose(['context_attention'], [good])
    with pytest.raises(ValueError):
        led.promote(c1['id'])                                   # not evaluated yet
    assert led.evaluate(c1['id'], compatibility_tests_passed=True)['passed']
    entry = led.promote(c1['id'])
    assert entry['mechanisms'] == ['context_attention']
    assert led.stable_spec().mechanisms == ('context_attention',)
    assert led.stable_spec().enabled is False and led.stable_spec().gain == 0.0   # still opt-in / neutral
    assert not list(tmp_path.glob('*.tmp-*'))

    good2 = _improved(by['multi_soma_integration'])
    c2 = led.propose(['multi_soma', 'extended_somas', 'dendritic_compartments', 'context_attention'], [good2])
    led.evaluate(c2['id'], compatibility_tests_passed=True)

    # atomic: a failing os.replace leaves both the file and the in-memory ledger unchanged
    on_disk = path.read_text(encoding='utf-8')
    snapshot = led.to_dict()

    def boom(*a, **k):
        raise OSError('simulated crash during replace')
    monkeypatch.setattr(os, 'replace', boom)
    with pytest.raises(OSError):
        led.promote(c2['id'])
    monkeypatch.undo()
    assert path.read_text(encoding='utf-8') == on_disk and led.to_dict() == snapshot
    assert not list(tmp_path.glob('*.tmp-*'))

    led.promote(c2['id'])
    assert set(led.stable_spec().mechanisms) == {'multi_soma', 'extended_somas', 'dendritic_compartments',
                                                 'context_attention'}
    restored = led.rollback()
    assert restored['candidate_id'] == c1['id'] and led.stable_spec().mechanisms == ('context_attention',)
    assert led.archive[-1]['status'] == 'rolled_back' and led.archive[-1]['receipts'][0]['digest'] == good2['digest']
    assert led.rollback() is None and STABLE_PRESET not in led.presets
    with pytest.raises(ValueError):
        led.rollback()

    events = [h['event'] for h in led.history]
    assert events.count('promote') == 2 and events.count('rollback') == 2 and events[0] == 'propose'
    reloaded = PromotionLedger(path)
    assert reloaded.to_dict() == led.to_dict()

    from synapse.config import SynapseConfig
    assert SynapseConfig().neuroweave is None and 'neuroweave' not in SynapseConfig().to_dict()
