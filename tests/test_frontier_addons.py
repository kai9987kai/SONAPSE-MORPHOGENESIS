"""Frontier add-ons: experiment_design (decision-sensitive selection) and morphogenesis (growth + regeneration)."""
import copy
import math

import numpy as np
import pytest

from synapse.frontier import experiment_design as ed
from synapse.frontier import morphogenesis as mg
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, LAB_SCHEMA
from synapse.neuroweave.topology import topology_metrics

ED_TINY = dict(quick=True, seeds=(0, 1), studies=8, budget=2)
MG_TINY = dict(quick=True, seeds=(0, 1), size=12, generations=2, offspring=3, lesions_per_seed=2)


# ============================================================== experiment_design

def _divergent_problem(**kw):
    """Experiment 0 separates hypotheses inside each decision group (1 bit, decision-useless);
    experiment 1 separates the groups (less information, decision-relevant)."""
    return ed.DecisionProblem([0.25] * 4, [[0.0, 0.0], [10.0, 0.0], [0.0, 2.0], [10.0, 2.0]], 1.0,
                              [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]], **kw)


def test_decision_risk_and_information_gain_disagree_in_the_expected_direction():
    p = _divergent_problem()
    rr, ig = p.risk_reduction(p.prior), p.information_gain(p.prior)
    assert rr[0] == pytest.approx(0.0, abs=1e-9) and rr[1] > 0.3
    assert ig[0] == pytest.approx(math.log(2.0), abs=1e-3) and ig[1] < ig[0]
    assert ed.DecisionSensitiveSelector(p).choose(p.prior) == 1
    assert ed.InformationGainSelector(p).choose(p.prior) == 0


def test_two_hypothesis_case_picks_the_informative_experiment_and_abstains_when_nothing_helps():
    loss = [[0.0, 1.0], [1.0, 0.0]]
    p = ed.DecisionProblem([0.5, 0.5], [[0.0, 0.0], [0.0, 3.0]], 1.0, loss)
    assert ed.DecisionSensitiveSelector(p).choose(p.prior) == 1
    flat = ed.DecisionProblem([0.5, 0.5], [[1.0, -2.0], [1.0, -2.0]], 1.0, loss)
    assert ed.DecisionSensitiveSelector(flat).choose(flat.prior) is None
    assert ed.InformationGainSelector(flat).choose(flat.prior) is None
    assert ed.make_selector('decision_risk', flat, abstain=False).choose(flat.prior) == 0


def test_costs_trigger_abstention_and_reduce_the_net_score():
    loss = [[0.0, 1.0], [1.0, 0.0]]
    free = ed.DecisionProblem([0.5, 0.5], [[0.0], [3.0]], 1.0, loss)
    value = float(free.risk_reduction(free.prior)[0])
    pricey = ed.DecisionProblem([0.5, 0.5], [[0.0], [3.0]], 1.0, loss, costs=[value + 0.01])
    cheap = ed.DecisionProblem([0.5, 0.5], [[0.0], [3.0]], 1.0, loss, costs=[value / 2])
    assert ed.DecisionSensitiveSelector(pricey).choose(pricey.prior) is None
    assert ed.DecisionSensitiveSelector(cheap).choose(cheap.prior) == 0
    assert ed.DecisionSensitiveSelector(cheap).scores(cheap.prior)[0] == pytest.approx(value / 2)


def test_bayes_update_matches_closed_form_and_value_bounds_hold():
    p = ed.DecisionProblem([0.3, 0.7], [[0.0], [2.0]], [1.5], [[0.0, 1.0], [2.0, 0.0]])
    y = 0.8
    l0 = math.exp(-0.5 * (y / 1.5) ** 2)
    l1 = math.exp(-0.5 * ((y - 2.0) / 1.5) ** 2)
    expected = np.array([0.3 * l0, 0.7 * l1]) / (0.3 * l0 + 0.7 * l1)
    assert np.allclose(p.update(p.prior, 0, y), expected)
    study = ed.make_study(3)
    w = study.prior
    rr, ig = study.risk_reduction(w), study.information_gain(w)
    assert np.all(rr >= 0.0) and np.all(rr <= study.bayes_risk(w) + 1e-9)
    entropy = -float(np.sum(w * np.log(w)))
    assert np.all(ig >= 0.0) and np.all(ig <= entropy + 1e-6)
    assert np.all(np.isfinite(p.update(p.prior, 0, 1e6)))            # extreme outcome stays finite


def test_problem_is_immutable_and_validates_inputs():
    p = _divergent_problem()
    for arr in (p.prior, p.means, p.sigmas, p.loss, p.costs):
        with pytest.raises(ValueError):
            arr[0] = 1.0
    good = dict(prior=[0.5, 0.5], means=[[0.0], [1.0]], sigmas=1.0, loss=[[0.0, 1.0], [1.0, 0.0]])
    bad_cases = [dict(prior=[-1.0, 2.0]), dict(prior=[0.0, 0.0]), dict(prior=[1.0]), dict(means=[[0.0], [np.nan]]),
                 dict(sigmas=0.0), dict(sigmas=[1.0, 2.0]), dict(loss=[[0.0, 1.0, 2.0]]), dict(means=[0.0, 1.0])]
    for case in bad_cases:
        with pytest.raises(ValueError):
            ed.DecisionProblem(**{**good, **case})
    with pytest.raises(ValueError):
        ed.DecisionProblem(**good, costs=[-0.1])
    with pytest.raises(ValueError):
        ed.DecisionProblem(**good, quadrature=1)
    prob = ed.DecisionProblem(**good)
    with pytest.raises(ValueError):
        prob.update(prob.prior, 5, 0.0)
    with pytest.raises(ValueError):
        prob.update([1.0, 2.0, 3.0], 0, 0.0)
    with pytest.raises(ValueError):
        prob.update(prob.prior, 0, float('inf'))
    with pytest.raises(ValueError):
        ed.make_selector('oracle', prob)
    with pytest.raises(ValueError):
        ed.DecisionSensitiveSelector(prob, abstain_threshold=-1.0)


def test_study_generator_and_simulation_are_deterministic_and_bounded():
    a, b = ed.make_study(11), ed.make_study(11)
    assert a.to_dict() == b.to_dict() and a.to_dict() != ed.make_study(12).to_dict()
    for bad in (dict(hypotheses=1), dict(decisions=9, hypotheses=4), dict(distractors=8), dict(noise=0.0),
                dict(cost_range=(0.2, 0.1))):
        with pytest.raises(ValueError):
            ed.make_study(0, **bad)
    with pytest.raises(ValueError):
        ed.make_study(-1)
    z = np.random.default_rng(0).standard_normal(3)
    runs = [ed.simulate_study(a, 2, ed.DecisionSensitiveSelector(a), 3, z) for _ in range(2)]
    assert runs[0] == runs[1]
    assert len(runs[0]['regret_curve']) == 4 and all(r >= 0.0 for r in runs[0]['regret_curve'])
    r1 = ed.simulate_study(a, 2, ed.RandomSelector(a, seed=5), 3, z)
    r2 = ed.simulate_study(a, 2, ed.RandomSelector(a, seed=5), 3, z)
    assert r1['experiments'] == r2['experiments']
    with pytest.raises(ValueError):
        ed.simulate_study(a, 99, ed.DecisionSensitiveSelector(a), 3, z)
    with pytest.raises(ValueError):
        ed.simulate_study(a, 0, ed.DecisionSensitiveSelector(a), 3, z[:2])


def test_abstention_stops_spending():
    loss = [[0.0, 1.0], [1.0, 0.0]]
    flat = ed.DecisionProblem([0.9, 0.1], [[0.0], [0.0]], 1.0, loss, costs=[0.5])
    res = ed.simulate_study(flat, 0, ed.DecisionSensitiveSelector(flat), 4, np.zeros(4))
    assert res['abstained_at'] == 0 and res['experiments'] == [] and res['cost'] == 0.0


@pytest.mark.parametrize('runner', [ed.run_experiment, ed.run_costed_experiment])
def test_experiment_design_receipts_are_sealed_and_deterministic(runner):
    first, second = runner(**ED_TINY), runner(**ED_TINY)
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['schema'] == LAB_SCHEMA and first['evidence_tier'] == EVIDENCE_TIER
    assert first['verdict'] in VERDICTS and first['verdict_basis']
    assert first['config']['higher_is_better'] is False
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'improved' if first['verdict'] != 'improved' else 'worse'
    assert not verify_receipt(tampered)
    for arm in first['arms'].values():
        s = arm['summary']
        assert len(s['mean_regret_by_budget']) == ED_TINY['budget'] + 1
        assert 0.0 <= s['decision_accuracy'] <= 1.0 and s['final_regret'] >= 0.0


def test_experiment_design_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ed.run_experiment(quick=1)
    with pytest.raises(ValueError):
        ed.run_experiment(quick=True, seeds=(0, 0))
    with pytest.raises(ValueError):
        ed.run_experiment(quick=True, studies=1)
    with pytest.raises(ValueError):
        ed.run_experiment(quick=True, warp=3)
    with pytest.raises(ValueError):
        ed.run_costed_experiment(quick=True, cost_low=0.5, cost_high=0.1)
    assert list(ed.EXPERIMENTS)[0] == ed.MODULE_ID
    assert all(k == ed.MODULE_ID or k.startswith(ed.MODULE_ID + '_') for k in ed.EXPERIMENTS)


# ============================================================== morphogenesis

def test_growth_is_deterministic_and_connected_to_the_organizer():
    prog = mg.DevelopmentalProgram()
    a, b = mg.grow(prog, 16, seed=3), mg.grow(prog, 16, seed=3)
    assert np.array_equal(a.alive, b.alive) and np.array_equal(a.cell_type, b.cell_type)
    assert a.alive[a.organizer] and a.cells > 1
    assert np.all((a.cell_type == mg.EMPTY) == ~a.alive)
    assert set(np.unique(a.cell_type[a.alive]).tolist()) <= set(range(len(mg.CELL_TYPES)))


def test_grown_topology_is_valid_and_metrics_are_computable():
    body = mg.grow(mg.DevelopmentalProgram(), 16, seed=0)
    topo = mg.body_to_topology(body)
    topo.validate()
    assert topo.n == body.cells and topo.edges > 0
    assert np.all(topo.src != topo.dst)
    keys = topo.src.astype(np.int64) * topo.n + topo.dst
    assert np.unique(keys).size == keys.size
    assert topo.delay.min() >= 1 and topo.delay.max() <= 3
    assert set(np.unique(topo.region).tolist()) <= set(mg.TYPE_REGION)
    m = topology_metrics(topo)
    assert m['n'] == topo.n and np.isfinite(m['global_efficiency']) and 0.0 <= m['density'] <= 1.0
    assert mg.edge_jaccard(body, body) == 1.0


def test_lesion_and_memory_controls():
    body = mg.grow(mg.DevelopmentalProgram(), 16, seed=0)
    oy, ox = body.organizer
    hurt = mg.lesion(body, (oy - 2, oy + 3, ox - 2, ox + 3))
    assert not hurt.alive[oy, ox] and hurt.cells < body.cells
    mem = mg.make_memory(body, block=2)
    ctrl = mg.shuffled_memory(mem, seed=1)
    assert ctrl.numbers == mem.numbers and ctrl.setpoint.shape == mem.setpoint.shape
    assert np.isclose(ctrl.setpoint.sum(), mem.setpoint.sum())
    assert mem.field(16).shape == (16, 16) and 0.0 <= mem.field(16).min() and mem.field(16).max() <= 1.0
    with pytest.raises(ValueError):
        mem.setpoint[0, 0] = 1.0
    with pytest.raises(ValueError):
        mg.lesion(body, (5, 5, 0, 3))
    with pytest.raises(ValueError):
        mg.make_memory(body, block=0)


def test_regrowth_with_pattern_memory_recovers_better_after_losing_the_organizer():
    size = 16
    body = mg.grow(mg.DevelopmentalProgram(), size, seed=0)
    oy, ox = body.organizer
    damaged = mg.lesion(body, (0, size, 0, ox + 2))            # remove the left half including the organizer
    assert not damaged.alive[oy, ox]
    mem = mg.make_memory(body)
    with_mem = mg.regrow(mg.DevelopmentalProgram(), damaged, memory=mem, seed=0)
    without = mg.regrow(mg.DevelopmentalProgram(), damaged, memory=None, seed=0)
    assert mg.iou(with_mem.alive, body.alive) > mg.iou(without.alive, body.alive)
    again = mg.regrow(mg.DevelopmentalProgram(), damaged, memory=mem, seed=0)
    assert np.array_equal(again.alive, with_mem.alive)


def test_evolution_never_gets_worse_and_is_deterministic():
    tgt = mg.target_mask('ellipse', 12)
    a = mg.evolve_program(tgt, seed=1, generations=3, offspring=3)
    b = mg.evolve_program(tgt, seed=1, generations=3, offspring=3)
    assert a['trace'] == b['trace'] and a['program'] == b['program']
    assert all(y >= x for x, y in zip(a['trace'], a['trace'][1:], strict=False))
    assert a['evaluations'] == 1 + 3 * 3 and 0.0 <= a['fitness'] <= 1.0


def test_morphogenesis_validation():
    with pytest.raises(ValueError):
        mg.DevelopmentalProgram(theta=(1.0, 2.0))
    with pytest.raises(ValueError):
        mg.DevelopmentalProgram(theta=(np.nan, 0, 0, 0, 0))
    with pytest.raises(ValueError):
        mg.DevelopmentalProgram(p_update=0.0)
    with pytest.raises(ValueError):
        mg.grow(mg.DevelopmentalProgram(), 4)
    with pytest.raises(ValueError):
        mg.grow(mg.DevelopmentalProgram(), 16, steps=0)
    with pytest.raises(ValueError):
        mg.target_mask('blob', 16)
    with pytest.raises(ValueError):
        mg.iou(np.zeros((3, 3)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        mg.run_experiment(quick=True, warp=1)
    with pytest.raises(ValueError):
        mg.run_experiment(quick=True, seeds=())
    with pytest.raises(ValueError):
        mg.run_experiment(quick=True, lesion_fraction=0.9)


def test_morphogenesis_receipt_is_sealed_and_deterministic():
    first, second = mg.run_experiment(**MG_TINY), mg.run_experiment(**MG_TINY)
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['schema'] == LAB_SCHEMA and first['evidence_tier'] == EVIDENCE_TIER
    assert first['verdict'] in VERDICTS and first['verdict_basis']
    assert set(first['arms']) == {'pattern_memory', 'no_memory', 'shuffled_memory'}
    for arm in first['arms'].values():
        assert 0.0 <= arm['summary']['iou'] <= 1.0
    assert first['parameter_budget']['pattern_memory_numbers'] == first['parameter_budget']['shuffled_memory_numbers']
    tampered = copy.deepcopy(first)
    tampered['arms']['no_memory']['summary']['iou'] = 0.0
    assert not verify_receipt(tampered)
    assert list(mg.EXPERIMENTS) == [mg.MODULE_ID]
