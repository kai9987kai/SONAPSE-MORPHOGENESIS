"""Regression tests for the NEUROWEAVE review findings (small and fast)."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from synapse.neuroweave import experiments as ex
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.memory import MemoryEcology
from synapse.neuroweave.spec import NeuroWeaveSpecV1


# ------------------------------------------------------------------ server: reset keeps NEUROWEAVE
def test_web_config_reset_preserves_an_active_neuroweave_profile(tmp_path):
    from synapse.config import SynapseConfig
    from synapse.server import Workbench

    workbench = Workbench.__new__(Workbench)
    workbench.root = tmp_path
    workbench.runtime = SimpleNamespace(config=SynapseConfig(
        nodes_per_source=16,
        neuroweave=NeuroWeaveSpecV1(enabled=True, neurons=16, max_neurons=24, seed=3),
    ))
    workbench.receipts = {}
    workbench.episode_recorder = object()
    workbench.lock = threading.RLock()
    workbench.aux_lock = threading.RLock()

    result = workbench.post('/api/reset', {'config': {'nodes_per_source': 16}})

    assert result['config']['neuroweave']['enabled'] is True
    assert workbench.runtime.agent.neuroweave_channel is not None
    assert workbench.runtime.agent.neuroweave_channel.spec.enabled is True


# ------------------------------------------------------------------ memory: unconfirmed predictions
def test_unconfirmed_predicted_stores_never_change_verified_items_or_traces():
    m = MemoryEcology(4, 2, seed=5, interference_value_tolerance=0.0, interference_penalty=1.0)
    key = np.array([1.0, 0.5, -0.2, 0.3])
    vid = m.store(key, [0.5, 0.5], provenance='sensory')
    conf = m.get(vid)['confidence']
    traces = {n: t.copy() for n, t in m.traces.items()}
    count, events = m.trace_count, m.counters['interference_events']
    surprise = m.default_surprise([0.5, 0.5])
    for _ in range(30):
        m.store(key, [-0.5, -0.5], provenance='predicted')
    assert m.get(vid)['confidence'] == conf
    for n in traces:
        np.testing.assert_array_equal(m.traces[n], traces[n])
    assert m.trace_count == count and m.counters['interference_events'] == events
    assert m.default_surprise([0.5, 0.5]) == surprise
    # a verified contradicting observation still interferes
    m.store(key, [-0.5, -0.5], provenance='sensory')
    assert m.get(vid)['confidence'] < conf


# ------------------------------------------------------------------ engine: flags that must act
def _run(mechs, steps=40, seed=4):
    e = NeuroWeaveEngine(NeuroWeaveSpecV1(enabled=True, neurons=24, max_neurons=32, seed=seed,
                                          mechanisms=tuple(mechs)))
    rng = np.random.default_rng(1)
    outs = []
    for i in range(steps):
        r = e.step(rng.uniform(-1, 1, 8), reward=(1.0 if i % 3 == 0 else -0.5))
        outs.append(np.asarray(r['output'] if isinstance(r, dict) and 'output' in r else e.brain.cells.output,
                               dtype=float).copy())
    return e, np.stack(outs)


def test_receptor_heterogeneity_changes_engine_dynamics_with_neuromodulation():
    base = ('multimodal_sensory', 'neuromodulator_field', 'affective_modulation')
    _, a = _run(base)
    e, b = _run(base + ('receptor_heterogeneity',))
    assert np.max(np.abs(a - b)) > 0.0
    g = e.neuromod.per_unit_gain(e.flags)
    assert g.shape == (e.n,) and np.all(np.isfinite(g))


def test_predictive_ensemble_controls_ensemble_size_and_gating():
    on = NeuroWeaveEngine(NeuroWeaveSpecV1(enabled=True, neurons=16, max_neurons=24, seed=2,
                                           mechanisms=('predictive_ensemble', 'uncertainty_gating')))
    off = NeuroWeaveEngine(NeuroWeaveSpecV1(enabled=True, neurons=16, max_neurons=24, seed=2,
                                            mechanisms=('uncertainty_gating',)))
    assert on.prediction.ensemble == 3 and off.prediction.ensemble == 1
    _, a = _run(('multimodal_sensory', 'uncertainty_gating'))
    _, b = _run(('multimodal_sensory', 'uncertainty_gating', 'predictive_ensemble'))
    assert np.max(np.abs(a - b)) > 0.0


# ------------------------------------------------------------------ statistics
def test_degenerate_ci_never_yields_a_directional_verdict():
    st = ex.paired_stats([0.75, 0.8125], [0.6875, 0.75])
    assert st['sd'] == 0.0
    v = ex.verdict_from_stats(st)
    assert v == {'verdict': 'no_detectable_difference', 'reason': 'degenerate_ci'}
    assert ex.verdict_from_stats(ex.paired_stats([0.6, 0.7, 0.8], [0.3, 0.3, 0.3]))['verdict'] == 'improved'


def test_topology_best_of_selection_uses_family_wise_adjustment():
    # passes an unadjusted 95 % t-interval but not a Bonferroni(10) one
    st = ex.paired_stats([0.30, 0.42, 0.36], [0.10, 0.10, 0.10])
    assert ex.verdict_from_stats(st)['verdict'] == 'improved'
    v = ex._bonferroni_verdict(st, 10)
    assert v['verdict'] == 'no_detectable_difference' and v['family_size'] == 10
    assert v['ci95_family_adjusted'][0] < st['ci95'][0]
    # null simulation: best-of-11 selection stays near the nominal one-sided rate
    rng = np.random.default_rng(0)
    hits = 0
    trials = 400
    for _ in range(trials):
        diff = rng.normal(0, 0.05, size=3)
        accs = [0.5 + diff + rng.normal(0, 0.03, size=3) for _ in range(11)]
        ref = 0.5 + diff + rng.normal(0, 0.03, size=3)
        best = max(range(11), key=lambda i: accs[i].mean())
        hits += ex._bonferroni_verdict(ex.paired_stats(accs[best], ref), 11)['verdict'] == 'improved'
    assert hits / trials <= 0.04


# ------------------------------------------------------------------ experiments: support + inert guard
def test_experiment_support_gives_named_mechanisms_a_consumer():
    assert 'memory_ecology' in ex.EXPERIMENT_SUPPORT['predictive_replay']
    assert 'reward_plasticity' in ex.EXPERIMENT_SUPPORT['action_self_model']
    assert 'neuromodulator_field' in ex.EXPERIMENT_SUPPORT['predictive_coding']


def test_inert_ablation_is_flagged_not_reported_as_a_verdict():
    r = ex.run_experiment('associative_memory', seeds=(0, 1), heldout_seeds=(100,), steps=8, neurons=16,
                          heldout_steps=4)
    assert ex.verify_receipt(r)
    # associative_memory / fast_weights are declared-only: their minus arms reproduce the candidate exactly
    assert {'associative_memory', 'fast_weights'} <= set(r['inert_mechanisms'])
    for m in ('associative_memory', 'fast_weights'):
        assert r['ablation_verdicts'][m] == 'inert_in_this_configuration'
        c = r['comparisons'][f'candidate_vs_minus_{m}']
        assert c['inert'] is True and c['verdict'] == 'no_detectable_difference'
    assert 'memory_ecology' not in r['inert_mechanisms']


# ------------------------------------------------------------------ package: bounded validation output
def test_hostile_legacy_package_report_is_bounded():
    from synapse.neuroweave.package import MAX_REPORT_ISSUES, inspect_package
    nodes = [{}] * 60_000
    pkg = {'format': 'NEUROWEAVE_MODEL_PACKAGE', 'formatVersion': '1.0.0',
           'network': {'neurons': nodes, 'synapses': [], 'neuronCount': len(nodes), 'synapseCount': 0}}
    text = json.dumps(pkg)
    t0 = time.perf_counter()
    rep = inspect_package(text)
    assert time.perf_counter() - t0 < 5.0
    assert rep['valid'] is False
    assert len(rep['errors']) <= MAX_REPORT_ISSUES + 1
    assert any('safe import limit' in e for e in rep['errors'])
    # within the limits, per-neuron issues are still reported but capped
    small = {**pkg, 'network': {'neurons': [{}] * 400, 'synapses': [], 'neuronCount': 400, 'synapseCount': 0}}
    rep2 = inspect_package(json.dumps(small))
    assert len(rep2['errors']) == MAX_REPORT_ISSUES + 1 and rep2['errors'][-1].endswith('truncated')


# ------------------------------------------------------------------ workbench: lab cost knobs + export note
class _Runtime:
    def snapshot(self): return {'step': 0}
    def step(self, steps=1, learn=False): return {'step': steps}


def test_lab_configure_bounds_cost_knobs_and_export_reports_inspectability(tmp_path):
    from synapse.server import Workbench
    wb = Workbench(_Runtime(), output_root=tmp_path)
    ok = NeuroWeaveSpecV1.all_mechanisms(enabled=True, neurons=24, max_neurons=32, seed=5).to_dict()
    wb.post('/api/neuroweave/configure', {'spec': ok})
    for knob, value in (('structural_interval', 1), ('sleep_interval', 1), ('fan_in', 64), ('micro_experts', 64),
                        ('tile_size', 1), ('max_synapses', 250_000)):
        bad = {**ok, knob: value}
        if knob == 'micro_experts':
            bad['micro_top_k'] = 1
        with pytest.raises(ValueError):
            wb.post('/api/neuroweave/configure', {'spec': bad})
    # the max preset at the lab population limit is still accepted
    wb.post('/api/neuroweave/configure', {'spec': NeuroWeaveSpecV1.max_preset(neurons=64).to_dict()})
    wb.post('/api/neuroweave/configure', {'spec': ok})
    out = wb.post('/api/neuroweave/export', {'kind': 'package', 'source': 'lab'})
    assert out['inspectable_in_workbench'] is True and 'inspect_limit_note' not in out


# ------------------------------------------------------------------ checkpoints: non-finite strings
def test_checkpoint_import_rejects_non_finite_or_string_scalars():
    import torch  # noqa: F401
    from synapse.neuroweave.bridge import NeuroWeaveBusChannel, _canonical_digest
    ch = NeuroWeaveBusChannel(NeuroWeaveSpecV1(enabled=True, neurons=16, max_neurons=24, seed=3), input_dim=32)
    good = ch.export_state()
    for value in ('Infinity', 'nan', float('inf'), True):
        st = json.loads(json.dumps(good))
        st['runtime']['last_current_inf_norm'] = value
        body = {k: v for k, v in st.items() if k != 'digest'}
        st['digest'] = _canonical_digest(body) if not isinstance(value, float) else 'x'
        with pytest.raises(ValueError):
            ch.import_state(st)
    ch.import_state(good)
    json.dumps(ch.export_state(), allow_nan=False)

    e = NeuroWeaveEngine(NeuroWeaveSpecV1(enabled=True, neurons=16, max_neurons=24, seed=3,
                                          mechanisms=('multimodal_sensory', 'representation_fusion')))
    for _ in range(3):
        e.step(np.zeros(8))
    st = e.export_state()
    assert st['runtime']['prev_packet_state'] is not None
    for value in ('nan', 'Infinity', None):
        bad = json.loads(json.dumps(st))
        bad['runtime']['prev_packet_state']['precision'] = value
        with pytest.raises(ValueError):
            NeuroWeaveEngine(e.spec).import_state(bad)
    bad = json.loads(json.dumps(st))
    bad['runtime']['authorship'] = '0.5'
    with pytest.raises(ValueError):
        NeuroWeaveEngine(e.spec).import_state(bad)
