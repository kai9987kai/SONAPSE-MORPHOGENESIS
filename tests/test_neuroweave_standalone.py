"""Standalone HTML runner: JS/Python frozen-inference parity and embedding safety (small and fast)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from synapse.neuroweave import package as pk
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.spec import MECHANISM_IDS, NeuroWeaveSpecV1
from synapse.neuroweave.standalone import (annotate_package, frozen_reference, script_safe_json,
                                           standalone_capabilities, standalone_html, standalone_model)

NODE = shutil.which('node')
STEPS = 12

_BRAIN_FORWARD = ('multi_soma', 'extended_somas', 'dendritic_compartments', 'membrane_dynamics', 'axonal_branches',
                  'astrocyte_regulation', 'homeostatic_governor', 'predictive_coding', 'hierarchical_temporal_memory',
                  'sparse_code', 'structural_encoder', 'context_attention', 'flyhash_projection', 'synaptic_delays',
                  'adaptive_conduction', 'adaptive_routing', 'uncertainty_gating', 'multi_region_routing')
MECH_SETS = {
    'all_off': ((), 'clustered'),
    'rich': (_BRAIN_FORWARD + ('micro_collective', 'expert_routing', 'reward_plasticity', 'stdp',
                               'neuromodulator_field', 'memory_ecology'), 'dense_multi_region'),
    'hardware': (('quantised_synapses', 'q88_state', 'tiled_execution', 'synaptic_delays', 'membrane_dynamics'),
                 'hub_skewed'),
    'everything': (MECHANISM_IDS, 'multi_region'),
}

NODE_HARNESS = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const inputs = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const grab = (id) => { const m = html.match(new RegExp('<script[^>]*id="' + id + '"[^>]*>([\\s\\S]*?)</script>')); return m[1]; };
const NW = new Function(grab('nw-engine') + '\nreturn NeuroWeaveStandalone;')();
const model = JSON.parse(grab('nw-model'));
let rt = NW.create(model);
if (process.argv[4] !== undefined) NW.setThreshold(rt, Number(process.argv[4]));
const out = inputs.map(([t, s]) => NW.step(rt, t, s));
process.stdout.write(JSON.stringify({results: out, rint: [2.5, 3.5, -2.5, -3.5, 0.5, 1.5].map(NW.rint)}));
"""


def _package(mechs, topology, *, neurons=24, steps=6, seed=5):
    spec = NeuroWeaveSpecV1(enabled=True, seed=seed, neurons=neurons, topology=topology, mechanisms=mechs,
                            structural_interval=3, sleep_interval=4, tile_size=7)
    eng = NeuroWeaveEngine(spec)
    rng = np.random.default_rng(seed)
    for t in range(steps):
        eng.step(rng.uniform(-1, 1, 8), rng.uniform(-1, 1, 16), reward=0.5 if t % 2 else -0.25)
    return eng.export_package()


def _inputs(seed=11):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-1.5, 1.5, 8).tolist(), rng.uniform(-1, 1, 16).tolist()) for _ in range(STEPS)]


def _run_node(tmp_path, html_text, inputs, threshold=None):
    page = tmp_path / 'runner.html'
    page.write_text(html_text, encoding='utf-8')
    harness = tmp_path / 'harness.js'
    harness.write_text(NODE_HARNESS, encoding='utf-8')
    data = tmp_path / 'inputs.json'
    data.write_text(json.dumps(inputs), encoding='utf-8')
    argv = [NODE, str(harness), str(page), str(data)] + ([] if threshold is None else [repr(float(threshold))])
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.skipif(NODE is None, reason='node is not installed')
@pytest.mark.parametrize('name', sorted(MECH_SETS))
def test_js_frozen_step_matches_python(tmp_path, name):
    mechs, topology = MECH_SETS[name]
    pkg = _package(mechs, topology)
    inputs = _inputs()
    ref = frozen_reference(pkg, inputs)
    js = _run_node(tmp_path, standalone_html(pkg), inputs)
    assert js['rint'] == [2, 4, -2, -4, 0, 2]
    got = js['results']
    assert len(got) == len(ref) == STEPS
    worst = 0.0
    for r, g in zip(ref, got):
        assert g['step'] == r['step']
        diff = np.max(np.abs(np.asarray(g['output']) - np.asarray(r['output'])))
        worst = max(worst, float(diff), abs(g['readout_probability'] - r['readout_probability']))
        assert abs(g['uncertainty']['readout_ambiguity'] - r['uncertainty']['readout_ambiguity']) <= 1e-9
        if r['micro'] is None:
            assert g['micro'] is None
        else:
            assert g['micro']['experts'] == r['micro']['experts']
            worst = max(worst, float(np.max(np.abs(np.asarray(g['micro']['prediction']) - r['micro']['prediction']))))
            assert abs(g['uncertainty']['router_entropy'] - r['uncertainty']['router_entropy']) <= 1e-9
        _assert_selective_parity(g['selective'], r['selective'])
    assert worst <= 1e-9, (name, worst)
    # the dynamics must actually move (not a trivially-zero comparison)
    assert np.max(np.abs(ref[-1]['output'])) > 1e-3


def test_reference_equals_brain_frozen_step_and_is_non_mutating():
    pkg = _package(MECH_SETS['rich'][0], 'dense_multi_region')
    snap = json.dumps(pkg, sort_keys=True)
    a = frozen_reference(pkg, _inputs())
    b = frozen_reference(pkg, _inputs())
    assert a == b and json.dumps(pkg, sort_keys=True) == snap


def test_capabilities_list_every_enabled_mechanism():
    pkg = _package(MECHANISM_IDS, 'multi_region')
    ann = annotate_package(pkg)
    assert pk.validate_package(json.loads(pk.dumps_package(ann)))['valid']
    st = ann['capabilities']['frozen_inference']['standalone']
    listed = set(st['executed']) | {x['id'] for x in st['partial']} | {x['id'] for x in st['not_executed']}
    assert listed == set(MECHANISM_IDS)
    assert {'stdp', 'neuromodulator_field', 'specialist_graft'} <= {x['id'] for x in st['not_executed']}
    assert 'micro_collective' in st['executed'] and st['runtime_capabilities']['frozen_inference']['status'] == 'supported'
    assert st['runtime_capabilities']['continued_learning']['status'] == 'not_available'
    html_text = standalone_html(ann)
    section = html_text.split('id="nw-not-executed"', 1)[1]
    for item in st['not_executed'] + st['partial']:
        assert item['id'] in section
    off = standalone_capabilities((), has_micro=False)
    assert off['executed'] == [] and off['not_executed'] == [] and off['partial'] == []


def test_all_off_model_has_no_flags_and_page_is_offline():
    pkg = _package((), 'clustered', steps=2)
    model = standalone_model(pkg)
    assert model['flags'] == [] and model['hash'] is None and model['structural_weight'] is None
    page = standalone_html(pkg)
    assert 'http://' not in page and 'https://' not in page and 'getUserMedia' not in page
    assert 'Every enabled mechanism is executed' in page


def test_script_safe_json_escapes_breakouts():
    hostile = {'x': '</script><script>alert(1)</script>', 'y': 'a\u2028b\u2029c', 'z': '<!-- & -->'}
    text = script_safe_json(hostile)
    assert '<' not in text and '>' not in text and '&' not in text
    assert '\u2028' not in text and '\u2029' not in text
    assert json.loads(text) == hostile


def test_hostile_package_text_cannot_inject_markup():
    pkg = _package((), 'clustered', steps=2)
    evil = '</script><script>alert("x")</script><img src=x onerror=alert(1)>\u2028'
    pkg['capabilities']['frozen_inference']['explanation'] = evil
    pkg.pop('integrity')
    pkg['integrity'] = {'algorithm': 'sha256', 'fingerprint': pk.package_fingerprint(pkg)}
    assert pk.validate_package(pkg)['valid']
    page = standalone_html(pkg)
    assert '<script>alert' not in page and '<img src=x' not in page and '\u2028' not in page
    assert '&lt;/script&gt;&lt;script&gt;alert(&quot;x&quot;)' in page
    assert page.count('</script>') == 3  # exactly the three page scripts
    data = re.search(r'<script type="application/json" id="nw-model">(.*?)</script>', page, re.S).group(1)
    assert json.loads(data)['meta']['package_capabilities']['frozen_inference']['explanation'] == evil


def test_rejects_invalid_or_tampered_packages():
    pkg = _package((), 'clustered', steps=1)
    with pytest.raises(ValueError):
        standalone_html({'format': 'something_else'})
    tampered = json.loads(json.dumps(pkg))
    tampered['brain']['readout']['b'] = 0.123
    with pytest.raises(ValueError, match='fingerprint'):
        standalone_html(tampered)


def _assert_selective_parity(js_sel, py_sel):
    assert js_sel['label'] == py_sel['label'] == 'heuristic, not calibrated'
    for key in ('abstain', 'decision', 'risk_source', 'threshold'):
        assert js_sel[key] == py_sel[key], key
    for key in ('score', 'margin', 'risk', 'raw_probability'):
        assert abs(js_sel[key] - py_sel[key]) <= 1e-9, key


def test_selective_outcome_rules():
    from synapse.neuroweave.standalone import default_selective_threshold, selective_outcome
    sure = selective_outcome(0.95, None, 0.12)
    assert sure['abstain'] is False and sure['decision'] == 'predict' and sure['risk_source'] == 'none'
    assert sure['score'] == pytest.approx(0.9) and sure['raw_probability'] == 0.95
    unsure = selective_outcome(0.52, None, 0.12)
    assert unsure['abstain'] is True and unsure['decision'] == 'abstain_review' and unsure['raw_probability'] == 0.52
    risky = selective_outcome(0.95, 0.9, 0.12)
    assert risky['risk'] == pytest.approx(0.9) and risky['score'] == pytest.approx(0.09) and risky['abstain'] is True
    assert selective_outcome(0.5, 2.0, 0.0)['abstain'] is False      # threshold 0 never abstains
    assert selective_outcome(0.9, -1.0, 0.1)['risk'] == 0.0
    for bad in (-0.1, 1.5, float('nan')):
        with pytest.raises(ValueError):
            selective_outcome(0.5, None, bad)
    assert default_selective_threshold(['uncertainty_gating']) == 0.12 and default_selective_threshold([]) == 0.0


def test_page_shows_raw_probability_and_heuristic_selective_label():
    pkg = _package(MECH_SETS['rich'][0], 'dense_multi_region')
    model = standalone_model(pkg)
    assert model['selective']['threshold'] == 0.12 and model['selective']['label'] == 'heuristic, not calibrated'
    assert standalone_model(_package((), 'clustered', steps=1))['selective']['threshold'] == 0.0
    page = standalone_html(pkg)
    for needle in ('id="nw-prob"', 'id="nw-selective"', 'id="nw-decision"', 'id="nw-abstain-threshold"',
                   'heuristic, not calibrated', 'Raw readout probability (always shown)', 'ABSTAIN'):
        assert needle in page
    ref = frozen_reference(pkg, _inputs())
    assert all(r['selective']['raw_probability'] == r['readout_probability'] for r in ref)


@pytest.mark.skipif(NODE is None, reason='node is not installed')
@pytest.mark.parametrize('threshold', [0.0, 0.5, 0.97])
def test_js_selective_score_matches_python_with_threshold_control(tmp_path, threshold):
    mechs, topology = MECH_SETS['rich']
    pkg = _package(mechs, topology)
    inputs = _inputs(seed=23)
    ref = frozen_reference(pkg, inputs, selective_threshold=threshold)
    got = _run_node(tmp_path, standalone_html(pkg), inputs, threshold=threshold)['results']
    for r, g in zip(ref, got):
        _assert_selective_parity(g['selective'], r['selective'])
        assert g['readout_probability'] == pytest.approx(r['readout_probability'], abs=1e-9)
    if threshold == 0.0:
        assert not any(r['selective']['abstain'] for r in ref)
    if threshold == 0.97:
        assert any(r['selective']['abstain'] for r in ref)
