"""Tests for synapse.neuroweave.hardware and synapse.neuroweave.package."""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from synapse.neuroweave import hardware as hw
from synapse.neuroweave import package as pk
from synapse.neuroweave.spec import MAX_POPULATION, MAX_SYNAPSES, MECHANISM_IDS, NeuroWeaveSpecV1


# --------------------------------------------------------------------------- fixtures
def make_brain_state(n: int = 24, edges_per: int = 4, seed: int = 3, spec: NeuroWeaveSpecV1 | None = None) -> dict:
    rng = np.random.default_rng(seed)
    sign = [(-1 if i % 5 == 0 else 1) for i in range(n)]
    pairs = set()
    while len(pairs) < n * edges_per:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            pairs.add((a, b))
    pairs = sorted(pairs)
    src = [a for a, _ in pairs]
    dst = [b for _, b in pairs]
    w = [float(sign[a] * rng.uniform(0.01, 0.9)) for a in src]
    delay = [int(rng.integers(1, 4)) for _ in src]
    spec = spec or NeuroWeaveSpecV1(neurons=max(8, n), max_neurons=max(8, n), seed=seed)
    cells = {name: [float(x) for x in rng.uniform(-1, 1, n)] for name in ('fast', 'slow', 'context', 'membrane', 'output')}
    cells['threshold'] = [0.53] * n
    cells['homeo_gain'] = [float(x) for x in rng.uniform(0.8, 1.2, n)]
    cells['myelin'] = [float(x) for x in rng.uniform(0.55, 0.9, n)]
    return {
        'schema': 'NeuroWeaveBrainStateV1',
        'spec': spec.to_dict(),
        'topology': {'n': n, 'src': list(src), 'dst': list(dst), 'weight': list(w), 'delay': list(delay),
                     'sign': sign, 'region': [i % 10 for i in range(n)], 'name': 'clustered', 'seed': seed},
        'cells': cells,
        'synapses': {'src': list(src), 'dst': list(dst), 'w': list(w), 'delay': list(delay),
                     'trace': [0.0] * len(src), 'eligibility': [0.01] * len(src), 'utility': [0.0] * len(src)},
        'readout': {'w': [float(x) for x in rng.normal(0, 0.1, n)], 'b': 0.05},
        'step': 17,
        'ledger': {'events': []},
    }


NODE_KEYS = ('fast', 'slow', 'context', 'trace', 'membrane', 'calcium', 'act')
LEGACY_REGIONS = ('visual', 'auditory', 'olfactory', 'gustatory', 'thermosensory', 'mechanosensory',
                  'interoceptive', 'association', 'memory', 'motor')


def _legacy_net(n: int, rng: np.random.Generator, full: bool) -> dict:
    nodes = []
    tricky = [1e-7, 0.1 + 0.2, 1.0, 123456789.123, -0.000001, 2.5e-5, 1e21, -0.0]
    for i in range(n):
        node = {'id': i, 'excitatory': i % 5 != 0}
        for j, k in enumerate(NODE_KEYS):
            node[k] = tricky[(i + j) % len(tricky)] if k != 'act' else float(rng.uniform(-1, 1))
        if full:
            node.update({
                'transmitter': 'glutamate' if node['excitatory'] else 'GABA',
                'region': LEGACY_REGIONS[i % 10], 'receptors': {'d1': 0.7, 'd2': 0.6},
                'neuromodLocal': {'da': 0, 'serotonin': 0.5}, 'transmitterPrediction': 'glutamate',
                'transmitterConfidence': 0.4, 'transmitterProbabilities': {'glutamate': 0.4},
                'transmitterGain': 1, 'threshold': 0.53, 'refractory': 0, 'adaptation': 0, 'homeo': 1,
                'astrocyte': 1, 'myelin': 0.6, 'utility': 0, 'consolidation': 0,
                'predictive': {'estimate': 0.1, 'error': 0.02, 'surprise': 0},
                'temporalMemory': {'fast': 0.1, 'mid': 0.05, 'slow': 0.01},
                'dendrites': {'proximal': 0.1, 'distal': 0, 'apical': 0.2, 'basal': 0},
                'soma': {'integrator': 0.3, 'burst': 0, 'gating': 0.5},
                'axon': {'hillock': 0, 'branchA': 0.2, 'branchB': 0.1, 'terminal': 0, 'release': 0}, 'spike': 0,
            })
        nodes.append(node)
    syn = []
    for i in range(n):
        for k in (1, 3):
            j = (i + k) % n
            mag = float(rng.uniform(0.05, 0.9))
            e = {'src': i, 'dst': j, 'weight': mag if nodes[i]['excitatory'] else -mag, 'delay': 1 + (i % 3)}
            if full:
                e.update({'eligibility': 0.01, 'trace': 0.0, 'preSpikeTrace': 0.2, 'postSpikeTrace': 0.1,
                          'routingGain': 1, 'utility': 0.0, 'myelin': 0.6})
            syn.append(e)
    return {'neuronCount': n, 'synapseCount': len(syn), 'neurons': nodes, 'synapses': syn,
            'neuromodulators': {'da': 0, 'serotonin': 0.5}}


def make_legacy(version: str, n: int = 12, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    full = version != '1.0.0'
    features = {k: (i % 3 == 0) for i, k in enumerate(pk.LEGACY_FEATURE_IDS)}
    m = {'format': 'NEUROWEAVE_MODEL_PACKAGE', 'formatVersion': version, 'project': 'NEUROWEAVE LAB',
         'config': {'topology': 'clustered', 'units': n, 'sparsity': 0.15, 'tile': 16, 'seed': 42},
         'features': features, 'network': _legacy_net(n, rng, full),
         'readout': {'weights': [float(x) for x in rng.normal(0, 0.2, 8)], 'bias': 0.1, 'learningRate': 0.03,
                     'inputDimension': 8, 'activation': 'sigmoid'},
         'training': {'steps': 250, 'structuralEvents': 3}}
    if version in ('1.1.0', '1.2.0'):
        m['agentController'] = {'config': {'units': 8}, 'features': {'multiSoma': True},
                                'network': _legacy_net(8, rng, full),
                                'environment': {'x': 10.5, 'y': 20, 'energy': 99.75}}
    if version == '1.2.0':
        m['microCollective'] = {
            'format': 'NEUROWEAVE_MICRO_COLLECTIVE', 'dimension': 16, 'expertCount': 4,
            'experts': [{'w': [[float(x) for x in rng.normal(0, 0.05, 16)] for _ in range(16)],
                         'b': [0.0] * 16} for _ in range(4)],
            'router': [[float(x) for x in rng.normal(0, 0.05, 16)] for _ in range(4)], 'updates': 5}
        m['sensory'] = {'channels': [f'c{i}' for i in range(16)], 'cameraEnabled': False,
                        'microphoneEnabled': False, 'rawMediaStored': False}
    if version in ('1.1.0', '1.2.0'):
        m['integrity'] = {'fingerprint': pk.legacy_fingerprint(m), 'fingerprintAlgorithm': 'FNV-1a-32'}
    return m


# =========================================================================== hardware
def test_quantise_int8_symmetric_and_bounded():
    w = np.array([-0.9, -0.45, 0.0, 0.3, 0.9, 0.0001])
    q, scale = hw.quantise_int8(w)
    assert q.dtype == np.int8 and scale == pytest.approx(0.9 / 127)
    assert q.tolist()[0] == -127 and q.tolist()[4] == 127 and q[2] == 0
    assert np.all(np.abs(q.astype(int)) <= 127)
    deq = hw.dequantise(q, scale)
    assert np.max(np.abs(deq - w)) <= scale / 2 + 1e-12
    stats = hw.quantisation_stats(w, q, scale)
    assert stats['max_abs_error'] <= scale / 2 + 1e-12 and stats['rmse'] > 0
    assert stats['zeroed_nonzero'] == 1 and stats['sign_flips'] == 0 and stats['saturated'] == 2
    # preserve_nonzero keeps the tiny weight alive with the right sign
    q2, _ = hw.quantise_int8(w, preserve_nonzero=True)
    assert q2[5] == 1 and hw.quantisation_stats(w, q2, scale)['zeroed_nonzero'] == 0
    # fixed range saturates
    q3, s3 = hw.quantise_int8([2.0, -2.0, 0.5], max_abs=1.0)
    assert q3.tolist() == [127, -127, 64] and s3 == pytest.approx(1 / 127)
    # all-zero input is well defined
    qz, sz = hw.quantise_int8(np.zeros(4))
    assert sz == 1.0 and not qz.any()


def test_quantise_rejects_bad_input():
    with pytest.raises(ValueError):
        hw.quantise_int8([0.1, float('nan')])
    with pytest.raises(ValueError):
        hw.dequantise([200], 0.1)
    with pytest.raises(ValueError):
        hw.dequantise([1], 0.0)
    with pytest.raises(ValueError):
        hw.quantise_int8([0.1], max_abs=float('inf'))


def test_q88_roundtrip_and_saturation():
    x = np.array([0.0, 1.0, -1.5, 0.00390625, 0.001, 200.0, -300.0, 127.99])
    raw = hw.to_q88(x)
    assert raw.dtype == np.int16
    back = hw.from_q88(raw)
    assert back[1] == 1.0 and back[2] == -1.5 and back[3] == 0.00390625 and back[4] == 0.0
    assert back[5] == pytest.approx(127.99609375) and back[6] == -128.0
    inrange = (x >= hw.Q88_MIN) & (x <= hw.Q88_MAX)
    assert np.all(np.abs(back[inrange] - x[inrange]) <= 1 / 512 + 1e-12)
    assert hw.q88_stats(x)['saturated'] == 2
    assert np.array_equal(hw.q88_round(x), back)
    with pytest.raises(ValueError):
        hw.from_q88([40000])
    with pytest.raises(ValueError):
        hw.to_q88([float('inf')])


def test_tile_schedule_groups_and_covers_edges():
    rng = np.random.default_rng(0)
    n, ts = 40, 16
    src = rng.integers(0, n, 300)
    dst = rng.integers(0, n, 300)
    sched = hw.tile_schedule(src, dst, n, ts, lanes=4)
    assert sorted(sched['edge_order']) == list(range(300))
    keys = [(t['dst_tile'], t['src_tile']) for t in sched['tiles']]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    for t in sched['tiles']:
        idx = np.array(t['edge_indices'])
        assert np.all(dst[idx] // ts == t['dst_tile']) and np.all(src[idx] // ts == t['src_tile'])
        assert 0 < t['utilization'] <= 1
    # exactly one writeback per destination block
    wb = [t['dst_tile'] for t in sched['tiles'] if t['writeback']]
    assert sorted(wb) == sorted(set(dst // ts)) and len(wb) == len(set(wb))
    # pipeline invariants: stages ordered within a tile and per stage across tiles
    prev = None
    for entry in sched['schedule']:
        st = entry['stages']
        assert [s['stage'] for s in st] == list(hw.PIPELINE_STAGES)
        for a, b in zip(st, st[1:]):
            assert b['start'] >= a['end']
        if prev is not None:
            for a, b in zip(prev, st):
                assert b['start'] >= a['end']
        prev = st
    cyc = sched['cycles']
    assert cyc['kind'] == 'estimated' and 0 < cyc['pipelined'] <= cyc['sequential']
    assert cyc['speedup_vs_sequential'] >= 1.0
    # more lanes -> fewer estimated cycles (real dependence on the parameter)
    assert hw.tile_schedule(src, dst, n, ts, lanes=16)['cycles']['pipelined'] < cyc['pipelined']
    # tile size changes the grouping
    assert hw.tile_schedule(src, dst, n, 8)['tile_count'] > sched['tile_count']
    json.dumps(sched)


def test_tile_schedule_rejects_bad_endpoints():
    with pytest.raises(ValueError):
        hw.tile_schedule([0, 5], [1, 2], 5, 4)
    with pytest.raises(ValueError):
        hw.tile_schedule([0], [1], 5, 0)
    empty = hw.tile_schedule([], [], 5, 4)
    assert empty['tile_count'] == 0 and empty['cycles']['pipelined'] == 0


def test_execution_profile_separates_measured_from_estimated():
    est_only = hw.execution_profile(100, 600, steps=10)
    assert est_only['measured'] == {'available': False, 'source': None}
    assert est_only['estimated']['kind'] == 'estimated'
    assert est_only['estimated']['mac_ops'] == 2 * 600 * 10
    sched = hw.tile_schedule([0, 1, 2], [1, 2, 0], 3, 2)
    prof = hw.execution_profile(100, 600, steps=10, wallclock_s=0.5, schedule=sched)
    assert prof['measured']['available'] and prof['measured']['source'] == 'caller_supplied_wallclock'
    assert prof['measured']['latency_ms_per_step'] == pytest.approx(50.0)
    assert prof['measured']['steps_per_s'] == pytest.approx(20.0)
    assert prof['estimated'] == {**est_only['estimated'], 'cycles_per_step': sched['cycles']['pipelined'],
                                 'cycles_total': sched['cycles']['pipelined'] * 10}
    assert 'wallclock_s' not in prof['estimated']
    assert hw.execution_profile(100, 1200)['estimated']['bytes_per_step'] > \
        hw.execution_profile(100, 600)['estimated']['bytes_per_step']
    with pytest.raises(ValueError):
        hw.execution_profile(10, 10, wallclock_s=float('nan'))
    json.dumps(prof)


def test_fpga_manifest_valid_and_honest():
    state = make_brain_state()
    before = copy.deepcopy(state)
    m = hw.fpga_mapping_manifest(state)
    assert state == before  # not mutated
    assert m['format'] == 'SONAPSE_NEUROWEAVE_FPGA_MAPPING' and m['version'] == '1.0.0'
    assert m['rtl_generated'] is False and m['hardware_validated'] is False and m['bitstream'] is None
    assert 'not measurements' in m['disclaimer']
    v = m['validation']
    assert v['endpoints_ok'] and v['signs_ok'] and v['delays_ok'] and v['weights_bounded'] and v['errors'] == []
    assert m['graph']['synapse_count'] == len(state['synapses']['src'])
    # quantised weights correspond to source weights
    q = np.array([e['weight_int8'] for e in m['graph']['edges']])
    assert np.allclose(q * m['scales']['weight'], state['synapses']['w'], atol=m['scales']['weight'] / 2 + 1e-12)
    # q88 hints for selected cell states
    assert set(m['neuron_hints']) == {'threshold', 'homeo_gain', 'myelin', 'membrane'}
    assert m['neuron_hints']['threshold']['q88'][0] == round(0.53 * 256)
    assert m['schedule']['cycles']['kind'] == 'estimated' and m['resource_estimate']['kind'] == 'estimated'
    assert m['schedule']['tile_size'] == state['spec']['tile_size']
    assert hw.validate_fpga_manifest(m)['valid']
    json.dumps(m)
    # manifest reflects current synapse weights, not the initial topology
    state2 = copy.deepcopy(state)
    state2['synapses']['w'] = [x * 0.5 for x in state2['synapses']['w']]
    state2['synapses']['w'][0] *= 0.1
    m2 = hw.fpga_mapping_manifest(state2)
    assert m2['graph']['edges'] != m['graph']['edges'] or m2['scales'] != m['scales']


def test_fpga_validation_catches_sign_flip():
    state = make_brain_state()
    # flip one excitatory synapse to negative in the source brain state
    i = next(k for k, a in enumerate(state['synapses']['src']) if state['topology']['sign'][a] == 1)
    state['synapses']['w'][i] = -abs(state['synapses']['w'][i])
    m = hw.fpga_mapping_manifest(state)
    assert m['validation']['signs_ok'] is False and m['validation']['errors']
    assert not hw.validate_fpga_manifest(m)['valid']
    # tampering with a valid manifest is also detected
    good = hw.fpga_mapping_manifest(make_brain_state())
    good['graph']['edges'][3]['weight_int8'] *= -1
    rep = hw.validate_fpga_manifest(good)
    assert not rep['valid'] and any('sign' in e for e in rep['errors'])


def test_fpga_validation_catches_endpoint_and_flags():
    state = make_brain_state()
    state['synapses']['dst'][0] = state['topology']['n'] + 5
    m = hw.fpga_mapping_manifest(state)
    assert m['validation']['endpoints_ok'] is False
    assert not hw.validate_fpga_manifest(m)['valid']
    good = hw.fpga_mapping_manifest(make_brain_state())
    for key, val in (('rtl_generated', True), ('hardware_validated', True), ('bitstream', 'x.bit')):
        bad = copy.deepcopy(good)
        bad[key] = val
        assert not hw.validate_fpga_manifest(bad)['valid']
    bad = copy.deepcopy(good)
    bad['graph']['edges'][0]['weight_int8'] = 300
    assert not hw.validate_fpga_manifest(bad)['valid']
    bad = copy.deepcopy(good)
    bad['tiles'][0]['edge_indices'] = bad['tiles'][0]['edge_indices'][1:]
    assert not hw.validate_fpga_manifest(bad)['valid']
    assert not hw.validate_fpga_manifest([])['valid']
    with pytest.raises(ValueError):
        hw.fpga_mapping_manifest({'topology': {}})


# =========================================================================== package
def full_package(**kw):
    state = make_brain_state()
    comps = {'memory': {'items': [{'key': [0.1, 0.2], 'provenance': 'experience'}]},
             'sensory': {'channels': ['visual_luminance'], 'raw_media_stored': False},
             'experiments': {'rows': [{'id': 'stdp', 'score': 0.5}]}}
    comps.update(kw.pop('components', {}))
    return state, pk.build_package(state, components=comps, provenance={'run': 'unit'}, **kw)


def test_build_roundtrip_and_determinism():
    state, pkg = full_package()
    assert pkg['format'] == 'SONAPSE_NEUROWEAVE_PACKAGE' and pkg['version'] == '1.0.0'
    assert pkg['integrity']['algorithm'] == 'sha256' and len(pkg['integrity']['fingerprint']) == 64
    text = pk.dumps_package(pkg)
    rep = pk.validate_package(json.loads(text))
    assert rep['valid'], rep['errors']
    assert rep['counts']['neurons'] == 24 and rep['counts']['synapses'] == len(state['synapses']['src'])
    assert pk.loads_package(text) == pkg
    # brain content round-trips exactly
    assert json.loads(text)['brain'] == state
    # determinism (no wall-clock in fingerprint)
    _, pkg2 = full_package()
    assert pkg2['integrity'] == pkg['integrity']
    # numpy inputs are converted
    st = make_brain_state()
    st['readout']['w'] = np.asarray(st['readout']['w'])
    st['synapses']['src'] = np.asarray(st['synapses']['src'], dtype=np.int64)
    p3 = pk.build_package(st)
    assert pk.validate_package(json.loads(pk.dumps_package(p3)))['valid']


def test_build_does_not_mutate_inputs():
    state = make_brain_state()
    comps = {'memory': {'items': [1, 2]}}
    snap_s, snap_c = copy.deepcopy(state), copy.deepcopy(comps)
    pkg = pk.build_package(state, components=comps)
    assert state == snap_s and comps == snap_c
    pkg['brain']['cells']['fast'][0] = 99.0
    assert state['cells']['fast'][0] != 99.0


def test_capability_matrix_reflects_contents():
    state = make_brain_state()
    bare = pk.build_package(state)['capabilities']
    assert set(bare) == set(pk.CAPABILITIES)
    assert bare['frozen_inference']['status'] == 'supported'
    assert bare['persistent_model_state']['status'] == 'supported'
    assert bare['continued_learning']['status'] == 'requires_runtime'
    assert bare['full_simulation']['status'] == 'unsupported'
    assert bare['live_sensory_operation']['status'] == 'unsupported'
    assert bare['exported_memory']['status'] == 'unsupported'
    for entry in bare.values():
        assert entry['status'] in pk.CAPABILITY_STATUSES and entry['explanation']
    st2 = copy.deepcopy(state)
    del st2['synapses']['trace'], st2['synapses']['eligibility']
    assert pk.build_package(st2)['capabilities']['continued_learning']['status'] == 'unsupported'
    spec = NeuroWeaveSpecV1(neurons=24, max_neurons=24, seed=3,
                            sensory_modes=(('visual_luminance', 'live'), ('audio_low', 'live')))
    full = pk.build_package(state, spec=spec, components={
        'agent': {'w': [0.1]}, 'environment': {'config': 'default'}, 'memory': {'items': [1]},
        'sensory': {'channels': ['visual_luminance'], 'raw_media_stored': False}})['capabilities']
    assert full['full_simulation']['status'] == 'requires_runtime'
    assert full['exported_memory']['status'] == 'supported'
    live = full['live_sensory_operation']
    assert live['status'] == 'requires_runtime' and live['live_channels'] == ['audio_low', 'visual_luminance']
    assert any('camera permission' in r for r in live['requirements'])
    assert any('microphone permission' in r for r in live['requirements'])
    only_vis = pk.build_package(state, components={'sensory': {'modes': {'visual_motion': 'live'}}})
    reqs = only_vis['capabilities']['live_sensory_operation']['requirements']
    assert any('camera' in r for r in reqs) and not any('microphone' in r for r in reqs)


def test_build_rejects_bad_inputs():
    state = make_brain_state()
    with pytest.raises(ValueError):
        pk.build_package(state, components={'bogus': {}})
    bad = copy.deepcopy(state)
    bad['cells']['fast'][0] = float('nan')
    with pytest.raises(ValueError):
        pk.build_package(bad)
    with pytest.raises(ValueError):
        pk.build_package(state, components={'sensory': {'raw_frame': [[1, 2], [3, 4]]}})
    with pytest.raises(ValueError):
        pk.build_package(state, components={'sensory': {'raw_media_stored': True}})
    with pytest.raises(ValueError):
        pk.build_package(state, spec={'schema': 'Nope'})
    with pytest.raises(ValueError):
        pk.build_package('not a dict')


def test_tamper_detection():
    _, pkg = full_package()
    text = pk.dumps_package(pkg)
    t = json.loads(text)
    t['brain']['synapses']['w'][0] = t['brain']['synapses']['w'][0] * 0.5
    rep = pk.validate_package(t)
    assert not rep['valid'] and any('fingerprint mismatch' in e for e in rep['errors'])
    t2 = json.loads(text)
    t2['provenance']['run'] = 'other'
    assert not pk.validate_package(t2)['valid']
    # recomputing the fingerprint does not hide a capability lie
    t3 = json.loads(text)
    t3['capabilities']['full_simulation']['status'] = 'supported'
    t3['integrity']['fingerprint'] = pk.package_fingerprint(t3)
    rep3 = pk.validate_package(t3)
    assert not rep3['valid'] and any('capability matrix' in e for e in rep3['errors'])
    ins = pk.inspect_package(json.dumps(t))
    assert ins['parsed'] and not ins['valid']


def test_validate_counts_endpoints_signs_nan():
    _, pkg = full_package()
    base = json.loads(pk.dumps_package(pkg))

    def resign(p):
        p['integrity']['fingerprint'] = pk.package_fingerprint(p)
        return pk.validate_package(p)

    p = copy.deepcopy(base)
    p['counts']['synapses'] += 1
    assert any('declared synapse count' in e for e in resign(p)['errors'])
    p = copy.deepcopy(base)
    p['brain']['synapses']['dst'][2] = 10_000
    assert any('out of range' in e for e in resign(p)['errors'])
    p = copy.deepcopy(base)
    p['brain']['synapses']['src'][1] = -1
    assert any('out of range' in e for e in resign(p)['errors'])
    p = copy.deepcopy(base)
    a = p['brain']['synapses']['src'][0]
    p['brain']['synapses']['w'][0] = -p['brain']['topology']['sign'][a] * 0.3
    assert any('violate source sign' in e for e in resign(p)['errors'])
    p = copy.deepcopy(base)
    p['brain']['cells']['slow'][3] = float('nan')
    rep = pk.validate_package(p)
    assert not rep['valid'] and any('non-finite' in e for e in rep['errors'])
    p = copy.deepcopy(base)
    p['components']['memory']['items'][0]['key'][0] = float('inf')
    assert any('non-finite' in e for e in pk.validate_package(p)['errors'])
    p = copy.deepcopy(base)
    p['brain']['synapses']['eligibility'] = p['brain']['synapses']['eligibility'][:-1]
    assert any('eligibility' in e for e in resign(p)['errors'])
    p = copy.deepcopy(base)
    p['privacy']['raw_media_stored'] = True
    assert not resign(p)['valid']
    p = copy.deepcopy(base)
    p['version'] = '9.9.9'
    assert any('version' in e for e in resign(p)['errors'])
    p = copy.deepcopy(base)
    p['mechanisms'] = ['not_a_mechanism']
    assert not resign(p)['valid']
    assert not pk.validate_package([1, 2])['valid']


def test_validate_population_limits():
    _, pkg = full_package()
    p = json.loads(pk.dumps_package(pkg))
    p['brain']['topology']['n'] = MAX_POPULATION + 1
    p['counts']['neurons'] = MAX_POPULATION + 1
    p['integrity']['fingerprint'] = pk.package_fingerprint(p)
    assert any('exceeds limit' in e for e in pk.validate_package(p)['errors'])
    errs, _, _ = pk.validate_brain_state({**make_brain_state(), 'synapses': {
        'src': [0] * (MAX_SYNAPSES + 1), 'dst': [1] * (MAX_SYNAPSES + 1), 'w': [0.1] * (MAX_SYNAPSES + 1)}})
    assert any('exceeds limit' in e for e in errs)


def test_inspect_oversize_rejected_before_parse(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('json.loads must not be called for oversize input')
    monkeypatch.setattr(pk.json, 'loads', boom)
    big = '{"a":"' + 'x' * (pk.MAX_PACKAGE_BYTES + 10) + '"}'
    rep = pk.inspect_package(big)
    assert not rep['valid'] and not rep['parsed'] and 'before parsing' in rep['errors'][0]
    rep_b = pk.inspect_package(big.encode())
    assert not rep_b['valid'] and not rep_b['parsed']
    # multi-byte characters: within the char limit but over the byte limit
    mb = '"' + 'é' * (pk.MAX_PACKAGE_BYTES // 2 + 10) + '"'
    assert len(mb) <= pk.MAX_PACKAGE_BYTES
    rep_mb = pk.inspect_package(mb)
    assert not rep_mb['parsed'] and 'bytes' in rep_mb['errors'][0]


def test_inspect_malformed_and_nan_reported_not_raised():
    for text in ('{not json', '', '[1,2', b'\xff\xfe\x00', 12345, None):
        rep = pk.inspect_package(text)
        assert rep['valid'] is False and rep['errors']
    _, pkg = full_package()
    text = pk.dumps_package(pkg).replace('"b":0.05', '"b":NaN', 1)
    assert 'NaN' in text
    rep = pk.inspect_package(text)
    assert not rep['valid'] and not rep['parsed'] and 'NaN' in rep['errors'][0]
    rep = pk.inspect_package('[' * 100_000 + ']' * 100_000)
    assert rep['valid'] is False
    with pytest.raises(ValueError):
        pk.loads_package('{bad')


def test_inspect_summary_is_read_only():
    state, pkg = full_package(spec=NeuroWeaveSpecV1(neurons=24, max_neurons=24, seed=3,
                                                    mechanisms=('stdp', 'multi_soma')))
    text = pk.dumps_package(pkg)
    snapshot = copy.deepcopy(pkg)
    r1 = pk.inspect_package(text)
    r2 = pk.inspect_package(text.encode('utf-8'))
    assert r1['valid'] and r1['format'] == pk.PACKAGE_FORMAT and r1['version'] == '1.0.0' and not r1['legacy']
    assert r1['mechanisms'] == ['multi_soma', 'stdp']
    assert r1['capabilities']['frozen_inference'] == 'supported'
    assert r1['counts']['neurons'] == 24 and r1['counts']['bytes'] == len(text.encode())
    assert r1['fingerprint'] == pkg['integrity']['fingerprint']
    assert {k: v for k, v in r1.items()} == {k: v for k, v in r2.items()}
    assert pkg == snapshot
    json.dumps(r1)


def test_package_does_not_touch_sonapse_checkpoint_format():
    _, pkg = full_package()
    assert pkg['compatibility']['sonapse_checkpoint_format_changed'] is False
    assert pkg['format'] != 'NEUROWEAVE_MODEL_PACKAGE'
    for mod in (pk, hw):
        src = open(mod.__file__, encoding='utf-8').read()
        assert 'import torch' not in src and 'from torch' not in src


# =========================================================================== legacy
@pytest.mark.parametrize('version', ['1.0.0', '1.1.0', '1.2.0'])
def test_legacy_acceptance_and_conversion(version):
    m = make_legacy(version)
    snap = copy.deepcopy(m)
    rep = pk.validate_legacy_package(m)
    assert rep['valid'], rep['errors']
    assert rep['legacy'] and rep['legacy_schema'] == (version == '1.0.0')
    ins = pk.inspect_package(json.dumps(m))
    assert ins['valid'] and ins['legacy'] and ins['version'] == version and ins['format'] == pk.LEGACY_FORMAT
    assert ins['mechanisms'] == [pk.LEGACY_FEATURE_MAP[k] for i, k in enumerate(pk.LEGACY_FEATURE_IDS) if i % 3 == 0]
    assert pk.validate_package(m)['valid']
    brain = pk.convert_legacy_neuroweave(m)
    assert m == snap  # read-only
    assert brain['provenance'] == 'imported_neuroweave_js'
    assert brain['schema'] == 'NeuroWeaveBrainStateV1'
    n = len(m['network']['neurons'])
    assert brain['topology']['n'] == n and len(brain['synapses']['src']) == len(m['network']['synapses'])
    assert brain['topology']['sign'] == [1 if nd['excitatory'] else -1 for nd in m['network']['neurons']]
    assert brain['synapses']['w'] == [e['weight'] for e in m['network']['synapses']]
    assert brain['cells']['output'] == [nd['act'] for nd in m['network']['neurons']]
    assert brain['readout']['w'] == m['readout']['weights'] and brain['step'] == 250
    assert brain['conversion']['source_version'] == version
    if version == '1.0.0':
        assert 'threshold' in brain['conversion']['defaulted_cells']
        assert 'routing_gain' in brain['conversion']['defaulted_synapse_fields']
    else:
        assert 'threshold' not in brain['conversion']['defaulted_cells']
        assert brain['cells']['branch_a'][0] == 0.2 and brain['cells']['tm_mid'][0] == 0.05
        assert brain['topology']['region'][0] == 0 and brain['topology']['region'][9] == 9
    errs, _, _ = pk.validate_brain_state(brain)
    assert errs == []
    pkg = pk.build_package(brain, provenance={'source': 'imported_neuroweave_js'})
    assert pk.validate_package(json.loads(pk.dumps_package(pkg)))['valid']
    assert pkg['mechanisms'] == brain['conversion']['mechanisms']


@pytest.mark.parametrize('version', ['1.1.0', '1.2.0'])
def test_legacy_tamper_and_coverage(version):
    m = make_legacy(version)
    t = copy.deepcopy(m)
    t['network']['synapses'][0]['weight'] = t['network']['synapses'][0]['weight'] * 0.5
    rep = pk.validate_legacy_package(t)
    assert not rep['valid'] and any('fingerprint' in e for e in rep['errors'])
    with pytest.raises(ValueError):
        pk.convert_legacy_neuroweave(t)
    # sensory is covered by the 1.2 fingerprint but not by 1.1
    t2 = copy.deepcopy(m)
    t2['sensory'] = {'channels': [f'c{i}' for i in range(16)], 'cameraEnabled': True,
                     'microphoneEnabled': False, 'rawMediaStored': False}
    assert pk.validate_legacy_package(t2)['valid'] == (version == '1.1.0')
    # training is outside coverage in both versions
    t3 = copy.deepcopy(m)
    t3['training']['steps'] = 999
    assert pk.validate_legacy_package(t3)['valid']


def test_legacy_rejections():
    m = make_legacy('1.2.0')
    cases = []
    c = copy.deepcopy(m); c['network']['synapses'][0]['dst'] = 999; cases.append(c)
    c = copy.deepcopy(m); c['network']['neuronCount'] += 1; cases.append(c)
    c = copy.deepcopy(m); c['readout']['weights'] = c['readout']['weights'][:7]; cases.append(c)
    c = copy.deepcopy(m); c['formatVersion'] = '2.0.0'; cases.append(c)
    c = copy.deepcopy(m); c['sensory']['rawMediaStored'] = True; cases.append(c)
    c = copy.deepcopy(m); c['microCollective']['experts'] = c['microCollective']['experts'][:3]; cases.append(c)
    c = copy.deepcopy(m); c['network']['neurons'][0]['fast'] = float('nan'); cases.append(c)
    c = copy.deepcopy(m); c['network']['synapses'][0]['weight'] = 1.5; cases.append(c)
    c = copy.deepcopy(m); del c['agentController']; cases.append(c)
    c = copy.deepcopy(m); c['features']['stdp'] = 'yes'; cases.append(c)
    for case in cases:
        assert not pk.validate_legacy_package(case)['valid']
    assert not pk.validate_legacy_package('nope')['valid']
    # Dale clamp on convert: a 1.0 package (no fingerprint) whose weight contradicts its source sign
    legacy10 = make_legacy('1.0.0')
    e = legacy10['network']['synapses'][0]
    e['weight'] = -abs(e['weight']) if legacy10['network']['neurons'][e['src']]['excitatory'] else abs(e['weight'])
    brain = pk.convert_legacy_neuroweave(legacy10)
    assert brain['synapses']['w'][0] == 0.0 and brain['conversion']['sign_clamped_synapses'] == 1


def test_fnv1a_known_vectors_and_js_number_format():
    # standard FNV-1a 32-bit test vectors (ASCII == UTF-16 code units here)
    assert pk.fnv1a32('') == '811c9dc5'
    assert pk.fnv1a32('a') == 'e40c292c'
    assert pk.fnv1a32('foobar') == 'bf9cf968'
    assert pk.js_json_stringify({'b': 1.0, 'a': [0.1, 1e-7, 1e21, 2.5e-5, -0.0, True, None, 'x"\n']}) == \
        '{"b":1,"a":[0.1,1e-7,1e+21,0.000025,0,true,null,"x\\"\\n"]}'
    assert pk.js_json_stringify([123456789012345680000.0, 1.5e300, 5e-324, 1e-6]) == \
        '[123456789012345680000,1.5e+300,5e-324,0.000001]'


NODE_SCRIPT = r"""
const fs = require('fs');
function fnv1a(text){let h=0x811c9dc5;for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,0x01000193);}return('00000000'+(h>>>0).toString(16)).slice(-8);}
const m = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const payload = {config:m.config,features:m.features,network:m.network,readout:m.readout,agentController:m.agentController};
if (m.formatVersion === '1.2.0') { payload.microCollective = m.microCollective; payload.sensory = m.sensory; }
const extra = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
process.stdout.write(JSON.stringify({pkg: fnv1a(JSON.stringify(payload)), strings: extra.map(s => fnv1a(s)),
  stringify: fnv1a(JSON.stringify(extra))}));
"""


@pytest.mark.skipif(shutil.which('node') is None, reason='node not available')
@pytest.mark.parametrize('version', ['1.1.0', '1.2.0'])
def test_fnv1a_matches_javascript(tmp_path, version):
    m = make_legacy(version)
    m['config']['label'] = 'café ☃ \U0001F600 "q" \\ \t'
    m['integrity']['fingerprint'] = pk.legacy_fingerprint(m)
    pkg_path = tmp_path / 'pkg.json'
    pkg_path.write_text(json.dumps(m, ensure_ascii=False), encoding='utf-8')
    strings = ['', 'a', 'NEUROWEAVE', 'café', '\U0001F600', 'x' * 300]
    str_path = tmp_path / 'strings.json'
    str_path.write_text(json.dumps(strings), encoding='utf-8')
    script = tmp_path / 'fnv.cjs'
    script.write_text(NODE_SCRIPT, encoding='utf-8')
    out = subprocess.run(['node', str(script), str(pkg_path), str(str_path)], capture_output=True,
                         text=True, timeout=30, check=True)
    js = json.loads(out.stdout)
    assert js['pkg'] == m['integrity']['fingerprint'] == pk.legacy_fingerprint(json.loads(pkg_path.read_text('utf-8')))
    assert js['strings'] == [pk.fnv1a32(s) for s in strings]
    assert js['stringify'] == pk.fnv1a32(pk.js_json_stringify(strings))
    assert pk.validate_legacy_package(json.loads(pkg_path.read_text('utf-8')))['valid']


# =========================================================================== CSV
def test_csv_rows():
    rows = [{'experiment': 'stdp', 'score': 0.5, 'seeds': [1, 2]},
            {'experiment': '=HYPERLINK("x")', 'delta': -0.25, 'ok': True, 'note': None}]
    text = pk.csv_rows(rows)
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0] == ['experiment', 'score', 'seeds', 'delta', 'ok', 'note']
    assert parsed[1] == ['stdp', '0.5', '[1,2]', '', '', '']
    assert parsed[2][0].startswith("'=") and parsed[2][3] == '-0.25' and parsed[2][4] == 'true'
    assert pk.csv_rows({'rows': rows}) == text
    assert pk.csv_rows([]) == '\n'
    with pytest.raises(ValueError):
        pk.csv_rows([{'x': float('nan')}])
    with pytest.raises(ValueError):
        pk.csv_rows('nope')


def test_mechanism_ids_exposed_in_spec_are_valid_for_packages():
    spec = NeuroWeaveSpecV1.all_mechanisms(neurons=24, max_neurons=24, seed=3)
    state = make_brain_state(spec=spec)
    pkg = pk.build_package(state)
    assert pkg['mechanisms'] == list(MECHANISM_IDS)
    assert pk.validate_package(json.loads(pk.dumps_package(pkg)))['valid']


# =========================================================================== adversarial review additions
def _real_brain_and_components():
    brain_mod = pytest.importorskip('synapse.neuroweave.brain')
    from synapse.neuroweave.memory import MemoryEcology
    from synapse.neuroweave.micro_collective import MicroCollective
    from synapse.neuroweave.sensory import SensoryHub
    spec = NeuroWeaveSpecV1.max_preset(seed=3, neurons=48)
    brain = brain_mod.NeuroWeaveBrain(spec)
    for _ in range(4):
        brain.step(np.full(8, 0.3))
    comps = {'micro_collective': MicroCollective().export_state(), 'sensory': SensoryHub(spec).manifest(),
             'memory': MemoryEcology(8, 4, capacity=16, seed=1).export_state()}
    return brain_mod, spec, brain, comps


def test_real_brain_export_packages_validates_and_reimports():
    # The real NeuroWeaveBrain.export_state keeps edges only in `synapses`, stores scalar
    # metadata in `cells` and per-neuron arrays in `synapses`; the micro-collective exports a
    # scalar `counters.samples`. All of this must package cleanly (it previously raised).
    brain_mod, spec, brain, comps = _real_brain_and_components()
    state = brain.export_state()
    pkg = pk.build_package(state, components=comps)
    loaded = json.loads(pk.dumps_package(pkg))
    report = pk.validate_package(loaded)
    assert report['valid'], report['errors']
    assert report['counts']['synapses'] == len(state['synapses']['src'])
    clone = brain_mod.NeuroWeaveBrain(spec)
    clone.import_state(loaded['brain'])
    assert clone.step_count == brain.step_count
    # the re-imported brain reproduces frozen inference exactly
    for x in (0.2, -0.4, 0.7):
        np.testing.assert_array_equal(clone.frozen_step(np.full(8, x)), brain.frozen_step(np.full(8, x)))
    manifest = hw.fpga_mapping_manifest(state)
    assert hw.validate_fpga_manifest(manifest)['valid']
    assert manifest['graph']['synapse_count'] == len(state['synapses']['src'])
    # consistency checks on the real shape actually bite
    bad = copy.deepcopy(loaded)
    bad['brain']['synapses']['sign'][0] *= -1
    assert any('synapses.sign' in e for e in pk.validate_package(bad)['errors'])
    bad = copy.deepcopy(loaded)
    bad['brain']['cells']['n'] = 999
    assert any('cells.n' in e for e in pk.validate_package(bad)['errors'])
    bad = copy.deepcopy(loaded)
    bad['brain']['synapses']['compartment'] = bad['brain']['synapses']['compartment'][:-1]
    assert any('compartment' in e for e in pk.validate_package(bad)['errors'])
    bad = copy.deepcopy(loaded)
    bad['brain']['synapses']['coact'].append(0.0)
    assert any('coact' in e for e in pk.validate_package(bad)['errors'])


def test_raw_media_detection_is_precise():
    state = make_brain_state()
    ok = pk.build_package(state, components={'training_history': {'samples': 1000, 'frames': 3},
                                             'micro_collective': {'counters': {'samples': 5}}})
    assert pk.validate_package(json.loads(pk.dumps_package(ok)))['valid']
    for comp in ({'sensory': {'samples': [0.1, 0.2, 0.3]}}, {'sensory': {'frame': 'AAAA'}},
                 {'sensory': {'pixels': 3}}, {'experiments': {'thumb': 'data:image/png;base64,AAAA'}},
                 {'sensory': {'clip': 'DATA:audio/wav;base64,UklG'}}):
        with pytest.raises(ValueError):
            pk.build_package(state, components=comp)
    loaded = json.loads(pk.dumps_package(ok))
    loaded['components']['training_history']['note'] = 'data:video/mp4;base64,AAAA'
    assert any('media' in e for e in pk.validate_package(loaded)['errors'])


def test_validators_never_raise_on_hostile_structure():
    pkg = json.loads(pk.dumps_package(full_package()[1]))
    for junk in (5, 1.5, True, 'x', {'a': 1}, [[1, 2, 3]], None):
        bad = copy.deepcopy(pkg)
        bad['spec']['sensory_modes'] = junk  # previously raised TypeError out of validate_package
        rep = pk.validate_package(bad)
        assert rep['valid'] is False and rep['errors']
        assert pk.inspect_package(json.dumps(bad))['valid'] is False
    m = hw.fpga_mapping_manifest(make_brain_state())
    for junk in ([None], ['a', 1], [{}], [[0]], [1.5]):
        bad = copy.deepcopy(m)
        bad['tiles'][0]['edge_indices'] = junk
        rep = hw.validate_fpga_manifest(bad)
        assert rep['valid'] is False


def test_fpga_validation_catches_edges_moved_between_tiles():
    m = hw.fpga_mapping_manifest(make_brain_state(n=40, edges_per=4), tile_size=8)
    assert hw.validate_fpga_manifest(m)['valid']
    tiles = m['tiles']
    a, b = tiles[0], next(t for t in tiles if (t['dst_tile'], t['src_tile']) != (tiles[0]['dst_tile'], tiles[0]['src_tile']))
    bad = copy.deepcopy(m)
    ta = next(t for t in bad['tiles'] if t['tile_index'] == a['tile_index'])
    tb = next(t for t in bad['tiles'] if t['tile_index'] == b['tile_index'])
    ta['edge_indices'][0], tb['edge_indices'][0] = tb['edge_indices'][0], ta['edge_indices'][0]
    rep = hw.validate_fpga_manifest(bad)  # coverage is still exact, but blocks are wrong
    assert not rep['valid'] and any('block' in e for e in rep['errors'])
    bad = copy.deepcopy(m)
    bad['schedule']['tile_size'] = 0
    assert not hw.validate_fpga_manifest(bad)['valid']


def test_fpga_delay_handling_and_profile_schedule_guard():
    state = make_brain_state()
    state['synapses']['delay'] = state['synapses']['delay'][:-1]
    with pytest.raises(ValueError):
        hw.fpga_mapping_manifest(state)
    state = make_brain_state()
    del state['synapses']['delay']
    del state['topology']['delay']
    m = hw.fpga_mapping_manifest(state)
    assert m['validation']['delays_defaulted'] is True and all(e['delay'] == 1 for e in m['graph']['edges'])
    assert hw.fpga_mapping_manifest(make_brain_state())['validation']['delays_defaulted'] is False
    for sched in ({'cycles': [1]}, {'cycles': {'kind': 'measured', 'pipelined': 3}},
                  {'cycles': {'kind': 'estimated', 'pipelined': -1}}, {'cycles': {'kind': 'estimated', 'pipelined': 'x'}}):
        with pytest.raises(ValueError):
            hw.execution_profile(8, 16, schedule=sched)
    good = hw.execution_profile(8, 16, steps=3, schedule={'cycles': {'kind': 'estimated', 'pipelined': 7}})
    assert good['estimated']['cycles_total'] == 21


def test_provenance_cannot_upgrade_evidence_and_deep_nesting_is_value_error():
    pkg = pk.build_package(make_brain_state(), provenance={'evidence_tier': 'measured_hardware', 'author': 'x'})
    assert pkg['provenance']['evidence_tier'] == 'synthetic_abstraction' and pkg['provenance']['author'] == 'x'
    deep: dict = {}
    cur = deep
    for _ in range(5000):
        cur['k'] = {}
        cur = cur['k']
    with pytest.raises(ValueError):
        pk.build_package(make_brain_state(), components={'experiments': deep})


def test_js_stringify_key_order_and_surrogate_pairs():
    obj = {'b': 1, '10': 2, '2': 3, '01': 4, '4294967295': 5, '-1': 6, '0': 7}
    assert pk.js_json_stringify(obj) == '{"0":7,"2":3,"10":2,"b":1,"01":4,"4294967295":5,"-1":6}'
    pair = '😀'  # surrogate pair held as two Python code points: JS sees a valid pair
    assert pk.js_json_stringify(pair) == '"' + pair + '"'
    assert pk.js_json_stringify('\ud83d') == '"\\ud83d"'
    assert pk.js_json_stringify('\ude00x\ud83d') == '"\\ude00x\\ud83d"'


NODE_ORDER_SCRIPT = r"""
const fs = require('fs');
function fnv1a(text){let h=0x811c9dc5;for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,0x01000193);}return('00000000'+(h>>>0).toString(16)).slice(-8);}
const m = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const payload = {config:m.config,features:m.features,network:m.network,readout:m.readout,agentController:m.agentController,microCollective:m.microCollective,sensory:m.sensory};
process.stdout.write(JSON.stringify({fp: fnv1a(JSON.stringify(payload)), text: JSON.stringify({o: m.config.order})}));
"""


@pytest.mark.skipif(shutil.which('node') is None, reason='node not available')
def test_legacy_fingerprint_matches_js_with_integer_like_keys(tmp_path):
    m = make_legacy('1.2.0')
    m['config']['order'] = {'z': 1, '10': 2, '2': 3, '007': 4, 'a': [1.0, 1e-7]}
    m['network']['neuromodulators'] = {'5': 0.5, 'da': 0, '1': 0.25}
    m['integrity']['fingerprint'] = pk.legacy_fingerprint(m)
    path = tmp_path / 'pkg.json'
    path.write_text(json.dumps(m, ensure_ascii=False), encoding='utf-8')
    script = tmp_path / 'order.cjs'
    script.write_text(NODE_ORDER_SCRIPT, encoding='utf-8')
    out = subprocess.run(['node', str(script), str(path)], capture_output=True, text=True, timeout=30, check=True)
    js = json.loads(out.stdout)
    assert js['text'] == pk.js_json_stringify({'o': m['config']['order']})
    assert js['fp'] == m['integrity']['fingerprint']
    assert pk.validate_legacy_package(json.loads(path.read_text('utf-8')))['valid']


def test_csv_header_formula_injection_guarded():
    text = pk.csv_rows([{'=cmd|calc': 1, '@sum': 2, 'safe': 3}])
    header = next(csv.reader(io.StringIO(text)))
    assert header == ["'=cmd|calc", "'@sum", 'safe']
