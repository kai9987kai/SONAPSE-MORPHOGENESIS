"""Tests for synapse.neuroweave prediction, packets and events (predpack)."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from synapse.neuroweave.prediction import EfferenceSelfModel, PredictiveHierarchy, GATE_MIN, GATE_MAX
from synapse.neuroweave import packets as P
from synapse.neuroweave.packets import RepresentationFusion, RepresentationPacket
from synapse.neuroweave.events import INTERFACES, TOPICS, Envelope, EventBus, validate_payload


# --------------------------------------------------------------------------- helpers
def _rot(theta: float, r: float) -> np.ndarray:
    return r * np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def _system(t1: float, t2: float) -> np.ndarray:
    A = np.zeros((4, 4))
    A[:2, :2] = _rot(t1, 0.998)
    A[2:, 2:] = _rot(t2, 0.998)
    return A


def _snapshot(model) -> str:
    return json.dumps(model.export_state(), sort_keys=True)


@pytest.fixture(scope='module')
def regime_run():
    """Linear dynamical system with a regime shift at t=500 (one shared run)."""
    rng = np.random.default_rng(0)
    ph = PredictiveHierarchy(4, seed=1)
    A = _system(0.2, 0.07)
    x = rng.normal(size=4) * 0.3
    log = {}
    for t in range(560):
        if t == 500:
            A = _system(-1.1, 0.9)
        r = ph.observe(x)
        if t == 60:
            log['early'] = {h: ph.error_medium[h] for h in ph.horizons}
        if t == 499:
            log['pre'] = {'errors': {h: ph.error_medium[h] for h in ph.horizons}, 'horizon': r['horizon'],
                          'state': ph.export_state()}
        if 500 < t < 515:
            log.setdefault('post_horizons', []).append(r['horizon'])
            log.setdefault('post_surprise', []).append(ph.surprise_trace[1])
        x = A @ x + rng.normal(size=4) * 0.01
    log['model'] = ph
    return log


# --------------------------------------------------------------------------- PredictiveHierarchy
def test_errors_shrink_on_linear_system(regime_run):
    early, pre = regime_run['early'], regime_run['pre']['errors']
    assert pre[1] < early[1] * 0.8
    assert pre[1] < 0.2


def test_longer_horizon_less_accurate(regime_run):
    e = regime_run['pre']['errors']
    assert e[1] < e[4] < e[16] < e[64]


def test_adaptive_horizon_shortens_after_regime_shift(regime_run):
    before = regime_run['pre']['horizon']
    assert before >= 4
    assert min(regime_run['post_horizons']) < before
    assert regime_run['post_horizons'][-1] < before
    assert max(regime_run['post_surprise']) > 1.0


def test_prequential_scoring_only_after_target_arrives():
    ph = PredictiveHierarchy(3, horizons=(1, 4), seed=3)
    for t in range(4):
        out = ph.observe(np.full(3, float(t)))
        assert (1 in out['scored']) == (t >= 1)
        assert 4 not in out['scored']
        assert ph.scored[4] == 0
    out = ph.observe(np.full(3, 4.0))
    assert 4 in out['scored'] and ph.scored[4] == 1
    # The h=4 prediction scored at t=4 was the one stored at t=0 (before any h=4 learning).
    W_before = [w.copy() for w in ph.W[4]]
    ph2 = PredictiveHierarchy(3, horizons=(1, 4), seed=3)
    phi0 = np.concatenate([np.zeros(3), [1.0]])
    stored = np.mean([w @ phi0 for w in ph2.W[4]], axis=0)
    expected = math.sqrt(np.mean((stored - 4.0) ** 2)) / ph._scale()
    assert ph.last_error[4] == pytest.approx(min(expected, 10.0), rel=1e-9)
    assert any(not np.array_equal(a, b) for a, b in zip(W_before, ph2.W[4]))


def test_forecast_and_update_shapes_and_learning():
    ph = PredictiveHierarchy(2, horizons=(1, 4), seed=5)
    fc = ph.forecast([0.5, -0.5])
    assert set(fc) == {1, 4}
    assert fc[1]['mean'].shape == (2,) and fc[1]['disagreement'] >= 0
    target = np.array([1.0, 2.0])
    errs = [ph.update([0.5, -0.5], target, horizon=1) for _ in range(30)]
    assert errs[-1] < errs[0]
    assert np.allclose(ph.forecast([0.5, -0.5])[1]['mean'], target, atol=0.05)
    with pytest.raises(ValueError):
        ph.update([0.5, -0.5], target, horizon=2)
    with pytest.raises(ValueError):
        ph.forecast([np.nan, 0.0])
    with pytest.raises(ValueError):
        ph.forecast([1.0, 2.0, 3.0])


def test_counterfactual_is_pure_and_action_sensitive():
    rng = np.random.default_rng(2)
    ph = PredictiveHierarchy(2, horizons=(1, 4), seed=4, action_dim=1)
    x = np.zeros(2)
    for _ in range(80):
        a = rng.normal(size=1)
        ph.observe(x, a)
        x = 0.9 * x + np.array([a[0], -a[0]]) + rng.normal(size=2) * 0.01
    before = _snapshot(ph)
    res = ph.counterfactual(x, [[-1.0], [0.0], [1.0]])
    assert _snapshot(ph) == before
    assert len(res) == 3
    m = [np.array(r['forecast'][1]['mean']) for r in res]
    assert m[2][0] > m[1][0] > m[0][0]
    assert m[2][1] < m[0][1]
    json.dumps(res)
    with pytest.raises(ValueError):
        PredictiveHierarchy(2).counterfactual([0, 0], [[1.0]])


def test_uncertainty_gate_bounds_and_response():
    calm = PredictiveHierarchy(2, horizons=(1,), seed=1, init_scale=0.0)
    calm.observe([1.0, 0.0])
    calm.observe([0.5, 0.5])
    assert calm.uncertainty_gate() > 0.999
    noisy = PredictiveHierarchy(2, horizons=(1,), seed=1, init_scale=3.0)
    for i in range(20):
        noisy.observe([math.sin(i), math.cos(i)], learn=False)
    g = noisy.uncertainty_gate()
    assert GATE_MIN <= g < 0.9
    assert GATE_MIN <= g <= GATE_MAX


def test_hierarchy_determinism_and_state_roundtrip():
    def run(seed, steps=40):
        ph = PredictiveHierarchy(3, horizons=(1, 4), seed=seed)
        rng = np.random.default_rng(9)
        for _ in range(steps):
            ph.observe(rng.normal(size=3))
        return ph
    a, b, c = run(11), run(11), run(12)
    assert _snapshot(a) == _snapshot(b)
    assert _snapshot(a) != _snapshot(c)
    state = json.loads(json.dumps(a.export_state()))
    d = PredictiveHierarchy(3, horizons=(1, 4), seed=11)
    d.import_state(state)
    assert _snapshot(d) == _snapshot(a)
    # Continued learning is identical after restore (incl. bootstrap RNG).
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(10, 3))
    for x in xs:
        a.observe(x)
        d.observe(x)
    assert _snapshot(a) == _snapshot(d)


def test_hierarchy_import_rejects_bad_state_without_mutation():
    ph = PredictiveHierarchy(2, horizons=(1, 4), seed=1)
    for i in range(6):
        ph.observe([i * 0.1, -i * 0.1])
    good = ph.export_state()
    before = _snapshot(ph)
    bad_cases = []
    s = copy.deepcopy(good); s['horizons']['1']['W'][0][0][0] = float('nan'); bad_cases.append(s)
    s = copy.deepcopy(good); s['horizons']['4']['P'] = s['horizons']['4']['P'][:1]; bad_cases.append(s)
    s = copy.deepcopy(good); s['config']['dim'] = 3; bad_cases.append(s)
    s = copy.deepcopy(good); s['buffer'][0]['step'] = 99; bad_cases.append(s)
    s = copy.deepcopy(good); s['rng'] = {'bit_generator': 'MT19937'}; bad_cases.append(s)
    s = copy.deepcopy(good); s['horizons']['1']['surprise'] = 99.0; bad_cases.append(s)
    s = copy.deepcopy(good); s['schema'] = 'nope'; bad_cases.append(s)
    for bad in bad_cases:
        with pytest.raises(ValueError):
            ph.import_state(bad)
        assert _snapshot(ph) == before


def test_hierarchy_state_bounded_under_extreme_input():
    ph = PredictiveHierarchy(2, horizons=(1, 4), seed=0)
    rng = np.random.default_rng(0)
    for i in range(120):
        ph.observe(rng.normal(size=2) * (1e5 if i % 7 == 0 else 1.0))
    for h in ph.horizons:
        for W, Pm in zip(ph.W[h], ph.P[h]):
            assert np.all(np.isfinite(W)) and np.all(np.abs(W) <= 25.0)
            assert np.all(np.isfinite(Pm)) and np.trace(Pm) <= 1e5 + 1e-6
    m = ph.metrics()
    assert all(0 <= v['surprise'] <= 10 and v['fast'] <= 10 for v in m.values())


# --------------------------------------------------------------------------- EfferenceSelfModel
def _train_efference(gain=0.8, steps=150, seed=0):
    rng = np.random.default_rng(seed)
    em = EfferenceSelfModel(3, seed=seed)
    for _ in range(steps):
        u = rng.normal(size=3)
        em.record(u, u, gain * u, gain * u + rng.normal(size=3) * 0.005)
    return em, rng


def test_efference_learns_forward_model_and_clean_authorship():
    em, rng = _train_efference()
    assert np.allclose(em.G[:, :3], 0.8 * np.eye(3), atol=0.02)
    u = rng.normal(size=3)
    r = em.record(u, u, 0.8 * u, 0.8 * u, learn=False)
    assert r['authorship'] > 0.95
    assert r['disturbance'] < 0.02 and r['actuator_slip'] < 0.03 and r['prediction_error'] < 1e-9
    assert em.control_reliability > 0.9


def test_efference_disturbance_injection():
    em, rng = _train_efference()
    u = np.array([1.0, 0.0, 0.0])
    push = np.array([0.0, 0.6, -0.4])  # orthogonal to the issued command
    clean = em.record(u, u, 0.8 * u, 0.8 * u, learn=False)
    dist = em.record(u, u, 0.8 * u, 0.8 * u + push, learn=False)
    assert dist['disturbance'] > clean['disturbance'] + 0.2
    assert dist['disturbance'] == pytest.approx(math.sqrt(np.mean(push ** 2)), abs=0.02)
    assert dist['authorship'] < clean['authorship'] - 0.3
    assert dist['gain_mismatch'] < 0.05  # disturbance is not mistaken for slip
    assert dist['prediction_error'] > 0.3


def test_efference_slip_injection():
    em, _ = _train_efference()
    intended = np.array([1.0, -1.0, 0.5])
    issued = np.array([0.5, -1.0, 0.5])
    slip = em.record(intended, issued, 0.8 * issued, 0.8 * issued, learn=False)
    assert slip['command_slip'] == pytest.approx(math.sqrt(0.25 / 3), rel=1e-9)
    assert slip['disturbance'] < 0.02
    gain = em.record(issued, issued, 0.8 * issued, 0.4 * issued, learn=False)
    assert gain['gain_mismatch'] == pytest.approx(0.4, abs=0.03)
    assert gain['actuator_slip'] >= gain['gain_mismatch']
    rel_before = em.control_reliability
    for _ in range(20):
        em.record(intended, issued, 0.8 * issued, 0.4 * issued + 0.3, learn=False)
    assert em.control_reliability < rel_before - 0.3
    assert 0.0 <= em.control_reliability <= 1.0


def test_efference_roundtrip_and_validation():
    em, _ = _train_efference(steps=20)
    st = json.loads(json.dumps(em.export_state()))
    other = EfferenceSelfModel(3)
    other.import_state(st)
    assert json.dumps(other.export_state(), sort_keys=True) == json.dumps(em.export_state(), sort_keys=True)
    before = json.dumps(other.export_state(), sort_keys=True)
    bad = copy.deepcopy(st); bad['G'][0][0] = float('inf')
    with pytest.raises(ValueError):
        other.import_state(bad)
    bad = copy.deepcopy(st); bad['control_reliability'] = 1.5
    with pytest.raises(ValueError):
        other.import_state(bad)
    assert json.dumps(other.export_state(), sort_keys=True) == before
    with pytest.raises(ValueError):
        em.record([1, 2], [1, 2, 3], [0, 0, 0], [0, 0, 0])


# --------------------------------------------------------------------------- packets
def test_packet_validation_and_checksum():
    p = P.encode_vector([1.0, 2.0, 3.0], source=('region:association_a', 'soma:primary'), provenance='sensory',
                        confidence=0.9, uncertainty=0.1, timestamp=5)
    assert p.schema == 'NeuroWeavePacketV1' and p.version == 1 and p.dim == 3
    assert len(p.checksum) == 64
    q = RepresentationPacket.from_dict(json.loads(p.to_json()))
    assert q == p and q.checksum == p.checksum
    assert P.encode_vector([1.0, 2.0, 3.0001]).checksum != P.encode_vector([1.0, 2.0, 3.0]).checksum
    tampered = p.to_dict(); tampered['payload'][0] = 9.0
    with pytest.raises(ValueError):
        RepresentationPacket.from_dict(tampered)
    bad = [
        dict(kind='nope', payload=(1.0,)),
        dict(kind='vector', payload=(float('nan'),)),
        dict(kind='vector', payload=()),
        dict(kind='vector', payload=(1.0,), confidence=1.5),
        dict(kind='vector', payload=(1.0,), uncertainty=-0.1),
        dict(kind='vector', payload=(1.0,), provenance='dream'),
        dict(kind='phase', payload=(4.0,)),
        dict(kind='sparse', payload=(0.0, 2.0), form='binary'),
        dict(kind='sparse', payload=(3.0, 1.0), form='index', dim=4),
        dict(kind='sparse', payload=(5.0,), form='index', dim=4),
        dict(kind='sparse', payload=(1.0,), form='dense'),
        dict(kind='vector', payload=(1.0, 2.0), dim=3),
        dict(kind='vector', payload=(1.0,), source=()),
        dict(kind='vector', payload=(1.0,), timestamp=-1),
        dict(kind='vector', payload=(1.0,), metadata=(('k', float('inf')),)),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            RepresentationPacket(**kw)
    with pytest.raises(Exception):
        p.kind = 'scalar'  # frozen


def test_neurowave_x2_roundtrip():
    from synapse.neurowave_x2.packets import NeuralPacket
    pkts = [P.encode_sparse([0.1, 0.9, 0.3, 0.8], 2, source='region:visual_sensory', provenance='sensory'),
            P.encode_phase(np.sin(np.arange(16) * 0.7), timestamp=12.5, confidence=0.7, uncertainty=0.2),
            P.encode_vsa({'color': 'red'}, dim=32, metadata={'note': 'x'})]
    for p in pkts:
        npk = P.to_neurowave_x2(p, sender_region=1, receiver_region=3, sequence_id=7)
        assert isinstance(npk, NeuralPacket)
        assert npk.phase_metadata == (p.timestamp, p.confidence, p.uncertainty)
        assert npk.protocol_id.startswith('nwp1:')
        back = P.from_neurowave_x2(npk)
        assert back == p and back.checksum == p.checksum
        wire = NeuralPacket.from_bytes(npk.to_bytes())
        assert P.from_neurowave_x2(wire) == p
    plain = NeuralPacket(0, 0, 0, 0, 0, 'other', (1.0,))
    with pytest.raises(ValueError):
        P.from_neurowave_x2(plain)
    npk = P.to_neurowave_x2(pkts[0])
    forged = NeuralPacket(0, 0, 0, 0, 0, npk.protocol_id, (0.0, 3.0), phase_metadata=npk.phase_metadata)
    with pytest.raises(ValueError):
        P.from_neurowave_x2(forged)


def test_encoders_compute_real_features():
    s = P.encode_scalar(0.25)
    assert s.kind == 'scalar' and s.payload == (0.25,)
    sig = np.sin(2 * np.pi * 4 * np.arange(64) / 64)
    sp = P.encode_spectral(sig)
    assert int(np.argmax(sp.payload)) == 4 and sp.payload[4] == pytest.approx(0.5, abs=1e-9)
    ph = P.encode_phase(np.cos(2 * np.pi * 3 * np.arange(32) / 32 + 0.4))
    assert ph.payload[3] == pytest.approx(0.4, abs=1e-9)
    assert all(-math.pi <= v <= math.pi for v in ph.payload)
    pop = P.encode_population(0.3, n=21)
    assert P.decode_population(pop) == pytest.approx(0.3, abs=0.02)
    assert np.argmax(pop.payload) == np.argmin(np.abs(np.linspace(-1, 1, 21) - 0.3))
    sx = P.encode_sparse([0.1, 0.9, 0.3, 0.8, 0.0], 2)
    assert sx.form == 'index' and sx.payload == (1.0, 3.0) and sx.dim == 5
    sb = P.encode_sparse([0.1, 0.9, 0.3, 0.8, 0.0], 2, form='binary')
    assert np.array_equal(sb.dense(), sx.dense()) and sb.dense().sum() == 2
    # VSA: unbinding the role recovers the right filler.
    dim = 256
    v = P.encode_vsa({'color': 'red', 'shape': 'circle'}, dim=dim, seed=1)
    assert set(v.payload) <= {-1.0, 1.0}
    rec = P.vsa_unbind(v.dense(), P.vsa_vector('color', dim, 1))
    sims = {f: float(rec @ P.vsa_vector(f, dim, 1)) for f in ('red', 'circle', 'blue')}
    assert sims['red'] > 3 * max(abs(sims['circle']), abs(sims['blue']))
    ev = P.encode_event(2.0, onset=True, duration=3.0, label='collision')
    assert ev.payload[:3] == (1.0, 2.0, pytest.approx(math.log1p(3.0)))
    assert sum(ev.payload[3:]) == 1.0
    sa, sb2 = P.encode_spatial(0.1, 0.1), P.encode_spatial(0.9, 0.9)
    assert np.argmax(sa.payload) == 0 and np.argmax(sb2.payload) == 15
    ring = np.zeros((6, 6))
    for i in range(6):
        ring[i, (i + 1) % 6] = ring[(i + 1) % 6, i] = 1
    g = P.encode_graph(ring, k_eig=3)
    assert g.payload[0] == 6 and g.payload[2] == 2.0 and g.payload[3] == 0.0
    assert g.payload[5] == pytest.approx(0.0, abs=1e-9)  # connected graph: one zero eigenvalue
    two = np.zeros((6, 6)); two[0, 1] = two[1, 0] = two[2, 3] = two[3, 2] = two[4, 5] = two[5, 4] = 1
    g2 = P.encode_graph(two, k_eig=3)
    assert g2.payload[5:8] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)  # three components
    tmp = P.encode_temporal([0.0] * 10 + [1.0] * 4, taus=(1, 4, 16))
    assert tmp.payload[0] == 1.0 and 1.0 > tmp.payload[1] > tmp.payload[2] > 0.0
    t2 = P.encode_temporal(np.ones((5, 2)), taus=(1, 2))
    assert t2.dim == 4


# --------------------------------------------------------------------------- fusion
def test_fusion_identity_with_zero_weights_and_no_packets():
    rng = np.random.default_rng(0)
    drive = rng.normal(size=8)
    fu = RepresentationFusion(8, mode='learned')
    pk = [P.encode_vector(rng.normal(size=5)), P.encode_spectral(rng.normal(size=16)), P.encode_scalar(3.0)]
    assert np.array_equal(fu.fuse(drive, []), drive)
    assert np.array_equal(fu.fuse(drive, pk), drive)
    out = fu.fuse(drive, pk)
    assert out is not drive


def test_fusion_configured_weights_change_output_and_reliability_downweights():
    drive = np.zeros(6)
    fu = RepresentationFusion(6, kinds=('vector', 'scalar'))
    vec = [1.0, -2.0, 0.5]
    fu.set_weight('vector', 1.0)
    full = fu.fuse(drive, [P.encode_vector(vec)])
    assert not np.allclose(full, 0)
    expected = fu.projection('vector', 3) @ np.array(vec)
    assert np.allclose(full, expected)
    half = fu.fuse(drive, [P.encode_vector(vec, confidence=0.5)])
    assert np.allclose(half, 0.5 * full)
    unc = fu.fuse(drive, [P.encode_vector(vec, uncertainty=0.75)])
    assert np.allclose(unc, 0.25 * full)
    assert np.array_equal(fu.fuse(drive, [P.encode_vector(vec, uncertainty=1.0)]), drive)
    before = fu.ignored
    assert np.array_equal(fu.fuse(drive, [P.encode_spectral(np.arange(8.0))]), drive)
    assert fu.ignored == before + 1
    with pytest.raises(ValueError):
        fu.set_weight('vector', 5.0)
    with pytest.raises(ValueError):
        fu.set_weight('phase', 0.1)
    with pytest.raises(ValueError):
        fu.learn(drive, [], drive)


def test_fusion_learned_delta_rule_converges_and_stays_bounded():
    rng = np.random.default_rng(1)
    fu = RepresentationFusion(10, kinds=('vector', 'population'), mode='learned', lr=0.2, weight_bound=2.0)
    vec = rng.normal(size=4)
    pk = [P.encode_vector(vec), P.encode_population(0.2, n=8)]
    drive = rng.normal(size=10)
    target = drive + 0.7 * (fu.projection('vector', 4) @ vec)
    losses = [fu.learn(drive, pk, target)['loss'] for _ in range(200)]
    assert losses[-1] < losses[0] * 0.05
    assert fu.weights['vector'] == pytest.approx(0.7, abs=0.05)
    far = drive + 50.0 * (fu.projection('vector', 4) @ vec)
    for _ in range(300):
        fu.learn(drive, pk, far)
    assert all(abs(w) <= 2.0 for w in fu.weights.values())
    st = json.loads(json.dumps(fu.export_state()))
    fu2 = RepresentationFusion(10, kinds=('vector', 'population'), mode='learned', lr=0.2, weight_bound=2.0)
    fu2.import_state(st)
    assert np.array_equal(fu2.fuse(drive, pk), fu.fuse(drive, pk))
    bad = copy.deepcopy(st); bad['weights']['vector'] = 9.0
    snap = fu2.export_state()
    with pytest.raises(ValueError):
        fu2.import_state(bad)
    assert fu2.export_state() == snap


def test_fusion_projection_deterministic_by_seed():
    a, b, c = RepresentationFusion(5, seed=1), RepresentationFusion(5, seed=1), RepresentationFusion(5, seed=2)
    assert np.array_equal(a.projection('vector', 3), b.projection('vector', 3))
    assert not np.array_equal(a.projection('vector', 3), c.projection('vector', 3))
    assert not np.array_equal(a.projection('vector', 3), a.projection('context', 3))


# --------------------------------------------------------------------------- events
def test_event_bus_publish_subscribe_history_counters():
    bus = EventBus(history_limit=3)
    seen, all_seen = [], []
    tok = bus.subscribe('prediction', seen.append)
    bus.subscribe('*', all_seen.append)
    for i in range(5):
        env = bus.publish('prediction', {'horizon': 4, 'error': 0.1 * i}, step=i, source='hierarchy')
        assert isinstance(env, Envelope) and env.schema == 'NeuroWeaveEventV1' and env.sequence == i
    bus.publish('telemetry', {'name': 'latency', 'value': 1.5, 'latency_ms': 1.5})
    assert len(seen) == 5 and len(all_seen) == 6
    hist = bus.history('prediction')
    assert [e.step for e in hist] == [2, 3, 4]
    assert bus.history('prediction', limit=1)[0].step == 4
    c = bus.counters()
    assert c['published']['prediction'] == 5 and c['delivered']['prediction'] == 10 and c['sequence'] == 6
    assert bus.unsubscribe(tok) and not bus.unsubscribe(tok)
    bus.publish('prediction', {'horizon': 1, 'error': 0.0})
    assert len(seen) == 5


def test_event_bus_isolates_failing_subscriber():
    bus = EventBus()
    good = []

    def boom(_env):
        raise RuntimeError('subscriber exploded')
    bus.subscribe('action', boom)
    bus.subscribe('action', good.append)
    bus.publish('action', {'action': [0.1, -0.2], 'authorship': 0.9})
    assert len(good) == 1
    assert bus.errors['action'] == 1 and bus.delivered['action'] == 1
    assert 'subscriber exploded' in bus.error_log[-1]['error']


def test_event_payload_validation_and_json_safety():
    assert set(INTERFACES) == set(TOPICS)
    for topic, spec in INTERFACES.items():
        assert spec['required'] and isinstance(spec['description'], str)
    validate_payload('memory', {'operation': 'store', 'id': 'm1', 'extra': [1, {'a': None}]})
    bus = EventBus()
    bad = [('prediction', {'horizon': 1}),
           ('prediction', {'horizon': '1', 'error': 0.1}),
           ('prediction', {'horizon': 1, 'error': float('nan')}),
           ('modulation', {'levels': {'dopamine': 'high'}}),
           ('input', {'channels': [0.1, True]}),
           ('telemetry', {'name': 'x', 'value': 1.0, 'blob': object()}),
           ('telemetry', {'name': 'x', 'value': 1.0, 'arr': np.zeros(2)}),
           ('nope', {})]
    for topic, payload in bad:
        with pytest.raises(ValueError):
            bus.publish(topic, payload)
    assert bus.sequence == 0
    # Non-strict mode still enforces JSON safety.
    bus.publish('state', {'anything': 1}, strict=False)
    with pytest.raises(ValueError):
        bus.publish('state', {'x': float('inf')}, strict=False)
    payload = {'activity_mean': 0.1, 'activity_std': 0.2, 'nested': {'a': [1]}}
    env = bus.publish('state', payload)
    payload['nested']['a'].append(2)
    assert env.payload['nested']['a'] == [1]  # envelope holds its own copy
    with pytest.raises(ValueError):
        Envelope(topic='state', step=-1, source='x', payload={}, sequence=0)


def test_event_bus_state_roundtrip_and_validation():
    bus = EventBus(history_limit=4)
    for i in range(6):
        bus.publish('learning', {'kind': 'rls', 'magnitude': i * 0.5, 'updates': i}, step=i)
    st = json.loads(json.dumps(bus.export_state()))
    other = EventBus(history_limit=4)
    seen = []
    other.subscribe('learning', seen.append)
    other.import_state(st)
    assert other.export_state() == bus.export_state()
    env = other.publish('learning', {'kind': 'rls', 'magnitude': 1.0})
    assert env.sequence == 6 and len(seen) == 1
    snap = other.export_state()
    bad = copy.deepcopy(st); bad['history']['learning'][0]['topic'] = 'action'
    with pytest.raises(ValueError):
        other.import_state(bad)
    bad = copy.deepcopy(st); bad['published']['input'] = -1
    with pytest.raises(ValueError):
        other.import_state(bad)
    bad = copy.deepcopy(st); bad['history_limit'] = 9
    with pytest.raises(ValueError):
        other.import_state(bad)
    assert other.export_state() == snap


# --------------------------------------------------------------------------- adversarial-review regressions
def test_update_only_path_normalises_errors_and_drives_gate():
    """Contract-style update(state, target) use must not saturate errors at the clip."""
    rng = np.random.default_rng(0)
    ph = PredictiveHierarchy(4, horizons=(1,), seed=1)
    A = np.eye(4) * 0.9
    x = rng.normal(size=4)
    errs = []
    for _ in range(300):
        y = A @ x + rng.normal(size=4) * 0.01
        errs.append(ph.update(x, y))
        x = y if np.linalg.norm(y) > 0.05 else rng.normal(size=4)
    assert ph.state_count == 300
    assert errs[-1] < 1.0 and ph.error_medium[1] < 0.5  # was pinned at ERROR_CLIP=10
    # disagreement EMA is fed by update(), so a disagreeing ensemble lowers the gate
    noisy = PredictiveHierarchy(2, horizons=(1,), seed=1, init_scale=3.0)
    for i in range(10):
        noisy.update([math.sin(i), math.cos(i)], [math.cos(i), math.sin(i)])
    assert noisy.disagreement[1] > 0.0 and noisy.uncertainty_gate() < 1.0
    with pytest.raises(ValueError):
        noisy.uncertainty_gate(horizon=5)
    with pytest.raises(ValueError):
        noisy.volatility(horizon=5)


def test_large_hierarchy_export_is_packed_and_roundtrips(monkeypatch):
    from synapse.neuroweave import prediction as pred_mod
    monkeypatch.setattr(pred_mod, 'PACKED_MIN_SIZE', 8)  # force packing at a small, fast size
    rng = np.random.default_rng(3)
    a = PredictiveHierarchy(5, horizons=(1, 4), seed=2)
    for _ in range(12):
        a.observe(rng.normal(size=5))
    st = json.loads(json.dumps(a.export_state()))
    assert isinstance(st['horizons']['1']['W'][0], dict) and 'b64' in st['horizons']['1']['W'][0]
    b = PredictiveHierarchy(5, horizons=(1, 4), seed=2)
    b.import_state(st)
    for h in a.horizons:
        for w1, w2 in zip(a.W[h], b.W[h]):
            assert np.array_equal(w1, w2)
    x = rng.normal(size=5)
    assert a.observe(x)['scored'] == b.observe(x)['scored']
    before = _snapshot(b)
    bad = copy.deepcopy(st); bad['horizons']['1']['W'][0]['b64'] = bad['horizons']['1']['W'][0]['b64'][:-8]
    with pytest.raises(ValueError):
        b.import_state(bad)
    bad = copy.deepcopy(st); bad['horizons']['1']['W'][0]['dtype'] = '<f4'
    with pytest.raises(ValueError):
        b.import_state(bad)
    assert _snapshot(b) == before


def test_hierarchy_import_rejects_bad_P_and_bias():
    ph = PredictiveHierarchy(2, horizons=(1, 4), seed=1)
    for i in range(6):
        ph.observe([i * 0.1, -i * 0.1])
    good = ph.export_state()
    before = _snapshot(ph)
    cases = []
    s = copy.deepcopy(good); s['horizons']['1']['P'][0][0][0] = 1e9; cases.append(s)       # trace bound
    s = copy.deepcopy(good); s['horizons']['1']['P'][0][0][1] = 5.0; cases.append(s)       # asymmetric
    s = copy.deepcopy(good); s['horizons']['1']['P'][0][1][1] = -1.0; cases.append(s)      # negative diag
    s = copy.deepcopy(good); s['buffer'][0]['phi'][-1] = 0.0; cases.append(s)               # bias lost
    s = copy.deepcopy(good); s['state_stats']['var'] = [1e20, 1.0]; cases.append(s)
    for bad in cases:
        with pytest.raises(ValueError):
            ph.import_state(bad)
        assert _snapshot(ph) == before
    em, _ = _train_efference(steps=5)
    st = em.export_state()
    st['P'][0][0] = 1e9
    with pytest.raises(ValueError):
        em.import_state(st)


def test_packet_rejects_string_payload_source_and_metadata():
    base = P.encode_vector([1.0, 2.0]).to_dict()
    for key, val in (('payload', '12'), ('source', 'abc'), ('metadata', 'ab'), ('metadata', ['ab'])):
        d = dict(base); d[key] = val; d.pop('checksum')
        with pytest.raises(ValueError):
            RepresentationPacket.from_dict(d)
    with pytest.raises(ValueError):
        RepresentationPacket(kind='vector', payload='12')
    with pytest.raises(ValueError):
        RepresentationPacket(kind='vector', payload=(np.bool_(True), 1.0))


def test_fusion_skips_zero_weight_projection_and_bounds_cache(monkeypatch):
    rng = np.random.default_rng(0)
    f = RepresentationFusion(16, seed=3)
    big = P.encode_sparse(rng.normal(size=4096), 7)
    drive = rng.normal(size=16)
    assert np.array_equal(f.fuse(drive, [big]), drive)
    assert f._proj_cache == {}  # zero weight: nothing projected
    f.set_weight('sparse', 1.0)
    out = f.fuse(drive, [big])
    expected = drive + f.projection('sparse', 4096) @ big.dense()
    assert np.allclose(out, expected, atol=1e-12)
    binary = P.encode_sparse(rng.normal(size=32), 4, form='binary')
    idx = RepresentationPacket(kind='sparse', payload=tuple(float(i) for i in np.flatnonzero(binary.dense())),
                               form='index', dim=32)
    assert np.allclose(f.fuse(drive, [binary]), f.fuse(drive, [idx]), atol=1e-12)
    # invalid packet in the list: no counter is touched
    f2 = RepresentationFusion(4, kinds=('scalar',))
    with pytest.raises(ValueError):
        f2.fuse(np.zeros(4), [P.encode_vector([1.0]), 'junk'])
    assert f2.ignored == 0
    from synapse.neuroweave import packets as pk_mod
    monkeypatch.setattr(pk_mod, 'PROJECTION_CACHE_ELEMENTS', 100)
    f3 = RepresentationFusion(10, seed=1)
    p1 = f3.projection('vector', 6).copy()
    for n in range(1, 30):
        f3.projection('vector', n)
        assert f3._proj_elements <= 100
    assert np.array_equal(f3.projection('vector', 6), p1)  # regenerated identically after eviction


def test_event_subscriber_cannot_corrupt_history_or_other_subscribers():
    bus = EventBus()
    seen = []
    bus.subscribe('telemetry', lambda e: e.payload.__setitem__('value', 999.0))
    bus.subscribe('telemetry', lambda e: seen.append(e.payload['value']))
    env = bus.publish('telemetry', {'name': 'x', 'value': 1.0})
    assert seen == [1.0]
    assert env.payload['value'] == 1.0 and bus.history('telemetry')[0].payload['value'] == 1.0
    assert bus.counters()['errors']['telemetry'] == 0
    with pytest.raises(ValueError):
        bus.publish('telemetry', [('name', 'x')], strict=False)
    st = bus.export_state()
    before = json.dumps(bus.export_state(), sort_keys=True)
    bad = copy.deepcopy(st); bad['strict'] = 'no'
    with pytest.raises(ValueError):
        bus.import_state(bad)
    bus.publish('state', {'activity_mean': 0.1, 'activity_std': 0.2})
    st2 = bus.export_state()
    bad = copy.deepcopy(st2); bad['history']['state'][0]['sequence'] = 0  # duplicates telemetry seq 0
    with pytest.raises(ValueError):
        bus.import_state(bad)
    assert json.dumps(bus.export_state(), sort_keys=True) == json.dumps(st2, sort_keys=True) != before
