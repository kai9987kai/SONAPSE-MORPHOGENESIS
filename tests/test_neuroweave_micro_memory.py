"""Tests for the NEUROWEAVE Micro-Collective, SpecialistGraft and MemoryEcology."""
from __future__ import annotations

import json

import numpy as np
import pytest

from synapse.neuroweave.memory import MemoryEcology, Provenance
from synapse.neuroweave.micro_collective import MicroCollective, SpecialistGraft
from synapse.neuroweave.spec import EVIDENCE_TIER, NeuroWeaveSpecV1


# ============================================================================ fixtures
def rotating_sinusoid(steps: int, omega: float = 0.5, dim: int = 16) -> np.ndarray:
    j = np.arange(dim)
    return np.array([np.sin(omega * t + 2 * np.pi * j / dim) for t in range(steps + 1)])


SEQ = rotating_sinusoid(400)


def train(mc: MicroCollective, steps: int, seq: np.ndarray = SEQ) -> list[dict]:
    return [mc.train_step(seq[t], seq[t + 1]) for t in range(steps)]


@pytest.fixture(scope='module')
def trained() -> MicroCollective:
    mc = MicroCollective(seed=3)
    train(mc, 300)
    return mc


def _json_roundtrip(state: dict) -> dict:
    return json.loads(json.dumps(state, allow_nan=False))


# ============================================================================ MicroCollective
def test_validation_loss_beats_persistence_after_training(trained):
    fresh = MicroCollective(seed=3)
    train(fresh, 40)
    early = fresh.evaluate()
    late = trained.evaluate()
    pers = trained.persistence_loss()
    assert pers is not None and pers > 0.05
    assert late < early
    assert late < pers
    r = trained.receipt()
    assert r['beats_persistence'] is True
    assert r['validation_loss'] == pytest.approx(late)
    assert r['persistence_loss'] == pytest.approx(pers)
    assert r['evidence_tier'] == EVIDENCE_TIER


def test_held_out_split_never_trained(trained):
    every = trained.validation_every
    assert len(trained.validation) == min(300 // every, trained.validation_capacity)
    held = {tuple(np.round(it['x'], 12)) for it in trained.validation}
    trained_x = {tuple(np.round(it['x'], 12)) for it in trained.replay}
    # sinusoid with irrational period: inputs never repeat, so held-out and replay are disjoint
    assert held.isdisjoint(trained_x)
    mc = MicroCollective(seed=1)
    out = train(mc, 10)
    assert [o['held_out'] for o in out] == [(i + 1) % every == 0 for i in range(10)]
    assert all(o['loss'] is None for o in out if o['held_out'])


def test_replay_buffer_bounded():
    mc = MicroCollective(seed=2, replay_capacity=20, validation_capacity=5)
    train(mc, 120)
    assert len(mc.replay) == 20
    assert len(mc.validation) == 5


def test_top_k_routing_exact():
    for experts, k in [(4, 2), (6, 3), (5, 1), (3, 3), (1, 1)]:
        mc = MicroCollective(experts=experts, top_k=k, seed=11)
        x = SEQ[7]
        y, routing = mc.predict(x)
        probs = np.array(routing['probabilities'])
        assert len(routing['experts']) == k
        assert set(routing['experts']) == set(np.argsort(-probs, kind='stable')[:k].tolist())
        assert sum(routing['weights']) == pytest.approx(1.0)
        # manual recomputation of the sparse mixture
        manual = np.zeros(16)
        for e, g in zip(routing['experts'], routing['weights']):
            h = np.tanh(mc.w1[e] @ x + mc.b1[e])
            manual += g * (mc.w2[e] @ h + mc.b2[e])
        np.testing.assert_allclose(y, manual, atol=1e-12)
        # unrouted experts have no influence on the output
        others = [e for e in range(experts) if e not in routing['experts']]
        if others:
            mc.w2[others] += 1.0
            np.testing.assert_allclose(mc.predict(x)[0], y, atol=1e-12)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        MicroCollective(experts=2, top_k=3)
    with pytest.raises(ValueError):
        MicroCollective(experts=0)
    mc = MicroCollective(seed=1)
    with pytest.raises(ValueError):
        mc.predict(np.zeros(15))
    with pytest.raises(ValueError):
        mc.train_step(np.full(16, np.nan), np.zeros(16))


def test_unrouted_experts_receive_no_gradient():
    mc = MicroCollective(seed=4, replay_batch=0)
    x, t = SEQ[0], SEQ[1]
    _, routing = mc.predict(x)
    before = mc.w1.copy()
    mc.train_step(x, t)
    changed = [e for e in range(mc.n_experts) if not np.array_equal(before[e], mc.w1[e])]
    assert sorted(changed) == sorted(routing['experts'])


def test_load_balancing_reduces_imbalance():
    bal = MicroCollective(seed=3, balanced=True)
    unbal = MicroCollective(seed=3, balanced=False)
    train(bal, 300)
    train(unbal, 300)
    assert np.all(unbal.balance_bias == 0.0)
    assert np.any(bal.balance_bias != 0.0)
    assert bal.expert_load['cv'] < unbal.expert_load['cv']
    assert bal.expert_load['cv'] < 0.6 * unbal.expert_load['cv']
    assert sum(bal.expert_load['counts']) == sum(unbal.expert_load['counts'])


def test_from_spec_flags_map_to_behaviour():
    off = MicroCollective.from_spec(NeuroWeaveSpecV1())
    assert (off.balanced, off.prioritized_replay, off.rollback_enabled) == (False, False, False)
    on = MicroCollective.from_spec(NeuroWeaveSpecV1(micro_experts=6, micro_top_k=3,
                                                    mechanisms=('expert_routing', 'predictive_replay', 'meta_plasticity')))
    assert (on.balanced, on.prioritized_replay, on.rollback_enabled) == (True, True, True)
    assert on.n_experts == 6 and on.top_k == 3


def test_novelty_prioritised_sampling():
    def counts(prioritized: bool) -> int:
        mc = MicroCollective(seed=5, prioritized_replay=prioritized)
        for i in range(40):
            mc.replay.append({'x': np.zeros(16), 'y': np.zeros(16), 'novelty': 0.001})
        mc.replay[17]['novelty'] = 5.0
        return sum(17 in mc.sample_replay(4) for _ in range(200))
    hi = counts(True)
    uni = counts(False)
    assert hi > 190          # nearly always drawn when prioritised
    assert uni < 60          # ~ 4/40 * 200 = 20 under uniform sampling


def test_rollback_on_corrupted_weights():
    mc = MicroCollective(seed=3)
    train(mc, 150)
    first = mc.validate()
    assert first['event'] in ('improved', 'none')
    best_pred = mc._predict_batch(np.stack([it['x'] for it in mc.validation]), mc.best)
    rollbacks = mc.rollbacks
    lr = mc.lr
    mc.w2 *= 0.0
    mc.w2 += 3.0                            # corrupt
    corrupted = mc.evaluate()
    res = mc.validate()
    assert res['rolled_back'] is True and res['event'] == 'rollback'
    assert mc.rollbacks == rollbacks + 1
    assert mc.lr < lr
    assert res['validation_loss'] < corrupted
    restored = mc._predict_batch(np.stack([it['x'] for it in mc.validation]))
    np.testing.assert_allclose(restored, best_pred, atol=1e-12)
    assert mc.receipts[-1]['event'] == 'rollback'


def test_no_rollback_when_disabled():
    mc = MicroCollective(seed=3, rollback=False)
    train(mc, 150)
    mc.validate()
    mc.w2 += 3.0
    corrupted = mc.w2.copy()
    res = mc.validate()
    assert res['rolled_back'] is False and mc.rollbacks == 0
    np.testing.assert_array_equal(mc.w2, corrupted)
    assert res['validation_loss'] > mc.best_loss


def test_receipt_sha256_tracks_weights_and_counters(trained):
    r1 = trained.receipt()
    assert len(r1['sha256']) == 64
    assert trained.receipt()['sha256'] == r1['sha256']
    mc = MicroCollective(seed=3)
    d0 = mc.weights_digest()
    mc.train_step(SEQ[0], SEQ[1])
    assert mc.weights_digest() != d0
    clone = MicroCollective(seed=3)
    clone.import_state(mc.export_state())
    assert clone.weights_digest() == mc.weights_digest()
    clone.samples += 1
    assert clone.weights_digest() != mc.weights_digest()
    json.dumps(r1, allow_nan=False)


def test_add_expert_growth():
    mc = MicroCollective(seed=3)
    train(mc, 30)
    idx = mc.add_expert()
    assert idx == 4 and mc.n_experts == 5 and mc.best is None
    assert mc.w1.shape[0] == 5 and len(mc.expert_load['counts']) == 5
    train(mc, 120, SEQ[30:])
    assert mc.expert_load['counts'][4] > 0            # new expert gets routed (load balancing)
    y, routing = mc.predict(SEQ[0])
    assert np.all(np.isfinite(y)) and len(routing['experts']) == 2
    state = _json_roundtrip(mc.export_state())
    other = MicroCollective(seed=3)
    other.import_state(state)
    assert other.n_experts == 5
    np.testing.assert_array_equal(other.predict(SEQ[3])[0], mc.predict(SEQ[3])[0])


def test_export_import_round_trip_and_continuation(trained):
    state = _json_roundtrip(trained.export_state())
    assert set(state) == {'schema', 'schema_version', 'evidence_tier', 'config', 'weights', 'load', 'counters',
                          'lr', 'replay', 'validation', 'best', 'best_loss', 'last_validation_loss',
                          'receipts', 'rng'}
    other = MicroCollective(seed=99)
    other.import_state(state)
    for t in range(20):
        np.testing.assert_array_equal(other.predict(SEQ[t])[0], trained.predict(SEQ[t])[0])
    # continued training is identical too (rng state restored)
    a = MicroCollective(seed=3); b = MicroCollective(seed=3)
    train(a, 60)
    b.import_state(_json_roundtrip(a.export_state()))
    for t in range(60, 90):
        a.train_step(SEQ[t], SEQ[t + 1]); b.train_step(SEQ[t], SEQ[t + 1])
    assert a.export_state() == b.export_state()


def test_import_rejects_bad_state_without_mutation(trained):
    good = trained.export_state()
    mc = MicroCollective(seed=3)
    before = mc.export_state()
    bad_cases = []
    s = json.loads(json.dumps(good)); s['weights']['w1'][0][0][0] = float('nan'); bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['weights']['w2'] = s['weights']['w2'][:-1]; bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['config']['top_k'] = 9; bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['schema'] = 'other'; bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['replay'][0]['x'] = [0.0]; bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['rng']['sample'] = {'bit_generator': 'MT19937'}; bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['config']['input_dim'] = 8; bad_cases.append(s)
    s = json.loads(json.dumps(good)); s['weights']['w1'][0][0][0] = 1e6; bad_cases.append(s)
    for bad in bad_cases:
        with pytest.raises(ValueError):
            mc.import_state(bad)
        assert mc.export_state() == before
    with pytest.raises(ValueError):
        mc.import_state('nope')


def test_deterministic_and_seed_sensitive():
    a = MicroCollective(seed=8); b = MicroCollective(seed=8); c = MicroCollective(seed=9)
    ra = train(a, 60); rb = train(b, 60); train(c, 60)
    assert a.export_state() == b.export_state()
    assert [r['loss'] for r in ra] == [r['loss'] for r in rb]
    assert a.weights_digest() != c.weights_digest()


def test_state_bounded_and_finite_under_extreme_input():
    mc = MicroCollective(seed=1, lr=1.0)
    rng = np.random.default_rng(0)
    for _ in range(60):
        mc.train_step(rng.normal(0, 1e3, 16), rng.normal(0, 1e3, 16))
    for arr in (mc.w1, mc.w2, mc.b1, mc.b2, mc.router_w):
        assert np.all(np.isfinite(arr)) and np.max(np.abs(arr)) <= 5.0
    assert np.max(np.abs(mc.router_b)) <= 2.0 and np.max(np.abs(mc.balance_bias)) <= 2.0
    y, _ = mc.predict(rng.normal(0, 1e3, 16))
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 10.0


def test_non_square_dims_have_no_persistence_baseline():
    mc = MicroCollective(input_dim=8, output_dim=4, seed=1)
    rng = np.random.default_rng(1)
    for _ in range(30):
        mc.train_step(rng.normal(size=8), rng.normal(size=4))
    assert mc.persistence_loss() is None
    assert mc.receipt()['beats_persistence'] is False
    assert mc.receipt()['zero_baseline_loss'] is not None


# ============================================================================ SpecialistGraft
def test_graft_zero_contribution_at_init():
    g = SpecialistGraft(16, 16, seed=5)
    assert g.coupling == 0.0
    for t in range(10):
        g.receive_main_prediction(SEQ[t + 1])
        g.train_step(SEQ[t], SEQ[t + 1])
        np.testing.assert_array_equal(g.contribute(SEQ[t]), np.zeros(16))
    assert np.any(g.emit_prediction() != 0.0)
    g.set_coupling(0.3)
    contrib = g.contribute(SEQ[3])
    assert np.any(contrib != 0.0)
    for bad in (0.6, -0.1, float('nan')):
        with pytest.raises(ValueError):
            g.set_coupling(bad)


def test_graft_coupling_zero_leaves_host_unchanged_and_detach_removes():
    host = MicroCollective(seed=3)
    plain = MicroCollective(seed=3)
    g = SpecialistGraft(16, 16, seed=5)
    host.attach_specialist(g)
    train(host, 79)
    train(plain, 79)
    sent = plain._predict_main(SEQ[79])
    host.train_step(SEQ[79], SEQ[80])
    plain.train_step(SEQ[79], SEQ[80])
    # zero coupling: identical predictions and identical main-model state
    np.testing.assert_array_equal(host.predict(SEQ[90])[0], plain.predict(SEQ[90])[0])
    assert g.samples == 80 and g.updates > 0
    # exchange: the graft received the main prediction as context
    np.testing.assert_array_equal(g.context, sent)
    g.set_coupling(0.5)
    y_coupled, routing = host.predict(SEQ[90])
    assert routing['specialist_coupling'] == 0.5
    assert not np.allclose(y_coupled, plain.predict(SEQ[90])[0])
    receipt = g.detach()
    assert receipt['attached'] is False and receipt['evidence_tier'] == EVIDENCE_TIER
    assert host.specialist is None and g._host is None
    np.testing.assert_array_equal(host.predict(SEQ[90])[0], plain.predict(SEQ[90])[0])
    hs, ps = host.export_state(), plain.export_state()
    # receipts written while attached legitimately record the graft; nothing else differs
    assert any(r['specialist'] is not None for r in hs['receipts'])
    strip = lambda rs: [{k: v for k, v in r.items() if k != 'specialist'} for r in rs]
    assert strip(hs.pop('receipts')) == strip(ps.pop('receipts'))
    assert hs == ps
    train(host, 10, SEQ[100:])
    train(plain, 10, SEQ[100:])
    assert host.weights_digest() == plain.weights_digest()
    np.testing.assert_array_equal(host.predict(SEQ[5])[0], plain.predict(SEQ[5])[0])
    np.testing.assert_array_equal(g.contribute(SEQ[0]), np.zeros(16))
    with pytest.raises(RuntimeError):
        g.train_step(SEQ[0], SEQ[1])
    with pytest.raises(ValueError):
        host.attach_specialist(g)


def test_graft_context_changes_its_prediction():
    g = SpecialistGraft(16, 16, seed=5)
    g.set_coupling(0.5)
    a = g.contribute(SEQ[0])
    g.receive_main_prediction(np.ones(16))
    b = g.contribute(SEQ[0])
    assert not np.allclose(a, b)


def test_graft_learns_with_independent_validation():
    g = SpecialistGraft(16, 16, seed=5)
    seq = rotating_sinusoid(400)
    first = None
    for t in range(300):
        g.receive_main_prediction(np.zeros(16))
        out = g.train_step(seq[t], seq[t + 1])
        if out['validation_loss'] is not None and first is None:
            first = out['validation_loss']
    r = g.receipt()
    assert r['validation_loss'] is not None and r['validation_loss'] < first
    assert r['validation_loss'] < r['persistence_loss']
    assert len(r['sha256']) == 64
    assert r['validation_samples'] == 60


def test_graft_round_trip_and_validation():
    g = SpecialistGraft(16, 16, seed=5)
    for t in range(40):
        g.receive_main_prediction(SEQ[t])
        g.train_step(SEQ[t], SEQ[t + 1])
    g.set_coupling(0.2)
    state = _json_roundtrip(g.export_state())
    h = SpecialistGraft(16, 16, seed=1)
    h.import_state(state)
    np.testing.assert_array_equal(h.contribute(SEQ[50]), g.contribute(SEQ[50]))
    np.testing.assert_array_equal(h.step(SEQ[50]), g.step(SEQ[50]))
    before = h.export_state()
    bad = json.loads(json.dumps(state)); bad['coupling'] = 0.9
    with pytest.raises(ValueError):
        h.import_state(bad)
    bad = json.loads(json.dumps(state)); bad['weights']['wo'] = [[0.0]]
    with pytest.raises(ValueError):
        h.import_state(bad)
    assert h.export_state() == before


# ============================================================================ MemoryEcology
def _unit(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def test_provenance_enum():
    assert {p.value for p in Provenance} == {'sensory', 'predicted', 'external', 'experience', 'synthetic'}
    m = MemoryEcology(8, 4, seed=1)
    with pytest.raises(ValueError):
        m.store(np.ones(8), np.ones(4), provenance='dreamed')


def test_flyhash_sparse_and_locality():
    m = MemoryEcology(16, 4, seed=2)
    rng = np.random.default_rng(0)
    a = _unit(rng)
    code = m.flyhash(a)
    assert len(code) == m.hash_k and m.hash_k < m.hash_dim / 4
    assert m.flyhash(a) == code
    assert m.flyhash(3.0 * a) == code                 # scale invariant
    near = len(code & m.flyhash(a + 0.05 * _unit(rng)))
    far = np.mean([len(code & m.flyhash(_unit(rng))) for _ in range(20)])
    assert near > far


def test_retrieval_uses_sparse_index():
    rng = np.random.default_rng(1)
    m = MemoryEcology(16, 4, capacity=400, seed=3)
    keys = [_unit(rng) for _ in range(300)]
    for i, k in enumerate(keys):
        m.store(k, np.full(4, i / 300), provenance=Provenance.SENSORY)
    q = keys[42] + 0.02 * _unit(rng)
    res = m.retrieve(q, k=1)
    assert res[0]['id'] == 42
    assert 0 < m.last_candidate_count < len(m)
    ex = m.retrieve(q, k=1, exhaustive=True)
    assert ex[0]['id'] == 42 and m.last_candidate_count == len(m)


def test_context_and_reward_bias_change_ranking():
    m = MemoryEcology(8, 2, seed=4, context_dim=4, interference_penalty=0.0)
    key = np.ones(8)
    a = m.store(key, [1, 0], provenance='sensory', context=[1, 0, 0, 0], reward=0.0)
    b = m.store(key, [0, 1], provenance='sensory', context=[0, 1, 0, 0], reward=1.0)
    assert m.retrieve(key, context=[1, 0, 0, 0], k=1)[0]['id'] == a
    assert m.retrieve(key, context=[0, 1, 0, 0], k=1)[0]['id'] == b
    no_bias = m.retrieve(key, k=2)
    assert no_bias[0]['score'] == pytest.approx(no_bias[1]['score'])
    biased = m.retrieve(key, k=2, reward_bias=0.5)
    assert biased[0]['id'] == b and biased[0]['score'] > biased[1]['score']


def test_verified_only_excludes_predicted_and_synthetic():
    m = MemoryEcology(8, 2, seed=5)
    key = np.arange(8, dtype=float)
    ids = {p: m.store(key + 0.01 * i, [i, i], provenance=p) for i, p in enumerate(Provenance)}
    all_ids = {r['id'] for r in m.retrieve(key, k=10)}
    ver_ids = {r['id'] for r in m.retrieve(key, k=10, verified_only=True)}
    assert all_ids == set(ids.values())
    assert ver_ids == {ids[Provenance.SENSORY], ids[Provenance.EXTERNAL], ids[Provenance.EXPERIENCE]}
    assert all(r['verified'] for r in m.retrieve(key, k=10, verified_only=True))


def test_confirm_promotes_or_records_contradiction():
    m = MemoryEcology(8, 3, seed=6)
    key = np.ones(8)
    pid = m.store(key, [1.0, 2.0, 3.0], provenance=Provenance.PREDICTED, confidence=0.6)
    assert m.retrieve(key, verified_only=True) == []
    bad = m.confirm(pid, [3.0, 0.0, 1.0], tol=0.1)
    assert bad['confirmed'] is False and m.get(pid)['provenance'] == 'predicted'
    assert m.get(pid)['contradictions'] == 1 and m.counters['contradictions'] == 1
    assert m.get(pid)['confidence'] == pytest.approx(0.3)
    assert m.contradiction_log[-1]['id'] == pid
    assert m.retrieve(key, verified_only=True) == []
    good = m.confirm(pid, [1.05, 2.0, 2.95], tol=0.1)
    assert good['confirmed'] is True and m.get(pid)['provenance'] == 'experience'
    np.testing.assert_allclose(m.get(pid)['value'], [1.05, 2.0, 2.95])
    assert [r['id'] for r in m.retrieve(key, verified_only=True)] == [pid]
    with pytest.raises(ValueError):
        m.confirm(pid, [1, 2, 3], tol=0.1)            # no longer PREDICTED
    sid = m.store(key, [0, 0, 0], provenance=Provenance.SYNTHETIC)
    with pytest.raises(ValueError):
        m.confirm(sid, [0, 0, 0], tol=1.0)            # synthetic cannot be promoted
    with pytest.raises(ValueError):
        m.confirm(999, [0, 0, 0], tol=1.0)


def test_interference_tracking():
    m = MemoryEcology(8, 2, seed=7)
    key = np.linspace(-1, 1, 8)
    a = m.store(key, [1.0, 0.0], provenance='sensory')
    m.store(key * 1.01, [1.0, 0.01], provenance='sensory')        # similar value: no interference
    assert m.counters['interference_events'] == 0
    conf_before = m.get(a)['confidence']
    m.store(key, [-1.0, 1.0], provenance='sensory')               # same key, conflicting value
    assert m.counters['interference_events'] >= 1
    assert m.get(a)['interference'] == 1
    assert m.get(a)['confidence'] < conf_before
    assert 0.0 < m.interference_score <= 1.0
    m.store(-key, [5.0, 5.0], provenance='sensory')               # dissimilar key: no new event
    assert m.get(a)['interference'] == 1


def test_multi_timescale_traces_and_default_surprise():
    m = MemoryEcology(4, 2, seed=8)
    for _ in range(30):
        m.store(np.ones(4), [0.0, 0.0], provenance='sensory')
    s_old = m.get(m.store(np.ones(4), [0.0, 0.0], provenance='sensory'))['surprise']
    for _ in range(5):
        m.store(np.ones(4), [4.0, 4.0], provenance='sensory')
    fast, mid, slow = (m.traces[n] for n in ('fast', 'mid', 'slow'))
    assert fast[0] > mid[0] > slow[0] > 0.0
    assert np.linalg.norm(m.trace_contrast()) > 0
    s_new = m.get(m.store(np.ones(4), [-4.0, -4.0], provenance='sensory'))['surprise']
    assert s_new > s_old


def test_replay_scheduler_priority_and_rotation():
    m = MemoryEcology(4, 1, seed=9)
    low = m.store([1, 0, 0, 0], [0], provenance='sensory', reward=0.0, surprise=0.1)
    high = m.store([0, 1, 0, 0], [0], provenance='sensory', reward=1.0, surprise=0.9)
    mid = m.store([0, 0, 1, 0], [0], provenance='sensory', reward=0.5, surprise=0.5)
    pred = m.store([0, 0, 0, 1], [0], provenance='predicted', reward=1.0, surprise=1.0)
    assert m.replay_priority(high) > m.replay_priority(mid) > m.replay_priority(low)
    first = m.replay_schedule(2, verified_only=True)
    assert [r['id'] for r in first] == [high, mid]
    assert pred not in [r['id'] for r in m.replay_schedule(10, verified_only=True)]
    m2 = MemoryEcology(4, 1, seed=9)
    a = m2.store([1, 0, 0, 0], [0], provenance='sensory', reward=1.0, surprise=0.6)
    b = m2.store([0, 1, 0, 0], [0], provenance='sensory', reward=1.0, surprise=0.5)
    assert m2.replay_schedule(1)[0]['id'] == a
    assert m2.replay_schedule(1)[0]['id'] == b       # a's surprise was decayed by replay
    assert m2.get(a)['replays'] == 1 and m2.counters['replays'] == 2
    # recency: an old item loses priority relative to a fresh identical one
    m3 = MemoryEcology(4, 1, seed=9, recency_tau=5.0)
    old = m3.store([1, 0, 0, 0], [0], provenance='sensory', reward=1.0, surprise=0.5)
    m3.tick(20)
    new = m3.store([0, 1, 0, 0], [0], provenance='sensory', reward=1.0, surprise=0.5)
    assert m3.replay_priority(new) > m3.replay_priority(old)


def test_stochastic_replay_deterministic_per_seed():
    def run(seed):
        m = MemoryEcology(4, 1, seed=seed)
        rng = np.random.default_rng(0)
        for i in range(20):
            m.store(rng.normal(size=4), [i], provenance='sensory', reward=float(rng.uniform(-1, 1)))
        return [r['id'] for r in m.replay_schedule(5, stochastic=True)]
    assert run(1) == run(1)
    assert len(set(run(1))) == 5


def test_confidence_decay_and_forgetting():
    m = MemoryEcology(4, 1, seed=10, confidence_decay=0.9, forget_threshold=0.1)
    a = m.store([1, 0, 0, 0], [0], provenance='sensory', confidence=1.0)
    b = m.store([0, 1, 0, 0], [0], provenance='sensory', confidence=0.15)
    out = m.tick(5)
    assert b in out['forgotten'] and b not in m and a in m
    assert m.get(a)['confidence'] == pytest.approx(0.9 ** 5)
    assert m.counters['forgotten'] == 1


def test_downscale_experiment():
    m = MemoryEcology(4, 1, seed=11, forget_threshold=0.1)
    a = m.store([1, 0, 0, 0], [0], provenance='sensory', confidence=1.0, reward=0.8)
    b = m.store([0, 1, 0, 0], [0], provenance='sensory', confidence=0.3, reward=-0.4)
    res = m.downscale(0.5)
    assert res['forgotten'] == [] and res['mean_confidence_after'] == pytest.approx(0.5 * res['mean_confidence_before'])
    assert m.get(a)['reward'] == pytest.approx(0.4)
    res = m.downscale(0.5)
    assert res['forgotten'] == [b] and a in m
    with pytest.raises(ValueError):
        m.downscale(1.5)


def test_capacity_eviction_lowest_priority():
    m = MemoryEcology(4, 1, capacity=3, seed=12)
    keep1 = m.store([1, 0, 0, 0], [0], provenance='sensory', reward=1.0, surprise=0.9)
    weak = m.store([0, 1, 0, 0], [0], provenance='sensory', reward=0.0, surprise=0.0, confidence=0.2)
    keep2 = m.store([0, 0, 1, 0], [0], provenance='sensory', reward=0.7, surprise=0.5)
    new = m.store([0, 0, 0, 1], [0], provenance='sensory', reward=0.5, surprise=0.5)
    assert len(m) == 3 and weak not in m
    assert {keep1, keep2, new} == set(m.items)
    assert m.counters['evictions'] == 1
    assert all(r['id'] != weak for r in m.retrieve([0, 1, 0, 0], k=3, exhaustive=True))


def test_cross_region_sharing():
    m = MemoryEcology(4, 2, seed=13)
    key = np.array([1.0, 0.5, 0.0, -0.5])
    vis = m.store(key, [1, 1], provenance='sensory', region='visual_sensory')
    m.store(key, [1, 1], provenance='sensory', region='working_memory')
    assert [r['id'] for r in m.retrieve(key, region='visual_sensory', k=5)] == [vis]
    assert m.retrieve(key, region='episodic_memory') == []
    shared = m.share(vis, 'episodic_memory')
    got = m.retrieve(key, region='episodic_memory')
    assert [r['id'] for r in got] == [shared] and m.get(shared)['shared_from'] == vis
    assert got[0]['provenance'] == 'sensory'
    assert m.region_counts() == {'episodic_memory': 1, 'visual_sensory': 1, 'working_memory': 1}
    assert len(m.retrieve(key, k=5)) == 3
    pid = m.store(key, [0, 0], provenance='predicted', region='visual_sensory')
    sp = m.share(pid, 'working_memory')
    assert m.get(sp)['provenance'] == 'predicted'     # sharing never upgrades provenance
    assert sp not in [r['id'] for r in m.retrieve(key, region='working_memory', verified_only=True, k=5)]


def test_memory_export_import_round_trip():
    rng = np.random.default_rng(2)
    m = MemoryEcology(16, 4, capacity=50, seed=14, context_dim=3)
    for i in range(60):
        prov = list(Provenance)[i % 5]
        m.store(_unit(rng), rng.normal(size=4), provenance=prov, reward=float(rng.uniform(-1, 1)),
                context=rng.normal(size=3) if i % 2 else None, region='association_a' if i % 3 else None)
    pid = next(i for i, it in m.items.items() if it['provenance'] is Provenance.PREDICTED)
    m.confirm(pid, m.get(pid)['value'] + 5.0, tol=0.1)
    m.tick(3)
    state = _json_roundtrip(m.export_state())
    assert state['raw_media_stored'] is False and state['evidence_tier'] == EVIDENCE_TIER
    other = MemoryEcology(16, 4, capacity=50, seed=14, context_dim=3)
    other.import_state(state)
    assert other.export_state() == m.export_state()
    q = _unit(rng)
    assert [r['id'] for r in other.retrieve(q, k=5, touch=False)] == [r['id'] for r in m.retrieve(q, k=5, touch=False)]
    assert [r['id'] for r in other.replay_schedule(3, stochastic=True)] == \
        [r['id'] for r in m.replay_schedule(3, stochastic=True)]


def test_memory_import_validation_atomic():
    m = MemoryEcology(4, 2, seed=15)
    m.store([1, 0, 0, 0], [1, 1], provenance='sensory')
    good = m.export_state()
    target = MemoryEcology(4, 2, seed=15)
    target.store([0, 1, 0, 0], [2, 2], provenance='external')
    before = target.export_state()
    cases = []
    s = json.loads(json.dumps(good)); s['items'][0]['value'] = [float('nan'), 0.0]; cases.append(s)
    s = json.loads(json.dumps(good)); s['items'][0]['provenance'] = 'imagined'; cases.append(s)
    s = json.loads(json.dumps(good)); s['config']['seed'] = 1; cases.append(s)
    s = json.loads(json.dumps(good)); s['raw_media_stored'] = True; cases.append(s)
    s = json.loads(json.dumps(good)); s['items'].append(dict(s['items'][0])); cases.append(s)
    s = json.loads(json.dumps(good)); s['items'][0]['confidence'] = 1.5; cases.append(s)
    s = json.loads(json.dumps(good)); s['traces']['fast'] = [0.0]; cases.append(s)
    for bad in cases:
        with pytest.raises(ValueError):
            target.import_state(bad)
        assert target.export_state() == before


def test_memory_store_validation():
    m = MemoryEcology(4, 2, seed=16)
    with pytest.raises(ValueError):
        m.store(np.ones(3), [1, 1], provenance='sensory')
    with pytest.raises(ValueError):
        m.store(np.ones(4), [np.inf, 1], provenance='sensory')
    with pytest.raises(ValueError):
        m.store(np.ones(4), [1, 1], provenance='sensory', confidence=2.0)
    with pytest.raises(ValueError):
        m.store(np.ones(4), [1, 1], provenance='sensory', region='')
    i = m.store(np.ones(4), [1, 1], provenance='sensory', reward=50.0)
    assert m.get(i)['reward'] == 1.0                      # reward clipped to [-1, 1]
    assert len(m) == 1 and m.stats()['raw_media_stored'] is False


# ============================================================================ adversarial-review regressions
def test_memory_default_surprise_uses_all_three_timescales():
    m = MemoryEcology(4, 2, seed=20)
    for _ in range(10):
        m.store(np.ones(4), [0.0, 0.0], provenance='sensory')
    probe = np.array([1.0, 1.0])
    base = m.default_surprise(probe)
    for name in ('fast', 'mid', 'slow'):
        saved = m.traces[name].copy()
        m.traces[name] = probe.copy()                 # this timescale now "expects" the probe
        assert m.default_surprise(probe) < base, name
        m.traces[name] = saved
    sid = m.store(np.ones(4), probe, provenance='sensory')
    assert m.get(sid)['surprise'] == pytest.approx(base)
    assert MemoryEcology(4, 2, seed=20).default_surprise(probe) == 1.0


def test_memory_import_rejects_id_collision_with_next_id():
    m = MemoryEcology(4, 2, seed=21)
    m.store([1, 0, 0, 0], [1, 1], provenance='sensory')
    m.store([0, 1, 0, 0], [1, 1], provenance='sensory')
    target = MemoryEcology(4, 2, seed=21)
    before = target.export_state()
    for next_id, ids in ((0, (0, 1)), (1, (0, 1)), (2, (0, 2))):
        s = json.loads(json.dumps(m.export_state()))
        s['next_id'] = next_id
        for item, iid in zip(s['items'], ids):
            item['id'] = iid
        with pytest.raises(ValueError):
            target.import_state(s)
        assert target.export_state() == before


def test_memory_interference_keeps_confidence_and_score_bounded():
    with pytest.raises(ValueError):
        MemoryEcology(4, 2, seed=22, interference_similarity=-0.5)
    m = MemoryEcology(4, 2, seed=22, interference_similarity=0.5, interference_value_tolerance=0.0,
                      interference_penalty=1.0)
    a = m.store([1, 0.2, 0, 0], [5, 5], provenance='sensory')
    for i in range(30):
        m.store([1, 0.2 + 0.01 * i, 0, 0], [-5 + i * 0.1, 5], provenance='sensory')
    confs = [it['confidence'] for it in m.items.values()]
    assert all(0.0 <= c <= 1.0 for c in confs) and m.get(a)['confidence'] < 1.0
    assert 0.0 <= m.interference_score <= 1.0
    other = MemoryEcology(4, 2, seed=22, interference_similarity=0.5, interference_value_tolerance=0.0,
                          interference_penalty=1.0)
    other.import_state(_json_roundtrip(m.export_state()))
    assert other.export_state() == m.export_state()


def test_memory_share_is_a_copy_not_an_observation():
    m = MemoryEcology(4, 2, seed=23, interference_value_tolerance=0.0)
    a = m.store([1, 0, 0, 0], [3, 3], provenance='sensory')
    m.store([0, 1, 0, 0], [-3, -3], provenance='sensory')
    traces = {n: t.copy() for n, t in m.traces.items()}
    count, events = m.trace_count, m.counters['interference_events']
    sid = m.share(a, 'episodic_memory')
    for n in traces:
        np.testing.assert_array_equal(m.traces[n], traces[n])
    assert m.trace_count == count and m.counters['interference_events'] == events
    assert m.get(sid)['shared_from'] == a and m.counters['shares'] == 1
    # a real observation does update the traces
    m.store([0, 0, 1, 0], [9, 9], provenance='sensory')
    assert not np.array_equal(m.traces['fast'], traces['fast'])


def test_memory_vectorised_scores_match_formula():
    rng = np.random.default_rng(24)
    m = MemoryEcology(8, 3, seed=24, context_dim=4, context_weight=0.4)
    for i in range(40):
        m.store(rng.normal(size=8), rng.normal(size=3), provenance=list(Provenance)[i % 5],
                reward=float(rng.uniform(-1, 1)), confidence=float(rng.uniform(0.1, 1)),
                context=rng.normal(size=4) if i % 2 else None)
    q, c = rng.normal(size=8), rng.normal(size=4)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    expected = []
    for iid, it in m.items.items():
        sim = cos(q, it['key'])
        if it['context'] is not None:
            sim = 0.6 * sim + 0.4 * cos(c, it['context'])
        expected.append((sim * (0.5 + 0.5 * it['confidence']) + 0.3 * it['reward'], iid))
    expected.sort(key=lambda s: (-s[0], s[1]))
    got = m.retrieve(q, context=c, k=10, reward_bias=0.3, exhaustive=True, touch=False)
    assert [r['id'] for r in got] == [iid for _, iid in expected[:10]]
    np.testing.assert_allclose([r['score'] for r in got], [s for s, _ in expected[:10]], atol=1e-12)


def test_memory_capacity_512_eviction_scales():
    import time
    rng = np.random.default_rng(25)
    m = MemoryEcology(32, 16, capacity=512, seed=25)
    t0 = time.perf_counter()
    for i in range(800):
        m.store(rng.normal(size=32), rng.normal(size=16), provenance='sensory',
                reward=float(rng.uniform(-1, 1)))
    for _ in range(50):
        m.retrieve(rng.normal(size=32), k=8)
    elapsed = time.perf_counter() - t0
    assert len(m) == 512 and m.counters['evictions'] == 288
    assert elapsed < 20.0
    # the evicted victim was always the lowest retention score at the time: spot-check the next one
    ids = list(m.items)
    worst = min(ids, key=lambda i: (m.retention_score(i), i))
    m.store(rng.normal(size=32), rng.normal(size=16), provenance='sensory', reward=1.0, surprise=1.0)
    assert worst not in m


def test_rng_state_overflow_rejected_atomically():
    mc = MicroCollective(seed=1)
    before = mc.export_state()
    for bad_val in (-5, 2**200):
        s = json.loads(json.dumps(before))
        s['rng']['sample']['state']['state'] = bad_val
        with pytest.raises(ValueError):
            mc.import_state(s)
        assert mc.export_state() == before
    m = MemoryEcology(4, 2, seed=1)
    mb = m.export_state()
    s = json.loads(json.dumps(mb)); s['rng']['state']['inc'] = 2**300
    with pytest.raises(ValueError):
        m.import_state(s)
    assert m.export_state() == mb


def test_graft_import_of_detached_state_unlinks_host():
    host = MicroCollective(seed=3)
    plain = MicroCollective(seed=3)
    g = SpecialistGraft(16, 16, seed=5)
    host.attach_specialist(g)
    dead = SpecialistGraft(16, 16, seed=5)
    dead.detach()
    g.import_state(_json_roundtrip(dead.export_state()))
    assert host.specialist is None and g._host is None and g.attached is False
    train(host, 12); train(plain, 12)            # host keeps working, identical to never-attached twin
    assert host.weights_digest() == plain.weights_digest()
