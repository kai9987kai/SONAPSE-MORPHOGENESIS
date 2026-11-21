"""Frontier module ``synchrony``: CTM-style synchronisation extension, live observer and the MiniCTM benchmark."""
import copy
import json
import math
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from synapse.frontier import synchrony as sy
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import MODULE_IDS, OBSERVER_MODULE_IDS, FrontierSpecV1
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.spec import NeuroWeaveSpecV1

HALF = 1.0 / math.log(2.0)          # timescale whose per-step decay exp(-1/tau) is 0.5


def _bound(n, pairs, **kw):
    ext = sy.SynchronyExtension(pairs, **kw)
    ext.bind(types.SimpleNamespace(n=n))
    return ext


def _run(engine, steps=12, seed=3):
    rng = np.random.default_rng(seed)
    outs = []
    for t in range(steps):
        res = engine.step(rng.uniform(-1, 1, 8), rng.uniform(-1, 1, 16), reward=0.5 if t % 3 else -0.4)
        outs.append(res['output'].copy())
    return np.stack(outs)


# ====================================================================== shared maths
def test_sample_pairs_self_pairs_first_then_distinct_cross_pairs():
    left, right = sy.sample_pairs(10, 20, 4, np.random.default_rng(1))
    assert left.shape == right.shape == (20,)
    assert np.array_equal(left[:4], right[:4]) and len(set(left[:4].tolist())) == 4      # self pairs, distinct units
    cross = list(zip(left[4:].tolist(), right[4:].tolist()))
    assert all(i < j for i, j in cross) and len(set(cross)) == 16
    again = sy.sample_pairs(10, 20, 4, np.random.default_rng(1))
    assert np.array_equal(again[0], left) and np.array_equal(again[1], right)
    dense_l, dense_r = sy.sample_pairs(5, 12, 2, np.random.default_rng(2))               # all 10 cross pairs
    assert set(zip(dense_l[2:].tolist(), dense_r[2:].tolist())) == {(i, j) for i in range(5) for j in range(i + 1, 5)}
    over_l, over_r = sy.sample_pairs(3, 8, 1, np.random.default_rng(3))                  # more pairs than exist
    assert set(zip(over_l[1:].tolist(), over_r[1:].tolist())) == {(0, 1), (0, 2), (1, 2)}
    one = sy.sample_pairs(1, 3, 1, np.random.default_rng(4))
    assert one[0].tolist() == one[1].tolist() == [0, 0, 0]
    for bad in ((0, 4, 1), (5, 0, 0), (5, 4, 5), (5, True, 0)):
        with pytest.raises(ValueError):
            sy.sample_pairs(*bad, np.random.default_rng(0))


def test_timescales_are_log_uniform_and_observer_ladder():
    tau = sy.log_uniform_timescales(8, 1.0, 256.0, np.random.default_rng(0))
    logs = np.diff(np.log(np.sort(tau)))
    assert np.all((tau >= 1.0) & (tau <= 256.0)) and np.allclose(logs, logs[0])
    assert sy.observer_timescales(256) == (2.0, 8.0, 32.0, 128.0)
    assert sy.observer_timescales(16) == (2.0, 8.0) and sy.observer_timescales(4) == (2.0,)
    with pytest.raises(ValueError):
        sy.log_uniform_timescales(4, 10.0, 1.0, np.random.default_rng(0))


# ====================================================================== engine extension
def test_extension_recursion_matches_hand_computed_closed_form():
    ext = _bound(2, 2, seed=0, self_pairs=1, timescales=(HALF, HALF))
    left, right = ext.pair_indices
    assert np.allclose(ext.decays, 0.5) and (left[1], right[1]) == (0, 1) and left[0] == right[0]
    history = np.array([[1.0, 0.5], [-1.0, 0.5], [0.5, 1.0]])
    for t, z in enumerate(history):
        ext.after_step(z, {'learn': True})
        assert np.allclose(ext.synchronisation(), sy.decayed_synchrony(history[:t + 1], left, right, ext.decays),
                           rtol=1e-12, atol=0.0)
    # by hand, decay 0.5: alpha = 0.25*(1*0.5) + 0.5*(-1*0.5) + 1*(0.5*1) = 0.375, beta = 0.25 + 0.5 + 1 = 1.75
    s = ext.synchronisation()
    assert s[1] == pytest.approx(0.375 / math.sqrt(1.75), rel=1e-12)
    self_alpha = {0: 0.25 * 1.0 + 0.5 * 1.0 + 0.25, 1: 0.25 * 0.25 + 0.5 * 0.25 + 1.0}[int(left[0])]
    assert s[0] == pytest.approx(self_alpha / math.sqrt(1.75), rel=1e-12)
    assert np.array_equal(ext.features(), np.clip(s, -10, 10))
    unit = _bound(2, 2, seed=0, self_pairs=1, timescales=(HALF, HALF), feature_scale='unit')
    for z in history:
        unit.after_step(z)
    assert np.allclose(unit.features(), s * math.sqrt(0.5), rtol=1e-12)


def test_extension_recursion_matches_closed_form_on_random_history():
    ext = _bound(10, 12, seed=3, self_pairs=3, timescales=(1.0, 64.0))
    history = np.random.default_rng(0).uniform(-1, 1, (40, 10))
    for z in history:
        ext.after_step(z)
    left, right = ext.pair_indices
    assert np.max(np.abs(ext.synchronisation() - sy.decayed_synchrony(history, left, right, ext.decays))) < 1e-12
    taus = -1.0 / np.log(ext.decays)
    assert taus.min() >= 1.0 - 1e-9 and taus.max() <= 64.0 + 1e-9 and np.ptp(np.log(taus)) > 3.0


def test_extension_on_engine_is_bounded_deterministic_and_readout_only():
    spec = NeuroWeaveSpecV1(seed=5, neurons=24)
    plain = NeuroWeaveEngine(spec)
    feats, outs = [], []
    for _ in range(2):
        engine = NeuroWeaveEngine(spec)
        ext = sy.SynchronyExtension(16, seed=3)
        engine.attach_extension(ext)
        outs.append(_run(engine, steps=20))
        feats.append(ext.features().copy())
    assert np.array_equal(feats[0], feats[1]) and feats[0].shape == (16,) and np.all(np.isfinite(feats[0]))
    assert np.all(np.abs(feats[0]) <= 10.0) and np.any(feats[0] != 0.0)
    assert np.array_equal(outs[0], _run(plain, steps=20))            # never changes the brain's drive
    other = sy.SynchronyExtension(16, seed=4)
    other.bind(engine)
    assert not np.array_equal(other.pair_indices[0], ext.pair_indices[0])
    json.dumps(ext.telemetry())
    unit = _bound(24, 16, seed=3, feature_scale='unit')
    for _ in range(50):
        unit.after_step(np.full(24, 1e6))                            # clipped to [-1, 1] before use
    assert np.all(np.abs(unit.features()) <= 1.0 + 1e-12)
    # z = 1 everywhere: alpha = beta = sum_k d^k, so S = sqrt(beta) and S * sqrt(1 - d) = sqrt(1 - d^50)
    assert np.allclose(unit.features(), np.sqrt(1.0 - unit.decays ** 50), rtol=1e-9)


def test_extension_learn_flag_born_neurons_and_deepcopy():
    a, b = _bound(6, 8, seed=1), _bound(6, 8, seed=1)
    rng = np.random.default_rng(2)
    for _ in range(10):
        z = rng.uniform(-1, 1, 9)
        a.after_step(z, {'learn': True})
        b.after_step(z[:6], {'learn': False})                       # extra (later-born) units are ignored
    assert np.array_equal(a.features(), b.features())
    short = _bound(6, 8, seed=1)
    short.after_step(np.array([0.5, -0.5, 0.25, 1.0]))                 # missing units count as silent
    pad = _bound(6, 8, seed=1)
    pad.after_step(np.array([0.5, -0.5, 0.25, 1.0, 0.0, 0.0]))
    assert np.array_equal(short.features(), pad.features())
    twin = copy.deepcopy(a)
    twin.after_step(np.ones(6))
    assert not np.array_equal(twin.features(), a.features())
    a.reset()
    assert np.array_equal(a.features(), np.zeros(8)) and a.telemetry()['steps'] == 0


def test_extension_state_round_trip_and_validation():
    ext = _bound(12, 10, seed=2, self_pairs=2)
    rng = np.random.default_rng(5)
    for _ in range(15):
        ext.after_step(rng.uniform(-1, 1, 12))
    state = json.loads(json.dumps(ext.export_state()))
    fresh = _bound(12, 10, seed=2, self_pairs=2)
    fresh.import_state(state)
    assert np.array_equal(fresh.features(), ext.features())
    z = rng.uniform(-1, 1, 12)
    fresh.after_step(z)
    ext.after_step(z)
    assert np.array_equal(fresh.features(), ext.features())
    before = fresh.export_state()
    num_big = copy.deepcopy(state)
    num_big['num'][0] = state['den'][0] + 1.0
    bad_steps = {**state, 'steps': state['steps'] + 3}
    nan_num = copy.deepcopy(state)
    nan_num['num'][1] = float('nan')
    for bad in ({**state, 'id': 'other'}, {**state, 'seed': 3}, {**state, 'left': state['right'][::-1]}, num_big,
                bad_steps, nan_num, {**state, 'den': state['den'][:-1]}, 'not a mapping'):
        with pytest.raises(ValueError):
            fresh.import_state(bad)
        assert fresh.export_state() == before                           # validated before mutating
    with pytest.raises(ValueError):
        sy.SynchronyExtension(10, seed=2, self_pairs=2).import_state(state)   # not bound


@pytest.mark.parametrize('kwargs', [{'pairs': 0}, {'pairs': True}, {'self_pairs': 99}, {'seed': -1},
                                    {'timescales': (4.0, 2.0)}, {'timescales': (0.0, 2.0)}, {'timescales': 'ab'},
                                    {'feature_scale': 'raw'}])
def test_extension_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        sy.SynchronyExtension(**{'pairs': 8, **kwargs})


def test_extension_rejects_bad_activity():
    with pytest.raises(ValueError):
        sy.SynchronyExtension(4).after_step(np.zeros(4))              # not bound
    ext = _bound(4, 4)
    for bad in (np.array([0.0, np.nan, 0.0, 0.0]), np.zeros((2, 2))):
        with pytest.raises(ValueError):
            ext.after_step(bad)
    assert ext.telemetry()['steps'] == 0


# ====================================================================== live observer
def test_observer_identical_units_are_perfectly_synchronised():
    obs = sy.SynchronyObserver(6, pairs=18, self_pairs=3, history=64)
    for t in range(200):
        obs.observe(np.full(6, math.sin(0.3 * t)))
    sm = obs.summary()
    assert sm['cross_pairs'] == 15 and sm['observed_steps'] == 200 and sm['timescales'] == [2.0, 8.0, 32.0]
    assert sm['global_synchrony_index'] == pytest.approx(1.0, abs=1e-12)
    assert sm['synchrony_entropy'] == pytest.approx(1.0, abs=1e-12)
    assert all(row['mean_abs_corr'] == pytest.approx(1.0, abs=1e-12) for row in sm['per_timescale'])
    assert len(sm['top_pairs']) == 8 and all(p['corr'] == pytest.approx(1.0, abs=1e-12) for p in sm['top_pairs'])
    # CTM scaling: S = alpha / sqrt(beta) grows with the window for a steady product
    s = [row['mean_abs_S'] for row in sm['per_timescale']]
    assert s[0] < s[1] < s[2]


def test_observer_finds_planted_in_phase_and_anti_phase_pairs():
    n = 8
    obs = sy.SynchronyObserver(n, pairs=32, self_pairs=4, history=64, seed=3)
    rng = np.random.default_rng(11)
    for _ in range(400):
        x = rng.standard_normal(n)
        x[1] = x[0]
        x[2] = -x[0]
        obs.observe(x)
    sm = obs.summary()
    top = {(p['i'], p['j']): p['corr'] for p in sm['top_pairs'][:3]}
    assert set(top) == {(0, 1), (0, 2), (1, 2)}
    assert top[(0, 1)] == pytest.approx(1.0, abs=1e-12)
    assert top[(0, 2)] == pytest.approx(-1.0, abs=1e-12) and top[(1, 2)] == pytest.approx(-1.0, abs=1e-12)
    assert all(abs(p['corr']) < 0.6 for p in sm['top_pairs'][3:])
    assert 3 / 28 < sm['global_synchrony_index'] < 0.5 and 0.0 < sm['synchrony_entropy'] < 1.0


def test_observer_noise_synchrony_is_low_and_falls_with_timescale():
    obs = sy.SynchronyObserver(16, pairs=64, history=256, seed=1)
    rng = np.random.default_rng(4)
    for _ in range(1000):
        obs.observe(rng.standard_normal(16))
    sm = obs.summary()
    corr = [row['mean_abs_corr'] for row in sm['per_timescale']]
    assert all(a > b for a, b in zip(corr, corr[1:])) and corr[-1] < 0.15
    assert all(0.0 <= row['max_abs_corr'] <= 1.0 for row in sm['per_timescale'])
    assert len(json.dumps(sm)) < 16_000


def test_observer_is_read_only_validates_input_and_resets():
    obs = sy.SynchronyObserver(5, pairs=6)
    x = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
    keep = x.copy()
    obs.observe(x)
    assert np.array_equal(x, keep)
    for bad in (np.zeros(4), np.array([0.0, np.nan, 0.0, 0.0, 0.0]), np.zeros((5, 1)), np.array([np.inf] * 5)):
        with pytest.raises(ValueError):
            obs.observe(bad)
    assert obs.observed_steps == 1
    obs.reset()
    assert obs.observed_steps == 0 and obs.summary()['global_synchrony_index'] == 0.0
    for bad in ({'n_units': 0}, {'pairs': 0}, {'history': 2}, {'zclip': 0.5}, {'self_pairs': 9}):
        with pytest.raises(ValueError):
            sy.SynchronyObserver(**{'n_units': 5, 'pairs': 6, **bad})


def test_observer_state_round_trip_and_validation():
    rng = np.random.default_rng(8)
    obs = sy.SynchronyObserver(10, pairs=12, history=32, seed=4)
    for _ in range(50):
        obs.observe(rng.standard_normal(10))
    state = json.loads(json.dumps(obs.export_state()))
    twin = sy.SynchronyObserver(10, pairs=12, history=32, seed=4)
    twin.import_state(state)
    assert twin.summary() == obs.summary()
    for _ in range(5):
        x = rng.standard_normal(10)
        obs.observe(x)
        twin.observe(x)
    assert twin.summary() == obs.summary()
    before = twin.summary()
    cross = int(np.flatnonzero(obs.pair_indices[0] != obs.pair_indices[1])[0])
    over = copy.deepcopy(state)
    over['num'][0][cross] = 100.0                                    # breaks Cauchy-Schwarz |alpha| <= sqrt(q_i q_j)
    bad_var = copy.deepcopy(state)
    bad_var['var'][0] = -1.0
    bad_mean = copy.deepcopy(state)
    bad_mean['mean'][2] = float('nan')
    for bad in ({**state, 'module': 'other'}, {**state, 'seed': 5}, {**state, 'steps': 49}, over, bad_var, bad_mean,
                {**state, 'sq': state['sq'][:1]}, {**state, 'den': [x * 2 for x in state['den']]}, None):
        with pytest.raises(ValueError):
            twin.import_state(bad)
        assert twin.summary() == before


def test_observer_from_spec_determinism_and_registry():
    spec = FrontierSpecV1(enabled=True, modules=('synchrony',), seed=9, sync_pairs=16, history=64)
    obs = sy.SynchronyObserver.from_spec(32, spec)
    assert (obs.pairs, obs.history, obs.seed) == (16, 64, 9) and 'synchrony' in OBSERVER_MODULE_IDS
    same = sy.SynchronyObserver(32, seed=9, history=64, pairs=16)
    other = sy.SynchronyObserver(32, seed=10, history=64, pairs=16)
    assert np.array_equal(obs.pair_indices[0], same.pair_indices[0])
    assert not np.array_equal(obs.pair_indices[0], other.pair_indices[0])
    big = sy.SynchronyObserver(256, pairs=4096, history=8192)
    big.observe(np.random.default_rng(0).standard_normal(256))
    big.observe(np.random.default_rng(1).standard_normal(256))
    assert len(json.dumps(big.summary())) < 16_000
    assert sy.MODULE_ID in MODULE_IDS and list(sy.EXPERIMENTS) == ['synchrony', 'synchrony_ctm']


# ====================================================================== PyTorch pieces
def test_neuron_level_models_are_private_per_neuron():
    D, B, M, H, k = 6, 5, 4, 3, 2
    nlm = sy.NeuronLevelModels(D, M, H, seed=1)
    with torch.no_grad():
        nlm.b1.normal_(generator=torch.Generator().manual_seed(0))
    gen = torch.Generator().manual_seed(3)
    hist = torch.randn(D, B, M, generator=gen)

    def grads_of_neuron(history):
        h = history.clone().requires_grad_(True)
        nlm.zero_grad(set_to_none=True)
        out = nlm(h)
        out[k].sum().backward()
        return out.detach(), h.grad, nlm.w1.grad.clone(), nlm.w2.grad.clone(), nlm.b1.grad.clone()

    out, g_hist, g_w1, g_w2, g_b1 = grads_of_neuron(hist)
    assert out.shape == (D, B)
    others = [j for j in range(D) if j != k]
    assert torch.all(g_hist[others] == 0) and torch.any(g_hist[k] != 0)
    for g in (g_w1, g_w2, g_b1):
        assert torch.all(g[others] == 0) and torch.any(g[k] != 0)
    moved = hist.clone()
    moved[others] += torch.randn(len(others), B, M, generator=gen)  # change every other neuron's history
    out2, _, g_w1b, g_w2b, _ = grads_of_neuron(moved)
    assert torch.equal(out2[k], out[k]) and not torch.equal(out2[others], out[others])
    assert torch.equal(g_w1b, g_w1) and torch.equal(g_w2b, g_w2)   # neuron k's gradient sees only its history
    with torch.no_grad():                                               # same contraction as the reference einsum
        ref = torch.einsum('BDM,MHD->BDH', hist.permute(1, 0, 2), nlm.w1.permute(1, 2, 0)) + nlm.b1.permute(1, 0, 2)
        mine = torch.einsum('dbm,dmh->dbh', hist, nlm.w1) + nlm.b1
        assert torch.allclose(ref, mine.permute(1, 0, 2), atol=1e-6)


def _small_ctm(seed=0, **kw):
    return sy.MiniCTM(6, ticks=4, d_model=10, memory=3, nlm_hidden=2, d_input=8, n_sync_out=8, n_sync_action=6,
                      self_pairs=2, seed=seed, **kw)


def test_minictm_shapes_ticks_seeding_and_validation():
    state = torch.random.get_rng_state().clone()
    model, same, other = _small_ctm(0), _small_ctm(0), _small_ctm(1)
    assert torch.equal(state, torch.random.get_rng_state())             # global RNG untouched
    sd, sd_same = model.state_dict(), same.state_dict()
    assert all(torch.equal(sd[key], sd_same[key]) for key in sd)
    assert any(not torch.equal(sd[key], other.state_dict()[key]) for key in sd)
    tokens = torch.randint(0, 2, (5, 6), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, cert = model(tokens)
        one, _ = model(tokens, ticks=1)
    assert logits.shape == (5, 4, 6, 2) and cert.shape == (5, 4)
    assert torch.all((cert >= 0) & (cert <= 1))
    assert torch.allclose(one[:, 0], logits[:, 0], atol=1e-6)            # tick 1 does not depend on later ticks
    assert not torch.allclose(logits[:, 0], logits[:, -1], atol=1e-4)    # internal ticks change the output
    assert sy._trainable(model) == sy._ctm_params({'seq_len': 6, 'd_model': 10, 'memory': 3, 'nlm_hidden': 2,
                                                   'd_input': 8, 'n_sync_out': 8, 'n_sync_action': 6})
    for bad in (tokens.float(), tokens[:, :5], tokens + 1, torch.zeros(0, 6, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(bad)
    with pytest.raises(ValueError):
        model(tokens, ticks=0)
    for bad in ({'self_pairs': 6}, {'decay_init': (1.0, 0.1)}, {'decay_init': 3}, {'seq_len': 0}, {'seed': -1}):
        with pytest.raises(ValueError):
            sy.MiniCTM(**{'seq_len': 6, 'n_sync_action': 6, 'n_sync_out': 8, **bad})


def test_models_initialise_from_private_generators_only():
    def build():
        return [_small_ctm(3), sy.AttentiveGRU(6, hidden=7, ticks=4, d_input=8, seed=3), sy.ParityMLP(6, hidden=9, seed=3)]

    states = []
    for global_seed in (11, 12345):
        with torch.random.fork_rng(devices=[]):                        # a different global RNG state each time
            torch.manual_seed(global_seed)
            before = torch.random.get_rng_state().clone()
            models = build()
            assert torch.equal(before, torch.random.get_rng_state())    # construction draws nothing globally
        states.append([m.state_dict() for m in models])
    for a, b in zip(*states):
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    ctm, gru, mlp = build()
    linear = sy._define_torch_classes()['_Linear']
    for model in (ctm, gru, mlp):
        assert all(torch.all(torch.isfinite(p)) for p in model.parameters())
    with torch.no_grad():                                               # nn.Linear's default U(+-1/sqrt(fan_in))
        for lin in [m for model in (ctm, gru, mlp) for m in model.modules() if isinstance(m, linear)]:
            bound = 1.0 / math.sqrt(lin.in_features) + 1e-6
            assert float(lin.weight.abs().max()) <= bound and float(lin.bias.abs().max()) <= bound
        assert all(float(p.abs().max()) <= 1.0 / math.sqrt(7) + 1e-6 for p in gru.cell.parameters())
    assert torch.equal(ctm.synapse[2].weight, torch.ones(10)) and torch.equal(ctm.reader.norm.bias, torch.zeros(8))


def test_building_blocks_match_their_torch_nn_counterparts():
    blocks = sy._define_torch_classes()
    g = torch.Generator().manual_seed(0)
    mine = {'linear': blocks['_Linear'](5, 3, g), 'norm': blocks['_LayerNorm'](3), 'cell': blocks['_GRUCell'](4, 3, g)}
    with torch.random.fork_rng(devices=[]):                             # torch.nn constructors draw from the global RNG
        ref = {'linear': torch.nn.Linear(5, 3), 'norm': torch.nn.LayerNorm(3), 'cell': torch.nn.GRUCell(4, 3)}
    for key, module in mine.items():
        ref[key].load_state_dict(module.state_dict())                  # same parameter names and shapes
    gen = torch.Generator().manual_seed(1)
    x5, x3, x4, h = (torch.randn(6, k, generator=gen) for k in (5, 3, 4, 3))
    with torch.no_grad():
        assert torch.allclose(mine['linear'](x5), ref['linear'](x5), atol=1e-6)
        assert torch.allclose(mine['norm'](x3), ref['norm'](x3), atol=1e-6)
        assert torch.allclose(mine['cell'](x4, h), ref['cell'](x4, h), atol=1e-6)


def test_reader_fused_attention_equals_explicit_softmax_attention():
    reader = sy.SequenceReader(6, 2, 8, seed=1)
    kv = reader.keys(torch.randint(0, 2, (5, 6), generator=torch.Generator().manual_seed(2)))
    q = torch.randn(5, 8, generator=torch.Generator().manual_seed(3), requires_grad=True)
    o, w = reader.attend(q, kv, weights=True)
    w_ref = torch.softmax((kv @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(8), dim=-1)
    o_ref = (w_ref.unsqueeze(-1) * kv).sum(1)
    assert torch.allclose(o, o_ref, atol=1e-6) and torch.allclose(w, w_ref, atol=1e-6)
    assert torch.allclose(w.sum(-1), torch.ones(5)) and reader.attend(q, kv)[1] is None
    g, g_ref = (torch.autograd.grad(x.square().sum(), q, retain_graph=True)[0] for x in (o, o_ref))
    assert torch.allclose(g, g_ref, atol=1e-6)


def test_minictm_synchronisation_equals_closed_form():
    model = _small_ctm(2)
    tokens = torch.randint(0, 2, (3, 6), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        logits, _, rec = model(tokens, track=True)
        decay = torch.exp(-torch.nn.functional.softplus(model.rho)).double().numpy()
        z0 = model.start_z.detach().expand(3, -1).unsqueeze(1)
        Z = torch.cat([z0, rec['post_activations']], 1).double().numpy()   # start state counts as tick 0
        assert torch.allclose(logits, model.output(rec['sync_out']).view(3, 4, 6, 2))
    left, right = model.pair_left.numpy(), model.pair_right.numpy()
    pa = model.n_sync_action
    for b in range(3):
        for t in range(4):
            out = sy.decayed_synchrony(Z[b, :t + 2], left[pa:], right[pa:], decay[pa:])
            act = sy.decayed_synchrony(Z[b, :t + 1], left[:pa], right[:pa], decay[:pa])
            assert np.allclose(rec['sync_out'][b, t].double().numpy(), out, rtol=1e-4, atol=1e-7)
            assert np.allclose(rec['sync_action'][b, t].double().numpy(), act, rtol=1e-4, atol=1e-7)
    r_act, r_out = model.decay_rates()
    assert r_act.shape == (6,) and r_out.shape == (8,) and torch.all(r_act > 0) and torch.all(r_out > 0)
    assert np.ptp(np.log(torch.cat([r_act, r_out]).detach().numpy())) > 2.0    # several timescales at init


def test_ctm_tick_loss_selects_min_loss_and_most_certain_ticks():
    logits = torch.tensor([[[[0.0, 0.0]], [[2.0, 0.0]], [[-5.0, 5.0]]]])     # (B=1, T=3, L=1, C=2), target 0
    target = torch.tensor([[0]])
    loss, info = sy.ctm_tick_loss(logits, target)
    ce = [math.log(2.0), math.log1p(math.exp(-2.0)), math.log1p(math.exp(10.0))]
    assert info['per_tick_loss'][0].tolist() == pytest.approx(ce, rel=1e-5)
    assert info['min_loss_tick'].tolist() == [1] and info['most_certain_tick'].tolist() == [2]
    assert float(loss) == pytest.approx(0.5 * (ce[1] + ce[2]), rel=1e-5)
    single = torch.randn(4, 1, 5, 2, generator=torch.Generator().manual_seed(0))
    tgt = torch.randint(0, 2, (4, 5), generator=torch.Generator().manual_seed(1))
    plain = torch.nn.functional.cross_entropy(single.reshape(-1, 2), tgt.reshape(-1))
    assert float(sy.ctm_tick_loss(single, tgt)[0]) == pytest.approx(float(plain), rel=1e-6)
    with pytest.raises(ValueError):
        sy.ctm_tick_loss(single, tgt[:, :4])


def test_cumulative_parity_data_known_answer_and_split():
    data = sy.cumulative_parity_data(4, 0, 5)
    x = np.concatenate([data['heldout_x'], data['train_x']])
    y = np.concatenate([data['heldout_y'], data['train_y']])
    assert data['heldout_x'].shape == (5, 4) and data['train_x'].shape == (11, 4)
    assert {tuple(r) for r in x.tolist()} == {tuple((i >> np.arange(4)) & 1) for i in range(16)}   # each once
    assert np.array_equal(y, np.cumsum(x, axis=1) % 2)
    row = x.tolist().index([1, 1, 0, 1])
    assert y[row].tolist() == [1, 0, 0, 1]
    again, other = sy.cumulative_parity_data(4, 0, 5), sy.cumulative_parity_data(4, 1, 5)
    assert np.array_equal(again['heldout_x'], data['heldout_x'])
    assert not np.array_equal(other['heldout_x'], data['heldout_x'])
    for bad in ((1, 0, 1), (4, 0, 16), (4, -1, 2)):
        with pytest.raises(ValueError):
            sy.cumulative_parity_data(*bad)


@pytest.mark.parametrize('quick', [True, False])
def test_parameter_counts_are_analytic_and_matched(quick):
    cfg = sy._ctm_config(quick, {})
    L, d = cfg['seq_len'], cfg['d_input']
    ctm = sy.MiniCTM(L, **{k: cfg[k] for k in ('ticks', 'd_model', 'memory', 'nlm_hidden', 'd_input', 'n_sync_out',
                                               'n_sync_action', 'self_pairs')})
    target = sy._ctm_params(cfg)
    assert sy._trainable(ctm) == target
    h_gru = sy._closest_width(lambda h: sy._gru_params(L, d, h), target, 4096)
    h_mlp = sy._closest_width(lambda h: sy._mlp_params(L, h), target, 1_000_000)
    gru, mlp = sy.AttentiveGRU(L, hidden=h_gru, d_input=d), sy.ParityMLP(L, hidden=h_mlp)
    assert sy._trainable(gru) == sy._gru_params(L, d, h_gru) and sy._trainable(mlp) == sy._mlp_params(L, h_mlp)
    assert abs(sy._trainable(gru) - target) / target <= 0.10 and abs(sy._trainable(mlp) - target) / target <= 0.10


# ====================================================================== experiments and receipts
TINY = dict(steps=3, heldout=16, train_eval=16, batch=8)


def test_ctm_benchmark_is_deterministic_and_leaves_torch_state_alone():
    threads, rng_state = torch.get_num_threads(), torch.random.get_rng_state().clone()
    first = sy.run_ctm_benchmark(quick=True, seeds=(0,), **TINY)
    second = sy.run_ctm_benchmark(quick=True, seeds=(0,), **TINY)
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert torch.get_num_threads() == threads and torch.equal(rng_state, torch.random.get_rng_state())
    for bad in ({'quick': 1}, {'seeds': ()}, {'seeds': (0, 0)}, {'quick': True, 'nonsense': 1},
                {'quick': True, 'heldout': 256}, {'quick': True, 'self_pairs': 12}):
        with pytest.raises(ValueError):
            sy.run_ctm_benchmark(**bad)


def test_ctm_benchmark_quick_receipt_is_sealed_and_honest():
    r = sy.EXPERIMENTS['synchrony_ctm'](quick=True, seeds=(0, 1))
    assert verify_receipt(r) and r['experiment_id'] == 'synchrony_ctm' and r['verdict'] in VERDICTS
    assert r['kind'] == 'standalone_model_comparison' and r['evidence_tier'] == 'synthetic_abstraction'
    budget = r['parameter_budget']
    assert budget['matched_within_10pct'] and budget['max_relative_gap'] <= 0.10
    assert set(r['models']) == {'mini_ctm', 'gru', 'mlp'}
    for model in r['models'].values():
        assert len(model['per_seed']) == 2 and len(model['summary']['heldout_per_position_accuracy']) == 8
        assert all(0.0 <= s['heldout']['accuracy'] <= 1.0 for s in model['per_seed'])
        assert model['summary']['heldout_per_position_accuracy'][0] > 0.9   # position 0 is the copied bit
    best = r['comparisons']['mini_ctm_vs_best_baseline']
    assert best['best_baseline'] in ('gru', 'mlp') and r['verdict'] == best['verdict'] and best['n'] == 2
    assert 'estimated_cost' in r and r['measured_wallclock_s']['total'] > 0
    tampered = copy.deepcopy(r)
    tampered['models']['mini_ctm']['summary']['heldout_accuracy'] = 1.0
    assert not verify_receipt(tampered)


def test_run_experiment_quick_receipt_is_sealed_deterministic_and_controlled():
    first = sy.run_experiment(quick=True, seeds=(0, 1))
    second = sy.EXPERIMENTS['synchrony'](quick=True, seeds=(0, 1))
    assert first['digest'] == second['digest'] and verify_receipt(first) and first['verdict'] in VERDICTS
    assert set(first['arms']) == {'baseline', 'candidate', 'control'}
    assert first['config']['task'] == 'rhythm' and first['config']['control'] == 'random_trace'
    assert first['config']['candidate_feature_dim'] == 16 and first['config']['extension']['pairs'] == 16
    cand = first['arms']['candidate']['per_seed'][0]
    assert cand['extensions'] == ['synchrony'] and cand['budget']['extension_parameters'] == 0
    assert first['arms']['control']['per_seed'][0]['budget']['extension_features'] == 16
    for bad in ({'quick': 'yes'}, {'seeds': ()}, {'quick': True, 'nonsense': 1}, {'quick': True, 'pairs': 0}):
        with pytest.raises(ValueError):
            sy.run_experiment(**bad)


def test_importing_the_module_does_not_import_torch():
    code = 'import sys, synapse.frontier.synchrony as s; s.SynchronyObserver(4); print("torch" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == 'False'
