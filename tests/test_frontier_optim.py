"""Frontier pack, optimisation group: Muon-style optimiser (muon.py) and RigL dynamic sparse training (rigl.py)."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from synapse.frontier import muon, rigl
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.muon import (NS_COEFFICIENTS, MuonWithAuxAdam, graft_input_layers, graft_output_layers,
                                   make_graft_optimizer, zeropower_via_newtonschulz5)
from synapse.frontier.rigl import RigLScheduler, cosine_drop_fraction, layer_active_counts, layer_densities
from synapse.frontier.spec import EVIDENCE_TIER, LAB_SCHEMA, MODULE_IDS
from synapse.neural_graft import _ResidualNetwork


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _p5(x, steps=5):
    a, b, c = NS_COEFFICIENTS
    for _ in range(steps):
        x = a * x + b * x ** 3 + c * x ** 5
    return x


def _ref_ns(g, steps=5):
    """Independent re-implementation of the quintic Newton-Schulz iteration (float64)."""
    a, b, c = NS_COEFFICIENTS
    x = g.double()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (torch.linalg.matrix_norm(x) + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


def _graft_net(hidden=32, seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _ResidualNetwork(hidden)


def _graft_batch(seed=0, n=16):
    g = _gen(seed)
    state = torch.randn(n, 256, generator=g)
    return state, state[:, :155], torch.randn(n, 160, generator=g)


def _graft_loss(net, batch):
    state, obs, dyn = batch
    return net.policy(state).square().mean() + (net.value(obs) - 1.0).square().mean() + net.dynamics(dyn).square().mean()


# ====================================================================== Newton-Schulz
@pytest.mark.parametrize('shape', [(16, 64), (64, 16), (32, 96), (8, 256), (5, 32), (48, 96)])
def test_newton_schulz_singular_values_lie_in_the_quintic_band(shape):
    g = torch.randn(*shape, generator=_gen(sum(shape)))
    before = g.clone()
    out = zeropower_via_newtonschulz5(g)
    assert torch.equal(g, before) and out.shape == g.shape and out.dtype == g.dtype
    s_in = torch.linalg.svdvals(g.double())
    assert float(s_in.min() / torch.linalg.vector_norm(s_in)) >= 0.01          # precondition of the band
    s = torch.linalg.svdvals(out.double())
    assert float(s.min()) >= 0.68 and float(s.max()) <= 1.135                  # band [0.6818, 1.1344] for x0 >= 0.01
    assert torch.allclose(out.double(), _ref_ns(g).to(torch.float64), atol=2e-5)


def test_newton_schulz_is_the_scalar_quintic_on_a_known_svd():
    q1, _ = torch.linalg.qr(torch.randn(6, 6, generator=_gen(1), dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(9, 9, generator=_gen(2), dtype=torch.float64))
    s = torch.tensor([3.0, 1.0, 0.5, 0.2, 0.05, 0.001], dtype=torch.float64)
    g = q1 @ torch.diag(s) @ q2[:6]
    out = zeropower_via_newtonschulz5(g)
    expected = q1 @ torch.diag(_p5(s / (torch.linalg.vector_norm(s) + 1e-7))) @ q2[:6]
    assert out.dtype == torch.float64 and torch.allclose(out, expected, atol=1e-12)
    # tall input = transpose of the wide computation; positive rescaling is removed by the normalisation
    assert torch.equal(zeropower_via_newtonschulz5(g.T), out.T)
    assert torch.allclose(zeropower_via_newtonschulz5(g * 7.5), out, atol=1e-9)
    sq = torch.randn(24, 24, generator=_gen(3))          # square: tiny singular values leave the band ...
    sv = torch.linalg.svdvals(zeropower_via_newtonschulz5(sq).double())
    assert float(sv.min()) > 0.0 and float(sv.max()) <= 1.2025                 # ... but stay in (0, 1.2024]


def test_newton_schulz_handles_entries_whose_squares_overflow():
    g = torch.randn(5, 7, generator=_gen(4))
    huge = g * 1e30                                      # finite in float32, but its Frobenius norm overflows
    assert math.isinf(float(torch.linalg.matrix_norm(huge)))
    assert torch.allclose(zeropower_via_newtonschulz5(huge), zeropower_via_newtonschulz5(g), atol=1e-5)


@pytest.mark.parametrize('bad', [dict(matrix=torch.ones(3)), dict(matrix=torch.ones(0, 3)),
                                 dict(matrix=torch.ones(2, 2), steps=-1), dict(matrix=torch.ones(2, 2), steps=1.0),
                                 dict(matrix=torch.ones(2, 2), eps=0.0), dict(matrix=[[1.0]]),
                                 dict(matrix=torch.ones(2, 2, dtype=torch.int64)),
                                 dict(matrix=torch.tensor([[1.0, float('nan')], [0.0, 1.0]])),
                                 dict(matrix=torch.tensor([[1.0, float('inf')], [0.0, 1.0]]))])
def test_newton_schulz_validates_input(bad):
    with pytest.raises(ValueError):
        zeropower_via_newtonschulz5(**bad)


# ====================================================================== MuonWithAuxAdam
def test_vector_parameters_follow_adamw_exactly():
    def run(make_opt):
        g = _gen(4)
        ps = [torch.randn(7, generator=g).requires_grad_(), torch.randn(3, generator=g).requires_grad_()]
        opt = make_opt(ps)
        traj = []
        for _ in range(6):
            loss = sum(((p ** 2).sum() * (i + 1) + p.sin().sum()) for i, p in enumerate(ps))
            opt.zero_grad()
            loss.backward()
            opt.step()
            traj.append([p.detach().clone() for p in ps])
        return traj, opt
    mine, opt = run(lambda ps: MuonWithAuxAdam(ps, lr=0.01, weight_decay=0.1))
    ref, _ = run(lambda ps: torch.optim.AdamW(ps, lr=0.01, weight_decay=0.1))
    assert opt.muon_parameters() == [] and len(opt.adam_parameters()) == 2
    assert all(torch.equal(a, b) for sa, sb in zip(mine, ref) for a, b in zip(sa, sb))


@pytest.mark.parametrize('nesterov', [True, False])
def test_matrix_update_matches_liu_et_al_equation_4(nesterov):
    w = torch.randn(6, 4, generator=_gen(5), dtype=torch.float64, requires_grad=True)
    lr, mu, wd = 0.02, 0.9, 0.1
    opt = MuonWithAuxAdam([w], lr=lr, momentum=mu, nesterov=nesterov, weight_decay=wd)
    ref_w, buf = w.detach().clone(), torch.zeros(6, 4, dtype=torch.float64)
    for k in range(3):
        grad = torch.randn(6, 4, generator=_gen(10 + k), dtype=torch.float64)
        w.grad = grad.clone()
        opt.step()
        buf = mu * buf + (1 - mu) * grad
        u = (1 - mu) * grad + mu * buf if nesterov else buf
        ref_w = ref_w - lr * (0.2 * math.sqrt(6) * _ref_ns(u) + wd * ref_w)
        assert torch.allclose(w.detach(), ref_w, atol=1e-12)
    assert opt.state_numbers() == 24                                           # one momentum buffer


def test_non_finite_gradients_and_updates_are_refused_without_any_change():
    w = torch.randn(4, 3, generator=_gen(6), requires_grad=True)
    b = torch.randn(3, generator=_gen(7), requires_grad=True)
    opt = MuonWithAuxAdam([w, b], lr=0.01)
    w.grad, b.grad = torch.ones(4, 3), torch.ones(3)
    opt.step()
    snap_w, snap_b, snap_state = w.detach().clone(), b.detach().clone(), copy.deepcopy(opt.state_dict())
    for bad_w, bad_b in ((torch.ones(4, 3), torch.tensor([1.0, float('nan'), 0.0])),     # non-finite gradient
                         (torch.ones(4, 3), torch.tensor([1.0, float('inf'), 0.0])),
                         (torch.ones(4, 3), torch.full((3,), 1e30))):                   # AdamW's grad**2 overflows
        w.grad, b.grad = bad_w, bad_b
        with pytest.raises(ValueError):
            opt.step()
        assert torch.equal(w, snap_w) and torch.equal(b, snap_b)            # the Muon part was not applied either
        assert str(opt.state_dict()) == str(snap_state)


def test_muon_update_is_invariant_to_gradient_scale_even_when_the_norm_overflows():
    def first_step(scale):
        w = torch.randn(4, 3, generator=_gen(6)).requires_grad_()
        opt = MuonWithAuxAdam([w], lr=0.01)
        w.grad = torch.randn(4, 3, generator=_gen(17)) * scale
        opt.step()
        return w.detach()
    base = first_step(1.0)            # (scale-free up to the 1e-7 eps added to the Frobenius norm)
    assert torch.allclose(first_step(10.0), base, atol=1e-6)
    assert torch.allclose(first_step(1e25), base, atol=1e-6)                 # Frobenius norm overflows float32


def test_add_param_group_validates_and_routes():
    w = torch.randn(4, 3, generator=_gen(18), requires_grad=True)
    opt = MuonWithAuxAdam([w], lr=0.01)
    v = torch.randn(5, generator=_gen(19), requires_grad=True)
    m = torch.randn(2, 2, generator=_gen(20), requires_grad=True)
    opt.add_param_group({'params': [v], 'lr': 0.05})
    opt.add_param_group({'params': m})
    assert [g['use_muon'] for g in opt.param_groups] == [True, False, True]
    assert opt.param_groups[1]['lr'] == 0.05 and opt.param_groups[2]['lr'] == 0.01
    n = len(opt.param_groups)
    for bad in ({'params': [torch.zeros(3, 3, requires_grad=True), torch.zeros(3, requires_grad=True)]},
                {'params': [torch.zeros(3, requires_grad=True)], 'use_muon': True},
                {'params': [torch.zeros(3, 3, requires_grad=True)], 'use_muon': 1},
                {'params': [torch.zeros(3, 3, requires_grad=True)], 'lr': -1.0},
                {'params': [torch.zeros(3, 3, requires_grad=True)], 'surprise': 1},
                {'params': [w]},                                             # already in a group
                [torch.zeros(3, 3, requires_grad=True)]):
        with pytest.raises(ValueError):
            opt.add_param_group(bad)
    assert len(opt.param_groups) == n
    for p in (w, v, m):
        p.grad = torch.ones_like(p)
    opt.step()
    assert set(opt.state[v]) == {'step', 'exp_avg', 'exp_avg_sq'} and set(opt.state[m]) == {'momentum_buffer'}


def test_state_dict_round_trip_continues_identically_and_bad_state_is_rejected():
    def make():
        net = nn.Sequential(nn.Linear(6, 8), nn.LayerNorm(8), nn.Tanh(), nn.Linear(8, 3))
        with torch.no_grad():
            g = _gen(8)
            for p in net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
        return net
    x, y = torch.randn(16, 6, generator=_gen(9)), torch.randn(16, 3, generator=_gen(10))

    def train(net, opt, n):
        for _ in range(n):
            loss = F.mse_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    a = make()
    opt_a = MuonWithAuxAdam(a, lr=0.01, weight_decay=0.05)
    train(a, opt_a, 3)
    saved = copy.deepcopy(opt_a.state_dict())
    b = make()
    b.load_state_dict(a.state_dict())
    opt_b = MuonWithAuxAdam(b, lr=0.01, weight_decay=0.05)
    opt_b.load_state_dict(saved)
    saved['state'][0]['momentum_buffer'].fill_(123.0)                         # loaded state is a private copy
    train(a, opt_a, 4)
    train(b, opt_b, 4)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    fresh = MuonWithAuxAdam(make(), lr=0.01)
    good = copy.deepcopy(opt_a.state_dict())
    for mutate in (lambda s: s['state'][0].__setitem__('momentum_buffer', torch.full((8, 6), float('nan'))),
                   lambda s: s['state'][0].__setitem__('momentum_buffer', torch.zeros(3, 3)),
                   lambda s: s['param_groups'][0].__setitem__('use_muon', False),
                   lambda s: s['param_groups'][0].__setitem__('lr', -1.0),
                   lambda s: s['param_groups'].pop()):
        bad = copy.deepcopy(good)
        mutate(bad)
        before = str(fresh.state_dict())
        with pytest.raises(ValueError):
            fresh.load_state_dict(bad)
        assert str(fresh.state_dict()) == before


def test_constructor_validation():
    w = torch.zeros(3, 3, requires_grad=True)
    for kwargs in ({'lr': -1.0}, {'momentum': 1.0}, {'nesterov': 1}, {'ns_steps': 2.0}, {'betas': (0.9,)},
                   {'eps': 0.0}, {'update_scale': 0.0}, {'weight_decay': float('nan')}):
        with pytest.raises(ValueError):
            MuonWithAuxAdam([w], **kwargs)
    for params in ([{'params': [torch.zeros(3, requires_grad=True)], 'use_muon': True}],
                   [{'params': [w], 'surprise': 1}], [{'params': [w], 'momentum': 2.0}], [], w, 3,
                   {w}, [w, 'x'], [{'lr': 0.1}]):
        with pytest.raises(ValueError):
            MuonWithAuxAdam(params)


# ====================================================================== graft helper
def test_make_graft_optimizer_adam_is_exactly_torch_adam():
    net = _graft_net()
    twin = copy.deepcopy(net)
    mine = make_graft_optimizer(net, 'adam', 3e-4)
    ref = torch.optim.Adam(twin.parameters(), lr=3e-4)
    assert type(mine) is torch.optim.Adam and mine.defaults == ref.defaults
    for k in range(4):
        batch = _graft_batch(k)
        for model, opt in ((net, mine), (twin, ref)):
            loss = _graft_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert all(torch.equal(p, q) for p, q in zip(net.parameters(), twin.parameters()))
    assert str(mine.state_dict()) == str(ref.state_dict())
    params = list(_graft_net().parameters())
    assert type(make_graft_optimizer(params, 'adam', 1e-3)) is torch.optim.Adam


def test_make_graft_optimizer_muon_routes_heads_and_vectors_to_adamw():
    net = _graft_net()
    names = {id(p): n for n, p in net.named_parameters()}
    opt = make_graft_optimizer(net, 'muon', 1e-3)
    assert isinstance(opt, MuonWithAuxAdam)
    heads = {id(p) for m in graft_output_layers(net) for p in m.parameters()}
    assert graft_output_layers(net) == (net.actor_head, net.value[-1], net.dynamics[-1])
    inputs = {id(p) for m in graft_input_layers(net) for p in m.parameters()}
    muon_names = sorted(names[id(p)] for p in opt.muon_parameters())
    # default (the live opt-in): input and output layers on AdamW (Jordan et al.), hidden-to-hidden on Muon
    assert muon_names == ['actor_encoder.3.weight', 'dynamics.2.weight', 'value.2.weight']
    assert muon_names == sorted(n for n, p in net.named_parameters()
                                if p.ndim == 2 and id(p) not in heads and id(p) not in inputs)
    assert all(p.ndim != 2 or id(p) in heads or id(p) in inputs for p in opt.adam_parameters())
    explicit = make_graft_optimizer(net, 'muon', 1e-3, inputs_on_adam=True)
    assert [id(p) for p in explicit.muon_parameters()] == [id(p) for p in opt.muon_parameters()]
    with_inputs = make_graft_optimizer(net, 'muon', 1e-3, inputs_on_adam=False)       # the routing ablation
    with_names = sorted(names[id(p)] for p in with_inputs.muon_parameters())
    assert with_names == sorted(n for n, p in net.named_parameters() if p.ndim == 2 and id(p) not in heads)
    assert len(with_names) == 6 and 'actor_encoder.1.weight' not in with_names                  # LayerNorm gain
    every = make_graft_optimizer(net, 'muon', 1e-3, heads_on_adam=False, inputs_on_adam=False)
    assert len(every.muon_parameters()) == 9
    assert len(make_graft_optimizer(net, 'muon', 1e-3, heads_on_adam=False).muon_parameters()) == 6
    assert graft_input_layers(net) == (net.actor_encoder[0], net.value[0], net.dynamics[0])
    # modules without the graft structure and parameter iterables: no input layers to identify
    plain = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    assert len(make_graft_optimizer(plain, 'muon', 1e-3).muon_parameters()) == 2
    assert len(make_graft_optimizer(list(plain.parameters()), 'muon', 1e-3).muon_parameters()) == 2
    assert graft_output_layers(nn.Linear(3, 3)) == () == graft_input_layers(nn.Sequential(nn.Linear(3, 3)))
    loss = _graft_loss(net, _graft_batch(1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(bool(torch.isfinite(p).all()) for p in net.parameters())
    for bad in (dict(kind='sgd'), dict(kind='adam', momentum=0.9), dict(kind='adam', heads_on_adam=False),
                dict(kind='adam', inputs_on_adam=True), dict(kind='muon', surprise=1), dict(kind='muon', lr=0.0),
                dict(kind='muon', inputs_on_adam=1), dict(kind='muon', inputs_on_adam='yes')):
        with pytest.raises(ValueError):
            make_graft_optimizer(net, **{'lr': 1e-3, **bad})
    for not_a_graft in (list(net.parameters()), nn.Linear(3, 3)):
        with pytest.raises(ValueError):
            make_graft_optimizer(not_a_graft, 'muon', 1e-3, inputs_on_adam=True)


# ====================================================================== Muon experiment
def test_muon_quick_receipt_is_sealed_deterministic_and_leaves_global_state_alone():
    rng_before, threads_before = torch.random.get_rng_state(), torch.get_num_threads()
    first = muon.run_experiment(quick=True, seeds=(0, 1))
    second = muon.run_experiment(quick=True, seeds=(0, 1))
    assert torch.equal(torch.random.get_rng_state(), rng_before) and torch.get_num_threads() == threads_before
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['schema'] == LAB_SCHEMA and first['evidence_tier'] == EVIDENCE_TIER
    assert first['experiment_id'] == 'muon_optimizer' and first['kind'] == 'optimizer_comparison'
    assert first['verdict'] in VERDICTS and set(first['arms']) == set(muon.ARMS)
    assert first['verdict'] == first['comparisons']['muon_vs_adam']['verdict']
    assert first['comparisons']['muon_with_inputs_vs_adam']['role'].startswith('routing_ablation')
    assert first['config']['lr_grid'] == list(muon.QUICK_CONFIG['lr_grid'])
    for arm in first['arms'].values():
        assert arm['selected_lr'] in first['config']['lr_grid'] and len(arm['per_seed']) == 2
        assert all(math.isfinite(r['test_mse']) for r in arm['per_seed'])
    budget, routing = first['budget'], first['parameter_routing']
    for arm, b in budget['per_arm'].items():
        assert b['muon_parameters'] + b['adamw_or_adam_parameters'] == budget['parameters']
        assert b['optimizer_state_numbers'] == b['muon_parameters'] + 2 * b['adamw_or_adam_parameters']
    assert budget['per_arm']['adam']['muon_parameters'] == 0
    assert len(routing['muon']['muon']) == 3 and len(routing['muon_with_inputs']['muon']) == 6
    assert 'measured_wallclock_s' in first and 'estimated_cost' in first and first['notes']
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'improved' if first['verdict'] != 'improved' else 'worse'
    assert not verify_receipt(tampered)
    small = muon.run_experiment(quick=True, seeds=(0, 1), routing_ablation=False, steps=3, lr_grid=(1e-3,))
    assert set(small['arms']) == {'adam', 'muon'} and set(small['comparisons']) == {'muon_vs_adam'}
    assert verify_receipt(small) and small['config']['ablation'] is None


@pytest.mark.parametrize('bad', [dict(seeds=()), dict(seeds=(1, 1)), dict(seeds=(-1,)), dict(quick=1),
                                 dict(surprise=1), dict(lr_grid=()), dict(lr_grid=(0.0,)), dict(hidden=8),
                                 dict(steps=0), dict(heads_on_adam='yes'), dict(routing_ablation=1),
                                 dict(lr_grid=tuple(1e-4 * (k + 1) for k in range(33))), dict(momentum=1.0)])
def test_muon_experiment_validates_arguments(bad):
    with pytest.raises(ValueError):
        muon.run_experiment(**bad)


def test_muon_baseline_gets_the_same_weight_decay_as_the_muon_arms():
    cfg = muon._config(True, {'weight_decay': 0.1})
    net = muon._graft_network(cfg['hidden'], _gen(0))
    base, cand = muon._optimizer('adam', net, 1e-3, cfg), muon._optimizer('muon', net, 1e-3, cfg)
    assert isinstance(base, torch.optim.AdamW) and all(g['weight_decay'] == 0.1 for g in base.param_groups)
    assert all(g['weight_decay'] == 0.1 for g in cand.param_groups)
    assert type(muon._optimizer('adam', net, 1e-3, muon._config(True, {}))) is torch.optim.Adam    # default: graft's
    small = muon.run_experiment(quick=True, seeds=(0, 1), routing_ablation=False, steps=2, lr_grid=(1e-3,),
                                weight_decay=0.1)
    assert verify_receipt(small) and 'AdamW' in small['config']['baseline']


# ====================================================================== RigL schedule and allocation
def test_cosine_drop_fraction_follows_the_paper_formula():
    assert cosine_drop_fraction(0, 0.3, 100) == pytest.approx(0.3)
    assert cosine_drop_fraction(50, 0.3, 100) == pytest.approx(0.15)
    assert cosine_drop_fraction(25, 0.3, 100) == pytest.approx(0.15 * (1 + math.cos(math.pi / 4)))
    assert cosine_drop_fraction(100, 0.3, 100) == 0.0 and cosine_drop_fraction(150, 0.3, 100) == 0.0
    for bad in ((-1, 0.3, 10), (1, 1.5, 10), (1, 0.3, 0), (1.0, 0.3, 10)):
        with pytest.raises(ValueError):
            cosine_drop_fraction(*bad)


def test_erk_allocation_follows_erdos_renyi_and_keeps_the_budget():
    shapes = [(128, 64), (128, 128), (4, 128)]
    dens = layer_densities(shapes, 0.9, 'erk')
    raw = [(o + i) / (o * i) for o, i in shapes]
    assert dens[2] == 1.0                                                     # would exceed 1: made dense
    assert dens[0] / dens[1] == pytest.approx(raw[0] / raw[1])
    counts = layer_active_counts(shapes, 0.9, 'erk')
    uniform = layer_active_counts(shapes, 0.9, 'uniform')
    assert uniform == [820, 1639, 52]                                         # N - floor(0.9 N)
    assert counts[2] == 512 and abs(sum(counts) - sum(uniform)) <= len(shapes)    # same budget up to rounding
    assert layer_densities(shapes, 0.9, 'uniform') == pytest.approx([0.1] * 3)
    assert layer_active_counts([(3, 4)], 0.0, 'erk') == [12]
    for bad in (dict(shapes=[], sparsity=0.5), dict(shapes=[(0, 3)], sparsity=0.5),
                dict(shapes=[(3, 3)], sparsity=1.0), dict(shapes=[(3, 3)], sparsity=0.5, distribution='random')):
        with pytest.raises(ValueError):
            layer_densities(**bad)


def _mlp(sizes=(8, 16, 16, 3), seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        return nn.Sequential(*layers[:-1])


def test_initial_masks_are_seeded_applied_and_budgeted():
    net = _mlp()
    sched = RigLScheduler(net, 0.8, 5, 0.3, 40, 'erk', seed=3)
    again = RigLScheduler(_mlp(), 0.8, 5, 0.3, 40, 'erk', seed=3)
    other = RigLScheduler(_mlp(), 0.8, 5, 0.3, 40, 'erk', seed=4)
    assert all(torch.equal(a, b) for a, b in zip(sched.masks, again.masks))
    assert not all(torch.equal(a, b) for a, b in zip(sched.masks, other.masks))
    assert sched.active_counts() == sched.target_counts == layer_active_counts(sched.shapes, 0.8, 'erk')
    for layer, mask in zip(sched.layers, sched.masks):
        assert not bool(layer.weight.detach()[~mask].any())
    assert len(sched.layers) == 3 and sched.summary()['n_updates'] == 0


def test_update_connections_known_answer():
    layer = nn.Linear(4, 3)
    net = nn.Sequential(layer)
    sched = RigLScheduler(net, 0.5, 10, 0.5, 100, 'uniform', seed=0)
    assert sched.target_counts == [6]
    mask = torch.zeros(12, dtype=torch.bool)
    mask[[0, 1, 2, 5, 7, 10]] = True
    state = sched.state_dict()
    state['masks'] = [mask.reshape(3, 4)]
    sched.load_state_dict(state)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.9, -0.1, 0.5, 0, 0, 0.05, 0, -0.7, 0, 0, 0.2, 0]).reshape(3, 4))
    grad = torch.tensor([0.01 * i for i in range(12)])
    grad[3], grad[1], grad[6], grad[11] = 5.0, -4.0, 3.0, -3.0               # 6 and 11 tie: lower index wins
    layer.weight.grad = grad.reshape(3, 4).clone()
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    opt.state[layer.weight]['momentum_buffer'] = torch.ones(3, 4)
    record = sched.update_connections(0.5, opt)
    # k = 3: drop |w| = 0.05 (5), 0.1 (1), 0.2 (10); grow by |g| among the rest: 3, 1 (just dropped), 6
    expected = torch.zeros(12, dtype=torch.bool)
    expected[[0, 1, 2, 3, 6, 7]] = True
    assert torch.equal(sched.masks[0].reshape(-1), expected)
    assert torch.equal(layer.weight.detach().reshape(-1),
                       torch.tensor([0.9, -0.1, 0.5, 0, 0, 0, 0, -0.7, 0, 0, 0, 0]))   # new ones start at 0
    assert record['layers'] == [{'k': 3, 'activated': 2, 'regrown_same': 1}] and record['fraction'] == 0.5
    kept = torch.zeros(12)
    kept[[0, 1, 2, 7]] = 1.0                                                   # state zeroed elsewhere
    assert torch.equal(opt.state[layer.weight]['momentum_buffer'].reshape(-1), kept)
    assert sched.active_counts() == [6] and sched.n_updates == 1


def test_training_keeps_budget_zero_inactive_weights_and_the_cosine_schedule():
    torch_gen = _gen(11)
    x, y = torch.randn(64, 8, generator=torch_gen), torch.randint(0, 3, (64,), generator=torch_gen)
    net = _mlp()
    sched = RigLScheduler(net, 0.8, delta_t=5, alpha=0.4, t_end=40, distribution='erk', seed=3)
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-2)
    targets = list(sched.target_counts)
    sparse = [c < layer.weight.numel() for c, layer in zip(targets, sched.layers)]
    assert any(sparse)
    for t in range(1, 51):
        before = sched.masks
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        stepped = sched.step(opt)
        update = t % 5 == 0 and t < 40
        assert stepped is (not update) and sched.t == t
        assert sched.active_counts() == targets                               # budget is constant
        for layer, old, new in zip(sched.layers, before, sched.masks):
            w = layer.weight.detach()
            assert not bool(w[~new].any())                                   # masked weights stay exactly zero
            if update:
                assert not bool(w[new & ~old].any())                         # regrown weights start at zero
                buf = opt.state[layer.weight]['momentum_buffer']
                assert not bool(buf[~new | (new & ~old)].any())
            else:
                assert torch.equal(old, new)
    hist = sched.history
    assert [h['t'] for h in hist] == list(range(5, 40, 5)) and sched.n_updates == 7
    for h in hist:
        f = 0.4 / 2 * (1 + math.cos(math.pi * h['t'] / 40))
        assert h['fraction'] == pytest.approx(f, abs=1e-15)
        ks = [r['k'] for r in h['layers']]
        assert ks == [math.floor(f * c + 1e-9) if s else 0 for c, s in zip(targets, sparse)]
    assert any(r['activated'] > 0 for h in hist for r in h['layers'])
    first_k = hist[0]['layers'][0]['k']
    assert first_k > hist[-1]['layers'][0]['k']                                # fraction decays


def test_static_scheduler_never_changes_its_masks():
    torch_gen = _gen(12)
    x, y = torch.randn(32, 8, generator=torch_gen), torch.randint(0, 3, (32,), generator=torch_gen)
    net = _mlp()
    sched = RigLScheduler(net, 0.8, 5, 0.4, 40, 'erk', seed=3, dynamic=False)
    start = sched.masks
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(20):
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        assert sched.step(opt) is True
    assert all(torch.equal(a, b) for a, b in zip(start, sched.masks)) and sched.n_updates == 0
    assert all(not bool(layer.weight.detach()[~m].any()) for layer, m in zip(sched.layers, start))


def test_rigl_regrows_toward_the_inputs_that_matter():
    gen = _gen(13)
    x = torch.randn(512, 32, generator=gen)
    y = (x[:, :4] @ torch.tensor([1.5, -2.0, 1.0, 2.5])).unsqueeze(1)        # only inputs 0..3 carry signal

    def share(mask):
        return float(mask[:, :4].sum() / mask.sum())
    result = {}
    for dynamic in (False, True):
        net = _mlp((32, 32, 1), seed=1)
        sched = RigLScheduler(net, 0.9, delta_t=10, alpha=0.5, t_end=300, distribution='uniform',
                              layers=[net[0]], seed=2, dynamic=dynamic)
        initial = share(sched.masks[0])
        opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)
        bgen = _gen(14)
        for _ in range(300):
            idx = torch.randint(0, 512, (64,), generator=bgen)
            loss = F.mse_loss(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            sched.step(opt)
        with torch.no_grad():
            result[dynamic] = (initial, share(sched.masks[0]), float(F.mse_loss(net(x), y)))
    assert result[False][0] == result[True][0] == result[False][1]            # same start; static never moves
    assert result[True][1] >= 2.5 * result[True][0]       # growth follows the gradient (measured: 0.107 -> 0.369)
    assert result[True][2] < result[False][2]             # full-data MSE (measured: RigL 0.0016, static 0.0039)


def test_update_refuses_bad_gradients_and_changes_nothing():
    net = _mlp()
    sched = RigLScheduler(net, 0.8, 1, 0.5, 100, 'erk', seed=1)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    snap = ([m.clone() for m in sched.masks], [p.detach().clone() for p in net.parameters()], sched.t)
    with pytest.raises(ValueError):                                           # no gradients yet
        sched.step(opt)
    loss = net(torch.ones(2, 8)).sum()
    loss.backward()
    net[0].weight.grad[0, 0] = float('nan')
    with pytest.raises(ValueError):
        sched.step(opt)
    assert all(torch.equal(a, b) for a, b in zip(snap[0], sched.masks)) and sched.t == snap[2]
    assert all(torch.equal(a, b) for a, b in zip(snap[1], net.parameters()))
    with pytest.raises(ValueError):
        sched.update_connections(1.5)
    with pytest.raises(ValueError):
        sched.step('not an optimizer')


@pytest.mark.parametrize('bad', [dict(sparsity=1.0), dict(sparsity=-0.1), dict(delta_t=0), dict(alpha=1.5),
                                 dict(t_end=0), dict(distribution='random'), dict(seed=-1), dict(dynamic=1),
                                 dict(history=0), dict(layers=[nn.Linear(2, 2)]), dict(layers=nn.Linear(8, 16))])
def test_scheduler_validation(bad):
    with pytest.raises(ValueError):
        RigLScheduler(_mlp(), **{'sparsity': 0.5, **bad})


def test_scheduler_rejects_models_without_linear_layers_and_bounds_history():
    with pytest.raises(ValueError):
        RigLScheduler(nn.Sequential(nn.ReLU()), 0.5)
    net = _mlp()
    sched = RigLScheduler(net, 0.8, 1, 0.5, 100, 'erk', seed=1, history=3)
    x = torch.randn(4, 8, generator=_gen(15))
    for _ in range(6):
        net.zero_grad()
        net(x).square().sum().backward()
        sched.update_connections(0.2)
    assert sched.n_updates == 6 and len(sched.history) == 3


def test_scheduler_state_round_trip_continues_identically():
    torch_gen = _gen(16)
    x, y = torch.randn(64, 8, generator=torch_gen), torch.randint(0, 3, (64,), generator=torch_gen)

    def train(net, opt, sched, n):
        for _ in range(n):
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            sched.step(opt)
    a = _mlp(seed=0)
    opt_a = torch.optim.SGD(a.parameters(), lr=0.05, momentum=0.9)
    sched_a = RigLScheduler(a, 0.8, 5, 0.4, 40, 'erk', seed=3)
    train(a, opt_a, sched_a, 12)
    saved = copy.deepcopy((a.state_dict(), opt_a.state_dict(), sched_a.state_dict()))
    b = _mlp(seed=9)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.05, momentum=0.9)
    sched_b = RigLScheduler(b, 0.8, 5, 0.4, 40, 'erk', seed=3)
    b.load_state_dict(saved[0])
    opt_b.load_state_dict(saved[1])
    sched_b.load_state_dict(saved[2])
    assert sched_b.t == 12 and sched_b.history == sched_a.history
    train(a, opt_a, sched_a, 12)
    train(b, opt_b, sched_b, 12)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert all(torch.equal(p, q) for p, q in zip(sched_a.masks, sched_b.masks)) and sched_b.n_updates == sched_a.n_updates
    other = RigLScheduler(_mlp(), 0.7, 5, 0.4, 40, 'erk', seed=3)
    good = sched_a.state_dict()
    for mutate in (lambda s: s['masks'][0].fill_(True), lambda s: s.__setitem__('t', -1),
                   lambda s: s['masks'].pop(), lambda s: s.__setitem__('schema', 'x')):
        bad = copy.deepcopy(good)
        mutate(bad)
        before = (sched_a.masks, sched_a.t)
        with pytest.raises(ValueError):
            sched_a.load_state_dict(bad)
        assert all(torch.equal(p, q) for p, q in zip(before[0], sched_a.masks)) and sched_a.t == before[1]
    with pytest.raises(ValueError):
        other.load_state_dict(good)                                           # different configuration


# ====================================================================== RigL experiment
def test_rigl_quick_receipt_is_sealed_deterministic_and_budget_matched():
    rng_before, threads_before = torch.random.get_rng_state(), torch.get_num_threads()
    first = rigl.run_experiment(quick=True, seeds=(0, 1))
    second = rigl.run_experiment(quick=True, seeds=(0, 1))
    assert torch.equal(torch.random.get_rng_state(), rng_before) and torch.get_num_threads() == threads_before
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['schema'] == LAB_SCHEMA and first['evidence_tier'] == EVIDENCE_TIER
    assert first['experiment_id'] == 'rigl_sparse' and first['kind'] == 'sparse_training_comparison'
    assert first['verdict'] in VERDICTS and set(first['arms']) == {'dense', 'static_sparse', 'rigl'}
    budget = first['budget']
    assert budget['static_and_rigl_share_initial_masks'] and budget['rigl_budget_constant']
    assert budget['inactive_weights_zero'] and budget['sparse_parameters'] < budget['dense_parameters']
    rigl_arm, static_arm = first['arms']['rigl'], first['arms']['static_sparse']
    assert rigl_arm['summary']['parameters'] == static_arm['summary']['parameters'] == budget['sparse_parameters']
    for r in rigl_arm['per_seed']:
        assert r['connectivity_updates'] == first['estimated_cost']['rigl_update_iterations']
        assert r['optimizer_steps'] == first['config']['steps'] - r['connectivity_updates']
    assert all(r['connectivity_updates'] == 0 and r['initial_mask_digest'] == r['final_mask_digest']
               for r in static_arm['per_seed'])
    assert set(first['comparisons']) == {'rigl_vs_static_sparse', 'rigl_vs_dense', 'static_sparse_vs_dense'}
    tampered = copy.deepcopy(first)
    tampered['arms']['rigl']['summary']['test_accuracy'] = 1.0
    assert not verify_receipt(tampered)


@pytest.mark.parametrize('bad', [dict(seeds=()), dict(seeds=(2, 2)), dict(quick='no'), dict(surprise=1),
                                 dict(sparsity=1.0), dict(distribution='random'), dict(n_informative=0),
                                 dict(t_end_fraction=0.0), dict(steps=0)])
def test_rigl_experiment_validates_arguments(bad):
    with pytest.raises(ValueError):
        rigl.run_experiment(**bad)


def test_experiment_registries():
    assert muon.MODULE_ID in MODULE_IDS and rigl.MODULE_ID in MODULE_IDS
    assert muon.EXPERIMENTS == {'muon_optimizer': muon.run_experiment}
    assert rigl.EXPERIMENTS == {'rigl_sparse': rigl.run_experiment}
