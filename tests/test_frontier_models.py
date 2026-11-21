"""Frontier pack, model group: recursive_refiner (tiny recursive model) and dendritic_layer (dANN)."""
import copy
import math

import numpy as np
import pytest
import torch

from synapse.frontier import dendritic as dd
from synapse.frontier import recursive_refiner as rr
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, LAB_SCHEMA, MODULE_IDS

VALID = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]])
RR_TINY = dict(quick=True, seeds=(0, 1), train_puzzles=48, val_puzzles=16, test_puzzles=16, train_eval_puzzles=16,
               updates=3, batch_size=8, latent_dim=8, hidden=8, token_hidden=8, n_latent_steps=1, recursions=1,
               supervision_steps=2)
DD_TINY = dict(quick=True, seeds=(0, 1), train_per_class=3, val_per_class=2, test_per_class=2, updates=3, batch_size=8,
               somas=6, dendrites_per_soma=4, synapses_per_dendrite=9)       # 340 dANN vs 326 dense parameters


def _global_state():
    return torch.random.get_rng_state().clone(), torch.get_num_threads()


def _same_global_state(before):
    rng, threads = before
    return torch.equal(torch.random.get_rng_state(), rng) and torch.get_num_threads() == threads


def _trm(**kw):
    args = dict(latent_dim=8, hidden=16, n_latent_steps=3, supervision_steps=2, recursions=2, tokens=16,
                token_hidden=8, seed=1)
    args.update(kw)
    return rr.TinyRecursiveModel(rr.DIGITS + 1, rr.DIGITS, **args)


# ====================================================================== registry
@pytest.mark.parametrize('mod', [rr, dd])
def test_module_ids_and_experiment_registry(mod):
    assert mod.MODULE_ID in MODULE_IDS
    assert list(mod.EXPERIMENTS) == [mod.MODULE_ID] and mod.EXPERIMENTS[mod.MODULE_ID] is mod.run_experiment


# ====================================================================== Sudoku utilities
def test_sudoku_enumeration_has_288_distinct_valid_grids_closed_under_relabelling():
    grids = rr.enumerate_sudoku4()
    assert grids.shape == (288, 16) and rr.GRID_COUNT == 288
    assert all(rr.is_valid_sudoku4(g) for g in grids)
    table = {tuple(int(v) for v in g) for g in grids}
    assert len(table) == 288
    relabel = np.array([0, 3, 1, 4, 2])                     # digits 1..4 -> 3, 1, 4, 2
    assert all(tuple(int(v) for v in relabel[g]) in table for g in grids[::17])
    assert tuple(int(v) for v in VALID.reshape(-1)) in table
    grids[0, 0] = 9                                         # a copy: the cached table stays intact
    assert rr.enumerate_sudoku4()[0, 0] != 9


def test_validity_and_consistency_checkers_on_known_cases():
    assert rr.is_valid_sudoku4(VALID) and rr.is_valid_sudoku4(VALID.reshape(-1))
    swapped = VALID.copy()
    swapped[0, [0, 1]] = swapped[0, [1, 0]]                 # rows still fine, columns broken
    assert not rr.is_valid_sudoku4(swapped)
    assert not rr.is_valid_sudoku4(np.ones(16, dtype=int))
    bad_digit = VALID.copy()
    bad_digit[0, 0] = 5
    assert not rr.is_valid_sudoku4(bad_digit) and not rr.is_valid_sudoku4(np.zeros(16, dtype=int))
    puzzle = VALID.copy()
    puzzle[0, :2] = 0
    assert rr.is_consistent_puzzle(puzzle) and not rr.is_valid_sudoku4(puzzle)
    clash = np.zeros((4, 4), dtype=int)
    clash[0, 0], clash[1, 1] = 2, 2                         # same box
    assert not rr.is_consistent_puzzle(clash)
    with pytest.raises(ValueError):
        rr.is_valid_sudoku4(np.ones(15, dtype=int))
    with pytest.raises(ValueError):
        rr.is_valid_sudoku4(VALID.astype(float))


def test_solution_count_and_unique_puzzle_generation():
    assert rr.solution_count(np.zeros(16, dtype=int)) == 288
    assert rr.solution_count(VALID) == 1
    grids = rr.enumerate_sudoku4()[:10]
    kw = dict(min_blanks=8, max_blanks=12)
    puzzles, solutions = rr.make_puzzles(grids, 30, rng=np.random.default_rng(3), **kw)
    again = rr.make_puzzles(grids, 30, rng=np.random.default_rng(3), **kw)
    assert np.array_equal(puzzles, again[0]) and np.array_equal(solutions, again[1])
    known = {tuple(int(v) for v in g) for g in grids}
    for p, s in zip(puzzles, solutions):
        blanks = int((p == 0).sum())
        assert 8 <= blanks <= 12 and rr.solution_count(p) == 1
        assert np.array_equal(p[p > 0], s[p > 0]) and tuple(int(v) for v in s) in known
    for bad in (dict(min_blanks=9, max_blanks=8), dict(min_blanks=1, max_blanks=13), dict(min_blanks=0, max_blanks=4)):
        with pytest.raises(ValueError):
            rr.make_puzzles(grids, 2, rng=np.random.default_rng(0), **bad)
    with pytest.raises(ValueError):
        rr.make_puzzles(np.ones((2, 16), dtype=int), 2, rng=np.random.default_rng(0), **kw)
    with pytest.raises(ValueError):
        rr.solution_count(np.full(16, 7))


def test_encode_puzzles_is_one_hot_per_cell():
    puzzles = np.array([[0] * 16, list(range(5)) * 3 + [4]])
    x = rr.encode_puzzles(puzzles)
    assert x.shape == (2, rr.INPUT_DIM) and x.dtype == torch.float32
    cells = x.view(2, 16, 5)
    assert torch.equal(cells.sum(-1), torch.ones(2, 16))
    assert torch.equal(cells.argmax(-1), torch.from_numpy(puzzles))
    for bad in (np.full((1, 16), 5), np.zeros((1, 15), dtype=int), np.zeros((0, 16), dtype=int)):
        with pytest.raises(ValueError):
            rr.encode_puzzles(bad)


def test_benchmark_splits_grids_disjointly_and_puzzles_are_unique():
    cfg = rr._config(True, dict(train_puzzles=24, val_puzzles=8, test_puzzles=12, train_eval_puzzles=8))
    data = rr._seed_data(0, cfg)
    split = {k: set(v.tolist()) for k, v in data['split'].items()}
    assert not (split['train'] & split['test']) and not (split['train'] & split['val'])
    assert not (split['val'] & split['test'])
    table = rr.enumerate_sudoku4()
    for name in ('train', 'val', 'test'):
        puzzles, solutions = data[name][0], data[name][1]
        allowed = {tuple(int(v) for v in table[i]) for i in split[name]}
        assert all(tuple(int(v) for v in s) in allowed for s in solutions)
        assert all(rr.solution_count(p) == 1 for p in puzzles)
    assert np.array_equal(rr._seed_data(0, cfg)['test'][0], data['test'][0])          # deterministic per seed


# ====================================================================== tiny recursive model
def test_trm_recursion_follows_z_then_y_updates_with_gradient_only_in_the_last_recursion():
    torch.manual_seed(0)       # only for the probe inputs below; the model itself uses its own generator
    model = _trm()
    original = model.net
    calls = []

    def record(s):
        calls.append((s.detach().clone(), torch.is_grad_enabled()))
        return original(s)

    model.net = record
    x = rr.encode_puzzles(np.vstack([VALID.reshape(-1), np.zeros(16, dtype=int)]))
    e = model.embed(x)
    y0, z0 = model.initial_state(2)
    y1, z1, logits, q = model.supervision_step(e, y0, z0)
    n = model.n_latent_steps
    assert len(calls) == model.recursions * (n + 1) == model.calls_per_supervision_step()
    assert [g for _, g in calls] == [False] * (n + 1) + [True] * (n + 1)
    with torch.no_grad():                                   # replay the first recursion by hand
        z = z0
        assert torch.allclose(calls[0][0], z0 + y0 + e)
        for i in range(n):
            assert torch.allclose(calls[i][0], z + y0 + e, atol=1e-6)
            z = original(z + y0 + e)
        assert torch.allclose(calls[n][0], y0 + z, atol=1e-6)    # y <- f(y + z): no question embedding
    assert not y1.requires_grad and not z1.requires_grad and y1.grad_fn is None   # detached between steps
    assert logits.shape == (2, rr.ANSWER_DIM) and q.shape == (2,) and logits.requires_grad
    loss = logits.sum() + q.sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_trm_counts_network_applications_for_forward_and_unrolled():
    model = _trm(supervision_steps=3, recursions=2, n_latent_steps=2)
    x = rr.encode_puzzles(np.zeros((4, 16), dtype=int))
    out = model(x)
    assert model.net_calls == 3 * 2 * (2 + 1)
    assert out['step_logits'].shape == (3, 4, rr.ANSWER_DIM) and out['halt_prob'].shape == (3, 4)
    assert torch.equal(out['logits'], out['step_logits'][-1])
    model.net_calls = 0
    logits, q = model.unrolled(x)
    assert model.net_calls == 3 * 2 * (2 + 1) and logits.shape == (4, rr.ANSWER_DIM) and q.shape == (4,)
    with torch.no_grad():                                    # same computation, different gradient path
        assert torch.allclose(logits, model(x)['logits'], atol=1e-5)
    model.net_calls = 0
    model(x, steps=1)
    assert model.net_calls == 2 * 3


def test_halting_head_outputs_probabilities_and_starts_near_sigmoid_minus_five():
    model = _trm()
    x = rr.encode_puzzles(rr.make_puzzles(rr.enumerate_sudoku4()[:4], 6, rng=np.random.default_rng(1),
                                          min_blanks=6, max_blanks=10)[0])
    halt = model(x)['halt_prob']
    assert torch.allclose(halt, torch.full_like(halt, 1.0 / (1.0 + math.exp(5.0))))   # w = 0, b = -5 at init
    with torch.no_grad():
        model.q_head.weight.normal_(0.0, 50.0, generator=torch.Generator().manual_seed(2))
    halt = model(x)['halt_prob']
    assert torch.all((halt >= 0) & (halt <= 1)) and torch.all(torch.isfinite(halt))
    assert torch.all((model.halt_probability(torch.randn(3, 16, 8, generator=torch.Generator().manual_seed(4))) >= 0))


def test_trm_and_mlp_initialisation_is_seeded_locally():
    before = _global_state()
    a, b, c = _trm(seed=3), _trm(seed=3), _trm(seed=4)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.net_in.weight, c.net_in.weight)
    m1, m2 = rr.SinglePassMLP(80, 64, 12, seed=5), rr.SinglePassMLP(80, 64, 12, seed=5)
    assert all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))
    assert _same_global_state(before)
    flat = rr.TinyRecursiveModel(80, 64, 16, 32, 2, 2, seed=0)          # tokens=1: plain vector model
    assert flat.token_in is None and flat(rr.encode_puzzles(np.zeros((2, 16), dtype=int)))['logits'].shape == (2, 64)


@pytest.mark.parametrize('kwargs', [dict(latent_dim=1), dict(n_latent_steps=0), dict(supervision_steps=0),
                                    dict(recursions=0), dict(tokens=0), dict(token_hidden=0), dict(seed=-1),
                                    dict(hidden=2.5)])
def test_trm_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        _trm(**kwargs)


def test_trm_rejects_bad_inputs_and_flat_token_hidden():
    model = _trm()
    for bad in (torch.zeros(2, 79), torch.zeros(80), torch.zeros(2, 80, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(bad)
    with pytest.raises(ValueError):
        rr.TinyRecursiveModel(80, 64, 16, 32, 2, 2, token_hidden=4)
    with pytest.raises(ValueError):
        rr.SinglePassMLP(80, 64, 8)(torch.zeros(2, 81))


def test_parameter_matching_within_tolerance_for_default_and_quick_configs():
    for cfg in (rr.DEFAULTS, rr.QUICK):
        models, hidden = rr._models(0, rr._config(cfg is rr.QUICK, {}))
        counts = {k: m.parameter_count() for k, m in models.items()}
        assert counts['trm'] == counts['unrolled'] == counts['shallow']
        assert abs(counts['mlp'] - counts['trm']) / counts['trm'] <= cfg['param_tolerance']
        assert counts['mlp'] == sum(p.numel() for p in models['mlp'].parameters()) and models['mlp'].hidden == hidden
        trm, unrolled, shallow = models['trm'], models['unrolled'], models['shallow']
        assert all(torch.equal(p, q) for p, q in zip(trm.parameters(), unrolled.parameters()))
        assert all(torch.equal(p, q) for p, q in zip(trm.parameters(), shallow.parameters()))
        assert shallow.supervision_steps == shallow.recursions == 1
    assert rr.matched_mlp_hidden(10_000, 80, 64, 2) == min(range(1, 200), key=lambda h: abs(
        sum(p.numel() for p in rr.SinglePassMLP(80, 64, h).parameters()) - 10_000))


def test_deep_supervision_carries_slots_until_the_step_limit():
    """With a near-zero learning rate the halting head stays at q = -5, so every slot halts exactly at the
    supervision-step limit: 6 updates with N_sup = 2 refill all 8 slots three times (known count)."""
    model = _trm(supervision_steps=2, recursions=1, n_latent_steps=1)
    p, s = rr.make_puzzles(rr.enumerate_sudoku4()[:8], 40, rng=np.random.default_rng(0), min_blanks=4, max_blanks=8)
    cfg = dict(updates=6, batch_size=8, lr=1e-6, weight_decay=0.0, grad_clip=1.0, halt_exploration=0.0, _seed=0)
    log = rr.train_recursive(model, rr.encode_puzzles(p), rr._targets(s), cfg, torch.Generator().manual_seed(0))
    assert log['examples_drawn'] == 8 + 3 * 8
    assert log['mean_supervision_steps_per_finished_example'] == 2.0 and math.isfinite(log['final_loss'])
    model.net_calls = 0
    rr.train_unrolled(model, rr.encode_puzzles(p), rr._targets(s), {**cfg, 'updates': 2},
                      torch.Generator().manual_seed(0))
    assert model.net_calls == 2 * (2 * 1 * 2)               # updates x supervision_steps x recursions x (n + 1)


def test_positive_halting_logit_retires_slots_after_one_step_or_after_the_explored_minimum():
    """Halting branch with a known count: with q = +5 (and a negligible learning rate) every slot halts as soon
    as it is allowed to; with exploration on, not before its random minimum of 2..N_sup steps."""
    p, s = rr.make_puzzles(rr.enumerate_sudoku4()[:8], 40, rng=np.random.default_rng(0), min_blanks=4, max_blanks=8)
    x, t = rr.encode_puzzles(p), rr._targets(s)
    cfg = dict(updates=5, batch_size=8, lr=1e-6, weight_decay=0.0, grad_clip=1.0, halt_exploration=0.0, _seed=0)
    for explore, n_sup in ((0.0, 3), (1.0, 3)):
        model = _trm(supervision_steps=n_sup, recursions=1, n_latent_steps=1)
        with torch.no_grad():
            model.q_head.bias.fill_(5.0)
        log = rr.train_recursive(model, x, t, {**cfg, 'halt_exploration': explore}, torch.Generator().manual_seed(0))
        if explore == 0.0:
            assert log['examples_drawn'] == 8 + 5 * 8 and log['mean_supervision_steps_per_finished_example'] == 1.0
        else:
            assert 2.0 <= log['mean_supervision_steps_per_finished_example'] <= 3.0 and log['examples_drawn'] < 8 + 5 * 8


def test_recursive_metrics_on_a_constructed_case():
    """Final-step, per-step, adaptive (first step with halting probability > 0.5) and calibration metrics."""
    target = torch.stack([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    puzzles = np.zeros((2, 16), dtype=np.int64)
    puzzles[:, :4] = target[:, :4].numpy() + 1                                  # 4 givens, 12 blanks each
    step0 = torch.stack([torch.zeros(16, dtype=torch.long), torch.full((16,), 2)])   # example 1 all wrong
    step1 = target.clone()
    step1[1, 10] = 3                                                                 # one wrong blank cell
    logits = torch.stack([torch.nn.functional.one_hot(p, 4).float().reshape(2, 64) * 5 for p in (step0, step1)])
    halt = torch.tensor([[0.9, 0.6], [0.95, 0.4]])                                  # (steps, examples)

    class Stub:
        def eval(self):
            return self

        def __call__(self, x):
            return {'step_logits': logits, 'halt_prob': halt}

    out = rr.evaluate_recursive(Stub(), None, target, puzzles)
    assert out['cell_accuracy'] == pytest.approx(23 / 24) and out['exact_solve'] == 0.5
    assert out['per_step_exact_solve'] == [0.5, 0.5]
    assert out['adaptive_cell_accuracy'] == pytest.approx(12 / 24) and out['adaptive_exact_solve'] == 0.5
    assert out['adaptive_mean_steps'] == 1.0                                        # both halt at the first step
    assert out['halting_brier'] == pytest.approx((0.01 + 0.36 + 0.0025 + 0.16) / 4)
    assert out['halting_decision_accuracy'] == 0.75 and out['halt_prob_range'] == pytest.approx([0.4, 0.95])


def test_model_state_round_trips_with_fixed_buffers():
    x = rr.encode_puzzles(rr.make_puzzles(rr.enumerate_sudoku4()[:4], 3, rng=np.random.default_rng(2),
                                          min_blanks=6, max_blanks=8)[0])
    a, b = _trm(seed=3), _trm(seed=4)
    assert not torch.equal(a.y_init, b.y_init)
    b.load_state_dict(a.state_dict())                                           # y_init / z_init travel too
    assert torch.equal(a(x)['logits'], b(x)['logits']) and torch.equal(a.z_init, b.z_init)
    la, lb = dd.DendriticNet(144, 10, 4, 3, 9, 'local', seed=1), dd.DendriticNet(144, 10, 4, 3, 9, 'local', seed=2)
    assert not torch.equal(la.dendritic.synaptic_mask, lb.dendritic.synaptic_mask)
    lb.load_state_dict(la.state_dict())                                         # the fixed mask is part of the state
    images = dd.make_glyph_dataset(1, seed=4)[0]
    assert torch.equal(la.dendritic.synaptic_mask, lb.dendritic.synaptic_mask) and torch.equal(la(images), lb(images))


def test_recursive_refiner_tiny_run_is_sealed_deterministic_and_leaves_global_state_alone():
    before = _global_state()
    first = rr.run_experiment(**RR_TINY)
    second = rr.run_experiment(**RR_TINY)
    assert _same_global_state(before)
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['schema'] == LAB_SCHEMA and first['evidence_tier'] == EVIDENCE_TIER
    assert first['experiment_id'] == 'recursive_refiner' and first['kind'] == 'model_comparison'
    assert first['verdict'] in VERDICTS and set(first['arms']) == set(rr.ARMS)
    assert first['best_baseline']['arm'] in ('unrolled', 'shallow', 'mlp')
    budget = first['parameter_budget']
    assert budget['within_tolerance'] and budget['max_relative_gap_vs_trm'] <= first['config']['param_tolerance']
    assert first['data']['grid_overlap_between_splits'] == [0, 0]
    for arm in first['arms'].values():
        for seed_run in arm['per_seed']:
            for split in ('train', 'val', 'heldout'):
                assert 0.0 <= seed_run[split]['exact_solve'] <= 1.0 and 0.0 <= seed_run[split]['cell_accuracy'] <= 1.0
    lo, hi = first['arms']['trm']['per_seed'][0]['heldout']['halt_prob_range']
    assert 0.0 <= lo <= hi <= 1.0
    assert first['estimated_cost']['network_applications_per_puzzle']['trm'] == 2 * 1 * 2
    tampered = copy.deepcopy(first)
    tampered['arms']['trm']['summary']['heldout_exact_solve'] = 1.0
    assert not verify_receipt(tampered)
    assert rr.run_experiment(**{**RR_TINY, 'seeds': (2, 3)})['digest'] != first['digest']


@pytest.mark.parametrize('bad', [dict(seeds=()), dict(seeds=(1, 1)), dict(seeds=(-1,)), dict(seeds=(0.5,)),
                                 dict(quick=1), dict(surprise=1), dict(min_blanks=9, max_blanks=8),
                                 dict(train_grids=200, val_grids=50, test_grids=50), dict(lr=0.0),
                                 dict(updates=2.0), dict(latent_dim=True),
                                 dict(latent_dim=2, hidden=1, token_hidden=1)])     # 83-parameter TRM < 211 MLP min
def test_recursive_refiner_rejects_bad_options(bad):
    with pytest.raises(ValueError):
        rr.run_experiment(**{'quick': True, **bad})


# ====================================================================== dendritic layer
def test_neighbourhoods_follow_the_block_then_ring_rule():
    rng = np.random.default_rng(0)
    shape = (12, 12)
    centre = sorted((r * 12 + c) for r in (4, 5, 6) for c in (4, 5, 6))
    assert dd._neighbourhood((5, 5), 9, shape, rng) == centre                     # full 3x3 block
    assert dd._neighbourhood((0, 0), 4, shape, rng) == [0, 1, 12, 13]              # clipped block at a corner
    assert dd._neighbourhood((0, 0), 9, shape, rng) == [0, 1, 2, 12, 13, 14, 24, 25, 26]   # + the radius-2 ring
    sub = dd._neighbourhood((5, 5), 4, shape, rng)
    assert len(sub) == 4 and set(sub) <= set(centre)                              # subsampled block
    big = dd._neighbourhood((5, 5), 12, shape, rng)
    assert set(centre) <= set(big) and len(big) == 12 and all(max(abs(i // 12 - 5), abs(i % 12 - 5)) <= 2 for i in big)


@pytest.mark.parametrize('sampling', dd.SAMPLING_MODES)
def test_receptive_fields_are_fixed_sparse_and_one_soma_per_dendrite(sampling):
    layer = dd.DendriticLayer(144, 6, 4, 9, sampling, seed=3)
    mask = layer.synaptic_mask
    assert mask.shape == (24, 144) and torch.equal(mask.sum(1), torch.full((24,), 9.0))
    assert layer.mask_density() == pytest.approx(9 / 144)
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    fields = layer.fields.numpy()
    assert all(len(set(row.tolist())) == 9 and list(row) == sorted(row) for row in fields)
    again = dd.DendriticLayer(144, 6, 4, 9, sampling, seed=3)
    assert torch.equal(again.synaptic_mask, mask) and torch.equal(again.synaptic_weight, layer.synaptic_weight)
    cable = layer.cable_mask()
    assert cable.shape == (6, 24) and torch.equal(cable.sum(0), torch.ones(24)) and torch.equal(cable.sum(1), torch.full((6,), 4.0))


def _extent(rows):
    r, c = np.asarray(rows) // 12, np.asarray(rows) % 12
    return max(r.max() - r.min(), c.max() - c.min())


def test_local_and_global_sampling_are_spatially_restricted():
    rng = lambda: np.random.default_rng(11)                                       # noqa: E731
    local = dd.sample_receptive_fields(144, 20, 4, 9, 'local', rng=rng())
    random = dd.sample_receptive_fields(144, 20, 4, 9, 'random', rng=rng())
    glob = dd.sample_receptive_fields(144, 20, 4, 9, 'global', rng=rng())
    assert max(_extent(f) for f in local) <= 4                                    # a dendrite sees one small patch
    assert np.mean([_extent(f) for f in random]) > 7
    per_soma = lambda fields: [_extent(fields[j * 4:(j + 1) * 4].reshape(-1)) for j in range(20)]   # noqa: E731
    assert max(per_soma(glob)) <= 6 < np.mean(per_soma(local))                   # a soma's dendrites cluster


def test_dendritic_layer_forward_matches_hand_computation():
    layer = dd.DendriticLayer(4, 2, 2, 2, 'random', image_shape=(1, 4), seed=0)
    fields = layer.fields.numpy()
    with torch.no_grad():
        layer.synaptic_weight.copy_(torch.tensor([[1.0, -2.0, 0.5, 3.0]] * 4) * layer.synaptic_mask)
        layer.dendritic_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -5.0]))
        layer.cable_weight.copy_(torch.tensor([[1.0, -1.0], [2.0, 0.5]]))
        layer.soma_bias.copy_(torch.tensor([0.0, -0.25]))
    x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
    w = np.array([1.0, -2.0, 0.5, 3.0])
    lrelu = lambda v: v if v > 0 else 0.1 * v                                     # noqa: E731
    dend = [lrelu(sum(w[i] * x[0, i].item() for i in fields[k]) + [0.1, -0.2, 0.3, -5.0][k]) for k in range(4)]
    soma = [lrelu(1.0 * dend[0] - 1.0 * dend[1] + 0.0), lrelu(2.0 * dend[2] + 0.5 * dend[3] - 0.25)]
    assert torch.allclose(layer.dendritic_activity(x)[0], torch.tensor(dend, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(layer(x)[0], torch.tensor(soma, dtype=torch.float32), atol=1e-6)


def test_masked_synapses_get_no_gradient_and_stay_zero_during_training():
    net = dd.DendriticNet(144, 10, 8, 4, 9, 'local', seed=2)
    layer = net.dendritic
    mask = layer.synaptic_mask.clone()
    off = mask == 0
    x, y = dd.make_glyph_dataset(4, seed=9)
    loss = torch.nn.functional.cross_entropy(net(x), y)
    loss.backward()
    assert torch.count_nonzero(layer.synaptic_weight.grad[off]) == 0
    assert torch.count_nonzero(layer.synaptic_weight.grad[~off]) > 0
    start = layer.synaptic_weight.detach().clone()
    log = dd.train_classifier(net, x, y, updates=15, batch_size=16, lr=0.01, generator=torch.Generator().manual_seed(1))
    assert math.isfinite(log['final_loss'])
    assert torch.equal(layer.synaptic_mask, mask)                                 # the mask is fixed
    assert torch.count_nonzero(layer.synaptic_weight.detach()[off]) == 0
    assert not torch.equal(layer.synaptic_weight.detach()[~off], start[~off])     # existing synapses learned
    with torch.no_grad():
        layer.synaptic_weight.add_(1.0)                                            # corrupt, then re-apply W <- W * M
    net.enforce_mask()
    assert torch.count_nonzero(layer.synaptic_weight.detach()[off]) == 0


def test_parameter_counts_include_only_existing_connections_and_dense_matching():
    layer = dd.DendriticLayer(144, 16, 8, 9, 'random', seed=0)
    assert layer.parameter_count() == 16 * 8 * 9 + 16 * 8 + 16 * 8 + 16
    assert layer.parameter_count() < sum(p.numel() for p in layer.parameters())  # dense storage, sparse count
    net = dd.DendriticNet(144, 10, 16, 8, 9, 'random', seed=0)
    assert net.parameter_count() == layer.parameter_count() + 16 * 10 + 10 == dd.dann_parameter_count(16, 8, 9, 10)
    for sampling, shape in (('local', (5, 3, 7)), ('global', (3, 9, 4)), ('random', (1, 1, 1))):
        assert dd.DendriticNet(144, 10, *shape, sampling).parameter_count() == dd.dann_parameter_count(*shape, 10)
    for h in (5, 18, 40):
        assert dd.DenseMLP(144, 10, h).parameter_count() == dd.dense_parameter_count(144, 10, h)
    for cfg in (dd.DEFAULTS, dd.QUICK):
        shape = {k: cfg[k] for k in ('somas', 'dendrites_per_soma', 'synapses_per_dendrite')}
        target = dd.DendriticNet(144, 10, **shape).parameter_count()
        dense = dd.dense_parameter_count(144, 10, dd.matched_dense_hidden(target, 144, 10))
        assert abs(dense - target) / target <= cfg['param_tolerance']


def test_glyphs_are_deterministic_balanced_bounded_and_symmetric():
    x1, y1 = dd.make_glyph_dataset(3, seed=5)
    x2, y2 = dd.make_glyph_dataset(3, seed=5)
    assert torch.equal(x1, x2) and torch.equal(y1, y2) and x1.shape == (30, 144)
    assert torch.equal(torch.bincount(y1, minlength=10), torch.full((10,), 3))
    assert torch.all(torch.isfinite(x1))
    fixed = dict(noise=0.0, shift=0.0, size=(7.0, 7.0), thickness=(0.6, 0.6), rotation=0.0)
    images = {name: dd.render_glyph(i, np.random.default_rng(0), **fixed).reshape(12, 12)
              for i, name in enumerate(dd.GLYPH_NAMES)}
    for img in images.values():
        assert img.min() >= 0.0 and img.max() <= 1.0 and img.sum() > 5.0
    assert np.allclose(images['bar'], images['bar'][:, ::-1])                     # mirror-symmetric
    assert np.allclose(images['plus'], images['plus'].T) and np.allclose(images['box'], images['box'][::-1])
    assert not np.allclose(images['plus'], images['cross'])
    noisy = dd.render_glyph(0, np.random.default_rng(0), **{**fixed, 'noise': 0.3}).reshape(12, 12)
    assert not np.allclose(noisy, images['box'])


@pytest.mark.parametrize('call', [
    lambda: dd.DendriticLayer(144, 4, 2, 9, 'nearest'),
    lambda: dd.DendriticLayer(16, 4, 2, 17, 'random'),
    lambda: dd.DendriticLayer(144, 4, 2, 9, 'local', image_shape=(10, 10)),
    lambda: dd.DendriticLayer(4, 1, 5, 1, 'global', image_shape=(2, 2)),
    lambda: dd.DendriticLayer(144, 4, 2, 9, 'random', negative_slope=1.5),
    lambda: dd.DendriticLayer(144, 0, 2, 9, 'random'),
    lambda: dd.DendriticNet(144, 1, 4, 2, 9),
    lambda: dd.DendriticLayer(144, 4, 2, 9)(torch.zeros(3, 143)),
    lambda: dd.make_glyph_dataset(0, seed=0),
    lambda: dd.make_glyph_dataset(2, seed=0, noise=-1.0),
    lambda: dd.render_glyph(10, np.random.default_rng(0)),
    lambda: dd.train_classifier(dd.DenseMLP(144, 10, 4), torch.zeros(4, 144), torch.zeros(4, dtype=torch.long),
                                updates=1, batch_size=2, lr=0.0, generator=torch.Generator()),
])
def test_dendritic_rejects_bad_arguments(call):
    with pytest.raises(ValueError):
        call()


def test_dendritic_tiny_run_is_sealed_deterministic_and_leaves_global_state_alone():
    before = _global_state()
    first = dd.run_experiment(**DD_TINY)
    second = dd.run_experiment(**DD_TINY)
    assert _same_global_state(before)
    assert first['digest'] == second['digest'] and verify_receipt(first)
    assert first['schema'] == LAB_SCHEMA and first['evidence_tier'] == EVIDENCE_TIER
    assert first['experiment_id'] == 'dendritic_layer' and first['kind'] == 'model_comparison'
    assert first['verdict'] in VERDICTS and first['robustness_verdict'] in VERDICTS
    names = {f'dann_{m}' for m in dd.SAMPLING_MODES} | {'dense_mlp'}
    assert set(first['arms']) == names and first['best_dann']['arm'] in names - {'dense_mlp'}
    budget = first['parameter_budget']
    assert budget['within_tolerance'] and budget['max_relative_gap'] <= first['config']['param_tolerance']
    assert budget['trainable_parameters'] == {'dann_random': 340, 'dann_local': 340, 'dann_global': 340,
                                              'dense_mlp': 326} and budget['dense_hidden'] == 2
    for name in names - {'dense_mlp'}:
        for seed_run in first['arms'][name]['per_seed']:
            assert seed_run['masked_weights_nonzero'] == 0 and seed_run['mask_density'] == pytest.approx(9 / 144)
    for arm in first['arms'].values():
        for k in ('train_accuracy', 'val_accuracy', 'heldout_accuracy', 'heldout_noisy_accuracy'):
            assert 0.0 <= arm['summary'][k] <= 1.0
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'improved' if first['verdict'] != 'improved' else 'worse'
    assert not verify_receipt(tampered)


@pytest.mark.parametrize('bad', [dict(seeds=()), dict(seeds=(2, 2)), dict(seeds=(-3,)), dict(quick='yes'),
                                 dict(surprise=1), dict(noise=-0.1), dict(somas=0), dict(updates=1.5),
                                 dict(synapses_per_dendrite=145), dict(dendrites_per_soma=145),
                                 dict(somas=4, dendrites_per_soma=2, synapses_per_dendrite=4)])   # 102 < 167 dense min
def test_dendritic_rejects_bad_options(bad):
    with pytest.raises(ValueError):
        dd.run_experiment(**{'quick': True, **bad})
