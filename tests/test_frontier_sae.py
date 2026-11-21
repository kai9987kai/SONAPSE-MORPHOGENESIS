"""Frontier pack, module ``sparse_dictionary``: online TopK sparse autoencoder, live observer, causal feature ablation."""
import copy
import json
import math

import numpy as np
import pytest

from synapse.frontier import FrontierSpecV1, OBSERVER_MODULE_IDS
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.sparse_dictionary import (EXPERIMENTS, MODULE_ID, SparseDictionaryObserver, TopKSAE, _topk,
                                                ablation_phase, fraction_of_variance_unexplained, geometric_median,
                                                refit_readout, run_experiment, select_predictive_features, topk_relu,
                                                usage_histogram)
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import ReadoutHead, _state_digest, clone_engine, make_spec, run_task_phase
from synapse.neuroweave.tasks import make_stream

TASK = 'delayed_recall'


def _planted(d, n_atoms, k, n, seed, offset=0.0):
    """Data made of sparse non-negative combinations of ``k`` known unit-norm atoms (the known answer)."""
    rng = np.random.default_rng(seed)
    atoms = rng.standard_normal((d, n_atoms))
    atoms /= np.linalg.norm(atoms, axis=0, keepdims=True)
    codes = np.zeros((n, n_atoms))
    for i in range(n):
        idx = rng.choice(n_atoms, size=k, replace=False)
        codes[i, idx] = rng.uniform(0.5, 1.5, size=k)
    return atoms, codes @ atoms.T + offset


def _trained_engine(seed=0, neurons=16, steps=40):
    engine = NeuroWeaveEngine(make_spec((), seed=seed, neurons=neurons, topology='clustered', steps=steps))
    head = ReadoutHead(engine.n, seed)
    run_task_phase(engine, {TASK: head}, make_stream(seed, steps), learn=True)
    return engine, head


# ---------------------------------------------------------------------- registration
def test_module_is_registered_as_an_observer_with_one_experiment():
    assert MODULE_ID == 'sparse_dictionary' and MODULE_ID in OBSERVER_MODULE_IDS
    assert list(EXPERIMENTS) == [MODULE_ID] and EXPERIMENTS[MODULE_ID] is run_experiment


# ---------------------------------------------------------------------- primitives
def test_topk_relu_is_exact_against_brute_force():
    rng = np.random.default_rng(0)
    pre = np.round(rng.standard_normal((40, 9)), 1)          # rounding creates ties and negatives
    for k in range(1, 10):
        z, idx = topk_relu(pre, k)
        for r in range(pre.shape[0]):
            order = sorted(range(9), key=lambda j: (-pre[r, j], j))[:k]          # ties -> lower index
            assert idx[r].tolist() == order
            expect = np.zeros(9)
            expect[order] = np.maximum(pre[r, order], 0.0)
            assert np.array_equal(z[r], expect)
        assert np.all(np.count_nonzero(z, axis=1) <= k)
    positive, _ = topk_relu(np.abs(pre) + 0.1, 3)
    assert np.all(np.count_nonzero(positive, axis=1) == 3)                     # L0 == k when the top k are > 0
    for bad_k in (0, 10, 2.5, True):
        with pytest.raises(ValueError):
            topk_relu(pre, bad_k)
    with pytest.raises(ValueError):
        topk_relu(np.array([[np.nan, 1.0]]), 1)


def test_metrics_and_geometric_median_known_answers():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 4))
    assert fraction_of_variance_unexplained(x, x) == 0.0
    assert fraction_of_variance_unexplained(x, np.broadcast_to(x.mean(axis=0), x.shape)) == pytest.approx(1.0)
    const = np.ones((5, 3))
    assert fraction_of_variance_unexplained(const, const) == 0.0
    assert fraction_of_variance_unexplained(const, const + 0.1) is None        # no variance to explain
    with pytest.raises(ValueError):
        fraction_of_variance_unexplained(x, x[:, :3])
    hist = usage_histogram([0.0, 5e-4, 5e-3, 0.05, 0.5, 1.0])
    assert hist['counts'] == [1, 1, 1, 1, 2] and len(hist['bins']) == 5
    with pytest.raises(ValueError):
        usage_histogram([1.5])
    assert geometric_median(np.array([[0.0], [1.0], [10.0]]))[0] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(geometric_median(np.array([[0., 0.], [0., 2.], [2., 0.], [2., 2.]])), [1.0, 1.0])
    # minimiser on the diagonal at t = 1/2 + sqrt(3)/6 (solve d/dt of the summed distances = 0)
    gm = geometric_median(np.array([[0., 0.], [1., 0.], [0., 1.], [5., 5.], [1., 1.]]))
    assert np.allclose(gm, 0.5 + math.sqrt(3.0) / 6.0, atol=1e-6)


# ---------------------------------------------------------------------- TopK sparse autoencoder
def test_gradients_match_finite_differences_including_auxk():
    rng = np.random.default_rng(5)
    d, f, k = 6, 10, 3
    x = rng.standard_normal((9, d)) + 0.5
    sae = TopKSAE(d, f, k, seed=2, optimizer='sgd', grad_clip=None, aux_coef=0.25, k_aux=3, dead_after=1)
    sae.init_from_data(rng.standard_normal((20, d)))
    sae.W_enc += 0.3 * rng.standard_normal(sae.W_enc.shape)
    sae.b_enc = 0.1 * rng.standard_normal(f)
    captured = {}
    sae._apply = lambda g: captured.update({name: v.copy() for name, v in g.items()})   # capture, do not step
    res = sae.partial_fit(x)
    dead = np.flatnonzero(sae.dead_mask())
    assert res['aux_loss'] > 0.0 and 0 < dead.size < f
    base = {name: getattr(sae, name).copy() for name in ('W_enc', 'b_enc', 'W_dec', 'b_pre')}

    def loss(p, e_fixed=None):
        pre = (x - p['b_pre']) @ p['W_enc'].T + p['b_enc']
        z = _topk(pre, k)[0]
        err = x - (z @ p['W_dec'].T + p['b_pre'])
        e = err if e_fixed is None else e_fixed               # the AuxK target is a constant (detached)
        za = np.zeros_like(z)
        za[:, dead] = _topk(pre[:, dead], min(3, dead.size))[0]
        aux = np.mean(((e - za @ p['W_dec'].T) ** 2).sum(axis=1) / (e ** 2).sum(axis=1))
        return sae.mse_scale * np.mean(err ** 2) + 0.25 * aux, err

    e0 = loss(base)[1]
    h = 1e-6
    for name, value in base.items():
        g = np.zeros_like(value)
        for idx in np.ndindex(value.shape):
            plus, minus = copy.deepcopy(base), copy.deepcopy(base)
            plus[name][idx] += h
            minus[name][idx] -= h
            g[idx] = (loss(plus, e0)[0] - loss(minus, e0)[0]) / (2 * h)
        if name == 'W_dec':                                    # the step uses the tangent (projected) gradient
            g -= value * np.sum(g * value, axis=0, keepdims=True)
        assert np.max(np.abs(g - captured[name])) <= 1e-6 * max(1.0, np.max(np.abs(g))), name


def test_planted_sparse_dictionary_is_recovered_with_low_fvu():
    atoms, x = _planted(12, 16, 2, 800, seed=0, offset=0.3)
    sae = TopKSAE(12, 16, 2, seed=0, lr=1e-2, dead_after=800)
    untrained = sae.evaluate(x[:200])['fvu']          # b_pre = 0, random dictionary
    sae.fit(x, epochs=30, batch_size=32)
    ev = sae.evaluate(x)
    best_cos = (atoms.T @ sae.W_dec).max(axis=1)      # both unit-norm: max cosine per planted atom
    assert untrained > 0.5 and ev['fvu'] < 0.1
    assert int(np.sum(best_cos > 0.9)) >= 14
    assert 2 - 0.05 <= ev['l0'] <= 2 and ev['never_fired_fraction'] <= 0.125
    assert np.max(np.abs(sae.decoder_norms() - 1.0)) < 1e-12
    assert sum(ev['usage_histogram']['counts']) == 16 and len(ev['usage']) == 16
    assert np.array_equal(sae.encode(x[:7]), topk_relu(sae.pre_activations(x[:7]), 2)[0])
    assert np.allclose(sae.reconstruct(x[3]), sae.decode(sae.encode(x[3])))


@pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
def test_decoder_columns_stay_unit_norm_and_parameters_bounded(optimizer):
    rng = np.random.default_rng(3)
    sae = TopKSAE(8, 12, 3, seed=1, optimizer=optimizer, lr=0.2, grad_clip=None, dead_after=16, max_param=50.0)
    for _ in range(60):
        sae.partial_fit(rng.standard_normal((6, 8)) * 5.0 + 2.0)
        assert np.max(np.abs(sae.decoder_norms() - 1.0)) < 1e-12
    for name in ('W_enc', 'b_enc', 'W_dec', 'b_pre'):
        value = getattr(sae, name)
        assert np.all(np.isfinite(value)) and np.max(np.abs(value)) <= 50.0
    assert sae.updates == 60 and sae.samples_seen == 360


def test_dead_latent_bookkeeping_counts_samples_since_last_fire():
    sae = TopKSAE(4, 6, 1, seed=0, dead_after=7)
    rng = np.random.default_rng(0)
    assert sae.dead_fraction() == 0.0 and not sae.dead_mask().any()        # nothing is dead before training
    expected, fires = np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)
    for rows in (5, 3, 4, 6):
        res = sae.partial_fit(rng.standard_normal((rows, 4)))
        fired = res['z'] > sae.fire_threshold                  # the pre-update codes of this batch
        for j in range(6):
            hits = np.flatnonzero(fired[:, j])
            expected[j] = rows - 1 - hits[-1] if hits.size else expected[j] + rows
        fires += fired.sum(axis=0)
        assert np.array_equal(sae.since_fired, expected) and np.array_equal(sae.fire_count, fires)
        assert np.array_equal(sae.dead_mask(), expected >= 7)
    assert sae.dead_fraction() == pytest.approx(float(np.mean(expected >= 7)))


@pytest.mark.parametrize('seed', [1, 2])
def test_auxk_revives_dead_latents(seed):
    _, x = _planted(12, 16, 2, 800, seed=0)
    killed = np.arange(4)
    runs = {}
    for aux in (0.0, 1.0 / 32.0):
        sae = TopKSAE(12, 16, 2, seed=seed, lr=5e-3, dead_after=64, aux_coef=aux, k_aux=6)
        sae.init_from_data(x)
        sae.W_enc[killed] *= 0.02          # these latents can never win the top-k on their own
        sae.fit(x, epochs=22, batch_size=32)
        runs[aux] = (sae, sae.evaluate(x))
    plain, plain_ev = runs[0.0]
    auxk, auxk_ev = runs[1.0 / 32.0]
    plain_rates, auxk_rates = np.array(plain_ev['usage'])[killed], np.array(auxk_ev['usage'])[killed]
    assert plain_rates.max() < 0.01 and plain.dead_mask()[killed].all()       # without AuxK they stay dead
    assert int(np.sum(auxk_rates > 0.05)) >= 2                                 # AuxK brings them back
    assert auxk_ev['fvu'] < plain_ev['fvu']


def test_fit_is_deterministic_and_resumable():
    _, x = _planted(8, 10, 2, 120, seed=4)
    a, b = (TopKSAE(8, 12, 2, seed=9, lr=1e-2, dead_after=100) for _ in range(2))
    a.fit(x, epochs=3, batch_size=16)
    b.fit(x, epochs=2, batch_size=16)
    b.fit(x, epochs=1, batch_size=16)
    assert a.export_state() == b.export_state()                 # shuffles are keyed by (seed, epoch)
    other = TopKSAE(8, 12, 2, seed=10, lr=1e-2, dead_after=100)
    other.fit(x, epochs=3, batch_size=16)
    assert not np.array_equal(other.W_dec, a.W_dec)


def test_sae_state_round_trip_is_exact_and_validated():
    rng = np.random.default_rng(6)
    sae = TopKSAE(6, 8, 2, seed=4, dead_after=20)
    sae.fit(rng.standard_normal((40, 6)), epochs=2, batch_size=8)
    state = json.loads(json.dumps(sae.export_state()))
    twin = TopKSAE.from_state(state)
    assert twin.export_state() == sae.export_state()
    batch = rng.standard_normal((5, 6))
    sae.partial_fit(batch)
    twin.partial_fit(batch)
    assert twin.export_state() == sae.export_state()

    good = sae.export_state()
    bad_states = []
    for mutate in (lambda s: s.update(schema='Other'),
                   lambda s: s['config'].update(k=3),
                   lambda s: s['W_enc'][0].__setitem__(0, float('nan')),
                   lambda s: s['W_dec'][0].__setitem__(0, s['W_dec'][0][0] + 0.5),
                   lambda s: s['adam_v']['b_pre'].__setitem__(0, -1.0),
                   lambda s: s['since_fired'].__setitem__(0, -1),
                   lambda s: s['since_fired'].__setitem__(0, 1.5),
                   lambda s: s.update(W_enc=s['W_enc'][:-1]),
                   lambda s: s.update(mse_scale=0.0)):
        s = copy.deepcopy(good)
        mutate(s)
        bad_states.append(s)
    for s in bad_states:
        with pytest.raises(ValueError):
            sae.import_state(s)
        assert sae.export_state() == good                  # nothing changed on a rejected import
    no_bias = TopKSAE(6, 8, 2, encoder_bias=False)
    s = no_bias.export_state()
    s['b_enc'][0] = 0.5
    with pytest.raises(ValueError):
        no_bias.import_state(s)


def test_sae_rejects_bad_arguments_without_changing_state():
    for kwargs in ({'n_inputs': 0}, {'k': 9}, {'optimizer': 'rmsprop'}, {'lr': 0.0}, {'aux_coef': -1.0},
                   {'dead_after': 0}, {'encoder_bias': 1}, {'betas': (0.9,)}, {'n_inputs': 2048, 'n_features': 1024}):
        args = {'n_inputs': 6, 'n_features': 8, 'k': 2, **kwargs}
        with pytest.raises(ValueError):
            TopKSAE(args.pop('n_inputs'), args.pop('n_features'), args.pop('k'), **args)
    sae = TopKSAE(6, 8, 2)
    before = sae.export_state()
    for bad in (np.ones((3, 5)), np.full((3, 6), np.nan), np.full((3, 6), 1e9), np.ones(6), 'text'):
        with pytest.raises(ValueError):
            sae.partial_fit(bad)                           # a single row cannot initialise b_pre either
        assert sae.export_state() == before


# ---------------------------------------------------------------------- live observer
def test_observer_learns_online_is_bounded_and_read_only():
    _, x = _planted(10, 8, 2, 400, seed=0)
    obs = SparseDictionaryObserver(10, seed=3, history=64, features=16, k=2, batch_size=8)
    assert obs.summary()['ready'] is False and obs.summary()['fvu_ema'] is None
    early = None
    for i, row in enumerate(x):
        frozen = row.copy()
        frozen.setflags(write=False)                  # the observer must not need to write to its input
        obs.observe(frozen)
        if i == 49:
            early = obs.summary()['fvu_ema']
    row = x[0].copy()
    obs.observe(row)
    row[:] = 99.0                                     # the observer kept a copy, not a view
    s = obs.summary()
    assert np.max(np.abs(np.array(obs.export_state()['buffer']))) < 99.0
    assert early > 0.5 and s['fvu_ema'] < 0.3                                  # it learns the planted dictionary
    assert s['ready'] and s['observed'] == 401 and s['history_fill'] == 64 and s['steps'] == 401 - 32 + 1
    assert 0.0 <= s['dead_fraction'] <= 1.0 and s['l0_ema'] <= 2.0 and s['decoder_norm_max_error'] < 1e-12
    assert 1 <= len(s['top_features']) <= SparseDictionaryObserver.TOP_FEATURES
    usage = [t['usage'] for t in s['top_features']]
    assert usage == sorted(usage, reverse=True)
    for t in s['top_features']:
        assert len(t['top_units']) == 5
        weights = [abs(u['weight']) for u in t['top_units']]
        column = obs.dictionary.W_dec[:, t['feature']]
        assert weights == sorted(weights, reverse=True) and weights[0] == pytest.approx(np.max(np.abs(column)))
    text = json.dumps(s)
    assert len(text) < 16_000 and 'wallclock' not in text
    assert len(obs.export_state()['buffer']) == 64


def test_observer_round_trip_continues_identically_and_rejects_bad_states():
    _, x = _planted(10, 8, 2, 90, seed=1)
    a = SparseDictionaryObserver(10, seed=4, history=32, features=8, k=2, batch_size=4)
    for row in x[:60]:
        a.observe(row)
    state = json.loads(json.dumps(a.export_state()))
    b = SparseDictionaryObserver(10, seed=4, history=32, features=8, k=2, batch_size=4)
    b.import_state(state)
    assert b.export_state() == a.export_state() and b.summary() == a.summary()
    for row in x[60:]:
        a.observe(row)
        b.observe(row)
    assert b.export_state() == a.export_state()

    good = a.export_state()
    mutations = (lambda s: s.update(module='other'),
                 lambda s: s['config'].update(k=3),
                 lambda s: s['buffer'][0].__setitem__(0, float('nan')),
                 lambda s: s.update(buffer=s['buffer'][:-1]),
                 lambda s: s.update(trained=s['observed'] + 1),
                 lambda s: s['fire_ema'].__setitem__(0, 1.5),
                 lambda s: s.update(mse_ema=None),
                 lambda s: s['sae'].update(updates=s['sae']['updates'] + 1),
                 lambda s: s['sae']['W_dec'][0].__setitem__(0, s['sae']['W_dec'][0][0] + 0.5),
                 lambda s: s['var'].__setitem__(0, -1.0))
    for mutate in mutations:
        s = copy.deepcopy(good)
        mutate(s)
        with pytest.raises(ValueError):
            a.import_state(s)
        assert a.export_state() == good
    a.reset()
    assert a.export_state() == SparseDictionaryObserver(10, seed=4, history=32, features=8, k=2,
                                                        batch_size=4).export_state()


def test_observer_validates_inputs_and_parameters():
    obs = SparseDictionaryObserver(6, history=16, features=8, k=2)
    before = obs.export_state()
    for bad in (np.zeros(5), np.zeros((2, 6)), np.full(6, np.nan), np.full(6, np.inf), np.full(6, 2e6), 'abcdef'):
        with pytest.raises(ValueError):
            obs.observe(bad)
    assert obs.export_state() == before and obs.observed == 0
    for kwargs in ({'n_units': 0}, {'history': 8}, {'features': 2}, {'k': 9}, {'seed': -1}, {'lr': 0.0},
                   {'batch_size': 0}, {'ema': 0.0}, {'n_units': 4096, 'features': 4096}):
        args = {'n_units': 6, 'history': 16, 'features': 8, 'k': 2, **kwargs}
        with pytest.raises(ValueError):
            SparseDictionaryObserver(args.pop('n_units'), **args)
    spec = FrontierSpecV1(enabled=True, seed=5, history=64, sae_features=16, sae_k=4, modules=('sparse_dictionary',))
    from_spec = SparseDictionaryObserver.from_spec(12, spec)
    sized = lambda o: (o.config['seed'], o.history, o.config['features'], o.config['k'])      # noqa: E731
    assert sized(from_spec) == (5, 64, 16, 4) and sized(SparseDictionaryObserver(12)) == (7, 256, 64, 8)
    with pytest.raises(ValueError):
        SparseDictionaryObserver.from_spec(12, spec.to_dict())


# ---------------------------------------------------------------------- experiment helpers
def test_select_predictive_features_ranks_alive_features_by_absolute_correlation():
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.5).astype(float)
    codes = np.zeros((200, 6))
    codes[:, 0] = 1.0 - y                                   # r = -1
    codes[:, 2] = y * rng.uniform(0.5, 1.0, 200)            # strong, below 1
    codes[:, 3] = rng.uniform(0.0, 1.0, 200) * (rng.random(200) < 0.3)
    codes[:5, 4] = 1.0                                      # rare: 5 firing rows
    sel = select_predictive_features(codes, y, 3, min_fires=10)
    assert sel['pool'] == [0, 2, 3] and sel['targeted'] == [0, 2, 3]
    assert sel['corr'][0] == pytest.approx(-1.0) and 0.5 < sel['corr'][2] < 1.0
    assert select_predictive_features(codes, y, 1, min_fires=10)['targeted'] == [0]
    with pytest.raises(ValueError):
        select_predictive_features(codes, y[:-1], 2)
    with pytest.raises(ValueError):
        select_predictive_features(codes, y, 0)


def test_refit_readout_reaches_the_regularised_optimum():
    rng = np.random.default_rng(8)
    head = ReadoutHead(4, seed=1)
    x = rng.standard_normal((300, head.dim)) * 0.5
    w_true = rng.standard_normal(head.dim)
    y = (rng.random(300) < 1.0 / (1.0 + np.exp(-(x @ w_true)))).astype(float)
    info = refit_readout(head, x, y, l2=1e-2)
    p = 1.0 / (1.0 + np.exp(-(x @ head.w + head.b)))
    grad_w = x.T @ (p - y) / 300 + 1e-2 * head.w
    grad_b = float(np.mean(p - y))
    assert np.max(np.abs(grad_w)) < 1e-6 and abs(grad_b) < 1e-6 and info['iterations'] < 50
    with pytest.raises(ValueError):
        refit_readout(head, x, y + 0.5)


def test_ablation_matches_the_lab_protocol_and_only_touches_clones():
    engine, head = _trained_engine()
    digest, weights = _state_digest(engine), head.w.copy()
    held = make_stream(100, 20)
    base = ablation_phase(engine, head, held, TASK, record=True)
    lab = run_task_phase(clone_engine(engine), {TASK: head.copy()}, held, learn=False)['tasks'][TASK]
    assert (base['accuracy'], base['log_loss'], base['brier']) == (lab['accuracy'], lab['log_loss'], lab['brier'])
    assert base['removed_norm'] == 0.0 and base['ablated'] == []
    sae = TopKSAE(head.n0, 8, 2, seed=1, lr=1e-2, dead_after=40)
    sae.fit(base['outputs'], epochs=10, batch_size=8)
    feats = np.flatnonzero((sae.encode(base['outputs']) > 0.0).any(axis=0))[:3].tolist()
    assert feats
    abl = ablation_phase(engine, head, held, TASK, sae=sae, ablate=feats, record=True)
    x0 = base['outputs'][0]                      # step 0: same clone state and input, so the raw output matches
    expected = np.clip(x0 - sae.W_dec[:, feats] @ sae.encode(x0)[feats], -1.0, 1.0)
    assert np.allclose(abl['outputs'][0], expected, atol=1e-12)
    assert abl['ablated'] == sorted(feats) and abl['removed_norm'] > 0.0
    assert not np.array_equal(abl['outputs'], base['outputs'])
    assert _state_digest(engine) == digest and np.array_equal(head.w, weights)      # original never stepped
    for kwargs in ({'ablate': [0]}, {'sae': sae, 'ablate': [8]}, {'sae': TopKSAE(5, 8, 2), 'ablate': [0]}):
        with pytest.raises(ValueError):
            ablation_phase(engine, head, held, TASK, **kwargs)
    with pytest.raises(ValueError):
        ablation_phase(engine, head, held, 'nope')


# ---------------------------------------------------------------------- experiment
def test_run_experiment_validates_arguments_before_running():
    for kwargs in ({'bogus': 1}, {'seeds': ()}, {'seeds': (0, 0)}, {'seeds': (0, 100)}, {'task': 'nope'},
                   {'readout': 'magic'}, {'n_ablate': 999}, {'topology': 'nope'}, {'steps': 8},
                   {'mechanisms': ('not_a_mechanism',)}, {'neurons': 4096, 'features': 4096}):
        with pytest.raises(ValueError):
            run_experiment(quick=True, **kwargs)
    with pytest.raises(ValueError):
        run_experiment(quick='yes')


@pytest.fixture(scope='module')
def quick_receipts():
    return run_experiment(quick=True), run_experiment(quick=True)


def test_quick_receipt_is_sealed_deterministic_and_states_what_improved_means(quick_receipts):
    first, second = quick_receipts
    assert first['digest'] == second['digest'] and verify_receipt(first) and verify_receipt(second)
    assert first['experiment_id'] == MODULE_ID and first['kind'] == 'feature_ablation'
    assert first['evidence_tier'] == 'synthetic_abstraction' and first['config']['quick'] is True
    assert first['verdict'] in VERDICTS
    assert first['verdict'] == first['comparisons']['targeted_vs_random_log_loss']['verdict']
    assert 'targeted' in first['verdict_basis'] and 'random' in first['verdict_basis']
    assert "'improved' means" in first['notes'] and 'Gao et al.' in first['source']
    assert first['integrity']['original_engines_unchanged'] and first['integrity']['decoder_columns_unit_norm']
    assert set(first['arms']) == {'none', 'targeted', 'random'}
    for seed in first['per_seed']:
        sel = seed['selection']
        assert seed['original_engine_unchanged'] and len(sel['targeted']) == first['config']['n_ablate']
        assert all(len(s) == len(sel['targeted']) for s in sel['random_sets'])
        assert all(t['fires'] >= sel['min_fires'] for t in sel['targeted'])
        assert seed['dictionary']['train_fvu'] < 0.5 and seed['dictionary']['train_l0'] <= first['config']['k']
        assert seed['heldout']['none']['removed_norm'] == 0.0 and seed['heldout']['targeted']['removed_norm'] > 0.0
    assert json.loads(json.dumps(first)) == first
    timed = copy.deepcopy(first)
    timed['measured_wallclock_s']['total'] += 1.0           # wall-clock is outside the digest
    assert verify_receipt(timed)
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'worse' if first['verdict'] != 'worse' else 'improved'
    assert not verify_receipt(tampered)
