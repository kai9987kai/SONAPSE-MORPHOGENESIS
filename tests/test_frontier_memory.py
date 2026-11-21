"""Frontier modules ``continuum_memory`` and ``btsp_memory``: memories, engine extensions and benchmarks."""
import base64
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from synapse.frontier import btsp as BT
from synapse.frontier import continuum_memory as CM
from synapse.frontier.extension import EngineExtension
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, MODULE_IDS, OBSERVER_MODULE_IDS
from synapse.neuroweave.engine import NeuroWeaveEngine
from synapse.neuroweave.experiments import make_spec
from synapse.neuroweave.spec import SENSORY_CHANNEL_IDS, SENSORY_DIM, TASK_INPUT_DIM
from synapse.neuroweave.tasks import make_stream


def _engine(seed=0, neurons=24):
    return NeuroWeaveEngine(make_spec((), seed=seed, neurons=neurons, topology='clustered', steps=16))


def _drive(engine, steps, seed=0, learn=True):
    stream = make_stream(seed, steps)
    outs = []
    for i in range(steps):
        outs.append(engine.step(stream.inputs[i], stream.sensory[i], reward=0.2 if i % 2 else -0.1,
                                learn=learn)['output'].copy())
    return outs


def _contexts(seed, steps):
    """Engine-shaped step contexts (sensory frame, task input, reward) from a NEUROWEAVE task stream.

    The stream's sensory rows are channel -> value dicts (``None`` = missing); they become the 16-channel frame in
    ``SENSORY_CHANNEL_IDS`` order with missing channels at 0."""
    stream = make_stream(seed, steps)
    frames = [np.array([0.0 if row[c] is None else float(row[c]) for c in SENSORY_CHANNEL_IDS]) for row in stream.sensory]
    return [{'sensory': frames[i], 'task_input': np.asarray(stream.inputs[i], dtype=float),
             'reward': 0.3 if i % 3 else -0.4} for i in range(steps)]


def _unit(rng, n):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


_LEAVES = (np.ndarray, np.random.Generator, int, float, str, bool, type(None), np.integer, np.floating, np.bool_)


def _assert_holds_no_engine(obj, depth=0):
    """Only arrays, numbers, strings, generators, containers and the module's own memory objects are reachable."""
    assert depth < 6
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_holds_no_engine(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_holds_no_engine(v, depth + 1)
    elif isinstance(obj, (CM.ContinuumMemory, CM.ContinuumMemoryExtension, BT.BTSPMemory, BT.BTSPExtension)):
        for v in vars(obj).values():
            _assert_holds_no_engine(v, depth + 1)
    else:
        assert isinstance(obj, _LEAVES), type(obj)


def _global_random_state():
    torch = sys.modules.get('torch')
    return np.random.get_state(), (None if torch is None else (torch.get_rng_state().clone(), torch.get_num_threads()))


def _same_global_random_state(before, after):
    (np_a, torch_a), (np_b, torch_b) = before, after
    same = np_a[0] == np_b[0] and np.array_equal(np_a[1], np_b[1]) and np_a[2:] == np_b[2:]
    if torch_a is not None and torch_b is not None:
        same = same and bool((torch_a[0] == torch_b[0]).all()) and torch_a[1] == torch_b[1]
    return same


@pytest.fixture(scope='module')
def quick_receipts():
    """Every registered experiment of both modules, run once in quick mode (shared by the receipt tests)."""
    before = _global_random_state()
    receipts = {key: runner(quick=True) for mod in (CM, BT) for key, runner in mod.EXPERIMENTS.items()}
    return receipts, _same_global_random_state(before, _global_random_state())


# ====================================================================== registry
def test_every_registered_experiment_runs_quick_and_returns_a_sealed_receipt(quick_receipts):
    receipts, global_state_untouched = quick_receipts
    assert global_state_untouched                       # no global NumPy / torch RNG or thread-count change
    assert set(receipts) == {'continuum_memory', 'continuum_memory_recall', 'btsp_memory', 'btsp_memory_capacity'}
    for mod in (CM, BT):
        for key in mod.EXPERIMENTS:
            r = receipts[key]
            assert verify_receipt(r) and r['experiment_id'] == key and r['evidence_tier'] == EVIDENCE_TIER
            assert r['verdict'] in VERDICTS and isinstance(r['verdict_basis'], str) and isinstance(r['notes'], str)
            assert r['module'] == mod.MODULE_ID and r['source'] == mod.SOURCE and r['config']['quick'] is True
            assert r['measured_wallclock_s'] and all(v >= 0.0 for v in r['measured_wallclock_s'].values())
            json.dumps(r)


def test_modules_are_registered_with_their_experiments():
    assert CM.MODULE_ID == 'continuum_memory' and BT.MODULE_ID == 'btsp_memory'
    for mod in (CM, BT):
        assert mod.MODULE_ID in MODULE_IDS and mod.MODULE_ID not in OBSERVER_MODULE_IDS
        assert list(mod.EXPERIMENTS)[0] == mod.MODULE_ID and mod.EXPERIMENTS[mod.MODULE_ID] is mod.run_experiment
        assert all(k == mod.MODULE_ID or k.startswith(mod.MODULE_ID + '_') for k in mod.EXPERIMENTS)
    assert list(CM.EXPERIMENTS) == ['continuum_memory', 'continuum_memory_recall']
    assert CM.EXPERIMENTS['continuum_memory_recall'] is CM.run_recall_benchmark
    assert list(BT.EXPERIMENTS) == ['btsp_memory', 'btsp_memory_capacity']
    assert BT.EXPERIMENTS['btsp_memory_capacity'] is BT.run_capacity_benchmark
    assert issubclass(CM.ContinuumMemoryExtension, EngineExtension) and CM.ContinuumMemoryExtension.id == CM.MODULE_ID
    assert issubclass(BT.BTSPExtension, EngineExtension) and BT.BTSPExtension.id == BT.MODULE_ID


# ====================================================================== continuum memory: core
def test_single_write_then_read_recovers_the_value():
    rng = np.random.default_rng(0)
    k, v = _unit(rng, 6), rng.standard_normal(3)
    mem = CM.ContinuumMemory(6, 3, levels=1, lr=0.5, forget=0.0)
    assert np.array_equal(mem.predict(k), np.zeros(3))
    info = mem.write(k, v)
    # one delta-rule step with lr = 1/2 on a unit key is an exact write: M = v k^T
    assert np.allclose(mem.M[0], np.outer(v, k), atol=1e-12) and np.allclose(mem.predict(k), v, atol=1e-12)
    assert info['updated_levels'] == [0] and info['surprise'] == 0.5            # empty memory: surprise 1/2
    assert info['error'] == pytest.approx(float(v @ v) / 3)                     # MSE of the read before the write
    k2 = rng.standard_normal(6)
    k2 -= (k2 @ k) * k
    k2 /= np.linalg.norm(k2)
    assert np.allclose(mem.predict(k2), 0.0, atol=1e-12)                       # an orthogonal key reads nothing
    v2 = rng.standard_normal(3)
    mem.write(k2, v2)
    assert np.allclose(mem.predict(k), v, atol=1e-12) and np.allclose(mem.predict(k2), v2, atol=1e-12)
    # a shared key: only the period-1 level has updated after one write, and it holds the whole value
    multi = CM.ContinuumMemory(6, 3, levels=3, lr=0.5, forget=0.0)
    multi.write(k, v)
    reads = multi.read(k)
    assert reads.shape == (3, 3) and np.allclose(reads[0], v, atol=1e-12) and not reads[1:].any()


def test_level_l_updates_exactly_every_base_period_times_two_to_the_l_writes():
    rng = np.random.default_rng(1)
    mem = CM.ContinuumMemory(4, 2, levels=4, base_period=3, lr=0.1, forget=0.05)
    assert mem.periods == (3, 6, 12, 24)
    for t in range(1, 49):
        before = mem.M.copy()
        info = mem.write(rng.standard_normal((4, 4)), rng.standard_normal(2))
        due = [level for level, p in enumerate(mem.periods) if t % p == 0]
        assert info['updated_levels'] == due
        assert [not np.array_equal(mem.M[level], before[level]) for level in range(4)] == [lv in due for lv in range(4)]
    state = mem.export_state()
    assert state['update_counts'] == [16, 8, 4, 2] and state['writes'] == 48 and state['acc_count'] == [0, 0, 0, 0]


def test_chunk_gradient_is_the_sum_of_per_write_gradients_and_mean_divides_by_the_period():
    rng = np.random.default_rng(2)
    keys, vals, lr = rng.standard_normal((2, 2, 3)), rng.standard_normal((2, 2)), 0.2
    for reduce, scale in (('sum', 1.0), ('mean', 0.5)):
        mem = CM.ContinuumMemory(3, 2, levels=2, base_period=1, lr=lr, forget=0.0, chunk_reduce=reduce)
        mem.write(keys[0], vals[0])
        mem.write(keys[1], vals[1])
        # by hand: level 0 (period 1) steps after each write; level 1 (period 2) applies its two-write chunk once,
        # each gradient taken from the joint read at that write (Nested Learning Eq. 71, constant learning rate)
        m0, m1 = np.zeros((2, 3)), np.zeros((2, 3))
        e1 = m0 @ keys[0, 0] + m1 @ keys[0, 1] - vals[0]
        m0 = m0 - lr * 2 * np.outer(e1, keys[0, 0])
        e2 = m0 @ keys[1, 0] + m1 @ keys[1, 1] - vals[1]
        m0 = m0 - lr * 2 * np.outer(e2, keys[1, 0])
        m1 = m1 - lr * scale * (2 * np.outer(e1, keys[0, 1]) + 2 * np.outer(e2, keys[1, 1]))
        assert np.allclose(mem.M[0], m0, atol=1e-12) and np.allclose(mem.M[1], m1, atol=1e-12)


def test_momentum_and_forgetting_follow_the_titans_recurrences():
    rng = np.random.default_rng(3)
    keys, vals = rng.standard_normal((4, 4)), rng.standard_normal((4, 2))
    mu, lr, alpha = 0.6, 0.15, 0.1
    mem = CM.ContinuumMemory(4, 2, levels=1, lr=lr, momentum=mu, forget=alpha)
    m, s = np.zeros((2, 4)), np.zeros((2, 4))
    for k, v in zip(keys, vals):
        s = mu * s - lr * 2 * np.outer(m @ k - v, k)        # S_t = eta S_{t-1} - theta grad   (Titans Eq. 10)
        m = (1 - alpha) * m + s                              # M_t = (1 - alpha) M_{t-1} + S_t (Titans Eq. 13)
        mem.write(k, v)
        assert np.allclose(mem.S[0], s, atol=1e-12) and np.allclose(mem.M[0], m, atol=1e-12)


def test_forgetting_decays_each_level_at_its_own_update_rate():
    rng = np.random.default_rng(4)
    m0 = rng.uniform(-1.0, 1.0, (3, 2, 3))
    zero_k, zero_v = np.zeros(3), np.zeros(2)
    for forget in (0.1, 0.0):
        mem = CM.ContinuumMemory(3, 2, levels=3, base_period=1, lr=0.3, forget=forget)
        state = mem.export_state()
        state['M'] = m0.tolist()
        mem.import_state(state)
        for _ in range(8):
            info = mem.write(zero_k, zero_v)        # zero error and zero gradient: only the decay acts
            assert info['error'] == 0.0 and info['surprise'] == 0.0
        for level, period in enumerate(mem.periods):         # periods 1, 2, 4 -> 8, 4 and 2 decays
            assert np.allclose(mem.M[level], (1.0 - forget) ** (8 // period) * m0[level], atol=1e-12)
    # a stored association fades geometrically
    one = CM.ContinuumMemory(3, 2, levels=1, lr=0.5, forget=0.2)
    k, v = _unit(rng, 3), rng.standard_normal(2)
    one.write(k, v)
    for n in range(1, 4):
        one.write(zero_k, zero_v)
        assert np.allclose(one.predict(k), 0.8 ** n * v, atol=1e-12)


def test_surprise_raises_the_forgetting_gate_up_to_its_cap():
    k, v = np.array([1.0, 0.0]), np.array([2.0])
    for cap, expected in ((0.5, 0.02 + 0.2 * 0.5), (0.05, 0.05)):
        mem = CM.ContinuumMemory(2, 1, levels=1, lr=0.25, forget=0.02, surprise_forget=0.2, max_forget=cap)
        mem.write(k, v)                                       # empty memory: surprise 1/2
        assert mem.last_surprise == 0.5 and mem.last_alpha[0] == pytest.approx(expected)
    mem = CM.ContinuumMemory(2, 1, levels=1, lr=0.25, forget=0.0)
    mem.write(k, v)                                           # reads 1.0 next time (lr 1/4 halves the error)
    assert mem.predict(k) == pytest.approx([1.0])
    mem.write(k, -v)                                          # confidently wrong: e = 9, ||v||^2 = 4
    assert mem.last_surprise == pytest.approx(9.0 / 13.0) and mem.last_error == pytest.approx(9.0)


def test_state_stays_finite_and_bounded():
    rng = np.random.default_rng(5)
    mem = CM.ContinuumMemory(4, 3, levels=3, lr=10.0, momentum=0.9, forget=0.0, bound=2.0)
    for _ in range(40):
        mem.write(rng.uniform(-1e3, 1e3, 4), rng.uniform(-1e3, 1e3, 3))
        assert np.all(np.isfinite(mem.M)) and np.abs(mem.M).max() <= 2.0 and np.abs(mem.S).max() <= 2.0
    assert mem.parameter_count == 3 * 3 * 4 and mem.state_numbers > 3 * mem.parameter_count


def test_each_memory_mechanism_changes_the_recall_trace():
    stream = CM.make_recall_stream(0, 160, key_dim=16, value_dim=4)

    def trace(levels=4, base_period=1, **kw):
        cfg = {'lr': 0.25, 'momentum': 0.0, 'forget': 0.0, **kw}
        mem = CM.ContinuumMemory(16 // levels, 4, levels, base_period, **cfg)
        preds = []
        for k, v in zip(stream['keys'], stream['values']):
            kk = k.reshape(mem.levels, mem.key_dim)
            preds.append(mem.predict(kk))
            mem.write(kk, v)
        return np.array(preds)

    base = trace()
    assert np.array_equal(base, trace())                                     # deterministic
    for toggle in ({'momentum': 0.5}, {'forget': 0.02}, {'surprise_forget': 0.3, 'max_forget': 0.5},
                   {'chunk_reduce': 'mean'}, {'base_period': 2}, {'levels': 2}, {'lr': 0.1}):
        assert not np.allclose(trace(**toggle), base), toggle


def test_with_every_period_one_the_cms_equals_its_single_level_control():
    """The recall benchmark's capacity-matched control differs from the CMS only in the update schedule: with every
    period forced to 1, per-level key blocks reproduce one memory reading the concatenated key exactly."""
    rng = np.random.default_rng(12)
    kw = dict(lr=0.2, momentum=0.5, forget=0.05, surprise_forget=0.1)
    cms = CM.ContinuumMemory(4, 3, levels=3, **kw)
    single = CM.ContinuumMemory(12, 3, levels=1, **kw)
    cms.periods = (1, 1, 1)                                     # schedule ablation for this test only
    for _ in range(20):
        k, v = rng.standard_normal(12), rng.standard_normal(3)
        assert np.allclose(cms.predict(k.reshape(3, 4)), single.predict(k), atol=1e-12)
        assert cms.write(k.reshape(3, 4), v)['surprise'] == pytest.approx(single.write(k, v)['surprise'], abs=1e-12)
    assert np.allclose(np.concatenate(list(cms.M), axis=1), single.M[0], atol=1e-12)
    assert np.allclose(np.concatenate(list(cms.S), axis=1), single.S[0], atol=1e-12)
    normal = CM.ContinuumMemory(4, 3, levels=3, **kw)          # the real schedule (periods 1, 2, 4) differs:
    fresh = CM.ContinuumMemory(12, 3, levels=1, **kw)
    k, v = rng.standard_normal(12), rng.standard_normal(3)
    normal.write(k.reshape(3, 4), v)
    fresh.write(k, v)
    assert np.allclose(normal.M[0], fresh.M[0][:, :4], atol=1e-12)   # the period-1 level matches its key block,
    assert not normal.M[1:].any() and fresh.M[0][:, 4:].all()      # the slower levels wait for their chunk


@pytest.mark.parametrize('kwargs', [{'key_dim': 0}, {'value_dim': 5000}, {'levels': 13}, {'levels': True},
                                    {'base_period': 0}, {'lr': -0.1}, {'lr': float('nan')}, {'momentum': 1.0},
                                    {'forget': 1.0}, {'forget': 0.2, 'max_forget': 0.1}, {'surprise_forget': 2.0},
                                    {'chunk_reduce': 'max'}, {'bound': 0.0}, {'lr': '0.5'}])
def test_continuum_memory_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        CM.ContinuumMemory(**{'key_dim': 4, 'value_dim': 2, **kwargs})


def test_bad_reads_and_writes_raise_without_changing_state():
    mem = CM.ContinuumMemory(3, 2, levels=2)
    mem.write(np.ones(3), np.ones(2))
    before = mem.export_state()
    for key, value in ((np.ones(4), np.ones(2)), (np.ones((3, 3)), np.ones(2)), (np.ones(3), np.ones(3)),
                       (np.full(3, np.nan), np.ones(2)), (np.ones(3), np.array([np.inf, 0.0])),
                       (np.full(3, 1e7), np.ones(2)), (['a', 'b', 'c'], np.ones(2))):
        with pytest.raises(ValueError):
            mem.write(key, value)
    with pytest.raises(ValueError):
        mem.read(np.ones(5))
    assert mem.export_state() == before


def test_continuum_memory_state_round_trip_and_rejection():
    rng = np.random.default_rng(6)
    kw = dict(levels=3, lr=0.2, momentum=0.3, forget=0.01, surprise_forget=0.1)
    mem = CM.ContinuumMemory(3, 2, **kw)
    for _ in range(5):
        mem.write(rng.standard_normal(3), rng.standard_normal(2))
    state = json.loads(json.dumps(mem.export_state()))
    twin = CM.ContinuumMemory(3, 2, **kw)
    twin.import_state(state)
    assert twin.export_state() == mem.export_state()
    for _ in range(3):
        k, v = rng.standard_normal(3), rng.standard_normal(2)
        assert twin.write(k, v) == mem.write(k, v)
    assert twin.export_state() == mem.export_state()
    before = twin.export_state()
    big = copy.deepcopy(state)
    big['M'][0][0][0] = 1e3                                   # beyond the bound of 100
    nan = copy.deepcopy(state)
    nan['S'][1][0][0] = float('nan')
    stray = copy.deepcopy(state)                              # 5 writes, periods (1, 2, 4): level 0's chunk is empty
    stray['acc'][0][0][0] = 1.0
    huge = copy.deepcopy(state)                               # level 2's chunk is open, but no valid write gets here
    huge['acc'][2][0][0] = 1e200
    for bad in ({**state, 'config': {**state['config'], 'lr': 0.3}}, {**state, 'M': state['M'][:2]}, big, nan,
                stray, huge, {**state, 'writes': -1}, {**state, 'writes': 6}, {**state, 'acc_count': [0, 0, 0]},
                {**state, 'update_counts': [5, 2, 2]}, {**state, 'acc_count': [0.0, 1.0, 1.0]},
                {**state, 'update_counts': [5, 2, True]}, {**state, 'acc_count': 0},
                {**state, 'acc_surprise': [-0.1, 0.0, 0.0]},
                {**state, 'last_alpha': [0.9, 0.0, 0.0]}, {**state, 'last_surprise': 2.0},
                {k: v for k, v in state.items() if k != 'S'}, 'not a mapping'):
        with pytest.raises(ValueError):
            twin.import_state(bad)
    assert twin.export_state() == before                      # failed imports leave no trace


# ====================================================================== continuum memory: engine extension
def test_continuum_extension_never_changes_the_engine_and_keeps_no_engine_reference():
    plain, hooked = _engine(seed=2), _engine(seed=2)
    ext = CM.ContinuumMemoryExtension(seed=1)
    hooked.attach_extension(ext)
    _assert_holds_no_engine(ext)
    a, b = _drive(plain, 10), _drive(hooked, 10)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert plain.export_state() == hooked.export_state()
    tel = ext.telemetry()
    assert tel['writes'] == 9 and tel['steps'] == 10 and tel['periods'] == [1, 2, 4, 8]   # no write before a context
    assert tel['update_counts'] == [9, 4, 2, 1]
    feats = ext.features()
    assert feats.shape == (ext.feature_dim,) == (8,) and np.all(np.isfinite(feats)) and np.abs(feats).max() <= 10.0
    assert np.abs(feats).max() > 0.0
    json.dumps(tel)
    _assert_holds_no_engine(ext)
    assert ext.parameter_count() == 4 * SENSORY_DIM * 8


def test_continuum_extension_learns_a_periodic_sensory_forecast():
    frames = np.random.default_rng(0).uniform(-1.0, 1.0, (3, SENSORY_DIM))
    ext = CM.ContinuumMemoryExtension(seed=1)
    errors = []
    for t in range(60):
        ext.after_step(np.zeros(4), {'learn': True, 'sensory': frames[t % 3], 'task_input': None, 'reward': 0.0})
        context = np.concatenate([np.zeros(TASK_INPUT_DIM), frames[t % 3]])
        forecast = ext.memory.predict(ext.key(context))
        errors.append(float(np.mean((forecast - frames[(t + 1) % 3]) ** 2)))
    assert errors[-1] < 1e-2 < errors[0] and errors[-1] < 0.01 * errors[0]
    # the features are fixed projections of the per-level reads of the current context
    reads = ext.memory.read(ext.key(np.concatenate([np.zeros(TASK_INPUT_DIM), frames[59 % 3]])))
    assert np.allclose(ext.features(), np.clip(np.einsum('lfv,lv->lf', ext._P, reads).ravel(), -10, 10))


def test_continuum_extension_is_frozen_without_learning_and_deep_copies_are_independent():
    ctxs = _contexts(3, 12)
    ext = CM.ContinuumMemoryExtension(seed=2)
    for c in ctxs[:6]:
        ext.after_step(np.zeros(4), {**c, 'learn': True})
    memory_before = ext.memory.export_state()
    frozen = copy.deepcopy(ext)
    feats = []
    for c in ctxs[6:]:
        frozen.after_step(np.zeros(4), {**c, 'learn': False})
        feats.append(frozen.features())
    assert frozen.memory.export_state() == memory_before and frozen.telemetry()['frozen_steps'] == 6
    assert not np.allclose(feats[0], feats[1])                                # features still follow the context
    learner = copy.deepcopy(ext)
    for c in ctxs[6:]:
        learner.after_step(np.zeros(4), {**c, 'learn': True})
    assert learner.memory.writes == ext.memory.writes + 6
    assert ext.memory.export_state() == memory_before                         # the original never moved
    assert not np.shares_memory(learner.memory.M, ext.memory.M)
    twin = CM.ContinuumMemoryExtension(seed=2)
    for c in ctxs[:6]:
        twin.after_step(np.zeros(4), {**c, 'learn': True})
    assert twin.export_state() == ext.export_state()                         # same seed and inputs: same state
    other = CM.ContinuumMemoryExtension(seed=3)
    for c in ctxs[:6]:
        other.after_step(np.zeros(4), {**c, 'learn': True})
    assert not np.allclose(other.features(), ext.features())
    ext.reset()
    assert not ext.features().any() and ext.export_state()['prev_key'] is None
    assert ext.memory.export_state() == memory_before                         # reset is an episode boundary only


def test_continuum_extension_state_round_trip_and_rejection():
    ctxs = _contexts(4, 10)
    ext = CM.ContinuumMemoryExtension(seed=5, levels=3, key_dim=4, momentum=0.2)
    for c in ctxs[:7]:
        ext.after_step(np.zeros(4), {**c, 'learn': True})
    state = json.loads(json.dumps(ext.export_state()))
    clone = CM.ContinuumMemoryExtension(seed=5, levels=3, key_dim=4, momentum=0.2)
    clone.import_state(state)
    assert clone.export_state() == ext.export_state()
    for c in ctxs[7:]:
        ext.after_step(np.zeros(4), {**c, 'learn': True})
        clone.after_step(np.zeros(4), {**c, 'learn': True})
    assert np.array_equal(clone.features(), ext.features()) and clone.export_state() == ext.export_state()
    before = clone.export_state()
    bad_key = copy.deepcopy(state)
    bad_key['prev_key'][0][0] = 0.3
    bad_memory = copy.deepcopy(state)
    bad_memory['memory']['writes'] = 1
    for bad in ({**state, 'id': 'other'}, {**state, 'config': {**state['config'], 'seed': 6}}, bad_key, bad_memory,
                {**state, 'features': [11.0] + state['features'][1:]}, {**state, 'features': state['features'][:-1]},
                {**state, 'steps': -1}, {**state, 'frozen_steps': state['steps'] + 1}, {**state, 'memory': None}):
        with pytest.raises(ValueError):
            clone.import_state(bad)
    assert clone.export_state() == before


def test_continuum_extension_validates_context_and_configuration():
    ext = CM.ContinuumMemoryExtension()
    good = {'learn': True, 'sensory': np.zeros(SENSORY_DIM), 'task_input': None}
    for bad in ({**good, 'learn': 1}, {**good, 'sensory': np.zeros(3)}, {**good, 'sensory': np.full(SENSORY_DIM, np.nan)},
                {**good, 'task_input': [np.inf]}, 'context'):
        with pytest.raises(ValueError):
            ext.after_step(np.zeros(4), bad)
    assert ext.export_state()['steps'] == 0
    ext.after_step(np.zeros(4), {**good, 'task_input': np.ones(3)})        # shorter task inputs are zero-padded
    for bad in ({'levels': 0}, {'key_dim': 0}, {'features_per_level': 0}, {'seed': -1}, {'lr': -1.0},
                {'chunk_reduce': 'median'}):
        with pytest.raises(ValueError):
            CM.ContinuumMemoryExtension(**bad)


# ====================================================================== continuum memory: recall benchmark
def test_recall_stream_is_deterministic_with_recurring_items():
    first = CM.make_recall_stream(3, 300, key_dim=16, value_dim=4, short_gap=(3, 10), long_gap=(40, 80))
    again = CM.make_recall_stream(3, 300, key_dim=16, value_dim=4, short_gap=(3, 10), long_gap=(40, 80))
    assert all(np.array_equal(first[k], again[k]) for k in first)
    assert not np.array_equal(CM.make_recall_stream(4, 300, key_dim=16, value_dim=4)['keys'], first['keys'])
    kind, item, gap = first['kind'], first['item'], first['gap']
    assert np.allclose(np.linalg.norm(first['keys'], axis=1), 1.0)
    short, long_ = kind == CM.KIND_CODES['short'], kind == CM.KIND_CODES['long']
    assert short.sum() > 20 and long_.sum() > 10 and (kind == CM.KIND_CODES['first']).sum() == 12
    assert gap[short].min() >= 3 and gap[long_].min() >= 40              # waiting can only lengthen a gap
    assert np.all(item[short] < 4) and np.all(item[long_] >= 4) and np.all(item[kind == 0] == -1)
    for i in range(12):                                                  # an item always carries the same pair
        rows = np.flatnonzero(item == i)
        assert np.allclose(first['keys'][rows], first['keys'][rows[0]]) and np.allclose(first['values'][rows],
                                                                                         first['values'][rows[0]])
    remapped = CM.make_recall_stream(3, 300, key_dim=16, value_dim=4, remap_prob=1.0)
    assert (remapped['kind'] == CM.KIND_CODES['short']).sum() == 0 and (remapped['kind'] == CM.KIND_CODES['remap']).sum() > 0
    for bad in ({'seed': -1}, {'steps': 0}, {'short_gap': (5, 2)}, {'remap_prob': 1.5}, {'n_short': -1}):
        with pytest.raises(ValueError):
            CM.make_recall_stream(**{'seed': 0, 'steps': 10, **bad})


def test_recall_benchmark_quick_receipt_is_sealed_deterministic_and_capacity_matched(quick_receipts):
    first = quick_receipts[0]['continuum_memory_recall']
    second = CM.run_recall_benchmark(quick=True)
    assert verify_receipt(first) and first['digest'] == second['digest']
    assert first['experiment_id'] == 'continuum_memory_recall' and first['kind'] == 'memory_benchmark'
    assert first['evidence_tier'] == EVIDENCE_TIER and first['verdict'] in VERDICTS
    assert first['verdict'] == first['comparisons']['cms_vs_single_long_gap']['verdict']
    cfg = first['config']
    assert cfg['quick'] is True and cfg['seeds'] == [0, 1, 2] and cfg['steps'] == 240 and not set(cfg['tune_seeds']) & {0, 1, 2}
    budget = first['budget']
    assert budget['cms_memory_entries'] == budget['single_memory_entries'] == 8 * 4 * 32
    assert set(first['tuning']) == {'cms', 'single'} and len(first['tuning']['cms']['table']) == 3 * 3 * 2
    for arm in ('cms', 'single'):                     # both arms tuned on the same grid; the optimum is inside it
        tun = first['tuning'][arm]
        assert [(r['lr'], r['momentum'], r['forget']) for r in tun['table']] == [
            (r['lr'], r['momentum'], r['forget']) for r in first['tuning']['cms']['table']]
        assert tun['chosen_on_grid_edge'] == []
    for row in first['per_seed']:
        assert row['cms']['long_queries'] == row['single']['long_queries'] == row['none']['long_queries'] > 0
        assert row['cms']['short_queries'] > 0
    # any associative memory beats the no-memory running mean on recurring associations
    assert first['comparisons']['cms_vs_none_long_gap']['verdict'] == 'improved'
    assert first['comparisons']['single_vs_none_long_gap']['verdict'] == 'improved'
    assert first['measured_wallclock_s']['tuning'] > 0.0 and isinstance(first['notes'], str)
    tampered = copy.deepcopy(first)
    tampered['summary']['cms']['long_mse'] += 1.0
    assert not verify_receipt(tampered)
    assert CM.run_recall_benchmark(quick=True, seeds=(3, 4))['digest'] != first['digest']


def test_grid_edge_flags_mark_a_tuned_optimum_on_the_boundary_of_its_grid():
    grids = {'lr': (0.1, 0.2, 0.4), 'momentum': (0.0, 0.5), 'forget': (0.0,)}
    natural = {'momentum': (0.0,), 'forget': (0.0,)}
    assert CM._grid_edges({'lr': 0.2, 'momentum': 0.0, 'forget': 0.0}, grids, natural) == []
    assert CM._grid_edges({'lr': 0.4, 'momentum': 0.5, 'forget': 0.0}, grids, natural) == ['lr', 'momentum']
    assert CM._grid_edges({'lr': 0.1, 'momentum': 0.0, 'forget': 0.0}, grids) == ['lr', 'momentum']   # 1 value: untuned
    # a grid of tiny learning rates: the larger one wins and the receipt says the optimum may lie beyond the grid
    r = CM.run_recall_benchmark(quick=True, seeds=(0, 1), lr_grid=(0.001, 0.002), momentum_grid=(0.0,),
                                forget_grid=(0.0,))
    assert r['tuning']['cms']['chosen']['lr'] == 0.002 and r['tuning']['cms']['chosen_on_grid_edge'] == ['lr']
    assert verify_receipt(r)


@pytest.mark.parametrize('kwargs', [{'quick': 'yes'}, {'seeds': ()}, {'seeds': (1, 1)}, {'seeds': (1.5,)},
                                    {'surprise': 1}, {'tune_seeds': (0,)}, {'lr_grid': ()}, {'steps': 8},
                                    {'max_forget': 'x'}, {'chunk_reduce': 'max'}, {'levels': 0},
                                    {'momentum_grid': (1.0,)}, {'short_gap': (3,)}])
def test_recall_benchmark_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        CM.run_recall_benchmark(**{'quick': True, **kwargs})


def test_continuum_run_experiment_quick_receipt_is_sealed_and_deterministic(quick_receipts):
    first = quick_receipts[0]['continuum_memory']
    assert verify_receipt(first) and first['verdict'] in VERDICTS and first['evidence_tier'] == EVIDENCE_TIER
    assert first['experiment_id'] == 'continuum_memory' and set(first['arms']) == {'baseline', 'candidate', 'control'}
    cfg = first['config']
    assert cfg['task'] == 'delayed_recall' and cfg['control'] == 'random_trace' and cfg['quick'] is True
    assert cfg['candidate_feature_dim'] == 8 and cfg['extension']['levels'] == 4 and cfg['seeds'] == [0, 1, 2]
    cand = first['arms']['candidate']['per_seed'][0]
    assert cand['extensions'] == ['continuum_memory'] and cand['budget']['extension_features'] == 8
    assert cand['budget']['extension_parameters'] == 4 * SENSORY_DIM * 8
    assert cand['telemetry'][0]['writes'] == 23 and first['arms']['control']['per_seed'][0]['extensions'] == [
        'random_trace_control']
    assert first['digest'] == CM.run_experiment(quick=True)['digest']
    tampered = copy.deepcopy(first)
    tampered['verdict'] = 'worse' if first['verdict'] != 'worse' else 'improved'
    assert not verify_receipt(tampered)
    for bad in ({'quick': 1}, {'surprise': 1}, {'levels': 0}, {'seeds': (0, 0)}, {'heldout_seeds': (0,)}):
        with pytest.raises(ValueError):
            CM.run_experiment(**{'quick': True, **bad})


# ====================================================================== BTSP: metrics and cues
def test_overlap_hamming_and_reconstruction_error_on_hand_cases():
    a = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    b = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
    assert BT.overlap(a, b) == pytest.approx(2.0 / 3.0) and BT.hamming(a, b) == 2
    assert BT.overlap(a, a) == 1.0 and BT.overlap(a, np.zeros(6)) == 0.0 and BT.overlap([1, 0, 1], [1, 1, 1]) == pytest.approx(
        2.0 / np.sqrt(6.0))
    cue = np.array([1, 0, 0, 0, 1, 0], dtype=bool)                      # one active bit deleted
    assert BT.reconstruction_error(a, cue, a) == 0.0 and BT.reconstruction_error(a, cue, cue) == 1.0
    assert BT.reconstruction_error(a, cue, b) == 2.0 and BT.reconstruction_error(a, a, b) is None
    for bad in ((a, a[:5]), (a, np.array([0, 1, 2, 0, 0, 0])), (np.ones((2, 3)), np.ones((2, 3)))):
        with pytest.raises(ValueError):
            BT.overlap(*bad)


def test_cues_delete_or_move_the_requested_number_of_active_bits():
    rng = np.random.default_rng(0)
    pats = BT.random_patterns(5, 100, 20, rng)
    assert pats.shape == (5, 100) and pats.dtype == bool and np.all(pats.sum(axis=1) == 20)
    x = pats[0]
    deleted = BT.make_cue(x, 0.3, 'delete', rng)
    assert deleted.sum() == 14 and not np.any(deleted & ~x)
    flipped = BT.make_cue(x, 0.3, 'flip', rng)
    assert flipped.sum() == 20 and (flipped & x).sum() == 14
    assert np.array_equal(BT.make_cue(x, 0.0, 'flip', rng), x) and not BT.make_cue(x, 1.0, 'delete', rng).any()
    for args in ((x, 0.3, 'swap'), (x, 1.5, 'delete'), (np.array([0, 2, 1]), 0.5, 'delete'),
                 (np.ones(4, dtype=bool), 0.5, 'flip')):
        with pytest.raises(ValueError):
            BT.make_cue(*args, rng)
    for bad_rng in (0, np.random.RandomState(0)):                          # only seeded Generators are accepted
        with pytest.raises(ValueError):
            BT.make_cue(x, 0.3, 'delete', bad_rng)
        with pytest.raises(ValueError):
            BT.random_patterns(2, 10, 3, bad_rng)
    with pytest.raises(ValueError):
        BT.random_patterns(1_000_000, 1_000, 5, rng)                        # bounded allocation


# ====================================================================== BTSP: memory
def test_store_follows_the_plateau_gated_flip_rule_on_a_deterministic_case():
    m, n = 40, 30
    x = np.zeros(m, dtype=bool)
    x[:10] = True
    mem = BT.BTSPMemory(m, n, density=0.6, plateau_prob=1.0, flip_prob=1.0, threshold=0.35, seed=3)
    fwd, fbk = mem.forward_mask, mem.feedback_mask
    info = mem.store(x)
    # every unit gets a plateau and every eligible flip happens: exactly the connected synapses from active inputs
    assert info['plateau_units'] == n and info['ltd_flips'] == 0 and info['ltp_flips'] == int(fwd[:, x].sum())
    expected = np.zeros((n, m), dtype=bool)
    expected[:, x] = fwd[:, x]
    assert np.array_equal(mem.W, expected)
    trace = fwd[:, x].sum(axis=1) > 0.35 * 0.6 * 10                        # McCulloch-Pitts trace z(x)
    assert np.array_equal(mem.code(x), trace) and info['trace_size'] == trace.sum() > 0
    feedback = np.zeros((m, n), dtype=bool)                               # one-shot Hebbian feedback, saturating
    feedback[np.ix_(x, trace)] = fbk[np.ix_(x, trace)]
    assert np.array_equal(mem.B, feedback) and info['new_feedback_synapses'] == feedback.sum()
    recon = mem.reconstruct(trace)
    assert not np.any(recon & ~x)                                         # feedback only reaches stored inputs
    # the same pattern again with certain flips is pure depression (1 -> 0): every forward weight returns to 0
    info2 = mem.store(x)
    assert info2['ltp_flips'] == 0 and info2['ltd_flips'] == info['ltp_flips'] and not mem.W.any()
    assert info2['trace_size'] == 0 and np.array_equal(mem.B, feedback)    # feedback weights never decrease
    assert mem.stores == 2 and mem.plateaus == 2 * n


def test_without_a_plateau_nothing_changes_and_inactive_inputs_never_change():
    rng = np.random.default_rng(1)
    x = BT.random_patterns(1, 50, 10, rng)[0]
    silent = BT.BTSPMemory(50, 40, plateau_prob=0.0, seed=1)
    assert silent.store(x) == {'plateau_units': 0, 'ltp_flips': 0, 'ltd_flips': 0, 'trace_size': 0,
                               'new_feedback_synapses': 0}
    assert not silent.W.any() and not silent.B.any()
    mem = BT.BTSPMemory(50, 40, plateau_prob=0.3, seed=1)
    for p in BT.random_patterns(5, 50, 10, rng):
        mem.store(p)
    before = mem.W.copy()
    mem.store(x)
    changed = mem.W != before
    assert changed.any() and not changed[:, ~x].any()                    # only synapses from active inputs move
    assert mem.W.dtype == bool and mem.B.dtype == bool                     # weights stay binary
    assert not np.any(mem.W & ~mem.forward_mask) and not np.any(mem.B & ~mem.feedback_mask)


def test_flip_probabilities_set_the_potentiation_and_depression_rates():
    m = n = 200
    x = np.ones(m, dtype=bool)
    mem = BT.BTSPMemory(m, n, density=0.5, plateau_prob=1.0, flip_prob=(0.3, 0.8), threshold=10.0, seed=4)
    connected = int(mem.forward_mask.sum())
    first = mem.store(x)
    assert first['ltp_flips'] / connected == pytest.approx(0.3, abs=0.02) and first['trace_size'] == 0
    ones = int(mem.W.sum())
    second = mem.store(x)
    assert second['ltd_flips'] / ones == pytest.approx(0.8, abs=0.02)
    assert second['ltp_flips'] / (connected - ones) == pytest.approx(0.3, abs=0.02)
    no_ltd = BT.BTSPMemory(m, n, density=0.5, plateau_prob=1.0, flip_prob=(0.5, 0.0), threshold=10.0, seed=4)
    no_ltd.store(x)
    kept = no_ltd.W.copy()
    no_ltd.store(x)
    assert np.all(no_ltd.W[kept]) and no_ltd.ltd_flips == 0               # without depression weights only grow


def test_each_btsp_mechanism_changes_what_is_recalled():
    pats = BT.random_patterns(12, 120, 12, np.random.default_rng(11))
    cues = [BT.make_cue(p, 0.4, 'flip', np.random.default_rng(i)) for i, p in enumerate(pats)]

    def outcome(**kw):
        cfg = {'density': 0.6, 'plateau_prob': 0.1, 'flip_prob': 0.5, 'threshold': 0.35, 'feedback_threshold': 0.5,
               'seed': 3, **kw}
        mem = BT.BTSPMemory(120, 100, **cfg)
        for p in pats:
            mem.store(p)
        recalls = [mem.recall(c) for c in cues]
        return np.concatenate([np.concatenate([r['code'], r['reconstruction']]) for r in recalls]), mem.ltd_flips

    base, ltd = outcome()
    assert np.array_equal(base, outcome()[0]) and ltd > 0                   # deterministic; depression does occur
    for toggle in ({'plateau_prob': 0.2}, {'flip_prob': (0.5, 0.0)}, {'flip_prob': (0.8, 0.5)}, {'threshold': 0.5},
                   {'feedback_threshold': 0.7}, {'density': 0.4}, {'seed': 4}):
        assert not np.array_equal(outcome(**toggle)[0], base), toggle


def _recall_profile(seed, *, patterns=10, fractions=(0.0, 0.2, 0.4, 0.6), kind='delete'):
    rng = np.random.default_rng(seed)
    pats = BT.random_patterns(patterns, 200, 20, rng)
    mem = BT.BTSPMemory(200, 200, density=0.6, plateau_prob=0.1, threshold=0.35, feedback_threshold=0.5, seed=seed)
    for p in pats:
        mem.store(p)                                                      # each pattern is presented exactly once
    cue_rng = np.random.default_rng(seed + 50)
    out = {}
    for frac in fractions:
        rows = []
        for p in pats:
            cue = BT.make_cue(p, frac, kind, cue_rng)
            rec = mem.recall(cue)
            rows.append((BT.overlap(p, rec['reconstruction']), BT.overlap(p, cue), BT.overlap(mem.code(p), rec['code'])))
        out[frac] = np.mean(rows, axis=0)
    return out, mem


def test_one_shot_storage_recalls_full_cues_and_degrades_gracefully():
    profiles = [_recall_profile(seed)[0] for seed in (0, 1, 2)]
    mean = {f: np.mean([p[f] for p in profiles], axis=0) for f in profiles[0]}
    recon = [mean[f][0] for f in (0.0, 0.2, 0.4, 0.6)]
    traces = [mean[f][2] for f in (0.0, 0.2, 0.4, 0.6)]
    assert recon[0] > 0.95 and traces[0] == 1.0                           # full cue: the stored pattern comes back
    assert all(mean[f][0] > mean[f][1] for f in (0.2, 0.4, 0.6))          # completion beats the degraded cue
    assert all(a >= b - 0.01 for a, b in zip(recon, recon[1:])) and recon[-1] > 0.8
    assert all(a > b for a, b in zip(traces, traces[1:])) and traces[-1] > 0.6
    flip = [_recall_profile(seed, fractions=(0.0, 0.3), kind='flip')[0] for seed in (0, 1, 2)]
    assert np.mean([p[0.3][0] for p in flip]) > np.mean([p[0.3][1] for p in flip])   # wrong bits are cleaned up too
    _, mem = _recall_profile(0)
    assert mem.W.dtype == bool and mem.B.dtype == bool and 0.0 < mem.weight_fractions()['forward'] < 0.5


def test_more_stored_patterns_reduce_recall_and_unstored_cues_find_little():
    few, mem = _recall_profile(0, patterns=10, fractions=(0.4,))
    many, _ = _recall_profile(0, patterns=40, fractions=(0.4,))
    assert many[0.4][0] < few[0.4][0] - 0.1                                # a palimpsest: interference grows with load
    novel = BT.random_patterns(10, 200, 20, np.random.default_rng(999))
    stored = BT.random_patterns(10, 200, 20, np.random.default_rng(0))
    recalls = [mem.recall(p)['reconstruction'] for p in novel]
    best = [max(BT.overlap(r, s) for s in stored) for r in recalls]
    assert np.mean([BT.overlap(p, r) for p, r in zip(novel, recalls)]) < 0.5 and np.mean(best) < 0.5


def test_btsp_memory_is_deterministic_and_seeded():
    pats = BT.random_patterns(6, 60, 8, np.random.default_rng(2))

    def build(seed):
        mem = BT.BTSPMemory(60, 50, plateau_prob=0.2, seed=seed)
        for p in pats:
            mem.store(p)
        return mem

    a, b, c = build(5), build(5), build(6)
    assert np.array_equal(a.W, b.W) and np.array_equal(a.B, b.B) and a.export_state() == b.export_state()
    assert not np.array_equal(a.forward_mask, c.forward_mask)
    assert a.synapse_count == int(a.forward_mask.sum() + a.feedback_mask.sum())
    a.reset()
    assert not a.W.any() and not a.B.any() and a.stores == 0 and np.array_equal(a.forward_mask, b.forward_mask)
    for p in pats:
        a.store(p)
    assert np.array_equal(a.W, b.W)                                       # reset restarts the plasticity stream


def test_btsp_memory_state_round_trip_and_rejection():
    pats = BT.random_patterns(6, 60, 8, np.random.default_rng(7))
    mem = BT.BTSPMemory(60, 50, plateau_prob=0.2, flip_prob=(0.5, 0.4), seed=5)
    for p in pats[:4]:
        mem.store(p)
    state = json.loads(json.dumps(mem.export_state()))
    twin = BT.BTSPMemory(60, 50, plateau_prob=0.2, flip_prob=(0.5, 0.4), seed=5)
    twin.import_state(state)
    assert np.array_equal(twin.W, mem.W) and np.array_equal(twin.B, mem.B)
    for p in pats[4:]:
        assert twin.store(p) == mem.store(p)                              # the plasticity stream resumes exactly
    assert twin.export_state() == mem.export_state()
    before = twin.export_state()
    long_bits = base64.b64encode(base64.b64decode(state['W_bits']) + b'\x00').decode('ascii')
    for bad in ({**state, 'config': {**state['config'], 'seed': 6}}, {**state, 'W_bits': long_bits},
                {**state, 'W_bits': '***'}, {**state, 'B_bits': 7}, {**state, 'stores': -1},
                {**state, 'plateaus': 4 * 50 + 1}, {**state, 'rng': {**state['rng'], 'bit_generator': 'MT19937'}},
                {**state, 'rng': {'bit_generator': 'PCG64', 'state': {'state': 'x'}}}, 'state'):
        with pytest.raises(ValueError):
            twin.import_state(bad)
    assert twin.export_state() == before


@pytest.mark.parametrize('kwargs', [{'input_dim': 0}, {'memory_dim': 2_000_001}, {'density': 0.0},
                                    {'plateau_prob': 1.5}, {'flip_prob': (0.5,)}, {'flip_prob': (0.5, 1.2)},
                                    {'flip_prob': np.array(0.5)}, {'flip_prob': np.full((2, 1), 0.5)},
                                    {'threshold': -0.1}, {'feedback_threshold': 11.0}, {'seed': -1}])
def test_btsp_memory_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        BT.BTSPMemory(**{'input_dim': 20, 'memory_dim': 10, **kwargs})


def test_btsp_memory_rejects_bad_patterns_without_changing_state():
    mem = BT.BTSPMemory(20, 10, plateau_prob=0.5, seed=0)
    before = mem.export_state()
    for bad in (np.ones(19), np.full(20, 0.5), np.full(20, 2), np.full(20, np.nan), ['a'] * 20, np.ones((20, 1))):
        with pytest.raises(ValueError):
            mem.store(bad)
    with pytest.raises(ValueError):
        mem.store(np.ones(20), plateau_prob=2.0)
    with pytest.raises(ValueError):
        mem.reconstruct(np.ones(11))
    assert mem.export_state() == before


# ====================================================================== baselines
def test_willshaw_baseline_completes_patterns_and_saturates():
    m = 30
    pats = np.zeros((3, m), dtype=bool)
    pats[0, :6], pats[1, 10:16], pats[2, 20:26] = True, True, True
    will = BT.WillshawMemory(m)
    for p in pats:
        will.store(p)
    assert np.array_equal(will.A, will.A.T) and will.A.sum() == 3 * 36 and will.synapse_count == m * m
    for p in pats:
        cue = p.copy()
        cue[np.flatnonzero(p)[:3]] = False                                # half the active bits deleted
        assert np.array_equal(will.recall(cue), p)                        # threshold 1: exact completion
    noisy = pats[0].copy()
    noisy[0], noisy[29] = False, True                                     # one bit missing, one wrong
    assert not will.recall(noisy).any()                                   # threshold 1 needs every cue unit
    tolerant = BT.WillshawMemory(m, threshold=0.8)
    tolerant.import_state({**will.export_state(), 'config': {'input_dim': m, 'threshold': 0.8}})
    assert np.array_equal(tolerant.recall(noisy), pats[0])
    assert not will.recall(np.zeros(m)).any()
    will.store(np.ones(m, dtype=bool))                                    # saturated: every cue recalls everything
    assert will.recall(pats[1]).all()
    state = json.loads(json.dumps(will.export_state()))
    bits = np.zeros((m, m), dtype=bool)
    bits[0, 1] = True
    for bad in ({**state, 'config': {'input_dim': m, 'threshold': 0.5}}, {**state, 'stores': -1},
                {**state, 'A_bits': base64.b64encode(np.packbits(bits.ravel()).tobytes()).decode('ascii')}):
        with pytest.raises(ValueError):
            BT.WillshawMemory(m).import_state(bad)
    with pytest.raises(ValueError):
        BT.WillshawMemory(m, threshold=0.0)


def test_random_recall_control_is_at_chance():
    ctl = BT.RandomRecallControl(200, 20, seed=0)
    x = BT.random_patterns(1, 200, 20, np.random.default_rng(1))[0]
    recalls = [ctl.recall(x) for _ in range(400)]
    assert all(r.sum() == 20 for r in recalls) and ctl.store(x) == {}
    assert np.mean([BT.overlap(x, r) for r in recalls]) == pytest.approx(20 / 200, abs=0.01)
    with pytest.raises(ValueError):
        ctl.recall(np.ones(5))


# ====================================================================== BTSP: engine extension
def test_btsp_extension_binarises_by_top_k():
    ext = BT.BTSPExtension(memory_dim=16, active_fraction=0.25, seed=0)
    with pytest.raises(ValueError):
        ext.binarise(np.zeros(24))                                        # not bound yet
    ext.bind(_engine(neurons=24))
    assert ext.telemetry()['n'] == 24 and ext.telemetry()['k'] == 6
    assert np.array_equal(np.flatnonzero(ext.binarise(np.arange(24.0))), np.arange(18, 24))
    assert np.array_equal(np.flatnonzero(ext.binarise(np.zeros(24))), np.arange(6))       # ties: lower index
    assert np.array_equal(np.flatnonzero(ext.binarise(np.r_[np.zeros(24), np.ones(5)])), np.arange(6))   # truncated
    assert np.array_equal(np.flatnonzero(ext.binarise(-np.ones(10))), np.arange(10, 16))  # zero-padded
    for bad in (np.full(24, np.nan), {}, object(), 1j, ['a'] * 24):
        with pytest.raises(ValueError):
            ext.binarise(bad)


def test_btsp_extension_never_changes_the_engine_and_keeps_no_engine_reference():
    plain, hooked = _engine(seed=2), _engine(seed=2)
    ext = BT.BTSPExtension(seed=1)
    hooked.attach_extension(ext)
    _assert_holds_no_engine(ext)
    a, b = _drive(plain, 10), _drive(hooked, 10)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert plain.export_state() == hooked.export_state()
    tel = ext.telemetry()
    assert tel['steps'] == 10 and tel['frozen_steps'] == 0 and 1 <= tel['stores'] <= 10
    feats = ext.features()
    assert feats.shape == (ext.feature_dim,) == (8,) and np.all(np.isfinite(feats)) and np.abs(feats).max() <= 10.0
    assert 0.0 <= feats[0] <= 1.0 and 0.0 <= feats[1] <= 1.0
    json.dumps(tel)
    _assert_holds_no_engine(ext)
    assert ext.parameter_count() == ext.memory.synapse_count > 0


def test_btsp_extension_stores_novel_states_and_recognises_them():
    ext = BT.BTSPExtension(memory_dim=64, plateau_prob=0.3, seed=1)
    ext.bind(_engine(neurons=24))
    out = np.random.default_rng(0).standard_normal(24)
    ext.after_step(out, {'learn': True, 'reward': 0.0})
    assert ext.features()[0] == 0.0 and ext.telemetry()['last_stored'] is True   # novel: nothing recalled, stored
    ext.after_step(out, {'learn': True, 'reward': 0.0})
    tel = ext.telemetry()
    assert ext.features()[0] > 0.9 and ext.features()[1] > 0.0                   # one exposure makes it familiar
    assert tel['last_stored'] is False and tel['stores'] == 1                     # familiar and unrewarded: no store
    ext.after_step(out, {'learn': True, 'reward': -1.0})
    assert ext.telemetry()['stores'] == 2                                         # a salient outcome stores again
    gated = BT.BTSPExtension(memory_dim=64, plateau_prob=0.3, gate=1.0, seed=1)
    gated.bind(_engine(neurons=24))
    gated.after_step(out, {'learn': True, 'reward': 1.0})
    assert gated.telemetry()['stores'] == 0 and not gated.memory.W.any()          # gate 1 never opens


def test_btsp_extension_is_frozen_without_learning_and_deep_copies_are_independent():
    engine = _engine(seed=3)
    ext = BT.BTSPExtension(seed=2)
    engine.attach_extension(ext)
    outs = _drive(engine, 8)
    memory_before = ext.memory.export_state()
    frozen = copy.deepcopy(ext)
    for o in outs:
        frozen.after_step(o, {'learn': False, 'reward': 1.0})
    assert frozen.memory.export_state() == memory_before and frozen.telemetry()['frozen_steps'] == 8
    learner = copy.deepcopy(ext)
    rng = np.random.default_rng(9)
    for _ in range(4):
        learner.after_step(rng.standard_normal(24), {'learn': True, 'reward': 1.0})
    assert learner.memory.stores > ext.memory.stores and ext.memory.export_state() == memory_before
    assert not np.shares_memory(learner.memory.W, ext.memory.W)
    other = BT.BTSPExtension(seed=4)
    other.bind(_engine(seed=3))
    for o in outs:
        other.after_step(o, {'learn': True, 'reward': 0.0})
    assert not np.array_equal(other.memory.forward_mask, ext.memory.forward_mask)
    ext.reset()
    assert not ext.features().any() and ext.memory.export_state() == memory_before   # memories survive reset


def test_btsp_extension_is_deterministic_given_the_same_inputs():
    outs = [np.tanh(np.random.default_rng(i).standard_normal(24)) for i in range(12)]

    def run(seed):
        ext = BT.BTSPExtension(seed=seed)
        ext.bind(_engine(neurons=24))
        for i, o in enumerate(outs):
            ext.after_step(o, {'learn': True, 'reward': 0.5 if i % 4 == 0 else 0.0})
        return ext

    a, b = run(3), run(3)
    assert a.export_state() == b.export_state() and np.array_equal(a.features(), b.features())
    assert run(4).export_state() != a.export_state()


def test_btsp_extension_state_round_trip_and_rejection():
    outs = [np.random.default_rng(i).standard_normal(24) for i in range(10)]
    ext = BT.BTSPExtension(seed=5, plateau_prob=0.2)
    ext.bind(_engine(neurons=24))
    for o in outs[:7]:
        ext.after_step(o, {'learn': True, 'reward': 0.3})
    state = json.loads(json.dumps(ext.export_state()))
    clone = BT.BTSPExtension(seed=5, plateau_prob=0.2)                        # unbound: sized from the state
    clone.import_state(state)
    assert clone.export_state() == ext.export_state()
    for o in outs[7:]:
        ext.after_step(o, {'learn': True, 'reward': 0.3})
        clone.after_step(o, {'learn': True, 'reward': 0.3})
    assert np.array_equal(clone.features(), ext.features()) and clone.export_state() == ext.export_state()
    before = clone.export_state()
    bad_memory = copy.deepcopy(state)
    bad_memory['memory']['stores'] = -1
    for bad in ({**state, 'id': 'other'}, {**state, 'config': {**state['config'], 'gate': 0.4}}, {**state, 'n': 12},
                {**state, 'n': 0}, bad_memory, {**state, 'features': [11.0] * 8}, {**state, 'steps': -1},
                {**state, 'stores': state['steps'] + 1}, {**state, 'last': {**state['last'], 'stored': 1}},
                {**state, 'last': {**state['last'], 'overlap': 1.5}}, {**state, 'memory': None}):
        with pytest.raises(ValueError):
            clone.import_state(bad)
    assert clone.export_state() == before
    unbound = BT.BTSPExtension(seed=5, plateau_prob=0.2)
    blank = unbound.export_state()
    assert blank['n'] == 0 and blank['memory'] is None
    unbound.import_state(json.loads(json.dumps(blank)))
    for bad in ({**blank, 'memory': state['memory']}, {**blank, 'steps': 3},              # an unbound extension
                {**blank, 'features': [0.5] + blank['features'][1:]}):                   # has never stepped
        with pytest.raises(ValueError):
            unbound.import_state(bad)
    assert unbound.export_state() == blank


def test_btsp_extension_validates_context_and_configuration():
    ext = BT.BTSPExtension(memory_dim=16)
    ext.bind(_engine(neurons=24))
    for bad in ({'learn': 'yes'}, {'learn': True, 'reward': float('nan')}, {'learn': True, 'reward': True},
                {'learn': True, 'reward': '1'}, 'context'):
        with pytest.raises(ValueError):
            ext.after_step(np.zeros(24), bad)
    assert ext.telemetry()['steps'] == 0
    for bad in ({'memory_dim': 0}, {'active_fraction': 0.0}, {'density': 1.5}, {'gate': 1.5}, {'projection_dim': 65},
                {'flip_prob': (0.1, 0.2, 0.3)}, {'seed': -1}):
        with pytest.raises(ValueError):
            BT.BTSPExtension(**bad)
    with pytest.raises(ValueError):
        BT.BTSPExtension(memory_dim=4096).bind(SimpleNamespace(n=2000))          # 2000 x 4096 synapses: too large
    for n in (None, 24.0, float('inf'), True, '24', 0):
        with pytest.raises(ValueError):
            BT.BTSPExtension(memory_dim=16).bind(SimpleNamespace(n=n))
    BT.BTSPExtension(memory_dim=16).bind(SimpleNamespace(n=np.int64(24)))     # NumPy integers are fine


# ====================================================================== BTSP: benchmarks
def test_capacity_benchmark_quick_receipt_is_sealed_deterministic_and_synapse_matched(quick_receipts):
    first = quick_receipts[0]['btsp_memory_capacity']
    second = BT.run_capacity_benchmark(quick=True)
    assert verify_receipt(first) and first['digest'] == second['digest']
    tuning = first['tuning']
    assert len(tuning['btsp']['table']) == 3 * 4 * 3 and len(tuning['willshaw']['table']) == 4
    assert tuning['btsp']['chosen_on_grid_edge'] == [] and tuning['willshaw']['chosen_on_grid_edge'] == []
    assert first['experiment_id'] == 'btsp_memory_capacity' and first['kind'] == 'memory_benchmark'
    assert first['evidence_tier'] == EVIDENCE_TIER and first['verdict'] in VERDICTS
    assert first['verdict'] == first['comparisons']['btsp_vs_willshaw']['verdict']
    cfg = first['config']
    assert cfg['quick'] is True and cfg['input_dim'] == 400 and cfg['memory_dim'] == 333 and cfg['loads'] == [30, 60, 90, 120]
    assert cfg['conditions'] == ['delete_0.3', 'delete_0.5', 'flip_0.3', 'flip_0.5']
    budget = first['budget']
    assert budget['willshaw_synapses'] == 400 ** 2 and abs(budget['btsp_to_willshaw_synapse_ratio'] - 1.0) < 0.02
    summary = first['summary']
    assert summary['random_recall']['curve_mean_overlap'] == pytest.approx(20 / 400, abs=0.01)
    assert len(set(summary['cue_only']['overlap_by_load'])) == 1                  # the cue does not depend on load
    curve = summary['btsp']['overlap_by_load']
    assert all(a > b for a, b in zip(curve, curve[1:]))                         # interference grows with load
    assert first['comparisons']['btsp_vs_random_recall']['verdict'] == 'improved'
    for row in first['per_seed']:
        fractions = [d['weight_fractions']['forward'] for d in row['btsp_diagnostics']]
        assert all(0.0 < a < b < 0.5 for a, b in zip(fractions, fractions[1:]))
    assert first['measured_wallclock_s']['evaluation'] > 0.0 and isinstance(first['notes'], str)
    tampered = copy.deepcopy(first)
    tampered['summary']['btsp']['curve_mean_overlap'] = 1.0
    assert not verify_receipt(tampered)
    assert BT.run_capacity_benchmark(quick=True, seeds=(3, 4))['digest'] != first['digest']


@pytest.mark.parametrize('kwargs', [{'quick': 'yes'}, {'seeds': ()}, {'seeds': (2, 2)}, {'surprise': 1},
                                    {'active': 0}, {'input_dim': 4}, {'cue_kinds': ('swap',)}, {'cue_kinds': None},
                                    {'cue_kinds': ('flip', 'flip')}, {'density': 0.0},
                                    {'tune_seeds': (1,)}, {'willshaw_grid': (0.0,)}, {'plateau_grid': ()},
                                    {'checkpoints': 0}])
def test_capacity_benchmark_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        BT.run_capacity_benchmark(**{'quick': True, **kwargs})


def test_btsp_run_experiment_quick_receipt_is_sealed_and_deterministic(quick_receipts):
    first = quick_receipts[0]['btsp_memory']
    assert verify_receipt(first) and first['verdict'] in VERDICTS and first['evidence_tier'] == EVIDENCE_TIER
    assert first['experiment_id'] == 'btsp_memory' and set(first['arms']) == {'baseline', 'candidate', 'control'}
    cfg = first['config']
    assert cfg['task'] == 'delayed_recall' and cfg['control'] == 'random_trace' and cfg['quick'] is True
    assert cfg['candidate_feature_dim'] == 8 and cfg['extension']['memory_dim'] == 64 and cfg['seeds'] == [0, 1, 2]
    cand = first['arms']['candidate']['per_seed'][0]
    assert cand['extensions'] == ['btsp_memory'] and cand['budget']['extension_features'] == 8
    assert cand['budget']['extension_parameters'] == cand['telemetry'][0]['synapses'] > 0
    assert cand['telemetry'][0]['stores'] >= 1 and cand['telemetry'][0]['frozen_steps'] == 0
    assert first['digest'] == BT.run_experiment(quick=True)['digest']
    tampered = copy.deepcopy(first)
    tampered['arms']['candidate']['summary']['heldout_accuracy'] = 1.0
    assert not verify_receipt(tampered)
    for bad in ({'quick': 1}, {'surprise': 1}, {'memory_dim': 0}, {'seeds': (0, 0)}, {'heldout_seeds': (0,)}):
        with pytest.raises(ValueError):
            BT.run_experiment(**{'quick': True, **bad})


def test_extension_parameters_change_the_features():
    ctxs = _contexts(5, 14)

    def cms(**kw):
        ext = CM.ContinuumMemoryExtension(**{'seed': 1, **kw})
        feats = []
        for c in ctxs:
            ext.after_step(np.zeros(4), {**c, 'learn': True})
            feats.append(ext.features())
        return np.concatenate(feats)

    base = cms()
    assert np.array_equal(base, cms()) and np.abs(base).max() > 0.0
    for toggle in ({'lr': 0.1}, {'momentum': 0.5}, {'forget': 0.2}, {'surprise_forget': 0.5}, {'base_period': 2},
                   {'chunk_reduce': 'mean'}, {'key_dim': 4}, {'levels': 3}, {'features_per_level': 3}, {'seed': 2}):
        other = cms(**toggle)
        assert other.shape != base.shape or not np.allclose(other, base), toggle
    outs = [np.random.default_rng(i).standard_normal(24) for i in range(10)]

    def btsp(**kw):
        ext = BT.BTSPExtension(**{'seed': 1, **kw})
        ext.bind(SimpleNamespace(n=24))                                    # bind only reads the engine size
        feats = []
        for i, o in enumerate(outs):
            ext.after_step(o, {'learn': True, 'reward': 0.5 if i % 3 == 0 else 0.0})
            feats.append(ext.features())
        return np.concatenate(feats), ext.telemetry()['stores']

    base, stores = btsp()
    assert np.array_equal(base, btsp()[0]) and 1 <= stores < 10
    for toggle in ({'gate': 0.95}, {'active_fraction': 0.4}, {'plateau_prob': 0.3}, {'flip_prob': (1.0, 0.0)},
                   {'threshold': 0.6}, {'feedback_threshold': 0.8}, {'density': 0.3}, {'memory_dim': 32},
                   {'projection_dim': 3}, {'seed': 2}):
        other = btsp(**toggle)[0]
        assert other.shape != base.shape or not np.allclose(other, base), toggle
    assert btsp(gate=0.95)[1] < stores                                    # a higher gate stores less often


def test_core_operations_leave_global_random_and_thread_state_alone():
    state = np.random.get_state()
    torch = sys.modules.get('torch')
    torch_before = (torch.get_rng_state().clone(), torch.get_num_threads()) if torch is not None else None
    mem = CM.ContinuumMemory(4, 2, levels=3)
    ext = CM.ContinuumMemoryExtension(seed=1)
    for c in _contexts(0, 5):
        mem.write(np.ones(4), np.ones(2))
        ext.after_step(np.zeros(4), {**c, 'learn': True})
    CM.make_recall_stream(0, 50)
    pats = BT.random_patterns(3, 40, 5, np.random.default_rng(0))
    bt = BT.BTSPMemory(40, 30, plateau_prob=0.3, seed=0)
    for p in pats:
        bt.store(p)
        bt.recall(BT.make_cue(p, 0.4, 'flip', np.random.default_rng(1)))
    bext = BT.BTSPExtension(seed=0)
    bext.bind(_engine(neurons=24))
    bext.after_step(np.ones(24), {'learn': True, 'reward': 1.0})
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    if torch is not None:
        assert torch.equal(torch.get_rng_state(), torch_before[0]) and torch.get_num_threads() == torch_before[1]
