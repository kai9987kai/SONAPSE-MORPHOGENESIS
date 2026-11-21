"""NEUROWEAVE associative trace memory and universal fast weights (engine-level; tiny, fast).

Covers the two mechanisms that used to be declared only: exact no-op without the
flags, known-answer recall/association, causality of each flag, learn=False
freezing, bounds, determinism and validated persistence (including states saved
before the mechanisms existed).
"""
import copy
import hashlib
import json

import numpy as np
import pytest

from synapse.frontier.extension import EngineExtension
from synapse.neuroweave import experiments as ex
from synapse.neuroweave.brain import CONSUMED_FLAGS, NeuroWeaveBrain
from synapse.neuroweave.engine import (ASSOC_CAPACITY, ASSOC_COSINE_EPS, ASSOC_GAIN, ASSOC_KEY_DIM,
                                       ASSOC_REWARD_THRESHOLD, ASSOC_STORE_EVERY, ASSOC_THRESHOLD, ENGINE_FLAGS,
                                       FAST_DECAY, FAST_DIM, FAST_ETA, FAST_READ_GAIN, AssociativeTraceMemory,
                                       FastWeightMemory, NeuroWeaveEngine)
from synapse.neuroweave.spec import MECHANISM_IDS, NeuroWeaveSpecV1
from synapse.neuroweave.standalone import standalone_capabilities

NEW = ('associative_memory', 'fast_weights')
OTHER = ('multi_soma', 'membrane_dynamics', 'synaptic_delays', 'reward_plasticity', 'neuromodulator_field',
         'multimodal_sensory', 'memory_ecology', 'predictive_ensemble', 'micro_collective')


def _spec(mechs=(), **kw):
    base = dict(neurons=24, seed=5, sleep_interval=6, structural_interval=4, max_neurons=32)
    base.update(kw)
    return NeuroWeaveSpecV1(mechanisms=tuple(m for m in MECHANISM_IDS if m in set(mechs)), **base)


def _x(t):
    return np.sin(np.arange(8) * 0.7 + 0.4 * t)


def _s(t):
    return 0.5 * np.cos(np.arange(16) * 0.3 + 0.5 * t)


def _reward(t):
    return 0.8 if t % 4 == 0 else -0.3          # > 0.35 on every 12th step, so episodes are stored


def _run(engine, steps, *, start=0, learn=True, reward=_reward):
    outs, results = [], []
    for t in range(start, start + steps):
        r = engine.step(_x(t), _s(t), reward=reward(t), learn=learn)
        outs.append(r['output'].copy())
        results.append(r)
    return outs, results


def _record_brain_calls(engine):
    calls = []
    original = engine.brain.step

    def wrapped(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    engine.brain.step = wrapped
    return calls


def _digest(engine):
    return hashlib.sha256(json.dumps(engine.export_state(), sort_keys=True).encode()).hexdigest()


def _unit(i, d=ASSOC_KEY_DIM):
    v = np.zeros(d)
    v[i] = 1.0
    return v


# ------------------------------------------------------------------ registry / standalone status
def test_flags_are_engine_consumed_and_standalone_says_why_they_are_not_executed():
    assert set(NEW) <= set(ENGINE_FLAGS) <= set(MECHANISM_IDS)
    # with these two, every declared mechanism id has a runtime consumer (none is declared only)
    assert set(CONSUMED_FLAGS) | set(ENGINE_FLAGS) == set(MECHANISM_IDS)
    caps = standalone_capabilities(NEW, has_micro=False)
    assert caps['executed'] == [] and caps['partial'] == []
    reasons = {x['id']: x['reason'] for x in caps['not_executed']}
    assert set(reasons) == set(NEW)
    for reason in reasons.values():
        assert 'engine-level' in reason and 'not consumed' not in reason


def test_constants_pin_the_reference_rule():
    # the values the documentation states (reference: the SONAPSE-EVO port of the LAB switches); the
    # known-answer tests below use the names, so a changed constant must fail here
    assert (ASSOC_KEY_DIM, ASSOC_CAPACITY, ASSOC_STORE_EVERY) == (32, 32, 12)
    assert (ASSOC_THRESHOLD, ASSOC_GAIN, ASSOC_REWARD_THRESHOLD, ASSOC_COSINE_EPS) == (0.35, 0.11, 0.35, 1e-9)
    assert (FAST_DIM, FAST_READ_GAIN, FAST_ETA, FAST_DECAY) == (8, 0.03, 0.0012, 0.999)


# ------------------------------------------------------------------ exact no-op without the flags
def test_neither_flag_keeps_the_previous_code_path():
    e = NeuroWeaveEngine(_spec(OTHER))
    calls = _record_brain_calls(e)
    _, results = _run(e, 14)
    assert e.associative is None and e.fast_weights is None
    assert calls and all('unit_input' not in c for c in calls)
    assert all('associative_memory' not in r and 'fast_weights' not in r for r in results)
    state = e.export_state()
    assert 'associative_memory' not in state and 'fast_weights' not in state
    assert 'associative_memory' not in e.observatory()['learning']


@pytest.mark.parametrize('learn,reward', [(True, lambda t: 0.0), (False, _reward)])
def test_empty_store_and_zero_matrix_contribute_exactly_nothing(learn, reward):
    # no rewarded learning step -> nothing is ever written -> bit-identical to the engine without the flags
    a, b = NeuroWeaveEngine(_spec(OTHER)), NeuroWeaveEngine(_spec(OTHER + NEW))
    calls = _record_brain_calls(b)
    oa, _ = _run(a, 26, learn=learn, reward=reward)
    ob, rb = _run(b, 26, learn=learn, reward=reward)
    assert all(np.array_equal(x, y) for x, y in zip(oa, ob))
    assert all('unit_input' not in c for c in calls)
    assert b.associative.stores == 0 and not np.any(b.fast_weights.matrix) and b.fast_weights.updates == 0
    assert all(not r['associative_memory']['hit'] and not r['fast_weights']['updated'] for r in rb)


# ------------------------------------------------------------------ each flag alone acts (causally)
@pytest.mark.parametrize('flag', NEW)
def test_each_flag_alone_changes_outputs_from_the_step_after_the_first_write(flag):
    base, on = NeuroWeaveEngine(_spec()), NeuroWeaveEngine(_spec((flag,)))
    ob, _ = _run(base, 16)
    oo, res = _run(on, 16)
    # step 0 writes (reward 0.8 on step 0, a multiple of 12); its read can only act from step 1
    assert np.array_equal(ob[0], oo[0])
    assert not np.array_equal(ob[1], oo[1])
    assert sum(not np.array_equal(x, y) for x, y in zip(ob, oo)) == 15
    if flag == 'associative_memory':
        assert res[0]['associative_memory']['stored'] and res[1]['associative_memory']['hit']
        assert [r['associative_memory']['stored'] for r in res].count(True) == 2      # steps 0 and 12
    else:
        assert res[0]['fast_weights']['updated'] and res[1]['fast_weights']['read_norm'] > 0.0
        assert on.fast_weights.updates == 16


def test_the_two_flags_together_differ_from_each_alone():
    outs = {}
    for name, mechs in (('assoc', NEW[:1]), ('fast', NEW[1:]), ('both', NEW)):
        outs[name], _ = _run(NeuroWeaveEngine(_spec(mechs)), 14)
    assert not np.array_equal(outs['both'][-1], outs['assoc'][-1])
    assert not np.array_equal(outs['both'][-1], outs['fast'][-1])


# ------------------------------------------------------------------ associative memory: known answers
def test_associative_recall_picks_best_cosine_match_and_scales_the_stored_state():
    mem = AssociativeTraceMemory()
    states = [np.linspace(-1.0, 1.0, 6), np.full(6, 0.5), -np.ones(6)]
    keys = [_unit(0), _unit(1), (_unit(0) + _unit(1)) / np.sqrt(2.0)]
    for i, (k, s) in enumerate(zip(keys, states)):
        assert mem.maybe_store(k, s, 0.9, ASSOC_STORE_EVERY * i)
    query = _unit(0) + 0.1 * _unit(1)
    read, info = mem.recall(query, 6)
    # cos(query, e0) = 1 / sqrt(1.01) is the largest of the three (the reference cosine adds eps = 1e-9)
    sim = 1.0 / (np.sqrt(1.01) + ASSOC_COSINE_EPS)
    assert info['hit'] and info['episode'] == 0 and info['similarity'] == pytest.approx(sim, abs=1e-15)
    np.testing.assert_allclose(read, ASSOC_GAIN * sim * states[0], rtol=0, atol=1e-12)
    # population changed since storing: zero-padded / truncated state
    np.testing.assert_allclose(mem.recall(query, 9)[0], ASSOC_GAIN * sim * np.r_[states[0], 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(mem.recall(query, 4)[0], ASSOC_GAIN * sim * states[0][:4], atol=1e-12)
    assert mem.hits == 3
    # orthogonal or anti-aligned keys and sub-threshold matches contribute nothing (and are not hits)
    for q in (_unit(5), -_unit(0), 0.30 * _unit(0) + np.sqrt(1 - 0.30 ** 2) * _unit(7)):
        read, info = mem.recall(q, 6)
        assert read is None and not info['hit']
    above = 0.40 * _unit(0) + np.sqrt(1 - 0.40 ** 2) * _unit(7)
    assert mem.recall(above, 6)[1]['similarity'] == pytest.approx(0.40 / (1 + ASSOC_COSINE_EPS), abs=1e-15)
    assert 0.40 / (1 + ASSOC_COSINE_EPS) > ASSOC_THRESHOLD and mem.hits == 4
    # ties resolve to the earliest episode
    tie = AssociativeTraceMemory()
    tie.maybe_store(_unit(2), np.ones(3), 0.9, 0)
    tie.maybe_store(_unit(2), -np.ones(3), 0.9, ASSOC_STORE_EVERY)
    np.testing.assert_allclose(tie.recall(_unit(2), 3)[0], ASSOC_GAIN * np.ones(3) / (1 + ASSOC_COSINE_EPS),
                               rtol=0, atol=1e-15)
    # a zero key (no drive) matches nothing
    assert tie.recall(np.zeros(ASSOC_KEY_DIM), 3)[0] is None


def test_associative_store_gating_fifo_capacity_and_clipping():
    mem = AssociativeTraceMemory()
    k = _unit(0)
    assert not mem.maybe_store(k, np.ones(4), ASSOC_REWARD_THRESHOLD, 0)        # reward must exceed 0.35
    assert not mem.maybe_store(k, np.ones(4), 0.9, ASSOC_STORE_EVERY + 1)        # only every 12th step
    assert mem.stores == 0 and mem.episodes == []
    for i in range(ASSOC_CAPACITY + 8):
        assert mem.maybe_store(k, 3.0 * np.ones(4), 0.9, ASSOC_STORE_EVERY * i)
    assert len(mem.episodes) == ASSOC_CAPACITY and mem.stores == ASSOC_CAPACITY + 8
    assert [ep['step'] for ep in mem.episodes] == [ASSOC_STORE_EVERY * i for i in range(8, ASSOC_CAPACITY + 8)]
    assert all(np.array_equal(ep['state'], np.ones(4)) for ep in mem.episodes)          # clipped to [-1, 1]


def test_engine_keys_are_the_drive_before_any_extension_and_the_first_write_is_exact():
    class _Shift(EngineExtension):
        id = 'shift'

        def transform_drive(self, external, ctx):
            return None if external is None else external + 0.25

    e = NeuroWeaveEngine(_spec(NEW))
    e.attach_extension(_Shift())
    _run(e, 1)                                      # step 0, reward 0.8: one store and one write
    key = e.associative.episodes[0]['key']
    assert key.shape == (ASSOC_KEY_DIM,)
    np.testing.assert_array_equal(key, np.r_[np.clip(_x(0), -1, 1), np.zeros(ASSOC_KEY_DIM - 8)])
    np.testing.assert_array_equal(e.associative.episodes[0]['state'], e.brain.cells.output)
    # fast-weight write: key = first 8 drive values, value = first 8 pooled readout features after the step
    v = np.clip(e.brain.readout_features()[:FAST_DIM], -1, 1)
    np.testing.assert_allclose(e.fast_weights.matrix, FAST_ETA * 0.8 * 0.8 * np.outer(v, np.clip(_x(0), -1, 1)),
                               rtol=0, atol=1e-18)


def test_engine_step_feeds_exactly_recall_plus_fast_read_into_the_brain():
    e = NeuroWeaveEngine(_spec(NEW))
    calls = _record_brain_calls(e)
    _run(e, 1)                                      # step 0: one store and one write, nothing to read yet
    assert 'unit_input' not in calls[0]
    ep, m = e.associative.episodes[0], e.fast_weights.matrix.copy()
    _run(e, 1, start=1)                             # step 1 reads both
    k = np.r_[np.clip(_x(1), -1, 1), np.zeros(ASSOC_KEY_DIM - 8)]
    sim = float(k @ ep['key']) / (np.linalg.norm(k) * np.linalg.norm(ep['key']) + ASSOC_COSINE_EPS)
    assert sim > ASSOC_THRESHOLD and ep['state'].size == e.brain.n
    expected = ASSOC_GAIN * sim * ep['state'] + FAST_READ_GAIN * (m @ k[:FAST_DIM])[np.arange(e.brain.n) % FAST_DIM]
    np.testing.assert_allclose(calls[1]['unit_input'], expected, rtol=0, atol=1e-15)


# ------------------------------------------------------------------ fast weights: known answers
def test_fast_weight_write_and_read_follow_the_rule_exactly():
    fw = FastWeightMemory()
    rng = np.random.default_rng(3)
    v, k, k2 = rng.uniform(-1, 1, FAST_DIM), rng.uniform(-1, 1, FAST_DIM), rng.uniform(-1, 1, FAST_DIM)
    assert fw.read(k, 19)[0] is None                              # a zero matrix contributes nothing
    change = fw.write(v, k, 0.5)
    expected = np.clip(FAST_DECAY * np.zeros((FAST_DIM, FAST_DIM)) + FAST_ETA * 0.5 * 0.5 * np.outer(v, k), -1, 1)
    np.testing.assert_array_equal(fw.matrix, expected)
    assert change == pytest.approx(np.abs(expected).max(), abs=0) and fw.updates == 1
    read, f = fw.read(k2, 19)
    np.testing.assert_allclose(f, expected @ k2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(read, FAST_READ_GAIN * f[np.arange(19) % FAST_DIM], rtol=0, atol=1e-15)
    # a zero key only decays the matrix
    m = fw.matrix.copy()
    fw.write(v, np.zeros(FAST_DIM), -1.0)
    np.testing.assert_allclose(fw.matrix, FAST_DECAY * m, rtol=0, atol=1e-18)


def test_fast_weights_store_reward_signed_value_key_associations():
    fw = FastWeightMemory()
    e = np.eye(FAST_DIM)
    v1, v2 = np.linspace(-1, 1, FAST_DIM), np.cos(np.arange(FAST_DIM))
    fw.write(v1, e[0], 1.0)          # rewarded pair
    fw.write(v2, e[1], -1.0)         # punished pair
    _, f1 = fw.read(e[0], FAST_DIM)
    _, f2 = fw.read(e[1], FAST_DIM)
    _, f3 = fw.read(e[2], FAST_DIM)
    np.testing.assert_allclose(f1, FAST_DECAY * FAST_ETA * v1, rtol=0, atol=1e-15)   # recalls v1 (scaled)
    np.testing.assert_allclose(f2, -FAST_ETA * v2, rtol=0, atol=1e-15)               # recalls -v2
    assert not np.any(f3)                                                            # unseen key: nothing


def test_fast_weights_saturate_inside_the_bounds():
    fw = FastWeightMemory()
    ones = np.ones(FAST_DIM)
    for _ in range(3000):
        fw.write(ones, ones, 1.0)
    assert np.abs(fw.matrix).max() == 1.0
    read, _ = fw.read(ones, 16)
    assert np.abs(read).max() <= FAST_READ_GAIN * FAST_DIM + 1e-12


# ------------------------------------------------------------------ learning only when learn=True
def test_frozen_steps_read_but_never_write():
    e = NeuroWeaveEngine(_spec(NEW))
    _run(e, 13)                                     # stores at steps 0 and 12, writes on every step
    stored = e.export_state()
    assert len(stored['associative_memory']['episodes']) == 2 and stored['fast_weights']['updates'] == 13
    blank_state = copy.deepcopy(stored)
    del blank_state['associative_memory'], blank_state['fast_weights']
    blank = NeuroWeaveEngine(_spec(NEW))
    blank.import_state(blank_state)                 # same engine, memories at their initial state
    frozen, res = _run(e, 12, start=13, learn=False, reward=lambda t: 0.9)
    fresh, _ = _run(blank, 12, start=13, learn=False, reward=lambda t: 0.9)
    after = e.export_state()
    assert after['associative_memory']['episodes'] == stored['associative_memory']['episodes']
    assert after['fast_weights'] == stored['fast_weights']
    assert any(r['associative_memory']['hit'] for r in res)
    assert not np.array_equal(frozen[0], fresh[0])  # the learned memories are still read while frozen


# ------------------------------------------------------------------ bounds under hostile inputs
def test_contributions_and_state_stay_bounded():
    e = NeuroWeaveEngine(_spec(NEW + ('neurogenesis', 'multimodal_sensory')))
    calls = _record_brain_calls(e)
    rng = np.random.default_rng(4)
    for t in range(60):
        r = e.step(50.0 * rng.normal(size=8), 5.0 * rng.normal(size=16),
                   reward=float(5.0 * np.sign(rng.normal())) if t % 12 else 5.0)
        assert np.all(np.isfinite(r['output']))
    inputs = [c['unit_input'] for c in calls if 'unit_input' in c]
    assert inputs and max(float(np.abs(u).max()) for u in inputs) <= ASSOC_GAIN + FAST_READ_GAIN * FAST_DIM + 1e-12
    assert np.abs(e.fast_weights.matrix).max() <= 1.0
    assert 0 < len(e.associative.episodes) <= ASSOC_CAPACITY
    for ep in e.associative.episodes:
        assert np.abs(ep['key']).max() <= 1.0 and np.abs(ep['state']).max() <= 1.0
        assert ep['state'].size <= e.spec.max_neurons and -1.0 <= ep['reward'] <= 1.0


# ------------------------------------------------------------------ determinism and persistence
def test_determinism_round_trip_and_no_global_rng_use():
    global_state = np.random.get_state()
    a, b = NeuroWeaveEngine(_spec(OTHER + NEW)), NeuroWeaveEngine(_spec(OTHER + NEW))
    oa, _ = _run(a, 15)
    ob, _ = _run(b, 15)
    assert all(np.array_equal(x, y) for x, y in zip(oa, ob)) and _digest(a) == _digest(b)
    c = NeuroWeaveEngine(_spec(OTHER + NEW))
    c.import_state(json.loads(json.dumps(a.export_state())))
    assert _digest(c) == _digest(a)
    ra, _ = _run(a, 10, start=15)
    rc, _ = _run(c, 10, start=15)
    assert all(np.array_equal(x, y) for x, y in zip(ra, rc)) and _digest(c) == _digest(a)
    ob_obs = a.observatory()['learning']
    assert ob_obs['associative_memory']['stores'] == 3 and ob_obs['fast_weights']['updates'] == 25
    gs = np.random.get_state()
    assert gs[0] == global_state[0] and np.array_equal(gs[1], global_state[1]) and gs[2:] == global_state[2:]


def test_states_saved_before_the_mechanisms_existed_import_with_fresh_memories():
    e = NeuroWeaveEngine(_spec(NEW))
    _run(e, 13)
    old = e.export_state()
    del old['associative_memory']
    old['fast_weights'] = None                      # absent or None: both mean "not saved"
    fresh = NeuroWeaveEngine(_spec(NEW))
    fresh.import_state(old)
    assert fresh.associative.episodes == [] and fresh.associative.stores == 0
    assert not np.any(fresh.fast_weights.matrix) and fresh.fast_weights.updates == 0
    assert fresh.step_count == 13


def _bad_states(state):
    """(description, mutated state) pairs that import_state must reject."""
    def mut(fn):
        s = copy.deepcopy(state)
        fn(s)
        return s
    a, f = 'associative_memory', 'fast_weights'
    ep0 = state[a]['episodes'][0]
    return [
        ('assoc extra key', mut(lambda s: s[a].update(extra=1))),
        ('assoc missing key', mut(lambda s: s[a].pop('hits'))),
        ('assoc hits bool', mut(lambda s: s[a].update(hits=True))),
        ('assoc stores negative', mut(lambda s: s[a].update(stores=-1))),
        ('assoc more episodes than stores', mut(lambda s: s[a].update(stores=0))),
        ('assoc over capacity', mut(lambda s: s[a].update(episodes=[copy.deepcopy(ep0)] * (ASSOC_CAPACITY + 1),
                                                          stores=10 ** 6))),
        ('assoc episodes not a list', mut(lambda s: s[a].update(episodes={'0': ep0}))),
        ('assoc key too short', mut(lambda s: s[a]['episodes'][0].update(key=[0.1] * 31))),
        ('assoc key out of range', mut(lambda s: s[a]['episodes'][0]['key'].__setitem__(0, 1.5))),
        ('assoc key non-finite', mut(lambda s: s[a]['episodes'][0]['key'].__setitem__(0, float('nan')))),
        ('assoc state empty', mut(lambda s: s[a]['episodes'][0].update(state=[]))),
        ('assoc state too long', mut(lambda s: s[a]['episodes'][0].update(state=[0.0] * (s['spec']['max_neurons'] + 1)))),
        ('assoc state string', mut(lambda s: s[a]['episodes'][0].update(state='abc'))),
        ('assoc reward out of range', mut(lambda s: s[a]['episodes'][0].update(reward=1.5))),
        ('assoc reward never stored', mut(lambda s: s[a]['episodes'][0].update(reward=ASSOC_REWARD_THRESHOLD))),
        ('assoc reward bool', mut(lambda s: s[a]['episodes'][0].update(reward=True))),
        ('assoc step float', mut(lambda s: s[a]['episodes'][0].update(step=12.0))),
        ('assoc step off schedule', mut(lambda s: s[a]['episodes'][0].update(step=ASSOC_STORE_EVERY + 1))),
        ('assoc step not before engine step', mut(lambda s: s[a]['episodes'][-1].update(step=10 * ASSOC_STORE_EVERY))),
        ('assoc steps not increasing', mut(lambda s: s[a]['episodes'][1].update(step=s[a]['episodes'][0]['step']))),
        ('assoc episode extra key', mut(lambda s: s[a]['episodes'][0].update(extra=0))),
        ('assoc hits above engine step', mut(lambda s: s[a].update(hits=s['runtime']['step'] + 1))),
        ('assoc stores above schedule', mut(lambda s: s[a].update(stores=s[a]['stores'] + 1))),
        ('assoc stored episode missing', mut(lambda s: s[a]['episodes'].pop(0))),
        ('fast extra key', mut(lambda s: s[f].update(extra=1))),
        ('fast updates negative', mut(lambda s: s[f].update(updates=-1))),
        ('fast updates above engine step', mut(lambda s: s[f].update(updates=s['runtime']['step'] + 1))),
        ('fast written matrix without updates', mut(lambda s: s[f].update(updates=0))),
        ('fast wrong shape', mut(lambda s: s[f].update(matrix=[[0.0] * FAST_DIM] * (FAST_DIM - 1)))),
        ('fast ragged', mut(lambda s: s[f]['matrix'].__setitem__(0, [0.0]))),
        ('fast out of range', mut(lambda s: s[f]['matrix'][0].__setitem__(0, 1.5))),
        ('fast non-finite', mut(lambda s: s[f]['matrix'][0].__setitem__(0, float('inf')))),
        ('fast not numeric', mut(lambda s: s[f].update(matrix='zeros'))),
    ]


def test_import_validates_everything_before_mutating():
    e = NeuroWeaveEngine(_spec(NEW))
    _run(e, 13)
    state = e.export_state()
    before = json.dumps(e.export_state(), sort_keys=True)
    for name, bad in _bad_states(state):
        with pytest.raises(ValueError):
            e.import_state(bad)
        assert json.dumps(e.export_state(), sort_keys=True) == before, name
    # memories in a state whose spec does not enable them are rejected, also without mutation
    plain = NeuroWeaveEngine(_spec())
    plain_before = json.dumps(plain.export_state(), sort_keys=True)
    for key in NEW:
        with_mem = plain.export_state()
        with_mem[key] = copy.deepcopy(state[key])
        with pytest.raises(ValueError):
            plain.import_state(with_mem)
    assert json.dumps(plain.export_state(), sort_keys=True) == plain_before


def test_from_state_enforces_capacity_and_per_step_counts():
    def eps(steps):
        return [{'key': [0.0] * ASSOC_KEY_DIM, 'state': [0.5], 'reward': 0.9, 'step': s} for s in steps]

    load = AssociativeTraceMemory.from_state
    n = 5 * ASSOC_STORE_EVERY + 1                   # engine steps 0..60: six scheduled store steps
    sched = [ASSOC_STORE_EVERY * i for i in range(6)]
    ok = load({'episodes': eps(sched), 'hits': n, 'stores': 6}, max_units=4, before_step=n)
    assert ok.stores == 6 and ok.hits == n and [ep['step'] for ep in ok.episodes] == sched
    for bad in ({'episodes': eps(sched), 'hits': n + 1, 'stores': 6},        # more than one hit per step
                {'episodes': eps(sched[1:]), 'hits': 0, 'stores': 6}):       # a stored episode is missing
        with pytest.raises(ValueError):
            load(bad, max_units=4, before_step=n)
    # past capacity exactly the ASSOC_CAPACITY newest episodes remain (FIFO), and the store count is
    # bounded by the number of scheduled store steps before the engine step
    steps = [ASSOC_STORE_EVERY * i for i in range(ASSOC_CAPACITY + 2)]
    big = steps[-1] + 1                             # ASSOC_CAPACITY + 2 scheduled store steps
    full = load({'episodes': eps(steps[2:]), 'hits': 0, 'stores': ASSOC_CAPACITY + 2}, max_units=4, before_step=big)
    assert len(full.episodes) == ASSOC_CAPACITY and full.stores == ASSOC_CAPACITY + 2
    for episodes, stores in ((eps(steps[1:]), ASSOC_CAPACITY + 2),      # more than ASSOC_CAPACITY episodes
                             (eps(steps[2:]), ASSOC_CAPACITY + 3)):     # more stores than scheduled steps
        with pytest.raises(ValueError):
            load({'episodes': episodes, 'hits': 0, 'stores': stores}, max_units=4, before_step=big)
    # fast weights: at most one write per engine step; a written matrix needs a recorded write
    zero = [[0.0] * FAST_DIM for _ in range(FAST_DIM)]
    fw = FastWeightMemory.from_state({'matrix': zero, 'updates': 3}, max_updates=3)     # zero-key writes
    assert fw.updates == 3 and not np.any(fw.matrix)
    one = [row[:] for row in zero]
    one[2][5] = 0.25
    for bad, cap in (({'matrix': zero, 'updates': 4}, 3), ({'matrix': one, 'updates': 0}, None)):
        with pytest.raises(ValueError):
            FastWeightMemory.from_state(bad, max_updates=cap)
    assert FastWeightMemory.from_state({'matrix': one, 'updates': 1}).matrix[2, 5] == 0.25


# ------------------------------------------------------------------ engine-extension hook stays intact
def test_neutral_extension_is_still_exact_with_the_new_flags():
    plain, hooked = NeuroWeaveEngine(_spec(NEW)), NeuroWeaveEngine(_spec(NEW))
    hooked.attach_extension(EngineExtension())
    oa, _ = _run(plain, 14)
    ob, rb = _run(hooked, 14)
    assert all(np.array_equal(x, y) for x, y in zip(oa, ob))
    assert plain.export_state() == hooked.export_state() and rb[-1]['extensions'] == ['extension']


# ------------------------------------------------------------------ brain-level per-unit input
def test_brain_unit_input_is_added_to_the_feed_forward_drive():
    spec = NeuroWeaveSpecV1(seed=3, structural_interval=5, sleep_interval=10)
    d = NeuroWeaveBrain(spec, mechanisms=())
    syn = d.synapses
    rng = np.random.default_rng(0)
    fast, prev = np.zeros(d.n), np.zeros(d.n)
    for _ in range(5):
        x, extra = rng.normal(size=8), 0.2 * rng.normal(size=d.n)
        inc = np.bincount(syn.dst, weights=prev[syn.src] * syn.w, minlength=d.n)
        fast = 0.58 * fast + (d.project_input(x) + extra) + 0.52 * inc
        prev = d.step(x, reward=0.5, unit_input=extra)
        np.testing.assert_allclose(prev, np.tanh(0.84 * fast + 0.16 * inc), atol=1e-12)
    out0, step0 = d.cells.output.copy(), d.step_count
    for bad in (np.zeros(d.n + 1), np.full(d.n, np.nan)):
        with pytest.raises(ValueError):
            d.step(np.ones(8), unit_input=bad)
    assert np.array_equal(d.cells.output, out0) and d.step_count == step0


# ------------------------------------------------------------------ lab: what the registered experiment now measures
def test_lab_associative_experiment_measures_fast_weights_and_flags_the_short_run_store_as_inert():
    kw = dict(seeds=(0, 1), heldout_seeds=(100,), steps=8, neurons=16, heldout_steps=4)
    r = ex.run_experiment('associative_memory', **kw)
    assert ex.verify_receipt(r)
    # an episode can only be stored on step 0 or 12; step 0 carries no reward, so 8 steps never store one
    # and the lab's inert guard (minus arm bit-identical to the candidate) must say so instead of a verdict
    assert r['inert_mechanisms'] == ['associative_memory']
    assert r['ablation_verdicts']['associative_memory'] == 'inert_in_this_configuration'
    c = r['comparisons']['candidate_vs_minus_associative_memory']
    assert c['inert'] is True and c['verdict'] == 'no_detectable_difference'
    # fast weights are written from the first rewarded step, so their ablation is a real comparison
    assert r['ablation_verdicts']['fast_weights'] in ex.VERDICTS
    assert 'inert' not in r['comparisons']['candidate_vs_minus_fast_weights']
    # same seeds and settings -> the same sealed receipt
    assert ex.run_experiment('associative_memory', **kw)['digest'] == r['digest']


def test_lab_quick_settings_exercise_both_mechanisms():
    # registered quick settings (24 training steps): the lagged reward on step 12 exceeds 0.35 for at
    # least one seed, so the store fires and neither minus arm reproduces the candidate
    r = ex.run_all_experiments(True, experiment_ids=['associative_memory'])[0]
    assert ex.verify_receipt(r) and r['inert_mechanisms'] == []
    assert set(r['ablation_verdicts']) == {'associative_memory', 'fast_weights', 'memory_ecology'}
    assert all(v in ex.VERDICTS for v in r['ablation_verdicts'].values())
