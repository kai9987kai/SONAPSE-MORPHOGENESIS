import json
import math

import numpy as np
import pytest

from synapse.neuroweave.ecology import (CONFIG_NAMES, ECOLOGY_SCHEMA, GRAFTBRAIN_TASKS, HEIGHT, WIDTH,
                                        NeuroWeaveEcology)
from synapse.neuroweave.spec import SENSORY_CHANNEL_IDS


def _calm(config='default', **kw):
    """Arena without obstacles or wind so geometry tests are exact."""
    params = {'n_obstacles': 0, 'disturbance_scale': 0.0}
    params.update(kw)
    return NeuroWeaveEcology(config, seed=3, **params)


def _rollout(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        out.append(env.step(rng.uniform(-1, 1, (env.n_agents, 2))))
    return out


def test_arena_and_default_inventory():
    env = NeuroWeaveEcology('default', seed=0)
    s = env.summary()
    assert (s['width'], s['height']) == (1200.0, 660.0)
    assert s['obstacles'] == 10 and s['food_patches'] == 18 and s['thermal_zones'] == 3
    assert s['raw_media_stored'] is False and s['evidence_tier'] == 'synthetic_abstraction'
    assert np.all(env.th_sign != 0) and (env.th_sign > 0).any() and (env.th_sign < 0).any()


def test_determinism_and_seed_sensitivity():
    a, b, c = (NeuroWeaveEcology('default', seed=s) for s in (5, 5, 6))
    ra, rb = _rollout(a, 40), _rollout(b, 40)
    _rollout(c, 40)
    assert [r[1] for r in ra] == [r[1] for r in rb]
    assert a.export_state() == b.export_state()
    assert a.export_state()['arrays']['pos'] != c.export_state()['arrays']['pos']


def test_export_import_clone_roundtrip_continues_identically():
    env = NeuroWeaveEcology('multi_agent', seed=2)
    _rollout(env, 25)
    state = json.loads(json.dumps(env.export_state()))
    assert state['schema'] == ECOLOGY_SCHEMA
    assert set(state) >= {'schema', 'version', 'config', 'seed', 'params', 't', 'rng', 'arrays', 'evidence_tier'}
    twin = env.clone()
    other = NeuroWeaveEcology('default', seed=99)
    other.import_state(state)
    acts = np.random.default_rng(7).uniform(-1, 1, (15, 3, 2))
    for act in acts:
        r1 = env.step(act)
        r2 = twin.step(act)
        r3 = other.step(act)
        assert r1[1] == r2[1] == r3[1]
    assert env.export_state() == twin.export_state() == other.export_state()


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(schema='bogus'),
    lambda s: s['arrays']['pos'][0].__setitem__(0, float('nan')),
    lambda s: s['arrays'].__setitem__('energy', [1.0, 2.0]),
    lambda s: s['arrays'].__setitem__('food_amount', [2.0] * len(s['arrays']['food_amount'])),
    lambda s: s.update(rule_sign=0.5),
    lambda s: s['params'].__setitem__('agents', 0),
    lambda s: s.update(rng={'bit_generator': 'nope'}),
    lambda s: s['arrays'].pop('gust'),
])
def test_import_state_validates_without_mutation(mutate):
    env = NeuroWeaveEcology('default', seed=1)
    _rollout(env, 5)
    before = env.export_state()
    bad = json.loads(json.dumps(before))
    mutate(bad)
    with pytest.raises(ValueError):
        env.import_state(bad)
    assert env.export_state() == before


def test_food_depletes_rewards_and_replenishes():
    env = _calm()
    j = 0
    env.pos[0] = env.food_pos[j]
    env.food_amount[j] = 1.0
    e0 = env.energy[0]
    _, r, _, info = env.step([0.0, 0.0])
    assert info['eaten'][0] > 0 and r[0] > 0
    assert env.food_amount[j] < 1.0
    assert env.energy[0] > e0
    for _ in range(4):
        env.step([0.0, 0.0])
    depleted = env.food_amount[j]
    assert depleted <= 0.05
    # Walk away and let the patch regrow.
    env.pos[0] = np.array([WIDTH - 20.0, HEIGHT - 20.0])
    env.food_pos[j] = np.array([100.0, 100.0])
    env.step([0.0, 0.0])
    a1 = env.food_amount[j]
    for _ in range(10):
        env.step([0.0, 0.0])
    assert env.food_amount[j] == pytest.approx(min(1.0, a1 + 10 * env.params['regen']))


def test_scarce_regrows_slower_and_has_fewer_patches():
    d, s = NeuroWeaveEcology('default', seed=0), NeuroWeaveEcology('scarce', seed=0)
    assert s.food_pos.shape[0] < d.food_pos.shape[0]
    assert s.params['regen'] < d.params['regen']


def test_obstacle_collision_counted_once_per_contact_and_penalised():
    env = _calm(n_obstacles=1)
    env.obs_vel[:] = 0.0
    env.obs_r[:] = 20.0
    env.pos[0] = np.array([600.0, 330.0])
    env.heading[0] = 0.0
    env.obs_pos[0] = np.array([600.0 + 12 + 20 + 2, 330.0])
    env.food_amount[:] = 0.0
    _, r, _, info = env.step([0.0, 1.0])            # thrust into the obstacle
    assert info['collisions'] == [1] and env.collisions[0] == 1
    assert info['contact'] == [True] and r[0] < 0
    assert np.hypot(*(env.pos[0] - env.obs_pos[0])) >= 32.0 - 1e-9
    _, _, _, info = env.step([0.0, 1.0])            # still pressing: same contact
    assert info['collisions'] == [0] and env.collisions[0] == 1
    assert env.sensory_vector(0)[SENSORY_CHANNEL_IDS.index('mechanosensation')] == 1.0


def test_obstacles_move_and_bounce_inside_arena():
    env = NeuroWeaveEcology('default', seed=4)
    p0 = env.obs_pos.copy()
    for _ in range(400):
        env.step([0.0, 0.0])
    assert not np.allclose(env.obs_pos, p0)
    assert np.all(env.obs_pos[:, 0] >= env.obs_r - 1e-9) and np.all(env.obs_pos[:, 0] <= WIDTH - env.obs_r + 1e-9)
    assert np.all(env.obs_pos[:, 1] >= env.obs_r - 1e-9) and np.all(env.obs_pos[:, 1] <= HEIGHT - env.obs_r + 1e-9)


def test_maze_wall_line_circle_collision_blocks_crossing():
    env = _calm('maze')
    assert env.walls.shape[0] > 0
    env.pos[0] = np.array([280.0, 200.0])            # left of the x=300 wall
    env.heading[0] = 0.0
    env.food_amount[:] = 0.0
    hits = 0
    for _ in range(12):
        _, _, _, info = env.step([0.0, 1.0])
        hits += info['collisions'][0]
    assert env.pos[0, 0] < 300.0 - 12.0 + 1e-6
    assert hits == 1 and env.collisions[0] == 1
    dist, kind = env.rays(0)
    assert kind[0] == 2                                # wall straight ahead
    assert dist[0] < 0.05


def test_thermal_zones_affect_energy_sense_and_reward():
    def run(pos, zone_sign):
        env = _calm(n_food=0)
        env.th_pos[:] = np.array([[600.0, 330.0]] * env.th_pos.shape[0])
        env.th_pos[1:] = np.array([2000.0, 2000.0])     # only zone 0 active (others far away)
        env.th_r[:] = 150.0
        env.th_intensity[:] = 1.0
        env.th_sign[0] = zone_sign
        env.pos[0] = np.array(pos)
        e0 = env.energy[0]
        _, r, _, info = env.step([0.0, 0.0])
        return e0 - env.energy[0], info['thermal'][0], r[0], env.sensory_vector(0)
    d_hot, t_hot, r_hot, v_hot = run([600.0, 330.0], 1.0)
    d_cold, t_cold, r_cold, v_cold = run([600.0, 330.0], -1.0)
    d_far, t_far, r_far, _ = run([100.0, 600.0], 1.0)
    ti = SENSORY_CHANNEL_IDS.index('thermal')
    assert t_hot > 0.9 and t_cold < -0.9 and t_far == 0.0
    assert v_hot[ti] > 0.9 and v_cold[ti] < -0.9
    assert d_hot > d_far + 0.2 and d_cold > d_far + 0.2
    assert r_hot < r_far and r_cold < r_far


def test_shifting_rules_flip_food_value_and_thermal_sign():
    env = NeuroWeaveEcology('shifting_rules', seed=0, flip_period=5, n_obstacles=0, disturbance_scale=0.0)
    ctx0 = env.sensory_vector(0)[15]
    env.pos[0] = np.array([30.0, 30.0])
    env.food_pos[:] = np.array([1000.0, 600.0])
    th_before = env._thermal_field(env.th_pos[0])
    for _ in range(5):
        _, _, _, info = env.step([0.0, 0.0])
    assert info['rule_sign'] == -1.0 and any(e['kind'] == 'rule_flip' for e in info['events'])
    assert env._thermal_field(env.th_pos[0]) == pytest.approx(-th_before)
    assert env.sensory_vector(0)[15] != ctx0
    env.food_pos[0] = env.pos[0]
    env.food_amount[0] = 1.0
    e0 = env.energy[0]
    _, r, _, info = env.step([0.0, 0.0])
    assert r[0] < 0 and info['eaten'][0] < 0 and env.energy[0] < e0
    # toxic food is rendered purple in the 155-d retina, healthy food green
    env.food_amount[0] = 1.0
    env.pos[0] = np.array([200.0, 330.0])
    env.heading[0] = 0.0
    env.food_pos[0] = np.array([280.0, 330.0])
    o = env.to_synapse_observation(0)
    assert (o[33 + 16], o[66 + 16], o[99 + 16]) == pytest.approx((0.7, 0.3, 1.0))
    for _ in range(4):
        env.step([0.0, 0.0])
    assert env.t == 10 and env.rule_sign == 1.0


def test_hidden_food_requires_landmark_memory():
    env = NeuroWeaveEcology('hidden_food', seed=1, n_obstacles=0, disturbance_scale=0.0)
    j = 0
    lm = int(env.food_landmark[j])
    food = env.food_pos[j].copy()
    olf = SENSORY_CHANNEL_IDS.index('olfactory')
    # Adjacent to the food but its landmark never visited: sensors are blind to it.
    env.landmarks_seen[:] = False
    env.pos[0] = food + np.array([40.0, 0.0]) if food[0] < WIDTH - 60 else food - np.array([40.0, 0.0])
    assert not env.food_visible(0).any()
    assert env.sensory_vector(0)[olf] == 0.0
    assert 4 not in env.rays(0)[1]
    assert np.allclose(env.to_synapse_observation(0)[149:151], 0.5)
    # Visit the landmark (memory formed), then return near the food.
    env.pos[0] = env.landmarks[lm].copy()
    env.step([0.0, 0.0])
    assert env.landmarks_seen[0, lm]
    env.pos[0] = food + np.array([40.0, 0.0]) if food[0] < WIDTH - 60 else food - np.array([40.0, 0.0])
    env.step([0.0, 0.0])
    assert env.food_visible(0)[j]
    assert env.sensory_vector(0)[olf] > 0.0
    # Memory survives persistence.
    twin = env.clone()
    assert twin.landmarks_seen[0, lm] and twin.food_visible(0)[j]
    # Forgetting (fresh agent memory) hides the food again from the same spot.
    env.landmarks_seen[:] = False
    assert not env.food_visible(0)[j]


def test_hidden_food_blind_before_landmark_visit_strict():
    env = NeuroWeaveEcology('hidden_food', seed=2, n_obstacles=0, disturbance_scale=0.0)
    env.landmarks_seen[:] = False
    env.pos[0] = env.food_pos[0] + np.array([0.0, 30.0 if env.food_pos[0, 1] < HEIGHT - 50 else -30.0])
    # Same placement in a non-hidden arena *is* visible.
    assert not env.food_visible(0).any()
    assert env.sensory_vector(0)[SENSORY_CHANNEL_IDS.index('olfactory')] == 0.0
    plain = NeuroWeaveEcology('default', seed=2, n_obstacles=0, disturbance_scale=0.0)
    plain.food_pos = env.food_pos.copy()
    plain.food_amount = env.food_amount.copy()
    plain.pos[0] = env.pos[0]
    assert plain.food_visible(0).any()
    assert plain.sensory_vector(0)[SENSORY_CHANNEL_IDS.index('olfactory')] > 0.0


def test_long_horizon_reward_is_delayed_until_sequence_complete():
    env = NeuroWeaveEcology('long_horizon', seed=0, n_obstacles=0, disturbance_scale=0.0, n_food=0)
    n = env.waypoints.shape[0]
    rewards = []
    for k in range(n):
        env.pos[0] = env.waypoints[k].copy()
        _, r, _, info = env.step([0.0, 0.0])
        rewards.append(r[0])
        if k < n - 1:
            assert info['waypoint_index'][0] == k + 1
    assert all(abs(x) < 0.2 for x in rewards[:-1])
    assert rewards[-1] >= env.params['completion_reward'] - 0.2
    assert env.sequences_completed[0] == 1 and env.waypoint_index[0] == 0


def test_multi_agent_competition_and_contacts():
    env = NeuroWeaveEcology('multi_agent', seed=0, n_obstacles=0, disturbance_scale=0.0)
    assert env.n_agents == 3
    obs, r, done, info = env.step(np.zeros((3, 2)))
    assert len(obs) == 3 and len(r) == 3 and [o['agent'] for o in obs] == [0, 1, 2]
    # Two agents share one patch holding 0.3: agent 0 bites first, agent 1 gets the rest.
    env.food_amount[:] = 0.0
    env.food_amount[0] = 0.3
    env.pos[0] = env.food_pos[0] + np.array([-6.0, 0.0])
    env.pos[1] = env.food_pos[0] + np.array([6.0, 0.0])
    env.pos[2] = np.array([30.0, 30.0])
    _, r, _, info = env.step(np.zeros((3, 2)))
    assert info['eaten'][0] == pytest.approx(0.2)
    assert info['eaten'][1] == pytest.approx(0.1)
    assert info['contact'][0] and info['contact'][1]
    assert np.hypot(*(env.pos[0] - env.pos[1])) > 12.0
    assert any(e['kind'] == 'agent_contact' for e in info['events'])
    with pytest.raises(ValueError):
        env.step([0.0, 1.0])                              # wrong number of agents
    v = env.sensory_vector(2)
    assert v.shape == (16,)


def test_disturbance_is_recorded_ground_truth():
    env = NeuroWeaveEcology('default', seed=0, n_obstacles=0, n_food=0)
    env.pos[0] = np.array([600.0, 330.0])
    env.gust[:] = [1.5, -0.5]
    env.gust_remaining = 5
    p0 = env.pos[0].copy()
    _, _, _, info = env.step([0.0, 0.0])
    push = np.array(info['disturbance'][0])
    assert np.hypot(*push) > 0.5
    assert np.allclose(env.pos[0] - p0, push)
    env2 = NeuroWeaveEcology('default', seed=0, n_obstacles=0, n_food=0, disturbance_scale=0.0)
    env2.pos[0] = p0.copy()
    _, _, _, info2 = env2.step([0.0, 0.0])
    assert info2['disturbance'][0] == [0.0, 0.0] and np.allclose(env2.pos[0], p0)
    es = SENSORY_CHANNEL_IDS.index('environment_state')
    assert env.sensory_vector(0)[es] > env2.sensory_vector(0)[es]


def test_action_formats_are_equivalent_and_validated():
    a, b = NeuroWeaveEcology('default', seed=8), NeuroWeaveEcology('default', seed=8)
    a.step([0.3, 0.8])                                 # [turn, thrust]
    b.step([0.8, 0.3, 0.0, 0.0, 0.0])                  # Sonapse [drive, turn, reach, height, grip]
    assert a.export_state() == b.export_state()
    assert a.speed[0] == pytest.approx(0.8 * 4.0)
    with pytest.raises(ValueError):
        a.step([float('nan'), 0.0])
    with pytest.raises(ValueError):
        a.step([0.0, 0.0, 0.0])
    a.step({'action': [0.0, 0.0, 0.0, 0.0, 0.0]})


def test_energy_decay_and_exhaustion():
    env = _calm(n_food=0)
    env.th_pos[:] = 5000.0
    e0 = env.energy[0]
    env.step([0.0, 0.0])
    idle = e0 - env.energy[0]
    e1 = env.energy[0]
    env.step([0.0, 1.0])
    moving = e1 - env.energy[0]
    assert idle > 0 and moving > idle
    env.energy[0] = 0.01
    _, r, done, info = env.step([0.0, 0.0])
    assert done and info['alive'] == [False] and r[0] <= -1.0
    assert any(e['kind'] == 'exhausted' for e in info['events'])


@pytest.mark.parametrize('config', CONFIG_NAMES)
def test_sensory_vector_and_adapter_bounds(config):
    env = NeuroWeaveEcology(config, seed=11)
    vecs = []
    for _ in range(25):
        env.step(np.random.default_rng(env.t).uniform(-1, 1, (env.n_agents, 2)))
        for i in range(env.n_agents):
            v = env.sensory_vector(i)
            assert v.shape == (16,) and np.isfinite(v).all()
            assert np.all(v >= -1.0) and np.all(v <= 1.0)
            vecs.append(v)
            o = env.to_synapse_observation(i)
            assert o.shape == (155,) and o.dtype == np.float32 and np.isfinite(o).all()
            assert np.all(o[:132] >= 0) and np.all(o[:132] <= 1)
            assert -1 <= o[137] <= 1 and o[138] ** 2 + o[139] ** 2 == pytest.approx(1.0, abs=1e-5)
            assert o[140:149].sum() == 1.0 and np.all((o[149:153] >= 0) & (o[149:153] <= 1))
            assert o[152] == 1.0 and 0 <= o[154] <= 1
    assert np.ptp(np.array(vecs), axis=0).max() > 0     # channels actually vary
    assert set(env.sensory_dict(0)) == set(SENSORY_CHANNEL_IDS)


def test_visual_channels_respond_to_scene():
    env = _calm(n_food=0)
    env.th_pos[:] = 5000.0
    env.pos[0] = np.array([600.0, 330.0])
    env.heading[0] = 0.0
    env.step([0.0, 0.0])
    empty = env.sensory_vector(0)
    env2 = _calm(n_food=1)
    env2.th_pos[:] = 5000.0
    env2.pos[0] = np.array([600.0, 330.0])
    env2.heading[0] = 0.0
    env2.food_pos[0] = np.array([660.0, 330.0])
    env2.food_amount[0] = 1.0
    env2.step([0.0, 0.0])
    with_food = env2.sensory_vector(0)
    assert with_food[0] > empty[0]           # luminance
    assert with_food[1] > empty[1]           # red/green opponency toward food
    assert with_food[8] > empty[8]           # olfactory
    o = env2.to_synapse_observation(0)
    centre = 16
    assert o[centre] > 0.85                  # depth of food straight ahead
    assert o[66 + centre] == pytest.approx(1.0)  # green channel
    # approaching obstacle produces positive motion signal
    env3 = _calm(n_food=0, n_obstacles=1)
    env3.th_pos[:] = 5000.0
    env3.pos[0] = np.array([600.0, 330.0])
    env3.heading[0] = 0.0
    env3.obs_pos[0] = np.array([800.0, 330.0])
    env3.obs_vel[0] = np.array([-2.0, 0.0])
    env3.step([0.0, 0.0])
    assert env3.sensory_vector(0)[4] > 0.2


def test_adapter_layout_and_obs_dict():
    env = NeuroWeaveEcology('maze', seed=0)
    obs, _, _, _ = env.step([0.5, 0.4])
    o = env.to_synapse_observation(obs[0])
    assert np.array_equal(o, env.to_synapse_observation(0))
    assert o[137] == pytest.approx(0.5, abs=1e-6)                # turn command -> compass input
    assert o[136] == pytest.approx(0.4, abs=1e-6)                # forward speed
    assert o[141] == 1.0                                          # graftbrain 'maze' task slot
    assert o[138] == pytest.approx(math.sin(env.heading[0]), abs=1e-6)
    env.step([0.0, 0.0])
    with pytest.raises(ValueError):
        env.to_synapse_observation(obs[0])                       # stale
    with pytest.raises(ValueError):
        env.to_synapse_observation(3)


def test_graftbrain_task_order_matches():
    try:
        from synapse.complex_world import BASE_TASKS
    except Exception:  # pragma: no cover - vendored world unavailable
        pytest.skip('graftbrain world unavailable')
    assert tuple(BASE_TASKS) == GRAFTBRAIN_TASKS


class _FakeAgent:
    def __init__(self):
        self.feedback_calls = []
        self.pending_transition = None
        self.obs_seen = []

    def act(self, obs, learn=False):
        assert obs.shape == (155,) and np.isfinite(obs).all()
        self.obs_seen.append(obs.copy())
        self.pending_transition = {}
        return {'action': np.array([1.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)}

    def feedback(self, reward, done=False):
        self.feedback_calls.append((reward, done))


def test_run_synapse_agent_with_fake_agent():
    env = NeuroWeaveEcology('default', seed=0)
    agent = _FakeAgent()
    out = env.run_synapse_agent(agent, 12, learn=False)
    assert out['steps'] == 12 and agent.feedback_calls == [] and agent.pending_transition is None
    assert set(out) >= {'total_reward', 'food_eaten', 'collisions', 'final_energy', 'mean_energy'}
    assert out['actions_finite'] and len(out['rewards']) == 12
    assert out['total_reward'] == pytest.approx(sum(out['rewards']))
    out2 = env.run_synapse_agent(agent, 5, learn=True)
    assert len(agent.feedback_calls) == 5
    assert [r for r, _ in agent.feedback_calls] == out2['rewards']
    with pytest.raises(ValueError):
        env.run_synapse_agent(agent, 0)


def test_run_synapse_agent_real_sonapse_smoke():
    pytest.importorskip('torch')
    from synapse.agent import IntegratedAgent
    from synapse.config import SynapseConfig
    agent = IntegratedAgent(SynapseConfig(nodes_per_source=16))
    env = NeuroWeaveEcology('default', seed=0)
    p0 = env.pos[0].copy()
    out = env.run_synapse_agent(agent, 5, learn=False)
    assert out['steps'] == 5 and out['actions_finite']
    assert np.isfinite(out['total_reward']) and 0 <= out['final_energy'] <= 1
    assert not np.allclose(env.pos[0], p0)
    out = env.run_synapse_agent(agent, 2, learn=True)
    assert out['steps'] == 2 and np.isfinite(out['total_reward'])


# --- adversarial review additions -------------------------------------------------

@pytest.mark.parametrize('mutate', [
    lambda s: s['arrays'].__setitem__('energy', []),                       # empty list for non-empty array
    lambda s: s['arrays'].__setitem__('pos', []),
    lambda s: s['arrays'].__setitem__('energy', ['50.0']),                 # strings are not numbers
    lambda s: s['arrays'].__setitem__('energy', [True]),                   # bools are not numbers
    lambda s: s['arrays'].__setitem__('alive', [1]),                       # ints are not bools
    lambda s: s['arrays'].__setitem__('collisions', [-5]),
    lambda s: s['arrays']['obs_vel'].__setitem__(0, [1e300, 0.0]),
    lambda s: s['arrays'].__setitem__('gust', [1e6, 0.0]),
    lambda s: s['arrays'].__setitem__('obs_r', [1e9] * len(s['arrays']['obs_r'])),
    lambda s: s.update(rule_sign=True),
    lambda s: s.update(seed=-3),
    lambda s: s['params'].__setitem__('disturbance_scale', 1e300),
])
def test_import_state_rejects_subtle_malformed_values(mutate):
    env = NeuroWeaveEcology('default', seed=1)
    _rollout(env, 5)
    before = env.export_state()
    bad = json.loads(json.dumps(before))
    mutate(bad)
    with pytest.raises(ValueError):
        env.import_state(bad)
    assert env.export_state() == before


def test_maze_walls_must_match_layout_on_import():
    env = NeuroWeaveEcology('maze', seed=1)
    bad = json.loads(json.dumps(env.export_state()))
    bad['arrays']['walls'][0] = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        env.import_state(bad)


@pytest.mark.parametrize('key,value', [('obstacle_speed', 1e6), ('disturbance_scale', 50.0),
                                       ('thermal_cost', float('inf')), ('completion_reward', 1e9)])
def test_param_upper_bounds(key, value):
    with pytest.raises(ValueError):
        NeuroWeaveEcology('default', seed=0, **{key: value})


def test_episode_counter_persists_and_layout_varies_deterministically():
    env = NeuroWeaveEcology('default', seed=4, max_steps=3)
    first = env.food_pos.copy()
    for _ in range(3):
        _, _, done, _ = env.step([0.0, 0.0])
    assert done and env.episodes == 1
    env.reset()
    assert env.episodes == 1                                # reset no longer erases the counter
    assert not np.allclose(env.food_pos, first)             # new episode, new layout
    twin = NeuroWeaveEcology('default', seed=4, max_steps=3)
    for _ in range(3):
        twin.step([0.0, 0.0])
    twin.reset()
    assert env.export_state() == twin.export_state()        # still fully deterministic
    clone = env.clone()
    clone.reset()
    env.reset()
    assert env.export_state() == clone.export_state()


def test_run_synapse_agent_counts_episodes_across_resets():
    env = NeuroWeaveEcology('default', seed=0, max_steps=4)
    out = env.run_synapse_agent(_FakeAgent(), 10, learn=False)
    assert out['episodes_finished'] == 2 and env.episodes == 2 and out['steps'] == 10


def test_exhausted_agent_is_inert_in_multi_agent_arena():
    env = NeuroWeaveEcology('multi_agent', seed=0, n_obstacles=0, n_food=0)
    env.th_pos[:] = 5000.0
    env.energy[1] = 0.01
    _, r, done, info = env.step(np.zeros((3, 2)))
    assert info['alive'] == [True, False, True] and not done and r[1] <= -1.0
    env.pos[1] = np.array([6.0 + 12.0, 330.0])               # pressed against the boundary
    p1 = env.pos[1].copy()
    env.gust[:] = [-2.0, 0.0]
    env.gust_remaining = 5
    acts = np.zeros((3, 2))
    acts[1] = [0.0, 1.0]
    _, r, _, info = env.step(acts)
    assert r[1] == 0.0 and info['disturbance'][1] == [0.0, 0.0]
    assert np.allclose(env.pos[1], p1)
    assert np.hypot(*info['disturbance'][0]) > 0.0            # living agents still pushed
