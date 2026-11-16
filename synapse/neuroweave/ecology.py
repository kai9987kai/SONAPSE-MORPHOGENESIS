"""NEUROWEAVE ecology: a 1200 x 660 continuous foraging arena for Sonapse.

Re-implements (in NumPy) the NEUROWEAVE-LAB 0.4.0 expanded ecology - moving
obstacles, regrowing food patches, thermal zones, energy, collisions - and
extends it with configurable task variants and a ground-truth disturbance
field for action-grounded self-model experiments.

This is a synthetic control benchmark. It makes no claim of biological
fidelity; "smell", "taste", "sound" and "temperature" are named proxies for
geometric quantities computed from the simulated arena.

Configs
-------
``default``         10 bouncing obstacles, 18 food patches, 3 thermal zones, wind.
``maze``            fixed wall segments (line-circle collision) plus a few obstacles.
``scarce``          few patches with slow regrowth and lower starting energy.
``shifting_rules``  every ``flip_period`` steps food value and thermal sign flip.
``hidden_food``     food is invisible to sensors unless the agent has previously
                    visited the food's landmark and is still within its reveal
                    radius (memory-dependent foraging).
``long_horizon``    ordered waypoint sequence; reward only on completion (delayed).
``multi_agent``     several agents competing for the same patches.

Coordinates are screen pixels (x right, y down); heading is radians.

Sonapse 155-d adapter layout (mirrors ``graftbrain.world.World.observe``):

=========  =====================================================================
index      meaning
=========  =====================================================================
0:33       retina depth ``1 - d / RETINA_RANGE`` for 33 rays over a 120 deg FOV
33:66      retina red channel of the first hit
66:99      retina green channel
99:132     retina blue channel
132        arm reach (no arm here -> 0.0, the graftbrain rest value)
133        arm height (0.0)
134        grip (0.0)
135        holding flag (0.0)
136        forward speed / MAX_SPEED (graftbrain ``v / 1.4``; range [-0.5, 1])
137        turn command, scaled so ``obs[137] * 2.5`` is the angular velocity in
           rad/s at graftbrain's DT=0.1 (compass input of the Sonapse agent)
138, 139   sin(heading), cos(heading)
140:149    one-hot over graftbrain TASKS (config -> nearest task, see CONFIG_TASK)
149:153    cue: [0.5+0.5*target_lateral, 0.5+0.5*target_forward, energy01, 1.0]
           (target = sensed food gradient, or the current waypoint in
           long_horizon; this is the mushroom-body input of the Sonapse agent)
153        last reward clipped to [-10, 10]
154        t / max_steps
=========  =====================================================================

Retina colours reuse graftbrain's palette: food green (purple while food is
toxic in ``shifting_rules``), obstacles red, walls/boundary grey, landmarks
gold, other agents cyan, current waypoint yellow.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .spec import EVIDENCE_TIER, SENSORY_CHANNEL_IDS

ECOLOGY_SCHEMA = 'NeuroWeaveEcologyV1'
ECOLOGY_VERSION = 1

WIDTH = 1200.0
HEIGHT = 660.0
AGENT_RADIUS = 12.0
MAX_SPEED = 4.0            # px / step at thrust 1
REVERSE_FRACTION = 0.5     # reverse thrust is weaker
TURN_RATE = 0.25           # rad / step at turn 1
N_RAYS = 16
RAY_FOV = 2.0 * math.pi    # egocentric ring of rays
RAY_RANGE = 360.0
RETINA_RAYS = 33
RETINA_FOV = math.radians(120.0)
RETINA_RANGE = 600.0
MAX_ENERGY = 100.0
SMELL_SCALE = 150.0
EAT_BITE = 0.2
FOOD_REWARD = 2.0
FOOD_ENERGY = 25.0
COLLISION_PENALTY = 0.25
AGENT_CONTACT_PENALTY = 0.1
EXHAUSTION_PENALTY = 1.0
THERMAL_STRESS_THRESHOLD = 0.35
THERMAL_STRESS_PENALTY = 0.05
WAYPOINT_RADIUS = 30.0
LANDMARK_VISIT_RADIUS = 90.0
LANDMARK_REVEAL_RADIUS = 360.0

# graftbrain.world.TASKS (copied names; order is part of the 155-d layout).
GRAFTBRAIN_TASKS = ('forage', 'maze', 'push', 'grasp_place', 'peg_insert', 'delayed_match',
                    'go_nogo', 'button_sequence', 'webcam_chase')

CONFIG_NAMES = ('default', 'maze', 'scarce', 'shifting_rules', 'hidden_food', 'long_horizon', 'multi_agent')
CONFIG_TASK = {
    'default': 'forage', 'maze': 'maze', 'scarce': 'forage', 'shifting_rules': 'go_nogo',
    'hidden_food': 'delayed_match', 'long_horizon': 'button_sequence', 'multi_agent': 'forage',
}

# Ray hit categories.
HIT_NONE, HIT_BOUNDARY, HIT_WALL, HIT_OBSTACLE, HIT_FOOD, HIT_LANDMARK, HIT_AGENT, HIT_WAYPOINT = range(8)
HIT_NAMES = ('none', 'boundary', 'wall', 'obstacle', 'food', 'landmark', 'agent', 'waypoint')
_COLOURS = {
    HIT_NONE: (0.0, 0.0, 0.0), HIT_BOUNDARY: (0.45, 0.45, 0.5), HIT_WALL: (0.45, 0.45, 0.5),
    HIT_OBSTACLE: (1.0, 0.2, 0.2), HIT_FOOD: (0.2, 1.0, 0.3), HIT_LANDMARK: (1.0, 0.75, 0.1),
    HIT_AGENT: (0.2, 0.9, 1.0), HIT_WAYPOINT: (1.0, 0.9, 0.2),
}
_TOXIC_FOOD_COLOUR = (0.7, 0.3, 1.0)
_BRIGHTNESS = {HIT_NONE: 0.0, HIT_BOUNDARY: 0.3, HIT_WALL: 0.3, HIT_OBSTACLE: 0.5, HIT_FOOD: 1.0,
               HIT_LANDMARK: 0.8, HIT_AGENT: 0.6, HIT_WAYPOINT: 0.9}

_MAZE_WALLS = (
    (300.0, 0.0, 300.0, 420.0),
    (600.0, 240.0, 600.0, 660.0),
    (900.0, 0.0, 900.0, 420.0),
    (120.0, 540.0, 420.0, 540.0),
    (720.0, 120.0, 840.0, 120.0),
    (960.0, 540.0, 1140.0, 540.0),
)

_BASE_PARAMS: dict[str, Any] = {
    'n_obstacles': 10, 'n_food': 18, 'n_thermal': 3, 'n_landmarks': 0, 'n_waypoints': 0,
    'agents': 1, 'walls': False, 'regen': 0.004, 'start_energy': 80.0, 'energy_decay': 0.04,
    'move_cost': 0.06, 'thermal_cost': 0.25, 'disturbance_scale': 1.0, 'flip_period': 0,
    'hidden_food': False, 'max_steps': 2000, 'obstacle_speed': 2.0, 'completion_reward': 5.0,
}
CONFIGS: dict[str, dict[str, Any]] = {
    'default': {},
    'maze': {'n_obstacles': 4, 'n_food': 12, 'n_thermal': 2, 'walls': True},
    'scarce': {'n_food': 5, 'regen': 0.0008, 'start_energy': 60.0},
    'shifting_rules': {'flip_period': 150},
    'hidden_food': {'n_food': 12, 'n_landmarks': 4, 'hidden_food': True},
    'long_horizon': {'n_obstacles': 6, 'n_food': 8, 'n_waypoints': 4, 'energy_decay': 0.02},
    'multi_agent': {'agents': 3, 'n_food': 14},
}
_PARAM_TYPES = {k: type(v) for k, v in _BASE_PARAMS.items()}
# Upper bounds keeping every derived quantity (positions, speeds, rewards) finite and bounded.
_PARAM_UPPER = {'energy_decay': 10.0, 'move_cost': 10.0, 'thermal_cost': 10.0, 'disturbance_scale': 10.0,
                'obstacle_speed': 20.0, 'completion_reward': 100.0, 'max_steps': 10_000_000, 'flip_period': 10_000_000}
# Bounds applied to imported state (anything the simulator itself can produce lies inside them).
_MAX_OBSTACLE_SPEED = 2.0 * _PARAM_UPPER['obstacle_speed']
_MAX_GUST = 25.0
_MAX_RADIUS = 400.0


def config_params(name: str) -> dict[str, Any]:
    if name not in CONFIGS:
        raise ValueError(f'unknown ecology config {name!r}; expected one of {CONFIG_NAMES}')
    return {**_BASE_PARAMS, **CONFIGS[name]}


def _segment_point_distance(segs: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from point p to each segment and the closest points (S,2)."""
    a = segs[:, 0:2]
    b = segs[:, 2:4]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    u = np.clip(((p - a) * ab).sum(1) / denom, 0.0, 1.0)
    closest = a + ab * u[:, None]
    return np.hypot(*(p - closest).T), closest


def _ray_circles(ox: float, oy: float, dx: np.ndarray, dy: np.ndarray, centres: np.ndarray,
                 radii: np.ndarray) -> np.ndarray:
    """Ray-circle entry distance (R, C); inf for misses."""
    if centres.shape[0] == 0:
        return np.full((dx.shape[0], 0), np.inf)
    cx = centres[:, 0] - ox
    cy = centres[:, 1] - oy
    b = dx[:, None] * cx[None] + dy[:, None] * cy[None]
    c = cx * cx + cy * cy - radii * radii
    disc = b * b - c[None]
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = b - sq
    t = np.where(t < 0.0, b + sq, t)            # origin inside the circle -> exit point
    hit = (disc >= 0.0) & (t > 0.0)
    return np.where(hit, t, np.inf)


def _ray_segments(ox: float, oy: float, dx: np.ndarray, dy: np.ndarray, segs: np.ndarray) -> np.ndarray:
    if segs.shape[0] == 0:
        return np.full((dx.shape[0], 0), np.inf)
    ex = segs[:, 2] - segs[:, 0]
    ey = segs[:, 3] - segs[:, 1]
    qx = segs[:, 0] - ox
    qy = segs[:, 1] - oy
    den = dx[:, None] * ey[None] - dy[:, None] * ex[None]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (qx[None] * ey[None] - qy[None] * ex[None]) / den
        u = (qx[None] * dy[:, None] - qy[None] * dx[:, None]) / den
    ok = np.isfinite(t) & (t > 0) & (u >= 0) & (u <= 1)
    return np.where(ok, t, np.inf)


def _ray_box(ox: float, oy: float, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        tx = np.where(dx > 1e-12, (WIDTH - ox) / dx, np.where(dx < -1e-12, -ox / dx, np.inf))
        ty = np.where(dy > 1e-12, (HEIGHT - oy) / dy, np.where(dy < -1e-12, -oy / dy, np.inf))
    return np.maximum(np.minimum(tx, ty), 0.0)


class NeuroWeaveEcology:
    """Continuous 1200 x 660 ecology with configurable task variants."""

    width = WIDTH
    height = HEIGHT

    def __init__(self, config: str = 'default', seed: int = 0, agents: int | None = None,
                 **overrides: Any) -> None:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise ValueError('seed must be an integer')
        params = config_params(config)
        if agents is not None:
            params['agents'] = agents
        for key, value in overrides.items():
            if key not in _BASE_PARAMS:
                raise ValueError(f'unknown ecology parameter {key!r}')
            params[key] = value
        self._validate_params(params)
        self.config = config
        self.seed = int(seed)
        self.params = params
        self.episodes = 0
        self.reset()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _validate_params(p: Mapping[str, Any]) -> None:
        if set(p) != set(_BASE_PARAMS):
            raise ValueError('ecology params must contain exactly the documented keys')
        for key, typ in _PARAM_TYPES.items():
            v = p[key]
            if typ is bool:
                if not isinstance(v, bool):
                    raise ValueError(f'{key} must be bool')
            elif typ is int:
                if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0:
                    raise ValueError(f'{key} must be a non-negative int')
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)) \
                        or not math.isfinite(float(v)) or float(v) < 0:
                    raise ValueError(f'{key} must be a finite non-negative number')
        if not 1 <= int(p['agents']) <= 16:
            raise ValueError('agents must be in [1, 16]')
        if int(p['n_obstacles']) > 64 or int(p['n_food']) > 128 or int(p['n_thermal']) > 16 \
                or int(p['n_landmarks']) > 16 or int(p['n_waypoints']) > 16:
            raise ValueError('ecology object counts exceed bounds')
        if p['hidden_food'] and int(p['n_landmarks']) < 1:
            raise ValueError('hidden_food requires landmarks')
        if int(p['max_steps']) < 1 or float(p['start_energy']) > MAX_ENERGY or float(p['regen']) > 1.0:
            raise ValueError('invalid max_steps/start_energy/regen')
        for key, hi in _PARAM_UPPER.items():
            if float(p[key]) > hi:
                raise ValueError(f'{key} must be <= {hi}')

    @property
    def n_agents(self) -> int:
        return int(self.params['agents'])

    def _clear_of_walls(self, p: np.ndarray, r: float) -> bool:
        if self.walls.shape[0] == 0:
            return True
        d, _ = _segment_point_distance(self.walls, p)
        return bool(np.all(d > r + 8.0))

    def _sample_point(self, rng: np.random.Generator, r: float, avoid: Sequence[tuple[np.ndarray, float]] = (),
                      near: tuple[np.ndarray, float, float] | None = None) -> np.ndarray:
        for _ in range(400):
            if near is None:
                p = np.array([rng.uniform(r + 10, WIDTH - r - 10), rng.uniform(r + 10, HEIGHT - r - 10)])
            else:
                c, lo, hi = near
                ang = rng.uniform(0, 2 * math.pi)
                p = c + rng.uniform(lo, hi) * np.array([math.cos(ang), math.sin(ang)])
                if not (r + 10 <= p[0] <= WIDTH - r - 10 and r + 10 <= p[1] <= HEIGHT - r - 10):
                    continue
            if not self._clear_of_walls(p, r):
                continue
            if all(np.hypot(*(p - q)) > r + rq + 6.0 for q, rq in avoid):
                return p
        # Deterministic fallback (bounded attempts exhausted): keep last point clipped.
        return np.clip(p, r + 10, [WIDTH - r - 10, HEIGHT - r - 10])

    def reset(self) -> list[dict[str, Any]]:
        """Start a new episode. The layout is a deterministic function of (seed, episodes),
        so successive episodes differ while the run stays reproducible; ``episodes`` is kept."""
        p = self.params
        ep = int(getattr(self, 'episodes', 0))
        rng = np.random.default_rng(self.seed + 7100 + 7919 * ep)
        self.walls = np.array(_MAZE_WALLS, dtype=np.float64) if p['walls'] else np.zeros((0, 4))
        placed: list[tuple[np.ndarray, float]] = []

        n_ag = self.n_agents
        self.pos = np.zeros((n_ag, 2))
        for i in range(n_ag):
            self.pos[i] = self._sample_point(rng, AGENT_RADIUS + 20, placed)
            placed.append((self.pos[i].copy(), AGENT_RADIUS + 20))
        self.heading = rng.uniform(-math.pi, math.pi, n_ag)
        self.vel = np.zeros((n_ag, 2))
        self.speed = np.zeros(n_ag)
        self.turn = np.zeros(n_ag)
        self.energy = np.full(n_ag, float(p['start_energy']))
        self.alive = np.ones(n_ag, dtype=bool)
        self.contact = np.zeros(n_ag, dtype=bool)
        self.collisions = np.zeros(n_ag, dtype=np.int64)
        self.contact_steps = np.zeros(n_ag, dtype=np.int64)
        self.collided_last = np.zeros(n_ag, dtype=np.int64)
        self.food_eaten = np.zeros(n_ag)
        self.bites = np.zeros(n_ag, dtype=np.int64)
        self.eaten_last = np.zeros(n_ag)
        self.total_reward = np.zeros(n_ag)
        self.last_reward = np.zeros(n_ag)
        self.disturbance_last = np.zeros((n_ag, 2))
        self.thermal_last = np.zeros(n_ag)
        self.waypoint_index = np.zeros(n_ag, dtype=np.int64)
        self.sequences_completed = np.zeros(n_ag, dtype=np.int64)

        n_obs = int(p['n_obstacles'])
        self.obs_r = rng.uniform(14.0, 28.0, n_obs)
        self.obs_pos = np.zeros((n_obs, 2))
        for k in range(n_obs):
            self.obs_pos[k] = self._sample_point(rng, float(self.obs_r[k]), placed)
            placed.append((self.obs_pos[k].copy(), float(self.obs_r[k])))
        ang = rng.uniform(-math.pi, math.pi, n_obs)
        spd = rng.uniform(0.5, 1.0, n_obs) * float(p['obstacle_speed'])
        self.obs_vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1) if n_obs else np.zeros((0, 2))
        self.obs_contact = np.zeros((n_ag, n_obs), dtype=bool)
        self.wall_contact = np.zeros(n_ag, dtype=bool)
        self.agent_contact = np.zeros((n_ag, n_ag), dtype=bool)

        n_lm = int(p['n_landmarks'])
        self.landmarks = np.zeros((n_lm, 2))
        for k in range(n_lm):
            self.landmarks[k] = self._sample_point(rng, 18.0, placed)
            placed.append((self.landmarks[k].copy(), 120.0))
        self.landmark_r = np.full(n_lm, 18.0)
        self.landmarks_seen = np.zeros((n_ag, n_lm), dtype=bool)

        n_food = int(p['n_food'])
        self.food_pos = np.zeros((n_food, 2))
        self.food_r = np.full(n_food, 14.0)
        self.food_landmark = np.full(n_food, -1, dtype=np.int64)
        for j in range(n_food):
            near = None
            if p['hidden_food']:
                lm = j % n_lm
                self.food_landmark[j] = lm
                near = (self.landmarks[lm], 70.0, 200.0)
            self.food_pos[j] = self._sample_point(rng, 14.0, placed, near)
            placed.append((self.food_pos[j].copy(), 14.0))
        self.food_amount = rng.uniform(0.6, 1.0, n_food)

        n_th = int(p['n_thermal'])
        self.th_pos = np.stack([rng.uniform(150, WIDTH - 150, n_th), rng.uniform(120, HEIGHT - 120, n_th)], 1) \
            if n_th else np.zeros((0, 2))
        self.th_r = rng.uniform(110.0, 170.0, n_th)
        self.th_sign = np.array([1.0 if k % 3 != 2 else -1.0 for k in range(n_th)])
        self.th_intensity = rng.uniform(0.7, 1.0, n_th)

        n_wp = int(p['n_waypoints'])
        self.waypoints = np.zeros((n_wp, 2))
        self._new_waypoints(rng)

        self.rule_sign = 1.0
        self.thermal_sign = 1.0
        self.wind_phase = rng.uniform(0, 2 * math.pi, 2)
        self.gust = np.zeros(2)
        self.gust_remaining = 0
        self.t = 0
        self.episodes = ep
        self.rng = np.random.default_rng(self.seed + 7101 + 7919 * ep)
        self._update_landmarks()
        return [self.observation(i) for i in range(n_ag)]

    def _new_waypoints(self, rng: np.random.Generator) -> None:
        for k in range(self.waypoints.shape[0]):
            self.waypoints[k] = self._sample_point(rng, WAYPOINT_RADIUS, ())

    # --------------------------------------------------------------- physics
    def _parse_actions(self, actions: Any) -> np.ndarray:
        """Return (N, 2) [thrust, turn]. Accepts [turn, thrust] 2-d or Sonapse 5-d actions."""
        n = self.n_agents
        if actions is None:
            return np.zeros((n, 2))
        if isinstance(actions, Mapping):
            actions = actions.get('action')
        try:
            arr = np.asarray(actions, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError('actions must be numeric') from exc
        if arr.ndim == 1:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] not in (2, 5):
            raise ValueError(f'actions must have shape ({n},2) or ({n},5)')
        if not np.isfinite(arr).all():
            raise ValueError('actions must be finite')
        arr = np.clip(arr, -1.0, 1.0)
        if arr.shape[1] == 2:          # [turn, thrust]
            return np.stack([arr[:, 1], arr[:, 0]], 1)
        return arr[:, 0:2].copy()       # Sonapse: drive, turn

    def _disturbance(self) -> np.ndarray:
        """Ground-truth wind/current displacement for each agent this step (px)."""
        scale = float(self.params['disturbance_scale'])
        if scale == 0.0:
            return np.zeros((self.n_agents, 2))
        base = 0.6 * np.array([math.cos(0.013 * self.t + self.wind_phase[0]),
                               math.sin(0.009 * self.t + self.wind_phase[1])])
        spatial = 0.6 + 0.4 * np.sin(2.0 * math.pi * self.pos[:, 1] / HEIGHT)
        return scale * (base[None] * spatial[:, None] + self.gust[None])

    def _advance_gust(self) -> None:
        if self.gust_remaining > 0:
            self.gust_remaining -= 1
            if self.gust_remaining == 0:
                self.gust = np.zeros(2)
        elif self.rng.random() < 0.02:
            ang = self.rng.uniform(-math.pi, math.pi)
            mag = self.rng.uniform(0.8, 2.5)
            self.gust = mag * np.array([math.cos(ang), math.sin(ang)])
            self.gust_remaining = int(self.rng.integers(10, 31))

    def _move_obstacles(self) -> None:
        if self.obs_pos.shape[0] == 0:
            return
        self.obs_pos += self.obs_vel
        r = self.obs_r
        for axis, lim in ((0, WIDTH), (1, HEIGHT)):
            lo = self.obs_pos[:, axis] < r
            hi = self.obs_pos[:, axis] > lim - r
            self.obs_vel[lo | hi, axis] *= -1.0
            self.obs_pos[:, axis] = np.clip(self.obs_pos[:, axis], r, lim - r)
        if self.walls.shape[0] == 0:
            return
        for k in range(self.obs_pos.shape[0]):
            d, cl = _segment_point_distance(self.walls, self.obs_pos[k])
            for s in np.nonzero(d < r[k])[0]:
                nrm = self.obs_pos[k] - cl[s]
                nn = np.hypot(*nrm)
                nrm = nrm / nn if nn > 1e-9 else np.array([1.0, 0.0])
                self.obs_pos[k] = cl[s] + nrm * (r[k] + 0.5)
                vn = float(self.obs_vel[k] @ nrm)
                if vn < 0:
                    self.obs_vel[k] -= 2.0 * vn * nrm

    def _thermal_field(self, p: np.ndarray) -> float:
        if self.th_pos.shape[0] == 0:
            return 0.0
        d = np.hypot(*(self.th_pos - p).T)
        v = self.th_sign * self.th_intensity * np.maximum(0.0, 1.0 - d / self.th_r)
        return float(np.clip(self.thermal_sign * v.sum(), -1.0, 1.0))

    def _update_landmarks(self) -> None:
        if self.landmarks.shape[0] == 0:
            return
        d = np.hypot(self.pos[:, None, 0] - self.landmarks[None, :, 0], self.pos[:, None, 1] - self.landmarks[None, :, 1])
        self.landmarks_seen |= d < LANDMARK_VISIT_RADIUS

    def food_visible(self, i: int) -> np.ndarray:
        """Boolean mask of food patches agent ``i``'s sensors can detect."""
        n_food = self.food_pos.shape[0]
        present = self.food_amount > 0.02
        if not self.params['hidden_food'] or n_food == 0:
            return present
        lm = self.food_landmark
        d_lm = np.hypot(*(self.landmarks[lm] - self.pos[i]).T)
        return present & self.landmarks_seen[i, lm] & (d_lm < LANDMARK_REVEAL_RADIUS)

    def step(self, actions: Any) -> tuple[list[dict[str, Any]], list[float], bool, dict[str, Any]]:
        cmd = self._parse_actions(actions)
        p = self.params
        n = self.n_agents
        rewards = np.zeros(n)
        events: list[dict[str, Any]] = []
        self.t += 1

        # Rule flips (shifting_rules).
        period = int(p['flip_period'])
        if period > 0 and self.t % period == 0:
            self.rule_sign *= -1.0
            self.thermal_sign *= -1.0
            events.append({'kind': 'rule_flip', 'step': self.t, 'rule_sign': self.rule_sign})

        self._advance_gust()
        self._move_obstacles()
        dist = self._disturbance()

        alive_start = self.alive.copy()
        dist[~alive_start] = 0.0          # exhausted agents are inert: no push, no further reward
        thrust = np.where(self.alive, cmd[:, 0], 0.0)
        turn = np.where(self.alive, cmd[:, 1], 0.0)
        self.heading = (self.heading + turn * TURN_RATE + math.pi) % (2 * math.pi) - math.pi
        self.speed = np.where(thrust >= 0, thrust, thrust * REVERSE_FRACTION) * MAX_SPEED
        self.turn = turn
        self.vel = np.stack([np.cos(self.heading) * self.speed, np.sin(self.heading) * self.speed], 1)
        self.pos = self.pos + self.vel + dist
        self.disturbance_last = dist.copy()

        self.collided_last[:] = 0
        contact = np.zeros(n, dtype=bool)
        # Boundary + wall (line-circle) collisions.
        for i in range(n):
            wall_hit = False
            lo = self.pos[i] < AGENT_RADIUS
            hi = self.pos[i] > np.array([WIDTH, HEIGHT]) - AGENT_RADIUS
            if lo.any() or hi.any():
                wall_hit = True
                self.pos[i] = np.clip(self.pos[i], AGENT_RADIUS, [WIDTH - AGENT_RADIUS, HEIGHT - AGENT_RADIUS])
            if self.walls.shape[0]:
                d, cl = _segment_point_distance(self.walls, self.pos[i])
                for s in np.nonzero(d < AGENT_RADIUS)[0]:
                    nrm = self.pos[i] - cl[s]
                    nn = np.hypot(*nrm)
                    nrm = nrm / nn if nn > 1e-9 else np.array([1.0, 0.0])
                    self.pos[i] = cl[s] + nrm * (AGENT_RADIUS + 0.5)
                    wall_hit = True
            if wall_hit:
                contact[i] = True
                if not self.wall_contact[i]:
                    self.collisions[i] += 1
                    self.collided_last[i] += 1
                    rewards[i] -= COLLISION_PENALTY
                    events.append({'kind': 'collision', 'agent': i, 'with': 'wall', 'step': self.t})
            self.wall_contact[i] = wall_hit
        # Agent-obstacle circle-circle collisions.
        if self.obs_pos.shape[0]:
            for i in range(n):
                delta = self.pos[i] - self.obs_pos
                d = np.hypot(*delta.T)
                touching = d < self.obs_r + AGENT_RADIUS
                for k in np.nonzero(touching)[0]:
                    nrm = delta[k] / d[k] if d[k] > 1e-9 else np.array([1.0, 0.0])
                    self.pos[i] = self.obs_pos[k] + nrm * (self.obs_r[k] + AGENT_RADIUS + 0.5)
                    if not self.obs_contact[i, k]:
                        self.collisions[i] += 1
                        self.collided_last[i] += 1
                        rewards[i] -= COLLISION_PENALTY
                        events.append({'kind': 'collision', 'agent': i, 'with': 'obstacle', 'index': int(k), 'step': self.t})
                contact[i] |= bool(touching.any())
                self.obs_contact[i] = touching
                self.pos[i] = np.clip(self.pos[i], AGENT_RADIUS, [WIDTH - AGENT_RADIUS, HEIGHT - AGENT_RADIUS])
        # Agent-agent contact (multi-agent).
        for i in range(n):
            for j in range(i + 1, n):
                delta = self.pos[i] - self.pos[j]
                d = float(np.hypot(*delta))
                touching = d < 2 * AGENT_RADIUS
                if touching:
                    nrm = delta / d if d > 1e-9 else np.array([1.0, 0.0])
                    push = 0.5 * (2 * AGENT_RADIUS - d) + 0.25
                    self.pos[i] += nrm * push
                    self.pos[j] -= nrm * push
                    contact[i] = contact[j] = True
                    if not self.agent_contact[i, j]:
                        rewards[i] -= AGENT_CONTACT_PENALTY
                        rewards[j] -= AGENT_CONTACT_PENALTY
                        events.append({'kind': 'agent_contact', 'agents': [i, j], 'step': self.t})
                self.agent_contact[i, j] = self.agent_contact[j, i] = touching
        self.pos = np.clip(self.pos, AGENT_RADIUS, [WIDTH - AGENT_RADIUS, HEIGHT - AGENT_RADIUS])
        self.contact = contact
        self.contact_steps += contact.astype(np.int64)

        # Eating (agents processed in index order -> competition for patches).
        self.eaten_last[:] = 0.0
        if self.food_pos.shape[0]:
            for i in range(n):
                if not self.alive[i]:
                    continue
                d = np.hypot(*(self.food_pos - self.pos[i]).T)
                for j in np.nonzero((d < self.food_r + AGENT_RADIUS) & (self.food_amount > 0.02))[0]:
                    bite = float(min(self.food_amount[j], EAT_BITE))
                    self.food_amount[j] -= bite
                    self.eaten_last[i] += bite * self.rule_sign
                    self.food_eaten[i] += bite
                    self.bites[i] += 1
                    rewards[i] += FOOD_REWARD * bite * self.rule_sign
                    self.energy[i] += FOOD_ENERGY * bite * self.rule_sign
            self.food_amount = np.minimum(1.0, self.food_amount + float(p['regen']))

        # Thermal + energy.
        for i in range(n):
            th = self._thermal_field(self.pos[i])
            self.thermal_last[i] = th
            if not self.alive[i]:
                continue
            self.energy[i] -= float(p['energy_decay']) + float(p['move_cost']) * abs(thrust[i]) \
                + float(p['thermal_cost']) * abs(th)
            if abs(th) > THERMAL_STRESS_THRESHOLD:
                rewards[i] -= THERMAL_STRESS_PENALTY * abs(th)
        self.energy = np.clip(self.energy, 0.0, MAX_ENERGY)

        # Waypoints (long_horizon, delayed reward).
        n_wp = self.waypoints.shape[0]
        if n_wp:
            for i in range(n):
                k = int(self.waypoint_index[i])
                if self.alive[i] and np.hypot(*(self.waypoints[k] - self.pos[i])) < WAYPOINT_RADIUS:
                    self.waypoint_index[i] = k + 1
                    events.append({'kind': 'waypoint', 'agent': i, 'index': k, 'step': self.t})
                    if k + 1 >= n_wp:
                        rewards[i] += float(p['completion_reward'])
                        self.sequences_completed[i] += 1
                        self.waypoint_index[:] = 0
                        self._new_waypoints(self.rng)
                        events.append({'kind': 'sequence_complete', 'agent': i, 'step': self.t})
                        break

        self._update_landmarks()

        rewards[~alive_start] = 0.0
        newly_dead = self.alive & (self.energy <= 0.0)
        rewards[newly_dead] -= EXHAUSTION_PENALTY
        for i in np.nonzero(newly_dead)[0]:
            events.append({'kind': 'exhausted', 'agent': int(i), 'step': self.t})
        self.alive &= ~newly_dead

        self.last_reward = rewards.copy()
        self.total_reward += rewards
        done = bool((not self.alive.any()) or self.t >= int(p['max_steps']))
        info = {
            'step': self.t,
            'config': self.config,
            'collisions': self.collided_last.tolist(),
            'collisions_total': self.collisions.tolist(),
            'eaten': self.eaten_last.tolist(),
            'energy': self.energy.tolist(),
            'thermal': self.thermal_last.tolist(),
            'contact': self.contact.tolist(),
            'disturbance': self.disturbance_last.tolist(),
            'rule_sign': self.rule_sign,
            'thermal_sign': self.thermal_sign,
            'waypoint_index': self.waypoint_index.tolist(),
            'alive': self.alive.tolist(),
            'events': events,
            'evidence_tier': EVIDENCE_TIER,
        }
        if done:
            self.episodes += 1
        return [self.observation(i) for i in range(n)], rewards.tolist(), done, info

    # ---------------------------------------------------------------- senses
    def _cast(self, i: int, angles: np.ndarray, rng_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cast rays for agent i. Returns (distance, hit type, colour (R,3), hit index)."""
        ox, oy = float(self.pos[i, 0]), float(self.pos[i, 1])
        dx, dy = np.cos(angles), np.sin(angles)
        R = angles.shape[0]
        best = np.minimum(_ray_box(ox, oy, dx, dy), rng_max)
        kind = np.where(best < rng_max, HIT_BOUNDARY, HIT_NONE).astype(np.int64)
        index = np.full(R, -1, dtype=np.int64)

        def take(t: np.ndarray, code: int) -> None:
            nonlocal best
            if t.shape[1] == 0:
                return
            tmin = t.min(1)
            arg = t.argmin(1)
            m = tmin < best
            best = np.where(m, tmin, best)
            kind[m] = code
            index[m] = arg[m]

        take(_ray_segments(ox, oy, dx, dy, self.walls), HIT_WALL)
        take(_ray_circles(ox, oy, dx, dy, self.obs_pos, self.obs_r), HIT_OBSTACLE)
        vis = self.food_visible(i)
        fidx = np.nonzero(vis)[0]
        tf = _ray_circles(ox, oy, dx, dy, self.food_pos[fidx], self.food_r[fidx])
        if tf.shape[1]:
            tmin = tf.min(1)
            m = tmin < best
            best = np.where(m, tmin, best)
            kind[m] = HIT_FOOD
            index[m] = fidx[tf.argmin(1)][m]
        take(_ray_circles(ox, oy, dx, dy, self.landmarks, self.landmark_r), HIT_LANDMARK)
        if self.n_agents > 1:
            others = np.array([j for j in range(self.n_agents) if j != i])
            ta = _ray_circles(ox, oy, dx, dy, self.pos[others], np.full(others.shape[0], AGENT_RADIUS))
            tmin = ta.min(1)
            m = tmin < best
            best = np.where(m, tmin, best)
            kind[m] = HIT_AGENT
            index[m] = others[ta.argmin(1)][m]
        if self.waypoints.shape[0]:
            k = int(self.waypoint_index[i])
            take(_ray_circles(ox, oy, dx, dy, self.waypoints[k:k + 1], np.array([WAYPOINT_RADIUS])), HIT_WAYPOINT)
        colour = np.array([_COLOURS[int(c)] for c in kind], dtype=np.float64).reshape(R, 3)
        if self.rule_sign < 0:
            colour[kind == HIT_FOOD] = _TOXIC_FOOD_COLOUR
        return best, kind, colour, index

    def rays(self, i: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Egocentric ring of N_RAYS rays: (distance / RAY_RANGE in [0,1], hit type)."""
        self._check_agent(i)
        angles = self.heading[i] + np.arange(N_RAYS) * (RAY_FOV / N_RAYS)
        d, kind, _, _ = self._cast(i, angles, RAY_RANGE)
        return d / RAY_RANGE, kind

    def food_gradient(self, i: int) -> tuple[float, float, float]:
        """(smell intensity, egocentric lateral, egocentric forward) from sensor-visible food."""
        vis = self.food_visible(i)
        if not vis.any():
            return 0.0, 0.0, 0.0
        delta = self.food_pos[vis] - self.pos[i]
        d = np.maximum(np.hypot(*delta.T), 1e-6)
        w = self.food_amount[vis] * np.exp(-d / SMELL_SCALE)
        g = (w[:, None] * delta / d[:, None]).sum(0)
        c, s = math.cos(self.heading[i]), math.sin(self.heading[i])
        forward = g[0] * c + g[1] * s
        lateral = -g[0] * s + g[1] * c
        norm = max(1e-9, math.hypot(forward, lateral))
        mag = math.tanh(float(w.sum()))
        return mag, float(lateral / norm * mag), float(forward / norm * mag)

    def _target_direction(self, i: int) -> tuple[float, float]:
        if self.waypoints.shape[0]:
            k = int(self.waypoint_index[i])
            delta = self.waypoints[k] - self.pos[i]
            c, s = math.cos(self.heading[i]), math.sin(self.heading[i])
            fwd = delta[0] * c + delta[1] * s
            lat = -delta[0] * s + delta[1] * c
            n = max(1e-9, math.hypot(fwd, lat))
            return float(lat / n), float(fwd / n)
        _, lat, fwd = self.food_gradient(i)
        return lat, fwd

    def _check_agent(self, i: int) -> None:
        if not isinstance(i, (int, np.integer)) or isinstance(i, bool) or not 0 <= int(i) < self.n_agents:
            raise ValueError(f'agent index must be in [0, {self.n_agents})')

    def observation(self, i: int) -> dict[str, Any]:
        self._check_agent(i)
        dist, kind = self.rays(i)
        smell, lat, fwd = self.food_gradient(i)
        return {
            'agent': int(i),
            'step': self.t,
            'rays': dist.tolist(),
            'ray_types': kind.tolist(),
            'position': [float(self.pos[i, 0] / WIDTH), float(self.pos[i, 1] / HEIGHT)],
            'heading': float(self.heading[i]),
            'energy': float(self.energy[i] / MAX_ENERGY),
            'thermal': float(self.thermal_last[i]),
            'contact': bool(self.contact[i]),
            'food_smell': smell,
            'food_gradient': [lat, fwd],
            'speed': float(self.speed[i] / MAX_SPEED),
            'turn': float(self.turn[i]),
            'eaten': float(self.eaten_last[i]),
            'disturbance': self.disturbance_last[i].tolist(),
            'alive': bool(self.alive[i]),
        }

    def _task_context(self, i: int) -> float:
        cfg_code = CONFIG_NAMES.index(self.config) / (len(CONFIG_NAMES) - 1) * 2.0 - 1.0
        phase = 0.0
        if int(self.params['flip_period']) > 0:
            phase = self.rule_sign
        elif self.waypoints.shape[0]:
            phase = self.waypoint_index[i] / self.waypoints.shape[0] * 2.0 - 1.0
        elif self.params['hidden_food']:
            phase = float(self.landmarks_seen[i].mean()) * 2.0 - 1.0
        return float(np.clip(0.6 * cfg_code + 0.4 * phase, -1.0, 1.0))

    def sensory_vector(self, i: int = 0) -> np.ndarray:
        """Sixteen synthetic channels in [-1, 1], ordered as spec.SENSORY_CHANNELS."""
        self._check_agent(i)
        angles = self.heading[i] + np.arange(N_RAYS) * (RAY_FOV / N_RAYS)
        d, kind, _, index = self._cast(i, angles, RAY_RANGE)
        prox = 1.0 - d / RAY_RANGE
        bright = np.array([_BRIGHTNESS[int(k)] for k in kind])
        luminance = float(np.mean(prox * bright))
        food_p = float(np.sum(prox[kind == HIT_FOOD]))
        obst_p = float(np.sum(prox[kind == HIT_OBSTACLE]))
        red_green = (food_p - obst_p) / N_RAYS * 4.0
        cool = float(np.sum(prox[(kind == HIT_LANDMARK) | (kind == HIT_AGENT) | (kind == HIT_WAYPOINT)]))
        warm = float(np.sum(prox[(kind == HIT_WALL) | (kind == HIT_BOUNDARY)]))
        blue_yellow = (cool - warm) / N_RAYS * 2.0
        edge = float(np.std(d / RAY_RANGE)) * 2.0
        # Obstacle closing speed seen by obstacle-hit rays (positive = approaching).
        approach = 0.0
        hits = np.nonzero(kind == HIT_OBSTACLE)[0]
        if hits.size:
            ks = index[hits]
            delta = self.obs_pos[ks] - self.pos[i]
            dn = np.maximum(np.hypot(*delta.T), 1e-6)
            rel = self.obs_vel[ks] - self.vel[i]
            approach = float(np.mean(-(delta * rel).sum(1) / dn))
        motion = math.tanh(approach / 3.0)
        # Synthetic audio: obstacle rumble, own locomotion, impact.
        if self.obs_pos.shape[0]:
            od = np.hypot(*(self.obs_pos - self.pos[i]).T)
            ospd = np.hypot(*self.obs_vel.T)
            rumble = float(np.sum(ospd * np.maximum(0.0, 1.0 - od / 300.0)))
            nearest = float(od.min() - self.obs_r[od.argmin()] - AGENT_RADIUS)
        else:
            rumble, nearest = 0.0, RAY_RANGE
        audio_low = math.tanh(rumble / 2.0)
        audio_mid = abs(self.speed[i]) / MAX_SPEED
        audio_high = min(1.0, float(self.collided_last[i]) + max(0.0, 0.5 * motion) * (nearest < 60.0))
        smell, _, _ = self.food_gradient(i)
        vec = np.array([
            luminance, red_green, blue_yellow, edge, motion,
            audio_low, audio_mid, audio_high,
            smell, float(np.clip(self.eaten_last[i] / EAT_BITE, -1, 1)),
            self.thermal_last[i], self.energy[i] / MAX_ENERGY * 2.0 - 1.0,
            0.7 * self.speed[i] / MAX_SPEED + 0.3 * self.turn[i],
            1.0 if self.contact[i] else 0.0,
            math.tanh(float(np.hypot(*self.disturbance_last[i])) / 1.5),
            self._task_context(i),
        ], dtype=np.float64)
        assert vec.shape[0] == len(SENSORY_CHANNEL_IDS)
        return np.clip(np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    def sensory_dict(self, i: int = 0) -> dict[str, float]:
        return dict(zip(SENSORY_CHANNEL_IDS, self.sensory_vector(i).tolist()))

    def to_synapse_observation(self, obs: int | Mapping[str, Any] = 0) -> np.ndarray:
        """Map agent state into Sonapse's 155-d observation layout (see module docstring).

        ``obs`` is an agent index or an observation dict returned by ``reset``/``step``
        for the *current* step (its ``agent`` key selects the agent).
        """
        if isinstance(obs, Mapping):
            if obs.get('step') != self.t:
                raise ValueError('observation dict is stale; pass the current step observation')
            i = obs.get('agent')
        else:
            i = obs
        self._check_agent(i)
        i = int(i)
        angles = self.heading[i] + np.linspace(-RETINA_FOV / 2, RETINA_FOV / 2, RETINA_RAYS)
        d, _, colour, _ = self._cast(i, angles, RETINA_RANGE)
        depth = 1.0 - d / RETINA_RANGE
        prop = [0.0, 0.0, 0.0, 0.0, self.speed[i] / MAX_SPEED, self.turn[i] * TURN_RATE / 0.25,
                math.sin(self.heading[i]), math.cos(self.heading[i])]
        onehot = np.zeros(len(GRAFTBRAIN_TASKS))
        onehot[GRAFTBRAIN_TASKS.index(CONFIG_TASK[self.config])] = 1.0
        lat, fwd = self._target_direction(i)
        cue = [0.5 + 0.5 * lat, 0.5 + 0.5 * fwd, self.energy[i] / MAX_ENERGY, 1.0]
        out = np.concatenate([depth, colour[:, 0], colour[:, 1], colour[:, 2], prop, onehot, cue,
                              [float(np.clip(self.last_reward[i], -10, 10)), self.t / int(self.params['max_steps'])]])
        out = np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        assert out.shape == (155,)
        return out

    # ------------------------------------------------------- Sonapse driver
    def run_synapse_agent(self, agent: Any, steps: int, learn: bool = False, agent_index: int = 0,
                          reset_on_done: bool = True) -> dict[str, Any]:
        """Drive a Sonapse agent (``act(obs, learn=...)`` / ``feedback(reward, done)``).

        Mirrors ``synapse.runtime`` stepping: the actor action is governed by the
        agent's homeostatic thrust limit when present, homeostasis is stepped
        with the transition, and learning hooks run only when ``learn`` is True
        (otherwise the pending transition is discarded so nothing is learnt).
        Other agents in a multi-agent arena receive zero actions.
        """
        if type(steps) is not int or not 1 <= steps <= 100000:
            raise ValueError('steps must be an integer in [1, 100000]')
        if type(learn) is not bool:
            raise ValueError('learn must be bool')
        self._check_agent(agent_index)
        from synapse.actions import govern_action, validate_action  # numpy-only Sonapse contract

        total_reward = 0.0
        food_eaten = 0.0
        bites = 0
        collisions = 0
        energies: list[float] = []
        rewards: list[float] = []
        episodes = 0
        actions_finite = True
        obs = self.to_synapse_observation(agent_index)
        for _ in range(steps):
            out = agent.act(obs, learn=learn)
            proposed = out['action'] if isinstance(out, Mapping) else out
            homeo = getattr(agent, 'homeostasis', None)
            if homeo is not None and hasattr(homeo, 'get_max_allowed_thrust'):
                act = govern_action(proposed, homeo.get_max_allowed_thrust())
            else:
                act = np.asarray(proposed, dtype=np.float64)
                act = validate_action(act) if act.shape == (5,) else np.clip(act, -1, 1)
            actions_finite &= bool(np.isfinite(act).all())
            full = np.zeros((self.n_agents, act.shape[0]))
            full[agent_index] = act
            pending = getattr(agent, 'pending_transition', None)
            if isinstance(pending, dict):
                pending['action'] = np.asarray(act).copy()
                pending['learning'] = learn
            _, rew, done, info = self.step(full)
            r = float(rew[agent_index])
            obs = self.to_synapse_observation(agent_index)
            if homeo is not None and hasattr(homeo, 'step') and np.asarray(act).shape == (5,):
                od = np.hypot(*(self.obs_pos - self.pos[agent_index]).T) if self.obs_pos.shape[0] else np.array([1200.0])
                homeo.step(applied_action=act, reward=r, hazard_dist=float(od.min() / 120.0),
                           consumed_food=bool(self.eaten_last[agent_index] > 0), in_friction_zone=False)
            if learn:
                if hasattr(agent, 'observe_transition'):
                    agent.observe_transition(obs, r, done, action=act)
                agent.feedback(r, done)
            elif hasattr(agent, 'pending_transition'):
                agent.pending_transition = None
            total_reward += r
            food_eaten += abs(float(self.eaten_last[agent_index]))
            bites += int(self.eaten_last[agent_index] != 0.0)
            collisions += int(self.collided_last[agent_index])
            rewards.append(r)
            energies.append(float(self.energy[agent_index]))
            if done:
                episodes += 1
                if not reset_on_done:
                    break
                self.reset()
                obs = self.to_synapse_observation(agent_index)
        return {
            'steps': len(rewards),
            'total_reward': total_reward,
            'food_eaten': food_eaten,
            'eating_steps': bites,
            'collisions': collisions,
            'final_energy': energies[-1] / MAX_ENERGY,
            'mean_energy': float(np.mean(energies)) / MAX_ENERGY,
            'min_energy': float(np.min(energies)) / MAX_ENERGY,
            'episodes_finished': episodes,
            'actions_finite': actions_finite,
            'rewards': rewards,
            'learn': learn,
            'evidence_tier': EVIDENCE_TIER,
        }

    # ------------------------------------------------------------ persistence
    _ARRAYS: dict[str, tuple[str, tuple[Any, ...]]] = {
        # name: (dtype kind, shape spec using 'A' agents, 'O' obstacles, 'F' food, 'Z' thermal,
        #        'L' landmarks, 'W' waypoints, 'S' walls)
        'pos': ('f', ('A', 2)), 'heading': ('f', ('A',)), 'vel': ('f', ('A', 2)), 'speed': ('f', ('A',)),
        'turn': ('f', ('A',)), 'energy': ('f', ('A',)), 'alive': ('b', ('A',)), 'contact': ('b', ('A',)),
        'collisions': ('i', ('A',)), 'contact_steps': ('i', ('A',)), 'collided_last': ('i', ('A',)),
        'food_eaten': ('f', ('A',)), 'bites': ('i', ('A',)), 'eaten_last': ('f', ('A',)),
        'total_reward': ('f', ('A',)), 'last_reward': ('f', ('A',)), 'disturbance_last': ('f', ('A', 2)),
        'thermal_last': ('f', ('A',)), 'waypoint_index': ('i', ('A',)), 'sequences_completed': ('i', ('A',)),
        'obs_pos': ('f', ('O', 2)), 'obs_vel': ('f', ('O', 2)), 'obs_r': ('f', ('O',)),
        'obs_contact': ('b', ('A', 'O')), 'wall_contact': ('b', ('A',)), 'agent_contact': ('b', ('A', 'A')),
        'landmarks': ('f', ('L', 2)), 'landmark_r': ('f', ('L',)), 'landmarks_seen': ('b', ('A', 'L')),
        'food_pos': ('f', ('F', 2)), 'food_r': ('f', ('F',)), 'food_landmark': ('i', ('F',)),
        'food_amount': ('f', ('F',)), 'th_pos': ('f', ('Z', 2)), 'th_r': ('f', ('Z',)),
        'th_sign': ('f', ('Z',)), 'th_intensity': ('f', ('Z',)), 'waypoints': ('f', ('W', 2)),
        'walls': ('f', ('S', 4)), 'wind_phase': ('f', (2,)), 'gust': ('f', (2,)),
    }

    def export_state(self) -> dict[str, Any]:
        arrays = {name: getattr(self, name).tolist() for name in self._ARRAYS}
        return {
            'schema': ECOLOGY_SCHEMA,
            'version': ECOLOGY_VERSION,
            'config': self.config,
            'seed': self.seed,
            'params': {k: (bool(v) if isinstance(v, bool) else (int(v) if _PARAM_TYPES[k] is int else float(v)))
                       for k, v in self.params.items()},
            't': int(self.t),
            'episodes': int(self.episodes),
            'rule_sign': float(self.rule_sign),
            'thermal_sign': float(self.thermal_sign),
            'gust_remaining': int(self.gust_remaining),
            'rng': copy.deepcopy(self.rng.bit_generator.state),
            'arrays': arrays,
            'evidence_tier': EVIDENCE_TIER,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError('state must be a mapping')
        if state.get('schema') != ECOLOGY_SCHEMA or state.get('version') != ECOLOGY_VERSION:
            raise ValueError('unsupported ecology state schema/version')
        config = state.get('config')
        if config not in CONFIGS:
            raise ValueError('unknown config in state')
        seed = state.get('seed')
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('seed must be a non-negative int')
        params = state.get('params')
        if not isinstance(params, Mapping):
            raise ValueError('params must be a mapping')
        params = dict(params)
        self._validate_params(params)
        for key in ('t', 'episodes', 'gust_remaining'):
            v = state.get(key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f'{key} must be a non-negative int')
        for key in ('rule_sign', 'thermal_sign'):
            v = state.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v not in (1.0, -1.0):
                raise ValueError(f'{key} must be +/-1')
        rng_state = state.get('rng')
        rng = np.random.default_rng(0)
        try:
            rng.bit_generator.state = copy.deepcopy(dict(rng_state))  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError('invalid rng state') from exc
        arrays = state.get('arrays')
        if not isinstance(arrays, Mapping) or set(arrays) != set(self._ARRAYS):
            raise ValueError('arrays must contain exactly the documented keys')
        dims = {'A': int(params['agents']), 'O': int(params['n_obstacles']), 'F': int(params['n_food']),
                'Z': int(params['n_thermal']), 'L': int(params['n_landmarks']), 'W': int(params['n_waypoints']),
                'S': len(_MAZE_WALLS) if params['walls'] else 0}
        parsed: dict[str, np.ndarray] = {}
        for name, (kind, shape_spec) in self._ARRAYS.items():
            shape = tuple(dims[s] if isinstance(s, str) else s for s in shape_spec)
            raw = arrays[name]
            dtype = {'b': bool, 'i': np.int64, 'f': np.float64}[kind]
            try:
                obj = np.asarray(raw, dtype=object)
                if obj.size == 0:
                    # An empty list is only acceptable when the expected array is empty too.
                    if int(np.prod(shape)) != 0:
                        raise ValueError
                    arr = np.zeros(shape, dtype=dtype)
                else:
                    leaves = obj.ravel()
                    if kind == 'b':
                        ok = all(isinstance(x, (bool, np.bool_)) for x in leaves)
                    elif kind == 'i':
                        ok = all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                                 for x in leaves)
                    else:
                        ok = all(isinstance(x, (int, float, np.integer, np.floating))
                                 and not isinstance(x, (bool, np.bool_)) for x in leaves)
                    if not ok:
                        raise ValueError
                    arr = obj.astype(dtype)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'array {name} has invalid values') from exc
            if arr.shape != shape:
                raise ValueError(f'array {name} has shape {arr.shape}, expected {shape}')
            if kind == 'f' and not np.isfinite(arr).all():
                raise ValueError(f'array {name} must be finite')
            parsed[name] = arr
        # Semantic bounds.
        if np.any(parsed['energy'] < 0) or np.any(parsed['energy'] > MAX_ENERGY):
            raise ValueError('energy out of bounds')
        if np.any(parsed['food_amount'] < 0) or np.any(parsed['food_amount'] > 1):
            raise ValueError('food_amount out of bounds')
        p = parsed['pos']
        if np.any(p[:, 0] < 0) or np.any(p[:, 0] > WIDTH) or np.any(p[:, 1] < 0) or np.any(p[:, 1] > HEIGHT):
            raise ValueError('agent position outside arena')
        if np.any(parsed['waypoint_index'] < 0) or (dims['W'] and np.any(parsed['waypoint_index'] >= dims['W'])):
            raise ValueError('waypoint_index out of range')
        if np.any(parsed['food_landmark'] < -1) or np.any(parsed['food_landmark'] >= dims['L']):
            raise ValueError('food_landmark out of range')
        if params['hidden_food'] and np.any(parsed['food_landmark'] < 0):
            raise ValueError('hidden_food requires every food to have a landmark')
        if np.any(np.abs(parsed['th_sign']) != 1.0):
            raise ValueError('th_sign must be +/-1')
        for name in ('obs_r', 'food_r', 'th_r', 'landmark_r'):
            if np.any(parsed[name] <= 0) or np.any(parsed[name] > _MAX_RADIUS):
                raise ValueError(f'{name} must be in (0, {_MAX_RADIUS}]')
        for name in ('collisions', 'contact_steps', 'collided_last', 'bites', 'sequences_completed'):
            if np.any(parsed[name] < 0):
                raise ValueError(f'{name} must be non-negative')
        if np.any(parsed['food_eaten'] < 0) or np.any(np.abs(parsed['th_intensity']) > 1.0):
            raise ValueError('food_eaten/th_intensity out of bounds')
        if parsed['obs_vel'].size and np.any(np.hypot(*parsed['obs_vel'].T) > _MAX_OBSTACLE_SPEED):
            raise ValueError('obstacle velocity out of bounds')
        if np.hypot(*parsed['gust']) > _MAX_GUST or np.any(np.abs(parsed['speed']) > MAX_SPEED + 1e-9) \
                or np.any(np.abs(parsed['turn']) > 1.0 + 1e-9):
            raise ValueError('gust/speed/turn out of bounds')
        if params['walls'] and not np.array_equal(parsed['walls'], np.array(_MAZE_WALLS, dtype=np.float64)):
            raise ValueError('walls must match the maze layout')
        # Commit.
        self.config = config
        self.seed = seed
        self.params = params
        self.t = int(state['t'])
        self.episodes = int(state['episodes'])
        self.rule_sign = float(state['rule_sign'])
        self.thermal_sign = float(state['thermal_sign'])
        self.gust_remaining = int(state['gust_remaining'])
        self.rng = rng
        for name, arr in parsed.items():
            setattr(self, name, arr.copy())

    def clone(self) -> 'NeuroWeaveEcology':
        other = NeuroWeaveEcology.__new__(NeuroWeaveEcology)
        other.import_state(self.export_state())
        return other

    def summary(self) -> dict[str, Any]:
        return {
            'config': self.config, 'step': self.t, 'agents': self.n_agents,
            'width': WIDTH, 'height': HEIGHT,
            'obstacles': int(self.obs_pos.shape[0]), 'food_patches': int(self.food_pos.shape[0]),
            'thermal_zones': int(self.th_pos.shape[0]), 'walls': int(self.walls.shape[0]),
            'collisions': self.collisions.tolist(), 'food_eaten': self.food_eaten.tolist(),
            'energy': (self.energy / MAX_ENERGY).tolist(), 'rule_sign': self.rule_sign,
            'raw_media_stored': False, 'evidence_tier': EVIDENCE_TIER,
        }


__all__ = ['NeuroWeaveEcology', 'CONFIGS', 'CONFIG_NAMES', 'CONFIG_TASK', 'GRAFTBRAIN_TASKS', 'config_params',
           'WIDTH', 'HEIGHT', 'N_RAYS', 'ECOLOGY_SCHEMA', 'HIT_NAMES']
