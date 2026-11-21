"""Developmental growth and regeneration of a connectome (frontier module ``morphogenesis``).

Inspired by the user's morpheus, prometheus-alpha, GenesisEngine and MOLT projects
(kai9987kai), by Growing Neural Cellular Automata (Mordvintsev, Randazzo,
Niklasson & Levin, Distill, 2020) and by the bioelectric "pattern memory" view of
regeneration (Pezzulo & Levin, Integrative Biology, 2015). It is a synthetic
abstraction: no claim of biological fidelity and no reproduction of either paper.

Model
-----
* A 2-D grid of sites. A body grows from one seed cell (the *organizer*). Every
  step, an empty site next to living tissue becomes alive, and a living cell
  dies, according to a *growth logit* evaluated under a seeded stochastic update
  mask (as in neural cellular automata, only a random subset of sites updates)::

      logit = theta . [1, n/8, dx^2, dy^2, dx^4]

  where ``n`` is the number of living 8-neighbours and ``(dx, dy)`` is the
  position relative to the organizer in units of a quarter of the grid side.
  The developmental program is the five numbers ``theta``. A site is born when
  ``logit > 0``; a living cell dies when ``logit < -death_margin``.
* Cell types come from positional information around the organizer
  (French-flag style): ``sensory`` at one pole, ``action`` at the other,
  ``memory`` in the core and ``association`` elsewhere.
* :func:`body_to_topology` turns a body into a valid
  :class:`synapse.neuroweave.topology.Topology` (cells -> neurons, types ->
  regions, proximity and type-rank rules -> edges, distance -> delays), so the
  NEUROWEAVE graph metrics apply.
* :func:`evolve_program` fits ``theta`` to a target shape with a seeded
  (1 + lambda) evolution strategy.
* Regeneration: a lesion removes a block of tissue (possibly the organizer).
  Regrowth runs the same developmental rule with (a) a stored **pattern memory**
  (a coarse setpoint map of the intact body and the organizer position, held by
  every surviving cell), (b) no memory (the organizer, if lost, is re-derived
  from the centroid of the surviving tissue), or (c) a capacity-matched control
  memory with the same size but shuffled content.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..neuroweave.spec import REGIONS
from ..neuroweave.topology import KIND_RANK, WEIGHT_MIN, WEIGHT_SPAN, Topology, topology_metrics
from . import harness

__all__ = ['MODULE_ID', 'CELL_TYPES', 'TARGETS', 'DevelopmentalProgram', 'PatternMemory', 'Body', 'target_mask',
           'grow', 'lesion', 'regrow', 'make_memory', 'shuffled_memory', 'body_to_topology', 'iou',
           'edge_jaccard', 'evolve_program', 'run_experiment', 'EXPERIMENTS', 'DEFAULTS']

MODULE_ID = 'morphogenesis'
CELL_TYPES = ('sensory', 'association', 'memory', 'action')
TYPE_REGION = (0, 3, 5, 8)        # visual_sensory, association_a, working_memory, premotor (REGIONS indices)
TARGETS = ('two_lobes', 'ellipse', 'cross')
N_PARAMS = 5
MIN_GRID, MAX_GRID = 8, 48
MAX_STEPS = 400
_GROW_OFFSET = 31_337
_ES_OFFSET = 271_828
_LESION_OFFSET = 161_803
_SHUFFLE_OFFSET = 141_421
EMPTY = -1

for _t, _r in zip(CELL_TYPES, TYPE_REGION, strict=True):
    assert REGIONS[_r][1] == _t, (REGIONS[_r], _t)


def _check_grid(size: Any) -> int:
    if type(size) is not int or not MIN_GRID <= size <= MAX_GRID:
        raise ValueError(f'grid size must be an int in [{MIN_GRID}, {MAX_GRID}]')
    return size


def _check_seed(seed: Any) -> int:
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError('seed must be a non-negative int')
    return seed


@dataclass(frozen=True)
class DevelopmentalProgram:
    """Five-number growth rule plus fixed update settings."""

    theta: tuple[float, ...] = (2.0, 1.0, -1.2, -1.2, 0.0)
    p_update: float = 0.5
    death_margin: float = 2.0

    def __post_init__(self) -> None:
        th = tuple(float(v) for v in self.theta)
        if len(th) != N_PARAMS or not all(math.isfinite(v) and abs(v) <= 50.0 for v in th):
            raise ValueError(f'theta must hold {N_PARAMS} finite numbers with |value| <= 50')
        if not isinstance(self.p_update, (int, float)) or not 0.05 <= float(self.p_update) <= 1.0:
            raise ValueError('p_update must lie in [0.05, 1]')
        if not isinstance(self.death_margin, (int, float)) or not 0.0 <= float(self.death_margin) <= 50.0:
            raise ValueError('death_margin must lie in [0, 50]')
        object.__setattr__(self, 'theta', th)
        object.__setattr__(self, 'p_update', float(self.p_update))
        object.__setattr__(self, 'death_margin', float(self.death_margin))

    def to_dict(self) -> dict[str, Any]:
        return {'theta': list(self.theta), 'p_update': self.p_update, 'death_margin': self.death_margin}


@dataclass(frozen=True)
class PatternMemory:
    """Coarse setpoint map (block-averaged body mask) plus the organizer position, held by every cell."""

    setpoint: np.ndarray           # (size // block, size // block) in [0, 1]
    block: int
    organizer: tuple[int, int]
    strength: float = 3.0

    def field(self, size: int) -> np.ndarray:
        """Up-sampled setpoint in [0, 1] at grid resolution."""
        up = np.kron(self.setpoint, np.ones((self.block, self.block)))
        out = np.zeros((size, size))
        n = min(size, up.shape[0])
        out[:n, :n] = up[:n, :n]
        return out

    @property
    def numbers(self) -> int:
        return int(self.setpoint.size + 2)


@dataclass
class Body:
    alive: np.ndarray              # (size, size) bool
    cell_type: np.ndarray          # (size, size) int8, EMPTY where dead
    organizer: tuple[int, int]

    @property
    def size(self) -> int:
        return int(self.alive.shape[0])

    @property
    def cells(self) -> int:
        return int(self.alive.sum())

    def copy(self) -> Body:
        return Body(self.alive.copy(), self.cell_type.copy(), tuple(self.organizer))


# ---------------------------------------------------------------------- geometry

def target_mask(name: str, size: int) -> np.ndarray:
    """Boolean target morphology centred on the grid."""
    size = _check_grid(size)
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = (xx - c) / size, (yy - c) / size
    if name == 'two_lobes':
        r = 0.2
        mask = ((dx - 0.17) ** 2 + dy ** 2 <= r * r) | ((dx + 0.17) ** 2 + dy ** 2 <= r * r)
    elif name == 'ellipse':
        mask = (dx / 0.4) ** 2 + (dy / 0.25) ** 2 <= 1.0
    elif name == 'cross':
        mask = ((np.abs(dx) <= 0.12) & (np.abs(dy) <= 0.38)) | ((np.abs(dy) <= 0.12) & (np.abs(dx) <= 0.38))
    else:
        raise ValueError(f'target must be one of {TARGETS}')
    return mask


def _features(size: int, organizer: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    scale = size / 4.0
    dx = (xx - organizer[1]) / scale
    dy = (yy - organizer[0]) / scale
    return dx, dy, np.stack([dx * dx, dy * dy, dx ** 4])


def _neighbours(alive: np.ndarray) -> np.ndarray:
    a = np.pad(alive.astype(np.int16), 1)
    s = alive.shape[0]
    total = np.zeros(alive.shape, dtype=np.int16)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                total += a[1 + di:1 + di + s, 1 + dj:1 + dj + s]
    return total


def _assign_types(size: int, organizer: tuple[int, int]) -> np.ndarray:
    dx, dy, _ = _features(size, organizer)
    types = np.full((size, size), 1, dtype=np.int8)                     # association
    types[dx * dx + dy * dy <= 0.36] = 2                                 # memory core
    types[dx < -1.0] = 0                                                 # sensory pole
    types[dx > 1.0] = 3                                                  # action pole
    return types


def _logit(program: DevelopmentalProgram, alive: np.ndarray, organizer: tuple[int, int],
           memory: PatternMemory | None) -> np.ndarray:
    size = alive.shape[0]
    _, _, quad = _features(size, organizer)
    th = program.theta
    out = th[0] + th[1] * _neighbours(alive) / 8.0 + th[2] * quad[0] + th[3] * quad[1] + th[4] * quad[2]
    if memory is not None:
        out = out + memory.strength * (2.0 * memory.field(size) - 1.0)
    return out


def _develop(program: DevelopmentalProgram, body: Body, steps: int, rng: np.random.Generator,
             memory: PatternMemory | None) -> Body:
    alive = body.alive.copy()
    organizer = body.organizer
    for _ in range(steps):
        logit = _logit(program, alive, organizer, memory)
        mask = rng.random(alive.shape) < program.p_update
        has_nb = _neighbours(alive) > 0
        born = (~alive) & has_nb & (logit > 0.0) & mask
        die = alive & (logit < -program.death_margin) & mask
        die[organizer] = False                                          # the organizer persists while alive
        alive = (alive | born) & ~die
        if not alive.any():
            break
    types = _assign_types(alive.shape[0], organizer)
    cell_type = np.where(alive, types, EMPTY).astype(np.int8)
    return Body(alive, cell_type, organizer)


def grow(program: DevelopmentalProgram, size: int, *, seed: int = 0, steps: int | None = None) -> Body:
    """Grow a body from a single organizer cell at the grid centre (deterministic for a given seed)."""
    if not isinstance(program, DevelopmentalProgram):
        raise ValueError('program must be a DevelopmentalProgram')
    size = _check_grid(size)
    seed = _check_seed(seed)
    steps = 2 * size if steps is None else steps
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f'steps must be an int in [1, {MAX_STEPS}]')
    c = size // 2
    alive = np.zeros((size, size), dtype=bool)
    alive[c, c] = True
    start = Body(alive, np.where(alive, 2, EMPTY).astype(np.int8), (c, c))
    return _develop(program, start, steps, np.random.default_rng(seed + _GROW_OFFSET), None)


def lesion(body: Body, box: tuple[int, int, int, int]) -> Body:
    """Remove all tissue inside the half-open box ``(row0, row1, col0, col1)``."""
    if not isinstance(body, Body):
        raise ValueError('body must be a Body')
    r0, r1, c0, c1 = (int(v) for v in box)
    s = body.size
    if not (0 <= r0 < r1 <= s and 0 <= c0 < c1 <= s):
        raise ValueError('lesion box must lie inside the grid and be non-empty')
    out = body.copy()
    out.alive[r0:r1, c0:c1] = False
    out.cell_type[r0:r1, c0:c1] = EMPTY
    return out


def make_memory(body: Body, *, block: int = 2, strength: float = 3.0) -> PatternMemory:
    """Coarse pattern memory of an intact body: block-averaged occupancy plus the organizer position."""
    if not isinstance(body, Body):
        raise ValueError('body must be a Body')
    if type(block) is not int or not 1 <= block <= body.size // 2:
        raise ValueError('block must be an int in [1, size // 2]')
    if not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 50.0:
        raise ValueError('strength must lie in [0, 50]')
    n = body.size // block
    occ = body.alive[:n * block, :n * block].reshape(n, block, n, block).mean(axis=(1, 3))
    setpoint = occ.astype(np.float64)          # fractional occupancy: 0.5 is neutral, so block edges stay rule-driven
    setpoint.setflags(write=False)
    return PatternMemory(setpoint, block, tuple(body.organizer), float(strength))


def shuffled_memory(memory: PatternMemory, *, seed: int = 0) -> PatternMemory:
    """Capacity-matched control: same number of setpoint cells and numbers, shuffled content and organizer."""
    if not isinstance(memory, PatternMemory):
        raise ValueError('memory must be a PatternMemory')
    rng = np.random.default_rng(_check_seed(seed) + _SHUFFLE_OFFSET)
    flat = memory.setpoint.ravel().copy()
    rng.shuffle(flat)
    n = memory.setpoint.shape[0]
    side = n * memory.block
    organizer = (int(rng.integers(side)), int(rng.integers(side)))
    setpoint = flat.reshape(memory.setpoint.shape)
    setpoint.setflags(write=False)
    return PatternMemory(setpoint, memory.block, organizer, memory.strength)


def regrow(program: DevelopmentalProgram, damaged: Body, *, memory: PatternMemory | None = None, seed: int = 0,
           steps: int | None = None) -> Body:
    """Regrow a lesioned body with the same developmental rule.

    With a pattern memory the organizer position comes from the memory and the setpoint biases growth and
    death. Without one, a lost organizer is re-derived from the centroid of the surviving tissue (nearest
    living cell)."""
    if not isinstance(program, DevelopmentalProgram):
        raise ValueError('program must be a DevelopmentalProgram')
    if not isinstance(damaged, Body):
        raise ValueError('damaged must be a Body')
    if memory is not None and not isinstance(memory, PatternMemory):
        raise ValueError('memory must be a PatternMemory or None')
    seed = _check_seed(seed)
    size = damaged.size
    steps = 2 * size if steps is None else steps
    if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f'steps must be an int in [1, {MAX_STEPS}]')
    if not damaged.alive.any():
        return damaged.copy()                        # nothing left to regrow from
    body = damaged.copy()
    oy, ox = body.organizer
    if memory is not None:
        my, mx = (min(size - 1, max(0, int(v))) for v in memory.organizer)
        body.organizer = (my, mx)                    # re-established by growth if it was lost
    elif not body.alive[oy, ox]:
        ys, xs = np.nonzero(body.alive)
        cy, cx = ys.mean(), xs.mean()
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        body.organizer = (int(ys[k]), int(xs[k]))
    return _develop(program, body, steps, np.random.default_rng(seed + _GROW_OFFSET + 1), memory)


# ---------------------------------------------------------------------- connectome

def body_to_topology(body: Body, *, radius: float = 1.5, name: str = 'morphogenesis', seed: int = 0) -> Topology:
    """Cells -> neurons (row-major order), types -> regions, proximity + type-rank rules -> edges.

    * same type within ``radius``: reciprocal (local recurrent) edges;
    * different types within ``radius``: feedforward edge from lower to higher processing rank, plus a feedback
      edge when the cells are direct (4-)neighbours;
    * delay 1 / 2 / 3 by distance; weight magnitude decays with distance; sign fixed by site position
      (~20 % inhibitory, spread across the grid) so the same site keeps its sign after regrowth."""
    if not isinstance(body, Body):
        raise ValueError('body must be a Body')
    if not isinstance(radius, (int, float)) or not 1.0 <= float(radius) <= 4.0:
        raise ValueError('radius must lie in [1, 4]')
    ys, xs = np.nonzero(body.alive)
    n = int(ys.size)
    types = body.cell_type[ys, xs].astype(np.int64)
    rank = np.array([KIND_RANK[CELL_TYPES[t]] for t in types], dtype=np.int64)
    region = np.array([TYPE_REGION[t] for t in types], dtype=np.int64)
    sign = np.where((ys * 7 + xs * 13) % 5 == 0, -1, 1).astype(np.int8)
    if n >= 2:
        dy = ys[:, None] - ys[None, :]
        dx = xs[:, None] - xs[None, :]
        dist = np.sqrt(dy * dy + dx * dx)
        near = (dist <= radius + 1e-9) & ~np.eye(n, dtype=bool)
        same = types[:, None] == types[None, :]
        ff = rank[:, None] < rank[None, :]
        fb = (rank[:, None] > rank[None, :]) & (dist <= 1.0 + 1e-9)
        adj = near & (same | ff | fb)
        src, dst = np.nonzero(adj)
        d = dist[src, dst]
    else:
        src = dst = np.zeros(0, dtype=np.int64)
        d = np.zeros(0)
    delay = (1 + (d > 1.5 + 1e-9) + (d > 2.5 + 1e-9)).astype(np.int64)
    mag = WEIGHT_MIN + WEIGHT_SPAN * np.clip(1.0 - (d - 1.0) / max(float(radius), 1.0), 0.0, 1.0)
    weight = sign[src].astype(np.float64) * mag
    topo = Topology(n=n, src=src.astype(np.int64), dst=dst.astype(np.int64), weight=weight, delay=delay,
                    sign=sign, region=region, name=str(name), seed=int(seed), regions=REGIONS)
    topo.validate()
    return topo


def _edge_keys(body: Body, topo: Topology) -> set[tuple[int, int]]:
    ys, xs = np.nonzero(body.alive)
    pos = ys * body.size + xs
    return set(zip(pos[topo.src].tolist(), pos[topo.dst].tolist(), strict=True))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('masks must share a shape')
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def edge_jaccard(body_a: Body, body_b: Body, *, radius: float = 1.5) -> float:
    """Jaccard similarity of position-keyed directed edge sets of the two grown connectomes."""
    ea = _edge_keys(body_a, body_to_topology(body_a, radius=radius))
    eb = _edge_keys(body_b, body_to_topology(body_b, radius=radius))
    union = len(ea | eb)
    return float(len(ea & eb) / union) if union else 1.0


def _type_agreement(ref: Body, other: Body) -> float:
    both = ref.alive & other.alive
    return float(np.mean(ref.cell_type[both] == other.cell_type[both])) if both.any() else 0.0


def _metric_similarity(a: Body, b: Body) -> float:
    keys = ('density', 'clustering_mean', 'modularity', 'global_efficiency')
    if a.cells < 2 or b.cells < 2:
        return 0.0
    ma, mb = topology_metrics(body_to_topology(a)), topology_metrics(body_to_topology(b))
    diffs = [abs(ma[k] - mb[k]) / max(abs(ma[k]), abs(mb[k]), 1e-9) for k in keys]
    return float(1.0 - np.mean(diffs))


# ---------------------------------------------------------------------- evolution

def evolve_program(target: np.ndarray, *, seed: int = 0, generations: int = 30, offspring: int = 8,
                   sigma: float = 0.4, growth_seed: int = 0, steps: int | None = None) -> dict[str, Any]:
    """Seeded (1 + lambda)-ES on ``theta`` maximising IoU between the grown body and ``target``.

    Ties keep the parent (no drift on plateaus). Returns the best program and the fitness trace."""
    tgt = np.asarray(target, dtype=bool)
    if tgt.ndim != 2 or tgt.shape[0] != tgt.shape[1]:
        raise ValueError('target must be a square boolean mask')
    size = _check_grid(int(tgt.shape[0]))
    seed = _check_seed(seed)
    for nm, v, lo, hi in (('generations', generations, 0, 500), ('offspring', offspring, 1, 64)):
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError(f'{nm} must be an int in [{lo}, {hi}]')
    if not isinstance(sigma, (int, float)) or not 0.0 < float(sigma) <= 5.0:
        raise ValueError('sigma must lie in (0, 5]')
    rng = np.random.default_rng(seed + _ES_OFFSET)
    parent = DevelopmentalProgram()

    def fitness(p: DevelopmentalProgram) -> float:
        return iou(grow(p, size, seed=growth_seed, steps=steps).alive, tgt)

    best = fitness(parent)
    initial = best
    trace = [best]
    evaluations = 1
    for _ in range(generations):
        base = np.asarray(parent.theta)
        children = [DevelopmentalProgram(tuple(np.clip(base + float(sigma) * rng.standard_normal(N_PARAMS),
                                                       -50.0, 50.0)))
                    for _ in range(offspring)]
        scores = [fitness(c) for c in children]
        evaluations += offspring
        k = int(np.argmax(scores))
        if scores[k] > best:
            parent, best = children[k], scores[k]
        trace.append(best)
    return {'program': parent, 'fitness': best, 'initial_fitness': initial, 'trace': trace,
            'evaluations': evaluations}


# ---------------------------------------------------------------------- experiment

DEFAULTS: dict[str, Any] = {'size': 20, 'target': 'two_lobes', 'generations': 30, 'offspring': 8, 'sigma': 0.4,
                            'lesions_per_seed': 4, 'lesion_fraction': 0.4, 'memory_block': 2,
                            'memory_strength': 3.0, 'radius': 1.5}
_QUICK: dict[str, Any] = {'size': 14, 'generations': 10, 'offspring': 6, 'lesions_per_seed': 3}


def _config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'unknown overrides: {sorted(unknown)}')
    cfg = {**DEFAULTS, **(_QUICK if quick else {}), **overrides}
    _check_grid(cfg['size'])
    if cfg['target'] not in TARGETS:
        raise ValueError(f'target must be one of {TARGETS}')
    if type(cfg['lesions_per_seed']) is not int or not 1 <= cfg['lesions_per_seed'] <= 64:
        raise ValueError('lesions_per_seed must be an int in [1, 64]')
    if not isinstance(cfg['lesion_fraction'], (int, float)) or not 0.1 <= cfg['lesion_fraction'] <= 0.7:
        raise ValueError('lesion_fraction must lie in [0.1, 0.7]')
    if type(cfg['memory_block']) is not int or not 1 <= cfg['memory_block'] <= cfg['size'] // 2:
        raise ValueError('memory_block must be an int in [1, size // 2]')
    return cfg


def _lesion_box(body: Body, rng: np.random.Generator, fraction: float) -> tuple[int, int, int, int]:
    """A square box centred on a random living cell, sized to cover ``fraction`` of the body's bounding box."""
    ys, xs = np.nonzero(body.alive)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    side = max(2, int(round(math.sqrt(fraction * h * w))))
    k = int(rng.integers(ys.size))
    cy, cx = int(ys[k]), int(xs[k])
    s = body.size
    r0 = min(max(0, cy - side // 2), s - side)
    c0 = min(max(0, cx - side // 2), s - side)
    return r0, r0 + side, c0, c0 + side


def _run_seed(seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    size = int(cfg['size'])
    tgt = target_mask(cfg['target'], size)
    evo = evolve_program(tgt, seed=seed, generations=cfg['generations'], offspring=cfg['offspring'],
                         sigma=cfg['sigma'], growth_seed=seed)
    program: DevelopmentalProgram = evo['program']
    body = grow(program, size, seed=seed)
    memory = make_memory(body, block=cfg['memory_block'], strength=cfg['memory_strength'])
    rng = np.random.default_rng(seed + _LESION_OFFSET)
    arms: dict[str, dict[str, list[float]]] = {a: {'iou': [], 'edge_jaccard': [], 'type_agreement': [],
                                                   'metric_similarity': []}
                                               for a in ('pattern_memory', 'no_memory', 'shuffled_memory')}
    lesions = []
    for j in range(int(cfg['lesions_per_seed'])):
        box = _lesion_box(body, rng, float(cfg['lesion_fraction']))
        damaged = lesion(body, box)
        oy, ox = body.organizer
        lesions.append({'box': list(box), 'removed_cells': int(body.cells - damaged.cells),
                        'organizer_lost': bool(not damaged.alive[oy, ox])})
        mems = {'pattern_memory': memory, 'no_memory': None,
                'shuffled_memory': shuffled_memory(memory, seed=seed * 1_000 + j)}
        for a, mem in mems.items():
            healed = regrow(program, damaged, memory=mem, seed=seed * 1_000 + j)
            rec = arms[a]
            rec['iou'].append(iou(healed.alive, body.alive))
            rec['edge_jaccard'].append(edge_jaccard(body, healed, radius=cfg['radius']))
            rec['type_agreement'].append(_type_agreement(body, healed))
            rec['metric_similarity'].append(_metric_similarity(body, healed))
    topo = body_to_topology(body, radius=cfg['radius'])
    metrics = topology_metrics(topo)
    return {
        'seed': int(seed),
        'development': {'target_iou': float(evo['fitness']), 'initial_program_iou': float(evo['initial_fitness']),
                        'es_trace': [float(v) for v in evo['trace']], 'evaluations': int(evo['evaluations']),
                        'theta': list(program.theta), 'cells': body.cells,
                        'type_counts': {t: int(np.sum(body.cell_type == i)) for i, t in enumerate(CELL_TYPES)}},
        'connectome': {k: metrics[k] for k in ('n', 'edges', 'density', 'clustering_mean', 'modularity',
                                               'global_efficiency', 'reachable_pair_fraction')},
        'lesions': lesions,
        'arms': {a: {k: float(np.mean(v)) for k, v in rec.items()} for a, rec in arms.items()},
    }


def run_experiment(*, quick: bool = False, seeds: Iterable[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Regeneration with a stored pattern memory vs without one vs a shuffled (capacity-matched) memory.

    Per seed: evolve a developmental program towards the target shape, grow a body, apply several random lesions
    and regrow under each arm. Metric: IoU between the regrown and the pre-lesion body (higher is better)."""
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique non-negative ints')
    cfg = _config(quick, overrides)
    t0 = time.perf_counter()
    runs = [_run_seed(s, cfg) for s in seeds]
    arm_names = ('pattern_memory', 'no_memory', 'shuffled_memory')
    vals = {a: [r['arms'][a]['iou'] for r in runs] for a in arm_names}
    vs_none = harness.compare(vals['pattern_memory'], vals['no_memory'])
    vs_shuffled = harness.compare(vals['pattern_memory'], vals['shuffled_memory'])
    head = harness.headline_verdict(vs_none, vs_shuffled)
    edge = {a: [r['arms'][a]['edge_jaccard'] for r in runs] for a in arm_names}
    receipt = harness.base_receipt(MODULE_ID, 'regeneration_comparison', {
        **cfg, 'quick': quick, 'seeds': list(seeds), 'metric': 'recovery_iou_vs_pre_lesion_body',
        'program_parameters': N_PARAMS, 'cell_types': list(CELL_TYPES),
        'type_regions': [REGIONS[r][0] for r in TYPE_REGION]})
    receipt['per_seed'] = runs
    receipt['arms'] = {a: {'summary': {k: float(np.mean([r['arms'][a][k] for r in runs]))
                                       for k in ('iou', 'edge_jaccard', 'type_agreement', 'metric_similarity')}}
                       for a in arm_names}
    receipt['development_summary'] = {
        'target_iou': float(np.mean([r['development']['target_iou'] for r in runs])),
        'initial_program_iou': float(np.mean([r['development']['initial_program_iou'] for r in runs]))}
    receipt['comparisons'] = {'pattern_memory_vs_no_memory': vs_none,
                              'pattern_memory_vs_shuffled_memory': vs_shuffled,
                              'edge_jaccard_pattern_memory_vs_no_memory': harness.compare(edge['pattern_memory'],
                                                                                          edge['no_memory'])}
    receipt['verdict'] = head['verdict']
    receipt['verdict_basis'] = head['basis'] + ' (paired 95% t interval over seeds on recovery IoU)'
    mem_numbers = (cfg['size'] // cfg['memory_block']) ** 2 + 2
    receipt['parameter_budget'] = {'program_parameters': N_PARAMS, 'pattern_memory_numbers': int(mem_numbers),
                                   'shuffled_memory_numbers': int(mem_numbers), 'no_memory_numbers': 0}
    receipt['estimated_cost'] = {
        'estimated_site_updates_per_growth': int(2 * cfg['size'] * cfg['size'] ** 2),
        'estimated_growths_per_seed': int(1 + cfg['generations'] * cfg['offspring'] + 1 + 3 * cfg['lesions_per_seed'])}
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'Synthetic developmental abstraction inspired by growing neural cellular automata (Mordvintsev et al., Distill '
        '2020) and bioelectric pattern memory (Pezzulo & Levin, Integr. Biol. 2015). The rule is a hand-designed '
        'five-parameter growth logit, not a trained neural CA. The pattern memory is a coarse copy of the intact body, '
        'so it helps by construction when lesions remove the organizer; the shuffled-memory arm checks that the gain '
        'comes from the content of the memory rather than its presence or size.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
