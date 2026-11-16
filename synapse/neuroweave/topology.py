"""Deterministic NEUROWEAVE topology generators and graph metrics (NumPy only).

Re-implements the NEUROWEAVE-LAB 0.4.0 connectome generators (clustered
small-world, hub-skewed, uniform random, degree-shuffled double-edge swaps,
Erdos-Renyi, hub-capped, ring lattice, multi-region and dense multi-region) and
adds Sonapse-side regional generators (hierarchical modular, multi-scale
regional, hybrid). Every generator returns a :class:`Topology` with ten region
identities assigned in contiguous blocks, Dale-style per-neuron signs (an
interneuron-like inhibitory population spread across every region), signed
weights and integer delays in ``[1, 3]``. There are no self loops and no
duplicate edges.

These are synthetic graph abstractions; nothing here claims biological
fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .spec import REGIONS, TOPOLOGIES

__all__ = [
    'Topology', 'build_topology', 'topology_metrics', 'assign_regions', 'assign_signs',
    'region_kind_rank', 'edge_compartments', 'KIND_RANK',
]

_TOPOLOGY_OFFSET = 1_009
WEIGHT_MIN = 0.16
WEIGHT_SPAN = 0.48

# Processing hierarchy used for feedforward/feedback classification.
KIND_RANK: dict[str, int] = {'sensory': 0, 'association': 1, 'memory': 2, 'action': 3}


@dataclass
class Topology:
    n: int
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    delay: np.ndarray
    sign: np.ndarray
    region: np.ndarray
    name: str
    seed: int
    regions: tuple = field(default=REGIONS)

    @property
    def edges(self) -> int:
        return int(self.src.size)

    def validate(self) -> None:
        n = self.n
        if self.src.shape != self.dst.shape or self.src.shape != self.weight.shape or self.src.shape != self.delay.shape:
            raise ValueError('edge arrays must share one shape')
        if self.sign.shape != (n,) or self.region.shape != (n,):
            raise ValueError('per-neuron arrays must have length n')
        if self.src.size:
            if self.src.min() < 0 or self.src.max() >= n or self.dst.min() < 0 or self.dst.max() >= n:
                raise ValueError('edge endpoint out of range')
            if np.any(self.src == self.dst):
                raise ValueError('self loops are not allowed')
            keys = self.src.astype(np.int64) * n + self.dst
            if np.unique(keys).size != keys.size:
                raise ValueError('duplicate edges are not allowed')
            if self.delay.min() < 1 or self.delay.max() > 3:
                raise ValueError('delays must lie in [1, 3]')
            if not np.all(np.sign(self.weight) == self.sign[self.src]):
                raise ValueError('edge weight sign must match source sign')
        if not np.all(np.isfinite(self.weight)):
            raise ValueError('weights must be finite')
        if not np.all(np.isin(self.sign, (-1, 1))):
            raise ValueError('signs must be +1/-1')

    def to_dict(self) -> dict[str, Any]:
        return {
            'n': int(self.n), 'name': self.name, 'seed': int(self.seed),
            'src': self.src.tolist(), 'dst': self.dst.tolist(), 'weight': self.weight.tolist(),
            'delay': self.delay.tolist(), 'sign': self.sign.tolist(), 'region': self.region.tolist(),
            'regions': [list(r) for r in self.regions],
        }


# --------------------------------------------------------------------------- helpers

def assign_regions(n: int, n_regions: int) -> np.ndarray:
    """Contiguous blocks proportional to n; every region has >=1 neuron when n >= n_regions."""
    idx = np.arange(n, dtype=np.int64)
    return np.minimum(n_regions - 1, (idx * n_regions) // max(1, n)).astype(np.int64)


def assign_signs(region: np.ndarray, inhibitory_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Inhibitory interneuron-like units spread across every region (per-region quota)."""
    n = region.size
    sign = np.ones(n, dtype=np.int8)
    if inhibitory_fraction <= 0 or n == 0:
        return sign
    for r in np.unique(region):
        members = np.flatnonzero(region == r)
        k = int(round(inhibitory_fraction * members.size))
        k = min(k, members.size - 1) if members.size > 1 else 0
        if k > 0:
            chosen = rng.choice(members, size=k, replace=False)
            sign[chosen] = -1
    return sign


def region_kind_rank(regions: Sequence) -> np.ndarray:
    """Hierarchy rank per region (-1 for modulatory/unknown kinds)."""
    return np.array([KIND_RANK.get(str(r[1]), -1) for r in regions], dtype=np.int64)


def edge_compartments(src: np.ndarray, dst: np.ndarray, region: np.ndarray, kind_rank: np.ndarray) -> np.ndarray:
    """Dendritic target compartment per edge.

    0 proximal: same region (local recurrent); 1 distal: lateral / modulatory;
    2 apical: feedback (higher -> lower rank); 3 basal: feedforward.
    """
    rs, rd = region[src], region[dst]
    ks, kd = kind_rank[rs], kind_rank[rd]
    comp = np.ones(src.size, dtype=np.int8)
    valid = (ks >= 0) & (kd >= 0)
    comp[valid & (ks < kd)] = 3
    comp[valid & (ks > kd)] = 2
    comp[rs == rd] = 0
    return comp


class _EdgeBuilder:
    """Collects candidate edges in generation order; dedupes keeping first occurrence."""

    def __init__(self, n: int):
        self.n = n
        self.parts_src: list[np.ndarray] = []
        self.parts_dst: list[np.ndarray] = []

    def add(self, src, dst) -> None:
        s = np.asarray(src, dtype=np.int64).ravel()
        d = np.asarray(dst, dtype=np.int64).ravel()
        if s.size:
            self.parts_src.append(s % self.n)
            self.parts_dst.append(d % self.n)

    def finish(self, cap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if not self.parts_src:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        s = np.concatenate(self.parts_src)
        d = np.concatenate(self.parts_dst)
        keep = s != d
        s, d = s[keep], d[keep]
        keys = s * self.n + d
        _, first = np.unique(keys, return_index=True)
        first.sort()
        s, d = s[first], d[first]
        if cap is not None and s.size:
            order = np.argsort(d, kind='stable')
            ds = d[order]
            starts = np.r_[0, np.flatnonzero(np.diff(ds)) + 1]
            rank = np.arange(ds.size) - np.repeat(starts, np.diff(np.r_[starts, ds.size]))
            ok = np.zeros(s.size, dtype=bool)
            ok[order[rank < cap]] = True
            s, d = s[ok], d[ok]
        return s, d


def _ring(builder: _EdgeBuilder, n: int, radius: int) -> None:
    i = np.arange(n)
    for r in range(1, radius + 1):
        builder.add(i, i + r)
        builder.add(i, i - r + n)


def _local_random(builder: _EdgeBuilder, n: int, count: int, rng: np.random.Generator, local_p: float = 0.72) -> None:
    src = rng.integers(0, n, size=count)
    dst = rng.integers(0, n, size=count)
    local = rng.random(count) < local_p
    off = rng.integers(1, 6, size=count) * np.where(rng.random(count) < 0.5, 1, -1)
    dst = np.where(local, (src + off) % n, dst)
    builder.add(src, dst)


def _sample_pairs(rng: np.random.Generator, pool_src: np.ndarray, pool_dst: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    total = pool_src.size * pool_dst.size
    if total == 0 or p <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    count = int(rng.binomial(total, min(1.0, p)))
    return pool_src[rng.integers(0, pool_src.size, count)], pool_dst[rng.integers(0, pool_dst.size, count)]


def _members(region: np.ndarray, n_regions: int) -> list[np.ndarray]:
    return [np.flatnonzero(region == r) for r in range(n_regions)]


def _block_local(builder: _EdgeBuilder, members: list[np.ndarray], p: float, rng: np.random.Generator, *, reciprocal: bool = True) -> None:
    """Local recurrent circuits: random within-region edges plus reciprocal ring motifs."""
    for m in members:
        s = m.size
        if s < 2:
            continue
        mask = rng.random((s, s)) < p
        np.fill_diagonal(mask, False)
        a, b = np.nonzero(mask)
        builder.add(m[a], m[b])
        if reciprocal:
            nxt = np.roll(m, -1)
            builder.add(m, nxt)
            builder.add(nxt, m)


def _pathways(builder: _EdgeBuilder, members: list[np.ndarray], regions: Sequence, ff: int, fb: int) -> None:
    """Sensory->association->memory->action feedforward, action->association and
    memory->sensory predictive feedback, modulatory hub broadcast."""
    kinds = [str(r[1]) for r in regions]
    by_kind: dict[str, list[int]] = {}
    for idx, k in enumerate(kinds):
        by_kind.setdefault(k, []).append(idx)
    routes: list[tuple[str, str, int]] = [
        ('sensory', 'association', ff), ('association', 'memory', ff), ('memory', 'action', ff),
        ('association', 'action', max(1, ff // 2)),
        ('action', 'association', fb), ('memory', 'sensory', fb), ('association', 'sensory', max(1, fb // 2)),
        ('modulatory', 'sensory', 1), ('modulatory', 'association', 1), ('modulatory', 'memory', 1),
        ('modulatory', 'action', 1), ('association', 'modulatory', 1), ('memory', 'modulatory', 1),
    ]
    for a_kind, b_kind, fan in routes:
        for ra in by_kind.get(a_kind, []):
            for rb in by_kind.get(b_kind, []):
                srcs, dsts = members[ra], members[rb]
                if not srcs.size or not dsts.size:
                    continue
                k = min(fan, dsts.size)
                for j in range(k):
                    builder.add(srcs, dsts[(j * 17 + srcs + ra * 3 + rb) % dsts.size])


def _long_range(builder: _EdgeBuilder, region: np.ndarray, n: int, p: float, rng: np.random.Generator) -> None:
    allv = np.arange(n, dtype=np.int64)
    s, d = _sample_pairs(rng, allv, allv, p)
    cross = region[s] != region[d]
    builder.add(s[cross], d[cross])


def _double_edge_swaps(src: np.ndarray, dst: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    src = src.copy()
    dst = dst.copy()
    e = src.size
    if e < 2:
        return src, dst
    keys = set((src * n + dst).tolist())
    attempts = min(e * 24, 200_000)
    ii = rng.integers(0, e, attempts)
    jj = rng.integers(0, e, attempts)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i == j:
            continue
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if a == c or b == d or a == d or c == b:
            continue
        n1, n2 = a * n + d, c * n + b
        if n1 in keys or n2 in keys:
            continue
        keys.discard(a * n + b)
        keys.discard(c * n + d)
        keys.add(n1)
        keys.add(n2)
        dst[i], dst[j] = d, b
    return src, dst


# --------------------------------------------------------------------------- builder

def build_topology(name: str, n: int, *, seed: int, fan_in: int = 6, inhibitory_fraction: float = 0.2,
                   regions: Sequence = REGIONS) -> Topology:
    if name not in TOPOLOGIES:
        raise ValueError(f'unknown topology {name!r}; expected one of {TOPOLOGIES}')
    if type(n) is not int or n < 2:
        raise ValueError('n must be an int >= 2')
    if type(fan_in) is not int or fan_in < 1:
        raise ValueError('fan_in must be a positive int')
    if not 0.0 <= float(inhibitory_fraction) < 1.0:
        raise ValueError('inhibitory_fraction must be in [0, 1)')
    regions = tuple(tuple(r) for r in regions)
    if not regions:
        raise ValueError('regions must be non-empty')
    # degree_shuffled shares the clustered stream so its pre-swap graph is exactly the clustered graph.
    stream = 'clustered' if name == 'degree_shuffled' else name
    rng = np.random.default_rng(int(seed) + _TOPOLOGY_OFFSET + TOPOLOGIES.index(stream))
    n_regions = len(regions)
    region = assign_regions(n, n_regions)
    sign = assign_signs(region, float(inhibitory_fraction), rng)
    members = _members(region, n_regions)
    b = _EdgeBuilder(n)
    cap: int | None = None

    if name in ('clustered', 'degree_shuffled'):
        _ring(b, n, 2)
        _local_random(b, n, n * max(1, fan_in - 3), rng)
    elif name == 'ring_lattice':
        _ring(b, n, max(1, min((n - 1) // 2, max(2, fan_in // 2))))
    elif name == 'hub_capped':
        cap = max(fan_in + 1, int(np.ceil(1.5 * fan_in)))
        _ring(b, n, 2)
        b.add(rng.integers(0, n, n * fan_in), rng.integers(0, n, n * fan_in))
    elif name == 'hub_skewed':
        order = rng.permutation(n)
        w = np.empty(n)
        w[order] = 1.0 / np.power(1.0 + np.arange(n), 1.1)
        w /= w.sum()
        cnt = n * fan_in
        b.add(rng.integers(0, n, cnt), rng.choice(n, size=cnt, p=w))
    elif name == 'uniform_random':
        b.add(rng.integers(0, n, n * fan_in), rng.integers(0, n, n * fan_in))
    elif name == 'erdos_renyi':
        p = min(0.18 if n > 2 * fan_in else 0.9, fan_in / max(1, n - 1))
        allv = np.arange(n, dtype=np.int64)
        s, d = _sample_pairs(rng, allv, allv, p)
        b.add(s, d)
    elif name in ('multi_region', 'dense_multi_region'):
        dense = name == 'dense_multi_region'
        size = max(2, n // n_regions)
        local_p = min(0.8 if dense else 0.6, max(0.20 if dense else 0.105, (1.0 if dense else 0.5) * fan_in / max(1, size - 1)))
        _block_local(b, members, local_p, rng)
        _pathways(b, members, regions, ff=4 if dense else 3, fb=2 if dense else 1)
        _long_range(b, region, n, 0.035 if dense else 0.012, rng)
    elif name == 'hierarchical_modular':
        size = max(2, n // n_regions)
        _block_local(b, members, min(0.7, max(0.15, 0.6 * fan_in / max(1, size - 1))), rng)
        # second level: modules of the same kind are densely interconnected
        kinds = [str(r[1]) for r in regions]
        for k in sorted(set(kinds)):
            grp = np.concatenate([members[i] for i, kk in enumerate(kinds) if kk == k] or [np.zeros(0, np.int64)])
            if grp.size > 1:
                s, d = _sample_pairs(rng, grp, grp, min(0.5, 1.5 / max(1, grp.size)))
                cross = region[s] != region[d]
                b.add(s[cross], d[cross])
        _pathways(b, members, regions, ff=3, fb=1)
        _long_range(b, region, n, 0.006, rng)
    elif name == 'multi_scale_regional':
        # scale 1: micro rings inside each region; scale 2: regional random; scale 3: global long-range
        for m in members:
            for r in (1, 2):
                if m.size > r:
                    b.add(m, np.roll(m, -r))
                    b.add(np.roll(m, -r), m)
        _block_local(b, members, min(0.4, max(0.08, 0.3 * fan_in / max(1, n // n_regions))), rng, reciprocal=False)
        _pathways(b, members, regions, ff=2, fb=1)
        _long_range(b, region, n, min(0.05, 1.5 / max(1, n)), rng)
    elif name == 'hybrid':
        _ring(b, n, 2)
        _local_random(b, n, n * max(1, fan_in // 3), rng)
        hubs = rng.choice(n, size=max(1, n // 12), replace=False)
        cnt = n * max(1, fan_in // 3)
        b.add(rng.integers(0, n, cnt), hubs[rng.integers(0, hubs.size, cnt)])
        b.add(hubs[rng.integers(0, hubs.size, cnt // 2)], rng.integers(0, n, cnt // 2))
        _pathways(b, members, regions, ff=2, fb=1)
    else:  # pragma: no cover - guarded by TOPOLOGIES check
        raise ValueError(name)

    src, dst = b.finish(cap)
    if name == 'degree_shuffled':
        src, dst = _double_edge_swaps(src, dst, n, rng)
    order = np.lexsort((dst, src))
    src, dst = src[order].astype(np.int64), dst[order].astype(np.int64)
    mag = WEIGHT_MIN + WEIGHT_SPAN * rng.random(src.size)
    weight = sign[src].astype(np.float64) * mag
    delay = (1 + np.floor(rng.random(src.size) * 3)).astype(np.int64)
    topo = Topology(n=n, src=src, dst=dst, weight=weight, delay=np.clip(delay, 1, 3), sign=sign,
                    region=region, name=name, seed=int(seed), regions=regions)
    topo.validate()
    return topo


# --------------------------------------------------------------------------- metrics

def _degree_stats(deg: np.ndarray) -> dict[str, Any]:
    mean = float(deg.mean()) if deg.size else 0.0
    std = float(deg.std()) if deg.size else 0.0
    return {
        'histogram': np.bincount(deg).tolist() if deg.size else [],
        'mean': mean, 'std': std, 'max': int(deg.max()) if deg.size else 0,
        'cv': float(std / mean) if mean > 0 else 0.0,
    }


def _gini(x: np.ndarray) -> float:
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    s = np.sort(x.astype(np.float64))
    k = np.arange(1, s.size + 1)
    return float((2.0 * (k * s).sum()) / (s.size * s.sum()) - (s.size + 1.0) / s.size)


def _bfs_distances(n: int, src: np.ndarray, dst: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Unweighted directed shortest-path lengths from `sources` (inf if unreachable).

    Multi-source level-synchronous BFS over the edge list: each level gathers
    frontier bits at edge sources and OR-reduces them per destination.
    """
    dist = np.full((sources.size, n), np.inf)
    frontier = np.zeros((sources.size, n), dtype=bool)
    frontier[np.arange(sources.size), sources] = True
    visited = frontier.copy()
    dist[frontier] = 0.0
    if src.size == 0:
        return dist
    order = np.argsort(dst, kind='stable')
    s_sorted, d_sorted = src[order], dst[order]
    starts = np.r_[0, np.flatnonzero(np.diff(d_sorted)) + 1]
    targets = d_sorted[starts]
    level = 0
    while frontier.any():
        level += 1
        reached = np.logical_or.reduceat(frontier[:, s_sorted], starts, axis=1)
        nxt = np.zeros_like(frontier)
        nxt[:, targets] = reached
        nxt &= ~visited
        dist[nxt] = level
        visited |= nxt
        frontier = nxt
    return dist


def topology_metrics(topo: Topology, *, max_bfs_sources: int = 1024) -> dict[str, Any]:
    n = int(topo.n)
    src, dst = topo.src, topo.dst
    e = int(src.size)
    n_regions = len(topo.regions)
    indeg = np.bincount(dst, minlength=n)
    outdeg = np.bincount(src, minlength=n)

    und = np.zeros((n, n), dtype=bool)
    und[src, dst] = True
    und |= und.T
    np.fill_diagonal(und, False)
    deg_u = und.sum(1)
    clus = np.zeros(n)
    for i in range(n):
        nb = np.flatnonzero(und[i])
        k = nb.size
        if k >= 2:
            links = und[np.ix_(nb, nb)].sum() / 2.0
            clus[i] = links / (k * (k - 1) / 2.0)
    m_u = und.sum() / 2.0
    modularity = 0.0
    if m_u > 0:
        for r in range(n_regions):
            mem = topo.region == r
            if not mem.any():
                continue
            l_c = und[np.ix_(mem, mem)].sum() / 2.0
            d_c = deg_u[mem].sum()
            modularity += l_c / m_u - (d_c / (2.0 * m_u)) ** 2

    if n <= max_bfs_sources:
        sources = np.arange(n)
    else:
        sources = np.unique(np.linspace(0, n - 1, max_bfs_sources).astype(np.int64))
    dist = _bfs_distances(n, src, dst, sources)
    offdiag = np.ones_like(dist, dtype=bool)
    offdiag[np.arange(sources.size), sources] = False
    finite = np.isfinite(dist) & offdiag
    mean_sp = float(dist[finite].mean()) if finite.any() else float('inf')
    inv = np.zeros_like(dist)
    inv[finite] = 1.0 / dist[finite]
    pairs = offdiag.sum()
    efficiency = float(inv[offdiag].sum() / pairs) if pairs else 0.0

    top = max(1, int(np.ceil(0.1 * n)))
    top_share = float(np.sort(indeg)[::-1][:top].sum() / e) if e else 0.0

    rp = np.zeros((n_regions, n_regions), dtype=np.int64)
    np.add.at(rp, (topo.region[src], topo.region[dst]), 1)
    reg_eff = np.zeros((n_regions, n_regions))
    src_region = topo.region[sources]
    for a in range(n_regions):
        rows = src_region == a
        if not rows.any():
            continue
        for bb in range(n_regions):
            cols = topo.region == bb
            if not cols.any():
                continue
            block_inv = inv[np.ix_(rows, cols)]
            block_valid = offdiag[np.ix_(rows, cols)]
            cnt = block_valid.sum()
            reg_eff[a, bb] = float(block_inv[block_valid].sum() / cnt) if cnt else 0.0
    present = np.array([(topo.region == r).any() for r in range(n_regions)])
    off = ~np.eye(n_regions, dtype=bool) & present[:, None] & present[None, :]
    inter = int(rp[off].sum()) if e else 0
    return {
        'n': n, 'edges': e,
        'density': float(e / (n * (n - 1))) if n > 1 else 0.0,
        'in_degree': _degree_stats(indeg), 'out_degree': _degree_stats(outdeg),
        'clustering_mean': float(clus.mean()),
        'modularity': float(modularity),
        'mean_shortest_path': mean_sp,
        'global_efficiency': efficiency,
        'reachable_pair_fraction': float(finite.sum() / pairs) if pairs else 0.0,
        'bfs_sources': int(sources.size),
        'hub_top10_in_share': top_share,
        'in_degree_gini': _gini(indeg),
        'region_pair_edges': rp.tolist(),
        'regional_efficiency': reg_eff.tolist(),
        'regional_communication_efficiency': float(reg_eff[off].mean()) if off.any() else 0.0,
        'inter_region_edge_fraction': float(inter / e) if e else 0.0,
    }
