"""Signed synapse table with delays, routing, plasticity and structural change.

NumPy re-implementation of the NEUROWEAVE-LAB 0.4.0 synapse layer: per-edge
delay queues (ring buffer of past outputs), tiled traversal (numerically the
same totals), INT8-like quantised transmission, context-adaptive routing,
uncertainty gating, region-pair routing gains, reward-gated eligibility
updates, pair-based STDP, meta-plasticity, utility-guided pruning/regrowth
with a bounded reasoned event ledger, downscaling, consolidation and lesions.

Dale-style source signs are never flipped: every weight update operates on the
magnitude and re-applies the source neuron's sign.
"""
from __future__ import annotations

from collections import deque
import copy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cells import CompartmentDrive
from .spec import REGIONS
from .topology import Topology, edge_compartments, region_kind_rank

__all__ = ['SynapseTable', 'EventLedger', 'quantise_int8', 'dequantise_int8', 'W_MIN', 'W_MAX',
           'ROUTING_BOUNDS', 'UNCERTAINTY_BOUNDS', 'ROUTE_GAIN_BOUNDS', 'EVENT_KINDS', 'MYELIN_BOUNDS']

W_MIN = 0.01
W_MAX = 2.0
TRACE_MAX = 1.5
ROUTING_BOUNDS = (0.62, 1.14)
UNCERTAINTY_BOUNDS = (0.58, 1.0)
ROUTE_GAIN_BOUNDS = (0.0, 2.0)
LEARNED_ROUTE_BOUNDS = (0.5, 1.5)
LR_SCALE_BOUNDS = (0.25, 4.0)
MYELIN_BOUNDS = (0.2, 0.98)
UTILITY_MAX = 10.0
MAX_DELAY = 3
EVENT_KINDS = ('prune', 'grow', 'rewire', 'neuron_birth', 'repurpose')
STDP_A_PLUS = 0.01
STDP_A_MINUS = 0.0105

EDGE_FLOAT = ('w', 'trace', 'eligibility', 'utility', 'pre_spike', 'post_spike', 'routing_gain', 'myelin',
              'last_signal', 'lr_scale', 'last_dw', 'last_pre')
EDGE_INT = ('src', 'dst', 'delay', 'updates', 'created_step', 'compartment')
EDGE_ARRAYS = EDGE_INT + EDGE_FLOAT


def quantise_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor signed INT8 (127-level) quantisation that never zeroes or flips a sign."""
    w = np.asarray(w, dtype=np.float64)
    scale = float(np.max(np.abs(w))) if w.size else 1.0
    if scale <= 0:
        return np.zeros(w.shape, dtype=np.int8), 1.0
    q = np.clip(np.round(w / scale * 127.0), -127, 127)
    q = np.where((q == 0) & (w != 0), np.sign(w), q)
    return q.astype(np.int8), scale


def dequantise_int8(q: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * (float(scale) / 127.0)


class EventLedger:
    """Bounded structural-event ledger with cumulative counters by kind and reason."""

    def __init__(self, max_events: int = 512):
        self.max_events = int(max_events)
        self.events: deque = deque(maxlen=self.max_events)
        self.counts_by_kind: dict[str, int] = {k: 0 for k in EVENT_KINDS}
        self.counts_by_reason: dict[str, int] = {}
        self.total = 0

    def record(self, *, step: int, kind: str, src: int, dst: int, reason: str, utility: float = 0.0) -> dict:
        if kind not in EVENT_KINDS:
            raise ValueError(f'unknown event kind {kind!r}')
        event = {'step': int(step), 'kind': kind, 'src': int(src), 'dst': int(dst), 'reason': str(reason),
                 'utility': float(utility) if np.isfinite(utility) else 0.0}
        self.events.append(event)
        self.counts_by_kind[kind] += 1
        self.counts_by_reason[event['reason']] = self.counts_by_reason.get(event['reason'], 0) + 1
        self.total += 1
        return event

    def export_state(self) -> dict[str, Any]:
        return {'max_events': self.max_events, 'events': [dict(e) for e in self.events],
                'counts_by_kind': dict(self.counts_by_kind), 'counts_by_reason': dict(self.counts_by_reason),
                'total': int(self.total)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> 'EventLedger':
        if not isinstance(state, Mapping):
            raise ValueError('ledger must be a mapping')
        m = state.get('max_events')
        if type(m) is not int or not 1 <= m <= 100_000:
            raise ValueError('ledger max_events invalid')
        events = state.get('events')
        if not isinstance(events, list) or len(events) > m:
            raise ValueError('ledger events invalid')
        for e in events:
            if (not isinstance(e, Mapping) or e.get('kind') not in EVENT_KINDS or not isinstance(e.get('reason'), str)
                    or any(type(e.get(k)) is not int for k in ('step', 'src', 'dst'))
                    or not isinstance(e.get('utility'), (int, float)) or not np.isfinite(e['utility'])):
                raise ValueError('ledger event malformed')
        kinds = state.get('counts_by_kind')
        reasons = state.get('counts_by_reason')
        if (not isinstance(kinds, Mapping) or set(kinds) != set(EVENT_KINDS)
                or any(type(v) is not int or v < 0 for v in kinds.values())):
            raise ValueError('ledger counts_by_kind invalid')
        if not isinstance(reasons, Mapping) or any(type(v) is not int or v < 0 for v in reasons.values()):
            raise ValueError('ledger counts_by_reason invalid')
        if type(state.get('total')) is not int or state['total'] < 0:
            raise ValueError('ledger total invalid')
        led = cls(m)
        for e in events:
            led.events.append({'step': e['step'], 'kind': e['kind'], 'src': e['src'], 'dst': e['dst'],
                               'reason': e['reason'], 'utility': float(e['utility'])})
        led.counts_by_kind = {k: int(kinds[k]) for k in EVENT_KINDS}
        led.counts_by_reason = {str(k): int(v) for k, v in reasons.items()}
        led.total = int(state['total'])
        return led


class SynapseTable:
    """Directed signed synapse arrays plus a delay ring buffer of past outputs."""

    def __init__(self, n: int, region: np.ndarray, sign: np.ndarray, *, regions: Sequence = REGIONS,
                 seed: int = 0, max_events: int = 512):
        self.n = int(n)
        self.region = np.asarray(region, dtype=np.int64).copy()
        self.sign = np.asarray(sign, dtype=np.int8).copy()
        self.regions = tuple(tuple(r) for r in regions)
        self.kind_rank = region_kind_rank(self.regions)
        self.seed = int(seed)
        for name in EDGE_INT:
            setattr(self, name, np.zeros(0, dtype=np.int64))
        for name in EDGE_FLOAT:
            setattr(self, name, np.zeros(0, dtype=np.float64))
        self.hist = np.zeros((MAX_DELAY, self.n))
        self.hist_count = 0
        self.coact = np.zeros(self.n)
        r = len(self.regions)
        self.route_gain = np.ones((r, r))
        self.ledger = EventLedger(max_events)
        self.version = 0
        self.profile: dict[str, int] = {'edge_traversals_last': 0, 'edge_traversals_total': 0, 'tiles_last': 0,
                                        'quantised_last': 0}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_topology(cls, topo: Topology, seed: int) -> 'SynapseTable':
        topo.validate()
        table = cls(topo.n, topo.region, topo.sign, regions=topo.regions, seed=seed)
        rng = np.random.default_rng(int(seed) + 3_011)
        e = topo.src.size
        table.src = topo.src.astype(np.int64).copy()
        table.dst = topo.dst.astype(np.int64).copy()
        table.w = topo.weight.astype(np.float64).copy()
        table.delay = topo.delay.astype(np.int64).copy()
        table.updates = np.zeros(e, dtype=np.int64)
        table.created_step = np.zeros(e, dtype=np.int64)
        table.compartment = edge_compartments(table.src, table.dst, table.region, table.kind_rank).astype(np.int64)
        for name in EDGE_FLOAT:
            if name != 'w':
                setattr(table, name, np.zeros(e))
        table.routing_gain = np.ones(e)
        table.lr_scale = np.ones(e)
        table.myelin = 0.5 + 0.4 * rng.random(e)
        return table

    @property
    def size(self) -> int:
        return int(self.src.size)

    def copy(self) -> 'SynapseTable':
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ transmission
    def push(self, output: np.ndarray) -> None:
        """Record the newest output into the delay ring buffer (hist[0] = newest)."""
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (self.n,):
            raise ValueError('output length mismatch')
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = output
        self.hist_count = min(MAX_DELAY, self.hist_count + 1)

    def effective_delay(self, flags: Iterable[str]) -> np.ndarray:
        flags = frozenset(flags)
        if 'synaptic_delays' not in flags:
            return np.ones(self.size, dtype=np.int64)
        d = self.delay.copy()
        if 'adaptive_conduction' in flags:
            d = np.maximum(1, d - (self.myelin > 0.78).astype(np.int64))
        return d

    def transmit(self, output: np.ndarray, flags: Iterable[str], *, tile_size: int | None = None,
                 uncertainty_gate: float = 1.0, route_gain: np.ndarray | None = None, cells: Any = None,
                 src_weight: np.ndarray | None = None) -> CompartmentDrive:
        flags = frozenset(flags)
        output = np.asarray(output, dtype=np.float64)
        if output.shape != (self.n,) or not np.all(np.isfinite(output)):
            raise ValueError('output must be a finite vector of length n')
        src, dst = self.src, self.dst
        pre = output[src]
        if 'synaptic_delays' in flags:
            d = self.effective_delay(flags)
            for k in range(2, MAX_DELAY + 1):
                if self.hist_count >= k:
                    sel = d == k
                    if sel.any():
                        pre[sel] = self.hist[k - 1][src[sel]]
        if 'quantised_synapses' in flags:
            q, scale = quantise_int8(self.w)
            w = dequantise_int8(q, scale)
            self.profile['quantised_last'] = 1
        else:
            w = self.w
            self.profile['quantised_last'] = 0
        signal = pre * w
        if 'adaptive_routing' in flags and cells is not None:
            routing = np.clip(0.86 + 0.24 * np.tanh(np.abs(cells.gating[dst]) + 0.3 * np.abs(cells.apical[dst]) - 0.12),
                              *ROUTING_BOUNDS)
            self.routing_gain = routing
            signal = signal * routing
        if 'uncertainty_gating' in flags:
            signal = signal * float(np.clip(uncertainty_gate, *UNCERTAINTY_BOUNDS))
        if 'multi_region_routing' in flags:
            rg = self.route_gain if route_gain is None else self.validate_route_gain(route_gain)
            signal = signal * rg[self.region[src], self.region[dst]]
        if 'adaptive_conduction' in flags:
            signal = signal * (0.85 + 0.25 * self.myelin)
        if src_weight is not None:
            sw = np.asarray(src_weight, dtype=np.float64)
            if sw.shape != (self.n,) or not np.all(np.isfinite(sw)):
                raise ValueError('src_weight must be finite with length n')
            signal = signal * sw[src]
        self.last_signal = signal
        self.last_pre = pre

        n = self.n
        comp = self.compartment
        e = self.size
        acc = np.zeros((4, n))
        if 'tiled_execution' in flags and tile_size:
            tile = max(1, int(tile_size))
            tiles = 0
            for base in range(0, e, tile):
                sl = slice(base, min(e, base + tile))
                idx = comp[sl] * n + dst[sl]
                acc += np.bincount(idx, weights=signal[sl], minlength=4 * n).reshape(4, n)
                tiles += 1
        else:
            acc += np.bincount(comp * n + dst, weights=signal, minlength=4 * n).reshape(4, n)
            tiles = 1 if e else 0
        self.profile['edge_traversals_last'] = e
        self.profile['edge_traversals_total'] += e
        self.profile['tiles_last'] = tiles
        return CompartmentDrive(proximal=acc[0], distal=acc[1], apical=acc[2], basal=acc[3], external=np.zeros(n))

    def update_myelin(self, scale: float = 1.0) -> None:
        """Activity-dependent myelin proxy (learning-time only): grows with recent presynaptic activity."""
        if self.last_pre.shape != self.src.shape:
            return
        s = float(np.clip(scale, 0.0, 4.0))
        self.myelin = np.clip(self.myelin + s * (0.0005 * np.abs(self.last_pre) - 0.00008), *MYELIN_BOUNDS)

    def validate_route_gain(self, route_gain: Any) -> np.ndarray:
        r = len(self.regions)
        rg = np.asarray(route_gain, dtype=np.float64)
        if rg.shape != (r, r) or not np.all(np.isfinite(rg)):
            raise ValueError(f'route_gain must be a finite ({r}, {r}) matrix')
        return np.clip(rg, *ROUTE_GAIN_BOUNDS)

    # ------------------------------------------------------------------ learning
    def update_traces(self, post: np.ndarray) -> None:
        post = np.asarray(post, dtype=np.float64)
        pre = self.last_pre if self.last_pre.shape == self.src.shape else np.zeros(self.size)
        pp = pre * post[self.dst]
        self.trace = np.clip(0.92 * self.trace + 0.08 * pp, -TRACE_MAX, TRACE_MAX)
        self.eligibility = np.clip(0.96 * self.eligibility + 0.04 * pp, -TRACE_MAX, TRACE_MAX)
        self.utility = np.clip(0.995 * self.utility + 0.005 * np.abs(pp), 0.0, UTILITY_MAX)
        self.coact = np.clip(0.95 * self.coact + 0.05 * np.abs(post), 0.0, 1.0)

    def _apply_dw(self, dw: np.ndarray, flags: frozenset) -> int:
        if 'meta_plasticity' in flags:
            dw = dw * self.lr_scale
        mag = np.abs(self.w)
        new_mag = np.clip(mag + dw, W_MIN, W_MAX)
        actual = new_mag - mag
        changed = actual != 0
        self.w = self.sign[self.src].astype(np.float64) * new_mag
        if 'meta_plasticity' in flags:
            agree = changed & (np.sign(actual) == np.sign(self.last_dw)) & (self.last_dw != 0)
            disagree = changed & (np.sign(actual) == -np.sign(self.last_dw)) & (self.last_dw != 0)
            self.lr_scale = np.clip(np.where(agree, self.lr_scale * 1.05, np.where(disagree, self.lr_scale * 0.8, self.lr_scale)),
                                    *LR_SCALE_BOUNDS)
        self.last_dw = np.where(changed, actual, self.last_dw)
        self.updates = self.updates + changed.astype(np.int64)
        return int(changed.sum())

    def reward_update(self, reward: float, lr: float, flags: Iterable[str]) -> int:
        """Reward-gated eligibility update; bounded and sign-preserving. Returns #changed synapses."""
        flags = frozenset(flags)
        r = float(np.clip(reward, -1.0, 1.0))
        if r == 0.0 or lr <= 0 or not self.size:
            return 0
        dw = float(lr) * r * (self.trace + 0.2 * self.eligibility)
        return self._apply_dw(dw, flags)

    def stdp_update(self, spikes: np.ndarray, flags: Iterable[str], *, scale: float = 1.0) -> int:
        """Pair-based STDP: post spike with recent pre trace potentiates, pre spike with recent post trace depresses."""
        flags = frozenset(flags)
        spikes = np.asarray(spikes, dtype=np.float64)
        if spikes.shape != (self.n,):
            raise ValueError('spikes length mismatch')
        pre_s, post_s = spikes[self.src], spikes[self.dst]
        dw = float(scale) * (STDP_A_PLUS * post_s * self.pre_spike - STDP_A_MINUS * pre_s * self.post_spike)
        changed = self._apply_dw(dw, flags) if self.size and np.any(dw != 0) else 0
        self.pre_spike = np.clip(0.92 * self.pre_spike + pre_s, 0.0, TRACE_MAX)
        self.post_spike = np.clip(0.92 * self.post_spike + post_s, 0.0, TRACE_MAX)
        return changed

    def metaplastic_scale(self) -> np.ndarray:
        return self.lr_scale.copy()

    def update_route_gain(self, reward: float, post: np.ndarray, lr: float = 0.02) -> None:
        """Reward x region-pair co-activity adapts the region routing matrix (bounded)."""
        r = float(np.clip(reward, -1.0, 1.0))
        if r == 0.0 or not self.size:
            return
        nr = len(self.regions)
        pp = self.last_pre * np.asarray(post)[self.dst]
        flat = self.region[self.src] * nr + self.region[self.dst]
        tot = np.bincount(flat, weights=pp, minlength=nr * nr)
        cnt = np.bincount(flat, minlength=nr * nr)
        co = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0).reshape(nr, nr)
        self.route_gain = np.clip(self.route_gain + lr * r * co, *LEARNED_ROUTE_BOUNDS)

    def downscale(self, factor: float, *, protect: np.ndarray | None = None) -> None:
        """Multiplicative synaptic downscaling (sign-preserving, bounded).

        ``protect`` (per-edge values in [0, 1]) shields consolidated synapses:
        the per-edge factor is ``f + (1 - f) * protect`` (1 = fully protected).
        """
        f = float(factor)
        if not 0.0 < f <= 1.0:
            raise ValueError('downscale factor must be in (0, 1]')
        fe: Any = f
        if protect is not None:
            p = np.asarray(protect, dtype=np.float64)
            if p.shape != (self.size,) or not np.all(np.isfinite(p)):
                raise ValueError('protect must be a finite per-edge array')
            fe = f + (1.0 - f) * np.clip(p, 0.0, 1.0)
        self.w = self.sign[self.src].astype(np.float64) * np.clip(np.abs(self.w) * fe, W_MIN, W_MAX)
        self.utility = self.utility * fe

    def consolidate(self) -> dict[str, float]:
        """Sleep-like consolidation: decay transient traces, relax meta-plastic rates toward 1."""
        before = float(np.abs(self.trace).mean()) if self.size else 0.0
        self.trace = 0.5 * self.trace
        self.eligibility = 0.5 * self.eligibility
        self.pre_spike = np.zeros(self.size)
        self.post_spike = np.zeros(self.size)
        self.lr_scale = 1.0 + 0.5 * (self.lr_scale - 1.0)
        return {'trace_before': before, 'trace_after': float(np.abs(self.trace).mean()) if self.size else 0.0}

    # ------------------------------------------------------------------ structure
    def add_neurons(self, regions: Sequence[int], signs: Sequence[int]) -> None:
        reg = np.asarray(regions, dtype=np.int64)
        sg = np.asarray(signs, dtype=np.int8)
        k = reg.size
        if sg.shape != (k,) or not np.all(np.isin(sg, (-1, 1))):
            raise ValueError('signs must be +1/-1 per new neuron')
        self.region = np.concatenate([self.region, reg])
        self.sign = np.concatenate([self.sign, sg])
        self.hist = np.concatenate([self.hist, np.zeros((MAX_DELAY, k))], axis=1)
        self.coact = np.concatenate([self.coact, np.zeros(k)])
        self.n += k
        self.version += 1

    def has_edge(self, a: int, b: int) -> bool:
        return bool(np.any((self.src == a) & (self.dst == b)))

    def add_edges(self, src: Sequence[int], dst: Sequence[int], *, step: int, rng: np.random.Generator,
                  magnitude: np.ndarray | None = None) -> int:
        s = np.asarray(src, dtype=np.int64)
        d = np.asarray(dst, dtype=np.int64)
        if s.size == 0:
            return 0
        if s.min() < 0 or d.min() < 0 or s.max() >= self.n or d.max() >= self.n or np.any(s == d):
            raise ValueError('invalid new edge endpoints')
        k = s.size
        mag = 0.12 + 0.2 * rng.random(k) if magnitude is None else np.clip(np.asarray(magnitude, dtype=np.float64), W_MIN, W_MAX)
        delay = (1 + np.floor(rng.random(k) * 3)).astype(np.int64)
        new = {
            'src': s, 'dst': d, 'w': self.sign[s].astype(np.float64) * mag, 'delay': np.clip(delay, 1, MAX_DELAY),
            'updates': np.zeros(k, np.int64), 'created_step': np.full(k, int(step), np.int64),
            'compartment': edge_compartments(s, d, self.region, self.kind_rank).astype(np.int64),
            'routing_gain': np.ones(k), 'myelin': np.full(k, 0.5), 'lr_scale': np.ones(k),
        }
        for name in EDGE_ARRAYS:
            add = new.get(name, np.zeros(k))
            setattr(self, name, np.concatenate([getattr(self, name), add.astype(getattr(self, name).dtype)]))
        self.version += 1
        return k

    def remove_edges(self, mask: np.ndarray) -> int:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError('edge mask length mismatch')
        keep = ~mask
        for name in EDGE_ARRAYS:
            setattr(self, name, getattr(self, name)[keep])
        self.version += 1
        return int(mask.sum())

    def _sample_growth(self, count: int, rng: np.random.Generator, candidates: int = 32) -> list[tuple[int, int, float]]:
        n = self.n
        keys = set((self.src * n + self.dst).tolist())
        chosen: list[tuple[int, int, float]] = []
        for _ in range(count):
            a = rng.integers(0, n, candidates)
            b = rng.integers(0, n, candidates)
            score = np.abs(self.coact[a] * self.coact[b]) + 0.015 * rng.random(candidates)
            best = None
            for j in np.argsort(-score, kind='stable'):
                aj, bj = int(a[j]), int(b[j])
                if aj != bj and aj * n + bj not in keys:
                    best = (aj, bj, float(score[j]))
                    break
            if best is None:
                continue
            keys.add(best[0] * n + best[1])
            chosen.append(best)
        return chosen

    def structural_step(self, step: int, rng: np.random.Generator, max_synapses: int) -> list[dict]:
        """Prune low-utility synapses and grow high co-activity ones under a hard cap."""
        events: list[dict] = []
        max_synapses = int(max_synapses)
        if self.size > max_synapses:
            score = self.utility + 0.15 * np.abs(self.trace)
            drop = np.argsort(score, kind='stable')[: self.size - max_synapses]
            mask = np.zeros(self.size, dtype=bool)
            mask[drop] = True
            for i in drop:
                events.append(self.ledger.record(step=step, kind='prune', src=self.src[i], dst=self.dst[i],
                                                 reason='capacity', utility=self.utility[i]))
            self.remove_edges(mask)
        k = 0
        if self.size >= max(20, 2 * self.n):
            k = max(1, int(self.size * 0.008))
            score = self.utility + 0.15 * np.abs(self.trace)
            drop = np.argsort(score, kind='stable')[:k]
            mask = np.zeros(self.size, dtype=bool)
            mask[drop] = True
            for i in drop:
                events.append(self.ledger.record(step=step, kind='prune', src=self.src[i], dst=self.dst[i],
                                                 reason='low_utility', utility=self.utility[i]))
            self.remove_edges(mask)
        grow = min(max(k, 1), max(0, max_synapses - self.size))
        picks = self._sample_growth(grow, rng)
        if picks:
            self.add_edges([p[0] for p in picks], [p[1] for p in picks], step=step, rng=rng)
            for a, b, sc in picks:
                events.append(self.ledger.record(step=step, kind='grow', src=a, dst=b, reason='high_coactivity', utility=sc))
        return events

    def lesion(self, mask_edges: np.ndarray) -> 'SynapseTable':
        """Return a lesioned copy with masked edges removed; the original is untouched."""
        mask = np.asarray(mask_edges, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError('lesion mask length mismatch')
        clone = self.copy()
        clone.remove_edges(mask)
        return clone

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {name: getattr(self, name).tolist() for name in EDGE_ARRAYS}
        state.update({
            'n': self.n, 'seed': self.seed, 'region': self.region.tolist(), 'sign': self.sign.astype(int).tolist(),
            'regions': [list(r) for r in self.regions], 'hist': self.hist.tolist(), 'hist_count': int(self.hist_count),
            'coact': self.coact.tolist(), 'route_gain': self.route_gain.tolist(), 'version': int(self.version),
            'ledger': self.ledger.export_state(),
        })
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> 'SynapseTable':
        """Validate a full exported state and build a new table (never mutates anything)."""
        if not isinstance(state, Mapping):
            raise ValueError('synapse state must be a mapping')
        n = state.get('n')
        if type(n) is not int or n < 1:
            raise ValueError('synapse n invalid')
        if type(state.get('seed')) is not int or type(state.get('version')) is not int:
            raise ValueError('synapse seed/version invalid')
        regions = state.get('regions')
        if not isinstance(regions, list) or not regions or not all(isinstance(r, (list, tuple)) and len(r) == 2 for r in regions):
            raise ValueError('synapse regions invalid')
        nr = len(regions)
        try:
            region = np.asarray(state['region'], dtype=np.int64)
            sign = np.asarray(state['sign'], dtype=np.int64)
            hist = np.asarray(state['hist'], dtype=np.float64)
            coact = np.asarray(state['coact'], dtype=np.float64)
            rg = np.asarray(state['route_gain'], dtype=np.float64)
            arrays = {name: np.asarray(state[name], dtype=np.int64 if name in EDGE_INT else np.float64) for name in EDGE_ARRAYS}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'synapse state malformed: {exc}') from exc
        if region.shape != (n,) or sign.shape != (n,) or not np.all(np.isin(sign, (-1, 1))):
            raise ValueError('synapse region/sign malformed')
        if region.size and (region.min() < 0 or region.max() >= nr):
            raise ValueError('synapse region index out of range')
        if hist.shape != (MAX_DELAY, n) or not np.all(np.isfinite(hist)) or np.abs(hist).max(initial=0) > 1.0 + 1e-9:
            raise ValueError('synapse delay buffer malformed')
        hc = state.get('hist_count')
        if type(hc) is not int or not 0 <= hc <= MAX_DELAY:
            raise ValueError('synapse hist_count invalid')
        if coact.shape != (n,) or not np.all(np.isfinite(coact)) or (coact.size and (coact.min() < 0 or coact.max() > 1.0 + 1e-9)):
            raise ValueError('synapse coact malformed')
        if rg.shape != (nr, nr) or not np.all(np.isfinite(rg)) or rg.min() < ROUTE_GAIN_BOUNDS[0] or rg.max() > ROUTE_GAIN_BOUNDS[1]:
            raise ValueError('synapse route_gain malformed')
        e = arrays['src'].size
        for name, arr in arrays.items():
            if arr.shape != (e,):
                raise ValueError(f'synapse array {name} length mismatch')
            if arr.dtype == np.float64 and not np.all(np.isfinite(arr)):
                raise ValueError(f'synapse array {name} not finite')
        s, d, w = arrays['src'], arrays['dst'], arrays['w']
        if e:
            if s.min() < 0 or d.min() < 0 or s.max() >= n or d.max() >= n or np.any(s == d):
                raise ValueError('synapse endpoints invalid')
            if np.unique(s * n + d).size != e:
                raise ValueError('duplicate synapses')
            if np.any(np.sign(w) != sign[s]) or np.abs(w).min() < W_MIN - 1e-12 or np.abs(w).max() > W_MAX + 1e-12:
                raise ValueError('synapse weight sign/bounds invalid')
            if arrays['delay'].min() < 1 or arrays['delay'].max() > MAX_DELAY:
                raise ValueError('synapse delay out of range')
            if arrays['compartment'].min() < 0 or arrays['compartment'].max() > 3:
                raise ValueError('synapse compartment invalid')
            if arrays['lr_scale'].min() < LR_SCALE_BOUNDS[0] - 1e-12 or arrays['lr_scale'].max() > LR_SCALE_BOUNDS[1] + 1e-12:
                raise ValueError('synapse lr_scale out of bounds')
            if max(arrays['pre_spike'].max(), arrays['post_spike'].max()) > TRACE_MAX + 1e-12 or min(arrays['pre_spike'].min(), arrays['post_spike'].min()) < 0:
                raise ValueError('synapse spike traces out of bounds')
            bounds = {'trace': (-TRACE_MAX, TRACE_MAX), 'eligibility': (-TRACE_MAX, TRACE_MAX),
                      'utility': (0.0, UTILITY_MAX), 'routing_gain': ROUTING_BOUNDS, 'myelin': MYELIN_BOUNDS,
                      'last_pre': (-1.0, 1.0), 'last_dw': (-W_MAX, W_MAX)}
            for name, (lo, hi) in bounds.items():
                arr = arrays[name]
                if arr.min() < lo - 1e-9 or arr.max() > hi + 1e-9:
                    raise ValueError(f'synapse array {name} out of bounds [{lo}, {hi}]')
            if arrays['updates'].min() < 0 or arrays['created_step'].min() < 0:
                raise ValueError('synapse counters must be non-negative')
        ledger = EventLedger.from_state(state.get('ledger'))
        table = cls(n, region, sign.astype(np.int8), regions=[tuple(r) for r in regions], seed=int(state['seed']),
                    max_events=ledger.max_events)
        for name, arr in arrays.items():
            setattr(table, name, arr.copy())
        table.hist = hist.copy()
        table.hist_count = hc
        table.coact = coact.copy()
        table.route_gain = rg.copy()
        table.version = int(state['version'])
        table.ledger = ledger
        return table

    def import_state(self, state: Mapping[str, Any]) -> None:
        other = SynapseTable.from_state(state)
        self.__dict__.update(other.__dict__)
