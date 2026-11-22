"""NeuroWeaveBrain: topology + multi-compartment cells + synapse table + readout.

With zero mechanisms the step is the NEUROWEAVE v0.1 baseline update; every
mechanism consumed here (cells, connectome and brain-level representation
flags) changes the computation only when enabled. The brain is a synthetic
computational abstraction with no claim of biological fidelity.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

import numpy as np

from .cells import CellPopulation, NEURON_FLAGS, validate_controls
from .connectome import EventLedger, SynapseTable
from .spec import NeuroWeaveSpecV1, REGIONS
from .topology import KIND_RANK, build_topology

__all__ = ['NeuroWeaveBrain', 'BRAIN_SCHEMA', 'BRAIN_FLAGS', 'CONNECTOME_FLAGS', 'CONSUMED_FLAGS']

BRAIN_SCHEMA = 'NeuroWeaveBrainStateV1'
BRAIN_FLAGS: tuple[str, ...] = ('structural_encoder', 'context_attention', 'flyhash_projection', 'sparse_code',
                                'neurogenesis', 'sleep_consolidation')
CONNECTOME_FLAGS: tuple[str, ...] = ('synaptic_delays', 'adaptive_conduction', 'quantised_synapses', 'adaptive_routing',
                                     'uncertainty_gating', 'multi_region_routing', 'reward_plasticity', 'stdp',
                                     'meta_plasticity', 'structural_plasticity', 'tiled_execution')
CONSUMED_FLAGS: tuple[str, ...] = tuple(dict.fromkeys(NEURON_FLAGS + BRAIN_FLAGS + CONNECTOME_FLAGS))
_PLASTIC = frozenset({'reward_plasticity', 'stdp', 'structural_plasticity', 'meta_plasticity', 'multi_region_routing',
                      'neurogenesis'})
_KIND_INPUT_GAIN = {'sensory': 1.0, 'association': 0.6, 'memory': 0.4, 'modulatory': 0.5, 'action': 0.3}
_MAX_INPUT_DIM = 256
_READOUT_OFFSET = 331
_BRAIN_OFFSET = 4_001
_PROJ_OFFSET = 5_003
_HASH_OFFSET = 6_007
HASH_CHANNELS = 16
HASH_WINNERS = 4


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-z))


class NeuroWeaveBrain:
    def __init__(self, spec: NeuroWeaveSpecV1 | None = None, mechanisms: Iterable[str] | None = None):
        spec = spec if spec is not None else NeuroWeaveSpecV1()
        if not isinstance(spec, NeuroWeaveSpecV1):
            raise ValueError('spec must be a NeuroWeaveSpecV1')
        if mechanisms is not None:
            spec = spec.with_mechanisms(tuple(mechanisms))
        self.spec = spec
        self.flags = frozenset(spec.mechanisms)
        self.regions = REGIONS
        topo = build_topology(spec.topology, spec.neurons, seed=spec.seed, fan_in=spec.fan_in,
                              inhibitory_fraction=spec.inhibitory_fraction, regions=REGIONS)
        self.cells = CellPopulation(topo.n, seed=spec.seed, regions=topo.region, signs=topo.sign)
        self.synapses = SynapseTable.from_topology(topo, seed=spec.seed)
        rr = np.random.default_rng(spec.seed + _READOUT_OFFSET)
        self.readout_w = (rr.random(self.feature_dim) - 0.5) * 0.12
        self.readout_b = 0.0
        self.readout_lr = 0.035
        self.rng = np.random.default_rng(spec.seed + _BRAIN_OFFSET)
        self.step_count = 0
        self.counters = {'sleep_cycles': 0, 'neurons_born': 0, 'repurposed': 0, 'structural_steps': 0}
        self._proj_cache: dict[int, np.ndarray] = {}
        self._hash_cache: dict[int, np.ndarray] = {}
        self._fp_version = -1
        self._fp_weight: np.ndarray | None = None
        self._region_counts_n = -1
        self._region_counts: np.ndarray | None = None
        self.measured: dict[str, float] = {'wallclock_ms_last': 0.0, 'wallclock_ms_total': 0.0,
                                           'edge_traversals_last': 0, 'edge_traversals_total': 0, 'tiles_last': 0,
                                           'steps': 0}
        self.estimated: dict[str, float] = {'ops_last': 0, 'ops_total': 0, 'bytes': 0}

    # ------------------------------------------------------------------ properties
    @property
    def n(self) -> int:
        return self.cells.n

    @property
    def feature_dim(self) -> int:
        return 8 + 2 * len(self.regions)

    @property
    def output(self) -> np.ndarray:
        return self.cells.output.copy()

    @property
    def ledger(self) -> EventLedger:
        return self.synapses.ledger

    def profile(self) -> dict[str, dict[str, float]]:
        return {'measured': dict(self.measured), 'estimated': dict(self.estimated)}

    # ------------------------------------------------------------------ input
    def _prepare_input(self, external: Any) -> np.ndarray:
        x = np.asarray(external if external is not None else [], dtype=np.float64).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError('external input must be finite')
        x = np.clip(x, -10.0, 10.0)
        if x.size > _MAX_INPUT_DIM:  # fixed folding of long inputs into 256 channels
            bins = np.arange(x.size) * _MAX_INPUT_DIM // x.size
            x = np.bincount(bins, weights=x, minlength=_MAX_INPUT_DIM) / np.maximum(1, np.bincount(bins, minlength=_MAX_INPUT_DIM))
        return x

    def _projection(self, d: int) -> np.ndarray:
        if d not in self._proj_cache:
            rng = np.random.default_rng(self.spec.seed + _PROJ_OFFSET + d)
            self._proj_cache[d] = 0.3 * rng.standard_normal((self.spec.max_neurons, d)) / np.sqrt(d)
        return self._proj_cache[d]

    def _hash_matrix(self, d: int) -> np.ndarray:
        if d not in self._hash_cache:
            rng = np.random.default_rng(self.spec.seed + _HASH_OFFSET + d)
            h = np.zeros((HASH_CHANNELS, d))
            fan = max(1, int(round(0.3 * d)))
            for c in range(HASH_CHANNELS):
                h[c, rng.choice(d, size=fan, replace=False)] = 1.0
            self._hash_cache[d] = h
        return self._hash_cache[d]

    def flyhash_code(self, x: np.ndarray) -> np.ndarray:
        """FlyHash-style sparse binary code: fixed sparse projection + top-k winners."""
        x = self._prepare_input(x)
        code = np.zeros(HASH_CHANNELS)
        if x.size:
            act = self._hash_matrix(x.size) @ x
            code[np.argsort(-act, kind='stable')[:HASH_WINNERS]] = 1.0
        return code

    def project_input(self, external: Any) -> np.ndarray:
        x = self._prepare_input(external)
        n = self.n
        ext = np.zeros(n)
        if x.size:
            idx = np.arange(n)
            polarity = np.where(idx % 2 == 0, 1.0, -0.75)
            kinds = np.array([_KIND_INPUT_GAIN.get(kind, 0.5) for _, kind in self.regions])[self.cells.region]
            ext = kinds * (polarity * x[idx % x.size] + self._projection(x.size)[:n] @ x)
        if 'flyhash_projection' in self.flags:
            code = self.flyhash_code(x)
            ext = ext + 0.16 * (code[np.arange(n) % HASH_CHANNELS] - HASH_WINNERS / HASH_CHANNELS)
        return ext

    # ------------------------------------------------------------------ graph-derived terms
    def structural_fingerprints(self) -> np.ndarray:
        """Per-neuron graph fingerprint: in/out degree, strength, reciprocity, two-hop reach."""
        syn, n = self.synapses, self.n
        indeg = np.bincount(syn.dst, minlength=n).astype(np.float64)
        outdeg = np.bincount(syn.src, minlength=n).astype(np.float64)
        s_in = np.bincount(syn.dst, weights=np.abs(syn.w), minlength=n)
        keys = syn.src * n + syn.dst
        recip_edge = np.isin(syn.dst * n + syn.src, keys).astype(np.float64)
        recip = np.bincount(syn.dst, weights=recip_edge, minlength=n) / np.maximum(1.0, indeg)
        two_hop = np.bincount(syn.dst, weights=indeg[syn.src], minlength=n) / np.maximum(1.0, indeg * n)
        return np.stack([indeg / n, outdeg / n, s_in / np.maximum(1.0, indeg), recip, two_hop], axis=1)

    def _structural_weight(self) -> np.ndarray:
        if self._fp_weight is None or self._fp_version != self.synapses.version or self._fp_weight.size != self.n:
            fp = self.structural_fingerprints()
            out_rel = fp[:, 1] / max(1e-9, fp[:, 1].mean())
            self._fp_weight = np.clip(1.0 + 0.35 * np.tanh(out_rel - 1.0) + 0.2 * fp[:, 3] - 0.1 + 0.5 * fp[:, 4], 0.75, 1.5)
            self._fp_version = self.synapses.version
        return self._fp_weight

    def context_attention(self, prev: np.ndarray) -> np.ndarray:
        """Softmax attention over each unit's (up to) eight strongest incoming neighbours."""
        syn, n = self.synapses, self.n
        att = np.zeros(n)
        if not syn.size:
            return att
        score = np.abs(syn.w * prev[syn.src])
        order = np.lexsort((-score, syn.dst))
        d_sorted = syn.dst[order]
        starts = np.r_[0, np.flatnonzero(np.diff(d_sorted)) + 1]
        rank = np.arange(order.size) - np.repeat(starts, np.diff(np.r_[starts, order.size]))
        keep = order[rank < 8]
        q = 2.2 * score[keep]
        dk = syn.dst[keep]
        qmax = np.full(n, -np.inf)
        np.maximum.at(qmax, dk, q)
        ex = np.exp(q - qmax[dk])
        den = np.bincount(dk, weights=ex, minlength=n)
        num = np.bincount(dk, weights=ex * prev[syn.src[keep]], minlength=n)
        np.divide(num, den, out=att, where=den > 0)
        return att

    # ------------------------------------------------------------------ dynamics
    def step(self, external: Any = None, *, reward: float = 0.0, controls: Mapping[str, Any] | None = None,
             learn: bool = True, uncertainty: float = 0.0, route_gain: Any = None,
             unit_gain: Any = None, unit_input: Any = None) -> np.ndarray:
        started = time.perf_counter()
        reward = float(reward)
        uncertainty = float(uncertainty)
        if not (np.isfinite(reward) and np.isfinite(uncertainty)):
            raise ValueError('reward and uncertainty must be finite')
        validate_controls(controls, extra=('learning_rate_scale',))  # fail before any state is touched
        flags = self.flags
        n = self.n
        ug = None
        if unit_gain is not None:
            # optional per-unit excitability multiplier (receptor-weighted neuromodulation)
            ug = np.asarray(unit_gain, dtype=np.float64).ravel()
            if ug.size != n or not np.all(np.isfinite(ug)):
                raise ValueError(f'unit_gain must be a finite vector of length {n}')
            ug = np.clip(ug, 0.0, 4.0)
        ui = None
        if unit_input is not None:
            # optional per-unit additive feed-forward input (the engine's associative-recall and
            # fast-weight reads); None leaves the update exactly as before
            ui = np.asarray(unit_input, dtype=np.float64).ravel()
            if ui.size != n or not np.all(np.isfinite(ui)):
                raise ValueError(f'unit_input must be a finite vector of length {n}')
            ui = np.clip(ui, -10.0, 10.0)
        prev = self.cells.output.copy()
        ext = self.project_input(external)
        gate = float(np.clip(1.0 / (1.0 + 2.2 * max(0.0, uncertainty)), 0.58, 1.0)) if 'uncertainty_gating' in flags else 1.0
        rg = None
        if 'multi_region_routing' in flags and route_gain is not None:
            rg = self.synapses.validate_route_gain(route_gain)
        drive = self.synapses.transmit(
            prev, flags, tile_size=self.spec.tile_size if 'tiled_execution' in flags else None,
            uncertainty_gate=gate, route_gain=rg, cells=self.cells,
            src_weight=self._structural_weight() if 'structural_encoder' in flags else None)
        if 'context_attention' in flags:
            drive.apical = drive.apical + 0.3 * self.context_attention(prev)
        drive.external = ext if ug is None else ext * ug
        if ui is not None:
            drive.external = drive.external + ui        # added after any per-unit gain
        sparse_k = max(1, int(round(n * self.spec.sparsity)))
        out = self.cells.step(drive, flags, controls, sparse_k=sparse_k)
        self.synapses.push(out)
        if learn:
            if 'adaptive_conduction' in flags:
                ps = validate_controls(controls)['plasticity_scale']
                self.synapses.update_myelin(ps)
            self._learn(reward, controls, route_gain_given=rg is not None)
        self.step_count += 1
        if learn:
            self._periodic()
        self._profile(started, n)
        return out.copy()

    def frozen_step(self, external: Any = None, *, reward: float = 0.0, controls: Mapping[str, Any] | None = None,
                    uncertainty: float = 0.0, route_gain: Any = None) -> np.ndarray:
        """Identical forward math with all learning / structural change disabled."""
        return self.step(external, reward=reward, controls=controls, learn=False, uncertainty=uncertainty,
                         route_gain=route_gain)

    def _learn(self, reward: float, controls: Mapping[str, Any] | None, *, route_gain_given: bool) -> None:
        flags = self.flags
        if not flags & _PLASTIC:
            return
        c = dict(controls or {})
        ps = float(np.clip(c.get('plasticity_scale', 1.0) if c.get('plasticity_scale') is not None else 1.0, 0.0, 4.0))
        lrs = float(np.clip(c.get('learning_rate_scale', 1.0) if c.get('learning_rate_scale') is not None else 1.0, 0.0, 4.0))
        out = self.cells.output
        self.synapses.update_traces(out)
        if 'reward_plasticity' in flags and reward != 0.0:
            self.synapses.reward_update(reward, 0.02 * lrs * ps, flags)
        if 'stdp' in flags:
            self.synapses.stdp_update(self.cells.spike, flags, scale=ps * lrs)
        if 'multi_region_routing' in flags and reward != 0.0 and not route_gain_given:
            self.synapses.update_route_gain(reward, out, lr=0.05 * lrs)

    def _periodic(self) -> None:
        spec, flags, step = self.spec, self.flags, self.step_count
        if step % spec.structural_interval == 0:
            if 'structural_plasticity' in flags:
                self.synapses.structural_step(step, self.rng, spec.max_synapses)
                self.counters['structural_steps'] += 1
            if 'neurogenesis' in flags:
                self._neurogenesis()
        if 'sleep_consolidation' in flags and step % spec.sleep_interval == 0:
            self.sleep()

    def sleep(self) -> dict[str, float]:
        """Sleep-like consolidation: trace consolidation plus bounded synaptic downscaling."""
        c = self.cells
        c.consolidation = np.clip(0.96 * c.consolidation + 0.04 * np.abs(c.trace) * 10.0, 0.0, 2.0)
        c.trace = 0.998 * c.trace
        # consolidated destination units shield their incoming synapses from downscaling
        protect = np.clip(c.consolidation[self.synapses.dst] / 2.0, 0.0, 1.0)
        self.synapses.downscale(0.98, protect=protect)
        summary = self.synapses.consolidate()
        self.counters['sleep_cycles'] += 1
        return summary

    # ------------------------------------------------------------------ growth
    def _region_index(self, region: Any) -> int:
        if isinstance(region, str):
            names = [r[0] for r in self.regions]
            if region not in names:
                raise ValueError(f'unknown region {region!r}')
            return names.index(region)
        if type(region) is not int and not isinstance(region, np.integer):
            raise ValueError('region must be an index or a name')
        region = int(region)
        if not 0 <= region < len(self.regions):
            raise ValueError('region index out of range')
        return region

    def _ranked_partners(self, pool: np.ndarray, count: int, exclude: int) -> np.ndarray:
        pool = pool[pool != exclude]
        if pool.size == 0 or count <= 0:
            return np.zeros(0, dtype=np.int64)
        score = self.synapses.coact[pool] + 0.01 * self.rng.random(pool.size)
        return pool[np.argsort(-score, kind='stable')[:count]]

    def grow_neurons(self, k: int, region: Any, reason: str = 'manual') -> list[int]:
        """Add up to k neurons in `region` (capped by spec.max_neurons), wired by co-activity."""
        if type(k) is not int or k < 0:
            raise ValueError('k must be a non-negative int')
        r = self._region_index(region)
        room = self.spec.max_neurons - self.n
        k_eff = max(0, min(k, room))
        if k_eff == 0:
            return []
        syn = self.synapses
        signs = []
        members = self.cells.region == r
        inhib = int(np.sum(self.cells.sign[members] < 0))
        size = int(members.sum())
        for _ in range(k_eff):
            size += 1
            if inhib / size < self.spec.inhibitory_fraction:
                signs.append(-1)
                inhib += 1
            else:
                signs.append(1)
        new_ids = self.cells.grow(k_eff, r, signs)
        syn.add_neurons([r] * k_eff, signs)
        rank = KIND_RANK.get(self.regions[r][1], -1)
        kinds = np.array([KIND_RANK.get(kind, -1) for _, kind in self.regions])[self.cells.region]
        same = np.flatnonzero(self.cells.region == r)
        upstream = np.flatnonzero((kinds >= 0) & (kinds < rank)) if rank > 0 else np.zeros(0, np.int64)
        downstream = np.flatnonzero(kinds > rank) if rank >= 0 else np.zeros(0, np.int64)
        fan_in = self.spec.fan_in
        for nid in new_ids.tolist():
            src_pool = np.unique(np.concatenate([same, upstream]))
            srcs = self._ranked_partners(src_pool[src_pool < nid] if src_pool.size else src_pool, fan_in, nid)
            dst_pool = np.unique(np.concatenate([same, downstream]))
            dsts = self._ranked_partners(dst_pool, max(1, fan_in // 2), nid)
            pairs = [(int(s), nid) for s in srcs] + [(nid, int(d)) for d in dsts]
            pairs = [p for p in pairs if not syn.has_edge(*p)]
            room_syn = self._make_room(len(pairs), protect_from=int(new_ids[0]), reason='birth_capacity')
            pairs = pairs[:room_syn]
            if pairs:
                syn.add_edges([p[0] for p in pairs], [p[1] for p in pairs], step=self.step_count, rng=self.rng)
                for a, b in pairs:
                    syn.ledger.record(step=self.step_count, kind='grow', src=a, dst=b, reason='neuron_birth_wiring')
            syn.ledger.record(step=self.step_count, kind='neuron_birth', src=nid, dst=nid, reason=reason, utility=0.0)
        self.counters['neurons_born'] += k_eff
        self._fp_weight = None
        return new_ids.tolist()

    def _make_room(self, needed: int, *, protect_from: int | None = None, protect_dst: int | None = None,
                   reason: str = 'capacity') -> int:
        """Prune lowest-utility synapses (ledgered) so `needed` new ones fit under max_synapses."""
        syn = self.synapses
        deficit = syn.size + needed - self.spec.max_synapses
        if needed <= 0 or deficit <= 0:
            return max(0, needed)
        eligible = np.ones(syn.size, dtype=bool)
        if protect_from is not None:
            eligible &= (syn.src < protect_from) & (syn.dst < protect_from)
        if protect_dst is not None:
            eligible &= syn.dst != protect_dst
        cand = np.flatnonzero(eligible)
        score = syn.utility[cand] + 0.15 * np.abs(syn.trace[cand])
        drop = cand[np.argsort(score, kind='stable')[:deficit]]
        if drop.size:
            mask = np.zeros(syn.size, dtype=bool)
            mask[drop] = True
            for i in drop:
                syn.ledger.record(step=self.step_count, kind='prune', src=syn.src[i], dst=syn.dst[i], reason=reason,
                                  utility=syn.utility[i])
            syn.remove_edges(mask)
        return int(max(0, min(needed, self.spec.max_synapses - syn.size)))

    def repurpose_neuron(self, idx: int, reason: str = 'underused') -> int:
        """Reset an underused unit and rewire its inputs from co-active partners."""
        idx = int(idx)
        if not 0 <= idx < self.n:
            raise ValueError('neuron index out of range')
        syn = self.synapses
        utility = float(self.cells.utility[idx])
        self.cells.reset_units([idx])
        incoming = syn.dst == idx
        removed = int(incoming.sum())
        if removed:
            syn.remove_edges(incoming)
        r = int(self.cells.region[idx])
        pool = np.flatnonzero((self.cells.region == r) | (syn.coact > np.median(syn.coact)))
        srcs = self._ranked_partners(pool, max(1, min(self.spec.fan_in, removed if removed else self.spec.fan_in)), idx)
        srcs = srcs[:self._make_room(srcs.size, protect_dst=idx, reason='repurpose_capacity')]
        if srcs.size:
            syn.add_edges(srcs, np.full(srcs.size, idx), step=self.step_count, rng=self.rng)
            for s in srcs.tolist():
                syn.ledger.record(step=self.step_count, kind='rewire', src=s, dst=idx, reason=reason)
        syn.ledger.record(step=self.step_count, kind='repurpose', src=idx, dst=idx, reason=reason, utility=utility)
        self.counters['repurposed'] += 1
        self._fp_weight = None
        return idx

    def _neurogenesis(self) -> None:
        out = np.abs(self.cells.output)
        loads = np.array([out[self.cells.region == r].mean() if np.any(self.cells.region == r) else -1.0
                          for r in range(len(self.regions))])
        r = int(np.argmax(loads))
        if loads[r] > 0.25 and self.n < self.spec.max_neurons:
            self.grow_neurons(1, r, reason='region_load')
        u = self.cells.utility
        candidate = int(np.argmin(u))
        if u[candidate] < 0.25 * float(u.mean()):
            self.repurpose_neuron(candidate, reason='underused')

    # ------------------------------------------------------------------ readout
    def readout_features(self) -> np.ndarray:
        c = self.cells
        a = c.output
        abs_a = np.abs(a)
        nr = len(self.regions)
        out = np.empty(8 + 2 * nr, dtype=np.float64)
        out[0] = float(a.mean())
        out[1] = float(abs_a.mean())
        out[2] = float(a.std())
        out[3] = float(np.tanh(c.fast).mean())
        out[4] = float(np.tanh(c.slow).mean())
        out[5] = float(np.tanh(c.context).mean())
        out[6] = float(c.trace.mean())
        out[7] = float((abs_a > 0.1).mean())
        if self._region_counts_n != c.n or self._region_counts is None:
            self._region_counts = np.maximum(1, np.bincount(c.region, minlength=nr)[:nr]).astype(np.float64)
            self._region_counts_n = c.n
        inv_cnt = 1.0 / self._region_counts
        out[8:8 + nr] = np.bincount(c.region, weights=a, minlength=nr)[:nr] * inv_cnt
        out[8 + nr:] = np.bincount(c.region, weights=abs_a, minlength=nr)[:nr] * inv_cnt
        return out

    def predict(self, features: np.ndarray | None = None) -> float:
        x = self.readout_features() if features is None else np.asarray(features, dtype=np.float64)
        return _sigmoid(self.readout_b + float(self.readout_w @ x))

    def train_readout(self, target: float, features: np.ndarray | None = None) -> float:
        """One logistic SGD step toward `target` in [0, 1]; returns the pre-update prediction."""
        y = float(target)
        if not (np.isfinite(y) and 0.0 <= y <= 1.0):
            raise ValueError('target must be in [0, 1]')
        x = self.readout_features() if features is None else np.asarray(features, dtype=np.float64)
        p = self.predict(x)
        e = y - p
        self.readout_w = np.clip(self.readout_w + self.readout_lr * e * x, -4.0, 4.0)
        self.readout_b = float(np.clip(self.readout_b + self.readout_lr * e, -4.0, 4.0))
        return p

    # ------------------------------------------------------------------ profile
    def _profile(self, started: float, n: int) -> None:
        syn = self.synapses
        e = syn.profile['edge_traversals_last']
        f = self.flags
        ops = (e * (5 if f & {'adaptive_routing', 'uncertainty_gating'} else 3) + n * 42
               + (n * 12 if 'structural_encoder' in f else 0) + (n * 8 if 'context_attention' in f else 0)
               + (int(n * np.log2(max(2, n))) if 'sparse_code' in f else 0) + (HASH_CHANNELS * 8 if 'flyhash_projection' in f else 0))
        ms = (time.perf_counter() - started) * 1000.0
        self.measured['wallclock_ms_last'] = ms
        self.measured['wallclock_ms_total'] += ms
        self.measured['edge_traversals_last'] = e
        self.measured['edge_traversals_total'] += e
        self.measured['tiles_last'] = syn.profile['tiles_last']
        self.measured['steps'] += 1
        self.estimated['ops_last'] = int(ops)
        self.estimated['ops_total'] += int(ops)
        self.estimated['bytes'] = int(syn.size * 40 + n * 224)

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        syn_state = self.synapses.export_state()
        ledger = syn_state.pop('ledger')
        ledger['counters'] = dict(self.counters)
        return {
            'schema': BRAIN_SCHEMA,
            'spec': self.spec.to_dict(),
            'topology': {'name': self.spec.topology, 'seed': int(self.spec.seed), 'n': int(self.n),
                         'region': self.cells.region.tolist(), 'sign': self.cells.sign.astype(int).tolist(),
                         'regions': [list(r) for r in self.regions]},
            'cells': self.cells.export_state(),
            'synapses': syn_state,
            'readout': {'w': self.readout_w.tolist(), 'b': float(self.readout_b)},
            'step': int(self.step_count),
            'ledger': ledger,
            'rng': self.rng.bit_generator.state,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate the whole state first, then assign; malformed input raises ValueError without mutation."""
        try:
            staged = self._validate_state(state)
        except ValueError:
            raise
        except (TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
            raise ValueError(f'malformed brain state: {type(exc).__name__}: {exc}') from exc
        spec, cells, synapses, w, b, step, rng, counters = staged
        self.spec = spec
        self.flags = frozenset(spec.mechanisms)
        self.cells = cells
        self.synapses = synapses
        self.readout_w = w.copy()
        self.readout_b = b
        self.step_count = step
        self.rng = rng
        self.counters = {k: int(v) for k, v in counters.items()}
        self._proj_cache = {}
        self._hash_cache = {}
        self._fp_weight = None
        self._fp_version = -1
        self._region_counts = None
        self._region_counts_n = -1

    def _validate_state(self, state: Mapping[str, Any]) -> tuple:
        if not isinstance(state, Mapping) or state.get('schema') != BRAIN_SCHEMA:
            raise ValueError('not a NeuroWeaveBrainStateV1 state')
        required = ('spec', 'topology', 'cells', 'synapses', 'readout', 'step', 'ledger', 'rng')
        missing = [k for k in required if k not in state]
        if missing:
            raise ValueError(f'brain state missing {missing}')
        try:
            spec = NeuroWeaveSpecV1.from_dict(state['spec'])
        except (TypeError, KeyError) as exc:
            raise ValueError(f'invalid spec: {exc}') from exc
        cells = CellPopulation(1, seed=0, regions=[0], signs=[1])
        cells.import_state(state['cells'])
        if not isinstance(state['synapses'], Mapping) or not isinstance(state['cells'], Mapping):
            raise ValueError('cells and synapses must be mappings')
        syn_state = dict(state['synapses'])
        ledger_state = dict(state['ledger']) if isinstance(state['ledger'], Mapping) else None
        if ledger_state is None:
            raise ValueError('ledger must be a mapping')
        counters = ledger_state.pop('counters', None)
        if (not isinstance(counters, Mapping) or set(counters) != {'sleep_cycles', 'neurons_born', 'repurposed', 'structural_steps'}
                or any(type(v) is not int or v < 0 for v in counters.values())):
            raise ValueError('ledger counters invalid')
        syn_state['ledger'] = ledger_state
        synapses = SynapseTable.from_state(syn_state)
        topo = state['topology']
        if not isinstance(topo, Mapping):
            raise ValueError('topology must be a mapping')
        if topo.get('name') != spec.topology or topo.get('seed') != spec.seed:
            raise ValueError('topology name/seed do not match spec')
        if (topo.get('n') != cells.n or synapses.n != cells.n or list(topo.get('region', [])) != cells.region.tolist()
                or list(topo.get('sign', [])) != cells.sign.astype(int).tolist()
                or not np.array_equal(synapses.region, cells.region) or not np.array_equal(synapses.sign, cells.sign)):
            raise ValueError('topology / cells / synapses are inconsistent')
        if cells.n > spec.max_neurons:
            raise ValueError('state exceeds spec.max_neurons')
        if len(synapses.regions) != len(REGIONS) or cells.region.max() >= len(REGIONS):
            raise ValueError('region set mismatch')
        readout = state['readout']
        if not isinstance(readout, Mapping):
            raise ValueError('readout must be a mapping')
        try:
            w = np.asarray(readout.get('w'), dtype=np.float64)
            b = float(readout.get('b'))
        except (TypeError, ValueError) as exc:
            raise ValueError('readout malformed') from exc
        if w.shape != (self.feature_dim,) or not np.all(np.isfinite(w)) or not np.isfinite(b) or np.abs(w).max() > 4 or abs(b) > 4:
            raise ValueError('readout malformed')
        step = state['step']
        if type(step) is not int or step < 0:
            raise ValueError('step must be a non-negative int')
        rng = np.random.default_rng(0)
        try:
            rng.bit_generator.state = state['rng']
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError('rng state invalid') from exc
        return spec, cells, synapses, w, b, step, rng, counters

    def clone(self) -> 'NeuroWeaveBrain':
        other = NeuroWeaveBrain.__new__(NeuroWeaveBrain)
        other.__init__(self.spec)
        other.import_state(self.export_state())
        return other
