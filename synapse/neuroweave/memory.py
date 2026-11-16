"""Memory ecology: provenance-tagged, sparse-indexed, multi-timescale memory.

A bounded store of key/value items with:

* multi-timescale traces (fast/mid/slow EMAs of stored values) whose
  contrast drives default surprise estimates;
* a FlyHash-style sparse index (fixed sparse binary random expansion followed
  by top-k winner binarisation) used to find retrieval candidates;
* cosine key similarity plus optional context similarity, with an optional
  reward bias, weighted by item confidence;
* confidence decay, interference tracking (a new item whose key closely
  matches an existing item but whose value differs), a replay scheduler
  (priority = surprise x reward magnitude x recency), forgetting via
  ``downscale``, capacity eviction of the lowest-retention item and
  region-tagged cross-region sharing;
* provenance discipline: ``PREDICTED`` and ``SYNTHETIC`` items are never
  returned by verified retrieval, and a ``PREDICTED`` item is promoted to
  ``EXPERIENCE`` only by ``confirm`` against an observation within tolerance
  (otherwise a contradiction is recorded).

Synthetic computational abstraction; no biological-fidelity claim. Only
derived vectors are stored (never raw camera frames or audio buffers).
"""
from __future__ import annotations

from enum import Enum
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .spec import EVIDENCE_TIER

MEMORY_SCHEMA = 'NeuroWeaveMemoryEcologyV1'
SCHEMA_VERSION = 1
MEMORY_SEED_OFFSET = 15013
VALUE_BOUND = 1e3
MAX_DIM = 4096
MAX_CAPACITY = 1_000_000
MAX_CONTRADICTION_LOG = 64
MAX_REGION_LEN = 64
SURPRISE_TRACE_WEIGHTS = (0.5, 0.3, 0.2)   # fast, mid, slow


class Provenance(str, Enum):
    SENSORY = 'sensory'
    PREDICTED = 'predicted'
    EXTERNAL = 'external'
    EXPERIENCE = 'experience'
    SYNTHETIC = 'synthetic'


VERIFIED_PROVENANCE = frozenset({Provenance.SENSORY, Provenance.EXTERNAL, Provenance.EXPERIENCE})


# ----------------------------------------------------------------------------- helpers
def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _check_float(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return value


def _vector(value: Any, dim: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.shape != (dim,):
        raise ValueError(f'{name} must have shape ({dim},), got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    if np.any(np.abs(arr) > VALUE_BOUND):
        raise ValueError(f'{name} exceeds bound {VALUE_BOUND}')
    return arr.copy()


def _provenance(value: Any) -> Provenance:
    if isinstance(value, Provenance):
        return value
    try:
        return Provenance(value)
    except ValueError as exc:
        raise ValueError(f'unknown provenance {value!r}') from exc


def _region(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > MAX_REGION_LEN:
        raise ValueError('region must be a non-empty string (<= 64 chars) or None')
    return value


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


class MemoryEcology:
    """Bounded provenance-aware memory with a FlyHash candidate index."""

    def __init__(self, key_dim: int, value_dim: int, capacity: int = 256, seed: int = 7, *,
                 context_dim: int | None = None, hash_dim: int | None = None, hash_k: int | None = None,
                 hash_connections: int | None = None, trace_rates: tuple[float, float, float] = (0.5, 0.1, 0.01),
                 confidence_decay: float = 0.995, forget_threshold: float = 0.02,
                 interference_similarity: float = 0.9, interference_value_tolerance: float = 0.25,
                 interference_penalty: float = 0.1, recency_tau: float = 50.0,
                 context_weight: float = 0.3) -> None:
        self.key_dim = _check_int('key_dim', key_dim, 1, MAX_DIM)
        self.value_dim = _check_int('value_dim', value_dim, 1, MAX_DIM)
        self.capacity = _check_int('capacity', capacity, 1, MAX_CAPACITY)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.context_dim = _check_int('context_dim', self.key_dim if context_dim is None else context_dim, 1, MAX_DIM)
        self.hash_dim = _check_int('hash_dim', max(64, 8 * self.key_dim) if hash_dim is None else hash_dim, 2, 65536)
        self.hash_k = _check_int('hash_k', max(4, self.hash_dim // 20) if hash_k is None else hash_k, 1, self.hash_dim)
        self.hash_connections = _check_int(
            'hash_connections', min(6, self.key_dim) if hash_connections is None else hash_connections, 1, self.key_dim)
        if len(tuple(trace_rates)) != 3:
            raise ValueError('trace_rates must have three entries (fast, mid, slow)')
        rates = tuple(_check_float('trace_rate', r, 1e-6, 1.0) for r in trace_rates)
        if not rates[0] >= rates[1] >= rates[2]:
            raise ValueError('trace_rates must be ordered fast >= mid >= slow')
        self.trace_rates = rates
        self.confidence_decay = _check_float('confidence_decay', confidence_decay, 0.0, 1.0)
        self.forget_threshold = _check_float('forget_threshold', forget_threshold, 0.0, 1.0)
        # A non-positive threshold would let anti-correlated keys "interfere" with a
        # negative strength (raising confidence above 1), so the threshold is (0, 1].
        self.interference_similarity = _check_float('interference_similarity', interference_similarity, 1e-6, 1.0)
        self.interference_value_tolerance = _check_float('interference_value_tolerance', interference_value_tolerance, 0.0, 1e6)
        self.interference_penalty = _check_float('interference_penalty', interference_penalty, 0.0, 1.0)
        self.recency_tau = _check_float('recency_tau', recency_tau, 1e-6, 1e9)
        self.context_weight = _check_float('context_weight', context_weight, 0.0, 1.0)

        # Fixed sparse binary random expansion (FlyHash): each hash unit samples
        # ``hash_connections`` input dimensions.
        rng = np.random.default_rng(self.seed + MEMORY_SEED_OFFSET)
        proj = np.zeros((self.hash_dim, self.key_dim))
        for row in range(self.hash_dim):
            proj[row, rng.choice(self.key_dim, size=self.hash_connections, replace=False)] = 1.0
        self._projection = proj
        self._rng = np.random.default_rng(self.seed + MEMORY_SEED_OFFSET + 1)

        self.items: dict[int, dict[str, Any]] = {}
        self._index: dict[int, set[int]] = {}
        self.clock = 0
        self.next_id = 0
        self.traces = {name: np.zeros(self.value_dim) for name in ('fast', 'mid', 'slow')}
        self.trace_count = 0
        self.counters = {'stored': 0, 'evictions': 0, 'forgotten': 0, 'interference_events': 0,
                         'contradictions': 0, 'confirmations': 0, 'replays': 0, 'shares': 0,
                         'retrievals': 0, 'index_candidates': 0}
        self.interference_score = 0.0
        self.contradiction_log: list[dict[str, Any]] = []
        self.last_candidate_count = 0

    # ------------------------------------------------------------------ hashing
    def flyhash(self, key: Any) -> frozenset[int]:
        """Sparse binary code: top-``hash_k`` active units of the random expansion."""
        k = _vector(key, self.key_dim, 'key')
        norm = float(np.linalg.norm(k))
        if norm > 0.0:
            k = k / norm
        act = self._projection @ k
        winners = np.argsort(-act, kind='stable')[: self.hash_k]
        return frozenset(int(i) for i in winners)

    def _index_add(self, item_id: int, code: Iterable[int]) -> None:
        for unit in code:
            self._index.setdefault(unit, set()).add(item_id)

    def _index_remove(self, item_id: int, code: Iterable[int]) -> None:
        for unit in code:
            bucket = self._index.get(unit)
            if bucket is not None:
                bucket.discard(item_id)
                if not bucket:
                    del self._index[unit]

    def _candidates(self, code: frozenset[int]) -> set[int]:
        out: set[int] = set()
        for unit in code:
            out |= self._index.get(unit, set())
        return out

    # ------------------------------------------------------------------ priorities
    def _recency(self, item: Mapping[str, Any]) -> float:
        return math.exp(-(self.clock - item['last_access']) / self.recency_tau)

    def replay_priority(self, item_id: int) -> float:
        """surprise x reward magnitude x recency (small floors keep it strictly positive)."""
        item = self._get(item_id)
        return (0.05 + item['surprise']) * (0.05 + abs(item['reward'])) * self._recency(item)

    def retention_score(self, item_id: int) -> float:
        """Eviction score: replay priority weighted by confidence. Lowest is evicted."""
        return (0.1 + self._get(item_id)['confidence']) * self.replay_priority(item_id)

    def _priorities(self, ids: list[int]) -> np.ndarray:
        items = [self.items[i] for i in ids]
        surprise = np.fromiter((it['surprise'] for it in items), float, len(items))
        reward = np.fromiter((it['reward'] for it in items), float, len(items))
        last = np.fromiter((it['last_access'] for it in items), float, len(items))
        return (0.05 + surprise) * (0.05 + np.abs(reward)) * np.exp(-(self.clock - last) / self.recency_tau)

    def _retention_scores(self, ids: list[int]) -> np.ndarray:
        conf = np.fromiter((self.items[i]['confidence'] for i in ids), float, len(ids))
        return (0.1 + conf) * self._priorities(ids)

    def _get(self, item_id: int) -> dict[str, Any]:
        if isinstance(item_id, bool) or not isinstance(item_id, (int, np.integer)) or int(item_id) not in self.items:
            raise ValueError(f'unknown memory id {item_id!r}')
        return self.items[int(item_id)]

    def _remove(self, item_id: int) -> None:
        item = self.items.pop(item_id)
        self._index_remove(item_id, item['code'])

    def trace_contrast(self) -> np.ndarray:
        """Fast minus slow trace: what is recent relative to long-run content."""
        return self.traces['fast'] - self.traces['slow']

    # ------------------------------------------------------------------ store
    def store(self, key: Any, value: Any, *, provenance: Provenance | str, reward: float = 0.0,
              context: Any = None, confidence: float = 1.0, region: str | None = None,
              surprise: float | None = None) -> int:
        """Store an item and return its id. Values are derived vectors only."""
        k = _vector(key, self.key_dim, 'key')
        v = _vector(value, self.value_dim, 'value')
        prov = _provenance(provenance)
        rew = float(np.clip(_check_float('reward', reward, -1e6, 1e6), -1.0, 1.0))
        conf = _check_float('confidence', confidence, 0.0, 1.0)
        ctx = None if context is None else _vector(context, self.context_dim, 'context')
        reg = _region(region)
        surp = self.default_surprise(v) if surprise is None else _check_float('surprise', surprise, 0.0, 1.0)
        # Only verified provenance counts as an observation: unconfirmed
        # PREDICTED/REPLAYED/... content must never erode verified items'
        # confidence (interference) or move the novelty traces.
        return self._insert(k, v, prov, rew, ctx, conf, reg, surp, observe=prov in VERIFIED_PROVENANCE)

    def default_surprise(self, value: Any) -> float:
        """Multi-timescale novelty of ``value`` in [0, 1].

        Weighted mean of ``d / (1 + d)`` where ``d`` is the distance from the
        fast, mid and slow traces (weights ``SURPRISE_TRACE_WEIGHTS``): a value
        that departs from recent *and* long-run content is most surprising; a
        value that returns to long-run content after a recent excursion is
        less surprising than a genuinely new one. 1.0 before any observation."""
        v = _vector(value, self.value_dim, 'value')
        if self.trace_count == 0:
            return 1.0
        total = 0.0
        for name, weight in zip(('fast', 'mid', 'slow'), SURPRISE_TRACE_WEIGHTS):
            d = float(np.linalg.norm(v - self.traces[name]))
            total += weight * d / (1.0 + d)
        return float(min(1.0, max(0.0, total)))

    def _insert(self, k: np.ndarray, v: np.ndarray, prov: Provenance, rew: float, ctx: np.ndarray | None,
                conf: float, reg: str | None, surp: float, *, observe: bool) -> int:
        """Insert a validated item. ``observe`` (a new observation, not a copy)
        updates the multi-timescale traces and tracks interference."""
        self.clock += 1
        code = self.flyhash(k)
        # ---- interference: similar key, different value (vectorised over index candidates)
        cand = sorted(self._candidates(code)) if observe else []
        if cand:
            keys = np.stack([self.items[c]['key'] for c in cand])
            vals = np.stack([self.items[c]['value'] for c in cand])
            kn = float(np.linalg.norm(k))
            norms = np.linalg.norm(keys, axis=1)
            denom = norms * kn
            sims = np.where(denom > 0.0, (keys @ k) / np.where(denom > 0.0, denom, 1.0), 0.0)
            sims = np.clip(sims, -1.0, 1.0)
            scale = np.maximum(1.0 + float(np.linalg.norm(v)), 1.0 + np.linalg.norm(vals, axis=1))
            vdiff = np.linalg.norm(vals - v, axis=1) / scale
            hits = np.nonzero((sims >= self.interference_similarity) & (vdiff > self.interference_value_tolerance))[0]
            for j in hits:
                other = self.items[cand[int(j)]]
                strength = float(max(0.0, sims[j]) * min(1.0, vdiff[j]))
                other['interference'] += 1
                other['confidence'] = float(min(1.0, max(0.0, other['confidence'] * (1.0 - self.interference_penalty * strength))))
                self.counters['interference_events'] += 1
                n = self.counters['interference_events']
                self.interference_score = float(min(1.0, max(0.0, self.interference_score + (strength - self.interference_score) / n)))
        # ---- capacity eviction (vectorised retention scores)
        while len(self.items) >= self.capacity:
            ids = list(self.items)
            scores = self._retention_scores(ids)
            order = np.lexsort((np.array(ids), scores))
            victim = ids[int(order[0])]
            self._remove(victim)
            self.counters['evictions'] += 1
        # ---- traces
        if observe:
            for name, rate in zip(('fast', 'mid', 'slow'), self.trace_rates):
                if self.trace_count == 0:
                    self.traces[name] = v.copy()
                else:
                    self.traces[name] += rate * (v - self.traces[name])
                np.clip(self.traces[name], -VALUE_BOUND, VALUE_BOUND, out=self.traces[name])
            self.trace_count += 1

        item_id = self.next_id
        self.next_id += 1
        self.items[item_id] = {
            'key': k, 'value': v, 'context': ctx, 'provenance': prov, 'reward': rew,
            'confidence': conf, 'surprise': surp, 'region': reg, 'code': code,
            'created': self.clock, 'last_access': self.clock, 'access_count': 0,
            'interference': 0, 'contradictions': 0, 'replays': 0, 'shared_from': None,
        }
        self._index_add(item_id, code)
        self.counters['stored'] += 1
        return item_id

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, key: Any, *, context: Any = None, k: int = 4, verified_only: bool = False,
                 region: str | None = None, reward_bias: float = 0.0, min_confidence: float = 0.0,
                 exhaustive: bool = False, touch: bool = True) -> list[dict[str, Any]]:
        """Top-k items by score.

        Candidates come from the FlyHash index (items sharing at least one
        active hash unit with the query) unless ``exhaustive``. Score =
        ``(1 - w_ctx) * cos(key) + w_ctx * cos(context)`` (context term only
        when both sides have context), multiplied by ``0.5 + 0.5 * confidence``,
        plus ``reward_bias * reward``."""
        q = _vector(key, self.key_dim, 'key')
        ctx = None if context is None else _vector(context, self.context_dim, 'context')
        k = _check_int('k', k, 0, MAX_CAPACITY)
        if type(verified_only) is not bool or type(exhaustive) is not bool or type(touch) is not bool:
            raise ValueError('verified_only, exhaustive and touch must be booleans')
        reg = _region(region)
        rb = _check_float('reward_bias', reward_bias, -10.0, 10.0)
        minc = _check_float('min_confidence', min_confidence, 0.0, 1.0)
        cand = set(self.items) if exhaustive else self._candidates(self.flyhash(q))
        self.last_candidate_count = len(cand)
        self.counters['retrievals'] += 1
        self.counters['index_candidates'] += len(cand)
        keep = []
        for cid in sorted(cand):
            item = self.items[cid]
            if verified_only and item['provenance'] not in VERIFIED_PROVENANCE:
                continue
            if reg is not None and item['region'] != reg:
                continue
            if item['confidence'] < minc:
                continue
            keep.append(cid)
        if not keep or k == 0:
            return []
        keys = np.stack([self.items[c]['key'] for c in keep])
        qn = float(np.linalg.norm(q))
        denom = np.linalg.norm(keys, axis=1) * qn
        ksims = np.clip(np.where(denom > 0.0, (keys @ q) / np.where(denom > 0.0, denom, 1.0), 0.0), -1.0, 1.0)
        conf = np.fromiter((self.items[c]['confidence'] for c in keep), float, len(keep))
        reward = np.fromiter((self.items[c]['reward'] for c in keep), float, len(keep))
        csims: list[float | None] = [None] * len(keep)
        sims = ksims.copy()
        if ctx is not None:
            for j, c in enumerate(keep):
                ictx = self.items[c]['context']
                if ictx is not None:
                    csims[j] = _cosine(ctx, ictx)
                    sims[j] = (1.0 - self.context_weight) * ksims[j] + self.context_weight * csims[j]
        scores = sims * (0.5 + 0.5 * conf) + rb * reward
        order = np.lexsort((np.array(keep), -scores))[:k]
        scored = [(float(scores[j]), keep[int(j)], float(ksims[j]), csims[int(j)]) for j in order]
        out = []
        for score, cid, ksim, csim in scored:
            item = self.items[cid]
            if touch:
                item['last_access'] = self.clock
                item['access_count'] += 1
            out.append({'id': cid, 'value': item['value'].copy(), 'key': item['key'].copy(),
                        'score': float(score), 'similarity': ksim, 'context_similarity': csim,
                        'provenance': item['provenance'].value, 'reward': item['reward'],
                        'confidence': item['confidence'], 'region': item['region'],
                        'verified': item['provenance'] in VERIFIED_PROVENANCE})
        return out

    def get(self, item_id: int) -> dict[str, Any]:
        item = self._get(item_id)
        return {'id': int(item_id), 'key': item['key'].copy(), 'value': item['value'].copy(),
                'context': None if item['context'] is None else item['context'].copy(),
                'provenance': item['provenance'].value, 'reward': item['reward'],
                'confidence': item['confidence'], 'surprise': item['surprise'], 'region': item['region'],
                'interference': item['interference'], 'contradictions': item['contradictions'],
                'replays': item['replays'], 'access_count': item['access_count'],
                'shared_from': item['shared_from']}

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self.items

    # ------------------------------------------------------------------ provenance discipline
    def confirm(self, item_id: int, observed_value: Any, tol: float) -> dict[str, Any]:
        """Check a PREDICTED item against an observation.

        Within ``tol`` (RMS error) it becomes EXPERIENCE with the observed value;
        otherwise it stays PREDICTED, loses confidence and a contradiction is logged."""
        item = self._get(item_id)
        obs = _vector(observed_value, self.value_dim, 'observed_value')
        tol = _check_float('tol', tol, 0.0, 1e6)
        if item['provenance'] is not Provenance.PREDICTED:
            raise ValueError('only PREDICTED items can be confirmed')
        err = float(np.sqrt(np.mean((item['value'] - obs) ** 2)))
        self.clock += 1
        if err <= tol:
            item['provenance'] = Provenance.EXPERIENCE
            item['value'] = obs
            item['confidence'] = float(min(1.0, max(item['confidence'], 0.5) + 0.25))
            item['last_access'] = self.clock
            self.counters['confirmations'] += 1
            confirmed = True
        else:
            item['contradictions'] += 1
            item['confidence'] = float(item['confidence'] * 0.5)
            item['surprise'] = float(max(item['surprise'], min(1.0, err / (1.0 + err))))
            self.counters['contradictions'] += 1
            self.contradiction_log.append({'id': int(item_id), 'error': err, 'tol': tol, 'clock': self.clock})
            if len(self.contradiction_log) > MAX_CONTRADICTION_LOG:
                self.contradiction_log.pop(0)
            confirmed = False
        return {'id': int(item_id), 'confirmed': confirmed, 'error': err, 'tol': tol,
                'provenance': item['provenance'].value, 'confidence': item['confidence']}

    # ------------------------------------------------------------------ dynamics
    def tick(self, steps: int = 1) -> dict[str, Any]:
        """Advance the clock; decay confidence; forget items below threshold."""
        steps = _check_int('steps', steps, 0, 1_000_000)
        self.clock += steps
        factor = self.confidence_decay ** steps
        for item in self.items.values():
            item['confidence'] = float(item['confidence'] * factor)
        return {'forgotten': self._forget_below_threshold(), 'decay_factor': factor}

    def _forget_below_threshold(self) -> list[int]:
        gone = sorted(i for i, it in self.items.items() if it['confidence'] < self.forget_threshold)
        for i in gone:
            self._remove(i)
        self.counters['forgotten'] += len(gone)
        return gone

    def downscale(self, factor: float) -> dict[str, Any]:
        """Forgetting experiment: scale every confidence and reward association by
        ``factor`` in (0, 1], then forget items that fall below threshold."""
        factor = _check_float('factor', factor, 1e-9, 1.0)
        before = float(np.mean([it['confidence'] for it in self.items.values()])) if self.items else 0.0
        for item in self.items.values():
            item['confidence'] = float(item['confidence'] * factor)
            item['reward'] = float(item['reward'] * factor)
        forgotten = self._forget_below_threshold()
        after = float(np.mean([it['confidence'] for it in self.items.values()])) if self.items else 0.0
        return {'factor': factor, 'forgotten': forgotten, 'mean_confidence_before': before,
                'mean_confidence_after': after, 'remaining': len(self.items)}

    def forget(self, item_id: int) -> None:
        self._get(item_id)
        self._remove(int(item_id))
        self.counters['forgotten'] += 1

    def replay_schedule(self, n: int = 4, *, verified_only: bool = False, stochastic: bool = False,
                        surprise_decay: float = 0.5) -> list[dict[str, Any]]:
        """Pick items to replay by priority (surprise x |reward| x recency).

        Deterministic top-n by default; ``stochastic`` samples proportionally to
        priority. Replayed items have their surprise multiplied by
        ``surprise_decay`` so the schedule rotates over time."""
        n = _check_int('n', n, 0, MAX_CAPACITY)
        decay = _check_float('surprise_decay', surprise_decay, 0.0, 1.0)
        ids = [i for i, it in self.items.items()
               if not verified_only or it['provenance'] in VERIFIED_PROVENANCE]
        n = min(n, len(ids))
        if n == 0:
            return []
        prio = self._priorities(ids)
        if stochastic:
            pick = self._rng.choice(len(ids), size=n, replace=False, p=prio / prio.sum())
        else:
            pick = np.lexsort((np.array(ids), -prio))[:n]
        out = []
        for j in pick:
            cid = ids[int(j)]
            item = self.items[cid]
            out.append({'id': cid, 'priority': float(prio[int(j)]), 'key': item['key'].copy(),
                        'value': item['value'].copy(), 'provenance': item['provenance'].value,
                        'reward': item['reward'], 'surprise': item['surprise']})
            item['replays'] += 1
            item['surprise'] = float(item['surprise'] * decay)
        self.counters['replays'] += len(out)
        return out

    def share(self, item_id: int, region: str) -> int:
        """Copy an item into another region (provenance preserved). Returns the new id."""
        item = self._get(item_id)
        reg = _region(region)
        if reg is None:
            raise ValueError('target region is required')
        # A share is a copy, not a new observation: traces and interference are untouched.
        new_id = self._insert(item['key'].copy(), item['value'].copy(), item['provenance'], item['reward'],
                              None if item['context'] is None else item['context'].copy(),
                              item['confidence'], reg, item['surprise'], observe=False)
        self.items[new_id]['shared_from'] = int(item_id)
        self.counters['shares'] += 1
        return new_id

    def region_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.items.values():
            name = item['region'] if item['region'] is not None else '__global__'
            out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items()))

    def stats(self) -> dict[str, Any]:
        prov = {p.value: 0 for p in Provenance}
        for item in self.items.values():
            prov[item['provenance'].value] += 1
        conf = [it['confidence'] for it in self.items.values()]
        return {'evidence_tier': EVIDENCE_TIER, 'size': len(self.items), 'capacity': self.capacity,
                'clock': self.clock, 'provenance': prov, 'counters': dict(self.counters),
                'interference_score': self.interference_score,
                'mean_confidence': float(np.mean(conf)) if conf else 0.0,
                'regions': self.region_counts(), 'raw_media_stored': False,
                'trace_contrast_norm': float(np.linalg.norm(self.trace_contrast()))}

    # ------------------------------------------------------------------ persistence
    def _config(self) -> dict[str, Any]:
        return {'key_dim': self.key_dim, 'value_dim': self.value_dim, 'capacity': self.capacity,
                'seed': self.seed, 'context_dim': self.context_dim, 'hash_dim': self.hash_dim,
                'hash_k': self.hash_k, 'hash_connections': self.hash_connections,
                'trace_rates': list(self.trace_rates), 'confidence_decay': self.confidence_decay,
                'forget_threshold': self.forget_threshold,
                'interference_similarity': self.interference_similarity,
                'interference_value_tolerance': self.interference_value_tolerance,
                'interference_penalty': self.interference_penalty, 'recency_tau': self.recency_tau,
                'context_weight': self.context_weight}

    def export_state(self) -> dict[str, Any]:
        state = self._rng.bit_generator.state
        return {
            'schema': MEMORY_SCHEMA, 'schema_version': SCHEMA_VERSION, 'evidence_tier': EVIDENCE_TIER,
            'raw_media_stored': False,
            'config': self._config(),
            'clock': self.clock, 'next_id': self.next_id, 'trace_count': self.trace_count,
            'traces': {n: t.tolist() for n, t in self.traces.items()},
            'counters': dict(self.counters),
            'interference_score': self.interference_score,
            'contradiction_log': [dict(e) for e in self.contradiction_log],
            'items': [{'id': i, 'key': it['key'].tolist(), 'value': it['value'].tolist(),
                       'context': None if it['context'] is None else it['context'].tolist(),
                       'provenance': it['provenance'].value, 'reward': it['reward'],
                       'confidence': it['confidence'], 'surprise': it['surprise'], 'region': it['region'],
                       'created': it['created'], 'last_access': it['last_access'],
                       'access_count': it['access_count'], 'interference': it['interference'],
                       'contradictions': it['contradictions'], 'replays': it['replays'],
                       'shared_from': it['shared_from']}
                      for i, it in sorted(self.items.items())],
            'rng': {'bit_generator': str(state['bit_generator']),
                    'state': {'state': int(state['state']['state']), 'inc': int(state['state']['inc'])},
                    'has_uint32': int(state['has_uint32']), 'uinteger': int(state['uinteger'])},
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate fully, then replace contents. The config must match this instance
        (the FlyHash projection is derived from it)."""
        if not isinstance(state, Mapping):
            raise ValueError('state must be a mapping')
        if state.get('schema') != MEMORY_SCHEMA or state.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported memory ecology schema')
        if state.get('raw_media_stored') is not False:
            raise ValueError('raw_media_stored must be False')
        if state.get('config') != self._config():
            raise ValueError('memory config mismatch')
        clock = _check_int('clock', state.get('clock'), 0, 2**62)
        next_id = _check_int('next_id', state.get('next_id'), 0, 2**62)
        trace_count = _check_int('trace_count', state.get('trace_count'), 0, 2**62)
        traces_raw = state.get('traces')
        if not isinstance(traces_raw, Mapping) or set(traces_raw) != {'fast', 'mid', 'slow'}:
            raise ValueError('traces must have fast/mid/slow')
        traces = {n: _vector(traces_raw[n], self.value_dim, f'traces.{n}') for n in ('fast', 'mid', 'slow')}
        counters_raw = state.get('counters')
        if not isinstance(counters_raw, Mapping) or set(counters_raw) != set(self.counters):
            raise ValueError('counters keys mismatch')
        counters = {c: _check_int(c, counters_raw[c], 0, 2**62) for c in self.counters}
        iscore = _check_float('interference_score', state.get('interference_score'), 0.0, 1.0)
        log_raw = state.get('contradiction_log')
        if not isinstance(log_raw, list) or len(log_raw) > MAX_CONTRADICTION_LOG:
            raise ValueError('contradiction_log must be a bounded list')
        log = []
        for e in log_raw:
            if not isinstance(e, Mapping):
                raise ValueError('contradiction_log entries must be mappings')
            log.append({'id': _check_int('id', e.get('id'), 0, 2**62),
                        'error': _check_float('error', e.get('error'), 0.0, 1e12),
                        'tol': _check_float('tol', e.get('tol'), 0.0, 1e6),
                        'clock': _check_int('clock', e.get('clock'), 0, 2**62)})
        items_raw = state.get('items')
        if not isinstance(items_raw, list) or len(items_raw) > self.capacity:
            raise ValueError('items must be a list no longer than capacity')
        items: dict[int, dict[str, Any]] = {}
        for raw in items_raw:
            if not isinstance(raw, Mapping):
                raise ValueError('items must be mappings')
            if next_id == 0:
                raise ValueError('items present but next_id is 0')
            iid = _check_int('id', raw.get('id'), 0, next_id - 1)
            if iid in items:
                raise ValueError('duplicate memory id')
            key = _vector(raw.get('key'), self.key_dim, 'item.key')
            ctx = raw.get('context')
            shared = raw.get('shared_from')
            items[iid] = {
                'key': key, 'value': _vector(raw.get('value'), self.value_dim, 'item.value'),
                'context': None if ctx is None else _vector(ctx, self.context_dim, 'item.context'),
                'provenance': _provenance(raw.get('provenance')),
                'reward': _check_float('reward', raw.get('reward'), -1.0, 1.0),
                'confidence': _check_float('confidence', raw.get('confidence'), 0.0, 1.0),
                'surprise': _check_float('surprise', raw.get('surprise'), 0.0, 1.0),
                'region': _region(raw.get('region')),
                'code': self.flyhash(key),
                'created': _check_int('created', raw.get('created'), 0, clock),
                'last_access': _check_int('last_access', raw.get('last_access'), 0, clock),
                'access_count': _check_int('access_count', raw.get('access_count'), 0, 2**62),
                'interference': _check_int('interference', raw.get('interference'), 0, 2**62),
                'contradictions': _check_int('contradictions', raw.get('contradictions'), 0, 2**62),
                'replays': _check_int('replays', raw.get('replays'), 0, 2**62),
                'shared_from': None if shared is None else _check_int('shared_from', shared, 0, 2**62),
            }
        rng_raw = state.get('rng')
        if not isinstance(rng_raw, Mapping) or rng_raw.get('bit_generator') != 'PCG64':
            raise ValueError('rng must be a PCG64 state')
        rng = np.random.Generator(np.random.PCG64())
        try:
            rng.bit_generator.state = {'bit_generator': 'PCG64',
                                       'state': {'state': int(rng_raw['state']['state']),
                                                 'inc': int(rng_raw['state']['inc'])},
                                       'has_uint32': int(rng_raw['has_uint32']),
                                       'uinteger': int(rng_raw['uinteger'])}
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid rng state') from exc
        # ---- validated: assign
        self.items = items
        self._index = {}
        for iid, it in items.items():
            self._index_add(iid, it['code'])
        self.clock, self.next_id, self.trace_count = clock, next_id, trace_count
        self.traces = traces
        self.counters = counters
        self.interference_score = iscore
        self.contradiction_log = log
        self._rng = rng
        self.last_candidate_count = 0


__all__ = ['Provenance', 'MemoryEcology', 'VERIFIED_PROVENANCE', 'MEMORY_SCHEMA']
