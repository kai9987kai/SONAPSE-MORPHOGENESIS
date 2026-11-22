"""v93 Supermix-Archimedes Connectome Neurogenesis & Synaptic Plasticity Engine.

Adapted from Kai9987kai's Supermix Archimedes (v93 / mimomix) architecture.
Implements continuous activity-dependent structural plasticity:
1. **Apoptosis**: Pruning stagnant, weak or inactive synapses below softplus threshold.
2. **Synaptogenesis**: Sprouting new synaptic connections between pairs of neurons
   exhibiting high pre-post firing rate covariance.
3. **Sign-aware Growth**: Reuses consistent frozen-source polarity when available;
   otherwise retains the replaced slot's sign. No measured cell-type guarantee.
4. **Neuromodulatory & Astrocytic Gating**: Synaptic growth is facilitated by elevated Dopamine (positive RPE)
   and Astrocytic [Ca2+] spillover, and suppressed under high stress/threat.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class NeurogenesisStats:
    step: int = 0
    synapses_born: int = 0
    synapses_pruned: int = 0
    active_synapses: int = 0
    mean_weight: float = 0.0
    synaptogenesis_events: int = 0
    apoptosis_events: int = 0
    recent_events: List[Dict[str, Any]] = field(default_factory=list)


class ConnectomeNeurogenesisEngine:
    """Orchestrates activity-dependent structural remodeling of the grafted connectome substrate."""

    _COVARIANCE_TEMP_BYTES = 8 * 1024 * 1024
    _COVARIANCE_MAX_BLOCK_ROWS = 256

    def __init__(
        self,
        num_neurons: int,
        prune_threshold: float = 0.02,
        growth_rate: float = 0.05,
        covariance_decay: float = 0.95,
        max_new_synapses_per_event: int = 8,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate
        self.covariance_decay = covariance_decay
        self.max_new_synapses = max_new_synapses_per_event
        self.rng = np.random.default_rng(seed)

        # Running rate covariance accumulator [N, N]
        self.rate_mean = np.zeros(num_neurons, dtype=np.float32)
        self._rate_cov = np.zeros((num_neurons, num_neurons), dtype=np.float32)
        self._cov_upper_bound = 0.0
        self._row_upper_bound = np.zeros(num_neurons, dtype=np.float32)
        self._bounds_valid = True
        self.stats = NeurogenesisStats()
        self.activity_steps = 0

    @property
    def rate_cov(self) -> np.ndarray:
        # External callers may mutate the returned array in place (e.g. direct tests).
        self._bounds_valid = False
        return self._rate_cov

    @rate_cov.setter
    def rate_cov(self, value: np.ndarray) -> None:
        self._rate_cov = value
        self._bounds_valid = False

    def record_activity(self, rates: np.ndarray | torch.Tensor) -> None:
        """Accumulates neural firing rate covariance across microsteps."""
        if isinstance(rates, torch.Tensor):
            r = rates.detach().cpu().numpy().reshape(-1)
        else:
            r = np.asarray(rates, dtype=np.float32).reshape(-1)

        n = min(len(r), self.num_neurons)
        r = r[:n]

        # Online mean and covariance update (Welford-style decay)
        delta = r - self.rate_mean[:n]
        self.rate_mean[:n] += 0.05 * delta
        decay = float(self.covariance_decay)
        coefficient = 1.0 - decay
        # Zero-temporary fused SIMD rank-one update via PyTorch addr_ when backed by
        # float32 contiguous storage; falls back to bounded-block NumPy otherwise.
        if (
            n > 0
            and self._rate_cov.dtype == np.float32
            and self._rate_cov.flags['C_CONTIGUOUS']
        ):
            delta_f32 = np.ascontiguousarray(delta, dtype=np.float32)
            delta_t = torch.from_numpy(delta_f32)
            torch.from_numpy(self._rate_cov[:n, :n]).addr_(
                delta_t, delta_t, beta=decay, alpha=coefficient
            )
        else:
            delta_f32 = np.asarray(delta, dtype=np.float32)
            scaled_delta = delta_f32 * np.float32(coefficient)
            block_rows = max(1, min(
                self._COVARIANCE_MAX_BLOCK_ROWS,
                self._COVARIANCE_TEMP_BYTES // max(1, n * np.dtype(np.float32).itemsize),
                n,
            ))
            for start in range(0, n, block_rows):
                stop = min(n, start + block_rows)
                covariance_block = self._rate_cov[start:stop, :n]
                np.multiply(covariance_block, decay, out=covariance_block)
                covariance_delta = scaled_delta[start:stop, None] * delta_f32[None, :]
                np.add(covariance_block, covariance_delta, out=covariance_block)

        if self._bounds_valid and n > 0:
            d_min = float(delta_f32.min())
            d_max = float(delta_f32.max())
            max_abs = max(-d_min, d_max)
            max_outer = max(
                d_max * d_max if d_max > 0.0 else 0.0,
                d_min * d_min if d_min < 0.0 else 0.0,
            )
            self._cov_upper_bound = decay * self._cov_upper_bound + coefficient * max_outer
            if len(self._row_upper_bound) < n:
                self._bounds_valid = False
            else:
                rb = self._row_upper_bound[:n]
                rb *= np.float32(decay)
                rb += np.abs(delta_f32) * np.float32(coefficient * max_abs)
        self.activity_steps += 1

    @staticmethod
    def _ranked_candidates_rows(
        cov_mat: np.ndarray,
        n: int,
        active_rows: np.ndarray,
        threshold: float = 0.01,
        head: int = 256,
    ):
        """Yield off-diagonal flat indices >= threshold scanning only ``active_rows``."""
        if active_rows.size == 0:
            return
        sub = cov_mat[active_rows]
        local_rows, cols = np.nonzero(sub >= threshold)
        if local_rows.size == 0:
            return
        rows = active_rows[local_rows]
        off_diag = rows != cols
        rows = rows[off_diag]
        cols = cols[off_diag]
        if rows.size == 0:
            return
        cand = rows.astype(np.int64) * n + cols.astype(np.int64)
        vals = sub[local_rows[off_diag], cols]
        if cand.size <= head:
            for k in np.argsort(vals)[::-1]:
                yield int(cand[k])
            return
        part = np.argpartition(vals, cand.size - head)
        top, rest = part[cand.size - head:], part[:cand.size - head]
        for k in top[np.argsort(vals[top])[::-1]]:
            yield int(cand[k])
        for k in rest[np.argsort(vals[rest])[::-1]]:
            yield int(cand[k])

    @staticmethod
    def _ranked_candidates(cov_mat: np.ndarray, n: int, threshold: float = 0.01, head: int = 256):
        """Lazily yields flat indices of off-diagonal entries >= threshold, descending.

        A small head is ranked first with argpartition; the remainder is only
        sorted if the caller keeps consuming (rare: sprouting stops after
        ``max_new_synapses``).
        """
        flat = cov_mat.reshape(-1) if cov_mat.flags['C_CONTIGUOUS'] else np.ascontiguousarray(cov_mat).reshape(-1)
        cand = np.flatnonzero(flat >= threshold)
        if cand.size:
            cand = cand[(cand % (n + 1)) != 0]  # drop self-loops (diagonal)
        if cand.size == 0:
            return
        vals = flat[cand]
        if cand.size <= head:
            for k in np.argsort(vals)[::-1]:
                yield int(cand[k])
            return
        part = np.argpartition(vals, cand.size - head)
        top, rest = part[cand.size - head:], part[:cand.size - head]
        for k in top[np.argsort(vals[top])[::-1]]:
            yield int(cand[k])
        for k in rest[np.argsort(vals[rest])[::-1]]:
            yield int(cand[k])

    def evaluate_structural_remodeling(
        self,
        agent: Any,
        dopamine: float = 1.0,
        astrocyte_calcium: float = 1.0,
        threat: float = 0.0,
    ) -> Dict[str, Any]:
        """Executes one round of Supermix Archimedes apoptosis & synaptogenesis."""
        self.stats.step += 1
        event_record: Dict[str, Any] = {
            'step': self.stats.step,
            'pruned': 0,
            'sprouted': 0,
            'dopamine': float(dopamine),
            'calcium': float(astrocyte_calcium),
        }

        brain = getattr(agent, 'brain', None)
        graph = getattr(agent, 'graph', None)
        if brain is None or graph is None:
            return event_record

        # Current graft weights and edges
        g_w = brain.graft_w.detach().cpu().numpy().copy()
        pre = np.array(graph.graft_pre, dtype=np.int64)
        post = np.array(graph.graft_post, dtype=np.int64)
        n_edges = len(g_w)
        changed_slots = []

        # -------------------------------------------------------------
        # 1. APOPTOSIS (Synaptic Pruning)
        # -------------------------------------------------------------
        # Prune synapses with near-zero transmission efficacy
        alive_mask = np.abs(g_w) >= self.prune_threshold
        pruned_count = int(np.sum(~alive_mask))

        if pruned_count > 0:
            # Weaken pruned synapses toward zero
            g_w[~alive_mask] *= 0.1
            self.stats.synapses_pruned += pruned_count
            self.stats.apoptosis_events += 1
            event_record['pruned'] = pruned_count

        # -------------------------------------------------------------
        # 2. SYNAPTOGENESIS (Sprouting between covarying neurons)
        # -------------------------------------------------------------
        # Neurogenesis gating: promoted by Dopamine RPE and Astrocyte [Ca2+]
        growth_drive = float(np.clip(dopamine * 0.7 + astrocyte_calcium * 0.3 - threat * 0.8, 0.0, 2.0))

        if growth_drive > 0.4 and self.activity_steps > 10:
            n = min(self.num_neurons, self._rate_cov.shape[0])
            # Exact upper-bound skip: if the proven maximum entry is < 0.01,
            # no candidate pair can qualify and we avoid scanning N^2 entries.
            if not (self._bounds_valid and self._cov_upper_bound < 0.01):
                cov_mat = self._rate_cov[:n, :n]
                use_row_filter = (
                    self._bounds_valid
                    and len(self._row_upper_bound) >= n
                    and '_ranked_candidates' not in self.__dict__
                )
                if use_row_filter:
                    active_rows = np.flatnonzero(self._row_upper_bound[:n] >= 0.01)
                    flat_indices = self._ranked_candidates_rows(cov_mat, n, active_rows, threshold=0.01)
                else:
                    flat_indices = self._ranked_candidates(cov_mat, n, threshold=0.01)

                existing_pairs = None
                all_compatible = None
                origin_groups = None
                has_origin = hasattr(graph, 'origin')
                origins = graph.origin if has_origin else None
                base_w_arr = None
                base_pre_arr = None
                sprouted = 0

                for idx in flat_indices:
                    if sprouted >= self.max_new_synapses:
                        break
                    p_i, p_j = divmod(int(idx), n)
                    cov_val = cov_mat[p_i, p_j]
                    if cov_val < 0.01:
                        break

                    if existing_pairs is None:
                        existing_pairs = set(zip(pre.tolist(), post.tolist()))
                        if hasattr(agent, 'plasticity'):
                            active_np = agent.plasticity.active_mask.detach().cpu().numpy().astype(bool, copy=False)
                            all_compatible = np.flatnonzero(active_np)
                        else:
                            all_compatible = np.arange(n_edges)
                        if has_origin:
                            origin_groups = {}
                            if all_compatible.size:
                                pre_o = origins[pre[all_compatible]]
                                post_o = origins[post[all_compatible]]
                                for slot_idx, o_pre, o_post in zip(
                                    all_compatible.tolist(), pre_o.tolist(), post_o.tolist()
                                ):
                                    origin_groups.setdefault((o_pre, o_post), []).append(slot_idx)
                                origin_groups = {
                                    k: np.asarray(v, dtype=np.int64) for k, v in origin_groups.items()
                                }

                    # If connection already exists in graft, strengthen it
                    if (p_i, p_j) in existing_pairs:
                        continue

                    # Sprout new synaptic coupling within existing graft weight slots
                    if has_origin:
                        compatible = origin_groups.get((int(origins[p_i]), int(origins[p_j])))
                        if compatible is None or not len(compatible):
                            continue
                    else:
                        compatible = all_compatible
                        if not len(compatible):
                            continue
                    weakest_idx = int(compatible[np.argmin(np.abs(g_w[compatible]))])
                    if np.abs(g_w[weakest_idx]) < self.prune_threshold * 1.5:
                        old_sign = float(np.sign(g_w[weakest_idx])) or 1.0
                        pre[weakest_idx] = p_i
                        post[weakest_idx] = p_j
                        if hasattr(graph, 'base_w'):
                            if base_w_arr is None:
                                base_w_arr = np.asarray(graph.base_w)
                                base_pre_arr = np.asarray(graph.base_pre)
                            source_signs = np.sign(base_w_arr[base_pre_arr == p_i])
                        else:
                            source_signs = np.array([])
                        source_signs = source_signs[source_signs != 0]
                        sign = float(source_signs[0]) if len(source_signs) and np.all(source_signs == source_signs[0]) else old_sign
                        g_w[weakest_idx] = sign * (0.05 + 0.03 * growth_drive)
                        changed_slots.append(weakest_idx)
                        existing_pairs.add((p_i, p_j))
                        sprouted += 1

                if sprouted > 0:
                    self.stats.synapses_born += sprouted
                    self.stats.synaptogenesis_events += 1
                    event_record['sprouted'] = sprouted

        # Apply updated weights and topology back to brain
        if changed_slots and hasattr(agent, 'rewire_graft_slots'):
            agent.rewire_graft_slots(pre, post, g_w, changed_slots)
        elif pruned_count > 0 or changed_slots:
            with torch.no_grad():
                brain.graft_w.copy_(torch.as_tensor(g_w, dtype=torch.float32))
                if changed_slots:
                    if hasattr(brain, 'graft_pre'):
                        brain.graft_pre.copy_(torch.as_tensor(pre))
                        brain.graft_post.copy_(torch.as_tensor(post))
                    graph.graft_pre = pre
                    graph.graft_post = post

        # Update stats
        self.stats.active_synapses = int(np.sum(np.abs(g_w) >= self.prune_threshold))
        self.stats.mean_weight = float(np.mean(np.abs(g_w)))
        self.stats.recent_events.append(event_record)
        if len(self.stats.recent_events) > 32:
            self.stats.recent_events.pop(0)

        return event_record

    def export_state(self) -> Dict[str, Any]:
        return {
            'rate_mean': self.rate_mean.copy(),
            'rate_cov': self._rate_cov.copy(),
            'activity_steps': self.activity_steps,
            'rng': self.rng.bit_generator.state,
            'stats': asdict(self.stats),
        }

    def import_state(self, d: Dict[str, Any]) -> None:
        if 'rate_mean' in d:
            self.rate_mean = np.array(d['rate_mean'], dtype=np.float32)
        if 'rate_cov' in d:
            self._rate_cov = np.array(d['rate_cov'], dtype=np.float32)
            self._row_upper_bound = np.zeros(self._rate_cov.shape[0], dtype=np.float32)
            self._bounds_valid = False
        self.activity_steps = d.get('activity_steps', 0)
        if 'rng' in d:
            self.rng.bit_generator.state = d['rng']
        if 'stats' in d:
            st = d['stats']
            self.stats = NeurogenesisStats(**st)

    def summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable telemetry dictionary."""
        return {
            'step': self.stats.step,
            'active_synapses': self.stats.active_synapses,
            'synapses_born': self.stats.synapses_born,
            'synapses_pruned': self.stats.synapses_pruned,
            'mean_weight': round(self.stats.mean_weight, 5),
            'synaptogenesis_events': self.stats.synaptogenesis_events,
            'apoptosis_events': self.stats.apoptosis_events,
            'last_event': self.stats.recent_events[-1] if self.stats.recent_events else None,
        }
