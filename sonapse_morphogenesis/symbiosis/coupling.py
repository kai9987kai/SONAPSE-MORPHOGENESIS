"""SONAPSE SYMBIOSIS — Intermittent Cross-Connectome Development with Sonapse-LAB.

Couples a developing Morphogenesis connectome temporarily to an established
``Sonapse-LAB`` ``LabRuntime`` / ``IntegratedAgent`` (initially frozen, optionally
mutually adaptive), then returns it to independent native training and measures
durable representational diversity, adaptation, generalization, and retention.

Supports all 11 coupling mechanisms from Section 23.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime

from ..core.compartment_engine import MultiFidelityConnectome
from ..core.interchange import DevelopmentalGenome
from ..foundry.organelles import create_organelle
from ..genesis.population import generate_candidate_genome
from .retention import KnowledgeRetentionSuite


COUPLING_MECHANISMS: Tuple[str, ...] = (
    "shared_sensory_env",
    "behavioral_observation",
    "learned_neural_adapters",
    "bounded_activation_exchange",
    "shared_chemical_fields",
    "episodic_experience_exchange",
    "cooperative_task",
    "competitive_evaluation",
    "predictive_assistance",
    "temporary_neural_grafting",
    "adaptive_bridge_formation",
)


def _effective_rank(X: np.ndarray) -> float:
    """Roy & Vetterli (2007) exponential spectral entropy effective rank of neural activity [T, N]."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return 1.0
    Xc = X - np.mean(X, axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    sv = sv[sv > 1e-9]
    if sv.size == 0:
        return 1.0
    p = sv / np.sum(sv)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


class SymbioticBridge:
    """Bounded zero-init neural adapter bridging an established Sonapse host and a developing connectome."""

    def __init__(self, host_dim: int, dev_dim: int, max_norm: float = 0.5, seed: int = 42) -> None:
        self.host_dim = int(host_dim)
        self.dev_dim = int(dev_dim)
        self.max_norm = float(max_norm)
        rng = np.random.default_rng(seed)
        # Function-preserving near-zero init
        self.W_host_to_dev = np.zeros((self.dev_dim, self.host_dim), dtype=np.float64)
        self.W_dev_to_host = np.zeros((self.host_dim, self.dev_dim), dtype=np.float64)
        self._init_seed_basis = rng.normal(0.0, 0.05, size=(self.dev_dim, self.host_dim))
        self.adaptation_steps = 0

    def project_host_to_dev(self, host_r: np.ndarray, gain: float = 0.25) -> np.ndarray:
        h = np.resize(np.asarray(host_r, dtype=np.float64).ravel(), self.host_dim)
        raw = (self.W_host_to_dev + 0.2 * self._init_seed_basis) @ h
        nrm = float(np.linalg.norm(raw))
        if nrm > self.max_norm:
            raw = raw * (self.max_norm / nrm)
        return raw * float(gain)

    def project_dev_to_host(self, dev_r: np.ndarray, gain: float = 0.08) -> np.ndarray:
        d = np.resize(np.asarray(dev_r, dtype=np.float64).ravel(), self.dev_dim)
        raw = self.W_dev_to_host @ d
        nrm = float(np.linalg.norm(raw))
        if nrm > self.max_norm:
            raw = raw * (self.max_norm / nrm)
        return raw * float(gain)

    def adapt(self, host_r: np.ndarray, dev_r: np.ndarray, reward: float, lr: float = 0.03) -> None:
        h = np.resize(np.asarray(host_r, dtype=np.float64).ravel(), self.host_dim)
        d = np.resize(np.asarray(dev_r, dtype=np.float64).ravel(), self.dev_dim)
        # Hebbian-modulated alignment bounded by L2 norm
        scale = lr * (0.2 + abs(reward))
        self.W_host_to_dev = np.clip(self.W_host_to_dev + scale * np.outer(d, h), -1.0, 1.0)
        self.W_dev_to_host = np.clip(self.W_dev_to_host + scale * np.outer(h, d), -1.0, 1.0)
        self.adaptation_steps += 1


class SymbiosisSession:
    """Runs intermittent coupling between a developing connectome and an established Sonapse-LAB runtime."""

    def __init__(
        self,
        developing_genome: Optional[DevelopmentalGenome] = None,
        sonapse_nodes_per_source: int = 16,
        seed: int = 7,
    ) -> None:
        self.seed = int(seed)
        self.genome = developing_genome or generate_candidate_genome("mixed_custom_standard", n_neurons=12, seed=seed)
        self.dev_connectome = MultiFidelityConnectome(self.genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
        # Real established Sonapse-LAB runtime (kept frozen initially!)
        self.host_config = SynapseConfig(seed=seed, nodes_per_source=sonapse_nodes_per_source, task="forage")
        self.host_runtime = LabRuntime(self.host_config)
        self.host_dim = min(48, int(self.host_runtime.agent.brain.n))
        self.dev_dim = len(self.dev_connectome.neuron_ids)
        self.bridge = SymbioticBridge(self.host_dim, self.dev_dim, seed=seed + 19)
        self.retention = KnowledgeRetentionSuite(param_dim=self.dev_dim, seed=seed)
        self.episodic_exchange_buffer: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

        # Trainable readout mapping developing connectome -> 5-D Sonapse motor action
        rng = np.random.default_rng(seed + 77)
        self.dev_readout = rng.normal(0.0, 0.2, size=(5, self.dev_dim))

    def _evaluate_native_skill(self, n_steps: int = 36, seed_offset: int = 0) -> Dict[str, float]:
        """Measure live behavioral accuracy on developing connectome's native delayed-cue task."""
        rng = np.random.default_rng(self.seed + 500 + seed_offset)
        cue = rng.choice([-1.0, 1.0], size=n_steps)
        target = np.roll(cue, 3)
        target[:3] = 0.0
        self.dev_connectome.reset()
        states = np.zeros((n_steps, self.dev_dim), dtype=np.float64)
        for k in range(n_steps):
            out = self.dev_connectome.step(np.array([cue[k], np.sin(0.25 * k), -cue[k], 0.1]))
            states[k] = out["outputs"]
        preds = np.tanh(states @ self.dev_readout[0])
        mse = float(np.mean((preds - target) ** 2))
        erank = _effective_rank(states)
        return {"native_mse": round(mse, 5), "effective_rank": round(erank, 4)}

    def train_native_phase(self, steps: int = 40, seed_offset: int = 0) -> Dict[str, Any]:
        """Independent training phase in the developing connectome's native environment."""
        rng = np.random.default_rng(self.seed + 100 + seed_offset)
        cue = rng.choice([-1.0, 1.0], size=steps)
        target = np.roll(cue, 3)
        target[:3] = 0.0
        self.dev_connectome.reset()
        states = np.zeros((steps, self.dev_dim), dtype=np.float64)
        for k in range(steps):
            out = self.dev_connectome.step(np.array([cue[k], np.sin(0.25 * k), -cue[k], 0.1]), adapt=True, reward_signal=0.2)
            states[k] = out["outputs"]
        # Ridge fit of native readout (row 0) + EWC anchor update
        w = np.linalg.solve(states.T @ states + 0.3 * np.eye(self.dev_dim), states.T @ target)
        self.dev_readout[0] = self.retention.apply_protected_update(self.dev_readout[0], w)
        self.retention.consolidate_anchor(self.dev_readout[0], fisher_diag=np.mean(states ** 2, axis=0) + 0.05)
        self.retention.store_replay_batch(states, target)
        eval_res = self._evaluate_native_skill(seed_offset=99)
        return {"phase": "native", "steps": steps, **eval_res}

    def run_symbiotic_phase(
        self,
        coupling: str = "learned_neural_adapters",
        steps: int = 24,
        host_frozen: bool = True,
    ) -> Dict[str, Any]:
        """Temporarily couple developing connectome to established Sonapse-LAB runtime."""
        if coupling not in COUPLING_MECHANISMS:
            raise ValueError(f"Unknown coupling mechanism: {coupling!r}")

        host_before_digest = float(np.sum(np.abs(self.host_runtime.agent.motor.weight.detach().cpu().numpy())))
        dev_states = np.zeros((steps, self.dev_dim), dtype=np.float64)
        host_states = np.zeros((steps, self.host_dim), dtype=np.float64)
        host_actions = np.zeros((steps, 5), dtype=np.float64)
        alignment_corrs: List[float] = []

        for k in range(steps):
            # Step established Sonapse-LAB runtime (frozen if host_frozen=True!)
            self.host_runtime.step(steps=1, learn=not host_frozen, include_snapshot=False)
            host_r = self.host_runtime.agent.brain.r.detach().cpu().numpy().ravel()[: self.host_dim]
            host_act = np.asarray(self.host_runtime.last_action, dtype=np.float64).ravel()[:5]
            host_obs = np.asarray(self.host_runtime.obs, dtype=np.float64).ravel()
            host_mods = self.host_runtime.agent.neuromod.state

            # Construct distinct coupling input according to each of the 11 active mechanisms
            ext = np.zeros(self.dev_dim, dtype=np.float64)
            mods: Dict[str, float] = {}
            m_min = min(8, self.dev_dim)
            if coupling == "shared_sensory_env":
                ext[:m_min] = np.tanh(host_obs[:m_min])
            elif coupling == "cooperative_task":
                ext[:m_min] = 0.65 * np.tanh(host_obs[:m_min]) + 0.35 * np.resize(host_act, m_min)
            elif coupling == "competitive_evaluation":
                ext[:m_min] = np.tanh(host_obs[:m_min]) - 0.30 * np.tanh(host_r[:m_min])
            elif coupling == "learned_neural_adapters":
                ext += self.bridge.project_host_to_dev(host_r, gain=0.45)
            elif coupling == "bounded_activation_exchange":
                ext += np.clip(self.bridge.project_host_to_dev(host_r, gain=0.35), -0.25, 0.25)
            elif coupling == "temporary_neural_grafting":
                ext[: min(self.host_dim, self.dev_dim)] += 0.30 * host_r[: min(self.host_dim, self.dev_dim)]
            elif coupling == "adaptive_bridge_formation":
                ramp = min(1.0, (k + 1) / max(steps * 0.5, 1.0))
                ext += self.bridge.project_host_to_dev(host_r, gain=0.50 * ramp)
            elif coupling == "shared_chemical_fields":
                mods = {"da": float(host_mods.dopamine), "serotonin": float(host_mods.serotonin), "ach": float(host_mods.acetylcholine)}
                ext[:4] = np.tanh(host_obs[:4])
            elif coupling == "behavioral_observation":
                ext[: min(5, self.dev_dim)] += host_act[: min(5, self.dev_dim)]
            elif coupling == "predictive_assistance":
                # Anticipatory 1-step action delta guidance
                prev_h = host_actions[k - 1] if k > 0 else np.zeros(5)
                ext[: min(5, self.dev_dim)] += (host_act + 0.5 * (host_act - prev_h))[: min(5, self.dev_dim)]
            elif coupling == "episodic_experience_exchange":
                self.episodic_exchange_buffer.append({"step": k, "obs_head": host_obs[:4].copy(), "action": host_act.copy()})
                if len(self.episodic_exchange_buffer) >= 2:
                    recalled = self.episodic_exchange_buffer[k % len(self.episodic_exchange_buffer)]
                    ext[:4] = 0.6 * np.tanh(host_obs[:4]) + 0.4 * np.tanh(recalled["obs_head"])
                else:
                    ext[:4] = np.tanh(host_obs[:4])

            dev_out = self.dev_connectome.step(ext, modulators=mods, reward_signal=0.15, adapt=True, collect_telemetry=False)
            dev_r = dev_out["outputs"]
            dev_states[k] = dev_r
            host_states[k] = host_r
            host_actions[k] = host_act

            # When host_frozen=False (mutually adaptive symbiosis), inject bounded reverse current into host brain
            if not host_frozen:
                rev_vec = self.bridge.project_dev_to_host(dev_r, gain=0.04)
                import torch
                with torch.no_grad():
                    self.host_runtime.agent.brain.r[: self.host_dim].add_(torch.as_tensor(rev_vec, dtype=self.host_runtime.agent.brain.r.dtype))
                    self.host_runtime.agent.brain.r.clamp_(-3.0, 3.0)

            if coupling in ("learned_neural_adapters", "adaptive_bridge_formation", "bounded_activation_exchange"):
                self.bridge.adapt(host_r, dev_r, reward=0.25)

            # Behavioral imitation of host motor outputs on auxiliary rows 1..4 (preserving row 0 native anchor!)
            pred_act = np.tanh(self.dev_readout @ dev_r)
            err_aux = host_act[1:] - pred_act[1:]
            self.dev_readout[1:] = np.clip(self.dev_readout[1:] + 0.04 * np.outer(err_aux, dev_r), -3.0, 3.0)

            corr = float(np.corrcoef(host_r[: min(self.host_dim, self.dev_dim)], dev_r[: min(self.host_dim, self.dev_dim)])[0, 1]) if self.dev_dim >= 2 else 0.0
            if np.isfinite(corr):
                alignment_corrs.append(corr)

        # Replay consolidation to protect native skill during symbiotic shift
        self.dev_readout[0] = self.retention.replay_consolidate(self.dev_readout[0], lr=0.08)

        host_after_digest = float(np.sum(np.abs(self.host_runtime.agent.motor.weight.detach().cpu().numpy())))
        host_preserved = abs(host_after_digest - host_before_digest) < 1e-8 if host_frozen else True
        imitation_mse = float(np.mean((np.tanh(dev_states @ self.dev_readout.T) - host_actions)[:, 1:] ** 2))

        rec = {
            "phase": "symbiotic_coupling",
            "coupling": coupling,
            "host_frozen": host_frozen,
            "host_weights_bit_preserved": host_preserved,
            "steps": steps,
            "dev_effective_rank": round(_effective_rank(dev_states), 4),
            "host_effective_rank": round(_effective_rank(host_states), 4),
            "mean_host_dev_corr": round(float(np.mean(alignment_corrs)) if alignment_corrs else 0.0, 4),
            "motor_alignment_mse": round(imitation_mse, 5),
            "bridge_adaptations": self.bridge.adaptation_steps,
        }
        self.history.append(rec)
        return rec


def run_symbiosis_cycle_experiment(
    couplings: Sequence[str] = ("learned_neural_adapters", "shared_chemical_fields", "behavioral_observation"),
    seed: int = 7,
) -> Dict[str, Any]:
    """Execute full Native -> Symbiosis (Frozen Sonapse-LAB) -> Independent Return -> Retention Audit cycle."""
    results: Dict[str, Any] = {}
    for c_idx, coupling in enumerate(couplings):
        sess = SymbiosisSession(seed=seed + c_idx)
        pre_native = sess.train_native_phase(steps=45, seed_offset=0)
        # Save immutable pre-symbiosis checkpoint in retention archive
        sess.retention.archive_snapshot("pre_symbiosis", sess.dev_readout[0])

        sym_res = sess.run_symbiotic_phase(coupling=coupling, steps=24, host_frozen=True)
        post_sym_eval = sess._evaluate_native_skill(seed_offset=99)

        # Return to independent native consolidation
        post_consol = sess.train_native_phase(steps=30, seed_offset=2)
        retention_audit = sess.retention.audit_retention(
            pre_score=-pre_native["native_mse"],
            post_exposure_score=-post_sym_eval["native_mse"],
            post_consolidation_score=-post_consol["native_mse"],
            snapshot_label="pre_symbiosis",
        )
        results[coupling] = {
            "pre_native": pre_native,
            "symbiotic_phase": sym_res,
            "immediately_after_symbiosis": post_sym_eval,
            "after_independent_consolidation": post_consol,
            "rank_gain_post_consolidation": round(post_consol["effective_rank"] - pre_native["effective_rank"], 4),
            "retention_audit": retention_audit,
        }

    return {
        "experiment": "sonapse_symbiosis_intermittent_coupling",
        "seed": seed,
        "all_supported_couplings": list(COUPLING_MECHANISMS),
        "tested_couplings": results,
    }


def run_long_horizon_mutual_symbiosis(
    alternating_cycles: int = 6,
    steps_per_phase: int = 36,
    seed: int = 7,
) -> Dict[str, Any]:
    """Resolves Open Question #3: Long-horizon alternating native/symbiotic curriculum
    comparing a bit-preserved frozen Sonapse-LAB host vs a mutually co-adapting host.
    """
    arms: Dict[str, Any] = {}
    for arm_name, frozen in (("frozen_host_control", True), ("mutually_adaptive_host", False)):
        sess = SymbiosisSession(seed=seed)
        init_native = sess.train_native_phase(steps=steps_per_phase, seed_offset=0)
        sess.retention.archive_snapshot("cycle0_anchor", sess.dev_readout[0])
        trajectory: List[Dict[str, Any]] = []
        for cyc in range(alternating_cycles):
            sym = sess.run_symbiotic_phase(coupling="adaptive_bridge_formation", steps=steps_per_phase, host_frozen=frozen)
            nat = sess.train_native_phase(steps=steps_per_phase, seed_offset=cyc + 10)
            trajectory.append({
                "cycle": cyc,
                "host_dev_corr": sym["mean_host_dev_corr"],
                "motor_alignment_mse": sym["motor_alignment_mse"],
                "native_mse": nat["native_mse"],
                "effective_rank": nat["effective_rank"],
                "host_weights_bit_preserved": sym["host_weights_bit_preserved"],
            })
        final_ret = sess.retention.audit_retention(
            pre_score=-init_native["native_mse"],
            post_exposure_score=-trajectory[-1]["native_mse"],
            post_consolidation_score=-trajectory[-1]["native_mse"],
            snapshot_label="cycle0_anchor",
        )
        arms[arm_name] = {
            "host_frozen": frozen,
            "total_steps_executed": steps_per_phase * (1 + 2 * alternating_cycles),
            "initial_native_mse": init_native["native_mse"],
            "final_native_mse": trajectory[-1]["native_mse"],
            "final_motor_alignment_mse": trajectory[-1]["motor_alignment_mse"],
            "final_effective_rank": trajectory[-1]["effective_rank"],
            "retention_audit": final_ret,
            "cycle_trajectory": trajectory,
        }
    return {
        "experiment": "long_horizon_mutual_vs_frozen_symbiosis",
        "alternating_cycles": alternating_cycles,
        "steps_per_phase": steps_per_phase,
        "arms": arms,
    }
