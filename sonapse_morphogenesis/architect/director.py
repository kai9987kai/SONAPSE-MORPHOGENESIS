"""SONAPSE ARCHITECT — Independent 14-Step Autonomous Research Director & Training-Pack Generator.

Operates strictly outside the main connectome (Section 25 & 26):
1. Observe performance & neural activity
2. Diagnose weaknesses / anomalies
3. Formulate explicit testable hypotheses
4. Design controlled experimental protocols
5. Construct candidate training packs (delayed memory, math, failure correction, curriculum)
6. Generate miniature connectome clones
7. Test alternative neural morphologies
8. Test alternative learning rules
9. Evaluate outcomes on isolated screening suites
10. Reject ineffective or forgetting-inducing proposals
11. Select promising interventions
12. Verify scale-transfer to larger connectomes (never assuming micro-scale proves large-scale!)
13. Record all findings & failed hypotheses in ExperimentRegistry
14. Recommend subsequent research directions.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from ..core.compartment_engine import MultiFidelityConnectome
from ..core.interchange import (
    DevelopmentalGenome,
    EvidenceState,
    ExperimentReceipt,
    ObjectiveCategory,
    make_id,
)
from ..core.registry import ExperimentRegistry
from ..foundry.organelles import create_organelle
from ..genesis.genome_evolution import apply_genome_mutation
from ..genesis.population import generate_candidate_genome
from .experts import InternalExpertPanel
from .learning_rules import DynamicLearningEngine


@dataclass
class CandidateTrainingPack:
    pack_id: str
    kind: str  # delayed_memory | math_transform | corrective_failure | multimodal_mix | curriculum_step
    difficulty: float
    X: np.ndarray
    y: np.ndarray
    rationale: str


def construct_training_pack(weakness: str, steps: int = 64, seed: int = 42) -> CandidateTrainingPack:
    """Autonomously synthesize a targeted training pack in response to diagnosed weakness."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=np.float64)
    if weakness == "temporal_delay_deficit":
        cue = rng.choice([-1.0, 1.0], size=steps)
        X = np.stack([cue, rng.normal(0, 0.25, size=steps), np.sin(0.2 * t), np.cos(0.2 * t)], axis=1)
        y = np.roll(cue, 5)
        y[:5] = 0.0
        return CandidateTrainingPack(make_id("pack"), "delayed_memory", 0.7, X, y, "Targets 5-step delayed recall deficit.")
    if weakness == "nonlinear_math_deficit":
        a = rng.uniform(-1.0, 1.0, size=steps)
        b = rng.uniform(-1.0, 1.0, size=steps)
        X = np.stack([a, b, a + b, a - b], axis=1)
        y = np.tanh(a * b + 0.4 * (a ** 2 - b ** 2))
        return CandidateTrainingPack(make_id("pack"), "math_transform", 0.65, X, y, "Targets multiplicative algebraic relation deficit.")
    # corrective_failure / general stability pack
    sig = np.sin(0.3 * t)
    corrupt = sig + np.where(rng.random(steps) < 0.25, rng.choice([-2.5, 2.5], size=steps), 0.0)
    X = np.stack([sig, corrupt, np.abs(corrupt - sig), np.cos(0.3 * t)], axis=1)
    y = sig
    return CandidateTrainingPack(make_id("pack"), "corrective_failure", 0.6, X, y, "Targets robustness against corrupted outlier cues.")


class AutonomousResearchDirector:
    """Independent SONAPSE ARCHITECT orchestrator running isolated miniature & scale-transfer studies."""

    def __init__(self, registry: Optional[ExperimentRegistry] = None, seed: int = 42) -> None:
        self.seed = int(seed)
        self.registry = registry
        self.expert_panel = InternalExpertPanel(feat_dim=8, seed=seed)
        self.learning_engine = DynamicLearningEngine(seed=seed)
        self.research_log: List[Dict[str, Any]] = []

    def _eval_connectome_on_pack(self, genome: DevelopmentalGenome, pack: CandidateTrainingPack, seed: int) -> Tuple[float, float]:
        sim = MultiFidelityConnectome(genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
        T = pack.X.shape[0]
        n = len(sim.neuron_ids)
        S = np.zeros((T, n), dtype=np.float64)
        for k in range(T):
            S[k] = sim.step(pack.X[k])["outputs"]
        sp = int(0.6 * T)
        w = np.linalg.solve(S[:sp].T @ S[:sp] + 0.35 * np.eye(n), S[:sp].T @ pack.y[:sp])
        mse = float(np.mean((S[sp:] @ w - pack.y[sp:]) ** 2))
        return mse, sim.total_energy

    def run_research_cycle(
        self,
        observed_diagnostics: Optional[Mapping[str, Any]] = None,
        mini_neurons: int = 8,
        scale_neurons: int = 16,
    ) -> Dict[str, Any]:
        """Execute one full 14-step autonomous research cycle strictly on isolated clones."""
        t0 = time.perf_counter()
        diag = dict(observed_diagnostics or {"delay_mse": 0.62, "math_mse": 0.45, "forgetting": 0.04})

        # Step 1-3: Observe, diagnose, and form explicit testable hypothesis
        weakness = "temporal_delay_deficit" if diag.get("delay_mse", 0.5) >= diag.get("math_mse", 0.4) else "nonlinear_math_deficit"
        hypothesis = f"Adding targeted morphology mutations + dynamic learning rules improves {weakness} without degrading retention or failing scale transfer."

        # Step 4-5: Construct candidate training pack + baseline control pack
        pack = construct_training_pack(weakness, steps=64, seed=self.seed)
        retention_pack = construct_training_pack("corrective_failure", steps=48, seed=self.seed + 77)

        # Step 6: Generate isolated miniature connectome baseline
        base_mini = generate_candidate_genome("mixed_custom_standard", n_neurons=mini_neurons, seed=self.seed)
        base_mse, base_eng = self._eval_connectome_on_pack(base_mini, pack, seed=self.seed + 1)
        base_ret_mse, _ = self._eval_connectome_on_pack(base_mini, retention_pack, seed=self.seed + 2)

        # Step 7-10: Screen candidate morphological + organelle + rule proposals, consulting non-LLM experts
        candidate_ops = ("attach_organelle", "add_axonal_pathway", "specialize_neuron", "insert_circuit_motif", "add_secondary_soma")
        trials: List[Dict[str, Any]] = []
        best_trial: Optional[Dict[str, Any]] = None
        best_genome: Optional[DevelopmentalGenome] = None
        rejected_trials: List[Dict[str, Any]] = []

        for idx, op in enumerate(candidate_ops):
            feat = [float(idx) / 5.0, base_mse, base_ret_mse, float(mini_neurons) / 32.0, pack.difficulty, 1.0, 0.5, 0.2]
            advice = self.expert_panel.consult(feat)
            cand_g, tx = apply_genome_mutation(base_mini, operator=op, seed=self.seed + 200 + idx)
            cand_mse, cand_eng = self._eval_connectome_on_pack(cand_g, pack, seed=self.seed + 1)
            cand_ret_mse, _ = self._eval_connectome_on_pack(cand_g, retention_pack, seed=self.seed + 2)

            benefit = base_mse - cand_mse
            forgetting_increase = max(0.0, cand_ret_mse - base_ret_mse)
            cost_delta = max(0.0, (cand_eng - base_eng) / max(base_eng, 1.0))
            risk = float(np.clip(forgetting_increase * 2.0, 0.0, 1.0))

            # Update all 11 internal research experts with REAL measured outcomes
            self.expert_panel.update_all(feat, benefit, cost_delta, risk)

            accepted = bool(benefit > 0.002 and forgetting_increase < 0.05)
            rec = {
                "operator": op,
                "mini_mse": round(cand_mse, 5),
                "benefit_delta_mse": round(benefit, 5),
                "forgetting_delta": round(forgetting_increase, 5),
                "cost_delta": round(cost_delta, 4),
                "expert_consensus_before": advice["consensus_benefit"],
                "accepted_at_micro_scale": accepted,
            }
            trials.append(rec)
            if accepted and (best_trial is None or benefit > best_trial["benefit_delta_mse"]):
                best_trial = rec
                best_genome = cand_g
            else:
                rejected_trials.append(rec)
                if self.registry is not None:
                    self.registry.record_failed_hypothesis("architect", f"{op} on {weakness}", "no_gain_or_forgetting_exceeded", rec)

        # Step 11-12: Scale-transfer verification on larger connectome (never trust micro scale alone!)
        scale_transfer: Dict[str, Any] = {"tested": False, "transferred": False}
        if best_trial is not None:
            base_large = generate_candidate_genome("mixed_custom_standard", n_neurons=scale_neurons, seed=self.seed + 50)
            large_base_mse, _ = self._eval_connectome_on_pack(base_large, pack, seed=self.seed + 51)
            large_mut, _ = apply_genome_mutation(base_large, operator=best_trial["operator"], seed=self.seed + 52)
            large_mut_mse, _ = self._eval_connectome_on_pack(large_mut, pack, seed=self.seed + 51)
            scale_gain = large_base_mse - large_mut_mse
            scale_transfer = {
                "tested": True,
                "scale_neurons": scale_neurons,
                "operator": best_trial["operator"],
                "large_baseline_mse": round(large_base_mse, 5),
                "large_mutated_mse": round(large_mut_mse, 5),
                "scale_gain_mse": round(scale_gain, 5),
                "transferred": bool(scale_gain > 0.0),
            }

        # Step 13-14: Audit experts, record receipt, and propose next research directions
        expert_audit = self.expert_panel.meta_auditor.audit_panel(self.expert_panel.experts)
        next_recommendations = [
            f"Replicate winning operator '{best_trial['operator']}' across 8 fresh seeds under SENTINEL held-out suite."
            if best_trial else "Expand search to multi-step organelle + learning-rule combinations.",
            "Verify pack transfer across non-stationary curriculum difficulty increments.",
        ]

        cycle_result = {
            "cycle_id": make_id("arch"),
            "diagnosed_weakness": weakness,
            "hypothesis": hypothesis,
            "training_pack": {"pack_id": pack.pack_id, "kind": pack.kind, "rationale": pack.rationale},
            "micro_baseline_mse": round(base_mse, 5),
            "trials": trials,
            "accepted_count": sum(1 for t in trials if t["accepted_at_micro_scale"]),
            "rejected_count": len(rejected_trials),
            "selected_proposal": best_trial,
            "scale_transfer_verification": scale_transfer,
            "expert_meta_audit": {
                "experts_audited": expert_audit["experts_audited"],
                "flagged_experts": expert_audit["flagged_experts"],
            },
            "next_recommendations": next_recommendations,
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

        if self.registry is not None:
            ev_state = EvidenceState.CANDIDATE_DISCOVERY if scale_transfer.get("transferred") else EvidenceState.UNVERIFIED_OBSERVATION
            receipt = ExperimentReceipt(
                experiment_id=cycle_result["cycle_id"],
                title=f"Architect Autonomous Cycle ({weakness})",
                module="architect",
                hypothesis=hypothesis,
                objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
                evidence_state=ev_state,
                seeds=[self.seed],
                config={"mini_neurons": mini_neurons, "scale_neurons": scale_neurons},
                metrics={"micro_baseline_mse": base_mse, "selected": best_trial, "scale_transfer": scale_transfer},
                failed_experiments=rejected_trials,
                resource_usage={"wall_ms": cycle_result["wall_ms"]},
            ).seal()
            self.registry.record_experiment(receipt)

        self.research_log.append(cycle_result)
        return cycle_result
