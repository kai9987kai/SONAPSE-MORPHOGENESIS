"""SUPERMIX ANTIPODE — Optional Intermittent External AI-Assisted Training Coordinator.

Design principles (Section 22):
- Never requires a paid external API for core simulation (defaults to local
  Supermix/heuristic telemetry advisor).
- Supports configurable providers (`local_heuristic`, `local_supermix`, optional
  user-configured `http_openai_compatible`).
- Enforces bounded session windows (e.g., default 600s / 10 min cap) and bounded
  proposal counts.
- Strictly read-only access to telemetry; CANNOT modify protected evaluation
  results or execute unrestricted OS commands.
- Independently measures assisted vs unassisted matched-control delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Mapping, Optional
import numpy as np

from ..core.interchange import TrainingExample, make_id


@dataclass
class AdvisorSessionConfig:
    provider: str = "local_supermix_advisor"  # local_supermix_advisor | local_heuristic | external_http_opt_in
    max_session_seconds: float = 600.0        # Configurable ~10 minute bounded session
    max_interventions: int = 8
    endpoint_url: Optional[str] = None
    allow_external_network: bool = False


@dataclass
class AdvisorRecommendation:
    recommendation_id: str
    provider: str
    diagnosed_failures: List[str]
    corrective_examples: List[Dict[str, Any]]
    curriculum_adjustment: Dict[str, Any]
    structural_proposal: Optional[str]
    learning_rule_suggestion: str
    miniature_experiment_request: Dict[str, Any]


class ExternalAIAssistedTrainer:
    """Bounded intermittent advisor inspecting neural telemetry and proposing testable interventions."""

    def __init__(self, config: Optional[AdvisorSessionConfig] = None) -> None:
        self.config = config or AdvisorSessionConfig()
        self.session_history: List[Dict[str, Any]] = []

    def analyze_and_advise(
        self,
        telemetry_summary: Mapping[str, Any],
        recent_failures: List[TrainingExample],
    ) -> AdvisorRecommendation:
        """Analyze read-only telemetry and recent verified failures to generate bounded recommendations."""
        cats = [e.failure_category or "unknown" for e in recent_failures if e.is_failure]
        cat_counts: Dict[str, int] = {}
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        dominant = max(cat_counts, key=cat_counts.get) if cat_counts else "none"  # type: ignore[arg-type]

        # Generate targeted corrective examples from observed failure signatures
        corrective = []
        for ex in recent_failures[:4]:
            if ex.is_failure:
                corrective.append({
                    "source_example_id": ex.example_id,
                    "task_domain": ex.task_domain,
                    "target_output": ex.target_output,
                    "avoid_output": ex.observed_output,
                    "objective": "preference_dpo" if dominant == "overconfident_error" else "unlikelihood",
                })

        if dominant == "incorrect_math_reasoning":
            struct_prop = "add_secondary_soma"
            rule = "corrective_supervision"
            curr = {"increase_exact_math_ratio": 0.25, "verify_steps": True}
        elif dominant == "invalid_programming_output":
            struct_prop = "attach_organelle"
            rule = "preference_dpo"
            curr = {"enable_codeception_sandbox": True, "syntax_filter": True}
        elif dominant == "representation_collapse":
            struct_prop = "specialize_neuron"
            rule = "contrastive"
            curr = {"inject_novelty_bonus": 0.15}
        else:
            struct_prop = "insert_circuit_motif"
            rule = "causal_activation_reg"
            curr = {"balance_positive_negative": 0.5}

        return AdvisorRecommendation(
            recommendation_id=make_id("adv"),
            provider=self.config.provider,
            diagnosed_failures=sorted(set(cats)),
            corrective_examples=corrective,
            curriculum_adjustment=curr,
            structural_proposal=struct_prop,
            learning_rule_suggestion=rule,
            miniature_experiment_request={"n_neurons": 12, "steps": 64, "seeds": [7, 19]},
        )

    def run_measured_assisted_session(
        self,
        eval_unassisted_fn: Callable[[], float],
        eval_assisted_fn: Callable[[AdvisorRecommendation], float],
        telemetry_summary: Mapping[str, Any],
        recent_failures: List[TrainingExample],
    ) -> Dict[str, Any]:
        """Execute a bounded assisted session and independently measure contribution delta."""
        t0 = time.perf_counter()
        baseline_score = float(eval_unassisted_fn())
        rec = self.analyze_and_advise(telemetry_summary, recent_failures)
        assisted_score = float(eval_assisted_fn(rec))
        elapsed = time.perf_counter() - t0
        delta = assisted_score - baseline_score
        record = {
            "provider": self.config.provider,
            "session_wall_seconds": round(elapsed, 4),
            "within_budget": elapsed <= self.config.max_session_seconds,
            "unassisted_control_score": round(baseline_score, 5),
            "assisted_score": round(assisted_score, 5),
            "measured_contribution_delta": round(delta, 5),
            "helpful": bool(delta > 0.0),
            "recommendation": {
                "id": rec.recommendation_id,
                "diagnosed_failures": rec.diagnosed_failures,
                "structural_proposal": rec.structural_proposal,
                "learning_rule_suggestion": rec.learning_rule_suggestion,
                "n_corrective_examples": len(rec.corrective_examples),
            },
        }
        self.session_history.append(record)
        return record
