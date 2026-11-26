"""SONAPSE ARCHITECT — 11 Lightweight Non-LLM Internal Research Experts & Independent Meta-Auditor.

Every expert is a genuine online statistical/regularized linear predictor trained
on the historical experimental record—predicting likely benefit, cost, risk, and
epistemic uncertainty for proposed interventions, and requesting isolated miniature
experiments when uncertainty exceeds threshold.

An independent ``ExpertMetaAuditor`` tracks calibration error and optimism bias
across all 11 experts and flags unreliable or systematically biased advisors.
Experts advise only; they NEVER hold direct write authority over the main learner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


EXPERT_ROLES: Tuple[str, ...] = (
    "morphology_assessor",
    "plasticity_assessor",
    "memory_retention",
    "stability_assessor",
    "resource_efficiency",
    "novelty_detector",
    "curriculum_selector",
    "cross_connectome_transfer",
    "causal_inference",
    "experimental_design",
    "scientific_anomaly_detector",
)


@dataclass
class ExpertPrediction:
    expert_name: str
    predicted_benefit: float
    predicted_cost: float
    predicted_risk: float
    uncertainty: float
    recommends_miniature_experiment: bool


class LearnedResearchExpert:
    """Lightweight non-LLM Bayesian-ridge online predictor trained on real experiment histories.

    Uses role-specialized orthogonal feature emphasis + Sherman-Morrison O(d^2) rank-1
    inverse covariance updates so all 11 experts maintain genuine epistemic diversity
    and constant-time online learning over multi-hour campaigns.
    """

    def __init__(self, name: str, feat_dim: int = 8, ridge: float = 0.5, seed: int = 42) -> None:
        self.name = name
        self.feat_dim = int(feat_dim)
        self.ridge = float(ridge)
        self.rng = np.random.default_rng(seed)
        # Role-specific feature lens & weak prior breaking identical-clone degeneracy
        raw_lens = 0.65 + 0.70 * self.rng.random(self.feat_dim)
        self.feature_lens = raw_lens / float(np.mean(raw_lens))
        self.prior_ben = self.rng.normal(0.0, 0.015, size=self.feat_dim)
        self.w_ben = self.prior_ben.copy()
        self.w_cost = np.abs(self.rng.normal(0.02, 0.01, size=self.feat_dim))
        self.w_risk = np.abs(self.rng.normal(0.02, 0.01, size=self.feat_dim))
        self.cov_inv = np.eye(self.feat_dim, dtype=np.float64) / self.ridge
        self.xty_ben = self.ridge * self.prior_ben.copy()
        self.xty_cost = self.ridge * self.w_cost.copy()
        self.xty_risk = self.ridge * self.w_risk.copy()
        self.sample_count = 0
        # Bounded history & calibration ledger for Meta-Auditor
        self.pred_history: List[float] = []
        self.actual_history: List[float] = []

    def _encode_proposal(self, proposal_features: Sequence[float]) -> np.ndarray:
        arr = np.asarray(proposal_features, dtype=np.float64).ravel()
        out = np.zeros(self.feat_dim, dtype=np.float64)
        m = min(arr.size, self.feat_dim)
        out[:m] = np.nan_to_num(arr[:m], nan=0.0)
        return out * self.feature_lens

    def fit_observation(self, proposal_features: Sequence[float], actual_benefit: float, actual_cost: float, actual_risk: float) -> None:
        x = self._encode_proposal(proposal_features)
        pred_before = float(np.dot(self.w_ben, x))
        self.pred_history.append(pred_before)
        self.actual_history.append(float(actual_benefit))
        if len(self.pred_history) > 256:
            self.pred_history = self.pred_history[-256:]
            self.actual_history = self.actual_history[-256:]

        # Exact Sherman-Morrison rank-1 update: (A + x x^T)^-1 = A^-1 - (A^-1 x x^T A^-1) / (1 + x^T A^-1 x)
        ax = self.cov_inv @ x
        denom = 1.0 + float(np.dot(x, ax))
        self.cov_inv = self.cov_inv - np.outer(ax, ax) / max(denom, 1e-9)
        self.xty_ben = self.xty_ben + x * float(actual_benefit)
        self.xty_cost = self.xty_cost + x * float(actual_cost)
        self.xty_risk = self.xty_risk + x * float(actual_risk)
        self.w_ben = self.cov_inv @ self.xty_ben
        self.w_cost = self.cov_inv @ self.xty_cost
        self.w_risk = self.cov_inv @ self.xty_risk
        self.sample_count += 1

    def predict(self, proposal_features: Sequence[float]) -> ExpertPrediction:
        x = self._encode_proposal(proposal_features)
        ben = float(np.dot(self.w_ben, x))
        cost = float(max(0.0, np.dot(self.w_cost, x)))
        risk = float(np.clip(np.dot(self.w_risk, x), 0.0, 1.0))
        unc = float(math.sqrt(max(float(x @ self.cov_inv @ x), 1e-6)))
        return ExpertPrediction(
            expert_name=self.name,
            predicted_benefit=round(ben, 5),
            predicted_cost=round(cost, 5),
            predicted_risk=round(risk, 5),
            uncertainty=round(unc, 5),
            recommends_miniature_experiment=bool(unc > 0.65 or risk > 0.45),
        )


class ExpertMetaAuditor:
    """Independent auditor detecting unreliable or systematically biased internal research experts."""

    def __init__(self, max_allowed_rmse: float = 0.45, max_abs_bias: float = 0.20) -> None:
        self.max_allowed_rmse = float(max_allowed_rmse)
        self.max_abs_bias = float(max_abs_bias)

    def audit_panel(self, experts: Mapping[str, LearnedResearchExpert]) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        flagged: List[str] = []
        for name, exp in experts.items():
            if len(exp.actual_history) < 3:
                report[name] = {"samples": len(exp.actual_history), "status": "warmup", "trusted": True}
                continue
            preds = np.asarray(exp.pred_history[-25:], dtype=np.float64)
            acts = np.asarray(exp.actual_history[-25:], dtype=np.float64)
            residuals = preds - acts
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            bias = float(np.mean(residuals))  # >0 => systematic optimism bias!
            unreliable = (rmse > self.max_allowed_rmse) or (abs(bias) > self.max_abs_bias)
            if unreliable:
                flagged.append(name)
            report[name] = {
                "samples": len(exp.actual_history),
                "prequential_rmse": round(rmse, 5),
                "signed_bias": round(bias, 5),
                "optimism_biased": bool(bias > self.max_abs_bias),
                "trusted": not unreliable,
                "status": "flagged_unreliable" if unreliable else "calibrated_ok",
            }
        return {
            "experts_audited": len(experts),
            "flagged_experts": flagged,
            "details": report,
        }


class InternalExpertPanel:
    """Coordinates all 11 specialized non-LLM research experts plus the independent Meta-Auditor."""

    def __init__(self, feat_dim: int = 8, seed: int = 42) -> None:
        self.experts: Dict[str, LearnedResearchExpert] = {
            role: LearnedResearchExpert(role, feat_dim=feat_dim, seed=seed + idx)
            for idx, role in enumerate(EXPERT_ROLES)
        }
        self.meta_auditor = ExpertMetaAuditor()

    def consult(self, proposal_features: Sequence[float]) -> Dict[str, Any]:
        audit = self.meta_auditor.audit_panel(self.experts)
        preds = {}
        trusted_benefits = []
        request_mini = False
        for role, exp in self.experts.items():
            p = exp.predict(proposal_features)
            preds[role] = {
                "benefit": p.predicted_benefit,
                "cost": p.predicted_cost,
                "risk": p.predicted_risk,
                "uncertainty": p.uncertainty,
                "mini_exp": p.recommends_miniature_experiment,
            }
            if audit["details"][role]["trusted"]:
                trusted_benefits.append(p.predicted_benefit)
            if p.recommends_miniature_experiment:
                request_mini = True
        return {
            "consensus_benefit": round(float(np.mean(trusted_benefits)) if trusted_benefits else 0.0, 5),
            "recommend_miniature_experiment": request_mini,
            "flagged_experts": audit["flagged_experts"],
            "per_expert": preds,
        }

    def update_all(self, proposal_features: Sequence[float], actual_benefit: float, actual_cost: float, actual_risk: float) -> None:
        for exp in self.experts.values():
            exp.fit_observation(proposal_features, actual_benefit, actual_cost, actual_risk)

    def score_fantasy_decision_risk(self, candidate_features: Sequence[Sequence[float]], cvar_alpha: float = 0.25) -> List[Dict[str, Any]]:
        """COUNTERFORM v0.2 inspired CVaR sensitivity-particle fantasy decision-risk acquisition."""
        scored: List[Dict[str, Any]] = []
        for idx, feat in enumerate(candidate_features):
            panel = self.consult(feat)
            per_exp = list(panel["per_expert"].values())
            utilities = np.array([
                float(p["benefit"]) - 0.15 * float(p["cost"]) - 0.35 * float(p["risk"])
                for p in per_exp
            ], dtype=np.float64)
            k_tail = max(1, int(math.ceil(cvar_alpha * len(utilities))))
            cvar_tail = float(np.mean(np.sort(utilities)[:k_tail]))
            disagreement = float(np.std(utilities))
            acq_score = 0.55 * float(np.mean(utilities)) + 0.30 * cvar_tail + 0.15 * disagreement
            scored.append({
                "candidate_idx": idx,
                "mean_utility": round(float(np.mean(utilities)), 5),
                "cvar_25": round(cvar_tail, 5),
                "epistemic_disagreement": round(disagreement, 5),
                "acquisition_score": round(acq_score, 5),
                "abstain_exploratory": bool(cvar_tail < -0.15),
            })
        scored.sort(key=lambda r: r["acquisition_score"], reverse=True)
        return scored
