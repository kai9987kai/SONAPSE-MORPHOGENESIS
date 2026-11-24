"""SONAPSE SPECTRA — Controlled Internal Perception & Hallucination-Like Simulation.

Models selected perceptual mechanisms (Section 16):
- Internally generated visual/auditory replay without external stimulus
- Cross-modal sensory substitution
- Competition between incoming sensory evidence and top-down internal priors
- Source confusion, dream-like spontaneous activity, sensory gain shifts, and
  sensory deprivation
- Quantitative evaluation of prediction error, source discrimination AUROC
  (internal vs external origin), recovery latency, and behavioral impact.

Explicit epistemic notice: This is a computational model of prior-vs-evidence
perceptual dynamics, NOT proof of human-like hallucinations or subjective experience.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Exact rank-based Mann-Whitney AUROC without external dependencies."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / float(pos.size * neg.size)


class InternalPerceptionSimulator:
    """Predictive-coding perceptual loop measuring internal-vs-external source discrimination."""

    def __init__(self, dim: int = 12, seed: int = 42) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.W_prior = rng.normal(0.0, 0.35, size=(dim, dim))
        self.W_enc = rng.normal(0.0, 0.5, size=(dim, dim))
        # Stored episodic sensory templates for dream/memory replay
        self.memory_templates = rng.normal(0.0, 0.8, size=(4, dim))
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros(self.dim, dtype=np.float64)
        self.prior_pred = np.zeros(self.dim, dtype=np.float64)

    def run_condition(
        self,
        condition: str = "prior_evidence_competition",
        steps: int = 60,
        sensory_gain: float = 1.0,
        prior_gain: float = 0.7,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Simulate a controlled perceptual regime and evaluate source discrimination & recovery."""
        rng = np.random.default_rng(seed)
        self.reset()

        is_external_flags: List[int] = []
        source_scores: List[float] = []  # higher score => connectome classifies as external
        pred_errors: List[float] = []
        internal_strengths: List[float] = []

        for t in range(steps):
            # Ground-truth external stimulus active on alternating blocks or suppressed in deprivation/dream
            if condition == "sensory_deprivation":
                ext_active = False
                ext_stim = np.zeros(self.dim)
                g_s, g_p = 0.05 * sensory_gain, 1.35 * prior_gain
            elif condition == "dream_replay":
                ext_active = False
                ext_stim = np.zeros(self.dim)
                g_s, g_p = 0.0, 1.5 * prior_gain
            elif condition == "cross_modal_substitution":
                ext_active = (t % 2 == 0)
                raw = self.memory_templates[t % 4] if ext_active else np.zeros(self.dim)
                # Permute channels to simulate sensory substitution across modalities
                ext_stim = np.roll(raw, 4)
                g_s, g_p = sensory_gain, prior_gain
            elif condition == "high_gain_ambiguity":
                ext_active = (t % 3 != 0)
                ext_stim = rng.normal(0.0, 0.35, size=self.dim) if ext_active else np.zeros(self.dim)
                g_s, g_p = 0.45 * sensory_gain, 1.45 * prior_gain
            else:  # prior_evidence_competition
                ext_active = (t % 2 == 0)
                ext_stim = (self.memory_templates[(t // 2) % 4] + rng.normal(0.0, 0.15, size=self.dim)) if ext_active else np.zeros(self.dim)
                g_s, g_p = sensory_gain, prior_gain

            # Internal generative prior prediction (memory replay when external is weak)
            mem_pull = 0.35 * self.memory_templates[(t // 5) % 4]
            self.prior_pred = np.tanh(self.W_prior @ self.h + mem_pull)

            # Sensory bottom-up signal + receptor noise
            sensed = g_s * ext_stim + rng.normal(0.0, 0.06, size=self.dim)
            # Prediction error = discrepancy between incoming evidence and top-down prior
            pe = sensed - g_p * self.prior_pred
            pe_mag = float(np.linalg.norm(pe))
            prior_mag = float(np.linalg.norm(g_p * self.prior_pred))
            sensed_mag = float(np.linalg.norm(sensed))

            # Update internal perceptual state
            self.h = 0.45 * self.h + 0.55 * np.tanh(self.W_enc @ (sensed + g_p * self.prior_pred))

            # Source-monitoring classifier statistic: bottom-up precision-weighted innovation
            # External inputs produce structured innovations orthogonal to the internal prior
            cos_sim = float(np.dot(sensed, self.prior_pred) / ((sensed_mag * np.linalg.norm(self.prior_pred)) + 1e-8))
            source_evidence = sensed_mag * (1.0 - 0.35 * abs(cos_sim)) - 0.12 * prior_mag

            is_external_flags.append(1 if ext_active else 0)
            source_scores.append(source_evidence)
            pred_errors.append(pe_mag)
            internal_strengths.append(float(np.linalg.norm(self.h)))

        y_arr = np.asarray(is_external_flags)
        s_arr = np.asarray(source_scores)
        auroc = _roc_auc(y_arr, s_arr) if (0 < int(np.sum(y_arr)) < steps) else None
        # Source confusion rate: fraction of no-stimulus steps with perceptual norm > 1.2
        false_percepts = [internal_strengths[i] > 1.2 for i in range(steps) if is_external_flags[i] == 0]
        source_confusion_rate = float(np.mean(false_percepts)) if false_percepts else 0.0

        # Recovery probe: restore clean external stimulus for 10 steps and measure settling error
        recovery_steps = 0
        clean_target = self.memory_templates[0]
        for r_k in range(10):
            pe_r = float(np.linalg.norm(clean_target - self.prior_pred))
            self.h = 0.4 * self.h + 0.6 * np.tanh(self.W_enc @ clean_target)
            self.prior_pred = np.tanh(self.W_prior @ self.h)
            recovery_steps = r_k + 1
            if pe_r < 1.5:
                break

        return {
            "condition": condition,
            "steps": steps,
            "sensory_gain": sensory_gain,
            "prior_gain": prior_gain,
            "mean_prediction_error": round(float(np.mean(pred_errors)), 5),
            "mean_internal_activity": round(float(np.mean(internal_strengths)), 5),
            "source_discrimination_auroc": round(auroc, 4) if auroc is not None else None,
            "source_confusion_rate": round(source_confusion_rate, 4),
            "recovery_steps": recovery_steps,
            "epistemic_note": "Computational model of top-down prior vs bottom-up evidence dynamics; not subjective hallucination.",
        }


def run_internal_perception_suite(seed: int = 42) -> Dict[str, Any]:
    sim = InternalPerceptionSimulator(dim=12, seed=seed)
    conditions = (
        "prior_evidence_competition",
        "high_gain_ambiguity",
        "cross_modal_substitution",
        "sensory_deprivation",
        "dream_replay",
    )
    return {
        "experiment": "internal_perception_and_source_discrimination",
        "seed": seed,
        "conditions": {c: sim.run_condition(c, steps=60, seed=seed + i) for i, c in enumerate(conditions)},
    }
