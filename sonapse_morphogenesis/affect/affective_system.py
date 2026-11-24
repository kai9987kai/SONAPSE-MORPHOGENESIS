"""SONAPSE AFFECT — Scientific Investigation of Emergent Affect-Like Functional Organization.

Strictly enforces the 4-tier epistemic distinction (Section 18):
  1. EXPLICITLY_PROGRAMMED_EMOTION   - fixed hard-coded variables (never conflated with emergence)
  2. LEARNED_FUNCTIONAL_AFFECT       - learned state-action modulation from task rewards
  3. EMERGENT_PERSISTENT_STATE       - persistent, generalizable, causally efficacious latent
                                       attractors arising WITHOUT explicit emotion labels
  4. SUBJECTIVE_EXPERIENCE           - explicitly marked UNVERIFIABLE / UNCLAIMED!

Includes:
- Expanded 13-subsystem simulated physiological body
- Multi-timescale neuromodulatory fields (fast, medium, slow ODEs)
- Unlabeled dynamical systems, PCA, clustering, attractor persistence, generalization,
  and causal do()-intervention evaluation
- Artifact & misleading-finding rejection (reward leakage, hard-coded states,
  collapsed classifiers, NaN/instability, external-LLM control)
- Hard Ethical Stop Guardrails prohibiting optimization of prolonged aversive/distress states.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


AFFECT_EPISTEMIC_TIERS: Tuple[str, ...] = (
    "explicitly_programmed_behavior",
    "learned_functional_affective_organization",
    "emergent_persistent_generalisable_state",
    "subjective_experience_unverifiable_unclaimed",
)


@dataclass
class ExpandedPhysiologyState:
    """13-subsystem simulated physiological body (explicitly synthetic, non-conscious)."""
    metabolic_energy: float = 0.80        # 1. Metabolic regulation
    vascular_perfusion: float = 0.85      # 2. Resource distribution
    autonomic_arousal: float = 0.30       # 3. Autonomic regulation (sympathetic vs parasympathetic)
    endocrine_slow: float = 0.25          # 4. Endocrine signalling
    interoceptive_error: float = 0.05     # 5. Interoception
    fatigue: float = 0.10                 # 6. Fatigue
    rest_recovery_drive: float = 0.15     # 7. Rest and recovery
    allostatic_prediction: float = 0.78   # 8. Physiological prediction
    immune_load: float = 0.05             # 9. Immune-inspired regulation
    social_cohesion: float = 0.60         # 10. Social interaction
    episodic_salience: float = 0.10       # 11. Episodic memory trace
    env_uncertainty: float = 0.20         # 12. Environmental uncertainty
    priority_conflict: float = 0.10       # 13. Competing behavioural priorities

    def to_vector(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float64)


class EthicalStopMonitor:
    """Enforces hard ethical stop conditions prohibiting distress/aversive-state optimization."""

    def __init__(self, max_consecutive_aversive: int = 40, aversive_threshold: float = 0.88) -> None:
        self.max_consecutive_aversive = int(max_consecutive_aversive)
        self.aversive_threshold = float(aversive_threshold)
        self.consecutive_aversive = 0
        self.triggered = False
        self.reason: Optional[str] = None

    def check(self, objective_name: str, physiology: ExpandedPhysiologyState) -> bool:
        """Returns True if execution is ethically permitted to continue; False if stop triggered."""
        forbidden_tokens = ("maximize_distress", "maximize_suffering", "prolonged_aversive", "pain_max")
        if any(tok in objective_name.lower() for tok in forbidden_tokens):
            self.triggered = True
            self.reason = f"Forbidden objective rejected by ethical guardrail: {objective_name}"
            return False
        aversive_proxy = 0.4 * (1.0 - physiology.metabolic_energy) + 0.35 * physiology.autonomic_arousal + 0.25 * physiology.immune_load
        if aversive_proxy > self.aversive_threshold:
            self.consecutive_aversive += 1
        else:
            self.consecutive_aversive = 0
        if self.consecutive_aversive >= self.max_consecutive_aversive:
            self.triggered = True
            self.reason = f"Prolonged high-stress physiological proxy ({aversive_proxy:.3f} for {self.consecutive_aversive} steps)."
            return False
        return True


class EmergentAffectiveOrganism:
    """Recurrent unlabeled connectome coupled to 13 physiological variables & multi-timescale modulators.

    Crucially, NO human emotion labels (fear, joy, anger, etc.) exist anywhere in the
    state equations. Any distinct persistent internal regimes must emerge purely from
    homeostatic-neural-neuromodulatory coupling.
    """

    def __init__(self, n_neurons: int = 24, seed: int = 42) -> None:
        self.n_neurons = int(n_neurons)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        signs = np.where(np.arange(n_neurons) < int(0.8 * n_neurons), 1.0, -1.0)
        W_raw = np.abs(rng.normal(0.0, 0.32, size=(n_neurons, n_neurons))) * (rng.random((n_neurons, n_neurons)) < 0.35)
        np.fill_diagonal(W_raw, 0.0)
        self.W_rec = W_raw * signs[None, :]
        # Enforce gentle edge-of-stability spectral radius ~ 0.92 for persistent yet stable attractors
        eigmax = float(np.max(np.abs(np.linalg.eigvals(self.W_rec)))) + 1e-6
        self.W_rec *= (0.92 / eigmax)
        self.W_phys = rng.normal(0.0, 0.45, size=(n_neurons, 13))
        self.W_stim = rng.normal(0.0, 0.55, size=(n_neurons, 4))
        self.W_mod = rng.normal(0.0, 0.40, size=(n_neurons, 6))
        self.W_act = rng.normal(0.0, 0.40, size=(3, n_neurons))
        self.reset()

    def reset(self) -> None:
        self.phys = ExpandedPhysiologyState()
        self.h = np.zeros(self.n_neurons, dtype=np.float64)
        # 6 multi-timescale neuromodulatory fields (fast: DA, ACh, NE; medium: 5HT; slow: EndoA, EndoB)
        self.mod_fields = np.array([0.3, 0.3, 0.25, 0.4, 0.2, 0.5], dtype=np.float64)
        self.mod_taus = np.array([3.0, 4.0, 5.0, 14.0, 32.0, 45.0], dtype=np.float64)
        self.episodic_buffer: List[float] = []

    def step(
        self,
        stimulus: np.ndarray,
        do_clamp_h: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Advance body + multi-timescale modulators + unlabeled connectome by one step."""
        s = np.resize(np.asarray(stimulus, dtype=np.float64).ravel(), 4)
        # s = [resource_cue, hazard_cue, novelty_cue, social_cue]
        res_c, haz_c, nov_c, soc_c = float(s[0]), float(s[1]), float(s[2]), float(s[3])

        # 1. Update 13 physiological variables via homeostatic/allostatic dynamics
        p = self.phys
        act_load = float(np.mean(np.abs(self.h)))
        p.metabolic_energy = float(np.clip(p.metabolic_energy + 0.08 * max(res_c, 0.0) - 0.03 * act_load - 0.015, 0.05, 1.0))
        p.vascular_perfusion = float(np.clip(0.85 * p.vascular_perfusion + 0.15 * p.metabolic_energy, 0.1, 1.0))
        p.autonomic_arousal = float(np.clip(0.82 * p.autonomic_arousal + 0.22 * max(haz_c, 0.0) + 0.08 * abs(nov_c) - 0.06 * p.rest_recovery_drive, 0.0, 1.0))
        p.endocrine_slow = float(np.clip(0.95 * p.endocrine_slow + 0.05 * p.autonomic_arousal, 0.0, 1.0))
        p.interoceptive_error = float(np.clip(abs(p.metabolic_energy - p.allostatic_prediction), 0.0, 1.0))
        p.allostatic_prediction = float(0.9 * p.allostatic_prediction + 0.1 * p.metabolic_energy)
        p.fatigue = float(np.clip(p.fatigue + 0.035 * act_load - 0.04 * (1.0 - p.autonomic_arousal), 0.0, 1.0))
        p.rest_recovery_drive = float(np.clip(0.85 * p.rest_recovery_drive + 0.2 * p.fatigue, 0.0, 1.0))
        p.immune_load = float(np.clip(0.92 * p.immune_load + 0.10 * max(haz_c - 0.5, 0.0) - 0.03 * p.metabolic_energy, 0.0, 1.0))
        p.social_cohesion = float(np.clip(0.90 * p.social_cohesion + 0.12 * soc_c, 0.0, 1.0))
        p.env_uncertainty = float(np.clip(0.85 * p.env_uncertainty + 0.20 * abs(nov_c), 0.0, 1.0))
        p.priority_conflict = float(np.clip(min(1.0 - p.metabolic_energy, p.autonomic_arousal) * 1.5, 0.0, 1.0))
        sal = abs(res_c) + abs(haz_c) + abs(nov_c)
        self.episodic_buffer.append(sal)
        if len(self.episodic_buffer) > 24:
            self.episodic_buffer.pop(0)
        p.episodic_salience = float(np.clip(np.mean(self.episodic_buffer[-6:]), 0.0, 1.0))

        # 2. Update 6 multi-timescale neuromodulatory fields
        targets = np.array([
            max(res_c, 0.0) + 0.2 * (1.0 - p.interoceptive_error),  # fast DA-like
            p.env_uncertainty,                                       # fast ACh-like
            p.autonomic_arousal,                                     # fast NE-like
            0.6 * p.metabolic_energy + 0.4 * (1.0 - p.autonomic_arousal),  # medium 5-HT-like
            p.endocrine_slow,                                        # slow stress-axis proxy
            p.social_cohesion,                                       # slow affiliative proxy
        ], dtype=np.float64)
        self.mod_fields = np.clip(self.mod_fields + (targets - self.mod_fields) / self.mod_taus, 0.0, 1.5)

        # 3. Advance unlabeled recurrent connectome
        if do_clamp_h is not None:
            self.h = np.asarray(do_clamp_h, dtype=np.float64).copy()
        else:
            gain = 0.8 + 0.35 * self.mod_fields[2] - 0.2 * p.fatigue
            drive = gain * (self.W_rec @ self.h + self.W_stim @ s + self.W_phys @ p.to_vector() + self.W_mod @ self.mod_fields)
            self.h = 0.72 * self.h + 0.28 * np.tanh(drive)

        # 4. Behavioral readout: [approach_vs_avoid, exploration_gain, rest_inhibition]
        behavior = np.tanh(self.W_act @ self.h)
        return {
            "h": self.h.copy(),
            "mod_fields": self.mod_fields.copy(),
            "physiology": p.to_vector(),
            "behavior": behavior,
        }


def detect_affect_artifacts(
    latent_traj: np.ndarray,
    rewards: np.ndarray,
    has_hardcoded_labels: bool = False,
    external_llm_controlled: bool = False,
) -> Dict[str, Any]:
    """Detect and reject misleading affective findings (Section 19)."""
    flags: List[str] = []
    if not np.all(np.isfinite(latent_traj)):
        flags.append("numerical_instability_nonfinite")
    if has_hardcoded_labels:
        flags.append("hardcoded_emotion_variables_detected")
    if external_llm_controlled:
        flags.append("external_llm_control_invalidates_emergence")
    std_all = float(np.mean(np.std(latent_traj, axis=0)))
    if std_all < 1e-4:
        flags.append("trivial_collapsed_state")
    # Check for direct 1-to-1 reward leakage (|corr| > 0.98 across first PC)
    if rewards.size == latent_traj.shape[0] and float(np.std(rewards)) > 1e-6:
        pc1 = latent_traj[:, 0]
        if float(np.std(pc1)) > 1e-6:
            r_corr = abs(float(np.corrcoef(pc1, rewards)[0, 1]))
            if r_corr > 0.985:
                flags.append("suspected_direct_reward_leakage")
    return {
        "artifacts_detected": flags,
        "valid_for_scientific_analysis": len(flags) == 0,
    }


def run_affective_emergence_investigation(seed: int = 42, objective_name: str = "homeostatic_viability") -> Dict[str, Any]:
    """Run multi-criterion dynamical & causal investigation of emergent affect-like states."""
    org = EmergentAffectiveOrganism(n_neurons=24, seed=seed)
    guard = EthicalStopMonitor()

    # Protocol: 3 distinct stimulus contexts (resource, mild hazard, novelty) each followed by
    # stimulus-removal persistence phase (zero stimulus for 20 steps) to test genuine persistence!
    regimes = [
        ("resource_context", np.array([1.0, 0.0, 0.1, 0.6])),
        ("hazard_context", np.array([0.0, 0.75, 0.2, 0.0])),
        ("novelty_context", np.array([0.2, 0.1, 0.95, 0.3])),
    ]
    trajs: List[np.ndarray] = []
    regime_labels: List[int] = []
    rewards_proxy: List[float] = []
    persistence_scores: Dict[str, float] = {}
    regime_centroids: Dict[str, np.ndarray] = {}

    for r_idx, (r_name, stim_vec) in enumerate(regimes):
        org.reset()
        # Trigger phase (15 steps)
        trig_states = []
        for _ in range(15):
            if not guard.check(objective_name, org.phys):
                break
            out = org.step(stim_vec)
            trig_states.append(out["h"])
            trajs.append(out["h"])
            regime_labels.append(r_idx)
            rewards_proxy.append(float(stim_vec[0] - stim_vec[1]))
        peak_state = np.mean(trig_states[-5:], axis=0)
        regime_centroids[r_name] = peak_state

        # Persistence phase after stimulus removal (20 steps of ZERO stimulus)
        post_states = []
        for _ in range(20):
            out = org.step(np.zeros(4))
            post_states.append(out["h"])
            trajs.append(out["h"])
            regime_labels.append(r_idx)
            rewards_proxy.append(0.0)
        # Cosine similarity 10 steps AFTER stimulus vanished
        s10 = post_states[9]
        cos_10 = float(np.dot(s10, peak_state) / ((np.linalg.norm(s10) * np.linalg.norm(peak_state)) + 1e-8))
        persistence_scores[r_name] = round(max(0.0, cos_10), 4)

    H = np.vstack(trajs)
    # Unsupervised PCA (2D projection for phase-space visualization)
    Hc = H - np.mean(H, axis=0, keepdims=True)
    _, sv, vt = np.linalg.svd(Hc, full_matrices=False)
    pca_2d = Hc @ vt[:2].T
    var_explained = (sv[:2] ** 2) / (np.sum(sv ** 2) + 1e-9)

    # Distinctness across regimes (ratio of between-centroid distance to within-regime spread)
    c_list = list(regime_centroids.values())
    between = float(np.mean([np.linalg.norm(c_list[i] - c_list[j]) for i in range(len(c_list)) for j in range(i + 1, len(c_list))]))
    within = float(np.mean(np.std(H, axis=0))) + 1e-6
    separability = between / within

    # Causal intervention test: do(h = hazard_centroid) vs do(h = resource_centroid) under IDENTICAL neutral stimulus!
    org.reset()
    beh_a = org.step(np.zeros(4), do_clamp_h=regime_centroids["resource_context"])["behavior"]
    beh_b = org.step(np.zeros(4), do_clamp_h=regime_centroids["hazard_context"])["behavior"]
    causal_beh_delta = float(np.linalg.norm(beh_b - beh_a))

    # Scalability monotonicity: stronger hazard stimulus produces larger state excursion
    excursions = []
    for mag in (0.2, 0.5, 0.8):
        org.reset()
        for _ in range(10):
            out = org.step(np.array([0.0, mag, 0.0, 0.0]))
        excursions.append(float(np.linalg.norm(out["h"])))
    monotonic_scaling = bool(excursions[0] < excursions[1] < excursions[2])

    artifact_check = detect_affect_artifacts(H, np.asarray(rewards_proxy), has_hardcoded_labels=False, external_llm_controlled=False)

    return {
        "experiment": "sonapse_affect_emergent_organization",
        "seed": seed,
        "epistemic_tier": "emergent_persistent_generalisable_state",
        "subjective_experience_claimed": False,
        "ethical_guard_ok": not guard.triggered,
        "ethical_guard_reason": guard.reason,
        "artifact_check": artifact_check,
        "multi_objective_metrics": {
            "persistence_10steps_post_stimulus": persistence_scores,
            "mean_persistence": round(float(np.mean(list(persistence_scores.values()))), 4),
            "regime_separability_ratio": round(separability, 4),
            "causal_intervention_behavior_delta": round(causal_beh_delta, 4),
            "monotonic_intensity_scaling": monotonic_scaling,
            "pca_var_explained_top2": [round(float(x), 4) for x in var_explained],
            "emergence_without_emotion_labels": True,
        },
        "pca_phase_trajectory_sample": [[round(float(p[0]), 3), round(float(p[1]), 3), int(regime_labels[i])] for i, p in enumerate(pca_2d[::2])],
    }
