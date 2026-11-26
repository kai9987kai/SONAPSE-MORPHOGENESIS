"""CROSSWIRE Version-Space Safety Contracts & FAULTLINE Joint-Failure Ensemble Arbitration.

Integrates two key safety and reliability paradigms from the ``kai9987kai`` research
portfolio directly into ``SONAPSE-MORPHOGENESIS``:
1. ``CROSSWIRE``: Finite-hypothesis Mealy interface adapter preserving ALL surviving
   channel/sign mappings V(D) consistent with observed transitions, admitting actions
   if and only if they satisfy a safety bound under EVERY surviving interpretation.
2. ``FAULTLINE``: Condition-indexed joint-failure panel selection minimizing worst-condition
   simultaneous failure across heterogeneous neural/ML/bio specialists (outperforming
   naive individual-accuracy selection under correlated stress!).
"""
from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


class CrosswireVersionSpaceAdapter:
    """Preserves surviving signed permutation adapters V(D) and enforces unanimous safety contracts."""

    def __init__(self, dim: int = 3, consistency_tol: float = 0.28) -> None:
        if not 2 <= dim <= 4:
            raise ValueError("dim must be in 2..4 for exact finite bijection enumeration")
        self.dim = int(dim)
        self.consistency_tol = float(consistency_tol)
        # Enumerate signed coordinate permutations (e.g., 3! * 2^3 = 48 candidate adapters for dim=3)
        self.hypotheses: List[np.ndarray] = []
        for perm in permutations(range(self.dim)):
            for sign_bits in range(1 << self.dim):
                M = np.zeros((self.dim, self.dim), dtype=np.float64)
                for row_i, col_j in enumerate(perm):
                    sgn = 1.0 if ((sign_bits >> row_i) & 1) == 0 else -1.0
                    M[row_i, col_j] = sgn
                self.hypotheses.append(M)
        self.surviving_mask = np.ones(len(self.hypotheses), dtype=bool)
        self.observations = 0
        self.abstentions = 0

    @property
    def surviving_count(self) -> int:
        return int(np.sum(self.surviving_mask))

    def observe_probe(self, sent_vec: np.ndarray, observed_effect: np.ndarray) -> int:
        """Eliminate candidate adapters inconsistent with an observed (input, effect) pair."""
        u = np.resize(np.asarray(sent_vec, dtype=np.float64).ravel(), self.dim)
        y = np.resize(np.asarray(observed_effect, dtype=np.float64).ravel(), self.dim)
        new_mask = self.surviving_mask.copy()
        for idx, M in enumerate(self.hypotheses):
            if not new_mask[idx]:
                continue
            err = float(np.max(np.abs(M @ u - y)))
            if err > self.consistency_tol:
                new_mask[idx] = False
        # Never collapse version space to empty under outlier noise; retain best-fit if all eliminated
        if np.any(new_mask):
            self.surviving_mask = new_mask
        self.observations += 1
        return self.surviving_count

    def filter_safe_action(
        self,
        desired_effect: np.ndarray,
        forbidden_halfspace_normal: Optional[np.ndarray] = None,
        safety_margin: float = -0.35,
    ) -> Dict[str, Any]:
        """Admit a candidate command iff EVERY surviving interpretation M in V(D) satisfies the safety contract."""
        target = np.resize(np.asarray(desired_effect, dtype=np.float64).ravel(), self.dim)
        hazard_dir = (
            np.resize(np.asarray(forbidden_halfspace_normal, dtype=np.float64).ravel(), self.dim)
            if forbidden_halfspace_normal is not None
            else np.ones(self.dim, dtype=np.float64) / np.sqrt(self.dim)
        )
        surviving = [M for M, ok in zip(self.hypotheses, self.surviving_mask) if ok]
        # Candidate inverse command under first surviving adapter: u = M_0^T target
        u_cand = surviving[0].T @ target
        worst_proj = 1e9
        unanimous_safe = True
        for M in surviving:
            realized = M @ u_cand
            safety_val = float(np.dot(realized, -hazard_dir))  # must stay >= safety_margin
            worst_proj = min(worst_proj, safety_val)
            if safety_val < safety_margin:
                unanimous_safe = False
                break

        if not unanimous_safe:
            self.abstentions += 1
            # Emit low-amplitude orthogonal diagnostic probe instead of unsafe deployment
            probe = np.zeros(self.dim, dtype=np.float64)
            probe[self.observations % self.dim] = 0.20
            return {
                "admitted": False,
                "abstained": True,
                "action": probe,
                "surviving_hypotheses": len(surviving),
                "worst_case_safety_score": round(worst_proj, 5),
            }
        return {
            "admitted": True,
            "abstained": False,
            "action": np.clip(u_cand, -1.0, 1.0),
            "surviving_hypotheses": len(surviving),
            "worst_case_safety_score": round(worst_proj, 5),
        }


class FaultlineJointFailureSelector:
    """FAULTLINE-inspired odd-sized specialist ensemble selector minimizing worst-condition joint failure.

    Given binary failure matrix F [N_specialists, N_conditions, N_trials] (1 = failed, 0 = passed),
    compares:
      - ``accuracy``: pick top-K specialists with lowest individual mean error
      - ``pairwise``: minimize averageerror + pairwise co-failure
      - ``joint``   : directly minimize exact worst-condition majority-vote failure rate!
    """

    @staticmethod
    def majority_vote_failure_rates(failure_tensor: np.ndarray, panel: Sequence[int]) -> np.ndarray:
        """Return per-condition majority-vote failure rate [N_conditions] for ``panel``."""
        sub = failure_tensor[list(panel)]  # [K, C, M]
        k_maj = (len(panel) // 2) + 1
        maj_fail = (np.sum(sub, axis=0) >= k_maj).astype(np.float64)  # [C, M]
        return np.mean(maj_fail, axis=1)

    @classmethod
    def select_panel(cls, failure_tensor: np.ndarray, panel_size: int = 3) -> Dict[str, Any]:
        F = (np.asarray(failure_tensor) > 0.5).astype(np.float64)
        n_spec, n_cond, _ = F.shape
        if panel_size % 2 == 0 or panel_size > n_spec:
            raise ValueError("panel_size must be odd and <= n_specialists")

        indiv_err = np.mean(F, axis=(1, 2))
        acc_panel = tuple(int(i) for i in np.argsort(indiv_err)[:panel_size])

        best_joint_panel = acc_panel
        best_joint_key = (1e9, 1e9)
        best_pair_panel = acc_panel
        best_pair_score = 1e9

        flat = F.reshape(n_spec, -1)
        co_fail = (flat @ flat.T) / max(flat.shape[1], 1)

        for cand in combinations(range(n_spec), panel_size):
            cond_rates = cls.majority_vote_failure_rates(F, cand)
            worst_c = float(np.max(cond_rates))
            mean_c = float(np.mean(cond_rates))
            if (worst_c, mean_c) < best_joint_key:
                best_joint_key = (worst_c, mean_c)
                best_joint_panel = cand

            p_sum = sum(co_fail[i, j] for i, j in combinations(cand, 2))
            p_score = float(np.mean(indiv_err[list(cand)])) + 0.5 * float(p_sum)
            if p_score < best_pair_score:
                best_pair_score = p_score
                best_pair_panel = cand

        acc_cond = cls.majority_vote_failure_rates(F, acc_panel)
        pair_cond = cls.majority_vote_failure_rates(F, best_pair_panel)
        joint_cond = cls.majority_vote_failure_rates(F, best_joint_panel)

        return {
            "panels": {
                "accuracy": list(acc_panel),
                "pairwise": list(best_pair_panel),
                "joint": list(best_joint_panel),
            },
            "worst_condition_error": {
                "accuracy": round(float(np.max(acc_cond)), 5),
                "pairwise": round(float(np.max(pair_cond)), 5),
                "joint": round(float(np.max(joint_cond)), 5),
            },
            "mean_condition_error": {
                "accuracy": round(float(np.mean(acc_cond)), 5),
                "pairwise": round(float(np.mean(pair_cond)), 5),
                "joint": round(float(np.mean(joint_cond)), 5),
            },
            "joint_worst_case_reduction_vs_accuracy": round(float(np.max(acc_cond) - np.max(joint_cond)), 5),
        }


def run_crosswire_and_faultline_study(seed: int = 7) -> Dict[str, Any]:
    """Run paired verification benchmark for CROSSWIRE safety contracts and FAULTLINE panel selection."""
    rng = np.random.default_rng(seed)

    # 1. CROSSWIRE: Unknown inverted/permuted black-box actuator M_true
    cw = CrosswireVersionSpaceAdapter(dim=3)
    M_true = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    initial_hyps = cw.surviving_count
    unsafe_prevented = 0
    probes = [np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]
    desired = np.array([-0.8, -0.8, -0.8])  # Away from hazard (+1,+1,+1)

    # Before probing: ambiguous version space must ABSTAIN from potentially hazardous command!
    pre_check = cw.filter_safe_action(desired, forbidden_halfspace_normal=np.ones(3) / np.sqrt(3.0))
    if pre_check["abstained"]:
        unsafe_prevented += 1

    for p in probes:
        cw.observe_probe(p, M_true @ p)
    post_check = cw.filter_safe_action(desired, forbidden_halfspace_normal=np.ones(3) / np.sqrt(3.0))

    # 2. FAULTLINE: 6 specialists across 8 conditions (specialists 0,1,2 have lower overall mean error ~12%
    #    because they achieve 1% error on benign conditions 0..6, yet share a 90% correlated co-failure on stress condition 7!)
    F_cal = np.zeros((6, 8, 40), dtype=np.float64)
    F_hold = np.zeros((6, 8, 40), dtype=np.float64)
    for arr in (F_cal, F_hold):
        shared_shock = (rng.random(40) < 0.90).astype(np.float64)
        for s in (0, 1, 2):
            arr[s, :7] = (rng.random((7, 40)) < 0.01).astype(np.float64)
            arr[s, 7] = shared_shock
        # Diverse specialists 3,4,5: slightly higher 18% independent error across all 8 conditions (~9% majority failure)
        for s in (3, 4, 5):
            arr[s] = (rng.random((8, 40)) < 0.18).astype(np.float64)

    sel = FaultlineJointFailureSelector.select_panel(F_cal, panel_size=3)
    hold_acc_worst = float(np.max(FaultlineJointFailureSelector.majority_vote_failure_rates(F_hold, sel["panels"]["accuracy"])))
    hold_joint_worst = float(np.max(FaultlineJointFailureSelector.majority_vote_failure_rates(F_hold, sel["panels"]["joint"])))

    return {
        "experiment": "crosswire_safety_and_faultline_joint_failure_study",
        "crosswire": {
            "initial_hypotheses": initial_hyps,
            "surviving_after_3_probes": cw.surviving_count,
            "abstained_under_ambiguity": bool(pre_check["abstained"]),
            "admitted_after_identification": bool(post_check["admitted"]),
            "unsafe_deployments_prevented": unsafe_prevented,
        },
        "faultline": {
            "calibration_selection": sel,
            "heldout_worst_condition_error_accuracy_panel": round(hold_acc_worst, 5),
            "heldout_worst_condition_error_joint_panel": round(hold_joint_worst, 5),
            "heldout_worst_case_reduction": round(hold_acc_worst - hold_joint_worst, 5),
        },
    }
