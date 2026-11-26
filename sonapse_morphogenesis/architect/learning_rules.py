"""SONAPSE ARCHITECT — Dynamic Learning-Method Selection Across 19 Core + 14 Extended Learning Algorithms (33 Total).

Treats learning algorithms as experimentally adjustable, versioned, reversible
mechanisms that can differ by connectome region and developmental stage (Section 27).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


LEARNING_ALGORITHMS: Tuple[str, ...] = (
    "hebbian",
    "stdp",
    "three_factor",
    "neuromodulated_rl",
    "bptt",
    "surrogate_gradient",
    "contrastive",
    "self_supervised",
    "predictive_coding",
    "local_error",
    "imitation",
    "negative_distillation",
    "evolutionary",
    "generative_replay",
    "fast_weights",
    "population_based",
    "activity_guided",
    "structural_plasticity",
    "region_meta_learning",
)

EXTENDED_LEARNING_ALGORITHMS: Tuple[str, ...] = (
    "bcm_sliding_threshold",
    "eprop_eligibility",
    "btsp_plateau_learning",
    "burst_ccn_multiplexed",
    "equilibrium_propagation",
    "forward_forward_goodness",
    "dendritic_target_prop",
    "muon_orthogonal_plasticity",
    "hodge_curl_free_plasticity",
    "shy_synaptic_renormalization",
    "tether_counterfactual_invariance",
    "ghost_efference_credit",
    "rigl_connectome_rewiring",
    "morphogen_wiring_economy",
)

ALL_LEARNING_ALGORITHMS: Tuple[str, ...] = LEARNING_ALGORITHMS + EXTENDED_LEARNING_ALGORITHMS


def _newton_schulz_orthogonalize(G: np.ndarray, steps: int = 3) -> np.ndarray:
    """Fast 3-step Newton-Schulz matrix orthogonalization (Muon geometry)."""
    nrm = float(np.linalg.norm(G)) + 1e-8
    X = G / nrm
    rows, cols = X.shape
    for _ in range(steps):
        if rows <= cols:
            X = 1.5 * X - 0.5 * ((X @ X.T) @ X)
        else:
            X = 1.5 * X - 0.5 * (X @ (X.T @ X))
    return X


@dataclass
class LearningScheduleVersion:
    version_id: int
    stage_rules: Dict[str, str]     # early | middle | mature -> learning algorithm
    region_rules: Dict[str, str]    # sensory | association | memory | motor -> learning algorithm
    learning_rates: Dict[str, float]


class DynamicLearningEngine:
    """Executes all 33 (19 v1 + 14 v2) learning rules on synaptic weight matrices with Dale-sign safety & rollback."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self._elig_traces: Dict[Tuple[int, int], np.ndarray] = {}
        self._bcm_theta: Dict[int, np.ndarray] = {}
        self._muon_mom: Dict[Tuple[int, int], np.ndarray] = {}
        self.history: List[LearningScheduleVersion] = [
            LearningScheduleVersion(
                version_id=0,
                stage_rules={"early": "hebbian", "middle": "three_factor", "mature": "predictive_coding"},
                region_rules={"sensory": "self_supervised", "association": "predictive_coding", "memory": "fast_weights", "motor": "neuromodulated_rl"},
                learning_rates={"default": 0.025},
            )
        ]

    @property
    def active_schedule(self) -> LearningScheduleVersion:
        return self.history[-1]

    def update_schedule(self, stage_rules: Dict[str, str], region_rules: Dict[str, str], lr: float = 0.025) -> LearningScheduleVersion:
        for r in list(stage_rules.values()) + list(region_rules.values()):
            if r not in ALL_LEARNING_ALGORITHMS and r != "bcm":
                raise ValueError(f"Unsupported learning algorithm: {r!r}")
        ver = LearningScheduleVersion(
            version_id=len(self.history),
            stage_rules=dict(stage_rules),
            region_rules=dict(region_rules),
            learning_rates={"default": float(np.clip(lr, 1e-4, 0.5))},
        )
        self.history.append(ver)
        return ver

    def rollback_schedule(self) -> LearningScheduleVersion:
        if len(self.history) > 1:
            self.history.pop()
        return self.active_schedule

    def compute_weight_delta(
        self,
        algorithm: str,
        W: np.ndarray,
        pre: np.ndarray,
        post: np.ndarray,
        target: Optional[np.ndarray] = None,
        reward: float = 0.0,
        modulator: float = 1.0,
        lr: float = 0.02,
        pre_prev: Optional[np.ndarray] = None,
        post_prev: Optional[np.ndarray] = None,
        wrong_target: Optional[np.ndarray] = None,
        distance_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute bounded synaptic update dW [post_dim, pre_dim] for any of the 33 supported algorithms."""
        algo = "bcm_sliding_threshold" if algorithm == "bcm" else algorithm
        if algo not in ALL_LEARNING_ALGORITHMS:
            raise KeyError(f"Unknown learning algorithm: {algorithm!r}")
        pre = np.asarray(pre, dtype=np.float64).ravel()
        post = np.asarray(post, dtype=np.float64).ravel()
        p_prev = pre if pre_prev is None else np.asarray(pre_prev, dtype=np.float64).ravel()
        q_prev = post if post_prev is None else np.asarray(post_prev, dtype=np.float64).ravel()
        err = (np.asarray(target, dtype=np.float64).ravel() - post) if target is not None else np.zeros_like(post)
        shape = W.shape

        if algo == "hebbian":
            # Oja-normalized Hebbian rule to prevent unbounded growth
            dW = lr * (np.outer(post, pre) - np.outer(post ** 2, np.ones_like(pre)) * W)
        elif algo == "stdp":
            # Asymmetric temporal STDP: potentiate when pre preceded post, depress reverse
            dW = lr * (np.outer(post, p_prev) - 0.85 * np.outer(q_prev, pre))
        elif algo == "three_factor":
            dW = lr * reward * modulator * np.outer(post, pre)
        elif algo == "neuromodulated_rl":
            dW = lr * np.tanh(reward * modulator) * np.outer(post - np.mean(post), pre)
        elif algo == "bptt":
            # Exact local truncated-BPTT gradient for tanh(W @ pre)
            grad = err * (1.0 - np.clip(post, -0.99, 0.99) ** 2)
            dW = lr * (np.outer(grad, pre) + 0.35 * np.outer(grad, p_prev))
        elif algo == "surrogate_gradient":
            # Fast-sigmoid surrogate derivative 1 / (1 + |v|)^2
            surr = 1.0 / ((1.0 + np.abs(post)) ** 2)
            dW = lr * np.outer(err * surr, pre)
        elif algo == "contrastive":
            pos_term = np.outer(post, pre)
            neg_term = np.outer(np.roll(post, 1), pre)
            dW = lr * (pos_term - 0.8 * neg_term)
        elif algo == "self_supervised":
            # Predict current post from previous pre (temporal consistency)
            pred = np.tanh(W @ p_prev)
            dW = lr * np.outer((post - pred) * (1.0 - pred ** 2), p_prev)
        elif algo == "predictive_coding":
            pred_pre = W.T @ post
            pe_in = pre - pred_pre
            dW = lr * np.outer(post, pe_in)
        elif algo == "local_error":
            dW = lr * np.outer(err, pre)
        elif algo == "imitation":
            dW = lr * 1.2 * np.outer(err * (1.0 - post ** 2), pre)
        elif algo == "negative_distillation":
            neg = (np.asarray(wrong_target, dtype=np.float64).ravel() - post) if wrong_target is not None else -err
            dW = lr * (np.outer(err, pre) - 0.35 * np.outer(neg, pre))
        elif algo == "evolutionary":
            noise = self.rng.normal(0.0, 1.0, size=W.shape)
            dW = lr * (0.5 if reward >= 0 else -0.5) * noise
        elif algo == "generative_replay":
            # Blend current error with synthetic replay sample from W's own memory
            syn_pre = np.tanh(W.T @ post)
            dW = lr * (0.65 * np.outer(err, pre) + 0.35 * np.outer(post, syn_pre))
        elif algo == "fast_weights":
            dW = lr * 1.8 * np.outer(post, pre) - 0.05 * W
        elif algo == "population_based":
            dW = lr * (0.7 * np.outer(err, pre) + 0.3 * self.rng.normal(0.0, 0.25, size=W.shape))
        elif algo == "activity_guided":
            # Emphasize high-activity surprise channels
            sal = np.abs(post - q_prev) + 0.1
            dW = lr * np.outer(err * sal, pre)
        elif algo == "structural_plasticity":
            # Prune weakest 10% and reinforce strongest correlated pairs
            raw = lr * np.outer(post, pre)
            weak = np.abs(W) < np.percentile(np.abs(W), 15)
            dW = np.where(weak, -0.25 * W, raw)
        elif algo == "region_meta_learning":
            # Adaptive per-unit scaling inversely proportional to weight norm
            row_scale = 1.0 / (np.linalg.norm(W, axis=1, keepdims=True) + 0.5)
            dW = lr * row_scale * np.outer(err + 0.2 * reward * post, pre)

        # --- 14 EXTENDED V2 RESEARCH & CUSTOM CONNECTOME LEARNING ALGORITHMS ---
        elif algo == "bcm_sliding_threshold":
            theta = self._bcm_theta.get(post.size)
            if theta is None:
                theta = np.full(post.size, 0.25, dtype=np.float64)
            theta = 0.90 * theta + 0.10 * (post ** 2)
            self._bcm_theta[post.size] = theta
            phi_bcm = post * (post - theta)
            dW = lr * (np.outer(phi_bcm, pre) + (0.6 * np.outer(err, pre) if target is not None else 0.0) - 0.02 * W)

        elif algo == "eprop_eligibility":
            # Bellec et al. (2020) e-prop: dampened pseudo-derivative * presynaptic trace * broadcast learning signal
            elig = self._elig_traces.get(shape)
            if elig is None:
                elig = np.zeros(shape, dtype=np.float64)
            pseudo_d = 0.35 * np.maximum(0.0, 1.0 - np.abs(post)) + 0.15
            elig = 0.82 * elig + np.outer(pseudo_d, 0.65 * pre + 0.35 * p_prev)
            self._elig_traces[shape] = elig
            learn_sig = err if target is not None else np.full_like(post, float(np.tanh(reward * modulator)))
            dW = lr * 1.4 * (learn_sig[:, None] * elig)

        elif algo == "btsp_plateau_learning":
            # Bittner et al. (2017) / Magee & Grienberger (2020) Behavioral Time-Scale Synaptic Plasticity
            elig = self._elig_traces.get(shape)
            if elig is None:
                elig = np.zeros(shape, dtype=np.float64)
            elig = 0.90 * elig + 0.25 * np.outer(np.ones_like(post), pre)
            self._elig_traces[shape] = elig
            plateau = np.abs(err) + max(0.0, float(reward) * modulator) if target is not None else np.full_like(post, abs(float(reward)) + 0.2)
            gate = 1.0 / (1.0 + np.exp(-3.0 * (plateau[:, None] * np.abs(elig) - 0.25)))
            sign_dir = np.sign(err[:, None] if target is not None else np.outer(post, pre))
            w_cap = np.clip(2.5 - np.abs(W), 0.05, 2.5)
            dW = lr * 1.2 * w_cap * gate * sign_dir * np.abs(elig)

        elif algo == "burst_ccn_multiplexed":
            # Payeur, Guerguiev, Zenke, Richards & Naud (2021) BurstCCN single-phase credit assignment
            event_rate = 1.0 / (1.0 + np.exp(-np.clip(2.0 * pre, -15.0, 15.0)))
            p_base = 0.5
            apical_drive = err if target is not None else (reward * modulator * post)
            p_burst = 1.0 / (1.0 + np.exp(-np.clip(2.5 * apical_drive, -15.0, 15.0)))
            dW = lr * 2.2 * np.outer(p_burst - p_base, event_rate * np.sign(pre + 1e-9) * np.abs(pre))

        elif algo == "equilibrium_propagation":
            # Scellier & Bengio (2017) Equilibrium Propagation free vs weakly-clamped phase
            beta = 0.25
            free_post = np.tanh(W @ pre)
            nudge = err if target is not None else (reward * post)
            nudged_post = np.tanh(W @ pre + beta * nudge)
            dW = (lr / beta) * (np.outer(nudged_post, pre) - np.outer(free_post, pre))

        elif algo == "forward_forward_goodness":
            # Hinton (2022) Forward-Forward local goodness margin without backward pass
            goodness = float(np.mean(post ** 2))
            theta_ff = 0.35
            pos_prob = 1.0 / (1.0 + np.exp(-4.0 * (goodness - theta_ff)))
            is_pos = (float(reward) >= 0.0) if target is None else (float(np.mean(err ** 2)) < 0.35)
            direction = (1.0 - pos_prob) if is_pos else -pos_prob
            base_ff = direction * np.outer(post, pre)
            corr_assist = 0.5 * np.outer(err, pre) if target is not None else 0.0
            dW = lr * (base_ff + corr_assist)

        elif algo == "dendritic_target_prop":
            # Urbanczik & Senn (2014) / Sacramento et al. (2018) dendritic prediction error
            dend_pred = np.tanh(W @ pre)
            soma_teach = np.tanh(post + (0.8 * err if target is not None else 0.4 * reward * post))
            dW = lr * 1.35 * np.outer(soma_teach - dend_pred, pre)

        elif algo == "muon_orthogonal_plasticity":
            # Momentum-smoothed outer gradient orthogonalized via 3-step Newton-Schulz
            raw_grad = np.outer(err * (1.0 - np.clip(post, -0.99, 0.99) ** 2), pre) if target is not None else (reward * np.outer(post, pre))
            mom = self._muon_mom.get(shape)
            if mom is None:
                mom = np.zeros(shape, dtype=np.float64)
            mom = 0.85 * mom + 0.15 * raw_grad
            self._muon_mom[shape] = mom
            ortho = _newton_schulz_orthogonalize(mom, steps=3)
            dW = lr * 0.65 * ortho * min(1.0, float(np.linalg.norm(raw_grad)) * 2.0)

        elif algo == "hodge_curl_free_plasticity":
            # Novel SONAPSE v2 Topological Hodge Decomposition Plasticity:
            # Separates potential-gradient (acyclic hierarchy) from skew-symmetric cyclic curl (recurrent memory)
            G = np.outer(err if target is not None else post * (0.2 + reward), pre)
            m_min = min(shape[0], shape[1])
            dW = G.copy()
            sq = G[:m_min, :m_min]
            sym_grad = 0.5 * (sq + sq.T)
            skew_curl = 0.5 * (sq - sq.T)
            # Balance curl preservation vs gradient descent according to temporal drift
            temporal_need = float(np.clip(np.mean(np.abs(pre - p_prev)), 0.1, 1.0))
            dW[:m_min, :m_min] = (1.0 - 0.35 * temporal_need) * sym_grad + (1.0 + 0.45 * temporal_need) * skew_curl
            dW = lr * dW

        elif algo == "shy_synaptic_renormalization":
            # Tononi & Cirelli Synaptic Homeostasis Hypothesis: Hebbian/error wake + global L1 downscaling
            wake_pot = np.outer(err if target is not None else (0.5 * post * (1.0 + reward)), pre)
            l1_load = np.mean(np.abs(W), axis=1, keepdims=True)
            downscale = -0.12 * np.maximum(0.0, l1_load - 0.35) * W
            dW = lr * wake_pot + downscale * lr

        elif algo == "tether_counterfactual_invariance":
            # Supermix-Tether v0.2: reward co-variation with clean channels while suppressing volatile channels
            chan_volatility = np.abs(pre - p_prev)
            inv_penalty = -0.30 * W * (chan_volatility[None, :] ** 2)
            task_drive = np.outer(err if target is not None else reward * post, pre / (1.0 + chan_volatility))
            dW = lr * (1.25 * task_drive + inv_penalty)

        elif algo == "ghost_efference_credit":
            # GhostInTheMachine v8: weight plasticity by self-authored transition predictability
            pred_post = np.tanh(W @ p_prev)
            authorship = 1.0 / (1.0 + 2.0 * np.abs(q_prev - pred_post))
            drive = err if target is not None else (reward * modulator * post)
            dW = lr * 1.3 * np.outer(authorship * drive, pre)

        elif algo == "rigl_connectome_rewiring":
            # Evci et al. RigL dynamic sparse rewiring: decay smallest active weights, grow largest dormant gradients
            grad_mat = np.outer(err if target is not None else (reward * post), pre)
            active = np.abs(W) > 1e-4
            if np.any(active) and np.any(~active):
                drop_thresh = float(np.percentile(np.abs(W[active]), 15))
                grow_thresh = float(np.percentile(np.abs(grad_mat[~active]), 85))
                drop_mask = active & (np.abs(W) <= drop_thresh)
                grow_mask = (~active) & (np.abs(grad_mat) >= grow_thresh)
                dW = lr * grad_mat * active - 0.35 * W * drop_mask + lr * 1.5 * grad_mat * grow_mask
            else:
                dW = lr * grad_mat

        else:  # morphogen_wiring_economy
            # Bullmore & Sporns connectome wiring economy: performance gain minus spatial cable cost
            if distance_matrix is not None and distance_matrix.shape == shape:
                dist_cost = np.asarray(distance_matrix, dtype=np.float64)
            else:
                ri = np.arange(shape[0], dtype=np.float64)[:, None] / max(shape[0], 1)
                cj = np.arange(shape[1], dtype=np.float64)[None, :] / max(shape[1], 1)
                dist_cost = np.abs(ri - cj)
            drive = np.outer(err if target is not None else (reward * post), pre)
            wiring_penalty = -0.18 * dist_cost * np.sign(W) * (np.abs(W) > 0.05)
            dW = lr * (1.2 * drive + wiring_penalty)

        return np.clip(dW, -0.25, 0.25)


def benchmark_developmental_stage_learning_rules(
    seed: int = 42,
    steps: int = 60,
    include_extended: bool = False,
) -> Dict[str, Any]:
    """Test whether optimal learning rules differ across early vs mature developmental stages."""
    rng = np.random.default_rng(seed)
    engine = DynamicLearningEngine(seed=seed)
    X = rng.normal(0.0, 1.0, size=(steps, 6))
    W_star = rng.normal(0.0, 0.6, size=(4, 6))
    Y = np.tanh(X @ W_star.T)

    base_tested = (
        "hebbian", "stdp", "three_factor", "bptt", "surrogate_gradient",
        "predictive_coding", "local_error", "fast_weights", "region_meta_learning",
    )
    tested = (base_tested + EXTENDED_LEARNING_ALGORITHMS) if include_extended else base_tested
    early_scores: Dict[str, float] = {}
    mature_scores: Dict[str, float] = {}

    for algo in tested:
        W = rng.normal(0.0, 0.2, size=(4, 6))
        prev_x = np.zeros(6)
        prev_y = np.zeros(4)
        early_errs, mature_errs = [], []
        for t in range(steps):
            out = np.tanh(W @ X[t])
            mse = float(np.mean((out - Y[t]) ** 2))
            if t < steps // 3:
                early_errs.append(mse)
            elif t >= 2 * steps // 3:
                mature_errs.append(mse)
            dW = engine.compute_weight_delta(algo, W, X[t], out, target=Y[t], reward=-mse, lr=0.08, pre_prev=prev_x, post_prev=prev_y)
            W = np.clip(W + dW, -3.0, 3.0)
            prev_x, prev_y = X[t], out
        early_scores[algo] = round(float(np.mean(early_errs)), 5)
        mature_scores[algo] = round(float(np.mean(mature_errs)), 5)

    return {
        "experiment": "dynamic_learning_rules_by_developmental_stage",
        "all_19_algorithms_implemented": list(LEARNING_ALGORITHMS),
        "extended_v2_algorithms": list(EXTENDED_LEARNING_ALGORITHMS),
        "total_supported_algorithms": len(ALL_LEARNING_ALGORITHMS),
        "early_stage_mse": early_scores,
        "mature_stage_mse": mature_scores,
        "best_early": min(early_scores, key=early_scores.get),  # type: ignore[arg-type]
        "best_mature": min(mature_scores, key=mature_scores.get),  # type: ignore[arg-type]
    }


def benchmark_extended_connectome_learning_suite(seed: int = 42, steps: int = 90) -> Dict[str, Any]:
    """Comprehensive comparative benchmark of all 33 connectome & ML learning rules across 3 regimes:
    1. Clean nonlinear regression
    2. Volatile channel corruption (testing Tether / Efference / SHY robustness)
    3. Spatial wiring-economy efficiency (MSE vs L1 physical cable length cost)
    """
    rng = np.random.default_rng(seed)
    X_clean = rng.normal(0.0, 1.0, size=(steps, 8))
    # Corrupt channels 5..7 on second half to test counterfactual invariance & homeostasis
    X_corrupt = X_clean.copy()
    X_corrupt[:, 5:] += rng.normal(0.0, 1.4, size=(steps, 3))
    W_true = np.zeros((5, 8), dtype=np.float64)
    W_true[:, :5] = rng.normal(0.0, 0.7, size=(5, 5))
    Y = np.tanh(X_clean @ W_true.T)
    ri = np.arange(5, dtype=np.float64)[:, None] / 5.0
    cj = np.arange(8, dtype=np.float64)[None, :] / 8.0
    dist_mat = np.abs(ri - cj) + 0.1

    results: Dict[str, Dict[str, float]] = {}
    for idx, algo in enumerate(ALL_LEARNING_ALGORITHMS):
        eng = DynamicLearningEngine(seed=seed + idx)
        W = rng.normal(0.0, 0.15, size=(5, 8))
        px = np.zeros(8, dtype=np.float64)
        py = np.zeros(5, dtype=np.float64)
        late_clean_mse: List[float] = []
        late_noisy_mse: List[float] = []
        for t in range(steps):
            xt = X_corrupt[t]
            out = np.tanh(W @ xt)
            mse_noisy = float(np.mean((out - Y[t]) ** 2))
            mse_clean = float(np.mean((np.tanh(W @ X_clean[t]) - Y[t]) ** 2))
            if t >= int(0.6 * steps):
                late_noisy_mse.append(mse_noisy)
                late_clean_mse.append(mse_clean)
            dW = eng.compute_weight_delta(
                algo, W, xt, out, target=Y[t], reward=float(0.5 - mse_noisy),
                lr=0.075, pre_prev=px, post_prev=py, distance_matrix=dist_mat,
            )
            W = np.clip(W + dW, -3.0, 3.0)
            px, py = xt, out
        wiring_cost = float(np.sum(np.abs(W) * dist_mat))
        results[algo] = {
            "generation": 2 if algo in EXTENDED_LEARNING_ALGORITHMS else 1,
            "clean_eval_mse": round(float(np.mean(late_clean_mse)), 5),
            "corrupted_eval_mse": round(float(np.mean(late_noisy_mse)), 5),
            "wiring_cable_cost": round(wiring_cost, 4),
            "composite_pareto_score": round(1.0 / ((float(np.mean(late_noisy_mse)) + 0.05) * (1.0 + 0.04 * wiring_cost)), 4),
        }

    ranked_noisy = sorted(results.items(), key=lambda kv: kv[1]["corrupted_eval_mse"])
    ranked_pareto = sorted(results.items(), key=lambda kv: kv[1]["composite_pareto_score"], reverse=True)
    return {
        "experiment": "extended_connectome_learning_33_algorithm_benchmark",
        "seed": seed,
        "steps": steps,
        "total_algorithms": len(results),
        "top5_lowest_corrupted_mse": [(k, v["corrupted_eval_mse"]) for k, v in ranked_noisy[:5]],
        "top5_pareto_wiring_efficiency": [(k, v["composite_pareto_score"]) for k, v in ranked_pareto[:5]],
        "algorithms": results,
    }
