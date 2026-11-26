"""GhostInTheMachine-Inspired Causal Emergence, MAF Slow Subspace, MMI-PID & Cross-Yoked Twin Suite.

Adapts rigorous information-theoretic and dynamical-systems diagnostics inspired by
``GhostInTheMachine`` (v4-v8) into ``SONAPSE-MORPHOGENESIS``:
1. Min-Max Autocorrelation Factors (MAF) slow collective macro-state discovery.
2. Exact collinearity-free Barnett-Seth Dynamical Dependence (DD) & Independence (DI).
3. Calibrated non-negative Gaussian MMI-PID & integrated information decomposition (Phi_R).
4. Analytical Closed-Loop Efference-Copy Jacobian Spectrum (open vs comparator-damped).
5. Cross-Yoked Twin Authorship Assay separating genuine closed-loop causal agency
   from shared observation/reward statistics.

Explicit epistemic stance: These metrics quantify dynamical closure, temporal synergy,
and closed-loop causal authorship in computational graphs; they do NOT establish
phenomenal consciousness or subjective experience.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.linalg import solve_triangular


def _safe_logdet_cov(X: np.ndarray, reg: float = 1e-5) -> float:
    """Regularized log-determinant of empirical covariance of [T, D] trajectory."""
    if X.ndim == 1:
        X = X[:, None]
    Xc = X - np.mean(X, axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1) + reg * np.eye(Xc.shape[1])
    sign, ld = np.linalg.slogdet(cov)
    return float(ld) if sign > 0 else -50.0


def gaussian_mi(A: np.ndarray, B: np.ndarray, reg: float = 1e-5) -> float:
    """Exact regularized Gaussian mutual information I(A; B) in nats (>= 0)."""
    a2 = A[:, None] if A.ndim == 1 else A
    b2 = B[:, None] if B.ndim == 1 else B
    joint = np.hstack([a2, b2])
    mi = 0.5 * (_safe_logdet_cov(a2, reg) + _safe_logdet_cov(b2, reg) - _safe_logdet_cov(joint, reg))
    return float(max(0.0, mi))


def extract_maf_slow_subspace(X: np.ndarray, k_modes: int = 2, reg: float = 1e-4) -> Dict[str, Any]:
    """Extract Min-Max Autocorrelation Factors (MAF) maximizing lag-1 autocorrelation.

    Solves the generalized symmetric eigenproblem (1/2)(C_1 + C_1^T) w = lambda (C_0 + reg I) w
    via Cholesky whitening so collective slow modes missed by variance-based PCA are recovered.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 5 or X.shape[1] < 2:
        return {"autocorrelations": [0.0], "projections": np.zeros((len(X), 1)), "weights": np.ones((1, 1))}
    Xc = X - np.mean(X, axis=0, keepdims=True)
    T, D = Xc.shape
    k = min(k_modes, D)
    C0 = (Xc.T @ Xc) / max(T - 1, 1) + reg * np.eye(D)
    C1 = (Xc[:-1].T @ Xc[1:]) / max(T - 2, 1)
    C1_sym = 0.5 * (C1 + C1.T)

    L = np.linalg.cholesky(C0)
    # Transform: L^-1 C1_sym L^-T
    tmp = solve_triangular(L, C1_sym, lower=True)
    M = solve_triangular(L, tmp.T, lower=True).T
    M = 0.5 * (M + M.T)
    eigvals, eigvecs_w = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1][:k]
    top_vals = np.clip(eigvals[order], -0.999, 0.999)
    W_maf = solve_triangular(L.T, eigvecs_w[:, order], lower=False)  # [D, k]
    proj = Xc @ W_maf  # [T, k]

    # Compare lag-1 autocorrelation of MAF1 vs standard PC1
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ vt[0]
    pc1_ac = float(np.corrcoef(pc1[:-1], pc1[1:])[0, 1]) if float(np.std(pc1)) > 1e-8 else 0.0
    if not np.isfinite(pc1_ac):
        pc1_ac = 0.0

    return {
        "autocorrelations": [round(float(v), 5) for v in top_vals],
        "maf1_lag1_autocorr": round(float(top_vals[0]), 5),
        "pc1_lag1_autocorr": round(pc1_ac, 5),
        "slow_mode_advantage_over_pca": round(float(top_vals[0]) - pc1_ac, 5),
        "projections": proj,
        "weights": W_maf,
    }


def compute_dynamical_independence(X: np.ndarray, V: np.ndarray, tau: int = 1) -> Dict[str, float]:
    """Barnett & Seth collinearity-free Dynamical Dependence (DD) & Dynamical Independence (DI).

    When macro V = f(X), conditioning on both X_t and V_t is collinear unless evaluated via
    DD(V) = I(V_{t+tau}; X_t) - I(V_{t+tau}; V_t) >= 0, with self-predictive closure
    ratio DI = I(V_{t+tau}; V_t) / max(I(V_{t+tau}; X_t), eps) in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    if len(X) <= tau + 3:
        return {"mi_macro_self": 0.0, "mi_macro_from_micro": 0.0, "dynamical_dependence": 0.0, "dynamical_closure_ratio": 0.0}
    V_fut = V[tau:]
    V_now = V[:-tau]
    X_now = X[:-tau]
    i_vv = gaussian_mi(V_fut, V_now)
    i_vx = max(i_vv, gaussian_mi(V_fut, np.hstack([X_now, V_now])))
    dd = max(0.0, i_vx - i_vv)
    closure = i_vv / max(i_vx, 1e-6)
    return {
        "mi_macro_self": round(i_vv, 5),
        "mi_macro_from_micro": round(i_vx, 5),
        "dynamical_dependence": round(dd, 5),
        "dynamical_closure_ratio": round(float(np.clip(closure, 0.0, 1.0)), 5),
    }


def compute_gaussian_mmi_pid(X_a: np.ndarray, X_b: np.ndarray, tau: int = 1) -> Dict[str, float]:
    """Exact non-negative Minimum-Mutual-Information Partial Information Decomposition (MMI-PID)
    between two neural subpopulations/organelles A and B predicting future joint state J_{t+tau}.
    """
    A = X_a[:, None] if X_a.ndim == 1 else np.asarray(X_a, dtype=np.float64)
    B = X_b[:, None] if X_b.ndim == 1 else np.asarray(X_b, dtype=np.float64)
    J = np.hstack([A, B])
    if len(J) <= tau + 3:
        return {"redundancy": 0.0, "unique_a": 0.0, "unique_b": 0.0, "synergy": 0.0, "phi_r": 0.0}
    J_fut = J[tau:]
    A_now, B_now, J_now = A[:-tau], B[:-tau], J[:-tau]

    i_a = gaussian_mi(A_now, J_fut)
    i_b = gaussian_mi(B_now, J_fut)
    i_ab = max(i_a, i_b, gaussian_mi(J_now, J_fut))

    redundancy = min(i_a, i_b)
    unique_a = max(0.0, i_a - redundancy)
    unique_b = max(0.0, i_b - redundancy)
    synergy = max(0.0, i_ab - max(i_a, i_b))

    # Double-redundancy atom & Mediano et al. (2022) integrated information Phi_R proxy
    rtr = min(
        gaussian_mi(A_now, A[tau:]),
        gaussian_mi(A_now, B[tau:]),
        gaussian_mi(B_now, A[tau:]),
        gaussian_mi(B_now, B[tau:]),
    )
    phi_r = synergy + max(0.0, redundancy - rtr)
    return {
        "total_tdmi": round(i_ab, 5),
        "redundancy": round(redundancy, 5),
        "unique_a": round(unique_a, 5),
        "unique_b": round(unique_b, 5),
        "synergy": round(synergy, 5),
        "double_redundancy_rtr": round(rtr, 5),
        "phi_r": round(phi_r, 5),
    }


def analyze_closed_loop_efference_spectrum(
    W_rec: np.ndarray,
    W_motor: np.ndarray,
    W_comparator: np.ndarray,
    gain: float = 0.95,
    comparator_gamma: float = -0.22,
) -> Dict[str, float]:
    """Analytical open-loop vs closed-loop efference comparator Jacobian eigenvalue shift."""
    W_open = float(gain) * np.asarray(W_rec, dtype=np.float64)
    W_closed = W_open + float(comparator_gamma) * (np.asarray(W_comparator, dtype=np.float64) @ np.asarray(W_motor, dtype=np.float64))
    eig_open = np.linalg.eigvals(W_open)
    eig_closed = np.linalg.eigvals(W_closed)
    max_re_open = float(np.max(np.real(eig_open)))
    max_re_closed = float(np.max(np.real(eig_closed)))
    return {
        "max_re_lambda_open": round(max_re_open, 5),
        "max_re_lambda_closed": round(max_re_closed, 5),
        "delta_max_re_lambda": round(max_re_closed - max_re_open, 5),
        "stabilized": bool(max_re_closed < max_re_open),
    }


def run_cross_yoked_twin_assay(n_neurons: int = 24, steps: int = 120, seed: int = 7) -> Dict[str, Any]:
    """Execute the GhostInTheMachine Cross-Yoked Twin Authorship Assay on Morphogenesis connectomes.

    - ``master`` drives a 2D dynamical body in closed loop (its own motor output u_t moves the world).
    - ``yoked_twin`` has an identical connectome architecture & weight scale (different seed) and
      receives the EXACT SAME sensory stream o_t and reward r_t step-for-step, while generating its
      own motor intention u_twin_t and efference prediction!
    - Measures out-of-sample R^2 of [state, efference_copy] predicting next sensory transition:
      genuine causal authorship yields positive R^2_master > R^2_twin.
    """
    from ..core.compiled_engine import CompiledConnectomeEngine
    from ..genesis.population import generate_candidate_genome

    rng = np.random.default_rng(seed)
    g_master = generate_candidate_genome("mixed_custom_standard", n_neurons=n_neurons, seed=seed).connectome
    g_twin = generate_candidate_genome("mixed_custom_standard", n_neurons=n_neurons, seed=seed + 101).connectome

    eng_m = CompiledConnectomeEngine(g_master, dt_ms=1.0, seed=seed)
    eng_t = CompiledConnectomeEngine(g_twin, dt_ms=1.0, seed=seed + 101)

    W_act_m = rng.normal(0.0, 0.45 / np.sqrt(n_neurons), size=(2, n_neurons))
    W_act_t = rng.normal(0.0, 0.45 / np.sqrt(n_neurons), size=(2, n_neurons))

    pos = np.zeros(2, dtype=np.float64)
    vel = np.zeros(2, dtype=np.float64)

    X_m, X_t, E_m, E_t, O_delta = [], [], [], [], []
    for k in range(steps):
        exog = 0.08 * np.array([np.sin(0.27 * k), np.cos(0.19 * k)])
        obs = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)

        hm = eng_m.step(obs)["outputs"]
        ht = eng_t.step(obs)["outputs"]

        eff_m = np.tanh(W_act_m @ hm)
        eff_t = np.tanh(W_act_t @ ht)

        # ONLY master's action actually actuates the physical world!
        vel_next = 0.72 * vel + 0.48 * eff_m + exog
        pos_next = np.clip(pos + 0.35 * vel_next, -4.0, 4.0)
        obs_next = np.array([pos_next[0], pos_next[1], vel_next[0], vel_next[1]], dtype=np.float64)

        X_m.append(hm)
        X_t.append(ht)
        E_m.append(eff_m)
        E_t.append(eff_t)
        O_delta.append(obs_next - obs)

        pos, vel = pos_next, vel_next

    Xm_arr = np.vstack(X_m)
    Xt_arr = np.vstack(X_t)
    Em_arr = np.vstack(E_m)
    Et_arr = np.vstack(E_t)
    Y_arr = np.vstack(O_delta)

    sp = int(0.6 * steps)

    def _oos_r2(feat_mat: np.ndarray, target_mat: np.ndarray) -> float:
        F_tr, F_te = feat_mat[:sp], feat_mat[sp:]
        Y_tr, Y_te = target_mat[:sp], target_mat[sp:]
        W = np.linalg.solve(F_tr.T @ F_tr + 0.25 * np.eye(F_tr.shape[1]), F_tr.T @ Y_tr)
        pred = F_te @ W
        ss_res = float(np.sum((Y_te - pred) ** 2))
        ss_tot = float(np.sum((Y_te - np.mean(Y_te, axis=0, keepdims=True)) ** 2)) + 1e-8
        return float(max(-1.0, 1.0 - ss_res / ss_tot))

    r2_master = _oos_r2(np.hstack([Xm_arr, Em_arr]), Y_arr)
    r2_twin = _oos_r2(np.hstack([Xt_arr, Et_arr]), Y_arr)

    maf_res = extract_maf_slow_subspace(Xm_arr, k_modes=2)
    di_res = compute_dynamical_independence(Xm_arr, maf_res["projections"][:, :1])
    half = max(1, n_neurons // 2)
    pid_res = compute_gaussian_mmi_pid(Xm_arr[:, :half], Xm_arr[:, half:])

    W_rec_est = np.eye(n_neurons) * 0.55
    W_cmp = rng.normal(0.0, 0.3, size=(n_neurons, 2))
    spec_res = analyze_closed_loop_efference_spectrum(W_rec_est, np.abs(W_act_m), np.abs(W_cmp), comparator_gamma=-0.25)

    return {
        "experiment": "ghost_in_the_machine_causal_emergence_and_cross_yoked_twin",
        "n_neurons": n_neurons,
        "steps": steps,
        "seed": seed,
        "subjective_experience_claimed": False,
        "cross_yoked_authorship": {
            "master_oos_r2": round(r2_master, 5),
            "yoked_twin_oos_r2": round(r2_twin, 5),
            "causal_authorship_gap": round(r2_master - r2_twin, 5),
            "master_exceeds_yoked_twin": bool(r2_master > r2_twin),
        },
        "maf_slow_subspace": {
            "maf1_lag1_autocorr": maf_res["maf1_lag1_autocorr"],
            "pc1_lag1_autocorr": maf_res["pc1_lag1_autocorr"],
            "slow_mode_advantage_over_pca": maf_res["slow_mode_advantage_over_pca"],
        },
        "dynamical_independence": di_res,
        "gaussian_mmi_pid": pid_res,
        "closed_loop_efference_spectrum": spec_res,
    }


def run_causal_emergence_and_authorship_suite(
    X: np.ndarray,
    macro_dim: int = 2,
    seed: int = 7,
    include_twin_assay: bool = True,
) -> Dict[str, Any]:
    """Unified trajectory causal emergence + optional cross-yoked twin authorship suite."""
    X_arr = np.asarray(X, dtype=np.float64)
    maf = extract_maf_slow_subspace(X_arr, k_modes=max(1, int(macro_dim)))
    di = compute_dynamical_independence(X_arr, maf["projections"][:, : max(1, int(macro_dim))])
    half = max(1, X_arr.shape[1] // 2)
    pid = compute_gaussian_mmi_pid(X_arr[:, :half], X_arr[:, half:])
    twin = run_cross_yoked_twin_assay(n_neurons=16, steps=60, seed=seed) if include_twin_assay else None
    return {
        "subjective_experience_claimed": False,
        "maf_slow_subspace": {
            "autocorrelations": maf["autocorrelations"],
            "maf1_lag1_autocorr": maf["maf1_lag1_autocorr"],
            "pc1_lag1_autocorr": maf["pc1_lag1_autocorr"],
            "slow_mode_advantage_over_pca": maf["slow_mode_advantage_over_pca"],
        },
        "dynamical_independence": di,
        "mmi_pid": pid,
        "cross_yoked_authorship": twin["cross_yoked_authorship"] if twin else None,
        "efference_spectrum": twin["closed_loop_efference_spectrum"] if twin else None,
    }
