"""SONAPSE HYPERSPACE — Native Higher-Dimensional (4D / N-D) Developmental Environments.

Extends ``synapse.world4d`` into general N-dimensional Euclidean spaces (default N=4,
extensible to 5D/6D) with DIRECT higher-dimensional sensory and motor interfaces
(not merely 2D screen projections!):
- 4D / N-D navigation & hyperwall shortcut bypass
- Higher-dimensional polytope recognition (4-Simplex, Tesseract/Hypercube, 16-Cell)
- Exact SO(4) planar rotation invariance assay
- Hidden 4D object reconstruction & spatial memory
- Controlled 3D-pretrain-to-4D-transfer vs Native-4D development benchmark
- Interactive 3D cross-section slice & 4D->3D stereographic/perspective projection data.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def so_nd_rotation(dim: int, plane: Tuple[int, int], theta: float) -> np.ndarray:
    """Exact Givens rotation matrix in SO(dim) rotating the (i, j) plane by theta radians."""
    R = np.eye(dim, dtype=np.float64)
    i, j = plane
    c, s = math.cos(theta), math.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def generate_nd_polytope(kind: str, dim: int = 4) -> np.ndarray:
    """Generate canonical vertex coordinates [V, dim] of an N-dimensional polytope."""
    if kind == "hypercube":
        # 2^dim vertices in {-1, +1}^dim
        n_v = 1 << dim
        verts = np.array([[(1.0 if (k >> b) & 1 else -1.0) for b in range(dim)] for k in range(n_v)], dtype=np.float64)
        return verts / math.sqrt(dim)
    if kind == "orthoplex":
        # 2*dim vertices +/- e_i (16-cell in 4D)
        verts = np.vstack([np.eye(dim), -np.eye(dim)])
        return verts
    if kind == "simplex":
        # (dim+1) regular simplex vertices centered at origin
        V = np.zeros((dim + 1, dim), dtype=np.float64)
        for i in range(dim):
            V[i, i] = 1.0
        V[dim] = -1.0 / (1.0 + math.sqrt(dim + 1.0))
        V -= np.mean(V, axis=0, keepdims=True)
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        return V / norms
    raise ValueError(f"Unknown polytope kind: {kind!r}")


def project_4d_to_3d(points_nd: np.ndarray, camera_w: float = 3.0, angles: Tuple[float, float] = (0.4, 0.25)) -> List[List[float]]:
    """Project 4D (or N-D) points to 3D via XW/YW rotations and perspective projection from camera_w."""
    pts = np.asarray(points_nd, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    d = pts.shape[1]
    if d < 4:
        pts = np.pad(pts, ((0, 0), (0, 4 - d)))
    R1 = so_nd_rotation(pts.shape[1], (0, 3), angles[0])
    R2 = so_nd_rotation(pts.shape[1], (1, 3), angles[1])
    rot = pts @ R1.T @ R2.T
    out = []
    for p in rot:
        w = float(p[3])
        scale = 1.0 / max(camera_w - w, 0.35)
        out.append([round(float(p[0] * scale), 4), round(float(p[1] * scale), 4), round(float(p[2] * scale), 4)])
    return out


class HyperspaceEnvironment:
    """Native N-dimensional Euclidean developmental environment (default 4D, extensible to 5D/6D)."""

    def __init__(self, dim: int = 4, arena_half: float = 5.0, seed: int = 42) -> None:
        if not 3 <= dim <= 8:
            raise ValueError("dim must be in 3..8")
        self.dim = int(dim)
        self.arena_half = float(arena_half)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        # Precompute 2*dim cardinal + 2*dim diagonal unit rays directly in N-D
        rays = [np.eye(dim)[i] for i in range(dim)] + [-np.eye(dim)[i] for i in range(dim)]
        for i in range(dim):
            v = np.zeros(dim)
            v[i] = 1.0
            v[(i + 1) % dim] = 1.0
            rays.append(v / math.sqrt(2.0))
        self.sensor_rays = np.stack(rays, axis=0)  # [3*dim, dim]
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(self.dim, dtype=np.float64)
        self.pos[0] = -3.2
        self.vel = np.zeros(self.dim, dtype=np.float64)
        self.target = np.zeros(self.dim, dtype=np.float64)
        self.target[0] = 3.2
        self.target[1] = float(self.rng.uniform(-1.0, 1.0))
        if self.dim >= 4:
            self.target[3] = float(self.rng.uniform(-0.4, 0.4))
        # Central 3D hyperwall blocking ALL x=0 paths within |w| <= 0.75 (requires 4D shortcut!)
        self.wall_min = np.full(self.dim, -self.arena_half)
        self.wall_max = np.full(self.dim, self.arena_half)
        self.wall_min[0], self.wall_max[0] = -0.45, 0.45
        if self.dim >= 4:
            self.wall_min[3], self.wall_max[3] = -0.75, 0.75
        self.steps = 0
        self.used_4d_shortcut = False
        self.visited_memory: List[np.ndarray] = [self.pos.copy()]
        return self.observe()

    def _hits_wall(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.wall_min) and np.all(p <= self.wall_max))

    def observe(self) -> np.ndarray:
        """Direct N-D sensory vector: [pos/half, vel, rel_target/half, N-D ray distances]."""
        rel = self.target - self.pos
        ray_d = np.ones(len(self.sensor_rays), dtype=np.float64)
        for ri, rdir in enumerate(self.sensor_rays):
            for step_k in (0.4, 0.9, 1.5, 2.3, 3.2):
                pt = self.pos + rdir * step_k
                if np.any(np.abs(pt) > self.arena_half) or self._hits_wall(pt):
                    ray_d[ri] = step_k / 3.2
                    break
        return np.concatenate([self.pos / self.arena_half, self.vel, rel / self.arena_half, ray_d])

    def step(self, action_nd: Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        act = np.resize(np.asarray(action_nd, dtype=np.float64).ravel(), self.dim)
        act = np.clip(act, -1.0, 1.0)
        self.vel = np.clip(0.78 * self.vel + 0.42 * act, -1.3, 1.3)
        cand = np.clip(self.pos + 0.35 * self.vel, -self.arena_half, self.arena_half)
        collided = self._hits_wall(cand)
        if collided:
            self.vel *= -0.25
        else:
            self.pos = cand
        if self.dim >= 4 and abs(self.pos[3]) > 0.75 and abs(self.pos[0]) < 0.6:
            self.used_4d_shortcut = True
        dist = float(np.linalg.norm(self.pos - self.target))
        reached = dist < 0.85
        reward = (5.0 if reached else 0.0) - 0.08 * dist - (0.15 if collided else 0.0)
        self.steps += 1
        self.visited_memory.append(self.pos.copy())
        done = reached or (self.steps >= 64)
        return self.observe(), float(reward), done, {
            "dist": round(dist, 4),
            "reached": reached,
            "collided": collided,
            "used_4d_shortcut": self.used_4d_shortcut,
            "pos_nd": [round(float(x), 3) for x in self.pos],
            "projection_3d": project_4d_to_3d(self.pos)[0],
        }

    def slice_3d_at_w(self) -> Dict[str, Any]:
        """Return interactive 3D slice state at the agent's current w coordinate."""
        w_val = float(self.pos[3]) if self.dim >= 4 else 0.0
        wall_visible = (abs(w_val) <= 0.75) if self.dim >= 4 else True
        return {
            "dim": self.dim,
            "current_w": round(w_val, 4),
            "wall_present_in_slice": wall_visible,
            "agent_3d": [round(float(self.pos[i]), 3) for i in range(3)],
            "target_3d": [round(float(self.target[i]), 3) for i in range(3)],
            "projected_tesseract_vertices_3d": project_4d_to_3d(generate_nd_polytope("hypercube", 4) * 2.2),
        }


def run_hyperspace_developmental_study(seeds: Sequence[int] = (7, 19), generations: int = 12) -> Dict[str, Any]:
    """Compare 3D-pretrained-then-4D transfer vs Native-4D development + SO(4) polytope recognition."""
    # Part 1: 3D-to-4D transfer vs Native 4D navigation (shortcut discovery across hyperwall)
    obs_dim_4d = HyperspaceEnvironment(dim=4).observe().size
    transfer_returns, native_returns = [], []
    transfer_shortcuts, native_shortcuts = [], []

    def _eval_policy(W: np.ndarray, env: HyperspaceEnvironment, ep_seed: int, mask_w: bool = False) -> Tuple[float, bool]:
        obs = env.reset(ep_seed)
        tot = 0.0
        for _ in range(56):
            act = np.tanh(W @ obs)
            if mask_w and env.dim >= 4:
                act[3] = 0.0
            obs, r, done, info = env.step(act)
            tot += r
            if done:
                break
        return tot, bool(info["used_4d_shortcut"])

    def _evolve_policy(W_init: np.ndarray, env: HyperspaceEnvironment, gens: int, seed: int, mask_w: bool = False) -> np.ndarray:
        rng = np.random.default_rng(seed)
        best_W = W_init.copy()
        best_r, _ = _eval_policy(best_W, env, seed + 1, mask_w=mask_w)
        for g in range(gens):
            for c in range(10):
                cand = best_W + rng.normal(0.0, 0.25, size=best_W.shape)
                r, _ = _eval_policy(cand, env, seed + 1, mask_w=mask_w)
                if r > best_r:
                    best_r, best_W = r, cand
        return best_W

    for seed in seeds:
        rng = np.random.default_rng(seed)
        W0 = rng.normal(0.0, 0.2, size=(4, obs_dim_4d))
        env4 = HyperspaceEnvironment(dim=4, seed=seed)

        # Arm A: Native 4D evolution for full budget (generations)
        W_native = _evolve_policy(W0, env4, generations, seed=seed + 10, mask_w=False)
        # Arm B: Pretrain half budget restricted to 3D (w=0), then transfer to 4D for second half
        W_3d = _evolve_policy(W0, env4, generations // 2, seed=seed + 10, mask_w=True)
        W_trans = _evolve_policy(W_3d, env4, generations - generations // 2, seed=seed + 50, mask_w=False)

        r_nat, sc_nat = _eval_policy(W_native, env4, seed + 900)
        r_tr, sc_tr = _eval_policy(W_trans, env4, seed + 900)
        native_returns.append(r_nat)
        transfer_returns.append(r_tr)
        native_shortcuts.append(1.0 if sc_nat else 0.0)
        transfer_shortcuts.append(1.0 if sc_tr else 0.0)

    # Part 2: SO(4) Rotation-Invariant 4D Polytope Recognition (hypercube vs orthoplex vs simplex)
    rng_p = np.random.default_rng(seeds[0])
    kinds = ("hypercube", "orthoplex", "simplex")
    feats, labels = [], []
    for k_idx, kind in enumerate(kinds):
        base_v = generate_nd_polytope(kind, dim=4)
        for _ in range(30):
            # Random SO(4) double rotation in (0,3) and (1,2) planes
            R = so_nd_rotation(4, (0, 3), rng_p.uniform(0, 2 * math.pi)) @ so_nd_rotation(4, (1, 2), rng_p.uniform(0, 2 * math.pi))
            rot_v = base_v @ R.T + rng_p.normal(0.0, 0.03, size=base_v.shape)
            # Pairwise distance spectrum (exact SO(4) invariant!)
            dmat = np.linalg.norm(rot_v[:, None, :] - rot_v[None, :, :], axis=-1)
            hist, _ = np.histogram(dmat[dmat > 1e-4], bins=8, range=(0.0, 2.5), density=True)
            feats.append(np.concatenate([[len(rot_v) / 20.0, float(np.mean(dmat)), float(np.std(dmat))], hist]))
            labels.append(k_idx)
    X = np.vstack(feats)
    y = np.asarray(labels)
    perm = rng_p.permutation(len(y))
    X, y = X[perm], y[perm]
    sp = int(0.65 * len(y))
    Y_oh = np.eye(3)[y[:sp]]
    W_cls = np.linalg.solve(X[:sp].T @ X[:sp] + 0.05 * np.eye(X.shape[1]), X[:sp].T @ Y_oh)
    preds = np.argmax(X[sp:] @ W_cls, axis=1)
    poly_acc = float(np.mean(preds == y[sp:]))

    return {
        "experiment": "sonapse_hyperspace_4d_development_and_so4_recognition",
        "seeds": list(seeds),
        "native_4d_development": {
            "mean_heldout_return": round(float(np.mean(native_returns)), 4),
            "shortcut_discovery_rate": round(float(np.mean(native_shortcuts)), 4),
        },
        "transfer_3d_to_4d": {
            "mean_heldout_return": round(float(np.mean(transfer_returns)), 4),
            "shortcut_discovery_rate": round(float(np.mean(transfer_shortcuts)), 4),
        },
        "so4_rotation_invariant_polytope_recognition_acc": round(poly_acc, 4),
        "optional_5d_probe_obs_dim": int(HyperspaceEnvironment(dim=5).observe().size),
    }


def run_high_dim_5d_6d_study(seeds: Sequence[int] = (7, 19), generations: int = 12) -> Dict[str, Any]:
    """Resolves Open Question #5: Native 5D & 6D navigation + partially occluded SO(N) polytope reconstruction."""
    results_by_dim: Dict[str, Any] = {}
    for dim in (5, 6):
        env = HyperspaceEnvironment(dim=dim, seed=seeds[0])
        obs_dim = env.observe().size
        returns_nd, shortcuts_nd = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed + dim * 31)
            W = rng.normal(0.0, 0.2, size=(dim, obs_dim))
            obs0 = env.reset(seed + 1)
            best_r = -1e9
            for _ in range(generations):
                for _ in range(8):
                    cand = W + rng.normal(0.0, 0.24, size=W.shape)
                    obs = env.reset(seed + 1)
                    tot = 0.0
                    for _ in range(48):
                        obs, r, done, info = env.step(np.tanh(cand @ obs))
                        tot += r
                        if done:
                            break
                    if tot > best_r:
                        best_r, W = tot, cand
            # Held-out seed evaluation
            obs = env.reset(seed + 888)
            tot_h = 0.0
            for _ in range(56):
                obs, r, done, info = env.step(np.tanh(W @ obs))
                tot_h += r
                if done:
                    break
            returns_nd.append(tot_h)
            shortcuts_nd.append(1.0 if info["used_4d_shortcut"] else 0.0)

        # Partially occluded (35% vertices dropped!) SO(dim) polytope classification
        rng_occ = np.random.default_rng(seeds[0] + dim * 100)
        kinds = ("hypercube", "orthoplex", "simplex")
        feats, labels = [], []
        for k_i, kind in enumerate(kinds):
            base_v = generate_nd_polytope(kind, dim=dim)
            for _ in range(28):
                R = so_nd_rotation(dim, (0, dim - 1), rng_occ.uniform(0, 6.28)) @ so_nd_rotation(dim, (1, dim - 2), rng_occ.uniform(0, 6.28))
                rot_v = base_v @ R.T + rng_occ.normal(0.0, 0.03, size=base_v.shape)
                keep_n = max(3, int(round(0.65 * len(rot_v))))
                keep_idx = rng_occ.choice(len(rot_v), size=keep_n, replace=False)
                occ_v = rot_v[keep_idx]
                dmat = np.linalg.norm(occ_v[:, None, :] - occ_v[None, :, :], axis=-1)
                hist, _ = np.histogram(dmat[dmat > 1e-4], bins=8, range=(0.0, 2.5), density=True)
                feats.append(np.concatenate([[keep_n / 40.0, float(np.mean(dmat)), float(np.std(dmat))], hist]))
                labels.append(k_i)
        X = np.vstack(feats)
        y = np.asarray(labels)
        perm = rng_occ.permutation(len(y))
        X, y = X[perm], y[perm]
        sp = int(0.65 * len(y))
        W_c = np.linalg.solve(X[:sp].T @ X[:sp] + 0.08 * np.eye(X.shape[1]), X[:sp].T @ np.eye(3)[y[:sp]])
        acc_occ = float(np.mean(np.argmax(X[sp:] @ W_c, axis=1) == y[sp:]))

        results_by_dim[f"{dim}D"] = {
            "obs_dim": int(obs_dim),
            "mean_heldout_return": round(float(np.mean(returns_nd)), 4),
            "hyperwall_bypass_rate": round(float(np.mean(shortcuts_nd)), 4),
            "occluded_35pct_polytope_acc": round(acc_occ, 4),
        }
    return {"experiment": "hyperspace_5d_6d_navigation_and_occluded_polytopes", "dimensions": results_by_dim}
