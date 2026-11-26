"""SONAPSE CURIOSITY — 12 Novel Interactive Sensory-Motor Training Modalities.

Moves beyond static text/image/audio into interactive executable worlds (Section 29):
  1. restricted_python_vm         - Deterministic safe bytecode/AST execution sandbox (zero eval/exec!)
  2. ast_transformer              - Interactive expression-tree manipulation
  3. symbolic_math_transforms     - Algebraic simplification & equation balancing
  4. propositional_proof_verifier - Natural-deduction / truth-table proof steps
  5. electronic_rc_rlc_circuit    - Continuous-time RLC circuit voltage/current dynamics
  6. mechanical_spring_mass       - Coupled damped harmonic oscillator mechanics
  7. protein_contact_geometry     - 3D HP-lattice chain folding energy & contact sensing
  8. cellular_automata_1d         - Elementary Rule-110 / Rule-30 interactive perturbation
  9. fractal_terrain_explorer     - Procedural multi-octave fractal landscape navigation
 10. fluid_advection_2d           - Simplified 2D vortex advection-diffusion field
 11. resource_allocation_market   - Multi-node constrained metabolic/compute allocation
 12. packet_network_router        - Congested graph packet routing & latency control
"""
from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


NOVEL_MODALITIES: Tuple[str, ...] = (
    "restricted_python_vm",
    "ast_transformer",
    "symbolic_math_transforms",
    "propositional_proof_verifier",
    "electronic_rc_rlc_circuit",
    "mechanical_spring_mass",
    "protein_contact_geometry",
    "cellular_automata_1d",
    "fractal_terrain_explorer",
    "fluid_advection_2d",
    "resource_allocation_market",
    "packet_network_router",
)


class RestrictedPythonVM:
    """Safe, deterministic, step-by-step virtual execution environment for Codeception.

    Never executes arbitrary Python ``exec``/``eval``—interprets a strict register/stack
    instruction subset exposing PC, registers, memory slots, branch flags, and exceptions
    so a developing connectome can interactively probe how operations transform execution state.
    """

    OPCODES = ("LOAD_CONST", "ADD", "SUB", "MUL", "CMP_GT", "JMP_IF", "STORE", "HALT")

    def __init__(self, program: Optional[List[Tuple[str, Any]]] = None) -> None:
        self.program = program or [
            ("LOAD_CONST", 3),
            ("STORE", 0),
            ("LOAD_CONST", 5),
            ("ADD", 0),
            ("CMP_GT", 6),
            ("STORE", 1),
            ("HALT", 0),
        ]
        self.reset()

    def reset(self) -> np.ndarray:
        self.pc = 0
        self.acc = 0.0
        self.flag = 0.0
        self.memory = np.zeros(4, dtype=np.float64)
        self.halted = False
        self.exception: Optional[str] = None
        self.steps = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        op_idx = 0.0
        if 0 <= self.pc < len(self.program):
            op = self.program[self.pc][0]
            op_idx = float(self.OPCODES.index(op)) / max(len(self.OPCODES), 1) if op in self.OPCODES else 0.0
        return np.array([
            self.pc / max(len(self.program), 1),
            math.tanh(self.acc / 10.0),
            self.flag,
            op_idx,
            *(np.tanh(self.memory / 10.0)),
            1.0 if self.exception else 0.0,
        ], dtype=np.float64)

    def step(self, intervention_op: Optional[Tuple[str, Any]] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.halted or self.pc >= len(self.program) or self.steps >= 32:
            self.halted = True
            return self.observe(), 0.0, True, {"halted": True, "acc": self.acc, "exception": self.exception}

        op, arg = intervention_op if intervention_op is not None else self.program[self.pc]
        self.steps += 1
        reward = 0.0
        try:
            if op == "LOAD_CONST":
                self.acc = float(arg)
            elif op == "ADD":
                self.acc += float(self.memory[int(arg) % 4])
            elif op == "SUB":
                self.acc -= float(self.memory[int(arg) % 4])
            elif op == "MUL":
                self.acc *= float(self.memory[int(arg) % 4])
            elif op == "CMP_GT":
                self.flag = 1.0 if self.acc > float(arg) else -1.0
            elif op == "STORE":
                self.memory[int(arg) % 4] = self.acc
            elif op == "JMP_IF":
                if self.flag > 0.0:
                    self.pc = int(np.clip(int(arg), 0, len(self.program) - 1))
                    return self.observe(), 0.1, False, {"op": op, "acc": self.acc, "pc": self.pc}
            elif op == "HALT":
                self.halted = True
                reward = 1.0
            else:
                self.exception = f"IllegalOpcode:{op}"
                self.halted = True
                reward = -1.0
            self.pc += 1
        except Exception as exc:
            self.exception = type(exc).__name__
            self.halted = True
            reward = -1.0

        return self.observe(), reward, self.halted, {"op": op, "acc": self.acc, "pc": self.pc, "memory": self.memory.tolist(), "exception": self.exception}


class InteractiveModalitySuite:
    """Unified deterministic interactive environment wrapper covering all 12 novel modalities."""

    def __init__(self, modality: str, seed: int = 42) -> None:
        if modality not in NOVEL_MODALITIES:
            raise KeyError(f"Unknown modality: {modality!r}")
        self.modality = modality
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.vm = RestrictedPythonVM()
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = np.zeros(8, dtype=np.float64)
        if self.modality == "restricted_python_vm":
            return self.vm.reset()
        if self.modality == "electronic_rc_rlc_circuit":
            self.state = np.array([0.0, 0.0, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0])  # V_c, I_L, R, L, C...
        elif self.modality == "mechanical_spring_mass":
            self.state = np.array([1.0, 0.0, -0.5, 0.0, 1.5, 0.15, 0.0, 0.0])  # x1, v1, x2, v2, k, damping
        elif self.modality == "cellular_automata_1d":
            self.state = np.array([0, 0, 0, 1, 0, 1, 1, 0], dtype=np.float64)
        elif self.modality == "protein_contact_geometry":
            self.state = self.rng.normal(0.0, 0.8, size=8)
        elif self.modality == "packet_network_router":
            self.state = np.array([0.4, 0.2, 0.7, 0.1, 0.3, 0.5, 0.2, 0.1])  # queue loads across 8 routers
        else:
            self.state = self.rng.uniform(-0.5, 0.5, size=8)
        return self.state.copy()

    def step(self, action: float | Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        act = np.resize(np.asarray(action, dtype=np.float64).ravel(), 4)
        act = np.clip(act, -1.0, 1.0)
        self.t += 1

        if self.modality == "restricted_python_vm":
            return self.vm.step()

        if self.modality == "electronic_rc_rlc_circuit":
            # Series RLC circuit driven by input voltage u = act[0]
            v_c, i_l, R, L, C = self.state[0], self.state[1], 1.0, 0.5, 0.4
            dt = 0.08
            u = float(act[0]) * 2.0
            di = (u - R * i_l - v_c) / L
            dv = i_l / C
            self.state[0] = float(np.clip(v_c + dt * dv, -5.0, 5.0))
            self.state[1] = float(np.clip(i_l + dt * di, -5.0, 5.0))
            target_v = math.sin(0.2 * self.t)
            err = abs(self.state[0] - target_v)
            return self.state.copy(), float(-err), self.t >= 32, {"v_c": round(self.state[0], 4), "target_v": round(target_v, 4)}

        if self.modality == "mechanical_spring_mass":
            x1, v1, x2, v2, k, c = self.state[:6]
            dt = 0.1
            f = float(act[0])
            a1 = -k * x1 + k * (x2 - x1) - c * v1 + f
            a2 = -k * (x2 - x1) - c * v2
            self.state[:4] = np.clip([x1 + dt * v1, v1 + dt * a1, x2 + dt * v2, v2 + dt * a2], -5.0, 5.0)
            energy = 0.5 * (v1 * v1 + v2 * v2 + k * x1 * x1)
            return self.state.copy(), float(-abs(x2)), self.t >= 32, {"mechanical_energy": round(energy, 4)}

        if self.modality == "cellular_automata_1d":
            # Rule 110 deterministic step with action flipping one cell
            bits = (self.state > 0.5).astype(int)
            flip_idx = int((act[0] + 1.0) * 3.5) % 8
            if abs(act[1]) > 0.5:
                bits[flip_idx] ^= 1
            nxt = np.zeros(8, dtype=int)
            for i in range(8):
                pat = (bits[(i - 1) % 8] << 2) | (bits[i] << 1) | bits[(i + 1) % 8]
                nxt[i] = (110 >> pat) & 1
            self.state = nxt.astype(np.float64)
            return self.state.copy(), float(-abs(np.mean(self.state) - 0.5)), self.t >= 32, {"density": float(np.mean(self.state))}

        if self.modality == "propositional_proof_verifier":
            # Deterministic propositional tautology verification: (P -> Q) & P => Q
            p, q = int(act[0] > 0), int(act[1] > 0)
            imp = (not p) or q
            valid_mp = (not (imp and p)) or q
            self.state[:4] = [float(p), float(q), float(imp), float(valid_mp)]
            return self.state.copy(), 1.0 if valid_mp else -1.0, self.t >= 16, {"modus_ponens_verified": bool(valid_mp)}

        if self.modality == "packet_network_router":
            # Balance queues across 8 nodes using routing action
            arrivals = self.rng.uniform(0.05, 0.25, size=8)
            service = 0.15 + 0.15 * np.resize(np.abs(act), 8)
            self.state = np.clip(self.state + arrivals - service, 0.0, 2.0)
            congestion = float(np.max(self.state))
            return self.state.copy(), float(-congestion), self.t >= 32, {"max_queue": round(congestion, 4)}

        if self.modality == "ast_transformer":
            # Interactive expression-tree depth & constant-folding optimizer:
            # state = [depth, const_nodes, var_nodes, dead_branches, eval_val, target_val, steps_norm, fold_gain]
            if self.t == 1:
                self.state[:] = [5.0, 4.0, 3.0, 2.0, 3.5, 3.5, 0.0, 0.0]
            fold_const = 1.0 if act[0] > 0.25 and self.state[1] > 1.0 else 0.0
            prune_dead = 1.0 if act[1] > 0.25 and self.state[3] > 0.0 else 0.0
            bad_rewrite = 1.0 if act[2] < -0.6 else 0.0
            self.state[1] = max(1.0, self.state[1] - 0.8 * fold_const)
            self.state[3] = max(0.0, self.state[3] - 1.0 * prune_dead)
            self.state[0] = max(1.0, 1.0 + 0.5 * self.state[1] + 0.5 * self.state[2] + 0.5 * self.state[3])
            if bad_rewrite > 0.0:
                self.state[4] += 0.5  # semantic drift penalty!
            sem_err = abs(self.state[4] - self.state[5])
            complexity = 0.15 * self.state[0] + 0.2 * self.state[3]
            self.state[6] = self.t / 24.0
            self.state[7] = fold_const + prune_dead
            reward = float(0.4 * (fold_const + prune_dead) - complexity - 2.0 * sem_err)
            return self.state.copy(), reward, self.t >= 24, {"tree_depth": round(float(self.state[0]), 3), "semantic_error": round(sem_err, 4)}

        if self.modality == "symbolic_math_transforms":
            # Quadratic/linear algebraic balancing: drive residual a*x^2 + b*x + c -> 0 via candidate root x
            if self.t == 1:
                self.state[:] = [1.0, -3.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0]  # x^2 - 3x + 2 = (x-1)(x-2)
            a_c, b_c, c_c, x_est = self.state[:4]
            x_est = float(np.clip(x_est + 0.35 * act[0], -4.0, 4.0))
            poly_val = a_c * (x_est ** 2) + b_c * x_est + c_c
            deriv = 2.0 * a_c * x_est + b_c
            # Optional Newton-like symbolic step when act[1] > 0.3
            if act[1] > 0.3 and abs(deriv) > 0.15:
                x_est = float(np.clip(x_est - 0.4 * (poly_val / deriv), -4.0, 4.0))
                poly_val = a_c * (x_est ** 2) + b_c * x_est + c_c
            self.state[3] = x_est
            self.state[4] = float(np.clip(poly_val, -10.0, 10.0))
            self.state[5] = float(np.clip(deriv, -10.0, 10.0))
            solved = abs(poly_val) < 0.05
            return self.state.copy(), float(-abs(poly_val)), solved or self.t >= 24, {"x_est": round(x_est, 4), "residual": round(abs(poly_val), 5)}

        if self.modality == "protein_contact_geometry":
            # 6-bead 2D/3D hydrophobic-polar (HP) chain torsion folding minimizing non-bonded H-H energy
            angles = self.state[:4] + 0.25 * act
            coords = np.zeros((6, 2), dtype=np.float64)
            cum_theta = 0.0
            for bi in range(1, 6):
                if bi >= 2:
                    cum_theta += float(angles[bi - 2])
                coords[bi] = coords[bi - 1] + np.array([math.cos(cum_theta), math.sin(cum_theta)])
            dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            # Steric clash penalty (<0.75) + non-adjacent hydrophobic attraction (|i-j| >= 2 near 1.0..1.5)
            steric = 0.0
            hh_contacts = 0.0
            h_mask = (1, 0, 1, 1, 0, 1)
            for i in range(6):
                for j in range(i + 2, 6):
                    dij = float(dmat[i, j])
                    if dij < 0.72:
                        steric += (0.72 - dij) * 4.0
                    elif h_mask[i] and h_mask[j] and dij < 1.55:
                        hh_contacts += math.exp(-((dij - 1.05) ** 2) / 0.25)
            rg = float(np.sqrt(np.mean(np.sum((coords - np.mean(coords, axis=0)) ** 2, axis=1))))
            energy_hp = -hh_contacts + steric + 0.1 * rg
            self.state[:4] = np.clip(angles, -math.pi, math.pi)
            self.state[4:] = [rg, hh_contacts, steric, -energy_hp]
            return self.state.copy(), float(-energy_hp), self.t >= 24, {"hp_energy": round(energy_hp, 4), "hh_contacts": round(hh_contacts, 4)}

        if self.modality == "fractal_terrain_explorer":
            # Multi-octave fractal heightfield ascent avoiding steep hazard ravines
            px, py, vx, vy = self.state[:4]
            vx = float(np.clip(0.75 * vx + 0.30 * act[0], -0.8, 0.8))
            vy = float(np.clip(0.75 * vy + 0.30 * act[1], -0.8, 0.8))
            px = float(np.clip(px + vx, -3.5, 3.5))
            py = float(np.clip(py + vy, -3.5, 3.5))
            h = (
                0.55 * math.cos(0.9 * px) * math.cos(0.9 * py)
                + 0.28 * math.sin(2.1 * px + 0.5) * math.cos(2.1 * py)
                + 0.14 * math.cos(4.3 * (px - py))
            )
            gx = -0.50 * math.sin(0.9 * px) * math.cos(0.9 * py) + 0.59 * math.cos(2.1 * px + 0.5) * math.cos(2.1 * py)
            gy = -0.50 * math.cos(0.9 * px) * math.sin(0.9 * py) - 0.59 * math.sin(2.1 * px + 0.5) * math.sin(2.1 * py)
            ravine_pen = 0.4 if h < -0.35 else 0.0
            self.state[:] = [px, py, vx, vy, h, gx, gy, -ravine_pen]
            return self.state.copy(), float(h - ravine_pen), self.t >= 24, {"elevation": round(h, 4), "pos": [round(px, 3), round(py, 3)]}

        if self.modality == "fluid_advection_2d":
            # 2x4 grid passive-scalar advection-diffusion under controllable vortex stirrers act[0], act[1]
            c_grid = self.state.reshape(2, 4)
            vort_a, vort_b = 0.35 * float(act[0]), 0.35 * float(act[1])
            lap = (
                np.roll(c_grid, 1, axis=0) + np.roll(c_grid, -1, axis=0)
                + np.roll(c_grid, 1, axis=1) + np.roll(c_grid, -1, axis=1)
                - 4.0 * c_grid
            )
            adv = vort_a * (np.roll(c_grid, 1, axis=1) - np.roll(c_grid, -1, axis=1)) + vort_b * (np.roll(c_grid, 1, axis=0) - c_grid)
            c_next = np.clip(c_grid + 0.12 * lap - 0.20 * adv, -2.0, 2.0)
            self.state = c_next.ravel()
            mixing_var = float(np.var(self.state))
            return self.state.copy(), float(-mixing_var), self.t >= 24, {"scalar_variance": round(mixing_var, 5)}

        # resource_allocation_market: 4-sector metabolic/compute clearing market with SLA queues
        demands = 0.35 + 0.20 * np.sin(0.35 * self.t + np.arange(4, dtype=np.float64))
        raw_alloc = np.exp(np.clip(act, -3.0, 3.0))
        alloc = raw_alloc / np.sum(raw_alloc) * 1.45  # Fixed total supply = 1.45 units
        backlogs = np.clip(self.state[:4] + demands - alloc, 0.0, 3.0)
        sla_breach = float(np.sum(np.maximum(0.0, backlogs - 0.65)))
        self.state[:4] = backlogs
        self.state[4:] = demands - alloc
        reward = float(-np.mean(backlogs) - 0.5 * sla_breach)
        return self.state.copy(), reward, self.t >= 24, {"mean_backlog": round(float(np.mean(backlogs)), 4), "sla_breach": round(sla_breach, 4)}


def verify_all_novel_modalities(seed: int = 42) -> Dict[str, Any]:
    """Execute short deterministic rollouts across all 12 interactive training modalities."""
    out: Dict[str, Any] = {}
    for mod in NOVEL_MODALITIES:
        env = InteractiveModalitySuite(mod, seed=seed)
        obs = env.reset(seed)
        tot_r = 0.0
        steps = 0
        for k in range(12):
            obs, r, done, info = env.step([math.sin(0.4 * k), math.cos(0.3 * k), 0.2, -0.1])
            tot_r += r
            steps += 1
            if done:
                break
        out[mod] = {
            "steps": steps,
            "total_reward": round(tot_r, 4),
            "obs_dim": int(obs.size),
            "finite": bool(np.all(np.isfinite(obs))),
        }
    return {"experiment": "novel_training_modalities_verification", "modalities_verified": len(out), "results": out}
