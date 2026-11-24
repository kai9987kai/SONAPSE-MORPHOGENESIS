"""SONAPSE FOUNDRY — 13 Software/ML-Inspired Hybrid Neurons & 3-Variant Comparison.

Investigates how computational mechanisms from ML/software can be physically
approximated inside a structured neuron or connectome.

All 13 mechanisms implemented:
 1. dendritic_attention         - Transformer-inspired dendritic attention
 2. moe_branch_specialisation   - Mixture-of-experts branch specialisation
 3. state_space_compartment     - Continuous-time selective state-space (SSM) compartment
 4. reservoir_organelle         - Intracellular echo-state reservoir organelle
 5. fast_weight_memory          - Hebbian outer-product fast-weight memory
 6. recurrent_gating            - GRU/LSTM-style cellular gating
 7. memristive_synapse          - Hysteretic memristive conductance memory
 8. error_correcting_pathway    - Redundant parity / majority error-correcting axons
 9. wave_photonic_abstraction   - Complex wave-interference optical/photonic unit
10. local_predictive_coding     - Hierarchical local prediction-error minimizer
11. differentiable_routing      - Soft Gumbel/softmax differentiable pathway router
12. associative_hopfield_memory - Modern continuous Hopfield / dense associative memory
13. external_ai_adapter         - Modular bounded adapter to external/Supermix models

For core mechanisms, provides ALL THREE requested variants:
  (1) exact_software            - exact mathematical implementation
  (2) cellular_compartment      - multi-compartment continuous cellular approximation
  (3) biologically_constrained  - Dale's-law signed, bounded-gain, noisy/conductance-constrained approximation

And empirically measures approximation error, task MSE, robustness under noise,
and CPU cost!
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(np.clip(x - np.max(x), -30.0, 30.0))
    return e / (np.sum(e) + 1e-12)


class MLHybridMechanism:
    """Base wrapper supporting exact_software, cellular_compartment, and biologically_constrained variants."""

    def __init__(self, mechanism: str, dim: int = 8, variant: str = "exact_software", seed: int = 42) -> None:
        if variant not in ("exact_software", "cellular_compartment", "biologically_constrained"):
            raise ValueError(f"Unsupported variant: {variant}")
        self.mechanism = mechanism
        self.dim = int(dim)
        self.variant = variant
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._init_params()

    def _init_params(self) -> None:
        d = self.dim
        self.W1 = self.rng.normal(0.0, 0.5 / math.sqrt(d), size=(d, d))
        self.W2 = self.rng.normal(0.0, 0.5 / math.sqrt(d), size=(d, d))
        self.W3 = self.rng.normal(0.0, 0.5 / math.sqrt(d), size=(d, d))
        # Dale's-law sign separation for biological variant (80% exc, 20% inh)
        signs = np.where(np.arange(d) < int(0.8 * d), 1.0, -1.0)
        self.dale_W1 = np.abs(self.W1) * signs[None, :]
        self.dale_W2 = np.abs(self.W2) * signs[None, :]
        self.reset()

    def reset(self) -> None:
        d = self.dim
        self.h = np.zeros(d, dtype=np.float64)
        self.M = np.zeros((d, d), dtype=np.float64)  # fast weights / memristive / Hopfield
        self.mem_keys = self.rng.normal(0.0, 0.5, size=(6, d))
        self.mem_vals = self.rng.normal(0.0, 0.5, size=(6, d))
        self.memristor_g = np.full(d, 0.5, dtype=np.float64)
        self.phase = np.zeros(d, dtype=np.float64)
        self.pred = np.zeros(d, dtype=np.float64)
        self.flops = 0

    def step(self, x: np.ndarray, context: Optional[np.ndarray] = None, noise_std: float = 0.0) -> np.ndarray:
        d = self.dim
        u = np.resize(np.asarray(x, dtype=np.float64).ravel(), d)
        c = np.resize(np.asarray(context, dtype=np.float64).ravel(), d) if context is not None else u
        if noise_std > 0.0:
            u = u + self.rng.normal(0.0, noise_std, size=d)

        m = self.mechanism
        v = self.variant

        if m == "dendritic_attention":
            if v == "exact_software":
                q = self.W1 @ c
                k = self.W2 @ u
                val = self.W3 @ u
                scores = np.outer(q, k) / math.sqrt(d)
                att = np.array([_softmax(row) for row in scores])
                out = att @ val
                self.flops += 6 * d * d
            elif v == "cellular_compartment":
                # Multiplicative apical-basal coincidence gating across compartments
                apical = np.tanh(self.W1 @ c)
                basal = np.tanh(self.W2 @ u)
                gate = 1.0 / (1.0 + np.exp(-3.0 * (apical * basal)))
                out = gate * np.tanh(self.W3 @ u)
                self.flops += 4 * d * d
            else:
                # Biologically constrained: Dale-signed conductances + NMDA Mg2+ block + ATP saturation
                apical = np.maximum(0.0, self.dale_W1 @ c)
                basal = np.maximum(0.0, self.dale_W2 @ u)
                nmda = (apical * basal) / (0.3 + apical * basal)
                out = np.tanh(0.8 * nmda + 0.2 * (self.dale_W1 @ u) + self.rng.normal(0.0, 0.015, size=d))
                self.flops += 4 * d * d

        elif m == "moe_branch_specialisation":
            # 4 branches / experts
            e_outs = np.stack([np.tanh(np.roll(self.W1, k, axis=0) @ u) for k in range(4)])
            logits = np.array([float(np.dot(self.W2[k % d], c)) for k in range(4)])
            if v == "exact_software":
                g = _softmax(logits)
                out = np.tensordot(g, e_outs, axes=1)
            elif v == "cellular_compartment":
                # Soft lateral inhibition between dendritic branches
                g = np.maximum(0.0, logits - 0.5 * np.mean(logits))
                g = g / (np.sum(g) + 1e-6) if np.sum(g) > 1e-6 else np.full(4, 0.25)
                out = np.tensordot(g, e_outs, axes=1)
            else:
                # Winner-take-most inhibitory interneuron competition (Dale-compliant)
                win = int(np.argmax(logits))
                out = np.tanh(0.85 * e_outs[win] + 0.15 * np.mean(e_outs, axis=0))
            self.flops += 5 * d * d

        elif m == "state_space_compartment":
            A = -np.linspace(0.15, 0.85, d)
            if v == "exact_software":
                delta = 0.25 / (1.0 + np.exp(-(self.W1 @ u)))
                a_bar = np.exp(delta * A)
                self.h = a_bar * self.h + delta * (self.W2 @ u)
                out = self.W3 @ self.h
            elif v == "cellular_compartment":
                tau_inv = np.clip(0.1 + 0.4 * (1.0 / (1.0 + np.exp(-(self.W1 @ u)))), 0.05, 0.8)
                self.h = (1.0 - tau_inv) * self.h + tau_inv * np.tanh(self.W2 @ u)
                out = self.h.copy()
            else:
                # Shunting conductance changes effective membrane tau_m = C / (g_L + g_syn)
                g_shunt = np.abs(self.dale_W1 @ u)
                decay = np.exp(-0.2 * (0.2 + g_shunt))
                self.h = decay * self.h + (1.0 - decay) * np.tanh(self.dale_W2 @ u)
                out = self.h.copy()
            self.flops += 4 * d * d

        elif m == "reservoir_organelle":
            W = self.W1 * 0.9 if v != "biologically_constrained" else self.dale_W1 * 0.45
            self.h = 0.35 * self.h + 0.65 * np.tanh(W @ self.h + self.W2 @ u)
            out = self.h.copy()
            self.flops += 2 * d * d

        elif m == "fast_weight_memory":
            k = np.tanh(self.W1 @ u)
            val = np.tanh(self.W2 @ c)
            q = np.tanh(self.W3 @ u)
            if v == "exact_software":
                self.M = 0.92 * self.M + 0.25 * np.outer(val, k)
                out = self.M @ q
            elif v == "cellular_compartment":
                self.M = np.clip(0.90 * self.M + 0.2 * np.outer(val, k), -2.0, 2.0)
                out = np.tanh(self.M @ q)
            else:
                # Bounded presynaptic short-term facilitation trace (non-negative potentiation)
                self.M = np.clip(0.88 * self.M + 0.18 * np.outer(np.maximum(val, 0), np.maximum(k, 0)), 0.0, 1.5)
                out = np.tanh(self.M @ np.maximum(q, 0) - 0.25 * np.mean(self.M))
            self.flops += 5 * d * d

        elif m == "recurrent_gating":
            if v == "exact_software":
                z = 1.0 / (1.0 + np.exp(-(self.W1 @ u + self.W2 @ self.h)))
                r = 1.0 / (1.0 + np.exp(-(self.W2 @ u + self.W1 @ self.h)))
                cand = np.tanh(self.W3 @ u + self.W1 @ (r * self.h))
                self.h = (1.0 - z) * self.h + z * cand
            elif v == "cellular_compartment":
                # Dual-compartment fast/slow gated integration
                g = 1.0 / (1.0 + np.exp(-2.0 * (self.W1 @ u)))
                self.h = (1.0 - 0.5 * g) * self.h + 0.5 * g * np.tanh(self.W2 @ u + 0.5 * self.h)
            else:
                # Shunting inhibition controls persistence
                shunt = 1.0 / (1.0 + np.abs(self.dale_W1 @ u))
                self.h = 0.85 * shunt * self.h + (1.0 - 0.85 * shunt) * np.tanh(self.dale_W2 @ u)
            out = self.h.copy()
            self.flops += 6 * d * d

        elif m == "memristive_synapse":
            # Hysteretic conductance state g in [0.1, 2.0]
            self.memristor_g = np.clip(self.memristor_g + 0.12 * u * np.tanh(self.h + 0.1) - 0.02 * (self.memristor_g - 0.5), 0.1, 2.0)
            self.h = np.tanh(self.memristor_g * u)
            out = self.h.copy()
            self.flops += 6 * d

        elif m == "error_correcting_pathway":
            # Triple-modular redundant branches with median/majority voting
            b1 = np.tanh(self.W1 @ u)
            b2 = np.tanh(self.W1 @ u + self.rng.normal(0, 0.05 if v == "biologically_constrained" else 0.01, size=d))
            b3 = np.tanh(self.W1 @ u)
            out = np.median(np.stack([b1, b2, b3], axis=0), axis=0)
            self.flops += 3 * d * d

        elif m == "wave_photonic_abstraction":
            # Complex-amplitude phase interference
            self.phase = (self.phase + 0.35 + 0.4 * u) % (2.0 * math.pi)
            wave = np.exp(1j * self.phase)
            mixed = (self.W1 + 1j * self.W2 * 0.3) @ wave
            out = np.real(mixed) * np.abs(mixed) / (1.0 + np.abs(mixed))
            self.flops += 4 * d * d

        elif m == "local_predictive_coding":
            err = u - self.pred
            self.h = np.tanh(0.75 * self.h + 0.45 * (self.W1 @ err))
            self.pred = np.tanh(self.W2 @ self.h)
            out = err if v == "exact_software" else self.h.copy()
            self.flops += 3 * d * d

        elif m == "differentiable_routing":
            probs = _softmax(self.W1 @ c)
            out = probs * np.tanh(self.W2 @ u)
            self.flops += 3 * d * d

        elif m == "associative_hopfield_memory":
            beta = 3.0 if v == "exact_software" else (2.0 if v == "cellular_compartment" else 1.2)
            sims = beta * (self.mem_keys @ u)
            w = _softmax(sims)
            out = w @ self.mem_vals
            self.flops += 2 * len(self.mem_keys) * d

        elif m == "external_ai_adapter":
            # Zero-centered residual projection bridging external embeddings into cellular range
            proj = np.tanh(self.W1 @ u)
            out = 0.5 * u + 0.5 * proj
            self.flops += 2 * d * d
        else:
            raise KeyError(f"Unknown ML hybrid mechanism: {m}")

        self.h = np.clip(self.h, -5.0, 5.0)
        return np.asarray(out, dtype=np.float64)


ML_HYBRID_MECHANISMS: Tuple[str, ...] = (
    "dendritic_attention",
    "moe_branch_specialisation",
    "state_space_compartment",
    "reservoir_organelle",
    "fast_weight_memory",
    "recurrent_gating",
    "memristive_synapse",
    "error_correcting_pathway",
    "wave_photonic_abstraction",
    "local_predictive_coding",
    "differentiable_routing",
    "associative_hopfield_memory",
    "external_ai_adapter",
)


def compare_three_variants(
    mechanisms: Sequence[str] = ("dendritic_attention", "state_space_compartment", "fast_weight_memory", "recurrent_gating", "moe_branch_specialisation"),
    dim: int = 8,
    steps: int = 80,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare exact_software vs cellular_compartment vs biologically_constrained across tasks & noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(steps, dim))
    C = np.roll(X, 2, axis=0)
    # Target: delayed nonlinear contextual interaction
    y = np.tanh(0.6 * np.sum(X[:, :4] * C[:, :4], axis=1) + 0.4 * X[:, 0])

    report: Dict[str, Any] = {}
    for mech in mechanisms:
        variant_outputs: Dict[str, np.ndarray] = {}
        variant_stats: Dict[str, Any] = {}
        for variant in ("exact_software", "cellular_compartment", "biologically_constrained"):
            unit = MLHybridMechanism(mech, dim=dim, variant=variant, seed=seed)
            t0 = time.perf_counter()
            clean_feats = np.stack([unit.step(X[t], C[t], noise_std=0.0) for t in range(steps)])
            wall_ms = (time.perf_counter() - t0) * 1000.0
            flops = unit.flops

            # Fit linear readout on clean train split, test on clean + noisy test split
            split = int(0.6 * steps)
            X_tr, y_tr = clean_feats[:split], y[:split]
            X_te, y_te = clean_feats[split:], y[split:]
            w = np.linalg.solve(X_tr.T @ X_tr + 0.2 * np.eye(dim), X_tr.T @ y_tr)
            clean_mse = float(np.mean((X_te @ w - y_te) ** 2))

            # Noise robustness test (sigma = 0.35)
            unit.reset()
            noisy_feats = np.stack([unit.step(X[t], C[t], noise_std=0.35) for t in range(steps)])
            noisy_mse = float(np.mean((noisy_feats[split:] @ w - y_te) ** 2))

            variant_outputs[variant] = clean_feats
            variant_stats[variant] = {
                "clean_test_mse": round(clean_mse, 6),
                "noisy_test_mse": round(noisy_mse, 6),
                "noise_degradation": round(noisy_mse - clean_mse, 6),
                "wall_ms": round(wall_ms, 3),
                "flops": int(flops),
            }

        # Approximation error relative to exact_software
        ref = variant_outputs["exact_software"]
        ref_norm = float(np.linalg.norm(ref) + 1e-8)
        for vname in ("cellular_compartment", "biologically_constrained"):
            approx_err = float(np.linalg.norm(variant_outputs[vname] - ref) / ref_norm)
            variant_stats[vname]["relative_approx_error_vs_exact"] = round(approx_err, 5)
        variant_stats["exact_software"]["relative_approx_error_vs_exact"] = 0.0
        report[mech] = variant_stats

    return {
        "experiment": "ml_hybrid_three_variant_comparison",
        "dim": dim,
        "steps": steps,
        "seed": seed,
        "mechanisms": report,
    }
