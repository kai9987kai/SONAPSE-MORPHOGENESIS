"""SONAPSE FOUNDRY — 12 Non-Neural Biological System Computational Abstractions.

Explores how mechanisms from non-neural biological systems inspire new neural
computational structures. Strictly classified as ``SPECULATIVE_SYNTHETIC_BIOLOGY``
computational abstractions—explicitly distinguished from established biological
cellular mechanisms, with zero living-tissue or wet-lab genetic engineering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type
import numpy as np

from ..core.interchange import ObjectiveCategory


class BaseBioHybrid(ABC):
    """Common interface for non-neural biological-system computational abstractions."""
    hybrid_id: str = "base"
    biological_inspiration: str = ""
    objective_category: ObjectiveCategory = ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY
    caveat: str = "Synthetic computational abstraction; not a validated physiological or wet-lab biological model."

    def __init__(self, dim: int = 8, seed: int = 42) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        pass


class CardiacPacemakerHybrid(BaseBioHybrid):
    """1. Sinoatrial-inspired FitzHugh-Nagumo autonomous rhythmic pacemaker coordinating population phase."""
    hybrid_id = "cardiac_pacemaker"
    biological_inspiration = "Cardiac sinoatrial node pacemaker spontaneous depolarization & phase resetting"

    def reset(self) -> None:
        self.v = -0.8
        self.w = -0.5
        self.phase = 0.0

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        drive = float(np.mean(x))
        dt = 0.15
        # FitzHugh-Nagumo oscillator with tonic pacemaker current I0=0.45
        dv = self.v - (self.v ** 3) / 3.0 - self.w + 0.45 + 0.25 * drive
        dw = 0.08 * (self.v + 0.7 - 0.8 * self.w)
        self.v = float(np.clip(self.v + dt * dv, -3.0, 3.0))
        self.w = float(np.clip(self.w + dt * dw, -3.0, 3.0))
        self.phase = math.atan2(self.w, self.v)
        beat = 1.0 if self.v > 1.0 else 0.0
        return {"output": np.full(self.dim, self.v * 0.5), "pacemaker_v": round(self.v, 5), "beat": beat, "phase": round(self.phase, 4)}


class ImmuneRecognitionHybrid(BaseBioHybrid):
    """2. Adaptive-immune-inspired self/non-self affinity maturation & anomaly memory detector."""
    hybrid_id = "immune_recognition"
    biological_inspiration = "Clonal selection, negative selection of self-patterns, and immune memory cells"

    def reset(self) -> None:
        self.self_prototypes: List[np.ndarray] = []
        self.memory_antigens: List[np.ndarray] = []
        self.threshold = 0.65

    def train_self(self, normal_patterns: np.ndarray) -> None:
        for row in np.asarray(normal_patterns, dtype=np.float64):
            self.self_prototypes.append(row / (np.linalg.norm(row) + 1e-8))

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        v = np.asarray(x, dtype=np.float64).ravel()
        vn = v / (np.linalg.norm(v) + 1e-8)
        if not self.self_prototypes:
            self.self_prototypes.append(vn.copy())
        max_self_sim = max(float(np.dot(vn, p)) for p in self.self_prototypes)
        max_mem_sim = max((float(np.dot(vn, m)) for m in self.memory_antigens), default=0.0)
        anomaly_score = float(np.clip(1.0 - max_self_sim + 0.5 * max_mem_sim, 0.0, 2.0))
        is_abnormal = anomaly_score > self.threshold
        if is_abnormal and len(self.memory_antigens) < 32 and max_mem_sim < 0.9:
            self.memory_antigens.append(vn.copy())
        # Suppress anomalous patterns proportionally
        out = v * (0.2 if is_abnormal else 1.0)
        return {"output": out, "anomaly_score": round(anomaly_score, 5), "flagged_abnormal": is_abnormal, "memory_cells": len(self.memory_antigens)}


class EndocrineSignallingHybrid(BaseBioHybrid):
    """3. Multi-timescale slow humoral hormone secretion with negative-feedback hypothalamic axis."""
    hybrid_id = "endocrine_signalling"
    biological_inspiration = "Slow endocrine humoral hormones with negative-feedback regulation"

    def reset(self) -> None:
        self.hormones = np.array([0.2, 0.2, 0.2], dtype=np.float64)  # metabolic, arousal, recovery

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        stress = float(np.std(x))
        energy = float(np.mean(np.abs(x)))
        # Slow ODE with negative feedback
        dh0 = 0.04 * (energy - self.hormones[0]) - 0.02 * self.hormones[0]
        dh1 = 0.06 * (stress - self.hormones[1]) - 0.03 * self.hormones[2]
        dh2 = 0.03 * (self.hormones[1] - self.hormones[2])
        self.hormones = np.clip(self.hormones + np.array([dh0, dh1, dh2]), 0.0, 2.0)
        gain = float(1.0 + 0.3 * self.hormones[1] - 0.25 * self.hormones[2])
        return {"output": np.tanh(x * gain), "hormones": [round(float(h), 4) for h in self.hormones], "gain": round(gain, 4)}


class AstrocyticRegulationHybrid(BaseBioHybrid):
    """4. Tripartite-synapse astrocytic calcium reservoir & gliotransmitter gain clamping."""
    hybrid_id = "astrocytic_regulation"
    biological_inspiration = "Astrocytic tripartite synapse calcium buffering and heterosynaptic homeostasis"

    def reset(self) -> None:
        self.ca2 = 0.1
        self.glutamate_clearance = 1.0

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        mean_act = float(np.mean(np.abs(x)))
        self.ca2 = float(np.clip(0.92 * self.ca2 + 0.15 * mean_act, 0.0, 3.0))
        self.glutamate_clearance = float(np.clip(1.0 / (1.0 + 0.4 * max(0.0, self.ca2 - 0.8)), 0.3, 1.2))
        return {"output": x * self.glutamate_clearance, "ca2": round(self.ca2, 4), "clearance_gain": round(self.glutamate_clearance, 4)}


class CochlearFilterbankHybrid(BaseBioHybrid):
    """5. Tonotopic basilar-membrane gammatone-like bandpass bank for temporal signals."""
    hybrid_id = "cochlear_mechanics"
    biological_inspiration = "Cochlear basilar-membrane tonotopic traveling-wave frequency decomposition"

    def reset(self) -> None:
        self.freqs = np.linspace(0.08, 0.95, self.dim)
        self.re = np.zeros(self.dim, dtype=np.float64)
        self.im = np.zeros(self.dim, dtype=np.float64)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        s = float(np.mean(x))
        decay = 0.85
        new_re = decay * (self.re * np.cos(self.freqs) - self.im * np.sin(self.freqs)) + (1.0 - decay) * s
        new_im = decay * (self.re * np.sin(self.freqs) + self.im * np.cos(self.freqs))
        self.re, self.im = new_re, new_im
        env = np.sqrt(self.re ** 2 + self.im ** 2)
        return {"output": env, "dominant_band": int(np.argmax(env))}


class RetinalCenterSurroundHybrid(BaseBioHybrid):
    """6. Retinal ON/OFF center-surround spatial contrast & transient adaptation."""
    hybrid_id = "retinal_processing"
    biological_inspiration = "Retinal bipolar/ganglion center-surround spatial contrast & light adaptation"

    def reset(self) -> None:
        self.bg_adapt = np.zeros(self.dim, dtype=np.float64)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        arr = np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        surround = 0.5 * (np.roll(arr, 1) + np.roll(arr, -1))
        dog = arr - 0.65 * surround - 0.35 * self.bg_adapt
        self.bg_adapt = 0.9 * self.bg_adapt + 0.1 * arr
        return {"output": np.tanh(dog), "contrast_energy": round(float(np.mean(dog ** 2)), 5)}


class MuscleMechanicalAdapterHybrid(BaseBioHybrid):
    """7. Hill-type force-velocity & fatigue-hypertrophy actuator dynamics."""
    hybrid_id = "muscle_adaptation"
    biological_inspiration = "Skeletal muscle Hill-type viscoelasticity, fatigue, and usage-driven hypertrophy"

    def reset(self) -> None:
        self.strength = 1.0
        self.fatigue = 0.0
        self.tension = np.zeros(self.dim, dtype=np.float64)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        arr = np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        load = float(np.mean(np.abs(arr)))
        self.fatigue = float(np.clip(self.fatigue + 0.04 * load - 0.025 * (1.0 - load), 0.0, 0.85))
        if load > 0.4 and self.fatigue < 0.6:
            self.strength = min(2.0, self.strength + 0.004)
        eff = self.strength * (1.0 - 0.7 * self.fatigue)
        self.tension = 0.6 * self.tension + 0.4 * eff * np.tanh(arr)
        return {"output": self.tension.copy(), "strength": round(self.strength, 4), "fatigue": round(self.fatigue, 4)}


class VascularResourceDistributorHybrid(BaseBioHybrid):
    """8. Neurovascular coupling allocating metabolic flow dynamically toward high-demand subcircuits."""
    hybrid_id = "vascular_resource_distribution"
    biological_inspiration = "Functional hyperemia / neurovascular blood-flow redistribution"

    def reset(self) -> None:
        self.flow = np.full(self.dim, 1.0 / max(self.dim, 1), dtype=np.float64)
        self.total_supply = float(self.dim)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        arr = np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        demand = np.abs(arr) + 0.05
        target_flow = demand / np.sum(demand)
        # Lagged hemodynamic dilation response
        self.flow = 0.82 * self.flow + 0.18 * target_flow
        supply = self.flow * self.total_supply
        return {"output": arr * np.clip(supply, 0.2, 2.0), "flow_entropy": round(float(-np.sum(self.flow * np.log(self.flow + 1e-12))), 4)}


class CellularRepairHybrid(BaseBioHybrid):
    """9. Autophagy / chaperone-inspired outlier-state repair & noise scrubbing."""
    hybrid_id = "cellular_repair"
    biological_inspiration = "Intracellular molecular chaperones and damage-triggered repair"

    def reset(self) -> None:
        self.template = np.zeros(self.dim, dtype=np.float64)
        self.repairs = 0

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        arr = np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        broken = (~np.isfinite(arr)) | (np.abs(arr) > 3.0)
        if np.any(broken):
            self.repairs += int(np.sum(broken))
            arr = np.where(broken, self.template, arr)
        self.template = 0.95 * self.template + 0.05 * np.clip(arr, -2.0, 2.0)
        return {"output": arr, "repair_count": self.repairs}


class DevelopmentalDifferentiatorHybrid(BaseBioHybrid):
    """10. Morphogen-gradient Turing/French-flag gene-regulatory fate commitment."""
    hybrid_id = "developmental_differentiation"
    biological_inspiration = "Embryonic morphogen gradients and toggle-switch fate commitment"

    def reset(self) -> None:
        self.fate_a = np.full(self.dim, 0.5)
        self.fate_b = np.full(self.dim, 0.5)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        grad = np.linspace(-1.0, 1.0, self.dim) + 0.3 * np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        # Mutual-repression genetic toggle switch
        da = 0.15 * (1.0 / (1.0 + np.exp(-(3.0 * grad - 2.0 * self.fate_b))) - self.fate_a)
        db = 0.15 * (1.0 / (1.0 + np.exp(-(-3.0 * grad - 2.0 * self.fate_a))) - self.fate_b)
        self.fate_a = np.clip(self.fate_a + da, 0.0, 1.0)
        self.fate_b = np.clip(self.fate_b + db, 0.0, 1.0)
        return {"output": self.fate_a - self.fate_b, "polarization": round(float(np.mean(np.abs(self.fate_a - self.fate_b))), 4)}


class ChemotaxisGradientSensorHybrid(BaseBioHybrid):
    """11. Bacterial chemotaxis temporal derivative & exact adaptation (Barkai-Leibler style)."""
    hybrid_id = "chemical_gradient_sensing"
    biological_inspiration = "Chemotaxis receptor methylation providing integral exact adaptation"

    def reset(self) -> None:
        self.methylation = np.zeros(self.dim, dtype=np.float64)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        arr = np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        activity = np.tanh(arr - self.methylation)
        # Integral feedback brings steady-state activity back to 0, reporting gradients only
        self.methylation += 0.18 * activity
        return {"output": activity, "mean_gradient": round(float(np.mean(activity)), 5)}


class IntracellularSignallingCascadeHybrid(BaseBioHybrid):
    """12. MAPK-like 3-tier ultrasensitive phosphorylation cascade amplifying weak sustained cues."""
    hybrid_id = "intracellular_signalling_cascade"
    biological_inspiration = "MAPK three-tier ultrasensitive enzymatic phosphorylation cascade"

    def reset(self) -> None:
        self.k1 = np.zeros(self.dim, dtype=np.float64)
        self.k2 = np.zeros(self.dim, dtype=np.float64)
        self.k3 = np.zeros(self.dim, dtype=np.float64)

    @staticmethod
    def _hill(u: np.ndarray, k_half: float = 0.35, n: float = 3.0) -> np.ndarray:
        up = np.maximum(u, 0.0) ** n
        return up / (up + k_half ** n + 1e-12)

    def step(self, x: np.ndarray, context: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
        arr = np.resize(np.asarray(x, dtype=np.float64).ravel(), self.dim)
        sign = np.sign(arr)
        self.k1 = 0.7 * self.k1 + 0.3 * self._hill(np.abs(arr), 0.3, 2.0)
        self.k2 = 0.7 * self.k2 + 0.3 * self._hill(self.k1, 0.4, 3.0)
        self.k3 = 0.7 * self.k3 + 0.3 * self._hill(self.k2, 0.45, 4.0)
        return {"output": sign * self.k3, "cascade_gain": round(float(np.mean(self.k3)), 4)}


BIO_HYBRID_REGISTRY: Dict[str, Type[BaseBioHybrid]] = {
    cls.hybrid_id: cls
    for cls in (
        CardiacPacemakerHybrid,
        ImmuneRecognitionHybrid,
        EndocrineSignallingHybrid,
        AstrocyticRegulationHybrid,
        CochlearFilterbankHybrid,
        RetinalCenterSurroundHybrid,
        MuscleMechanicalAdapterHybrid,
        VascularResourceDistributorHybrid,
        CellularRepairHybrid,
        DevelopmentalDifferentiatorHybrid,
        ChemotaxisGradientSensorHybrid,
        IntracellularSignallingCascadeHybrid,
    )
}


def run_bio_hybrid_assays(seed: int = 42) -> Dict[str, Any]:
    """Run concrete verification assays on cardiac synchrony and immune anomaly detection."""
    rng = np.random.default_rng(seed)
    # 1. Cardiac pacemaker rhythmic coordination
    pm = CardiacPacemakerHybrid(dim=6, seed=seed)
    beats = 0
    vs = []
    for _ in range(120):
        r = pm.step(rng.normal(0.0, 0.1, size=6))
        beats += int(r["beat"] > 0.5)
        vs.append(r["pacemaker_v"])

    # 2. Immune familiar vs abnormal pattern discrimination
    imm = ImmuneRecognitionHybrid(dim=8, seed=seed)
    base_patterns = np.vstack([
        np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=float),
        np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=float),
        np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=float),
    ])
    imm.train_self(base_patterns)
    tp = fp = tn = fn = 0
    for _ in range(30):
        # Normal noisy pattern
        p = base_patterns[int(rng.integers(0, len(base_patterns)))] + rng.normal(0.0, 0.12, size=8)
        if imm.step(p)["flagged_abnormal"]:
            fp += 1
        else:
            tn += 1
        # Abnormal orthogonal/inverted pattern
        abn = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=float) + rng.normal(0.0, 0.12, size=8)
        if imm.step(abn)["flagged_abnormal"]:
            tp += 1
        else:
            fn += 1

    return {
        "objective_category": ObjectiveCategory.SPECULATIVE_SYNTHETIC_BIOLOGY.value,
        "registered_hybrids": list(BIO_HYBRID_REGISTRY.keys()),
        "cardiac_pacemaker_assay": {
            "steps": 120,
            "beat_steps": beats,
            "v_min": round(float(min(vs)), 4),
            "v_max": round(float(max(vs)), 4),
            "oscillatory": bool(max(vs) - min(vs) > 1.0),
        },
        "immune_recognition_assay": {
            "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "memory_cells_formed": len(imm.memory_antigens),
        },
    }
