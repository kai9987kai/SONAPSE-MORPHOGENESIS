"""SONAPSE FOUNDRY — Modular Library of 14 Synthetic Cellular Organelles.

All 14 organelles expose a uniform, independently testable interface:
- ``step(compartment_state, external_drive, reward, modulators, adapt)`` -> dict with ``current``
- ``reset()``
- ``mutate(rng, scale)`` -> returns mutated OrganelleAttachment params
- ``state_dict()``
- ``resource_cost`` & ``compatible_kinds``

Explicit epistemic label: SPECULATIVE_SYNTHETIC_BIOLOGY (computational cellular
abstractions, not biological organelles).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type
import numpy as np

from ..core.interchange import CompartmentKind, OrganelleAttachment


class BaseOrganelle(ABC):
    """Standardized interface for all attachable synthetic organelles."""

    organelle_type: str = "base"
    description: str = ""
    compatible_kinds: Tuple[CompartmentKind, ...] = tuple(CompartmentKind)
    base_resource_cost: float = 0.25

    def __init__(self, attachment: OrganelleAttachment, seed: int = 42) -> None:
        self.attachment = attachment
        self.size_scale = float(attachment.size_scale)
        self.sensitivity = float(attachment.sensitivity)
        self.params = dict(attachment.params)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def resource_cost(self) -> float:
        return round(self.base_resource_cost * self.size_scale, 4)

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(
        self,
        compartment_state: float,
        external_drive: float,
        reward: float = 0.0,
        modulators: Optional[Mapping[str, float]] = None,
        adapt: bool = False,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        pass

    def mutate_params(self, rng: np.random.Generator, scale: float = 0.15) -> Dict[str, Any]:
        out = dict(self.params)
        for k, v in out.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(np.clip(v + rng.normal(0.0, scale * max(abs(v), 0.1)), 0.01, 20.0))
        return out


class MnemosomeOrganelle(BaseOrganelle):
    """Associative memory organelle maintaining transient + consolidated activity traces."""
    organelle_type = "mnemosome"
    description = "Internal fast/slow associative memory trace organelle."
    base_resource_cost = 0.45

    def reset(self) -> None:
        self.slots = int(self.params.get("slots", 8))
        self.fast_decay = float(self.params.get("fast_decay", 0.88))
        self.slow_decay = float(self.params.get("slow_decay", 0.992))
        self.keys = np.zeros((self.slots, 2), dtype=np.float64)
        self.vals_fast = np.zeros(self.slots, dtype=np.float64)
        self.vals_slow = np.zeros(self.slots, dtype=np.float64)
        self.ptr = 0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        q = np.array([compartment_state, external_drive], dtype=np.float64)
        self.vals_fast *= self.fast_decay
        self.vals_slow *= self.slow_decay
        # Similarity retrieval
        dists = np.linalg.norm(self.keys - q[None, :], axis=1) + 1e-4
        w = np.exp(-2.5 * dists)
        w /= (np.sum(w) + 1e-9)
        recalled = float(np.dot(w, 0.55 * self.vals_fast + 0.45 * self.vals_slow))
        # Write transient trace and consolidate salient/rewarded events
        salience = abs(external_drive) + abs(reward) + float((modulators or {}).get("ach", 0.0))
        if salience > 0.35 or adapt:
            self.keys[self.ptr] = q
            self.vals_fast[self.ptr] = compartment_state
            if abs(reward) > 0.1 or salience > 0.9:
                self.vals_slow[self.ptr] = 0.7 * self.vals_slow[self.ptr] + 0.3 * compartment_state
            self.ptr = (self.ptr + 1) % self.slots
        current = float(np.clip(0.35 * self.sensitivity * recalled, -1.5, 1.5))
        return {"current": current, "recalled": round(recalled, 5), "slow_norm": round(float(np.linalg.norm(self.vals_slow)), 5)}

    def state_dict(self) -> Dict[str, Any]:
        return {"vals_fast": self.vals_fast.tolist(), "vals_slow": self.vals_slow.tolist(), "ptr": self.ptr}


class PredictonOrganelle(BaseOrganelle):
    """Local single-compartment autoregressive next-state predictor."""
    organelle_type = "predicton"
    description = "Predicts next local compartment state from recent activity & drive."
    base_resource_cost = 0.30

    def reset(self) -> None:
        self.lr = float(self.params.get("lr", 0.08))
        self.w = np.array([0.65, 0.25, 0.0], dtype=np.float64)
        self.last_pred = 0.0
        self.last_feat = np.zeros(3, dtype=np.float64)
        self.last_error = 0.0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        err = compartment_state - self.last_pred
        self.last_error = float(err)
        if adapt or abs(err) > 0.05:
            self.w = np.clip(self.w + self.lr * err * self.last_feat, -3.0, 3.0)
        feat = np.array([compartment_state, external_drive, 1.0], dtype=np.float64)
        pred = float(np.tanh(np.dot(self.w, feat)))
        self.last_feat = feat
        self.last_pred = pred
        current = float(np.clip(0.2 * self.sensitivity * pred, -1.0, 1.0))
        return {"current": current, "prediction": round(pred, 5), "pred_error": round(err, 5)}

    def state_dict(self) -> Dict[str, Any]:
        return {"w": self.w.tolist(), "last_pred": self.last_pred, "last_error": self.last_error}


class ContrastonOrganelle(BaseOrganelle):
    """Compares expected (EMA) vs observed activity to emit local prediction-error signals."""
    organelle_type = "contraston"
    description = "Computes signed contrast between expected and observed compartment activity."
    base_resource_cost = 0.20

    def reset(self) -> None:
        self.ema_alpha = float(self.params.get("ema_alpha", 0.15))
        self.expected = 0.0
        self.contrast = 0.0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        self.contrast = compartment_state - self.expected
        self.expected = (1.0 - self.ema_alpha) * self.expected + self.ema_alpha * compartment_state
        current = float(np.clip(0.30 * self.sensitivity * self.contrast, -1.5, 1.5))
        return {"current": current, "expected": round(self.expected, 5), "contrast": round(self.contrast, 5)}

    def state_dict(self) -> Dict[str, Any]:
        return {"expected": self.expected, "contrast": self.contrast}


class ChronosomeOrganelle(BaseOrganelle):
    """Adjustable temporal delay line organelle supporting multi-lag echo taps."""
    organelle_type = "chronosome"
    description = "Programmable intracellular delay buffer for temporal coincidence detection."
    base_resource_cost = 0.30

    def reset(self) -> None:
        self.max_delay = max(2, int(self.params.get("max_delay", 12)))
        self.tap_delay = min(self.max_delay - 1, max(1, int(self.params.get("tap_delay", 4))))
        self.buf: deque[float] = deque([0.0] * self.max_delay, maxlen=self.max_delay)

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        delayed = float(self.buf[-self.tap_delay])
        self.buf.append(compartment_state)
        if adapt and abs(reward) > 0.25:
            # Shift tap slightly toward higher reward correlation
            if reward < 0:
                self.tap_delay = 1 + (self.tap_delay % (self.max_delay - 1))
        current = float(np.clip(0.35 * self.sensitivity * delayed, -1.5, 1.5))
        return {"current": current, "delayed_signal": round(delayed, 5), "tap_delay": self.tap_delay}

    def state_dict(self) -> Dict[str, Any]:
        return {"tap_delay": self.tap_delay, "buf": list(self.buf)}


class AttentosomeOrganelle(BaseOrganelle):
    """Selective gain amplifier prioritizing high-salience or high-SNR inputs."""
    organelle_type = "attentosome"
    description = "Context- and salience-gated selective input amplification organelle."
    base_resource_cost = 0.25

    def reset(self) -> None:
        self.running_var = 0.1
        self.gain = 1.0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        ach = float((modulators or {}).get("ach", 0.0))
        self.running_var = 0.92 * self.running_var + 0.08 * (external_drive ** 2)
        snr = abs(external_drive) / (math.sqrt(self.running_var) + 0.15)
        self.gain = float(np.clip(1.0 / (1.0 + math.exp(-(snr - 0.8 + 0.5 * ach))), 0.1, 2.0))
        current = float(np.clip(0.35 * self.sensitivity * (self.gain - 0.5) * external_drive, -1.5, 1.5))
        return {"current": current, "attention_gain": round(self.gain, 4), "snr": round(snr, 4)}

    def state_dict(self) -> Dict[str, Any]:
        return {"running_var": self.running_var, "gain": self.gain}


class RoutosomeOrganelle(BaseOrganelle):
    """Context-sensitive signal routing organelle modulating channel preference."""
    organelle_type = "routosome"
    description = "Computes soft routing weights selecting between feed-forward and recurrent paths."
    base_resource_cost = 0.25

    def reset(self) -> None:
        self.route_logits = np.zeros(2, dtype=np.float64)
        self.route_probs = np.array([0.5, 0.5], dtype=np.float64)

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        ctx = float((modulators or {}).get("ne", 0.0)) + 0.5 * compartment_state
        self.route_logits = np.array([external_drive + ctx, -external_drive + 0.5 * compartment_state])
        ex = np.exp(np.clip(self.route_logits - np.max(self.route_logits), -20, 20))
        self.route_probs = ex / np.sum(ex)
        routed = float(self.route_probs[0] * external_drive - self.route_probs[1] * 0.25 * compartment_state)
        return {"current": float(np.clip(0.3 * self.sensitivity * routed, -1.5, 1.5)), "route_probs": [round(float(x), 4) for x in self.route_probs]}

    def state_dict(self) -> Dict[str, Any]:
        return {"route_probs": self.route_probs.tolist()}


class PlasticityNucleusOrganelle(BaseOrganelle):
    """Local meta-plasticity regulator controlling cellular learning rate & eligibility."""
    organelle_type = "plasticity_nucleus"
    description = "Regulates local synaptic plasticity gate based on surprise and dopamine."
    base_resource_cost = 0.30

    def reset(self) -> None:
        self.eligibility = 0.0
        self.plasticity_gate = 0.5

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        da = float((modulators or {}).get("da", 0.0))
        self.eligibility = 0.85 * self.eligibility + 0.15 * (compartment_state * external_drive)
        self.plasticity_gate = float(np.clip(0.5 + 0.4 * math.tanh(da + reward), 0.05, 1.5))
        current = float(np.clip(0.12 * self.sensitivity * self.plasticity_gate * self.eligibility, -1.0, 1.0))
        return {"current": current, "eligibility": round(self.eligibility, 5), "plasticity_gate": round(self.plasticity_gate, 4)}

    def state_dict(self) -> Dict[str, Any]:
        return {"eligibility": self.eligibility, "plasticity_gate": self.plasticity_gate}


class CheckpointVesicleOrganelle(BaseOrganelle):
    """Reversible computational state preservation vesicle for rapid state rollback."""
    organelle_type = "checkpoint_vesicle"
    description = "Snapshots high-value compartment states and restores them on sharp error drops."
    base_resource_cost = 0.25

    def reset(self) -> None:
        self.saved_state = 0.0
        self.saved_reward = -1e9
        self.restores = 0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        if reward > self.saved_reward:
            self.saved_reward = float(reward)
            self.saved_state = float(compartment_state)
        current = 0.0
        if reward < -0.5 and self.saved_reward > 0.0:
            current = float(np.clip(0.4 * self.sensitivity * (self.saved_state - compartment_state), -1.5, 1.5))
            self.restores += 1
        return {"current": current, "saved_state": round(self.saved_state, 5), "restores": self.restores}

    def state_dict(self) -> Dict[str, Any]:
        return {"saved_state": self.saved_state, "saved_reward": self.saved_reward, "restores": self.restores}


class RepairosomeOrganelle(BaseOrganelle):
    """Compensatory structural & gain adaptation when compartment drops silent or saturates."""
    organelle_type = "repairosome"
    description = "Detects hypo/hyper-activity damage and injects homeostatic compensatory gain."
    base_resource_cost = 0.30

    def reset(self) -> None:
        self.activity_ema = 0.2
        self.compensation = 0.0
        self.repair_events = 0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        self.activity_ema = 0.92 * self.activity_ema + 0.08 * abs(compartment_state)
        if self.activity_ema < 0.04:
            self.compensation = min(1.0, self.compensation + 0.05)
            self.repair_events += 1
        elif self.activity_ema > 0.92:
            self.compensation = max(-1.0, self.compensation - 0.05)
            self.repair_events += 1
        else:
            self.compensation *= 0.98
        current = float(np.clip(0.35 * self.sensitivity * self.compensation * (external_drive if abs(external_drive) > 1e-4 else 0.2), -1.2, 1.2))
        return {"current": current, "activity_ema": round(self.activity_ema, 4), "compensation": round(self.compensation, 4), "repair_events": self.repair_events}

    def state_dict(self) -> Dict[str, Any]:
        return {"activity_ema": self.activity_ema, "compensation": self.compensation, "repair_events": self.repair_events}


class UncertaintosomeOrganelle(BaseOrganelle):
    """Local prediction-reliability & epistemic dispersion estimator."""
    organelle_type = "uncertaintosome"
    description = "Tracks rolling squared residual variance to estimate local state uncertainty."
    base_resource_cost = 0.25

    def reset(self) -> None:
        self.mean = 0.0
        self.m2 = 0.05
        self.last_state = 0.0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        delta = compartment_state - self.last_state
        self.mean = 0.9 * self.mean + 0.1 * delta
        self.m2 = 0.9 * self.m2 + 0.1 * ((delta - self.mean) ** 2)
        self.last_state = compartment_state
        uncertainty = math.sqrt(max(self.m2, 1e-8))
        # Dampens overconfident saturation when uncertainty is high
        current = float(np.clip(-0.18 * self.sensitivity * uncertainty * compartment_state, -1.0, 1.0))
        return {"current": current, "uncertainty": round(uncertainty, 5)}

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "m2": self.m2}


class TranslatorMembraneOrganelle(BaseOrganelle):
    """Converts between rate, spike, and latent neural signal formats across mixed fidelities."""
    organelle_type = "translator_membrane"
    description = "Bidirectional format converter bridging rate, spiking, and ML vector encodings."
    base_resource_cost = 0.20

    def reset(self) -> None:
        self.rate_filter = 0.0
        self.spike_accum = 0.0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        self.rate_filter = 0.8 * self.rate_filter + 0.2 * math.tanh(external_drive)
        self.spike_accum += max(0.0, compartment_state)
        emitted_pulse = 1.0 if self.spike_accum >= 1.0 else 0.0
        if emitted_pulse > 0.0:
            self.spike_accum -= 1.0
        current = float(np.clip(0.25 * self.sensitivity * (self.rate_filter + 0.4 * emitted_pulse), -1.2, 1.2))
        return {"current": current, "translated_rate": round(self.rate_filter, 5), "translated_pulse": emitted_pulse}

    def state_dict(self) -> Dict[str, Any]:
        return {"rate_filter": self.rate_filter, "spike_accum": self.spike_accum}


class PhaseOrganelle(BaseOrganelle):
    """Oscillatory phase-coupled resonator organelle encoding frequency & phase."""
    organelle_type = "phase_organelle"
    description = "Internal Kuramoto/Hopf oscillator supporting phase-coherent gating."
    base_resource_cost = 0.30

    def reset(self) -> None:
        self.omega = float(self.params.get("omega", 0.35))
        self.phase = float(self.params.get("init_phase", 0.0))

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        # Phase-locking perturbation from external drive
        self.phase = (self.phase + self.omega + 0.25 * external_drive * math.cos(self.phase)) % (2.0 * math.pi)
        osc = math.sin(self.phase)
        current = float(np.clip(0.28 * self.sensitivity * osc, -1.2, 1.2))
        return {"current": current, "phase_rad": round(self.phase, 5), "osc": round(osc, 5)}

    def state_dict(self) -> Dict[str, Any]:
        return {"omega": self.omega, "phase": self.phase}


class ResourceRegulatorOrganelle(BaseOrganelle):
    """Local metabolic budget controller throttling high-cost firing when reserves drop."""
    organelle_type = "resource_regulator"
    description = "Maintains local energy reserve and suppresses runaway firing under scarcity."
    base_resource_cost = 0.15

    def reset(self) -> None:
        self.reserve = 1.0

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        cost = 0.06 * (compartment_state ** 2)
        self.reserve = float(np.clip(self.reserve - cost + 0.035 * (1.0 - self.reserve), 0.05, 1.0))
        throttle = -0.35 * (1.0 - self.reserve) * compartment_state
        return {"current": float(np.clip(self.sensitivity * throttle, -1.0, 1.0)), "reserve": round(self.reserve, 4)}

    def state_dict(self) -> Dict[str, Any]:
        return {"reserve": self.reserve}


class DevelopmentalGenomeOrganelle(BaseOrganelle):
    """Bounded intracellular differentiation rule engine guiding phenotype maturation."""
    organelle_type = "developmental_genome"
    description = "Tracks cellular maturation age and shifts excitability/phenotype cleanly."
    base_resource_cost = 0.20

    def reset(self) -> None:
        self.age_steps = 0
        self.maturation_horizon = int(self.params.get("maturation_horizon", 60))
        self.phenotype_stage = "progenitor"

    def step(self, compartment_state: float, external_drive: float, reward: float = 0.0,
             modulators: Optional[Mapping[str, float]] = None, adapt: bool = False) -> Dict[str, Any]:
        self.age_steps += 1
        frac = min(1.0, self.age_steps / max(self.maturation_horizon, 1))
        if frac < 0.33:
            self.phenotype_stage = "progenitor"
            bias = 0.12 * external_drive
        elif frac < 0.75:
            self.phenotype_stage = "differentiating"
            bias = 0.08 * (external_drive + 0.5 * compartment_state)
        else:
            self.phenotype_stage = "mature"
            bias = -0.05 * compartment_state + 0.15 * math.tanh(compartment_state)
        return {"current": float(np.clip(self.sensitivity * bias, -1.0, 1.0)), "stage": self.phenotype_stage, "maturation": round(frac, 4)}

    def state_dict(self) -> Dict[str, Any]:
        return {"age_steps": self.age_steps, "phenotype_stage": self.phenotype_stage}


ORGANELLE_REGISTRY: Dict[str, Type[BaseOrganelle]] = {
    cls.organelle_type: cls
    for cls in (
        MnemosomeOrganelle,
        PredictonOrganelle,
        ContrastonOrganelle,
        ChronosomeOrganelle,
        AttentosomeOrganelle,
        RoutosomeOrganelle,
        PlasticityNucleusOrganelle,
        CheckpointVesicleOrganelle,
        RepairosomeOrganelle,
        UncertaintosomeOrganelle,
        TranslatorMembraneOrganelle,
        PhaseOrganelle,
        ResourceRegulatorOrganelle,
        DevelopmentalGenomeOrganelle,
    )
}


def create_organelle(attachment: OrganelleAttachment, seed: int = 42) -> BaseOrganelle:
    if attachment.organelle_type not in ORGANELLE_REGISTRY:
        raise KeyError(f"Unknown organelle_type: {attachment.organelle_type!r}. Valid: {sorted(ORGANELLE_REGISTRY)}")
    return ORGANELLE_REGISTRY[attachment.organelle_type](attachment, seed=seed)


def list_organelles() -> List[Dict[str, Any]]:
    return [
        {
            "organelle_type": k,
            "class_name": cls.__name__,
            "description": cls.description,
            "base_resource_cost": cls.base_resource_cost,
        }
        for k, cls in ORGANELLE_REGISTRY.items()
    ]


def evaluate_organelle_interaction(
    org_a: str,
    org_b: str,
    seed: int = 42,
    steps: int = 80,
) -> Dict[str, Any]:
    """Empirically measure synergy vs interference when combining two organelles on a delayed-signal probe."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=np.float64)
    signal = np.sin(0.25 * t) + 0.5 * np.cos(0.11 * t) + rng.normal(0.0, 0.15, size=steps)
    target = np.roll(signal, 4)
    target[:4] = 0.0

    def _run_combo(types: Sequence[str]) -> Tuple[float, float]:
        orgs = [create_organelle(OrganelleAttachment(f"o_{i}", ot, "soma"), seed=seed + i) for i, ot in enumerate(types)]
        state = 0.0
        feats = np.zeros((steps, 2 + len(orgs)), dtype=np.float64)
        total_cost = sum(o.resource_cost for o in orgs)
        for k in range(steps):
            drive = float(signal[k])
            cur = 0.0
            for j, o in enumerate(orgs):
                r = o.step(state, drive, adapt=True)
                cur += float(r["current"])
                feats[k, 2 + j] = float(r["current"])
            state = float(np.tanh(0.6 * state + 0.5 * drive + cur))
            feats[k, 0] = state
            feats[k, 1] = drive
        # Fit ridge readout on first 60%, evaluate MSE on last 40%
        split = int(0.6 * steps)
        X_tr, y_tr = feats[:split], target[:split]
        X_te, y_te = feats[split:], target[split:]
        w = np.linalg.solve(X_tr.T @ X_tr + 0.2 * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
        mse = float(np.mean((X_te @ w - y_te) ** 2))
        return mse, total_cost

    mse_none, _ = _run_combo([])
    mse_a, cost_a = _run_combo([org_a])
    mse_b, cost_b = _run_combo([org_b])
    mse_ab, cost_ab = _run_combo([org_a, org_b])

    gain_a = mse_none - mse_a
    gain_b = mse_none - mse_b
    gain_ab = mse_none - mse_ab
    synergy = gain_ab - max(gain_a, gain_b)
    return {
        "org_a": org_a,
        "org_b": org_b,
        "mse_baseline": round(mse_none, 6),
        "mse_a": round(mse_a, 6),
        "mse_b": round(mse_b, 6),
        "mse_ab": round(mse_ab, 6),
        "synergy_delta_mse": round(synergy, 6),
        "interaction_label": "synergistic" if synergy > 0.005 else ("interfering" if synergy < -0.005 else "neutral"),
        "combined_resource_cost": round(cost_ab, 4),
    }
