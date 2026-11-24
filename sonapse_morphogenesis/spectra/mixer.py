"""SONAPSE SPECTRA — Multimodal Neural Signal Encoder, 13-Mode Mixer & Anatomical Injector.

Converts images, audio, video, text, mathematical functions, protein structures,
3D meshes, neural recordings, connectome streams, chemical fields, and procedural
generators into rate, spiking, event, temporal, frequency, or latent neural
representations—mixes them with 13 documented operators—and injects them into
any targeted anatomical site with exact replayable receipts and before/after diffs.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from ..core.compartment_engine import MultiFidelityConnectome
from ..core.interchange import DataSourceKind, content_digest, make_id
from .sensory_library import ExpandedSensoryEncoder


SIGNAL_ENCODINGS: Tuple[str, ...] = ("rate", "spiking", "event", "temporal", "frequency", "latent")

MIXING_OPERATORS: Tuple[str, ...] = (
    "weighted_addition",
    "multiplicative_binding",
    "amplitude_modulation",
    "frequency_modulation",
    "phase_coupling",
    "temporal_interleaving",
    "spatial_remapping",
    "temporal_compression",
    "delay_distortion",
    "signal_inversion",
    "cross_modal_attention",
    "chemical_modulation",
    "learned_nonlinear",
)

INJECTION_TARGETS: Tuple[str, ...] = (
    "ordinary_sensory",
    "alternative_sensory",
    "selected_population",
    "random_population",
    "specific_neuron",
    "multiple_neurons",
    "individual_compartment",
    "dendrite",
    "secondary_soma",
    "synthetic_organelle",
    "cross_connectome_bridge",
)


def encode_multimodal_stream(
    modality_type: str,
    payload: Any,
    encoding: str = "rate",
    length: int = 32,
    channels: int = 8,
    source_kind: DataSourceKind = DataSourceKind.SYNTHETIC_SIMULATION,
    seed: int = 42,
) -> Dict[str, Any]:
    """Convert any supported user modality into a documented [T, C] simulated neural signal."""
    if encoding not in SIGNAL_ENCODINGS:
        raise ValueError(f"Unsupported encoding: {encoding!r}")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * math.pi, length)

    # 1. Produce normalized base temporal tensor [T, C]
    if modality_type == "text":
        txt = str(payload)
        ords = np.array([ord(ch) / 128.0 - 0.5 for ch in txt] or [0.0], dtype=np.float64)
        base = np.zeros((length, channels), dtype=np.float64)
        for ti in range(length):
            for ci in range(channels):
                base[ti, ci] = math.tanh(ords[(ti * channels + ci) % ords.size] * (1.0 + 0.1 * ci))
    elif modality_type in ("image", "video"):
        arr = np.asarray(payload, dtype=np.float64)
        flat = arr.ravel()
        base = np.resize(flat, (length, channels))
        base = np.tanh((base - np.mean(base)) / (np.std(base) + 1e-6))
    elif modality_type in ("audio", "microphone", "neural_recording"):
        wav = np.asarray(payload, dtype=np.float64).ravel()
        if wav.size < length:
            wav = np.pad(wav, (0, length - wav.size))
        base = np.zeros((length, channels), dtype=np.float64)
        # Filterbank channels across audio/recording stream
        for ci in range(channels):
            freq = 0.15 + 0.25 * ci
            base[:, ci] = np.tanh(wav[:length] * np.cos(freq * np.arange(length)))
    elif modality_type == "math_function":
        expr = str(payload)
        # Safe procedural math signal synthesis (no eval!)
        base = np.zeros((length, channels), dtype=np.float64)
        for ci in range(channels):
            k = ci + 1
            if "chirp" in expr:
                base[:, ci] = np.sin((1.0 + 0.3 * k) * t * t * 0.2)
            elif "square" in expr:
                base[:, ci] = np.sign(np.sin(k * t))
            elif "decay" in expr:
                base[:, ci] = np.exp(-0.25 * t) * np.cos(k * t)
            else:
                base[:, ci] = np.sin(k * t + 0.3 * ci)
    elif modality_type in ("protein_structure", "3d_object"):
        enc = ExpandedSensoryEncoder(out_dim=channels)
        fvec = enc.encode("protein_perception", payload)["vector"]
        base = np.outer(np.cos(t), fvec)
    elif modality_type in ("connectome_stream", "chemical_data"):
        arr = np.asarray(payload, dtype=np.float64)
        base = np.tanh(np.resize(arr, (length, channels)))
    else:  # procedural_generator
        base = np.stack([np.sin((c + 1) * 0.6 * t + rng.uniform(0, 1)) for c in range(channels)], axis=1)

    # 2. Apply requested neural representation encoding
    if encoding == "rate":
        signal = np.clip(base, -1.0, 1.0)
    elif encoding == "spiking":
        # Poisson-like deterministic threshold spike train
        p_spike = 0.5 * (np.clip(base, -1.0, 1.0) + 1.0)
        signal = (rng.random(base.shape) < p_spike * 0.65).astype(np.float64)
    elif encoding == "event":
        diff = np.vstack([np.zeros((1, channels)), np.diff(base, axis=0)])
        signal = np.where(np.abs(diff) > 0.15, np.sign(diff), 0.0)
    elif encoding == "temporal":
        # Phase-latency coded pulses
        signal = np.tanh(np.cumsum(base, axis=0) / max(math.sqrt(length), 1.0))
    elif encoding == "frequency":
        fft_mag = np.abs(np.fft.rfft(base, axis=0))
        signal = np.tanh(np.resize(fft_mag, (length, channels)) / (np.max(fft_mag) + 1e-6))
    else:  # latent
        proj = rng.normal(0.0, 1.0 / math.sqrt(channels), size=(channels, channels))
        signal = np.tanh(base @ proj)

    return {
        "stream_id": make_id("spec"),
        "modality_type": modality_type,
        "encoding": encoding,
        "source_kind": source_kind.value,
        "is_synthetic_encoding": source_kind != DataSourceKind.BIOLOGICAL_RECORDING,
        "shape": list(signal.shape),
        "signal": signal,
    }


def mix_neural_signals(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    operator: str = "weighted_addition",
    alpha: float = 0.5,
    delay_steps: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Apply any of the 13 SONAPSE SPECTRA signal mixing operators to two [T, C] streams."""
    if operator not in MIXING_OPERATORS:
        raise ValueError(f"Unknown mixing operator: {operator!r}")
    A = np.asarray(sig_a, dtype=np.float64)
    B = np.resize(np.asarray(sig_b, dtype=np.float64), A.shape)
    T, C = A.shape
    a = float(np.clip(alpha, 0.0, 1.0))

    if operator == "weighted_addition":
        return np.clip((1.0 - a) * A + a * B, -2.0, 2.0)
    if operator == "multiplicative_binding":
        return np.tanh(A * B + 0.25 * (A + B))
    if operator == "amplitude_modulation":
        return np.tanh(A * (1.0 + a * B))
    if operator == "frequency_modulation":
        t = np.linspace(0.0, 2.0 * math.pi, T)[:, None]
        return np.sin(2.0 * t + 3.0 * a * B) * (0.5 + 0.5 * np.abs(A))
    if operator == "phase_coupling":
        phase_a = np.arctan2(np.gradient(A, axis=0), A + 1e-6)
        phase_b = np.arctan2(np.gradient(B, axis=0), B + 1e-6)
        return np.cos(phase_a - a * phase_b)
    if operator == "temporal_interleaving":
        out = A.copy()
        out[1::2] = B[1::2]
        return out
    if operator == "spatial_remapping":
        shift = max(1, int(round(a * C)))
        return 0.5 * A + 0.5 * np.roll(B, shift, axis=1)
    if operator == "temporal_compression":
        idx = np.linspace(0, T - 1, T)
        idx_fast = np.clip((idx * (1.0 + a)).astype(int), 0, T - 1)
        return 0.5 * A + 0.5 * B[idx_fast]
    if operator == "delay_distortion":
        d_s = max(1, min(T - 1, int(delay_steps)))
        return (1.0 - a) * A + a * np.roll(B, d_s, axis=0)
    if operator == "signal_inversion":
        return (1.0 - a) * A - a * B
    if operator == "cross_modal_attention":
        # Softmax cross-channel attention from A querying B
        scores = (A.T @ B) / max(math.sqrt(T), 1.0)
        ex = np.exp(np.clip(scores - np.max(scores, axis=1, keepdims=True), -20, 20))
        att = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-9)
        return np.tanh(B @ att.T)
    if operator == "chemical_modulation":
        # Slow chemical envelope derived from B modulates A's gain
        env = np.zeros(T, dtype=np.float64)
        c_lvl = 0.0
        for ti in range(T):
            c_lvl = 0.88 * c_lvl + 0.12 * float(np.mean(np.abs(B[ti])))
            env[ti] = c_lvl
        return np.tanh(A * (1.0 + 1.5 * a * env[:, None]))
    # learned_nonlinear
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.5 / math.sqrt(2 * C), size=(2 * C, C))
    return np.tanh(np.hstack([A, B]) @ W)


def inject_signal_into_connectome(
    connectome: MultiFidelityConnectome,
    mixed_signal: np.ndarray,
    target_scope: str = "ordinary_sensory",
    strength: float = 0.8,
    start_step: int = 4,
    duration: Optional[int] = None,
    noise_std: float = 0.0,
    specific_ids: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Inject a SPECTRA signal into any targeted anatomical site & compare before/after baseline."""
    if target_scope not in INJECTION_TARGETS:
        raise ValueError(f"Unknown injection target scope: {target_scope!r}")
    sig = np.asarray(mixed_signal, dtype=np.float64)
    T, C = sig.shape
    dur = int(duration) if duration is not None else max(1, T - start_step)
    end_step = min(T, start_step + dur)
    rng = np.random.default_rng(seed)

    # 1. Baseline unperturbed rollout (exact matched initial reset)
    connectome.reset()
    base_trace = np.zeros((T, len(connectome.neuron_ids)), dtype=np.float64)
    for k in range(T):
        base_trace[k] = connectome.step({})["outputs"]

    # 2. Resolve target neuron IDs by anatomical scope
    all_ids = connectome.neuron_ids
    if target_scope == "ordinary_sensory":
        targets = connectome.graph.sensory_ids or all_ids[: min(4, len(all_ids))]
    elif target_scope == "alternative_sensory":
        targets = all_ids[len(all_ids) // 2 : len(all_ids) // 2 + min(4, len(all_ids))] or all_ids[:1]
    elif target_scope in ("selected_population", "multiple_neurons") and specific_ids:
        targets = [nid for nid in specific_ids if nid in connectome.neurons] or all_ids[:2]
    elif target_scope == "specific_neuron":
        targets = [specific_ids[0]] if (specific_ids and specific_ids[0] in connectome.neurons) else all_ids[:1]
    elif target_scope == "random_population":
        k_cnt = max(1, len(all_ids) // 3)
        targets = [all_ids[int(i)] for i in rng.choice(len(all_ids), size=k_cnt, replace=False)]
    else:
        # Compartment-level targets: dendrite, secondary_soma, individual_compartment, synthetic_organelle, bridge
        targets = list(specific_ids) if specific_ids else all_ids[: min(4, len(all_ids))]

    # 3. Perturbed injection rollout
    connectome.reset()
    inj_trace = np.zeros((T, len(all_ids)), dtype=np.float64)
    for k in range(T):
        ext_map: Dict[str, Any] = {}
        mods: Dict[str, float] = {}
        if start_step <= k < end_step:
            row = sig[k] * float(strength)
            if noise_std > 0.0:
                row = row + rng.normal(0.0, noise_std, size=row.shape)
            for t_i, nid in enumerate(targets):
                val = float(row[t_i % C])
                inst = connectome.neurons[nid]
                if target_scope == "secondary_soma":
                    # Inject directly into secondary soma compartment vector
                    vec = np.zeros(inst.n_comp, dtype=np.float64)
                    for ci, comp in enumerate(inst.spec.compartments):
                        if comp.kind.value == "secondary_soma":
                            vec[ci] = val
                    if np.all(vec == 0.0):
                        vec[-1] = val
                    ext_map[nid] = vec
                elif target_scope == "dendrite":
                    vec = np.zeros(inst.n_comp, dtype=np.float64)
                    for ci, comp in enumerate(inst.spec.compartments):
                        if "dendrite" in comp.kind.value:
                            vec[ci] = val
                    if np.all(vec == 0.0):
                        vec[0] = val
                    ext_map[nid] = vec
                elif target_scope == "synthetic_organelle":
                    mods = {"da": val, "ach": abs(val)}
                    ext_map[nid] = val * 0.5
                else:
                    ext_map[nid] = val
        inj_trace[k] = connectome.step(ext_map, modulators=mods)["outputs"]

    delta = inj_trace - base_trace
    l2_response = float(np.linalg.norm(delta))
    per_neuron_delta = {all_ids[i]: round(float(np.mean(np.abs(delta[:, i]))), 5) for i in range(len(all_ids))}
    config_rec = {
        "target_scope": target_scope,
        "targets": targets,
        "strength": strength,
        "start_step": start_step,
        "duration": dur,
        "noise_std": noise_std,
        "seed": seed,
    }
    return {
        "intervention_id": make_id("inj"),
        "config": config_rec,
        "config_digest": content_digest(config_rec),
        "baseline_mean_activity": round(float(np.mean(np.abs(base_trace))), 5),
        "injected_mean_activity": round(float(np.mean(np.abs(inj_trace))), 5),
        "l2_response_delta": round(l2_response, 5),
        "peak_response_delta": round(float(np.max(np.abs(delta))), 5),
        "per_neuron_mean_abs_delta": per_neuron_delta,
        "baseline_waveform": [round(float(x), 4) for x in np.mean(base_trace, axis=1)],
        "injected_waveform": [round(float(x), 4) for x in np.mean(inj_trace, axis=1)],
    }
