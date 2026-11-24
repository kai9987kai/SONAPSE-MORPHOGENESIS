"""Physically Structured Multi-Compartment Neuron Simulation Engine.

Replaces the point-neuron assumption with a modular compartment-graph engine
where physical geometry (length, radius, myelination, membrane capacitance,
axial resistance, receptor densities) directly alters computational dynamics.

Supports 5 simulation fidelity levels AND mixed-fidelity connectomes:
  1. ABSTRACT_RATE                - fast leaky rate neuron
  2. MULTI_COMPARTMENT_RECURRENT  - coupled multi-compartment continuous dynamics
  3. SPIKING                      - multi-compartment LIF / adaptive Izhikevich
  4. CONDUCTANCE                  - Hodgkin-Huxley Na/K/Leak + AMPA/NMDA/GABA (Exp-Euler)
  5. REFERENCE_CABLE              - RK4 sub-stepped Hodgkin-Huxley cable reference

Reuses verified Hodgkin-Huxley rate equations from ``synapse.frontier.hh_cable``.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from synapse.frontier.hh_cable import hh_rates, steady_state_gates, temperature_factor

from .interchange import (
    CompartmentEdge,
    CompartmentKind,
    CompartmentNode,
    ConnectomeGraph,
    FidelityLevel,
    NeuronMorphologySpec,
    OrganelleAttachment,
)


def _nmda_mg_block(v_mV: np.ndarray, mg_mM: float = 1.0) -> np.ndarray:
    """Jahr & Stevens (1990) voltage-dependent Mg2+ block for NMDA receptors."""
    return 1.0 / (1.0 + (mg_mM / 3.57) * np.exp(-0.062 * np.clip(v_mV, -120.0, 60.0)))


class StructuredNeuronInstance:
    """Executable multi-compartment cellular simulation compiled from NeuronMorphologySpec."""

    def __init__(
        self,
        spec: NeuronMorphologySpec,
        dt_ms: float = 0.5,
        temperature_c: float = 18.5,
        organelle_factory: Optional[Callable[[OrganelleAttachment], Any]] = None,
        seed: int = 42,
    ) -> None:
        self.spec = spec
        self.dt_ms = float(dt_ms)
        self.temperature_c = float(temperature_c)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._organelle_factory = organelle_factory

        self.comp_ids: List[str] = [c.compartment_id for c in spec.compartments]
        self.id_to_idx: Dict[str, int] = {cid: i for i, cid in enumerate(self.comp_ids)}
        self.n_comp: int = len(self.comp_ids)
        self.input_indices: List[int] = [self.id_to_idx[c] for c in spec.input_compartments if c in self.id_to_idx]
        self.output_indices: List[int] = [self.id_to_idx[c] for c in spec.output_compartments if c in self.id_to_idx]

        # Precompute physical parameters per compartment
        self._compile_physics()

        # Instantiate attached synthetic organelles if factory supplied
        self.organelles: Dict[str, Any] = {}
        self.organelle_comp_idx: Dict[str, int] = {}
        if organelle_factory is not None:
            for att in spec.organelles:
                if att.active and att.compartment_id in self.id_to_idx:
                    self.organelles[att.organelle_id] = organelle_factory(att)
                    self.organelle_comp_idx[att.organelle_id] = self.id_to_idx[att.compartment_id]

        self.reset()

    def _compile_physics(self) -> None:
        n = self.n_comp
        self.active_mask = np.array([1.0 if c.active else 0.0 for c in self.spec.compartments], dtype=np.float64)
        self.cm = np.array([c.effective_cm_uF_cm2 for c in self.spec.compartments], dtype=np.float64)
        self.cap_nF = np.array([c.capacitance_nF for c in self.spec.compartments], dtype=np.float64)
        self.r_axial = np.array([c.axial_resistance_MOhm for c in self.spec.compartments], dtype=np.float64)
        self.g_leak = np.array([c.g_leak_mS_cm2 for c in self.spec.compartments], dtype=np.float64)
        self.e_rest = np.array([c.e_rest_mV for c in self.spec.compartments], dtype=np.float64)
        self.v_thresh = np.array([c.threshold_mV for c in self.spec.compartments], dtype=np.float64)
        self.tau_ms = np.array([c.effective_time_constant_ms for c in self.spec.compartments], dtype=np.float64)
        self.base_metabolic = np.array([c.baseline_metabolic_cost for c in self.spec.compartments], dtype=np.float64)
        self.ampa_gain = np.array([c.receptors.ampa for c in self.spec.compartments], dtype=np.float64)
        self.nmda_gain = np.array([c.receptors.nmda for c in self.spec.compartments], dtype=np.float64)
        self.gaba_gain = np.array([c.receptors.gaba_a + 0.5 * c.receptors.gaba_b for c in self.spec.compartments], dtype=np.float64)
        self.da_sens = np.array([c.receptors.dopamine for c in self.spec.compartments], dtype=np.float64)
        self.ach_sens = np.array([c.receptors.acetylcholine for c in self.spec.compartments], dtype=np.float64)
        self.paracrine_sens = np.array([c.receptors.paracrine for c in self.spec.compartments], dtype=np.float64)

        # Compile edge coupling matrices and discrete delay queues
        self.edges_compiled: List[Dict[str, Any]] = []
        self.axial_g = np.zeros((n, n), dtype=np.float64)
        for e in self.spec.edges:
            si = self.id_to_idx[e.source_id]
            ti = self.id_to_idx[e.target_id]
            src_c = self.spec.compartments[si]
            # Physical transmission delay = edge base delay + source compartment conduction delay
            total_delay_ms = e.delay_ms + (src_c.transmission_delay_ms if src_c.kind in (CompartmentKind.AXON_SEGMENT, CompartmentKind.AXON_FEEDBACK) else 0.0)
            delay_steps = max(0, int(round(total_delay_ms / max(self.dt_ms, 1e-3))))
            # Physical axial conductance inversely proportional to axial resistance
            inv_r = 1.0 / max(0.5 * (self.r_axial[si] + self.r_axial[ti]), 0.1)
            eff_g = float(np.clip(inv_r * 0.25, 0.02, 2.5))
            if e.coupling_type == "electrical_axial":
                self.axial_g[ti, si] += abs(e.weight) * eff_g
                self.axial_g[si, ti] += abs(e.weight) * eff_g * 0.35
            self.edges_compiled.append({
                "src": si,
                "dst": ti,
                "weight": float(e.weight),
                "coupling_type": e.coupling_type,
                "eff_g": eff_g,
                "delay_steps": min(delay_steps, 64),
                "gated_by": self.id_to_idx[e.gated_by] if e.gated_by and e.gated_by in self.id_to_idx else None,
                "plasticity_rule": e.plasticity_rule,
            })

    def reset(self) -> None:
        n = self.n_comp
        self.step_count = 0
        # Continuous rate / state per compartment
        self.state = np.zeros(n, dtype=np.float64)
        self.prev_state = np.zeros(n, dtype=np.float64)
        # Membrane voltage (mV) for spiking / conductance / cable modes
        self.v_mV = self.e_rest.copy()
        self.u_adapt = np.zeros(n, dtype=np.float64)  # Izhikevich / adaptive recovery variable
        self.refractory_ms = np.zeros(n, dtype=np.float64)
        self.spikes = np.zeros(n, dtype=np.float64)
        # Hodgkin-Huxley gating variables
        m0, h0, n0 = steady_state_gates(self.v_mV)
        self.m_gate = m0.astype(np.float64).copy()
        self.h_gate = h0.astype(np.float64).copy()
        self.n_gate = n0.astype(np.float64).copy()
        # Synaptic conductance traces
        self.g_exc = np.zeros(n, dtype=np.float64)
        self.g_inh = np.zeros(n, dtype=np.float64)
        # Metabolic ATP energy pool (0..100) and cumulative work
        self.atp_pool = 100.0
        self.cumulative_energy = 0.0
        self.cumulative_flops = 0
        # Delay buffers per edge
        self._delay_queues: List[deque] = [
            deque([0.0] * (ec["delay_steps"] + 1), maxlen=ec["delay_steps"] + 1)
            for ec in self.edges_compiled
        ]
        # Internal memory trace & paracrine release
        self.paracrine_emission = 0.0
        for org in self.organelles.values():
            if hasattr(org, "reset"):
                org.reset()

    def validate_numerical_stability(self) -> Dict[str, Any]:
        """Verify numerical stability bounds for the active fidelity level."""
        fidelity = self.spec.fidelity
        max_axial = float(np.max(np.sum(self.axial_g, axis=1))) if self.n_comp > 0 else 0.0
        min_tau = float(np.min(self.tau_ms)) if self.n_comp > 0 else 1.0
        max_dt_rate = 1.5 * min_tau
        phi = temperature_factor(self.temperature_c)
        # Conservative RK4 eigenvalue estimate for HH cable (Na/K peak conductance + axial coupling)
        lambda_est = (120.0 + 36.0 + max_axial * 10.0) / max(float(np.min(self.cm)), 0.1)
        rk4_product = self.dt_ms * lambda_est
        if fidelity == FidelityLevel.REFERENCE_CABLE:
            substeps = max(1, int(math.ceil(rk4_product / 1.8)))
            stable = (self.dt_ms / substeps) * lambda_est < 2.5
        elif fidelity == FidelityLevel.CONDUCTANCE:
            # Rush-Larsen exponential Euler is A-stable for linear decay
            substeps = max(1, int(math.ceil(self.dt_ms / 0.1)))
            stable = True
        else:
            substeps = 1
            stable = self.dt_ms <= max_dt_rate
        return {
            "fidelity": int(fidelity),
            "approximation": self.spec.approximation_label,
            "dt_ms": self.dt_ms,
            "min_tau_ms": round(min_tau, 4),
            "max_axial_g": round(max_axial, 4),
            "temperature_phi": round(phi, 4),
            "recommended_substeps": substeps,
            "stable": bool(stable),
            "finite_state": bool(np.all(np.isfinite(self.state)) and np.all(np.isfinite(self.v_mV))),
        }

    def _collect_edge_drive(self, source_signal: np.ndarray) -> np.ndarray:
        drive = np.zeros(self.n_comp, dtype=np.float64)
        for idx, ec in enumerate(self.edges_compiled):
            q = self._delay_queues[idx]
            delayed_val = q[0]
            q.append(float(source_signal[ec["src"]]))
            gate = 1.0
            if ec["gated_by"] is not None:
                gate = 1.0 / (1.0 + math.exp(-np.clip(3.0 * source_signal[ec["gated_by"]], -20.0, 20.0)))
            w = ec["weight"]
            if ec["coupling_type"] == "electrical_axial":
                drive[ec["dst"]] += w * ec["eff_g"] * gate * (delayed_val - source_signal[ec["dst"]])
            elif ec["coupling_type"] == "feedback_gated":
                drive[ec["dst"]] += w * gate * delayed_val
            else:
                drive[ec["dst"]] += w * gate * delayed_val
        return drive

    def step(
        self,
        external_input: float | Sequence[float] | np.ndarray,
        modulators: Optional[Mapping[str, float]] = None,
        reward_signal: float = 0.0,
        paracrine_field: float = 0.0,
        adapt: bool = False,
    ) -> Dict[str, Any]:
        """Advance the structured neuron by one time step ``dt_ms``."""
        n = self.n_comp
        mods = modulators or {}
        da = float(mods.get("da", 0.0))
        ach = float(mods.get("ach", 0.0))
        serotonin = float(mods.get("serotonin", 0.0))

        # Distribute external inputs across input compartments (or all compartments if matching length)
        ext = np.zeros(n, dtype=np.float64)
        arr_in = np.asarray(external_input, dtype=np.float64).ravel()
        if arr_in.size == 1:
            for ii in self.input_indices:
                ext[ii] = float(arr_in[0])
        elif arr_in.size == len(self.input_indices):
            for k, ii in enumerate(self.input_indices):
                ext[ii] = float(arr_in[k])
        elif arr_in.size == n:
            ext[:] = arr_in
        else:
            for k, ii in enumerate(self.input_indices):
                ext[ii] = float(arr_in[k % arr_in.size])

        # Receptor sensitivity & metabolic scaling
        atp_gain = 0.25 + 0.75 * (self.atp_pool / 100.0)
        exc_in = np.maximum(ext, 0.0) * self.ampa_gain
        inh_in = np.minimum(ext, 0.0) * self.gaba_gain
        mod_bias = 0.15 * da * self.da_sens + 0.12 * ach * self.ach_sens + 0.20 * paracrine_field * self.paracrine_sens
        total_ext = (exc_in + inh_in + mod_bias) * atp_gain * self.active_mask

        # Step synthetic organelles attached to compartments
        org_current = np.zeros(n, dtype=np.float64)
        org_telemetry: Dict[str, Any] = {}
        for oid, org in self.organelles.items():
            ci = self.organelle_comp_idx[oid]
            res = org.step(
                compartment_state=float(self.state[ci]),
                external_drive=float(total_ext[ci]),
                reward=reward_signal,
                modulators=mods,
                adapt=adapt,
            )
            org_current[ci] += float(res.get("current", 0.0))
            org_telemetry[oid] = res

        self.prev_state = self.state.copy()
        fidelity = self.spec.fidelity
        spikes_now = np.zeros(n, dtype=np.float64)

        if fidelity == FidelityLevel.ABSTRACT_RATE:
            # Level 1: Fast abstract rate dynamics
            alpha = np.clip(self.dt_ms / self.tau_ms, 0.02, 1.0)
            edge_drive = self._collect_edge_drive(self.state)
            pre = total_ext + edge_drive + org_current - 0.08 * serotonin * self.state
            self.state = ((1.0 - alpha) * self.state + alpha * np.tanh(pre)) * self.active_mask
            self.v_mV = self.e_rest + 25.0 * self.state
            flops = 8 * n + 4 * len(self.edges_compiled)

        elif fidelity == FidelityLevel.MULTI_COMPARTMENT_RECURRENT:
            # Level 2: Physically coupled multi-compartment continuous dynamics
            alpha = np.clip(self.dt_ms / self.tau_ms, 0.01, 0.95)
            edge_drive = self._collect_edge_drive(self.state)
            # NMDA supralinear coincidence boost on active depolarized dendrites
            nmda_boost = self.nmda_gain * np.maximum(self.state, 0.0) * np.maximum(total_ext, 0.0)
            pre = total_ext + nmda_boost + edge_drive + org_current - 0.1 * self.u_adapt
            self.state = ((1.0 - alpha) * self.state + alpha * np.tanh(pre)) * self.active_mask
            # Slow adaptation / calcium-like trace
            self.u_adapt = 0.94 * self.u_adapt + 0.06 * np.abs(self.state)
            self.v_mV = self.e_rest + 30.0 * self.state
            spikes_now = (self.state > 0.72).astype(np.float64)
            flops = 16 * n + 6 * len(self.edges_compiled)

        elif fidelity == FidelityLevel.SPIKING:
            # Level 3: Multi-compartment adaptive LIF / Izhikevich spiking dynamics
            edge_drive = self._collect_edge_drive(self.spikes * 1.5 + 0.25 * self.state)
            self.g_exc = 0.85 * self.g_exc + np.maximum(total_ext + edge_drive + org_current, 0.0)
            self.g_inh = 0.80 * self.g_inh + np.maximum(-(total_ext + edge_drive + org_current), 0.0)
            nmda_factor = _nmda_mg_block(self.v_mV)
            i_syn = (self.g_exc * (1.0 + 0.4 * self.nmda_gain * nmda_factor) * (0.0 - self.v_mV)
                     + self.g_inh * (-80.0 - self.v_mV)) * 0.18
            # Axial resistive current across coupled compartments
            i_axial = (self.axial_g @ self.v_mV) - np.sum(self.axial_g, axis=1) * self.v_mV
            not_ref = (self.refractory_ms <= 0.0).astype(np.float64)
            dv = ((self.e_rest - self.v_mV) + 18.0 * i_syn + 4.0 * i_axial - 8.0 * self.u_adapt) / self.tau_ms
            self.v_mV = np.clip(self.v_mV + self.dt_ms * dv * not_ref, -100.0, 50.0)
            fired = (self.v_mV >= self.v_thresh) & (self.active_mask > 0.5)
            spikes_now = fired.astype(np.float64)
            self.v_mV = np.where(fired, self.e_rest - 5.0, self.v_mV)
            self.refractory_ms = np.maximum(0.0, self.refractory_ms - self.dt_ms)
            self.refractory_ms = np.where(fired, 2.0, self.refractory_ms)
            self.u_adapt = 0.96 * self.u_adapt + 0.25 * spikes_now
            self.state = np.clip(0.8 * self.state + 0.55 * spikes_now + 0.015 * (self.v_mV - self.e_rest), -1.0, 1.0)
            flops = 26 * n + 8 * len(self.edges_compiled) + 2 * n * n

        elif fidelity in (FidelityLevel.CONDUCTANCE, FidelityLevel.REFERENCE_CABLE):
            # Level 4 (Rush-Larsen Exp-Euler) & Level 5 (RK4 sub-stepped HH cable)
            phi = temperature_factor(self.temperature_c)
            edge_drive = self._collect_edge_drive(self.state)
            i_stim = (total_ext + edge_drive + org_current) * 18.0  # scaled to uA/cm^2
            substeps = 5 if fidelity == FidelityLevel.CONDUCTANCE else 10
            sub_dt = self.dt_ms / substeps
            g_na_max, g_k_max = 120.0, 36.0
            e_na, e_k, e_l = 50.0, -77.0, -54.387
            v_prev_step = self.v_mV.copy()
            for _ in range(substeps):
                if fidelity == FidelityLevel.CONDUCTANCE:
                    # Rush-Larsen exponential Euler (unconditionally stable)
                    am, bm, ah, bh, an, bn = hh_rates(self.v_mV)
                    am, bm, ah, bh, an, bn = am * phi, bm * phi, ah * phi, bh * phi, an * phi, bn * phi
                    for g_attr, a, b in (("m_gate", am, bm), ("h_gate", ah, bh), ("n_gate", an, bn)):
                        rate = a + b + 1e-12
                        inf = a / rate
                        val = getattr(self, g_attr)
                        setattr(self, g_attr, inf + (val - inf) * np.exp(-sub_dt * rate))
                    g_na = g_na_max * (self.m_gate ** 3) * self.h_gate
                    g_k = g_k_max * (self.n_gate ** 4)
                    g_ax_row = np.sum(self.axial_g, axis=1)
                    g_tot = g_na + g_k + self.g_leak + g_ax_row + 1e-12
                    i_ax_in = self.axial_g @ self.v_mV
                    v_inf = (g_na * e_na + g_k * e_k + self.g_leak * e_l + i_ax_in + i_stim) / g_tot
                    self.v_mV = np.clip(v_inf + (self.v_mV - v_inf) * np.exp(-sub_dt * g_tot / self.cm), -110.0, 70.0)
                else:
                    # Level 5: Classical 4th-order Runge-Kutta on HH cable
                    def _deriv(v, m, h, ng):
                        am, bm, ah, bh, an, bn = hh_rates(v)
                        dm = phi * (am * (1.0 - m) - bm * m)
                        dh = phi * (ah * (1.0 - h) - bh * h)
                        dn = phi * (an * (1.0 - ng) - bn * ng)
                        i_ion = g_na_max * (m ** 3) * h * (v - e_na) + g_k_max * (ng ** 4) * (v - e_k) + self.g_leak * (v - e_l)
                        i_ax = (self.axial_g @ v) - np.sum(self.axial_g, axis=1) * v
                        dv = (i_stim + i_ax - i_ion) / self.cm
                        return dv, dm, dh, dn

                    k1 = _deriv(self.v_mV, self.m_gate, self.h_gate, self.n_gate)
                    k2 = _deriv(self.v_mV + 0.5 * sub_dt * k1[0], np.clip(self.m_gate + 0.5 * sub_dt * k1[1], 0, 1),
                                np.clip(self.h_gate + 0.5 * sub_dt * k1[2], 0, 1), np.clip(self.n_gate + 0.5 * sub_dt * k1[3], 0, 1))
                    k3 = _deriv(self.v_mV + 0.5 * sub_dt * k2[0], np.clip(self.m_gate + 0.5 * sub_dt * k2[1], 0, 1),
                                np.clip(self.h_gate + 0.5 * sub_dt * k2[2], 0, 1), np.clip(self.n_gate + 0.5 * sub_dt * k2[3], 0, 1))
                    k4 = _deriv(self.v_mV + sub_dt * k3[0], np.clip(self.m_gate + sub_dt * k3[1], 0, 1),
                                np.clip(self.h_gate + sub_dt * k3[2], 0, 1), np.clip(self.n_gate + sub_dt * k3[3], 0, 1))
                    self.v_mV = np.clip(self.v_mV + (sub_dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), -110.0, 70.0)
                    self.m_gate = np.clip(self.m_gate + (sub_dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0.0, 1.0)
                    self.h_gate = np.clip(self.h_gate + (sub_dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), 0.0, 1.0)
                    self.n_gate = np.clip(self.n_gate + (sub_dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]), 0.0, 1.0)
                upward = (v_prev_step < 0.0) & (self.v_mV >= 0.0)
                spikes_now = np.maximum(spikes_now, upward.astype(np.float64))
                v_prev_step = self.v_mV.copy()
            self.state = np.tanh((self.v_mV - self.e_rest) / 35.0) * self.active_mask
            flops = substeps * (85 if fidelity == FidelityLevel.CONDUCTANCE else 240) * n
        else:
            raise ValueError(f"Unsupported fidelity level: {fidelity}")

        self.spikes = spikes_now
        # Local internal plasticity if adapt=True
        if adapt and abs(reward_signal) > 1e-6:
            for ec in self.edges_compiled:
                if ec["plasticity_rule"] != "none":
                    dw = 0.015 * reward_signal * float(self.prev_state[ec["src"]] * self.state[ec["dst"]])
                    ec["weight"] = float(np.clip(ec["weight"] + dw, -4.0, 4.0))

        # Metabolic ATP consumption & homeostatic recovery
        step_energy = float(np.sum(self.base_metabolic * self.active_mask) * 0.05 + 0.12 * np.sum(spikes_now) + 0.02 * len(self.organelles))
        self.atp_pool = float(np.clip(self.atp_pool - step_energy + 0.45 * (1.0 - self.atp_pool / 100.0), 5.0, 100.0))
        self.cumulative_energy += step_energy
        self.cumulative_flops += int(flops)
        self.step_count += 1

        # Paracrine emission if any compartment is marked as paracrine broadcaster
        self.paracrine_emission = float(np.mean(self.state[self.output_indices])) * 0.25 if self.output_indices else 0.0
        outputs = self.state[self.output_indices].copy() if self.output_indices else np.array([0.0])

        return {
            "output": outputs,
            "scalar_output": float(np.mean(outputs)),
            "compartment_states": {self.comp_ids[i]: round(float(self.state[i]), 6) for i in range(n)},
            "v_mV": {self.comp_ids[i]: round(float(self.v_mV[i]), 3) for i in range(n)},
            "spikes": spikes_now.tolist(),
            "atp_pool": round(self.atp_pool, 4),
            "step_energy": round(step_energy, 6),
            "step_flops": int(flops),
            "organelles": org_telemetry,
            "paracrine_emission": round(self.paracrine_emission, 6),
        }

    def repair_compartment(self, compartment_id: str, gain_boost: float = 1.15) -> bool:
        """Structural repair: reactivate a damaged compartment or compensate via healthy neighbors."""
        if compartment_id not in self.id_to_idx:
            return False
        idx = self.id_to_idx[compartment_id]
        self.active_mask[idx] = 1.0
        self.spec.compartments[idx].active = True
        for ec in self.edges_compiled:
            if ec["dst"] == idx or ec["src"] == idx:
                ec["weight"] = float(np.clip(ec["weight"] * gain_boost, -4.0, 4.0))
        return True

    def lesion_compartment(self, compartment_id: str) -> bool:
        """Controlled causal lesion of a single internal compartment."""
        if compartment_id not in self.id_to_idx:
            return False
        idx = self.id_to_idx[compartment_id]
        self.active_mask[idx] = 0.0
        self.state[idx] = 0.0
        self.spec.compartments[idx].active = False
        return True


class MultiFidelityConnectome:
    """Executes a heterogeneous ConnectomeGraph where neurons of Fidelity 1..5 coexist.

    Uses spatial distance for paracrine chemical diffusion and discrete delay rings
    for synaptic transmission across heterogeneous neuron morphologies.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        dt_ms: float = 1.0,
        organelle_factory: Optional[Callable[[OrganelleAttachment], Any]] = None,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.dt_ms = float(dt_ms)
        self.seed = int(seed)
        self.neurons: Dict[str, StructuredNeuronInstance] = {}
        self.neuron_ids: List[str] = []
        for idx, nspec in enumerate(graph.neurons):
            self.neuron_ids.append(nspec.morphology_id)
            self.neurons[nspec.morphology_id] = StructuredNeuronInstance(
                nspec, dt_ms=dt_ms, organelle_factory=organelle_factory, seed=seed + idx
            )
        self.id_to_idx = {nid: i for i, nid in enumerate(self.neuron_ids)}
        self._synapses: List[Dict[str, Any]] = []
        self._syn_queues: List[deque] = []
        for e in graph.edges:
            if e.active and e.pre_id in self.neurons and e.post_id in self.neurons:
                delay_steps = max(0, int(round(e.delay_ms / max(self.dt_ms, 1e-3))))
                self._synapses.append({
                    "pre": e.pre_id,
                    "post": e.post_id,
                    "weight": float(e.weight),
                    "delay_steps": min(delay_steps, 64),
                    "plasticity_rule": e.plasticity_rule,
                    "dale_sign": self.neurons[e.pre_id].spec.dale_sign,
                })
                self._syn_queues.append(deque([0.0] * (min(delay_steps, 64) + 1), maxlen=min(delay_steps, 64) + 1))

        self.last_outputs = np.zeros(len(self.neuron_ids), dtype=np.float64)
        self.step_count = 0
        self.total_flops = 0
        self.total_energy = 0.0

    def reset(self) -> None:
        self.step_count = 0
        self.total_flops = 0
        self.total_energy = 0.0
        self.last_outputs.fill(0.0)
        for inst in self.neurons.values():
            inst.reset()
        for q in self._syn_queues:
            for i in range(len(q)):
                q[i] = 0.0

    def step(
        self,
        external_inputs: Mapping[str, float | Sequence[float]] | np.ndarray,
        modulators: Optional[Mapping[str, float]] = None,
        reward_signal: float = 0.0,
        adapt: bool = False,
    ) -> Dict[str, Any]:
        # Gather synaptic currents with physical delays
        syn_drive: Dict[str, float] = {nid: 0.0 for nid in self.neuron_ids}
        for s_idx, syn in enumerate(self._synapses):
            q = self._syn_queues[s_idx]
            pre_val = q[0]
            pre_idx = self.id_to_idx[syn["pre"]]
            q.append(float(self.last_outputs[pre_idx]))
            syn_drive[syn["post"]] += syn["weight"] * pre_val

        # Mean paracrine broadcast field from previous step
        paracrine_level = float(np.mean([inst.paracrine_emission for inst in self.neurons.values()])) if self.neurons else 0.0

        # Normalize external input mapping
        ext_map: Dict[str, Any] = {}
        if isinstance(external_inputs, Mapping):
            ext_map = dict(external_inputs)
        else:
            arr = np.asarray(external_inputs, dtype=np.float64).ravel()
            targets = self.graph.sensory_ids or self.neuron_ids
            for k, nid in enumerate(targets):
                if k < arr.size:
                    ext_map[nid] = float(arr[k])

        step_flops = len(self._synapses) * 3
        step_energy = 0.0
        new_outputs = np.zeros(len(self.neuron_ids), dtype=np.float64)
        fidelity_counts: Dict[int, int] = {}

        for i, nid in enumerate(self.neuron_ids):
            inst = self.neurons[nid]
            f_int = int(inst.spec.fidelity)
            fidelity_counts[f_int] = fidelity_counts.get(f_int, 0) + 1
            ext_v = ext_map.get(nid, 0.0)
            if isinstance(ext_v, (int, float)):
                total_in: Any = float(ext_v) + syn_drive[nid]
            else:
                total_in = np.asarray(ext_v, dtype=np.float64) + syn_drive[nid]
            res = inst.step(
                total_in,
                modulators=modulators,
                reward_signal=reward_signal,
                paracrine_field=paracrine_level,
                adapt=adapt,
            )
            new_outputs[i] = res["scalar_output"]
            step_flops += res["step_flops"]
            step_energy += res["step_energy"]

        # Optional Dale-compliant three-factor synaptic adaptation
        if adapt and abs(reward_signal) > 1e-6:
            for syn in self._synapses:
                if syn["plasticity_rule"] != "none":
                    pi = self.id_to_idx[syn["pre"]]
                    qi = self.id_to_idx[syn["post"]]
                    dw = 0.01 * reward_signal * float(self.last_outputs[pi] * new_outputs[qi])
                    new_w = syn["weight"] + dw
                    # Preserve Dale's sign
                    if syn["dale_sign"] > 0:
                        syn["weight"] = float(np.clip(new_w, 0.0, 5.0))
                    else:
                        syn["weight"] = float(np.clip(new_w, -5.0, 0.0))

        self.last_outputs = new_outputs
        self.step_count += 1
        self.total_flops += step_flops
        self.total_energy += step_energy

        motor_ids = self.graph.motor_ids or self.neuron_ids[-min(4, len(self.neuron_ids)):]
        motor_out = [float(self.last_outputs[self.id_to_idx[m]]) for m in motor_ids if m in self.id_to_idx]

        return {
            "step": self.step_count,
            "outputs": new_outputs.copy(),
            "motor_outputs": motor_out,
            "paracrine_field": round(paracrine_level, 6),
            "fidelity_distribution": fidelity_counts,
            "step_flops": step_flops,
            "step_energy": round(step_energy, 6),
        }
