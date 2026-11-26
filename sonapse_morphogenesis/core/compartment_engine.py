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
        self._input_idx_arr = np.asarray(self.input_indices, dtype=np.int32)
        self._output_idx_arr = np.asarray(self.output_indices, dtype=np.int32)

        # Precompute physical parameters & vectorized edge structures per compartment
        self._compile_physics()

        # Instantiate attached synthetic organelles if factory supplied
        self.organelles: Dict[str, Any] = {}
        self.organelle_comp_idx: Dict[str, int] = {}
        self._has_arbitrosome = bool(spec.metadata.get("organelle_arbitration", False))
        if organelle_factory is not None:
            for att in spec.organelles:
                if att.active and att.compartment_id in self.id_to_idx:
                    self.organelles[att.organelle_id] = organelle_factory(att)
                    self.organelle_comp_idx[att.organelle_id] = self.id_to_idx[att.compartment_id]
                    if att.organelle_type == "arbitrosome":
                        self._has_arbitrosome = True

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
        self.alpha_f1 = np.clip(self.dt_ms / np.maximum(self.tau_ms, 0.2), 0.02, 1.0)
        self.alpha_f2 = np.clip(self.dt_ms / np.maximum(self.tau_ms, 0.2), 0.01, 0.95)
        self.base_metabolic = np.array([c.baseline_metabolic_cost for c in self.spec.compartments], dtype=np.float64)
        self.ampa_gain = np.array([c.receptors.ampa for c in self.spec.compartments], dtype=np.float64)
        self.nmda_gain = np.array([c.receptors.nmda for c in self.spec.compartments], dtype=np.float64)
        self.gaba_gain = np.array([c.receptors.gaba_a + 0.5 * c.receptors.gaba_b for c in self.spec.compartments], dtype=np.float64)
        self.da_sens = np.array([c.receptors.dopamine for c in self.spec.compartments], dtype=np.float64)
        self.ach_sens = np.array([c.receptors.acetylcholine for c in self.spec.compartments], dtype=np.float64)
        self.paracrine_sens = np.array([c.receptors.paracrine for c in self.spec.compartments], dtype=np.float64)
        self.phi = float(temperature_factor(self.temperature_c))

        # Coupling type integer codes:
        # 0: electrical_axial, 1: gap_junction, 2: chemical_synapse, 3: feedback_gated, 4: paracrine, 5: ml_bridge
        ctype_map = {
            "electrical_axial": 0,
            "gap_junction": 1,
            "chemical_synapse": 2,
            "feedback_gated": 3,
            "paracrine": 4,
            "ml_bridge": 5,
        }

        self.edges_compiled: List[Dict[str, Any]] = []
        self.axial_g = np.zeros((n, n), dtype=np.float64)
        e_src, e_dst, e_w, e_eff_g, e_delay, e_gate, e_kind = [], [], [], [], [], [], []

        for e in self.spec.edges:
            si = self.id_to_idx[e.source_id]
            ti = self.id_to_idx[e.target_id]
            src_c = self.spec.compartments[si]
            total_delay_ms = e.delay_ms + (
                src_c.transmission_delay_ms
                if src_c.kind in (CompartmentKind.AXON_SEGMENT, CompartmentKind.AXON_FEEDBACK)
                else 0.0
            )
            delay_steps = min(64, max(0, int(round(total_delay_ms / max(self.dt_ms, 1e-3)))))
            inv_r = 1.0 / max(0.5 * (self.r_axial[si] + self.r_axial[ti]), 0.1)
            eff_g = float(np.clip(inv_r * 0.25, 0.02, 2.5))
            if e.coupling_type == "electrical_axial":
                self.axial_g[ti, si] += abs(e.weight) * eff_g
                self.axial_g[si, ti] += abs(e.weight) * eff_g * 0.35
            elif e.coupling_type == "gap_junction":
                self.axial_g[ti, si] += abs(e.weight) * eff_g * 0.85
                self.axial_g[si, ti] += abs(e.weight) * eff_g * 0.85
            g_idx = self.id_to_idx[e.gated_by] if e.gated_by and e.gated_by in self.id_to_idx else None
            self.edges_compiled.append({
                "src": si,
                "dst": ti,
                "weight": float(e.weight),
                "coupling_type": e.coupling_type,
                "eff_g": eff_g,
                "delay_steps": delay_steps,
                "gated_by": g_idx,
                "plasticity_rule": e.plasticity_rule,
            })
            e_src.append(si)
            e_dst.append(ti)
            e_w.append(float(e.weight))
            e_eff_g.append(eff_g)
            e_delay.append(delay_steps)
            e_gate.append(-1 if g_idx is None else g_idx)
            e_kind.append(ctype_map.get(e.coupling_type, 2))

        self._axial_g_row_sum = np.sum(self.axial_g, axis=1)
        self._e_src = np.asarray(e_src, dtype=np.int32)
        self._e_dst = np.asarray(e_dst, dtype=np.int32)
        self._e_w = np.asarray(e_w, dtype=np.float64)
        self._e_eff_g = np.asarray(e_eff_g, dtype=np.float64)
        self._e_delay = np.asarray(e_delay, dtype=np.int32)
        self._e_gate = np.asarray(e_gate, dtype=np.int32)
        self._e_kind = np.asarray(e_kind, dtype=np.int32)
        self._has_gates = bool(np.any(self._e_gate >= 0))
        self._ring_len = (int(np.max(self._e_delay)) if self._e_delay.size > 0 else 0) + 2
        self._edge_idx_arange = np.arange(self._e_src.size, dtype=np.int32)

    def _sync_edge_weights_from_compiled(self) -> None:
        if self.edges_compiled:
            self._e_w = np.array([ec["weight"] for ec in self.edges_compiled], dtype=np.float64)

    def reset(self) -> None:
        n = self.n_comp
        self.step_count = 0
        self.state = np.zeros(n, dtype=np.float64)
        self.prev_state = np.zeros(n, dtype=np.float64)
        self.v_mV = self.e_rest.copy()
        self.u_adapt = np.zeros(n, dtype=np.float64)
        self.refractory_ms = np.zeros(n, dtype=np.float64)
        self.spikes = np.zeros(n, dtype=np.float64)
        m0, h0, n0 = steady_state_gates(self.v_mV)
        self.m_gate = m0.astype(np.float64).copy()
        self.h_gate = h0.astype(np.float64).copy()
        self.n_gate = n0.astype(np.float64).copy()
        self.g_exc = np.zeros(n, dtype=np.float64)
        self.g_inh = np.zeros(n, dtype=np.float64)
        self.atp_pool = 100.0
        self.cumulative_energy = 0.0
        self.cumulative_flops = 0
        self._edge_ring = np.zeros((self._ring_len, self._e_src.size), dtype=np.float64)
        self._ring_ptr = 0
        self._paracrine_edge_ema = np.zeros(self._e_src.size, dtype=np.float64)
        self.paracrine_emission = 0.0
        for org in self.organelles.values():
            if hasattr(org, "reset"):
                org.reset()

    def validate_numerical_stability(self) -> Dict[str, Any]:
        """Verify numerical stability bounds for the active fidelity level."""
        fidelity = self.spec.fidelity
        max_axial = float(np.max(self._axial_g_row_sum)) if self.n_comp > 0 else 0.0
        min_tau = float(np.min(self.tau_ms)) if self.n_comp > 0 else 1.0
        max_dt_rate = 1.5 * min_tau
        phi = self.phi
        lambda_est = (120.0 + 36.0 + max_axial * 10.0) / max(float(np.min(self.cm)), 0.1)
        rk4_product = self.dt_ms * lambda_est
        if fidelity == FidelityLevel.REFERENCE_CABLE:
            substeps = max(1, int(math.ceil(rk4_product / 1.8)))
            stable = (self.dt_ms / substeps) * lambda_est < 2.5
        elif fidelity == FidelityLevel.CONDUCTANCE:
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
        m = self._e_src.size
        if m == 0:
            return np.zeros(self.n_comp, dtype=np.float64)

        # Read delayed values from circular ring buffer, then push current source_signal
        read_rows = (self._ring_ptr - self._e_delay - 1) % self._ring_len
        delayed_val = self._edge_ring[read_rows, self._edge_idx_arange]
        self._edge_ring[self._ring_ptr, :] = source_signal[self._e_src]
        self._ring_ptr = (self._ring_ptr + 1) % self._ring_len

        if self._has_gates:
            gates = np.ones(m, dtype=np.float64)
            gm = self._e_gate >= 0
            gates[gm] = 1.0 / (1.0 + np.exp(-np.clip(3.0 * source_signal[self._e_gate[gm]], -20.0, 20.0)))
        else:
            gates = 1.0

        # Distinct biophysical / hybrid coupling physics per edge kind:
        # 0: electrical_axial -> ohmic g * (v_src_delayed - v_dst)
        # 1: gap_junction     -> symmetric zero-rectification ohmic coupling
        # 2: chemical_synapse -> saturating synaptic release tanh(1.2 * delayed)
        # 3: feedback_gated   -> gated recurrent projection
        # 4: paracrine        -> slow low-pass volume-transmitted current
        # 5: ml_bridge        -> smooth residual adapter projection
        dst_val = source_signal[self._e_dst]
        contrib = self._e_w * gates * delayed_val
        k = self._e_kind
        if np.any(k == 0):
            m0 = k == 0
            g0 = gates[m0] if isinstance(gates, np.ndarray) else 1.0
            contrib[m0] = self._e_w[m0] * self._e_eff_g[m0] * g0 * (delayed_val[m0] - dst_val[m0])
        if np.any(k == 1):
            m1 = k == 1
            g1 = gates[m1] if isinstance(gates, np.ndarray) else 1.0
            contrib[m1] = self._e_w[m1] * self._e_eff_g[m1] * 1.15 * g1 * (delayed_val[m1] - dst_val[m1])
        if np.any(k == 2):
            m2 = k == 2
            g2 = gates[m2] if isinstance(gates, np.ndarray) else 1.0
            contrib[m2] = self._e_w[m2] * g2 * np.tanh(1.15 * delayed_val[m2])
        if np.any(k == 4):
            m4 = k == 4
            self._paracrine_edge_ema[m4] = 0.80 * self._paracrine_edge_ema[m4] + 0.20 * delayed_val[m4]
            contrib[m4] = self._e_w[m4] * self._paracrine_edge_ema[m4]
        if np.any(k == 5):
            m5 = k == 5
            nrm = float(np.sqrt(np.mean(delayed_val[m5] ** 2)) + 1.0)
            contrib[m5] = self._e_w[m5] * (delayed_val[m5] / nrm)

        return np.bincount(self._e_dst, weights=contrib, minlength=self.n_comp)

    def step(
        self,
        external_input: float | Sequence[float] | np.ndarray | Mapping[str, float],
        modulators: Optional[Mapping[str, float]] = None,
        reward_signal: float = 0.0,
        paracrine_field: float = 0.0,
        adapt: bool = False,
        collect_telemetry: bool = True,
    ) -> Dict[str, Any]:
        """Advance the structured neuron by one time step ``dt_ms``."""
        n = self.n_comp
        mods = modulators or {}
        da = float(mods.get("da", 0.0)) if mods else 0.0
        ach = float(mods.get("ach", 0.0)) if mods else 0.0
        serotonin = float(mods.get("serotonin", 0.0)) if mods else 0.0

        ext = np.zeros(n, dtype=np.float64)
        if isinstance(external_input, Mapping):
            for cid, cval in external_input.items():
                if cid in self.id_to_idx:
                    ext[self.id_to_idx[cid]] += float(cval)
        elif isinstance(external_input, (int, float)):
            if self._input_idx_arr.size > 0:
                ext[self._input_idx_arr] = float(external_input)
        else:
            arr_in = np.asarray(external_input, dtype=np.float64).ravel()
            if arr_in.size == 1:
                if self._input_idx_arr.size > 0:
                    ext[self._input_idx_arr] = float(arr_in[0])
            elif arr_in.size == self._input_idx_arr.size:
                ext[self._input_idx_arr] = arr_in
            elif arr_in.size == n:
                ext[:] = arr_in
            elif self._input_idx_arr.size > 0 and arr_in.size > 0:
                ext[self._input_idx_arr] = arr_in[np.arange(self._input_idx_arr.size) % arr_in.size]

        atp_gain = 0.25 + 0.75 * (self.atp_pool / 100.0)
        exc_in = np.maximum(ext, 0.0) * self.ampa_gain
        inh_in = np.minimum(ext, 0.0) * self.gaba_gain
        mod_bias = 0.15 * da * self.da_sens + 0.12 * ach * self.ach_sens + 0.20 * paracrine_field * self.paracrine_sens
        total_ext = (exc_in + inh_in + mod_bias) * atp_gain * self.active_mask

        # Step synthetic organelles attached to compartments
        org_current = np.zeros(n, dtype=np.float64)
        org_telemetry: Dict[str, Any] = {}
        if self.organelles:
            comp_org_counts = np.zeros(n, dtype=np.int32)
            for oid, org in self.organelles.items():
                ci = self.organelle_comp_idx[oid]
                res = org.step(
                    compartment_state=float(self.state[ci]),
                    external_drive=float(total_ext[ci]),
                    reward=reward_signal,
                    modulators=mods,
                    adapt=adapt,
                )
                cur_o = float(res.get("current", 0.0))
                org_current[ci] += cur_o
                comp_org_counts[ci] += 1
                if collect_telemetry:
                    org_telemetry[oid] = res
            if self._has_arbitrosome and np.any(comp_org_counts > 1):
                # Precision normalization preventing multi-organelle saturation on shared compartments
                multi = comp_org_counts > 1
                org_current[multi] /= np.sqrt(comp_org_counts[multi].astype(np.float64))

        self.prev_state = self.state.copy()
        fidelity = self.spec.fidelity
        spikes_now = np.zeros(n, dtype=np.float64)

        if fidelity == FidelityLevel.ABSTRACT_RATE:
            edge_drive = self._collect_edge_drive(self.state)
            pre = total_ext + edge_drive + org_current - 0.08 * serotonin * self.state
            self.state = ((1.0 - self.alpha_f1) * self.state + self.alpha_f1 * np.tanh(pre)) * self.active_mask
            self.v_mV = self.e_rest + 25.0 * self.state
            flops = 8 * n + 4 * self._e_src.size

        elif fidelity == FidelityLevel.MULTI_COMPARTMENT_RECURRENT:
            edge_drive = self._collect_edge_drive(self.state)
            nmda_boost = self.nmda_gain * np.maximum(self.state, 0.0) * np.maximum(total_ext, 0.0)
            pre = total_ext + nmda_boost + edge_drive + org_current - 0.1 * self.u_adapt
            self.state = ((1.0 - self.alpha_f2) * self.state + self.alpha_f2 * np.tanh(pre)) * self.active_mask
            self.u_adapt = 0.94 * self.u_adapt + 0.06 * np.abs(self.state)
            self.v_mV = self.e_rest + 30.0 * self.state
            spikes_now = (self.state > 0.72).astype(np.float64)
            flops = 16 * n + 6 * self._e_src.size

        elif fidelity == FidelityLevel.SPIKING:
            edge_drive = self._collect_edge_drive(self.spikes * 1.5 + 0.25 * self.state)
            tot_drv = total_ext + edge_drive + org_current
            self.g_exc = 0.85 * self.g_exc + np.maximum(tot_drv, 0.0)
            self.g_inh = 0.80 * self.g_inh + np.maximum(-tot_drv, 0.0)
            nmda_factor = _nmda_mg_block(self.v_mV)
            i_syn = (
                self.g_exc * (1.0 + 0.4 * self.nmda_gain * nmda_factor) * (-self.v_mV)
                + self.g_inh * (-80.0 - self.v_mV)
            ) * 0.18
            i_axial = (self.axial_g @ self.v_mV) - self._axial_g_row_sum * self.v_mV
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
            flops = 26 * n + 8 * self._e_src.size + 2 * n * n

        elif fidelity in (FidelityLevel.CONDUCTANCE, FidelityLevel.REFERENCE_CABLE):
            phi = self.phi
            edge_drive = self._collect_edge_drive(self.state)
            i_stim = (total_ext + edge_drive + org_current) * 18.0
            substeps = 5 if fidelity == FidelityLevel.CONDUCTANCE else 10
            sub_dt = self.dt_ms / substeps
            g_na_max, g_k_max = 120.0, 36.0
            e_na, e_k, e_l = 50.0, -77.0, -54.387
            v_prev_step = self.v_mV.copy()
            for _ in range(substeps):
                if fidelity == FidelityLevel.CONDUCTANCE:
                    am, bm, ah, bh, an, bn = hh_rates(self.v_mV)
                    am, bm, ah, bh, an, bn = am * phi, bm * phi, ah * phi, bh * phi, an * phi, bn * phi
                    rm = am + bm + 1e-12
                    rh = ah + bh + 1e-12
                    rn = an + bn + 1e-12
                    self.m_gate = (am / rm) + (self.m_gate - am / rm) * np.exp(-sub_dt * rm)
                    self.h_gate = (ah / rh) + (self.h_gate - ah / rh) * np.exp(-sub_dt * rh)
                    self.n_gate = (an / rn) + (self.n_gate - an / rn) * np.exp(-sub_dt * rn)
                    g_na = g_na_max * (self.m_gate ** 3) * self.h_gate
                    g_k = g_k_max * (self.n_gate ** 4)
                    g_tot = g_na + g_k + self.g_leak + self._axial_g_row_sum + 1e-12
                    i_ax_in = self.axial_g @ self.v_mV
                    v_inf = (g_na * e_na + g_k * e_k + self.g_leak * e_l + i_ax_in + i_stim) / g_tot
                    self.v_mV = np.clip(v_inf + (self.v_mV - v_inf) * np.exp(-sub_dt * g_tot / self.cm), -110.0, 70.0)
                else:
                    def _deriv(v, m, h, ng):
                        am, bm, ah, bh, an, bn = hh_rates(v)
                        dm = phi * (am * (1.0 - m) - bm * m)
                        dh = phi * (ah * (1.0 - h) - bh * h)
                        dn = phi * (an * (1.0 - ng) - bn * ng)
                        i_ion = g_na_max * (m ** 3) * h * (v - e_na) + g_k_max * (ng ** 4) * (v - e_k) + self.g_leak * (v - e_l)
                        i_ax = (self.axial_g @ v) - self._axial_g_row_sum * v
                        dv = (i_stim + i_ax - i_ion) / self.cm
                        return dv, dm, dh, dn

                    k1 = _deriv(self.v_mV, self.m_gate, self.h_gate, self.n_gate)
                    k2 = _deriv(
                        self.v_mV + 0.5 * sub_dt * k1[0],
                        np.clip(self.m_gate + 0.5 * sub_dt * k1[1], 0, 1),
                        np.clip(self.h_gate + 0.5 * sub_dt * k1[2], 0, 1),
                        np.clip(self.n_gate + 0.5 * sub_dt * k1[3], 0, 1),
                    )
                    k3 = _deriv(
                        self.v_mV + 0.5 * sub_dt * k2[0],
                        np.clip(self.m_gate + 0.5 * sub_dt * k2[1], 0, 1),
                        np.clip(self.h_gate + 0.5 * sub_dt * k2[2], 0, 1),
                        np.clip(self.n_gate + 0.5 * sub_dt * k2[3], 0, 1),
                    )
                    k4 = _deriv(
                        self.v_mV + sub_dt * k3[0],
                        np.clip(self.m_gate + sub_dt * k3[1], 0, 1),
                        np.clip(self.h_gate + sub_dt * k3[2], 0, 1),
                        np.clip(self.n_gate + sub_dt * k3[3], 0, 1),
                    )
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
        if adapt and abs(reward_signal) > 1e-6 and self.edges_compiled:
            changed = False
            for i_ec, ec in enumerate(self.edges_compiled):
                if ec["plasticity_rule"] != "none":
                    dw = 0.015 * reward_signal * float(self.prev_state[ec["src"]] * self.state[ec["dst"]])
                    nw = float(np.clip(ec["weight"] + dw, -4.0, 4.0))
                    ec["weight"] = nw
                    self._e_w[i_ec] = nw
                    changed = True
            if changed:
                for i_ec, ec in enumerate(self.edges_compiled):
                    self.spec.edges[i_ec].weight = ec["weight"]

        step_energy = float(
            np.sum(self.base_metabolic * self.active_mask) * 0.05
            + 0.12 * np.sum(spikes_now)
            + 0.02 * len(self.organelles)
        )
        self.atp_pool = float(np.clip(self.atp_pool - step_energy + 0.45 * (1.0 - self.atp_pool / 100.0), 5.0, 100.0))
        self.cumulative_energy += step_energy
        self.cumulative_flops += int(flops)
        self.step_count += 1

        if self._output_idx_arr.size > 0:
            outputs = self.state[self._output_idx_arr]
            scalar_out = float(np.mean(outputs))
            self.paracrine_emission = scalar_out * 0.25
        else:
            outputs = np.zeros(1, dtype=np.float64)
            scalar_out = 0.0
            self.paracrine_emission = 0.0

        if not collect_telemetry:
            return {
                "output": outputs.copy(),
                "scalar_output": scalar_out,
                "step_energy": round(step_energy, 6),
                "step_flops": int(flops),
                "paracrine_emission": round(self.paracrine_emission, 6),
            }

        return {
            "output": outputs.copy(),
            "scalar_output": scalar_out,
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
        for i_ec, ec in enumerate(self.edges_compiled):
            if ec["dst"] == idx or ec["src"] == idx:
                nw = float(np.clip(ec["weight"] * gain_boost, -4.0, 4.0))
                ec["weight"] = nw
                self._e_w[i_ec] = nw
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

    def export_state(self) -> Dict[str, Any]:
        """Lossless primitive/ndarray state export compatible with synapse.stateio."""
        return {
            "step_count": int(self.step_count),
            "state": self.state.copy(),
            "prev_state": self.prev_state.copy(),
            "v_mV": self.v_mV.copy(),
            "u_adapt": self.u_adapt.copy(),
            "refractory_ms": self.refractory_ms.copy(),
            "spikes": self.spikes.copy(),
            "m_gate": self.m_gate.copy(),
            "h_gate": self.h_gate.copy(),
            "n_gate": self.n_gate.copy(),
            "g_exc": self.g_exc.copy(),
            "g_inh": self.g_inh.copy(),
            "active_mask": self.active_mask.copy(),
            "atp_pool": float(self.atp_pool),
            "cumulative_energy": float(self.cumulative_energy),
            "cumulative_flops": int(self.cumulative_flops),
            "edge_weights": self._e_w.copy(),
            "edge_ring": self._edge_ring.copy(),
            "ring_ptr": int(self._ring_ptr),
            "paracrine_edge_ema": self._paracrine_edge_ema.copy(),
            "paracrine_emission": float(self.paracrine_emission),
            "organelles": {oid: org.state_dict() for oid, org in self.organelles.items()},
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Restore exact internal dynamical state from export_state()."""
        self.step_count = int(state["step_count"])
        for arr_key in (
            "state", "prev_state", "v_mV", "u_adapt", "refractory_ms", "spikes",
            "m_gate", "h_gate", "n_gate", "g_exc", "g_inh", "active_mask",
        ):
            if arr_key in state:
                setattr(self, arr_key, np.asarray(state[arr_key], dtype=np.float64).copy())
        self.atp_pool = float(state.get("atp_pool", 100.0))
        self.cumulative_energy = float(state.get("cumulative_energy", 0.0))
        self.cumulative_flops = int(state.get("cumulative_flops", 0))
        if "edge_weights" in state:
            self._e_w = np.asarray(state["edge_weights"], dtype=np.float64).copy()
            for i_ec, ec in enumerate(self.edges_compiled):
                ec["weight"] = float(self._e_w[i_ec])
        if "edge_ring" in state:
            self._edge_ring = np.asarray(state["edge_ring"], dtype=np.float64).copy()
        self._ring_ptr = int(state.get("ring_ptr", 0))
        if "paracrine_edge_ema" in state:
            self._paracrine_edge_ema = np.asarray(state["paracrine_edge_ema"], dtype=np.float64).copy()
        self.paracrine_emission = float(state.get("paracrine_emission", 0.0))
        org_states = state.get("organelles", {})
        if isinstance(org_states, Mapping):
            for oid, ostate in org_states.items():
                if oid in self.organelles and hasattr(self.organelles[oid], "load_state_dict"):
                    self.organelles[oid].load_state_dict(ostate)


class MultiFidelityConnectome:
    """Executes a heterogeneous ConnectomeGraph where neurons of Fidelity 1..5 coexist.

    Uses 3D spatial distance for paracrine chemical diffusion, compartment/receptor-aware
    synaptic routing, and vectorized circular delay rings across heterogeneous morphologies.
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
        coords = []
        for idx, nspec in enumerate(graph.neurons):
            self.neuron_ids.append(nspec.morphology_id)
            self.neurons[nspec.morphology_id] = StructuredNeuronInstance(
                nspec, dt_ms=dt_ms, organelle_factory=organelle_factory, seed=seed + idx
            )
            c0 = nspec.compartments[0].coords_3d if nspec.compartments else (float(idx), 0.0, 0.0)
            coords.append(c0)
        self.id_to_idx = {nid: i for i, nid in enumerate(self.neuron_ids)}
        n_cells = len(self.neuron_ids)

        # Precompute 3D spatial paracrine diffusion kernel K_ij = exp(-d_ij / lambda_diff) normalized row-wise
        if n_cells > 0:
            pos_mat = np.asarray(coords, dtype=np.float64)
            dmat = np.linalg.norm(pos_mat[:, None, :] - pos_mat[None, :, :], axis=-1)
            K = np.exp(-dmat / 15.0)
            row_s = np.sum(K, axis=1, keepdims=True) + 1e-8
            self._paracrine_kernel = K / row_s
        else:
            self._paracrine_kernel = np.zeros((0, 0), dtype=np.float64)

        self._synapses: List[Dict[str, Any]] = []
        s_pre, s_post, s_w, s_delay, s_dale, s_edge_idx = [], [], [], [], [], []
        receptor_scale = {
            "ampa": 1.0,
            "nmda": 1.15,
            "gaba_a": 1.0,
            "gaba_b": 0.75,
            "dopamine": 0.85,
            "acetylcholine": 0.90,
            "paracrine": 0.65,
        }
        for e_idx, e in enumerate(graph.edges):
            if e.active and e.pre_id in self.neurons and e.post_id in self.neurons:
                delay_steps = min(64, max(0, int(round(e.delay_ms / max(self.dt_ms, 1e-3)))))
                pi = self.id_to_idx[e.pre_id]
                qi = self.id_to_idx[e.post_id]
                dale_s = self.neurons[e.pre_id].spec.dale_sign
                r_gain = receptor_scale.get(e.receptor_type, 1.0)
                self._synapses.append({
                    "pre": e.pre_id,
                    "post": e.post_id,
                    "pre_idx": pi,
                    "post_idx": qi,
                    "edge_idx": e_idx,
                    "weight": float(e.weight),
                    "receptor_gain": r_gain,
                    "source_compartment": e.source_compartment,
                    "target_compartment": e.target_compartment,
                    "delay_steps": delay_steps,
                    "plasticity_rule": e.plasticity_rule,
                    "dale_sign": dale_s,
                })
                s_pre.append(pi)
                s_post.append(qi)
                s_w.append(float(e.weight))
                s_delay.append(delay_steps)
                s_dale.append(dale_s)
                s_edge_idx.append(e_idx)

        self._syn_pre = np.asarray(s_pre, dtype=np.int32)
        self._syn_post = np.asarray(s_post, dtype=np.int32)
        self._syn_w = np.asarray(s_w, dtype=np.float64)
        self._syn_rgain = np.array([s["receptor_gain"] for s in self._synapses], dtype=np.float64) if self._synapses else np.zeros(0)
        self._syn_delay = np.asarray(s_delay, dtype=np.int32)
        self._syn_dale = np.asarray(s_dale, dtype=np.float64)
        self._syn_edge_idx = np.asarray(s_edge_idx, dtype=np.int32)
        self._syn_arange = np.arange(self._syn_pre.size, dtype=np.int32)
        self._syn_ring_len = (int(np.max(self._syn_delay)) if self._syn_delay.size > 0 else 0) + 2
        self._syn_ring = np.zeros((self._syn_ring_len, self._syn_pre.size), dtype=np.float64)
        self._syn_ring_ptr = 0
        self._has_target_comp_routing = any(s["target_compartment"] is not None for s in self._synapses)

        self.last_outputs = np.zeros(n_cells, dtype=np.float64)
        self.prev_outputs = np.zeros(n_cells, dtype=np.float64)
        self.step_count = 0
        self.total_flops = 0
        self.total_energy = 0.0

    def reset(self) -> None:
        self.step_count = 0
        self.total_flops = 0
        self.total_energy = 0.0
        self.last_outputs.fill(0.0)
        self.prev_outputs.fill(0.0)
        self._syn_ring.fill(0.0)
        self._syn_ring_ptr = 0
        for inst in self.neurons.values():
            inst.reset()

    def export_weight_matrix(self) -> np.ndarray:
        """Export current [N, N] effective synaptic adjacency matrix [post, pre]."""
        n = len(self.neuron_ids)
        W = np.zeros((n, n), dtype=np.float64)
        if self._syn_pre.size > 0:
            np.add.at(W, (self._syn_post, self._syn_pre), self._syn_w)
        return W

    def apply_weight_delta(self, dW: np.ndarray) -> None:
        """Apply bounded [N, N] synaptic weight delta to existing synapses while enforcing Dale's sign."""
        if self._syn_pre.size == 0:
            return
        deltas = dW[self._syn_post, self._syn_pre]
        new_w = self._syn_w + deltas
        self._syn_w = np.where(self._syn_dale > 0, np.clip(new_w, 0.0, 5.0), np.clip(new_w, -5.0, 0.0))
        for k, syn in enumerate(self._synapses):
            w_val = float(self._syn_w[k])
            syn["weight"] = w_val
            self.graph.edges[syn["edge_idx"]].weight = w_val

    def step(
        self,
        external_inputs: Mapping[str, float | Sequence[float]] | np.ndarray,
        modulators: Optional[Mapping[str, float]] = None,
        reward_signal: float = 0.0,
        adapt: bool = False,
        collect_telemetry: bool = True,
    ) -> Dict[str, Any]:
        n_cells = len(self.neuron_ids)
        # Vectorized synaptic delay gather
        target_comp_drives: Dict[int, Dict[str, float]] = {}
        if self._syn_pre.size > 0:
            read_rows = (self._syn_ring_ptr - self._syn_delay - 1) % self._syn_ring_len
            pre_vals = self._syn_ring[read_rows, self._syn_arange]
            # Support optional source_compartment readouts
            src_signals = self.last_outputs[self._syn_pre]
            for s_i, syn in enumerate(self._synapses):
                sc = syn["source_compartment"]
                if sc is not None:
                    p_inst = self.neurons[syn["pre"]]
                    if sc in p_inst.id_to_idx:
                        src_signals[s_i] = float(p_inst.state[p_inst.id_to_idx[sc]])
            self._syn_ring[self._syn_ring_ptr, :] = src_signals
            self._syn_ring_ptr = (self._syn_ring_ptr + 1) % self._syn_ring_len

            weighted_syn = self._syn_w * self._syn_rgain * pre_vals
            syn_drive_arr = np.bincount(self._syn_post, weights=weighted_syn, minlength=n_cells)
            if self._has_target_comp_routing:
                for s_i, syn in enumerate(self._synapses):
                    tc = syn["target_compartment"]
                    if tc is not None:
                        qi = syn["post_idx"]
                        syn_drive_arr[qi] -= weighted_syn[s_i]
                        target_comp_drives.setdefault(qi, {})
                        target_comp_drives[qi][tc] = target_comp_drives[qi].get(tc, 0.0) + float(weighted_syn[s_i])
        else:
            syn_drive_arr = np.zeros(n_cells, dtype=np.float64)

        # 3D spatial paracrine diffusion field per neuron
        if n_cells > 0:
            emissions = np.array([self.neurons[nid].paracrine_emission for nid in self.neuron_ids], dtype=np.float64)
            paracrine_per_cell = self._paracrine_kernel @ emissions
            paracrine_level = float(np.mean(paracrine_per_cell))
        else:
            paracrine_per_cell = np.zeros(0, dtype=np.float64)
            paracrine_level = 0.0

        ext_map: Dict[str, Any] = {}
        if isinstance(external_inputs, Mapping):
            ext_map = dict(external_inputs)
        else:
            arr = np.asarray(external_inputs, dtype=np.float64).ravel()
            targets = self.graph.sensory_ids or self.neuron_ids
            for k, nid in enumerate(targets):
                if k < arr.size:
                    ext_map[nid] = float(arr[k])

        step_flops = int(self._syn_pre.size * 4)
        step_energy = 0.0
        new_outputs = np.zeros(n_cells, dtype=np.float64)
        fidelity_counts: Dict[int, int] = {}

        for i, nid in enumerate(self.neuron_ids):
            inst = self.neurons[nid]
            f_int = int(inst.spec.fidelity)
            fidelity_counts[f_int] = fidelity_counts.get(f_int, 0) + 1
            ext_v = ext_map.get(nid, 0.0)
            s_drv = float(syn_drive_arr[i])
            if i in target_comp_drives:
                cmap = dict(target_comp_drives[i])
                base_val = float(ext_v) if isinstance(ext_v, (int, float)) else float(np.mean(ext_v))
                for in_cid in inst.spec.input_compartments:
                    cmap[in_cid] = cmap.get(in_cid, 0.0) + base_val + s_drv
                total_in: Any = cmap
            elif isinstance(ext_v, (int, float)):
                total_in = float(ext_v) + s_drv
            else:
                total_in = np.asarray(ext_v, dtype=np.float64) + s_drv

            res = inst.step(
                total_in,
                modulators=modulators,
                reward_signal=reward_signal,
                paracrine_field=float(paracrine_per_cell[i]),
                adapt=adapt,
                collect_telemetry=collect_telemetry,
            )
            new_outputs[i] = res["scalar_output"]
            step_flops += res["step_flops"]
            step_energy += res["step_energy"]

        # Rule-specific Dale-compliant synaptic adaptation synced to ConnectomeGraph
        if adapt and self._syn_pre.size > 0 and (abs(reward_signal) > 1e-6 or any(s["plasticity_rule"] in ("hebbian", "stdp", "bcm", "predictive_coding") for s in self._synapses)):
            for k, syn in enumerate(self._synapses):
                rule = syn["plasticity_rule"]
                if rule == "none":
                    continue
                pi = syn["pre_idx"]
                qi = syn["post_idx"]
                pre_r = float(self.last_outputs[pi])
                post_r = float(new_outputs[qi])
                if rule == "stdp":
                    dw = 0.012 * (post_r * float(self.prev_outputs[pi]) - 0.85 * float(self.last_outputs[qi]) * pre_r)
                elif rule == "bcm":
                    dw = 0.012 * post_r * (post_r - 0.25) * pre_r
                elif rule == "hebbian":
                    dw = 0.010 * (post_r * pre_r - (post_r ** 2) * syn["weight"])
                elif rule == "predictive_coding":
                    dw = 0.012 * post_r * (pre_r - syn["weight"] * post_r)
                else:
                    dw = 0.012 * reward_signal * pre_r * post_r
                nw = syn["weight"] + dw
                nw = float(np.clip(nw, 0.0, 5.0)) if syn["dale_sign"] > 0 else float(np.clip(nw, -5.0, 0.0))
                syn["weight"] = nw
                self._syn_w[k] = nw
                self.graph.edges[syn["edge_idx"]].weight = nw

        self.prev_outputs = self.last_outputs.copy()
        self.last_outputs = new_outputs
        self.step_count += 1
        self.total_flops += step_flops
        self.total_energy += step_energy

        motor_ids = self.graph.motor_ids or self.neuron_ids[-min(4, n_cells) :]
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

    def export_state(self) -> Dict[str, Any]:
        """Lossless primitive/ndarray state export for checkpoint bit-identity."""
        return {
            "step_count": int(self.step_count),
            "total_flops": int(self.total_flops),
            "total_energy": float(self.total_energy),
            "last_outputs": self.last_outputs.copy(),
            "prev_outputs": self.prev_outputs.copy(),
            "syn_weights": self._syn_w.copy(),
            "syn_ring": self._syn_ring.copy(),
            "syn_ring_ptr": int(self._syn_ring_ptr),
            "neurons": {nid: inst.export_state() for nid, inst in self.neurons.items()},
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Restore exact connectome state from export_state()."""
        self.step_count = int(state["step_count"])
        self.total_flops = int(state.get("total_flops", 0))
        self.total_energy = float(state.get("total_energy", 0.0))
        if "last_outputs" in state:
            self.last_outputs = np.asarray(state["last_outputs"], dtype=np.float64).copy()
        if "prev_outputs" in state:
            self.prev_outputs = np.asarray(state["prev_outputs"], dtype=np.float64).copy()
        if "syn_weights" in state:
            self._syn_w = np.asarray(state["syn_weights"], dtype=np.float64).copy()
            for k, syn in enumerate(self._synapses):
                w_val = float(self._syn_w[k])
                syn["weight"] = w_val
                self.graph.edges[syn["edge_idx"]].weight = w_val
        if "syn_ring" in state:
            self._syn_ring = np.asarray(state["syn_ring"], dtype=np.float64).copy()
        self._syn_ring_ptr = int(state.get("syn_ring_ptr", 0))
        n_states = state.get("neurons", {})
        if isinstance(n_states, Mapping):
            for nid, nst in n_states.items():
                if nid in self.neurons:
                    self.neurons[nid].import_state(nst)
