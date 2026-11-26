"""Vectorized Structure-of-Arrays (SoA) & Sparse CSR Multi-Compartment Connectome Compiler.

Directly resolves Open Research Question #1 from ``LIMITATIONS_AND_OPEN_QUESTIONS.md``:
compiles an arbitrary heterogeneous ``ConnectomeGraph`` (coexisting Fidelity 1..4
multi-compartment neurons, physical cable conductances, discrete axonal/synaptic
delays, NMDA coincidence gates, LIF spiking, and Rush-Larsen Hodgkin-Huxley channels)
into contiguous flat arrays and SciPy CSR sparse operators so 256-4,096+ structured
neurons run at high throughput on standard laptop CPUs without Python per-cell loops.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy import sparse

from synapse.frontier.hh_cable import hh_rates, steady_state_gates, temperature_factor

from .compartment_engine import MultiFidelityConnectome, _nmda_mg_block
from .interchange import CompartmentKind, ConnectomeGraph, FidelityLevel


class CompiledConnectomeEngine:
    """High-throughput vectorized multi-compartment multi-fidelity connectome simulator."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        dt_ms: float = 1.0,
        temperature_c: float = 18.5,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.dt_ms = float(dt_ms)
        self.temperature_c = float(temperature_c)
        self.seed = int(seed)
        self.phi = float(temperature_factor(self.temperature_c))
        self._compile_soa()
        self.reset()

    def _compile_soa(self) -> None:
        neurons = self.graph.neurons
        self.n_neurons = len(neurons)
        self.neuron_ids = [u.morphology_id for u in neurons]
        self.nid_to_idx = {nid: i for i, nid in enumerate(self.neuron_ids)}

        comp_offsets: List[int] = []
        comp_counts: List[int] = []
        comp_lookup: Dict[Tuple[int, str], int] = {}
        input_global_by_neuron: List[np.ndarray] = []
        output_global_by_neuron: List[np.ndarray] = []

        fidelities: List[int] = []
        active_list: List[float] = []
        cm_list: List[float] = []
        r_ax_list: List[float] = []
        g_leak_list: List[float] = []
        e_rest_list: List[float] = []
        v_th_list: List[float] = []
        tau_list: List[float] = []
        ampa_list: List[float] = []
        nmda_list: List[float] = []
        gaba_list: List[float] = []
        metab_list: List[float] = []
        dale_by_neuron = np.zeros(self.n_neurons, dtype=np.float64)

        cursor = 0
        for ni, spec in enumerate(neurons):
            dale_by_neuron[ni] = float(spec.dale_sign)
            comp_offsets.append(cursor)
            k_len = len(spec.compartments)
            comp_counts.append(k_len)
            fid = int(spec.fidelity)
            if fid == int(FidelityLevel.REFERENCE_CABLE):
                # Map F5 to Rush-Larsen F4 in compiled batch mode; F5 RK4 remains available on StructuredNeuronInstance
                fid = int(FidelityLevel.CONDUCTANCE)
            for ci, c in enumerate(spec.compartments):
                gi = cursor + ci
                comp_lookup[(ni, c.compartment_id)] = gi
                fidelities.append(fid)
                active_list.append(1.0 if c.active else 0.0)
                cm_list.append(c.effective_cm_uF_cm2)
                r_ax_list.append(c.axial_resistance_MOhm)
                g_leak_list.append(c.g_leak_mS_cm2)
                e_rest_list.append(c.e_rest_mV)
                v_th_list.append(c.threshold_mV)
                tau_list.append(c.effective_time_constant_ms)
                ampa_list.append(c.receptors.ampa)
                nmda_list.append(c.receptors.nmda)
                gaba_list.append(c.receptors.gaba_a + 0.5 * c.receptors.gaba_b)
                metab_list.append(c.baseline_metabolic_cost)

            in_g = [comp_lookup[(ni, cid)] for cid in spec.input_compartments if (ni, cid) in comp_lookup]
            out_g = [comp_lookup[(ni, cid)] for cid in spec.output_compartments if (ni, cid) in comp_lookup]
            if not in_g and k_len > 0:
                in_g = [cursor]
            if not out_g and k_len > 0:
                out_g = [cursor + k_len - 1]
            input_global_by_neuron.append(np.asarray(in_g, dtype=np.int32))
            output_global_by_neuron.append(np.asarray(out_g, dtype=np.int32))
            cursor += k_len

        self.n_comp_total = cursor
        self.dale_by_neuron = dale_by_neuron
        self.input_global_by_neuron = input_global_by_neuron
        self.output_global_by_neuron = output_global_by_neuron

        # Flat output aggregation matrix P_out [N_neurons, C_total] averaging output compartments per neuron
        out_rows, out_cols, out_vals = [], [], []
        for ni, o_arr in enumerate(output_global_by_neuron):
            if o_arr.size > 0:
                w = 1.0 / float(o_arr.size)
                for gi in o_arr:
                    out_rows.append(ni)
                    out_cols.append(int(gi))
                    out_vals.append(w)
        self.P_out = sparse.csr_matrix(
            (out_vals, (out_rows, out_cols)),
            shape=(self.n_neurons, max(self.n_comp_total, 1)),
            dtype=np.float64,
        )

        # Sensory broadcast indices
        sens_ids = self.graph.sensory_ids or self.neuron_ids[: min(8, self.n_neurons)]
        self.sensory_neuron_indices = [self.nid_to_idx[s] for s in sens_ids if s in self.nid_to_idx]
        mot_ids = self.graph.motor_ids or self.neuron_ids[-min(4, self.n_neurons) :]
        self.motor_neuron_indices = [self.nid_to_idx[m] for m in mot_ids if m in self.nid_to_idx]

        self.fidelity_arr = np.asarray(fidelities, dtype=np.int32)
        self.mask_f1 = self.fidelity_arr == int(FidelityLevel.ABSTRACT_RATE)
        self.mask_f2 = self.fidelity_arr == int(FidelityLevel.MULTI_COMPARTMENT_RECURRENT)
        self.mask_f3 = self.fidelity_arr == int(FidelityLevel.SPIKING)
        self.mask_f4 = self.fidelity_arr == int(FidelityLevel.CONDUCTANCE)
        self.has_f3 = bool(np.any(self.mask_f3))
        self.has_f4 = bool(np.any(self.mask_f4))

        self.active_mask = np.asarray(active_list, dtype=np.float64)
        self.cm = np.asarray(cm_list, dtype=np.float64)
        self.r_axial = np.asarray(r_ax_list, dtype=np.float64)
        self.g_leak = np.asarray(g_leak_list, dtype=np.float64)
        self.e_rest = np.asarray(e_rest_list, dtype=np.float64)
        self.v_thresh = np.asarray(v_th_list, dtype=np.float64)
        self.tau_ms = np.asarray(tau_list, dtype=np.float64)
        self.alpha_f1 = np.clip(self.dt_ms / np.maximum(self.tau_ms, 0.2), 0.02, 1.0)
        self.alpha_f2 = np.clip(self.dt_ms / np.maximum(self.tau_ms, 0.2), 0.01, 0.95)
        self.ampa_gain = np.asarray(ampa_list, dtype=np.float64)
        self.nmda_gain = np.asarray(nmda_list, dtype=np.float64)
        self.gaba_gain = np.asarray(gaba_list, dtype=np.float64)
        self.base_metabolic = np.asarray(metab_list, dtype=np.float64)

        # Compile internal axial & directed couplings + inter-neuron synapses
        ax_rows, ax_cols, ax_vals = [], [], []
        fast_rows, fast_cols, fast_vals = [], [], []
        del_src, del_dst, del_w, del_steps, del_gate = [], [], [], [], []

        for ni, spec in enumerate(neurons):
            for e in spec.edges:
                si = comp_lookup.get((ni, e.source_id))
                ti = comp_lookup.get((ni, e.target_id))
                if si is None or ti is None:
                    continue
                inv_r = 1.0 / max(0.5 * (self.r_axial[si] + self.r_axial[ti]), 0.1)
                eff_g = float(np.clip(inv_r * 0.25, 0.02, 2.5))
                if e.coupling_type in ("electrical_axial", "gap_junction"):
                    g_val = abs(e.weight) * eff_g
                    ax_rows.extend([ti, si])
                    ax_cols.extend([si, ti])
                    ax_vals.extend([g_val, g_val * (1.0 if e.coupling_type == "gap_junction" else 0.35)])
                src_c = spec.compartments[si - comp_offsets[ni]]
                tot_d = e.delay_ms + (
                    src_c.transmission_delay_ms
                    if src_c.kind in (CompartmentKind.AXON_SEGMENT, CompartmentKind.AXON_FEEDBACK)
                    else 0.0
                )
                d_st = min(64, max(0, int(round(tot_d / max(self.dt_ms, 1e-3)))))
                g_idx = comp_lookup.get((ni, e.gated_by), -1) if e.gated_by else -1
                if d_st == 0 and g_idx < 0 and e.coupling_type != "electrical_axial":
                    fast_rows.append(ti)
                    fast_cols.append(si)
                    fast_vals.append(float(e.weight))
                else:
                    w_eff = float(e.weight) * (eff_g if e.coupling_type == "electrical_axial" else 1.0)
                    del_src.append(si)
                    del_dst.append(ti)
                    del_w.append(w_eff)
                    del_steps.append(d_st)
                    del_gate.append(g_idx)

        # Inter-neuron connectome synapses mapped directly to global source/target compartments
        self.syn_start_idx = len(del_src)
        self.syn_dale_signs: List[float] = []
        for syn in self.graph.edges:
            if not syn.active:
                continue
            p_ni = self.nid_to_idx.get(syn.pre_id)
            q_ni = self.nid_to_idx.get(syn.post_id)
            if p_ni is None or q_ni is None:
                continue
            src_gi = (
                comp_lookup.get((p_ni, syn.source_compartment))
                if syn.source_compartment
                else None
            )
            if src_gi is None:
                src_gi = int(output_global_by_neuron[p_ni][0])
            dst_gi = (
                comp_lookup.get((q_ni, syn.target_compartment))
                if syn.target_compartment
                else None
            )
            if dst_gi is None:
                dst_gi = int(input_global_by_neuron[q_ni][0])
            d_st = min(64, max(1, int(round(syn.delay_ms / max(self.dt_ms, 1e-3)))))
            del_src.append(src_gi)
            del_dst.append(dst_gi)
            del_w.append(float(syn.weight))
            del_steps.append(d_st)
            del_gate.append(-1)
            self.syn_dale_signs.append(dale_by_neuron[p_ni])

        c_dim = max(self.n_comp_total, 1)
        self.W_axial = sparse.csr_matrix((ax_vals, (ax_rows, ax_cols)), shape=(c_dim, c_dim), dtype=np.float64)
        self.axial_row_sum = np.asarray(self.W_axial.sum(axis=1)).ravel()
        self.W_fast = sparse.csr_matrix((fast_vals, (fast_rows, fast_cols)), shape=(c_dim, c_dim), dtype=np.float64)

        self.del_src = np.asarray(del_src, dtype=np.int32)
        self.del_dst = np.asarray(del_dst, dtype=np.int32)
        self.del_w = np.asarray(del_w, dtype=np.float64)
        self.del_steps = np.asarray(del_steps, dtype=np.int32)
        self.del_gate = np.asarray(del_gate, dtype=np.int32)
        self.syn_dale_arr = np.asarray(self.syn_dale_signs, dtype=np.float64)
        self.max_delay = int(np.max(self.del_steps)) if self.del_steps.size > 0 else 1
        self.ring_len = self.max_delay + 2

    def reset(self) -> None:
        c = self.n_comp_total
        self.step_count = 0
        self.state = np.zeros(c, dtype=np.float64)
        self.prev_state = np.zeros(c, dtype=np.float64)
        self.v_mV = self.e_rest.copy()
        self.u_adapt = np.zeros(c, dtype=np.float64)
        self.refractory_ms = np.zeros(c, dtype=np.float64)
        self.spikes = np.zeros(c, dtype=np.float64)
        self.g_exc = np.zeros(c, dtype=np.float64)
        self.g_inh = np.zeros(c, dtype=np.float64)
        if c > 0:
            m0, h0, n0 = steady_state_gates(self.v_mV)
            self.m_gate = m0.astype(np.float64).copy()
            self.h_gate = h0.astype(np.float64).copy()
            self.n_gate = n0.astype(np.float64).copy()
        else:
            self.m_gate = np.zeros(0, dtype=np.float64)
            self.h_gate = np.zeros(0, dtype=np.float64)
            self.n_gate = np.zeros(0, dtype=np.float64)
        self.ring_buf = np.zeros((self.ring_len, self.del_src.size), dtype=np.float64)
        self.ring_ptr = 0
        self.last_outputs = np.zeros(self.n_neurons, dtype=np.float64)
        self.total_flops = 0
        self.total_energy = 0.0

    def step(
        self,
        external_inputs: np.ndarray | Sequence[float] | Mapping[str, float],
        reward_signal: float = 0.0,
        adapt: bool = False,
    ) -> Dict[str, Any]:
        c = self.n_comp_total
        if c == 0:
            return {"step": self.step_count, "outputs": np.zeros(0), "motor_outputs": [], "step_flops": 0, "step_energy": 0.0}

        ext = np.zeros(c, dtype=np.float64)
        if isinstance(external_inputs, Mapping):
            for nid, val in external_inputs.items():
                ni = self.nid_to_idx.get(nid)
                if ni is not None:
                    ext[self.input_global_by_neuron[ni]] = float(val)
        else:
            arr = np.asarray(external_inputs, dtype=np.float64).ravel()
            limit = min(arr.size, len(self.sensory_neuron_indices))
            for k in range(limit):
                ni = self.sensory_neuron_indices[k]
                ext[self.input_global_by_neuron[ni]] = float(arr[k])

        total_ext = (np.maximum(ext, 0.0) * self.ampa_gain + np.minimum(ext, 0.0) * self.gaba_gain) * self.active_mask

        # Gather delayed & fast edge currents in one vectorized pass
        drive = self.W_fast.dot(self.state)
        if self.del_src.size > 0:
            read_rows = (self.ring_ptr - self.del_steps) % self.ring_len
            delayed_vals = self.ring_buf[read_rows, np.arange(self.del_src.size)]
            self.ring_buf[self.ring_ptr, :] = self.state[self.del_src]
            self.ring_ptr = (self.ring_ptr + 1) % self.ring_len
            gates = np.ones(self.del_src.size, dtype=np.float64)
            has_g = self.del_gate >= 0
            if np.any(has_g):
                g_states = self.state[self.del_gate[has_g]]
                gates[has_g] = 1.0 / (1.0 + np.exp(-np.clip(3.0 * g_states, -20.0, 20.0)))
            drive += np.bincount(self.del_dst, weights=self.del_w * gates * delayed_vals, minlength=c)

        self.prev_state = self.state.copy()
        new_state = self.state.copy()
        spikes_now = np.zeros(c, dtype=np.float64)

        # Fidelity 1: Abstract Rate
        if np.any(self.mask_f1):
            m1 = self.mask_f1
            a1 = self.alpha_f1[m1]
            new_state[m1] = ((1.0 - a1) * self.state[m1] + a1 * np.tanh(total_ext[m1] + drive[m1])) * self.active_mask[m1]
            self.v_mV[m1] = self.e_rest[m1] + 25.0 * new_state[m1]

        # Fidelity 2: Multi-Compartment Recurrent + NMDA coincidence
        if np.any(self.mask_f2):
            m2 = self.mask_f2
            a2 = self.alpha_f2[m2]
            ax_cur = self.W_axial.dot(self.state)[m2] - self.axial_row_sum[m2] * self.state[m2]
            nmda = self.nmda_gain[m2] * np.maximum(self.state[m2], 0.0) * np.maximum(total_ext[m2], 0.0)
            pre2 = total_ext[m2] + nmda + drive[m2] + 0.35 * ax_cur - 0.1 * self.u_adapt[m2]
            new_state[m2] = ((1.0 - a2) * self.state[m2] + a2 * np.tanh(pre2)) * self.active_mask[m2]
            self.u_adapt[m2] = 0.94 * self.u_adapt[m2] + 0.06 * np.abs(new_state[m2])
            self.v_mV[m2] = self.e_rest[m2] + 30.0 * new_state[m2]
            spikes_now[m2] = (new_state[m2] > 0.72).astype(np.float64)

        # Fidelity 3: Spiking LIF / Adaptive Izhikevich
        if self.has_f3:
            m3 = self.mask_f3
            tot3 = total_ext[m3] + drive[m3]
            self.g_exc[m3] = 0.85 * self.g_exc[m3] + np.maximum(tot3, 0.0)
            self.g_inh[m3] = 0.80 * self.g_inh[m3] + np.maximum(-tot3, 0.0)
            nmda_f = _nmda_mg_block(self.v_mV[m3])
            i_syn = (
                self.g_exc[m3] * (1.0 + 0.4 * self.nmda_gain[m3] * nmda_f) * (-self.v_mV[m3])
                + self.g_inh[m3] * (-80.0 - self.v_mV[m3])
            ) * 0.18
            i_ax = self.W_axial.dot(self.v_mV)[m3] - self.axial_row_sum[m3] * self.v_mV[m3]
            not_ref = (self.refractory_ms[m3] <= 0.0).astype(np.float64)
            dv = ((self.e_rest[m3] - self.v_mV[m3]) + 18.0 * i_syn + 4.0 * i_ax - 8.0 * self.u_adapt[m3]) / self.tau_ms[m3]
            v_next = np.clip(self.v_mV[m3] + self.dt_ms * dv * not_ref, -100.0, 50.0)
            fired = (v_next >= self.v_thresh[m3]) & (self.active_mask[m3] > 0.5)
            spk3 = fired.astype(np.float64)
            spikes_now[m3] = spk3
            self.v_mV[m3] = np.where(fired, self.e_rest[m3] - 5.0, v_next)
            ref_next = np.maximum(0.0, self.refractory_ms[m3] - self.dt_ms)
            self.refractory_ms[m3] = np.where(fired, 2.0, ref_next)
            self.u_adapt[m3] = 0.96 * self.u_adapt[m3] + 0.25 * spk3
            new_state[m3] = np.clip(0.8 * self.state[m3] + 0.55 * spk3 + 0.015 * (self.v_mV[m3] - self.e_rest[m3]), -1.0, 1.0)

        # Fidelity 4: Conductance Hodgkin-Huxley (Rush-Larsen Exp-Euler)
        if self.has_f4:
            m4 = self.mask_f4
            i_stim = (total_ext[m4] + drive[m4]) * 18.0
            substeps = 5
            sub_dt = self.dt_ms / substeps
            v_prev = self.v_mV[m4].copy()
            for _ in range(substeps):
                v4 = self.v_mV[m4]
                am, bm, ah, bh, an, bn = hh_rates(v4)
                am, bm, ah, bh, an, bn = am * self.phi, bm * self.phi, ah * self.phi, bh * self.phi, an * self.phi, bn * self.phi
                rm = am + bm + 1e-12
                rh = ah + bh + 1e-12
                rn = an + bn + 1e-12
                self.m_gate[m4] = (am / rm) + (self.m_gate[m4] - am / rm) * np.exp(-sub_dt * rm)
                self.h_gate[m4] = (ah / rh) + (self.h_gate[m4] - ah / rh) * np.exp(-sub_dt * rh)
                self.n_gate[m4] = (an / rn) + (self.n_gate[m4] - an / rn) * np.exp(-sub_dt * rn)
                g_na = 120.0 * (self.m_gate[m4] ** 3) * self.h_gate[m4]
                g_k = 36.0 * (self.n_gate[m4] ** 4)
                g_tot = g_na + g_k + self.g_leak[m4] + self.axial_row_sum[m4] + 1e-12
                i_ax_in = self.W_axial.dot(self.v_mV)[m4]
                v_inf = (g_na * 50.0 + g_k * (-77.0) + self.g_leak[m4] * (-54.387) + i_ax_in + i_stim) / g_tot
                self.v_mV[m4] = np.clip(v_inf + (v4 - v_inf) * np.exp(-sub_dt * g_tot / self.cm[m4]), -110.0, 70.0)
                spikes_now[m4] = np.maximum(spikes_now[m4], ((v_prev < 0.0) & (self.v_mV[m4] >= 0.0)).astype(np.float64))
                v_prev = self.v_mV[m4].copy()
            new_state[m4] = np.tanh((self.v_mV[m4] - self.e_rest[m4]) / 35.0) * self.active_mask[m4]

        self.state = new_state
        self.spikes = spikes_now
        outputs = self.P_out.dot(self.state)

        # Dale-compliant vectorized synaptic adaptation on inter-neuron edges
        if adapt and abs(reward_signal) > 1e-6 and self.del_src.size > self.syn_start_idx:
            s0 = self.syn_start_idx
            pre_s = self.prev_state[self.del_src[s0:]]
            post_s = self.state[self.del_dst[s0:]]
            dw = 0.012 * reward_signal * pre_s * post_s
            updated = self.del_w[s0:] + dw
            self.del_w[s0:] = np.where(
                self.syn_dale_arr > 0,
                np.clip(updated, 0.0, 5.0),
                np.clip(updated, -5.0, 0.0),
            )

        step_flops = int(16 * c + 4 * self.del_src.size + 2 * self.W_axial.nnz)
        step_energy = float(np.sum(self.base_metabolic * self.active_mask) * 0.05 + 0.12 * np.sum(spikes_now))
        self.last_outputs = outputs
        self.step_count += 1
        self.total_flops += step_flops
        self.total_energy += step_energy

        return {
            "step": self.step_count,
            "outputs": outputs.copy(),
            "motor_outputs": [float(outputs[i]) for i in self.motor_neuron_indices],
            "step_flops": step_flops,
            "step_energy": round(step_energy, 6),
        }

    def rollout(self, stimulus_seq: np.ndarray, reset_first: bool = True) -> np.ndarray:
        """Fast multi-step rollout returning [T, N_neurons] output matrix."""
        if reset_first:
            self.reset()
        T = stimulus_seq.shape[0]
        out = np.zeros((T, self.n_neurons), dtype=np.float64)
        for t in range(T):
            out[t] = self.step(stimulus_seq[t])["outputs"]
        return out

    def export_state(self) -> Dict[str, Any]:
        return {
            "step_count": int(self.step_count),
            "total_flops": int(self.total_flops),
            "total_energy": float(self.total_energy),
            "state": self.state.copy(),
            "prev_state": self.prev_state.copy(),
            "v_mV": self.v_mV.copy(),
            "u_adapt": self.u_adapt.copy(),
            "refractory_ms": self.refractory_ms.copy(),
            "spikes": self.spikes.copy(),
            "g_exc": self.g_exc.copy(),
            "g_inh": self.g_inh.copy(),
            "m_gate": self.m_gate.copy(),
            "h_gate": self.h_gate.copy(),
            "n_gate": self.n_gate.copy(),
            "del_w": self.del_w.copy(),
            "ring_buf": self.ring_buf.copy(),
            "ring_ptr": int(self.ring_ptr),
            "last_outputs": self.last_outputs.copy(),
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        self.step_count = int(state["step_count"])
        self.total_flops = int(state.get("total_flops", 0))
        self.total_energy = float(state.get("total_energy", 0.0))
        for key in (
            "state", "prev_state", "v_mV", "u_adapt", "refractory_ms", "spikes",
            "g_exc", "g_inh", "m_gate", "h_gate", "n_gate", "del_w", "ring_buf", "last_outputs",
        ):
            if key in state:
                setattr(self, key, np.asarray(state[key], dtype=np.float64).copy())
        self.ring_ptr = int(state["ring_ptr"])


def benchmark_compiled_vs_object_engine(
    neuron_counts: Sequence[int] = (32, 128, 512),
    steps: int = 40,
    seed: int = 7,
) -> Dict[str, Any]:
    """Benchmark speedup of CompiledConnectomeEngine vs MultiFidelityConnectome across scales."""
    from ..foundry.organelles import create_organelle
    from ..genesis.population import generate_candidate_genome

    rng = np.random.default_rng(seed)
    stim = rng.normal(0.0, 0.8, size=(steps, 4))
    rows: List[Dict[str, Any]] = []

    for n in neuron_counts:
        graph = generate_candidate_genome("mixed_custom_standard", n_neurons=n, seed=seed).connectome
        # Assign mixed fidelities (F1, F2, F3, F4)
        for i, u in enumerate(graph.neurons):
            u.fidelity = FidelityLevel(1 + (i % 4))

        t0 = time.perf_counter()
        compiled = CompiledConnectomeEngine(graph, dt_ms=1.0, seed=seed)
        out_comp = compiled.rollout(stim)
        comp_ms = (time.perf_counter() - t0) * 1000.0

        obj_ms: Optional[float] = None
        speedup: Optional[float] = None
        if n <= 256:
            t1 = time.perf_counter()
            obj_sim = MultiFidelityConnectome(graph, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
            for k in range(steps):
                obj_sim.step(stim[k], collect_telemetry=False)
            obj_ms = (time.perf_counter() - t1) * 1000.0
            speedup = round(obj_ms / max(comp_ms, 1e-4), 2)

        hz = round(steps / max(comp_ms / 1000.0, 1e-6), 1)
        rows.append({
            "neurons": int(n),
            "total_compartments": compiled.n_comp_total,
            "compiled_wall_ms": round(comp_ms, 3),
            "compiled_steps_per_sec_hz": hz,
            "object_engine_wall_ms": round(obj_ms, 3) if obj_ms is not None else None,
            "speedup_x": speedup,
            "finite_outputs": bool(np.all(np.isfinite(out_comp))),
        })

    # Also verify large-scale 2,048-neuron multi-compartment Fidelity 1-4 throughput
    g_2k = generate_candidate_genome("mixed_custom_standard", n_neurons=2048, seed=seed).connectome
    for i, u in enumerate(g_2k.neurons):
        u.fidelity = FidelityLevel(2 if i % 8 != 0 else 3)
    comp_2k = CompiledConnectomeEngine(g_2k, dt_ms=1.0, seed=seed)
    t2k = time.perf_counter()
    comp_2k.rollout(stim[:20])
    ms_2k = (time.perf_counter() - t2k) * 1000.0
    hz_2k = round(20.0 / max(ms_2k / 1000.0, 1e-6), 1)

    return {
        "experiment": "compiled_soa_vs_object_connectome_benchmark",
        "steps": steps,
        "scale_comparisons": rows,
        "large_2048_multicompartment_probe": {
            "neurons": 2048,
            "compartments": comp_2k.n_comp_total,
            "steps": 20,
            "wall_ms": round(ms_2k, 2),
            "steps_per_sec_hz": hz_2k,
        },
    }
