# SONAPSE MORPHOGENESIS — Known Limitations, Scaling Boundaries & Open Research Questions

## 1. Epistemic & Scientific Boundaries

1. **Biological Realism vs. Computational Usefulness vs. Speculative Synthetic Biology**:
   - **Fidelity 1–3** (`ABSTRACT_RATE`, `MULTI_COMPARTMENT_RECURRENT`, `SPIKING`) are synthetic computational approximations; they do not reproduce intracellular ion-channel biophysics.
   - **Fidelity 4–5** (`CONDUCTANCE`, `REFERENCE_CABLE`) integrate the 1952 Hodgkin–Huxley membrane equations via `synapse.frontier.hh_cable` with physical axial conductance and capacitance scaling, but assume uniform compartment parameters and are NOT clinically validated neurostimulation or in-vivo tissue simulators.
   - **Non-neural biological hybrids** (cardiac pacemaker, immune recognition, endocrine, astrocytic, cochlear, retinal, muscle, vascular, etc.) are strictly software computational abstractions. Zero wet-lab, genetic engineering, or living-tissue experimentation is conducted.
   - **SONAPSE AFFECT** investigates unlabeled emergent persistent dynamical attractors and causal behavior modulation. **No subjective experience, consciousness, or phenomenal feeling is claimed or measurable** by this software.
   - **Internal Perception Simulation** models top-down prior vs. bottom-up prediction-error dynamics and source discrimination; it is not human hallucination.

## 2. Measured Hardware & Scaling Limitations (CPU-First Target)

On the verification machine (Snapdragon X Plus 8C/8T @ 3.30 GHz, 15.61 GB RAM, CPU-only PyTorch 2.11.0):

| Scale Target | Neurons | Verified Practical Feasibility on Laptop CPU | Scaling Notes |
|---|---:|---|---|
| **Cellular experiment** | 1–32 | Fully interactive across Fidelities 1–5 (including RK4 HH reference cable) | Sub-millisecond to few-millisecond steps. |
| **Micro-connectome** | 32–256 | Vectorized batch screening runs >10,000 candidates/sec (Fidelity 1); full multi-compartment interactive at Fidelity 1–3 | Ideal tier for GENESIS population funnels & ARCHITECT screening. |
| **Small connectome** | 256–2,048 | Practical at Fidelity 1–2 via sparse CSR / vectorized stepping (`synapse` native substrate scales to 4,000–12,000+) | Full per-neuron Python-object Fidelity-4/5 stepping becomes CPU-bound above ~64–128 cells and requires compiled C/Numba/PyTorch kernels. |
| **Medium / Large connectome** | 2,048–150,000+ | Supported in `Sonapse-LAB`'s sparse rate engine (up to 12k+ neurons) and MaleCNS aggregate CSR (11,751 cell types / 3.83M edges); full multi-compartment Fidelity-4 at 150k neurons exceeds laptop CPU budgets and requires GPU/HPC cluster acceleration | Progressive selection funnel deliberately reserves high fidelity for focal subcircuits while keeping background populations at Fidelity 1–2. |

## 3. Methodological Limitations in Current Experiments

- **Readout Capacity Scaling in Single-Cell Benchmarks**: In single-cell Foundry assays, fitting a ridge readout over all internal compartment states gives multi-compartment neurons more linear basis functions than a 1-compartment rate baseline. Raw MSE must therefore ALWAYS be interpreted alongside compartment count, FLOPs, and metabolic energy—never as standalone proof that biological complexity beats parameter-matched RNNs/GRUs.
- **BPTT vs. Local-Plasticity Budget Asymmetry**: In Sentinel's matched-baseline suite, conventional `RNN`/`GRU`/`LSTM` train all recurrent parameters end-to-end via backpropagation through time, whereas fixed/plastic connectome reservoirs adapt only local or readout weights. Both trainable parameter counts and wall-clock latency are reported explicitly so differences are transparent.
- **Negative Distillation Ground-Truth Availability**: In Supermix Antipode's 6-way study, when clean ground-truth labels exist for all samples, direct equal-budget supervised learning performs comparably to causally filtered internal distillation. Failure-based activation distillation is most valuable when ground-truth labels are sparse, expensive, or only failure flags / verifier outcomes are available.
- **Archived Checkpoint vs. Live Retained Knowledge**: Saving pre-symbiosis checkpoints guarantees disk recoverability (`archived_checkpoint_recoverable = True`), NOT live zero-rollback retention. Both metrics are intentionally reported separately.
- **External Data Connectors**: `SONAPSE CURIOSITY` provides official-API catalog metadata and a verified quarantine/anti-leakage/anti-injection pipeline; offline/air-gapped test suites validate quarantine locally without scraping remote hosts.

## 4. Open Research Questions & Proposed Follow-Up Experiments

1. **Vectorized Multi-Compartment PyTorch/Triton Kernels**: Compile `StructuredNeuronInstance` compartment graphs into batched sparse PyTorch/C++ kernels so 2,048+ heterogeneous multi-compartment Fidelity-3/4 neurons run at >500 Hz on CPU/GPU.
2. **Organelle Interference vs. Gated Coordination**: Since naïve stacking of `mnemosome + chronosome` showed measurable interference (`synergy_delta_mse < 0`), test whether a `routosome` or `plasticity_nucleus` gating layer resolves multi-organelle interference.
3. **Long-Horizon Symbiosis With Mutually Adaptive Hosts**: Extend `Sonapse Symbiosis` beyond 24-step frozen-host exposure to multi-thousand-step alternating native/symbiotic curricula with mutually adaptive `Sonapse-LAB` hosts under strict catastrophic-forgetting stress assays.
4. **Cross-Scale Transfer Laws in Architect**: Empirically characterize which micro-scale (8–16 neuron) morphological discoveries survive scale-transfer to 256–4,000 neuron connectomes, and train the 11 non-LLM research experts across 500+ logged cycles to evaluate meta-auditor calibration over time.
5. **High-Dimensional (5D–6D) Generalization**: Extend `Hyperspace` to multi-obstacle 5D/6D spatial memory and partially occluded polytope reconstruction using Compass + Chronos + Nexus hybrid connectomes.
6. **Sparse-Label Antipode Distillation on Real Expanse Checkpoints**: Run causal activation patching directly on full 132M-parameter Supermix-Expanse checkpoints (when downloaded from Hugging Face) to localize exact MoE expert routing failures on the 0/25 code-generation and 0/25 PubMedQA tasks.
