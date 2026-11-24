# SONAPSE MORPHOGENESIS — Repository Audit & Capability Inventory

**Audit Date:** 2026-11-24  
**Base Repository Commit:** `Sonapse-LAB` (`0c48d67`, branched to `feat/sonapse-morphogenesis`)  
**Host Verification Hardware:** Snapdragon X Plus (8C/8T @ 3.30 GHz), 15.61 GB RAM, Windows ARM64, CPU-first PyTorch 2.11.0+cpu, NumPy 2.4.3, SciPy 1.17.1, scikit-learn 1.8.0, NetworkX 3.6.1.  

---

## 1. Epistemic Taxonomy & Research Objective Boundaries

Every subsystem and experiment in SONAPSE MORPHOGENESIS explicitly classifies its work into three distinct objectives so that metaphors never masquerade as evidence:

| Objective Category | Scope | Prohibited Claims |
|---|---|---|
| **`biological_realism`** | Biophysical cable equations (Hodgkin–Huxley ion channels, axial resistance, capacitance, myelinated delay, Dale's law, empirical Drosophila/C. elegans connectivity matrices). | Synthetic rate units or phenomenological conductances are never claimed to be validated clinical or in-vivo reconstructions. |
| **`computational_usefulness`** | Measurable task performance, sample efficiency, generalization, robustness, adaptation speed, memory retention, and FLOP/wall-clock efficiency on held-out benchmarks against matched baselines. | Complexity or biological inspiration never counts as evidence of superiority without multi-seed matched-budget evaluation. |
| **`speculative_synthetic_biology`** | Novel multi-soma neurons (Triarch, Ouroboros, Hydra, etc.), synthetic organelles (Mnemosome, Predicton, etc.), non-neural organ abstractions, artificial senses, and emergent affect-like organization. | Strictly software computational models; zero autonomous genetic engineering, living-tissue experimentation, or claims of subjective consciousness/sentience. |

---

## 2. Verified Audit of `kai9987kai` Ecosystem Repositories

All 25 inspected repositories under `https://github.com/kai9987kai` were verified via authenticated GitHub API inspection on 2026-11-24. None were assumed from memory.

| Repository | Verified Status | Scale | Primary Reusable Mechanisms Identified | Integration Strategy in Morphogenesis |
|---|---|---|---|---|
| **Sonapse-LAB** | Active Python repo (`0c48d67`, 1,436 files, 946 `.py`, 156 tests) | 70 MB | `IntegratedAgent`, `LabRuntime`, `SynapseConfig` (additive `None`-default schema), `stateio.py` (content-bound SHA-256 safe checkpoints), `graft_bus.py` (`NeuralGraftProtocol` zero-gain contract), `dsag/` (Dual-Soma Axodendritic Graft), `neuroweave/` (multi-compartment cells + topology), `neuroendocrine/` (WPAC + synthetic glands), `frontier/` (`hh_cable.py`, `morphogenesis.py` 2D NCA, `workbench_api.py`), `world4d.py`, `multi_connectome_fusion.py`, `artifacts/tidal_core.pt` (1.3 MB Poseidon Tidal MoE), `_vendor/expanse/` (MaleCNS `malecns_types.npz` 8.9 MB + real Expanse v1/v2/v3 comparison receipts). | **Direct base repository.** Preserved 100% intact; Morphogenesis added via `sonapse_morphogenesis/` package + additive default-off hooks (`SynapseConfig.morphogenesis = None`, `/api/morphogenesis/*`, Tab 08). |
| **Two-Soma-Neuron-Lab** | Active HTML/JS repo (5 files) | 121 KB | Seeded DSAG Experiment 01 reservoir equations (primary/memory decay $\rho_p=0.62, \rho_m=0.96$, gated axonal feedback), delayed recall & context-gated parity tasks, strict capacity/readout controls. | Reused as controlled cellular benchmark task family & baseline comparator in **FOUNDRY**. |
| **NEUROGRAFT** | Active Python/HTML repo (597 files, 80 `.pt` checkpoints) | 101 MB | Typed sparse provenance-preserving neural graphs, trainable routing layers, OpenWorm normalized fixtures, replay-learning & recency-revision controlled assays. | Provenance-preserving graph routing & matched replay/lesion methodology incorporated into **CHIMERA** and **SYMBIOSIS**. |
| **NEUROWEAVE-LAB** | Active HTML/JS spec & workbench (17 files, schema v1.2.0) | 265 KB | 4 dendritic compartments (`proximal`, `distal`, `apical`, `basal`), 6 soma channels (`fast`, `slow`, `context`, `integrator`, `burst`, `gating`), 31 toggleable mechanisms, FNV-1a/JSON portable export format, Micro-Collective 4-expert predictor. | Already ported to `synapse/neuroweave/cells.py`; generalized into arbitrary compartment graphs in `sonapse_morphogenesis.core.compartment_engine`. |
| **NEUROWEAVE-ANALOG** | Spec/pre-prototype repo (23 files) | 50 KB | Physical RC/op-amp conductance dynamics ($C_i dV_i/dt = \sum I_{ij} - (V_i - V_{rest})/R_i$), explicit 6-channel soma isolation & test-point acceptance criteria. | Physical capacitance/axial-resistance coupling & observability probe points implemented in **FOUNDRY** Fidelity 4. |
| **SONAPSE-FRACTAL** | Active single-file HTML sim (5 files) | 61 KB | Finite fractal-inspired graph topologies (Sierpiński, Cantor, Vicsek, Hilbert-like) vs small-world/random/hierarchical controls. | Fractal topology generators incorporated into **GENESIS** candidate seeding & **CURIOSITY** fractal modality. |
| **Supermix-Poseidon** | Active Python repo (519 files, 377 `.py`) | 226 MB | Frozen Tidal MoE core (`tidal_core.pt`, already vendored in `Sonapse-LAB/artifacts/`), Mnemosyne 3-tier memory, Horizon-X reserve guard, matched lesion protocols. | Real `artifacts/tidal_core.pt` instrumented directly in **ANTIPODE** and **SYMBIOSIS**. |
| **Supermix-expanse** & **Supermix-Expanse-v2** | Active Python repos (142 / 372 files, `malecns_types.npz`) | 35–41 MB | 132.6M-param multi-source graft (Archimedes + Omni v7 + Qwen2.5-Coder + BioMedLM + MaleCNS 11,751 cell types / 3.83M edges), zero-init function-preserving graft gates, **documented empirical failures** (`code pass_rate=0.00`, `PubMedQA=0.00`, `exact_accuracy=0.20–0.28` in `compare_v1_v2.json`). | Real `malecns_types.npz` used in **GENESIS** biological seeding; real Expanse v1/v2 failure records ingested directly in **ANTIPODE**! |
| **Supermix-Tether** | Active Python repo (309 files, 248 `.pt` checkpoints) | 104 MB | Dark-knowledge specialist reliability (tail entropy), counterfactual co-variation, perturbation-localized invariance, epistemic trust gating, split conformal calibration. | Counterfactual activation patching & reliability-gated specialization adapted into **ANTIPODE** and **SENTINEL**. |
| **SONAPSE-CONNECTOME** | Active Python/TS repo (236 files, 149 `.py`) | 867 KB | Separated structural vs functional connectivity, tract-length conduction delays $\tau_{ij} = l_{ij}/v_{ij}$, AMPA/NMDA/GABA conductance synapses, separated neuromodulator ODEs vs `ComputationalAffectState`. | Strict separation of structural/functional graphs & neuromodulator ODEs from affect proxies enforced in **CORE** and **AFFECT**. |
| **sonapse-evo** | Active Python repo (231 files, 156 `.py`) | 1.9 MB | 20 bounded structural variation operators, 8-axis Pareto optimization, 6-stage progressive multi-fidelity funnel, typed evidence taxonomy, Lyapunov criticality tracking. | Progressive selection funnel, reversible structural transactions, and typed evidence states adapted in **GENESIS** and **SENTINEL**. |
| **SONAPSE-ORIGIN** | Active Python repo (119 files, 111 `.py`) | 378 KB | 5-tier epistemic boundary framework, MAP-Elites / NSGA-II / Novelty Search, adversarial critics, SQLite evidence ledger. | MAP-Elites diversity preservation & adversarial audit concepts adapted in **GENESIS**, **ARCHITECT**, and **SENTINEL**. |
| **ODDITY-LAB** | Active HTML/JS sandbox (9 files) | 5.0 MB | Exact matched baseline-vs-grafted reservoir ablation protocol ("Ablate all -> identical outputs"), 6 temporal Boolean tasks (2-step/6-step echo, temporal XOR, 3-bit parity, 5-step majority, hybrid rule). | Exact zero-ablation identity check & temporal Boolean suite incorporated into **FOUNDRY** benchmarks. |
| **Supermix-Aether** | Active Python repo (103 files) | 6.8 MB | Analytical step FLOP counters, compute parity profiler, communication stress tasks. | Analytical FLOP/latency resource profiler adapted in `core/profiler.py`. |
| **Supermix / supermix-archimedes / Graft-brain / OMNIS-COGNITIVE / NeuroPulse-CableLab / morpheus / GenesisEngine / Evo-OMEGA / SYNAPSE-ATLAS / FAULTLINE** | All verified active | Combined >300 MB | `NeuroPulse-CableLab` HH equations already verified in `synapse/frontier/hh_cable.py`; `morpheus`/`GenesisEngine` developmental regeneration in `synapse/frontier/morphogenesis.py`; `FAULTLINE` correlated-failure analysis in `synapse/frontier/joint_failure.py`. | Reused directly from `synapse/frontier/` without duplication. |

---

## 3. Baseline Verification Before Modification

Prior to adding any Morphogenesis code, the following baseline regression suite was executed on `Sonapse-LAB` (`0c48d67`) on this host:

```powershell
python -m pytest tests/test_dsag.py tests/test_antigravity_contract_regressions.py tests/test_agent.py tests/test_9000_neurons_and_backward_compatibility.py -q
```

**Measured Baseline Outcome:** `37 passed in 135.30s` (0 failed, 0 skipped).  
All original files (`artifacts/tidal_core.pt`, `synapse/_vendor/expanse/expanse/data/malecns_types.npz`, `synapse/web/*`, all 156 existing tests) remain untouched and protected.
