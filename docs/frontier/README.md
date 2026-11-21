# Sonapse Frontier Pack (`synapse/frontier/`)

The **Frontier Pack** is an additive, default-off collection of experimental computational neuroscience, machine
learning, and methodology modules inspired by 2024–2026 research literature and by sibling `kai9987kai` repositories.

Every module follows the contract in [`CONTRACT.md`](CONTRACT.md) and carries full verified bibliography and portfolio
provenance in [`REFERENCES.md`](REFERENCES.md):

1. **Additive and default-off.** `SynapseConfig().frontier` defaults to `None` and is omitted from `to_dict()`.
   No existing Sonapse or NEUROWEAVE default, route, DOM id, CLI flag, or checkpoint changes behaviour.
2. **Read-only live observers.** Modules marked `observer=True` can observe a deterministic unit subsample of live
   brain activity (`FrontierObserverHub` in `synapse/frontier/runtime.py`) without ever writing back into the brain,
   graft bus, actor, or shared RNGs.
3. **Capacity-matched experimental harness.** Comparison experiments run through `synapse/frontier/harness.py`,
   reporting paired differences over seeds with a two-sided 95 % t confidence interval against both an unmodified
   baseline and (whenever capacity is added) a capacity-matched control, sealed with a wall-clock-independent SHA-256
   digest (`FrontierLabReceiptV1`, `evidence_tier: 'synthetic_abstraction'`).

---

## Module Catalog (`synapse/frontier/spec.py`)

| Module ID | File | Kind | Live Observer | Engine Extension | Inspiration / Source | Documentation |
|---|---|---|---|---|---|---|
| `synchrony` | `synchrony.py` | `extension` | `SynchronyObserver` | `SynchronyExtension`, `NeuronLevelExtension` | Continuous Thought Machines (Darlow et al., NeurIPS 2025) | [`modules/synchrony.md`](modules/synchrony.md) |
| `continuum_memory` | `continuum_memory.py` | `extension` | — | `ContinuumMemoryExtension` | Nested Learning / HOPE & Titans (Behrouz et al., NeurIPS 2025) | [`modules/continuum_memory.md`](modules/continuum_memory.md) |
| `btsp_memory` | `btsp.py` | `extension` | — | `BTSPExtension` | One-shot binary-synapse BTSP (Wu & Maass, *Nat. Commun.* 2025) | [`modules/btsp_memory.md`](modules/btsp_memory.md) |
| `kuramoto_binding` | `kuramoto.py` | `extension` | `KuramotoObserver` | `KuramotoExtension` | Artificial Kuramoto Oscillatory Neurons (Miyato et al., ICLR 2025) | [`modules/kuramoto_binding.md`](modules/kuramoto_binding.md) |
| `recursive_refiner` | `recursive_refiner.py` | `model` | — | — | Tiny Recursive Model (Jolicoeur-Martineau, 2025); HRM (Wang et al., 2025) | [`modules/recursive_refiner.md`](modules/recursive_refiner.md) |
| `dendritic_layer` | `dendritic.py` | `model` | — | — | Structured dendritic ANNs (Chavlis & Poirazi, *Nat. Commun.* 2025) | [`modules/dendritic_layer.md`](modules/dendritic_layer.md) |
| `muon_optimizer` | `muon.py` | `tool` | — | — | Muon orthogonalised-momentum optimiser (Jordan et al., 2024; Liu et al., 2025) | [`modules/muon_optimizer.md`](modules/muon_optimizer.md) |
| `rigl_sparse` | `rigl.py` | `tool` | — | — | RigL dynamic sparse training (Evci et al., ICML 2020) & `sonapse-evo` | [`modules/rigl_sparse.md`](modules/rigl_sparse.md) |
| `sparse_dictionary` | `sparse_dictionary.py` | `analysis` | `SparseDictionaryObserver` | — | TopK sparse autoencoders with AuxK & causal ablation (Gao et al., ICLR 2025) | [`modules/sparse_dictionary.md`](modules/sparse_dictionary.md) |
| `anytime_inference` | `anytime.py` | `methodology` | — | — | Betting confidence sequences & e-processes (Waudby-Smith & Ramdas, 2024; Ramdas et al., 2023) | [`modules/anytime_inference.md`](modules/anytime_inference.md) |
| `quality_diversity` | `quality_diversity.py` | `discovery` | — | — | Verifier-gated MAP-Elites & DGM parent selection (`SONAPSE-ORIGIN`, `sonapse-evo`; Mouret & Clune, 2015; Zhang et al., 2025) | [`modules/quality_diversity.md`](modules/quality_diversity.md) |
| `connectomics` | `connectomics.py` | `analysis` | `ConnectivityObserver` | — | Structural, functional, dynamic-FC, and ridge-VAR(1) effective connectivity (`SONAPSE-CONNECTOME`) | [`modules/connectomics.md`](modules/connectomics.md) |
| `causal_emergence` | `causal_emergence.py` | `analysis` | `CausalEmergenceObserver` | — | Effective information & macro-vs-micro causal emergence (`GhostInTheMachine`, `SONAPSE-ORIGIN`; Hoel et al., 2013) | [`modules/causal_emergence.md`](modules/causal_emergence.md) |
| `hh_cable` | `hh_cable.py` | `reference` | — | — | Multi-compartment Hodgkin–Huxley cable reference (`NeuroPulse-CableLab`; Hodgkin & Huxley, 1952) | [`modules/hh_cable.md`](modules/hh_cable.md) |
| `nir_export` | `nir_export.py` | `tool` | — | — | Neuromorphic Intermediate Representation export (Pedersen et al., *Nat. Commun.* 2024) | [`modules/nir_export.md`](modules/nir_export.md) |
| `joint_failure` | `joint_failure.py` | `methodology` | — | — | Condition-indexed joint-failure voting panel selection (`FAULTLINE`) | [`modules/joint_failure.md`](modules/joint_failure.md) |
| `experiment_design` | `experiment_design.py` | `methodology` | — | — | Decision-sensitive Bayes-risk acquisition (`COUNTERFORM`, `Mnemorph`; Lindley, 1956; Chaloner & Verdinelli, 1995) | [`modules/experiment_design.md`](modules/experiment_design.md) |
| `morphogenesis` | `morphogenesis.py` | `discovery` | — | — | Single-organizer connectome growth & pattern-memory regeneration (`morpheus`, `prometheus-alpha`, `GenesisEngine`, `MOLT`; Mordvintsev et al., 2020; Pezzulo & Levin, 2015) | [`modules/morphogenesis.md`](modules/morphogenesis.md) |

---

## Integration Surfaces

* **Config (`synapse/config.py`):** `SynapseConfig(frontier=FrontierSpecV1.max_preset())` (default `None`, omitted from `to_dict()`).
* **Live Runtime (`synapse/frontier/runtime.py`, `synapse/frontier/bridge.py`):** `FrontierObserverHub` attaches to `IntegratedAgent.frontier_hub`, steps read-only after each `act()` call, persists under `state['frontier_v1']`, and optionally enables `MuonWithAuxAdam` for `NeuralResidualGraft` (`graft_optimizer='muon'`).
* **CLI (`synapse/frontier/cli_commands.py`):** global `--frontier`, `--no-frontier`, `--frontier-config`, `--frontier-modules`, `--frontier-graft-optimizer`, and `python -m synapse.cli frontier {catalog,experiment,observe,laboratory}`.
* **Workbench HTTP API (`synapse/frontier/workbench_api.py`):** `GET /api/frontier/{catalog,status,observers,observers/lab}` and `POST /api/frontier/{configure,step,experiment}`.
* **Max Mode (`synapse/tandem_max.py`):** `SynapseConfig.max_tandem(...)` enables read-only `FrontierSpecV1.max_preset(seed=seed)` and records additive `section_07` (`Frontier Pack Laboratory`) in the Max Mode receipt.
