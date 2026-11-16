# Feature matrix: NEUROWEAVE-LAB 0.4.0 vs Sonapse

This compares the features of the NEUROWEAVE-LAB 0.4.0 browser prototype with
their state in Sonapse after the NEUROWEAVE evolution. The prototype's
features come from the `README.md` and `docs/ARCHITECTURE.md` of the
[NEUROWEAVE-LAB repository](https://github.com/kai9987kai/NEUROWEAVE-LAB)
(those files are not part of this repository).

Every Sonapse row points to code under `synapse/neuroweave/` unless it says
otherwise. Existing Sonapse modules named below were **not changed**: the
NEUROWEAVE versions are separate, independent implementations.

## Categories

- **Already present:** Sonapse already had a related, separate module before
  NEUROWEAVE. That module is listed for orientation only and is unchanged. In
  most of these rows a NEUROWEAVE-specific version was *also* added, so those
  rows also say "newly added".
- **Extended:** the NEUROWEAVE feature was carried over and given additional
  Sonapse-side capability.
- **Newly added:** no Sonapse counterpart existed; it is new in `synapse/neuroweave/`.
- **Adapted:** carried over with a changed form, for example from browser JS to
  NumPy, CLI or workbench.
- **Not ported:** not implemented in Sonapse.

## Neurons and connectome

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| Multi-compartment cell (4 dendrites, 6 soma channels, membrane, threshold, refractory, calcium, hillock, 2 axon branches, terminal, homeostatic gain, astrocyte reservoir, myelin) | **Extended** | `cells.py`: adds `primary`/`secondary` soma roles (DSAG-compatible), giving 8 roles. Each part has its own switch (`dendritic_compartments`, `extended_somas`, `membrane_dynamics`, `astrocyte_regulation`, …). Related pre-existing Sonapse modules: `synapse/dendritic_ltc.py`, `synapse/astrocyte.py`. |
| v0.1 baseline update and all-off identity | **Adapted** | `brain.py`; tested to `1e-12` (`test_all_off_identity_and_baseline_equivalence`) |
| 7 topologies (clustered, hub-skewed, uniform, degree-shuffled, Erdős–Rényi, hub-capped, ring lattice) | **Extended** | `topology.py`: 12 topologies (adds hierarchical_modular, multi_scale_regional, hybrid, multi_region, dense_multi_region) plus `topology_metrics` |
| Multi-region and dense multi-region connectomes (10 regions) | **Adapted** | `topology.py`, `spec.REGIONS`; region-pair gains via `multi_region_routing` (an extension) |
| Population ceiling 512 | **Extended** | spec validation allows up to 4096 (`MAX_POPULATION`), with neurogenesis caps |
| Synaptic delays, adaptive conduction | **Adapted** | `connectome.py` |
| Tiled execution | **Adapted** | `connectome.py`; numerically identical to untiled (tested) |
| Quantised INT8 synapses | **Adapted** | `connectome.py`, `hardware.py`; adds `q88_state` (an extension) |
| Dale-style sign constraint | **Already present** + **newly added** | NEUROWEAVE plasticity is sign-preserving (tested). Related pre-existing module: `synapse/dales_law.py`. |

## Plasticity and learning

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| Reward-gated plasticity | **Adapted** | `connectome.reward_update` |
| Pair-based STDP | **Already present** + **newly added** | `connectome.stdp_update`; related pre-existing module: `synapse/stdp.py` |
| Meta-plasticity | **Already present** + **newly added** | per-synapse lr scale in `connectome.py`, also in the Micro-Collective; related pre-existing modules: `synapse/meta_plasticity.py`, `synapse/metaplasticity.py` |
| Structural plasticity with event ledger | **Extended** | `connectome.structural_step` with reasoned events and caps; related pre-existing module: `synapse/structural_plasticity.py` |
| — (no neurogenesis switch in 0.4.0) | **Newly added** | `neurogenesis` mechanism with a capped ledger; related pre-existing module: `synapse/neurogenesis.py` |
| Sleep-like consolidation | **Already present** + **newly added** | `brain.py` (consolidation protects synapses from downscaling); related pre-existing module: `synapse/sleep.py` |
| Homeostatic governor | **Already present** + **newly added** | `cells.py`; related pre-existing module: `synapse/homeostasis.py` |
| Micro-Collective (4 experts, top-2, replay, held-out validation, rollback) | **Extended** | `micro_collective.py`: `add_expert()` growth, SHA-256 receipts, a persistence baseline and a parameter-matched dense-MLP comparison in the lab |
| Balanced expert routing | **Adapted** | `micro_collective.py` (tested to reduce load imbalance) |
| Predictive replay | **Adapted** | `micro_collective.py` replay plus `memory.py` replay scheduler |
| — | **Newly added** | `specialist_graft`: a zero-coupled, detachable specialist predictor |

## Memory

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| Associative trace memory | **Not ported as a runtime mechanism** | `associative_memory` is declared in the spec but no runtime code reads it. Its role is taken by `memory_ecology`. |
| Universal fast weights (8×8) | **Not ported as a runtime mechanism** in NEUROWEAVE; **already present** in Sonapse | `fast_weights` is declared but not read by the runtime. Related pre-existing module: `synapse/fast_weight_memory.py`. |
| Hierarchical temporal memory (fast/mid/slow per unit) | **Adapted** | `cells.py` |
| — | **Newly added** | `memory_ecology` (`memory.py`): provenance-tagged (sensory/predicted/external/experience/synthetic), FlyHash-indexed, interference tracking, confirm-to-promote; related pre-existing module: `synapse/multiscale_memory.py` |

## Prediction and self-model

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| Predictive world model (ensemble and disagreement) | **Extended** | `prediction.py`: per-horizon ensembles; related pre-existing modules: `synapse/world_model.py`, `synapse/world_model_v2.py` |
| — | **Newly added** | `hierarchical_prediction` (horizons 1/4/16/64 with adaptive selection) and `counterfactual_prediction` (pure, non-mutating) |
| Local predictive coding | **Already present** + **newly added** | `cells.py`; related pre-existing module: `synapse/predictive_coding.py` |
| Uncertainty-gated propagation | **Adapted** | `connectome.py` and `engine.py` |
| Standalone heuristic abstain/review outcome | **Adapted** | `standalone.selective_outcome`: score = \|2p−1\|·(1−clip(router entropy)); `ABSTAIN · REVIEW` below a threshold control (default 0.12 with `uncertainty_gating`, else 0); raw probability always shown; labelled "heuristic, not calibrated"; JS/Python parity tested in node |
| Action-grounded efference self-model | **Already present** + **newly added** | `prediction.EfferenceSelfModel`; related pre-existing module: `synapse/self_model.py` |
| Short-horizon candidate-heading planner in the agent | **Not ported** | NEUROWEAVE's ecology controller acts from engine output and noise; Sonapse's own planners are unchanged |

## Modulation and affect

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| DA/5-HT/OA/ACh/NE/eCB fields | **Already present** + **newly added** | `neuromod.py` (six modulators); related pre-existing module: `synapse/neuromodulation.py` |
| Valence/arousal/curiosity/stress/motivation | **Extended** | `neuromod.AffectState` adds novelty and reward-prediction error (7 variables). These are control signals, not emotions. |
| Heterogeneous receptor profiles | **Adapted** | `receptor_heterogeneity` |
| WireType-inspired transmitter inference | **Adapted** | `neuromod.infer_transmitters`; analysis only, never relabels signs |
| Connectivity-context attention, structural encoder, FlyHash projection, sparse code | **Adapted** | `brain.py` |

## Sensory and environment

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| 16 sensory channels (webcam visual stats, microphone bands, simulated chemical/somatic) | **Extended** | `sensory.py`: 4 modes per channel, reliability fusion, staleness, cross-modal imputation (`cross_modal_memory`); related pre-existing modules: `synapse/vision_encoder.py`, `synapse/bioacoustic_soundscape.py` |
| Webcam / microphone (permission-gated, scalars only) | **Adapted** | workbench "06 NEUROWEAVE Lab" tab plus `/api/neuroweave/sensory` (scalar-only validation). Not exercised with real devices in tests. |
| — | **Newly added** | `bridge.obs_to_sensory`: derives the channels from Sonapse's 155-d observation (3 channels missing, `audio_mid` is a speed proxy) |
| 1200×660 ecology (10 obstacles, 18 food patches, 3 thermal zones) | **Extended** | `ecology.py`: 7 configs (default, maze, scarce, shifting_rules, hidden_food, long_horizon, multi_agent) and a Sonapse 155-d adapter / `run_synapse_agent`; related pre-existing modules: `synapse/virtual_ecology.py`, `synapse/complex_world.py` |
| — | **Newly added** | typed representation packets and fusion (`packets.py`), with a `synapse/neurowave_x2` packet adapter; event bus (`events.py`) |

## Experiments

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| Topology assay (7 graphs × 4 seeds) | **Extended** | 12 topologies at matched neuron and edge budgets (`experiments.py`, CLI `neuroweave topology`) |
| Lesion assay (intact / targeted / random × 3 seeds) | **Extended** | `lesion.py`: 7 lesion kinds, 3 fractions, progressive mode, clone-only guarantee |
| Prequential metrics and session export | **Extended** | 24 experiments with ablations, frozen held-out evaluation, paired 95% CI verdicts, sealed receipts, CSV export |
| Fly-Protein-inspired sequence-variant search assay | **Ported** (NumPy RNG, so numbers differ from the browser) | `sequence_search.py`; supplementary experiment `sequence_search` (not in the 24-entry registry); CLI `neuroweave sequence-search`; `POST /api/neuroweave/sequence-search`; lab-panel button. Synthetic landscape, not protein modelling |
| — | **Newly added** | promotion ledger with atomic writes, archive and rollback (`promotion.py`) |

## Export and hardware

| NEUROWEAVE 0.4.0 feature | Sonapse status | Where / notes |
|---|---|---|
| `NEUROWEAVE_MODEL_PACKAGE` 1.0/1.1/1.2 | **Adapted** (read-only import) | `package.validate_legacy_package` / `convert_legacy_neuroweave` with a JS-compatible FNV-1a check |
| — | **Newly added** | `SONAPSE_NEUROWEAVE_PACKAGE` 1.0.0 with SHA-256 fingerprint and capability matrix; related pre-existing module: `synapse/portable_export.py` |
| Standalone HTML inference runner | **Adapted** | `standalone.py`: offline, CSP, no media access, per-mechanism executed/partial/not-executed matrix; JS ≡ Python `frozen_step` (test tolerance ≤ 1e-9; at most 6.1e-16 in one observed run) |
| Systolic-array FPGA mapping JSON | **Adapted** | `hardware.fpga_mapping_manifest`: `rtl_generated: False`, `hardware_validated: False`, independent validation |

## Sonapse integration (no NEUROWEAVE counterpart)

| Feature | Status | Where |
|---|---|---|
| `SynapseConfig.neuroweave` (default `None`) | **Newly added** | `synapse/config.py` |
| Residual graft-bus channel with gain ramp | **Newly added** | `bridge.NeuroWeaveBusChannel`, `synapse/agent.py` |
| Checkpoint namespace `neuroweave_v1` | **Newly added** | `synapse/agent.py` |
| Max Mode section 06 | **Newly added** | `synapse/tandem_max.run_neuroweave_laboratory` |
| CLI `--neuroweave`, `--neuroweave-config`, `neuroweave …` | **Newly added** | `synapse/cli.py`, `cli_commands.py` |
| Workbench `/api/neuroweave/*` and "06 NEUROWEAVE Lab" tab | **Newly added** | `synapse/server.py`, `workbench_api.py`, `synapse/web/*` |
| Additive-only compatibility manifest | **Newly added** | `scripts/neuroweave_compat_manifest.py` |
