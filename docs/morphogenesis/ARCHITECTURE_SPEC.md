# SONAPSE MORPHOGENESIS — Versioned Architecture Specification

**Schema Version:** `morphogenesis-schema-v1` (`1.0.0`)  
**Host Platform:** `Sonapse-LAB` (`synapse-connectome-lab 0.3.0`)  
**Execution Profile:** CPU-first deterministic Python/NumPy/SciPy/PyTorch with optional accelerator offload where available.

---

## 1. Unified Module Topology

`sonapse_morphogenesis` is structured as an interoperable package suite inside the unified `Sonapse-LAB` repository:

```text
sonapse_morphogenesis/
├── core/         # Common Interchange Format v1, 5-fidelity compartment graph engine,
│                 # SQLite/JSON experiment registry, safe checkpoints, Sonapse-LAB bridge, profiler
├── foundry/      # SONAPSE FOUNDRY: 21+3 custom graft neurons, 15 conventional neurons,
│                 # 14 synthetic organelles, 12 bio-system hybrids, 13 ML hybrids (3-variant), benchmarks
├── genesis/      # SONAPSE GENESIS: developmental genomes, reversible mutations, 1000+ micro-candidate
│                 # funnel, Successive Collective Experience Evolution, recursive connectome improvement
├── chimera/      # SONAPSE CHIMERA: 13 connectome fusion/grafting modes, Hungarian/Procrustes alignment
├── spectra/      # SONAPSE SPECTRA: 15 bio + 13 artificial senses, multimodal signal mixer & injector,
│                 # internal perception / hallucination-like source-discrimination simulator
├── hyperspace/   # SONAPSE HYPERSPACE: native N-dimensional (4D+) developmental environments & SO(4) tasks
├── affect/       # SONAPSE AFFECT: expanded 13-system physiology, multi-timescale neuromodulators,
│                 # dynamical attractor & causal evaluator, artifact rejection, ethical stop guards
├── antipode/     # SUPERMIX ANTIPODE: real Tidal MoE + Expanse failure instrumentation, 10 negative
│                 # training objectives, causal activation patching, 6-way benchmark, external AI advisor
├── symbiosis/    # SONAPSE SYMBIOSIS: 11 cross-connectome coupling modes to Sonapse-LAB, 11 retention
│                 # preservation techniques, archived-vs-live retention audit
├── curiosity/    # SONAPSE CURIOSITY: 12 novel interactive training modalities (incl. restricted Python VM)
│                 # + controlled public data discovery & quarantine pipeline
├── architect/    # SONAPSE ARCHITECT: independent 14-step autonomous research director, 19 selectable
│                 # learning algorithms, training-pack generator, 11 non-LLM experts + Meta-Auditor
├── sentinel/     # SONAPSE SENTINEL: immutable held-out evaluation suites, matched baselines, FDR stats,
│                 # 5-stage discovery state machine, automated reports, safety & governance guards
└── workbench/    # Interactive scientific workbench API (/api/morphogenesis/*) & decoupled browser UI
```

---

## 2. Common Interchange Format (`morphogenesis-v1`)

All modules communicate through versioned, JSON-serializable, SHA-256 content-digested dataclasses defined in `sonapse_morphogenesis.core.interchange`:

1. **`NeuronMorphologySpec`**: Internal compartment graph $(V_c, E_c)$ specifying soma(s), dendritic trees, axonal initiation zones, axonal segments, feedback loops, local inhibitory nodes, memory nodes, physical dimensions (`length_um`, `radius_um`, `myelin_factor`, `coords_3d`), electrical parameters (`cm_uF_cm2`, `ra_ohm_cm`, `g_leak`, `e_rest_mV`), receptor distributions, attached `OrganelleAttachment` list, and `FidelityLevel` (1–5).
2. **`DevelopmentalGenome`**: Versioned genome (`genome_version`, `genome_id`, `parent_ids`, `neurons`, `synapses`, `growth_rules`, `pruning_rules`, `differentiation_rules`, `neuromodulation_params`, `metabolic_budget`, `external_interfaces`, `mutation_history`, `provenance`).
3. **`ConnectomeGraph`**: Typed sparse graph (nodes with morphology/fidelity/coordinates, signed Dale-tagged edges with weights, physical lengths, conduction delays, receptor types, plasticity rule tags) convertible to/from `synapse.neuroweave.topology.Topology` and `graftbrain` sparse structures.
4. **`ActivationStream`**: Timestamped multi-channel neural recording (`timestamps`, `node_ids`, `compartment_values`, `spikes`, `chemical_fields`, `source_kind` distinguishing `biological_recording`, `artificial_model_activity`, and `synthetic_simulation`).
5. **`SensoryExperience`** & **`TrainingExample`**: Provenance-tagged sensory/motor/reward/correction tuples with explicit `source_kind`, license, hash, and quarantine status.
6. **`ExperimentReceipt`**: Immutable record binding `experiment_id`, `objective_category`, `evidence_state`, `code_commit`, `seed_list`, `config_digest`, `parent_checkpoint_id`, `metrics`, `baseline_comparisons`, `uncertainty_intervals`, `resource_usage`, `failed_attempts`, and `sha256_digest`.

---

## 3. Five-Level Multi-Fidelity Neuron Simulation Engine

| Fidelity | Name | Governing Equations | Approximation Label |
|---:|---|---|---|
| **1** | `ABSTRACT_RATE` | $\tau_i \dot{r}_i = -r_i + \phi(I_{\text{syn},i} + I_{\text{ext},i})$ | Continuous abstract rate (no spikes, no ionic conductances). |
| **2** | `MULTI_COMPARTMENT_RECURRENT` | Coupled soma/dendrite/axon graph with axial coupling $g_{ab} \propto \frac{\pi r^2}{\rho_a L}$, capacitance $\tau_c \propto C_m A_c$, delay rings $\tau_{\text{axon}} = L / v(\text{myelin}, r)$. | Synthetic multi-compartment continuous dynamics with physical geometry coupling. |
| **3** | `SPIKING` | Multi-compartment LIF / adaptive Izhikevich threshold-reset dynamics with refractory periods & synaptic conductance decays. | Phenomenological spiking neuron (subthreshold cable + spike threshold). |
| **4** | `CONDUCTANCE` | Compartment cable equations with active Hodgkin–Huxley $I_{\text{Na}}, I_{\text{K}}, I_L$ currents (reusing `synapse.frontier.hh_cable` rate functions) + AMPA/NMDA($\text{Mg}^{2+}$)/GABA conductances, Rush–Larsen exponential Euler stability. | Conductance-based biophysical approximation (uniform compartment parameters). |
| **5** | `REFERENCE_CABLE` | Sub-discretized multi-segment RK4 Hodgkin–Huxley reference cable with stability bound enforcement ($\Delta t \cdot \lambda_{\max} < 2.5$). | High-fidelity numerical reference for calibrating lower-fidelity models. |

Mixed-fidelity connectomes allow e.g. Fidelity-1/2 background populations to coexist with Fidelity-3/4 focal custom graft circuits via `TranslatorMembrane` / rate-to-current and spike-to-rate interfaces.

---

## 4. Separation of Authority & Safety Governance

SONAPSE SENTINEL enforces hard architectural boundaries:
- **Developing connectome** cannot edit evaluation suites, registry logs, or resource budgets.
- **SONAPSE ARCHITECT** runs on isolated clones and bounded budgets (`max_wall_seconds`, `max_memory_mb`, `max_candidates`); it cannot modify held-out evaluators, alter objective definitions, hide failed runs, or execute arbitrary OS commands.
- **SONAPSE CURIOSITY** quarantines all ingested external data and treats all external content strictly as untrusted passive data—never as executable prompts or code.
- **SONAPSE AFFECT** enforces ethical stop guards prohibiting optimization of sustained aversive/distress proxies.
- **Human Operator** holds exclusive emergency-stop, promotion-approval, and budget-configuration authority.
