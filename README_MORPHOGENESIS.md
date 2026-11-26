# SONAPSE MORPHOGENESIS
## Autonomous Developmental Connectome Intelligence, Synthetic Neuroscience & Neural Architecture Discovery

**Version:** `2.0.0` (`morphogenesis-schema-v1`)  
**Base Platform:** [`Sonapse-LAB`](README.md) (`synapse-connectome-lab 0.3.0`, branch `main`)  
**Research Paper:** [`docs/paper/SONAPSE_MORPHOGENESIS_V2_PAPER.md`](docs/paper/SONAPSE_MORPHOGENESIS_V2_PAPER.md)  
**Runtime Profile:** CPU-first deterministic Python 3.12 / NumPy / SciPy / PyTorch (Windows ARM64 verified).

> **Scientific Epistemic Stance:** SONAPSE MORPHOGENESIS explicitly separates **Biological Realism**, **Computational Usefulness**, and **Speculative Synthetic Biology** across every schema, experiment, and discovery report. Complexity is never assumed to equal intelligence, biological inspiration never establishes biological validity, behavioral imitation never proves subjective experience, and all claims are grounded in measured, reproducible multi-seed runs.

---

## 1. Platform Architecture (11 Interoperable Research Modules + Shared Core)

| Module | Package | Core Capability |
|---|---|---|
| **CORE** | `sonapse_morphogenesis.core` | Common Interchange Format v1 (`morphogenesis-schema-v1`), 5-level multi-fidelity physically structured compartment engine (Rate → Multi-Compartment → Spiking → HH Conductance → RK4 Cable), mixed-fidelity connectomes, real MaleCNS v1.0 induced subgraph loader, SQLite + safe `synapse.stateio` Experiment Registry, CPU profiler & zero-gain `Sonapse-LAB` graft bus bridge. |
| **SONAPSE FOUNDRY** | `sonapse_morphogenesis.foundry` | **24 custom synthetic graft neurons** (all 21 specified + 3 original: `SymbioteEcho`, `BifurcationGate`, `HolographicDendrite`), **15 improved conventional neurons**, **14 modular synthetic organelles** with pairwise interaction assays, **12 non-neural bio-system abstractions**, **13 ML/software hybrids** with 3-variant comparison (`exact_software` vs. `cellular_compartment` vs. `biologically_constrained`), SVG morphology renderer & cellular benchmarks. |
| **SONAPSE GENESIS** | `sonapse_morphogenesis.genesis` | Versioned developmental genomes, **15 reversible transactional mutation operators**, **7 candidate origins** (random, real MaleCNS, Maslov–Sneppen degree-shuffled null, fractal, mixed custom/standard, Sonapse prior, auto genome), **1,024+ candidate vectorized CPU screening funnel** (>6,000 cand/s), Successive Collective Experience Evolution (5 curricula + accidental-success rejection), and bounded recursive improvement. |
| **SUPERMIX ANTIPODE** | `sonapse_morphogenesis.antipode` | Real PyTorch forward-hook instrumentation of **`artifacts/tidal_core.pt`** (22 layers), real ingestion of **Supermix-Expanse `compare_v1_v2*.json` failures** (164 verified failures), causal activation-patching localization, trainable cross-architecture adapters, **10 negative neural training objectives**, 6-way comparative benchmark, and optional local/bounded AI-assisted training coordinator. |
| **SONAPSE SYMBIOSIS** | `sonapse_morphogenesis.symbiosis` | **11 intermittent cross-connectome coupling modes** connecting developing connectomes to a live frozen/adaptive **`Sonapse-LAB` `LabRuntime`**, plus **11 knowledge preservation mechanisms** (EWC anchors, replay, fast/slow compartments, reversible pruning, snapshots) explicitly separating archived recoverability from live behavioral retention. |
| **SONAPSE SPECTRA** | `sonapse_morphogenesis.spectra` | **15 biological + 13 artificial non-biological senses** (Topologyception, Codeception, Mathception, Probabiliception, SymmetryPerception, etc. with verified zero answer leakage), multimodal encoder (6 neural encodings), **13 signal mixing operators**, **11 anatomical injection sites** with replayable receipts, and **Internal Perception / Hallucination-Like Source-Discrimination Simulator**. |
| **SONAPSE HYPERSPACE** | `sonapse_morphogenesis.hyperspace` | General **$N$-dimensional (4D default, extensible to 5D/6D)** Euclidean developmental environment with direct $N$-D sensing/actuation, hyperwall 4D shortcut discovery, exact $\text{SO}(4)$ rotation-invariant polytope recognition, 3D→4D transfer vs. native 4D benchmarks, and 3D slice/perspective projections. |
| **SONAPSE CHIMERA** | `sonapse_morphogenesis.chimera` | **13 connectome mixing & fusion modes**, multi-cue Hungarian + orthogonal Procrustes functional/anatomical alignment, full provenance tracking, exact transactional parent rollback, and 5-way comparative evaluation. |
| **SONAPSE AFFECT** | `sonapse_morphogenesis.affect` | 4-tier epistemic separation, **13-subsystem simulated body**, 6 multi-timescale neuromodulatory fields, unlabeled PCA/attractor/persistence/causal `do()` evaluator, reward-leakage & artifact rejection suite, and **hard ethical stop guardrails** prohibiting optimization for distress/aversive states. |
| **SONAPSE CURIOSITY** | `sonapse_morphogenesis.curiosity` | **12 novel interactive training modalities** (including deterministic safe `RestrictedPythonVM` for Codeception, RLC circuits, spring-mass mechanics, Rule-110 CA, packet routing, propositional proof verifier) + controlled public data catalog & 6-gate Quarantine Pipeline (license, prompt-injection block, dedup, held-out leakage check). |
| **SONAPSE ARCHITECT** | `sonapse_morphogenesis.architect` | Independent 14-step autonomous research director on isolated clones, **19 experimentally selectable learning algorithms**, autonomous training-pack generator, **11 lightweight non-LLM learned research experts** + independent **Expert Meta-Auditor**, and micro-to-large scale-transfer verification. |
| **SONAPSE SENTINEL** | `sonapse_morphogenesis.sentinel` | Immutable held-out evaluator, **7 matched-budget baselines** (RNN, GRU, LSTM, Reservoir, LIF, Random, Degree-Shuffled Biological), paired bootstrap CIs, Benjamini–Hochberg FDR correction, **5-stage Discovery Evidence State Machine**, automated Markdown+JSON+SVG Discovery Reports, and **Governance Controller / Emergency Stop**. |
| **EMERGENCE** | `sonapse_morphogenesis.emergence` | Cross-yoked twin authorship assay, Slow-Mode Analysis (`MAF-1`), Gaussian MMI-PID ($\Phi_R$ & synergy), dynamical closure, and closed-loop efference spectrum. |
| **MAX MODE** | `sonapse_morphogenesis.max_mode` | **OMNI-MORPHOGENESIS 3-tier nested oscillatory coordinator** (Gamma/Theta/Circadian), 6-lane scheduler, 15-check completeness audit, atomic checkpoints, and continuous campaign runner. |
| **WORKBENCH UI** | `sonapse_morphogenesis.workbench` | Interactive browser scientific workbench strictly decoupled from simulation, mounted seamlessly as **Tab 08 (`/morphogenesis` & `/api/morphogenesis/*`)** inside `Sonapse-LAB` AND runnable standalone. |

---

## 2. Quickstart & Reproducibility

### Launch Interactive Scientific Workbench
```powershell
# Option A: Integrated inside Sonapse-LAB (adds Tab 08 + /morphogenesis)
python -m synapse serve --port 8765

# Option B: Standalone Morphogenesis Workbench
python -m sonapse_morphogenesis serve --port 8766
```
Open `http://127.0.0.1:8766/` (standalone) or `http://127.0.0.1:8765/morphogenesis` (integrated).

### Run Individual Research Modules via CLI
```powershell
python -m sonapse_morphogenesis status
python -m sonapse_morphogenesis foundry --seed 7
python -m sonapse_morphogenesis genesis --pop 1024 --seed 7
python -m sonapse_morphogenesis antipode --seed 7
python -m sonapse_morphogenesis symbiosis --seed 7
python -m sonapse_morphogenesis spectra --seed 7
python -m sonapse_morphogenesis hyperspace --seed 7
python -m sonapse_morphogenesis chimera --seed 7
python -m sonapse_morphogenesis affect --seed 7
python -m sonapse_morphogenesis curiosity --seed 7
python -m sonapse_morphogenesis architect --seed 7
python -m sonapse_morphogenesis sentinel
python -m sonapse_morphogenesis max-mode --audit-only
python -u -m sonapse_morphogenesis --seed 7 max-mode --duration-seconds 60 --host-neurons 192 --genesis-pop 256
```

### Reproduce the v2.0 Timed Section Campaigns
```powershell
python -u scripts/run_individual_modes_and_sections.py
```

### Reproduce the Complete End-to-End Experimental Suite (~74s on CPU)
```powershell
python scripts/run_morphogenesis_experiments.py
```

### Run Automated Tests & Backward-Compatibility Regressions
```powershell
python -m pytest tests/morphogenesis/test_sonapse_morphogenesis.py tests/test_dsag.py tests/test_antigravity_contract_regressions.py tests/test_agent.py tests/test_server.py tests/test_server_lock_isolation.py tests/test_experience_workbench.py tests/test_cli.py -v
```

---

## 3. Documentation, Schemas & Archived Artifacts

- **Repository Audit & Capability Inventory:** [`docs/morphogenesis/REPOSITORY_AUDIT.md`](docs/morphogenesis/REPOSITORY_AUDIT.md)
- **Versioned Architecture Specification:** [`docs/morphogenesis/ARCHITECTURE_SPEC.md`](docs/morphogenesis/ARCHITECTURE_SPEC.md)
- **Compatibility & Migration Guide:** [`docs/morphogenesis/COMPATIBILITY_AND_MIGRATION.md`](docs/morphogenesis/COMPATIBILITY_AND_MIGRATION.md)
- **Neuron & Synthetic Organelle Catalogue:** [`docs/morphogenesis/NEURON_ORGANELLE_CATALOG.md`](docs/morphogenesis/NEURON_ORGANELLE_CATALOG.md)
- **Actual Measured Scientific Findings:** [`docs/morphogenesis/SCIENTIFIC_FINDINGS.md`](docs/morphogenesis/SCIENTIFIC_FINDINGS.md)
- **Known Limitations & Open Questions:** [`docs/morphogenesis/LIMITATIONS_AND_OPEN_QUESTIONS.md`](docs/morphogenesis/LIMITATIONS_AND_OPEN_QUESTIONS.md)
- **Machine-Readable JSON Schemas:** [`schemas/morphogenesis_v1/`](schemas/morphogenesis_v1/)
- **Example Genomes & Connectomes:** [`examples/morphogenesis/`](examples/morphogenesis/)
- **Experiment Config:** [`configs/morphogenesis/default_research_suite.json`](configs/morphogenesis/default_research_suite.json)
- **Measured Results, SQLite Registry & Discovery Reports:** [`results/morphogenesis/`](results/morphogenesis/)
- **SVG Morphology Figures:** [`docs/morphogenesis/figures/`](docs/morphogenesis/figures/)
