# NEUROWEAVE changelog

## [Unreleased] — NEUROWEAVE evolution v1 (2026-11-15)

Baseline: git tag `sonapse-pre-neuroweave-baseline` (commit `6f2761b`). All
changes are additive and opt-in: `SynapseConfig().neuroweave is None`, and
`scripts/neuroweave_compat_manifest.py --check` passes.

### Added

**Subsystems.** New package `synapse/neuroweave/`:
- `spec`: `NeuroWeaveSpecV1`, with 45 mechanisms, 24 experiments, 16 sensory
  channels, 12 topologies and 10 regions.
- `topology`, `cells`, `connectome`, `brain`, `neuromod`, `sensory`,
  `micro_collective`, `memory`, `prediction`, `packets`, `events`, `ecology`,
  `hardware`, `package`.
- `engine`: `NeuroWeaveEngine` composes the subsystems.
- `bridge`: `NeuroWeaveBusChannel` (residual graft-bus current; zero when
  disabled or `observe_only`; gain ramps up from 0) and `obs_to_sensory`.

**Laboratory:**
- `tasks`, `experiments`, `lesion`, `promotion`: 24 seed-replicated, ablated
  experiments with frozen held-out evaluation, paired 95% CI verdicts and
  SHA-256-sealed receipts.
- A lesion assay that only ever touches clones.
- An atomic promotion ledger.

**Export:**
- `SONAPSE_NEUROWEAVE_PACKAGE` 1.0.0 with a capability matrix.
- Read-only legacy `NEUROWEAVE_MODEL_PACKAGE` 1.0.0/1.1.0/1.2.0 import.
- FPGA *mapping* manifest (no RTL).
- Offline standalone HTML runner whose JS frozen inference matches Python
  `frozen_step`: the parity test enforces ≤ 1e-9; one run on 2026-11-16
  (node v24.13.1) observed at most 6.1e-16.

**Interfaces:**
- CLI: global `--neuroweave` / `--no-neuroweave` / `--neuroweave-config`, and
  the `neuroweave` command group (`mechanisms`, `experiments`, `validate`,
  `smoke`, `topology`, `experiment`, `lesion`, `ecology`, `export`, `inspect`,
  `import-legacy`, `manifest-check`).
- Workbench: `/api/neuroweave/*` routes and the "06 NEUROWEAVE Lab" tab.
  Webcam and microphone are reduced to scalars in the browser; the server
  rejects raw media.

**Sonapse wiring:**
- `SynapseConfig.neuroweave` (default `None`, left out of `to_dict()`).
- `neuroweave_v1` audit entries.
- `IntegratedAgent.neuroweave_channel`.
- Checkpoint namespace `state['neuroweave_v1']`.
- Max Mode preset: `NeuroWeaveSpecV1.max_preset`, bounded to ≤ 256 neurons.
- `receipt['section_06']` from `run_neuroweave_laboratory`, with a new
  `neuroweave_steps` option.

**Tooling and docs:**
- `scripts/neuroweave_compat_manifest.py` and
  `docs/neuroweave/compatibility-manifest.json`.
- Docs in `docs/neuroweave/`: README, FEATURE_MATRIX, INTEGRATION_CONTRACT,
  BASELINE, CHANGELOG.

**Tests.** 14 new files `tests/test_neuroweave_*.py` with 468 tests in total,
including the follow-up `test_neuroweave_sequence_search.py` and
`test_neuroweave_review_fixes.py` (regression tests for the 13 issues found in
the adversarial review: memory provenance, webcam/microphone stream release,
bounded inspect reports, lab cost bounds and checkpoint scalar validation).

**Follow-up additions (same release):**
- Max Mode section 06 now also runs, under new bounded options,
  `neuroweave_experiments` (0/1, default 0 / opt-in: all 24 experiments in quick mode,
  per-experiment verdicts and receipt digests), `neuroweave_lesion` (0/1,
  default 0 / opt-in: quick lesion assay) and `neuroweave_agent_steps` (0–64, default 4:
  a deep copy of the live Max Mode agent driven in a `NeuroWeaveEcology`,
  live channel digest checked unchanged), plus the live channel telemetry and
  observatory summary. `sections_completed` still has five entries.
- `sequence_search.py`: budget-matched synthetic sequence-variant search
  (sparse random-projection guidance vs evolutionary vs random), exposed as the
  supplementary experiment `sequence_search` (`experiments.SUPPLEMENTARY_EXPERIMENTS`;
  `spec.EXPERIMENTS` still has 24 entries), CLI `neuroweave sequence-search`,
  `POST /api/neuroweave/sequence-search` and a lab-panel button.
- Standalone runner: heuristic abstain/review outcome (`selective_outcome`)
  with a threshold control; raw probability always shown; JS/Python parity
  extended to the selective score.
- New test file `tests/test_neuroweave_sequence_search.py`; additions to
  `tests/test_neuroweave_integration.py` and `tests/test_neuroweave_standalone.py`.

### Changed

Existing files, additive edits only. No existing default, flag, route, DOM id,
checkpoint namespace or test was removed or changed in behaviour.
- `synapse/config.py`
- `synapse/agent.py`
- `synapse/tandem_max.py`
- `synapse/cli.py`
- `synapse/server.py`
- `synapse/web/index.html`, `app.js`, `style.css`
- root `README.md`: section appended

### Known limitations

- `associative_memory` and `fast_weights` are declared in the spec but not
  consumed by the runtime.
- Every verdict so far for the 24 registered experiments (quick mode only) is
  `no_detectable_difference`; the full-size experiments have not been run.
  The supplementary sequence-variant search reports `worse` overall and
  `improved` only for sparse vs random in one synthetic setting.
- Not tested:
  - full Max Mode workbench runs
  - real webcam or microphone capture
  - the standalone page in a real browser
