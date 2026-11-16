# Pre-NEUROWEAVE baseline record

- Immutable reference: git tag `sonapse-pre-neuroweave-baseline` → commit `6f2761b`
  (`fix(max-mode): guard final checkpoint storage`) in the GitHub repository.
  Restore with `git checkout sonapse-pre-neuroweave-baseline`. The Hugging Face
  repository keeps a separate snapshot history; its equivalent of this baseline
  is commit `713d4ed` (`Sync GitHub 6f2761b with Hugging Face LFS assets`).
- Compatibility manifest: [`compatibility-manifest.json`](compatibility-manifest.json),
  verified by `python scripts/neuroweave_compat_manifest.py --check`
  (99 config defaults, 68 CLI sub-commands, 65 CLI flags, 138 HTTP routes,
  325 workbench DOM ids, 45 checkpoint namespaces, 123 test modules,
  71 Max Mode features).

## Full regression run at the baseline tag (2026-11-15, this machine)

Command: `python -m pytest -q -rf` in a clean worktree of the tag.

Result: **708 passed, 13 failed, 3 skipped** in 741 s.

The machine had ~99 MB free on `C:` and ~2.3 GB free RAM during the run.
Failure causes, from the captured tracebacks:

| Cause | Tests |
|---|---|
| RAM exhaustion (`ArrayMemoryError` / `DefaultCPUAllocator: not enough memory`) | 6 × `test_antigravity_contract_regressions`, 3 × `test_auto_programmer` |
| Disk exhaustion (checkpoint storage preflight refused) | `test_neuron_count::test_exact_total_builds_4000_neurons_and_replays`, 2 × `test_tandem_max_and_innovations` |
| **Pre-existing logic failure** | `test_planning_study::test_measured_planning_study_pairs_worlds_freezes_queries_and_checks_tampering` — the receipt now also contains the `task_value_h3` arm added by commit `5de56f5`; the test still expects only `{actor, horizon_1, horizon_3}` |

These failures exist before any NEUROWEAVE change. Post-integration
comparison uses this list: a NEUROWEAVE change is a regression only if it
fails a test that passed here, or changes the failure reason of one listed here.

## Accepted Max Mode change (recorded explicitly)

`SynapseConfig.max_tandem(...)` now also carries a NEUROWEAVE residual channel
(`NeuroWeaveSpecV1.max_preset`, not observe-only, gain 0.02, ramped), following
the DSAG precedent. Consequences for callers who pass no new options:

- the Max Mode agent trajectory differs from the baseline (the channel and its
  gate add a bounded residual current); pre-NEUROWEAVE Max checkpoints still
  load, but their continuation is not bit-identical to the baseline's
  (baseline Max continuation is itself not reproducible across processes on
  this machine, so the difference cannot be attributed cleanly);
- section 06's core lab receipt took 2.6–2.8 s at the max preset (256 neurons,
  16 steps) on one CPU thread in three runs on 2026-11-16. The quick experiment
  suite (`neuroweave_experiments`; 36.4 s in one run that day) and the quick
  lesion assay (`neuroweave_lesion`) are **opt-in** (default 0) so they do not
  add to default Max Mode wall-clock.

Default (non-Max) runs are unaffected: `SynapseConfig().neuroweave is None`.

## Post-integration full regression (2026-11-16, same machine)

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -rf`
on the working tree with every NEUROWEAVE change applied (C: had ~11 GB free).

Result: **1191 passed, 1 failed** in 1473 s.

The only failure is the pre-existing `test_planning_study` arm-set assertion
listed above (unchanged reason). The 12 environmental baseline failures all
pass once disk and RAM are available. No test that passed at the baseline fails.
