# Frontier pack: module contract (v1)

The frontier pack (`synapse/frontier/`) adds experimental modules inspired by
2024–2026 research and by sibling `kai9987kai` projects. This contract fixes the
rules every module follows so the pack stays additive, testable and honest.

Read with: `synapse/frontier/spec.py` (module registry and `FrontierSpecV1`),
`synapse/frontier/extension.py` (engine-extension interface v1) and
`synapse/frontier/harness.py` (shared experiment harness).

## Non-negotiable rules

1. **Additive and default-off.** No existing Sonapse or NEUROWEAVE default,
   API, route, DOM id, CLI flag, checkpoint format or test changes behaviour.
   `scripts/neuroweave_compat_manifest.py --check` keeps passing.
2. **Live observers are read-only.** An observer receives a *copy* of activity
   and never writes into the brain, graft bus, actor or any shared RNG.
3. **Influence only by explicit attachment.** A module may change computation
   only as an extension attached to a NEUROWEAVE engine (lab experiments), as a
   standalone model in its own experiment, or through an explicitly selected
   opt-in option.
4. **Deterministic.** All randomness from `numpy.random.default_rng(seed + offset)`
   or a seeded `torch.Generator`; never the global RNGs. Torch code sets
   `torch.manual_seed` only inside its own local generator usage and must not
   change global torch state (use `torch.Generator` objects; restore
   `torch.get_num_threads()` if changed).
5. **Bounded and finite.** Every buffer has a fixed maximum size; every state
   value is finite; inputs are validated (`ValueError` on bad input) before any
   state changes. `import_state` validates fully before mutating.
6. **Honest evidence.** Receipts carry `evidence_tier: 'synthetic_abstraction'`.
   Module titles name the idea a module is *inspired by*; never claim to
   reproduce a paper's results, biological validity, consciousness, emotion,
   or measured hardware performance. Every citation is verified (title,
   authors, venue, month/year, URL) before it is written down.
7. **Dependencies.** NumPy, SciPy and PyTorch only (already required). Any
   other package (for example `nir`) is optional: import lazily, degrade
   gracefully, and test the fallback.

## Module file API

Each module lives in `synapse/frontier/<file>.py` and exposes:

- `MODULE_ID: str` — the id from `spec.MODULE_IDS`.
- `run_experiment(*, quick: bool = False, seeds=(0, 1, 2), **overrides) -> dict`
  returning a **sealed frontier receipt** (see below). `quick=True` must finish
  in under ~20 s on one CPU thread; the default size should finish in a few
  minutes at most.
- `EXPERIMENTS: dict[str, Callable[..., dict]]` mapping every experiment id the
  module provides to its runner (same signature as `run_experiment`). The first
  entry is `MODULE_ID: run_experiment`; extra ids start with `MODULE_ID + '_'`.
  The frontier lab registry is built from these dictionaries.
- Observer modules (`spec.OBSERVER_MODULE_IDS`) also expose an observer class:
  `__init__(self, n_units: int, *, seed: int = 7, history: int = 256, **params)`,
  `observe(self, x: np.ndarray) -> None` (1-D, length `n_units`, finite),
  `summary(self) -> dict` (JSON-safe, at most ~16 KB, no wall-clock),
  `reset()`, `export_state() -> dict`, `import_state(state) -> None`.
  Observer parameters come from `FrontierSpecV1` (`history`, `sync_pairs`,
  `sae_features`, `sae_k`, `fc_window`).
- Engine extensions subclass `extension.EngineExtension` and follow its rules
  (no engine reference kept, deep-copyable, frozen when `ctx['learn']` is false).

## Receipts and verdicts

Build receipts with `harness.base_receipt(experiment_id, kind, config)`, add
fields, and finish with `harness.seal(receipt)` (adds the SHA-256 digest; keys
containing `wallclock` are excluded from the digest).

- **Comparison experiments** (`kind` other than `validation`) set
  `verdict` in `('improved', 'no_detectable_difference', 'worse')` using
  `harness.compare` (paired 95 % t interval over seeds; `improved` only when
  the interval excludes zero) and, when the candidate adds capacity,
  `harness.headline_verdict` against a capacity-matched control. Report the
  basis in `verdict_basis`. Never declare `improved` from complexity alone.
- **Validation experiments** (`kind: 'validation'`, e.g. interchange export,
  biophysical reference, statistical coverage checks) set `verdict` in
  `('pass', 'fail')` with an explicit `checks` mapping of named booleans.
- Always include: `config` (enough to reproduce), parameter/state budgets where
  models are compared, `estimated_*` cost fields kept apart from
  `measured_wallclock_s`, and `notes` stating limits.

## Tests

One test file per implementation group, `tests/test_frontier_<group>.py`, fast
(tiny sizes; well under a minute in total per file), covering: determinism
(same seed ⇒ same digest), bounds and finiteness, validation errors, state
round-trips, the module's core behaviour on a case with a known answer, and
`run_experiment(quick=True)` producing a sealed, verifiable receipt.

## Documentation

Each module gets `docs/frontier/modules/<module_id>.md`: what it does (with the
key equations), the verified source citation(s), what is implemented and what is
not, how it connects to Sonapse, and the measured quick-mode result with its
date and settings. No claim without a test or a recorded measurement.
