# SYNAPSE Task-Aware Delayed-Reward Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve delayed-reward control with a task-aware value-guided planner, measure it against the current actor and planner on held-out native-world episodes, then expose the resulting evidence clearly in SYNAPSE.

**Architecture:** Keep the existing 16-dimensional experience model and memory as the legacy/baseline path. Add a parallel 40-dimensional task-aware learner with 14 predicted coordinates and 26 fixed context coordinates, then select it through an opt-in planner algorithm on validated native tasks. Extend the existing paired planning-study and its receipt; use that receipt to drive the final workbench and documentation changes.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, existing local HTML/CSS/JavaScript workbench. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-11-14-whole-project-advancement-design.md`

## Global Constraints

- Keep current Python, NumPy, PyTorch, and local CPU execution paths; do not add runtime dependencies or require an accelerator.
- Preserve observation-only access to task identity. No privileged evaluator data enters action selection or model updates.
- Keep backward checkpoint compatibility and the default-off policy until the held-out decision rule passes.
- The builder may choose exact seed counts after measuring CPU cost and variance, but must freeze them before confirmation and report the resulting precision.
- This is a local research result. Receipts establish recorded provenance and arithmetic, not authenticity or real-world reliability.
- Preserve the 16-value legacy experience model and surprise memory; the new task-aware learner begins cold when it is missing from a checkpoint.
- Use the task-aware path only for validated native task codes; retain the legacy path for aliased complex-task codes.
- Do not add or run automated tests in this execution. The requested frozen-query world study, receipt verification, and manual workbench review are the validation route.

## Review Focus

1. **State schema and task identity:** 40 finite values in the documented order; task one-hot matches one of the nine native task IDs; ambiguous or unsupported task identity falls back to the legacy path.
2. **Recurrent context:** graph summary and the 16 donor-state values remain fixed during each imagined rollout; only the 14 predicted coordinates receive dynamics updates.
3. **Checkpoint migration:** old state restores the 16-value learner and memory unchanged; the new learner starts cold; malformed new model state rejects without partial restore.
4. **Frozen queries and completion outcomes:** only environment-reported episode outcomes count; support updates may train the models, but query hashes remain unchanged and unfinished episodes are recorded as incomplete.
5. **Budget and receipt integrity:** study bounds reject over-budget configurations; raw records, source/config fingerprints, cost totals, and paired summaries recompute from the receipt.

---

### Task 1: Add the task-aware experience path

**Files:**
- Modify: `synapse/experience.py`
- Modify: `synapse/agent.py`
- Modify: `synapse/config.py`
- Modify: `synapse/server.py`

**Interfaces:**
- Add `encode_task_aware_experience_state(obs, features, task_id) -> np.ndarray`, returning 40 finite float64 values: `[0:14]` predicted sensory/motion/cue/time summaries, `[14:15]` graph summary, `[15:24]` native-task one-hot, `[24:40]` donor recurrent state.
- Add `task_aware_model: ExperienceWorldModel(state_dim=40, action_dim=5, ensemble_size=3, context_dim=26)` to `IntegratedAgent`; retain `experience_model` as the 16-value baseline.
- Add planner algorithm `task_value_mpc`; it uses `task_aware_model` with `ExperienceMPC(..., value_guided=True)`, with no surprise-memory input.
- Store a task-aware pending state only when the configured task is a supported base task and its observed one-hot agrees with the task name.

- [x] Create or reuse branch `codex/whole-project-advancement`; commit the approved design and plan before source changes.
- [x] Add `encode_task_aware_experience_state(obs, features, task_id)` with bounds, exact native task-code validation, and the fixed 40-value coordinate order above.
- [x] Initialize the separate 40-value model and update it from actual outcomes on supported native tasks; keep the existing 16-value model and memory path intact.
- [x] Add `task_value_mpc` config selection and route its acting path through the task-aware model, value-guided MPC, support gate, trust region, and actor fallback. Keep it off by default.
- [x] Route the pure preview endpoint through the same selected model and provide explicit fallback reasons for `auto`, complex, or mismatched task identities.
- [x] Add the task-aware model to agent export/import. If absent, leave the old model and memory untouched and initialize the new model cold; reject malformed task-aware state atomically.
- [x] Manually inspect a native-task preview, an unsupported-task fallback, and loading an existing 16-value checkpoint before starting the study.
- [x] Commit the task-aware model path after its manual review.

### Task 2: Extend the paired planning study and receipt

**Files:**
- Modify: `synapse/planning_study.py`
- Modify: `synapse/cli.py`
- Modify: `synapse/runtime.py`
- Modify: `docs/results/2026-11-14-task-aware-planning.md` (create after the study)

**Interfaces:**
- Extend study schema to `synapse-planning-study-v2` with arms `actor`, `horizon_1`, `horizon_3`, and `task_value_h3`.
- Each query record contains one episode outcome (`success`, `fail`, `timeout`, or `incomplete`), actual query steps, reward, planner-use/abstention counts, model work, frozen-state digests, and trace digest. Terminal labels come from the environment `info` returned by `world.step`.
- The study configuration bounds the total executed transitions at 32,768 and the per-episode query limit at 256 steps.

- [x] Update the learning-state digest and source fingerprints to include the task-aware model.
- [x] Add the `task_value_h3` arm and train both learner paths during shared support; export the same support state to each query arm and freeze all learned state during query.
- [x] Add the environment's terminal `info['outcome']` to the runtime's recent transition record; keep the value absent on nonterminal steps.
- [x] Change query collection to one episode per seed/task/arm, stopping at a terminal result or 256 steps; capture the environment outcome and record failures, timeouts, and incomplete episodes separately.
- [x] Add aggregate and task-specific success rates, returns, paired intervals, planner costs, and incomplete counts; keep seed/task pairs as bootstrap units.
- [x] Extend `verify_planning_study` for schema v2 to recompute arm pairing, episode outcomes, frozen-state hashes, metrics, costs, source hashes, and integrity while preserving v1 verification.
- [x] Add bounded CLI options for task-aware horizon and query-step limit; constrain the total run to 32,768 world transitions.
- [x] Run a bounded development pilot. Before viewing confirmation outcomes, freeze fresh confirmation seeds, support/query budgets, and this promotion rule: task-aware pooled completion improves at least 10 percentage points over actor with a positive paired 95% descriptive bootstrap interval; `forage` completion is no worse than 5 percentage points; the 32,768-transition run cap is respected. Then run and verify the held-out study.
- [x] Write the result report with the frozen protocol, all per-task outcomes, intervals, costs, source fingerprints, and a clear promote/keep-opt-in decision. Retain negative and inconclusive outcomes.
- [x] Commit the verified v2 study and report as one measured-evidence change.

### Task 3: Surface measured capability and align project claims

**Files:**
- Modify: `synapse/web/index.html`
- Modify: `synapse/web/app.js`
- Modify: `synapse/server.py`
- Modify: `synapse/capabilities.py`
- Modify: `README.md`
- Modify: `docs/MODEL_CARD.md`
- Modify: `docs/DATA_CARD.md`
- Modify: `docs/research/2026-11-06-sail-roadmap.md`
- Modify: `docs/design-audit.md`

**Interfaces:**
- Add a read-only endpoint for loading a named planning receipt from the configured output directory; names pass through `safe_name`, and the existing verifier determines whether a receipt is valid.
- Add a focused panel for task-aware model status, active/fallback reason, task completion, return, planner use/abstentions, and model cost. Keep forecasts visually distinct from realized outcomes.

- [x] Add the task-aware planner option and a clear unsupported-task fallback label to the existing experience controls.
- [x] Add receipt selection/loading and display the validated protocol, task-level summary, costs, and evidence scope without exposing arbitrary filesystem paths.
- [x] Manually review the flow on a supported native task, a complex task using the legacy fallback, and a valid/invalid planning receipt.
- [x] Reconcile README, model card, data card, SAIL roadmap, design audit, and capability catalog against the measured result and current runtime. Label standalone, opt-in, illustrative, unavailable, and measured behavior distinctly.
- [x] Commit the workbench and evidence-boundary updates after manual review.

### Task 4: Finish the local delivery

**Files:**
- Modify: `docs/superpowers/specs/2026-11-14-whole-project-advancement-design.md`
- Modify: `docs/superpowers/plans/2026-11-14-synapse-whole-project-advancement.md`

**Observed outcome (14 November 2026):** The frozen confirmation did not meet
the promotion rule. Task-aware minus actor completion was 0.000, return was
−0.2283, and 17/18 episodes per arm were incomplete at the query cap. The
planner remains opt-in and off by default; this capped run leaves terminal
performance mostly unresolved. The confirmation had no separate context
ablation, so it does not identify the contribution of individual context
inputs.

- [x] Set the final evidence status in the design and plan from the observed study outcome; keep the feature opt-in if the promotion rule is not met.
- [x] Review the final diff and confirm the working tree contains only this approved project scope.
- [x] Commit the approved design, implementation plan, capability work, measured receipt/report, and polish on a `codex/` branch with clear task-level commits.

## Review Gate

The capability, study, and interface are sequentially dependent and share one task-aware model/receipt contract. Use the native execution path, with one whole-change review after all three tasks rather than parallel independent implementations.
