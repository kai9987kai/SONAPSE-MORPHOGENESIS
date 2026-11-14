# SYNAPSE whole-project advancement

**Status:** Approved for implementation  
**Date:** 14 November 2026

## Intent

Advance SYNAPSE across capability, evidence quality, usability, and maintainability.
The work is staged capability first, then measurement, then workbench and project
polish. The first capability is improved control on delayed-reward tasks, chosen
by the builder after the user delegated the technical choice.

SYNAPSE already learns an online transition model and TD(0) value estimate from
executed outcomes, with a bounded optional MPC planner. Its 16-value encoding
omits the task identifier and compresses the actor's 16-value donor memory state
to a single mean. Both task identity and donor memory are observable to the agent.
Existing development pilots do not establish that planning improves task return,
and planning remains off by default. This work will retain those inputs in the
live planner and evaluate whether that gives useful behavior.

## Goals

- Give the live transition and value learners the agent's observable task
  identity and donor recurrent state, so multiple tasks and delayed cues are
  distinguishable without access to hidden world state.
- Improve delayed-reward control by using the existing bounded action planner and
  TD value estimate on `delayed_match` and `button_sequence`; include a simpler
  task such as `forage` as a regression control.
- Learn only from real executed transitions. Keep imaginary rollouts read-only;
  use environment rewards and terminal outcomes rather than synthetic success
  labels or hidden coordinates.
- Measure the capability on paired, held-out seeds before any default or promotion
  change. Preserve all task-level results, costs, abstentions, and regressions.
- After measurement, make the workbench explain what is active, what evidence is
  available, and how the controller performed; align project documentation with
  current executable behavior.

## Non-goals and anti-goals

- Do not add another standalone planner or grow the capability catalog for its
  own sake.
- Do not turn on planning by default unless the held-out evaluation meets its
  predeclared promotion rule.
- Do not feed hidden object locations, evaluator-only subgoal state, or future
  outcomes into the controller.
- Do not claim biological fidelity, physical safety, general intelligence, or
  broad task improvement from local synthetic experiments.
- Do not rewrite the workbench or vendor trees wholesale, add dependencies,
  publish results remotely, or change unrelated experiments.

## Approaches considered

1. **Expand the frontier architecture:** add more planning, memory, or cognitive
   modules in parallel. This creates new surfaces without first solving the
   current gap between optional components and measured task improvement.
2. **Capability first, then measure and polish (selected):** extend the actual
   executed-transition learner with visible task context, improve the integrated
   value-guided planner for delayed rewards, measure it under frozen held-out
   queries, then polish the workbench and project contracts. This is the user's
   selected sequence and keeps capability work tied to outcomes.
3. **Evidence or workbench first:** consolidate documentation and usability before
   attempting new behavior. This is a lower-risk first step, but does not match the
   user's chosen order; the same work is retained as later phases.

## Design

### Phase 1: task-aware delayed-reward control

- Extend the experience-state schema from 16 to 40 values. Keep the current 14
  predicted sensory, motion, cue, and time inputs, and use 26 fixed context
  values: the existing graph summary, the 9-value visible base-task one-hot, and
  the 16-value donor recurrent state. Configure the task-aware model with 40
  state values and 26 conditioning values. Validate the selected task codes and
  bound all numeric inputs.
- Keep the current 16-value experience model intact as the legacy/baseline path;
  add a separate task-aware model that starts cold when absent from a checkpoint.
  Do not silently reinterpret the old model weights or memory as 40-value state.
  Continue using the legacy path on complex tasks whose public task identity is
  aliased in the current 9-value observation code; task-aware planning is
  supported only for the validated native task IDs in this phase.
- Train the existing transition ensemble and TD value model only after observed
  outcomes. Keep the task and donor recurrent coordinates fixed through each
  short imagined rollout, as the planner does for other conditioning inputs. Keep
  candidate-action changes within the existing bounded trust region, compare
  against the actor baseline, and retain support/uncertainty abstention. Use the
  value estimate only as the short-horizon terminal value; do not add untrained
  imagined reward shaping.
- Exercise task-conditioned value-guided MPC for bounded horizons on
  `delayed_match` and `button_sequence`. Use `forage` to detect regressions on a
  simpler objective. Preserve the actor as the control path when support is
  insufficient or a forecast fails its gates.
- Keep the feature opt-in during development and report which context, gates, and
  fallback selected the executed action.

### Phase 2: frozen-query measurement

- Before inspecting query outcomes, record the compared configurations, task
  list, support and query budgets, seeds, primary outcome, promotion rule, and
  source/configuration fingerprints.
- Compare the unchanged actor, current planner, and task-aware value-guided
  planner. Use shared support data where appropriate; use independent held-out
  query worlds and disable all fitting during query. Add a task-context ablation
  if it can be kept within the same compute budget.
- Primary outcome: per-episode verified task completion rate on held-out worlds.
  Secondary outcomes: native return, time to completion, value/model errors,
  planner-use and abstention rates, action changes, compute evaluations, and
  wall-clock cost. Report task-specific and aggregate paired differences with
  uncertainty intervals; do not treat time steps as independent samples.
- Select a bounded CPU budget before the run. Use a development pilot to estimate
  variance, then freeze the confirmation sample and analysis before inspecting
  its results. No automatic promotion from a pilot.
- Promote only if the predeclared held-out primary comparison clears the
  practical improvement threshold without a material regression on the control
  task or a compute-cost violation. Otherwise keep the feature opt-in and record
  the negative or inconclusive result.

### Phase 3: workbench and project polish

- Add a focused capability/evidence view showing the active planner and task
  context, support sufficiency, prediction/abstention reason, realized task
  outcomes, and compute cost. Distinguish observed measurements from model
  forecasts and illustrative services.
- Make a reproducible study receipt inspectable from the workbench and CLI,
  including task-level records, frozen-query status, fingerprints, and known
  limitations.
- Reconcile the README, model/data cards, roadmap, and capability catalog with
  current source. Mark optional, standalone, illustrative, unavailable, and
  measured behavior distinctly; link each measured claim to its receipt.
- Apply only targeted usability and maintainability changes that the new flow
  exposes. Do not undertake a broad frontend rewrite as part of this design.

## Constraints and decisions left to the builder

- Keep current Python, NumPy, PyTorch, and local CPU execution paths; do not add
  runtime dependencies or require an accelerator.
- Preserve observation-only access to task identity. No privileged evaluator data
  enters action selection or model updates.
- Keep backward checkpoint compatibility and the default-off policy until the
  held-out decision rule passes.
- The builder may choose exact seed counts after measuring CPU cost and variance,
  but must freeze them before confirmation and report the resulting precision.
- This is a local research result. Receipts establish recorded provenance and
  arithmetic, not authenticity or real-world reliability.
