# `joint_failure`: joint-failure ensemble selection

Module file: `synapse/frontier/joint_failure.py` · tests: `tests/test_frontier_methodology.py` ·
experiment ids: `joint_failure` (kind `selection_benchmark`) and `joint_failure_neuroweave` (kind
`selection_application`) · no live observer · evidence tier: `synthetic_abstraction`.

The module chooses an odd-sized majority-vote **panel** of components from boolean failure observations
(component × condition × trial). It compares four selection rules and scores the frozen panels on held-out
trials.

The module is *inspired by* FAULTLINE, a kai9987kai portfolio laboratory for correlated and higher-order
failures in small voting ensembles. The following follow FAULTLINE's design:

* the four rules and their tie-breaking;
* the second factorial-moment pairwise bound;
* the 3-bit parity example.

These parts are this module's own:

* the NumPy array format;
* the seeded family generator with exact population errors;
* the three further constructed trap cases;
* the NEUROWEAVE application.

It reproduces none of FAULTLINE's recorded numbers.

## What it computes

**Data.** `failures[i, c, t]` is true when component `i` is wrong on trial `t` of condition `c`. Every
condition has the same number of trials `T`.

**Majority-vote error.** A panel `P` of odd size `k` votes by majority, with threshold `s = (k + 1) / 2`:

```
W_P(c, t) = sum_{i in P} F[i, c, t]             Err(P | c) = (1/T) sum_t 1[ W_P(c, t) >= s ]
```

* **Two labels.** With binary labels every wrong member votes for the single wrong label, so this is the
  exact vote error.
* **More labels.** With more labels it is an upper bound on plurality-vote error. If fewer than `s` members
  are wrong, the correct label has a strict majority.
* **Signatures.** `Err` depends only on the joint failure pattern of each trial. `failure_signatures` lists,
  per condition, every distinct set of failing components with its count. These are the empirical failure
  hyperedges, and the list includes the empty set.
* **Signature path.** `majority_error_from_signatures` recomputes `Err` from those counts alone. The tests
  check it against the array computation on every 3-member panel of random correlated data.

**Pairwise bound.** Since `1[W >= s] <= C(W, 2) / C(s, 2)`, taking expectations gives:

```
B(P | c) = sum_{i<j in P} Pr(F_i and F_j | c) / C(s, 2)  >=  Err(P | c)          (k = 1: the individual error)
k = 3:   B = Pr(W = 2) + 3 Pr(W = 3)        but        Err = Pr(W = 2) + Pr(W = 3)
```

A unanimous failure therefore counts three times in the pairwise score. This is FAULTLINE's argument for
why pairwise co-failure misranks panels with rare higher-order faults.

**Selectors.** All four use calibration trials only. Every key is an integer count, so ties are exact. A tie
goes to the next key and finally to the lexicographically first panel.

| policy | ranking key (minimised) |
|---|---|
| `accuracy` | `sum_{i in P}` individual failures (pooled individual error) |
| `pairwise` | worst-condition `B(P \| c)`, then `sum_c B(P \| c)`, then accuracy |
| `pooled` | pooled majority error, then worst-condition majority error, then accuracy |
| `joint` | worst-condition majority error `max_c Err(P \| c)`, then pooled, then accuracy |

* **Equivalence to FAULTLINE.** With equal trials per condition, the sums over conditions are proportional
  to FAULTLINE's pooled rates. The keys therefore match FAULTLINE's `core.select_panel`, which uses exact
  fractions and breaks the last tie by component id order.
* **Tested against a direct implementation.** The tests compare every selector with a direct exact-fraction
  re-implementation of those keys. They cover 12 random correlated arrays, `k ∈ {1, 3, 5}` and all four
  policies.
* **`run_selection`.** It selects on calibration, freezes the panels and scores them on the held-out test
  trials. It also reports an `oracle`: the panel with the lowest *test* worst-condition error. The oracle
  is chosen on test data, so it is a reference, never a selector.
* **`evaluate_panel`.** It reports per-condition error, worst condition, pooled error, unanimous-failure rate
  and the worst pairwise bound.
* **Bounds.** At most 24 components, 64 conditions, 100,000 trials, 5,000,000 cells and 100,000 candidate
  panels. Panels are scored in chunks of at most 2,000,000 panel-condition-trial cells. Bad input raises
  `ValueError` before any computation.

## Constructed known-answer cases (`constructed_cases`, `check_constructed_cases`)

| case | construction | known answer |
|---|---|---|
| `parity` (FAULTLINE's example) | 3 components; `even` holds the 4 even-weight failure patterns, `odd` the 4 odd-weight ones | marginals 1/2 and pair rates 1/4 in both, majority error **3/4 vs 1/4** |
| `accuracy_trap` | 0–2 individually best (error 0.10) but fail together on 20 % of `stress` trials; 3–5 (0.15) never co-fail | `accuracy` picks (0, 1, 2), worst-condition error 0.20; `joint` finds a panel with 0 |
| `pairwise_trap` | (2, 3, 4) fails only unanimously, 10 % (bound 30 %); (0, 1, 2) fails in pairs, 18 % (bound 18 %); cross pairs make mixed panels worse | `pairwise` picks (0, 1, 2), error 0.18; `joint` picks (2, 3, 4), error 0.10 |
| `pooled_trap` | 0–2 fail together on 30 % of `b` trials only (pooled 0.15, worst 0.30); 3–5 fail together on 20 % of trials in both conditions | `pooled` picks worst-condition error 0.30; `joint` picks 0.20 (pooled 0.20) |

* **In the receipt.** The nine named checks are recorded in every benchmark receipt as `constructed_checks`.
* **Further hand cases in the tests.** One has failing sets `{}, {0}, {0,1}, {1,2}, {0,1,2}`, where
  `Err = 3/5` exactly and the pairwise sum is `5/5`. Another is a 5-member case.

## Synthetic generator (`make_failure_data`)

**Components and conditions.** There are `families × family_size` components (default 3 × 3 = 9),
shuffled per seed so that tie-breaking cannot favour a family. The conditions are `nominal` plus one
`stress_f` per family (default 4).

* **`shared`.**
  * Base error: family 0 (the "accurate" family) draws from `U(0.01, 0.03)`, the others from
    `U(0.03, 0.07)`.
  * In `stress_f`, family `f`'s stressor is active with probability `U(0.25, 0.40)`, drawn once per seed and
    family.
  * When active, it hits each member of family `f` with probability 0.85.
* **`independent`** (negative control).
  * There is no stressor. Component `i` fails with `e_i · mu_c`, where `e_i ~ U(0.04, 0.16)`, `mu_0 = 1` and
    `mu_c ~ U(1.0, 1.6)`.
  * Every condition ranks the components identically. Poisson-binomial tails increase in each `p_i`, so the
    `k` individually best components form the optimal panel in every condition.
* **`shift`** (unobserved common cause).
  * Calibration is identical to `shared`.
  * On held-out `nominal` trials a common cause is active with probability 0.6. It then hits each member of
    a cross-family set with probability 0.9.
  * With `shift_target='hardened'` (the default) the set is the lowest-base-error member of every family.
    That is adversarial to diversity-based selection by construction. `'random'` draws one member per
    family.

**Population errors.** `expected_condition_errors(structure, panel, split)` gives the exact population
`Err(P | c)`. Given the state of the (at most one) channel acting in a condition, components fail
independently, so `Err` is a mixture of Poisson-binomial tails. The tests compare it with 40,000-trial
Monte Carlo (within 5 standard errors).

**Digest.** Each dataset carries a SHA-256 digest of its packed arrays.

## Benchmark (`run_experiment`, id `joint_failure`)

* **Protocol.** For every seed and scenario, all four selectors see the same calibration trials and the same
  `C(9, 3) = 84` candidate panels. The frozen panels are scored on fresh test trials.
* **Verdict.** `harness.compare` of joint vs accuracy held-out worst-condition error in the `shared` scenario
  (lower is better). `improved` requires the paired 95 % t interval over seeds to exclude zero.
* **Also reported.**
  * Joint vs pairwise and joint vs pooled (`shared`), and joint vs accuracy (`independent`, `shift`).
  * The exact population error of every chosen panel.
  * The oracle.
  * The constructed checks.
  * A budget: candidate panels, calibration component calls (`9 × 4 × T_cal`), `k` deployment calls per case
    and 0 trainable parameters.
* **No capacity-matched control.** Every rule deploys `k` components with no trainable parameters and the
  same calibration data, so the comparison is between selection rules at equal cost.
* **Sizes.** Defaults: `k = 3`, 64 calibration and 512 test trials per condition. `quick=True` uses 48 and 256.
  Generator parameters can be overridden, and unknown keys raise `ValueError`.

## NEUROWEAVE application (`run_neuroweave_application`, id `joint_failure_neuroweave`)

* **Components.** Each component is a NEUROWEAVE engine that differs only in its mechanism set. Seed,
  `clustered` topology, neuron count, training stream and head initialisation are shared.
* **Training.** Each engine is trained with the lab's `run_task_phase(learn=True)`, with one `ReadoutHead`
  per task. The first task, `delayed_recall`, supplies the engine reward.
* **Conditions.** The four temporal tasks: `delayed_recall`, `xor`, `rhythm` and `context_switch`. Targets
  are binary, so majority-of-failures equals majority-vote error exactly.
* **Failure recording.** Failures are recorded frozen (`learn=False`, cloned engines and head copies) on a
  calibration stream (`seed + 50,000`) and a disjoint test stream (`seed + 60,000`). Training, calibration
  and test stream seeds must be pairwise disjoint.
* **Verdict.** `harness.compare` of joint vs accuracy held-out worst-task error over seeds.
* **Mechanism sets.** `()`, `sparse_code`, `homeostatic_governor`, `astrocyte_regulation`, `multi_soma`,
  `reward_plasticity` and `dendritic_compartments`. Quick mode uses the first five.
  * **How they were chosen.** A decision-diversity probe on seed 99, which is not an evaluation seed. It
    trained each single mechanism at 16 neurons for 120 steps and at 24 neurons for 160 steps (clustered
    topology, one head per task) and ranked mechanisms by how many training-phase head decisions they changed
    relative to the unmodified engine. Accuracy and selector outcomes played no part. The six sets are the
    probe's top six, in rank order (244, 127, 116, 71, 66 and 62 changed decisions over both sizes).
  * **Many mechanisms do nothing here.** 17 of the 45 single mechanisms changed no decision in that probe.
    A re-run of the probe on 2026-11-20 gave the same ranking and the same 17.
* **Sizes.** Defaults: 7 sets, 24 neurons, 160 training / 96 calibration / 96 test steps. Quick: 5 sets,
  16 neurons, 120 / 32 / 32 steps.

## Public API

```python
POLICIES = ('accuracy', 'pairwise', 'pooled', 'joint');  SCENARIOS = ('shared', 'independent', 'shift')
validate_failures(failures, *, name='failures') -> bool ndarray (components, conditions, trials)
majority_threshold(panel_size) -> int
majority_error(failures, panel) -> ndarray (per condition)
pairwise_bound(failures, panel) -> ndarray (per condition)
failure_signatures(failures) -> list[dict];   majority_error_from_signatures(signatures, panel) -> ndarray
panel_scores(failures, panel_size) -> dict;   evaluate_panel(failures, panel) -> dict
select_panels(failures, panel_size, policies=POLICIES) -> dict[str, tuple];  select_panel(failures, panel_size, policy)
run_selection(calibration, test, panel_size) -> dict          # every policy + oracle, frozen panels on test
make_failure_data(seed, *, scenario='shared', **params) -> dict
expected_condition_errors(structure, panel, *, split='test') -> ndarray
constructed_cases() -> dict;   check_constructed_cases() -> dict[str, bool]
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> sealed receipt
run_neuroweave_application(*, quick=False, seeds=(0, 1, 2), **overrides) -> sealed receipt
EXPERIMENTS = {'joint_failure': run_experiment, 'joint_failure_neuroweave': run_neuroweave_application}
```

Randomness comes only from `numpy.random.default_rng(seed + offset)`. The module never touches the global
NumPy RNG, never imports torch, and has no stateful observer or extension to export or import.

## Connection to Sonapse

* **What it consumes.** This is a methodology module. It does not attach to an engine or observe live
  activity. It consumes failure arrays, so any set of Sonapse or NEUROWEAVE components with per-trial
  correctness under named conditions can be compared. The application builds those arrays from per-task
  readout correctness of differently configured engines.
* **What it uses.** The NEUROWEAVE lab helpers (`make_spec`, `ReadoutHead`, `run_task_phase`,
  `clone_engine`, `make_stream`) and `synapse.frontier.harness` (`base_receipt`, `compare`, `seal`,
  `TASKS`).
* **Scope.** It does not change any engine or any other module.

## Differences from FAULTLINE and not implemented

* **Data.** FAULTLINE reads `faultline-dataset-v1` JSON (ids, truth, predictions; 3–12 components; unequal
  trials per condition allowed). This module takes boolean failure arrays with equal trials per condition
  and up to 24 components.
* **Fixtures.**
  * FAULTLINE's fixtures are 8 named sensor specialists with hand-designed regimes, driven by event-keyed
    SHA-256 draws. They include `glare_surge` faults that hit five components at once and a latent clock
    fault on a hardened triad.
  * The generator here is parametric and has no composed higher-order faults. On it the `pairwise`,
    `pooled` and `joint` rules chose the same panels in every measured seed of the `shared` scenario.
  * Their differences are therefore shown only on the constructed traps.
* **Not implemented.**
  * FAULTLINE's receipts and replay verifier, HTTP API, browser UI, test witnesses, bootstrap intervals and
    dataset import.
  * Plurality or weighted voting.
  * Any defence against common causes absent from calibration. No rule here protects against them, and the
    `shift` scenario exists to show that.

## Source (portfolio)

* **FAULTLINE: Joint-Failure Ensemble Selection Laboratory** (kai9987kai).
  <https://github.com/kai9987kai/FAULTLINE>.
* **Files read.** In a read-only clone at commit `b7ae1b9` (24 October 2026), the following were read on
  2026-11-20:
  * `README.md`;
  * `docs/design.md` (design decision dated 24 October 2026);
  * `PAPER.md` (dated 24 October 2026);
  * `faultline/core.py` (selector keys and metrics);
  * `faultline/fixtures.py` (scenarios and the parity demo).
* **Nothing executed.** No FAULTLINE code was run.
* **Not re-verified.** The literature FAULTLINE cites was not re-checked for this module and is not cited
  here.

## Measured results

All numbers were measured on 2026-11-20 (UTC) on Windows 11 ARM64 with Python 3.12.10 and NumPy 2.4.3, with
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Other agents shared the CPU, so wall-clock is
indicative only. Errors are held-out (test) majority-vote errors averaged over seeds `(0, 1, 2)`. "pop." is
the exact population worst-condition error of the chosen panels under the generator.

**Benchmark, quick** (`run_experiment(quick=True)`: 9 components, 4 conditions, `k = 3`, 48 calibration and
256 test trials per condition).

| scenario | accuracy | pairwise | pooled | joint | oracle | joint − accuracy, 95 % CI |
|---|---|---|---|---|---|---|
| `shared` | 0.160 (pop. 0.155) | 0.038 | 0.038 | 0.038 (pop. 0.029) | 0.021 | −0.122 [−0.387, +0.143] |
| `independent` | 0.030 (pop. 0.027) | 0.035 | 0.035 | 0.035 (pop. 0.034) | 0.022 | +0.005 [−0.010, +0.020] |
| `shift` | 0.398 (pop. 0.414) | 0.197 | 0.197 | 0.197 (pop. 0.197) | 0.027 | −0.202 [−0.756, +0.352] |

* **Verdict.** **no_detectable_difference**.
* **Per-seed `shared` differences.** 0.000, −0.195 and −0.172. In seed 0 `accuracy` happened to pick the
  same panel as the other rules.
* **Other values.** All 9 constructed checks pass. The digest prefix is `2ef7216676bbedb6` (identical across
  separate processes). Wall-clock was 0.08–0.10 s.

**Benchmark, default** (`run_experiment()`: 64 calibration and 512 test trials per condition).

| scenario | accuracy | pairwise | pooled | joint | oracle | joint − accuracy, 95 % CI |
|---|---|---|---|---|---|---|
| `shared` | 0.258 (pop. 0.259) | 0.027 | 0.027 | 0.027 (pop. 0.028) | 0.023 | **−0.231 [−0.239, −0.224]** |
| `independent` | 0.029 (pop. 0.026) | 0.038 | 0.028 | 0.028 (pop. 0.028) | 0.025 | −0.001 [−0.004, +0.002] |
| `shift` | 0.327 (pop. 0.335) | 0.382 | 0.382 | 0.382 (pop. 0.373) | 0.029 | +0.055 [−0.977, +1.086] |

* **Verdict.** **improved**: joint selection lowered held-out worst-condition error in `shared`.
* **Other values.** All 9 constructed checks pass. The digest prefix is `1bfc6d4ab8e769a7` (identical in two
  separate runs). Wall-clock was 0.10–0.20 s.
* **Same panels.** `pairwise`, `pooled` and `joint` chose the same panel in every `shared` seed of the quick,
  default and ten-seed runs.
* **Ten seeds.** A further run, `run_experiment(seeds=range(10))` at default size, gave these joint −
  accuracy differences:
  * `shared`: −0.224 [−0.243, −0.206], improved;
  * `independent`: +0.002 [−0.007, +0.011];
  * `shift`: −0.023 [−0.236, +0.189].

**NEUROWEAVE application.**

| | quick (5 sets, 16 neurons, 120/32/32 steps) | default (7 sets, 24 neurons, 160/96/96 steps) |
|---|---|---|
| verdict | **no_detectable_difference** | **no_detectable_difference** |
| joint − accuracy worst-task error (95 % CI) | 0.000 [0.000, 0.000] | −0.031 [−0.166, +0.103] (per seed −0.094, 0, 0) |
| mean worst-task error: accuracy / pairwise / pooled / joint / oracle | 0.563 / 0.552 / 0.563 / 0.563 / 0.552 | 0.552 / 0.521 / 0.552 / 0.521 / 0.507 |
| best single component's worst-task error (per seed) | 0.438 / 0.563 / 0.500 | 0.365 / 0.583 / 0.469 |
| oracle panel's worst-task error (per seed; chosen on test data) | 0.594 / 0.563 / 0.500 | 0.469 / 0.583 / 0.469 |
| components with identical calibration failures (pairs) | 1 / 6 / 1 of 10 | 0 / 0 / 0 of 21 |
| mean pairwise calibration-failure agreement | 0.68 / 0.86 / 0.79 | 0.88 / 0.87 / 0.67 |
| a component's mean test accuracy over tasks (min–max over components and seeds) | 0.49–0.65 | 0.53–0.72 |
| digest (prefix) | `1f7709e86ceea33e` (identical across separate processes) | `6f1e1af36c5f24f6` (identical in separate runs) |
| measured wall-clock | 9.7–10.9 s | 25.2–27.9 s |

Reading:

* **Synthetic benchmark.** Joint-failure selection removes the shared-stressor failure mode that accuracy
  selection walks into. The improvement shows at default size and over ten seeds. Quick mode is too small,
  with 3 seeds and one tie.
* **Negative controls.** The independent control shows no difference, as it should. In the unobserved
  common-cause scenario the advantage disappears: every rule is far from the oracle, and joint is not
  significantly better or worse than accuracy.
* **What this generator cannot separate.** It does not distinguish `joint` from `pairwise` or `pooled`.
* **NEUROWEAVE.** The readouts are close to chance on most tasks and their failures are strongly correlated.
  No selector beats another detectably. In all six measured seed runs (quick and default), even the oracle
  panel was no better than the best single component on test worst-task error: worse in seed 0, equal in
  seeds 1 and 2. These runs exercise the pipeline on real engine outputs; they are not evidence for any
  mechanism or selector.
