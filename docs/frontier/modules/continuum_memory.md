# `continuum_memory`: continuum memory system

Module file: `synapse/frontier/continuum_memory.py` · tests: `tests/test_frontier_memory.py` ·
experiment ids: `continuum_memory`, `continuum_memory_recall` · engine extension: `ContinuumMemoryExtension`
(id `continuum_memory`) · evidence tier: `synthetic_abstraction`.

The module combines two ideas from Behrouz and colleagues and uses them at toy scale:

* the **continuum memory system** (CMS) of *Nested Learning*: several memories, each updated at its own
  frequency, so fast levels adapt quickly and slow levels keep information longer (the paper's Hope module is
  a self-modifying Titans followed by a CMS);
* the **neural long-term memory update** of *Titans*: a gradient step on an associative loss, accumulated
  with momentum ("surprise") and shrunk by a forgetting gate.

It is *inspired by* these papers and does not reproduce either. The memories are linear maps, not MLP blocks.
Momentum, learning rate and forgetting are fixed numbers, not learned functions of the input. There is no
attention, no self-modifying Titans, no meta-learned initial state and no language modelling. None of the
papers' results are attempted.

## What it computes

`ContinuumMemory(key_dim, value_dim, levels=4, base_period=1, lr=0.5, momentum=0.0, forget=0.01, *,
surprise_forget=0.0, max_forget=0.5, chunk_reduce='sum', bound=100.0)` holds `L = levels` linear memories
`M_l` (`value_dim x key_dim`), one momentum buffer `S_l` per level, and one gradient accumulator per level.
Level `l` has period `P_l = base_period * 2**l` (1, 2, 4, 8 with the defaults). A key is either one
`(key_dim,)` vector shared by every level or a `(levels, key_dim)` array with one key block per level.

```
read       r_l = M_l k_l                       y = sum_l r_l                      (read / predict)
loss       L_t = ||y_t - v_t||^2               g_{l,t} = dL_t/dM_l = 2 (y_t - v_t) k_{l,t}^T
chunk      G_l = sum of g_{l,t} over the P_l writes of the chunk   ('sum')    or  G_l / P_l   ('mean')
when the write count is a multiple of P_l:
           S_l <- clip(momentum * S_l - lr * G_l)
           M_l <- clip((1 - alpha_l) * M_l + S_l)
           alpha_l = min(max_forget, forget + surprise_forget * (mean surprise of the chunk))
surprise   s_t = e_t / (e_t + ||v_t||^2),   e_t = ||y_t - v_t||^2   (s_t = 0 when e_t = 0)
```

Every gradient is taken from the joint read `y_t` with the memories as they are at that write, so a slow
level accumulates the errors left after the fast levels have adapted. `clip` bounds `M` and `S` to
`[-bound, bound]`. Every array has a fixed shape, so the state stays finite and bounded. The memory has no
randomness.

How this maps onto the sources:

* **Nested Learning, Eq. 71.** The parameters of level `l` change only once every `C(l)` steps, by the sum of
  `eta_t f(theta_t; x_t)` over the chunk, where `f` is the optimiser's error term. Here `f` is the gradient of
  `L_t` with respect to `M_l`. `chunk_reduce='sum'` is a constant `eta`, and `'mean'` is `eta = lr / P_l`.
* **Nested Learning, Eq. 74.** This is the independent ("head-wise") CMS: the levels read the input
  independently and an aggregation combines their outputs. The paper suggests a learnable weighted sum. This
  module uses a fixed, unweighted sum.
* **Titans, Eqs. 10, 12 and 13.** The associative loss is `||M(k) - v||^2`. The surprise recurrence is
  `S_t = eta_t S_{t-1} - theta_t grad`, and the forgetting gate gives `M_t = (1 - alpha_t) M_{t-1} + S_t`. Here
  `momentum` is `eta`, `lr` is `theta`, and `alpha` is a fixed number (`forget`), optionally raised by the
  chunk's surprise. In Titans' experiments `eta_t`, `theta_t` and `alpha_t` are functions of the token.
* **Surprise score.** `s_t` is this module's own bounded score. It is 0 for a perfect read, 1/2 for an empty
  memory, and approaches 1 for a confidently wrong read. Titans calls the gradient term itself the "momentary
  surprise". `s_t` only feeds the optional surprise-raised forgetting (`surprise_forget > 0`, off by default).

Tests check these properties on cases with a known answer:

* One write with `lr = 1/2` on a unit key stores exactly `v k^T`. The value reads back, and an orthogonal key
  reads zero.
* Level `l` changes exactly on the writes divisible by `P_l`. This was checked over 48 writes with periods
  3, 6, 12 and 24.
* The chunk update equals a hand computation (sum of per-write gradients, or their mean).
* Momentum and forgetting follow the Titans recurrences step by step.
* With zero gradient, level `l` decays by `(1 - forget)` per own update, so it decays at its own rate. A stored
  association fades as `(1 - forget)^n`.
* Surprise raises `alpha` up to `max_forget`.
* The state stays within `bound` under extreme inputs.
* Each mechanism (momentum, forgetting, surprise forgetting, chunk mean, base period, level count, learning
  rate) changes the predictions when toggled.

## Engine extension (`ContinuumMemoryExtension`, id `continuum_memory`)

`ContinuumMemoryExtension(levels=4, key_dim=8, base_period=1, lr=0.25, momentum=0.0, forget=0.01, *,
features_per_level=2, surprise_forget=0.0, max_forget=0.5, chunk_reduce='sum', seed=0)` is a readout-only
extension.

* **Context.** `c_t` is the task input (zero-padded or truncated to `TASK_INPUT_DIM = 8`; zeros when absent)
  followed by the 16-channel fused sensory frame from the step context (24 numbers).
* **Key.** The key is a fixed binary random-feature map, `k(c) = sign(R c + b) / sqrt(levels * key_dim)`, with
  `R` and `b` drawn from `default_rng(seed + 51407)`. It is split into one `key_dim` block per level and has
  unit norm overall.
* **Write.** After every engine step, and only when `ctx['learn']` is true, the extension writes the
  association `k(c_{t-1}) -> sensory_t`, so the memory learns to forecast the next sensory frame from the
  current context. Nothing is written before a previous context exists, for example right after `reset()`.
* **Features.** The extension reads every level with `k(c_t)`. Each level's 16-channel read is projected onto
  `features_per_level` fixed unit-norm directions (from `default_rng(seed + 51419)`) and clipped to
  `[-10, 10]`. The result is `feature_dim = levels * features_per_level` features (8 by default).
* **What it does not touch.** The extension never changes the drive and does not read the brain output. It
  keeps no engine reference, and `copy.deepcopy` gives an independent clone. With `ctx['learn']` false it does
  not write; it counts `frozen_steps` instead. `reset()` forgets the previous context and the features, and
  keeps the memory.
* **Budget.** `parameter_count()` is the number of memory entries, `levels * 16 * key_dim` (512 by default).
  `state_numbers()` adds the momentum buffers, accumulators, random features, projections and counters.

A test feeds a period-3 sequence of sensory frames. Within 60 steps, the forecast error falls below 1 % of its
initial value.

## Recall benchmark (`run_recall_benchmark`, id `continuum_memory_recall`)

The benchmark tests the CMS schedule on a task with a known structure. It does not use the NEUROWEAVE engine.

* **Stream** (`make_recall_stream`). Keys are random unit vectors in `levels * key_dim` dimensions (128 by
  default). Values are standard normal (8 dimensions).
  * 4 short-gap items recur after gaps of 3–10 steps, and 8 long-gap items recur after gaps of 40–80 steps.
    When several items are due, the earliest-due one is shown and the others wait, so a realised gap can be
    longer than the drawn one.
  * Every step without a due item shows a fresh one-off association, which acts as interference.
  * An optional `remap_prob` gives a short-gap item a new value at a recurrence. This is off in the experiment.
* **Scoring.** Scoring is prequential: predict the value from the key, score the squared error at short-gap
  and long-gap recurrences, then write.
* **Arms.**
  * `cms`: 4 levels with periods 1, 2, 4 and 8. Level `l` reads block `l` of the key.
  * `single`: one Titans-style memory (momentum and forgetting) that reads the whole key with period 1.
  * `none`: the running mean of the values seen so far.

  The two memory arms read the same keys and hold the same number of entries (`8 x 128 = 1024`). If every
  CMS level had period 1, the two arms would compute the same thing. The difference between them is therefore
  the update schedule, including how often forgetting acts.
* **Fair tuning.** For each memory arm, `(lr, momentum, forget)` is chosen from the same grid on separate
  tuning seeds (1000 and 1001; 1000 in quick mode). The objective is the mean of short-gap and long-gap MSE.
  `tuning.<arm>.chosen_on_grid_edge` names any chosen value that sits on a grid edge, where the optimum may lie
  outside the grid. A value of 0 is treated as a natural bound, not an edge.
* **Verdict.** The verdict is a paired 95 % t interval over seeds on long-gap MSE (lower is better), `cms`
  against `single`. Receipts also report the short-gap comparison and both memories against `none`.
* **Sizes.** The default uses 800 steps, `lr` in {1/64, …, 1/2} (6 values), `momentum` in
  {0, 0.5, 0.75, 0.875, 0.9375} and `forget` in {0, 0.01}. Quick mode uses 240 steps, `lr` in
  {0.125, 0.25, 0.5}, `momentum` in {0, 0.25, 0.5} and `forget` in {0, 0.02}.

## Lab experiment (`run_experiment`, id `continuum_memory`)

`harness.run_extension_experiment('continuum_memory', ...)` runs the NEUROWEAVE `delayed_recall` task (the
target is the input bit from two steps ago) with three arms:

* `baseline`: the unmodified engine and readout.
* `candidate`: the baseline plus `ContinuumMemoryExtension` features.
* `control`: the baseline plus the same number of random leaky-trace features of the brain output (the
  harness's capacity-matched `random_trace` control).

All arms share seeds, streams and the readout protocol, and held-out streams are evaluated frozen. The headline
verdict is `improved` only if the candidate beats both the baseline and the control. The control matches the
candidate's readout feature count (8), not its stored numbers: the measured receipts report 2,525 extension
state numbers for the candidate and 208 (quick) or 400 (default size) for the control.

The defaults are the harness's: seeds 0, 1, 2; held-out seeds 100, 101; 300 steps; 48 neurons; clustered
topology; no extra NEUROWEAVE mechanisms. The extension uses its defaults. Quick mode uses the harness
`QUICK_ARGS` sizes (24 neurons, 24 steps, one 12-step held-out stream) with the caller's seeds. The extension
keys (`levels`, `key_dim`, `base_period`, `lr`, `momentum`, `forget`, `features_per_level`,
`surprise_forget`, `max_forget`, `chunk_reduce`) and the harness keys (`task`, `heldout_seeds`, `steps`,
`heldout_steps`, `neurons`, `topology`, `mechanisms`) can be overridden.

## Public API

```python
MODULE_ID = 'continuum_memory'; RECALL_ID = 'continuum_memory_recall'
EXPERIMENTS = {'continuum_memory': run_experiment, 'continuum_memory_recall': run_recall_benchmark}
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict            # sealed receipt
run_recall_benchmark(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict      # sealed receipt
ContinuumMemory(key_dim, value_dim, levels=4, base_period=1, lr=0.5, momentum=0.0, forget=0.01, *,
                surprise_forget=0.0, max_forget=0.5, chunk_reduce='sum', bound=100.0)
    .read(query) -> (levels, value_dim); .predict(query) -> (value_dim,)
    .write(key, value) -> {'error', 'surprise', 'updated_levels'}
    .periods; .parameter_count; .state_numbers; .config(); .reset(); .export_state(); .import_state(state)
ContinuumMemoryExtension(levels=4, key_dim=8, base_period=1, lr=0.25, momentum=0.0, forget=0.01, *,
                         features_per_level=2, surprise_forget=0.0, max_forget=0.5, chunk_reduce='sum', seed=0)
    .key(context); .after_step(output, ctx); .features(); .telemetry(); .export_state(); .import_state(state)
make_recall_stream(seed, steps, *, key_dim=128, value_dim=8, n_short=4, n_long=8, short_gap=(3, 10),
                   long_gap=(40, 80), remap_prob=0.0) -> {'keys', 'values', 'kind', 'item', 'gap'}
RECALL_DEFAULTS, RECALL_QUICK, KIND_CODES, CHUNK_REDUCTIONS, SOURCE
```

Every public function and constructor validates its input and raises `ValueError` before any state changes.
`import_state` validates the whole state before replacing anything. It checks the configuration, shapes,
finiteness, the bound, and that the chunk and update counters are consistent with the write count. A failed
import leaves the memory unchanged. No global NumPy or torch random or thread state is touched.

## Connection to Sonapse

* **Not a live observer.** `continuum_memory` is not in `spec.OBSERVER_MODULE_IDS`.
* **Engine extension.** It influences nothing unless it is attached to a NEUROWEAVE engine with
  `attach_extension`, as in the lab experiment. Even when attached, it only adds readout features. It never
  changes the drive.
* **Standalone benchmark.** `run_recall_benchmark` uses only `ContinuumMemory` and synthetic streams.
* **Defaults.** No Sonapse or NEUROWEAVE default, API or checkpoint format changes.

## Not implemented, and differences from the papers

* MLP memory blocks (Titans uses MLPs with `L_M >= 1` layers; Hope's CMS uses MLP blocks). Here every level is
  one linear map.
* Learned, input-dependent `eta_t`, `theta_t` and `alpha_t`, and learned key, value and query projections.
  The extension uses a fixed random binary key map.
* The learnable aggregation weights of the independent CMS, and the nested and sequential CMS variants
  (Nested Learning Eqs. 72 and 73), including meta-learned initial states.
* Hope's self-modifying Titans module (and the softmax attention of the paper's Hope-Attention variant), the
  parallel chunk-wise training of Titans (mini-batch gradients with a parallel associative scan for the
  surprise terms), and every language-modelling, continual-learning and long-context result.
* Per-level momentum and forgetting applied at chunk boundaries. This combination of the two papers' update
  rules is this module's own design.
* Periods on a fixed geometric ladder (`base_period * 2**l`). Nested Learning defines each level's chunk size
  from its update frequency, `C(l) = max_i C(i) / f_l`; the geometric ladder is this module's choice.
* The surprise score `s_t` (see above).

## Verified references

Each reference was checked on 2026-11-20 against its arXiv listing and arXiv HTML full text, and against the
NeurIPS 2025 virtual-conference pages.

1. Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni. **Nested Learning: The Illusion of Deep
   Learning Architectures.** *Advances in Neural Information Processing Systems (NeurIPS) 2025*, poster
   (https://neurips.cc/virtual/2025/poster/116123). arXiv:2512.24695, v1 submitted 31 December 2025 (the only
   version listed on 2026-11-20); the arXiv comment says a version of the work is published at NeurIPS 2025.
   https://arxiv.org/abs/2512.24695. Checked in the arXiv HTML of v1:
   * Section 7.1, Eq. 70: the CMS chain of MLP blocks.
   * Eq. 71: level `l` is updated every `C(l)` steps with the summed error terms of its chunk, and
     `C(l) = max_i C(i) / f_l`.
   * Eqs. 72–73: the nested and sequential variants.
   * Eq. 74: the independent ("head-wise") variant with an aggregation function; the text suggests "a learnable
     weighted sum".
   * Definition 2: the update frequency is the "number of updates per unit of time".
   * Section 8.3: Hope is self-modifying Titans followed by a CMS; the Hope-Attention variant replaces the
     self-modifying Titans with softmax attention.
2. Ali Behrouz, Peilin Zhong, Vahab Mirrokni. **Titans: Learning to Memorize at Test Time.** *Advances in
   Neural Information Processing Systems (NeurIPS) 2025*, poster
   (https://neurips.cc/virtual/2025/loc/san-diego/poster/119639). arXiv:2501.00663, v1 submitted
   31 December 2024. https://arxiv.org/abs/2501.00663. Checked in the arXiv HTML:
   * Eqs. 9–10: surprise with momentum, with the "past surprise" and "momentary surprise" terms.
   * Eqs. 11–12: keys and values, and the loss `||M_{t-1}(k_t) - v_t||^2`.
   * Eqs. 13–14: the forgetting gate `alpha_t` in `[0, 1]`.
   * Eq. 15: retrieval without a weight update.
   * Section 3.1: the memory is an MLP with `L_M >= 1` layers.
   * Section 3.2: chunk-wise mini-batch training, a parallel associative scan for the surprise terms, and the
     statement that, in their experiments, `alpha_t`, `theta_t` and `eta_t` are functions of the tokens.

## Measured results

All runs were measured on 2026-11-20 on Windows 11 on ARM64, with Python 3.12.10, NumPy 2.4 and one BLAS/OpenMP
thread (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`). Every call used its defaults apart from
`quick`, with seeds 0, 1, 2. Each quick run was made three times and gave the same receipt digest every time;
each default-size run was made once (an earlier run on the same day gave the same numbers). The tests also
check that a second run gives the same digest. The digest prefix is listed so a rerun on this setup can be
compared; another platform may round floating-point results differently. Wall-clock covers the whole call
and is given as the range over the runs. The machine was shared with other jobs, so wall-clock is only
indicative.

| Run | Verdict | Key numbers | Digest | Wall-clock |
|---|---|---|---|---|
| `run_recall_benchmark(quick=True)` | `no_detectable_difference` | Long-gap MSE: cms 0.192 / single 0.194 / none 1.149. cms − single −0.0016, CI [−0.0089, 0.0058]. Short-gap MSE: cms 0.0156 / single 0.0146 / none 0.664; cms − single +0.0010, CI [−0.0007, 0.0028]. Both memories beat `none` (cms − none −0.957, CI [−1.228, −0.687]). Both arms chose `lr` 0.25, `momentum` 0.25, `forget` 0; no grid-edge flags. Seed 0 scores 133 short-gap and 24 long-gap recurrences | `cb276e79` | 1.98–3.14 s |
| `run_experiment(quick=True)` | `no_detectable_difference` | Held-out accuracy 0.500 in all three arms (chance 0.500). The CI is degenerate (all paired differences 0). Seed 0 telemetry: 23 writes; level update counts 23, 11, 5, 2 | `2bea12ce` | 1.57–2.07 s |
| `run_recall_benchmark()` | `no_detectable_difference` | Long-gap MSE: cms 0.111 / single 0.112 / none 1.142. cms − single −0.0015, CI [−0.0048, 0.0019]. Short-gap MSE: cms 0.0119 / single 0.0105 / none 0.652; cms − single +0.0014, CI [0.0006, 0.0021], so cms is `worse` on short gaps. Chosen: cms `lr` 1/16, `momentum` 0.75; single `lr` 1/32, `momentum` 0.875; both `forget` 0; no grid-edge flags. Seed 0 scores 463 short-gap and 99 long-gap recurrences | `e736bf69` | 43.7 s |
| `run_experiment()` | `no_detectable_difference` | Held-out accuracy: baseline 0.558 / candidate 0.559 / control 0.561 (chance 0.517). Candidate − baseline +0.001, CI [−0.004, 0.006]; candidate − control −0.002, CI [−0.015, 0.010]. Seed 0 telemetry: 299 writes; level update counts 299, 149, 74, 37 | `b26b9c5f` | 18.1 s |

At equal entries and equal keys, with each arm tuned on the same grid, the multi-period schedule did not
detectably improve long-gap recall over a single Titans-style memory at these sizes, and it was slightly worse
on short gaps in the default run. Tuning gave both arms no forgetting and the same effective step
`lr / (1 - momentum)`: 0.25 in the default run, and 1/3 in quick mode, where both chose the same grid point.
Both memories beat the no-memory predictor by a wide margin. On `delayed_recall` the sensory-forecast
features made no detectable difference to the NEUROWEAVE readout. In the quick run every arm stays at chance,
so it only checks that the pipeline runs.

These are statements about a linear toy on synthetic streams. They say nothing about language modelling or
about the published results.
