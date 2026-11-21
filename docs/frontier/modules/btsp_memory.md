# `btsp_memory`: one-shot BTSP memory with binary synapses

Module file: `synapse/frontier/btsp.py` · tests: `tests/test_frontier_memory.py` ·
experiment ids: `btsp_memory`, `btsp_memory_capacity` · engine extension: `BTSPExtension` (id `btsp_memory`) ·
baselines: `WillshawMemory`, `RandomRecallControl` · evidence tier: `synthetic_abstraction`.

The module implements the binary-weight rule for **behavioural time-scale synaptic plasticity** (BTSP) of
Wu & Maass (*Nature Communications*, 2025). In this rule, random plateau potentials gate stochastic flips of
binary synapses, so a pattern is stored after a single presentation. The module also implements the paper's
feedback pathway for completing a pattern from a partial cue. It uses the memory in three settings:

* a capacity benchmark against a classic Willshaw memory with the same number of synapses,
* a NEUROWEAVE engine extension that adds familiarity features to the readout,
* direct use as a content-addressable memory (`BTSPMemory`).

It is *inspired by* the paper and does not reproduce it. The paper's default network has 25,000 input and
39,000 memory neurons, while this module's largest benchmark uses 1,000 inputs and 833 memory units. Thresholds
are set more simply here (see below). None of the paper's figures or results are attempted. The model says nothing about
hippocampal biology.

## What it computes

`BTSPMemory(input_dim, memory_dim, *, density=0.6, plateau_prob=0.02, flip_prob=0.5, threshold=0.35,
feedback_threshold=0.5, seed=0)` has `m = input_dim` binary input units and `n = memory_dim` binary memory
units.

```
masks     C (n x m) forward, D (m x n) feedback: independent Bernoulli(density) connections, fixed by the seed
weights   W (n x m) forward, B (m x n) feedback: binary, initially all 0

store x   plateau set Q: every memory unit independently with probability plateau_prob (firing is irrelevant)
          for j in Q and every input i with x_i = 1 and C_ji = 1:
              W_ji: 0 -> 1 with probability p_ltp,  1 -> 0 with probability p_ltd       (paper Eq. 1: both 0.5)
          then, with the updated W:  B_ij <- B_ij OR (D_ij AND x_i AND z_j(x))     (one-shot Hebbian, saturating)

trace     z_j(c) = 1[ (W c)_j > threshold * density * |c| ]                    (McCulloch-Pitts memory units)
recall    r_i(c) = 1[ (B z(c))_i > feedback_threshold * density * |z(c)| ]     (McCulloch-Pitts input units)
```

`|c|` is the number of ones. `flip_prob` is one number (`p_ltp = p_ltd`) or a `(p_ltp, p_ltd)` pair. Masks come
from `default_rng(seed + 61403)`. Plateau and flip draws come from `default_rng(seed + 61417)`, so the same seed
and store sequence give identical weights. Weights are boolean arrays. Only existing synapses ever change.

How this maps onto the paper:

* **Plasticity (Eq. 1).** The binary-weight rule flips a weight from an active input to a memory neuron with
  a plateau, with probability 0.5: 0 becomes 1 and 1 becomes 0. The paper derives the 0.5 from the plateau's
  time window: input in the inner part of the window potentiates a weight that is not yet at its maximum,
  input in the outer part depresses a weight that is not yet at its minimum, and both parts are taken as
  equally likely, so with probability 0.5 a weight is not updated.
  * Every memory neuron receives a plateau independently with probability `f_q` for each new item. The default
    `f_q` is 0.005.
  * The paper notes that "BTSP does not depend on the firing of the postsynaptic neuron".
  * Initial weights are a zero matrix, and each item is presented once.
  * The paper also varies the LTD probability, which `flip_prob=(p_ltp, p_ltd)` allows.
* **Neurons.** Memory and input neurons are McCulloch-Pitts units. They fire when the weighted sum *exceeds* the
  threshold, as here.
* **Thresholds (simplified here).** The paper uses grid search to set one absolute memory threshold for its
  memory-trace experiments. For input completion, it sets both layers' thresholds by linear functions of the
  average fraction of ones in the input, fitted in preliminary simulations. This module sets the memory
  threshold proportional to the number of ones in the cue (within one cue condition every cue has the same
  number of ones, so this is a linear function of that fraction with zero intercept) and the input-layer
  threshold proportional to the number of active units in the memory trace.
* **Feedback.** Random feedback connections from memory to input units have the same density as the forward
  ones (0.6 in the paper). They are trained in one step by a Hebbian-like rule: a weight is incremented when
  both its memory unit and its input unit are 1, and saturates at one. Here the memory units' activity is the
  trace of the stored pattern under the updated forward weights.
* **Recall.** As in the paper, input completion is one forward and one feedback pass with no iteration (the
  paper produces it at the next time step after the cue; only its Hopfield comparison networks iterate).

Metrics:

* `overlap(a, b) = |a & b| / sqrt(|a| |b|)` is this module's recall metric, a cosine similarity. It is not the
  paper's "overlap ratio" of two memory traces, which divides the common ones by the average number of ones
  in a trace.
* `reconstruction_error(x, x', r) = HD(r, x) / HD(x, x')` is the paper's scaled reconstruction error. 0 is
  perfect, and 1 is no better than the cue.
* `make_cue` deletes a fraction of the active bits. With `kind='flip'`, it also turns on the same number of
  inactive bits.

Tests check these properties on cases with a known answer:

* With `plateau_prob = 1` and `flip_prob = 1`, a store sets exactly the connected synapses from active inputs.
  The trace and the feedback weights equal a hand computation. Storing the same pattern again is then pure
  depression and returns every forward weight to 0, while feedback weights never decrease.
* Without a plateau nothing changes, and synapses from inactive inputs never change. Weights stay boolean and
  stay on existing synapses.
* On a 200 x 200 network, the measured flip rates match `p_ltp = 0.3` and `p_ltd = 0.8` to within ±0.02.
* **One-shot storage.** 10 patterns are each stored once (200 inputs, 20 active, 200 memory units, plateau
  probability 0.1; means over 3 seeds and the 10 patterns).
  * Every stored pattern evokes a non-empty memory trace, and a full cue reconstructs the pattern with overlap
    above 0.95.
  * With 20 %, 40 % and 60 % of the active bits deleted, the overlap degrades gradually (never rising by more
    than 0.01 from one level to the next) and stays above 0.8.
  * At every deletion level the mean reconstruction overlap beats the mean overlap of the degraded cue, and
    cues with 30 % of their bits moved are also cleaned up.
* Storing 40 patterns instead of 10 lowers recall from 40 %-deleted cues by more than 0.1 (interference grows
  with load). Cues that were never stored retrieve little: on average their reconstruction overlaps neither
  the cue nor the closest stored pattern by more than 0.5.
* Each parameter (plateau probability, LTP and LTD probabilities, both thresholds, density, seed) changes what
  is recalled.

## Baselines

* `WillshawMemory(input_dim, *, threshold=1.0)` is the auto-associative form of the clipped-Hebbian binary
  memory of Willshaw, Buneman & Longuet-Higgins (1969).
  * Storing sets `A_ii' = 1` for every pair of co-active units, including `i = i'`.
  * Recall is `r_i = 1[(A c)_i >= threshold * |c|]`. `threshold = 1`, where every cue unit must agree, is the
    classic rule as usually described.
  * Tests: exact completion of half-deleted cues at threshold 1, tolerance of a wrong bit at threshold 0.8, and
    saturation after an all-ones pattern.
* `RandomRecallControl(input_dim, active, seed=0)` ignores the cue and returns a seeded random pattern with
  `active` ones. That is chance level, with expected overlap `active / input_dim`, which a test checks.

## Engine extension (`BTSPExtension`, id `btsp_memory`)

`BTSPExtension(memory_dim=64, *, active_fraction=0.25, density=0.6, plateau_prob=0.1, flip_prob=0.5,
threshold=0.35, feedback_threshold=0.5, gate=0.5, projection_dim=6, seed=0)` is a readout-only extension over
the binarised brain output.

* **Size.** `bind(engine)` copies `engine.n`, an integer up to 4096, and builds an empty `BTSPMemory` with that
  many inputs. Units born later are ignored.
* **Binarising.** After each engine step, the output is turned into a pattern `x_t` with the top
  `k = max(1, round(active_fraction * n))` units set (ties go to the lower index).
* **Features.** Recall happens *before* any store and gives three kinds of feature:
  * `recall_overlap = overlap(x_t, r(z(x_t)))`, how familiar the current state is;
  * `recall_confidence = min(1, |z(x_t)| / max(1, plateau_prob * memory_dim))`, the trace size relative to the
    expected trace size;
  * `projection_dim` fixed `+-1` projections of the trace, `P z / sqrt(max(1, |z|))`.

  In total `feature_dim = 2 + projection_dim` (8 by default).
* **Storing.** Only when `ctx['learn']` is true, the pattern is stored one-shot when its salience
  `max(min(|reward|, 1), 1 - recall_overlap)` exceeds `gate`. The plateau probability is scaled by that
  salience. This novelty-and-reward gate is this module's design, not the paper's, whose plateaus are random.
* **What it does not touch.** The extension never changes the drive. It keeps no engine reference, and
  `copy.deepcopy` gives an independent clone, including the plasticity random generator. With `ctx['learn']`
  false it never stores; it counts `frozen_steps` instead. `reset()` clears the features and keeps the stored
  memories.
* **Budget.** `parameter_count()` is the number of existing plastic synapses, forward plus feedback.

## Capacity benchmark (`run_capacity_benchmark`, id `btsp_memory_capacity`)

* **Patterns.** `patterns` random binary patterns with exactly `active` ones are stored once each, in order.
  At `checkpoints` evenly spaced loads, `test_per_checkpoint` stored patterns are recalled from degraded cues
  under every condition in `cue_kinds x cue_fractions` (delete and flip, 30 % and 50 %). Every arm sees the
  same cues.
* **Metric.** The metric is the normalised overlap between the stored pattern and its reconstruction,
  averaged over checkpoints and conditions (the mean of the capacity curve). Receipts also give the
  reconstruction error per checkpoint and condition.
* **Arms.**
  * `btsp`: `BTSPMemory` with `memory_dim = round(input_dim / (2 * density))`. Its forward plus feedback
    synapses then match the Willshaw matrix's `input_dim**2` in expectation; receipts report the exact counts.
  * `willshaw`: `WillshawMemory`.
  * `random_recall`: chance level.
  * `cue_only`: no memory; the degraded cue is its own reconstruction. This shows whether a memory improves on
    the cue at all.
* **Fair tuning.** Free parameters are tuned per arm on separate seeds (1000 and 1001; 1000 in quick mode) by
  the same metric. For BTSP these are the plateau probability, the threshold and the feedback threshold. For
  Willshaw it is the threshold. BTSP gets the larger grid because it has three free parameters.
  `tuning.<arm>.chosen_on_grid_edge` names any chosen value on a grid edge.
* **Verdict.** The verdict is a paired 95 % t interval over seeds, BTSP against Willshaw (higher is better).
  Receipts also report each memory against `random_recall` and against `cue_only`. Per-seed diagnostics give
  BTSP's fraction of forward weights at 1 and its mean trace size at each load.
* **Sizes.** The default uses 1,000 inputs with 50 active, 400 patterns, loads of 100, 200, 300 and 400, 64
  tested patterns per load, and 833 memory units at density 0.6. Quick mode uses 400 inputs with 20 active,
  120 patterns, loads of 30, 60, 90 and 120, 24 tested patterns per load, and 333 memory units.

## Lab experiment (`run_experiment`, id `btsp_memory`)

`harness.run_extension_experiment('btsp_memory', ...)` runs the NEUROWEAVE `delayed_recall` task (the target
is the input bit from two steps ago) with three arms:

* `baseline`: the unmodified engine and readout.
* `candidate`: the baseline plus `BTSPExtension` features.
* `control`: the baseline plus the same number of random leaky-trace features (the harness's capacity-matched
  `random_trace` control).

All arms share seeds, streams and the readout protocol, and held-out streams are evaluated frozen. The headline
verdict is `improved` only if the candidate beats both the baseline and the control. The control matches the
candidate's readout feature count (8), not its stored numbers: for seed 0 the measured receipts report 6,544
(quick) or 12,688 (default size) extension state numbers for the candidate and 208 or 400 for the control.

The defaults are the harness's: seeds 0, 1, 2; held-out seeds 100, 101; 300 steps; 48 neurons; clustered
topology; no extra mechanisms. The extension uses its defaults. Quick mode uses the harness `QUICK_ARGS` sizes
(24 neurons, 24 steps, one 12-step held-out stream) with the caller's seeds. The extension keys and the harness
keys (`task`, `heldout_seeds`, `steps`, `heldout_steps`, `neurons`, `topology`, `mechanisms`) can be
overridden.

## Public API

```python
MODULE_ID = 'btsp_memory'; CAPACITY_ID = 'btsp_memory_capacity'
EXPERIMENTS = {'btsp_memory': run_experiment, 'btsp_memory_capacity': run_capacity_benchmark}
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict             # sealed receipt
run_capacity_benchmark(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict     # sealed receipt
BTSPMemory(input_dim, memory_dim, *, density=0.6, plateau_prob=0.02, flip_prob=0.5, threshold=0.35,
           feedback_threshold=0.5, seed=0)
    .store(pattern, *, plateau_prob=None) -> {'plateau_units', 'ltp_flips', 'ltd_flips', 'trace_size',
                                              'new_feedback_synapses'}
    .drive(cue); .code(cue); .reconstruct(code); .recall(cue) -> {'code', 'reconstruction'}
    .forward_mask; .feedback_mask; .synapse_count; .state_numbers; .weight_fractions(); .config()
    .reset(); .export_state(); .import_state(state)
WillshawMemory(input_dim, *, threshold=1.0)       # .store, .recall, .synapse_count, .export_state, .import_state
RandomRecallControl(input_dim, active, seed=0)    # .store, .recall
BTSPExtension(memory_dim=64, *, active_fraction=0.25, density=0.6, plateau_prob=0.1, flip_prob=0.5,
              threshold=0.35, feedback_threshold=0.5, gate=0.5, projection_dim=6, seed=0)
    .binarise(output); .after_step(output, ctx); .features(); .telemetry(); .config(); .export_state(); .import_state(state)
overlap(a, b); hamming(a, b); reconstruction_error(target, cue, reconstruction)
make_cue(pattern, fraction, kind, rng); random_patterns(count, size, active, rng)
CAPACITY_DEFAULTS, CAPACITY_QUICK, CUE_KINDS, SOURCE
```

Every public function and constructor validates its input and raises `ValueError` before any state changes.
`export_state()` is JSON-safe: weights on existing synapses are stored as packed bits in base64, together with
the counters and the PCG64 state of the plasticity generator. `import_state` checks the configuration, the
exact bit lengths, the zero padding, the counters and the generator state before replacing anything, so a
failed import leaves the memory unchanged. After an import, further stores continue exactly as they would have
without the export. No global NumPy or torch random or thread state is touched.

## Connection to Sonapse

* **Not a live observer.** `btsp_memory` is not in `spec.OBSERVER_MODULE_IDS`.
* **Engine extension.** It influences nothing unless it is attached to a NEUROWEAVE engine with
  `attach_extension`, as in the lab experiment. Even when attached, it only adds readout features. It never
  changes the drive.
* **Standalone benchmark.** `run_capacity_benchmark` uses only the memories and synthetic patterns.
* **Defaults.** No Sonapse or NEUROWEAVE default, API or checkpoint format changes.

## Not implemented, and differences from the paper

* The paper's network sizes and sparsity. Its default is 25,000 inputs, 39,000 memory neurons, 0.5 % active
  bits and a plateau probability of 0.005, so about 195 memory neurons receive a plateau for each item. In the
  capacity benchmark, the tuned plateau probabilities give an expected 8.3 (default size) and 13.3 (quick mode)
  plateau units per stored pattern, and the measured mean traces hold about 8–15 units.
* The paper's threshold procedures: grid search for an absolute memory threshold, and fitted linear functions
  of the input's fraction of ones for input completion. Thresholds here are proportional to the active count.
* The time course of plateau potentials and the asymmetric time window. As in the paper's simplified rule, they
  are summarised by the flip probability.
* The paper's analyses and comparisons: random projections, Hopfield networks, the earlier
  postsynaptic-firing-dependent one-shot rule, repulsion of similar memory traces, and the line-drawing task.
  The paper does not compare BTSP with a Willshaw memory; this module does.
* Forgetting in the feedback pathway. As in the paper, feedback weights only saturate, so under a long store
  sequence they fill up (see the measured weight fractions below).
* The extension's salience gate and the plateau probability scaled by salience are this module's own design.

## Verified references

Each reference was checked on 2026-11-20.

1. Yujie Wu, Wolfgang Maass. **A simple model for Behavioral Time Scale Synaptic Plasticity (BTSP) provides
   content addressable memory with binary synapses and one-shot learning.** *Nature Communications* 16, 342
   (2025), published online 2 January 2025 (PMC record). doi:10.1038/s41467-024-55563-6,
   https://doi.org/10.1038/s41467-024-55563-6 · PMC11695864 · PMID 39747916 (the identifiers were matched with
   the PMC ID converter). The PMC record says the article was corrected, *Nat. Commun.* 16, 1150 (29 January
   2025). The following were checked in the PMC full text:
   * Eq. 1, applied "with probability 0.5" to inputs that arrive within the plateau's time window.
   * The per-neuron, per-item plateau probability `f_q` (default 0.005), and that BTSP does not depend on
     postsynaptic firing.
   * The zero initial weights and the strict firing threshold.
   * The grid-searched memory threshold, and the fitted linear threshold functions for input completion.
   * The Hebbian-like, saturating feedback learning at density 0.6.
   * The scaled reconstruction error `HD(r(x'), x) / HD(x, x')` and the overlap-ratio definition.
   * The default sizes: m = 25,000, n = 39,000, f_w = 0.6, f_p = 0.005.
   * Input completion is produced at the next time step after the cue (one pass); the Hopfield comparison
     networks are iterated 100 times.
   * Fig. 6D varies the LTD probability, and the main text points to Figs. S4–S5 for variants with a higher
     LTD and a lower LTP probability.

   The paper compares BTSP with random projections, Hopfield networks and an earlier one-shot rule that depends
   on postsynaptic firing. Neither its text nor its reference list mentions Willshaw. The supplementary
   material itself was not checked.
2. D. J. Willshaw, O. P. Buneman, H. C. Longuet-Higgins. **Non-Holographic Associative Memory.** *Nature*
   222(5197), 960–962, June 1969 (Crossref). doi:10.1038/222960a0. Only the bibliographic record was verified;
   the full text could not be opened in this build. The all-cue-units retrieval threshold (`threshold = 1`) is
   taken from later descriptions of the Willshaw model and was not checked against the 1969 text.

## Measured results

All runs were measured on 2026-11-20 on Windows 11 on ARM64, with Python 3.12.10, NumPy 2.4 and one BLAS/OpenMP
thread (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`). Every call used its defaults apart from
`quick`, with seeds 0, 1, 2. Each quick run was made three times and gave the same receipt digest every time;
each default-size run was made once (earlier runs on the same day gave the same numbers). The tests also check
that a second run gives the same digest. The digest prefix is listed so a rerun on this setup can be compared;
another platform may round floating-point results differently. Wall-clock covers the whole call and is given
as the range over the runs. The machine was shared with other jobs, so wall-clock is only indicative.

| Run | Verdict | Key numbers | Digest | Wall-clock |
|---|---|---|---|---|
| `run_capacity_benchmark(quick=True)` | `worse` | Curve-mean overlap: btsp 0.693 / willshaw 0.873 / random 0.050 / cue only 0.686. btsp − willshaw −0.180, CI [−0.255, −0.105]; btsp − cue only +0.007, CI [−0.063, 0.078]. BTSP by load (30, 60, 90, 120): 0.794, 0.749, 0.656, 0.574; Willshaw: 0.848, 0.889, 0.899, 0.858. Chosen: BTSP plateau 0.04, threshold 0.4, feedback 0.65; Willshaw threshold 0.6; no grid-edge flags; synapse ratio 0.999. Mean trace size 9.5–15.1 units over seeds and loads | `6fc711b5` | 2.07–2.67 s |
| `run_experiment(quick=True)` | `no_detectable_difference` | Held-out accuracy 0.500 in all three arms (chance 0.500). The CI is degenerate. Seed 0: 20 stores in 24 steps | `7ff1f576` | 1.62–2.03 s |
| `run_capacity_benchmark()` | `worse` | Curve-mean overlap: btsp 0.666 / willshaw 0.703 / random 0.050 / cue only 0.686. btsp − willshaw −0.038, CI [−0.052, −0.023]; btsp − cue only −0.020, CI [−0.038, −0.003]; willshaw − cue only +0.017, CI [0.014, 0.020]. BTSP by load (100, 200, 300, 400): 0.813, 0.729, 0.622, 0.499; Willshaw: 0.626, 0.769, 0.799, 0.619. Chosen: BTSP plateau 0.01, threshold 0.3, feedback 0.65; Willshaw threshold 0.8; no grid-edge flags; synapse ratio 0.999. Seed 0: forward weights at 1 rise from 2.4 % to 9.1 % of synapses, and the mean trace grows from 7.7 to 10.7 units (7.6–11.5 over all seeds and loads) | `9f015b29` | 53.1 s |
| `run_experiment()` | `no_detectable_difference` | Held-out accuracy: baseline 0.558 / candidate 0.549 / control 0.561 (chance 0.517). Candidate − baseline −0.009, CI [−0.028, 0.010]; candidate − control −0.012, CI [−0.039, 0.014]. Seed 0: 202 stores in 300 steps; 36 % of forward and 81 % of feedback synapses at weight 1 | `3abe23be` | 20.0 s |

At the same synapse count and at these toy sizes, the BTSP memory recalled worse than a tuned Willshaw memory
on average over the capacity curve.

* **Reconstruction against the cue.** Averaged over loads, BTSP's reconstruction did not improve on the
  degraded cue: no detectable difference in quick mode, and slightly worse at default size.
* **By load.** At the lowest default load, its mean overlap (0.813) was above both the cue (0.686) and
  Willshaw (0.626), whose single threshold is tuned for the whole curve. At higher loads it fell below both.
  Over the same loads the fraction of forward weights at 1 and the trace size grew (seed 0 above), which is
  consistent with growing interference between stored patterns; this was not tested further.
* **Engine extension.** On `delayed_recall`, the BTSP familiarity features made no detectable difference to
  the NEUROWEAVE readout. In the quick run every arm stays at chance, so it only checks that the pipeline runs.
  In the default run the feedback pathway filled up: 81 % of feedback synapses were at weight 1 on seed 0.

A possible reason for the gap to the paper is scale. Here an expected 8.3 (default size) or 13.3 (quick mode)
memory units receive a plateau for each stored pattern, and the measured mean traces hold about 8–15 units. In
the paper's default network about 195 neurons receive a plateau for each item. This explanation was not
tested. These results are statements about this toy only. They do not test the paper's network sizes or
claims.
