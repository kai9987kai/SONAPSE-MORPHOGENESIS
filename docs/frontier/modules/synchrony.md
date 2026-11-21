# `synchrony`: synchronisation representation and neuron-level models

Module file: `synapse/frontier/synchrony.py` · tests: `tests/test_frontier_synchrony.py` ·
experiment ids: `synchrony`, `synchrony_ctm` · engine extension: `SynchronyExtension` (id `synchrony`) ·
live observer: `SynchronyObserver` · standalone model: `MiniCTM` · evidence tier: `synthetic_abstraction`.

The module borrows three ideas from **Continuous Thought Machines** (CTM; Darlow et al., NeurIPS 2025) and uses
them at toy scale:

* **Synchronisation as a representation.** The exponentially decayed inner product of two units' activity
  histories, kept by a two-number recursion per unit pair. It is used as extra readout features of a NEUROWEAVE
  brain (engine extension) and as a read-only activity statistic (live observer).
* **Private neuron-level models (NLMs).** Every neuron of `MiniCTM` has its own tiny MLP over its last `M`
  pre-activations.
* **Internal ticks with a tick-selecting loss.** `MiniCTM` "thinks" for `T` ticks per input and is trained
  on the minimum-loss tick and the most-certain tick.

It is *inspired by* the CTM and does not reproduce it. The models here are thousands of parameters, not
millions. The benchmark is cumulative parity of 12-bit sequences (8 in quick mode), not 64. Nothing below says
anything about the paper's mazes, ImageNet or parity results, about biological synchrony, or about thought.

## What it computes

**Synchronisation (CTM Sec. 3.4, Eq. 10; App. H, Eqs. 13–17).** For a unit pair `(i, j)` with decay rate
`r >= 0` and activities `z^1 … z^t`:

```
S_ij^t = sum_{tau=1..t} e^{-r (t - tau)} z_i^tau z_j^tau  /  sqrt( sum_{tau=1..t} e^{-r (t - tau)} )

alpha <- e^{-r} alpha + z_i z_j        beta <- e^{-r} beta + 1        S = alpha / sqrt(beta)
```

The recursion needs two numbers per pair and no stored history. `decayed_synchrony(history, left, right,
decay)` is the closed form. Tests check that the extension's recursion equals it on a hand-computed
three-step example, to 1e-12 relative error, and on a random 40-step history, to 1e-12 absolute. They also
check that `MiniCTM`'s batched recursion equals it at every tick for both of its synchronisation
representations.

**Bounds.** With `|z| <= 1`, `|alpha| <= beta`, so `|S| <= sqrt(beta) < 1 / sqrt(1 - e^{-r})`. For a steady
product, `S` grows like the square root of the effective window; that is the paper's normalisation. The
optional "unit" scale `S * sqrt(1 - e^{-r})` stays in `[-1, 1]`; with `z = 1` everywhere it equals
`sqrt(1 - e^{-r t})` exactly (tested).

**Pair sampling (CTM App. C.2, "random pairing").** `sample_pairs(n, pairs, self_pairs, rng)` returns
`self_pairs` self pairs `(i, i)` first, then distinct unordered cross pairs `i < j`. The CTM adds self pairs
so the representation can recover a snapshot of single-unit activity. One difference from the reference code:
it samples indices with replacement, while here self pairs use distinct units and cross pairs repeat only
when more are requested than `n (n - 1) / 2` exist.

**Timescales.** The CTM learns one rate `r_ij` per pair, starting at 0 (no decay). The extension and the
observer fix the rates instead: `log_uniform_timescales` spreads `tau_p = 1 / r_p` log-uniformly (stratified,
in a seeded order) over a range, so features cover several timescales at once.

## Engine extension (`SynchronyExtension`, id `synchrony`)

```python
SynchronyExtension(pairs=32, *, seed=0, self_pairs=None, timescales=(1.0, 32.0), feature_scale='ctm')
```

* `bind(engine)` samples `pairs` unit pairs (default `self_pairs = pairs // 8`) from the `engine.n` units
  present at bind time. It keeps no reference to the engine. Units born later are ignored and missing units
  count as silent, so the feature shape never changes.
* `after_step(output, ctx)` runs the recursion on the brain output `z` (clipped to `[-1, 1]`; brain outputs
  already lie there). Pair `p` has the fixed decay `exp(-1 / tau_p)`, with `tau_p` log-uniform over
  `timescales` steps.
* `features()` returns the `pairs` values of `S` (`feature_scale='ctm'`), or `S * sqrt(1 - decay)` (`'unit'`),
  clipped to `[-10, 10]`. Before the first step it returns zeros.
* Nothing adapts: there are no trainable parameters (`parameter_count() == 0`), and `ctx['learn']` changes
  nothing. The accumulators still follow activity, which is inference. `transform_drive` is not overridden,
  so the brain's drive and outputs are bit-identical with and without the extension (tested).
* `export_state()` / `import_state()` round-trip exactly. `import_state` validates everything before it
  changes anything: the configuration and pair indices must match, the arrays must be finite and of the
  right shape, `beta` must equal its closed form `(1 - d^steps) / (1 - d)`, and `|alpha| <= beta`.
* `copy.deepcopy` gives an independent clone. `telemetry()` reports mean and max `|S|`.

## Live observer (`SynchronyObserver`)

```python
SynchronyObserver(n_units, *, seed=7, history=256, pairs=64, self_pairs=None, zclip=5.0)
SynchronyObserver.from_spec(n_units, spec)        # FrontierSpecV1: seed, history, sync_pairs
```

* **z-scoring.** Each unit has a running mean and variance. They are exact (Welford) for the first `history`
  observations and exponentially weighted with rate `1/history` afterwards. z-scores are clipped to
  `[-zclip, zclip]`.
* **Timescales.** `observer_timescales(history)` gives `2 * 4^k` steps up to `history / 2`, so
  `(2, 8, 32, 128)` for `history = 256`. At each timescale every sampled pair keeps the recursion above on
  z-scores, and every unit in a pair keeps `q_u <- d q_u + z_u^2`.
* **Two measures per pair and timescale.** `S` is the CTM-scaled synchrony. The decayed correlation
  `C = alpha / sqrt(q_i q_j)` lies in `[-1, 1]` by Cauchy–Schwarz.
* **`summary()`** (JSON-safe, no wall-clock, under 16 KB even at 4096 pairs; tested):
  `module`, `n_units`, `pairs`, `self_pairs`, `cross_pairs`, `timescales`, `zscore_window`,
  `observed_steps`;
  `per_timescale` (one row per timescale with `tau`, `mean_abs_S`, `max_abs_S`, `mean_abs_corr` and
  `max_abs_corr` over cross pairs);
  `global_synchrony_index` (mean `|C|` over cross pairs at the slowest timescale);
  `top_pairs` (the 8 cross pairs with the largest `|C|` at the slowest timescale, each with `i` and `j`
  indices into the observed units, plus `corr`, `corr_by_tau` and `S_by_tau`);
  `synchrony_entropy` (Shannon entropy of the `|C|` distribution over cross pairs divided by `log(cross pairs)`:
  1 when synchrony is spread evenly, lower when it is concentrated; `None` when undefined).
* **Known answers (tested).** Identical units give `C = 1` on every pair and timescale and entropy 1. A planted
  in-phase pair and two planted anti-phase pairs are the top three pairs, with `C = +1` and `C = -1`. For
  independent noise, mean `|C|` falls monotonically with the timescale and is below 0.15 at `tau = 128`.
* The observer copies its input and never writes to it. Every buffer has a fixed size: `O(n_units)` moments
  and `O(timescales × pairs)` accumulators, with no activity history. `import_state` validates fully before
  mutating, including the `beta`/`steps` identity and the Cauchy–Schwarz bound.

## `MiniCTM` (PyTorch)

```python
MiniCTM(seq_len, *, vocab=2, classes=2, d_model=32, memory=4, nlm_hidden=4, ticks=8, d_input=16,
        n_sync_out=32, n_sync_action=16, self_pairs=4, decay_init=(0.03, 1.0), seed=0)
model(tokens, ticks=None, *, track=False) -> (logits (B, T, L, C), certainty (B, T)) [, record]
```

The input (token ids, shape `(B, L)`) is read whole. Token and learned position embeddings pass through
Linear and LayerNorm to give identical keys and values (as in the paper's parity setup, App. F.4). From learnable
start states `z^0` and pre-activation history `A^0`, each tick `t = 1 … T` runs, in the reference code's order:

```
q   = W_in S_action                                   action synchrony -> attention query     (Eq. 7)
o   = softmax(kv q / sqrt(d_input)) kv                single-head attention over the input     (Eq. 8)
a   = LayerNorm(GLU(W_syn [z ; o] + b))               synapse model, depth 1                   (Eq. 1)
A   = [A[:, 1:], a]                                   last M pre-activations per neuron        (Eq. 2)
z_d = GLU(GLU(A_d W1_d + b1_d) W2_d + b2_d)           private NLM of neuron d                  (Eq. 3)
alpha <- e^{-r} alpha + z_i z_j ;  S_out = alpha / sqrt(beta)
y_t = W_out S_out                                     per-tick logits for all L positions      (Eq. 6)
```

* **NLMs** (`NeuronLevelModels`). The weights `W1` with shape `(D, M, 2H)` and `W2` with shape `(D, H, 2)`
  hold one private slice per neuron. Each layer is one batched einsum over the neuron axis
  (`'dbm,dmh->dbh'`), the reference's `einsum('BDM,MHD->BDH')` with the neuron axis first. A test checks the
  two contractions agree. It also checks privacy: the gradient of neuron `k`'s output reaches only neuron
  `k`'s weights and history, and changing every other neuron's history leaves neuron `k`'s output and
  weight gradients bit-identical.
* **Synchronisation** uses random pairing with `self_pairs` self pairs per representation. Each pair has a
  learnable rate `r = softplus(rho)`, initialised log-uniform over `decay_init`, which gives timescales of
  1–33 ticks. The action pairs and the output pairs share one recursion state, because both read the same
  post-activations. The action representation at tick `t` is therefore built from `z^0 … z^{t-1}`, and the
  output representation from `z^0 … z^t`. Both start from the learnable `z^0`, as in the reference code.
* **Certainty** `C_t` is 1 minus the normalised entropy, averaged over output positions.
* **Loss** (`ctm_tick_loss`, Sec. 3.5, Eq. 11). `L_t` is the cross-entropy averaged over positions. Per sample,
  `t1 = argmin_t L_t` and `t2 = argmax_t C_t`, and `loss = (L_t1 + L_t2) / 2`. With one tick it reduces to
  plain cross-entropy (tested). Evaluation reads each sample's most-certain tick.
* **Tested.** Output shapes. A 1-tick run equals the first tick of a longer run. Logits differ across ticks.
  Parameter counts match the analytic formula. Construction never touches the global torch RNG, and the same
  seed gives the same weights whatever the global RNG state. The fused attention equals explicit softmax
  attention in values and gradients.

**Baselines.** `AttentiveGRU` is a `GRUCell` unrolled for the same `T` ticks. It reads the input through the
same attention reader, with the query taken from its hidden state, and is trained with the same tick loss. It
stands in for the paper's parameter-matched LSTMs run for the same number of ticks. `ParityMLP` is one hidden
ReLU layer over the whole ±1 sequence, with one tick.

## CTM benchmark (`run_ctm_benchmark`, id `synchrony_ctm`)

* **Task.** Cumulative parity (CTM Sec. 6, App. F.3): the target at position `l` is the parity of bits
  `0 … l`, and the whole sequence is presented at once. `cumulative_parity_data(L, seed, heldout)` enumerates
  all `2^L` sequences and splits off `heldout` of them by a seeded permutation, so held-out sequences are
  never trained on.
* **Arms.** `MiniCTM`, `AttentiveGRU` and `ParityMLP`. The GRU and MLP widths are chosen so their trainable
  parameter counts match `MiniCTM`'s within 10 % (else `ValueError`). Counts are computed analytically and
  checked against the instantiated models.
* **Protocol.** Per seed, all three models share the split, the minibatch index sequence (NumPy generator),
  AdamW (fused CPU kernel where available), the learning rate, weight decay 0.01, gradient clipping at 1.0
  and the step count. PyTorch runs on one thread, and the caller's thread count is restored. The global torch
  RNG is never used.
* **Verdict.** `harness.compare` on per-seed held-out per-position accuracy (mean over positions and held-out
  sequences), MiniCTM against the best baseline. The best baseline is the one with the higher mean, chosen
  after the fact, which favours the baseline. `verdict_basis` names it. Comparisons against both baselines are
  also stored.
* **Receipt.** `config` (every size, the optimiser, the readout, the loss, the torch version and threads);
  `models.<name>` with `trainable_parameters`, `per_seed` (held-out and train accuracy, per-position accuracy,
  whole-sequence accuracy, last-tick accuracy, mean most-certain tick, a loss curve, the optimiser
  implementation and wall-clock) and a `summary`; `parameter_budget`; `comparisons`; `verdict`;
  `verdict_basis`; `estimated_cost` (forward multiply-accumulates per sequence and for training), kept apart
  from `measured_wallclock_s`; and `notes`.

| Setting | quick | default |
|---|---|---|
| `seq_len` L / held-out of 2^L | 8 / 64 of 256 | 12 / 512 of 4096 |
| `ticks`, `d_model`, `d_input`, `memory`, `nlm_hidden` | 4, 24, 12, 4, 4 | 8, 32, 16, 4, 4 |
| `n_sync_action`, `n_sync_out`, `self_pairs` | 12, 24, 3 | 16, 32, 4 |
| steps × batch, learning rate | 100 × 32, 0.01 | 800 × 64, 0.01 |
| trainable parameters: MiniCTM / GRU / MLP | 4036 / 3978 (h 25) / 4041 (h 161) | 6600 / 6680 (h 32) / 6610 (h 178) |

## Lab experiment (`run_experiment`, id `synchrony`)

`harness.run_extension_experiment('synchrony', factory, task='rhythm', control='random_trace',
metric='accuracy', ...)` runs three arms: the NEUROWEAVE baseline readout, the baseline plus
`SynchronyExtension` features, and the baseline plus the same number of random leaky-trace features (the
capacity-matched control). All arms share seeds, streams and the readout protocol, and held-out streams are
evaluated frozen. The verdict is `improved` only if the candidate beats both the baseline and the control. The
defaults are those of the harness (seeds 0, 1, 2; held-out seeds 100, 101; 300 steps; 48 neurons; clustered
topology; no extra mechanisms), with 32 pairs, `timescales=(1, 32)` and `feature_scale='unit'`, so the
features share the control's `[-1, 1]` range. Quick mode uses the harness `QUICK_ARGS` sizes with the
caller's seeds and 16 pairs. Overrides: `pairs`, `self_pairs`, `timescales`, `feature_scale`, `task`,
`heldout_seeds`, `steps`, `heldout_steps`, `neurons`, `topology`, `mechanisms`.

## Public API

```python
MODULE_ID = 'synchrony'
EXPERIMENTS = {'synchrony': run_experiment, 'synchrony_ctm': run_ctm_benchmark}
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict        # sealed receipt
run_ctm_benchmark(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict     # sealed receipt
SynchronyExtension(pairs=32, *, seed=0, self_pairs=None, timescales=(1.0, 32.0), feature_scale='ctm')
SynchronyObserver(n_units, *, seed=7, history=256, pairs=64, self_pairs=None, zclip=5.0)
sample_pairs(n, pairs, self_pairs, rng); log_uniform_timescales(count, low, high, rng)
decayed_synchrony(history, left, right, decay); observer_timescales(history)
cumulative_parity_data(seq_len, seed, heldout); ctm_tick_loss(logits, targets, certainty=None)
# PyTorch classes, created on first attribute access (importing the module does not import torch):
MiniCTM, NeuronLevelModels, SequenceReader, AttentiveGRU, ParityMLP
```

NumPy randomness comes from `numpy.random.default_rng(seed + offset)`. PyTorch parameters are drawn from
private seeded `torch.Generator` objects by small private layer classes (linear, layer norm, GRU cell) that
create their own parameters with PyTorch's default initialisation distributions instead of calling the
`torch.nn` constructors, so the global torch RNG is never read or advanced. Tests check each layer against its
`torch.nn` counterpart and check that construction leaves the global RNG state unchanged. Every public function
and constructor validates its input and raises `ValueError` on bad input.

## Connection to Sonapse

* **Live observer.** `synchrony` is in `spec.OBSERVER_MODULE_IDS`, so a `FrontierSpecV1` may list it in
  `modules`. `SynchronyObserver.from_spec(n_units, spec)` takes `seed`, `history` and `sync_pairs` from the
  spec. The observer reads a copy of activity and never writes to the brain, graft bus, actor or any shared
  RNG. This module does not wire it into the live runtime; the runtime hook lives elsewhere.
* **Engine extension.** It influences nothing unless attached to a NEUROWEAVE engine with
  `attach_extension`, as in the lab experiment. Even then it only adds readout features.
* **Standalone model.** `MiniCTM` and its baselines exist only inside `run_ctm_benchmark`.
* **Defaults.** No Sonapse or NEUROWEAVE default, API or checkpoint format changes.

## Not implemented, and differences from the paper

* **Decay rates.** They are fixed in the extension and the observer; the CTM learns them. In `MiniCTM` they
  are learned as `r = softplus(rho)` with a log-uniform start. The reference code clamps `r` to `[0, 15]` and
  starts it at 0.
* **Attention.** It is single-head, without the separate query/key/value and output projections of the
  reference's `nn.MultiheadAttention`; the paper's parity models used 8 heads. Keys and values come from a
  learned embedding reader, not a convolutional backbone.
* **Synapse model.** Only the depth-1 Linear–GLU–LayerNorm block is implemented, not the U-Net synapse.
* **NLMs.** They omit the reference `SuperLinear`'s learnable output temperature, its optional LayerNorm and
  dropout.
* **Pairing.** Only random pairing is implemented; the paper's parity runs used semi-dense pairing.
* **Training schedule.** Constant learning rate, with no warmup or cosine schedule (the paper used both).
* **Scale.** The paper used 64-bit sequences, `d_model` 1024, `d_input` 512, up to 100 ticks and 200,000
  iterations. Here L is 12 or 8, `d_model` is 32 or 24, there are 4–8 ticks and 100–800 AdamW
  steps. These runs measure early learning of short parities. They are not a test of the
  paper's claim.
* **Baselines.** The GRU replaces the paper's LSTM. It is trained with the same tick loss and read at its
  most certain tick; the paper trained its LSTMs on the final tick, reporting that both losses were unstable
  for them.
* **Not attempted.** Adaptive halting by certainty threshold at inference, and any of the paper's other tasks
  (mazes, ImageNet, sorting, Q&A, RL).

## Verified references

Checked on 2026-11-17 against the arXiv listing, the NeurIPS virtual-conference page, the paper's full text
(arXiv PDF) and the reference code. Re-checked on 2026-11-20 (review) against the reference code: the
`compute_synchronisation` recursion (`alpha <- r alpha + z_i z_j`, `beta <- r beta + 1`, `S = alpha / sqrt(beta)`),
the decay clamp to `[0, 15]`, the initial output-synchronisation call before the tick loop, the loop order
(action synchronisation, attention, synapses, trace, neuron-level models, output synchronisation, prediction)
and `parity_loss` (per-position mean, `argmin` loss tick, `argmax` of `1 - normalised entropy`, mean of the
two); `MiniCTM` and `ctm_tick_loss` match them.

1. Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, Llion Jones (Sakana AI). **Continuous Thought
   Machines.** *Advances in Neural Information Processing Systems (NeurIPS) 2025*, spotlight poster.
   arXiv:2505.05522 (v1 8 May 2025, v4 3 October 2025). https://arxiv.org/abs/2505.05522 ·
   https://neurips.cc/virtual/2025/loc/san-diego/poster/115192. Checked against the full text: Sec. 3.2–3.5
   (Eqs. 1–11: synapse, pre-activation history, NLMs, synchronisation and its decay rescaling, the
   min-loss/max-certainty loss); App. C.2 (random pairing with `n_self` self pairs); App. F.3–F.5 (parity
   data, architecture, optimisation and parameter-matched LSTM baselines); and App. H (Eqs. 12–17, the
   recursive computation).
2. Reference implementation: **SakanaAI/continuous-thought-machines**,
   https://github.com/SakanaAI/continuous-thought-machines. Read on 2026-11-17: `models/ctm.py`
   (`compute_synchronisation` recursion; decay parameters initialised at 0 and clamped to `[0, 15]`;
   random-pairing selection with self pairs; depth-1 synapse; tick-loop order), `models/modules.py`
   (`SuperLinear`: per-neuron `einsum('BDM,MHD->BDH')` and a learnable temperature) and `utils/losses.py`
   (`parity_loss`: cross-entropy averaged over positions, then the mean of the min-loss and most-certain
   ticks).

## Measured results

All runs were measured on **2026-11-20** on Windows 11 on ARM64, with Python 3.12.10, torch 2.11 (CPU), NumPy 2.4
and `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` (the CTM benchmark also runs torch on one thread).
The machine was shared with other jobs, so wall-clock is indicative only. Receipts are deterministic: two calls
in one process gave the same digest, and the quick digests below were identical in two separate processes that
day. Seeds are `(0, 1, 2)` throughout.

**CTM benchmark, quick** (`run_ctm_benchmark(quick=True)`; digest `c24433021c0c76a0`; 17.8 s under load).
Cumulative parity of 8 bits, 64 held-out of 256 sequences, 100 steps × 32.

| Model | Parameters | Held-out per-position accuracy (per seed) | Mean | Train accuracy (mean) |
|---|---:|---|---:|---:|
| `MiniCTM` (4 ticks) | 4036 | 0.553, 0.572, 0.576 | 0.567 | 0.611 |
| `AttentiveGRU` (4 ticks) | 3978 | 0.537, 0.572, 0.596 | 0.568 | 0.591 |
| `ParityMLP` | 4041 | 0.658, 0.678, 0.662 | 0.666 | 0.841 |

* `mini_ctm − gru`: −0.0013, 95 % CI [−0.0451, +0.0425]; `mini_ctm − mlp`: −0.0990, CI [−0.1270, −0.0709].
  The verdict compares against the MLP (the higher held-out mean): **`worse`**.

**CTM benchmark, default** (`run_ctm_benchmark()`; digest `d838aee28cb46d8e`; 228.9 s under load). 12 bits, 512
held-out of 4096 sequences, 800 steps × 64.

| Model | Parameters | Held-out per-position accuracy (per seed) | Mean | Whole-sequence accuracy (mean) |
|---|---:|---|---:|---:|
| `MiniCTM` (8 ticks) | 6600 | 0.639, 0.648, 0.668 | 0.652 | 0.003 |
| `AttentiveGRU` (8 ticks) | 6680 | 0.622, 0.652, 0.708 | 0.660 | 0.005 |
| `ParityMLP` | 6610 | 0.908, 0.877, 0.889 | 0.891 | 0.210 |

* `mini_ctm − gru`: −0.0087, CI [−0.0806, +0.0631]; `mini_ctm − mlp`: −0.2399, CI [−0.3043, −0.1754] →
  **`worse`**.
* Per position, `MiniCTM` is at 1.0 for the first three positions, 0.89 for the fourth and near 0.5 (chance)
  from the fifth on: at this size and training budget it does not learn cumulative parity beyond a few bits.
  Its most certain tick varies by seed (mean tick 2.4, 4.5 and 6.4 of 8). The paper's parity models are far
  larger and trained far longer; this run says nothing about them.

**Lab experiment** (`run_experiment`, task `rhythm`, metric held-out accuracy).

| Run | Digest | Held-out accuracy: baseline / candidate / control | Candidate − baseline | Candidate − control | Verdict | Wall-clock |
|---|---|---|---|---|---|---|
| quick (24 neurons, 24 / 12 steps, 16 pairs) | `94d638da05ea2d3e` | 0.750 / 0.750 / 0.750 (chance 0.75) | 0.000 (all seeds equal) | 0.000 | `no_detectable_difference` | 1.9 s |
| default (48 neurons, 300 / 150 steps, 32 pairs) | `dfaebbab4e3fcbae` | 0.799 / 0.809 / 0.797 (chance 0.80) | +0.010, CI [−0.015, +0.035] | +0.012, CI [−0.021, +0.046] | `no_detectable_difference` | 17.9 s |

In quick mode every arm predicts the majority class on the 12 held-out steps, so accuracies tie exactly; the
synchrony features do change the readout (held-out log-loss 0.624 / 0.576 / 0.559 for the baseline and
0.635 / 0.580 / 0.559 for the candidate, per seed). At the default size the readout is close to the
majority-class rate for all arms; the features neither help nor hurt measurably.

**Observer cost.** `SynchronyObserver.from_spec(256, FrontierSpecV1())` (64 pairs, `history = 256`): 0.09 ms
per `observe()` (mean of 300 calls on uniform random activity), and a `summary()` of 2.9 KB of JSON.

Unit tests (`tests/test_frontier_synchrony.py`, 36 tests) check, among other things: the recursion against the
closed form `decayed_synchrony`; the CTM loss against a hand-computed min-loss / max-certainty selection; the
`MiniCTM` synchronisation against a direct sum over ticks; identical, in-phase and anti-phase units in the
observer; validated state round-trips; and sealed, deterministic receipts that leave the global RNGs and thread
count unchanged.
