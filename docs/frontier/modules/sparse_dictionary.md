# `sparse_dictionary`: sparse dictionary of brain activity

Module file: `synapse/frontier/sparse_dictionary.py` · tests: `tests/test_frontier_sae.py` ·
experiment id: `sparse_dictionary` · live observer: `SparseDictionaryObserver` ·
evidence tier: `synthetic_abstraction`.

The module learns an online **TopK sparse autoencoder** (a sparse dictionary) of unit activity. It also runs a
lab experiment that asks whether the dictionary features that best predict a task target matter more to a
frozen readout than random features do. The dictionary is *inspired by* the TopK autoencoders of Gao et al.
(2024). It does not reproduce that paper's setting or results: the paper trains on language-model activations
with millions of latents on GPUs, while this module uses float64 NumPy with tens of units and latents on one
CPU thread. Dictionary features are directions in a toy model's activity space. They are not cell types or
biological circuits.

## What it computes

**Autoencoder (`TopKSAE`).** For an input `x` (one activity vector, `d` units) and `n` latents:

```
z     = ReLU( TopK_k( W_enc (x - b_pre) + b_enc ) )      # keep the k largest pre-activations, zero the rest
x_hat = W_dec z + b_pre                                    # ||W_dec[:, j]||_2 = 1 for every latent j
```

* **TopK.** The k largest pre-activations of each row are kept and the rest are set to zero. Ties go to the
  lower index. A ReLU on the kept values makes `L0 <= k`, with equality whenever the top k are positive.
  `b_enc` is optional (`encoder_bias=True` by default): the paper's main equation leaves it out, its Appendix C.3
  reports that it does not change the converged MSE, and the authors' reference code keeps it.
* **Initialisation.**
  * Decoder columns are random unit vectors.
  * The encoder is the decoder's transpose. The two are tied only at initialisation.
  * `b_pre` is set to the geometric median of a data sample (Weiszfeld iterations).
* **Main loss.** `L = s * mean((x - x_hat)^2)`. The scale `s = 1 / Var(x)` is computed once, on the
  initialisation sample.
* **Dead latents and AuxK.**
  * A latent is *dead* once it has not fired (`z_j > 1e-3`) for `dead_after` training samples.
  * The firing statistics are updated with the current batch before the dead mask is read.
  * `e = x - x_hat` is the main residual, treated as a constant. `z_aux` keeps the top-`k_aux` dead latents by
    pre-activation, with ReLU applied, and `e_hat = W_dec z_aux`.
  * The auxiliary loss is `L_aux = mean_i ||e_i - e_hat_i||^2 / ||e_i||^2`. The paper (App. A.6) says the
    AuxK normalisation is computed "per token" because the error scale changes during training, and
    per-sample division by `||e_i||^2` is this module's reading of that. The authors' reference code does it
    differently: its `normalized_mse` divides the batch MSE of the AuxK reconstruction by the MSE of
    predicting the batch-mean residual (a per-batch, mean-centred normalisation; re-read on 2026-11-20 in
    `sparse_autoencoder/kernels.py` and `train.py`). Both are scale-free; they are not numerically identical.
  * The total loss is `L + alpha * L_aux`, with `alpha = 1/32` and `k_aux = d/2` by default. A non-finite AuxK
    term is dropped.
* **Optimiser.**
  * Adam with `beta = (0.9, 0.999)`, `eps = 6.25e-10` and a constant learning rate. Plain SGD is also available.
  * The gradient component parallel to each decoder column is projected out before the step. The decoder
    columns are renormalised to unit norm after every step.
* **Additions for bounded online use.** These are not from the paper: global gradient-norm clipping (default
  1.0), parameter clipping (`max_param`), counter caps, and input validation (finite values, `|x| <= 1e6`).
* **Metrics** (`evaluate`):
  * Fraction of variance unexplained, `FVU = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2`. This is the paper's
    "normalized MSE".
  * MSE and mean `L0`.
  * The fraction of features that never fired.
  * Per-feature firing rate, and a usage histogram over the bins `never, (0,1e-3], (1e-3,1e-2], (1e-2,1e-1], (1e-1,1]`.
  * The training dead fraction.

Training is fully deterministic. Initialisation uses `default_rng(seed + offset)`, and each epoch's shuffle uses
a generator keyed by `(seed + offset, epoch)`. The object holds no generator state, so `export_state` /
`import_state` round-trip exactly and `fit(2); fit(1)` equals `fit(3)`. `import_state` validates everything
before it changes anything: schema, configuration, shapes, finiteness, unit-norm decoder, integer counters and
non-negative Adam second moments.

**Live observer (`SparseDictionaryObserver(n_units, *, seed=7, history=256, features=64, k=8, ...)`).**
`observe(x)` follows these steps:

1. It copies the activity vector into a fixed ring buffer of `history` rows. The caller's array is never aliased
   or written.
2. Once `min(history, 32)` rows exist, it initialises `b_pre` from their geometric median.
3. Every `train_every` observations it takes one optimiser step on the new vector plus `batch_size - 1` replayed
   buffer rows. The replay positions come from a generator keyed by `(seed + offset, observation count)`.
4. Its statistics are prequential: they are measured on each new vector before the update that uses it.
   * `fvu_ema` is the exponential moving average (EMA) of the per-element squared error, divided by the EMA
     variance of the units.
   * `l0_ema` is the EMA of the number of active features.
   * A per-feature usage EMA feeds the usage figures in the summary.
5. The default `dead_after` is `batch_size * history` samples, about one buffer's worth of observations.

`summary()` is JSON-safe, about 3 KB, and contains no wall-clock. It returns:

* `fvu_ema`, `mse_ema`, `l0_ema` and `dead_fraction`;
* `steps`, the optimiser updates;
* `observed` and `history_fill`;
* `top_features`, up to 8 features by usage EMA, each with its 5 units of largest absolute decoder weight;
* `usage_histogram`;
* `decoder_norm_max_error`;
* `evidence_tier`.

`export_state` / `import_state` validate fully and continue bit-identically. `reset()` returns the observer to
its constructed state. `from_spec(n_units, FrontierSpecV1)` maps `seed`, `history`, `sae_features` and `sae_k`
to the observer's arguments.

## Lab experiment (`run_experiment`, `EXPERIMENTS = {'sparse_dictionary': run_experiment}`)

Per training seed (default `seeds=(0, 1, 2)`):

1. **Train.** A NEUROWEAVE engine (`make_spec`, default `delayed_recall`, 48 neurons, `clustered`, no extra
   mechanisms) and the lab `ReadoutHead` are trained prequentially on `make_stream(seed, steps)` with
   `run_task_phase(..., learn=True)`.
2. **Record (training data only).** A frozen clone replays the training stream and records the first `n0` unit
   outputs and the head's features. With `readout='refit'` (default), the head is refit on those features by
   L2-regularised logistic regression (damped Newton) to give a competent frozen probe. `readout='online'`
   keeps the online head. Held-out streams are never used for fitting.
3. **Dictionary.** A `TopKSAE` is fit to the recorded outputs (default 64 features, `k=8`, `dead_after` = one
   pass over the training stream).
4. **Selection (training data only).** Only "alive" features are eligible: those that fired on at least
   `min_fire_rate` of the rows. The `n_ablate` alive features with the largest `|Pearson r|` between activation
   and the training target are *targeted*. `random_draws` uniform draws of the same number of alive features
   form the random arm. A draw may overlap the targeted set, and the receipt reports the overlap.
5. **Causal ablation (held-out, frozen clones).** For every held-out stream, three arms run on fresh frozen
   clones of the trained engine: **none**, **targeted** and **random**. After every engine step the chosen
   features' decoder contributions `sum_j z_j W_dec[:, j]` are subtracted from the clone's
   `engine.brain.cells.output` (clipped to the cells' `[-1, 1]` range) before the readout reads it. This is the
   place where `run_task_phase`'s `silence` zeroes units, so the change also reaches the next step's recurrent
   input. With no features, this phase reproduces `run_task_phase(..., learn=False)` number for number (tested).
   The original engine is never stepped, and its state digest is checked before and after.
6. **Verdict.** The verdict uses `harness.compare` (paired 95 % t interval over seeds) on held-out log-loss of
   targeted vs random ablation, and the receipt also reports Cohen's d_z.
   **`improved` here means only that the targeted features matter more to the frozen readout than random
   ones**: ablating them raised held-out log-loss more, and the interval excludes zero. `worse` means they
   matter less. It does not mean that any model got better. The receipt also carries these supporting
   comparisons:
   * targeted vs random accuracy;
   * targeted vs none log-loss;
   * random vs none log-loss.

The receipt also reports:

* dictionary train and held-out FVU, L0, dead and never-fired fractions, and the usage histogram;
* the selected features with their correlation and fire counts;
* the energy that the targeted and random sets carry on the training data, and the per-step norm actually
  removed;
* budgets, estimated operation counts and measured wall-clock.

Ablation can remove different amounts of activity in the two arms. Before reading a difference as
feature-specific, compare `selection.targeted_train_energy`, `selection.random_train_energy` and
`heldout.*.removed_norm`.

## Connection to Sonapse

* **Live (read-only).** `sparse_dictionary` is in `spec.OBSERVER_MODULE_IDS`. A runtime hook that receives
  `FrontierSpecV1(enabled=True, modules=('sparse_dictionary', ...))` builds the observer with
  `SparseDictionaryObserver.from_spec(n_units, spec)` and feeds it copies of the observed unit activity. The
  observer never writes into the brain, graft bus, actor or any shared RNG. Wiring that hook into the Sonapse
  runtime is outside this module.
* **Lab.** `EXPERIMENTS['sparse_dictionary']` is the frontier-lab entry. It uses the NEUROWEAVE lab helpers
  (`make_spec`, `ReadoutHead`, `run_task_phase`, `clone_engine`) and `synapse.frontier.harness` (`base_receipt`,
  `compare`, `seal`).

## Not implemented

* **From the paper.**
  * Weight EMA (paper A.5).
  * Encoder-magnitude matching at initialisation. The paper reports that it has no effect or a weak negative
    effect.
  * Rounding `k_aux` to a power of two.
  * Multi-TopK.
  * Sharded or fused kernels.
  * Scaling laws.
  * The paper's feature-quality metrics: probe recovery, explainability and downstream-effect sparsity.
* **In this module.**
  * Energy-matched random controls.
  * Any claim about biological interpretation.

## Verified references

* Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike,
  Jeffrey Wu. **Scaling and evaluating sparse autoencoders.** arXiv:2406.04093 (submitted 6 June 2024);
  International Conference on Learning Representations (ICLR) 2025. <https://arxiv.org/abs/2406.04093> ·
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/42ef3308c230942d223c411adf182c88-Abstract-Conference.html>.
  The following were checked against the paper text on 2026-11-17:
  * TopK, eq. 2 and footnote 5 (ReLU in code);
  * tied initialisation, Sec. 2.4 and A.1;
  * geometric-median `b_pre` and the unit-norm decoder after each step, A.1;
  * AuxK with "typically 1/32", zeroed when NaN, A.2;
  * Adam settings and the parallel-gradient projection, A.3;
  * normalisation, A.6;
  * "power of two close to d_model/2", B.1;
  * `b_enc`, C.3;
  * normalized MSE, Sec. 2.1.

  The firing threshold `1e-3` and the `latent_bias` follow the authors' reference code,
  <https://github.com/openai/sparse_autoencoder>, read the same day.
* **Auxiliary dead-latent loss lineage.** Adam Jermyn and Adly Templeton. **Ghost Grads: An improvement on
  resampling.** In *Circuits Updates – January 2024*, Transformer Circuits Thread, January 2024.
  <https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning-resampling>. Gao et al. describe
  AuxK as similar to ghost grads. This module implements AuxK, not ghost grads.
* **TopK activation origin.** Alireza Makhzani and Brendan Frey. **k-Sparse Autoencoders.** arXiv:1312.5663
  (December 2013, revised March 2014); listed by DBLP under ICLR 2014. <https://arxiv.org/abs/1312.5663>.

## Measured results

All numbers were measured on 2026-11-17 on Windows 11 ARM64 (Qualcomm ARMv8 CPU), Python 3.12.10, NumPy 2.4.3, with
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Other agents shared the CPU, so wall-clock is
indicative only.

**Quick mode** (`run_experiment(quick=True)`): seeds `(0, 1, 2)`, held-out seed `(100,)`, `delayed_recall`, 24
neurons, `clustered`, no mechanisms, 120 training / 60 held-out steps, 32 features, `k=4`, `n_ablate=3`,
`random_draws=2`, 40 dictionary epochs (batch 16, lr 3e-3), refit readout (L2 1e-2).

| | value |
|---|---|
| verdict | **no_detectable_difference** |
| targeted − random held-out log-loss | +0.0139, 95 % CI [−0.0579, 0.0857], d_z 0.48 |
| targeted − none held-out log-loss | +0.0236, 95 % CI [−0.0367, 0.0839] |
| random − none held-out log-loss | +0.0097, 95 % CI [−0.0238, 0.0433] |
| held-out log-loss (none / targeted / random) | 0.638 / 0.661 / 0.647; constant predictor 0.691 |
| held-out accuracy (none / targeted / random) | 0.628 / 0.567 / 0.625; chance 0.533 |
| dictionary FVU, train / held-out (per seed) | 0.047 / 0.126, 0.032 / 0.067, 0.032 / 0.183 |
| dead fraction at end of training (per seed) | 0.34, 0.31, 0.38 |
| original engines unchanged | true |
| digest (prefix) | `0562be1283cf59b4` (identical across two processes) |
| measured wall-clock | 2.70 s; 6.39 s in an earlier run under heavier CPU load |

**Default size** (`run_experiment()`): seeds `(0, 1, 2)`, held-out seeds `(100, 101)`, 48 neurons, 300 / 150
steps, 64 features, `k=8`, `n_ablate=4`, `random_draws=4`, 60 epochs (batch 32).

| | value |
|---|---|
| verdict | **no_detectable_difference** |
| targeted − random held-out log-loss | +0.0053, 95 % CI [−0.0373, 0.0478] |
| targeted − none held-out log-loss | +0.0350, 95 % CI [0.0085, 0.0616] (interval excludes zero) |
| random − none held-out log-loss | +0.0298, 95 % CI [−0.0367, 0.0962] |
| readout skill over the constant predictor (held-out, none arm) | 0.127 nats |
| dictionary FVU, train (per seed) | 0.013, 0.016, 0.014 |
| digest (prefix) | `93a7ef790ae2b8a9` |
| measured wall-clock | 28.9 s under concurrent load |

**Observer cost** at the default size: 256 units, 64 features, `k=8`, batch 8, train every observation.

* About 1.4 ms per `observe` and 0.6 ms per `summary()`.
* `summary()` JSON is 2.9 KB.
* `export_state()` JSON is 3.65 MB. That covers the ring buffer, parameters and Adam moments, which an exact
  continuation needs.
* Measured over 500 observations on random dense activity. FVU EMA is 0.91 there, because that data has no
  sparse structure.

Reading: in these small engines, removing the dictionary features most correlated with the delayed-recall target
does raise held-out log-loss at default size. The paired interval does not show that it raises log-loss more
than removing the same number of random alive features. The features selected by correlation are only weakly
predictive (|r| ≈ 0.14–0.26), and the readout has little skill to lose (0.05–0.13 nats over a constant).

Unit-level checks in `tests/test_frontier_sae.py` (21 tests, about 7 s single-threaded):

* TopK matches a brute-force sort exactly.
* Gradients, including AuxK, match central finite differences to within 1e-6.
* A planted dictionary is recovered. With 16 unit-norm atoms in 12 dimensions, 2 active per sample, 800 samples
  and 30 epochs, FVU is below 0.1 and at least 14 of 16 atoms have a decoder column with cosine above 0.9.
* Decoder norms stay within 1e-12 of 1.
* AuxK revives latents whose encoder rows were shrunk 50×. With AuxK, at least 2 of 4 fire on more than 5 % of
  samples and FVU is lower. Without AuxK they stay dead.
* Observer FVU EMA falls from above 0.5 to below 0.3 on planted data.
* Exact state round-trips with full validation.
* Ablation on clones only, with the original engine's digest unchanged.
* The quick receipt is sealed and deterministic.
