# `dendritic_layer`: structured dendritic layer

Module file: `synapse/frontier/dendritic.py` · tests: `tests/test_frontier_models.py` ·
experiment id: `dendritic_layer` · kind: standalone model (no engine extension, no live observer) ·
evidence tier: `synthetic_abstraction`.

The module implements a **dendritic artificial neural network** (dANN) layer inspired by Chavlis & Poirazi (2025).
Each dendrite receives a small, fixed subset of the inputs, which gives the layer structured sparsity through a
fixed binary mask. Each dendrite applies a nonlinearity, and each soma sums only its own dendrites through
trainable cable weights. The lab experiment compares dANNs with three input-sampling rules against a dense MLP at
matched trainable parameters, on a procedurally generated glyph dataset.

This is not a reproduction. The paper's datasets (MNIST variants, CIFAR-10), sizes, efficiency scores and
sparse-ANN controls are not used. The units are abstract nodes, not models of real dendrites.

## What it computes

**Layer (`DendriticLayer`).** There are `somas` somata, each with `dendrites_per_soma` dendrites (`D` dendrites
in total). Dendrite `k` has exactly `S = synapses_per_dendrite` synapses:

```
d_k = lrelu( sum_i M_ki W_ki x_i + b_k )                 # dendrite k; M is a fixed 0/1 mask, S ones per row
s_j = lrelu( sum_{k in dendrites(j)} c_jk d_k + a_j )    # soma j sums only its own dendrites
lrelu(u) = max(u, 0.1 u)                                 # leaky ReLU, slope 0.1, as in the paper
```

* **Masking.**
  * The weights are multiplied by the mask in the forward pass, so masked weights get exactly zero gradient.
    This is the paper's Eq. 3, gradient masking.
  * `W <- W ⊙ M` is re-applied after every optimiser step (`enforce_mask`, Eq. 1).
  * The mask is a registered buffer. It never changes and travels with `state_dict()`.
* **Tree structure.** Every dendrite connects to exactly one soma. `cable_weight` therefore has shape
  `(somas, dendrites_per_soma)`; `cable_mask()` shows the equivalent dense 0/1 soma-by-dendrite matrix.
* **Initialisation** uses a `torch.Generator`. Synapses and dendritic biases are uniform in
  ±1/√S, the dendrite's own fan-in. Cable weights and somatic biases are uniform in ±1/√`dendrites_per_soma`.
* **`parameter_count()`** counts only parameters that exist: unmasked synapses, dendritic biases, cable weights
  and somatic biases. The weight tensor is stored densely, but masked entries are not counted. `macs()` gives the
  multiply-accumulates a sparse implementation would need.

**Input sampling (`sample_receptive_fields`).** The mask is fixed at construction from
`default_rng(seed + 41011)`:

* **`random`** (dANN-R): each dendrite samples `S` distinct inputs uniformly.
* **`local`** (dANN-LRF): each dendrite gets a uniformly drawn centre pixel and takes the `S` pixels nearest to it.
  It starts from the centre's 3×3 block (a random subset when `S < 9`). It then adds Chebyshev rings of radius 2,
  3, … and takes a random subset of the last ring it needs. Blocks are clipped at the image border.
* **`global`** (dANN-GRF): each soma gets a uniformly drawn centre. Its dendrites' centres are
  `dendrites_per_soma` distinct pixels nearest that centre, chosen by the same rule. Each dendrite then samples
  locally around its own centre.

The paper's Methods text says each dendrite takes 16 pixels from "the 4 × 4 neighbourhood" of its centre. The
authors' released `receptive_fields.py` instead builds the 3 × 3 block and grows it ring by ring, as described
above, and that is what this module follows. The default here is `S = 9`, a full 3 × 3 block.

In the released somatic (GRF) mode, dendrite centres are drawn without replacement from the 3 × 3 block around
the soma centre. An inverted guard there never grows the block, so a soma with more than 9 dendrites, or one at
an image corner, leaves some dendrites without inputs. The ring rule used here avoids that case.

**Networks.**

* `DendriticNet` is a `DendriticLayer` followed by a dense linear read-out, the paper's "fully connected output
  layer".
* `DenseMLP` is the control: two dense hidden layers of equal width `h` with the same leaky ReLU, then a linear
  read-out. `matched_dense_hidden` picks the `h` whose parameter count is closest to the dANN's.

The trainable parameter counts are:

```
dANN  = D·S + 2·D + somas + somas·classes + classes          (dann_parameter_count; D = somas · dendrites_per_soma)
dense = 144·h + h + h² + h + h·classes + classes            (dense_parameter_count)
```

## Benchmark (`run_experiment`, `EXPERIMENTS = {'dendritic_layer': run_experiment}`)

**Glyphs (`render_glyph`, `make_glyph_dataset`).** There are ten stroke classes, defined as polylines in the unit
square: `box, bar, plus, cross, ell, tee, zed, caret, aitch, cup`. Each image is rendered anti-aliased into
12 × 12 pixels, so the input is 144-dimensional:

* random scale 6–8.5 px;
* rotation ±0.25 rad;
* translation ±2 px;
* half stroke width 0.35–0.9 px;
* additive Gaussian pixel noise with σ = 0.2.

The pixel value is `clip(half_width + 0.5 − distance to the nearest segment, 0, 1)` plus noise. Nothing is
downloaded.

**Per seed** (default `seeds=(0, 1, 2)`):

* independent training (60 per class), validation (50 per class) and held-out (200 per class) sets, drawn from
  different generator seeds;
* a noisy copy of the held-out set with extra Gaussian noise σ = 0.5, drawn from a `torch.Generator`. This copy
  measures robustness.

**Arms.**

* `dann_random`, `dann_local` and `dann_global`: 16 somata × 8 dendrites × 9 synapses.
* `dense_mlp`: the dense control.

All arms share the data, the minibatch order (one generator seed), Adam (lr 3e-3, β = (0.9, 0.999)), batch 64
and 1000 updates. The paper uses Adam with lr 1e-3. Here 3e-3 is used because, with this update budget, lr 1e-3
left the models short of fitting their training set; see the calibration row under *Measured results*. The
parameter budget is computed **before training**. If the dense control cannot be matched within
`param_tolerance` (10 %), `run_experiment` raises `ValueError`. That happens, for example, when the dANN is
smaller than the narrowest dense MLP, which has 167 parameters.

**Verdict.**

1. The best dANN is the sampling mode with the highest mean *validation* accuracy. The held-out set plays no part
   in that choice.
2. `verdict = harness.compare(best dANN, dense MLP)` on held-out accuracy per seed, a paired 95 % t interval over
   seeds.
3. `robustness_verdict` is the same comparison on the extra-noise held-out copy.

The receipt also compares every sampling mode with the dense MLP, on clean and on noisy data, and records:

* the per-arm train, validation, held-out and noisy accuracies;
* the mask density, and a count of nonzero masked weights after training (always 0);
* the parameter budget;
* the estimated MACs;
* the measured wall-clock, which is kept out of the digest.

| option | default | quick |
|---|---|---|
| images per class train / val / held-out | 60 / 50 / 200 | 40 / 20 / 50 |
| noise / extra noise (σ) | 0.2 / 0.5 | same |
| somata × dendrites × synapses | 16 × 8 × 9 (mask density 9/144 = 0.0625) | same |
| trainable parameters dANN / dense (width) | 1,594 / 1,670 (10) | same |
| Adam updates / batch / lr | 1000 / 64 / 3e-3 | 250 / 64 / 3e-3 |

## Public API

```python
MODULE_ID = 'dendritic_layer'
EXPERIMENTS = {'dendritic_layer': run_experiment}
run_experiment(*, quick: bool = False, seeds=(0, 1, 2), **overrides) -> dict   # overrides: any DEFAULTS key

DendriticLayer(in_features, somas, dendrites_per_soma, synapses_per_dendrite, sampling='random', *,
               image_shape=None, seed=0, negative_slope=0.1)
    .forward(x) -> somatic activity (batch, somas)      .dendritic_activity(x) -> (batch, D)
    .enforce_mask()  .cable_mask()  .parameter_count()  .mask_density()  .macs()
    buffers: synaptic_mask (D, in_features), fields (D, S) input indices per dendrite
DendriticNet(in_features, classes, somas, dendrites_per_soma, synapses_per_dendrite, sampling='random', *,
             image_shape=None, seed=0)
DenseMLP(in_features, classes, hidden, *, seed=0)
sample_receptive_fields(in_features, somas, dendrites_per_soma, synapses_per_dendrite, sampling, *,
                        image_shape=None, rng) -> np.ndarray (D, S)
dann_parameter_count(somas, dendrites_per_soma, synapses_per_dendrite, classes) -> int
dense_parameter_count(in_features, classes, hidden) -> int
matched_dense_hidden(target_params, in_features, classes) -> int
render_glyph(label, rng, *, noise=0.0, shift=2.0, size=(6.0, 8.5), thickness=(0.35, 0.9), rotation=0.25)
make_glyph_dataset(n_per_class, *, seed, noise=0.2, **render_kw) -> (x float32 (n, 144), y int64 (n,))
train_classifier(model, x, y, *, updates, batch_size, lr, generator) -> dict
accuracy(model, x, y) -> float
SAMPLING_MODES = ('random', 'local', 'global'), GLYPHS, GLYPH_NAMES, GLYPH_SIDE (12), DEFAULTS, QUICK
```

Image shapes default to a square, `(sqrt(n), sqrt(n))`, when `in_features` is a perfect square, and to
`(1, n)` otherwise. Pass `image_shape` to override.

Determinism and global state:

* masks use NumPy `default_rng`, and weights, minibatch order and test noise use `torch.Generator`;
* the global torch RNG is never touched;
* `run_experiment` runs on one torch thread and restores the caller's thread count afterwards.

## Connection to Sonapse

This is a standalone lab model (contract rule 3). It is not wired into the brain, a graft or the actor, and it
changes no Sonapse default. The frontier lab registry picks it up from `EXPERIMENTS`.

Sonapse already has two different dendrite abstractions:

* `synapse/dendritic_ltc.py` is a two-compartment, basal/apical liquid time-constant sub-graft;
* `synapse/dsag/` is the dual-soma axodendritic graft.

`DendriticLayer` is a plain `nn.Module`. It could replace a dense input projection in such a graft behind an
opt-in flag. That is not done here.

## Not implemented

* **Paper controls and analyses:**
  * dANN-F (all-to-all dendrites) and the sparse-ANN (sANN) controls;
  * vANNs with unequal hidden widths;
  * the accuracy and loss efficiency scores (Eq. 4–5);
  * the sequential-learning, dropout and depth studies;
  * the entropy and selectivity analyses.
* **Datasets:** MNIST, Fashion-MNIST, Kuzushiji-MNIST, EMNIST and CIFAR-10. No data is downloaded.
* **Biological constraints:** positive or fixed cable weights, which the paper names as a more bio-realistic
  option.
* **Sparse kernels.** The masked layer runs as a dense matrix product, so it is not faster than the dense MLP.
  `estimated_cost.macs_per_example` reports what a sparse implementation would need.

## Verified references

* Spyridon Chavlis and Panayiota Poirazi. **Dendrites endow artificial neural networks with accurate, robust and
  parameter-efficient learning.** *Nature Communications* 16, 943 (2025), published 22 January 2025 (received
  4 April 2024, accepted 13 January 2025). DOI [10.1038/s41467-025-56297-9](https://doi.org/10.1038/s41467-025-56297-9);
  open access as PMC11754790 (PMID 39843414). Preprint: arXiv:2404.03708, v1 4 April 2024, v2 13 September 2024,
  <https://arxiv.org/abs/2404.03708>.
  * **Re-checked on 2026-11-20 (review)** against PMC11754790 (title, authors, volume 16, article 943,
    22 January 2025, received 4 April 2024, accepted 13 January 2025; R / LRF / GRF sampling, 16 synapses
    per dendrite, leaky ReLU 0.1, `W ⊙ M` with masked gradients) and `codes_/receptive_fields.py` (3 × 3
    block, ring expansion with a random subset of the last ring, centres with replacement, the inverted
    somatic expansion guard). `sample_receptive_fields` matches the code, with the guard fixed.
  * **Checked on 2026-11-17 against the PMC full text and the arXiv preprint:**
    * Eq. 1, `W ← W ⊙ M`, and Eq. 3, gradient masking;
    * one soma per dendrite, and a fully connected output layer;
    * the three restricted sampling strategies R, LRF and GRF, and 16 inputs per dendrite;
    * leaky ReLU with negative slope 0.1 on all hidden nodes;
    * Adam with lr 0.001, betas 0.9 and 0.999, minibatch 128;
    * a 90 / 10 train / validation split.
  * **Checked in the authors' code** the same day: <https://github.com/Poirazi-Lab/dendritic_anns>,
    `codes_/receptive_fields.py`.
    * `allocate_synapses` uses the 3 × 3 base block, random subsampling, and ring expansion with a random subset
      of the last ring.
    * Centres are drawn uniformly with replacement.
    * In somatic mode, dendrite centres come from the 3 × 3 block without replacement, behind the inverted
      expansion guard noted above.

## Measured results

Measured on **2026-11-20** (`quick=True`, `seeds=(0, 1, 2)`, single CPU thread `OPENBLAS_NUM_THREADS=1`, digest `beb329f8b497c3e0`, wall-clock `5.03 s`):

| Arm | Train acc | Val acc | Held-out acc | Held-out noisy acc | Held-out Δ vs `dense_mlp` (95 % CI) |
|---|---:|---:|---:|---:|---|
| `dann_random` | 0.9067 | 0.6650 | 0.6560 | 0.4007 | −0.0127 (`[-0.0844, +0.0591]`, `no_detectable_difference`) |
| `dann_local` (best val) | 0.8625 | 0.6667 | 0.6780 | 0.4440 | +0.0093 (`[-0.0424, +0.0611]`, `no_detectable_difference`) |
| `dann_global` | 0.8158 | 0.6400 | 0.6433 | 0.4227 | −0.0253 (`[-0.1536, +0.1029]`, `no_detectable_difference`) |
| `dense_mlp` (baseline) | 0.9758 | 0.6633 | 0.6687 | 0.4080 | — |

* **Headline verdict (`best_dann(dann_local)_vs_dense_mlp`):** `no_detectable_difference` (mean +0.0093, 95 % CI `[-0.0424, +0.0611]`).
* **Robustness verdict (`best_dann_vs_dense_noisy`):** `no_detectable_difference` (mean +0.0360, 95 % CI `[-0.0372, +0.1092]`).
