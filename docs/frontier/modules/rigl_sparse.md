# `rigl_sparse`: dynamic sparse training (RigL)

Module file: `synapse/frontier/rigl.py` · tests: `tests/test_frontier_optim.py` ·
experiment id: `rigl_sparse` · kind: `tool` (a training-loop scheduler; no engine extension, no live observer) ·
evidence tier: `synthetic_abstraction`.

The module keeps the `nn.Linear` layers of a PyTorch model at a fixed number of active weights and, every
`delta_t` iterations until `t_end`, moves a cosine-decayed fraction of them: the smallest-magnitude active weights
are dropped and the same number of inactive weights with the largest gradient magnitude are grown, starting at
zero. It is *inspired by* **RigL** (Evci et al., ICML 2020) and by the `AuthenticRigL` class of the sibling
project `kai9987kai/sonapse-evo`. It does not reproduce the paper's ImageNet or CIFAR results, and it stores
weights densely with binary masks, so it saves no time or memory.

## What it computes

**Layer budgets.** For managed layers with weight shapes `(n_out, n_in)` and a target overall sparsity `S`:

* `'uniform'`: every layer has density `1 - S`.
* `'erk'` (Erdős–Rényi; for linear layers ERK and ER coincide): density `eps * (n_in + n_out) / (n_in * n_out)`,
  with one scale `eps` chosen so the total number of active weights equals the uniform budget. A layer whose
  density would exceed 1 is made dense and `eps` is recomputed for the others, as in the reference code.

A layer with `N` weights and density `d` keeps `a = N - floor((1 - d) N)` active weights (at least 1). The initial
masks are random (`numpy.random.default_rng(seed + offset)`), applied to the weights at construction, and every
inactive weight is exactly `0.0` from then on.

**Schedule.** Training iterations are counted `t = 1, 2, …`. Iteration `t` is a *connectivity update* when

```
t mod delta_t == 0   and   t < t_end            (RigL Algorithm 1)
f(t) = alpha / 2 * (1 + cos(pi * t / t_end))     (cosine update fraction of Evci et al.)
k    = floor(f(t) * a)                           (per layer; a = active weights of the layer)
```

**Update (per sparse layer).**

1. *Drop* the `k` active weights with the smallest `|w|` (ties: lowest flat index).
2. *Grow* the `k` weights with the largest `|dL/dw|` among all weights that were not kept, i.e. the inactive ones
   plus the ones just dropped (ties: lowest flat index). The gradient is the dense gradient of the current batch,
   which autograd gives at every position because inactive weights are zeros of a dense tensor.
3. Newly activated weights are set to `0.0`, so the network output does not change. A weight that is dropped and
   grown back in the same update keeps its value (the reference code's default, `reinit_when_same=False`).
4. Every per-element optimiser state tensor of the layer (SGD momentum, Adam moments) is zeroed at inactive and
   newly activated positions.
5. No optimiser step is taken on an update iteration (Algorithm 1). On every other iteration the gradients of
   inactive weights are zeroed before `optimizer.step()` and the masks are re-applied after it, so inactive
   weights stay exactly zero even with weight decay or momentum.

The number of active weights per layer never changes. Biases stay dense.

## Public API

```python
MODULE_ID = 'rigl_sparse'
DISTRIBUTIONS = ('uniform', 'erk')
STATE_SCHEMA = 'RigLSchedulerStateV1'
ARMS = ('dense', 'static_sparse', 'rigl')
EXPERIMENTS = {'rigl_sparse': run_experiment}

cosine_drop_fraction(t, alpha, t_end) -> float
layer_densities(shapes, sparsity, distribution='erk') -> list[float]
layer_active_counts(shapes, sparsity, distribution='erk') -> list[int]

RigLScheduler(model, sparsity, delta_t=100, alpha=0.3, t_end=1000, distribution='erk', *,
              layers=None, seed=0, dynamic=True, history=256)
    .step(optimizer) -> bool              # call once per iteration after loss.backward(); False = connectivity update
    .update_connections(fraction, optimizer=None) -> dict
    .apply_masks()  .mask_gradients()
    .is_update_step(t) -> bool  .drop_fraction(t) -> float
    .active_counts() -> list[int]  .parameter_budget() -> dict  .summary() -> dict
    .masks (copies)  .history (copies; at most `history` records)
    .state_dict() -> dict  .load_state_dict(state)

run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict      # sealed receipt
```

`layers` selects which `nn.Linear` modules are managed (default: all, in `model.modules()` order; pass a subset to
keep, for example, the first layer dense). `dynamic=False` keeps the initial masks for the whole run (static sparse
training with identical masks). Inputs are validated before anything changes: `update_connections` refuses
missing, sparse or non-finite gradients without touching masks, weights or optimiser state, and
`load_state_dict` checks the schema, the configuration, the counters and every mask's dtype, shape and active
count before loading. The scheduler state holds the masks, counters and history, not the optimiser state.

## Benchmark (`run_experiment`)

* **Task (`sparse_teacher`).** Gaussian inputs (`n_inputs = 64`), of which only a seeded subset of
  `n_informative = 8` drives the label: the label is the argmax of a fixed random tanh teacher on those inputs
  (4 classes, logits centred on the training split). The other 56 inputs are distractors, so which connections
  exist matters.
* **Model.** ReLU MLP `64-hidden-hidden-4` with PyTorch's default `nn.Linear` initialisation drawn from a private
  `torch.Generator` (the global RNG is untouched).
* **Arms.** `dense`; `static_sparse` (the scheduler with `dynamic=False`); `rigl`. Per seed all three share the
  initial weights, the minibatch sequence, `torch.optim.SGD(lr, momentum, weight_decay)` and the number of
  training iterations; `static_sparse` and `rigl` also share the initial masks (checked in the receipt).
* **Verdict.** `harness.compare` (paired 95 % t interval over seeds) of held-out test accuracy, `rigl` minus
  `static_sparse`: the two arms have the same parameter budget, so a win cannot come from capacity. `rigl` vs
  `dense` and `static_sparse` vs `dense` are reported as context. If a sparse arm diverges, the verdict basis
  becomes `training_diverged`.
* **Receipt.** `config` (every size and hyper-parameter, `t_end`), per-arm `per_seed` results (accuracies, final
  loss, optimiser steps, mask digests, update trace, connections activated, the share of first-layer
  connections that read an informative input), `budget` (active and dense parameters, per-layer densities, shared
  masks, constant budget, zero inactive weights), `estimated_cost` (analytic FLOPs for ideal sparse kernels,
  apart from `measured_wallclock_s`), `comparisons`, `verdict`, `verdict_basis` and `notes`.

| Setting | quick | default |
|---|---|---|
| `hidden`, train / val / test examples | 64, 1024 / 256 / 512 | 128, 4096 / 1024 / 2048 |
| iterations, batch, SGD lr / momentum / weight decay | 300, 64, 0.05 / 0.9 / 1e-4 | 2000, 64, 0.05 / 0.9 / 1e-4 |
| sparsity, distribution | 0.9, `erk` | 0.9, `erk` |
| `delta_t`, `alpha`, `t_end` | 25, 0.3, 225 (0.75 × iterations) | 50, 0.3, 1500 |

The paper uses `delta_t = 100`, `alpha = 0.3` and `T_end` at about 75 % of training; the shorter runs here use a
smaller `delta_t` so that several updates happen.

## Connection to Sonapse

A standalone lab tool (`EXPERIMENTS['rigl_sparse']`). Nothing in Sonapse or NEUROWEAVE calls it, and no default
training loop, graft schedule, API or checkpoint format changes. `sonapse-evo`'s `AuthenticRigL`
(`src/sonapse_evo/evaluation/rigl.py`, read in the local read-only clone) applies the same cosine drop/regrow idea
to a single recurrent edge vector and regrows weights at `±0.05` with a random sign; this module follows the
paper instead (per-layer masks, ERK budgets, zero-initialised growth, no optimiser step on update iterations).

## Not implemented, and differences from the paper

* **No sparse kernels.** Masks over dense tensors: no measured speed-up or memory saving is claimed.
  `estimated_cost` gives analytic FLOPs for ideal sparse kernels.
* **`nn.Linear` only.** No convolutions, so ERK equals ER here.
* **Tie-breaking.** Ties go to the lowest flat index; the reference code adds small Gaussian noise
  (std `1e-5`) to the drop scores.
* **Update condition.** Algorithm 1's `t mod delta_t == 0 and t < t_end`. The reference code's TensorFlow
  scheduler also allows an update at `t = t_end` and uses a `begin_step`.
* **Optimiser state.** All per-element state at inactive and newly grown positions is zeroed (the reference
  resets the momentum of new connections; inactive positions have zero gradient here anyway).
* **First layer under `'uniform'`.** The paper keeps the first layer dense for the uniform distribution; here
  that is the caller's choice (`layers=`). The benchmark uses ERK and sparsifies every layer.
* **Not attempted.** SET (random growth) and SNFS (momentum growth), the ImageNet/ResNet-50, CIFAR-10 and
  character-level experiments, and the paper's longer training schedules.

## Verified references

* Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, Erich Elsen. **Rigging the Lottery: Making All
  Tickets Winners.** *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR
  vol. 119, pp. 2943–2952, July 2020. <https://proceedings.mlr.press/v119/evci20a.html> · arXiv:1911.11134,
  <https://arxiv.org/abs/1911.11134>. Checked on 2026-11-20 against the PMLR page (title, authors, volume, pages,
  date) and the arXiv text: Algorithm 1 (update condition, `k = f_decay(t) (1 - s) N`, drop by `ArgTopK(-|θ|)`,
  grow by gradient magnitude among the non-kept connections, plain gradient step otherwise), the cosine
  `f_decay(t; α, T_end) = α/2 (1 + cos(tπ / T_end))`, the defaults `ΔT = 100`, `α = 0.3`, `T_end` ≈ 75 % of
  training, zero-initialised growth, the ER/ERK distributions, and the dense first layer under the uniform
  distribution.
* Reference code: <https://github.com/google-research/rigl>, `rigl/sparse_optimizers_base.py`, read on
  2026-11-20: `n_prune` is a truncating cast of `n_ones * drop_fraction`; kept connections get the lowest grow
  score; `reinit_when_same=False` by default, with new connections defined as grown and previously inactive;
  `grow_init='zeros'`; momentum of new connections reset to zero; cosine decay over `end_step - begin_step`.
* Portfolio provenance: `kai9987kai/sonapse-evo`, `src/sonapse_evo/evaluation/rigl.py` (`AuthenticRigL`).

## Measured results

Measured on **2026-11-20**, Windows 11 on ARM64, Python 3.12.10, torch 2.11 CPU, NumPy 2.4,
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` (the experiment also runs torch on one thread). The
machine was shared with other jobs, so wall-clock is indicative only. Receipts are deterministic: two calls in
one process gave the same digest, and the quick digest was identical in two separate processes run about twelve
hours apart that day.

**Quick** (`run_experiment(quick=True)`, seeds 0, 1, 2; digest `906545bf0ed9d08c`; 6.7–7.4 s):

| Arm | Parameters (active weights + biases) | Train acc | Held-out test acc | First-layer informative share (chance 0.125) |
|---|---:|---:|---:|---:|
| `dense` | 8580 | 0.9997 | 0.6986 | 0.125 |
| `static_sparse` | 980 | 0.6849 | 0.6510 | 0.117 (unchanged) |
| `rigl` | 980 | 0.8128 | 0.7357 | 0.117 → 0.186 |

* Realised density 0.100 (848 of 8448 weights); ERK gives the per-layer densities 0.082, 0.082 and 0.695.
  RigL made 8 connectivity updates (t = 25 … 200) and 292 optimiser steps per run; it activated 710–780
  connections per run, and its first-layer share of informative inputs rose on every seed
  (0.099 → 0.158, 0.134 → 0.194, 0.119 → 0.206).
* `rigl − static_sparse`: +0.0846, 95 % CI [−0.0126, +0.1818] → **`no_detectable_difference`** (verdict).
  `rigl − dense`: +0.0371, CI [−0.0282, +0.1024]; `static_sparse − dense`: −0.0475, CI [−0.1806, +0.0856].
* Quick mode is a pipeline check with three seeds; its interval is wide.

**Default** (`run_experiment()`, seeds 0, 1, 2; digest `555ce009ccb76e7b`; 51.6 s; hidden 128, 2000 iterations,
`delta_t = 50`, `t_end = 1500`):

| Arm | Parameters (active weights + biases) | Train acc | Held-out test acc (per seed) | Mean | First-layer informative share |
|---|---:|---:|---|---:|---:|
| `dense` | 25348 | 0.9928 | 0.714, 0.767, 0.766 | 0.7492 | 0.125 |
| `static_sparse` | 2772 | 0.8805 | 0.758, 0.813, 0.818 | 0.7961 | 0.110, 0.149, 0.142 (unchanged) |
| `rigl` | 2772 | 0.9107 | 0.809, 0.795, 0.824 | 0.8091 | 0.110 → 0.165, 0.149 → 0.207, 0.142 → 0.182 |

* Realised density 0.100 (2512 of 25088 weights); ERK gives 0.105, 0.070 and 1.0 (the 128 × 4 output layer is
  dense). RigL made 29 connectivity updates (t = 50 … 1450) and 1971 optimiser steps per run and activated
  6203–6636 connections per run.
* `rigl − static_sparse`: +0.0130, 95 % CI [−0.0733, +0.0993] → **`no_detectable_difference`** (verdict).
  `rigl − dense`: +0.0599, CI [−0.0227, +0.1425]; `static_sparse − dense`: +0.0469, CI [+0.0361, +0.0577]. On this
  distractor-heavy task the dense network fits the training set almost perfectly and generalises worse than
  both sparse arms. Static random masks alone already beat it at this size, so that gap does not depend on mask
  updates, and the mask updates add no detectable accuracy over the static masks.

Unit tests with known answers (`tests/test_frontier_optim.py`): the cosine fraction at fixed points; the ERK
ratios and budget; a hand-worked drop/grow example (including a tie, a regrown-same weight that keeps its value,
zero-initialised growth and optimiser-state zeroing); constant budgets, exactly-zero inactive weights and the
update trace over a training run; static masks never moving; growth moving first-layer connections toward the
inputs that carry the signal on a planted regression (measured share 0.107 → 0.369, full-data MSE 0.0016 vs
0.0039 for static masks); refusal of bad gradients without any change; and state round-trips that continue
identically.
