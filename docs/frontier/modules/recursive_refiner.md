# `recursive_refiner`: recursive latent refinement

Module file: `synapse/frontier/recursive_refiner.py` · tests: `tests/test_frontier_models.py` ·
experiment id: `recursive_refiner` · kind: standalone model (no engine extension, no live observer) ·
evidence tier: `synthetic_abstraction`.

The module implements a tiny recursive model in the style of the **Tiny Recursive Model** (TRM;
Jolicoeur-Martineau, 2025), which simplifies the **Hierarchical Reasoning Model** (HRM; Wang et al., 2025).
One small network repeatedly refines a latent state `z` and an answer state `y`. It is trained with deep
supervision: one loss and one optimiser step per supervision step, with the state detached in between. A halting
head is trained to predict whether the current answer is already correct. The benchmark is 4×4 Sudoku completion
on grids the model never saw during training.

The module is *inspired by* these papers and does not reproduce them. The papers use 9×9 Sudoku-Extreme, mazes
and ARC-AGI, 5–27 M parameters, EMA weights, heavy data augmentation and GPUs. This module uses 4×4 Sudoku,
about 1.8–3.5 k parameters, no EMA, no augmentation and one CPU thread.

## What it computes

**Model (`TinyRecursiveModel`).** States have shape `tokens × latent_dim`; for Sudoku there is one token per cell
(16 tokens). Each supervision step embeds the question once, `e = E x`, where `x` is the per-cell one-hot code
over {blank, 1, 2, 3, 4}. The single shared network is a residual mixer block:

```
f(s) = N( t + W2 relu(W1 t + b1) + b2 )        # channel MLP, hidden width `hidden`
t    = N( s + Mix(s) )                         # token-mixing MLP across the 16 positions (width `token_hidden`)
N    = layer norm without learned scale or shift (keeps recursed states bounded)
```

`Mix` is skipped when `tokens == 1` (a plain vector model). TRM uses an MLP over the sequence instead of
self-attention for Sudoku, with RMSNorm and SwiGLU in 2-layer blocks. Here one block with ReLU and an affine-free
layer norm is used, which is cheaper on CPU.

One **latent recursion**:

```
repeat n times:   z <- f(z + y + e)            # latent reasoning: sees the question
once:             y <- f(y + z)                # answer refinement: no question
```

The same `f` performs both updates. Whether `e` is in the sum is what tells the network which update it is doing,
as in TRM.

One **supervision step** (TRM's "deep recursion"):

1. `T - 1` latent recursions without gradient (`recursions = T`), then one latent recursion with gradient.
2. Read-outs from `y`:
   * answer logits `O y`, 4 digits per cell;
   * halting logit `q = w · mean_cells(y) + b`, with `P(halt) = sigmoid(q)`.
3. `(y, z)` are detached before the next supervision step.

The initial states `y_init` and `z_init` are fixed truncated-normal vectors (std 1, cut at ±2), shared by all
cells and stored as non-trainable buffers. The halting head starts at `w = 0, b = -5`, so early training rarely
halts. Both choices follow the TRM code. `net_calls` counts applications of `f`; one supervision step is
`T (n + 1)` applications.

**Training (`train_recursive`): deep supervision with ACT-style halting.**

* A batch of `batch_size` slots is carried across optimiser updates. Each update runs one supervision step for
  every slot.
* Each update takes one AdamW step (β = (0.9, 0.95), weight decay 0.01, gradient-norm clip 1.0) on
  `L = CE(logits, solution) + BCE(sigmoid(q), 1[all 16 cells correct])`. These are the two loss terms of TRM's
  Figure 3 pseudocode.
* A slot halts when `q > 0` (once its minimum step count is reached) or after `N_sup = supervision_steps` steps.
  It is then refilled with the next example of a deterministic shuffled stream, and its `(y, z)` are reset to the
  initial states. There is no Q-continue head, as in TRM.
* Exploration: with probability `halt_exploration` (0.1) a new example gets a minimum of 2..`N_sup` steps. This
  is HRM's `M_min` rule. Here it is drawn once per example, as the HRM paper describes it; the reference code
  redraws it at every step.

**Evaluation (`evaluate_recursive`).** All `N_sup` supervision steps are run, as TRM does at test time, and the
headline metrics use the last step:

* `exact_solve`: fraction of puzzles with all 16 cells right;
* `cell_accuracy`: fraction of *blank* cells right;
* `per_step_exact_solve`: exact-solve after each supervision step;
* `adaptive_*`: the answer taken at the first step with `P(halt) > 0.5` (or the last step), with the mean number
  of steps used;
* `halting_brier` and `halting_decision_accuracy`: calibration of `P(halt)` against per-step correctness, and
  `halt_prob_range`.

## Benchmark (`run_experiment`, `EXPERIMENTS = {'recursive_refiner': run_experiment}`)

Per seed (default `seeds=(0, 1, 2)`):

1. **Grids.** `enumerate_sudoku4()` returns all 288 valid 4×4 grids, found by backtracking. The tests check the
   count, validity, distinctness and closure under digit relabelling.
2. **Disjoint split by grid.** The grids are permuted by `default_rng(seed + 31001)` and split into 160 training,
   32 validation and 96 held-out grids. The receipt records the overlap between splits, which is 0 for every seed.
3. **Puzzles (`make_puzzles`).** Pick a grid of the split and draw a target of 8–12 blanks. Blank cells in random
   order, skipping any removal that would admit a second solution. The check is an exact count of the grids
   consistent with the givens, using bitsets over the 288 grids. Every puzzle therefore has exactly one solution.
   The default sizes are 4096 training, 512 validation and 1024 held-out puzzles; the first 512 training puzzles
   are re-scored as "train".
4. **Encoding.** One-hot per cell over {blank, 1, 2, 3, 4} (80 inputs); the targets are the 16 digits.

**Arms.** All arms share the data, the example stream (`torch.Generator(seed + 31301)`) and the number of AdamW
updates.

| arm | model | training |
|---|---|---|
| `trm` | `TinyRecursiveModel` | deep supervision with halting (`train_recursive`) |
| `unrolled` | the same network and initial weights, run to the same inference depth (`N_sup · T · (n + 1)` applications) | one loss at the end, full backpropagation through the whole unroll (`train_unrolled`); no deep supervision |
| `shallow` | the same network and initial weights with one latent recursion (`n + 1` applications) | one loss, full backpropagation |
| `mlp` | `SinglePassMLP`: flat 2-hidden-layer ReLU MLP on the 80 inputs | cross-entropy, single pass |

The MLP width is chosen by `matched_mlp_hidden` so that its trainable parameters are closest to `trm`'s. All
budgets are computed **before training**. If any arm is more than `param_tolerance` (10 %) away from `trm`,
`run_experiment` raises `ValueError`.

**Verdict.** The best baseline is the one with the highest mean *validation* exact-solve; held-out grids play no
part in that choice. `verdict = harness.compare(trm, best baseline)` on held-out exact-solve per seed, a paired
95 % t interval over seeds. The receipt also compares `trm` with each baseline separately, and with the best
baseline on cell accuracy:

* `trm_vs_unrolled` isolates deep supervision and the one-step gradient at equal depth;
* `trm_vs_shallow` isolates recursion depth;
* `trm_vs_mlp` mixes recursion with per-cell weight sharing.

**Receipt fields.**

* `config`, including `quick` and every option;
* `data`: overlap between splits, mean blanks, uniqueness note;
* `parameter_budget`: per-arm counts, MLP width, relative gap, tolerance;
* `arms`: per-seed train / validation / held-out metrics, training logs and summaries;
* `best_baseline`, `comparisons`, `verdict`, `verdict_basis`;
* `estimated_cost`: inference MACs per puzzle, network applications per puzzle, backpropagated applications per
  update, examples drawn by `trm`;
* `measured_wallclock_s`, which is kept out of the digest;
* `notes`.

| option | default | quick |
|---|---|---|
| grids train / val / held-out | 160 / 32 / 96 | same |
| puzzles train / val / held-out / train-eval | 4096 / 512 / 1024 / 512 | 1024 / 128 / 256 / 128 |
| blanks per puzzle | 8–12 | same |
| `latent_dim` / `hidden` / `token_hidden` | 24 / 48 / 24 | 16 / 32 / 16 |
| `n_latent_steps` (n) / `recursions` (T) / `supervision_steps` (N_sup) | 2 / 2 / 4 | 1 / 2 / 2 |
| applications of `f` per puzzle (trm, unrolled) | 24 | 8 |
| AdamW updates / batch / lr | 600 / 32 / 2e-3 | 100 / 64 / 4e-3 |
| trainable parameters trm / mlp (MLP width) | 3,453 / 3,384 (20) | 1,797 / 1,791 (11) |

## Public API

```python
MODULE_ID = 'recursive_refiner'
EXPERIMENTS = {'recursive_refiner': run_experiment}
run_experiment(*, quick: bool = False, seeds=(0, 1, 2), **overrides) -> dict   # overrides: any DEFAULTS key

TinyRecursiveModel(input_dim, answer_dim, latent_dim, hidden, n_latent_steps, supervision_steps, *,
                   recursions=2, tokens=1, token_hidden=None, seed=0)
    .net(s) / .embed(x) / .initial_state(batch) / .latent_recursion(e, y, z) / .read_out(y)
    .supervision_step(e, y, z) -> (y_detached, z_detached, logits, q)
    .forward(x, *, steps=None) -> {'logits', 'step_logits', 'halt_prob'}
    .unrolled(x) -> (logits, q)          # same depth, gradient through every recursion
    .halt_probability(y), .calls_per_supervision_step(), .parameter_count(), .inference_macs(), .net_calls
SinglePassMLP(input_dim, answer_dim, hidden, *, depth=2, seed=0)
matched_mlp_hidden(target_params, input_dim, answer_dim, depth=2) -> int

enumerate_sudoku4() -> np.ndarray (288, 16)       is_valid_sudoku4(grid) -> bool
is_consistent_puzzle(puzzle) -> bool               solution_count(puzzle) -> int
make_puzzles(grids, count, *, rng, min_blanks, max_blanks, max_attempts=64) -> (puzzles, solutions)
encode_puzzles(puzzles) -> torch.Tensor (n, 80)
train_recursive / train_unrolled / train_single_pass(model, x, target, cfg, generator) -> dict
evaluate_recursive / evaluate_single_pass(model, x, target, puzzles) -> dict
ARMS, GRID_COUNT (288), CELLS, DIGITS, INPUT_DIM (80), ANSWER_DIM (64), UNITS, DEFAULTS, QUICK
```

Determinism and global state:

* all initialisation uses `torch.Generator(seed + offset)` and all data uses `numpy.random.default_rng(seed + offset)`;
* the global torch RNG is never used;
* `run_experiment` runs on one torch thread and restores the caller's thread count afterwards.

The tests check that the global RNG state and the thread count are unchanged after a run.

## Connection to Sonapse

This is a standalone lab model (contract rule 3). It does not attach to a NEUROWEAVE engine, has no live
observer, and changes no Sonapse default. The frontier lab registry picks it up from `EXPERIMENTS`.

A possible follow-up, not implemented: use `TinyRecursiveModel` as a refinement readout over NEUROWEAVE
features, where `tokens=1` gives the plain-vector form.

## Not implemented

* **From the TRM paper and code:**
  * 2-layer SwiGLU / RMSNorm blocks, self-attention and rotary embeddings;
  * EMA of the weights (0.999);
  * the 1000-fold data augmentation;
  * the puzzle-identifier embedding token (TRM's halting head reads that first position; here it reads the
    mean over cells);
  * the 0.5 weight on the halting loss (here 1, as in the paper's pseudocode);
  * the stable-max cross-entropy option;
  * TRM's sizes (T = 3, n = 6, N_sup = 16, hidden 512).
* **From the HRM paper:**
  * two separate networks at two frequencies;
  * the Q-continue head and its Q-learning target;
  * the fixed-point / one-step-gradient justification. TRM's recursion is used instead.
* **Datasets:** 9×9 Sudoku-Extreme, Maze-Hard and ARC-AGI. 4×4 Sudoku is far easier, so the numbers here say
  nothing about those benchmarks.
* **A compute-matched protocol.** All arms get equal optimiser updates. `unrolled` therefore backpropagates
  through `N_sup` times more network applications per update than `trm`, while `trm` draws fewer distinct
  examples per update (`trm_examples_drawn` in the receipt).

## Verified references

Each was checked on 2026-11-17 against the arXiv abstract page and the paper's HTML text. TRM was also checked
against the authors' code. On 2026-11-20 (review) the TRM arXiv listing was re-checked (single author, v1 of
6 October 2025 is still the only version) and the recursion, deep-supervision loop and halting code paths of
this module were re-read against the Figure 3 pseudocode below.

* Alexia Jolicoeur-Martineau. **Less is More: Recursive Reasoning with Tiny Networks.** arXiv:2510.04871
  [cs.LG], v1 submitted 6 October 2025. <https://arxiv.org/abs/2510.04871>.
  * **Checked in the paper:**
    * Figure 3 pseudocode: `latent_recursion` (n × `z = net(x, y, z)`, then `y = net(y, z)`) and
      `deep_recursion` (T − 1 recursions under `no_grad`, one with gradient, return detached `(y, z)`, output
      head and Q head);
    * the deep-supervision loop: cross-entropy plus binary cross-entropy of `q_hat` against
      `(y_hat == y_true)`, an optimiser step every supervision step, and early stop when `q_hat > 0`;
    * the text forms `z ← f(x + y + z)` and `y ← f(y + z)`;
    * the same network for both updates, distinguished by the presence of `x`;
    * T = 3, n = 6, N_sup = 16, a 2-layer network, EMA 0.999;
    * an MLP over the sequence instead of self-attention on Sudoku;
    * no continue loss, a single forward pass;
    * Sudoku-Extreme with 1K training examples and 1000 augmentations each.
  * **Checked in the code**, <https://github.com/SamsungSAILMontreal/TinyRecursiveModels>
    (`models/recursive_reasoning/trm.py`):
    * `H_init` / `L_init` truncated-normal buffers with std 1;
    * `q_head` with weight 0 and bias −5, reading `z_H[:, 0]`;
    * injection `z_L ← L(z_L, z_H + input)` and `z_H ← L(z_H, z_L)`;
    * training halts on `q_halt > 0` (`no_ACT_continue`), with exploration minimum steps drawn from
      [2, `halt_max_steps`] with probability `halt_exploration_prob`;
    * evaluation runs to `halt_max_steps`;
    * post-norm RMSNorm blocks with SwiGLU.
* Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, Yasin Abbasi Yadkori.
  **Hierarchical Reasoning Model.** arXiv:2506.21734 [cs.AI], v1 submitted 26 June 2025, later revised.
  <https://arxiv.org/abs/2506.21734>. Checked in the paper:
  * deep supervision over segments, with the state detached between segments and an optimiser step per segment;
  * the one-step gradient approximation;
  * ACT: a Q-head on the final high-level state with halt / continue values;
  * halting when the segment count exceeds `M_max`, or when `Q_halt > Q_continue` and the count is at least
    `M_min`, where `M_min` is uniform on {2, …, M_max} with probability ε and 1 otherwise;
  * the halt target 1[ŷ = y];
  * Sudoku-Extreme with 1000 training examples.

## Measured results

Measured on **2026-11-20** (`quick=True`, `seeds=(0, 1, 2)`, single CPU thread `OPENBLAS_NUM_THREADS=1`, digest `79ccc3bbb0d6f94a`, wall-clock `8.63 s`):

| Arm | Train exact-solve | Val exact-solve | Held-out cell acc | Held-out exact-solve | Held-out Δ vs arm (95 % CI) |
|---|---:|---:|---:|---:|---|
| `trm` (candidate) | 0.2630 | 0.2734 | 0.8394 | 0.3242 | — (per-step: `0.2240 → 0.3242`, halting Brier `0.2236`) |
| `unrolled` (best baseline) | 0.4089 | 0.4089 | 0.8829 | 0.4245 | `trm − unrolled`: −0.1003 (`[-0.1999, -0.0007]`, `worse`) |
| `shallow` | 0.3073 | 0.2943 | 0.8503 | 0.3112 | `trm − shallow`: +0.0130 (`[-0.0501, +0.0762]`, `no_detectable_difference`) |
| `mlp` | 0.0026 | 0.0000 | 0.3459 | 0.0000 | `trm − mlp`: +0.3242 (`[+0.2286, +0.4198]`, `improved`) |

* **Headline verdict (`trm_vs_best_baseline(unrolled)`):** `worse` on held-out exact-solve at the tiny quick-mode update budget, while clearly outperforming the flat `mlp` baseline (`improved`).
