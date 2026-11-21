# `muon_optimizer`: orthogonalised-momentum optimiser (Muon)

Module file: `synapse/frontier/muon.py` · tests: `tests/test_frontier_optim.py` ·
experiment id: `muon_optimizer` · kind: `tool` (an optimiser plus a benchmark; no engine extension, no live
observer) · evidence tier: `synthetic_abstraction`.

The module implements a **Muon**-style optimiser (MomentUm Orthogonalized by Newton–Schulz) for matrix
parameters, with an internal **AdamW** for everything else, in one `torch.optim.Optimizer`
(`MuonWithAuxAdam`). It is *inspired by* Jordan et al. (2024), whose blog post and reference code define the
algorithm, and by Liu et al. (2025), whose update scale and decoupled weight decay it uses. It also provides
`make_graft_optimizer`, the opt-in optimiser for Sonapse's live neural graft
(`FrontierSpecV1(graft_optimizer='muon')`), and a small benchmark of Muon against Adam on the graft's own
architecture. Nothing here reproduces a published training result, and no speed or scale claim is made.

## What it computes

For a 2-D parameter `W` of shape `A × B` with gradient `G_t`, momentum `mu`, learning rate `lr` and weight decay
`wd`, one step is:

```
M_t = mu * M_{t-1} + (1 - mu) * G_t                    momentum (EMA form of Jordan's code: lerp)
U_t = (1 - mu) * G_t + mu * M_t                         Nesterov input (U_t = M_t with nesterov=False)
X_0 = U_t / (||U_t||_F + 1e-7)                          Frobenius normalisation: singular values in (0, 1]
X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k,   (a, b, c) = (3.4445, -4.7750, 2.0315), 5 steps
W_t = W_{t-1} - lr * (0.2 * sqrt(max(A, B)) * X_5 + wd * W_{t-1})        (Liu et al., Eq. 4)
```

* **Orientation.** The iteration runs on the wide orientation of the matrix (transposed when `A > B`), as in
  the reference code, so the Gram matrix is the smaller one.
* **What Newton–Schulz returns.** It keeps the singular vectors of `U_t` and maps every singular value `s` of
  `X_0` through `p^5`, with `p(x) = a x + b x^3 + c x^5`. It is not the exact polar factor `U V^T`. Iterating the
  scalar map on a dense grid (2,000,001 points, recomputed on 2026-11-20) gives outputs in
  `[0.6818, 1.1344]` for every start `s >= 0.01` and in `(0, 1.2024]` for every `s` in `(0, 1]`; the tests check
  this band on random matrices and the exact scalar map on a matrix with a known SVD.
* **Momentum form.** Liu et al. write `M_t = mu M_{t-1} + G_t` with the Nesterov input `mu M_t + G_t`. The EMA
  form above is the same sequence multiplied by `1 - mu`, and the Frobenius normalisation removes that factor
  (up to the `1e-7`), so both give the same update.
* **Update scale.** `update_scale * sqrt(max(A, B))` with `update_scale = 0.2` (Liu et al., Eq. 4). It lets the
  Muon part and the AdamW part share one learning rate. Jordan's reference code scales by
  `max(1, A / B) ** 0.5` instead and uses separate learning rates for the two parts; that variant is not
  implemented.
* **Other shapes.** A parameter with more than two dimensions that is routed to Muon is flattened to
  `(shape[0], -1)`, as the reference code does for convolution filters, and `max(A, B)` is taken on the flattened
  shape.
* **AdamW part.** Every parameter not routed to Muon (biases, norm gains, embeddings, output heads, other
  shapes) is updated with the arithmetic of `torch.optim.AdamW`'s single-tensor path, in the same order
  (decoupled decay, bias-corrected moments; defaults `betas = (0.9, 0.999)`, `eps = 1e-8`). A test checks that
  vector parameters follow `torch.optim.AdamW` bit for bit.
* **Overflow.** When the squares of finite entries overflow the Frobenius norm, the matrix is first divided by
  its largest absolute entry (the reference code would return zeros there). The update is invariant to the
  gradient's scale, which a test checks up to a gradient scale of `1e25`.

**Atomic steps.** `step()` computes the whole update first. A non-finite gradient, or an update that would make
a parameter or a state tensor non-finite, raises `ValueError` and changes no parameter and no state, the Muon
part included. Sparse gradients are refused. `load_state_dict` validates the group layout, the routing of every
parameter (`use_muon`), the hyper-parameters, the state keys, shapes, dtypes and finiteness, and
`exp_avg_sq >= 0`, before it changes anything, and then loads a private deep copy.

## Parameter routing

* `MuonWithAuxAdam(params, ...)` accepts an `nn.Module` (routed by `muon_param_groups`), an iterable of tensors
  (2-D tensors go to Muon, the rest to AdamW) or group dicts. A group may set `use_muon` explicitly (`True` needs
  at least 2 dimensions) and override any hyper-parameter. Groups added later with `add_param_group` are
  validated the same way. Sets are refused because their order is not deterministic.
* `muon_param_groups(module, *, aux_modules=(), **hyper)` puts every 2-D parameter on Muon except
  `nn.Embedding` / `nn.EmbeddingBag` weights and parameters owned by `aux_modules`; everything else goes to
  AdamW. Jordan et al. recommend AdamW for embeddings, classifier heads, input layers and scalar or vector
  parameters; Liu et al. use AdamW for the norm, LM-head and embedding parameters.
* `make_graft_optimizer(params_or_module, kind='adam', lr=3e-4, *, heads_on_adam=True, inputs_on_adam=None,
  **muon_options)`:
  * `kind='adam'` returns exactly `torch.optim.Adam(params, lr=lr)`, the graft's existing optimiser (a test
    checks identical trajectories and state); any other option is refused.
  * `kind='muon'` on a module with the graft's structure (`actor_encoder`, `value` and `dynamics`
    `nn.Sequential` paths plus `actor_head`) keeps the three output layers (`graft_output_layers`: the actor head
    and the last linear layer of the value and dynamics paths) and the three input layers
    (`graft_input_layers`: the first linear layer of each path) on AdamW, which leaves Muon the three
    hidden-to-hidden matrices. This follows Jordan et al., who write that input and output parameters should be
    optimised with AdamW even though they are 2-D. `inputs_on_adam=None` (the default) means `True` for a
    graft-structured module and `False` for any other module or a parameter iterable, where input layers cannot
    be identified; `inputs_on_adam=False` puts the graft's input layers on Muon as well (the routing ablation
    below); `inputs_on_adam=True` on a non-graft module is refused.

| routing on the graft network | matrices on Muon | numbers on Muon / AdamW (hidden 32) | (hidden 64) |
|---|---|---|---|
| `kind='muon'` (default, the live opt-in) | the 3 hidden-to-hidden matrices | 3,072 / 19,221 | 12,288 / 38,421 |
| `kind='muon', inputs_on_adam=False` (ablation) | the 3 input + 3 hidden matrices | 21,344 / 949 | 48,832 / 1,877 |

*Review fix (2026-11-20).* Until this review the live opt-in used the second row (input layers on Muon), which
contradicts both sources' guidance and measured worse than Adam in this module's own benchmark (see *Measured
results*); the default is now the first row.

## Benchmark (`run_experiment`)

* **Architecture.** `synapse.neural_graft._ResidualNetwork` (the live graft's network: policy, value and
  dynamics heads), built inside `torch.random.fork_rng` and then re-initialised from a private
  `torch.Generator`, so the global torch RNG is left as it was.
* **Task.** Teacher–student regression. Inputs are shaped like the graft's: a 256-wide state (155 observation
  values and 64 feature values from `N(0, 1)`, 5 baseline-action and 32 prior values from `U(-1, 1)`, clipped to
  `[-5, 5]`), the 155 observation values for the value head and observation plus a 5-wide action for the
  dynamics head. A fixed, never-trained teacher of the same architecture, with every linear layer (the actor
  head included) at `teacher_gain = 2` times PyTorch's default initialisation range, gives the targets, which
  are standardised per output with the training split's mean and standard deviation.
* **Student and training.** Every arm trains the same student initialisation (PyTorch's default
  `nn.Linear` distribution and the architecture's zero actor head) on the same minibatch sequence for the same
  number of steps, with the graft's loss form (the sum of the three heads' MSE) and gradient-norm clipping at 1.
* **Arms.** `adam`: `torch.optim.Adam` (with `weight_decay > 0`, `torch.optim.AdamW` with the same decay as
  the Muon arms); `muon`: `make_graft_optimizer(kind='muon')`, the routing the live opt-in uses (input and
  output layers on AdamW); `muon_with_inputs` (the routing ablation, `routing_ablation=True`):
  `inputs_on_adam=False`, the input layers on Muon too. Both Muon arms use the same `heads_on_adam`.
* **Learning rate.** For every arm separately, the learning rate is chosen from `lr_grid` by the lowest mean
  validation MSE over seeds. The receipt flags a choice at the edge of the grid
  (`selected_at_grid_edge`) and lists every run, so a grid that is too narrow is visible.
* **Verdict.** `harness.compare` (paired 95 % t interval over seeds, lower is better) of held-out test MSE,
  `muon` minus `adam`, at the selected learning rates. `muon_with_inputs` minus `adam` is reported as a
  secondary comparison and does not set the verdict. If an arm diverges at its selected learning rate, the
  basis becomes `training_diverged_at_selected_lr` (paired statistics drop non-finite pairs, so a divergence is
  never hidden).
* **Receipt.** `config` (every option), `parameter_routing` (parameter names, counts and shapes per arm),
  `arms` (selected learning rate, the grid of mean validation MSE with diverged seeds, per-seed results with
  per-head test MSE and optimiser state size, every run), `comparisons`, `verdict`, `verdict_basis`, `budget`
  (parameters, steps and runs, Muon/AdamW split and optimiser state numbers per arm), `estimated_cost`
  (analytic forward–backward and Newton–Schulz FLOPs per step) kept apart from `measured_wallclock_s`, the
  teacher's raw output scale and `notes`.

| setting | quick | default |
|---|---|---|
| hidden width, parameters | 32, 22,293 | 64, 50,709 |
| train / val / test examples | 512 / 256 / 256 | 2048 / 512 / 1024 |
| steps × batch | 40 × 128 | 300 × 128 |
| `lr_grid` | 1e-3, 1e-2 | 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 |
| momentum, Nesterov, Newton–Schulz steps, weight decay | 0.95, yes, 5, 0 | same |

## Public API

```python
MODULE_ID = 'muon_optimizer'
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ARMS = ('adam', 'muon', 'muon_with_inputs')
EXPERIMENTS = {'muon_optimizer': run_experiment}
DEFAULT_CONFIG, QUICK_CONFIG                                   # benchmark options (overridable by keyword)

zeropower_via_newtonschulz5(matrix, steps=5, eps=1e-7) -> torch.Tensor
MuonWithAuxAdam(params, lr=1e-3, *, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0,
                betas=(0.9, 0.999), eps=1e-8, update_scale=0.2)
    .step(closure=None)  .add_param_group(group)  .load_state_dict(state)  .state_dict()
    .muon_parameters()  .adam_parameters()  .state_numbers()
muon_param_groups(module, *, aux_modules=(), **hyper) -> list[dict]
graft_output_layers(network) -> tuple[nn.Module, ...]
graft_input_layers(network) -> tuple[nn.Module, ...]
make_graft_optimizer(params_or_module, kind='adam', lr=3e-4, *, heads_on_adam=True, inputs_on_adam=None,
                     **muon_options) -> torch.optim.Optimizer
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict           # sealed receipt
```

Every constructor and function validates its input and raises `ValueError` on bad input. The benchmark draws
data from `numpy.random.default_rng(seed + offset)` and weights from private `torch.Generator` objects, runs
PyTorch on one thread and restores the caller's thread count; a test checks that the global torch RNG state and
the thread count are unchanged after a run.

## Connection to Sonapse

* `FrontierSpecV1.graft_optimizer` defaults to `'adam'`: the live neural graft keeps `torch.optim.Adam`, and its
  checkpoint settings stay free of an `optimizer_kind` entry, so nothing changes by default.
* With `graft_optimizer='muon'`, `NeuralResidualGraft` builds its optimiser with
  `make_graft_optimizer(network, kind='muon', lr=learning_rate)`, that is, the `muon` routing above (input
  layers, output heads and vectors on AdamW; the hidden-to-hidden matrices on Muon). Its checkpoints record the
  optimiser kind and refuse to load into an Adam graft. A Muon-graft checkpoint written before the 2026-11-20
  routing fix stores the old routing and is refused by `load_state_dict`'s routing check.
* The benchmark measures that routing (`muon` arm) and the alternative with the input layers on Muon; see
  *Measured results* before choosing the opt-in.

## Not implemented, and differences from the sources

* **Precision and hardware.** The reference code runs Newton–Schulz in bfloat16 on GPUs. Here it runs in float32
  (float64 for float64 inputs) on the CPU. No speed, memory or scale benefit is claimed.
* **Scale and learning rates.** Liu et al.'s `0.2 sqrt(max(A, B))` scale with one shared learning rate is used.
  Jordan's `max(1, A / B) ** 0.5` scale with separate learning rates (Muon 0.02, Adam 3e-4 in the reference
  `MuonWithAuxAdam`, whose Adam part uses `betas = (0.9, 0.95)` and `eps = 1e-10`) is not.
* **Not attempted.** Distributed or sharded orthogonalisation, the language-model and speed-run benchmarks of
  both sources, other Newton–Schulz coefficient schedules, and learning-rate schedules (the benchmark uses a
  constant learning rate).
* **Benchmark scope.** One synthetic teacher–student task on one small architecture, with a short training
  budget. It does not show how Muon behaves on the live graft, whose targets come from executed transitions.

## Measured results

Measured on **2026-11-20**, Windows 11 on ARM64, Python 3.12.10, torch 2.11 CPU, NumPy 2.4,
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` (the benchmark also runs torch on one thread). The machine
was shared with other jobs, so wall-clock is indicative only. Seeds 0, 1, 2. Test MSE is the mean over the three
heads of the MSE on standardised targets (a constant prediction scores about 1); lower is better.

**Quick** (`run_experiment(quick=True)`; digest `fe48c4d94d312b14`, identical in two calls; 14.3 s):

| Arm | Selected lr | Held-out test MSE (per seed) | Mean |
|---|---|---|---:|
| `adam` | 1e-2 (grid edge) | 0.3895, 0.4110, 0.4056 | 0.4020 |
| `muon` (input and output layers on AdamW) | 1e-2 (grid edge) | 0.3880, 0.3999, 0.4081 | 0.3987 |
| `muon_with_inputs` (input layers on Muon) | 1e-2 (grid edge) | 0.5432, 0.5858, 0.5516 | 0.5602 |

* `muon − adam`: −0.0034, 95 % CI [−0.0207, +0.0140] → **`no_detectable_difference`** (verdict).
  `muon_with_inputs − adam`: +0.1582, CI [+0.1212, +0.1952] (`worse`).
* The quick grid has two values and every arm picked its upper edge; quick mode is a pipeline check, not
  evidence.

**Default** (`run_experiment()`; digest `c23a6ae52d6e23c7`; 207.6 s, of which 49.5 / 70.8 / 82.9 s were the 15
`adam` / `muon` / `muon_with_inputs` runs):

| Arm | Selected lr | Held-out test MSE (per seed) | Mean | Per head: policy / value / dynamics | Optimiser state numbers |
|---|---|---|---:|---|---:|
| `adam` | 1e-3 | 0.2668, 0.2761, 0.2797 | 0.2742 | 0.174 / 0.163 / 0.485 | 101,418 |
| `muon` | 1e-3 | 0.2628, 0.2707, 0.2726 | 0.2687 | 0.164 / 0.157 / 0.486 | 89,130 |
| `muon_with_inputs` | 3e-3 | 0.3351, 0.3420, 0.3507 | 0.3426 | 0.212 / 0.234 / 0.582 | 52,586 |

* `muon − adam`: −0.0055, 95 % CI [−0.0094, −0.0016] → **`improved`** (verdict): about 2 % lower test MSE,
  from the policy and value heads; the dynamics head is unchanged. No arm selected a learning rate at the edge
  of the five-value grid, and no run diverged.
* `muon_with_inputs − adam`: +0.0684, CI [+0.0622, +0.0746] (`worse`).
* Analytic cost per step (`estimated_cost`, not measured): forward–backward 38.5 MFLOP; Newton–Schulz
  23.6 MFLOP for `muon` and 78.2 MFLOP for `muon_with_inputs`.

**How to read this.** The `muon` routing was changed during the 2026-11-20 review, after both routings had
been measured: before the fix the candidate arm put the input layers on Muon and the input-layers-on-AdamW
routing was the ablation (then called `muon_hidden_only`). The receipts from before the fix (quick digest
`2aa131161a125fc2`, default `bab33558fdf5241c`, both run on 2026-11-20) contain the same per-seed numbers with
the two Muon arms swapped and gave the verdict `worse`. The change follows the routing both sources prescribe
(input and output layers on AdamW), not the result, but the order of events is stated here so the verdict can
be weighed accordingly. The effect is small, on one synthetic teacher–student task with three seeds and a
short budget; it is no evidence about the live graft. What the benchmark does show clearly, at both sizes, is
that putting the graft's input layers on Muon hurts.

Unit tests with known answers (`tests/test_frontier_optim.py`, Muon part): the Newton–Schulz output on a
matrix with a known SVD equals the scalar map `p^5` applied to its normalised singular values; the singular
value band on random matrices; scale invariance up to a gradient scale of `1e25`; one hand-computed Muon step
(momentum, Nesterov input, scale `0.2 sqrt(max(A, B))`, decoupled decay); the AdamW part equal to
`torch.optim.AdamW` bit for bit; `make_graft_optimizer('adam')` equal to `torch.optim.Adam`; the graft
routing (default and ablation); atomic refusal of non-finite steps; `load_state_dict` validation before any
change; and a sealed, deterministic receipt that leaves the global torch RNG and thread count unchanged.

## Verified references

* Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein.
  **Muon: An optimizer for hidden layers in neural networks.** Blog post, 8 December 2024.
  <https://kellerjordan.github.io/posts/muon/>. Checked on 2026-11-20: the coefficients
  `(3.4445, -4.7750, 2.0315)`, the replacement of `G` by `G / ||G||_F` before the iteration, Nesterov-style
  momentum as the default, and AdamW for scalar and vector parameters, input and output layers, embeddings and
  classifier heads.
* Reference code: <https://github.com/KellerJordan/Muon> (`muon.py`), read on 2026-11-20:
  `zeropower_via_newtonschulz5` (bfloat16, transpose when rows > cols, Frobenius normalisation with `1e-7`),
  `muon_update` (`momentum.lerp_(grad, 1 - beta)`, Nesterov `grad.lerp_(momentum, beta)`, the
  `max(1, rows / cols) ** 0.5` scale, defaults `beta = 0.95`, `ns_steps = 5`), decoupled weight decay
  `p.mul_(1 - lr * wd)`, and the `MuonWithAuxAdam` defaults quoted above.
* Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu,
  Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang,
  Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu,
  Xinyu Zhou, Zhilin Yang. **Muon is Scalable for LLM Training.** Technical report, arXiv:2502.16982
  (v1, 24 February 2025). <https://arxiv.org/abs/2502.16982>. Checked on 2026-11-20 against the arXiv listing
  and the HTML text: Eq. 1 (Muon with `M_t = mu M_{t-1} + grad`), Eq. 3 (weight decay), Eq. 4 (the
  `0.2 · O_t · sqrt(max(A, B))` scale), momentum 0.95 "same as Jordan et al.", the Nesterov-style input, and
  AdamW for the non-matrix parameters (RMSNorm, LM head, embeddings).
