# `connectomics`: structural, functional and effective connectivity

Module file: `synapse/frontier/connectomics.py` · tests: `tests/test_frontier_connectomics.py` ·
experiment id: `connectomics` · live observer: `ConnectivityObserver` · kind: `analysis` ·
evidence tier: `synthetic_abstraction`.

The module keeps three kinds of connectivity apart, as the sibling project `kai9987kai/SONAPSE-CONNECTOME`
does (`sonapse_connectome/connectome/functional.py`, `effective.py`): **structure** (the synapse table of a
NEUROWEAVE brain), **function** (statistical dependence between unit activities) and **effective connectivity**
(lagged, directed, predictive influence). It re-implements those ideas for NEUROWEAVE activity and checks every
estimator on a synthetic process whose answer is known before it is applied to a brain. It is a computational
abstraction: the "structure" is a toy synapse table and the "function" is the activity of abstract units, not
imaging data, and nothing here is evidence of biological fidelity.

## What it computes

**Functional connectivity (FC).** For a `(T, n)` activity array:

* Pearson FC `R_ij = corr(x_i, x_j)` and Spearman FC (Pearson correlation of per-unit ranks, average ranks for
  ties). Zero-variance units get zero rows/columns and a zero diagonal entry. Rescaling each channel into
  different measurement units preserves its Pearson and partial correlations.
* Partial correlation from the ridge-regularised precision of the correlation matrix,
  `Omega = (R + ridge I)^-1`, `P_ij = -Omega_ij / sqrt(Omega_ii Omega_jj)` (unit diagonal).
  `partial_from_covariance(cov, ridge=0)` is exact and raises `ValueError` on a singular matrix;
  `partial_correlation(X, ridge=1e-2)` is the sample version.

For a Gaussian chain `x -> y -> z` (`y = a x + e`, `z = a y + e`, unit noise) the precision matrix is
`[[1 + a^2, -a, 0], [-a, 1 + a^2, -a], [0, -a, 1]]`, so the exact partial correlations are
`P_xy = a / (1 + a^2)`, `P_yz = a / sqrt(1 + a^2)` and `P_xz = 0`, while the Pearson correlation
`R_xz = a^2 / sqrt(a^2 (a^2 + 1) + 1)` is not zero. The tests and the lab use this as the known answer.

**Dynamic FC (dFC).** Pearson FC in sliding rectangular windows (`window`, `step`); the upper triangles of the
window matrices are clustered with a deterministic k-means (Lloyd iterations, D²-weighted seeding,
`n_init` restarts, empty clusters re-seeded, labels in order of first appearance) for `k = 2 .. k_max`, and the
`k` with the largest mean silhouette width (Rousseeuw, 1987) wins. If no `k` reaches `min_silhouette`
(a heuristic default of 0.25) or there are fewer than three windows, the series is reported as one FC state.
Dwell times are run lengths of the window-state sequence (in windows and in samples, `x step`), with occupancy
and the number of state transitions.

**Effective connectivity (EC).** A ridge VAR(1) with intercepts,
`A = argmin_A sum_t ||x_{t+1} - A x_t||^2 + lambda ||A||_F^2`, `lambda = ridge * (T - 1)`, on (optionally z-scored)
series; `A[target, source]`. Significance comes from a **per-target conditional block-permutation null**: for
each draw, the past of all sources is block-permuted jointly in time (blocks of `round((T-1)^(1/3))` steps by
default, at least 2), except that each target keeps its own unpermuted past; every target's ridge regression is
refit (all targets in one batched solve). Each edge gets `z = (A - mean_null) / sd_null`, a two-sided normal
p-value and a rank p-value `(1 + #{|null - mean| >= |A - mean|}) / (1 + draws)`; the mask is
Benjamini-Hochberg at FDR `alpha` over the off-diagonal edges. This is lagged predictive (Granger-style)
influence, not an interventional effect.

**Structure and coupling.**

* `structural_matrix(n, src, dst, weight)`: dense `W[src, dst]` from an edge list (duplicates summed, `|w|` by
  default, diagonal zeroed).
* `weighted_communicability(S) = exp(D^-1/2 S D^-1/2)`, `D = diag(sum_k S_ik)`, the weighted communicability
  of Crofts & Higham (2009) (Estrada & Hatano, 2008, for the unweighted version); isolated nodes keep zero rows.
* `structure_function_coupling(S, FC)`: Spearman correlation over node pairs `i < j` between symmetrised
  structure and FC (all pairs and structurally connected pairs, signed and absolute FC), and the same for
  communicability, which also credits indirect walks; in the spirit of Honey et al. (2009).
* `modularity(W, labels)`: Newman-Girvan `Q = (1/2m) sum_ij (W_ij - k_i k_j / 2m) [c_i = c_j]`.
* `fc_communities(FC)`: leading-eigenvector communities (Newman, 2006) of the positive part of FC by
  recursive bisection with the generalised modularity matrix `B^(g)_ij = B_ij - delta_ij sum_{l in g} B_il`,
  without vertex-moving refinement, so the reported modularity is a proxy (a lower bound on the maximum).

## Live observer (`ConnectivityObserver`)

`ConnectivityObserver(n_units, *, seed=7, history=256, window=None, max_units=32, k_max=4, ec_null_draws=31,
ridge=0.05, alpha=0.05, min_silhouette=0.25, **params)`; `ConnectivityObserver.from_spec(n_units, spec)` takes
`seed`, `history` and `fc_window` (clipped to `history`) from a `FrontierSpecV1`. `window` defaults to 128; the
spec name `fc_window` is accepted as the same parameter (a conflicting pair raises `ValueError`), and the spec
parameters of other observers (`sync_pairs`, `sae_features`, `sae_k`, `max_observed_units`, `observe_every`)
are accepted and ignored; any other keyword raises `ValueError`.

* A seeded, sorted subset of at most `max_units` units is copied into a ring buffer of `history` rows;
  `observe(x)` validates `x` (1-D, length `n_units`, finite) before touching any state and never writes to it.
* Metrics are recomputed once `window` rows exist and then every `max(8, window // 4)` observations: static FC
  and EC on the last `window` rows (non-constant units only), dFC on the whole buffer with windows of
  `max(8, window // 4)` rows and half-window steps. The EC null is seeded by `(seed, observed steps)`, so results
  do not depend on how often `summary()` is called.
* `summary()` keys: `module`, `evidence_tier`, `n_units`, `analysed_units`, `unit_sample`, `history`, `window`,
  `recompute_every`, `observed_steps`, `buffered`, `ready`, `computed_at`, `rows`, `constant_units`,
  `mean_abs_fc`, `fc_modularity`, `fc_communities`, `dfc_window`, `dfc_windows`, `dfc_states`,
  `dfc_silhouette`, `dfc_mean_dwell_windows`, `dfc_mean_dwell_steps`, `dfc_transitions`, `ec_density`,
  `ec_significant` and `ec_top_edges` (at most five `{source, target, weight, z, significant}` records, by
  `|z|`). Values are `None` until the first recompute. Unit indices refer to positions in the observed vector.
* `export_state()` / `import_state(state)` round-trip through JSON; `import_state` validates the schema, the
  configuration, the unit sample, the buffer (shape, finiteness, zeros beyond the observed steps), the recompute
  schedule and every cached metric before replacing anything. `reset()` returns to the empty state.

The observer runs inside the frontier observer hub (`synapse/frontier/runtime.py`) when `connectomics` is in
`FrontierSpecV1.modules`; it is read-only and never writes into the brain or any shared RNG.

## Lab experiment (`run_experiment`, `EXPERIMENTS = {'connectomics': run_experiment}`)

`run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides)` returns a sealed `kind: 'validation'` receipt
with `verdict` in `('pass', 'fail')` and nine named checks, each required for every seed:

| Check | Known answer |
|---|---|
| `var_support_auroc_above_0.9` | Sparse stable VAR(1) (`var_units = 8`, edge probability 0.15, off-diagonal weights `±U(0.25, 0.5)`, self-coupling `U(0.2, 0.5)`, spectral radius ≤ 0.8): AUROC of `|A_hat|` for the true support > 0.9 |
| `var_sign_accuracy_at_least_0.9` | Signs of the estimated true edges |
| `var_mask_pooled_precision_at_least_0.8` | Precision of the BH mask pooled over seeds |
| `chain_partials_match_exact_values` | Gaussian chain `x -> y -> z` (`a = 0.8`): partials within `4 / sqrt(T)` of the exact values above, `P_xz` = 0 |
| `chain_fc_xz_clearly_nonzero` | `|R_xz| > 0.3` (the indirect link FC cannot remove) |
| `switching_two_states_recovered` | Six units alternating between two covariance states (blocks of 150 samples, within-state correlation 0.8 in one half of the units): dFC finds 2 states |
| `switching_pure_window_accuracy_at_least_0.9` | Label accuracy on windows that lie inside one block (up to label swap) |
| `stationary_control_single_state` | Same length, stationary, with the averaged covariance: dFC reports 1 state |
| `neuroweave_values_finite` | The descriptive part below is finite |

Then, for each topology (quick: `clustered`, `ring_lattice`, `hub_skewed`; default adds `multi_region`) and
seed, a frozen NEUROWEAVE engine (`learn=False`, so the synapse table that generates the activity stays fixed)
runs on the task stream; after burn-in the module reports structure metrics, mean |FC| and |partial|, the FC
modularity proxy, structure-function and communicability-function coupling, EC density, the AUROC of `|A_hat|`
against the synapse table, the fraction of significant EC edges that sit on a synapse, sign agreement, and the
dFC state count. This part is descriptive: no verdict is drawn from it beyond finiteness.

The receipt also holds `config`, `estimated_cost` (engine steps, ridge solves of the nulls, FC windows),
`measured_wallclock_s` (excluded from the digest) and `notes`.

## Public API

`MODULE_ID`, `ConnectivityObserver`, `pearson_fc`, `spearman_fc`, `partial_correlation`,
`partial_from_covariance`, `precision_to_partial`, `sliding_window_fc`, `kmeans`, `silhouette_score`,
`dwell_statistics`, `dynamic_fc_states`, `var1_fit`, `block_permutation`, `var1_permutation_null`,
`benjamini_hochberg`, `effective_connectivity`, `auroc`, `structural_matrix`, `weighted_communicability`,
`structure_function_coupling`, `modularity`, `fc_communities`, `run_experiment`, `EXPERIMENTS`.
All randomness uses `numpy.random.default_rng(seed + offset)`; inputs are validated with `ValueError`.
`block_permutation` is shared with `causal_emergence`.

## Connection to Sonapse

* The live observer gives the workbench FC, dFC and EC summaries of the running brain without touching it.
* The lab experiment uses `neuroweave.experiments.make_spec` / `engine_topology` and
  `neuroweave.topology.topology_metrics`, so the structural side is the engine's own synapse table.

Differences from SONAPSE-CONNECTOME: that project computes partial correlation from a shrinkage covariance,
labels dFC "states" by quantile bins of the mean |FC| of each window, and fits a ridge lagged regression without
intercepts or a significance test. Here dFC states are clustered with k-means and a silhouette criterion, EC has
intercepts, a conditional permutation null and FDR control, and every estimator has a known-answer check.

## Not implemented

* No imaging data, haemodynamic model or parcellation; no dynamic causal modelling; EC is a single-lag linear
  model (no higher-order lags, no nonlinear or interventional EC).
* The dFC pipeline is in the spirit of Allen et al. (2014) but simpler: they used tapered windows (a 22-TR
  rectangle convolved with a Gaussian), graphical-LASSO regularised covariance per window, k-means with the L1
  distance and the elbow criterion for k (k = 7); here windows are rectangular, FC is plain Pearson, k-means is
  Euclidean and k is chosen by the mean silhouette with a 0.25 floor (a heuristic, not a test).
* The FC community split has no vertex-moving refinement (a modularity proxy).
* The observer only analyses up to `max_units` units and recomputes on a fixed schedule.

## Verified references

Checked on 2026-11-20 (Crossref metadata and the publishers' or PMC pages).

* Christopher J. Honey, Olaf Sporns, Leila Cammoun, Xavier Gigandet, Jean-Philippe Thiran, Reto Meuli, Patric
  Hagmann. **Predicting human resting-state functional connectivity from structural connectivity.**
  *PNAS* 106(6), 2035–2040, 10 February 2009. <https://doi.org/10.1073/pnas.0811168106>
* Elena A. Allen, Eswar Damaraju, Sergey M. Plis, Erik B. Erhardt, Tom Eichele, Vince D. Calhoun. **Tracking
  Whole-Brain Connectivity Dynamics in the Resting State.** *Cerebral Cortex* 24(3), 663–676, March 2014
  (online 11 November 2012). <https://doi.org/10.1093/cercor/bhs352>
* Ernesto Estrada, Naomichi Hatano. **Communicability in complex networks.** *Physical Review E* 77(3), 036111,
  March 2008. <https://doi.org/10.1103/PhysRevE.77.036111>
* Jonathan J. Crofts, Desmond J. Higham. **A weighted communicability measure applied to complex brain
  networks.** *Journal of the Royal Society Interface* 6(33), 411–414, April 2009 (online 13 January 2009).
  <https://doi.org/10.1098/rsif.2008.0484>
* M. E. J. Newman. **Modularity and community structure in networks.** *PNAS* 103(23), 8577–8582, June 2006.
  <https://doi.org/10.1073/pnas.0601602103>
* C. W. J. Granger. **Investigating Causal Relations by Econometric Models and Cross-spectral Methods.**
  *Econometrica* 37(3), 424–438, August 1969. <https://doi.org/10.2307/1912791>
* Yoav Benjamini, Yosef Hochberg. **Controlling the False Discovery Rate: A Practical and Powerful Approach to
  Multiple Testing.** *Journal of the Royal Statistical Society, Series B* 57(1), 289–300, 1995.
  <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>
* Peter J. Rousseeuw. **Silhouettes: a graphical aid to the interpretation and validation of cluster
  analysis.** *Journal of Computational and Applied Mathematics* 20, 53–65, November 1987.
  <https://doi.org/10.1016/0377-0427(87)90125-7>
* Sibling project: `kai9987kai/SONAPSE-CONNECTOME` (read-only clone; ideas, not code, reused).

## Measured results

Measured on 2026-11-20, Windows 11 on ARM64, Python 3.12, NumPy 2.4, one BLAS thread
(`OPENBLAS_NUM_THREADS=1`), seeds 0, 1, 2, on a machine shared with other jobs (wall-clock is indicative).

| Run | Verdict | Key numbers | Wall-clock |
|---|---|---|---|
| `run_experiment(quick=True)` | `pass` (9/9 checks) | VAR(1) (8 units, 1500 steps, 99 null draws): support AUROC 1.000 / 1.000 / 1.000, sign accuracy 1.0, BH mask 24 true and 0 false positives (pooled precision 1.0), max coefficient error 0.052–0.056. Chain (2000 steps): `P_xz` 0.0085 / 0.0166 / −0.0042 (exact 0), max partial error 0.012–0.018 (tolerance 0.089), `R_xz` 0.42–0.47 (exact 0.447). Switching (6 blocks of 150): 2 states, silhouette 0.55–0.56 (k = 2) against 0.15–0.19 for the stationary control (1 state), pure-window accuracy 1.0 | 9.4 s (validation 3.0 s, NEUROWEAVE 6.4 s) on a lighter-loaded re-run; 16.3 s (validation 5.5 s, NEUROWEAVE 10.8 s) under load. Two runs in one process gave the same digest |
| `run_experiment()` | `pass` (9/9 checks) | VAR(1) (3000 steps, 199 null draws): support AUROC 1.000 for every seed, sign accuracy 1.0, mask 24 true and 2 false positives (pooled precision 0.92), max coefficient error 0.045–0.055. Chain (4000 steps): `P_xz` −0.016 / −0.014 / 0.011, max partial error 0.016–0.025 (tolerance 0.063), `R_xz` 0.45–0.46. Switching (8 blocks): 2 states, silhouette 0.55–0.57 against best 0.16–0.18 for the stationary control (1 state), pure-window accuracy 1.0 | 59.6 s (validation 4.0 s, NEUROWEAVE 55.6 s, four topologies × 32 neurons × 600 steps) |

NEUROWEAVE (quick, 24 neurons, 240 steps, 40 burn-in; descriptive, means over seeds): structure-FC Spearman
0.21 (`clustered`), 0.46 (`ring_lattice`), 0.15 (`hub_skewed`); communicability-FC Spearman 0.23 / 0.52 / 0.15;
EC-structure AUROC 0.62 / 0.71 / 0.58; EC density 0.10 / 0.17 / 0.09; dFC states 2.3 / 2.3 / 1.0.
Default size (32 neurons, 600 steps, 50 burn-in), `clustered` / `ring_lattice` / `hub_skewed` / `multi_region`:
structure-FC Spearman 0.16 / 0.23 / 0.15 / 0.12, communicability-FC Spearman 0.18 / 0.18 / 0.19 / 0.18,
EC-structure AUROC 0.73 / 0.76 / 0.63 / 0.69, dFC states 1.0 / 1.0 / 1.0 / 1.3. The quick and default numbers
differ (network size, run length), so neither is a stable property of a topology; these are small runs of a toy
model, reported, not tested.
