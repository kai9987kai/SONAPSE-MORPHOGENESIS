# `kuramoto_binding`: oscillatory binding units

Module file: `synapse/frontier/kuramoto.py` · tests: `tests/test_frontier_kuramoto.py` ·
experiment ids: `kuramoto_binding`, `kuramoto_binding_benchmark` · engine extension: `KuramotoExtension` ·
live observer: `KuramotoObserver` · evidence tier: `synthetic_abstraction`.

The module implements the discrete oscillator update of **Artificial Kuramoto Oscillatory Neurons** (AKOrN,
Miyato et al., ICLR 2025) and uses it in three small settings:

* a NEUROWEAVE engine extension that adds synchrony features to the readout,
* a read-only live observer of unit activity,
* a synthetic binding benchmark with a known answer.

It is *inspired by* AKOrN and does not reproduce it. Couplings here are fixed rather than learned. There are no
convolutional or attention couplings, no readout blocks and no training. None of the paper's tasks or results
(object discovery, adversarial robustness, calibration, Sudoku) are attempted. The oscillators are an abstract
computational device, not a model of neural oscillations.

## What it computes

**Oscillators.** Each oscillator is a unit vector `x_i` in `R^N`, with `N` even and `N = 2` by default. The
update is the paper's Eq. 2 in discrete form (Eqs. 4–5):

```
dx_i = Omega_i x_i + Proj_{x_i}( c_i + sum_j J_ij x_j )      Proj_x(y) = y - <y, x> x
x_i <- Pi( x_i + gamma * dx_i )                              Pi(v) = v / ||v||
```

* `Omega_i` is antisymmetric: a natural-frequency block `[[0, -w], [w, 0]]` on each coordinate plane
  `(2p, 2p+1)`.
* `c_i` is the conditional stimulus.
* `J_ij` is a scalar coupling (`J_ij * I_N`). A full `(C*N, C*N)` block matrix is also accepted, and
  `kron(J, I_N)` gives exactly the scalar result (tested).
* `K` iterations run per call (`akorn_step`, `akorn_run`).

Properties that are tested:

* The projection is tangent (`<Proj_x(y), x> = 0`) and idempotent, and every update returns unit rows.
* With `J = 0` and `c = 0`, each rotation plane advances by exactly `arctan(gamma * w)` per step. The plane
  amplitudes change when frequencies differ. As `gamma -> 0`, the phase follows `dtheta/dt = w`.
* Uniform positive coupling `2/(C-1)` takes `R` from 0.05 to 1 (to `1 - 1e-9` with identical frequencies, and
  above 0.99 with a 0.1 frequency spread, for both `N = 2` and `N = 4`). Uniform negative coupling drives
  `R` below `1e-6`.
* The energy `E = -1/2 sum_ij x_i^T J_ij x_j - sum_i c_i^T x_i` (paper Eq. 3, `kuramoto_energy`) never
  increases by more than `1e-12` per step under the paper's Lyapunov conditions. Those conditions are symmetric
  scalar couplings, a shared `Omega` (here 0) and `Omega c_i = 0`. The tested regime is `gamma = 0.1` with
  `||J||_2 = 1`.

**Measures.**

* Order parameter `R = ||mean_i x_i||`, and per-group coherence `||mean_{i in g} x_i||`.
* Phase `atan2(x_1, x_0)`. This is the first rotation plane, which is exact for `N = 2`.
* Phase-locking value `|mean_t exp(i (phi_a - phi_b))|`. It is a time-averaged variant of the PLV of Lachaux
  et al. (1999); the extension uses an exponentially weighted, bias-corrected average over engine steps.
* Phase-cluster count: the number of circular runs of *dense* bins in a 36-bin angle histogram. A bin is dense
  when it holds at least 10 % of the oscillators. An incoherent population, or one with no sparse bin at all,
  gives 0. A cluster smaller than about 20 % of the population can be missed if it straddles a bin edge.
* Shanahan (2010) indices from the community synchrony `phi_c(t)`, with the paper's sample variances:
  * Metastability `lambda`: the mean over communities of `1/(T-1) sum_t (phi_c(t) - <phi_c>)^2`.
  * Chimera index `chi`: the mean over time of `1/(M-1) sum_c (phi_c(t) - <phi(t)>)^2`.

  The tests reproduce the paper's reference values: `sigma_chi = 2/7` for a "perfect" chimera state,
  `chi_max = 1/7`, and `sigma_met -> 1/12` for uniformly distributed synchrony.
* Adjusted Rand index (Hubert & Arabie, 1985), implemented in NumPy from the contingency table. It returns 1
  for identical partitions, including the degenerate one-cluster and all-singleton cases. The tests check it
  against hand cases (`24/99`, `-1/2`, `4/7`) and against a brute-force pair-counting formula.
* Seeded k-means with k-means++ seeding (Arthur & Vassilvitskii, 2007). It runs several restarts, and the
  lowest inertia wins.

## Engine extension (`KuramotoExtension`, id `kuramoto_binding`)

```python
KuramotoExtension(seed=0, *, mode='region', units=16, dim=2, iterations=4, gamma=0.1, coupling_scale=1.0,
                  stimulus_scale=1.0, omega_mean=1.0, omega_spread=0.5, top_pairs=4, plv_rate=0.1)
```

**Oscillators.** In `mode='region'` there is one oscillator per NEUROWEAVE region (10). In `mode='units'`
there is one oscillator per unit, for `units` units sampled at `bind` with the extension seed.

**Coupling (copied at `bind`).** The engine itself is not kept.

* Synapse weights `brain.synapses.w` are summed over `src -> dst` into `J[group(dst), group(src)]`.
* Each row is divided by the size of its target group, which gives the mean input per target unit.
* The diagonal is zeroed, because self-coupling vanishes under the projection.
* `J` is scaled so that its largest absolute row sum equals `|coupling_scale|`.
* Inhibitory synapses give negative (anti-phase) couplings.
* `J` is a snapshot: later plasticity is not tracked.

**Stimulus.** `c_g = stimulus_scale * m_g * u_g`, where `m_g` is the group's current mean output and `u_g` is a
fixed, seeded unit vector.

**Dynamics.** Each engine step runs `iterations` AKOrN updates. Natural frequencies are
`omega_mean + omega_spread * N(0, 1)`, drawn per oscillator and plane.

**Features.** There are `feature_dim = C + 1 + top_pairs` of them: **15** in region mode with the defaults. All
lie in `[-1, 1]`.

1. `C` cosines between each oscillator and the global mean-field direction. These are 0 when `R` is about 0.
   Like the other two blocks, they only see relative phases: rotating every oscillator by the same angle leaves
   all features unchanged (tested to `1e-12`).
2. The global order parameter `R`.
3. The `top_pairs` largest pairwise PLVs, in descending order.

**Rules.**

* The extension only adds readout features. `transform_drive` is the neutral base method, and the tests check
  that the engine's trajectory and exported state are bit-identical with and without it.
* It has no trainable parameters (`parameter_count() == 0`). It is therefore frozen when `ctx['learn']` is
  false, and its oscillator state evolves the same way in both cases.
* It is deep-copyable.
* `export_state` and `import_state` round-trip exactly through JSON. `import_state` validates everything before
  mutating: config, `n`, shapes, finiteness, unit norm, `|PLV| <= 1` and a zero coupling diagonal.
* Accessors return copies: `coupling_matrix()` and `states()`.
* `telemetry()` reports the order parameter, mean and max PLV, and coupling statistics.

## Live observer (`KuramotoObserver`)

```python
KuramotoObserver(n_units, *, seed=7, history=256, oscillators=64, dim=2, iterations=2, gamma=0.1, omega_mean=1.0,
                 omega_spread=0.3, fan_in=4, coupling=0.5, hebbian_rate=0.01, hebbian_scale=2.0, stimulus_scale=0.5,
                 communities=8, groups=None, bins=36, min_cluster_fraction=0.1, **spec_params)
KuramotoObserver.from_spec(n_units, spec: FrontierSpecV1, **params)     # uses spec.seed and spec.history
```

**Oscillators and stimulus.** There are `M = min(oscillators, n_units)` oscillators, each driven by one
subsampled unit (seeded, sorted indices). The stimulus is `c_i = stimulus_scale * tanh(rate_i) * u_i`.

**Coupling.** `J = J0 + hebbian_scale * H / sqrt(M - 1)`.

* `J0` is fixed, seeded and sparse: `fan_in` partners per oscillator, each with weight `coupling / fan_in`.
* `H` is a slow Hebbian co-activity estimate, built as follows:
  * Rates are standardised with a running EWMA mean and variance (rate 0.05) and clipped to ±3 sd.
  * `S <- (1 - eta) S + eta z z^T` starts at the identity, a shrinkage prior.
  * `H_ij = S_ij / sqrt(S_ii S_jj)` off the diagonal, so `|H| <= 1`.

  The tests check that `H` recovers a two-block correlation structure: above 0.95 within blocks, where the true
  value is 0.99, and below 0.1 across blocks. They also check that `J - hebbian_scale * H / sqrt(M - 1)` stays
  equal to the fixed `J0`.

**Communities.** Shanahan's indices are computed over communities: contiguous blocks of the sampled unit order
(NEUROWEAVE regions are contiguous unit blocks), or the given `groups` labels.

**`summary()` keys.** The summary is JSON-safe and contains no wall-clock.

* Always present: `module`, `evidence_tier`, `observed_steps`, `history_len`, `history`, `oscillators`, `dim`,
  `communities`, `hebbian_mean_abs` and `hebbian_max_abs`.
* `None` before the first observation: `R_last`, `R_mean` and `R_std` (the standard deviation of `R` over the
  history, a metastability measure), `metastability_index`, `chimera_index`, `community_R_mean`,
  `phase_clusters` (the latest histogram count) and `phase_clusters_mean`.

**Rules.**

* The observer copies its input and never writes to it.
* All buffers have a fixed size.
* `export_state` and `import_state` (schema `KuramotoObserverStateV1`) round-trip bit-exactly. Imports are fully
  validated before any state changes.
* Unknown keyword parameters raise `ValueError`. `FrontierSpecV1` parameters meant for other observers
  (`sync_pairs`, `sae_features`, `sae_k`, `fc_window`, `max_observed_units`, `observe_every`) are accepted and
  ignored.

## Binding benchmark (`run_binding_benchmark`, id `kuramoto_binding_benchmark`)

**Task.** There are `groups` latent signals `s_g(t) ~ N(0, 1)`. Feature `i` of group `g` is
`a_i s_g(t) + noise * eps_i(t)`, with loading `a_i ~ U(0.5, 1.5)`.

**Candidate coupling.** The Pearson correlation of the feature rows, minus the mean off-diagonal correlation
(a uniform inhibition), with a zero diagonal. It is scaled so that the largest absolute row sum equals
`coupling_scale` (2.0).

**Dynamics.** The stimulus is `c_i = 0`, so binding comes from the coupling alone. Each seed draws its own
initial states and near-identical natural frequencies (`1 ± 0.05`), and all arms of that seed share them.

**Readout.** Seeded k-means (`k = groups`) on the final unit vectors after `K` iterations, scored by the ARI
against the ground truth. The receipt also records the ARI at the checkpoints `0, K/8, K/4, K/2, K`.

**Arms.**

* `candidate`: the correlation coupling.
* `shuffled_control`: the same multiset of coupling values, permuted and kept symmetric.
* `zero_control`: no coupling.
* `reference_correlation_kmeans`: a non-oscillatory reference, k-means on the correlation rows.

**Verdict.** The verdict is `harness.compare` (a paired 95 % t interval over seeds) of candidate against
shuffled-control ARI. The candidate and the shuffled control have the same coupling budget.

**Defaults.**

* Full: `groups=4, per_group=8, samples=256, dim=2, iterations=120, gamma=0.1`.
* Quick: `groups=3, per_group=6, samples=128`.

**Diagnostics.** Per seed, the receipt reports:

* `within_group_R`: the mean ground-truth group coherence.
* `global_R`.
* `max_group_cosine`: the largest cosine between two groups' mean-field directions. A value near 1 means two
  groups ended up in phase.

**Limitation (measured).** With symmetric scalar couplings, the between-group repulsion only constrains the
resultant of the group mean fields. Any arrangement whose resultant is zero is an equilibrium. For four or more
groups that includes two groups settling nearly in phase. The same happens for three groups whose latents are
correlated by chance in a short sample. Groups still synchronise internally (per-seed `within_group_R` between
0.89 and 1.00 in every cell of the sweep below), but the ARI drops, to 0.41 at worst.

The table below is a pilot sweep: mean (min) ARI over seeds 0–9 at 120 iterations, with `noise = 1`. It was
measured on 2026-11-17 with the shipped coupling rule, and a second run later that day gave identical numbers
(the runs are deterministic). In the same sweep, the shuffled-coupling controls stayed within ±0.05 mean ARI
(largest magnitude 0.033).

| groups × per_group, samples | N = 2 | N = 4 | N = 8 |
|---|---|---|---|
| 3 × 6, T = 48 | 0.902 (0.54) | 0.943 (0.60) | 0.948 (0.83) |
| 4 × 8, T = 64 | 0.705 (0.41) | 0.794 (0.58) | 0.791 (0.59) |
| 4 × 8, T = 256 | 0.910 (0.45) | 0.991 (0.91) | 1.000 (1.00) |
| 6 × 6, T = 128 | 0.749 (0.46) | 0.912 (0.77) | 0.916 (0.77) |

Two things drive the result. More samples give more accurate couplings, and larger `N` gives more room to keep
groups apart. At `N = 2` with four groups (4 × 8, seeds 0–9), running longer does not help. At 120, 160, 200 and
240 iterations the mean ARI was 0.910, 0.917, 0.904 and 0.905 with `T = 256`, and 0.913, 0.957, 0.929 and 0.846
with `T = 512`: no upward trend.

The task is also easy for direct clustering of the correlation matrix. Over the same seeds 0–9, the
non-oscillatory reference arm scored mean ARI 1.000 in every sweep cell except 3 × 6 with `T = 48`, where it
scored 0.983 (min 0.83). That is at or above the oscillators in every cell. The benchmark therefore tests that
the oscillator dynamics *implement* binding. It does not show that they beat conventional clustering.

## Lab experiment (`run_experiment`, id `kuramoto_binding`)

`harness.run_extension_experiment('kuramoto_binding', ...)` on the NEUROWEAVE `context_switch` task compares
three arms:

* `baseline`: the unmodified engine and readout.
* `candidate`: `KuramotoExtension` with its 15 features.
* `control`: random leaky traces with the same feature count (the harness's `random_trace` control).

All arms share seeds, streams and the readout protocol, and held-out streams are evaluated frozen. The headline
verdict requires a win over both the baseline and the capacity-matched control. Extension parameters and
harness run parameters (`task`, `steps`, `neurons`, ...) can be overridden. `quick=True` uses the harness quick
sizes: 24 neurons, 24 training steps, and one 12-step held-out stream.

## Public API

```python
MODULE_ID = 'kuramoto_binding'; BENCHMARK_ID = 'kuramoto_binding_benchmark'
EXPERIMENTS = {'kuramoto_binding': run_experiment, 'kuramoto_binding_benchmark': run_binding_benchmark}
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict            # sealed receipt
run_binding_benchmark(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict     # sealed receipt
project_tangent(x, y); natural_frequency_matrices(frequencies, dim)
akorn_step(x, coupling=None, stimulus=None, omega=None, *, gamma=0.1)
akorn_run(x, coupling=None, stimulus=None, omega=None, *, gamma=0.1, iterations=16)
kuramoto_energy(x, coupling=None, stimulus=None); order_parameter(x); group_coherence(x, labels, n_groups=None)
phases(x); phase_locking_value(phase_a, phase_b); count_phase_clusters(phase, *, bins=36, min_fraction=0.1)
shanahan_indices(community_order); adjusted_rand_index(labels_true, labels_pred)
kmeans_labels(points, k, *, seed=0, restarts=8, max_iter=100)
make_binding_task(seed, *, groups=4, per_group=8, samples=64, noise=1.0)
correlation_coupling(features, *, scale=2.0, center=True); shuffle_coupling(coupling, seed)
```

All randomness comes from `numpy.random.default_rng(seed + offset)`. Every public function validates its input
and raises `ValueError` on bad input.

## Connection to Sonapse

* **Live observer.** `kuramoto_binding` is in `spec.OBSERVER_MODULE_IDS`, so a `FrontierSpecV1` can list it in
  `modules`. The observer reads a copy of activity and never writes to the brain, graft bus, actor or any
  shared RNG. This module does not wire it into the live runtime; the runtime hook lives elsewhere.
* **Engine extension.** It influences nothing unless it is attached to a NEUROWEAVE engine with
  `attach_extension`, as in the lab experiment. Even when attached, it only adds readout features.
* **Defaults.** No Sonapse or NEUROWEAVE default, API or checkpoint format changes.

## Not implemented

* Learned couplings, frequencies or step size. In the paper, `J`, `Omega` and `gamma` are learned per layer.
* Convolutional or attention connectivity, and multi-block AKOrN networks.
* The paper's readout block (norms of learned projections, Eq. 6).
* Training through the dynamics, and any of the paper's tasks or results.
* In the extension, couplings that track later synaptic plasticity: `J` is a snapshot taken at `bind`.
* A stimulus in the binding benchmark. A symmetry-breaking stimulus might resolve the group-collision
  degeneracy described above, but it is not tested here.

## Verified references

Each reference below was checked on 2026-11-17 against the arXiv, conference, publisher, Crossref or PubMed
listing (checked twice that day, the second time in a separate session). On 2026-11-20 (review) the AKOrN
title, authors and update rule (Eq. 2, the projection, and the discrete Eqs. 4–5 with `Pi(x) = x / ||x||`)
were re-checked against the arXiv HTML of v3; `akorn_step` matches them.

1. Takeru Miyato, Sindy Löwe, Andreas Geiger, Max Welling. **Artificial Kuramoto Oscillatory Neurons.**
   *International Conference on Learning Representations (ICLR) 2025*, oral, held 24–28 April 2025 at the
   Singapore EXPO. arXiv:2410.13821 (v1 17 October 2024, v2 14 February 2025, v3 16 May 2025; arXiv comment
   "Accepted for Oral presentation at ICLR2025"). https://arxiv.org/abs/2410.13821 ·
   https://iclr.cc/virtual/2025/oral/31776. The update rule (Sec. 3 Eq. 2 and Sec. 4 Eqs. 4–5), the projection
   `Proj_x(y) = y - <y, x> x`, the energy and its Lyapunov conditions (Eq. 3: `J_ij = J_ij I` with
   `J_ij = J_ji`, `Omega_i = Omega`, `Omega c_i = 0`), and the readout (Eq. 6) were checked against the paper's
   method section (arXiv HTML of v3).
2. Murray Shanahan. **Metastable chimera states in community-structured oscillator networks.**
   *Chaos* 20(1), 013108, March 2010 issue (published online 22 February 2010, per Crossref).
   doi:10.1063/1.3305451, https://pubs.aip.org/aip/cha/article/20/1/013108/280633 · arXiv:0908.3881. The
   definitions of `lambda` (variance over time with `1/(T-1)`) and `chi` (variance over communities with
   `1/(M-1)`), and the reference values 1/12, 2/7 (eight communities, half fully synchronised) and 1/7, were
   checked against the text of the arXiv PDF.
3. Jean-Philippe Lachaux, Eugenio Rodriguez, Jacques Martinerie, Francisco J. Varela. **Measuring phase
   synchrony in brain signals.** *Human Brain Mapping* 8(4), 194–208, 1999 (PubMed Central lists the issue date
   as 30 November 1999). doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C,
   https://pubmed.ncbi.nlm.nih.gov/10619414/ (PMID 10619414, PMC6873296). Only the bibliographic details and
   the abstract were verified; the abstract names the method "phase-locking statistics". The full-text PLV
   definition could not be opened in this build, so the module calls its measure a time-averaged *variant*.
4. Lawrence Hubert, Phipps Arabie. **Comparing partitions.** *Journal of Classification* 2(1), 193–218,
   December 1985 (Crossref). doi:10.1007/BF01908075, https://link.springer.com/article/10.1007/BF01908075.
5. David Arthur, Sergei Vassilvitskii. **k-means++: the advantages of careful seeding.** *Proceedings of the
   18th ACM-SIAM Symposium on Discrete Algorithms (SODA '07)*, New Orleans, 7–9 January 2007, SIAM,
   pp. 1027–1035. https://dl.acm.org/doi/10.5555/1283383.1283494.

## Measured results

All runs were measured on 2026-11-17 on Windows 11 on ARM64, with Python 3.12.10, NumPy 2.4 and one BLAS/OpenMP
thread (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`). Receipts are deterministic: the same
arguments give the same digest, and every number below was reproduced exactly by a second run later the same
day. Wall-clock covers the whole call. It is given as the range over those two runs, because the machine was
shared with other jobs running in parallel.

| Run | Seeds | Verdict | Key numbers | Wall-clock |
|---|---|---|---|---|
| `run_binding_benchmark(quick=True)` | 0, 1, 2 | `improved` | ARI: candidate 1.000 / shuffled 0.019 / zero −0.011 / reference 1.000. Candidate − shuffled 0.981, CI [0.764, 1.198] | 0.21–0.50 s |
| `run_experiment(quick=True)` | 0, 1, 2 | `no_detectable_difference` | Held-out accuracy: baseline 0.472 / candidate 0.500 / control 0.500 (chance 0.5). Candidate − baseline +0.028, CI [−0.092, 0.147] | 0.88–1.20 s |
| `run_binding_benchmark()` | 0, 1, 2 | `improved` | ARI: candidate 1.000 / shuffled 0.002 / zero 0.004. `within_group_R` 0.995, `max_group_cosine` 0.633 | 0.30–0.50 s |
| `run_binding_benchmark(seeds=range(8))` | 0–7 | `improved` | ARI: candidate 0.957 (one seed 0.653) / shuffled −0.023. CI [0.86, 1.10] | 0.65–1.01 s |
| `run_binding_benchmark(dim=4)` | 0, 1, 2 | `improved` | ARI: candidate 0.971 (min 0.914) / shuffled 0.050 | 0.25–0.39 s |
| `run_experiment()` | 0, 1, 2 | `no_detectable_difference` | Held-out accuracy: baseline 0.541 / candidate 0.513 / control 0.530 (chance 0.51). Candidate − baseline −0.028, CI [−0.083, 0.027]; candidate − control −0.017, CI [−0.050, 0.017] | 8.99–15.38 s |

On `context_switch` at these sizes, the synchrony features did not help the readout. The quick run is too short
for any arm to learn the task. The binding benchmark shows the mechanism working on a task with a known
partition: correlation-coupled oscillators synchronise by latent group, while shuffled and zero coupling stay at
chance. That is a statement about the dynamics on synthetic data only.

**Observer cost.** One `observe()` with 256 units, 64 oscillators, `N = 2` and 2 iterations took 0.33–0.73 ms
(mean of 300 calls; range over two runs on the shared machine). A `summary()` is about
0.5 KB of JSON (529 bytes measured), and `export_state()` is about 143 KB of JSON with `history = 256`.
