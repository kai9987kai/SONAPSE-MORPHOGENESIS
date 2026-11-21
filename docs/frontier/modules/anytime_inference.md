# `anytime_inference`: anytime-valid inference

Module file: `synapse/frontier/anytime.py` · tests: `tests/test_frontier_methodology.py` ·
experiment id: `anytime_inference` (kind `validation`, verdict `pass`/`fail`) · no live observer ·
evidence tier: `synthetic_abstraction`.

The module provides two betting-based tools that stay valid when you look at the data after every
observation and stop whenever you like:

* a **confidence sequence** for the mean of a bounded stream;
* a two-sided **e-process** for a zero mean of bounded paired differences.

Both are combined into `anytime_compare`, a paired comparison that returns the lab's verdict vocabulary.
The constructions are *inspired by* Waudby-Smith & Ramdas (2024), and the framing follows Ramdas, Grünwald,
Vovk & Shafer (2023). The module re-implements the constructions in NumPy and checks their guarantees in
seeded simulations. It reproduces none of the papers' experiments or figures.

## What it computes

### Betting confidence sequence (`betting_confidence_sequence`)

Observations `x_t` in `[lo, hi]` are rescaled to `X_t = (x_t - lo) / (hi - lo)` in `[0, 1]`. For a candidate
mean `m` (paper Sec. 4.4, Eqs. 24-26):

```
K+_t(m) = prod_{i<=t} (1 + lam+_i(m) (X_i - m))         K-_t(m) = prod_{i<=t} (1 - lam-_i(m) (X_i - m))
K±_t(m) = max{ theta K+_t(m), (1 - theta) K-_t(m) }                                   (hedged capital, Eq. 24)
lam+_t(m) = min(lam_t, c / m)      lam-_t(m) = min(lam_t, c / (1 - m))                (truncation, Eq. 25)
lam_t     = sqrt( 2 log(2/alpha) / (sigma2_{t-1} t log(1 + t)) )                     (plug-in bet, Eq. 26)
mu_t      = (1/2 + sum_{i<=t} X_i) / (t + 1)
sigma2_t  = (1/4 + sum_{i<=t} (X_i - mu_i)^2) / (t + 1),   sigma2_0 = 1/4
CS_t      = intersection_{i<=t} { m in [0, 1] : K±_i(m) < 1/alpha }                    (Theorem 3)
```

* **Validity.** Theorem 3 of the paper: for each fixed `m = mu` the capital processes are nonnegative
  martingales, so by Ville's inequality the true mean leaves the set at *some* time with probability at
  most `alpha`. The running intersection keeps the same guarantee.
* **Grid search.** The set at each time is an interval (Theorem 3), so the module evaluates `K±` on the grid
  `m = 0, 1/g, ..., 1` (`g = grid_size`, default 1000), as the paper suggests. Each bound is the first
  *excluded* grid point outside the run of included points. The result is a superset of the exact set,
  conservative by at most one grid step `(hi - lo) / g`.
* **No grid point inside.** If no grid point is inside, the bounds bracket the grid minimisers of `K±`.
  `K+` is non-increasing and `K-` non-decreasing in `m`, so `K±` is quasiconvex and the exact set lies in
  that bracket.
* **Returns.** `lower` and `upper` (running intersection), `lower_raw` and `upper_raw` (the per-time set),
  `final` and `empty`. The intersection is empty with probability at most `alpha`.

### Paired e-process (`paired_e_process`)

The null hypothesis is `H0: E[D_t | D_1..D_{t-1}] = 0`, for differences with `|D_t| <= B`. Map each
difference to `X_t = (1 + D_t / B) / 2` in `[0, 1]` and test `m = 1/2`:

```
lam_t  = clip( (mu_{t-1} - m) / (sigma2_{t-1} + (mu_{t-1} - m)^2), -c/(1-m), c/m )   (aGRAPA, paper App. B.3)
lam+_t = max(lam_t, 0)        lam-_t = max(-lam_t, 0)
E_t    = (K+_t(m) + K-_t(m)) / 2                     (the paper's mixture M±_t with theta = 1/2, Sec. 4.4)
```

* **Estimators.** `mu` and `sigma2` are the Eq. 26 estimators. App. B.3 writes the bet with `mu_{t-1}` and
  `sigma2_{t-1}` without restating them; the paper defines that notation only in Eq. 26, which the module
  follows.
* **Sign split.** The bet is split by sign: `K+` gathers evidence for a positive mean, `K-` for a negative
  one.
* **Validity.** Every factor is at least `1 - c > 0` and the bets are predictable. Under `H0`, `K+`, `K-`
  and `E` are therefore nonnegative martingales that start at 1. Ville's inequality then gives
  `P(exists t: E_t >= 1/alpha) <= alpha` at every stopping time, so optional stopping does not break it.
* **Finite outputs.** Reported values are capped at `exp(690)` so they stay finite. `log_e_process` is
  uncapped.

### Paired comparison (`anytime_compare`)

`D_t = candidate_t - reference_t` with a known bound `|D_t| <= bound`. The e-process is checked after every
observation.

* **Rejection.** `rejected` is true at the first time `E_t >= 1/alpha`. `stopped_at` is that time (1-based)
  and `e_value` is the stopped value. `e_value_final` is the value at the last observation.
* **Direction.** The direction comes from the larger one-sided process at the stopping time. Declaring
  `positive` needs `K+_tau >= E_tau >= 1/alpha`. When the true conditional mean is `<= 0`, `K+` is a
  nonnegative supermartingale (`lam+ >= 0`), so a wrong-direction declaration also has probability at most
  `alpha`.
* **Verdict.** `improved` or `worse` depends on the direction and `higher_is_better`. Without a rejection
  the verdict is `no_detectable_difference`.
* **CS bounds.** `cs_lower` and `cs_upper` are the hedged betting CS for the mean difference on
  `[-bound, bound]` at the last observation, at level `1 - alpha`. The verdict comes from the e-process
  only. The CS is reported beside it, with `cs_excludes_zero`, and the two can disagree near the threshold.

### Invalid baseline (`naive_peeking_t_test`)

A two-sided one-sample t-test of mean 0 repeated at every look `n >= min_n`. It reports whether *any* look
rejected and whether the single final look rejected. The tests check it against `scipy.stats.ttest_1samp`
at every look.

## Validation experiment (`run_experiment`, `EXPERIMENTS = {'anytime_inference': run_experiment}`)

For every seed the experiment draws `replicates` independent sequences of length `horizon` per scenario.
Every statistic is evaluated after every observation. Each scenario has its own generator,
`np.random.default_rng(seed + offset)`.

| scenario | data | used for |
|---|---|---|
| `null_paired_bernoulli` | `D = Bern(0.6) - Bern(0.6)`, independent, bound 1 | type-I error under optional stopping |
| `null_skewed_continuous` | `D = Beta(0.5, 4) - 1/9` (mean exactly 0, skewed), bound 1 | type-I error under optional stopping |
| `effect` | `D = Bern(0.75) - Bern(0.5)` (mean 0.25), bound 1 | power, stopping time, wrong-direction rate |
| `cover_beta` | `X ~ Beta(2, 5)` on `[0, 1]` (mean 2/7) | time-uniform CS coverage and width |
| `cover_bernoulli` | `X ~ Bern(0.3)` | time-uniform CS coverage and width |

The checks are pooled over seeds. Monte-Carlo slack is `mc_z = 3` standard errors. The verdict is `pass`
only if all 12 checks hold:

* `e_process_type1_le_alpha_{null}`: the rate of `sup_t E_t >= 1/alpha` is at most `alpha + 3 se`.
* `naive_peeking_t_exceeds_alpha_{null}`: the repeated t-test (looks from `n = 10`) rejects more often than
  `alpha` by more than 3 se.
* `mean_final_e_value_le_1_{null}`: the sample mean of `E_horizon` is at most `1 + 3 se`.
  * The expectation is exactly 1, but the terminal value is heavy-tailed.
  * Rare large values carry the mean, so the sample mean usually falls below 1, and the check is one-sided.
  * A 200,000-path run of the paired-Bernoulli null (the experiment's generator, `D = Bern(0.6) - Bern(0.6)`,
    `np.random.default_rng(12345)`, `c = 0.5`, horizon 200; measured 2026-11-20) gave sample means 1.000,
    0.9996, 1.0006, 1.007, 1.339 and 0.763 at `t = 5, 10, 20, 50, 100, 200`.
  * The `t = 100` mean is above 1 only because one path reached `E_100 ≈ 82,000`, which alone adds 0.41 to
    the mean. No path came close to that at `t = 200` (largest value about 16,000), and the mean fell to 0.763.
  * The tests check that the expectation is exactly 1 by enumerating every path of short sequences instead.
* `power_on_clear_effect`: the rejection rate under `effect` is at least `power_min = 0.8`.
* `wrong_direction_rate_le_alpha`: wrong-direction rejections under `effect` are at most `alpha`.
* `cs_uniform_coverage_ge_1_minus_alpha_{beta,bernoulli}`: the true mean stays inside the
  running-intersection CS at every one of the `horizon` times with frequency at least `1 - alpha - 3 se`.
* `cs_shrinks_{beta,bernoulli}`: the mean final width is below the mean width at `t = 20`.

Defaults: `alpha = 0.05`, `replicates = 400`, `horizon = 500`, `grid_size = 400`, `c = 0.5`, `theta = 0.5`,
`naive_min_n = 10`, seeds `(0, 1, 2)`. `quick=True` uses `replicates = 100`, `horizon = 200` and
`grid_size = 200`. Unknown overrides raise `ValueError`, and so do sizes above the memory bounds
(`replicates × horizon <= 4,000,000`, `replicates × (grid_size + 1) <= 20,000,000`).

## Public API

```python
betting_confidence_sequence(x, lo, hi, alpha=0.05, *, grid_size=1000, c=0.5, theta=0.5) -> dict
paired_e_process(differences, bound, *, c=0.5) -> dict
anytime_compare(candidate, reference, *, bound, alpha=0.05, higher_is_better=True, c=0.5, grid_size=1000) -> dict
    # keys: e_value, e_value_final, rejected, stopped_at, direction, cs_lower, cs_upper, cs_excludes_zero,
    #       cs_empty, mean_difference, verdict in ('improved', 'no_detectable_difference', 'worse'), ...
naive_peeking_t_test(differences, alpha=0.05, *, min_n=2) -> dict
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> sealed receipt
EXPERIMENTS = {'anytime_inference': run_experiment}
```

* **Input validation.** Inputs are validated before any computation, and bad input raises `ValueError`.
  This covers non-finite or out-of-range values, `alpha`, `c` or `theta` outside `(0, 1)`, a mismatched
  length and `|difference| > bound`.
* **Determinism.** Everything is deterministic. The module uses NumPy, plus `scipy.special.stdtrit` for the
  t-test baseline. It never touches the global NumPy RNG and never imports torch.

## Connection to Sonapse

* **Not wired in.** `harness.compare` decides lab verdicts with a paired 95 % t interval over a fixed set of
  seeds. That interval is valid only at a sample size fixed in advance.
* **When to use it.** `anytime_compare` is the anytime-valid alternative for studies that add seeds or
  evaluation episodes and check results as they arrive. Examples are held-out correctness streams of two
  NEUROWEAVE arms, or paired per-seed scores with a known range.
* **How to use it.** The caller supplies the bound on `|candidate - reference|`. For 0/1 correctness or
  accuracies it is 1.
* **Scope.** The module does not change `harness.compare` or any other module's verdict. Using it there
  would mean editing the shared harness.

## Not implemented

* **From Waudby-Smith & Ramdas (2024).**
  * The other betting strategies: GRAPA root-finding, LBOW, ONS-m, dKelly, ConBo and SRP (App. B).
  * Sampling without replacement (Sec. 5.2).
  * The fixed-sample-size CI tuning of Remark 3 / Eq. 27.
  * The closed-form predictable plug-in Hoeffding and empirical-Bernstein CSs (Sec. 3).
  * Root-finding in place of the grid.
  * The point estimator `argmin_m K±_t(m)`.
* **Not covered by the guarantees.**
  * The CS assumes a constant conditional mean.
  * Streams whose conditional mean drifts are not covered (the paper's Sec. E.5).
  * The e-process needs `E[D_t | past] = 0` under the null.
* **From Ramdas et al. (2023).**
  * Composite nulls beyond a zero mean.
  * Universal inference, e-value merging and p-to-e calibration.

## Verified references

* Ian Waudby-Smith and Aaditya Ramdas. **Estimating means of bounded random variables by betting.**
  *Journal of the Royal Statistical Society Series B: Statistical Methodology* 86(1):1–27, February 2024
  (read before the Royal Statistical Society, with discussion on pp. 28–61). doi:10.1093/jrsssb/qkad009.
  <https://academic.oup.com/jrsssb/article/86/1/1/7043257> · preprint arXiv:2010.09686 (v1 19 October 2020;
  v7 25 August 2022), <https://arxiv.org/abs/2010.09686>.
  * Bibliographic data checked against the publisher and arXiv pages on 2026-11-20.
  * The following were checked against the text of arXiv v7 the same day; section, equation and appendix
    numbers in this document follow v7:
    * the hedged capital process, Sec. 4.4, Eq. 24, and the mixture `M±`;
    * the truncation `|lam~| ∧ c/m`, `∧ c/(1-m)` with `c = 1/2` or `3/4`, Eq. 25;
    * Theorem 3 (valid CS, interval for each `t`, running intersection);
    * the plug-in bet and estimators, Eq. 26;
    * the grid-search remark, Sec. 4.4;
    * quasiconvexity, Remark 2;
    * aGRAPA, App. B.3, truncated to `[-c/(1-m), c/m]`.
* Aaditya Ramdas, Peter Grünwald, Vladimir Vovk and Glenn Shafer. **Game-theoretic statistics and safe
  anytime-valid inference.** *Statistical Science* 38(4):576–601, November 2023. doi:10.1214/23-STS894.
  <https://projecteuclid.org/journals/statistical-science/volume-38/issue-4/Game-Theoretic-Statistics-and-Safe-Anytime-Valid-Inference/10.1214/23-STS894.full>
  · preprint arXiv:2210.01948. Bibliographic data was checked against Project Euclid and the Royal Holloway
  repository on 2026-11-20. The CWI repository lists the page range as 576–597, and Project Euclid gives
  576–601.

## Measured results

All numbers were measured on 2026-11-20 (UTC) on Windows 11 ARM64 with Python 3.12.10 and NumPy 2.4.3, with
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Other agents shared the CPU, so wall-clock is
indicative only. Rates are pooled over seeds `(0, 1, 2)`. `se` is the binomial Monte-Carlo standard error.

**Quick mode** (`run_experiment(quick=True)`): 3 × 100 sequences per scenario, horizon 200, grid 200,
`alpha = 0.05`, `c = theta = 0.5`.

| | value |
|---|---|
| verdict | **pass** (12 of 12 checks) |
| paired-Bernoulli null: e-process / repeated t-test / single final t-test type-I | 0.020 (6/300) / 0.360 (108/300) / 0.033 (10/300) |
| skewed null: e-process / repeated t-test / single final t-test type-I | 0.000 (0/300) / 0.410 (123/300) / 0.057 (17/300) |
| mean final e-value (paired Bernoulli / skewed) | 0.52 (max 4.56) / 1.00 (max 4.89) |
| effect (mean difference 0.25): power, wrong direction, median stopping time | 0.98 (294/300), 0, 59 observations |
| CS uniform coverage, Beta(2, 5) / Bernoulli(0.3) | 0.990 (297/300) / 0.957 (287/300) |
| CS mean width at t = 20 → t = 200, Beta / Bernoulli | 0.323 → 0.072 / 0.515 → 0.212 |
| digest (prefix) | `a07208ae621e340b` (identical across separate processes) |
| measured wall-clock | 3.1–3.9 s |

**Default size** (`run_experiment()`): 3 × 400 sequences per scenario, horizon 500, grid 400, same `alpha`,
`c` and `theta`.

| | value |
|---|---|
| verdict | **pass** (12 of 12 checks) |
| paired-Bernoulli null: e-process / repeated t-test / single final t-test type-I | 0.018 (22/1200, se 0.004) / 0.409 (491/1200) / 0.058 (69/1200) |
| skewed null: e-process / repeated t-test / single final t-test type-I | 0.001 (1/1200) / 0.457 (548/1200) / 0.047 (56/1200) |
| mean final e-value (paired Bernoulli / skewed) | 0.49 (max 19.5) / 0.96 (max 14.9) |
| effect (mean difference 0.25): power, wrong direction, median stopping time | 1.00 (1200/1200), 0, 57.5 observations |
| CS uniform coverage, Beta(2, 5) / Bernoulli(0.3) | 0.985 (1182/1200) / 0.966 (1159/1200) |
| CS mean width at t = 20 → t = 500, Beta / Bernoulli | 0.319 → 0.045 / 0.510 → 0.148 |
| running intersection empty at t = 500 | 0 / 1 of 1200 (counted as a coverage miss) |
| digest (prefix) | `7b05678cbd3aa5b7` (identical in two separate runs) |
| measured wall-clock | 81.9–94.1 s, almost all of it in the 401-point grid search |

Reading:

* **Peeking breaks the t-test.** At default size the repeated t-test rejects a true null in 41–46 % of
  sequences when it looks after every observation. A single t-test at the final time stays near the nominal
  5 %.
* **The e-process stays valid.** It rejected at most 1.8 % of null sequences under the same monitoring, and
  it still detected the 0.25 mean difference in every sequence (median 57.5 paired observations).
* **The betting CS covers the mean.** It covered the true mean at all 500 times in 98.5 % and 96.6 % of
  sequences. That is above the stated 95 %, as expected from a conservative construction.
* **Limits of these checks.** They test the stated guarantees on these generators only. They say nothing
  about data that violate the assumptions.
