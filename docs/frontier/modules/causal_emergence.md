# `causal_emergence`: effective information and causal emergence

Module file: `synapse/frontier/causal_emergence.py` · tests: `tests/test_frontier_causal_emergence.py` ·
experiment id: `causal_emergence` · live observer: `CausalEmergenceObserver` · kind: `analysis` ·
evidence tier: `synthetic_abstraction`.

The module computes **effective information** (EI) of transition probability matrices (TPMs), its
determinism/degeneracy decomposition, and **causal emergence** (CE): whether a coarse-grained (macro) description
of a system has more EI than the micro description. It is inspired by Hoel, Albantakis & Tononi (PNAS 2013) and
Hoel (Entropy 2017), and by the macro-vs-micro EI experiment of the sibling project `kai9987kai/SONAPSE-ORIGIN`
(`sonapse_origin/theories/causal_emergence.py`, which builds a TPM from a weight matrix and compares one fixed
pairing of states against a 0.05-bit threshold). The sibling project `kai9987kai/GhostInTheMachine` measures
emergence with the Psi criterion of Rosas et al. (2020) instead; its practice of reading an emergence statistic
against a matched null is what the nulls here follow. Nothing here measures consciousness, agency or biological
causation, and no claim is made that NEUROWEAVE brains "have" causal emergence.

## What it computes

**Effective information.** For a row-stochastic TPM `T` over `n` states and maximum-entropy (uniform)
interventions on the current state:

```
E_D = (1/n) sum_i T_i                         (effect distribution)
EI(T) = (1/n) sum_i D_KL(T_i || E_D)  =  H(E_D) - (1/n) sum_i H(T_i)      [bits]
```

i.e. the mutual information between uniform interventions and their effects. Identity and permutation TPMs give
`EI = log2 n`; a TPM with identical rows gives `EI = 0`.

**Decomposition** (formulas as printed in Hoel 2017; Hoel et al. 2013 introduce the determinism and degeneracy
components of effectiveness), with `U` the uniform distribution:

```
determinism   = (1/n) sum_i D_KL(T_i || U) / log2 n  = (log2 n - mean_i H(T_i)) / log2 n
degeneracy    = D_KL(E_D || U) / log2 n             = (log2 n - H(E_D)) / log2 n
effectiveness = determinism - degeneracy,     EI = log2(n) * effectiveness
```

The normalised terms are `None` for a one-state TPM.

**Estimation from data.** `transition_counts(states, n)` counts consecutive pairs of a state sequence, or, with
`next_states=`, interventional pairs `(s_t, s'_t)`. `estimate_tpm(C, alpha=1)` is additive smoothing
`T_ij = (C_ij + alpha) / (sum_k C_ik + alpha n)` (`alpha = 1`: Laplace). Every row receives `alpha n`
pseudo-counts, so with few transitions per state the estimate is pulled towards uniform rows and EI is shrunk,
more for a many-state micro TPM than for a few-state macro TPM.

**Coarse-graining.** A partition of the micro states into macro states (`labels`, one per micro state). The macro
TPM is the effect of intervening uniformly on the micro states of each macro state (Hoel et al. 2013):

```
T_M[A, B] = (1/|A|) sum_{i in A} sum_{j in B} T[i, j]
```

Two candidate families are searched exhaustively, with the micro description itself excluded (so CE can be
negative):

* `state_partition_candidates(n)` / `best_state_partition(T)`: every set partition of `3 <= n <= 10` states with
  `2 <= m < n` blocks (restricted-growth strings; 4138 candidates for `n = 8`).
* `unit_sum_candidates(u)` / `best_unit_coarse_graining(T, u)`: for `2^u` binary micro states (`2 <= u <= 6`),
  every partition of the `u` units into groups except all-singletons; each group's macro variable is its number
  of active units (`m = prod_g (|g| + 1)` macro states; 202 candidates for `u = 6`).

All candidates of a family are evaluated in one sparse matrix product (`macro_ei_batch`), checked against a
direct `scipy.stats.entropy` computation in the tests.

**Causal emergence.** `CE = max_M EI(T_M) - EI(T)` over the candidates.

**Nulls.** Both nulls re-run the search for the best coarse-graining in every draw, so the selection of the
maximum is part of the null. Each reports `z = (CE - mean_null) / sd_null` and the one-sided rank p-value
`(1 + #{CE_null >= CE}) / (1 + draws)`.

* *Successor block permutation* (`successor_block_null`): the successor sequence `s_{1..T}` is block-permuted
  (blocks of `round(T^(1/3))` steps by default) against the unchanged `s_{0..T-1}`. Visit counts and the successor
  marginal are kept; lag-1 dependence is broken.
* *Independent units* (`independent_units_null`, binary-unit data only): each unit's bit sequence is circularly
  shifted by its own random offset, which keeps each unit's occupancy and autocorrelation but makes the units
  independent.

Neither null tests "the true macro EI exceeds the true micro EI". Because smoothing shrinks the micro TPM more
than a macro TPM, the estimated CE of *any* temporally dependent sequence with few transitions per micro state
lies above the successor null, including independent slow units whose exact CE is negative (measured below).
Against the independent-units null, a large `z` says that the coarse-graining gains from coordination between
units beyond what the units' own dynamics give.

## Live observer (`CausalEmergenceObserver`)

`CausalEmergenceObserver(n_units, *, seed=7, history=256, units=6, alpha=1.0, null_draws=19, block=None,
min_transitions=None, **params)`; `CausalEmergenceObserver.from_spec(n_units, spec, **params)` takes `seed`
and `history` from a `FrontierSpecV1`. The spec parameters of other observers (`sync_pairs`, `sae_features`,
`sae_k`, `fc_window`, `max_observed_units`, `observe_every`) are accepted and ignored; any other keyword, and
any out-of-range value, raises `ValueError`.

* `u = min(units, n_units)` units (a seeded, sorted sample when `n_units > units`) are binarised online: a value
  is on when it is above the median of that unit's last `history` raw values (including the new one). A value
  tied with the median is on when more of the window lies strictly below the median than above it, so a
  discrete 0/1 unit that is on most of the time is binarised as itself rather than as constant 0 (a constant unit
  is off). The `2^u` micro states of the last `history` observations are kept in a ring buffer, so transition
  counts are bounded.
* `summary()` estimates the micro TPM from the buffered transitions (smoothing `alpha`) and reports micro EI with
  determinism/degeneracy/effectiveness, the best count-per-group coarse-graining, its macro EI and
  determinism/degeneracy, and `ce`. Once `min_transitions` transitions exist (default `min(32, history - 1)`) it
  adds both nulls with `null_draws` draws each, seeded by `(seed, observed steps)`; the smallest attainable p is
  `1 / (null_draws + 1)`. Results are a pure function of the buffered state and are cached per step, so they do
  not depend on how often `summary()` is called.
* `summary()` keys: `module`, `evidence_tier`, `n_units`, `analysed_units`, `unit_sample`, `history`,
  `micro_states`, `alpha`, `null_draws`, `min_transitions`, `observed_steps`, `buffered`, `transitions`,
  `transitions_per_micro_state`, `visited_states`, `ei_micro`, `determinism_micro`, `degeneracy_micro`,
  `effectiveness_micro`, `ei_macro`, `determinism_macro`, `degeneracy_macro`, `macro_partition` (groups of
  observed-unit indices), `macro_states`, `ce`, `null_ready`, `null_block`, successor null `null_ce_mean`,
  `null_ce_sd`, `ce_excess`, `ce_z`, `ce_p`, and independent-units null `indep_ce_mean`, `indep_ce_sd`,
  `ce_excess_indep`, `ce_z_indep`, `ce_p_indep`. Metric values are `None` before the first transition and the
  null fields are `None` until `null_ready`.
* `export_state()` / `import_state(state)` round-trip through JSON; `import_state` checks the schema, the
  configuration, the unit sample, buffer shapes and finiteness, the counters and ring position, that nothing is
  stored beyond the observed steps, and that stored states are the median-binarisation of the stored raw values
  (all of them while every raw value is still buffered, the newest one afterwards) before replacing anything.
  `reset()` returns to the empty state.
* Cost: one `summary()` with nulls evaluates `2 * null_draws + 1` TPMs against 202 candidates at `u = 6`
  (about 0.4–0.9 s measured on 2026-11-20 at history 256–2048); `observe()` costs one median per unit. In its
  per-step telemetry the frontier hub (`runtime.py`) runs full `causal_emergence` summaries only for the first
  `heavy_every` (32) observed steps and on multiples of it.

**Reading the observer.** At the defaults (6 units, 64 micro states, history 256) there are about 4
transitions per micro state against 64 Laplace pseudo-counts per row: measured EI values are a small fraction of
the exact ones and the best grouping is not stable across seeds (on a six-unit pair-count copy ring, measured
2026-11-20: micro EI 0.03–0.05 bits against an exact 3.29 bits, best grouping varying with the seed at history
256, the pair grouping found for 6 of 6 seeds at history 1024 with `ce_z_indep` 16.6–23.3). Use fewer units, a
longer history or a smaller `alpha` when the estimate matters, and read `ce` against `ce_z_indep`.

## Lab experiment (`run_experiment`, `EXPERIMENTS = {'causal_emergence': run_experiment}`)

`run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides)` returns a sealed `kind: 'validation'` receipt
with `verdict` in `('pass', 'fail')` and 22 named checks.

Exact checks (no sampling):

| Check | Known answer |
|---|---|
| `identity_ei_equals_log2_n`, `permutation_ei_equals_log2_n`, `uniform_ei_is_zero` | `n` in `sizes` (2–32): `EI = log2 n`, determinism 1, degeneracy 0; uniform rows: EI, determinism and degeneracy 0 |
| `ei_equals_size_times_effectiveness` | `EI = log2(n) * effectiveness` on every example TPM |
| `hoel2017_M1_M2_M3_exact` | Hoel (2017) M1, M2, M3: 2, 1, 0 bits |
| `hoel2017_chain8_micro_closed_form` | 8-state chain (seven states mix uniformly, one absorbing): `EI = 3 - (7/8) log2 7 = 0.5436` bits |
| `hoel2017_chain8_matches_printed` | Printed EI 0.55 bits, effectiveness 0.18, CE 0.45 bits, within 0.01 / 0.005 / 0.01 (the closed form gives 0.5436 and 0.4564 bits, so two-decimal equality is not expected) |
| `hoel2017_chain8_search_finds_1_bit_seven_plus_one` | Exhaustive search over 4138 partitions finds the macro `{s1..s7}, {s8}` with 1 bit |
| `hoel2017_chain8_varied_matches_printed` | Unequal-row variant: printed 0.81 bits micro, 1 bit macro, same partition |
| `hoel2017_channel4_matches_printed` | 4-state channel: printed EI 0.81 bits, `H(X|Y)` 1.19 bits, macro code 1 bit with `{x1, x2, x3}, {x4}` |
| `and_gate_network_matches_printed` | Six AND gates in three pairs, each pair reading the next (wiring reconstructed from the text of Hoel et al. 2013 / Hoel 2017): micro EI 2.43 bits (closed form `3 H(1/4, 3/4)`), determinism 1, degeneracy 0.59, effectiveness 0.41; the pair-AND macro is three COPY elements with 3 bits; CE 0.57 bits |
| `weakly_symmetric_tpm_has_no_emergence` | Noisy permutations `(1 - e) P + e / n` (n = 5, 8): no partition has CE > 0 (uniform inputs achieve the capacity of a weakly symmetric channel) |
| `unit_pair_copy_exact_ce_positive_for_pair_counts` | Constructed six-unit ring whose pair *counts* copy the next pair's count (noise 0.1, count 1 realised by either unit): best count-per-group grouping is the three pairs, exact CE > 0 |
| `unit_independent_exact_ce_negative` | Six independent units that keep their state with probability 0.9: micro EI `6 (1 - H(0.9))`, every grouping has CE < 0 |

Sampled checks (every seed):

| Check | Criterion |
|---|---|
| `estimated_tpm_recovers_ei_and_ce` | Laplace-smoothed TPM from `estimation_samples` interventional samples of the 8-state chain: micro EI and CE (seven-plus-one) within `estimation_tolerance` (0.05) |
| `iid_null_false_positive_rate_at_most_0.25`, `iid_null_mean_z_within_1` | i.i.d. 8-state sequences: successor-null `p <= 0.05` in at most 25 % of replicates, mean `z` within ±1 (CE relative to the null is about 0) |
| `noisy_chain_ce_positive_and_above_successor_null` | Ergodic noisy version of the 8-state chain (noise 0.1, exact CE 0.278 bits for seven-plus-one): estimated CE > 0, `z > 3`, `p <= 0.05` |
| `permutation_control_ce_negative_and_not_above_successor_null` | Noisy 8-state permutation of the same length (temporally dependent, exact CE ≤ 0): estimated CE < 0, `p > 0.05` |
| `unit_pair_copy_above_independent_units_null_with_pairs_recovered` | Six units sampled from the pair-count ring: independent-units `z > 3`, `p <= 0.05`, pair grouping recovered |
| `unit_independent_units_null_calibrated` | Independent slow units: independent-units null flags at most half of the replicates, mean `z` within ±2 |
| `neuroweave_values_finite` | The descriptive part is finite |

How often the successor null flags the independent slow units is reported (`independent_successor_null_*`),
not checked: it is the documented bias above.

**NEUROWEAVE report (descriptive).** For each mechanism set (default `baseline = ()` and
`dynamics = (sparse_code, synaptic_delays, membrane_dynamics)`) and seed, a frozen engine (`learn=False`,
`clustered` topology) runs on the task stream; after burn-in a `CausalEmergenceObserver` over `units = 4` output
units (16 micro states, chosen so the run gives about 11 (quick) or 34 (default) transitions per micro state)
reports EI, CE and both nulls. No verdict is drawn from these values.

The receipt also holds `config`, `estimated_cost` (TPM evaluations of each part, engine steps),
`measured_wallclock_s` (excluded from the digest) and `notes`.

## Public API

`MODULE_ID`, `CausalEmergenceObserver`, `entropy_bits`, `as_tpm`, `effective_information`, `ei_decomposition`,
`transition_counts`, `estimate_tpm`, `median_binarize`, `encode_bits`, `decode_states`, `coarse_grain_tpm`,
`set_partitions`, `unit_sum_labels`, `unit_sum_candidates`, `state_partition_candidates`, `macro_ei_batch`,
`best_state_partition`, `best_unit_coarse_graining`, `causal_emergence`, `successor_block_null`,
`independent_units_null`, `hoel2017_examples`, `and_gate_network`, `pair_count_copy_tpm`,
`noisy_permutation_tpm`, `independent_flip_tpm`, `PRINTED_VALUES`, `run_experiment`, `EXPERIMENTS`.
All randomness uses `numpy.random.default_rng(seed + offset)`; no global NumPy or torch state is touched
(tested).

## Connection to Sonapse

* The live observer reports EI/CE of a few units of the running brain without touching it (frontier observer
  hub, `causal_emergence` in `FrontierSpecV1.modules`).
* The lab experiment runs NEUROWEAVE engines through `neuroweave.experiments.make_spec` and
  `neuroweave.tasks.make_stream`; the successor null reuses `connectomics.block_permutation`.

## Not implemented

* Only uniform (maximum-entropy) interventions; no other intervention distributions, no "black-boxing" or
  removal of units, no temporal coarse-graining, no search over partitions of more than 10 states or of more
  than 6 binary units.
* No integrated information (Phi) and no Rosas et al. Psi / synergy-based emergence measures.
* TPMs are estimated from observational sequences with additive smoothing; there is no bias correction of EI.
* The six-AND-gate wiring is reconstructed from the papers' text; the figure itself was not used.

## Verified references

Checked on 2026-11-20 (Crossref metadata; PMC full text for the 2013 paper; the arXiv v1 text of the 2017 paper
for every printed value used above).

* Erik P. Hoel, Larissa Albantakis, Giulio Tononi. **Quantifying causal emergence shows that macro can beat
  micro.** *Proceedings of the National Academy of Sciences* 110(49), 19790–19795, issue of 3 December 2013
  (online 18 November 2013). <https://doi.org/10.1073/pnas.1314922110> — EI as the mutual information between
  uniform interventions and effects, macro interventions by averaging over grouped micro states; Fig. 4 values:
  micro EI 2.43 bits, effectiveness 0.4, determinism coefficient 1, degeneracy coefficient 0.6, macro 3 bits,
  CE 0.57 bits.
* Erik P. Hoel. **When the Map Is Better Than the Territory.** *Entropy* 19(5), 188, 26 April 2017.
  <https://doi.org/10.3390/e19050188> (preprint arXiv:1612.09592, v1 of 30 December 2016) — determinism,
  degeneracy, effectiveness and `EI = size * eff`; M1/M2/M3 (2, 1, 0 bits); the 8-state chain (eff 0.18,
  0.55 bits, macro 1 bit, CE 0.45 bits); its unequal-row variant (0.81 bits); the AND-gate network (2.43 bits,
  eff 0.41, degeneracy 0.59, macro 3 bits); the channel example (`H(X|Y)` 1.19 bits, EI 0.81 bits, macro 1 bit).
* Fernando E. Rosas, Pedro A. M. Mediano, Henrik J. Jensen, Anil K. Seth, Adam B. Barrett, Robin L.
  Carhart-Harris, Daniel Bor. **Reconciling emergences: An information-theoretic approach to identify causal
  emergence in multivariate data.** *PLOS Computational Biology* 16(12), e1008289, 21 December 2020.
  <https://doi.org/10.1371/journal.pcbi.1008289> — the Psi criterion used by GhostInTheMachine; not implemented
  here.
* Sibling projects: `kai9987kai/SONAPSE-ORIGIN` and `kai9987kai/GhostInTheMachine` (read-only clones; ideas,
  not code, reused).

## Measured results

Measured on 2026-11-20, Windows 11 on ARM64, Python 3.12, NumPy 2.4, one BLAS thread
(`OPENBLAS_NUM_THREADS=1`), seeds 0, 1, 2, on a machine shared with other jobs (wall-clock varies with load).

| Run | Verdict | Key numbers | Wall-clock |
|---|---|---|---|
| `run_experiment(quick=True)` | `pass` (22/22) | Exact: chain8 0.5436 → 1.000 bits (CE 0.4564, 4138 partitions), variant 0.8059 → 1.000, channel 0.8113 → 1.000 (`H(X|Y)` 1.1887), AND gates 2.4338 → 3.000 (CE 0.5662, degeneracy 0.5944); noisy permutations best CE −0.162 (n = 5) and −0.103 (n = 8); pair-count ring exact CE +0.207 with the pairs, independent units −0.280. Estimation (4000 samples): EI error 0.023–0.027, CE error 0.035–0.039. i.i.d. null: 0 of 6 replicates flagged, mean z −0.34. Noisy chain (384 steps): CE 0.151 / 0.159 / 0.212 (exact 0.278), z 38.7–67.9, p 0.025 (39 draws). Permutation control: CE −0.102 to −0.143, p 1.0. Units (512 steps, 19 draws): pairs recovered 3/3, min independent-units z 11.2; independent units flagged 0/3 by the independent-units null (mean z −0.69) but 3/3 by the successor null (mean z 8.26) | 7.3 s (validation 4.2 s, NEUROWEAVE 3.0 s) on a lighter-loaded re-run; 14.4 s (pytest) to 24.6 s (standalone; validation 14.7 s, NEUROWEAVE 9.8 s) under load. Two runs in one process gave the same digest |
| `run_experiment()` | `pass` (22/22) | Estimation (16000 samples): EI error 0.008–0.009, CE error 0.012. i.i.d. null: 1 of 30 replicates flagged, mean z 0.003. Noisy chain (512 steps): CE 0.151 / 0.169 / 0.223, z 47.7–79.6, p 0.005 (199 draws). Permutation control: CE −0.090 to −0.160, p 1.0. Units (1024 steps, 49 draws, 4 replicates): pairs recovered 12/12, min z 19.6; independent units flagged 2/12 by the independent-units null (mean z 0.33), 12/12 by the successor null (mean z 7.87) | 133.5 s (validation 111.7 s, NEUROWEAVE 21.8 s) |

NEUROWEAVE (descriptive; 4 units, `clustered`, means over seeds):

* quick (24 neurons, 200 steps, 24 burn-in, 175 transitions): `baseline` micro EI 0.257, macro 0.308, CE 0.051,
  independent-units z 4.96 (per-seed p 0.05 / 0.10 / 0.15); `dynamics` micro EI 0.223, macro 0.236, CE 0.013,
  z 2.53 (p 0.05 / 0.30 / 0.05).
* default (32 neurons, 600 steps, 50 burn-in, 549 transitions): `baseline` micro EI 0.745, macro 0.764,
  CE 0.019, independent-units z 2.45; `dynamics` micro EI 0.684, macro 0.665, CE −0.019, z 1.13.

The sign of the NEUROWEAVE CE changes between the quick and default sizes, and per-seed p-values are mostly at
or above the smallest attainable value; these numbers do not show causal emergence in NEUROWEAVE brains or a
difference between the mechanism sets.
