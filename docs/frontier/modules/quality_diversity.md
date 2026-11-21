# `quality_diversity`: quality-diversity architecture search

Module file: `synapse/frontier/quality_diversity.py` · tests: `tests/test_frontier_quality_diversity.py` ·
experiment id: `quality_diversity` · kind: discovery (a search over NEUROWEAVE configurations; no live observer) ·
evidence tier: `synthetic_abstraction`.

The module searches a small, declared space of NEUROWEAVE configurations with a **MAP-Elites** archive (Mouret &
Clune, 2015): a grid over two behaviour descriptors that keeps the best configuration found in each cell. Parents
are drawn from the **Darwin Gödel Machine** (DGM) parent distribution (Zhang et al., 2025), which favours fit
elites that have few verified children. Every candidate passes a **verification gate** before it may enter the
archive: its sealed receipt must verify and match what the search asked for, and every k-th candidate is run again
and must reproduce the same receipt digest. Rejected candidates are logged with reasons, every evaluated candidate
is a node of a lineage DAG, and the archive exports to JSON with a SHA-256 integrity hash.

The design also borrows from two sibling `kai9987kai` projects. SONAPSE-ORIGIN contributes a 2-D MAP-Elites grid
with edge-clipped bins, NSGA-II Pareto sorting, and mutation plus crossover. sonapse-evo contributes structural
operators with preconditions, audit receipts, a failure archive with reasons, and verify-before-commit
transactions.

The module is *inspired by* these sources and reproduces none of them. DGM evolves self-modifying coding agents
that are checked on coding benchmarks. Here each candidate is a tiny NEUROWEAVE engine (16–32 neurons) trained for
16–48 steps on a synthetic temporal task, and a run evaluates 5–48 candidates per arm.

## What it computes

### Genome and search space

```
genome  = (mechanisms ⊆ M, topology ∈ T, neurons ∈ N, sparsity ∈ S)
M       = 10 NEUROWEAVE switches (SEARCH_MECHANISMS): structural_encoder, sparse_code, reward_plasticity,
          synaptic_delays, homeostatic_governor, context_attention, flyhash_projection, adaptive_routing,
          hierarchical_temporal_memory, membrane_dynamics
T       = all 12 NEUROWEAVE topologies (synapse.neuroweave.spec.TOPOLOGIES)
N       = (16, 24, 32) neurons            S = (0.05, 0.15, 0.3, 0.5) k-winners-take-all fraction
|space| = 2^10 · 12 · 3 · 4 = 147,456 genomes
```

`Genome.to_spec(seed=, steps=)` is the NEUROWEAVE lab spec (`experiments.make_spec`) with the genome's `sparsity`.
The sparsity gene only acts when `sparse_code` is on. Without it, two genomes that differ only in sparsity give
identical metrics and descriptors but different specs, so their receipt digests differ (tested). Genomes are
canonical: mechanisms are kept in NEUROWEAVE order, and `Genome.key` (canonical JSON) identifies duplicates.

Measured on 2026-11-20 (16 neurons, `clustered`, 16 training steps, held-out seed 100 × 8 steps):

* five of the ten switches change the estimated operation count of a training run (17,840 with no switch):
  `structural_encoder` +3,072, `adaptive_routing` +2,848, `context_attention` +2,048, `flyhash_projection` +2,048
  and `sparse_code` +1,024; the other five change nothing;
* topology changes it too (15,872 to 21,200 with no switch);
* log2 ops per training step is 10.12 / 10.68 / 11.06 at 16 / 24 / 32 neurons (no switch, `clustered`).

### Variation (sonapse-evo style operators)

| Operator | Precondition | Change |
|---|---|---|
| `toggle_mechanism` | the space has a mechanism | flips one search-set mechanism on or off |
| `swap_mechanism` | 1 ≤ active < \|M\| | replaces one active mechanism with an inactive one |
| `change_topology` | ≥ 2 topologies | a different topology, uniformly |
| `step_neurons` | ≥ 2 neuron choices | an adjacent neuron count |
| `step_sparsity` | ≥ 2 sparsity choices | an adjacent k-WTA fraction |

`mutate` applies one applicable operator chosen uniformly. It then applies each further operator with
probability `extra_prob` (0.25), up to `max_operators` (3). A child that equals its parent (for example, the same
mechanism toggled twice) gets one more change, so a child always differs from its parent. Each change is logged
as an operator string, for example `toggle_mechanism:+sparse_code` or `step_neurons:16->24`; the tests replay these
trails and recover the child exactly. `crossover` is uniform: each mechanism's membership, and each other gene,
comes from either parent with probability 1/2.

### Evaluation, fitness and descriptors

`NeuroWeaveEvaluator(task, steps, heldout_seeds, heldout_steps)` follows the protocol of the lab's
`experiments._task_arm`:

* **Training.** A fresh engine is built from the genome's spec. It and a fresh `ReadoutHead` (logistic, learning
  rate 0.05) run prequentially, with learning on, over `make_stream(seed, steps)`.
* **Held out.** For each held-out seed `h`, the trained engine and a copy of the head run frozen (`learn=False`)
  over `make_stream(h, heldout_steps)`. Every stream starts from the trained state: a validated clone runs every
  stream except the last, and the trained engine itself runs the last. An `ActivityRecorder` is attached to each.

At `sparsity = 0.15` (the spec default), the metrics, estimated cost and budget equal `_task_arm`'s exactly
(tested). The recorder is a read-only engine extension: it adds no readout features and leaves the drive
untouched. Engine outputs and the engine's state digest are identical with and without it (tested).

```
fitness   f   = mean held-out accuracy                       (ties: lower mean held-out log loss, then earlier)
d_cost        = log2( estimated_cost.total_ops / training steps )        (from the candidate's own receipt)
d_activity    = mean over held-out steps t of H(y_t),   H(x) = (√n − ‖x‖₁ / ‖x‖₂) / (√n − 1)   (Hoyer, 2004)
```

`y_t` is the engine output vector. `H` is 1 for a single non-zero entry and 0 for equal magnitudes. All-zero
steps are skipped (H is undefined there) and counted as `silent_steps`; a run with no scored step counts as 1.0.
Each candidate's result is a sealed receipt (`experiment_id 'quality_diversity_candidate'`, `kind
'candidate_evaluation'`) with the genome, seed, spec, protocol, train and held-out metrics, activity statistics,
estimated cost, budget, fitness and descriptors. Wall-clock is kept outside the digest.

Measured on 2026-11-20 (same settings as above): without `sparse_code` the outputs are dense, with `d_activity`
0.0008 (0.0000 to 0.0796 across the 12 topologies). With `sparse_code`, `d_activity` is 0.829 / 0.741 / 0.507 /
0.339 at sparsity 0.05 / 0.15 / 0.3 / 0.5. In this space the activity axis therefore mostly sorts `sparse_code`
genomes by their k-WTA fraction.

`LandscapeEvaluator(target, space)` is a deterministic synthetic evaluator with a known answer, used by the tests.
It runs no NEUROWEAVE engine. Fitness is the fraction of genes equal to a hidden target genome, so the target is
the unique optimum with fitness 1. Its descriptors are analytic: ops per step `16 · neurons · (4 + #mechanisms)`,
and activity `1 − sparsity` with `sparse_code` or 0.05 without.

### Archive (MAP-Elites grid)

```
cell(d)  = ( bin(d_cost; lo_c, hi_c, B_c), bin(d_activity; lo_a, hi_a, B_a) )
bin(v)   = clip( ⌊(v − lo) / (hi − lo) · B⌋, 0, B − 1 )            # out-of-range values join the edge cells
admit x' into cell b'  iff  b' is empty  or  (f', −loss') > (f, −loss) of the elite in b'
```

This is the replacement rule of the paper's Fig. 2 (`P(b′) = ∅ or P(b′) < p′`), with log loss as tie-break.
Exact ties keep the earlier elite. The invariant is that each cell holds the verified candidate with the highest
`(f, −loss)` among all verified candidates that mapped to it. It is tested against a brute-force map of a
32-genome space, and checked again on import by replaying the lineage.

### Parent selection (DGM)

```
s_i = 1 / (1 + exp(−λ (f_i − f_0)))      h_i = 1 / (1 + n_i)      p_i = s_i h_i / Σ_j s_j h_j
```

* **Eligibility.** The distribution is taken over elites with `f_i < 1`, as in DGM. When every elite has
  `f_i = 1`, all elites are eligible: this fallback is ours, because DGM does not define that case. DGM writes
  the normaliser `Σ_j` over the whole archive while sampling from the eligible set; we normalise over the
  eligible set, so the probabilities sum to 1.
* **Values.** `λ = 10` and `f_0 = 0.5` are DGM's values. `n_i` is the number of the elite's children that passed
  verification; DGM counts children that kept their codebase-editing functionality.
* **Drawing parents.** DGM samples `k` parents with replacement, and each makes its own child. Here each variation
  step draws two parents with replacement and makes one child. With probability `crossover_rate` (0.25), and only
  when the two are distinct elites, the child is `mutate(crossover(a, b))`; otherwise it is `mutate(a)`.

`selection='uniform'` gives the paper's uniform random choice of elite and serves as the ablation. The first
`initial_random` candidates (G in the paper) are random genomes. Up to `max_resample` (16) proposals are drawn to
avoid re-proposing a genome that was already evaluated.

### Verification gate

`verify_candidate(genome, receipt, seed=, protocol=, space=)` never raises for a bad receipt. It returns the
failed reason codes, and an empty list means the candidate passed. Each check is made against what the search
asked for, not against the receipt itself.

| Reason | Meaning |
|---|---|
| `receipt_not_a_mapping` | the evaluator returned something other than a mapping |
| `receipt_digest_mismatch` | the SHA-256 seal does not verify (edited after sealing) |
| `receipt_schema_mismatch` | schema, experiment id, kind, evidence tier or module differ |
| `genome_outside_space` | the genome is outside the declared space |
| `genome_mismatch` | the receipt's genome is not the proposed one |
| `protocol_mismatch` | the protocol or the seed differ from the search's |
| `invalid_spec` | the receipt's spec is not a valid `NeuroWeaveSpecV1` |
| `spec_mismatch` | the spec is valid but is not the spec the genome implies |
| `invalid_metrics` | fitness or descriptor source fields are missing or out of range |
| `inconsistent_metrics` | the stated fitness or descriptors differ from those recomputed from the raw fields |
| `evaluation_error` | the evaluator raised; the message is logged, truncated to 200 characters |
| `spot_check_digest_mismatch` | the re-evaluation produced a different digest |
| `spot_check_error` | the re-evaluation raised |

**Spot checks.** With `spot_check_every = k` (default 4), candidates number k, 2k, 3k, … (in evaluation order)
that pass the static checks are evaluated again. The re-run calls the evaluator itself, never the cache, and must
reproduce the receipt digest. `k = 0` disables spot checks; it exists only as an ablation.

**Rejection.** A rejected candidate goes into the lineage and the rejection log. The grid and the child counts are
untouched.

What the gate catches (tested):

* receipts edited after sealing;
* re-sealed receipts whose stated fitness disagrees with their raw metrics;
* wrong specs, genomes, protocols and seeds;
* evaluator exceptions.

A *consistently* re-sealed lie passes every static check. Only a re-evaluation catches it, and only on candidates
that are spot-checked. In the tests, inflated answers were all rejected with `spot_check_every = 1`. With
`spot_check_every = 0` they were admitted, and a later `audit()`, which re-evaluates every elite, flagged them. A
SHA-256 seal detects accidental or naive edits; it is not a signature.

### Lineage, persistence and metrics

* **Lineage.** Every evaluated candidate is a node with: id, genome, origin (`random` / `mutation` /
  `crossover`), parents, operator trail, status (`elite` / `not_elite` / `rejected`), reasons, error, cell,
  fitness, log loss, descriptors, receipt digest, `spot_checked` and `displaced_by`. A child's parents were
  elites when it was created, so the lineage is a DAG in id order. It is bounded by the evaluation budget
  (`max_nodes`).
* **Export.** `export_json()` writes canonical JSON:
  `{schema: 'QualityDiversityArchiveV1', schema_version: 1, sha256, payload}`. The payload holds the grid
  configuration, the elites with their sealed receipts (wall-clock removed, so exports are deterministic), the
  lineage and the rejection log. `sha256` is the SHA-256 of the canonical payload (sorted keys, compact
  separators).
* **Import.** `import_json()` / `EliteArchive.from_json()` validate everything before changing anything:
  * the hash, the schema and the grid configuration;
  * every lineage node;
  * a full replay of the lineage, which must reproduce each node's status and `displaced_by`, the grid and the
    rejection log;
  * every elite's receipt, which must verify and match its entry and its lineage node.

  Eighteen tamper cases are tested, and none changes the target archive.
* **Metrics.**
  * QD-score is the total fitness over filled cells (Pugh et al., 2015). Coverage is filled cells over grid cells.
  * Reliability, precision and `coverage_of_union` (Mouret & Clune, 2015, Sec. 5 and 9.4) are computed per seed
    over the union of cells that either arm filled.
  * Our convention: a filled cell whose best value is 0 counts with ratio 1. The paper instead divides by the
    number of cells with a non-zero best value, and takes the best over all runs of all treatments, not over
    the two arms of one seed.
  * The summary also lists the Pareto front of the elites in (fitness, −log2 ops per step), cheapest first. This
    is the sibling projects' multi-objective view of the same archive.

## Lab experiment (`run_experiment`, id `quality_diversity`)

The two arms are `quality_diversity` (MAP-Elites with DGM selection) and `random_search`. Per seed they share:

* the evaluator: the task, the training stream `seed` and the held-out streams;
* the search space, the grid, the verification gate and the spot-check schedule;
* the candidate budget.

Their first `initial_random` candidates are identical (common random numbers). Those candidates are evaluated once
through a per-seed cache, which the spot checks bypass. After the search, each arm's best elite is re-evaluated on
fresh `test_seeds` streams (`best_test`), because `best_fitness` is selected on the held-out streams and is
therefore optimistic.

* **Metrics per arm and seed.** `qd_score`, `coverage`, `best_fitness`, `best_test_accuracy`, `reliability`,
  `precision` and `coverage_of_union`.
* **Comparisons.** `harness.compare` of QD − random for every metric, paired over seeds.
* **Verdict.** The `qd_score` comparison: `improved` only when the 95 % t interval excludes zero on the better
  side.

| Setting | Default | `quick=True` |
|---|---|---|
| task | `delayed_recall` | same |
| training / held-out steps | 48 / 24 | 16 / 8 |
| held-out seeds / test seeds | (100, 101) / (200, 201) | (100,) / (200,) |
| candidates per arm per seed (`budget`) | 16 | 5 |
| `initial_random` / `spot_check_every` | 6 / 4 | 3 / 2 |
| neurons | (16, 24, 32) | (16,) |
| grid | 4 × 4; cost (10.0, 12.0), activity (0.0, 0.8) | cost (10.1, 10.8) |
| selection / `crossover_rate` / `extra_mutation_prob` | DGM (λ 10, f₀ 0.5) / 0.25 / 0.25 | same |

`delayed_recall` is the default task because it separated genomes best in a scan on 2026-11-20. The scan covered
10 random genomes from the default space (drawn with `default_rng(0)`; seed 0; 48 training steps; held-out seeds
100 and 101 × 24 steps):

| Task | Distinct held-out accuracies | sd (population) | Best / majority-class rate |
|---|---|---|---|
| `delayed_recall` | 6 | 0.075 | 0.604 / 0.583 |
| `xor` | 4 | 0.062 | 0.688 / 0.542 |
| `parity` | 3 | 0.028 | 0.521 / 0.500 |
| `delayed_recall_6` | 2 | 0.115 | 0.625 / 0.625 |
| `rhythm` | 1 (all genomes equal) | 0 | 0.792 / 0.792 |
| `context_switch` | 1 (all genomes equal) | 0 | 0.500 / 0.542 |

Even on `delayed_recall` the best value (0.604) was close to the majority-class rate (0.583). On `xor` the best
genome rose further above its majority-class rate (0.688 vs 0.542) but the accuracies took fewer distinct values;
the choice of default task was made on the number of distinct values only.

The receipt (`kind: 'search_comparison'`) contains:

* `config`, with the space, grid, evaluator protocol and admission rule;
* per-seed rows for each arm: evaluations, spot checks, evaluator calls, cache hits, status counts, rejections by
  reason, QD-score and coverage trajectories, elites with their candidate-receipt digests, the best elite,
  `best_test`, the Pareto front, lineage statistics and the archive SHA-256;
* `comparisons`, `verdict` and `verdict_basis`;
* `budget`: equal candidate budgets and counted evaluator calls;
* `estimated_cost`: the estimated ops of the verified candidates, kept apart from `measured_wallclock_s`;
* `notes`.

Candidate digests are stored as `receipt_digest`, because receipt seals skip keys named `digest`.

## Public API

```python
MODULE_ID = 'quality_diversity'
EXPERIMENTS = {'quality_diversity': run_experiment}
run_experiment(*, quick: bool = False, seeds=(0, 1, 2), **overrides) -> dict    # overrides: any _DEFAULTS key

SEARCH_MECHANISMS, NEURON_CHOICES, SPARSITY_CHOICES, MUTATION_OPERATORS, REJECTION_REASONS, DESCRIPTOR_KEYS,
ARCHIVE_SCHEMA, CANDIDATE_RECEIPT_ID, MAX_BUDGET, SOURCE

Genome(mechanisms, topology, neurons, sparsity)            .to_dict() / .from_dict(d) / .key / .to_spec(seed=, steps=)
GenomeSpace(mechanisms=..., topologies=..., neurons=..., sparsity=...)
    .size / .contains(g) / .random_genome(rng) / .to_dict() / .from_dict(d)
applicable_operators(genome, space) -> tuple[str, ...]
mutate(genome, space, rng, *, extra_prob=0.25, max_operators=3) -> (Genome, operator_trail)
crossover(a, b, space, rng) -> Genome
dgm_parent_probabilities(fitness, children, *, sharpness=10.0, midpoint=0.5) -> np.ndarray
hoyer_sparseness(x) -> float | None
ActivityRecorder()                                          # read-only EngineExtension, id 'qd_activity_recorder'
NeuroWeaveEvaluator(*, task='delayed_recall', steps=48, heldout_seeds=(100, 101), heldout_steps=24)
    (genome, seed) -> sealed candidate receipt;  .evaluate(genome, seed, *, heldout_seeds=None);  .protocol()
LandscapeEvaluator(target, space, *, steps=16)              # known-answer synthetic evaluator
candidate_metrics(receipt) -> dict                          # recomputed fitness, log loss and descriptors
verify_candidate(genome, receipt, *, seed, protocol, space) -> list[str]
EliteArchive(*, cost_bins=4, cost_range=(9.75, 11.75), activity_bins=4, activity_range=(0.0, 0.8), max_nodes=10000)
    .record(...) / .cell_of(d) / .elites() / .elite_receipt(cell) / .best() / .pareto_front() / .qd_score()
    .coverage() / .lineage() / .rejections() / .children_count(id) / .sha256 / .export_json() / .import_json(text)
    EliteArchive.from_json(text)
QualityDiversitySearch(evaluator, *, seed=0, budget=32, method='qd'|'random', space=None, initial_random=8,
                       selection='dgm'|'uniform', crossover_rate=0.25, extra_mutation_prob=0.25, spot_check_every=4,
                       selection_sharpness=10.0, selection_midpoint=0.5, max_resample=16, cost_bins=4,
                       cost_range=(9.75, 11.75), activity_bins=4, activity_range=(0.0, 0.8), cache=None)
    .step() / .run() -> summary / .summary() / .audit() / .parent_probabilities() / .config() / .done / .archive
illumination_metrics({run: {cell: fitness}}) -> {run: {reliability, precision, coverage_of_union}}
pareto_front(points) -> list[int]
```

Determinism and global state: all randomness comes from `numpy.random.default_rng(seed + offset)`. The random
genome stream uses offset 67,001 and the variation stream 67,013; the NEUROWEAVE runs are seeded by the engine. The
module does not import torch and never touches the global NumPy or `random` state; the tests compare the NumPy,
`random` and (when it is loaded) torch RNG and thread state before and after a search and `run_experiment`. The
same arguments give the same receipt digest and the same archive export.

Each search switch is tested for an effect: DGM vs uniform selection, sigmoid sharpness 0 (only the child-count
term is left, checked against `1 / (1 + n_i)`), crossover off, single-operator mutation (one operator per child),
the random method and the spot-check frequency all change the archive. The shared evaluation cache, by design,
leaves the archive byte-identical.

## Connection to Sonapse

* The module uses the NEUROWEAVE lab read-only: `make_spec`, `NeuroWeaveEngine`, `ReadoutHead`,
  `run_task_phase`, `make_stream`, `clone_engine`, `_estimated_cost`, and the receipt helpers via
  `frontier.harness`. It reproduces the protocol of `_task_arm`.
* Its only engine attachment is `ActivityRecorder`, a read-only extension, and only on the module's own
  evaluation engines.
* It changes no Sonapse or NEUROWEAVE default and is not a live observer. The frontier lab picks it up from
  `EXPERIMENTS`.
* Each elite is a complete `NeuroWeaveSpecV1` (`Genome.to_spec`) with a sealed receipt. Possible follow-ups, not
  implemented: seed other lab experiments with archive elites, or search over topology-metric descriptors such as
  modularity.

## Not implemented

* **From MAP-Elites:**
  * any of the alternative variants discussed in the paper (Sec. 8);
  * the paper's global-performance metric, which is normalised by the best value found;
  * adaptive or centroidal grids. The descriptor ranges are fixed in the config.
* **From DGM:**
  * self-modification of code, foundation-model-proposed changes, staged benchmark evaluation, and `k` parents each
    producing their own child.
  * Only the parent-selection rule and the idea of admitting changes after empirical validation are reused.
* **Statistical strength of fitness.**
  * Fitness is held-out accuracy on one short stream per held-out seed. It is coarse (multiples of 1 / 48 by
    default), often near the majority-class rate, and noisy, so elites are partly selected on noise.
  * Candidates are not re-trained with other seeds.
  * The `best_test` re-evaluation on fresh streams is the only guard against this optimism.
* **Equal budget means equal candidate counts, not equal compute.** Candidates differ in estimated cost. The
  receipt reports the estimated ops of the verified candidates per arm.
* **Spot-check schedule.** It is periodic and deterministic, so an adversarial evaluator that knew `k` could cheat
  on the other candidates. The gate is designed for detecting faults, not for security.

## Verified references

Each reference was checked on 2026-11-20 against the listed pages and, where noted, the paper's text.

1. Jean-Baptiste Mouret, Jeff Clune. **Illuminating search spaces by mapping elites.** arXiv:1504.04909
   [cs.AI], v1 submitted 20 April 2015 (arXiv comment "Early draft"; no later version is listed).
   <https://arxiv.org/abs/1504.04909>. Checked in the PDF:
   * the Fig. 2 pseudocode: G random solutions, then a random elite, random variation, and a store when the cell
     is empty or `P(b′) < p′`;
   * the Sec. 5 criteria (global performance, global reliability, precision / opt-in reliability, coverage);
   * the Sec. 9.4.1–9.4.3 formulas, with the best per cell taken over all runs of all treatments and unfilled
     cells excluded.
2. Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune. **Darwin Gödel Machine: Open-Ended Evolution of
   Self-Improving Agents.** arXiv:2505.22954 (title spelled "Godel" on arXiv), v1 29 May 2025, v3 12 March 2026.
   ICLR 2026 poster: <https://iclr.cc/virtual/2026/poster/10007327> (OpenReview
   <https://openreview.net/forum?id=pUpzQZTvGY>). <https://arxiv.org/abs/2505.22954>. Checked in the v3 HTML,
   App. C.2 "Parent Selection":
   * the eligible set `α_i < 1`;
   * `s_i = 1/(1 + exp(−λ(α_i − α_0)))`;
   * `h_i = 1/(1 + n_i)`, with `n_i` the children with codebase-editing functionality;
   * `p_i = w_i / Σ w_j`, with λ = 10 and α_0 = 0.5;
   * `k` parents sampled with replacement: k = 2 for SWE-bench and 4 for Polyglot.
3. Justin K. Pugh, L. B. Soros, Paul A. Szerlip, Kenneth O. Stanley. **Confronting the Challenge of Quality
   Diversity.** GECCO '15, Madrid, 11–15 July 2015, pp. 967–974. doi:10.1145/2739480.2754664,
   <https://dl.acm.org/doi/10.1145/2739480.2754664>. Checked in the PDF: the QD score is the "total fitness across
   all filled grid bins".
4. Justin K. Pugh, Lisa B. Soros, Kenneth O. Stanley. **Quality Diversity: A New Frontier for Evolutionary
   Computation.** *Frontiers in Robotics and AI* 3, published 12 July 2016. doi:10.3389/frobt.2016.00040,
   <https://www.frontiersin.org/articles/10.3389/frobt.2016.00040/full>. Checked on the publisher page: the
   QD-score sums the highest fitness found in each grid bin, credited to Pugh et al. (2015).
5. Patrik O. Hoyer. **Non-negative Matrix Factorization with Sparseness Constraints.** *Journal of Machine
   Learning Research* 5, 1457–1469, November 2004. <https://www.jmlr.org/papers/v5/hoyer04a.html>. Checked in
   the PDF: `sparseness(x) = (√n − Σ|x_i| / √(Σ x_i²)) / (√n − 1)`.

The sibling projects were read from read-only clones on 2026-11-20; no code from them was executed or copied:

* SONAPSE-ORIGIN: `sonapse_origin/evolution/map_elites.py`, `search_engine.py` and `nsga2.py`. These hold a 2-D
  grid with clipped bins and strict-improvement replacement, uniform elite sampling, mutation and crossover, and
  NSGA-II fronts.
* sonapse-evo: `src/sonapse_evo/evolution/operators.py`, `evolution/transaction_manager.py` and
  `discovery/pareto_optimizer.py`. These hold operators with `can_apply` preconditions, mutation receipts, a
  `FailureArchive` with reasons, parent-digest checks before commit with rollback, and constrained Pareto sorting.

## Measured results

All runs were measured on 2026-11-20 on Windows 11 on ARM64, with Python 3.12.10, NumPy 2.4 and one BLAS/OpenMP
thread (`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`). The machine was shared with other jobs, so
wall-clock times are indicative. Receipts are deterministic: six quick runs gave the same digest, and re-running
the default and `budget=48` experiments later the same day reproduced their digests exactly.

Each row compares MAP-Elites with DGM selection (QD) against random search; numbers are QD vs random.

| Run | Seeds | Verdict (QD-score) | Key numbers (QD vs random) | Wall-clock |
|---|---|---|---|---|
| `run_experiment(quick=True)` | 0, 1, 2 | `worse` | QD-score 1.917 vs 2.625; difference −0.708, CI [−1.067, −0.350]. Coverage 0.229 vs 0.313. Best held-out accuracy 0.625 vs 0.625. Best elite on the test stream 0.417 vs 0.333, below the majority-class rate (0.75). No rejections; 2 spot checks per arm and seed, all reproduced | 3.3–4.0 s (6 runs) |
| `run_experiment()` | 0, 1, 2 | `worse` | QD-score 3.736 vs 4.924; −1.188, CI [−2.289, −0.086]. Coverage 0.417 vs 0.563 (difference CI [−0.383, 0.091]). Precision 0.983 vs 0.986. Best held-out accuracy 0.674 vs 0.660. Best elite on test streams 0.590 vs 0.604 (majority-class rate 0.542). No rejections; 4 spot checks per arm and seed, all reproduced | 43.9 s |
| `run_experiment(budget=48, initial_random=12)` | 0, 1, 2 | `no_detectable_difference` | QD-score 7.521 vs 7.806; −0.285, CI [−2.991, 2.422]. Coverage 0.833 vs 0.854. Best held-out accuracy 0.660 vs 0.667. Best elite on test streams 0.583 vs 0.646; −0.063, CI [−0.114, −0.011], so `worse` on that metric (majority-class rate 0.542). No rejections; 12 spot checks per arm and seed, all reproduced | 146.7 s |

The receipt digests are tied to the NEUROWEAVE code of that date:

* quick: `11618e7bd20da10be677b4b48c7d40523cc479e2f5172d3d0998c72e89259d53`;
* default: `c5c757eb4f4cca55832d39434ad5d09ab21048318084fdee7a4f23c544a10555`;
* `budget=48`: `8453ba874164518d5b99a426bb758443f95f5d4f30c5e3ba8bee05482bd9e414`.

Neither arm learned the quick task: no best elite beat the test stream's majority-class rate.

**Known-answer landscape** (from the tests, `LandscapeEvaluator`). The space has 6 mechanisms × 3 topologies ×
2 neuron counts × 2 sparsities (768 genomes) and a hidden target. The budget is 80 candidates; seeds 0–4; 4 × 4
grid.

| | MAP-Elites with DGM selection | Random search |
|---|---|---|
| Reached the unique optimum | 4 of 5 seeds | 0 of 5 |
| Mean best fitness | 0.978 | 0.844 |
| Mean QD-score | 4.756 (higher on all 5 seeds) | 4.422 |
| Filled cells | 6 on every seed | 6–7 |
| Mean fitness of verified candidates | 0.613 | 0.493 |

**Reading.** On the known landscape, MAP-Elites with DGM selection climbed within cells and reached a higher
QD-score while filling slightly fewer cells.

In the NEUROWEAVE space at 5 and 16 candidates per arm, it was the other way round. Held-out accuracy differed
little between configurations and stayed near the majority-class rate, so climbing had little to gain. Random
search filled more cells, and coverage decided the QD-score; precision was about the same for both arms.

At 48 candidates per arm, both arms filled most of the 16 cells and the QD-score difference was not detectable.
However, the best elites found by MAP-Elites scored lower on fresh test streams. One possible reason is stronger
selection on held-out noise; this was not tested.

These statements cover these settings only and say nothing about other tasks, sizes or budgets.
