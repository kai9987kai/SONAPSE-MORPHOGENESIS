# `experiment_design`: decision-sensitive experiment selection

Module file: `synapse/frontier/experiment_design.py` · tests: `tests/test_frontier_addons.py` ·
experiment ids: `experiment_design`, `experiment_design_costed` · kind: methodology (no engine extension, no live observer) ·
evidence tier: `synthetic_abstraction`.

The module implements **one-step decision-sensitive Bayesian experimental design** over a finite hypothesis set,
inspired by COUNTERFORM's decision-risk acquisition and Mnemorph (`kai9987kai`) and by classical Bayesian
decision-theoretic design (Lindley, 1956; Chaloner & Verdinelli, 1995). Instead of picking the experiment that
maximises expected information gain about which hypothesis is true, the selector picks the experiment that
maximises the expected reduction in the **downstream decision's Bayes risk**, net of any experiment cost, and
**abstains** (recommends stopping) when no experiment improves expected net risk beyond a threshold.

This is a synthetic finite-hypothesis abstraction. It does not reproduce COUNTERFORM's physical oscillator studies
or any published benchmark.

## What it computes

**Problem (`DecisionProblem`).** There are $K$ hypotheses with prior weights $w \in \Delta^{K-1}$, $M$ candidate
experiments, and $D$ downstream decisions with a loss matrix $L \in \mathbb{R}^{D \times K}$ and non-negative
per-experiment costs $c \in \mathbb{R}_{\ge 0}^M$. Under hypothesis $k$, experiment $e$ produces a scalar Gaussian
outcome

$$
y \mid (k, e) \sim \mathcal{N}\!\left(\mu_{k,e},\; \sigma_{k,e}^2\right).
$$

* **Bayes update (`update`).** Given belief $w$ and outcome $y$ from experiment $e$, the posterior $w'(y)$ is
  computed in log space:
  $$
  w'_k(y) \propto w_k \,\mathcal{N}\!\left(y;\; \mu_{k,e},\, \sigma_{k,e}^2\right),
  \qquad w_k = 0 \implies w'_k(y) = 0.
  $$
* **Bayes decision and Bayes risk (`decision`, `bayes_risk`).** The optimal decision at belief $w$ and its expected
  loss are
  $$
  d^*(w) = \arg\min_{d \in \{0,\dots,D-1\}} \sum_{k=0}^{K-1} L_{d,k}\, w_k,
  \qquad R(w) = \min_{d} (L w)_d,
  $$
  breaking ties by lowest decision index.
* **Deterministic posterior-predictive quadrature (`_fantasies`).** Under belief $w$, the marginal outcome
  distribution of experiment $e$ is the Gaussian mixture $\sum_{k} w_k \,\mathcal{N}(\mu_{k,e}, \sigma_{k,e}^2)$.
  Expectations over $y$ are evaluated with $Q$-point Gauss–Hermite quadrature (`quadrature=20` by default) applied
  to each mixture component:
  $$
  y_{k,q} = \mu_{k,e} + \sigma_{k,e}\sqrt{2}\,x_q, \qquad p_{k,q} = \frac{w_k\,\omega_q}{\sqrt{\pi}},
  $$
  where $(x_q, \omega_q)_{q=1}^Q$ are the standard Gauss–Hermite nodes and weights ($\sum_q \omega_q / \sqrt{\pi} = 1$).
  Because quadrature is deterministic, acquisition scores have zero Monte-Carlo variance.
* **Expected Bayes-risk reduction (`risk_reduction`).** The one-step value of information of experiment $e$ is
  $$
  \Delta R(e \mid w) = \max\!\left(0,\; R(w) - \sum_{k=0}^{K-1}\sum_{q=1}^{Q} p_{k,q}\, R\!\left(w'(y_{k,q})\right)\right).
  $$
* **Expected information gain (`information_gain`).** Lindley's mutual information (in nats) between the hypothesis
  and the outcome of experiment $e$:
  $$
  \mathrm{IG}(e \mid w) = \max\!\left(0,\; H(w) - \sum_{k=0}^{K-1}\sum_{q=1}^{Q} p_{k,q}\, H\!\left(w'(y_{k,q})\right)\right),
  \qquad H(w) = -\sum_{k:\,w_k>0} w_k \ln w_k.
  $$

**Selectors.**

* **`DecisionSensitiveSelector` (`decision_risk`)** scores each experiment by net value $\Delta R(e \mid w) - c_e$,
  picks the lowest index among maxima, and returns `None` (abstains) when
  $\max_e \{\Delta R(e \mid w) - c_e\} \le \tau$ (`abstain_threshold = 1e-9`). Once a study abstains, no further
  experiments are run and no further cost is paid.
* **`_NoAbstainDecisionSelector` (`decision_risk_no_abstain`)** uses the same net-value ranking but never abstains;
  it separates the effect of the decision-sensitive ranking from the stopping rule in costed studies.
* **`InformationGainSelector` (`information_gain`)** picks $\arg\max_e \mathrm{IG}(e \mid w)$ (abstaining only when
  $\max_e \mathrm{IG}(e \mid w) \le 10^{-12}$, i.e., when every experiment has identical outcome distributions across
  supported hypotheses).
* **`RandomSelector` (`random`)** draws uniformly from $\{0, \dots, M-1\}$ using a seeded NumPy generator
  (`default_rng(seed + 104729)`).

## Benchmarks (`EXPERIMENTS = {'experiment_design': run_experiment, 'experiment_design_costed': run_costed_experiment}`)

**Synthetic studies (`make_study`).** The generator constructs problems where **information about the hypothesis**
and **relevance to the decision** diverge:

1. The $K$ hypotheses are partitioned into $D$ decision groups ($g_k = k \bmod D$). Decision $d$ has zero loss on
   hypotheses in group $d$ and an asymmetric loss $L_{d,k} \sim \mathrm{Uniform}(0.5, 2.0)$ when $d \ne g_k$.
2. Prior weights follow $\mathrm{Dirichlet}(4, \dots, 4)$.
3. The $M - M_{\text{dist}}$ **relevant** experiments draw a group mean $\mathcal{N}(0, 1)$ per decision group plus
   small within-group jitter $\mathcal{N}(0, 0.15^2)$: they separate decision groups with modest signal-to-noise.
4. The $M_{\text{dist}}$ **distractor** experiments repeat the same large within-group spacing (amplitude
   $\mathrm{Uniform}(2, 4)$) inside every decision group: they carry high mutual information about *which*
   hypothesis holds within a group, but almost zero information about *which group* holds.
5. Experiment columns are randomly permuted so the selector must infer relevance from $(\mu, \sigma, L)$.

**Common random numbers (`simulate_study`).** For each study, the hidden true hypothesis $k^* \sim w^{(0)}$ and the
standard-normal observation noise draws $z_0, \dots, z_{B-1}$ are shared across every selector arm. After $B$
steps (or earlier abstention), the terminal decision $d^*(w^{(B)})$ is scored by its **decision regret** against an
oracle that knows $k^*$:

$$
\mathrm{regret}\!\left(d, k^*\right) = L_{d, k^*} - \min_{d'} L_{d', k^*}.
$$

* **`experiment_design` (`run_experiment`)** compares `decision_risk`, `information_gain`, and `random` at zero
  experiment cost (`c_e = 0`). Headline verdict: `harness.compare(decision_risk, information_gain, higher_is_better=False)`
  on mean final decision regret across seeds.
* **`experiment_design_costed` (`run_costed_experiment`)** assigns per-experiment costs
  $c_e \sim \mathrm{Uniform}(\text{cost\_low}, \text{cost\_high})$ and compares `decision_risk`,
  `decision_risk_no_abstain`, and `information_gain` on **mean total loss** ($\text{final regret} + \sum c_e$,
  lower is better).

| option | `experiment_design` default / quick | `experiment_design_costed` default / quick |
|---|---|---|
| studies per seed | 300 / 60 | 300 / 60 |
| experiment budget $B$ | 4 / 3 | 6 / 4 |
| hypotheses $K$ / decisions $D$ | 6 / 2 | 6 / 2 |
| experiments $M$ / distractors $M_{\text{dist}}$ | 8 / 3 | 8 / 3 |
| noise scale / quadrature $Q$ | 1.0 / 20 | 1.0 / 20 |
| per-experiment cost range | `(0.0, 0.0)` | `(0.02, 0.08)` |

## Public API

```python
MODULE_ID = 'experiment_design'
SELECTORS = ('decision_risk', 'information_gain', 'random')
EXPERIMENTS = {
    'experiment_design': run_experiment,
    'experiment_design_costed': run_costed_experiment,
}

DecisionProblem(prior, means, sigmas, loss, costs=None, *, quadrature=20)
    .n_hypotheses  .n_experiments  .n_decisions
    .log_likelihood(e, y) -> np.ndarray
    .update(w, e, y) -> np.ndarray
    .decision(w) -> (decision_idx: int, bayes_risk: float)
    .bayes_risk(w) -> float
    .risk_reduction(w) -> np.ndarray          # shape (M,)
    .information_gain(w) -> np.ndarray        # shape (M,), nats
    .to_dict() -> dict
DecisionSensitiveSelector(problem, *, abstain_threshold=1e-9)
    .scores(w) -> np.ndarray                  # risk_reduction(w) - costs
    .choose(w) -> int | None
InformationGainSelector(problem, *, abstain_threshold=1e-12)
    .scores(w) -> np.ndarray
    .choose(w) -> int | None
RandomSelector(problem, *, seed=0)
    .choose(w) -> int | None
make_selector(name, problem, *, seed=0, abstain=True)
make_study(seed, *, hypotheses=6, experiments=8, decisions=2, distractors=3,
           noise=1.0, cost_range=(0.0, 0.0), quadrature=20) -> DecisionProblem
simulate_study(problem, truth, selector, budget, noise) -> dict
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict
run_costed_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict
```

## Connection to Sonapse

This is a standalone methodology module (contract rule 3). It has no live observer and no engine extension, and it
changes no Sonapse or NEUROWEAVE default. Its `EXPERIMENTS` dictionary plugs into the frontier lab registry.

## Not implemented

* **Multi-step (non-myopic) lookahead.** Value of information is greedy (one step). When two experiments are
  individually weak but jointly decisive, greedy one-step lookahead underestimates their combined value and can
  abstain early.
* **Continuous hypothesis spaces or non-Gaussian likelihoods.** Hypotheses are finite and likelihoods are scalar
  Gaussian so that one-step expectations are exact up to Gauss–Hermite quadrature.
* **COUNTERFORM's physical oscillator simulator and adequacy gates.** Only the decision-theoretic acquisition
  principle is abstracted here.

## Verified references

* Dennis V. Lindley. **On a Measure of the Information Provided by an Experiment.** *The Annals of Mathematical
  Statistics* 27(4), 986–1005 (December 1956). DOI [10.1214/aoms/1177728069](https://doi.org/10.1214/aoms/1177728069);
  Project Euclid: <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-4/On-a-Measure-of-the-Information-Provided-by-an-Experiment/10.1214/aoms/1177728069.full>.
  * **Checked on 2026-11-19:** defines the expected Shannon information gain of an experiment as
    $I = H_{\text{prior}} - \mathbb{E}_y[H_{\text{posterior}}(y)]$ (used here for `InformationGainSelector`).
* Kathryn Chaloner and Isabella Verdinelli. **Bayesian Experimental Design: A Review.** *Statistical Science*
  10(3), 273–304 (August 1995). DOI [10.1214/ss/1177009939](https://doi.org/10.1214/ss/1177009939);
  Project Euclid: <https://projecteuclid.org/journals/statistical-science/volume-10/issue-3/Bayesian-Experimental-Design-A-Review/10.1214/ss/1177009939.full>.
  * **Checked on 2026-11-19:** reviews the decision-theoretic formulation of Bayesian experimental design (following
    Raiffa & Schlaifer, 1961, and DeGroot, 1970) where an experiment $e$ is chosen to maximise expected utility
    $\int \max_d \int u(d, \theta, y, e)\, p(\theta \mid y, e)\, d\theta\, p(y \mid e)\, dy$, equivalent to
    minimising expected posterior Bayes risk plus experiment cost (`DecisionSensitiveSelector`).
* Portfolio provenance: `kai9987kai/COUNTERFORM` (v0.2 decision-sensitive fantasy design-risk acquisition with
  abstention) and `kai9987kai/Mnemorph`, inspected in the local read-only portfolio clones on 2026-11-19.

## Measured results

Measured on **2026-11-19** with `seeds=(0, 1, 2)` on one CPU thread (`OPENBLAS_NUM_THREADS=1`).

### `experiment_design` (fixed budget, zero cost; metric: mean final decision regret, lower is better)

| mode | digest (prefix) | wall-clock | `decision_risk` regret (acc / abstain / exp run / entropy) | `information_gain` regret (acc / exp run / entropy) | `random` regret (acc) | `decision_risk - information_gain` (95 % CI) | verdict |
|---|---|---|---|---|---|---|---|
| `quick=True` (60 studies, $B=3$) | `a847c682598c5663` | 2.76 s | 0.0746 (93.9 % / 8.3 % / 2.92 / 1.062 nats) | 0.1170 (89.4 % / 3.00 / 0.254 nats) | 0.2453 (77.2 %) | −0.0424 (`[-0.1079, +0.0231]`, SD 0.0264) | `no_detectable_difference` |
| `quick=False` (300 studies, $B=4$) | `36526702c942f04e` | 12.26 s | 0.0481 (95.9 % / 25.7 % / 3.65 / 1.002 nats) | 0.0609 (94.8 % / 4.00 / 0.151 nats) | 0.2134 (80.8 %) | −0.0128 (`[-0.0384, +0.0129]`, SD 0.0103) | `no_detectable_difference` |

Regret by budget step (`quick=False`, steps $t = 0, 1, 2, 3, 4$):

* `decision_risk`: `[0.4885, 0.2164, 0.1157, 0.0698, 0.0481]`
* `information_gain`: `[0.4885, 0.4139, 0.1752, 0.0898, 0.0609]`
* `random`: `[0.4885, 0.3695, 0.2972, 0.2421, 0.2134]`

On step 1, `information_gain` typically chooses a high-entropy distractor experiment (regret drops only from 0.4885
to 0.4139 while posterior entropy falls sharply), whereas `decision_risk` chooses a group-separating experiment
(regret drops from 0.4885 to 0.2164). By step $B=4$, `information_gain` has also sampled relevant experiments and
nearly catches up in terminal regret, so the paired 95 % t interval across 3 seeds spans zero (`no_detectable_difference`),
while both active selectors beat `random` (`improved` at `quick=False`).

### `experiment_design_costed` (costs in `[0.02, 0.08]`; metric: mean total loss = final regret + cost, lower is better)

| mode | digest (prefix) | wall-clock | `decision_risk` total (regret + cost, abstain, exp run) | `decision_risk_no_abstain` total (regret + cost, exp run) | `information_gain` total (regret + cost, exp run) | `decision_risk - information_gain` (95 % CI) | `decision_risk - no_abstain` (95 % CI) | verdict |
|---|---|---|---|---|---|---|---|---|
| `quick=True` (60 studies, $B=4$) | `1bbab8d035695813` | 4.40 s | 0.1442 (0.0700 + 0.0742, 91.7 %, 1.63) | 0.1899 (0.0518 + 0.1380, 4.00) | 0.2606 (0.0670 + 0.1936, 4.00) | −0.1164 (`[-0.1693, -0.0635]`) | −0.0457 (`[-0.1131, +0.0218]`) | `improved` |
| `quick=False` (300 studies, $B=6$) | `ec83a4fa770b5f9c` | 10.68 s | 0.1504 (0.0750 + 0.0754, 98.4 %, 1.65) | 0.2388 (0.0422 + 0.1967, 6.00) | 0.3197 (0.0223 + 0.2974, 6.00) | −0.1693 (`[-0.2814, -0.0572]`) | −0.0885 (`[-0.1332, -0.0438]`) | `improved` |
