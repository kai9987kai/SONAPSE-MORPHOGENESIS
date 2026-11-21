"""Decision-sensitive experiment selection (frontier module ``experiment_design``).

Inspired by COUNTERFORM's decision-risk acquisition and Mnemorph (kai9987kai)
and by Bayesian decision-theoretic experimental design (Lindley, 1956;
Chaloner & Verdinelli, 1995). Nothing here reproduces a published result.

Setting: a finite set of hypotheses ``k`` with prior weights ``w``; a finite
menu of experiments ``e`` whose outcome under hypothesis ``k`` is Gaussian
``y ~ N(mu[k, e], sigma[k, e]^2)``; a finite set of downstream decisions ``d``
with a loss matrix ``L[d, k]`` and optional per-experiment costs ``c[e]``.

* Bayes risk of a belief: ``R(w) = min_d sum_k w_k L[d, k]``.
* Decision-sensitive value of experiment ``e``: the expected reduction in Bayes
  risk ``R(w) - E_y[R(w'(y))]`` where ``y`` follows the posterior predictive and
  ``w'`` is the Bayes update. The expectation uses deterministic Gauss-Hermite
  quadrature per mixture component (no Monte-Carlo noise).
* The selector picks the experiment with the largest net value
  ``value - cost`` and **abstains** (recommends stopping) when that net value
  does not exceed ``abstain_threshold``.
* Baselines: expected information gain (Lindley's mutual information
  ``H(w) - E_y[H(w'(y))]``, in nats) and a seeded random selector.

The value of information is greedy (one step). Greedy decision-risk acquisition
can stop too early when only a *combination* of experiments would change the
decision; receipts record abstentions so this is visible.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from . import harness

__all__ = ['MODULE_ID', 'SELECTORS', 'DecisionProblem', 'DecisionSensitiveSelector', 'InformationGainSelector',
           'RandomSelector', 'make_selector', 'make_study', 'simulate_study', 'run_experiment', 'run_costed_experiment',
           'EXPERIMENTS', 'DEFAULTS']

MODULE_ID = 'experiment_design'
SELECTORS = ('decision_risk', 'information_gain', 'random')

MAX_HYPOTHESES = 64
MAX_EXPERIMENTS = 64
MAX_DECISIONS = 32
MAX_QUADRATURE = 64
MAX_BUDGET = 64
_STUDY_OFFSET = 7_919
_TRUTH_OFFSET = 15_485_863
_NOISE_OFFSET = 32_452_843
_RANDOM_SELECT_OFFSET = 104_729
_EPS = 1e-300


def _finite_array(name: str, value: Any, ndim: int) -> np.ndarray:
    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}-dimensional')
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be non-empty and finite')
    return arr


def _entropy(w: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) along the last axis; 0 log 0 = 0."""
    safe = np.where(w > 0.0, w, 1.0)
    return -np.sum(np.where(w > 0.0, w * np.log(safe), 0.0), axis=-1)


class DecisionProblem:
    """Finite hypotheses x Gaussian experiments x decisions with a loss matrix (immutable after construction)."""

    def __init__(self, prior: Sequence[float], means: Any, sigmas: Any, loss: Any, costs: Sequence[float] | None = None,
                 *, quadrature: int = 20) -> None:
        prior_a = _finite_array('prior', prior, 1)
        means_a = _finite_array('means', means, 2)
        loss_a = _finite_array('loss', loss, 2)
        k, m = means_a.shape
        if not 1 <= k <= MAX_HYPOTHESES or not 1 <= m <= MAX_EXPERIMENTS:
            raise ValueError(f'means must be (hypotheses<= {MAX_HYPOTHESES}, experiments<= {MAX_EXPERIMENTS})')
        if prior_a.shape != (k,) or np.any(prior_a < 0.0) or not prior_a.sum() > 0.0:
            raise ValueError('prior must be non-negative, match the hypotheses and have positive mass')
        sig = np.array(sigmas, dtype=np.float64) if not np.isscalar(sigmas) else np.full(m, float(sigmas))
        if sig.ndim == 1:
            if sig.shape != (m,):
                raise ValueError('sigmas must have one value per experiment or shape (hypotheses, experiments)')
            sig = np.broadcast_to(sig, (k, m)).copy()
        if sig.shape != (k, m) or not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
            raise ValueError('sigmas must be positive and finite with shape (experiments,) or (hypotheses, experiments)')
        if loss_a.shape[1] != k or not 1 <= loss_a.shape[0] <= MAX_DECISIONS:
            raise ValueError(f'loss must be (decisions<= {MAX_DECISIONS}, hypotheses)')
        cost_a = np.zeros(m) if costs is None else _finite_array('costs', costs, 1)
        if cost_a.shape != (m,) or np.any(cost_a < 0.0):
            raise ValueError('costs must be non-negative with one value per experiment')
        if type(quadrature) is not int or not 2 <= quadrature <= MAX_QUADRATURE:
            raise ValueError(f'quadrature must be an int in [2, {MAX_QUADRATURE}]')
        self.prior = prior_a / prior_a.sum()
        self.means = means_a
        self.sigmas = sig
        self.loss = loss_a
        self.costs = cost_a
        self.quadrature = quadrature
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
        self._nodes = nodes * math.sqrt(2.0)              # N(0, 1) abscissae
        self._weights = weights / math.sqrt(math.pi)      # sum to 1
        for arr in (self.prior, self.means, self.sigmas, self.loss, self.costs, self._nodes, self._weights):
            arr.setflags(write=False)

    # ------------------------------------------------------------------ shapes
    @property
    def n_hypotheses(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_experiments(self) -> int:
        return int(self.means.shape[1])

    @property
    def n_decisions(self) -> int:
        return int(self.loss.shape[0])

    def _belief(self, w: Any) -> np.ndarray:
        arr = _finite_array('belief', w, 1)
        if arr.shape != (self.n_hypotheses,) or np.any(arr < 0.0) or not arr.sum() > 0.0:
            raise ValueError('belief must be a non-negative vector over the hypotheses with positive mass')
        return arr / arr.sum()

    def _experiment(self, e: Any) -> int:
        if type(e) not in (int, np.int64, np.int32) or not 0 <= int(e) < self.n_experiments:
            raise ValueError('experiment index out of range')
        return int(e)

    # ------------------------------------------------------------------ inference
    def log_likelihood(self, e: int, y: Any) -> np.ndarray:
        """``log p(y | k, e)`` for every hypothesis (``y`` scalar -> (K,), array (n,) -> (n, K))."""
        e = self._experiment(e)
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y_arr)):
            raise ValueError('outcome must be finite')
        mu, sd = self.means[:, e], self.sigmas[:, e]
        z = (y_arr[..., None] - mu) / sd
        return -0.5 * z * z - np.log(sd) - 0.5 * math.log(2.0 * math.pi)

    def update(self, w: Any, e: int, y: float) -> np.ndarray:
        """Bayes posterior after observing ``y`` from experiment ``e`` (computed in log space)."""
        belief = self._belief(w)
        logp = np.log(np.maximum(belief, _EPS)) + self.log_likelihood(e, y)
        logp = np.where(belief > 0.0, logp, -np.inf)
        logp -= logp.max()
        post = np.exp(logp)
        return post / post.sum()

    def decision(self, w: Any) -> tuple[int, float]:
        """Bayes decision (lowest index on ties) and its expected loss."""
        risks = self.loss @ self._belief(w)
        d = int(np.argmin(risks))
        return d, float(risks[d])

    def bayes_risk(self, w: Any) -> float:
        return self.decision(w)[1]

    def _fantasies(self, w: np.ndarray, e: int) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature outcomes from the posterior predictive: probabilities (K*Q,) and posteriors (K*Q, K)."""
        mu, sd = self.means[:, e], self.sigmas[:, e]
        ys = (mu[:, None] + sd[:, None] * self._nodes[None, :]).ravel()          # (K*Q,)
        probs = (w[:, None] * self._weights[None, :]).ravel()
        loglik = self.log_likelihood(e, ys)                                        # (K*Q, K)
        logp = np.where(w > 0.0, np.log(np.maximum(w, _EPS)), -np.inf)[None, :] + loglik
        logp -= logp.max(axis=1, keepdims=True)
        post = np.exp(logp)
        post /= post.sum(axis=1, keepdims=True)
        return probs, post

    def risk_reduction(self, w: Any) -> np.ndarray:
        """Expected reduction in Bayes risk for every experiment (clipped at 0 against quadrature error)."""
        belief = self._belief(w)
        base = self.bayes_risk(belief)
        out = np.empty(self.n_experiments)
        for e in range(self.n_experiments):
            probs, post = self._fantasies(belief, e)
            out[e] = base - float(probs @ np.min(post @ self.loss.T, axis=1))
        return np.maximum(out, 0.0)

    def information_gain(self, w: Any) -> np.ndarray:
        """Expected information gain (mutual information, nats) for every experiment (clipped at 0)."""
        belief = self._belief(w)
        base = float(_entropy(belief))
        out = np.empty(self.n_experiments)
        for e in range(self.n_experiments):
            probs, post = self._fantasies(belief, e)
            out[e] = base - float(probs @ _entropy(post))
        return np.maximum(out, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {'prior': self.prior.tolist(), 'means': self.means.tolist(), 'sigmas': self.sigmas.tolist(),
                'loss': self.loss.tolist(), 'costs': self.costs.tolist(), 'quadrature': self.quadrature}


# ---------------------------------------------------------------------- selectors

def _argmax_lowest(x: np.ndarray) -> int:
    return int(np.flatnonzero(x == x.max())[0])


class DecisionSensitiveSelector:
    """Choose the experiment with the largest expected Bayes-risk reduction net of cost, or abstain."""

    name = 'decision_risk'

    def __init__(self, problem: DecisionProblem, *, abstain_threshold: float = 1e-9) -> None:
        if not isinstance(problem, DecisionProblem):
            raise ValueError('problem must be a DecisionProblem')
        if not isinstance(abstain_threshold, (int, float)) or not math.isfinite(abstain_threshold) \
                or abstain_threshold < 0.0:
            raise ValueError('abstain_threshold must be a finite non-negative number')
        self.problem = problem
        self.abstain_threshold = float(abstain_threshold)

    def scores(self, w: Any) -> np.ndarray:
        return self.problem.risk_reduction(w) - self.problem.costs

    def choose(self, w: Any) -> int | None:
        net = self.scores(w)
        best = _argmax_lowest(net)
        return None if net[best] <= self.abstain_threshold else best


class InformationGainSelector:
    """Expected-information-gain baseline (stops only when no experiment carries information)."""

    name = 'information_gain'

    def __init__(self, problem: DecisionProblem, *, abstain_threshold: float = 1e-12) -> None:
        if not isinstance(problem, DecisionProblem):
            raise ValueError('problem must be a DecisionProblem')
        if not isinstance(abstain_threshold, (int, float)) or not math.isfinite(abstain_threshold) \
                or abstain_threshold < 0.0:
            raise ValueError('abstain_threshold must be a finite non-negative number')
        self.problem = problem
        self.abstain_threshold = float(abstain_threshold)

    def scores(self, w: Any) -> np.ndarray:
        return self.problem.information_gain(w)

    def choose(self, w: Any) -> int | None:
        ig = self.scores(w)
        best = _argmax_lowest(ig)
        return None if ig[best] <= self.abstain_threshold else best


class RandomSelector:
    """Seeded uniform choice over experiments (never abstains)."""

    name = 'random'

    def __init__(self, problem: DecisionProblem, *, seed: int = 0) -> None:
        if not isinstance(problem, DecisionProblem):
            raise ValueError('problem must be a DecisionProblem')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a non-negative int')
        self.problem = problem
        self._rng = np.random.default_rng(seed + _RANDOM_SELECT_OFFSET)

    def choose(self, w: Any) -> int | None:
        self.problem._belief(w)
        return int(self._rng.integers(self.problem.n_experiments))


def make_selector(name: str, problem: DecisionProblem, *, seed: int = 0, abstain: bool = True):
    """Build a selector by name; ``abstain=False`` gives the decision-risk ranking without its stopping rule."""
    if name == 'decision_risk':
        return DecisionSensitiveSelector(problem) if abstain else _NoAbstainDecisionSelector(problem)
    if name == 'information_gain':
        return InformationGainSelector(problem)
    if name == 'random':
        return RandomSelector(problem, seed=seed)
    raise ValueError(f'selector must be one of {SELECTORS}')


class _NoAbstainDecisionSelector(DecisionSensitiveSelector):
    """Ablation: decision-risk ranking but always runs an experiment (lowest index when all values are zero)."""

    name = 'decision_risk_no_abstain'

    def choose(self, w: Any) -> int | None:
        return _argmax_lowest(self.scores(w))


# ---------------------------------------------------------------------- synthetic studies

def make_study(seed: int, *, hypotheses: int = 6, experiments: int = 8, decisions: int = 2, distractors: int = 3,
               noise: float = 1.0, cost_range: tuple[float, float] = (0.0, 0.0), quadrature: int = 20) -> DecisionProblem:
    """Seeded synthetic study where information and decision relevance diverge.

    Hypotheses are split into ``decisions`` groups; decision ``d`` is right for group ``d`` and costs an
    asymmetric loss otherwise. *Relevant* experiments separate groups with modest effects. *Distractor*
    experiments repeat the same large within-group pattern in every group, so they are very informative about
    which hypothesis holds but nearly useless for the decision."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a non-negative int')
    for name, value, low, high in (('hypotheses', hypotheses, 2, MAX_HYPOTHESES),
                                   ('experiments', experiments, 1, MAX_EXPERIMENTS),
                                   ('decisions', decisions, 2, MAX_DECISIONS)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} must be an int in [{low}, {high}]')
    if decisions > hypotheses:
        raise ValueError('decisions must not exceed hypotheses')
    if type(distractors) is not int or not 0 <= distractors < experiments:
        raise ValueError('distractors must be an int in [0, experiments)')
    if not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise <= 0.0:
        raise ValueError('noise must be a positive finite number')
    lo, hi = (float(cost_range[0]), float(cost_range[1]))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0.0 or hi < lo:
        raise ValueError('cost_range must be finite with 0 <= low <= high')
    rng = np.random.default_rng(seed + _STUDY_OFFSET)
    group = np.arange(hypotheses) % decisions
    rank = np.array([int(np.sum(group[:k] == group[k])) for k in range(hypotheses)])
    size = np.bincount(group, minlength=decisions)
    prior = rng.dirichlet(np.full(hypotheses, 4.0))
    wrong = rng.uniform(0.5, 2.0, size=hypotheses)
    loss = np.where(np.arange(decisions)[:, None] == group[None, :], 0.0, wrong[None, :])
    relevant = experiments - distractors
    means = np.empty((hypotheses, experiments))
    group_effect = rng.normal(0.0, 1.0, size=(decisions, relevant))
    means[:, :relevant] = group_effect[group] + rng.normal(0.0, 0.15, size=(hypotheses, relevant))
    for j in range(distractors):
        amp = rng.uniform(2.0, 4.0)
        pos = np.where(size[group] > 1, rank / np.maximum(size[group] - 1, 1), 0.5)   # same pattern in every group
        means[:, relevant + j] = amp * (2.0 * pos - 1.0)
    order = rng.permutation(experiments)            # hide which experiments are distractors
    means = means[:, order]
    sigmas = rng.uniform(0.8, 1.2, size=experiments) * float(noise)
    costs = rng.uniform(lo, hi, size=experiments) if hi > 0.0 else np.zeros(experiments)
    return DecisionProblem(prior, means, sigmas, loss, costs, quadrature=quadrature)


def simulate_study(problem: DecisionProblem, truth: int, selector: Any, budget: int, noise: Sequence[float]) -> dict[str, Any]:
    """Run ``selector`` for up to ``budget`` experiments against the hidden ``truth``.

    ``noise[t]`` is the standard-normal draw used for the t-th experiment, shared across selectors (common random
    numbers). Returns the regret curve after 0..budget experiments, the chosen experiments and the total cost."""
    if type(truth) not in (int, np.int64) or not 0 <= int(truth) < problem.n_hypotheses:
        raise ValueError('truth out of range')
    if type(budget) is not int or not 0 <= budget <= MAX_BUDGET:
        raise ValueError(f'budget must be an int in [0, {MAX_BUDGET}]')
    z = np.asarray(noise, dtype=np.float64)
    if z.shape != (budget,) or not np.all(np.isfinite(z)):
        raise ValueError('noise must hold one finite draw per budget step')
    truth = int(truth)
    best_loss = float(problem.loss[:, truth].min())
    w = problem.prior.copy()

    def regret() -> float:
        d, _ = problem.decision(w)
        return float(problem.loss[d, truth]) - best_loss

    curve = [regret()]
    chosen: list[int] = []
    cost = 0.0
    abstained_at = None
    for t in range(budget):
        e = selector.choose(w) if abstained_at is None else None
        if e is None:
            abstained_at = t if abstained_at is None else abstained_at
            curve.append(curve[-1])
            continue
        y = float(problem.means[truth, e] + problem.sigmas[truth, e] * z[t])
        w = problem.update(w, e, y)
        chosen.append(int(e))
        cost += float(problem.costs[e])
        curve.append(regret())
    d_final, _ = problem.decision(w)
    return {'regret_curve': curve, 'final_regret': curve[-1], 'correct': bool(curve[-1] == 0.0),
            'decision': int(d_final), 'experiments': chosen, 'cost': cost, 'abstained_at': abstained_at,
            'final_entropy': float(_entropy(w))}


# ---------------------------------------------------------------------- experiments

DEFAULTS: dict[str, Any] = {'studies': 300, 'budget': 4, 'hypotheses': 6, 'experiments': 8, 'decisions': 2,
                            'distractors': 3, 'noise': 1.0, 'quadrature': 20}
_QUICK: dict[str, Any] = {'studies': 60, 'budget': 3}
COSTED_DEFAULTS: dict[str, Any] = {**DEFAULTS, 'budget': 6, 'cost_low': 0.02, 'cost_high': 0.08}
_COSTED_QUICK: dict[str, Any] = {'studies': 60, 'budget': 4}


def _check_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    out = tuple(seeds)
    if not out or any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in out) or len(set(out)) != len(out):
        raise ValueError('seeds must be unique non-negative ints')
    return out


def _config(defaults: Mapping[str, Any], quick_over: Mapping[str, Any], quick: bool,
            overrides: Mapping[str, Any]) -> dict[str, Any]:
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f'unknown overrides: {sorted(unknown)}')
    cfg = {**defaults, **(quick_over if quick else {}), **overrides}
    if type(cfg['studies']) is not int or not 2 <= cfg['studies'] <= 100_000:
        raise ValueError('studies must be an int in [2, 100000]')
    if type(cfg['budget']) is not int or not 1 <= cfg['budget'] <= MAX_BUDGET:
        raise ValueError(f'budget must be an int in [1, {MAX_BUDGET}]')
    return cfg


def _study_args(cfg: Mapping[str, Any], costed: bool) -> dict[str, Any]:
    args = {k: cfg[k] for k in ('hypotheses', 'experiments', 'decisions', 'distractors', 'noise', 'quadrature')}
    if costed:
        args['cost_range'] = (float(cfg['cost_low']), float(cfg['cost_high']))
    return args


def _run_seed(seed: int, cfg: Mapping[str, Any], arms: Sequence[str], *, costed: bool) -> dict[str, Any]:
    truth_rng = np.random.default_rng(seed + _TRUTH_OFFSET)
    noise_rng = np.random.default_rng(seed + _NOISE_OFFSET)
    budget = int(cfg['budget'])
    per_arm: dict[str, dict[str, list]] = {a: {'curves': [], 'costs': [], 'abstentions': [], 'correct': [],
                                               'experiments_run': [], 'entropy': []} for a in arms}
    study_args = _study_args(cfg, costed)
    for i in range(int(cfg['studies'])):
        problem = make_study(seed * 100_003 + i, **study_args)
        truth = int(truth_rng.choice(problem.n_hypotheses, p=problem.prior))
        z = noise_rng.standard_normal(budget)
        for a in arms:
            if a == 'decision_risk':
                sel = DecisionSensitiveSelector(problem)
            elif a == 'decision_risk_no_abstain':
                sel = _NoAbstainDecisionSelector(problem)
            elif a == 'information_gain':
                sel = InformationGainSelector(problem)
            else:
                sel = RandomSelector(problem, seed=seed * 100_003 + i)
            res = simulate_study(problem, truth, sel, budget, z)
            rec = per_arm[a]
            rec['curves'].append(res['regret_curve'])
            rec['costs'].append(res['cost'])
            rec['abstentions'].append(res['abstained_at'] is not None)
            rec['correct'].append(res['correct'])
            rec['experiments_run'].append(len(res['experiments']))
            rec['entropy'].append(res['final_entropy'])
    out = {}
    for a, rec in per_arm.items():
        curves = np.asarray(rec['curves'])
        regret = curves[:, -1]
        cost = np.asarray(rec['costs'])
        out[a] = {'seed': int(seed), 'mean_regret_by_budget': curves.mean(axis=0).tolist(),
                  'final_regret': float(regret.mean()), 'decision_accuracy': float(np.mean(rec['correct'])),
                  'mean_cost': float(cost.mean()), 'total_loss': float((regret + cost).mean()),
                  'abstention_rate': float(np.mean(rec['abstentions'])),
                  'mean_experiments_run': float(np.mean(rec['experiments_run'])),
                  'final_entropy_nats': float(np.mean(rec['entropy']))}
    return out


def _summary(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ('final_regret', 'decision_accuracy', 'mean_cost', 'total_loss', 'abstention_rate',
            'mean_experiments_run', 'final_entropy_nats')
    out = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
    out['mean_regret_by_budget'] = np.mean([p['mean_regret_by_budget'] for p in per_seed], axis=0).tolist()
    return out


def _estimated_cost(cfg: Mapping[str, Any], n_arms: int, seeds: int) -> dict[str, Any]:
    k, m, q = cfg['hypotheses'], cfg['experiments'], cfg['quadrature']
    per_selection = m * (k * q) * k * 4                 # likelihood, posterior, risk/entropy per fantasy outcome
    selections = seeds * cfg['studies'] * cfg['budget'] * n_arms
    return {'estimated_flops_per_selection': int(per_selection), 'estimated_selections_upper_bound': int(selections),
            'estimated_total_flops_upper_bound': int(per_selection * selections)}


def run_experiment(*, quick: bool = False, seeds: Iterable[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Decision-sensitive vs information-gain vs random acquisition at a fixed experiment budget.

    Each seed runs ``studies`` synthetic studies with a hidden true hypothesis; all selectors see the same study,
    truth and observation noise. Metric: mean final decision regret (lower is better)."""
    seeds = _check_seeds(seeds)
    cfg = _config(DEFAULTS, _QUICK, quick, overrides)
    make_study(0, **_study_args(cfg, False))                      # validate study settings before running
    arms = ('decision_risk', 'information_gain', 'random')
    t0 = time.perf_counter()
    runs = [_run_seed(s, cfg, arms, costed=False) for s in seeds]
    per_arm = {a: [r[a] for r in runs] for a in arms}
    regret = {a: [p['final_regret'] for p in per_arm[a]] for a in arms}
    comparisons = {
        'decision_risk_vs_information_gain': harness.compare(regret['decision_risk'], regret['information_gain'],
                                                             higher_is_better=False),
        'decision_risk_vs_random': harness.compare(regret['decision_risk'], regret['random'], higher_is_better=False),
        'information_gain_vs_random': harness.compare(regret['information_gain'], regret['random'],
                                                      higher_is_better=False),
    }
    receipt = harness.base_receipt(MODULE_ID, 'methodology_comparison', {
        **cfg, 'quick': quick, 'seeds': list(seeds), 'metric': 'mean_final_decision_regret',
        'higher_is_better': False, 'abstain_threshold': 1e-9, 'quadrature_rule': 'gauss_hermite_per_component',
        'noise_sharing': 'common_random_numbers_across_selectors'})
    receipt['arms'] = {a: {'per_seed': per_arm[a], 'summary': _summary(per_arm[a])} for a in arms}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['decision_risk_vs_information_gain']['verdict']
    receipt['verdict_basis'] = ('decision_risk_vs_information_gain: paired 95% t interval over seeds on mean final '
                                'decision regret (lower is better)')
    receipt['parameter_budget'] = {'trainable_parameters': 0, 'experiment_budget_per_study': cfg['budget'],
                                   'note': 'selectors have no trained parameters; all arms share the same budget'}
    receipt['estimated_cost'] = _estimated_cost(cfg, len(arms), len(seeds))
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'Synthetic finite-hypothesis studies with Gaussian outcomes, inspired by COUNTERFORM decision-risk acquisition '
        'and Bayesian decision-theoretic design (Lindley 1956; Chaloner & Verdinelli 1995). Distractor experiments are '
        'built to be informative about the hypothesis but nearly irrelevant to the decision, which favours a '
        'decision-sensitive criterion by construction; the result says how much, not whether it generalises. Values '
        'are one-step (greedy); abstention is rare at the default threshold and is recorded per arm.')
    return harness.seal(receipt)


def run_costed_experiment(*, quick: bool = False, seeds: Iterable[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """With per-experiment costs: decision-sensitive selection that may abstain vs information gain that always
    spends the budget. Metric: mean total loss = final regret + experiment cost (lower is better)."""
    seeds = _check_seeds(seeds)
    cfg = _config(COSTED_DEFAULTS, _COSTED_QUICK, quick, overrides)
    if not 0.0 <= cfg['cost_low'] <= cfg['cost_high'] or not math.isfinite(cfg['cost_high']):
        raise ValueError('cost_low/cost_high must satisfy 0 <= low <= high (finite)')
    make_study(0, **_study_args(cfg, True))
    arms = ('decision_risk', 'decision_risk_no_abstain', 'information_gain')
    t0 = time.perf_counter()
    runs = [_run_seed(s, cfg, arms, costed=True) for s in seeds]
    per_arm = {a: [r[a] for r in runs] for a in arms}
    total = {a: [p['total_loss'] for p in per_arm[a]] for a in arms}
    comparisons = {
        'decision_risk_vs_information_gain': harness.compare(total['decision_risk'], total['information_gain'],
                                                             higher_is_better=False),
        'decision_risk_vs_no_abstain_ablation': harness.compare(total['decision_risk'],
                                                                total['decision_risk_no_abstain'],
                                                                higher_is_better=False),
    }
    receipt = harness.base_receipt(MODULE_ID + '_costed', 'methodology_comparison', {
        **cfg, 'quick': quick, 'seeds': list(seeds), 'metric': 'mean_total_loss_regret_plus_cost',
        'higher_is_better': False, 'abstain_threshold': 1e-9, 'quadrature_rule': 'gauss_hermite_per_component',
        'noise_sharing': 'common_random_numbers_across_selectors'})
    receipt['arms'] = {a: {'per_seed': per_arm[a], 'summary': _summary(per_arm[a])} for a in arms}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['decision_risk_vs_information_gain']['verdict']
    receipt['verdict_basis'] = ('decision_risk_vs_information_gain: paired 95% t interval over seeds on mean total '
                                'loss (regret + cost, lower is better)')
    receipt['parameter_budget'] = {'trainable_parameters': 0, 'max_experiments_per_study': cfg['budget']}
    receipt['estimated_cost'] = _estimated_cost(cfg, len(arms), len(seeds))
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'Costs are synthetic and uniform per experiment. The information-gain baseline has no stopping rule, so part of '
        'any advantage is the abstention rule itself; the no-abstain ablation separates ranking from stopping.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment,
                                                         MODULE_ID + '_costed': run_costed_experiment}
