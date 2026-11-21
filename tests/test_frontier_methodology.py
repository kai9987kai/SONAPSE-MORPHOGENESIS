"""Frontier methodology modules: ``anytime_inference`` (betting CS, e-process) and ``joint_failure`` (panel selection)."""
import copy
import inspect
import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from synapse.frontier import anytime as A
from synapse.frontier import joint_failure as J
from synapse.frontier.harness import VERDICTS, verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, MODULE_IDS, MODULES


def _global_rng_state():
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2:]


def _same_rng_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]


# ====================================================================== registry
def test_modules_are_registered_with_their_experiments():
    kinds = {m.id: m.kind for m in MODULES}
    assert A.MODULE_ID == 'anytime_inference' and J.MODULE_ID == 'joint_failure'
    assert A.MODULE_ID in MODULE_IDS and J.MODULE_ID in MODULE_IDS
    assert kinds['anytime_inference'] == kinds['joint_failure'] == 'methodology'
    assert list(A.EXPERIMENTS) == ['anytime_inference'] and A.EXPERIMENTS['anytime_inference'] is A.run_experiment
    assert list(J.EXPERIMENTS) == ['joint_failure', 'joint_failure_neuroweave']
    assert J.EXPERIMENTS['joint_failure'] is J.run_experiment
    assert J.EXPERIMENTS['joint_failure_neuroweave'] is J.run_neuroweave_application
    for mod in (A, J):
        assert all(k == mod.MODULE_ID or k.startswith(mod.MODULE_ID + '_') for k in mod.EXPERIMENTS)
        assert 'torch' not in inspect.getsource(mod)          # NumPy/SciPy only: no global torch state to touch


# ====================================================================== anytime: confidence sequence
def _reference_hedged_inside(x01, alpha, grid, c, theta):
    """Direct loops over the paper's definitions (hedged capital, Eq. 24-26): grid points in the raw set per time."""
    n = len(x01)
    out = np.zeros((n, len(grid)), dtype=bool)
    for j, m in enumerate(grid):
        kp = km = 1.0
        s2, total, sq = 0.25, 0.0, 0.0                         # sigma2_0 = 1/4
        for t in range(1, n + 1):
            lam = math.sqrt(2.0 * math.log(2.0 / alpha) / (s2 * t * math.log(t + 1.0)))
            lam_p = lam if m == 0.0 else min(lam, c / m)
            lam_m = lam if m == 1.0 else min(lam, c / (1.0 - m))
            x = x01[t - 1]
            kp *= 1.0 + lam_p * (x - m)
            km *= 1.0 - lam_m * (x - m)
            out[t - 1, j] = max(theta * kp, (1.0 - theta) * km) < 1.0 / alpha
            total += x
            mu = (0.5 + total) / (t + 1.0)                     # mu_t uses X_t
            sq += (x - mu) ** 2
            s2 = (0.25 + sq) / (t + 1.0)
    return out


@pytest.mark.parametrize('seed,theta,c', [(0, 0.5, 0.5), (1, 0.3, 0.75)])
def test_confidence_sequence_matches_a_direct_implementation_of_the_hedged_capital_process(seed, theta, c):
    rng = np.random.default_rng(seed)
    lo, hi, g, alpha = 2.0, 6.0, 40, 0.1
    x = lo + (hi - lo) * rng.beta(2.0, 3.0, 30)
    cs = A.betting_confidence_sequence(x, lo, hi, alpha, grid_size=g, c=c, theta=theta)
    grid = np.linspace(0.0, 1.0, g + 1)
    inside = _reference_hedged_inside((x - lo) / (hi - lo), alpha, grid, c, theta)
    for t in range(len(x)):
        idx = np.flatnonzero(inside[t])
        assert idx.size and np.all(np.diff(idx) == 1)          # Theorem 3: the raw set is an interval
        assert math.isclose(cs['lower_raw'][t], lo + (hi - lo) * grid[max(idx[0] - 1, 0)])
        assert math.isclose(cs['upper_raw'][t], lo + (hi - lo) * grid[min(idx[-1] + 1, g)])
    assert np.array_equal(cs['lower'], np.maximum.accumulate(cs['lower_raw']))
    assert np.array_equal(cs['upper'], np.minimum.accumulate(cs['upper_raw']))
    assert cs['final'] == (cs['lower'][-1], cs['upper'][-1]) and not cs['empty']


def test_confidence_sequence_covers_the_true_mean_uniformly_at_the_stated_rate():
    alpha, reps, n = 0.2, 120, 60
    tol = 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)
    rng = np.random.default_rng(7)
    for draw, lo, hi, mu in ((lambda: rng.beta(2.0, 5.0, n), 0.0, 1.0, 2.0 / 7.0),
                             (lambda: 3.0 + 2.0 * (rng.random(n) < 0.3), 3.0, 5.0, 3.6)):
        misses, widths = 0, []
        for _ in range(reps):
            cs = A.betting_confidence_sequence(draw(), lo, hi, alpha, grid_size=100)
            misses += not bool(np.all((cs['lower'] <= mu) & (mu <= cs['upper'])))
            widths.append(cs['upper'][-1] - cs['lower'][-1])
            assert np.all(np.diff(cs['lower']) >= 0) and np.all(np.diff(cs['upper']) <= 0)
            assert lo <= cs['lower'][0] and cs['upper'][0] <= hi
        assert misses / reps <= alpha + tol                    # time-uniform coverage >= 1 - alpha
        assert np.mean(widths) < 0.6 * (hi - lo)               # and the sequence is informative


def test_confidence_sequence_concentrates_on_constant_data_and_mechanisms_change_it():
    x = np.full(400, 0.3)
    cs = A.betting_confidence_sequence(x, 0.0, 1.0, 0.05, grid_size=200)
    assert cs['lower'][-1] <= 0.3 <= cs['upper'][-1] and cs['upper'][-1] - cs['lower'][-1] <= 0.05
    data = np.random.default_rng(3).beta(2.0, 8.0, 80)
    base = A.betting_confidence_sequence(data, 0.0, 1.0, 0.05, grid_size=200)
    for kw in ({'theta': 0.9}, {'c': 0.9}, {'alpha': 0.3}):
        other = A.betting_confidence_sequence(data, 0.0, 1.0, **{'alpha': 0.05, 'grid_size': 200, **kw})
        assert not (np.array_equal(other['lower_raw'], base['lower_raw'])
                    and np.array_equal(other['upper_raw'], base['upper_raw'])), kw


# ====================================================================== anytime: e-process
@pytest.mark.parametrize('n,support,probs', [(7, (-1.0, 0.0, 1.0), (0.25, 0.5, 0.25)), (10, (-1.0, 1.0), (0.5, 0.5))])
def test_e_process_is_an_exact_test_martingale_on_an_enumerable_null(n, support, probs):
    """Exact expectation over every path: E[E_n] = 1, E[E_tau] = 1 at the stopping time, P(sup E >= 4) <= 1/4."""
    sup, pr = np.array(support), np.array(probs)
    e_final = e_plus = e_minus = e_stopped = p_cross = 0.0
    for idx in itertools.product(range(len(sup)), repeat=n):
        path = list(idx)
        p = float(np.prod(pr[path]))
        ep = A.paired_e_process(sup[path], 1.0)
        assert np.allclose(ep['e_process'], 0.5 * (ep['e_plus'] + ep['e_minus']))
        assert np.all(ep['e_process'] > 0.0)
        e_final += p * ep['e_process'][-1]
        e_plus += p * ep['e_plus'][-1]
        e_minus += p * ep['e_minus'][-1]
        hit = np.flatnonzero(ep['e_process'] >= 4.0)
        p_cross += p * bool(hit.size)
        e_stopped += p * ep['e_process'][hit[0] if hit.size else -1]
    for value in (e_final, e_plus, e_minus, e_stopped):
        assert abs(value - 1.0) < 1e-9
    assert 0.0 < p_cross <= 0.25                                # Ville: P(sup_t E_t >= 1/alpha) <= alpha


def test_e_values_have_mean_at_most_one_under_the_null_with_optional_stopping():
    alpha, reps, n = 0.1, 300, 80
    rng = np.random.default_rng(11)
    for draw in (lambda: rng.uniform(-1.0, 1.0, n), lambda: rng.beta(0.5, 4.0, n) - 1.0 / 9.0):
        finals, crossed = [], 0
        for _ in range(reps):
            ep = A.paired_e_process(draw(), 1.0)
            assert np.all(np.isfinite(ep['e_process'])) and np.all(ep['e_process'] > 0.0)
            finals.append(ep['e_process'][-1])
            crossed += bool(np.max(ep['e_process']) >= 1.0 / alpha)    # a look after every observation
        finals = np.array(finals)
        assert finals.mean() <= 1.0 + 3.0 * finals.std(ddof=1) / math.sqrt(reps)
        assert crossed / reps <= alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)


def test_naive_peeking_t_test_inflates_type_one_error_on_the_same_null_streams():
    rng = np.random.default_rng(5)
    reps, n, alpha = 200, 150, 0.05
    any_look = final_look = e_rej = 0
    for _ in range(reps):
        d = rng.uniform(-1.0, 1.0, n)
        naive = A.naive_peeking_t_test(d, alpha, min_n=10)
        any_look += naive['rejected_any_look']
        final_look += naive['rejected_final_look']
        e_rej += bool(np.max(A.paired_e_process(d, 1.0)['e_process']) >= 1.0 / alpha)
    assert any_look / reps > 2.0 * alpha                      # peeking breaks the fixed-n guarantee
    assert final_look / reps <= alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)
    assert e_rej / reps <= alpha + 3.0 * math.sqrt(alpha * (1.0 - alpha) / reps)


def test_naive_peeking_t_test_matches_scipy_at_every_look():
    from scipy.stats import ttest_1samp
    rng = np.random.default_rng(8)
    seen = set()
    for _ in range(20):
        d = rng.normal(0.3, 1.0, 25)
        res = A.naive_peeking_t_test(d, 0.05, min_n=5)
        pvalues = [ttest_1samp(d[:n], 0.0).pvalue for n in range(5, 26)]
        assert res['rejected_any_look'] == any(p < 0.05 for p in pvalues)
        assert res['rejected_final_look'] == (pvalues[-1] < 0.05)
        seen.add((res['rejected_any_look'], res['rejected_final_look']))
    assert (True, False) in seen                               # some streams were "significant" only at a peek


def test_e_process_bounds_and_mechanisms():
    big = A.paired_e_process(np.ones(5000), 1.0)                # log-capital ~ 2000 > the exp(690) cap
    assert np.all(np.isfinite(big['e_process'])) and big['e_process'][-1] == pytest.approx(math.exp(690.0))
    assert big['log_e_process'][-1] > 690.0 and big['e_minus'][-1] == 1.0
    d = np.random.default_rng(2).uniform(-0.5, 1.0, 60)
    assert not np.array_equal(A.paired_e_process(d, 1.0, c=0.5)['e_process'],
                              A.paired_e_process(d, 1.0, c=0.9)['e_process'])
    assert np.allclose(A.paired_e_process(d, 1.0)['e_process'], A.paired_e_process(2.0 * d, 2.0)['e_process'])
    flat = A.paired_e_process(np.zeros(50), 1.0)
    assert np.all(flat['e_process'] == 1.0)                     # no evidence, no bet: wealth stays at 1


def test_anytime_compare_detects_direction_and_stays_quiet_without_an_effect():
    rng = np.random.default_rng(3)
    cand = (rng.random(400) < 0.8).astype(float)
    ref = (rng.random(400) < 0.4).astype(float)
    res = A.anytime_compare(cand, ref, bound=1.0)
    assert res['rejected'] and res['verdict'] == 'improved' and res['direction'] == 'positive'
    assert res['e_value'] >= 20.0 and 1 <= res['stopped_at'] <= 400 and res['cs_lower'] > 0.0
    assert res['cs_lower'] <= res['mean_difference'] <= res['cs_upper']
    assert A.anytime_compare(cand, ref, bound=1.0, higher_is_better=False)['verdict'] == 'worse'
    swapped = A.anytime_compare(ref, cand, bound=1.0)
    assert swapped['verdict'] == 'worse' and swapped['direction'] == 'negative' and swapped['cs_upper'] < 0.0
    same = A.anytime_compare(cand, cand, bound=1.0)
    assert not same['rejected'] and same['verdict'] == 'no_detectable_difference' and same['e_value'] == 1.0
    assert same['stopped_at'] is None and same['cs_lower'] <= 0.0 <= same['cs_upper']
    assert set(res) >= {'e_value', 'rejected', 'cs_lower', 'cs_upper', 'verdict'}
    json.dumps(res)


@pytest.mark.parametrize('call', [
    lambda: A.betting_confidence_sequence([], 0.0, 1.0),
    lambda: A.betting_confidence_sequence([[0.1, 0.2]], 0.0, 1.0),
    lambda: A.betting_confidence_sequence([0.1, float('nan')], 0.0, 1.0),
    lambda: A.betting_confidence_sequence([0.1, 1.2], 0.0, 1.0),
    lambda: A.betting_confidence_sequence([0.1], 1.0, 1.0),
    lambda: A.betting_confidence_sequence([0.1], 0.0, 1.0, 0.0),
    lambda: A.betting_confidence_sequence([0.1], 0.0, 1.0, 1.0),
    lambda: A.betting_confidence_sequence([0.1], 0.0, 1.0, grid_size=4),
    lambda: A.betting_confidence_sequence([0.1], 0.0, 1.0, c=1.0),
    lambda: A.betting_confidence_sequence([0.1], 0.0, 1.0, theta=0.0),
    lambda: A.betting_confidence_sequence('0.1', 0.0, 1.0),
    lambda: A.paired_e_process([0.5], 0.0),
    lambda: A.paired_e_process([1.5], 1.0),
    lambda: A.paired_e_process([0.5], 1.0, c=0.0),
    lambda: A.anytime_compare([1.0, 0.0], [0.0], bound=1.0),
    lambda: A.anytime_compare([1.0], [0.0], bound=0.5),
    lambda: A.anytime_compare([1.0], [0.0], bound=1.0, higher_is_better=1),
    lambda: A.naive_peeking_t_test([0.1, 0.2], min_n=1),
    lambda: A.run_experiment(quick=1),
    lambda: A.run_experiment(seeds=()),
    lambda: A.run_experiment(seeds=(1, 1)),
    lambda: A.run_experiment(surprise=1),
    lambda: A.run_experiment(effect_p=0.4, effect_reference_p=0.5),
    lambda: A.run_experiment(replicates=5),
    lambda: A.run_experiment(replicates=10_000, horizon=1_000, grid_size=8),     # memory bound on simulated cells
])
def test_anytime_api_rejects_bad_input(call):
    with pytest.raises(ValueError):
        call()


@pytest.fixture(scope='module')
def anytime_quick():
    before = _global_rng_state()
    receipt = A.run_experiment(quick=True)
    assert _same_rng_state(before, _global_rng_state())
    return receipt


def test_anytime_quick_receipt_is_sealed_and_passes(anytime_quick):
    r = anytime_quick
    assert verify_receipt(r) and r['kind'] == 'validation' and r['experiment_id'] == 'anytime_inference'
    assert r['evidence_tier'] == EVIDENCE_TIER and r['verdict'] == 'pass' and r['verdict_basis'] == 'all_checks_pass'
    assert r['checks'] and all(r['checks'].values())
    cfg = r['config']
    assert cfg['quick'] is True and cfg['seeds'] == [0, 1, 2] and cfg['replicates'] == 100 and cfg['horizon'] == 200
    for null in ('null_paired_bernoulli', 'null_skewed_continuous'):
        res = r['results'][null]
        assert res['e_process_type1']['rate'] <= 0.05 + 3 * res['e_process_type1']['mc_se'] + 1e-12
        assert res['naive_peeking_t_type1']['rate'] > 0.1
        assert res['final_e_value']['mean'] <= 1.0 + 3 * res['final_e_value']['mc_se']
    for cover in ('cover_beta', 'cover_bernoulli'):
        assert r['results'][cover]['uniform_coverage']['rate'] >= 0.95 - 3 * r['results'][cover]['uniform_coverage']['mc_se']
    assert r['results']['effect']['power']['rate'] >= 0.8
    assert r['measured_wallclock_s'] > 0.0 and set(r['estimated_cost']) and r['notes']
    tampered = copy.deepcopy(r)
    tampered['results']['effect']['power']['rate'] = 0.0
    assert not verify_receipt(tampered)


def test_anytime_receipt_is_deterministic_and_seed_sensitive():
    small = {'replicates': 12, 'horizon': 30, 'grid_size': 16, 'power_min': 0.0, 'mc_z': 10.0}
    first = A.run_experiment(seeds=(4,), **small)
    assert verify_receipt(first) and first['digest'] == A.run_experiment(seeds=(4,), **small)['digest']
    assert A.run_experiment(seeds=(5,), **small)['digest'] != first['digest']
    assert A.run_experiment(seeds=(4,), **{**small, 'c': 0.75})['digest'] != first['digest']


# ====================================================================== joint failure: exact statistics
def _hand_case():
    """3 components, 1 condition, 5 trials with failing sets {}, {0}, {0,1}, {1,2}, {0,1,2}."""
    f = np.zeros((3, 1, 5), dtype=bool)
    for t, members in enumerate(((), (0,), (0, 1), (1, 2), (0, 1, 2))):
        f[list(members), 0, t] = True
    return f


def test_majority_vote_error_from_joint_failures_is_exact_on_hand_cases():
    f = _hand_case()                                           # wrong-member counts W = 0, 1, 2, 2, 3
    assert np.array_equal(J.majority_error(f, (0, 1, 2)), [3 / 5])
    assert np.array_equal(J.majority_error(f, (1,)), [3 / 5]) and np.array_equal(J.majority_error(f, (2,)), [2 / 5])
    assert np.array_equal(J.majority_error_from_signatures(J.failure_signatures(f), (2, 0, 1)), [3 / 5])
    # pairwise co-failures 2 + 1 + 2 = 5 of 5 trials = Pr(W = 2) + 3 Pr(W = 3) = 2/5 + 3/5
    assert J.pairwise_bound(f, (0, 1, 2))[0] == pytest.approx(1.0)
    ev = J.evaluate_panel(f, [0, 1, 2])
    assert ev['condition_error'] == [0.6] and ev['unanimous_failure_rate'] == pytest.approx(0.2)
    # 5-member panel: majority needs 3 wrong; trial patterns have W = 2, 3, 5, 0
    g = np.zeros((5, 2, 4), dtype=bool)
    for t, members in enumerate(((0, 1), (0, 1, 2), (0, 1, 2, 3, 4), ())):
        g[list(members), 0, t] = True
    g[[3, 4], 1, :2] = True
    assert np.array_equal(J.majority_error(g, range(5)), [0.5, 0.0])
    # the parity case of FAULTLINE: same marginals and pair rates, majority error 3/4 vs 1/4
    par = J.constructed_cases()['parity']['failures']
    assert np.array_equal(J.majority_error(par, (0, 1, 2)), [0.75, 0.25])
    assert np.allclose(par.mean(axis=2), 0.5) and np.allclose(J.pairwise_bound(par, (0, 1, 2)), [0.75, 0.75])


def test_signature_path_bound_and_identity_on_random_correlated_failures():
    rng = np.random.default_rng(9)
    for _ in range(10):
        shared = rng.random((1, 3, 40)) < 0.3
        f = (rng.random((6, 3, 40)) < 0.15) | (shared & (rng.random((6, 3, 40)) < 0.6))
        sigs = J.failure_signatures(f)
        assert all(sum(s['count'] for s in c['signatures']) == 40 and c['components'] == 6 for c in sigs)
        for panel in itertools.combinations(range(6), 3):
            err = J.majority_error(f, panel)
            assert np.array_equal(J.majority_error_from_signatures(sigs, panel), err)
            w = f[list(panel)].sum(axis=0)
            assert np.allclose(J.pairwise_bound(f, panel), (w == 2).mean(axis=1) + 3 * (w == 3).mean(axis=1))
            assert np.all(J.pairwise_bound(f, panel) >= err - 1e-12)
        assert np.array_equal(J.majority_error(f, (0, 2, 3, 4, 5)),
                              (f[[0, 2, 3, 4, 5]].sum(axis=0) >= 3).mean(axis=1))
    scores = J.panel_scores(f, 3)
    assert np.allclose(np.array(scores['majority_failures_by_condition'][5]) / 40,
                       J.majority_error(f, scores['panels'][5]))


def _reference_select(f, k, policy):
    """FAULTLINE's selector keys written directly with exact fractions (its core.select_panel, re-derived)."""
    m, kc, t = f.shape
    s = (k + 1) // 2
    best_key, best_panel = None, None
    for panel in itertools.combinations(range(m), k):
        idx = list(panel)
        ind = sum(Fraction(int(f[i].sum()), kc * t) for i in idx)
        wrong = f[idx].sum(axis=0)
        maj = [Fraction(int((wrong[c] >= s).sum()), t) for c in range(kc)]
        if k == 1:
            pair = [Fraction(int(f[idx[0], c].sum()), t) for c in range(kc)]
        else:
            pair = [sum(Fraction(int((f[i, c] & f[j, c]).sum()), t) for i, j in itertools.combinations(idx, 2))
                    / math.comb(s, 2) for c in range(kc)]
        key = {'accuracy': (ind,), 'pairwise': (max(pair), sum(pair) / kc, ind),
               'pooled': (sum(maj) / kc, max(maj), ind), 'joint': (max(maj), sum(maj) / kc, ind)}[policy] + (panel,)
        if best_key is None or key < best_key:
            best_key, best_panel = key, panel
    return best_panel


def test_selectors_match_a_direct_implementation_with_exact_tie_breaking():
    rng = np.random.default_rng(21)
    for trial in range(12):
        m, kc, t = 6, 2 + trial % 2, 10
        shared = rng.random((1, kc, t)) < 0.35
        f = (rng.random((m, kc, t)) < 0.2) | (shared & (rng.random((m, kc, t)) < 0.5))
        for k in (1, 3, 5):
            picks = J.select_panels(f, k)
            for policy in J.POLICIES:
                assert picks[policy] == _reference_select(f, k, policy), (trial, k, policy)
                assert J.select_panel(f, k, policy) == picks[policy]


def _accuracy_trap():
    """0-2 individually best but fail together in 'storm'; 3 and 4 slightly worse and never co-fail."""
    f = np.zeros((5, 2, 20), dtype=bool)
    f[3, 0, 0:3] = f[4, 0, 3:6] = True                        # calm
    f[[0, 1, 2], 1, 0:4] = True                               # storm: a common cause hits 0-2 together
    f[3, 1, 4:7] = f[4, 1, 7:10] = True
    return f


def test_selection_methods_differ_in_the_expected_direction_on_correlated_failures():
    f = _accuracy_trap()
    assert list(f.mean(axis=(1, 2))) == [0.1, 0.1, 0.1, 0.15, 0.15]
    picks = J.select_panels(f, 3)
    assert picks['accuracy'] == (0, 1, 2) and np.array_equal(J.majority_error(f, picks['accuracy']), [0.0, 0.2])
    assert picks['joint'] == (0, 3, 4) and np.array_equal(J.majority_error(f, picks['joint']), [0.0, 0.0])
    assert picks['pairwise'] == picks['pooled'] == (0, 3, 4)
    # held-out trials with the same structure: the frozen panels keep the same ordering
    test = np.concatenate([np.roll(f, 3, axis=2), f], axis=2)
    sel = J.run_selection(f, test, 3)
    worst = {p: sel['policies'][p]['test']['worst_condition_error'] for p in J.POLICIES}
    assert worst['joint'] < worst['accuracy'] and worst['joint'] == 0.0
    assert all(sel['oracle']['test']['worst_condition_error'] <= w for w in worst.values())
    # pairwise misranks rare unanimous failures; pooled hides a bad condition (module's own constructed cases)
    checks = J.check_constructed_cases()
    assert checks and all(checks.values())
    cases = J.constructed_cases()
    pt = cases['pairwise_trap']['failures']
    pick = J.select_panels(pt, 3)
    assert J.majority_error(pt, pick['joint'])[0] < J.majority_error(pt, pick['pairwise'])[0]
    assert J.pairwise_bound(pt, pick['joint'])[0] > J.pairwise_bound(pt, pick['pairwise'])[0]
    po = cases['pooled_trap']['failures']
    pick = J.select_panels(po, 3)
    assert J.majority_error(po, pick['joint']).max() < J.majority_error(po, pick['pooled']).max()
    assert J.majority_error(po, pick['joint']).mean() > J.majority_error(po, pick['pooled']).mean()


# ====================================================================== joint failure: generator
def test_generator_is_deterministic_and_documents_its_structure():
    a = J.make_failure_data(3, calibration_trials=40, test_trials=60)
    b = J.make_failure_data(3, calibration_trials=40, test_trials=60)
    assert a['digest'] == b['digest'] and np.array_equal(a['test'], b['test'])
    assert J.make_failure_data(4, calibration_trials=40, test_trials=60)['digest'] != a['digest']
    assert a['calibration'].shape == (9, 4, 40) and a['test'].shape == (9, 4, 60) and a['calibration'].dtype == bool
    assert a['conditions'] == ['nominal', 'stress_0', 'stress_1', 'stress_2']
    fam = np.array(a['structure']['family'])
    assert sorted(np.bincount(fam).tolist()) == [3, 3, 3] and list(fam) != sorted(fam)   # shuffled per seed
    shift = J.make_failure_data(3, scenario='shift', calibration_trials=40, test_trials=60)
    assert np.array_equal(shift['calibration'], a['calibration'])           # the common cause is unseen
    assert np.array_equal(shift['test'][:, 1:], a['test'][:, 1:]) and np.all(shift['test'] >= a['test'])
    assert shift['test'][:, 0].sum() > a['test'][:, 0].sum()
    hidden = shift['structure']['hidden_common_cause']['members']
    base = np.array(a['structure']['base_error'])
    for f in range(3):                                         # 'hardened': the most accurate member of each family
        members = np.flatnonzero(fam == f)
        assert members[np.argmin(base[members])] in hidden
    rand = J.make_failure_data(3, scenario='shift', shift_target='random', calibration_trials=40, test_trials=60)
    assert sorted(fam[rand['structure']['hidden_common_cause']['members']].tolist()) == [0, 1, 2]


def test_population_errors_match_monte_carlo_and_the_independent_control_ranks_like_accuracy():
    data = J.make_failure_data(1, scenario='shift', families=2, family_size=3, calibration_trials=8,
                               test_trials=40_000)
    for panel in ((0, 1, 2), (0, 3, 5), (1, 2, 4)):
        for split, arr in (('test', data['test']), ('calibration', None)):
            exp = J.expected_condition_errors(data['structure'], panel, split=split)
            assert np.all((exp >= 0.0) & (exp <= 1.0))
            if arr is not None:
                emp = J.majority_error(arr, panel)
                assert np.all(np.abs(emp - exp) <= 5.0 * np.sqrt(exp * (1.0 - exp) / 40_000) + 1e-3)
    hidden = data['structure']['hidden_common_cause']['members']
    assert (J.expected_condition_errors(data['structure'], hidden + [next(i for i in range(6) if i not in hidden)],
                                        split='test')[0]
            > J.expected_condition_errors(data['structure'], hidden + [next(i for i in range(6) if i not in hidden)],
                                          split='calibration')[0])
    ind = J.make_failure_data(2, scenario='independent')['structure']
    best = tuple(sorted(np.argsort(ind['independent_error'])[:3].tolist()))
    worst = {p: J.expected_condition_errors(ind, p).max() for p in itertools.combinations(range(9), 3)}
    assert min(worst, key=lambda p: (worst[p], p)) == best


# ====================================================================== joint failure: receipts
def test_joint_failure_quick_receipt_is_sealed_deterministic_and_seed_sensitive():
    before = _global_rng_state()
    first = J.run_experiment(quick=True)
    assert _same_rng_state(before, _global_rng_state())
    assert verify_receipt(first) and first['digest'] == J.run_experiment(quick=True)['digest']
    assert first['kind'] == 'selection_benchmark' and first['experiment_id'] == 'joint_failure'
    assert first['evidence_tier'] == EVIDENCE_TIER and first['verdict'] in VERDICTS
    assert first['verdict'] == first['comparisons']['shared_joint_vs_accuracy']['verdict']
    assert first['comparisons']['shared_joint_vs_accuracy']['higher_is_better'] is False
    assert first['constructed_checks_pass'] is True and first['config']['quick'] is True
    assert set(first['summary']) == set(J.SCENARIOS) and len(first['per_seed']['shared']) == 3
    for scenario in J.SCENARIOS:
        for row in first['per_seed'][scenario]:
            oracle = row['policies']['oracle']['test_worst_condition_error']
            for policy in J.POLICIES:
                rec = row['policies'][policy]
                assert 0.0 <= rec['test_worst_condition_error'] <= 1.0 and oracle <= rec['test_worst_condition_error']
                assert 0.0 <= rec['expected_worst_condition_error'] <= 1.0
    tampered = copy.deepcopy(first)
    tampered['summary']['shared']['joint']['test_worst_condition_error'] = 0.5
    assert not verify_receipt(tampered)
    assert J.run_experiment(quick=True, seeds=(5, 6))['digest'] != first['digest']
    assert J.run_experiment(quick=True, shift_target='random')['digest'] != first['digest']


@pytest.fixture(scope='module')
def neuroweave_quick():
    before = _global_rng_state()
    receipt = J.run_neuroweave_application(quick=True)
    assert _same_rng_state(before, _global_rng_state())
    return receipt


def test_neuroweave_application_quick_receipt(neuroweave_quick):
    r = neuroweave_quick
    assert verify_receipt(r) and r['kind'] == 'selection_application' and r['experiment_id'] == 'joint_failure_neuroweave'
    assert r['evidence_tier'] == EVIDENCE_TIER and r['verdict'] in VERDICTS
    assert r['verdict'] == r['comparisons']['joint_vs_accuracy']['verdict']
    cfg = r['config']
    assert cfg['quick'] is True and cfg['tasks'] == ['delayed_recall', 'xor', 'rhythm', 'context_switch']
    assert len(cfg['mechanism_sets']) == 5 and r['budget']['candidate_panels'] == math.comb(5, 3)
    for row in r['per_seed']:
        assert len(row['components']) == 5 and [c['mechanisms'] for c in row['components']] == cfg['mechanism_sets']
        assert len({c['train_stream_digest'] for c in row['components']}) == 1          # same streams for all
        for policy in J.POLICIES:
            rec = row['policies'][policy]
            assert len(rec['panel']) == 3 and row['oracle']['test_worst_condition_error'] <= rec['test_worst_condition_error']
            assert len(rec['test_condition_error']) == 4
        agree = row['calibration_failure_agreement']
        assert 0.0 <= agree['min_offdiagonal'] <= agree['mean_offdiagonal'] <= 1.0
    # the mechanism set changes the readout failures: in every seed at least one pair of components differs
    assert all(row['calibration_failure_agreement']['min_offdiagonal'] < 1.0 for row in r['per_seed'])
    assert len({json.dumps(row['policies'][p]['panel']) for row in r['per_seed'] for p in J.POLICIES}) > 1
    assert len(json.dumps(r)) < 200_000


def test_neuroweave_application_is_deterministic_and_validates_arguments():
    tiny = {'mechanism_sets': ((), ('sparse_code',), ('multi_soma',)), 'tasks': ('xor', 'rhythm'), 'neurons': 8,
            'train_steps': 8, 'calibration_steps': 6, 'test_steps': 6}
    first = J.run_neuroweave_application(seeds=(0,), **tiny)
    assert verify_receipt(first) and first['digest'] == J.run_neuroweave_application(seeds=(0,), **tiny)['digest']
    assert J.run_neuroweave_application(seeds=(1,), **tiny)['digest'] != first['digest']
    for bad in ({'mechanism_sets': (('no_such_mechanism',), ())}, {'mechanism_sets': ((), ())},
                {'mechanism_sets': ((), ('sparse_code',))}, {'tasks': ('xor', 'xor')}, {'tasks': ('nope',)},
                {'panel_size': 2}, {'neurons': 4}, {'test_seed_offset': 50_000}, {'test_seed_offset': 50_001},
                {'surprise': 1}, {'quick': 'yes'}):
        with pytest.raises(ValueError):
            J.run_neuroweave_application(**{**tiny, **bad})


@pytest.mark.parametrize('call', [
    lambda: J.validate_failures(np.zeros((2, 2))),
    lambda: J.validate_failures(np.zeros((0, 1, 1))),
    lambda: J.validate_failures(np.full((1, 1, 2), 2)),
    lambda: J.validate_failures(np.full((1, 1, 2), np.nan)),
    lambda: J.validate_failures(np.zeros((25, 1, 1))),
    lambda: J.validate_failures([[['x']]]),
    lambda: J.validate_failures('abc'),
    lambda: J.majority_error(_hand_case(), (0, 1)),
    lambda: J.majority_error(_hand_case(), (0, 0, 1)),
    lambda: J.majority_error(_hand_case(), (0, 1, 3)),
    lambda: J.majority_threshold(4),
    lambda: J.select_panels(_hand_case(), 5),
    lambda: J.select_panels(_hand_case(), 3, ('accuracy', 'magic')),
    lambda: J.select_panel(_hand_case(), 3, 'magic'),
    lambda: J.run_selection(_hand_case(), np.zeros((3, 2, 5), dtype=bool), 3),
    lambda: J.majority_error_from_signatures(J.failure_signatures(_hand_case()), (0, 1, 5)),
    lambda: J.majority_error_from_signatures([{'trials': 2, 'signatures': [{'members': [0], 'count': 1}]}], (0,)),
    lambda: J.majority_error_from_signatures([], (0,)),
    lambda: J.make_failure_data(0, scenario='storm'),
    lambda: J.make_failure_data(0, families=5, family_size=5),
    lambda: J.make_failure_data(0, stress_hit=1.5),
    lambda: J.make_failure_data(0, nonsense=1),
    lambda: J.expected_condition_errors({'scenario': 'x'}, (0,)),
    lambda: J.expected_condition_errors({'scenario': 'shared'}, (0,)),
    lambda: J.expected_condition_errors({**J.make_failure_data(0, calibration_trials=4, test_trials=4)['structure'],
                                         'base_error': [1.5] * 9}, (0,)),
    lambda: J.expected_condition_errors(J.make_failure_data(0, calibration_trials=4, test_trials=4)['structure'],
                                        (0,), split='train'),
    lambda: J.run_experiment(quick='yes'),
    lambda: J.run_experiment(seeds=(1, 1)),
    lambda: J.run_experiment(panel_size=4),
    lambda: J.run_experiment(panel_size=11),
    lambda: J.run_experiment(surprise=1),
])
def test_joint_failure_api_rejects_bad_input(call):
    with pytest.raises(ValueError):
        call()
