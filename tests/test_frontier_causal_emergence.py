"""Frontier module ``causal_emergence``: effective information and its decomposition, coarse-graining search,
TPM estimation, the successor and independent-units nulls, the live observer and the lab receipt."""
import copy
import json
import math
import sys

import numpy as np
import pytest
from scipy.stats import entropy as kl_entropy

from synapse.frontier import causal_emergence as CE
from synapse.frontier.harness import verify_receipt
from synapse.frontier.spec import EVIDENCE_TIER, FrontierSpecV1, MODULE_IDS, OBSERVER_MODULE_IDS

PAIRS = [[0, 1], [2, 3], [4, 5]]
SEVEN_PLUS_ONE = [0] * 7 + [1]


def _random_tpm(rng, n, power=3.0):
    T = rng.random((n, n)) ** power
    T[0] = 0.0
    T[0, n - 1] = 1.0                                           # one deterministic row (zeros in the TPM)
    return T / T.sum(axis=1, keepdims=True)


def _kl_reference_ei(T):
    """EI as the average KL divergence of each row from the mean row (scipy reference, bits)."""
    m = T.mean(axis=0)
    return float(np.mean([kl_entropy(row, m, base=2) for row in T]))


def _global_rng_state():
    out = {'numpy': np.random.get_state()[1].copy()}
    torch = sys.modules.get('torch')
    if torch is not None:
        out['torch'] = torch.random.get_rng_state().clone()
        out['threads'] = torch.get_num_threads()
    return out


def _same_global_state(a, b):
    if not np.array_equal(a['numpy'], b['numpy']) or set(a) != set(b):
        return False
    return 'torch' not in a or (bool((a['torch'] == b['torch']).all()) and a['threads'] == b['threads'])


# ====================================================================== registry
def test_module_is_registered_as_an_observer_with_one_experiment():
    assert CE.MODULE_ID == 'causal_emergence' and CE.MODULE_ID in MODULE_IDS and CE.MODULE_ID in OBSERVER_MODULE_IDS
    assert list(CE.EXPERIMENTS) == ['causal_emergence'] and CE.EXPERIMENTS['causal_emergence'] is CE.run_experiment
    assert CE.CausalEmergenceObserver.module_id == CE.MODULE_ID


# ====================================================================== effective information
def test_effective_information_known_answers_and_decomposition():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 8, 16):
        perm = np.eye(n)[rng.permutation(n)]
        assert CE.effective_information(np.eye(n)) == pytest.approx(math.log2(n), abs=1e-12)
        assert CE.effective_information(perm) == pytest.approx(math.log2(n), abs=1e-12)
        assert CE.effective_information(np.full((n, n), 1.0 / n)) == pytest.approx(0.0, abs=1e-12)
        ident, uni = CE.ei_decomposition(np.eye(n)), CE.ei_decomposition(np.full((n, n), 1.0 / n))
        assert (ident['determinism'], ident['degeneracy'], ident['effectiveness']) == pytest.approx((1.0, 0.0, 1.0))
        assert (uni['determinism'], uni['degeneracy'], uni['ei']) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
        sink = np.zeros((n, n))
        sink[:, 0] = 1.0                                         # deterministic but fully degenerate
        d = CE.ei_decomposition(sink)
        assert (d['ei'], d['determinism'], d['degeneracy']) == pytest.approx((0.0, 1.0, 1.0), abs=1e-12)
    for n in (3, 6, 11):                                         # EI = average KL from the mean row = size * eff
        T = _random_tpm(rng, n)
        d = CE.ei_decomposition(T)
        assert d['ei'] == pytest.approx(_kl_reference_ei(T), abs=1e-12)
        assert d['ei'] == pytest.approx(math.log2(n) * d['effectiveness'], abs=1e-12)
        assert d['ei'] == pytest.approx(d['effect_entropy'] - d['mean_row_entropy'], abs=1e-12)
    ex = CE.hoel2017_examples()                                  # Hoel (2017): EI(M1, M2, M3) = 2, 1, 0 bits
    assert [CE.effective_information(ex[k]) for k in ('M1', 'M2', 'M3')] == pytest.approx([2.0, 1.0, 0.0], abs=1e-12)
    m2 = CE.ei_decomposition(ex['M2'])
    assert m2['determinism'] == pytest.approx(1.0 - math.log2(3.0) / 4.0, abs=1e-12) and m2['effectiveness'] == 0.5
    ch = CE.ei_decomposition(ex['channel4'])                     # printed: EI 0.81 bits, H(X|Y) 1.19 bits
    assert round(ch['ei'], 2) == 0.81 and round(2.0 - ch['ei'], 2) == 1.19
    assert CE.entropy_bits([0.5, 0.5]) == 1.0 and CE.entropy_bits([1.0, 0.0]) == 0.0
    for bad in (np.ones((2, 3)) / 3, [[0.5, 0.6], [0.5, 0.5]], [[1.5, -0.5], [0.5, 0.5]], [[np.nan, 1.0], [0.5, 0.5]],
                np.zeros((0, 0)), np.full((2049, 2049), 1 / 2049), 'tpm'):
        with pytest.raises(ValueError):
            CE.effective_information(bad)
    with pytest.raises(ValueError):
        CE.entropy_bits([0.5, 0.4])


def test_hoel_published_examples_and_exhaustive_partition_search():
    ex, p = CE.hoel2017_examples(), CE.PRINTED_VALUES
    chain = CE.ei_decomposition(ex['chain8'])
    assert chain['ei'] == pytest.approx(3.0 - 7.0 / 8.0 * math.log2(7.0), abs=1e-12)        # 0.5436 bits
    # Hoel (2017) prints EI(Sm) = 0.55 bits and CE = 0.45 bits; the closed form gives 0.5436 and 0.4564, so the
    # printed values are checked to within 0.01, not to two decimals.
    assert abs(chain['ei'] - p['hoel2017_chain8_ei_micro']) < 0.01 and round(chain['effectiveness'], 2) == 0.18
    best = CE.best_state_partition(ex['chain8'])
    assert best['partitions_evaluated'] == 4140 - 2                     # Bell(8) minus the 1-state and 8-state ones
    assert best['labels'].tolist() == SEVEN_PLUS_ONE and best['ei_macro'] == pytest.approx(1.0, abs=1e-12)
    assert best['ce'] == pytest.approx(1.0 - chain['ei'], abs=1e-12) and abs(best['ce'] - p['hoel2017_chain8_ce']) < 0.01
    assert np.allclose(best['macro_tpm'], np.eye(2))
    varied = CE.best_state_partition(ex['chain8_varied'])
    assert round(varied['ei_micro'], 2) == p['hoel2017_chain8_varied_ei_micro'] and varied['ei_macro'] == pytest.approx(1.0)
    assert varied['labels'].tolist() == SEVEN_PLUS_ONE
    channel = CE.best_state_partition(ex['channel4'])
    assert channel['labels'].tolist() == [0, 0, 0, 1] and channel['ei_macro'] == pytest.approx(1.0, abs=1e-12)
    T, labels = CE.and_gate_network(3)                                   # six AND gates -> three COPY gates
    assert T.shape == (64, 64) and np.all(T.sum(axis=1) == 1.0) and np.all(np.isin(T, (0.0, 1.0)))
    r = CE.causal_emergence(T, labels)
    assert r['micro']['ei'] == pytest.approx(3.0 * CE.entropy_bits([0.25, 0.75]), abs=1e-12)
    assert round(r['micro']['ei'], 2) == p['hoel2013_and_ei_micro'] and r['micro']['determinism'] == 1.0
    assert round(r['micro']['degeneracy'], 2) == p['hoel2017_and_degeneracy']
    assert round(r['micro']['effectiveness'], 2) == p['hoel2017_and_effectiveness']
    assert round(r['micro']['degeneracy'], 1) == p['hoel2013_and_degeneracy']
    assert r['macro']['ei'] == pytest.approx(3.0, abs=1e-12) and round(r['ce'], 2) == p['hoel2013_and_ce']
    assert np.array_equal(r['macro_tpm'][np.argsort(r['macro_tpm'].argmax(axis=1))], np.eye(8)[np.sort(
        r['macro_tpm'].argmax(axis=1))]) and r['macro']['effectiveness'] == pytest.approx(1.0)   # macro is a permutation


# ====================================================================== coarse-graining
def test_coarse_graining_matches_hand_computation_and_the_batch_kernel_matches_a_reference():
    T = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]])
    assert np.allclose(CE.coarse_grain_tpm(T, [0, 0, 1]), [[0.75, 0.25], [1.0, 0.0]])
    perm = [2, 0, 1]                                                       # a relabelling, not a coarse-graining
    assert CE.effective_information(CE.coarse_grain_tpm(T, perm)) == pytest.approx(CE.effective_information(T))
    rng = np.random.default_rng(3)
    for n in (5, 6):
        T = _random_tpm(rng, n)
        parts = CE.set_partitions(n)
        fast = CE.macro_ei_batch(T, parts)
        ref = [_kl_reference_ei(CE.coarse_grain_tpm(T, lab)) for lab in parts]
        assert np.allclose(fast, ref, atol=1e-12, rtol=0.0)
    T64 = _random_tpm(rng, 64, power=6.0)
    cand = CE.unit_sum_candidates(6)
    fast = CE.macro_ei_batch(T64, cand['labels'])
    ref = [CE.effective_information(CE.coarse_grain_tpm(T64, lab)) for lab in cand['labels'][::17]]
    assert np.allclose(fast[::17], ref, atol=1e-12, rtol=0.0)
    for bad in ([0, 0], [0, 2, 2], [-1, 0, 1], [0.0, 1.0, 1.0]):
        with pytest.raises(ValueError):
            CE.coarse_grain_tpm(np.eye(3), bad)
    with pytest.raises(ValueError):
        CE.macro_ei_batch(np.eye(3), [[0, 2, 2]])


def test_set_partitions_unit_sum_labels_and_candidates():
    bell = [1, 2, 5, 15, 52, 203, 877, 4140]
    for n, b in enumerate(bell, start=1):
        parts = CE.set_partitions(n)
        assert parts.shape == (b, n) and len({tuple(r) for r in parts.tolist()}) == b
        assert np.all(parts[:, 0] == 0)
        assert all(r[i] <= max(r[:i]) + 1 for r in parts.tolist() for i in range(1, n))   # restricted growth
    assert CE.set_partitions(10).shape == (115975, 10)
    labels, m = CE.unit_sum_labels([0, 0, 1])                              # units 0+1 summed, unit 2 alone
    bits = CE.decode_states(np.arange(8), 3)
    assert m == 6 and labels.tolist() == (bits[:, 0] + bits[:, 1] + 3 * bits[:, 2]).tolist()
    cand = CE.unit_sum_candidates(3)
    assert cand['labels'].shape == (4, 8) and sorted(cand['m'].tolist()) == [4, 6, 6, 6]   # Bell(3) - 1 candidates
    assert CE.unit_sum_candidates(6)['labels'].shape == (202, 64)
    assert CE.state_partition_candidates(4)['labels'].shape == (15 - 2, 4)
    for bad in ([0, 2], [], [1, 1]):
        with pytest.raises(ValueError):
            CE.unit_sum_labels(bad)
    for call in (lambda: CE.set_partitions(11), lambda: CE.unit_sum_candidates(7), lambda: CE.state_partition_candidates(2)):
        with pytest.raises(ValueError):
            call()


def test_weakly_symmetric_tpms_never_show_emergence_but_asymmetric_ones_can():
    """Uniform inputs achieve the capacity of a weakly symmetric channel, and a coarse-graining only re-weights
    inputs and post-processes outputs, so no partition can raise EI (Hoel 2017)."""
    for n in (4, 5, 6, 7):
        for noise in (0.05, 0.3, 0.7):
            for seed in (0, 1):
                T = CE.noisy_permutation_tpm(n, noise, seed=seed)
                assert np.allclose(T.sum(axis=0), 1.0) and np.allclose(T.sum(axis=1), 1.0)
                assert CE.best_state_partition(T)['ce'] <= 1e-12
        circulant = np.array([np.roll([0.6, 0.3, 0.1] + [0.0] * (n - 3), k) for k in range(n)])
        assert CE.best_state_partition(circulant)['ce'] <= 1e-12
    assert CE.best_state_partition(CE.hoel2017_examples()['chain8'])['ce'] > 0.4
    assert np.array_equal(CE.noisy_permutation_tpm(6, 0.2, seed=4), CE.noisy_permutation_tpm(6, 0.2, seed=4))
    with pytest.raises(ValueError):
        CE.noisy_permutation_tpm(6, 1.5)


# ====================================================================== estimation
def test_transition_counts_laplace_smoothing_and_interventional_recovery():
    C = CE.transition_counts([0, 1, 1, 2, 0, 1], 3)
    assert C.tolist() == [[0, 2, 0], [0, 1, 1], [1, 0, 0]] and C.sum() == 5
    est = CE.estimate_tpm(C, alpha=1.0)
    assert np.allclose(est[0], [1 / 5, 3 / 5, 1 / 5]) and np.allclose(est[2], [2 / 4, 1 / 4, 1 / 4])
    raw = CE.estimate_tpm([[0, 0], [3, 1]], alpha=0.0)
    assert np.allclose(raw, [[0.5, 0.5], [0.75, 0.25]])                      # unseen row: uniform (no evidence)
    assert CE.effective_information(CE.estimate_tpm(C, alpha=1.0)) < CE.effective_information(CE.estimate_tpm(C, alpha=0.0))
    assert CE.transition_counts([0, 2, 2], 3, next_states=[1, 1, 0]).tolist() == [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
    T = CE.hoel2017_examples()['chain8']
    rng = np.random.default_rng(5)
    cur = rng.integers(0, 8, 20_000)                             # interventional samples: uniform current states
    nxt = (rng.random(cur.size)[:, None] >= np.cumsum(T, axis=1)[cur]).sum(axis=1)   # successors from the TPM
    counts = CE.transition_counts(cur, 8, next_states=nxt)
    assert counts.sum() == 20_000 and np.all(counts[:7, 7] == 0) and np.all(counts[7, :7] == 0)
    fit = CE.causal_emergence(CE.estimate_tpm(counts), SEVEN_PLUS_ONE)
    exact = CE.causal_emergence(T, SEVEN_PLUS_ONE)
    assert abs(fit['micro']['ei'] - exact['micro']['ei']) < 0.02 and abs(fit['ce'] - exact['ce']) < 0.02
    sparse = CE.causal_emergence(CE.estimate_tpm(CE.transition_counts(cur[:200], 8, next_states=nxt[:200])),
                                 SEVEN_PLUS_ONE)
    assert sparse['micro']['ei'] < fit['micro']['ei']             # Laplace smoothing shrinks EI when counts are few
    for bad in (lambda: CE.transition_counts([0, 3], 3), lambda: CE.transition_counts([0], 3),
                lambda: CE.transition_counts([0.5, 1.0], 3), lambda: CE.estimate_tpm([[-1, 1], [1, 1]]),
                lambda: CE.estimate_tpm(np.ones((2, 3))), lambda: CE.estimate_tpm(np.ones((2, 2)), alpha=-1.0),
                lambda: CE.transition_counts([0, 1], 3, next_states=[1]),
                lambda: CE.transition_counts([0, 1], 3, next_states=[1, 3])):
        with pytest.raises(ValueError):
            bad()


def test_bit_encoding_and_median_binarisation():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, (50, 5))
    states = CE.encode_bits(bits)
    assert states.max() < 32 and np.array_equal(CE.decode_states(states, 5), bits)
    assert CE.encode_bits([[1, 0, 1]]).tolist() == [5]
    X = np.column_stack([np.arange(6.0), [3.0, 1.0, 2.0, 0.0, 5.0, 4.0]])
    assert CE.median_binarize(X).astype(int).tolist() == [[0, 1], [0, 0], [0, 0], [1, 0], [1, 1], [1, 1]]
    for bad in (lambda: CE.encode_bits([[2, 0]]), lambda: CE.encode_bits(np.zeros((2, 31), dtype=int)),
                lambda: CE.decode_states([4], 2), lambda: CE.median_binarize([[np.inf, 1.0]])):
        with pytest.raises(ValueError):
            bad()


# ====================================================================== nulls
def test_successor_null_is_calibrated_on_iid_states_and_separates_chain_from_permutation():
    cand = CE.state_partition_candidates(8)['labels']
    zs, flagged = [], 0
    for r in range(4):
        iid = np.random.default_rng([40, r]).integers(0, 8, 256)
        out = CE.successor_block_null(iid, 8, cand, draws=19, seed=r)
        zs.append(out['z'])
        flagged += out['p'] <= 0.05
        assert out['null_ce'].shape == (19,) and out['transitions'] == 255 and out['block'] == round(255 ** (1 / 3))
    assert abs(float(np.mean(zs))) < 1.5 and flagged <= 1
    chain = CE._sample_chain(CE._noisy_chain(0.1), 384, np.random.default_rng(41))
    hit = CE.successor_block_null(chain, 8, cand, draws=19, seed=0)
    assert hit['ce'] > 0.0 and hit['z'] > 3.0 and hit['p'] == pytest.approx(1 / 20)
    perm = CE._sample_chain(CE.noisy_permutation_tpm(8, 0.3), 384, np.random.default_rng(42))
    ctrl = CE.successor_block_null(perm, 8, cand, draws=19, seed=0)
    assert ctrl['ce'] < 0.0 and ctrl['p'] > 0.05                          # temporal dependence, but no emergence
    again = CE.successor_block_null(chain, 8, cand, draws=19, seed=0)
    assert np.array_equal(again['null_ce'], hit['null_ce'])
    assert not np.array_equal(CE.successor_block_null(chain, 8, cand, draws=19, seed=1)['null_ce'], hit['null_ce'])
    assert not np.array_equal(CE.successor_block_null(chain, 8, cand, draws=19, seed=0, block=1)['null_ce'], hit['null_ce'])
    for bad in (dict(states=[0, 1], n_states=8), dict(states=[0, 9, 1], n_states=8), dict(states=chain, n_states=1),
                dict(states=chain, n_states=8, draws=1), dict(states=chain, n_states=8, block=0)):
        kw = {'draws': 19, **bad}
        with pytest.raises(ValueError):
            CE.successor_block_null(kw.pop('states'), kw.pop('n_states'), cand, **kw)


def test_independent_units_null_separates_coordination_from_slow_independent_units():
    pair_exact = CE.best_unit_coarse_graining(CE.pair_count_copy_tpm(3, 0.1), 6)
    assert pair_exact['groups'] == PAIRS and pair_exact['ce'] > 0.15            # exact: macro beats micro
    noiseless = CE.coarse_grain_tpm(CE.pair_count_copy_tpm(3, 0.0), CE.unit_sum_labels([0, 0, 1, 1, 2, 2])[0])
    assert CE.effective_information(noiseless) == pytest.approx(math.log2(27.0), abs=1e-12)   # counts copy exactly
    keep = 0.9
    ind_exact = CE.best_unit_coarse_graining(CE.independent_flip_tpm(6, keep), 6)
    assert ind_exact['ei_micro'] == pytest.approx(6.0 * (1.0 - CE.entropy_bits([keep, 1 - keep])), abs=1e-12)
    assert ind_exact['ce'] < -0.2                                               # exact: every grouping loses EI
    for T in (CE.pair_count_copy_tpm(2, 0.3), CE.independent_flip_tpm(4, 0.7)):
        assert np.allclose(T.sum(axis=1), 1.0) and np.all(T >= 0.0)
    cand = CE.unit_sum_candidates(6)
    w = 1 << np.arange(6)
    pair = CE._sample_pair_count_copy(3, 0.1, 384, np.random.default_rng(7)) @ w
    res = CE.independent_units_null(pair, 6, cand['labels'], draws=19, seed=0)
    assert res['z'] > 3.0 and res['p'] == pytest.approx(1 / 20) and res['ce'] > 0.0
    assert CE._groups(cand['groups'][res['best_index']], np.arange(6)) == PAIRS
    slow = CE._sample_independent_flips(6, keep, 384, np.random.default_rng(8)) @ w
    ind = CE.independent_units_null(slow, 6, cand['labels'], draws=19, seed=0)
    succ = CE.successor_block_null(slow, 64, cand['labels'], draws=19, seed=0)
    assert ind['p'] > 0.05 and succ['p'] == pytest.approx(1 / 20) and succ['z'] > 3.0   # documented successor bias
    assert np.array_equal(CE.independent_units_null(slow, 6, cand['labels'], draws=19, seed=0)['null_ce'], ind['null_ce'])
    with pytest.raises(ValueError):
        CE.independent_units_null(slow, 12, cand['labels'], draws=19)
    with pytest.raises(ValueError):
        CE.independent_units_null(slow, 6, cand['labels'][:, :32], draws=19)


# ====================================================================== live observer
def _pair_activity(seed, steps, n_units=6, noise=0.1):
    bits = CE._sample_pair_count_copy(3, noise, steps, np.random.default_rng(seed))
    X = 2.0 * bits - 1.0 + 0.05 * np.random.default_rng(seed + 1).standard_normal(bits.shape)
    if n_units > 6:
        X = np.column_stack([X, np.random.default_rng(seed + 2).standard_normal((steps, n_units - 6))])
    return X


def test_observer_is_bounded_read_only_json_safe_and_reports_both_nulls():
    obs = CE.CausalEmergenceObserver(10, seed=3, history=64, units=4, null_draws=9)
    empty = obs.summary()
    assert empty['transitions'] == 0 and empty['ei_micro'] is None and empty['null_ready'] is False
    assert obs.units_observed.tolist() == sorted(obs.units_observed.tolist()) and obs.units_observed.size == 4
    assert np.array_equal(CE.CausalEmergenceObserver(10, seed=3, history=64, units=4).units_observed, obs.units_observed)
    data = np.random.default_rng(0).standard_normal((150, 10))
    for t, row in enumerate(data):
        keep = row.copy()
        obs.observe(row)
        assert np.array_equal(row, keep)
        if t == 20:
            early = obs.summary()
            assert early['transitions'] == 20 and early['ei_micro'] is not None and early['null_ready'] is False
            assert early['null_ce_mean'] is None and early['ce_z_indep'] is None
    s = obs.summary()
    assert s['observed_steps'] == 150 and s['buffered'] == 64 and s['transitions'] == 63 and s['micro_states'] == 16
    assert s['null_ready'] and s['null_block'] == round(63 ** (1 / 3))
    for key in ('ei_micro', 'ei_macro', 'ce', 'null_ce_mean', 'null_ce_sd', 'ce_z', 'indep_ce_mean', 'ce_z_indep'):
        assert isinstance(s[key], float) and math.isfinite(s[key])
    assert 0.1 <= s['ce_p'] * 10 <= 10 and 0.1 <= s['ce_p_indep'] * 10 <= 10          # p in [1/10, 1]
    assert s['ce'] == pytest.approx(s['ei_macro'] - s['ei_micro'])
    assert 0.0 <= s['determinism_micro'] <= 1.0 and 0.0 <= s['degeneracy_micro'] <= 1.0
    assert sorted(u for g in s['macro_partition'] for u in g) == obs.units_observed.tolist()
    text = json.dumps(s)
    assert len(text) < 16_000 and 'wallclock' not in text and s['evidence_tier'] == EVIDENCE_TIER
    assert obs.summary() == s
    for bad in (np.zeros(9), np.full(10, np.nan), np.zeros((10, 1)), ['a'] * 10):
        with pytest.raises(ValueError):
            obs.observe(bad)
    assert obs.summary() == s and obs.observed_steps == 150
    lazy = CE.CausalEmergenceObserver(10, seed=3, history=64, units=4, null_draws=9)   # summary frequency is irrelevant
    for row in data:
        lazy.observe(row)
    assert lazy.summary() == s


def test_observer_finds_coordinated_units_and_its_parameters_change_the_output():
    """With ~16 transitions per micro state (history 1024, 64 states) the count-per-pair grouping of a pair-count
    copy ring is found and clears the independent-units null. With history 256 (~4 per state) Laplace smoothing
    dominates and the best grouping is not stable across seeds (measured 2026-11-20), so that is not asserted."""
    obs = CE.CausalEmergenceObserver(8, seed=1, history=1024, units=6)
    sample = obs.units_observed
    steps = 1074
    X = _pair_activity(5, steps, 8)
    full = np.zeros((steps, 8))
    full[:, sample] = X[:, :6]                                    # the six coordinated units are the observed ones
    full[:, [u for u in range(8) if u not in sample]] = X[:, 6:]
    for row in full:
        obs.observe(row)
    s = obs.summary()
    assert s['transitions'] == 1023 and s['transitions_per_micro_state'] == pytest.approx(1023 / 64)
    assert s['macro_partition'] == [[int(sample[a]), int(sample[b])] for a, b in PAIRS]
    assert s['ce'] > 0.0 and s['ce_z_indep'] > 3.0 and s['ce_p_indep'] == pytest.approx(1 / 20)
    exact = CE.best_unit_coarse_graining(CE.pair_count_copy_tpm(3, 0.1), 6)
    assert s['ei_micro'] < exact['ei_micro'] and s['ce'] < exact['ce']   # smoothed estimates are shrunk
    smooth = CE.CausalEmergenceObserver(8, seed=1, history=1024, units=6, alpha=0.1)
    for row in full:
        smooth.observe(row)
    assert smooth.summary()['ei_micro'] > s['ei_micro']             # less smoothing, less shrinkage
    fewer = CE.CausalEmergenceObserver(8, seed=1, history=256, units=4)
    reseeded = CE.CausalEmergenceObserver(8, seed=2, history=256, units=6)
    for row in full[:80]:
        fewer.observe(row)
        reseeded.observe(row)
    assert fewer.summary()['micro_states'] == 16 and not np.array_equal(reseeded.units_observed, sample)


def test_median_binarisation_keeps_discrete_units_whose_upper_value_is_the_majority():
    rng = np.random.default_rng(11)
    bits = (rng.random((200, 3)) < [0.7, 0.3, 0.5]).astype(float)   # on 70 %, 30 % and 50 % of the time
    bits[:, 2] = 1.0                                                  # ... and one constant unit
    assert np.array_equal(CE.median_binarize(bits)[:, :2], bits[:, :2].astype(bool))
    assert not CE.median_binarize(bits)[:, 2].any()                  # a constant unit is off, not on
    obs = CE.CausalEmergenceObserver(3, seed=0, history=64, units=3, null_draws=3)
    for row in bits:
        obs.observe(row)
    seq = obs._sequence()                                            # every buffered state, oldest first
    assert np.array_equal(CE.decode_states(seq, 3)[:, :2], bits[-64:, :2].astype(np.int8))


def test_observer_round_trip_reset_and_import_validates_before_mutating():
    kw = dict(seed=4, history=32, units=3, null_draws=4, min_transitions=8)
    data = np.random.default_rng(9).standard_normal((70, 5))
    obs, restored = CE.CausalEmergenceObserver(5, **kw), CE.CausalEmergenceObserver(5, **kw)
    for row in data[:20]:
        obs.observe(row)
    young = json.loads(json.dumps(obs.export_state()))
    restored.import_state(young)
    assert restored.summary() == obs.summary() and restored.export_state() == obs.export_state()
    for row in data[20:50]:
        obs.observe(row)
        restored.observe(row)
    assert restored.summary() == obs.summary()
    state = json.loads(json.dumps(obs.export_state()))
    clone = CE.CausalEmergenceObserver(5, **kw)
    clone.import_state(state)
    for row in data[50:]:
        obs.observe(row)
        clone.observe(row)
    assert clone.summary() == obs.summary() and clone.export_state() == obs.export_state()

    before = clone.export_state()
    flipped = copy.deepcopy(state)
    last = (flipped['pos'] - 1) % 32
    flipped['states'][last] ^= 1                                  # no longer the median-binarisation of its raw row
    nan_raw = copy.deepcopy(state)
    nan_raw['raw'][0][0] = float('nan')
    big_state = copy.deepcopy(state)
    big_state['states'][0] = 8
    beyond = copy.deepcopy(young)
    beyond['raw'][25][0] = 1.0                                    # a value beyond the observed steps
    bad_early = copy.deepcopy(young)
    bad_early['states'][3] ^= 1
    for bad in (flipped, nan_raw, big_state, beyond, bad_early, {**state, 'schema': 'Other'},
                {**state, 'module': 'connectomics'}, {**state, 'config': {**state['config'], 'alpha': 2.0}},
                {**state, 'units': [0, 1, 4]}, {**state, 'pos': 3}, {**state, 'steps': 49},
                {**state, 'filled': 31}, {**state, 'steps': 50.0}, {**state, 'raw': state['raw'][:-1]},
                {k: v for k, v in state.items() if k != 'states'}, 'not a state'):
        with pytest.raises(ValueError):
            clone.import_state(bad)
    assert clone.export_state() == before
    obs.reset()
    fresh = CE.CausalEmergenceObserver(5, **kw)
    assert obs.export_state() == fresh.export_state() and obs.summary() == fresh.summary()


def test_observer_from_spec_and_constructor_validation():
    spec = FrontierSpecV1(enabled=True, seed=21, history=128, modules=('causal_emergence',))
    obs = CE.CausalEmergenceObserver.from_spec(40, spec)
    assert (obs.seed, obs.history, obs.units, obs.null_draws) == (21, 128, 6, 19)
    assert CE.CausalEmergenceObserver.from_spec(40, spec, units=3).units == 3
    assert CE.CausalEmergenceObserver(40, fc_window=64, sync_pairs=8).units == 6     # other observers' spec params
    assert CE.CausalEmergenceObserver(3, units=6).summary()['analysed_units'] == 3
    with pytest.raises(ValueError):
        CE.CausalEmergenceObserver.from_spec(40, spec.to_dict())
    for bad in ({'units': 1}, {'units': 7}, {'history': 8}, {'alpha': -1.0}, {'null_draws': 1}, {'block': 0},
                {'min_transitions': 4}, {'min_transitions': 256}, {'seed': -1}, {'surprise': 1}):
        with pytest.raises(ValueError):
            CE.CausalEmergenceObserver(10, **bad)
    with pytest.raises(ValueError):
        CE.CausalEmergenceObserver(1)


# ====================================================================== lab experiment
def test_run_experiment_quick_passes_every_known_answer_check():
    r = CE.run_experiment(quick=True)
    assert verify_receipt(r) and r['kind'] == 'validation' and r['evidence_tier'] == EVIDENCE_TIER
    assert r['verdict'] == 'pass' and r['verdict_basis'] == 'all_known_answer_checks' and all(r['checks'].values())
    assert r['config']['quick'] is True and r['config']['seeds'] == [0, 1, 2] and len(r['checks']) == 22
    v = r['validations']
    assert v['hoel']['chain8']['partitions_evaluated'] == 4138 and v['candidate_partitions'] == 4138
    assert v['units']['summary']['pair_copy_pairs_recovered_fraction'] == 1.0
    assert v['units']['summary']['independent_successor_null_flag_rate_p_le_0.05'] > 0.5   # reported, not checked
    assert all(c['ce'] < 0 for c in v['null_permutation_control'])
    nw = r['neuroweave']['per_mechanism_set']
    assert set(nw) == {'baseline', 'dynamics'} and nw['dynamics']['mechanisms'] == list(CE._MECH_DYNAMICS)
    for runs in nw.values():
        for run in runs['per_seed']:
            assert run['null_ready'] and run['ce_z_indep'] is not None and run['transitions'] == 175
            assert run['transitions_per_micro_state'] == pytest.approx(175 / 16)       # 4 units, 16 micro states
    assert r['neuroweave']['units'] == 4
    base, dyn = (nw[k]['per_seed'] for k in ('baseline', 'dynamics'))
    assert all(b['ei_micro'] != d['ei_micro'] for b, d in zip(base, dyn))      # the mechanism set changes activity
    assert r['measured_wallclock_s']['total'] > 0.0 and 'tpm_evaluations_units' in r['estimated_cost']
    tampered = copy.deepcopy(r)
    tampered['checks']['uniform_ei_is_zero'] = False
    assert not verify_receipt(tampered)


def test_run_experiment_is_deterministic_rng_isolated_and_validates_arguments():
    tiny = dict(seeds=(0,), sizes=(2, 4), estimation_samples=400, null_steps=96, null_draws=19, null_replicates=1,
                unit_steps=96, unit_draws=19, unit_replicates=1, neurons=12, nw_steps=72, nw_burn_in=8,
                nw_null_draws=3, mechanism_sets={'baseline': ()})
    before = _global_rng_state()
    a = CE.run_experiment(quick=True, **tiny)
    b = CE.run_experiment(quick=True, **tiny)
    assert _same_global_state(before, _global_rng_state())
    assert verify_receipt(a) and a['digest'] == b['digest'] and a['verdict'] in ('pass', 'fail')
    assert set(a['checks']) == set(CE.run_experiment.__globals__['_exact_checks'](
        CE._validate_config({**CE._DEFAULTS, **CE._QUICK, **{k: v for k, v in tiny.items() if k != 'seeds'}}))[1]) | {
        'estimated_tpm_recovers_ei_and_ce', 'iid_null_false_positive_rate_at_most_0.25', 'iid_null_mean_z_within_1',
        'noisy_chain_ce_positive_and_above_successor_null',
        'permutation_control_ce_negative_and_not_above_successor_null',
        'unit_pair_copy_above_independent_units_null_with_pairs_recovered', 'unit_independent_units_null_calibrated',
        'neuroweave_values_finite'}
    assert CE.run_experiment(quick=True, **{**tiny, 'seeds': (1,)})['digest'] != a['digest']
    for bad in ({'quick': 1}, {'surprise': 1}, {'seeds': ()}, {'seeds': (1, 1)}, {'null_draws': 5},
                {'mechanism_sets': {'x': ('nope',)}}, {'mechanism_sets': {}}, {'topology': 'nope'},
                {'independent_keep': 0.5}, {'nw_burn_in': 60}, {'units': 7}, {'sizes': (1,)}):
        with pytest.raises(ValueError):
            CE.run_experiment(**{'quick': True, **tiny, **bad})
