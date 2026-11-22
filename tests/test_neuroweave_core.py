"""Core NEUROWEAVE tests: topology, cells, connectome and brain."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pytest

from synapse.neuroweave.brain import (BRAIN_SCHEMA, CONNECTOME_FLAGS, CONSUMED_FLAGS, NeuroWeaveBrain)
from synapse.neuroweave.cells import (NEURON_FLAGS, STATE_FIELDS, CellPopulation, CompartmentDrive, q88)
from synapse.neuroweave.connectome import (EVENT_KINDS, LR_SCALE_BOUNDS, ROUTING_BOUNDS, W_MAX, W_MIN, SynapseTable,
                                           dequantise_int8, quantise_int8)
from synapse.neuroweave.spec import MECHANISM_IDS, REGIONS, TOPOLOGIES, NeuroWeaveSpecV1
from synapse.neuroweave.topology import (assign_regions, build_topology, edge_compartments, region_kind_rank,
                                         topology_metrics)

REGIONAL = ('multi_region', 'dense_multi_region', 'hierarchical_modular', 'multi_scale_regional')
SENS, ASSOC, MEM, MOD, ACT = [0, 1, 2], [3, 4], [5, 6], [7], [8, 9]


def _block(m, a, b):
    return int(np.asarray(m)[np.ix_(a, b)].sum())


# ============================================================================ topology

@pytest.mark.parametrize('name', TOPOLOGIES)
def test_topology_invariants(name):
    t = build_topology(name, 64, seed=5)
    t.validate()
    assert t.edges > 64
    assert not np.any(t.src == t.dst)
    assert np.unique(t.src * t.n + t.dst).size == t.edges
    assert t.delay.min() >= 1 and t.delay.max() <= 3
    assert np.all(np.sign(t.weight) == t.sign[t.src])
    # regions: contiguous blocks, every region populated
    assert np.all(np.diff(t.region) >= 0)
    assert set(t.region.tolist()) == set(range(len(REGIONS)))
    # inhibitory interneuron-like units spread across regions
    for r in range(len(REGIONS)):
        members = t.region == r
        if members.sum() >= 3:
            assert np.any(t.sign[members] < 0), (name, r)
    assert 0.1 < np.mean(t.sign < 0) < 0.3
    # deterministic, seed-dependent
    t2 = build_topology(name, 64, seed=5)
    assert np.array_equal(t.src, t2.src) and np.array_equal(t.weight, t2.weight) and np.array_equal(t.delay, t2.delay)
    t3 = build_topology(name, 64, seed=6)
    assert not (t3.edges == t.edges and np.array_equal(t3.src, t.src) and np.array_equal(t3.dst, t.dst)
                and np.array_equal(t3.weight, t.weight))


def test_region_assignment_proportional():
    for n in (10, 11, 48, 97, 512):
        reg = assign_regions(n, 10)
        counts = np.bincount(reg, minlength=10)
        assert counts.min() >= 1 and counts.max() - counts.min() <= 1
    assert assign_regions(8, 10).max() <= 9


def test_topology_rejects_bad_args():
    with pytest.raises(ValueError):
        build_topology('nope', 32, seed=1)
    with pytest.raises(ValueError):
        build_topology('clustered', 1, seed=1)
    with pytest.raises(ValueError):
        build_topology('clustered', 32, seed=1, inhibitory_fraction=1.5)


def test_degree_shuffled_preserves_degree_sequence():
    for seed in (1, 7, 42):
        base = build_topology('clustered', 96, seed=seed)
        shuf = build_topology('degree_shuffled', 96, seed=seed)
        assert np.array_equal(np.bincount(base.src, minlength=96), np.bincount(shuf.src, minlength=96))
        assert np.array_equal(np.bincount(base.dst, minlength=96), np.bincount(shuf.dst, minlength=96))
        same = np.intersect1d(base.src * 96 + base.dst, shuf.src * 96 + shuf.dst).size
        assert same < 0.8 * base.edges  # swaps actually rewired the graph


@pytest.mark.parametrize('name', REGIONAL + ('hybrid',))
def test_regional_pathways_feedback_and_long_range(name):
    t = build_topology(name, 128, seed=3)
    m = topology_metrics(t)['region_pair_edges']
    assert _block(m, SENS, ASSOC) > 0          # sensory -> association
    assert _block(m, ASSOC, MEM) > 0           # association -> memory
    assert _block(m, MEM, ACT) > 0             # memory -> action
    assert _block(m, ACT, ASSOC) > 0           # action -> association feedback
    assert _block(m, MEM, SENS) > 0            # memory -> sensory predictive feedback
    assert _block(m, SENS, MEM) + _block(m, SENS, ACT) > 0   # long-range shortcuts
    assert np.trace(np.asarray(m)) > 0          # local circuits
    # local recurrent circuits: reciprocal edges inside regions
    keys = set((t.src * t.n + t.dst).tolist())
    recip_local = sum(1 for s, d in zip(t.src.tolist(), t.dst.tolist())
                      if t.region[s] == t.region[d] and d * t.n + s in keys)
    assert recip_local > 0


def test_hub_shapes():
    capped = build_topology('hub_capped', 128, seed=2, fan_in=6)
    assert np.bincount(capped.dst, minlength=128).max() <= 9
    skew = topology_metrics(build_topology('hub_skewed', 128, seed=2))
    unif = topology_metrics(build_topology('uniform_random', 128, seed=2))
    assert skew['in_degree_gini'] > unif['in_degree_gini'] + 0.2
    assert skew['hub_top10_in_share'] > unif['hub_top10_in_share']


def test_topology_metrics_contents_and_contrasts():
    ring = topology_metrics(build_topology('ring_lattice', 64, seed=1))
    er = topology_metrics(build_topology('erdos_renyi', 64, seed=1))
    mr = topology_metrics(build_topology('multi_region', 64, seed=1))
    for m in (ring, er, mr):
        for key in ('density', 'clustering_mean', 'modularity', 'mean_shortest_path', 'global_efficiency',
                    'hub_top10_in_share', 'in_degree_gini', 'region_pair_edges', 'regional_efficiency',
                    'regional_communication_efficiency', 'in_degree', 'out_degree'):
            assert key in m
        assert set(m['in_degree']) == {'histogram', 'mean', 'std', 'max', 'cv'}
        assert sum(m['in_degree']['histogram']) == 64
        assert 0 < m['global_efficiency'] <= 1 and 0 <= m['clustering_mean'] <= 1
        assert np.asarray(m['region_pair_edges']).sum() == m['edges']
        json.dumps(m)
    assert ring['clustering_mean'] > er['clustering_mean']
    assert ring['mean_shortest_path'] > er['mean_shortest_path']
    assert mr['modularity'] > er['modularity']
    # exact path length on a ring lattice: radius 3 both ways -> d(i, j) = ceil(ring distance / 3)
    t = build_topology('ring_lattice', 30, seed=1, fan_in=6)
    dist = [int(np.ceil(min(k, 30 - k) / 3)) for k in range(1, 30)]
    assert topology_metrics(t)['mean_shortest_path'] == pytest.approx(np.mean(dist))


def test_topology_metrics_fast_at_512():
    for name in ('dense_multi_region', 'clustered'):
        t = build_topology(name, 512, seed=4)
        start = time.perf_counter()
        m = topology_metrics(t)
        assert time.perf_counter() - start < 2.0
        assert m['n'] == 512 and np.isfinite(m['mean_shortest_path'])


def test_edge_compartments_classification():
    rank = region_kind_rank(REGIONS)
    region = np.arange(10)
    src = np.array([0, 3, 8, 5, 0, 7])
    dst = np.array([3, 5, 3, 0, 1, 2])
    comp = edge_compartments(src, dst, region, rank)
    # sens->assoc basal(3), assoc->mem basal, action->assoc apical(2), mem->sens apical, sens->sens distal(1), mod distal
    assert comp.tolist() == [3, 3, 2, 2, 1, 1]
    assert edge_compartments(np.array([4]), np.array([4]), np.array([0] * 5), rank).tolist() == [0]


# ============================================================================ cells

def _cells(n=12, seed=0):
    region = assign_regions(n, 10)
    sign = np.where(np.arange(n) % 5 == 0, -1, 1)
    return CellPopulation(n, seed=seed, regions=region, signs=sign)


def _drive(n, rng, scale=0.5):
    return CompartmentDrive(*(rng.normal(size=n) * scale for _ in range(5)))


def test_cells_baseline_is_v01_update():
    c = _cells()
    rng = np.random.default_rng(1)
    fast = np.zeros(12)
    slow = np.zeros(12)
    ctx = np.zeros(12)
    for _ in range(5):
        d = _drive(12, rng)
        inc = d.proximal + d.distal + d.apical + d.basal
        fast = 0.58 * fast + d.external + 0.52 * inc
        slow = 0.94 * slow + 0.14 * d.external + 0.18 * inc
        ctx = 0.69 * ctx + inc
        out = c.step(d, frozenset(), None)
        np.testing.assert_allclose(out, np.tanh(0.84 * fast + 0.16 * inc), atol=1e-12)
    np.testing.assert_allclose(c.fast, fast, atol=1e-12)
    np.testing.assert_allclose(c.slow, slow, atol=1e-12)
    np.testing.assert_allclose(c.context, ctx, atol=1e-12)


def _run_cells(flags, controls=None, steps=15, sparse_k=None):
    c = _cells()
    rng = np.random.default_rng(3)
    for _ in range(steps):
        out = c.step(_drive(12, rng, 0.8), frozenset(flags), controls, sparse_k=sparse_k)
    return c, out


@pytest.mark.parametrize('flag', [f for f in NEURON_FLAGS])
def test_each_neuron_flag_changes_cell_output(flag):
    _, base = _run_cells(())
    c, out = _run_cells((flag,), sparse_k=3)
    assert np.max(np.abs(out - base)) > 1e-6
    for name in STATE_FIELDS:
        assert np.all(np.isfinite(getattr(c, name)))


@pytest.mark.parametrize('controls', [{'gain': 1.5}, {'threshold_shift': 0.2}, {'attention': 0.3}])
def test_controls_change_output(controls):
    _, base = _run_cells(('multi_soma',))
    _, out = _run_cells(('multi_soma',), controls)
    assert np.max(np.abs(out - base)) > 1e-6


def test_plasticity_scale_control_changes_state_and_neutral_controls_are_identity():
    a, _ = _run_cells(())
    b, _ = _run_cells((), {'plasticity_scale': 2.0})
    assert np.max(np.abs(a.trace - b.trace)) > 1e-6
    c, _ = _run_cells(('homeostatic_governor',))
    d, _ = _run_cells(('homeostatic_governor',), {'plasticity_scale': 3.0})
    assert np.max(np.abs(c.output - d.output)) > 1e-6
    e, out_e = _run_cells(('multi_soma',), {'gain': 1.0, 'threshold_shift': 0.0, 'attention': 1.0, 'plasticity_scale': 1.0})
    _, out_f = _run_cells(('multi_soma',))
    assert np.array_equal(out_e, out_f)


def test_membrane_dynamics_spikes_refractory_and_calcium():
    c = _cells()
    d = CompartmentDrive(np.full(12, 1.5), np.zeros(12), np.zeros(12), np.zeros(12), np.full(12, 1.0))
    spikes = []
    for _ in range(12):
        c.step(d, frozenset({'membrane_dynamics'}))
        spikes.append(c.spike.copy())
        assert c.refractory.max() <= 2 and c.calcium.max() <= 4
    spikes = np.array(spikes)
    assert spikes.sum() > 0
    # refractory: a unit never spikes on two consecutive steps
    assert not np.any(spikes[1:] * spikes[:-1])
    assert c.calcium.max() > 0 and c.adaptation.max() > 0
    assert np.all((c.threshold >= 0.42) & (c.threshold <= 0.78))


def test_dendritic_apical_gates_basal_and_axon_myelin():
    c1, c2 = _cells(), _cells()
    basal = np.full(12, 0.5)
    lo = CompartmentDrive(np.zeros(12), np.zeros(12), np.full(12, -1.0), basal, np.zeros(12))
    hi = CompartmentDrive(np.zeros(12), np.zeros(12), np.full(12, 1.0), basal, np.zeros(12))
    for _ in range(10):
        c1.step(lo, {'dendritic_compartments'})
        c2.step(hi, {'dendritic_compartments'})
    assert np.all(c2.output > c1.output)  # positive apical context amplifies basal drive
    c = _cells()
    m0 = c.myelin.copy()
    d = CompartmentDrive(np.full(12, 1.0), np.zeros(12), np.zeros(12), np.zeros(12), np.full(12, 1.0))
    for _ in range(50):
        c.step(d, {'axonal_branches'})
    assert np.all(c.myelin > m0) and c.myelin.max() <= 0.98
    assert np.abs(c.release).max() > 0 and np.abs(c.branch_a).max() > 0


def test_extended_somas_primary_secondary_burst_gating():
    c, _ = _run_cells(('extended_somas',), steps=30)
    assert np.abs(c.primary).max() > 0 and np.abs(c.secondary).max() > 0
    assert c.burst.max() > 0
    assert np.all((c.gating >= 0.65) & (c.gating <= 1.35)) and np.any(c.gating != 1.0)


def test_q88_and_sparse():
    x = np.array([0.1234, -0.5, 3.00001])
    assert np.allclose(q88(x) * 256, np.round(q88(x) * 256))
    c, out = _run_cells(('q88_state',))
    assert np.allclose(out * 256, np.round(out * 256))
    c, out = _run_cells(('sparse_code',), sparse_k=3)
    big = np.sort(np.abs(out))[::-1]
    assert big[3] < big[2]


def test_cells_grow_reset_and_state_roundtrip():
    c, _ = _run_cells(('multi_soma', 'membrane_dynamics'))
    idx = c.grow(3, 4, signs=[1, -1, 1])
    assert idx.tolist() == [12, 13, 14] and c.n == 15
    assert c.region[-3:].tolist() == [4, 4, 4] and c.sign[-2] == -1
    for name in STATE_FIELDS:
        assert getattr(c, name).shape == (15,)
    c.reset_units([0, 1])
    assert c.fast[0] == 0 and c.gating[1] == 1.0 and c.threshold[0] == pytest.approx(0.53)
    state = c.export_state()
    json.dumps(state)
    d = _cells()
    d.import_state(state)
    assert d.n == 15 and np.array_equal(d.output, c.output)
    bad = copy.deepcopy(state)
    bad['fast'][0] = float('nan')
    before = d.export_state()
    with pytest.raises(ValueError):
        d.import_state(bad)
    assert d.export_state() == before
    bad = copy.deepcopy(state)
    bad['output'] = bad['output'][:-1]
    with pytest.raises(ValueError):
        d.import_state(bad)
    bad = copy.deepcopy(state)
    bad['output'][0] = 5.0
    with pytest.raises(ValueError):
        d.import_state(bad)


# ============================================================================ connectome

def _table(name='multi_region', n=64, seed=2):
    return SynapseTable.from_topology(build_topology(name, n, seed=seed), seed)


def test_tiled_traversal_matches_untiled():
    t = _table()
    out = np.random.default_rng(0).uniform(-1, 1, t.n)
    a = t.transmit(out, {'synaptic_delays'})
    for tile in (1, 7, 16, 1000):
        b = t.transmit(out, {'synaptic_delays', 'tiled_execution'}, tile_size=tile)
        for comp in ('proximal', 'distal', 'apical', 'basal'):
            np.testing.assert_allclose(getattr(a, comp), getattr(b, comp), atol=1e-12, rtol=0)
        assert t.profile['tiles_last'] == int(np.ceil(t.size / tile))
    total = np.bincount(t.dst, weights=out[t.src] * t.w, minlength=t.n)
    np.testing.assert_allclose(a.total, total, atol=1e-12)


def test_delay_ring_buffer():
    t = _table()
    rng = np.random.default_rng(1)
    outs = [rng.uniform(-1, 1, t.n) for _ in range(4)]
    # before any history every delay behaves as immediate
    np.testing.assert_allclose(t.transmit(outs[0], {'synaptic_delays'}).total, t.transmit(outs[0], set()).total, atol=1e-12)
    for o in outs:
        t.push(o)
    cur = outs[-1]
    drive = t.transmit(cur, {'synaptic_delays'})
    pre = np.where(t.delay == 1, cur[t.src], np.where(t.delay == 2, outs[-2][t.src], outs[-3][t.src]))
    np.testing.assert_allclose(drive.total, np.bincount(t.dst, weights=pre * t.w, minlength=t.n), atol=1e-12)
    no_delay = t.transmit(cur, set())
    assert np.max(np.abs(no_delay.total - drive.total)) > 1e-6
    # adaptive conduction: myelinated synapses shorten delays (never below 1)
    t.myelin[:] = 0.9
    d = t.effective_delay({'synaptic_delays', 'adaptive_conduction'})
    assert np.array_equal(d, np.maximum(1, t.delay - 1))


def test_quantised_int8():
    w = np.array([0.5, -0.25, 1e-6, -2.0, 0.0])
    q, scale = quantise_int8(w)
    assert q.dtype == np.int8 and np.abs(q).max() <= 127 and scale == 2.0
    assert np.all(np.sign(q[w != 0]) == np.sign(w[w != 0]))
    deq = dequantise_int8(q, scale)
    assert np.max(np.abs(deq[:2] - w[:2])) <= scale / 127
    t = _table()
    out = np.random.default_rng(3).uniform(-1, 1, t.n)
    a = t.transmit(out, set()).total
    b = t.transmit(out, {'quantised_synapses'}).total
    assert 0 < np.max(np.abs(a - b)) < 0.2


def test_routing_uncertainty_and_region_gain():
    t = _table()
    c = CellPopulation(t.n, seed=0, regions=t.region, signs=t.sign)
    c.gating[:] = np.linspace(0.65, 1.35, t.n)
    c.apical[:] = np.linspace(-3, 3, t.n)
    out = np.random.default_rng(4).uniform(-1, 1, t.n)
    base = t.transmit(out, set()).total
    routed = t.transmit(out, {'adaptive_routing'}, cells=c).total
    assert np.all((t.routing_gain >= ROUTING_BOUNDS[0]) & (t.routing_gain <= ROUTING_BOUNDS[1]))
    assert np.max(np.abs(routed - base)) > 1e-6
    gated = t.transmit(out, {'uncertainty_gating'}, uncertainty_gate=0.1).total
    np.testing.assert_allclose(gated, 0.58 * base, atol=1e-12)
    ident = t.transmit(out, {'multi_region_routing'}, route_gain=np.ones((10, 10))).total
    np.testing.assert_allclose(ident, base, atol=1e-12)
    rg = np.ones((10, 10))
    rg[:3, 3:5] = 1.8
    changed = t.transmit(out, {'multi_region_routing'}, route_gain=rg).total
    assert np.max(np.abs(changed - base)) > 1e-6
    with pytest.raises(ValueError):
        t.transmit(out, {'multi_region_routing'}, route_gain=np.ones((3, 3)))


def test_reward_update_bounded_sign_preserving_and_meta():
    t = _table()
    sign0 = np.sign(t.w).copy()
    rng = np.random.default_rng(5)
    for i in range(200):
        out = rng.uniform(-1, 1, t.n)
        t.transmit(out, set())
        t.update_traces(rng.uniform(-1, 1, t.n))
        t.reward_update(1.0 if i % 3 else -1.0, 0.5, {'meta_plasticity'})
    assert np.array_equal(np.sign(t.w), sign0)
    assert np.abs(t.w).max() <= W_MAX and np.abs(t.w).min() >= W_MIN
    assert np.all((t.lr_scale >= LR_SCALE_BOUNDS[0]) & (t.lr_scale <= LR_SCALE_BOUNDS[1]))
    assert np.any(t.lr_scale != 1.0) and t.updates.max() > 0
    assert np.abs(t.trace).max() <= 1.5 and np.abs(t.eligibility).max() <= 1.5
    assert t.reward_update(0.0, 0.5, set()) == 0


def test_stdp_traces_bounded_and_timing_rule():
    t = _table('clustered', 32)
    n = t.n
    e = 0
    a, b = int(t.src[e]), int(t.dst[e])
    pre_spike = np.zeros(n)
    pre_spike[a] = 1.0
    post_spike = np.zeros(n)
    post_spike[b] = 1.0
    w0 = abs(t.w[e])
    t.stdp_update(pre_spike, set())       # pre first
    t.stdp_update(post_spike, set())      # then post -> potentiation
    assert abs(t.w[e]) > w0
    t2 = _table('clustered', 32)
    t2.stdp_update(post_spike, set())     # post first
    t2.stdp_update(pre_spike, set())      # then pre -> depression
    assert abs(t2.w[e]) < w0
    for _ in range(50):
        t.stdp_update(np.ones(n), set())
    assert t.pre_spike.max() <= 1.5 and t.post_spike.max() <= 1.5
    assert np.all(np.sign(t.w) == t.sign[t.src])


def test_structural_step_cap_events_and_lesion():
    t = _table()
    rng = np.random.default_rng(6)
    for _ in range(30):
        t.transmit(rng.uniform(-1, 1, t.n), set())
        t.update_traces(rng.uniform(-1, 1, t.n))
    before = t.size
    events = t.structural_step(10, np.random.default_rng(1), max_synapses=before + 100)
    kinds = {e['kind'] for e in events}
    assert kinds == {'prune', 'grow'}
    assert all(set(e) == {'step', 'kind', 'src', 'dst', 'reason', 'utility'} for e in events)
    assert {e['reason'] for e in events} == {'low_utility', 'high_coactivity'}
    assert np.unique(t.src * t.n + t.dst).size == t.size and not np.any(t.src == t.dst)
    assert np.all(np.sign(t.w) == t.sign[t.src])
    cap = before - 50
    events = t.structural_step(20, np.random.default_rng(2), max_synapses=cap)
    assert t.size <= cap and any(e['reason'] == 'capacity' for e in events)
    assert t.ledger.counts_by_kind['prune'] >= 50
    # lesion returns a copy
    mask = np.zeros(t.size, dtype=bool)
    mask[:10] = True
    size = t.size
    w = t.w.copy()
    les = t.lesion(mask)
    assert les.size == size - 10 and t.size == size and np.array_equal(t.w, w)


def test_ledger_is_bounded():
    t = _table()
    for i in range(700):
        t.ledger.record(step=i, kind='grow', src=0, dst=1, reason='test')
    assert len(t.ledger.events) == 512 and t.ledger.counts_by_kind['grow'] == 700
    assert t.ledger.counts_by_reason['test'] == 700
    with pytest.raises(ValueError):
        t.ledger.record(step=0, kind='explode', src=0, dst=1, reason='x')


def test_downscale_consolidate_and_table_roundtrip():
    t = _table()
    m = np.abs(t.w).copy()
    t.downscale(0.5)
    np.testing.assert_allclose(np.abs(t.w), np.clip(m * 0.5, W_MIN, W_MAX))
    assert np.all(np.sign(t.w) == t.sign[t.src])
    with pytest.raises(ValueError):
        t.downscale(1.5)
    t.trace[:] = 1.0
    t.consolidate()
    assert np.allclose(t.trace, 0.5)
    t.push(np.zeros(t.n))
    state = t.export_state()
    json.dumps(state)
    u = SynapseTable.from_state(state)
    assert np.array_equal(u.w, t.w) and u.hist_count == 1
    bad = copy.deepcopy(state)
    bad['w'][0] = -bad['w'][0]  # Dale violation
    with pytest.raises(ValueError):
        SynapseTable.from_state(bad)
    bad = copy.deepcopy(state)
    bad['dst'][0] = bad['src'][0]
    with pytest.raises(ValueError):
        SynapseTable.from_state(bad)


# ============================================================================ brain

SPEC = NeuroWeaveSpecV1(seed=3, structural_interval=5, sleep_interval=10)


def _run_brain(mechs, steps=20, spec=SPEC, uncertainty=0.5):
    b = NeuroWeaveBrain(spec, mechanisms=mechs)
    rng = np.random.default_rng(0)
    out = None
    for t in range(steps):
        out = b.step(rng.normal(size=8), reward=float(np.sin(t)), uncertainty=uncertainty)
    return b, out


def test_all_off_identity_and_baseline_equivalence():
    a, oa = _run_brain(())
    b, ob = _run_brain(())
    np.testing.assert_allclose(oa, ob, atol=1e-12, rtol=0)
    c = NeuroWeaveBrain(SPEC)  # default spec has no mechanisms
    rng = np.random.default_rng(0)
    for t in range(20):
        oc = c.step(rng.normal(size=8), reward=float(np.sin(t)), uncertainty=0.5)
    np.testing.assert_allclose(oa, oc, atol=1e-12, rtol=0)
    # the all-off brain is the v0.1 update driven by the summed recurrent input
    d = NeuroWeaveBrain(SPEC, mechanisms=())
    syn = d.synapses
    fast = np.zeros(d.n)
    prev = np.zeros(d.n)
    for t in range(5):
        x = rng.normal(size=8)
        ext = d.project_input(x)
        inc = np.bincount(syn.dst, weights=prev[syn.src] * syn.w, minlength=d.n)
        fast = 0.58 * fast + ext + 0.52 * inc
        prev = d.step(x, reward=0.5)
        np.testing.assert_allclose(prev, np.tanh(0.84 * fast + 0.16 * inc), atol=1e-12)


_SINGLE = [f for f in CONSUMED_FLAGS if f not in ('tiled_execution', 'meta_plasticity')]


@pytest.mark.parametrize('flag', _SINGLE)
def test_each_consumed_mechanism_changes_output(flag):
    _, base = _run_brain(())
    b, out = _run_brain((flag,))
    assert out.shape != base.shape or np.max(np.abs(out - base)) > 1e-6, flag
    assert np.all(np.isfinite(out))


def test_tiled_transmission_preserves_per_tile_rounding_across_chunks():
    brain = NeuroWeaveBrain(NeuroWeaveSpecV1(neurons=1024, max_neurons=1024, fan_in=24, max_synapses=32768))
    table = brain.synapses
    assert table.size > 8192
    activity = np.random.default_rng(42).uniform(-1, 1, table.n)
    tile = 7
    drive = table.transmit(activity, {'tiled_execution'}, tile_size=tile)
    expected = np.zeros((4, table.n))
    bins = table.compartment * table.n + table.dst
    for start in range(0, table.size, tile):
        sl = slice(start, start + tile)
        expected += np.bincount(bins[sl], weights=table.last_signal[sl],
                                minlength=expected.size).reshape(expected.shape)
    np.testing.assert_array_equal([drive.proximal, drive.distal, drive.apical, drive.basal], expected)
    assert table.profile['tiles_last'] == (table.size + tile - 1) // tile


def test_meta_plasticity_and_tiled_execution():
    _, a = _run_brain(('reward_plasticity',))
    _, b = _run_brain(('reward_plasticity', 'meta_plasticity'))
    assert np.max(np.abs(a - b)) > 1e-9
    _, s1 = _run_brain(('stdp',))
    _, s2 = _run_brain(('stdp', 'meta_plasticity'))
    assert np.max(np.abs(s1 - s2)) > 1e-9
    base_b, base = _run_brain(())
    tb, tiled = _run_brain(('tiled_execution',))
    np.testing.assert_allclose(tiled, base, atol=1e-12, rtol=0)
    assert tb.profile()['measured']['tiles_last'] > base_b.profile()['measured']['tiles_last']


def test_route_gain_identity_and_learning():
    a = NeuroWeaveBrain(SPEC, mechanisms=())
    b = NeuroWeaveBrain(SPEC, mechanisms=('multi_region_routing',))
    rng = np.random.default_rng(1)
    for _ in range(10):
        x = rng.normal(size=8)
        np.testing.assert_allclose(a.frozen_step(x), b.frozen_step(x, route_gain=np.ones((10, 10))), atol=1e-12)
    for t in range(10):
        b.step(rng.normal(size=8), reward=1.0)
    assert np.any(b.synapses.route_gain != 1.0)
    assert b.synapses.route_gain.min() >= 0.5 and b.synapses.route_gain.max() <= 1.5


def test_dale_and_finite_after_200_steps_all_mechanisms():
    spec = NeuroWeaveSpecV1(seed=9, neurons=64, max_neurons=80, max_synapses=900, structural_interval=10,
                            sleep_interval=40, topology='multi_region')
    b = NeuroWeaveBrain(spec, mechanisms=MECHANISM_IDS)
    rng = np.random.default_rng(2)
    for t in range(200):
        out = b.step(rng.normal(size=12), reward=float(np.clip(rng.normal(), -1, 1)), uncertainty=abs(rng.normal()) * 0.3,
                     controls={'gain': 1.1, 'attention': 0.9, 'plasticity_scale': 1.2})
        assert np.all(np.isfinite(out)) and np.abs(out).max() <= 1.0
        if t % 25 == 0:
            b.train_readout(float(t % 2))
    syn = b.synapses
    assert np.all(np.sign(syn.w) == syn.sign[syn.src])
    assert np.all(syn.sign == b.cells.sign)
    assert np.abs(syn.w).max() <= W_MAX and np.abs(syn.w).min() >= W_MIN
    assert syn.pre_spike.max() <= 1.5 and syn.post_spike.max() <= 1.5
    for name in STATE_FIELDS:
        assert np.all(np.isfinite(getattr(b.cells, name)))
    assert b.n <= spec.max_neurons
    assert syn.size <= max(spec.max_synapses, 0) or b.counters['structural_steps'] == 0
    assert b.counters['sleep_cycles'] == 5
    assert b.ledger.total > 0 and len(b.ledger.events) <= 512
    assert set(b.ledger.counts_by_kind) == set(EVENT_KINDS)


def test_growth_caps_and_ledger():
    spec = NeuroWeaveSpecV1(seed=1, neurons=40, max_neurons=44, max_synapses=400, topology='multi_region')
    b = NeuroWeaveBrain(spec, mechanisms=('neurogenesis',))
    ids = b.grow_neurons(3, 'working_memory', reason='test_growth')
    assert ids == [40, 41, 42] and b.n == 43
    assert b.cells.region[40:].tolist() == [5, 5, 5]
    assert np.all(b.synapses.dst[np.isin(b.synapses.dst, ids)] >= 0)
    assert np.isin(ids, b.synapses.dst).all() and np.isin(ids, b.synapses.src).all()
    assert b.grow_neurons(5, 0) == [44 - 1] and b.n == 44
    assert b.grow_neurons(2, 0) == [] and b.n == 44
    assert b.ledger.counts_by_kind['neuron_birth'] == 4
    assert b.ledger.counts_by_reason['test_growth'] == 3
    assert b.synapses.size <= spec.max_synapses
    out = b.step(np.ones(8))
    assert out.shape == (44,) and np.all(np.isfinite(out))
    with pytest.raises(ValueError):
        b.grow_neurons(1, 'nowhere')
    # repurposing keeps the population size and is ledgered
    b.repurpose_neuron(3)
    assert b.n == 44 and b.ledger.counts_by_kind['repurpose'] == 1
    assert b.ledger.counts_by_kind['rewire'] >= 1


def test_structural_cap_respected_by_brain():
    spec = NeuroWeaveSpecV1(seed=2, neurons=48, max_synapses=250, structural_interval=2)
    b = NeuroWeaveBrain(spec, mechanisms=('structural_plasticity',))
    assert b.synapses.size > 250
    rng = np.random.default_rng(0)
    for _ in range(6):
        b.step(rng.normal(size=8), reward=0.5)
    assert b.synapses.size <= 250
    assert b.ledger.counts_by_reason.get('capacity', 0) > 0


def test_export_import_roundtrip_identical_next_step():
    spec = NeuroWeaveSpecV1(seed=4, neurons=48, structural_interval=5, sleep_interval=7, max_neurons=60)
    b = NeuroWeaveBrain(spec, mechanisms=MECHANISM_IDS)
    rng = np.random.default_rng(3)
    for _ in range(23):
        b.step(rng.normal(size=8), reward=0.4, uncertainty=0.2)
    state = b.export_state()
    assert set(state) >= {'schema', 'spec', 'topology', 'cells', 'synapses', 'readout', 'step', 'ledger'}
    assert state['schema'] == BRAIN_SCHEMA and state['step'] == 23
    text = json.dumps(state)
    c = NeuroWeaveBrain(spec)
    c.import_state(json.loads(text))
    d = b.clone()
    for _ in range(12):  # crosses structural, growth and sleep boundaries
        x = rng.normal(size=8)
        o1 = b.step(x, reward=-0.3, uncertainty=0.1)
        o2 = c.step(x, reward=-0.3, uncertainty=0.1)
        o3 = d.step(x, reward=-0.3, uncertainty=0.1)
        assert np.array_equal(o1, o2) and np.array_equal(o1, o3)
    assert b.export_state()['synapses'] == c.export_state()['synapses']


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(schema='bogus'),
    lambda s: s['cells']['fast'].__setitem__(0, float('inf')),
    lambda s: s['synapses']['w'].__setitem__(0, -s['synapses']['w'][0]),
    lambda s: s['readout'].update(w=[0.0]),
    lambda s: s['topology'].update(n=999),
    lambda s: s.update(step=-1),
    lambda s: s.update(rng={'bit_generator': 'nope'}),
    lambda s: s['ledger'].update(events='x'),
    lambda s: s.pop('synapses'),
    # adversarial-review additions: wrong container types and out-of-bound learned state
    lambda s: s.update(synapses=5),
    lambda s: s.update(cells=[1, 2]),
    lambda s: s['topology'].update(region=5),
    lambda s: s['topology'].update(name='ring_lattice'),
    lambda s: s['synapses'].update(coact=[5.0] * len(s['synapses']['coact'])),
    lambda s: s['synapses'].update(trace=[99.0] * len(s['synapses']['trace'])),
    lambda s: s['synapses'].update(utility=[-1.0] * len(s['synapses']['utility'])),
    lambda s: s['synapses'].update(myelin=[3.0] * len(s['synapses']['myelin'])),
    lambda s: s['synapses'].update(routing_gain=[9.0] * len(s['synapses']['routing_gain'])),
    lambda s: s['synapses'].update(updates=[-1] * len(s['synapses']['updates'])),
    lambda s: s['readout'].update(b='x'),
    lambda s: s['ledger'].update(counters=[1, 2]),
])
def test_import_rejects_bad_state_without_mutation(mutate):
    b, _ = _run_brain(('multi_soma', 'reward_plasticity'), steps=5)
    target, _ = _run_brain(('multi_soma',), steps=3)
    before = json.dumps(target.export_state(), sort_keys=True)
    state = copy.deepcopy(b.export_state())
    mutate(state)
    with pytest.raises(ValueError):
        target.import_state(state)
    assert json.dumps(target.export_state(), sort_keys=True) == before


def test_frozen_step_does_not_learn_and_readout_trains():
    b = NeuroWeaveBrain(SPEC, mechanisms=('reward_plasticity', 'stdp', 'membrane_dynamics', 'multi_region_routing'))
    rng = np.random.default_rng(5)
    for _ in range(5):
        b.step(rng.normal(size=8), reward=1.0)
    w = b.synapses.w.copy()
    rg = b.synapses.route_gain.copy()
    clone = b.clone()
    for _ in range(10):
        x = rng.normal(size=8)
        o1 = b.frozen_step(x, reward=1.0)
        o2 = clone.step(x, reward=1.0, learn=False)
        assert np.array_equal(o1, o2)
    assert np.array_equal(b.synapses.w, w) and np.array_equal(b.synapses.route_gain, rg)
    # logistic readout separates the brain states evoked by two different inputs
    r = NeuroWeaveBrain(SPEC, mechanisms=('multi_soma',))
    feats = {}
    for y in (0.0, 1.0):
        probe = r.clone()
        for _ in range(10):
            probe.step(np.full(8, 1.0 if y else -1.0))
        feats[y] = probe.readout_features()
    losses = []
    for t in range(400):
        y = float(t % 2)
        p = r.train_readout(y, feats[y])
        losses.append(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    assert np.mean(losses[-50:]) < np.mean(losses[:50]) and np.mean(losses[-50:]) < 0.5
    assert r.predict(feats[1.0]) > 0.5 > r.predict(feats[0.0])
    assert r.readout_features().shape == (r.feature_dim,)
    with pytest.raises(ValueError):
        r.train_readout(2.0)


def test_input_projection_any_length_and_flyhash():
    b = NeuroWeaveBrain(SPEC, mechanisms=('flyhash_projection',))
    for d in (1, 3, 8, 155, 1000):
        out = b.step(np.linspace(-1, 1, d))
        assert out.shape == (b.n,) and np.all(np.isfinite(out))
    code = b.flyhash_code(np.arange(8.0))
    assert code.sum() == 4 and set(code.tolist()) <= {0.0, 1.0}
    assert np.array_equal(code, b.flyhash_code(np.arange(8.0)))
    with pytest.raises(ValueError):
        b.step(np.array([np.nan]))
    assert b.step(None).shape == (b.n,)


def test_context_attention_matches_definition():
    b = NeuroWeaveBrain(SPEC, mechanisms=('context_attention',))
    prev = np.random.default_rng(7).uniform(-1, 1, b.n)
    att = b.context_attention(prev)
    syn = b.synapses
    i = int(np.bincount(syn.dst, minlength=b.n).argmax())
    inc = np.flatnonzero(syn.dst == i)
    score = np.abs(syn.w[inc] * prev[syn.src[inc]])
    top = inc[np.argsort(-score, kind='stable')[:8]]
    q = 2.2 * np.abs(syn.w[top] * prev[syn.src[top]])
    ex = np.exp(q - q.max())
    assert att[i] == pytest.approx(float((ex * prev[syn.src[top]]).sum() / ex.sum()), abs=1e-12)


def test_profile_measured_vs_estimated():
    b, _ = _run_brain(('tiled_execution', 'sparse_code'), steps=3)
    p = b.profile()
    assert set(p) == {'measured', 'estimated'}
    assert p['measured']['edge_traversals_total'] == 3 * b.synapses.size
    assert p['measured']['wallclock_ms_total'] > 0
    assert p['estimated']['ops_last'] > 0 and p['estimated']['bytes'] == b.synapses.size * 40 + b.n * 224
    assert 'wallclock_ms_last' not in json.dumps(b.export_state())


def test_dense_multi_region_512_runs():
    spec = NeuroWeaveSpecV1(seed=1, neurons=512, topology='dense_multi_region', max_synapses=30000)
    b = NeuroWeaveBrain(spec, mechanisms=MECHANISM_IDS)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(5):
        out = b.step(rng.normal(size=16), reward=0.2)
    assert out.shape == (512,) and np.all(np.isfinite(out))
    assert time.perf_counter() - start < 10.0


def test_consumed_flags_are_known_mechanisms():
    assert set(CONSUMED_FLAGS) <= set(MECHANISM_IDS)
    assert set(CONNECTOME_FLAGS) <= set(CONSUMED_FLAGS)


# ============================================================================ adversarial review regressions

def test_frozen_step_does_not_change_synaptic_myelin():
    b = NeuroWeaveBrain(SPEC, mechanisms=('synaptic_delays', 'adaptive_conduction'))
    rng = np.random.default_rng(11)
    m0 = b.synapses.myelin.copy()
    for _ in range(5):
        b.frozen_step(rng.normal(size=8))
    assert np.array_equal(b.synapses.myelin, m0)
    for _ in range(5):
        b.step(rng.normal(size=8))
    assert np.max(np.abs(b.synapses.myelin - m0)) > 0
    assert b.synapses.myelin.min() >= 0.2 and b.synapses.myelin.max() <= 0.98
    # plasticity_scale=0 freezes the myelin proxy as well
    c = NeuroWeaveBrain(SPEC, mechanisms=('adaptive_conduction',))
    m1 = c.synapses.myelin.copy()
    c.step(np.ones(8), controls={'plasticity_scale': 0.0})
    assert np.array_equal(c.synapses.myelin, m1)


@pytest.mark.parametrize('controls', [{'gain': float('nan')}, {'attention': 'abc'}, {'threshold_shift': [1, 2]},
                                      {'learning_rate_scale': float('inf')}, {'gain': True}, 'gain'])
def test_invalid_controls_rejected_before_mutation(controls):
    b, _ = _run_brain(('adaptive_conduction', 'reward_plasticity', 'synaptic_delays'), steps=3)
    before = json.dumps(b.export_state(), sort_keys=True)
    with pytest.raises(ValueError):
        b.step(np.ones(8), reward=1.0, controls=controls)
    assert json.dumps(b.export_state(), sort_keys=True) == before


def test_cell_myelin_sets_axonal_release_efficacy():
    lo, hi = _cells(), _cells()
    lo.myelin[:] = 0.2
    hi.myelin[:] = 0.95
    d = CompartmentDrive(np.full(12, 0.6), np.zeros(12), np.zeros(12), np.zeros(12), np.full(12, 0.4))
    for _ in range(6):
        lo.step(d, {'axonal_branches'})
        hi.step(d, {'axonal_branches'})
    assert np.all(np.abs(hi.release) > np.abs(lo.release))
    assert np.max(np.abs(hi.output - lo.output)) > 1e-4
    # without axonal_branches the myelin proxy neither changes nor matters
    a, b = _cells(), _cells()
    b.myelin[:] = 0.95
    m = a.myelin.copy()
    for _ in range(6):
        a.step(d, {'adaptive_conduction'})
        b.step(d, {'adaptive_conduction'})
    assert np.array_equal(a.output, b.output) and np.array_equal(a.myelin, m)


def test_consolidation_protects_synapses_from_sleep_downscaling():
    t = _table()
    w0 = np.abs(t.w).copy()
    protect = np.zeros(t.size)
    protect[: t.size // 2] = 1.0
    t.downscale(0.5, protect=protect)
    np.testing.assert_allclose(np.abs(t.w[: t.size // 2]), w0[: t.size // 2])
    np.testing.assert_allclose(np.abs(t.w[t.size // 2:]), np.clip(w0[t.size // 2:] * 0.5, W_MIN, W_MAX))
    assert np.all(np.sign(t.w) == t.sign[t.src])
    with pytest.raises(ValueError):
        t.downscale(0.5, protect=np.zeros(3))
    # brain-level: consolidated destination units keep more of their incoming weight
    b = NeuroWeaveBrain(SPEC, mechanisms=('sleep_consolidation',))
    c = b.clone()
    c.cells.consolidation[: b.n // 2] = 2.0
    wb, wc = np.abs(b.synapses.w).copy(), np.abs(c.synapses.w).copy()
    b.sleep()
    c.sleep()
    shielded = c.synapses.dst < b.n // 2
    assert np.all(np.abs(c.synapses.w[shielded]) >= np.abs(b.synapses.w[shielded]))
    assert np.max(np.abs(c.synapses.w[shielded]) - np.abs(b.synapses.w[shielded])) > 1e-6
    np.testing.assert_allclose(np.abs(c.synapses.w[~shielded]), np.abs(b.synapses.w[~shielded]))
    np.testing.assert_allclose(wb, wc)


def test_stdp_learns_with_membrane_spikes():
    b = NeuroWeaveBrain(SPEC, mechanisms=('membrane_dynamics', 'stdp'))
    w0 = b.synapses.w.copy()
    rng = np.random.default_rng(0)
    spikes = 0.0
    for _ in range(100):
        b.step(rng.normal(size=8), reward=0.3)
        spikes += b.cells.spike.sum()
    assert spikes > 0 and b.synapses.updates.sum() > 0
    assert np.max(np.abs(b.synapses.w - w0)) > 0
    assert np.all(np.sign(b.synapses.w) == b.synapses.sign[b.synapses.src])
