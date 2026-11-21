"""Cross-cutting contract audit tests enforcing docs/frontier/CONTRACT.md across synapse/frontier/."""
from __future__ import annotations

import ast
import json

import numpy as np
import pytest
import torch

from synapse.config import ROOT, SynapseConfig
from synapse.frontier.harness import _jsonable, verify_receipt
from synapse.frontier.runtime import (
    MODULE_FILES,
    OBSERVER_CLASSES,
    discover_experiments,
    load_frontier_module,
    make_observer,
    run_frontier_experiment,
)
from synapse.frontier.spec import EVIDENCE_TIER, OBSERVER_MODULE_IDS, FrontierSpecV1
from synapse.frontier.workbench_api import frontier_experiment_row

FRONTIER_DIR = ROOT / 'synapse' / 'frontier'


def test_frontier_modules_never_seed_global_rngs():
    """No module in synapse/frontier/ may call global np.random.seed, random.seed, or torch.manual_seed."""
    forbidden = {'np.random.seed', 'numpy.random.seed', 'random.seed', 'torch.manual_seed'}
    offenders: list[tuple[str, int, str]] = []
    for path in sorted(FRONTIER_DIR.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called = ast.unparse(node.func)
                if called in forbidden:
                    offenders.append((path.name, node.lineno, called))
    assert offenders == [], f'Forbidden global RNG seed calls found: {offenders}'


def test_torch_set_num_threads_always_paired_with_restore_in_finally():
    """Every torch.set_num_threads(1) call in synapse/frontier/ must restore inside a try...finally block."""
    for path in sorted(FRONTIER_DIR.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        set_calls = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'set_num_threads'
        ]
        if not set_calls:
            continue
        finally_restores = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and node.finalbody:
                for sub in node.finalbody:
                    for inner in ast.walk(sub):
                        if (isinstance(inner, ast.Call)
                                and isinstance(inner.func, ast.Attribute)
                                and inner.func.attr == 'set_num_threads'):
                            finally_restores += 1
        assert finally_restores >= 1, f'{path.name} calls torch.set_num_threads without a finally restore'


def test_present_frontier_modules_obey_export_and_experiment_naming_contract():
    """Every present frontier module in MODULE_FILES obeys MODULE_ID + EXPERIMENTS naming rules."""
    checked = 0
    for mod_id in MODULE_FILES:
        try:
            mod = load_frontier_module(mod_id)
        except ModuleNotFoundError:
            continue
        checked += 1
        assert getattr(mod, 'MODULE_ID', None) == mod_id
        exps = getattr(mod, 'EXPERIMENTS', None)
        assert isinstance(exps, dict) and exps, f'{mod_id} missing non-empty EXPERIMENTS dict'
        keys = list(exps.keys())
        assert keys[0] == mod_id, f'{mod_id} primary experiment key must equal MODULE_ID'
        for k, fn in exps.items():
            assert k == mod_id or k.startswith(f'{mod_id}_'), f'bad experiment id {k!r} in {mod_id}'
            assert callable(fn)
    assert checked >= 15
    assert set(discover_experiments(include_unregistered=True)) >= {
        'synchrony', 'kuramoto_binding', 'sparse_dictionary', 'anytime_inference',
        'experiment_design', 'morphogenesis',
    }


@pytest.mark.parametrize('mod_id', OBSERVER_MODULE_IDS)
def test_every_live_observer_obeys_readonly_rng_reset_and_checkpoint_contract(mod_id):
    """Every registered live observer obeys read-only, RNG-isolated, resettable, JSON-checkpointable rules."""
    assert mod_id in OBSERVER_CLASSES
    spec = FrontierSpecV1.max_preset(seed=13)
    obs = make_observer(mod_id, 16, spec)
    rng = np.random.default_rng(2026)
    vecs = [np.tanh(rng.standard_normal(16)) for _ in range(18)]

    t_before = torch.random.get_rng_state().clone()
    n_before = np.random.get_state()[1].copy()

    for v in vecs[:14]:
        snap = v.copy()
        obs.observe(v)
        assert np.array_equal(v, snap), f'{mod_id} mutated caller input vector'

    assert torch.equal(torch.random.get_rng_state(), t_before), f'{mod_id} mutated global torch RNG'
    assert np.array_equal(np.random.get_state()[1], n_before), f'{mod_id} mutated global numpy RNG'

    # JSON round-trip + post-restore exact determinism
    state = json.loads(json.dumps(_jsonable(obs.export_state())))
    summary_before = obs.summary()
    clone = make_observer(mod_id, 16, spec)
    clone.import_state(state)
    assert clone.summary() == summary_before

    for v in vecs[14:]:
        obs.observe(v.copy())
        clone.observe(v.copy())
    assert clone.summary() == obs.summary()

    # Reject malformed checkpoint without mutating live observer
    good_after = obs.summary()
    with pytest.raises((ValueError, TypeError)):
        obs.import_state({'schema': 'CorruptSchemaV0'})
    assert obs.summary() == good_after

    # Deterministic reset reproduces fresh seeded sequence
    obs.reset()
    fresh = make_observer(mod_id, 16, spec)
    for v in vecs[:5]:
        obs.observe(v.copy())
        fresh.observe(v.copy())
    assert obs.summary() == fresh.summary()


def test_frontier_experiment_row_formats_both_comparison_and_validation_receipts():
    val_receipt = run_frontier_experiment('anytime_inference', quick=True, seeds=(0, 1))
    assert verify_receipt(val_receipt) is True
    val_row = frontier_experiment_row(val_receipt)
    assert val_row['verdict'] == 'pass'
    assert val_row['checks_total'] == len(val_receipt['checks'])
    assert val_row['checks_passed'] == val_row['checks_total']
    assert val_row['primary_comparison'] == f"{val_row['checks_passed']}/{val_row['checks_total']} checks passed"
    assert val_row['evidence_tier'] == EVIDENCE_TIER

    cmp_receipt = run_frontier_experiment('joint_failure', quick=True, seeds=(0, 1))
    assert verify_receipt(cmp_receipt) is True
    cmp_row = frontier_experiment_row(cmp_receipt)
    assert cmp_row['primary_comparison'] in cmp_receipt['comparisons']
    assert isinstance(cmp_row['mean_difference'], float)
    assert isinstance(cmp_row['ci95'], list) and len(cmp_row['ci95']) == 2


def test_default_synapse_config_remains_strictly_frontier_off():
    cfg = SynapseConfig()
    assert cfg.frontier is None
    assert 'frontier' not in cfg.to_dict()
