"""Supplementary budget-matched synthetic sequence-variant search (small and fast)."""
import json
from pathlib import Path

import numpy as np
import pytest

from synapse.neuroweave import experiments as ex
from synapse.neuroweave.experiments import verify_receipt
from synapse.neuroweave.sequence_search import (DEFAULTS, STRATEGIES, make_landscape, run_sequence_search_assay,
                                                run_strategy)
from synapse.neuroweave.spec import EXPERIMENT_IDS, EXPERIMENTS

WEB = Path(__file__).resolve().parents[1] / 'synapse' / 'web'
TINY = dict(landscapes=2, seeds=2, budget=16, reference_samples=200)


@pytest.fixture(scope='module')
def receipt():
    return run_sequence_search_assay(**TINY)


def test_receipt_is_budget_matched_sealed_and_honest(receipt):
    assert receipt['schema'] == 'NeuroWeaveSequenceSearchV1' and receipt['supplementary'] is True
    assert receipt['evidence_tier'] == 'synthetic_abstraction' and receipt['protein_modelling'] is False
    assert 'not a protein-folding model' in receipt['disclaimer']
    assert verify_receipt(receipt)
    assert [r['strategy'] for r in receipt['results']] == list(STRATEGIES)
    for r in receipt['results']:
        assert r['runs'] == 4 and r['queries_per_run'] == 16
        assert 0.0 <= r['mean_best_fitness'] <= 1.0 and 0.0 <= r['top1pct_hit_rate'] <= 1.0
    assert receipt['budget_matched'] is True and receipt['reference_samples_counted_in_budget'] is False
    assert len(receipt['landscapes']) == 2
    for name in ('sparse_vs_evolution', 'sparse_vs_random'):
        c = receipt['comparisons'][name]
        assert c['n'] == 4 and c['verdict'] in ex.VERDICTS
    assert receipt['verdict'] == receipt['comparisons']['sparse_vs_evolution']['verdict']
    json.dumps(receipt, allow_nan=False)


def test_deterministic_excluding_wallclock(receipt):
    again = run_sequence_search_assay(**TINY)
    assert again['digest'] == receipt['digest']
    tampered = json.loads(json.dumps(receipt))
    tampered['results'][0]['mean_best_fitness'] += 0.01
    assert not verify_receipt(tampered)


def test_each_strategy_uses_exactly_the_budget_and_landscape_is_bounded():
    cfg = dict(DEFAULTS, budget=20)
    land = make_landscape(0, cfg)
    assert land.evaluate(land.target, count=False) == pytest.approx(1.0)
    assert land.queries == 0
    for name in STRATEGIES:
        res = run_strategy(name, land, 5, cfg)
        assert res['queries'] == 20 and 0.0 <= res['best'] <= 1.0
    assert land.queries == 60
    rng = np.random.default_rng(0)
    vals = [land.evaluate(rng.integers(0, 20, 24), count=False) for _ in range(50)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    with pytest.raises(ValueError):
        run_strategy('gradient', land, 0, cfg)


@pytest.mark.parametrize('kw', [dict(budget=5), dict(landscapes=0), dict(seeds=100), dict(quick='yes'),
                                dict(k_winners=64, projection_dim=8)])
def test_rejects_out_of_bounds_options(kw):
    with pytest.raises(ValueError):
        run_sequence_search_assay(**kw)


def test_supplementary_registry_does_not_change_the_24_experiments():
    assert len(EXPERIMENTS) == 24 and len(EXPERIMENT_IDS) == 24 and 'sequence_search' not in EXPERIMENT_IDS
    assert ex.SUPPLEMENTARY_IDS == ('sequence_search',)
    assert [e['id'] for e in ex.supplementary_catalog()] == ['sequence_search']
    out = ex.run_supplementary_experiment('sequence_search', **TINY)
    assert out['experiment_id'] == 'sequence_search' and verify_receipt(out)
    with pytest.raises(ValueError):
        ex.run_supplementary_experiment('xor')
    with pytest.raises(ValueError):
        ex.run_experiment('sequence_search')


def test_cli_sequence_search(capsys):
    from synapse.cli import main
    assert main(['neuroweave', 'sequence-search', '--landscapes', '1', '--seeds-per-landscape', '2',
                 '--budget', '14']) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['ok'] is True and out['command'] == 'neuroweave sequence-search'
    assert out['result']['config']['budget'] == 14 and out['result']['config']['landscapes'] == 1
    assert 'protein' in out['disclaimer']
    with pytest.raises(SystemExit):
        main(['neuroweave', 'sequence-search', '--budget', '3'])
    capsys.readouterr()
    assert main(['neuroweave', 'experiments']) == 0
    cat = json.loads(capsys.readouterr().out)
    assert cat['count'] == 24 and [e['id'] for e in cat['supplementary']] == ['sequence_search']


class _Runtime:
    def snapshot(self): return {'step': 0}
    def step(self, steps=1, learn=False): return {'step': steps}


def test_server_endpoint_and_ui_wiring(tmp_path):
    from synapse.server import Workbench
    wb = Workbench(_Runtime(), output_root=tmp_path)
    res = wb.post('/api/neuroweave/sequence-search', {'quick': True, 'landscapes': 1, 'seeds': 2, 'budget': 14})
    assert res['result']['config']['budget'] == 14 and verify_receipt(res['result'])
    assert res['evidence_tier'] == 'synthetic_abstraction' and 'protein' in res['disclaimer']
    for bad in ({'budget': 1000}, {'landscapes': 0}, {'unknown': 1}, {'quick': 'no'}):
        with pytest.raises(ValueError):
            wb.post('/api/neuroweave/sequence-search', bad)
    cat = wb.get('/api/neuroweave/catalog')
    assert [e['id'] for e in cat['supplementary_experiments']] == ['sequence_search']
    html = (WEB / 'index.html').read_text(encoding='utf-8')
    js = (WEB / 'app.js').read_text(encoding='utf-8')
    for ident in ('nwl-sequence-run', 'nwl-sequence-body', 'nwl-sequence-status', 'nwl-sequence-details'):
        assert f'id="{ident}"' in html and f"'{ident}'" in js
    assert "neuroweave/sequence-search" in js and 'not a protein-folding model' in html
