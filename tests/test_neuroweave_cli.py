"""`synapse.cli neuroweave ...` sub-commands and the opt-in global NEUROWEAVE flags (tiny settings)."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from synapse.cli import main
from synapse.neuroweave.cli_commands import cli_neuroweave_spec, parse_mechanisms
from synapse.neuroweave.spec import MECHANISM_IDS, NeuroWeaveSpecV1

_HAS_EXPERIMENTS = importlib.util.find_spec('synapse.neuroweave.experiments') is not None
_HAS_LESION = importlib.util.find_spec('synapse.neuroweave.lesion') is not None


def _run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out)


def _legacy_module():
    path = Path(__file__).with_name('test_neuroweave_hardware_package.py')
    spec = importlib.util.spec_from_file_location('_nw_hw_pkg_tests', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_catalog_commands(capsys):
    code, res = _run(capsys, 'neuroweave', 'mechanisms')
    assert code == 0 and res['ok'] and res['count'] == 45 and len(res['mechanisms']) == 45
    code, res = _run(capsys, 'neuroweave', 'experiments')
    assert code == 0 and res['count'] == 24


def test_validate_good_and_bad_config(capsys, tmp_path):
    good = tmp_path / 'good.json'
    good.write_text(json.dumps(NeuroWeaveSpecV1(enabled=True, mechanisms=('stdp',)).to_dict()), encoding='utf-8')
    code, res = _run(capsys, 'neuroweave', 'validate', '--config', str(good))
    assert code == 0 and res['valid'] and res['mechanisms'] == 1
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'mechanisms': ['not_a_mechanism']}), encoding='utf-8')
    code, res = _run(capsys, 'neuroweave', 'validate', '--config', str(bad))
    assert code == 1 and res['valid'] is False and 'not_a_mechanism' in res['errors'][0]
    nan = tmp_path / 'nan.json'
    nan.write_text('{"gain": NaN}', encoding='utf-8')
    code, res = _run(capsys, 'neuroweave', 'validate', '--config', str(nan))
    assert code == 1 and res['valid'] is False


def test_smoke_all_and_none(capsys):
    code, res = _run(capsys, 'neuroweave', 'smoke', '--steps', '3', '--neurons', '16', '--seed', '3',
                     '--topology', 'multi_region', '--mechanisms', 'all')
    assert code == 0 and res['ok'] and res['checkpoint_replay_exact'] and res['output_finite']
    assert len(res['mechanisms']) == 45 and res['neurons'] >= 16 and res['evidence_tier'] == 'synthetic_abstraction'
    assert 'wallclock_ms_last' not in res['telemetry']
    code, res2 = _run(capsys, 'neuroweave', 'smoke', '--steps', '2', '--neurons', '16', '--mechanisms', 'none')
    assert code == 0 and res2['mechanisms'] == []
    code, res3 = _run(capsys, 'neuroweave', 'smoke', '--steps', '2', '--neurons', '16', '--mechanisms', 'stdp,sparse_code')
    assert res3['mechanisms'] == ['sparse_code', 'stdp']


@pytest.mark.parametrize('argv', [
    ('--steps', '0'), ('--mechanisms', 'bogus'), ('--neurons', '4'), ('--topology', 'nope'),
])
def test_smoke_rejects_bad_options(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        main(['neuroweave', 'smoke', '--neurons', '16', *argv] if '--neurons' not in argv else ['neuroweave', 'smoke', *argv])
    assert exc.value.code == 2


def test_topology_sweep(capsys):
    code, res = _run(capsys, 'neuroweave', 'topology', '--neurons', '16', '--seeds', '1', '2')
    assert code == 0 and len(res['topologies']) == 12
    row = res['topologies']['clustered']
    assert len(row['per_seed']) == 2 and row['mean']['edges'] > 0


def test_export_inspect_and_tamper(capsys, tmp_path):
    out, page, fpga = tmp_path / 'p' / 'pkg.json', tmp_path / 'page.html', tmp_path / 'fpga.json'
    code, res = _run(capsys, 'neuroweave', 'export', '--steps', '2', '--neurons', '16', '--mechanisms', 'all',
                     '--out', str(out), '--html', str(page), '--fpga', str(fpga))
    assert code == 0 and res['valid'] and out.is_file() and page.is_file() and fpga.is_file()
    assert res['fpga']['rtl_generated'] is False and res['fpga']['hardware_validated'] is False
    assert 'stdp' in res['html']['not_executed']
    html_text = page.read_text(encoding='utf-8')
    assert 'id="nw-engine"' in html_text and 'Not executed in standalone' in html_text
    pkg = json.loads(out.read_text(encoding='utf-8'))
    assert 'standalone' in pkg['capabilities']['frozen_inference']
    before = out.read_bytes()
    code, ins = _run(capsys, 'neuroweave', 'inspect', str(out))
    assert code == 0 and ins['report']['valid'] and ins['read_only'] and out.read_bytes() == before
    pkg['brain']['readout']['b'] = 0.5
    bad = tmp_path / 'tampered.json'
    bad.write_text(json.dumps(pkg), encoding='utf-8')
    code, ins = _run(capsys, 'neuroweave', 'inspect', str(bad))
    assert code == 1 and not ins['report']['valid']


def test_import_legacy_summary_and_convert(capsys, tmp_path):
    legacy = _legacy_module().make_legacy('1.2.0')
    src = tmp_path / 'legacy.json'
    src.write_text(json.dumps(legacy), encoding='utf-8')
    code, res = _run(capsys, 'neuroweave', 'import-legacy', str(src))
    assert code == 0 and res['summary']['valid'] and res['converted'] is None
    dst = tmp_path / 'converted.json'
    code, res = _run(capsys, 'neuroweave', 'import-legacy', str(src), '--out', str(dst))
    assert code == 0 and res['converted']['valid'] and dst.is_file()
    code, ins = _run(capsys, 'neuroweave', 'inspect', str(dst))
    assert code == 0 and ins['report']['valid']
    junk = tmp_path / 'junk.json'
    junk.write_text('{"format": "NEUROWEAVE_MODEL_PACKAGE"', encoding='utf-8')
    code, res = _run(capsys, 'neuroweave', 'import-legacy', str(junk))
    assert code == 1 and not res['ok']


def test_ecology_engine_only(capsys):
    code, res = _run(capsys, 'neuroweave', 'ecology', '--steps', '3', '--neurons', '16', '--mechanisms', 'none',
                     '--config', 'scarce')
    assert code == 0 and res['engine']['steps'] == 3 and res['synapse_agent'] is None
    with pytest.raises(SystemExit):
        main(['neuroweave', 'ecology', '--steps', '2', '--config', 'no_such_world'])


def test_ecology_with_sonapse_agent_and_global_flag(capsys):
    code, res = _run(capsys, 'neuroweave', 'ecology', '--steps', '2', '--neurons', '16', '--mechanisms', 'none',
                     '--synapse-agent', '--agent-nodes', '16')
    assert code == 0 and res['synapse_agent']['neuroweave_attached'] is False
    assert res['synapse_agent']['result']['steps'] == 2
    code, res = _run(capsys, '--neuroweave', 'neuroweave', 'ecology', '--steps', '2', '--neurons', '16',
                     '--mechanisms', 'none', '--synapse-agent', '--agent-nodes', '16')
    assert code == 0 and res['synapse_agent']['neuroweave_attached'] is True


def test_global_flags_set_config_only_when_requested(monkeypatch, capsys, tmp_path):
    import synapse.runtime as runtime_mod
    seen = []

    class FakeRuntime:
        def __init__(self, config, output_root=None):
            seen.append(config)

        def step(self, steps, learn=False):
            return {'steps': steps, 'learn': learn}

    monkeypatch.setattr(runtime_mod, 'LabRuntime', FakeRuntime)
    assert main(['--output', str(tmp_path), '--nodes', '16', 'run', '--steps', '1']) == 0
    assert seen[-1].neuroweave is None  # default unchanged
    assert main(['--output', str(tmp_path), '--nodes', '16', '--neuroweave', 'train', '--steps', '1']) == 0
    assert seen[-1].neuroweave.enabled and seen[-1].neuroweave.mechanisms == MECHANISM_IDS
    cfg = tmp_path / 'nw.json'
    cfg.write_text(json.dumps(NeuroWeaveSpecV1(enabled=True, observe_only=True, mechanisms=('stdp',)).to_dict()),
                   encoding='utf-8')
    assert main(['--output', str(tmp_path), '--nodes', '16', '--neuroweave-config', str(cfg), 'run', '--steps', '1']) == 0
    assert seen[-1].neuroweave.observe_only and seen[-1].neuroweave.mechanisms == ('stdp',)
    assert main(['--output', str(tmp_path), '--nodes', '16', '--no-neuroweave', 'run', '--steps', '1']) == 0
    assert seen[-1].neuroweave is None
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(['--output', str(tmp_path), '--no-neuroweave', '--neuroweave-config', str(cfg), 'run'])
    assert exc.value.code == 2


def test_helpers():
    assert parse_mechanisms('all') == MECHANISM_IDS and parse_mechanisms('none') == () and parse_mechanisms(None) == ()
    with pytest.raises(ValueError):
        parse_mechanisms('stdp,unknown')
    assert cli_neuroweave_spec(None, None, 7) is None and cli_neuroweave_spec(False, None, 7) is None


@pytest.mark.skipif(not _HAS_EXPERIMENTS, reason='synapse.neuroweave.experiments (lab stage) is not present')
def test_experiment_quick(capsys, tmp_path):
    csv_path = tmp_path / 'exp.csv'
    code, res = _run(capsys, 'neuroweave', 'experiment', 'context_attention', '--quick', '--seeds', '0', '1',
                     '--steps', '8', '--csv', str(csv_path))
    assert code == 0 and res['result']['experiment_id'] == 'context_attention'
    assert res['result']['evidence_tier'] == 'synthetic_abstraction'
    assert csv_path.read_text(encoding='utf-8').startswith('experiment_id,')
    with pytest.raises(SystemExit):
        main(['neuroweave', 'experiment', 'no_such_experiment'])


@pytest.mark.skipif(not _HAS_LESION, reason='synapse.neuroweave.lesion (lab stage) is not present')
def test_lesion_quick(capsys):
    code, res = _run(capsys, 'neuroweave', 'lesion', '--quick', '--steps', '8', '--neurons', '16')
    assert code == 0 and res['options']['steps'] == 8 and isinstance(res['result'], dict)


@pytest.mark.skipif(shutil.which('git') is None, reason='manifest check needs git metadata')
def test_manifest_check_command(capsys):
    code, res = _run(capsys, 'neuroweave', 'manifest-check')
    assert code == 0 and res['ok'] and 'compatibility manifest OK' in res['output'][-1]
