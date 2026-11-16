"""Build or verify the pre-NEUROWEAVE Sonapse compatibility manifest.

The manifest is an additive-only contract: every public surface recorded here
(config fields and defaults, CLI sub-commands, HTTP routes, workbench DOM ids,
checkpoint namespaces and test modules) must remain present after the
NEUROWEAVE evolution. New surfaces may be added freely.

Usage::

    python scripts/neuroweave_compat_manifest.py --write   # (re)generate
    python scripts/neuroweave_compat_manifest.py --check   # verify superset
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs' / 'neuroweave' / 'compatibility-manifest.json'
BASELINE_TAG = 'sonapse-pre-neuroweave-baseline'
BASELINE_COMMIT = '6f2761b'

# Checkpoint namespaces written by Agent.export_state before NEUROWEAVE.
_STATE_KEY = re.compile(r"state\[['\"]([A-Za-z0-9_]+)['\"]\]\s*=")


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return repr(value)


def collect() -> dict:
    sys.path.insert(0, str(ROOT))
    from synapse.config import SynapseConfig

    cfg = SynapseConfig()
    config_defaults = {
        f.name: _json_default(getattr(cfg, f.name)) for f in fields(SynapseConfig)
    }
    cli_src = (ROOT / 'synapse' / 'cli.py').read_text(encoding='utf-8')
    cli_commands = sorted(set(re.findall(r"add_parser\(\s*['\"]([A-Za-z0-9_-]+)['\"]", cli_src)))
    cli_flags = sorted(set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9_-]+)['\"]", cli_src)))
    server_src = (ROOT / 'synapse' / 'server.py').read_text(encoding='utf-8')
    routes = sorted(set(re.findall(r"['\"](/api/[A-Za-z0-9_/.-]*)['\"]", server_src)))
    html = (ROOT / 'synapse' / 'web' / 'index.html').read_text(encoding='utf-8')
    dom_ids = sorted(set(re.findall(r'\bid="([^"]+)"', html)))
    agent_src = (ROOT / 'synapse' / 'agent.py').read_text(encoding='utf-8')
    checkpoint_keys = sorted(set(_STATE_KEY.findall(agent_src)))
    tests = sorted(p.name for p in (ROOT / 'tests').glob('test_*.py'))
    max_cfg = SynapseConfig.max_tandem(total_neurons=48)
    max_audit = max_cfg.max_tandem_feature_audit()
    return {
        'schema': 'sonapse-compatibility-manifest-v1',
        'baseline_tag': BASELINE_TAG,
        'baseline_commit': BASELINE_COMMIT,
        'contract': 'additive-only: every recorded item must remain present; defaults must stay equal',
        'config_defaults': config_defaults,
        'cli_commands': cli_commands,
        'cli_flags': cli_flags,
        'http_routes': routes,
        'web_dom_ids': dom_ids,
        'checkpoint_namespaces': checkpoint_keys,
        'test_modules': tests,
        'max_mode_enabled_features': sorted(max_audit['enabled_features']),
    }


def check(manifest: dict, current: dict) -> list[str]:
    problems: list[str] = []
    for key, old in manifest['config_defaults'].items():
        if key not in current['config_defaults']:
            problems.append(f'config field removed: {key}')
        elif current['config_defaults'][key] != old:
            problems.append(f'config default changed: {key}: {old!r} -> {current["config_defaults"][key]!r}')
    for section in ('cli_commands', 'cli_flags', 'http_routes', 'web_dom_ids',
                    'checkpoint_namespaces', 'test_modules', 'max_mode_enabled_features'):
        missing = sorted(set(manifest[section]) - set(current[section]))
        problems.extend(f'{section} missing: {item}' for item in missing)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    current = collect()
    if args.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        try:
            head = subprocess.run(['git', 'rev-parse', BASELINE_TAG], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
            current['baseline_commit_full'] = head
        except (OSError, subprocess.CalledProcessError):
            pass
        MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(f'wrote {MANIFEST.relative_to(ROOT)}')
        return 0
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    problems = check(manifest, current)
    for problem in problems:
        print(problem)
    print('compatibility manifest OK' if not problems else f'{len(problems)} compatibility problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
