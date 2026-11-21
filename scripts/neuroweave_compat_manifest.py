"""Build or verify a Sonapse additive-compatibility manifest.

The manifest is an additive-only contract: every public surface recorded here
(config fields and defaults, CLI sub-commands and flags, HTTP routes, workbench
DOM ids, checkpoint namespaces, test modules and Max Mode features) must remain
present in later trees. New surfaces may be added freely.

Two manifests are maintained:

* ``docs/neuroweave/compatibility-manifest.json`` (default; tag
  ``sonapse-pre-neuroweave-baseline``) - the pre-NEUROWEAVE contract.
* ``docs/compatibility/manifest-pre-frontier.json`` (tag
  ``sonapse-pre-frontier-baseline``) - the pre-Frontier contract.

Usage::

    python scripts/neuroweave_compat_manifest.py --check            # default manifest
    python scripts/neuroweave_compat_manifest.py --write            # regenerate default
    python scripts/neuroweave_compat_manifest.py --check \\
        --manifest docs/compatibility/manifest-pre-frontier.json \\
        --baseline-tag sonapse-pre-frontier-baseline

To (re)build a manifest for a historical tag, inspect a worktree of that tag
in a fresh interpreter::

    git worktree add <dir> sonapse-pre-frontier-baseline
    python scripts/neuroweave_compat_manifest.py --write --root <dir> \\
        --manifest docs/compatibility/manifest-pre-frontier.json \\
        --baseline-tag sonapse-pre-frontier-baseline
    git worktree remove --force <dir>

``--baseline-tag`` labels a written manifest; on ``--check`` it additionally
requires the manifest to record that tag. Defaults are unchanged, so a bare
``--check`` behaves exactly as before.
"""
from __future__ import annotations

import argparse
import json
import os
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
_ROUTE = re.compile(r"['\"](/api/[A-Za-z0-9_/.-]*)['\"]")
# Modules that register arguments on the main parser besides synapse/cli.py.
_EXTRA_CLI_SOURCES = (
    ('synapse', 'neuroweave', 'cli_commands.py'),
    ('synapse', 'frontier', 'cli_commands.py'),
)


def _json_default(value):
    if isinstance(value, tuple):
        return list(value)
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return repr(value)


def _iter_synapse_py(root: Path) -> list[Path]:
    """Every ``synapse/**/*.py`` file except the vendored third-party tree."""
    package = root / 'synapse'
    return [path for path in sorted(package.rglob('*.py'))
            if '_vendor' not in path.relative_to(package).parts]


def _http_routes(root: Path) -> list[str]:
    """``/api/...`` string literals in every non-vendored ``synapse`` module."""
    routes: set[str] = set()
    for py_file in _iter_synapse_py(root):
        routes.update(_ROUTE.findall(py_file.read_text(encoding='utf-8', errors='ignore')))
    return sorted(routes)


def _cli_sources(root: Path) -> list[Path]:
    """``synapse/cli.py`` plus the existing modules that register main-parser arguments."""
    extras = (root.joinpath(*parts) for parts in _EXTRA_CLI_SOURCES)
    return [root / 'synapse' / 'cli.py', *(path for path in extras if path.exists())]


def _cli_surfaces(sources: list[Path]) -> tuple[list[str], list[str]]:
    """Sub-command names and long flags registered in ``sources``."""
    text = '\n'.join(path.read_text(encoding='utf-8') for path in sources)
    commands = sorted(set(re.findall(r"add_parser\(\s*['\"]([A-Za-z0-9_-]+)['\"]", text)))
    flags = sorted(set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9_-]+)['\"]", text)))
    return commands, flags


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def _git_rev(tag: str, cwd: Path, *, short: bool) -> str | None:
    cmd = ['git', 'rev-parse', '--short', tag] if short else ['git', 'rev-parse', tag]
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def collect(root: Path | None = None, *, baseline_tag: str = BASELINE_TAG) -> dict:
    """Collect the public surfaces of the tree at ``root`` (default: this checkout).

    ``root`` other than this checkout must be inspected in a fresh interpreter:
    ``SynapseConfig`` is imported from ``root`` and a ``RuntimeError`` is raised
    if a different ``synapse`` package was already imported.
    """
    explicit_root = root is not None
    repo = ROOT if root is None else Path(root).resolve()
    inserted = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        inserted = True
    try:
        from synapse.config import SynapseConfig

        if explicit_root:
            loaded = Path(sys.modules['synapse.config'].__file__).resolve().parent
            if not _same_path(loaded, repo / 'synapse'):
                raise RuntimeError(
                    f'synapse.config was imported from {loaded}, not {repo / "synapse"}; '
                    'inspect another tree in a fresh interpreter (use --root)')
        cfg = SynapseConfig()
        config_defaults = {
            f.name: _json_default(getattr(cfg, f.name)) for f in fields(SynapseConfig)
        }
        cli_commands, cli_flags = _cli_surfaces(_cli_sources(repo))
        routes = _http_routes(repo)
        html = (repo / 'synapse' / 'web' / 'index.html').read_text(encoding='utf-8')
        dom_ids = sorted(set(re.findall(r'\bid="([^"]+)"', html)))
        agent_src = (repo / 'synapse' / 'agent.py').read_text(encoding='utf-8')
        checkpoint_keys = sorted(set(_STATE_KEY.findall(agent_src)))
        tests = sorted(p.name for p in (repo / 'tests').glob('test_*.py'))
        max_cfg = SynapseConfig.max_tandem(total_neurons=48)
        max_audit = max_cfg.max_tandem_feature_audit()
        fallback = BASELINE_COMMIT if baseline_tag == BASELINE_TAG else 'unresolved'
        commit_short = _git_rev(baseline_tag, repo, short=True) or fallback
        return {
            'schema': 'sonapse-compatibility-manifest-v1',
            'baseline_tag': baseline_tag,
            'baseline_commit': commit_short,
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
    finally:
        if inserted and str(repo) in sys.path:
            sys.path.remove(str(repo))


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    parser.add_argument('--manifest', type=Path, default=MANIFEST,
                        help='manifest JSON path, relative to the repository root '
                             '(default: docs/neuroweave/compatibility-manifest.json)')
    parser.add_argument('--baseline-tag', default=None,
                        help=f'git tag the manifest describes (default for --write: {BASELINE_TAG}); '
                             'with --check, the manifest must record this tag')
    parser.add_argument('--root', type=Path, default=None,
                        help='tree to inspect, e.g. a git worktree of the baseline tag (default: this checkout)')
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    tag = args.baseline_tag or BASELINE_TAG
    root = None if args.root is None else args.root.resolve()
    current = collect(root, baseline_tag=tag)
    if args.write:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        full = _git_rev(tag, ROOT if root is None else root, short=False)
        if full:
            current['baseline_commit_full'] = full
        manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        try:
            rel = manifest_path.relative_to(ROOT)
        except ValueError:
            rel = manifest_path
        print(f'wrote {rel}')
        return 0
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    problems = check(manifest, current)
    if args.baseline_tag is not None and manifest.get('baseline_tag') != args.baseline_tag:
        problems.insert(0, f'baseline tag mismatch: manifest records {manifest.get("baseline_tag")!r}, '
                           f'expected {args.baseline_tag!r}')
    for problem in problems:
        print(problem)
    print('compatibility manifest OK' if not problems else f'{len(problems)} compatibility problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
