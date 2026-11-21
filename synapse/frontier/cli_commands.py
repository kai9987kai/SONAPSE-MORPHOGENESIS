"""Handlers for ``python -m synapse.cli frontier ...`` and global ``--frontier*`` CLI flags (opt-in only).

Every subcommand handler returns a JSON-serialisable dict with an ``ok`` flag; nothing here changes any
Sonapse or NEUROWEAVE default.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .harness import _jsonable
from .runtime import (
    FrontierObserverHub,
    experiment_catalog,
    run_all_frontier_experiments,
    run_frontier_experiment,
    run_frontier_laboratory,
)
from .spec import (
    EVIDENCE_TIER,
    GRAFT_OPTIMIZERS,
    MODULE_IDS,
    OBSERVER_MODULE_IDS,
    FrontierSpecV1,
    module_catalog,
)

__all__ = [
    'parse_observer_modules',
    'parse_seeds',
    'cli_frontier_spec',
    'register_frontier_cli',
    'run_frontier_command',
]

MAX_CLI_JSON_BYTES = 1_000_000
MAX_CLI_OBSERVE_STEPS = 4096
MAX_CLI_OBSERVE_UNITS = 4096
_SYNTH_STREAM_OFFSET = 61_543


def _read_json_file(path: Path, limit: int = MAX_CLI_JSON_BYTES) -> Any:
    p = Path(path)
    size = p.stat().st_size
    if size > limit:
        raise ValueError(f'{p} exceeds {limit} bytes (checked before reading)')

    def _reject(token: str) -> Any:
        raise ValueError(f'non-standard JSON constant {token}')

    return json.loads(p.read_text(encoding='utf-8'), parse_constant=_reject)


def parse_observer_modules(text: str | None) -> tuple[str, ...]:
    """``'all'`` -> every observer module, ``'none'``/``''`` -> ``()``, else a comma list of observer ids."""
    if text is None:
        return ()
    t = text.strip()
    if t.lower() == 'all':
        return OBSERVER_MODULE_IDS
    if t.lower() in ('', 'none'):
        return ()
    items = [m.strip() for m in t.split(',') if m.strip()]
    unknown = [m for m in items if m not in MODULE_IDS]
    if unknown:
        raise ValueError(f'unknown frontier modules: {unknown}')
    non_obs = [m for m in items if m not in OBSERVER_MODULE_IDS]
    if non_obs:
        raise ValueError(f'modules without a live observer: {non_obs} (observers: {OBSERVER_MODULE_IDS})')
    return tuple(m for m in OBSERVER_MODULE_IDS if m in set(items))


def parse_seeds(text: str | None, default: tuple[int, ...] = (0, 1, 2), max_count: int = 16) -> tuple[int, ...]:
    if text is None or not str(text).strip():
        return default
    parts = [p.strip() for p in str(text).split(',') if p.strip()]
    if not 1 <= len(parts) <= max_count:
        raise ValueError(f'--seeds must list between 1 and {max_count} integers')
    seeds: list[int] = []
    for p in parts:
        try:
            val = int(p)
        except ValueError as exc:
            raise ValueError(f'invalid seed {p!r}') from exc
        if not 0 <= val <= 2**31 - 1:
            raise ValueError('seeds must be non-negative 31-bit ints')
        seeds.append(val)
    if len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique')
    return tuple(seeds)


def cli_frontier_spec(enabled: bool | None, config_path: Path | None, seed: int, *,
                      modules: str | None = None, graft_optimizer: str | None = None) -> FrontierSpecV1 | None:
    """Build a :class:`FrontierSpecV1` from global CLI flags, or ``None`` when not requested."""
    if config_path is not None:
        if enabled is False:
            raise ValueError('--no-frontier conflicts with --frontier-config')
        data = _read_json_file(Path(config_path))
        spec = FrontierSpecV1.from_dict(data)
        if modules is not None:
            spec = spec.with_modules(parse_observer_modules(modules))
        if graft_optimizer is not None:
            spec = replace(spec, graft_optimizer=graft_optimizer)
        return spec
    has_opt_in = bool(enabled) or (modules is not None) or (graft_optimizer not in (None, 'adam'))
    if not has_opt_in:
        return None
    if enabled is False:
        raise ValueError('--no-frontier conflicts with frontier module or graft-optimizer options')
    obs_mods = OBSERVER_MODULE_IDS if modules is None and enabled else parse_observer_modules(modules)
    return FrontierSpecV1(
        enabled=True,
        seed=int(seed),
        modules=obs_mods,
        graft_optimizer='adam' if graft_optimizer is None else graft_optimizer,
    )


def register_frontier_cli(parser: Any, subparsers: Any = None) -> None:
    """Add global ``--frontier*`` options to ``parser`` and optionally a ``frontier`` subcommand."""
    parser.add_argument('--frontier', dest='frontier_enabled', action='store_true', default=None,
                        help='Enable read-only live frontier observers on the brain')
    parser.add_argument('--no-frontier', dest='frontier_enabled', action='store_false',
                        help='Disable live frontier observers')
    parser.add_argument('--frontier-config', type=Path, default=None,
                        help='Path to a FrontierSpecV1 JSON file')
    parser.add_argument('--frontier-modules', type=str, default=None,
                        help='Comma-separated observer modules (or "all" / "none")')
    parser.add_argument('--frontier-graft-optimizer', choices=GRAFT_OPTIMIZERS, default=None,
                        help='Opt-in optimizer for the live neural-graft fit (adam or muon)')
    if subparsers is not None:
        fp = subparsers.add_parser('frontier', help='Frontier pack catalog, observers and experiments')
        fp.add_argument('frontier_action', choices=('catalog', 'experiment', 'observe', 'laboratory'),
                        nargs='?', default='catalog')
        fp.add_argument('--id', dest='frontier_id', type=str, default='all',
                        help='Experiment id to run (or "all")')
        fp.add_argument('--quick', dest='frontier_quick', action='store_true', default=False,
                        help='Run experiment(s) in fast quick mode')
        fp.add_argument('--seeds', dest='frontier_seeds', type=str, default='0,1,2',
                        help='Comma-separated integer seeds')
        fp.add_argument('--modules', dest='frontier_sub_modules', type=str, default='all',
                        help='Observer modules for observe/laboratory')
        fp.add_argument('--steps', dest='frontier_steps', type=int, default=32,
                        help='Synthetic observation steps for "frontier observe"')
        fp.add_argument('--units', dest='frontier_units', type=int, default=32,
                        help='Synthetic unit count for "frontier observe"')
        fp.add_argument('--out', dest='frontier_out', type=Path, default=None,
                        help='Optional path to write the JSON receipt')


def _write_optional(out_path: Path | None, payload: Any) -> dict[str, Any] | None:
    if out_path is None:
        return None
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True)
    p.write_text(text, encoding='utf-8')
    return {'path': str(p), 'bytes': len(text.encode('utf-8'))}


def run_frontier_command(args: Any) -> dict[str, Any]:
    """Execute ``python -m synapse.cli frontier <action>`` and return a JSON-safe response."""
    action = getattr(args, 'frontier_action', 'catalog')
    seed = int(getattr(args, 'seed', 7))
    out_path = getattr(args, 'frontier_out', None)
    try:
        if action == 'catalog':
            res: dict[str, Any] = {
                'ok': True,
                'action': 'catalog',
                'evidence_tier': EVIDENCE_TIER,
                'modules': module_catalog(),
                'observer_module_ids': list(OBSERVER_MODULE_IDS),
                'experiments': experiment_catalog(include_unregistered=True),
            }
        elif action == 'experiment':
            exp_id = str(getattr(args, 'frontier_id', 'all')).strip()
            quick = bool(getattr(args, 'frontier_quick', False))
            seeds = parse_seeds(getattr(args, 'frontier_seeds', '0,1,2'),
                                default=(0, 1) if quick else (0, 1, 2))
            if exp_id.lower() == 'all':
                receipts = run_all_frontier_experiments(
                    quick=quick, seeds=seeds, primary_only=True, include_unregistered=True)
                res = {'ok': True, 'action': 'experiment', 'count': len(receipts), 'receipts': receipts}
            else:
                receipt = run_frontier_experiment(exp_id, quick=quick, seeds=seeds, include_unregistered=True)
                res = {'ok': True, 'action': 'experiment', 'receipt': receipt}
        elif action == 'observe':
            steps = int(getattr(args, 'frontier_steps', 32))
            units = int(getattr(args, 'frontier_units', 32))
            if not 1 <= steps <= MAX_CLI_OBSERVE_STEPS:
                raise ValueError(f'--steps must be in [1, {MAX_CLI_OBSERVE_STEPS}]')
            if not 2 <= units <= MAX_CLI_OBSERVE_UNITS:
                raise ValueError(f'--units must be in [2, {MAX_CLI_OBSERVE_UNITS}]')
            mods = parse_observer_modules(getattr(args, 'frontier_sub_modules', 'all'))
            spec = FrontierSpecV1(enabled=True, seed=seed, modules=mods,
                                  history=max(16, min(256, steps * 2)),
                                  fc_window=max(8, min(128, max(16, steps))))
            hub = FrontierObserverHub(spec, n_units=units)
            rng = np.random.default_rng(seed + _SYNTH_STREAM_OFFSET)
            for _ in range(steps):
                hub.observe(np.tanh(rng.standard_normal(units)))
            res = {'ok': True, 'action': 'observe', 'summary': hub.summary()}
        elif action == 'laboratory':
            quick = bool(getattr(args, 'frontier_quick', True))
            seeds = parse_seeds(getattr(args, 'frontier_seeds', '0,1'), default=(0, 1))
            mods = parse_observer_modules(getattr(args, 'frontier_sub_modules', 'all'))
            spec = FrontierSpecV1(enabled=True, seed=seed, modules=mods)
            lab = run_frontier_laboratory(spec, experiments=True, quick=quick, seeds=seeds)
            res = {'ok': True, 'action': 'laboratory', 'receipt': lab}
        else:
            raise ValueError(f'unknown frontier action: {action!r}')
        written = _write_optional(out_path, res)
        if written is not None:
            res['written'] = written
        return _jsonable(res)
    except Exception as exc:
        return {'ok': False, 'action': action, 'error': str(exc)}
