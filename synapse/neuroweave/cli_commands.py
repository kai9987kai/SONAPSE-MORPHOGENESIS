"""Handlers for ``python -m synapse.cli neuroweave ...`` (JSON results, opt-in only).

Every handler returns a JSON-serialisable dict with an ``ok`` flag; the CLI
prints it to stdout and exits 1 when ``ok`` is False. Nothing here changes a
Sonapse default: the NEUROWEAVE engine/agent is built only inside these
commands. Lab modules (``experiments.py``, ``lesion.py``) are imported lazily.
"""
from __future__ import annotations

import importlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .spec import EVIDENCE_TIER, EXPERIMENT_IDS, MECHANISM_IDS, TOPOLOGIES, NeuroWeaveSpecV1

__all__ = ['run_neuroweave_command', 'parse_mechanisms', 'spec_from_cli', 'cli_neuroweave_spec']

_ROOT = Path(__file__).resolve().parents[2]
MAX_CLI_STEPS = 10_000
MAX_CLI_NEURONS = 1024
TOPOLOGY_METRIC_KEYS = ('density', 'clustering_mean', 'modularity', 'mean_shortest_path', 'global_efficiency',
                        'in_degree_gini', 'hub_top10_in_share', 'reachable_pair_fraction')


# =========================================================================== helpers
def _clean(value: Any) -> Any:
    """JSON-safe copy: numpy -> python, non-finite floats -> None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _bounded(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'--{name} must be an integer in [{low},{high}]')
    return value


def parse_mechanisms(text: str | None) -> tuple[str, ...]:
    """``all`` -> every mechanism, ``none``/empty -> none, else a comma list of known ids."""
    if text is None:
        return ()
    t = text.strip()
    if t.lower() == 'all':
        return MECHANISM_IDS
    if t.lower() in ('', 'none'):
        return ()
    items = [m.strip() for m in t.split(',') if m.strip()]
    unknown = [m for m in items if m not in MECHANISM_IDS]
    if unknown:
        raise ValueError(f'unknown NEUROWEAVE mechanisms: {unknown}')
    return tuple(items)


def spec_from_cli(args: Any) -> NeuroWeaveSpecV1:
    seed = getattr(args, 'nw_seed', 7)
    neurons = _bounded('neurons', getattr(args, 'nw_neurons', 48), 8, MAX_CLI_NEURONS)
    return NeuroWeaveSpecV1(enabled=True, seed=seed, neurons=neurons, max_neurons=max(neurons, min(4096, neurons * 2)),
                            topology=getattr(args, 'nw_topology', 'clustered'),
                            mechanisms=parse_mechanisms(getattr(args, 'nw_mechanisms', 'all')))


def _read_json_file(path: Path, limit: int) -> Any:
    size = Path(path).stat().st_size
    if size > limit:
        raise ValueError(f'{path} exceeds {limit} bytes (checked before reading)')

    def _reject(token: str) -> Any:
        raise ValueError(f'non-standard JSON constant {token}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject)


def cli_neuroweave_spec(enabled: bool | None, config_path: Path | None, seed: int) -> NeuroWeaveSpecV1 | None:
    """Spec for the global ``--neuroweave`` / ``--neuroweave-config`` flags (``None`` when not requested)."""
    if config_path is not None:
        if enabled is False:
            raise ValueError('--no-neuroweave conflicts with --neuroweave-config')
        data = _read_json_file(config_path, 1_000_000)
        return NeuroWeaveSpecV1.from_dict(data)
    if enabled:
        return NeuroWeaveSpecV1(enabled=True, gain=0.02, max_gain=0.1, seed=seed, mechanisms=MECHANISM_IDS)
    return None


def _write(path: Path, text: str) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return {'path': str(path), 'bytes': len(text.encode('utf-8'))}


def _lab(module: str, func: str) -> Callable[..., Any]:
    name = f'synapse.neuroweave.{module}'
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            raise RuntimeError(f'NEUROWEAVE lab module {module}.py is not available in this build') from exc
        raise
    fn = getattr(mod, func, None)
    if not callable(fn):
        raise RuntimeError(f'{name}.{func} is not available in this build')
    return fn


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a lab function passing only the keyword options its signature accepts."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _drive(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return rng.uniform(-1.0, 1.0, 8), rng.uniform(-1.0, 1.0, 16)


def _run_engine(spec: NeuroWeaveSpecV1, steps: int, seed: int):
    from .engine import NeuroWeaveEngine
    eng = NeuroWeaveEngine(spec)
    rng = np.random.default_rng(seed)
    probs = []
    for t in range(steps):
        task, sens = _drive(rng)
        res = eng.step(task, sens, reward=float(np.sin(0.7 * t)) * 0.5)
        probs.append(float(res['readout_probability']))
    return eng, probs, rng


# =========================================================================== commands
def _cmd_validate(args: Any) -> dict[str, Any]:
    try:
        spec = NeuroWeaveSpecV1.from_dict(_read_json_file(args.nw_config, 1_000_000))
    except (ValueError, TypeError) as exc:
        return {'ok': False, 'valid': False, 'errors': [str(exc)], 'path': str(args.nw_config)}
    return {'ok': True, 'valid': True, 'path': str(args.nw_config), 'spec': spec.to_dict(),
            'spec_digest': spec.spec_digest, 'mechanisms': len(spec.mechanisms), 'enabled': spec.enabled,
            'evidence_tier': EVIDENCE_TIER}


def _cmd_smoke(args: Any) -> dict[str, Any]:
    from .engine import NeuroWeaveEngine
    steps = _bounded('steps', args.nw_steps, 1, MAX_CLI_STEPS)
    spec = spec_from_cli(args)
    eng, probs, rng = _run_engine(spec, steps, spec.seed)
    clone = NeuroWeaveEngine(spec)
    clone.import_state(eng.export_state())
    replay = True
    for _ in range(2):
        task, sens = _drive(rng)
        a = eng.step(task, sens, reward=0.1)
        b = clone.step(task, sens, reward=0.1)
        replay = replay and bool(np.array_equal(a['output'], b['output'])) \
            and a['readout_probability'] == b['readout_probability']
    out = eng.brain.cells.output
    return {
        'ok': bool(np.all(np.isfinite(out))) and replay, 'command': 'neuroweave smoke',
        'spec_digest': spec.spec_digest, 'seed': spec.seed, 'topology': spec.topology,
        'mechanisms': list(spec.mechanisms), 'steps': steps, 'neurons': int(eng.n),
        'synapses': int(eng.brain.synapses.size),
        'readout_probability': {'first': probs[0], 'last': probs[-1], 'mean': float(np.mean(probs))},
        'activity_mean': float(out.mean()), 'output_finite': bool(np.all(np.isfinite(out))),
        'structural_events': int(eng.brain.ledger.total), 'checkpoint_replay_exact': replay,
        'telemetry': eng.telemetry(include_timing=False), 'evidence_tier': EVIDENCE_TIER,
    }


def _cmd_topology(args: Any) -> dict[str, Any]:
    from .topology import build_topology, topology_metrics
    neurons = _bounded('neurons', args.nw_neurons, 8, MAX_CLI_NEURONS)
    seeds = list(args.nw_seeds)
    if not seeds or len(seeds) > 16:
        raise ValueError('--seeds needs 1..16 values')
    rows: dict[str, Any] = {}
    for name in TOPOLOGIES:
        per_seed = []
        for seed in seeds:
            m = topology_metrics(build_topology(name, neurons, seed=seed))
            per_seed.append({'seed': seed, 'edges': int(m['edges']),
                             **{k: m.get(k) for k in TOPOLOGY_METRIC_KEYS if k in m}})
        mean = {}
        for k in TOPOLOGY_METRIC_KEYS + ('edges',):
            vals = [r[k] for r in per_seed if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
            mean[k] = float(np.mean(vals)) if vals else None
        rows[name] = {'per_seed': per_seed, 'mean': mean}
    return {'ok': True, 'command': 'neuroweave topology', 'neurons': neurons, 'seeds': seeds, 'topologies': rows,
            'evidence_tier': EVIDENCE_TIER}


def _seed_list(seeds: Any) -> list[int] | None:
    if not seeds:
        return None
    seeds = list(seeds)
    if len(seeds) > 16 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('--seeds needs 1..16 non-negative integers')
    return seeds


def _cmd_experiment(args: Any) -> dict[str, Any]:
    """``experiment <id|all>``: delegates to ``experiments.run_all_experiments`` (``quick`` presets included)."""
    overrides: dict[str, Any] = {}
    seeds = _seed_list(args.nw_seeds)
    if seeds is not None:
        overrides['seeds'] = tuple(seeds)
    if args.nw_steps is not None:
        overrides['steps'] = _bounded('steps', args.nw_steps, 1, MAX_CLI_STEPS)
    run_all = _lab('experiments', 'run_all_experiments')
    if args.experiment_id == 'all':
        receipts = _call(run_all, quick=bool(args.nw_quick), **overrides)
    else:
        if args.experiment_id not in EXPERIMENT_IDS:
            raise ValueError(f'unknown experiment {args.experiment_id!r}; choose from {list(EXPERIMENT_IDS)} or all')
        receipts = _call(run_all, quick=bool(args.nw_quick), experiment_ids=[args.experiment_id], **overrides)
    receipts = list(receipts)
    out: dict[str, Any] = {'ok': True, 'command': 'neuroweave experiment', 'experiment': args.experiment_id,
                           'quick': bool(args.nw_quick), 'overrides': overrides,
                           'result': receipts[0] if args.experiment_id != 'all' else receipts,
                           'evidence_tier': EVIDENCE_TIER}
    if args.nw_csv is not None:
        out['csv'] = _write(args.nw_csv, _lab('experiments', 'results_to_csv')(receipts))
    return out


LESION_QUICK = {'kinds': ('intact', 'random', 'hub'), 'fractions': (0.2,), 'seeds': (0,), 'neurons': 24,
                'steps': 24, 'recovery_steps': 8}


def _cmd_lesion(args: Any) -> dict[str, Any]:
    opts: dict[str, Any] = dict(LESION_QUICK) if args.nw_quick else {}
    seeds = _seed_list(args.nw_seeds)
    if seeds is not None:
        opts['seeds'] = tuple(seeds)
    if args.nw_steps is not None:
        opts['steps'] = _bounded('steps', args.nw_steps, 1, MAX_CLI_STEPS)
    if args.nw_neurons is not None:
        opts['neurons'] = _bounded('neurons', args.nw_neurons, 8, MAX_CLI_NEURONS)
    result = _call(_lab('lesion', 'run_lesion_assay'), **opts)
    return {'ok': True, 'command': 'neuroweave lesion', 'quick': bool(args.nw_quick), 'options': opts,
            'result': result, 'evidence_tier': EVIDENCE_TIER}


def _cmd_sequence_search(args: Any) -> dict[str, Any]:
    """``sequence-search``: supplementary budget-matched synthetic sequence-variant search assay."""
    opts: dict[str, Any] = {'quick': bool(getattr(args, 'nw_quick', False))}
    for attr, key, low, high in (('nw_landscapes', 'landscapes', 1, 50), ('nw_seq_seeds', 'seeds', 1, 32),
                                 ('nw_budget', 'budget', 13, 500)):
        value = getattr(args, attr, None)
        if value is not None:
            opts[key] = _bounded(key, value, low, high)
    result = _lab('experiments', 'run_supplementary_experiment')('sequence_search', **opts)
    return {'ok': True, 'command': 'neuroweave sequence-search', 'options': opts, 'result': result,
            'disclaimer': result.get('disclaimer'), 'evidence_tier': EVIDENCE_TIER}


def _cmd_ecology(args: Any, config: Any) -> dict[str, Any]:
    from .ecology import CONFIG_NAMES, NeuroWeaveEcology
    from .engine import NeuroWeaveEngine
    steps = _bounded('steps', args.nw_steps, 1, MAX_CLI_STEPS)
    if args.nw_eco_config not in CONFIG_NAMES:
        raise ValueError(f'--config must be one of {list(CONFIG_NAMES)}')
    spec = spec_from_cli(args)
    eng = NeuroWeaveEngine(spec)
    eng.attach_ecology(args.nw_eco_config)
    run = eng.run_ecology(steps, learn=not args.nw_frozen, config=args.nw_eco_config)
    out: dict[str, Any] = {'ok': True, 'command': 'neuroweave ecology', 'config': args.nw_eco_config,
                           'spec_digest': spec.spec_digest, 'mechanisms': len(spec.mechanisms),
                           'engine': run, 'engine_counters': dict(eng.counters), 'synapse_agent': None,
                           'evidence_tier': EVIDENCE_TIER}
    if args.nw_synapse_agent:
        from ..agent import IntegratedAgent
        from ..config import SynapseConfig
        nodes = _bounded('agent-nodes', args.nw_agent_nodes, 4, 256)
        agent_cfg = SynapseConfig(seed=spec.seed, nodes_per_source=nodes,
                                  neuroweave=getattr(config, 'neuroweave', None))
        agent = IntegratedAgent(agent_cfg)
        env = NeuroWeaveEcology(args.nw_eco_config, seed=spec.seed)
        res = env.run_synapse_agent(agent, steps, learn=False)
        out['synapse_agent'] = {'nodes_per_source': nodes,
                                'neuroweave_attached': agent_cfg.neuroweave is not None,
                                'result': res}
    return out


def _engine_for_export(args: Any):
    steps = _bounded('steps', args.nw_steps, 0, MAX_CLI_STEPS)
    spec = spec_from_cli(args)
    if steps:
        eng, _probs, _rng = _run_engine(spec, steps, spec.seed)
    else:
        from .engine import NeuroWeaveEngine
        eng = NeuroWeaveEngine(spec)
    return eng, spec, steps


def _cmd_export(args: Any) -> dict[str, Any]:
    from .package import dumps_package, validate_package
    from .standalone import annotate_package, standalone_html
    eng, spec, steps = _engine_for_export(args)
    pkg = annotate_package(eng.export_package())
    text = dumps_package(pkg)
    report = validate_package(json.loads(text))
    out: dict[str, Any] = {'ok': bool(report['valid']), 'command': 'neuroweave export', 'steps': steps,
                           'spec_digest': spec.spec_digest, 'fingerprint': pkg['integrity']['fingerprint'],
                           'counts': pkg['counts'], 'valid': report['valid'], 'errors': report['errors'][:8],
                           'package': _write(args.nw_out, text), 'html': None, 'fpga': None,
                           'raw_media_stored': False, 'evidence_tier': EVIDENCE_TIER}
    if args.nw_html is not None:
        out['html'] = _write(args.nw_html, standalone_html(pkg))
        st = pkg['capabilities']['frozen_inference']['standalone']
        out['html']['not_executed'] = [x['id'] for x in st['not_executed']] + [x['id'] for x in st['partial']]
    if args.nw_fpga is not None:
        manifest = _clean(eng.fpga_manifest())
        out['fpga'] = _write(args.nw_fpga, json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        out['fpga']['valid'] = bool(manifest['independent_validation'].get('valid'))
        out['fpga']['rtl_generated'] = manifest.get('rtl_generated')
        out['fpga']['hardware_validated'] = manifest.get('hardware_validated')
    return out


def _cmd_inspect(args: Any) -> dict[str, Any]:
    from .package import MAX_PACKAGE_BYTES, inspect_package
    path = Path(args.nw_file)
    size = path.stat().st_size
    if size > MAX_PACKAGE_BYTES:
        report = {'valid': False, 'errors': [f'package exceeds {MAX_PACKAGE_BYTES} bytes (checked before reading)'],
                  'counts': {'bytes': size}}
    else:
        report = inspect_package(path.read_bytes())
    return {'ok': bool(report.get('valid')), 'command': 'neuroweave inspect', 'path': str(path),
            'read_only': True, 'report': report}


def _cmd_import_legacy(args: Any) -> dict[str, Any]:
    from .package import (LEGACY_FORMAT, LEGACY_PROVENANCE, MAX_PACKAGE_BYTES, build_package,
                          convert_legacy_neuroweave, dumps_package, summarise_legacy, validate_package)
    try:
        legacy = _read_json_file(args.nw_file, MAX_PACKAGE_BYTES)
    except (ValueError, RecursionError) as exc:
        return {'ok': False, 'command': 'neuroweave import-legacy', 'path': str(args.nw_file),
                'errors': [f'malformed JSON: {exc}']}
    if not isinstance(legacy, dict) or legacy.get('format') != LEGACY_FORMAT:
        return {'ok': False, 'command': 'neuroweave import-legacy', 'path': str(args.nw_file),
                'errors': [f'not a {LEGACY_FORMAT} file']}
    summary = summarise_legacy(legacy)
    out: dict[str, Any] = {'ok': bool(summary['valid']), 'command': 'neuroweave import-legacy',
                           'path': str(args.nw_file), 'summary': summary, 'converted': None}
    if args.nw_out is not None and summary['valid']:
        brain = convert_legacy_neuroweave(legacy)
        pkg = build_package(brain, provenance={'source': LEGACY_PROVENANCE, 'source_file': Path(args.nw_file).name})
        text = dumps_package(pkg)
        report = validate_package(json.loads(text))
        out['converted'] = {**_write(args.nw_out, text), 'valid': report['valid'], 'errors': report['errors'][:8],
                            'fingerprint': pkg['integrity']['fingerprint'], 'conversion': brain.get('conversion')}
        out['ok'] = out['ok'] and bool(report['valid'])
    return out


def _cmd_manifest_check(_args: Any) -> dict[str, Any]:
    script = _ROOT / 'scripts' / 'neuroweave_compat_manifest.py'
    if not script.is_file():
        return {'ok': False, 'command': 'neuroweave manifest-check', 'errors': [f'{script} not found']}
    proc = subprocess.run([sys.executable, str(script), '--check'], cwd=str(_ROOT), capture_output=True,
                          text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return {'ok': proc.returncode == 0, 'command': 'neuroweave manifest-check', 'returncode': proc.returncode,
            'output': lines[-40:], 'stderr': proc.stderr.splitlines()[-10:]}


def run_neuroweave_command(args: Any, config: Any = None) -> dict[str, Any]:
    """Dispatch ``args.neuroweave_action`` and return a JSON-safe result dict."""
    from .spec import experiment_catalog, mechanism_catalog
    action = args.neuroweave_action
    if action == 'mechanisms':
        result: dict[str, Any] = {'ok': True, 'count': len(MECHANISM_IDS), 'mechanisms': mechanism_catalog()}
    elif action == 'experiments':
        result = {'ok': True, 'count': len(EXPERIMENT_IDS), 'experiments': experiment_catalog(),
                  'supplementary': _lab('experiments', 'supplementary_catalog')()}
    elif action == 'validate':
        result = _cmd_validate(args)
    elif action == 'smoke':
        result = _cmd_smoke(args)
    elif action == 'topology':
        result = _cmd_topology(args)
    elif action == 'experiment':
        result = _cmd_experiment(args)
    elif action == 'lesion':
        result = _cmd_lesion(args)
    elif action == 'ecology':
        result = _cmd_ecology(args, config)
    elif action == 'sequence-search':
        result = _cmd_sequence_search(args)
    elif action == 'export':
        result = _cmd_export(args)
    elif action == 'inspect':
        result = _cmd_inspect(args)
    elif action == 'import-legacy':
        result = _cmd_import_legacy(args)
    elif action == 'manifest-check':
        result = _cmd_manifest_check(args)
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f'unknown neuroweave action {action!r}')
    return _clean(result)
