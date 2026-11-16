"""Workbench HTTP API for the NEUROWEAVE Lab (``/api/neuroweave/*``, opt-in, additive).

The Sonapse ``Workbench`` delegates every ``/api/neuroweave/`` path to
``NeuroWeaveWorkbenchAPI``. Two engines can be observed:

* the **live** engine inside ``runtime.agent.neuroweave_channel`` (only when the
  runtime was configured with an enabled ``SynapseConfig.neuroweave``); it is
  read under the workbench runtime lock and is never stepped, reconfigured or
  fed sensory values from here, so Sonapse behaviour is unchanged;
* a standalone **lab** engine kept on the workbench, built lazily and replaced
  only by ``POST /api/neuroweave/configure``.

Experiments, lesion assays and topology sweeps are bounded (seeds/steps/neurons
capped) and can run as background jobs in the existing ``/api/jobs`` registry.
The sensory endpoint accepts only derived scalar features: payloads larger than
64 KB, any array longer than 64 items, long strings or raw-media keys are
rejected, so raw camera frames or audio buffers can never reach the server.
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .cli_commands import LESION_QUICK, _clean, _cmd_topology
from .spec import (AFFECT_VARIABLES, EVIDENCE_TIER, EXPERIMENT_IDS, MECHANISM_IDS, NEUROMODULATORS, REGIONS,
                   SENSORY_CHANNEL_IDS, SENSORY_CHANNELS, SENSORY_MODES, TOPOLOGIES, NeuroWeaveSpecV1,
                   experiment_catalog, mechanism_catalog)

__all__ = ['NeuroWeaveWorkbenchAPI', 'validate_sensory_payload', 'experiment_summary_row', 'default_lab_spec',
           'API_SCHEMA', 'MAX_SENSORY_BYTES', 'MAX_SENSORY_ARRAY', 'MAX_STEP', 'MAX_LAB_NEURONS',
           'MAX_INSPECT_BYTES', 'RAW_MEDIA_KEYS', 'LAB_SPEC_BOUNDS']

API_SCHEMA = 'NeuroWeaveWorkbenchApiV1'
MAX_SENSORY_BYTES = 64 * 1024
MAX_SENSORY_ARRAY = 64
MAX_SENSORY_STRING = 128
MAX_STEP = 256
MAX_LAB_NEURONS = 512
MAX_INSPECT_BYTES = 1_000_000
# Lab-engine cost knobs (a bounded /step must also be bounded in wall time).
LAB_SPEC_BOUNDS = {'max_synapses': (16, MAX_LAB_NEURONS * 48), 'structural_interval': (10, 100_000),
                   'sleep_interval': (10, 100_000), 'fan_in': (1, 16), 'micro_experts': (1, 8),
                   'tile_size': (4, 4096)}
MAX_EXPERIMENT_STEPS = 200
MAX_EXPERIMENT_SEEDS = 3
MAX_LESION_STEPS = 120
MAX_LESION_NEURONS = 64
MAX_TOPOLOGY_NEURONS = 256
MAX_TOPOLOGY_SEEDS = 4
MAX_SEQUENCE_LANDSCAPES = 10
MAX_SEQUENCE_SEEDS = 8
MAX_SEQUENCE_BUDGET = 200
RAW_MEDIA_KEYS = frozenset({'frame', 'frames', 'image', 'images', 'pixels', 'imagedata', 'samples', 'pcm',
                            'waveform', 'buffer', 'video', 'blob', 'data_url', 'dataurl', 'raw', 'bytes',
                            'base64', 'stream'})
_SENSORY_SECTIONS = ('visual', 'audio', 'external')


def _supplementary_catalog() -> list[dict[str, Any]]:
    from .experiments import supplementary_catalog
    return supplementary_catalog()


def default_lab_spec(seed: int = 7) -> NeuroWeaveSpecV1:
    """Small lab engine (48 neurons, every mechanism) used until ``/configure`` replaces it."""
    return NeuroWeaveSpecV1.all_mechanisms(enabled=True, seed=seed, neurons=48)


def _int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value


def _num(value: Any, name: str, low: float = -1e6, high: float = 1e6) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return v


def _seeds(value: Any, default: tuple[int, ...], cap: int) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        _int(value, 'seeds', 1, cap)
        return tuple(range(value))
    if not isinstance(value, list) or not 1 <= len(value) <= cap:
        raise ValueError(f'seeds must be a count in [1, {cap}] or a list of 1..{cap} integers')
    out = tuple(_int(s, 'seed', 0, 99) for s in value)
    if len(set(out)) != len(out):
        raise ValueError('seeds must be distinct')
    return out


# ---------------------------------------------------------------------------- sensory payload
def _scan(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str) or len(k) > 64:
                raise ValueError(f'{path}: keys must be short strings')
            if k.lower() in RAW_MEDIA_KEYS:
                raise ValueError(f'{path}.{k}: raw media is not accepted; send derived scalar features only')
            _scan(v, f'{path}.{k}')
    elif isinstance(value, list):
        if len(value) > MAX_SENSORY_ARRAY:
            raise ValueError(f'{path}: arrays longer than {MAX_SENSORY_ARRAY} are rejected (raw media is not accepted)')
        for i, v in enumerate(value):
            _scan(v, f'{path}[{i}]')
    elif isinstance(value, str):
        if len(value) > MAX_SENSORY_STRING or value.startswith('data:'):
            raise ValueError(f'{path}: long strings / data URLs are rejected (raw media is not accepted)')


def validate_sensory_payload(data: Any) -> dict[str, dict[str, float]]:
    """Validate a browser sensory payload; return ``{visual, audio, external}`` scalar dicts.

    Rejects: bodies > 64 KB, arrays > 64 items, raw-media keys, data URLs,
    unknown sections/features/channels and non-finite values.
    """
    from .sensory import AUDIO_FEATURE_KEYS, VISUAL_FEATURE_KEYS
    if not isinstance(data, Mapping):
        raise ValueError('sensory payload must be an object')
    try:
        size = len(json.dumps(data, allow_nan=False, default=str).encode('utf-8'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'sensory payload is not finite JSON: {exc}') from None
    if size > MAX_SENSORY_BYTES:
        raise ValueError(f'sensory payload is {size} bytes; the limit is {MAX_SENSORY_BYTES} (raw media is not accepted)')
    _scan(data, 'payload')
    unknown = [k for k in data if k not in _SENSORY_SECTIONS]
    if unknown:
        raise ValueError(f'unknown sensory sections {unknown}; expected any of {list(_SENSORY_SECTIONS)}')
    if not any(k in data for k in _SENSORY_SECTIONS):
        raise ValueError('sensory payload needs at least one of visual, audio, external')
    allowed = {'visual': VISUAL_FEATURE_KEYS, 'audio': AUDIO_FEATURE_KEYS, 'external': SENSORY_CHANNEL_IDS}
    out: dict[str, dict[str, float]] = {}
    for section in _SENSORY_SECTIONS:
        if section not in data:
            continue
        block = data[section]
        if not isinstance(block, Mapping) or not block:
            raise ValueError(f'{section} must be a non-empty object of scalar values')
        bad = [k for k in block if k not in allowed[section]]
        if bad:
            raise ValueError(f'unknown {section} keys {bad}; expected {list(allowed[section])}')
        out[section] = {k: _num(v, f'{section}.{k}', -1e3, 1e3) for k, v in block.items()}
    return out


# ---------------------------------------------------------------------------- experiment rows
def experiment_summary_row(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Table row: verdict plus the paired 95% CI of the comparison the verdict is based on."""
    basis = receipt.get('verdict_basis') or {}
    comp_name = basis.get('comparison')
    comp = (receipt.get('comparisons') or {}).get(comp_name) or {}
    primary = comp.get('primary') if isinstance(comp, Mapping) else None
    primary = primary if isinstance(primary, Mapping) else {}
    return _clean({
        'experiment_id': receipt.get('experiment_id'), 'kind': receipt.get('kind'), 'task': receipt.get('task'),
        'verdict': receipt.get('verdict'), 'comparison': comp_name, 'metric': basis.get('metric'),
        'section': basis.get('section'), 'mean_difference': primary.get('mean'), 'ci95': primary.get('ci95'),
        'n': primary.get('n'), 'candidate_mechanisms': receipt.get('candidate_mechanisms'),
        'digest': receipt.get('digest'), 'evidence_tier': receipt.get('evidence_tier'),
    })


# ---------------------------------------------------------------------------- API
class NeuroWeaveWorkbenchAPI:
    """Route handler for ``/api/neuroweave/*`` bound to one Sonapse ``Workbench``."""

    PREFIX = '/api/neuroweave/'

    def __init__(self, workbench: Any) -> None:
        self.wb = workbench
        self.lab_lock = threading.RLock()     # guards the lab engine (never held while taking wb.lock)
        self.run_lock = threading.Lock()      # one bounded experiment/lesion/topology run at a time
        self._lab = None
        self._lab_rng = None
        self.sensory_updates = 0
        self.last_sensory: dict[str, Any] | None = None

    # ------------------------------------------------------------ engines
    def _seed(self) -> int:
        seed = getattr(getattr(self.wb.runtime, 'config', None), 'seed', 7)
        return seed if type(seed) is int and 0 <= seed < 2**31 else 7

    def lab(self):
        """The workbench lab engine (built lazily from ``default_lab_spec``)."""
        with self.lab_lock:
            if self._lab is None:
                self._set_lab(default_lab_spec(self._seed()))
            return self._lab

    def _set_lab(self, spec: NeuroWeaveSpecV1) -> None:
        from .engine import NeuroWeaveEngine
        engine = NeuroWeaveEngine(spec)       # validates; raises before anything is swapped
        with self.lab_lock:
            self._lab = engine
            self._lab_rng = np.random.default_rng([spec.seed, 4242])

    def _live_channel(self):
        agent = getattr(self.wb.runtime, 'agent', None)
        channel = getattr(agent, 'neuroweave_channel', None)
        return channel if channel is not None and getattr(channel, 'enabled', False) else None

    def _with_engine(self, source: str, fn):
        """Run ``fn(engine, source)`` on the live engine (under the runtime lock) or the lab engine."""
        if source not in ('auto', 'live', 'lab'):
            raise ValueError("source must be 'auto', 'live' or 'lab'")
        if source in ('auto', 'live'):
            with self.wb.lock:
                channel = self._live_channel()
                if channel is not None:
                    return fn(channel.engine, 'live')
            if source == 'live':
                raise ValueError('the live runtime has no enabled NEUROWEAVE channel (start it with --neuroweave)')
        engine = self.lab()
        with self.lab_lock:
            return fn(engine, 'lab')

    # ------------------------------------------------------------ views
    @staticmethod
    def _region_connectivity(engine) -> dict[str, Any]:
        syn, region = engine.brain.synapses, engine.brain.cells.region
        k = len(REGIONS)
        counts = np.zeros((k, k), dtype=np.int64)
        weight = np.zeros((k, k))
        if syn.size:
            rs, rd = region[syn.src], region[syn.dst]
            np.add.at(counts, (rs, rd), 1)
            np.add.at(weight, (rs, rd), syn.w)
        return {'regions': [r for r, _ in REGIONS], 'counts': counts.tolist(), 'weight_sum': weight.tolist(),
                'orientation': 'rows=source region, columns=target region'}

    def _observatory(self, engine, source: str) -> dict[str, Any]:
        obs = engine.observatory()
        obs['neural']['region_connectivity'] = self._region_connectivity(engine)
        micro = engine.micro
        obs['learning']['micro']['persistence_loss'] = micro.persistence_loss()
        obs['learning']['micro']['zero_baseline_loss'] = (micro.zero_baseline_loss()
                                                         if hasattr(micro, 'zero_baseline_loss') else None)
        obs['source'] = source
        obs['spec'] = engine.spec.to_dict()
        obs['raw_media_stored'] = False
        return _clean(obs)

    @staticmethod
    def _scene(engine) -> dict[str, Any] | None:
        from .ecology import MAX_ENERGY
        eco = engine.ecology
        if eco is None:
            return None

        def rows(*arrays) -> list[list[float]]:
            if not arrays or len(arrays[0]) == 0:
                return []
            arrs = [np.asarray(a, dtype=np.float64).reshape(len(a), -1) for a in arrays]
            return np.concatenate(arrs, axis=1).tolist()

        return _clean({
            'width': eco.width, 'height': eco.height, 'config': eco.config, 'step': eco.t,
            'obstacles': rows(eco.obs_pos, eco.obs_r),
            'food': rows(eco.food_pos, eco.food_r, getattr(eco, 'food_amount', np.ones(len(eco.food_pos)))),
            'thermal': rows(eco.th_pos, eco.th_r, eco.th_sign, eco.th_intensity),
            'walls': np.asarray(eco.walls, dtype=np.float64).tolist(),
            'landmarks': rows(eco.landmarks, eco.landmark_r) if len(getattr(eco, 'landmarks', [])) else [],
            'agents': rows(eco.pos, eco.heading, eco.energy / MAX_ENERGY),
            'summary': eco.summary(), 'columns': {
                'obstacles': ['x', 'y', 'r'], 'food': ['x', 'y', 'r', 'amount'],
                'thermal': ['x', 'y', 'r', 'sign', 'intensity'], 'walls': ['x1', 'y1', 'x2', 'y2'],
                'landmarks': ['x', 'y', 'r'], 'agents': ['x', 'y', 'heading', 'energy_fraction']},
            'evidence_tier': EVIDENCE_TIER, 'raw_media_stored': False,
        })

    def catalog(self) -> dict[str, Any]:
        from .ecology import CONFIG_NAMES
        from .sensory import AUDIO_FEATURE_KEYS, VISUAL_FEATURE_KEYS
        return {
            'schema': API_SCHEMA, 'mechanisms': mechanism_catalog(), 'experiments': experiment_catalog(),
            'topologies': list(TOPOLOGIES),
            'sensory_channels': [{'id': c, 'modality': m} for c, m in SENSORY_CHANNELS],
            'sensory_modes': list(SENSORY_MODES), 'regions': [{'id': r, 'kind': k} for r, k in REGIONS],
            'neuromodulators': list(NEUROMODULATORS), 'affect_variables': list(AFFECT_VARIABLES),
            'ecology_configs': list(CONFIG_NAMES), 'visual_feature_keys': list(VISUAL_FEATURE_KEYS),
            'audio_feature_keys': list(AUDIO_FEATURE_KEYS),
            'limits': {'step': MAX_STEP, 'lab_neurons': MAX_LAB_NEURONS, 'experiment_steps': MAX_EXPERIMENT_STEPS,
                       'experiment_seeds': MAX_EXPERIMENT_SEEDS, 'lesion_steps': MAX_LESION_STEPS,
                       'lesion_neurons': MAX_LESION_NEURONS, 'topology_neurons': MAX_TOPOLOGY_NEURONS,
                       'topology_seeds': MAX_TOPOLOGY_SEEDS, 'sensory_bytes': MAX_SENSORY_BYTES,
                       'sensory_array': MAX_SENSORY_ARRAY, 'inspect_bytes': MAX_INSPECT_BYTES},
            'supplementary_experiments': _supplementary_catalog(),
            'evidence_tier': EVIDENCE_TIER,
        }

    def status(self) -> dict[str, Any]:
        with self.wb.lock:
            cfg = getattr(self.wb.runtime, 'config', None)
            spec = getattr(cfg, 'neuroweave', None)
            channel = self._live_channel()
            live = {'configured': spec is not None, 'enabled': channel is not None,
                    'spec': spec.to_dict() if spec is not None else None}
            if channel is not None:
                live.update(engine_steps=int(channel.engine_steps), effective_gain=float(channel.effective_gain()),
                            telemetry=channel.engine.telemetry(include_timing=False))
        with self.lab_lock:
            lab = None
            if self._lab is not None:
                lab = {'spec': self._lab.spec.to_dict(), 'spec_digest': self._lab.spec.spec_digest,
                       'telemetry': self._lab.telemetry(include_timing=False),
                       'ecology_attached': self._lab.ecology is not None}
        with self.wb.job_lock:
            jobs = [{'id': j['id'], 'kind': j.get('kind'), 'status': j['status']} for j in self.wb.jobs.values()
                    if str(j.get('kind', '')).startswith('neuroweave_')]
        return _clean({'schema': API_SCHEMA, 'live': live, 'lab': lab, 'lab_built': lab is not None,
                       'observed_source': 'live' if live['enabled'] else 'lab', 'run_active': self.run_lock.locked(),
                       'sensory_updates': self.sensory_updates, 'jobs': jobs, 'raw_media_stored': False,
                       'evidence_tier': EVIDENCE_TIER})

    def ledger(self, source: str = 'auto') -> dict[str, Any]:
        def view(engine, src):
            led = engine.brain.ledger
            return {'source': src, 'total': int(led.total), 'by_kind': dict(led.counts_by_kind),
                    'by_reason': dict(led.counts_by_reason), 'events': [dict(e) for e in led.events],
                    'max_events': int(led.max_events), 'step': int(engine.step_count),
                    'evidence_tier': EVIDENCE_TIER}
        return _clean(self._with_engine(source, view))

    # ------------------------------------------------------------ routes
    def get(self, path: str) -> dict[str, Any]:
        if path == '/api/neuroweave/catalog':
            return self.catalog()
        if path == '/api/neuroweave/status':
            return self.status()
        if path == '/api/neuroweave/observatory':
            return self._with_engine('auto', self._observatory)
        if path == '/api/neuroweave/observatory/lab':
            return self._with_engine('lab', self._observatory)
        if path == '/api/neuroweave/ledger':
            return self.ledger('auto')
        if path == '/api/neuroweave/ledger/lab':
            return self.ledger('lab')
        if path == '/api/neuroweave/ecology':
            engine = self.lab()
            with self.lab_lock:
                scene = self._scene(engine)
            return {'attached': scene is not None, 'scene': scene, 'source': 'lab'}
        raise KeyError('unknown NEUROWEAVE endpoint')

    def post(self, path: str, data: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise ValueError('JSON body must be an object')
        handlers = {
            '/api/neuroweave/configure': self.configure, '/api/neuroweave/step': self.step,
            '/api/neuroweave/experiment': self.experiment, '/api/neuroweave/topology': self.topology,
            '/api/neuroweave/lesion': self.lesion, '/api/neuroweave/sensory': self.sensory,
            '/api/neuroweave/export': self.export, '/api/neuroweave/inspect': self.inspect,
            '/api/neuroweave/ecology': self.ecology,
            '/api/neuroweave/sequence-search': self.sequence_search,
        }
        if path not in handlers:
            raise KeyError('unknown NEUROWEAVE endpoint')
        return handlers[path](data)

    # ------------------------------------------------------------ configure / step
    def configure(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Build/replace ONLY the lab engine. ``spec`` (full dict) and/or ``sensory_modes`` ({channel: mode})."""
        allowed = {'spec', 'sensory_modes', 'reset'}
        extra = set(data) - allowed
        if extra:
            raise ValueError(f'unknown configure fields {sorted(extra)}')
        if not any(k in data for k in allowed):
            raise ValueError('configure needs spec, sensory_modes or reset')
        try:
            if 'spec' in data:
                if not isinstance(data['spec'], Mapping):
                    raise ValueError('spec must be an object')
                spec = NeuroWeaveSpecV1.from_dict(data['spec'])
            else:
                spec = self.lab().spec
            if 'sensory_modes' in data:
                modes = data['sensory_modes']
                if not isinstance(modes, Mapping):
                    raise ValueError('sensory_modes must be an object {channel: mode}')
                merged = dict(spec.sensory_modes)
                merged.update({str(k): v for k, v in modes.items()})
                if any(not isinstance(v, str) for v in merged.values()):
                    raise ValueError('sensory modes must be strings')
                spec = NeuroWeaveSpecV1.from_dict({**spec.to_dict(), 'sensory_modes': sorted(merged.items())})
            if 'reset' in data:
                _bool(data['reset'], 'reset')
        except (TypeError, KeyError) as exc:          # KeyError would otherwise map to HTTP 404
            raise ValueError(f'invalid NEUROWEAVE spec: {exc}') from None
        if spec.neurons > MAX_LAB_NEURONS or spec.max_neurons > MAX_LAB_NEURONS:
            raise ValueError(f'the workbench lab engine is limited to {MAX_LAB_NEURONS} neurons')
        for knob, (low, high) in LAB_SPEC_BOUNDS.items():
            value = getattr(spec, knob)
            if not low <= value <= high:
                raise ValueError(f'the workbench lab engine requires {knob} in [{low}, {high}]')
        self._set_lab(spec)
        return {'configured': True, 'source': 'lab', 'spec': spec.to_dict(), 'spec_digest': spec.spec_digest,
                'live_runtime_changed': False, 'note': 'configure rebuilds the lab engine from step 0',
                'evidence_tier': EVIDENCE_TIER}

    def step(self, data: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(data) - {'steps', 'reward', 'learn', 'task_input'}
        if extra:
            raise ValueError(f'unknown step fields {sorted(extra)}')
        steps = _int(data.get('steps', 1), 'steps', 1, MAX_STEP)
        reward = _num(data.get('reward', 0.0), 'reward', -1.0, 1.0)
        learn = _bool(data.get('learn', True), 'learn')
        task = data.get('task_input')
        if task is not None:
            if not isinstance(task, list) or len(task) != 8:
                raise ValueError('task_input must be a list of 8 numbers')
            task = np.array([_num(v, 'task_input', -1.0, 1.0) for v in task])
        engine = self.lab()
        with self.lab_lock:
            probs = []
            for _ in range(steps):
                drive = task if task is not None else self._lab_rng.uniform(-1.0, 1.0, 8)
                res = engine.step(drive, None, reward=reward, learn=learn)
                probs.append(float(res['readout_probability']))
            return _clean({'source': 'lab', 'steps': steps, 'step': engine.step_count,
                           'task_input': 'given' if task is not None else 'seeded_uniform',
                           'readout_probability': {'last': probs[-1], 'mean': float(np.mean(probs))},
                           'uncertainty': res.get('uncertainty'), 'controls': res.get('controls'),
                           'telemetry': engine.telemetry(include_timing=False), 'evidence_tier': EVIDENCE_TIER})

    def ecology(self, data: Mapping[str, Any]) -> dict[str, Any]:
        from .ecology import CONFIG_NAMES
        extra = set(data) - {'config', 'steps', 'learn'}
        if extra:
            raise ValueError(f'unknown ecology fields {sorted(extra)}')
        config = data.get('config', 'default')
        if config not in CONFIG_NAMES:
            raise ValueError(f'config must be one of {list(CONFIG_NAMES)}')
        steps = _int(data.get('steps', 16), 'steps', 1, MAX_STEP)
        learn = _bool(data.get('learn', True), 'learn')
        engine = self.lab()
        with self.lab_lock:
            run = engine.run_ecology(steps, learn=learn, config=config)
            return _clean({'source': 'lab', 'run': run, 'scene': self._scene(engine), 'evidence_tier': EVIDENCE_TIER})

    # ------------------------------------------------------------ sensory
    def sensory(self, data: Mapping[str, Any]) -> dict[str, Any]:
        from .sensory import AUDIO_CHANNELS, VISUAL_CHANNELS
        clean = validate_sensory_payload(data)
        engine = self.lab()
        applied: dict[str, float] = {}
        with self.lab_lock:
            hub = engine.sensory
            if 'visual' in clean:
                feats = clean['visual']
                for cid, key in VISUAL_CHANNELS.items():
                    if key in feats:
                        hub.push_live(cid, feats[key])
                        applied[cid] = feats[key]
                hub.last_visual = dict(feats)
            if 'audio' in clean:
                feats = clean['audio']
                for cid, key in AUDIO_CHANNELS.items():
                    if key in feats:
                        hub.push_live(cid, feats[key])
                        applied[cid] = feats[key]
                hub.last_audio = dict(feats)
            for cid, value in clean.get('external', {}).items():
                hub.set_external(cid, value)
                applied[cid] = value
            modes = dict(hub.modes)
        self.sensory_updates += 1
        used = {cid: modes[cid] for cid in applied}
        self.last_sensory = {'channels': sorted(applied), 'update': self.sensory_updates}
        return _clean({'accepted': True, 'source': 'lab', 'applied': applied, 'channel_modes': used,
                       'reaches_engine': [c for c in applied
                                          if modes[c] == ('external' if c in clean.get('external', {}) else 'live')],
                       'note': "values are used on the next lab step only on channels whose mode is 'live' "
                               "(visual/audio) or 'external' (external values)",
                       'raw_media_stored': False, 'evidence_tier': EVIDENCE_TIER})

    # ------------------------------------------------------------ export / inspect
    def export(self, data: Mapping[str, Any]) -> dict[str, Any]:
        kind = data.get('kind', 'package')
        if kind not in ('package', 'html', 'fpga'):
            raise ValueError("kind must be 'package', 'html' or 'fpga'")
        source = data.get('source', 'lab')

        def build(engine, src):
            from .package import dumps_package, validate_package
            from .standalone import annotate_package, standalone_html
            step = int(engine.step_count)
            if kind == 'fpga':
                manifest = _clean(engine.fpga_manifest())
                text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
                return {'kind': kind, 'source': src, 'filename': f'neuroweave_fpga_manifest_step{step}.json',
                        'mime': 'application/json', 'text': text,
                        'valid': bool(manifest['independent_validation'].get('valid')),
                        'rtl_generated': manifest.get('rtl_generated'),
                        'hardware_validated': manifest.get('hardware_validated')}
            pkg = annotate_package(engine.export_package())
            fingerprint = pkg['integrity']['fingerprint']
            if kind == 'html':
                st = pkg['capabilities']['frozen_inference']['standalone']
                return {'kind': kind, 'source': src, 'filename': f'neuroweave_standalone_step{step}.html',
                        'mime': 'text/html', 'text': standalone_html(pkg), 'fingerprint': fingerprint,
                        'not_executed': [x['id'] for x in st['not_executed']] + [x['id'] for x in st['partial']]}
            text = dumps_package(pkg)
            report = validate_package(json.loads(text))
            return {'kind': kind, 'source': src, 'filename': f'neuroweave_package_step{step}.json',
                    'mime': 'application/json', 'text': text, 'fingerprint': fingerprint,
                    'valid': bool(report['valid']), 'errors': report['errors'][:8], 'counts': pkg['counts']}

        out = self._with_engine(source, build)
        out.update(bytes=len(out['text'].encode('utf-8')), raw_media_stored=False, evidence_tier=EVIDENCE_TIER)
        if kind == 'package':
            # the workbench inspect route reads at most MAX_INSPECT_BYTES; say so at export time
            out['inspectable_in_workbench'] = out['bytes'] <= MAX_INSPECT_BYTES
            if not out['inspectable_in_workbench']:
                out['inspect_limit_note'] = (f'package is {out["bytes"]} bytes; the workbench inspect limit is '
                                             f'{MAX_INSPECT_BYTES} bytes, so Inspect will reject this file '
                                             '(use the CLI or the Python API to inspect it)')
        return _clean(out)

    def inspect(self, data: Mapping[str, Any]) -> dict[str, Any]:
        from .package import inspect_package
        text = data.get('text')
        if not isinstance(text, str):
            raise ValueError('text must be a string holding the package JSON')
        size = len(text.encode('utf-8'))
        if size > MAX_INSPECT_BYTES:
            raise ValueError(f'package text is {size} bytes; the workbench inspect limit is {MAX_INSPECT_BYTES}')
        report = inspect_package(text)
        return _clean({'read_only': True, 'live_state_changed': False, 'bytes': size, 'report': report})

    # ------------------------------------------------------------ bounded runs
    def _run(self, kind: str, fn, background: bool, params: dict[str, Any]) -> dict[str, Any]:
        if not background:
            if not self.run_lock.acquire(blocking=False):
                raise ValueError('another NEUROWEAVE run is active; wait for it to finish')
            try:
                return _clean(fn())
            finally:
                self.run_lock.release()
        wb = self.wb
        with wb.job_lock:
            if any(j['status'] in ('queued', 'running') for j in wb.jobs.values()):
                raise ValueError('a background job is already active')
            if self.run_lock.locked():
                raise ValueError('another NEUROWEAVE run is active; wait for it to finish')
            ident = uuid.uuid4().hex
            event = threading.Event()
            wb.cancel[ident] = event
            job = {'id': ident, 'kind': f'neuroweave_{kind}', 'status': 'queued', 'created': time.time(),
                   'config': _clean(params), 'progress': {'phase': 'queued'}, 'result': None, 'error': None}
            wb.jobs[ident] = job
            wb.persist(job)

        def worker() -> None:
            try:
                with self.run_lock:
                    if event.is_set():
                        with wb.job_lock:
                            job.update(status='cancelled')
                        return
                    with wb.job_lock:
                        job.update(status='running', progress={'phase': kind})
                        wb.persist(job)
                    result = _clean(fn())
                with wb.job_lock:
                    # runs are bounded and not interruptible; a late cancel keeps the finished result
                    job.update(status='cancelled' if event.is_set() else 'completed', result=result,
                               progress={'phase': 'done'})
            except Exception as exc:
                with wb.job_lock:
                    job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                with wb.job_lock:
                    job['finished'] = time.time()
                    wb.persist(job)

        threading.Thread(target=worker, daemon=True, name=f'neuroweave-{kind}-{ident[:8]}').start()
        with wb.job_lock:
            return deepcopy(job)

    def experiment(self, data: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(data) - {'id', 'seeds', 'steps', 'quick', 'mechanisms', 'background'}
        if extra:
            raise ValueError(f'unknown experiment fields {sorted(extra)}')
        eid = data.get('id')
        if eid != 'all' and eid not in EXPERIMENT_IDS:
            raise ValueError(f'id must be one of {list(EXPERIMENT_IDS)} or all')
        quick = _bool(data.get('quick', True), 'quick')
        background = _bool(data.get('background', False), 'background')
        if eid == 'all' and not background:
            raise ValueError("id 'all' runs only as a background job (background: true)")
        seeds = _seeds(data.get('seeds'), (0, 1) if quick else (0, 1, 2), MAX_EXPERIMENT_SEEDS)
        steps = _int(data.get('steps', 24 if quick else 64), 'steps', 4, MAX_EXPERIMENT_STEPS)
        mechs = data.get('mechanisms', [])
        if not isinstance(mechs, list) or any(m not in MECHANISM_IDS for m in mechs):
            raise ValueError('mechanisms must be a list of known mechanism ids')
        overrides: dict[str, Any] = {'seeds': seeds, 'steps': steps, 'heldout_steps': max(4, steps // 2)}
        if not quick:
            overrides.update(heldout_seeds=(100, 101), neurons=48)
        if mechs:
            overrides['extra_mechanisms'] = tuple(mechs)
        ids = list(EXPERIMENT_IDS) if eid == 'all' else [eid]

        def run() -> dict[str, Any]:
            from .experiments import results_to_csv, run_all_experiments
            receipts = run_all_experiments(quick=quick, experiment_ids=ids, **overrides)
            return {'experiment': eid, 'quick': quick, 'overrides': overrides,
                    'rows': [experiment_summary_row(r) for r in receipts], 'receipts': receipts,
                    'csv': results_to_csv(receipts), 'evidence_tier': EVIDENCE_TIER}

        return self._run('experiment', run, background, {'id': eid, **overrides, 'quick': quick})

    def topology(self, data: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(data) - {'neurons', 'seeds', 'background'}
        if extra:
            raise ValueError(f'unknown topology fields {sorted(extra)}')
        neurons = _int(data.get('neurons', 48), 'neurons', 8, MAX_TOPOLOGY_NEURONS)
        seeds = _seeds(data.get('seeds'), (0,), MAX_TOPOLOGY_SEEDS)
        background = _bool(data.get('background', False), 'background')
        args = SimpleNamespace(nw_neurons=neurons, nw_seeds=list(seeds))
        return self._run('topology', lambda: _cmd_topology(args), background, {'neurons': neurons, 'seeds': seeds})

    def sequence_search(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Supplementary budget-matched synthetic sequence-variant search (never touches any engine)."""
        extra = set(data) - {'quick', 'landscapes', 'seeds', 'budget', 'background'}
        if extra:
            raise ValueError(f'unknown sequence-search fields {sorted(extra)}')
        quick = _bool(data.get('quick', True), 'quick')
        background = _bool(data.get('background', False), 'background')
        opts: dict[str, Any] = {'quick': quick}
        if 'landscapes' in data:
            opts['landscapes'] = _int(data['landscapes'], 'landscapes', 1, MAX_SEQUENCE_LANDSCAPES)
        if 'seeds' in data:
            opts['seeds'] = _int(data['seeds'], 'seeds', 1, MAX_SEQUENCE_SEEDS)
        if 'budget' in data:
            opts['budget'] = _int(data['budget'], 'budget', 13, MAX_SEQUENCE_BUDGET)

        def run() -> dict[str, Any]:
            from .experiments import run_supplementary_experiment
            result = run_supplementary_experiment('sequence_search', **opts)
            return {'options': opts, 'result': result, 'disclaimer': result['disclaimer'],
                    'evidence_tier': EVIDENCE_TIER}

        return self._run('sequence_search', run, background, dict(opts))

    def lesion(self, data: Mapping[str, Any]) -> dict[str, Any]:
        extra = set(data) - {'quick', 'seeds', 'steps', 'neurons', 'background'}
        if extra:
            raise ValueError(f'unknown lesion fields {sorted(extra)}')
        quick = _bool(data.get('quick', True), 'quick')
        background = _bool(data.get('background', False), 'background')
        opts: dict[str, Any] = dict(LESION_QUICK) if quick else {
            'fractions': (0.1, 0.2), 'seeds': (0, 1), 'neurons': 32, 'steps': 60, 'recovery_steps': 20}
        if 'seeds' in data:
            opts['seeds'] = _seeds(data['seeds'], (0,), MAX_EXPERIMENT_SEEDS)
        if 'steps' in data:
            opts['steps'] = _int(data['steps'], 'steps', 8, MAX_LESION_STEPS)
        if 'neurons' in data:
            opts['neurons'] = _int(data['neurons'], 'neurons', 16, MAX_LESION_NEURONS)
        opts['recovery_steps'] = min(int(opts['recovery_steps']), int(opts['steps']))

        def run() -> dict[str, Any]:
            from .lesion import run_lesion_assay
            return {'quick': quick, 'options': opts, 'result': run_lesion_assay(**opts),
                    'evidence_tier': EVIDENCE_TIER}

        return self._run('lesion', run, background, {'quick': quick, **opts})
