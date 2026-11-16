"""SONAPSE_NEUROWEAVE_PACKAGE: portable, validated NEUROWEAVE model packages.

This is a *separate* file format. It never alters existing Sonapse checkpoint
formats or namespaces; it wraps a duck-typed NEUROWEAVE brain-state dict plus
optional subsystem states into one JSON document with:

* a capability matrix stating honestly what a consumer can do with it;
* a SHA-256 fingerprint over the canonical JSON of everything except
  ``integrity`` (tamper/corruption detection, not authorship);
* a validator (limits, declared vs actual counts, endpoints, finite numbers,
  Dale-style sign consistency, fingerprint, privacy flags);
* a read-only inspector that checks size *before* parsing and never raises on
  malformed input;
* read-only compatibility with NEUROWEAVE-LAB ``NEUROWEAVE_MODEL_PACKAGE``
  1.0.0 / 1.1.0 / 1.2.0 (FNV-1a 32-bit fingerprint with the exact JS coverage)
  and a converter into the brain-state shape.

Nothing in this module touches live runtime state: all functions are pure
with respect to their inputs (inputs are deep-copied, never mutated).
"""
from __future__ import annotations

import copy
import csv
import decimal
import hashlib
import io
import json
import math
from typing import Any, Mapping

import numpy as np

from .spec import (EVIDENCE_TIER, MAX_POPULATION, MAX_SYNAPSES, MECHANISM_IDS,
                   NEUROWEAVE_ORIGINAL_IDS, REGIONS, SENSORY_CHANNELS, TOPOLOGIES,
                   NeuroWeaveSpecV1)

PACKAGE_FORMAT = 'SONAPSE_NEUROWEAVE_PACKAGE'
PACKAGE_VERSION = '1.0.0'
BRAIN_SCHEMA = 'NeuroWeaveBrainStateV1'
MAX_PACKAGE_BYTES = 25_000_000
MAX_JSON_DEPTH = 64
MAX_DELAY = 16
LEGACY_FORMAT = 'NEUROWEAVE_MODEL_PACKAGE'
LEGACY_VERSIONS = ('1.0.0', '1.1.0', '1.2.0')
LEGACY_PROVENANCE = 'imported_neuroweave_js'
PER_EDGE_FIELDS = frozenset({'trace', 'eligibility', 'utility', 'pre_spike', 'post_spike', 'routing_gain',
                             'myelin', 'last_signal', 'lr_scale', 'updates', 'created_step', 'compartment',
                             'last_dw', 'last_pre'})
# per-neuron arrays that NeuroWeaveBrain also stores inside its synapse table
PER_NEURON_SYNAPSE_FIELDS = frozenset({'sign', 'region', 'coact'})
COMPONENT_KEYS = ('neuromod', 'sensory', 'micro_collective', 'memory', 'prediction', 'agent',
                  'environment', 'training_history', 'experiments')
CAPABILITIES = ('frozen_inference', 'continued_learning', 'full_simulation',
                'live_sensory_operation', 'exported_memory', 'persistent_model_state')
CAPABILITY_STATUSES = ('supported', 'requires_runtime', 'unsupported')
# keys whose presence always indicates raw media was embedded (privacy invariant)
RAW_MEDIA_KEYS = frozenset({'raw_frame', 'raw_frames', 'pixels', 'raw_audio', 'audio_buffer', 'audio_samples',
                            'waveform', 'pcm', 'raw_media', 'raw_image', 'raw_video'})
# generic names that are raw media only when they hold a payload (a list or a string);
# scalar counters such as ``counters.samples`` (MicroCollective) are legitimate
RAW_MEDIA_PAYLOAD_KEYS = frozenset({'frame', 'frames', 'image', 'images', 'video', 'audio', 'samples'})
_MEDIA_URI_PREFIXES = ('data:image/', 'data:audio/', 'data:video/')
# NEUROWEAVE-LAB FEATURE_IDS, positionally aligned with the 31 original mechanism ids
LEGACY_FEATURE_IDS = ('multiSoma', 'structural', 'sparse', 'memory', 'plasticity', 'delays', 'glial',
                      'tiled', 'wiretype', 'flyHash', 'neuromod', 'neurogenesis', 'sleep', 'worldModel',
                      'myelination', 'quantized', 'axonBranches', 'fastWeights', 'selfModel',
                      'predictiveCoding', 'stdp', 'adaptiveRouting', 'hierarchicalMemory',
                      'uncertaintyGate', 'multimodalSensory', 'affectiveSignals', 'microCollective',
                      'expertRouting', 'metaLearning', 'predictiveReplay', 'wireTypeInference')
LEGACY_FEATURE_MAP = dict(zip(LEGACY_FEATURE_IDS, NEUROWEAVE_ORIGINAL_IDS))
_REGION_INDEX = {name: i for i, (name, _) in enumerate(REGIONS)}
LEGACY_REGION_MAP = {
    'visual': _REGION_INDEX['visual_sensory'], 'auditory': _REGION_INDEX['auditory_sensory'],
    'olfactory': _REGION_INDEX['chemical_somatic_sensory'], 'gustatory': _REGION_INDEX['chemical_somatic_sensory'],
    'thermosensory': _REGION_INDEX['chemical_somatic_sensory'],
    'mechanosensory': _REGION_INDEX['chemical_somatic_sensory'],
    'interoceptive': _REGION_INDEX['chemical_somatic_sensory'],
    'association': _REGION_INDEX['association_a'], 'memory': _REGION_INDEX['working_memory'],
    'motor': _REGION_INDEX['motor'],
}
# brain cell name -> (legacy path, default used by NEUROWEAVE-LAB at construction)
LEGACY_CELL_MAP: dict[str, tuple[tuple[str, ...], float]] = {
    'proximal': (('dendrites', 'proximal'), 0.0), 'distal': (('dendrites', 'distal'), 0.0),
    'apical': (('dendrites', 'apical'), 0.0), 'basal': (('dendrites', 'basal'), 0.0),
    'fast': (('fast',), 0.0), 'slow': (('slow',), 0.0), 'context': (('context',), 0.0),
    'integrator': (('soma', 'integrator'), 0.0), 'burst': (('soma', 'burst'), 0.0),
    'gating': (('soma', 'gating'), 0.0), 'membrane': (('membrane',), 0.0),
    'threshold': (('threshold',), 0.53), 'refractory': (('refractory',), 0.0), 'spike': (('spike',), 0.0),
    'calcium': (('calcium',), 0.0), 'adaptation': (('adaptation',), 0.0), 'trace': (('trace',), 0.0),
    'hillock': (('axon', 'hillock'), 0.0), 'branch_a': (('axon', 'branchA'), 0.0),
    'branch_b': (('axon', 'branchB'), 0.0), 'terminal': (('axon', 'terminal'), 0.0),
    'release': (('axon', 'release'), 0.0), 'homeo_gain': (('homeo',), 1.0),
    'astrocyte': (('astrocyte',), 1.0), 'myelin': (('myelin',), 0.55), 'utility': (('utility',), 0.0),
    'consolidation': (('consolidation',), 0.0), 'pred_estimate': (('predictive', 'estimate'), 0.0),
    'pred_error': (('predictive', 'error'), 0.0), 'surprise': (('predictive', 'surprise'), 0.0),
    'tm_fast': (('temporalMemory', 'fast'), 0.0), 'tm_mid': (('temporalMemory', 'mid'), 0.0),
    'tm_slow': (('temporalMemory', 'slow'), 0.0), 'output': (('act',), 0.0),
    # NEUROWEAVE-LAB has no primary/secondary soma; they are initialised neutral
    'primary': ((), 0.0), 'secondary': ((), 0.0),
}
LEGACY_SYNAPSE_MAP = {'w': ('weight', None), 'delay': ('delay', 1), 'trace': ('trace', 0.0),
                      'eligibility': ('eligibility', 0.0), 'utility': ('utility', 0.0),
                      'pre_spike': ('preSpikeTrace', 0.0), 'post_spike': ('postSpikeTrace', 0.0),
                      'routing_gain': ('routingGain', 1.0), 'myelin': ('myelin', 0.55)}


# =========================================================================== JSON helpers
def _to_jsonable(value: Any) -> Any:
    """Convert numpy containers/scalars to plain JSON types (copying)."""
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f'mapping keys must be strings, got {type(k).__name__}')
            out[k] = _to_jsonable(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError('non-finite number in package content')
        return f
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f'unsupported value type in package content: {type(value).__name__}')


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def package_fingerprint(pkg: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of canonical JSON of everything except ``integrity``."""
    body = {k: v for k, v in pkg.items() if k != 'integrity'}
    return hashlib.sha256(canonical_json(body).encode('utf-8')).hexdigest()


def _scan_tree(value: Any, path: str = '$') -> tuple[list[str], list[str]]:
    """Iteratively find non-finite numbers, non-JSON types and raw-media keys."""
    nonfinite: list[str] = []
    other: list[str] = []
    stack: list[tuple[Any, str, int]] = [(value, path, 0)]
    while stack:
        v, p, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            other.append(f'{p}: nesting deeper than {MAX_JSON_DEPTH}')
            continue
        if isinstance(v, str):
            if v[:12].lower().startswith(_MEDIA_URI_PREFIXES):
                other.append(f'{p}: embedded media data URI is not allowed in packages')
            continue
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                nonfinite.append(p)
            continue
        if isinstance(v, Mapping):
            for k, child in v.items():
                if not isinstance(k, str):
                    other.append(f'{p}: non-string key')
                    continue
                if k in RAW_MEDIA_KEYS or (k in RAW_MEDIA_PAYLOAD_KEYS and isinstance(child, (list, str))):
                    other.append(f'{p}.{k}: raw media field is not allowed in packages')
                if k == 'raw_media_stored' and child is not False:
                    other.append(f'{p}.{k}: raw_media_stored must be False')
                stack.append((child, f'{p}.{k}', depth + 1))
            continue
        if isinstance(v, list):
            for i, child in enumerate(v):
                stack.append((child, f'{p}[{i}]', depth + 1))
            continue
        other.append(f'{p}: unsupported type {type(v).__name__}')
    return nonfinite, other


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_integral(v: Any) -> bool:
    return _is_int(v) or (isinstance(v, float) and math.isfinite(v) and v.is_integer())


# =========================================================================== brain checks
def validate_brain_state(state: Any) -> tuple[list[str], list[str], dict[str, int]]:
    """Validate a duck-typed brain-state dict. Returns (errors, warnings, counts)."""
    errors: list[str] = []
    warnings: list[str] = []
    counts = {'neurons': 0, 'synapses': 0, 'cells': 0}
    if not isinstance(state, Mapping):
        return ['brain state must be an object'], warnings, counts
    if state.get('schema') != BRAIN_SCHEMA:
        errors.append(f'brain schema must be {BRAIN_SCHEMA!r}')
    topo = state.get('topology')
    if not isinstance(topo, Mapping):
        return errors + ['brain topology missing'], warnings, counts
    n = topo.get('n')
    if not _is_int(n) or n < 1:
        return errors + ['topology.n must be a positive integer'], warnings, counts
    counts['neurons'] = n
    if n > MAX_POPULATION:
        errors.append(f'neuron count {n} exceeds limit {MAX_POPULATION}')
    sign = topo.get('sign')
    if not isinstance(sign, list) or len(sign) != n:
        errors.append('topology.sign must be a list of length n')
        sign = None
    elif any(isinstance(s, bool) or s not in (-1, 1) for s in sign):
        errors.append('topology.sign entries must be +1 or -1')
        sign = None
    region = topo.get('region')
    if not isinstance(region, list) or len(region) != n:
        errors.append('topology.region must be a list of length n')
    elif any(not _is_int(r) or not 0 <= r < len(REGIONS) for r in region):
        errors.append('topology.region entries must index REGIONS')

    def _edges(table: Mapping[str, Any], wkey: str, label: str) -> int | None:
        src, dst, w, delay = table.get('src'), table.get('dst'), table.get(wkey), table.get('delay')
        tn = table.get('n')
        if tn is not None and tn != n:
            errors.append(f'{label}.n does not match topology.n')
        if not all(isinstance(x, list) for x in (src, dst, w)):
            errors.append(f'{label} src/dst/{wkey} must be lists')
            return None
        m = len(src)
        if len(dst) != m or len(w) != m:
            errors.append(f'{label} src/dst/{wkey} lengths differ')
            return None
        if m > MAX_SYNAPSES:
            errors.append(f'{label} synapse count {m} exceeds limit {MAX_SYNAPSES}')
        bad = sum(1 for a, b in zip(src, dst) if not (_is_int(a) and _is_int(b) and 0 <= a < n and 0 <= b < n))
        if bad:
            errors.append(f'{label}: {bad} synapse endpoint(s) out of range [0, {n})')
        if any(not _is_num(x) for x in w):
            errors.append(f'{label}: weights must be finite numbers')
        elif sign is not None and not bad:
            flips = sum(1 for a, x in zip(src, w) if x != 0 and (1 if x > 0 else -1) != sign[a])
            if flips:
                errors.append(f'{label}: {flips} weight(s) violate source sign')
        if isinstance(delay, list):
            if len(delay) != m:
                errors.append(f'{label} delay length differs')
            elif any(not _is_int(d) or not 1 <= d <= MAX_DELAY for d in delay):
                errors.append(f'{label}: delays must be integers in [1, {MAX_DELAY}]')
            elif any(d > 3 for d in delay):
                warnings.append(f'{label}: delays above 3 exceed the NEUROWEAVE 2-bit range')
        if label == 'synapses' and isinstance(table.get('sign'), list) and sign is not None \
                and list(table['sign']) != list(sign):
            errors.append('synapses.sign does not match topology.sign')
        for key, val in table.items():
            if key in ('src', 'dst', wkey, 'delay') or not isinstance(val, list):
                continue
            if key in PER_NEURON_SYNAPSE_FIELDS and label == 'synapses':
                if len(val) != n:
                    errors.append(f'{label}.{key} length {len(val)} != neuron count {n}')
                continue
            if len(val) == m:
                if key in PER_EDGE_FIELDS and any(not _is_num(x) for x in val):
                    errors.append(f'{label}.{key} must contain finite numbers')
                continue
            if key in PER_EDGE_FIELDS:
                errors.append(f'{label}.{key} length {len(val)} != synapse count {m}')
            elif all(not isinstance(x, list) for x in val):
                warnings.append(f'{label}.{key} is a flat list whose length differs from the synapse count')
        return m

    # NeuroWeaveBrain.export_state keeps edges only in ``synapses``; a topology
    # that also carries edges (the duck-typed shape) is validated as well.
    if any(k in topo for k in ('src', 'dst', 'weight')):
        _edges(topo, 'weight', 'topology')
    syn = state.get('synapses')
    if not isinstance(syn, Mapping):
        errors.append('brain synapses missing')
    else:
        m = _edges(syn, 'w', 'synapses')
        if m is not None:
            counts['synapses'] = m
    cells = state.get('cells')
    if not isinstance(cells, Mapping) or not cells:
        errors.append('brain cells missing')
    else:
        per_neuron = 0
        for name, vals in cells.items():
            if isinstance(vals, list):
                if len(vals) != n:
                    errors.append(f'cells.{name} must be a list of length n')
                elif any(not _is_num(x) for x in vals):
                    errors.append(f'cells.{name} contains non-finite values')
                else:
                    per_neuron += 1
            elif name == 'n':
                if vals != n:
                    errors.append('cells.n does not match topology.n')
            elif not (_is_num(vals) or isinstance(vals, str) or vals is None):
                errors.append(f'cells.{name} must be a per-neuron list or scalar metadata')
        counts['cells'] = per_neuron
        if not per_neuron:
            errors.append('cells must contain at least one per-neuron state array')
    readout = state.get('readout')
    if not isinstance(readout, Mapping) or not isinstance(readout.get('w'), list) or not readout['w']:
        errors.append('readout.w must be a non-empty list')
    else:
        if any(not _is_num(x) for x in readout['w']):
            errors.append('readout.w must be finite')
        if not _is_num(readout.get('b')):
            errors.append('readout.b must be finite')
    step = state.get('step')
    if not _is_int(step) or step < 0:
        errors.append('step must be a non-negative integer')
    return errors, warnings, counts


# =========================================================================== capability matrix
def _spec_object(spec: Any) -> NeuroWeaveSpecV1 | None:
    if spec is None:
        return None
    if isinstance(spec, NeuroWeaveSpecV1):
        return spec
    return NeuroWeaveSpecV1.from_dict(spec)


def _live_channels(spec_dict: Mapping[str, Any] | None, sensory: Any) -> list[str]:
    modes: dict[str, str] = {}
    if isinstance(spec_dict, Mapping) and isinstance(spec_dict.get('sensory_modes'), (list, tuple)):
        for item in spec_dict['sensory_modes']:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                modes[str(item[0])] = str(item[1])
    if isinstance(sensory, Mapping):
        extra = sensory.get('modes') or sensory.get('channel_modes')
        if isinstance(extra, Mapping):
            modes.update({str(k): str(v) for k, v in extra.items()})
    return sorted(c for c, m in modes.items() if m == 'live')


def capability_matrix(brain_state: Mapping[str, Any], components: Mapping[str, Any],
                      spec_dict: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Derive the honest capability matrix from what the package actually holds."""
    syn = brain_state.get('synapses') if isinstance(brain_state, Mapping) else None
    has_traces = isinstance(syn, Mapping) and any(isinstance(syn.get(k), list) for k in ('eligibility', 'trace'))
    out: dict[str, dict[str, Any]] = {}
    out['frozen_inference'] = {
        'status': 'supported',
        'explanation': 'Topology, synapse weights, neuron state and readout are stored; '
                       'NeuroWeaveBrain.frozen_step reproduces inference with learning disabled.',
        'requirements': ['python>=3.10', 'numpy', 'synapse.neuroweave.brain.NeuroWeaveBrain'],
    }
    if has_traces:
        out['continued_learning'] = {
            'status': 'requires_runtime',
            'explanation': 'Plasticity traces are exported; learning resumes only inside the NumPy '
                           'runtime (NeuroWeaveBrain.step with learn=True). No optimizer checkpoint is implied.',
            'requirements': ['synapse.neuroweave.brain.NeuroWeaveBrain.import_state', 'learn=True'],
        }
    else:
        out['continued_learning'] = {
            'status': 'unsupported',
            'explanation': 'Synapse plasticity traces (eligibility/trace) were not exported.',
            'requirements': [],
        }
    if 'environment' in components and 'agent' in components:
        out['full_simulation'] = {
            'status': 'requires_runtime',
            'explanation': 'Agent and ecology snapshots are included; the closed loop runs only with the '
                           'NeuroWeaveEcology and agent runtime.',
            'requirements': ['synapse.neuroweave.ecology.NeuroWeaveEcology', 'agent runtime'],
        }
    else:
        missing = [k for k in ('agent', 'environment') if k not in components]
        out['full_simulation'] = {
            'status': 'unsupported',
            'explanation': f'Package lacks the {" and ".join(missing)} snapshot(s) needed for closed-loop simulation.',
            'requirements': [],
        }
    sensory = components.get('sensory')
    live = _live_channels(spec_dict, sensory)
    if sensory is None and not live:
        out['live_sensory_operation'] = {
            'status': 'unsupported',
            'explanation': 'No sensory manifest is included.',
            'requirements': [],
        }
    else:
        kinds = {k for c, k in SENSORY_CHANNELS if c in live}
        reqs = ['synapse.neuroweave.sensory.SensoryHub']
        if 'visual' in kinds:
            reqs.append('camera permission granted by the user at runtime (frames processed locally, never stored)')
        if 'auditory' in kinds:
            reqs.append('microphone permission granted by the user at runtime (audio processed locally, never stored)')
        if not live:
            reqs.append('set a channel mode to live or supply external/simulated features')
        out['live_sensory_operation'] = {
            'status': 'requires_runtime',
            'explanation': 'Only the sensory manifest and derived features are packaged; no raw media is stored, '
                           'so live operation needs real devices at runtime.',
            'requirements': reqs,
            'live_channels': live,
        }
    memory = components.get('memory')
    if (isinstance(memory, Mapping) and memory) or (isinstance(memory, list) and memory):
        out['exported_memory'] = {
            'status': 'supported',
            'explanation': 'Memory ecology state is included and can be restored with MemoryEcology.import_state.',
            'requirements': ['synapse.neuroweave.memory.MemoryEcology'],
        }
    else:
        out['exported_memory'] = {
            'status': 'unsupported',
            'explanation': 'No memory state was exported.',
            'requirements': [],
        }
    out['persistent_model_state'] = {
        'status': 'supported',
        'explanation': 'The full brain state (cells, synapses, readout, step, ledger) round-trips through '
                       'NeuroWeaveBrain.import_state; this is a separate format from Sonapse checkpoints.',
        'requirements': ['synapse.neuroweave.brain.NeuroWeaveBrain.import_state'],
    }
    return out


# =========================================================================== build / dump
def build_package(brain_state: Mapping[str, Any], *, components: Mapping[str, Any] | None = None,
                  spec: Any = None, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble a validated ``SONAPSE_NEUROWEAVE_PACKAGE`` (inputs are not mutated)."""
    if not isinstance(brain_state, Mapping):
        raise ValueError('brain_state must be a mapping')
    try:
        brain = _to_jsonable(copy.deepcopy(dict(brain_state)))
        comps = _to_jsonable(copy.deepcopy(dict(components or {})))
        prov_in = _to_jsonable(dict(provenance or {}))
    except RecursionError as exc:
        raise ValueError(f'package content nested deeper than {MAX_JSON_DEPTH}') from exc
    errors, _warnings, counts = validate_brain_state(brain)
    if errors:
        raise ValueError('invalid brain state: ' + '; '.join(errors))
    unknown = sorted(set(comps) - set(COMPONENT_KEYS))
    if unknown:
        raise ValueError(f'unknown package components: {unknown}')
    comps = {k: v for k, v in comps.items() if v is not None}
    spec_src = spec if spec is not None else brain.get('spec')
    try:
        spec_obj = _spec_object(spec_src)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'invalid NEUROWEAVE spec: {exc}') from exc
    spec_dict = spec_obj.to_dict() if spec_obj is not None else None
    prov = {'source': 'sonapse_neuroweave', 'created_by': 'synapse.neuroweave.package',
            'synthetic_data': True, 'evidence_tier': EVIDENCE_TIER}
    prov.update(prov_in)
    prov['evidence_tier'] = EVIDENCE_TIER  # provenance may not upgrade the evidence claim
    pkg: dict[str, Any] = {
        'format': PACKAGE_FORMAT,
        'version': PACKAGE_VERSION,
        'evidence_tier': EVIDENCE_TIER,
        'spec': spec_dict,
        'mechanisms': list(spec_obj.mechanisms) if spec_obj is not None else [],
        'counts': {'neurons': counts['neurons'], 'synapses': counts['synapses']},
        'brain': brain,
        'components': comps,
        'capabilities': capability_matrix(brain, comps, spec_dict),
        'privacy': {'raw_media_stored': False},
        'provenance': prov,
        'compatibility': {'sonapse_checkpoint_format_changed': False,
                          'note': 'Separate file format; existing Sonapse checkpoints are untouched.'},
    }
    nonfinite, other = _scan_tree(pkg)
    if nonfinite or other:
        raise ValueError('package content invalid: ' + '; '.join((nonfinite + other)[:8]))
    pkg['integrity'] = {'algorithm': 'sha256', 'fingerprint': package_fingerprint(pkg)}
    return pkg


def dumps_package(pkg: Mapping[str, Any]) -> str:
    return json.dumps(pkg, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


# =========================================================================== validation
MAX_REPORT_ISSUES = 50   # a hostile package must not amplify into an unbounded error list


def _cap_issues(issues: list[str], limit: int = MAX_REPORT_ISSUES) -> list[str]:
    """Keep the first ``limit`` issues (in place) and summarise the rest in one entry."""
    if len(issues) > limit:
        extra = len(issues) - limit
        issues[:] = issues[:limit] + [f'{extra} more issue(s) truncated']
    return issues


def validate_package(obj: Any) -> dict[str, Any]:
    """Full validation report ``{valid, errors, warnings, counts}`` (never raises)."""
    try:
        report = _validate_package(obj)
        _cap_issues(report['errors'])
        _cap_issues(report['warnings'])
        return report
    except Exception as exc:  # defensive: hostile input must never escape as an exception
        return {'valid': False, 'errors': [f'package could not be validated: {type(exc).__name__}: {exc}'],
                'warnings': [], 'counts': {'neurons': 0, 'synapses': 0, 'components': [], 'bytes': 0},
                'legacy': isinstance(obj, Mapping) and obj.get('format') == LEGACY_FORMAT}


def _validate_package(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping) and obj.get('format') == LEGACY_FORMAT:
        return validate_legacy_package(obj)
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Any] = {'neurons': 0, 'synapses': 0, 'components': [], 'bytes': 0}
    report = {'valid': False, 'errors': errors, 'warnings': warnings, 'counts': counts, 'legacy': False}
    if not isinstance(obj, Mapping):
        errors.append('package must be a JSON object')
        return report
    if obj.get('format') != PACKAGE_FORMAT:
        errors.append(f'unknown package format {obj.get("format")!r}')
    if obj.get('version') != PACKAGE_VERSION:
        errors.append(f'unsupported package version {obj.get("version")!r}')
    nonfinite, other = _scan_tree(obj)
    if nonfinite:
        errors.append(f'{len(nonfinite)} non-finite number(s), first at {nonfinite[0]}')
    errors.extend(other[:16])
    try:
        size = len(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=True).encode('utf-8'))
        counts['bytes'] = size
        if size > MAX_PACKAGE_BYTES:
            errors.append(f'package size {size} exceeds {MAX_PACKAGE_BYTES} bytes')
    except (TypeError, ValueError, RecursionError):
        errors.append('package is not JSON-serialisable')
    brain = obj.get('brain')
    b_err, b_warn, b_counts = validate_brain_state(brain)
    errors.extend(b_err)
    warnings.extend(b_warn)
    counts['neurons'], counts['synapses'] = b_counts['neurons'], b_counts['synapses']
    declared = obj.get('counts')
    if not isinstance(declared, Mapping):
        errors.append('declared counts missing')
    else:
        if declared.get('neurons') != b_counts['neurons']:
            errors.append('declared neuron count does not match brain state')
        if declared.get('synapses') != b_counts['synapses']:
            errors.append('declared synapse count does not match brain state')
    comps = obj.get('components')
    if not isinstance(comps, Mapping):
        errors.append('components must be an object')
        comps = {}
    else:
        unknown = sorted(set(comps) - set(COMPONENT_KEYS))
        if unknown:
            errors.append(f'unknown components {unknown}')
        counts['components'] = sorted(comps)
    spec_dict = obj.get('spec')
    if spec_dict is not None:
        try:
            spec_obj = NeuroWeaveSpecV1.from_dict(spec_dict)
            if list(obj.get('mechanisms') or []) != list(spec_obj.mechanisms):
                errors.append('mechanism list does not match spec')
        except (ValueError, TypeError) as exc:
            errors.append(f'invalid spec: {exc}')
    mech = obj.get('mechanisms')
    if not isinstance(mech, list) or any(m not in MECHANISM_IDS for m in mech):
        errors.append('mechanisms must be a list of known mechanism ids')
    caps = obj.get('capabilities')
    if not isinstance(caps, Mapping) or set(caps) != set(CAPABILITIES):
        errors.append('capability matrix must list exactly the six capabilities')
    else:
        malformed = False
        for name, entry in caps.items():
            if not isinstance(entry, Mapping) or entry.get('status') not in CAPABILITY_STATUSES \
                    or not isinstance(entry.get('explanation'), str) or not isinstance(entry.get('requirements'), list):
                errors.append(f'capability {name} malformed')
                malformed = True
        if not malformed and not b_err and isinstance(brain, Mapping):
            expected = capability_matrix(brain, comps, spec_dict if isinstance(spec_dict, Mapping) else None)
            if {k: v['status'] for k, v in caps.items()} != {k: v['status'] for k, v in expected.items()}:
                errors.append('capability matrix does not match package contents')
    privacy = obj.get('privacy')
    if not isinstance(privacy, Mapping) or privacy.get('raw_media_stored') is not False:
        errors.append('privacy.raw_media_stored must be False')
    integ = obj.get('integrity')
    if not isinstance(integ, Mapping) or integ.get('algorithm') != 'sha256' or not isinstance(integ.get('fingerprint'), str):
        errors.append('integrity must declare a sha256 fingerprint')
    elif not nonfinite and not other:
        try:
            if package_fingerprint(obj) != integ['fingerprint']:
                errors.append('integrity fingerprint mismatch: package was modified or corrupted')
        except (TypeError, ValueError):
            errors.append('cannot compute integrity fingerprint')
    report['valid'] = not errors
    return report


# =========================================================================== legacy NEUROWEAVE
def fnv1a32(text: str) -> str:
    """FNV-1a 32-bit over UTF-16 code units (matches NEUROWEAVE-LAB ``fnv1a``)."""
    h = 0x811C9DC5
    data = text.encode('utf-16-le', 'surrogatepass')
    for i in range(0, len(data), 2):
        h ^= data[i] | (data[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f'{h:08x}'


def _js_number(x: float | int) -> str:
    """ECMAScript Number::toString for finite values (shortest round-trip digits)."""
    if isinstance(x, int) and not isinstance(x, bool):
        if abs(x) < 2 ** 53:
            return str(x)
        x = float(x)
    if not math.isfinite(x):
        return 'null'
    if x == 0:
        return '0'
    sign = '-' if x < 0 else ''
    tup = decimal.Decimal(repr(abs(x))).normalize().as_tuple()
    digits = ''.join(map(str, tup.digits))
    k = len(digits)
    n = tup.exponent + k
    if k <= n <= 21:
        body = digits + '0' * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + '.' + digits[n:]
    elif -6 < n <= 0:
        body = '0.' + '0' * (-n) + digits
    else:
        e = n - 1
        body = digits[0] + ('.' + digits[1:] if k > 1 else '') + 'e' + ('+' if e >= 0 else '-') + str(abs(e))
    return sign + body


def _js_string(s: str) -> str:
    """JSON.stringify string quoting; only *lone* surrogates are escaped (well-formed stringify).

    A Python string may hold a high/low surrogate pair as two code points; JS
    sees the same two UTF-16 units as one valid pair and emits them raw.
    """
    quoted = json.dumps(s, ensure_ascii=False)
    out: list[str] = []
    i, L = 0, len(quoted)
    while i < L:
        c = ord(quoted[i])
        if 0xD800 <= c <= 0xDBFF and i + 1 < L and 0xDC00 <= ord(quoted[i + 1]) <= 0xDFFF:
            out.append(quoted[i:i + 2])
            i += 2
            continue
        out.append('\\u%04x' % c if 0xD800 <= c <= 0xDFFF else quoted[i])
        i += 1
    return ''.join(out)


def _is_array_index(key: str) -> bool:
    """ECMAScript array-index property key (canonical uint32 string below 2**32 - 1)."""
    return key.isdigit() and key.isascii() and (key == '0' or key[0] != '0') and int(key) < 2 ** 32 - 1


def _js_key_order(keys: list[str]) -> list[str]:
    """OrdinaryOwnPropertyKeys order: array indices ascending, then insertion order."""
    idx = sorted((k for k in keys if _is_array_index(k)), key=int)
    return idx + [k for k in keys if not _is_array_index(k)]


def js_json_stringify(value: Any) -> str:
    """Compact ``JSON.stringify`` equivalent (insertion-ordered keys, JS numbers)."""
    parts: list[str] = []

    def emit(v: Any) -> None:
        if v is None:
            parts.append('null')
        elif v is True:
            parts.append('true')
        elif v is False:
            parts.append('false')
        elif isinstance(v, (int, float)):
            parts.append(_js_number(v))
        elif isinstance(v, str):
            parts.append(_js_string(v))
        elif isinstance(v, Mapping):
            parts.append('{')
            first = True
            items = {str(k): child for k, child in v.items()}
            for k in _js_key_order(list(items)):
                if not first:
                    parts.append(',')
                first = False
                parts.append(_js_string(k))
                parts.append(':')
                emit(items[k])
            parts.append('}')
        elif isinstance(v, (list, tuple)):
            parts.append('[')
            for i, child in enumerate(v):
                if i:
                    parts.append(',')
                emit(child)
            parts.append(']')
        else:
            raise ValueError(f'cannot stringify {type(v).__name__}')

    emit(value)
    return ''.join(parts)


def legacy_fingerprint(m: Mapping[str, Any]) -> str:
    """Recompute the NEUROWEAVE-LAB fingerprint with the exact schema coverage.

    1.1.0 covers config, features, network, readout, agentController; 1.2.0
    additionally covers microCollective and sensory. Absent keys are omitted,
    as ``JSON.stringify`` drops ``undefined`` members.
    """
    keys = ['config', 'features', 'network', 'readout', 'agentController']
    if m.get('formatVersion') == '1.2.0':
        keys += ['microCollective', 'sensory']
    payload = {k: m[k] for k in keys if k in m}
    return fnv1a32(js_json_stringify(payload))


def validate_legacy_package(m: Any) -> dict[str, Any]:
    """Port of NEUROWEAVE-LAB ``validateModelBundle`` (read-only, never raises)."""
    issues: list[str] = []
    counts: dict[str, Any] = {'neurons': 0, 'synapses': 0, 'components': [], 'bytes': 0}
    base: dict[str, Any] = {'valid': False, 'errors': issues, 'warnings': [], 'counts': counts, 'legacy': True}
    if not isinstance(m, Mapping):
        issues.append('Model package must be a JSON object.')
        return base
    version = m.get('formatVersion')
    base['schema'] = version
    if m.get('format') != LEGACY_FORMAT:
        issues.append('Unknown model package format.')
    if version not in LEGACY_VERSIONS:
        issues.append('Unsupported model package schema version.')
    net = m.get('network') if isinstance(m.get('network'), Mapping) else {}
    nodes, edges = net.get('neurons'), net.get('synapses')
    n = len(nodes) if isinstance(nodes, list) else 0
    if not isinstance(nodes, list) or not n:
        issues.append('Missing neuron array.')
    if not isinstance(edges, list):
        issues.append('Missing synapse array.')
    if n > MAX_POPULATION:
        issues.append(f'Neuron count exceeds the safe import limit ({MAX_POPULATION}).')
    if isinstance(edges, list) and len(edges) > MAX_SYNAPSES:
        issues.append(f'Synapse count exceeds the safe import limit ({MAX_SYNAPSES}).')
    # past the safe limits the package is already invalid: skip the per-element scans
    oversize = n > MAX_POPULATION or (isinstance(edges, list) and len(edges) > MAX_SYNAPSES)
    if not _is_integral(net.get('neuronCount')) or net.get('neuronCount') != n:
        issues.append('Declared neuron count does not match the neuron array.')
    if isinstance(edges, list) and (not _is_integral(net.get('synapseCount')) or net.get('synapseCount') != len(edges)):
        issues.append('Declared synapse count does not match the synapse array.')
    cfg = m.get('config')
    if version in ('1.1.0', '1.2.0') and (not isinstance(cfg, Mapping) or not isinstance(cfg.get('topology'), str)):
        issues.append('Schema 1.1 requires a valid configuration and topology label.')
    readout = m.get('readout') if isinstance(m.get('readout'), Mapping) else {}
    rw = readout.get('weights')
    if not isinstance(rw, list) or len(rw) != 8 or not all(_is_num(x) for x in rw):
        issues.append('Readout must contain eight finite weights.')
    if not _is_num(readout.get('bias')):
        issues.append('Readout bias must be finite.')
    feats = m.get('features')
    if not isinstance(feats, Mapping) or not all(isinstance(v, bool) for v in feats.values()):
        issues.append('Feature manifest must contain boolean flags.')
    for key, label in (('network', 'Network'), ('agentController', 'Agent controller'),
                       ('readout', 'Readout'), ('microCollective', 'Micro-Collective snapshot')):
        if m.get(key) and _scan_tree(m.get(key))[0]:
            issues.append(f'{label} contains a non-finite numeric value.')
    mc = m.get('microCollective')
    if version == '1.2.0':
        if not (isinstance(mc, Mapping) and mc.get('format') == 'NEUROWEAVE_MICRO_COLLECTIVE' and mc.get('dimension') == 16
                and mc.get('expertCount') == 4 and isinstance(mc.get('experts'), list) and len(mc['experts']) == 4
                and isinstance(mc.get('router'), list) and len(mc['router']) == 4):
            issues.append('Schema 1.2 requires a valid 16-channel, four-expert Micro-Collective snapshot.')
        if isinstance(mc, Mapping) and isinstance(mc.get('experts'), list) and any(
                not isinstance(e, Mapping) or not isinstance(e.get('w'), list) or len(e['w']) != 16
                or any(not isinstance(r, list) or len(r) != 16 for r in e['w'])
                or not isinstance(e.get('b'), list) or len(e['b']) != 16 for e in mc['experts']):
            issues.append('Micro-Collective expert matrix dimensions are invalid.')
        if isinstance(mc, Mapping) and isinstance(mc.get('router'), list) and any(
                not isinstance(r, list) or len(r) != 16 for r in mc['router']):
            issues.append('Micro-Collective router dimensions are invalid.')
        sens = m.get('sensory')
        if not (isinstance(sens, Mapping) and isinstance(sens.get('channels'), list) and len(sens['channels']) == 16
                and sens.get('rawMediaStored') is False):
            issues.append('Schema 1.2 requires a 16-channel sensory manifest and an explicit no-raw-media export flag.')
        if isinstance(nodes, list) and not oversize and any(
                not isinstance(nd, Mapping) or not nd.get('region') or not nd.get('receptors') or not nd.get('neuromodLocal')
                or not isinstance(nd.get('transmitterPrediction'), str) or not _is_num(nd.get('transmitterConfidence'))
                or not 0 <= nd['transmitterConfidence'] <= 1 for nd in nodes):
            issues.append('Schema 1.2 neuron region, receptor, neuromodulator or transmitter-hypothesis metadata is incomplete.')
    for e in edges if isinstance(edges, list) and not oversize else []:
        if not isinstance(e, Mapping):
            issues.append('Synapse endpoint out of bounds.')
            continue
        a, b = e.get('src'), e.get('dst')
        if not (_is_integral(a) and _is_integral(b) and 0 <= a < n and 0 <= b < n):
            issues.append('Synapse endpoint out of bounds.')
        w = e.get('weight')
        if not _is_num(w) or abs(w) > 0.951:
            issues.append('Synapse weight is non-finite or outside the supported bound.')
        if 'delay' in e and (not _is_integral(e['delay']) or not 1 <= e['delay'] <= 3):
            issues.append('Synapse delay must be an integer from 1 to 3.')
        for k in ('preSpikeTrace', 'postSpikeTrace'):
            if k in e and (not _is_num(e[k]) or e[k] < 0 or e[k] > 1.501):
                issues.append(f'Synapse {k} is outside its supported range.')
        if 'routingGain' in e and (not _is_num(e['routingGain']) or e['routingGain'] < 0.62 or e['routingGain'] > 1.14):
            issues.append('Synapse routing gain is outside its supported range.')
    for i, nd in enumerate(nodes if isinstance(nodes, list) and not oversize else []):
        if not isinstance(nd, Mapping):
            issues.append(f'Neuron {i} is not an object.')
            continue
        for key in ('fast', 'slow', 'context', 'trace', 'membrane', 'calcium', 'act'):
            if not _is_num(nd.get(key)):
                issues.append(f'Neuron {i} has non-finite {key}.')
        if not isinstance(nd.get('excitatory'), bool):
            issues.append(f'Neuron {i} has no valid source-sign identity.')
        for group in ('temporalMemory', 'predictive'):
            if isinstance(nd.get(group), Mapping):
                for k, v in nd[group].items():
                    if not _is_num(v):
                        issues.append(f'Neuron {i} has invalid {group} {k}.')
    integ = m.get('integrity') if isinstance(m.get('integrity'), Mapping) else {}
    fingerprint = integ.get('fingerprint') if isinstance(integ.get('fingerprint'), str) else None
    if version in ('1.1.0', '1.2.0'):
        if not fingerprint:
            issues.append('Schema 1.1 requires a payload integrity fingerprint.')
        ac = m.get('agentController')
        acn = ac.get('network') if isinstance(ac, Mapping) and isinstance(ac.get('network'), Mapping) else {}
        if not (isinstance(acn.get('neurons'), list) and acn['neurons']) or not isinstance(acn.get('synapses'), list):
            issues.append('Schema 1.1 requires an embodied-agent model snapshot.')
        if fingerprint:
            try:
                if legacy_fingerprint(m) != fingerprint:
                    issues.append('Model integrity fingerprint does not match its current payload.')
            except ValueError:
                issues.append('Model integrity fingerprint cannot be recomputed.')
    # deduplicate while keeping order (the JS validator repeats per-edge messages)
    seen: set[str] = set()
    issues[:] = [x for x in issues if not (x in seen or seen.add(x))]
    _cap_issues(issues)
    counts['neurons'] = n
    counts['synapses'] = len(edges) if isinstance(edges, list) else 0
    counts['components'] = [k for k in ('agentController', 'microCollective', 'sensory', 'training') if k in m]
    base.update({'valid': not issues, 'fingerprint': fingerprint or 'missing',
                 'legacy_schema': version == '1.0.0'})
    return base


def _legacy_get(node: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = node
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return None
        cur = cur[p]
    return cur


def convert_legacy_neuroweave(pkg: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a valid legacy package into a brain-state-shaped dict (read-only).

    Raises ``ValueError`` if the legacy package does not validate. The input is
    never mutated. Fields absent in the legacy schema are filled with the
    NEUROWEAVE-LAB construction defaults and listed in ``conversion``.
    """
    report = validate_legacy_package(pkg)
    if not report['valid']:
        raise ValueError('invalid legacy NEUROWEAVE package: ' + '; '.join(report['errors'][:6]))
    m = copy.deepcopy(dict(pkg))
    net = m['network']
    nodes, edges = net['neurons'], net['synapses']
    n = len(nodes)
    sign = [1 if nd['excitatory'] else -1 for nd in nodes]
    unknown_regions = sorted({str(nd.get('region')) for nd in nodes if nd.get('region') not in LEGACY_REGION_MAP})
    region = [LEGACY_REGION_MAP.get(nd.get('region'), _REGION_INDEX['association_a']) for nd in nodes]
    defaulted_cells: list[str] = []
    cells: dict[str, list[float]] = {}
    for name, (path, default) in LEGACY_CELL_MAP.items():
        vals = []
        missing = 0
        for nd in nodes:
            v = _legacy_get(nd, path) if path else None
            if _is_num(v):
                vals.append(float(v))
            else:
                vals.append(float(default))
                missing += 1
        if missing:
            defaulted_cells.append(name)
        cells[name] = vals
    syn: dict[str, list[Any]] = {'src': [int(e['src']) for e in edges], 'dst': [int(e['dst']) for e in edges]}
    defaulted_syn: list[str] = []
    for key, (lk, default) in LEGACY_SYNAPSE_MAP.items():
        col: list[Any] = []
        missing = 0
        for e in edges:
            v = e.get(lk)
            if _is_num(v):
                col.append(int(v) if key == 'delay' else float(v))
            else:
                col.append(default)
                missing += 1
        if missing:
            defaulted_syn.append(key)
        syn[key] = col
    # Dale-style consistency: a legacy weight whose sign contradicts its source
    # identity is clamped to zero (never flipped) and recorded.
    clamped = 0
    for i, (a, w) in enumerate(zip(syn['src'], syn['w'])):
        if w != 0 and (1 if w > 0 else -1) != sign[a]:
            syn['w'][i] = 0.0
            clamped += 1
    cfg = m.get('config') if isinstance(m.get('config'), Mapping) else {}
    topo_name = cfg.get('topology') if cfg.get('topology') in TOPOLOGIES else 'imported'
    seed = cfg.get('seed') if _is_int(cfg.get('seed')) and 0 <= cfg.get('seed') <= 2 ** 31 - 1 else 0
    feats = m.get('features') if isinstance(m.get('features'), Mapping) else {}
    mechanisms = [LEGACY_FEATURE_MAP[k] for k in LEGACY_FEATURE_IDS if feats.get(k) is True]
    spec_dict = None
    if 8 <= n <= MAX_POPULATION:
        spec_kwargs: dict[str, Any] = {'seed': seed, 'neurons': n, 'max_neurons': n,
                                       'mechanisms': tuple(mechanisms),
                                       'max_synapses': min(MAX_SYNAPSES, max(16, len(edges)))}
        if topo_name in TOPOLOGIES:
            spec_kwargs['topology'] = topo_name
        tile = cfg.get('tile')
        if _is_int(tile) and 1 <= tile <= 4096:
            spec_kwargs['tile_size'] = tile
        spar = cfg.get('sparsity')
        if _is_num(spar) and 0.01 <= spar <= 1.0:
            spec_kwargs['sparsity'] = float(spar)
        spec_dict = NeuroWeaveSpecV1(**spec_kwargs).to_dict()
    training = m.get('training') if isinstance(m.get('training'), Mapping) else {}
    step = training.get('steps') if _is_int(training.get('steps')) and training.get('steps') >= 0 else 0
    readout = m['readout']
    return {
        'schema': BRAIN_SCHEMA,
        'spec': spec_dict,
        'topology': {'n': n, 'src': list(syn['src']), 'dst': list(syn['dst']), 'weight': list(syn['w']),
                     'delay': list(syn['delay']), 'sign': sign, 'region': region,
                     'name': topo_name, 'seed': seed},
        'cells': cells,
        'synapses': syn,
        'readout': {'w': [float(x) for x in readout['weights']], 'b': float(readout['bias'])},
        'step': step,
        'ledger': {'events': [], 'imported_structural_events':
                   training.get('structuralEvents') if _is_int(training.get('structuralEvents')) else 0},
        'provenance': LEGACY_PROVENANCE,
        'conversion': {
            'source_format': LEGACY_FORMAT,
            'source_version': m.get('formatVersion'),
            'source_fingerprint': report.get('fingerprint'),
            'mechanisms': mechanisms,
            'defaulted_cells': defaulted_cells,
            'defaulted_synapse_fields': defaulted_syn,
            'sign_clamped_synapses': clamped,
            'unknown_regions': unknown_regions,
            'dropped': [k for k in ('agentController', 'microCollective') if k in m]
                       + [k for k in ('episodes', 'fastMemory', 'hashProjection', 'ensemble') if k in net],
        },
    }


def summarise_legacy(m: Any) -> dict[str, Any]:
    """Read-only summary of a legacy package (validation + mechanisms + capabilities)."""
    report = validate_legacy_package(m)
    feats = m.get('features') if isinstance(m, Mapping) and isinstance(m.get('features'), Mapping) else {}
    ok = report['valid']
    return {**report, 'format': LEGACY_FORMAT, 'version': m.get('formatVersion') if isinstance(m, Mapping) else None,
            'mechanisms': [LEGACY_FEATURE_MAP[k] for k in LEGACY_FEATURE_IDS if feats.get(k) is True],
            'capabilities': {
                'frozen_inference': 'requires_runtime' if ok else 'unsupported',
                'continued_learning': 'unsupported', 'full_simulation': 'unsupported',
                'live_sensory_operation': 'unsupported', 'exported_memory': 'unsupported',
                'persistent_model_state': 'requires_runtime' if ok else 'unsupported'},
            'fingerprint_algorithm': 'fnv1a32'}


# =========================================================================== inspection
def _reject_constant(token: str) -> Any:
    raise ValueError(f'non-standard JSON constant {token}')


def inspect_package(text: str | bytes) -> dict[str, Any]:
    """Read-only inspection of package text; never raises, never mutates state.

    The size limit is enforced before any JSON parsing; malformed JSON and
    NaN/Infinity literals are reported as errors.
    """
    report: dict[str, Any] = {'valid': False, 'errors': [], 'warnings': [], 'format': None, 'version': None,
                              'legacy': False, 'counts': {'neurons': 0, 'synapses': 0, 'components': [], 'bytes': 0},
                              'mechanisms': [], 'capabilities': {}, 'fingerprint': None,
                              'fingerprint_algorithm': None, 'parsed': False}
    if isinstance(text, (bytes, bytearray, memoryview)):
        raw = bytes(text)
        report['counts']['bytes'] = len(raw)
        if len(raw) > MAX_PACKAGE_BYTES:
            report['errors'].append(f'package exceeds {MAX_PACKAGE_BYTES} bytes (checked before parsing)')
            return report
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            report['errors'].append('package is not valid UTF-8')
            return report
    elif not isinstance(text, str):
        report['errors'].append('package must be str or bytes')
        return report
    if len(text) > MAX_PACKAGE_BYTES:
        report['counts']['bytes'] = len(text)
        report['errors'].append(f'package exceeds {MAX_PACKAGE_BYTES} characters (checked before parsing)')
        return report
    size = len(text.encode('utf-8', 'surrogatepass'))
    report['counts']['bytes'] = size
    if size > MAX_PACKAGE_BYTES:
        report['errors'].append(f'package exceeds {MAX_PACKAGE_BYTES} bytes (checked before parsing)')
        return report
    try:
        obj = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        report['errors'].append(f'malformed JSON: {exc}')
        return report
    report['parsed'] = True
    try:
        return _inspect_parsed(obj, report, size)
    except Exception as exc:  # defensive: inspection never raises
        report['valid'] = False
        report['errors'] = [f'package could not be inspected: {type(exc).__name__}: {exc}']
        return report


def _inspect_parsed(obj: Any, report: dict[str, Any], size: int) -> dict[str, Any]:
    if isinstance(obj, Mapping) and obj.get('format') == LEGACY_FORMAT:
        summ = summarise_legacy(obj)
        summ['counts']['bytes'] = size
        for k in ('valid', 'errors', 'warnings', 'counts', 'legacy', 'mechanisms', 'capabilities',
                  'fingerprint', 'fingerprint_algorithm'):
            report[k] = summ[k]
        report.update({'format': LEGACY_FORMAT, 'version': obj.get('formatVersion'),
                       'legacy_schema': summ.get('legacy_schema', False)})
        _cap_issues(report['errors'])
        _cap_issues(report['warnings'])
        return report
    val = validate_package(obj)
    val['counts']['bytes'] = size
    report.update({'valid': val['valid'], 'errors': val['errors'], 'warnings': val['warnings'],
                   'counts': val['counts']})
    if isinstance(obj, Mapping):
        report['format'] = obj.get('format') if isinstance(obj.get('format'), str) else None
        report['version'] = obj.get('version') if isinstance(obj.get('version'), str) else None
        if isinstance(obj.get('mechanisms'), list):
            report['mechanisms'] = [m for m in obj['mechanisms'] if isinstance(m, str)]
        caps = obj.get('capabilities')
        if isinstance(caps, Mapping):
            report['capabilities'] = {k: v.get('status') for k, v in caps.items() if isinstance(v, Mapping)}
        integ = obj.get('integrity')
        if isinstance(integ, Mapping):
            report['fingerprint'] = integ.get('fingerprint') if isinstance(integ.get('fingerprint'), str) else None
            report['fingerprint_algorithm'] = integ.get('algorithm') if isinstance(integ.get('algorithm'), str) else None
    return report


def loads_package(text: str | bytes) -> dict[str, Any]:
    """Parse and fully validate package text; raises ``ValueError`` if invalid."""
    report = inspect_package(text)
    if not report['valid']:
        raise ValueError('invalid package: ' + '; '.join(report['errors'][:6]))
    if isinstance(text, (bytes, bytearray, memoryview)):
        text = bytes(text).decode('utf-8')
    return json.loads(text)


# =========================================================================== CSV
_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r')


def _csv_cell(v: Any) -> str:
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError('experiment results must be finite')
        return repr(f)
    if isinstance(v, (Mapping, list, tuple, np.ndarray)):
        return canonical_json(_to_jsonable(v))
    s = str(v)
    # neutralise spreadsheet formula injection in text cells
    return "'" + s if s.startswith(_FORMULA_PREFIXES) else s


def csv_rows(experiment_results: Any) -> str:
    """Render experiment result rows as CSV text.

    Accepts a list of dicts, or a dict holding ``rows``/``results`` (list of
    dicts). Columns are the union of keys in first-seen order; nested values
    are canonical JSON; non-finite numbers raise ``ValueError``.
    """
    rows = experiment_results
    if isinstance(rows, Mapping):
        rows = rows.get('rows', rows.get('results'))
    if not isinstance(rows, (list, tuple)) or any(not isinstance(r, Mapping) for r in rows):
        raise ValueError('experiment results must be a list of mappings')
    columns: list[str] = []
    for r in rows:
        for k in r:
            if str(k) not in columns:
                columns.append(str(k))
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow([_csv_cell(c) for c in columns])  # headers are guarded against formula injection too
    for r in rows:
        lookup = {str(k): v for k, v in r.items()}
        writer.writerow([_csv_cell(lookup.get(c)) for c in columns])
    return buf.getvalue()


__all__ = [
    'PACKAGE_FORMAT', 'PACKAGE_VERSION', 'MAX_PACKAGE_BYTES', 'LEGACY_FORMAT', 'LEGACY_VERSIONS',
    'COMPONENT_KEYS', 'CAPABILITIES', 'build_package', 'dumps_package', 'loads_package', 'validate_package',
    'inspect_package', 'validate_brain_state', 'capability_matrix', 'package_fingerprint', 'canonical_json',
    'fnv1a32', 'js_json_stringify', 'legacy_fingerprint', 'validate_legacy_package', 'convert_legacy_neuroweave',
    'summarise_legacy', 'csv_rows',
]
