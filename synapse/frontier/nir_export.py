"""Neuromorphic interchange export (frontier module ``nir_export``).

Exports a NEUROWEAVE brain as a graph in the style of the Neuromorphic Intermediate Representation (NIR) of
J. E. Pedersen et al., "Neuromorphic intermediate representation: A unified instruction set for interoperable
brain-inspired computing", Nature Communications 15, 8122 (16 September 2024), doi:10.1038/s41467-024-52259-9,
using the node types and parameter names of the NIR Python package (github.com/neuromorphs/NIR; checked on
2026-11-20 against the main branch, PyPI release 1.0.8). Full, verified references:
``docs/frontier/modules/nir_export.md``.

Graph format (JSON-safe; lists, not arrays)::

    {'nodes': {name: {'type': <NIR primitive>, 'params': {field: value}}},
     'edges': [[src, dst], ...],                 # incoming edges of a node are summed element-wise (NIR)
     'metadata': {...}}                           # optional; ignored by graph equality

The export is a *lossy approximation* of the abstract NEUROWEAVE cell, and every NEUROWEAVE feature that cannot
be represented is listed (``lossy_features``). Exact parts: the linear input projection, the recurrent synapse
table split into one ``Delay`` + ``Linear`` pair per delay class (a frozen snapshot, with quantisation, region
routing, myelin gain and structural source weights folded in when those mechanisms are on), and, for isolated
cells, the synaptic current and (up to the documented adaptation current) the membrane of the ``CubaLIF`` cell,
whose parameters come from the zero-order-hold mapping ``tau = -dt / ln(decay per step)`` of the cell constants
in ``synapse/neuroweave/cells.py``. Approximated or dropped: graded ``tanh`` transmission (NIR edges carry spikes),
the direct recurrent feed-through, multiplicative reset, refractoriness, adaptive threshold, calcium feedback,
learning rules and the nonlinear mechanisms listed by :func:`unrepresentable_features`.

``write_nir`` writes a real NIR (HDF5) file with ``nir.write`` and reads it back with ``nir.read`` when the
optional ``nir`` package imports; otherwise it writes JSON tagged ``'format': 'nir-json-fallback'``. Either way it
reports whether the read-back graph equals the written one. ``simulate_graph`` is a small NumPy discrete-time
simulator for the node types used here (zero-order-hold updates; Delay nodes break recurrent loops). Nothing here
touches global random or thread state, and the exported brain is never mutated (the exporter works on a copy).
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ['MODULE_ID', 'SOURCE', 'EXPERIMENTS', 'FALLBACK_FORMAT', 'NIR_FORMAT', 'NODE_PARAMS', 'NIR_NODE_TYPES',
           'NEURON_TYPES', 'SPIKING_TYPES', 'CELL_CONSTANTS', 'cell_parameters', 'nir_status', 'brain_to_nir',
           'unrepresentable_features', 'validate_graph', 'node_io_shapes', 'canonical_graph', 'graphs_equal',
           'write_nir', 'read_nir', 'simulate_graph', 'run_experiment']

MODULE_ID = 'nir_export'
SOURCE = ('Pedersen, J. E., Abreu, S., Jobst, M., Lenz, G., Fra, V., Bauer, F. C., Muir, D. R., Zhou, P., '
          'Vogginger, B., Heckel, K., Urgese, G., Shankar, S., Stewart, T. C., Sheik, S. & Eshraghian, J. K., '
          '"Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired '
          'computing", Nature Communications 15, 8122, September 2024, doi:10.1038/s41467-024-52259-9')
FALLBACK_FORMAT = 'nir-json-fallback'
FALLBACK_VERSION = 1
NIR_FORMAT = 'nir-hdf5'
GRAPH_FORMAT = 'nir-style-graph'
_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'

# Parameter fields of the NIR primitives handled here (names as in the nir package's dataclasses).
NODE_PARAMS: dict[str, tuple[str, ...]] = {
    'Input': ('shape',),
    'Output': ('shape',),
    'Affine': ('weight', 'bias'),
    'Linear': ('weight',),
    'Scale': ('scale',),
    'Delay': ('delay',),
    'I': ('r',),
    'IF': ('r', 'v_threshold', 'v_reset'),
    'LI': ('tau', 'r', 'v_leak'),
    'LIF': ('tau', 'r', 'v_leak', 'v_threshold', 'v_reset'),
    'CubaLI': ('tau_syn', 'tau_mem', 'r', 'v_leak', 'w_in'),
    'CubaLIF': ('tau_syn', 'tau_mem', 'r', 'v_leak', 'v_threshold', 'v_reset', 'w_in'),
}
NIR_NODE_TYPES: tuple[str, ...] = tuple(NODE_PARAMS)
NEURON_TYPES: tuple[str, ...] = ('I', 'IF', 'LI', 'LIF', 'CubaLI', 'CubaLIF')
SPIKING_TYPES: tuple[str, ...] = ('IF', 'LIF', 'CubaLIF')
_POSITIVE = frozenset({'tau', 'tau_syn', 'tau_mem'})

# Constants of the NEUROWEAVE cell update (synapse/neuroweave/cells.py, CellPopulation.step); the parity checks in
# run_experiment and the tests confirm the exported dynamics against the cell code itself.
CELL_CONSTANTS: dict[str, float] = {
    'fast_decay': 0.58,          # fast = 0.58 fast + u + 0.52 eff
    'external_gain': 1.0,
    'recurrent_gain': 0.52,
    'pre_fast': 0.84,            # pre = 0.84 fast + 0.16 eff        (no multi_soma)
    'pre_direct': 0.16,
    'membrane_decay': 0.78,      # membrane = 0.78 membrane + 0.22 pre - 0.06 adaptation
    'membrane_input': 0.22,
    'adaptation_current': 0.06,
    'reset_factor': 0.26,        # membrane <- 0.26 membrane at a spike
    'refractory_steps': 2.0,
}

_MAX_NODES = 4096
_MAX_EDGES = 65536
_MAX_ERRORS = 64
_MAX_ELEMENTS = 4_000_000
_MAX_SIM_STEPS = 100_000
_MAX_DELAY_STEPS = 10_000
_INPUT_OFFSET = 7_919
_SIM_OFFSET = 7_927
_PARITY_OFFSET = 7_933


# ====================================================================== small helpers
def _int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _num(name: str, value: Any, low: float, high: float, *, open_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    v = float(value)
    if not math.isfinite(v) or v < low or v > high or (open_low and v <= low):
        raise ValueError(f'{name} must be a finite number in {"(" if open_low else "["}{low}, {high}]')
    return v


def _floats(x: Any) -> Any:
    """Nested lists of Python floats (exactly representable in JSON via repr)."""
    return np.asarray(x, dtype=np.float64).tolist()


def _tau(decay: float, dt: float) -> float:
    """Zero-order-hold time constant of a per-step decay: ``exp(-dt / tau) = decay``."""
    return -dt / math.log(decay)


# ====================================================================== optional nir package
def _import_nir() -> tuple[Any, str | None]:
    try:
        import nir  # type: ignore[import-not-found]
    except ImportError as exc:
        return None, f'ImportError: {exc}'
    except Exception as exc:  # a broken optional install must not break the module
        return None, f'{type(exc).__name__}: {exc}'
    return nir, None


def nir_status() -> dict[str, Any]:
    """Whether the optional ``nir`` package imports (and its version)."""
    nir, err = _import_nir()
    return {'available': nir is not None, 'version': None if nir is None else str(getattr(nir, '__version__', '?')),
            'error': err}


# ====================================================================== NEUROWEAVE -> graph
def cell_parameters(model: str, *, dt: float = 1.0) -> dict[str, float]:
    """Scalar NIR parameters that approximate one NEUROWEAVE cell.

    ``'CubaLIF'`` (cells with ``membrane_dynamics``): synaptic current = the fast channel, membrane = the membrane
    state; ``w_in = 1 / (1 - fast_decay)`` makes an external input of 1 add exactly 1 to the current per step.
    ``'LIF'`` (cells without it): the membrane is the fast channel and the threshold is the fast-channel level at
    which ``tanh(0.84 fast)`` crosses the base threshold; the source never resets the fast channel, so
    ``v_reset = v_threshold`` (closest constant reset)."""
    from ..neuroweave.cells import BASE_THRESHOLD
    dt = _num('dt', dt, 0.0, 1e6, open_low=True)
    k = CELL_CONSTANTS
    if model == 'CubaLIF':
        return {'tau_syn': _tau(k['fast_decay'], dt), 'tau_mem': _tau(k['membrane_decay'], dt),
                'r': k['pre_fast'] * k['membrane_input'] / (1.0 - k['membrane_decay']), 'v_leak': 0.0,
                'v_threshold': float(BASE_THRESHOLD), 'v_reset': k['reset_factor'] * float(BASE_THRESHOLD),
                'w_in': k['external_gain'] / (1.0 - k['fast_decay'])}
    if model == 'LIF':
        theta = math.atanh(float(BASE_THRESHOLD)) / k['pre_fast']
        return {'tau': _tau(k['fast_decay'], dt), 'r': k['external_gain'] / (1.0 - k['fast_decay']), 'v_leak': 0.0,
                'v_threshold': theta, 'v_reset': theta}
    raise ValueError("model must be 'CubaLIF' or 'LIF'")


def _brain_copy(source: Any) -> Any:
    """A private, validated NeuroWeaveBrain built from a brain, an engine or their exported states."""
    from ..neuroweave.brain import BRAIN_SCHEMA, NeuroWeaveBrain
    from ..neuroweave.engine import NeuroWeaveEngine
    from ..neuroweave.spec import NeuroWeaveSpecV1
    if isinstance(source, NeuroWeaveEngine):
        return source.brain.clone()
    if isinstance(source, NeuroWeaveBrain):
        return source.clone()
    if isinstance(source, Mapping):
        state = source
        if state.get('schema') != BRAIN_SCHEMA and isinstance(state.get('brain'), Mapping):
            state = state['brain']                         # an engine export carries the brain export
        if state.get('schema') != BRAIN_SCHEMA:
            raise ValueError(f'expected a NEUROWEAVE brain export (schema {BRAIN_SCHEMA!r}) or an engine export')
        try:
            spec = NeuroWeaveSpecV1.from_dict(state['spec'])
        except (KeyError, TypeError) as exc:
            raise ValueError(f'malformed brain export: {exc}') from exc
        brain = NeuroWeaveBrain(spec)
        brain.import_state(state)                         # validates fully before assigning
        return brain
    raise ValueError('source must be a NeuroWeaveBrain, a NeuroWeaveEngine or one of their exports')


def _lossy(feature: str, kind: str, treatment: str, reason: str) -> dict[str, str]:
    return {'feature': feature, 'kind': kind, 'treatment': treatment, 'reason': reason}


_NEURON_FLAG_REASONS = {
    'multi_soma': 'coupled fast/slow/context/integrator soma channels feed the spike input',
    'extended_somas': 'primary/secondary/burst soma roles and a multiplicative gating soma',
    'dendritic_compartments': 'per-compartment dendritic filters with a tanh apical gate',
    'predictive_coding': 'per-unit prediction, clipped residual and surprise trace',
    'hierarchical_temporal_memory': 'fast/medium/slow activity traces feed a temporal contrast',
    'axonal_branches': 'branch and terminal traces with myelin-scaled release feed the output',
    'homeostatic_governor': 'activity-dependent multiplicative output gain',
    'astrocyte_regulation': 'slow glial reservoir scales the output gain',
    'q88_state': 'Q8.8 fixed-point rounding of the output and several states',
    'sparse_code': 'global k-winners-take-all mask on the output',
}
_BRAIN_FLAG_REASONS = {
    'context_attention': 'softmax attention over the strongest incoming neighbours adds to the apical drive',
    'flyhash_projection': 'nonlinear top-k FlyHash code added to the input (only its linear part is exported)',
    'adaptive_routing': 'per-step gain from the destination cell state (gating, apical) scales transmission',
    'uncertainty_gating': 'runtime uncertainty input scales transmission (exported at gate 1.0, uncertainty 0)',
    'multi_region_routing': 'stored region-pair gains are folded in; runtime gains passed by the engine are not',
}
_PLASTIC_FLAGS = ('reward_plasticity', 'stdp', 'meta_plasticity', 'structural_plasticity', 'neurogenesis',
                  'sleep_consolidation', 'adaptive_conduction')


def unrepresentable_features(source: Any) -> list[dict[str, str]]:
    """Every NEUROWEAVE feature of ``source`` that the NIR-style graph cannot represent exactly.

    ``treatment`` is ``'approximated'`` (mapped to the closest NIR construct), ``'dropped'`` (absent from the
    graph) or ``'frozen_snapshot'`` (current values exported; the learning or adaptation rule is not)."""
    brain = _brain_copy(source)
    return _unrepresentable(brain)


def _unrepresentable(brain: Any) -> list[dict[str, str]]:
    from ..neuroweave.brain import CONSUMED_FLAGS
    flags = brain.flags
    out = [
        _lossy('graded_output_transmission', 'cell_model', 'approximated',
               'NEUROWEAVE synapses transmit the graded output tanh(gain * pre - shift) (+ hillock boost on spikes); '
               'NIR edges from a spiking node carry 0/1 spikes'),
        _lossy('direct_recurrent_feedthrough', 'cell_model', 'dropped',
               'pre = 0.84 fast + 0.16 recurrent drive: the 0.16 path bypasses the synaptic filter'),
        _lossy('readout_head', 'brain', 'dropped',
               'the logistic readout uses nonlinear population features (|a|, std, tanh of channels)'),
        _lossy('input_clipping', 'brain', 'dropped', 'external input is clipped to [-10, 10] before projection'),
        _lossy('control_inputs', 'cell_model', 'dropped',
               'runtime controls (gain, threshold shift, attention, plasticity scale) are exported at neutral values'),
    ]
    if 'membrane_dynamics' in flags:
        out += [
            _lossy('multiplicative_reset', 'cell_model', 'approximated',
                   'membrane <- 0.26 * membrane at a spike; NIR resets to a constant (0.26 * base threshold)'),
            _lossy('refractory_period', 'cell_model', 'dropped',
                   'two-step refractory hold (and 0.7x scaling of the output drive while held)'),
            _lossy('adaptive_threshold', 'cell_model', 'dropped',
                   'threshold = clip(0.53 + 0.18 * adaptation, 0.42, 0.78) and a -0.06 * adaptation current'),
            _lossy('calcium_feedback', 'cell_model', 'dropped', 'calcium trace feeds back into the output drive'),
        ]
    else:
        out += [
            _lossy('threshold_on_graded_output', 'cell_model', 'approximated',
                   'spikes are tanh(pre) > threshold; exported as a membrane threshold atanh(0.53) / 0.84 on the fast '
                   'channel (exact only without recurrent input)'),
            _lossy('no_reset_in_source', 'cell_model', 'approximated',
                   'the fast channel is never reset; NIR LIF resets (v_reset = v_threshold)'),
        ]
    for flag in sorted(flags):
        if flag in _NEURON_FLAG_REASONS:
            out.append(_lossy(flag, 'mechanism', 'dropped', _NEURON_FLAG_REASONS[flag]))
        elif flag in _BRAIN_FLAG_REASONS:
            treatment = 'approximated' if flag in ('multi_region_routing', 'uncertainty_gating',
                                                   'flyhash_projection') else 'dropped'
            out.append(_lossy(flag, 'mechanism', treatment, _BRAIN_FLAG_REASONS[flag]))
        elif flag in _PLASTIC_FLAGS:
            out.append(_lossy(flag, 'mechanism', 'frozen_snapshot',
                              'learning/adaptation rule; NIR has no plasticity, so the current weights, delays and '
                              'myelin gains are exported as a frozen snapshot'))
    for flag in sorted(set(brain.spec.mechanisms) - set(CONSUMED_FLAGS)):
        out.append(_lossy(flag, 'engine_mechanism', 'dropped',
                          'engine-level mechanism outside the brain graph (memory, modulation, prediction or sensing)'))
    return out


def _folded(flags: frozenset[str]) -> list[str]:
    out = ['synapse_table_snapshot']
    for flag in ('synaptic_delays', 'quantised_synapses', 'multi_region_routing', 'adaptive_conduction',
                 'structural_encoder', 'tiled_execution'):
        if flag in flags:
            out.append(flag)
    return out


def _input_projection(brain: Any, d: int) -> np.ndarray:
    """Exact linear input map (n, d) of ``brain.project_input`` without the nonlinear FlyHash term."""
    lin = brain.clone()
    lin.flags = frozenset(brain.flags - {'flyhash_projection'})
    eye = np.eye(d)
    return np.stack([lin.project_input(eye[j]) for j in range(d)], axis=1)


def _folded_weights(brain: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge transmission weights and delays of the current synapse table with the static gains folded in."""
    from ..neuroweave.connectome import dequantise_int8, quantise_int8
    syn, flags = brain.synapses, brain.flags
    w = syn.w.astype(np.float64).copy()
    if 'quantised_synapses' in flags:
        q, scale = quantise_int8(w)
        w = dequantise_int8(q, scale)
    if 'multi_region_routing' in flags:
        w = w * syn.route_gain[syn.region[syn.src], syn.region[syn.dst]]
    if 'adaptive_conduction' in flags:
        w = w * (0.85 + 0.25 * syn.myelin)
    if 'structural_encoder' in flags:
        w = w * brain._structural_weight()[syn.src]
    return w, syn.effective_delay(flags).astype(np.int64)


def brain_to_nir(source: Any, *, input_dim: int | None = None, step_ms: float = 1.0) -> dict[str, Any]:
    """NIR-style graph of a NEUROWEAVE brain (a brain, an engine, or their ``export_state()`` dicts).

    Layout: ``input`` (Input, ``input_dim``) -> ``input_projection`` (Linear) -> ``cells`` (CubaLIF with
    ``membrane_dynamics``, else LIF) -> ``output`` (Output); for every synaptic delay class ``k`` (steps)
    ``cells -> delay_k`` (Delay ``k * step_ms``) ``-> recurrent_k`` (Linear, 0.52 x folded weights) ``-> cells``.
    Time constants and delays are in ms with one engine step = ``step_ms`` ms (``metadata.dt``)."""
    from ..neuroweave.brain import BRAIN_SCHEMA
    from ..neuroweave.spec import TASK_INPUT_DIM
    d = TASK_INPUT_DIM if input_dim is None else _int('input_dim', input_dim, 1, 256)
    dt = _num('step_ms', step_ms, 0.0, 1e6, open_low=True)
    brain = _brain_copy(source)
    flags = brain.flags
    n = int(brain.n)
    model = 'CubaLIF' if 'membrane_dynamics' in flags else 'LIF'
    scalar = cell_parameters(model, dt=dt)
    P = _input_projection(brain, d)
    w, delay = _folded_weights(brain)
    syn = brain.synapses
    g = CELL_CONSTANTS['recurrent_gain']
    nodes: dict[str, Any] = {
        'input': {'type': 'Input', 'params': {'shape': [d]}},
        'input_projection': {'type': 'Linear', 'params': {'weight': _floats(P)}},
        'cells': {'type': model, 'params': {k: [float(v)] * n for k, v in scalar.items()}},
        'output': {'type': 'Output', 'params': {'shape': [n]}},
    }
    edges: list[list[str]] = [['input', 'input_projection'], ['input_projection', 'cells'], ['cells', 'output']]
    classes = sorted(int(k) for k in np.unique(delay)) if syn.size else []
    expected_nonzero = 0
    for k in classes:
        sel = delay == k
        W = np.zeros((n, n))
        np.add.at(W, (syn.dst[sel], syn.src[sel]), g * w[sel])
        expected_nonzero += int(np.count_nonzero(W))
        nodes[f'delay_{k}'] = {'type': 'Delay', 'params': {'delay': [float(k * dt)] * n}}
        nodes[f'recurrent_{k}'] = {'type': 'Linear', 'params': {'weight': _floats(W)}}
        edges += [['cells', f'delay_{k}'], [f'delay_{k}', f'recurrent_{k}'], [f'recurrent_{k}', 'cells']]
    triples = len(set(zip(syn.src.tolist(), syn.dst.tolist(), delay.tolist())))
    metadata = {
        'producer': 'synapse.frontier.nir_export', 'format': GRAPH_FORMAT, 'format_version': 1,
        'time_unit': 'ms', 'dt': dt, 'reference': SOURCE,
        'source': {'schema': BRAIN_SCHEMA, 'seed': int(brain.spec.seed), 'step': int(brain.step_count),
                   'neurons': n, 'synapses': int(syn.size), 'unique_src_dst_delay': int(triples),
                   'input_dim': d, 'mechanisms': list(brain.spec.mechanisms)},
        'neuron_model': model, 'cell_parameters': scalar, 'cell_constants': dict(CELL_CONSTANTS),
        'delay_classes_steps': classes, 'represented_synapses': int(expected_nonzero),
        'mapping': ('tau = -dt / ln(decay per step) (zero-order hold); recurrent Linear = 0.52 x folded weights '
                    '(W[dst, src], summed over duplicate edges); input Linear = exact linear part of project_input'),
        'folded_features': _folded(flags), 'lossy_features': _unrepresentable(brain),
    }
    return {'nodes': nodes, 'edges': edges, 'metadata': metadata}


# ====================================================================== validation
def _array(name: str, value: Any, errors: list[str]) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        errors.append(f'{name}: not numeric')
        return None
    if arr.size > _MAX_ELEMENTS:
        errors.append(f'{name}: more than {_MAX_ELEMENTS} elements')
        return None
    if not np.all(np.isfinite(arr)):
        errors.append(f'{name}: non-finite values')
        return None
    return arr


def _shape(name: str, value: Any, errors: list[str]) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 8:
        errors.append(f'{name}: shape must be a non-empty list of positive ints')
        return None
    out = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) < 1:
            errors.append(f'{name}: shape must be a non-empty list of positive ints')
            return None
        out.append(int(x))
    return tuple(out)


def node_io_shapes(name: str, node: Mapping[str, Any], errors: list[str] | None = None
                   ) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """``(input_shape, output_shape)`` of a node (``None`` for no input / no output); appends problems to
    ``errors`` (or raises ``ValueError`` when ``errors`` is None)."""
    own: list[str] = [] if errors is None else errors
    start = len(own)
    t = node.get('type') if isinstance(node, Mapping) else None
    params = node.get('params') if isinstance(node, Mapping) else None
    shapes: tuple[Any, Any] = (None, None)
    if not isinstance(t, str) or t not in NODE_PARAMS:
        own.append(f'{name}: unknown node type {t!r} (known: {", ".join(NIR_NODE_TYPES)})')
    elif not isinstance(params, Mapping):
        own.append(f'{name}: params must be a mapping')
    elif set(params) != set(NODE_PARAMS[t]):
        missing = sorted(set(NODE_PARAMS[t]) - set(params))
        extra = sorted(set(params) - set(NODE_PARAMS[t]))
        own.append(f'{name}: {t} params mismatch (missing {missing}, unexpected {extra})')
    elif t in ('Input', 'Output'):
        s = _shape(f'{name}.shape', params['shape'], own)
        if s is not None:
            shapes = (None, s) if t == 'Input' else (s, None)
    elif t in ('Affine', 'Linear'):
        W = _array(f'{name}.weight', params['weight'], own)
        if W is not None and (W.ndim != 2 or 0 in W.shape):
            own.append(f'{name}.weight: must be a non-empty 2-D (out, in) matrix')
        elif W is not None:
            shapes = ((W.shape[1],), (W.shape[0],))
            if t == 'Affine':
                b = _array(f'{name}.bias', params['bias'], own)
                if b is not None and b.shape != (W.shape[0],):
                    own.append(f'{name}.bias: shape {b.shape} does not match weight rows {W.shape[0]}')
    else:
        arrays = {k: _array(f'{name}.{k}', params[k], own) for k in NODE_PARAMS[t]}
        if all(a is not None for a in arrays.values()):
            ref = arrays[NODE_PARAMS[t][0]].shape
            if len(ref) != 1 or ref[0] < 1:
                own.append(f'{name}: parameters must be non-empty 1-D vectors')
            elif any(a.shape != ref for a in arrays.values()):
                own.append(f'{name}: parameter shapes differ ({ {k: a.shape for k, a in arrays.items()} })')
            else:
                if any(k in _POSITIVE and np.any(a <= 0.0) for k, a in arrays.items()):
                    own.append(f'{name}: time constants must be positive')
                elif t == 'Delay' and np.any(arrays['delay'] < 0.0):
                    own.append(f'{name}: delays must be non-negative')
                else:
                    shapes = (ref, ref)
    if errors is None and len(own) > start:
        raise ValueError('; '.join(own[start:]))
    return shapes


def validate_graph(graph: Any) -> dict[str, Any]:
    """Structural validation: known node types with the right parameter fields, finite parameters with
    consistent shapes, edge endpoints that exist, no edges into an Input or out of an Output, matching shapes
    along every edge, no duplicate edges, at least one Input and one Output. Returns a report; ``valid`` is
    False when ``errors`` is non-empty (``warnings`` flag nodes without incoming/outgoing edges)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(graph, Mapping):
        return {'valid': False, 'errors': ['graph must be a mapping with nodes and edges'], 'warnings': [],
                'node_count': 0, 'edge_count': 0, 'node_types': {}}
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if not isinstance(nodes, Mapping) or not 1 <= len(nodes) <= _MAX_NODES:
        errors.append(f'nodes must be a mapping of 1..{_MAX_NODES} named nodes')
        nodes = {}
    if not isinstance(edges, (list, tuple)) or len(edges) > _MAX_EDGES:
        errors.append(f'edges must be a list of at most {_MAX_EDGES} [src, dst] pairs')
        edges = []
    io: dict[str, tuple[Any, Any]] = {}
    for name, node in nodes.items():
        if not isinstance(name, str) or not name:
            errors.append(f'node names must be non-empty strings (got {name!r})')
            continue
        io[name] = node_io_shapes(name, node, errors)
    types = {name: node.get('type') for name, node in nodes.items() if isinstance(node, Mapping)}
    seen: set[tuple[str, str]] = set()
    indeg: Counter = Counter()
    outdeg: Counter = Counter()
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2 or not all(isinstance(x, str) for x in e):
            errors.append(f'edge {e!r} must be a [src, dst] pair of node names')
            continue
        src, dst = e
        if src not in nodes or dst not in nodes:
            errors.append(f'edge {src!r} -> {dst!r}: endpoint does not exist')
            continue
        if (src, dst) in seen:
            errors.append(f'edge {src!r} -> {dst!r}: duplicate')
            continue
        seen.add((src, dst))
        indeg[dst] += 1
        outdeg[src] += 1
        if types.get(src) == 'Output':
            errors.append(f'edge {src!r} -> {dst!r}: an Output node cannot have outgoing edges')
        if types.get(dst) == 'Input':
            errors.append(f'edge {src!r} -> {dst!r}: an Input node cannot have incoming edges')
        out_shape = io.get(src, (None, None))[1]
        in_shape = io.get(dst, (None, None))[0]
        if out_shape is not None and in_shape is not None and out_shape != in_shape:
            errors.append(f'edge {src!r} -> {dst!r}: shape {list(out_shape)} does not match input {list(in_shape)}')
    if nodes and not any(t == 'Input' for t in types.values()):
        errors.append('graph has no Input node')
    if nodes and not any(t == 'Output' for t in types.values()):
        errors.append('graph has no Output node')
    for name, t in types.items():
        if t != 'Input' and indeg[name] == 0:
            warnings.append(f'{name}: no incoming edges')
        if t != 'Output' and outdeg[name] == 0:
            warnings.append(f'{name}: no outgoing edges')
    if len(errors) > _MAX_ERRORS:
        errors = errors[:_MAX_ERRORS] + [f'... {len(errors) - _MAX_ERRORS} more errors']
    return {'valid': not errors, 'errors': errors, 'warnings': warnings[:_MAX_ERRORS], 'node_count': len(nodes),
            'edge_count': len(edges), 'node_types': dict(sorted(Counter(t for t in types.values()
                                                                        if isinstance(t, str)).items()))}


def _require_valid(graph: Any) -> None:
    report = validate_graph(graph)
    if not report['valid']:
        raise ValueError('invalid NIR-style graph: ' + '; '.join(report['errors']))


# ====================================================================== equality and serialisation
def canonical_graph(graph: Mapping[str, Any], *, include_metadata: bool = True) -> dict[str, Any]:
    """Validated, JSON-safe copy: params as nested float lists, shapes as int lists, edges as [src, dst] lists."""
    _require_valid(graph)
    nodes = {}
    for name in sorted(graph['nodes']):
        node = graph['nodes'][name]
        t = node['type']
        if t in ('Input', 'Output'):
            params = {'shape': [int(x) for x in node['params']['shape']]}
        else:
            params = {k: _floats(node['params'][k]) for k in NODE_PARAMS[t]}
        nodes[name] = {'type': t, 'params': params}
    out = {'nodes': nodes, 'edges': [[str(s), str(d)] for s, d in graph['edges']]}
    if include_metadata and isinstance(graph.get('metadata'), Mapping):
        out['metadata'] = json.loads(json.dumps(graph['metadata'], sort_keys=True, allow_nan=False))
    return out


def graphs_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Same node names, types, parameter values (exact, shape included) and edge set; metadata is ignored."""
    try:
        ca, cb = canonical_graph(a, include_metadata=False), canonical_graph(b, include_metadata=False)
    except ValueError:
        return False
    if set(ca['nodes']) != set(cb['nodes']):
        return False
    for name, na in ca['nodes'].items():
        nb = cb['nodes'][name]
        if na['type'] != nb['type']:
            return False
        for k, va in na['params'].items():
            x, y = np.asarray(va), np.asarray(nb['params'][k])
            if x.shape != y.shape or not np.array_equal(x, y):
                return False
    return sorted(map(tuple, ca['edges'])) == sorted(map(tuple, cb['edges']))


def _to_nir(graph: Mapping[str, Any], nir: Any) -> Any:
    nodes = {}
    for name, node in graph['nodes'].items():
        t, p = node['type'], node['params']
        if t == 'Input':
            nodes[name] = nir.Input(input_type=np.asarray(p['shape'], dtype=np.int64))
        elif t == 'Output':
            nodes[name] = nir.Output(output_type=np.asarray(p['shape'], dtype=np.int64))
        else:
            nodes[name] = getattr(nir, t)(**{k: np.asarray(p[k], dtype=np.float64) for k in NODE_PARAMS[t]})
    return nir.NIRGraph(nodes=nodes, edges=[(str(s), str(d)) for s, d in graph['edges']])


def _shape_of(types: Any, key: str) -> list[int]:
    value = types[key] if isinstance(types, Mapping) else types
    return [int(x) for x in np.asarray(value).ravel()]


def _from_nir(g: Any) -> dict[str, Any]:
    nodes = {}
    for name, node in g.nodes.items():
        t = type(node).__name__
        if t == 'Input':
            nodes[str(name)] = {'type': t, 'params': {'shape': _shape_of(node.input_type, 'input')}}
        elif t == 'Output':
            nodes[str(name)] = {'type': t, 'params': {'shape': _shape_of(node.output_type, 'output')}}
        elif t in NODE_PARAMS:
            nodes[str(name)] = {'type': t, 'params': {k: _floats(getattr(node, k)) for k in NODE_PARAMS[t]}}
        else:
            raise ValueError(f'NIR node {name!r} has a type this module does not handle: {t}')
    edges = [[a.decode('utf8') if isinstance(a, bytes) else str(a), b.decode('utf8') if isinstance(b, bytes) else str(b)]
             for a, b in g.edges]
    return {'nodes': nodes, 'edges': edges}


def write_nir(graph: Mapping[str, Any], path: str | Path, *, backend: str = 'auto') -> dict[str, Any]:
    """Write ``graph`` to ``path`` and read it back.

    ``backend='auto'`` writes a real NIR (HDF5) file with ``nir.write`` / ``nir.read`` when the optional ``nir``
    package imports and JSON tagged ``'format': 'nir-json-fallback'`` otherwise; ``'json'`` forces the fallback and
    ``'nir'`` requires the package. Returns ``format``, ``round_trip_equal`` (nodes, parameters and edges compared
    exactly), node/edge counts before and after, and the nir package status. NIR files carry no metadata here (the
    metadata stays in the JSON form and in the in-memory graph)."""
    if backend not in ('auto', 'json', 'nir'):
        raise ValueError("backend must be 'auto', 'json' or 'nir'")
    canon = canonical_graph(graph)                       # validates before anything is written
    p = Path(path)
    status = nir_status()
    nir = None
    if backend in ('auto', 'nir'):
        nir, _ = _import_nir()
        if nir is None and backend == 'nir':
            raise ValueError(f'the optional nir package is not importable ({status["error"]})')
    if nir is not None:
        nir.write(str(p), _to_nir(canon, nir))
        back = read_nir(p)
        fmt = NIR_FORMAT
    else:
        doc = {'format': FALLBACK_FORMAT, 'format_version': FALLBACK_VERSION, 'reference': SOURCE, 'graph': canon}
        p.write_text(json.dumps(doc, sort_keys=True, allow_nan=False, separators=(',', ':')), encoding='utf-8')
        back = read_nir(p)
        fmt = FALLBACK_FORMAT
    return {'format': fmt, 'round_trip_equal': graphs_equal(canon, back),
            'node_count': len(canon['nodes']), 'edge_count': len(canon['edges']),
            'read_node_count': len(back['nodes']), 'read_edge_count': len(back['edges']),
            'nir': status}


def read_nir(path: str | Path) -> dict[str, Any]:
    """Read a JSON fallback file, or a real NIR file when the optional ``nir`` package imports; returns a
    validated graph dict (JSON fallback files keep their metadata)."""
    p = Path(path)
    with p.open('rb') as fh:
        head = fh.read(8)
    if head == _HDF5_MAGIC:
        nir, err = _import_nir()
        if nir is None:
            raise ValueError(f'{p.name} is an HDF5 (NIR) file but the optional nir package is not importable ({err})')
        graph = _from_nir(nir.read(str(p)))
    else:
        try:
            doc = json.loads(p.read_text(encoding='utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'{p.name} is neither an HDF5 NIR file nor JSON') from exc
        if not isinstance(doc, Mapping) or doc.get('format') != FALLBACK_FORMAT:
            raise ValueError(f"JSON file is not tagged 'format': {FALLBACK_FORMAT!r}")
        if doc.get('format_version') != FALLBACK_VERSION or not isinstance(doc.get('graph'), Mapping):
            raise ValueError('unsupported nir-json-fallback version or missing graph')
        graph = dict(doc['graph'])
    _require_valid(graph)
    return graph


# ====================================================================== discrete-time simulation
def simulate_graph(graph: Mapping[str, Any], inputs: Any, *, dt: float | None = None, record_states: bool = False,
                   max_steps: int = _MAX_SIM_STEPS) -> dict[str, Any]:
    """Simulate a validated graph in discrete time with step ``dt`` (default ``metadata.dt`` or 1).

    Zero-order-hold updates: ``a = exp(-dt / tau)``; LI ``v <- a v + (1 - a)(v_leak + r x)``; CubaLI/CubaLIF
    ``i <- a_s i + (1 - a_s) w_in x`` then ``v <- a_m v + (1 - a_m)(v_leak + r i)``; I/IF ``v <- v + dt r x``;
    spiking types emit ``v > v_threshold`` and reset to ``v_reset``. Initial ``v = v_leak`` (0 for I/IF), ``i = 0``.
    Incoming edges are summed; Delay ``d`` outputs its input from ``round(d / dt)`` steps earlier (zeros before
    that; ``d`` must be a multiple of ``dt``), so Delay nodes with ``d >= dt`` break recurrent loops; any loop
    without one raises ``ValueError``. ``inputs`` is an array (T, k) for a single Input node or a mapping
    {input name: array (T, k)}. Returns per-Output arrays (T, k), spike counts per spiking node, ``finite`` and,
    with ``record_states``, neuron states ``v`` (and ``i``) of shape (T, k)."""
    _require_valid(graph)
    max_steps = _int('max_steps', max_steps, 1, _MAX_SIM_STEPS)
    meta = graph.get('metadata') if isinstance(graph.get('metadata'), Mapping) else {}
    step = _num('dt', meta.get('dt', 1.0) if dt is None else dt, 0.0, 1e6, open_low=True)
    nodes = {name: {'type': nd['type'], 'p': {k: np.asarray(v, dtype=np.float64) for k, v in nd['params'].items()}}
             for name, nd in graph['nodes'].items()}
    in_names = [n for n, nd in nodes.items() if nd['type'] == 'Input']
    if isinstance(inputs, Mapping):
        if set(inputs) != set(in_names):
            raise ValueError(f'inputs must provide exactly the Input nodes {sorted(in_names)}')
        X = dict(inputs)
    else:
        if len(in_names) != 1:
            raise ValueError('a bare input array needs exactly one Input node; pass a mapping instead')
        X = {in_names[0]: inputs}
    T = None
    for name in in_names:
        try:
            arr = np.asarray(X[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'inputs[{name!r}] must be numeric') from exc
        shape = tuple(int(x) for x in nodes[name]['p']['shape'])
        if len(shape) != 1 or arr.ndim != 2 or arr.shape[1] != shape[0]:
            raise ValueError(f'inputs[{name!r}] must have shape (T, {shape[0] if len(shape) == 1 else shape})')
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'inputs[{name!r}] must be finite')
        if T is not None and arr.shape[0] != T:
            raise ValueError('all inputs must have the same number of steps')
        T = int(arr.shape[0])
        X[name] = arr
    if T is None or not 1 <= T <= max_steps:
        raise ValueError(f'inputs must have 1..{max_steps} steps')
    for name, nd in nodes.items():
        if nd['type'] in ('Input', 'Output') and len(nd['p']['shape']) != 1:
            raise ValueError(f'{name}: the simulator handles 1-D shapes only')

    preds: dict[str, list[str]] = {name: [] for name in nodes}
    for s, d in graph['edges']:
        preds[d].append(s)
    lag: dict[str, np.ndarray] = {}
    for name, nd in nodes.items():
        if nd['type'] == 'Delay':
            q = nd['p']['delay'] / step
            k = np.rint(q)
            if np.any(np.abs(q - k) > 1e-9 * np.maximum(1.0, np.abs(q))) or np.any(k > _MAX_DELAY_STEPS):
                raise ValueError(f'{name}: delays must be multiples of dt up to {_MAX_DELAY_STEPS} steps')
            lag[name] = k.astype(np.int64)
    breaking = {name for name, k in lag.items() if k.min() >= 1}
    deps = {name: [s for s in preds[name]] for name in nodes if name not in breaking}
    order: list[str] = []
    state_mark: dict[str, int] = {}

    def visit(name: str, stack: list[str]) -> None:
        mark = state_mark.get(name, 0)
        if mark == 2:
            return
        if mark == 1:
            raise ValueError('algebraic loop without a delay: ' + ' -> '.join(stack + [name]))
        state_mark[name] = 1
        for s in deps.get(name, []):
            if s not in breaking:
                visit(s, stack + [name])
        state_mark[name] = 2
        order.append(name)

    for name in sorted(nodes):
        if name not in breaking:
            visit(name, [])

    out_dim = {name: (int(nd['p']['weight'].shape[0]) if nd['type'] in ('Affine', 'Linear')
                      else int(nd['p']['shape'][0]) if nd['type'] in ('Input', 'Output')
                      else int(next(iter(nd['p'].values())).shape[0])) for name, nd in nodes.items()}
    v: dict[str, np.ndarray] = {}
    cur: dict[str, np.ndarray] = {}
    decay: dict[str, dict[str, np.ndarray]] = {}
    for name, nd in nodes.items():
        t, p = nd['type'], nd['p']
        if t in ('LI', 'LIF'):
            v[name] = p['v_leak'].copy()
            decay[name] = {'m': np.exp(-step / p['tau'])}
        elif t in ('CubaLI', 'CubaLIF'):
            v[name] = p['v_leak'].copy()
            cur[name] = np.zeros_like(p['v_leak'])
            decay[name] = {'s': np.exp(-step / p['tau_syn']), 'm': np.exp(-step / p['tau_mem'])}
        elif t in ('I', 'IF'):
            v[name] = np.zeros_like(p['r'])
    hist = {name: np.zeros((int(k.max()) + 1, out_dim[name])) for name, k in lag.items()}
    outputs = {name: np.zeros((T, out_dim[name])) for name, nd in nodes.items() if nd['type'] == 'Output'}
    spikes = {name: 0 for name, nd in nodes.items() if nd['type'] in SPIKING_TYPES}
    states: dict[str, dict[str, np.ndarray]] = {}
    if record_states:
        for name, nd in nodes.items():
            if nd['type'] in NEURON_TYPES:
                states[name] = {'v': np.zeros((T, out_dim[name]))}
                if name in cur:
                    states[name]['i'] = np.zeros((T, out_dim[name]))
    finite = True
    chan = {name: np.arange(out_dim[name]) for name in lag}
    for t in range(T):
        val: dict[str, np.ndarray] = {}

        def delayed(name: str) -> np.ndarray:
            k = lag[name]
            idx = (t - k) % hist[name].shape[0]
            got = hist[name][idx, chan[name]]
            return np.where(t - k >= 0, got, 0.0)

        for name in breaking:
            val[name] = delayed(name)
        for name in order:
            nd = nodes[name]
            t_, p = nd['type'], nd['p']
            if t_ == 'Input':
                val[name] = X[name][t]
                continue
            x = np.zeros(out_dim[name] if t_ not in ('Affine', 'Linear') else p['weight'].shape[1])
            for s in preds[name]:
                x = x + val[s]
            if t_ == 'Output':
                outputs[name][t] = x
                val[name] = x
            elif t_ == 'Linear':
                val[name] = p['weight'] @ x
            elif t_ == 'Affine':
                val[name] = p['weight'] @ x + p['bias']
            elif t_ == 'Scale':
                val[name] = p['scale'] * x
            elif t_ == 'Delay':                            # some channel has zero delay: pass those through
                k = lag[name]
                idx = (t - k) % hist[name].shape[0]
                old = np.where(t - k >= 0, hist[name][idx, chan[name]], 0.0)
                val[name] = np.where(k == 0, x, old)
            else:
                if t_ in ('LI', 'LIF'):
                    a = decay[name]['m']
                    v[name] = a * v[name] + (1.0 - a) * (p['v_leak'] + p['r'] * x)
                elif t_ in ('CubaLI', 'CubaLIF'):
                    a_s, a_m = decay[name]['s'], decay[name]['m']
                    cur[name] = a_s * cur[name] + (1.0 - a_s) * p['w_in'] * x
                    v[name] = a_m * v[name] + (1.0 - a_m) * (p['v_leak'] + p['r'] * cur[name])
                else:                                      # I, IF
                    v[name] = v[name] + step * p['r'] * x
                if record_states:
                    states[name]['v'][t] = v[name]
                    if name in cur:
                        states[name]['i'][t] = cur[name]
                if t_ in SPIKING_TYPES:
                    # Thresholding can turn even an infinite potential into a
                    # finite spike and reset. Check before discarding the state.
                    if finite and not np.all(np.isfinite(v[name])):
                        finite = False
                    z = v[name] > p['v_threshold']
                    v[name] = np.where(z, p['v_reset'], v[name])
                    spikes[name] += int(z.sum())
                    val[name] = z.astype(np.float64)
                else:
                    val[name] = v[name].copy()
        for name in lag:
            x = np.zeros(out_dim[name])
            for s in preds[name]:
                x = x + val[s]
            hist[name][t % hist[name].shape[0]] = x
        if finite and not all(np.all(np.isfinite(a)) for a in val.values()):
            finite = False
    result: dict[str, Any] = {'steps': T, 'dt': step, 'outputs': outputs, 'spike_counts': spikes, 'finite': finite,
                              'order': order, 'delay_breaks': sorted(breaking)}
    if record_states:
        result['states'] = states
    return result


# ====================================================================== validation experiment
_DEFAULTS: dict[str, Any] = {
    'neurons': 48, 'topology': 'clustered',
    'mechanisms': ('membrane_dynamics', 'synaptic_delays', 'quantised_synapses', 'stdp'),
    'warmup_steps': 100, 'simulate_steps': 200, 'input_dim': 8, 'input_scale': 1.0, 'step_ms': 1.0,
    'parity_cells': 16, 'parity_steps': 200, 'parity_drive_max': 0.6, 'parity_tolerance': 1e-9, 'backend': 'auto',
}
_QUICK: dict[str, Any] = {'neurons': 24, 'warmup_steps': 30, 'simulate_steps': 80, 'parity_cells': 8,
                          'parity_steps': 80}
_LOSSY_CONNECTOME = frozenset({'adaptive_routing', 'uncertainty_gating', 'context_attention'})


def _seeds(seeds: Any) -> tuple[int, ...]:
    try:
        out = tuple(_int('seed', s, 0, 2**31 - 1) for s in seeds)
    except TypeError as exc:
        raise ValueError('seeds must be an iterable of ints') from exc
    if not out or len(set(out)) != len(out):
        raise ValueError('seeds must be non-empty and unique')
    return out


def _validate_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from ..neuroweave.spec import MECHANISMS, TOPOLOGIES
    c = dict(cfg)
    c['neurons'] = _int('neurons', c['neurons'], 4, 512)
    if c['topology'] not in TOPOLOGIES:
        raise ValueError(f'topology must be one of {TOPOLOGIES}')
    if isinstance(c['mechanisms'], str):
        raise ValueError('mechanisms must be a sequence of mechanism ids')
    mechs = tuple(c['mechanisms'])
    known = {m.id for m in MECHANISMS}
    if any(m not in known for m in mechs) or len(set(mechs)) != len(mechs):
        raise ValueError('mechanisms must be unique NEUROWEAVE mechanism ids')
    c['mechanisms'] = mechs
    c['warmup_steps'] = _int('warmup_steps', c['warmup_steps'], 0, 100_000)
    c['simulate_steps'] = _int('simulate_steps', c['simulate_steps'], 4, 100_000)
    c['input_dim'] = _int('input_dim', c['input_dim'], 1, 256)
    c['input_scale'] = _num('input_scale', c['input_scale'], 0.0, 10.0)
    c['step_ms'] = _num('step_ms', c['step_ms'], 0.0, 1e6, open_low=True)
    c['parity_cells'] = _int('parity_cells', c['parity_cells'], 1, 4096)
    c['parity_steps'] = _int('parity_steps', c['parity_steps'], 4, 100_000)
    c['parity_drive_max'] = _num('parity_drive_max', c['parity_drive_max'], 0.0, 10.0, open_low=True)
    c['parity_tolerance'] = _num('parity_tolerance', c['parity_tolerance'], 0.0, 1e-3, open_low=True)
    if c['backend'] not in ('auto', 'json', 'nir'):
        raise ValueError("backend must be 'auto', 'json' or 'nir'")
    return c


def _recurrent_parity(brain: Any, graph: Mapping[str, Any]) -> dict[str, Any]:
    """Recurrent drive of the exported Delay/Linear pairs on NEUROWEAVE's own output history vs the synapse
    table's transmission (same history, run on a copy). Exact unless state-dependent gains are on."""
    lossy = sorted(_LOSSY_CONNECTOME & brain.flags)
    if lossy:
        return {'applicable': False, 'reason': f'state-dependent gains: {lossy}', 'max_abs_difference': None}
    ref = brain.clone()
    syn = ref.synapses
    out = ref.cells.output.copy()
    drive = syn.transmit(out, ref.flags, tile_size=ref.spec.tile_size if 'tiled_execution' in ref.flags else None,
                         uncertainty_gate=1.0, cells=ref.cells,
                         src_weight=ref._structural_weight() if 'structural_encoder' in ref.flags else None)
    expect = drive.total
    got = np.zeros(ref.n)
    g = CELL_CONSTANTS['recurrent_gain']
    delayed_ok = 'synaptic_delays' in ref.flags
    for name, node in graph['nodes'].items():
        if node['type'] == 'Delay':
            k = int(round(float(node['params']['delay'][0]) / float(graph['metadata']['dt'])))
            src = out if k == 1 or not delayed_ok else (syn.hist[k - 1] if syn.hist_count >= k else out)
            W = np.asarray(graph['nodes'][f'recurrent_{k}']['params']['weight'])
            got = got + (W / g) @ src
    return {'applicable': True, 'reason': None, 'max_abs_difference': float(np.max(np.abs(got - expect)))
            if ref.n else 0.0, 'history_steps': int(syn.hist_count)}


def _cell_parity(graph: Mapping[str, Any], c: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Isolated cells (no recurrent input): NEUROWEAVE ``CellPopulation`` vs the exported neuron node driven
    through an identity Linear. The synaptic current must equal the fast channel; before a cell's first spike
    the membrane must equal NEUROWEAVE's membrane minus its (known, dropped) adaptation current."""
    from ..neuroweave.cells import CellPopulation, CompartmentDrive
    cells_node = graph['nodes']['cells']
    model = cells_node['type']
    k, T = c['parity_cells'], c['parity_steps']
    params = {name: [float(vals[0])] * k for name, vals in cells_node['params'].items()}
    tiny = {'nodes': {'x': {'type': 'Input', 'params': {'shape': [k]}},
                      'eye': {'type': 'Linear', 'params': {'weight': np.eye(k).tolist()}},
                      'cells': {'type': model, 'params': params},
                      'y': {'type': 'Output', 'params': {'shape': [k]}}},
            'edges': [['x', 'eye'], ['eye', 'cells'], ['cells', 'y']], 'metadata': {'dt': graph['metadata']['dt']}}
    rng = np.random.default_rng(int(seed) + _PARITY_OFFSET)
    u = rng.uniform(0.0, c['parity_drive_max'], size=(T, k))
    sim = simulate_graph(tiny, u, record_states=True)
    flags = frozenset({'membrane_dynamics'}) if model == 'CubaLIF' else frozenset()
    pop = CellPopulation(k, seed=int(seed), regions=np.zeros(k, dtype=np.int64), signs=np.ones(k, dtype=np.int64))
    zero = np.zeros(k)
    fast = np.zeros((T, k))
    memb = np.zeros((T, k))
    adapt = np.zeros((T, k))
    nw_spike = np.zeros((T, k), dtype=bool)
    for t in range(T):
        pop.step(CompartmentDrive(zero.copy(), zero.copy(), zero.copy(), zero.copy(), u[t].copy()), flags)
        fast[t], memb[t], adapt[t], nw_spike[t] = pop.fast, pop.membrane, pop.adaptation, pop.spike > 0
    nir_spike = sim['outputs']['y'] > 0
    i_dev = None
    if model == 'CubaLIF':
        i_dev = float(np.max(np.abs(sim['states']['cells']['i'] - fast)))
        corr = np.zeros(k)                                 # accumulated -0.06 * adaptation current
        ref_v = np.zeros((T, k))
        for t in range(T):
            corr = CELL_CONSTANTS['membrane_decay'] * corr - CELL_CONSTANTS['adaptation_current'] * adapt[t]
            ref_v[t] = memb[t] - corr
        v_nir = sim['states']['cells']['v']
    else:
        ref_v, v_nir = fast, sim['states']['cells']['v']

    def first(s: np.ndarray) -> list[int | None]:
        return [int(np.flatnonzero(s[:, j])[0]) if s[:, j].any() else None for j in range(k)]

    f_nw, f_nir = first(nw_spike), first(nir_spike)
    v_dev = 0.0
    for j in range(k):                                     # compare up to (and including) the first spike step
        stops = [s for s in (f_nw[j], f_nir[j]) if s is not None]
        end = min(stops) + 1 if stops else T
        if model == 'CubaLIF':
            seg = slice(0, end)
            # the spike step itself is compared before the reset: NEUROWEAVE stores the reset membrane
            pre_reset = np.abs(v_nir[seg, j] - ref_v[seg, j])
            if stops and nw_spike[end - 1, j]:
                pre_reset = pre_reset[:-1]
            v_dev = max(v_dev, float(pre_reset.max()) if pre_reset.size else 0.0)
        else:
            seg = slice(0, end - 1 if stops else T)        # LIF state is recorded before the NIR reset
            if seg.stop:
                v_dev = max(v_dev, float(np.max(np.abs(v_nir[seg, j] - ref_v[seg, j]))))
    both = [(a, b) for a, b in zip(f_nw, f_nir) if a is not None and b is not None]
    return {'model': model, 'cells': k, 'steps': T, 'drive_max': c['parity_drive_max'],
            'synaptic_current_max_abs_difference': i_dev, 'membrane_max_abs_difference_before_first_spike': v_dev,
            'first_spike_step_neuroweave': f_nw, 'first_spike_step_nir': f_nir,
            'first_spike_agreement': (sum(1 for a, b in both if a == b) / len(both)) if both else None,
            'spike_rate_neuroweave': float(nw_spike.mean()), 'spike_rate_nir': float(nir_spike.mean())}


def _seed_run(seed: int, c: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    from ..neuroweave.brain import NeuroWeaveBrain
    from ..neuroweave.experiments import make_spec
    spec = make_spec(c['mechanisms'], seed=seed, neurons=c['neurons'], topology=c['topology'],
                     steps=max(4, c['warmup_steps']))
    brain = NeuroWeaveBrain(spec)
    rng = np.random.default_rng(int(seed) + _INPUT_OFFSET)
    for _ in range(c['warmup_steps']):
        brain.step(rng.uniform(-1.0, 1.0, c['input_dim']) * c['input_scale'])
    export = brain.export_state()
    graph = brain_to_nir(export, input_dim=c['input_dim'], step_ms=c['step_ms'])
    report = validate_graph(graph)
    rt = write_nir(graph, out_dir / f'brain_seed{seed}.nir', backend=c['backend'])
    back = read_nir(out_dir / f'brain_seed{seed}.nir')
    srng = np.random.default_rng(int(seed) + _SIM_OFFSET)
    X = srng.uniform(-1.0, 1.0, (c['simulate_steps'], c['input_dim'])) * c['input_scale']
    sim = simulate_graph(graph, X)
    ref = brain.clone()                                   # NEUROWEAVE on the same inputs, frozen (descriptive)
    nw_spikes = 0
    for t in range(c['simulate_steps']):
        ref.frozen_step(X[t])
        nw_spikes += int(np.count_nonzero(ref.cells.spike))
    recurrent_nodes = [nd for name, nd in graph['nodes'].items() if name.startswith('recurrent_')]
    nnz = int(sum(np.count_nonzero(np.asarray(nd['params']['weight'])) for nd in recurrent_nodes))
    meta = graph['metadata']
    n = meta['source']['neurons']
    spikes = int(sum(sim['spike_counts'].values()))
    return {
        'seed': int(seed), 'neurons': n, 'synapses': meta['source']['synapses'],
        'unique_src_dst_delay': meta['source']['unique_src_dst_delay'],
        'represented_synapses': meta['represented_synapses'], 'recurrent_nonzero_weights': nnz,
        'delay_classes_steps': meta['delay_classes_steps'], 'node_types': report['node_types'],
        'node_count': report['node_count'], 'edge_count': report['edge_count'],
        'validation': {'valid': report['valid'], 'errors': report['errors'], 'warnings': report['warnings']},
        'round_trip': {k: rt[k] for k in ('format', 'round_trip_equal', 'node_count', 'edge_count',
                                          'read_node_count', 'read_edge_count')},
        'reread_equal': graphs_equal(graph, back),
        'simulation': {'steps': sim['steps'], 'finite': sim['finite'], 'spikes': spikes,
                       'spike_rate': spikes / float(sim['steps'] * n),
                       'output_abs_sum': float(np.abs(sim['outputs']['output']).sum()),
                       'neuroweave_spike_rate_same_input': nw_spikes / float(c['simulate_steps'] * n)},
        'recurrent_parity': _recurrent_parity(brain, graph),
        'cell_parity': _cell_parity(graph, c, seed),
        'lossy_features': meta['lossy_features'], 'folded_features': meta['folded_features'],
        'cell_parameters': meta['cell_parameters'],
    }


def run_experiment(*, quick: bool = False, seeds: Sequence[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """Export small NEUROWEAVE brains, validate, round-trip through a file, simulate the exported graph.

    Checks (``kind='validation'``, ``verdict`` in ``('pass', 'fail')``) per seed: graph valid with known node
    types; file round trip equal (real NIR file when the optional ``nir`` package imports, else the JSON fallback)
    with node and edge counts preserved; the exported recurrent weights reproduce every synapse (non-zero entries
    = unique src/dst/delay triples) and NEUROWEAVE's own recurrent drive on its output history; the exported cell
    reproduces the fast channel exactly and the membrane up to the dropped adaptation current (isolated cells);
    simulating the exported graph stays finite and spikes; lossy features are listed."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seed_list = _seeds(seeds)
    cfg = dict(_DEFAULTS)
    if quick:
        cfg.update(_QUICK)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(f'unknown run_experiment overrides: {unknown}')
    cfg.update(overrides)
    c = _validate_config(cfg)
    from . import harness
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='nir_export_') as tmp:
        rows = [_seed_run(s, c, Path(tmp)) for s in seed_list]
    tol = c['parity_tolerance']
    first = rows[0]
    applicable = [r['recurrent_parity'] for r in rows if r['recurrent_parity']['applicable']]
    checks = {
        'graph_valid': all(r['validation']['valid'] for r in rows),
        'node_types_known': all(set(r['node_types']) <= set(NIR_NODE_TYPES) for r in rows),
        'round_trip_equal': all(r['round_trip']['round_trip_equal'] and r['reread_equal'] for r in rows),
        'edge_and_node_counts_preserved': all(r['round_trip']['edge_count'] == r['round_trip']['read_edge_count']
                                              and r['round_trip']['node_count'] == r['round_trip']['read_node_count']
                                              for r in rows),
        'synapse_count_matches': all(r['recurrent_nonzero_weights'] == r['represented_synapses']
                                     and r['represented_synapses'] <= r['unique_src_dst_delay'] <= r['synapses']
                                     for r in rows),
        'recurrent_drive_parity': all(p['max_abs_difference'] <= tol for p in applicable),
        'synaptic_current_parity': all(r['cell_parity']['synaptic_current_max_abs_difference'] is None
                                       or r['cell_parity']['synaptic_current_max_abs_difference'] <= tol
                                       for r in rows),
        'membrane_parity_before_first_spike': all(r['cell_parity']['membrane_max_abs_difference_before_first_spike']
                                                  <= tol for r in rows),
        'simulation_finite': all(r['simulation']['finite'] for r in rows),
        'simulation_active': all(r['simulation']['spikes'] > 0 for r in rows),
        'lossy_features_listed': all(any(f['feature'] == 'graded_output_transmission' for f in r['lossy_features'])
                                     for r in rows),
    }
    receipt = harness.base_receipt(MODULE_ID, 'validation', {**c, 'quick': quick, 'seeds': list(seed_list)})
    receipt['module'] = MODULE_ID
    receipt['source'] = SOURCE
    receipt['nir_package'] = {k: v for k, v in nir_status().items() if k != 'error'}
    receipt['file_format'] = first['round_trip']['format']
    receipt['per_seed'] = rows
    receipt['lossy_features'] = first['lossy_features']
    receipt['folded_features'] = first['folded_features']
    receipt['summary'] = {
        'node_types': first['node_types'], 'neuron_model': next(iter(t for t in first['node_types']
                                                                     if t in NEURON_TYPES), None),
        'lossy_feature_count': len(first['lossy_features']),
        'max_recurrent_drive_difference': max((p['max_abs_difference'] for p in applicable), default=None),
        'recurrent_parity_applicable_seeds': len(applicable),
        'nir_spike_rate': [r['simulation']['spike_rate'] for r in rows],
        'neuroweave_spike_rate_same_input': [r['simulation']['neuroweave_spike_rate_same_input'] for r in rows],
        'first_spike_agreement': [r['cell_parity']['first_spike_agreement'] for r in rows],
    }
    if not applicable:
        checks['recurrent_drive_parity'] = True     # vacuous; recorded in the summary and per-seed rows
        receipt['summary']['recurrent_parity_note'] = 'not applicable: state-dependent transmission gains enabled'
    receipt['checks'] = checks
    receipt['verdict'] = 'pass' if all(checks.values()) else 'fail'
    receipt['verdict_basis'] = ('all_known_answer_checks' if receipt['verdict'] == 'pass'
                                else 'failed_checks: ' + ', '.join(k for k, ok in checks.items() if not ok))
    receipt['estimated_cost'] = {
        'neuroweave_brain_steps': int(len(seed_list) * (c['warmup_steps'] + c['simulate_steps'])),
        'graph_simulation_steps': int(len(seed_list) * c['simulate_steps']),
        'parity_cell_updates': int(len(seed_list) * c['parity_cells'] * c['parity_steps']),
        'files_written': len(seed_list)}
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t0}
    receipt['notes'] = (
        'NIR-style export of a frozen NEUROWEAVE brain snapshot. Exact: linear input projection, recurrent synapse '
        'table as Delay+Linear pairs per delay class (with static gains folded in), fast-channel synaptic current; '
        'membrane exact up to the dropped adaptation current. Approximated or dropped: graded tanh transmission '
        '(edges carry spikes), direct feed-through, multiplicative reset, refractoriness, adaptive threshold, '
        'calcium, plasticity rules and the listed mechanisms; the simulated spike rates of the exported graph and of '
        'NEUROWEAVE are reported side by side and are not expected to agree. Real NIR files are written only when '
        'the optional nir package imports (file_format says which). Not a claim of hardware deployment or of '
        'equivalence with any NIR backend.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
