"""Hardware-mapping helpers for NEUROWEAVE brains (NumPy only).

This module provides *numerical* building blocks for hardware experiments:

* symmetric signed INT8 weight quantisation (127 levels) with error statistics;
* Q8.8 fixed-point conversion with saturation;
* a tile schedule that groups edges by ``(dst_tile, src_tile)`` blocks and
  estimates cycles for a four-stage systolic-style pipeline;
* an execution profile that keeps *measured* (caller-supplied wall-clock)
  values strictly separate from *estimated* (operation/byte counts) values;
* an FPGA mapping manifest with endpoint and Dale-style sign validation.

Nothing here generates RTL, a bitstream or any evidence of FPGA execution.
Every manifest records ``rtl_generated: False`` and ``hardware_validated:
False``; cycle and resource numbers are analytic estimates only.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .spec import EVIDENCE_TIER, MAX_POPULATION, MAX_SYNAPSES

FPGA_FORMAT = 'SONAPSE_NEUROWEAVE_FPGA_MAPPING'
FPGA_VERSION = '1.0.0'
INT8_LEVELS = 127
Q88_SCALE = 256
Q88_RAW_MIN = -32768
Q88_RAW_MAX = 32767
Q88_MIN = Q88_RAW_MIN / Q88_SCALE          # -128.0
Q88_MAX = Q88_RAW_MAX / Q88_SCALE          # 127.99609375
PIPELINE_STAGES = ('load_weights', 'stream_activations', 'accumulate', 'writeback')
DEFAULT_CELL_HINTS = ('threshold', 'homeo_gain', 'myelin', 'membrane')
FPGA_DISCLAIMER = (
    'Machine-readable mapping manifest only. No RTL, testbench, synthesis, '
    'place-and-route, bitstream or FPGA execution was produced or validated; '
    'cycle and resource figures are analytic estimates, not measurements.'
)


# --------------------------------------------------------------------------- INT8
def _finite_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')
    return arr


def quantise_int8(w: Any, *, max_abs: float | None = None,
                  preserve_nonzero: bool = False) -> tuple[np.ndarray, float]:
    """Symmetric signed INT8 quantisation onto 127 levels per sign.

    ``scale = max_abs / 127`` (``max_abs`` defaults to ``max|w|``), and
    ``q = clip(round(w / scale), -127, 127)``. Values beyond ``max_abs``
    saturate. With ``preserve_nonzero`` any nonzero weight that would round to
    zero is kept at ``±1`` so sparse connectivity (and its sign) survives.
    Returns ``(q: int8 array, scale: float)``; see :func:`quantisation_stats`.
    """
    arr = _finite_array(w, 'weights')
    if max_abs is None:
        max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    else:
        if isinstance(max_abs, bool) or not math.isfinite(float(max_abs)) or float(max_abs) < 0:
            raise ValueError('max_abs must be a finite non-negative number')
        max_abs = float(max_abs)
    scale = max_abs / INT8_LEVELS if max_abs > 0 else 1.0
    q = np.clip(np.rint(arr / scale), -INT8_LEVELS, INT8_LEVELS)
    if preserve_nonzero:
        lost = (q == 0) & (arr != 0)
        q[lost] = np.sign(arr[lost])
    return q.astype(np.int8), float(scale)


def dequantise(q: Any, scale: float) -> np.ndarray:
    """Inverse of :func:`quantise_int8` (``q * scale`` as float64)."""
    qa = np.asarray(q)
    if qa.size and (not np.issubdtype(qa.dtype, np.integer)):
        if not np.all(np.isfinite(qa)) or not np.all(qa == np.rint(qa)):
            raise ValueError('quantised values must be integers')
    if np.any(np.abs(qa.astype(np.int64)) > INT8_LEVELS):
        raise ValueError('quantised values must lie in [-127, 127]')
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('scale must be a finite positive number')
    return qa.astype(np.float64).reshape(-1) * float(scale)


dequantise_int8 = dequantise


def quantisation_stats(w: Any, q: Any, scale: float) -> dict[str, Any]:
    """Error statistics of a quantisation (all values computed, JSON-safe)."""
    arr = _finite_array(w, 'weights')
    deq = dequantise(q, scale)
    if deq.shape != arr.shape:
        raise ValueError('weights and quantised values must have the same length')
    err = deq - arr
    nonzero = arr != 0
    sign_flips = int(np.sum(nonzero & (np.sign(deq) != 0) & (np.sign(deq) != np.sign(arr))))
    zeroed = int(np.sum(nonzero & (deq == 0)))
    signal = float(np.mean(arr ** 2)) if arr.size else 0.0
    mse = float(np.mean(err ** 2)) if arr.size else 0.0
    if mse == 0.0:
        sqnr = None  # exact (or empty): undefined/infinite ratio, reported as None
    else:
        sqnr = float(10.0 * math.log10(signal / mse)) if signal > 0 else None
    return {
        'count': int(arr.size),
        'scale': float(scale),
        'max_abs_error': float(np.max(np.abs(err))) if arr.size else 0.0,
        'mean_abs_error': float(np.mean(np.abs(err))) if arr.size else 0.0,
        'rmse': float(math.sqrt(mse)),
        'sqnr_db': sqnr,
        'saturated': int(np.sum(np.abs(np.asarray(q).astype(np.int64)) == INT8_LEVELS)),
        'zeroed_nonzero': zeroed,
        'sign_flips': sign_flips,
        'levels_used': int(np.unique(np.asarray(q)).size),
    }


# --------------------------------------------------------------------------- Q8.8
def to_q88(x: Any) -> np.ndarray:
    """Convert to raw Q8.8 int16 values with saturation to [-128, 127.996]."""
    arr = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError('Q8.8 input must be finite')
    return np.clip(np.rint(arr * Q88_SCALE), Q88_RAW_MIN, Q88_RAW_MAX).astype(np.int16)


def from_q88(raw: Any) -> np.ndarray:
    """Convert raw Q8.8 integers back to float64."""
    arr = np.asarray(raw)
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        f = arr.astype(np.float64)
        if not np.all(np.isfinite(f)) or not np.all(f == np.rint(f)):
            raise ValueError('Q8.8 raw values must be integers')
    a64 = arr.astype(np.int64)
    if np.any(a64 < Q88_RAW_MIN) or np.any(a64 > Q88_RAW_MAX):
        raise ValueError('Q8.8 raw values must lie in the int16 range')
    return a64.astype(np.float64) / Q88_SCALE


def q88_round(x: Any) -> np.ndarray:
    """Round-trip through Q8.8 (the ``q88_state`` mechanism's numeric effect)."""
    return from_q88(to_q88(x))


def q88_stats(x: Any) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    raw = to_q88(arr)
    err = from_q88(raw) - arr
    return {
        'count': int(arr.size),
        'saturated': int(np.sum((arr < Q88_MIN) | (arr > Q88_MAX))),
        'max_abs_error': float(np.max(np.abs(err))) if arr.size else 0.0,
        'range': [Q88_MIN, Q88_MAX],
    }


# --------------------------------------------------------------------------- tiling
def _int_edges(src: Any, dst: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError('n must be a positive integer')
    s = np.asarray(src)
    d = np.asarray(dst)
    if s.shape != d.shape or s.ndim != 1:
        raise ValueError('src and dst must be 1-D arrays of equal length')
    for arr, name in ((s, 'src'), (d, 'dst')):
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            f = arr.astype(np.float64)
            if not np.all(np.isfinite(f)) or not np.all(f == np.rint(f)):
                raise ValueError(f'{name} must contain integers')
    s = s.astype(np.int64)
    d = d.astype(np.int64)
    if s.size and (s.min() < 0 or d.min() < 0 or s.max() >= n or d.max() >= n):
        raise ValueError('edge endpoint out of range')
    return s, d


def tile_schedule(src: Any, dst: Any, n: int, tile_size: int, *, lanes: int = 8) -> dict[str, Any]:
    """Group edges into ``(dst_tile, src_tile)`` blocks and pipeline them.

    Tiles are ordered by destination block then source block, so all partial
    sums for one destination block are accumulated contiguously and written
    back once (after its last source block). Each tile passes through four
    stages; per-stage cycle counts are analytic estimates:

    * ``load_weights``: ``ceil(edges / lanes)`` weight-word loads;
    * ``stream_activations``: one cycle per distinct source neuron in the tile;
    * ``accumulate``: ``ceil(edges / lanes)`` MAC cycles;
    * ``writeback``: distinct destinations of the block on the block's last
      tile, otherwise 0.

    Pipelining: stage ``s`` of tile ``t`` starts when stage ``s`` of tile
    ``t-1`` and stage ``s-1`` of tile ``t`` have finished.
    """
    if isinstance(tile_size, bool) or not isinstance(tile_size, (int, np.integer)) or int(tile_size) < 1:
        raise ValueError('tile_size must be a positive integer')
    if isinstance(lanes, bool) or not isinstance(lanes, (int, np.integer)) or int(lanes) < 1:
        raise ValueError('lanes must be a positive integer')
    tile_size, lanes, n = int(tile_size), int(lanes), int(n)
    s, d = _int_edges(src, dst, n)
    st, dt = s // tile_size, d // tile_size
    order = np.lexsort((np.arange(s.size), st, dt))  # stable: dst_tile, src_tile, edge index
    tiles: list[dict[str, Any]] = []
    i = 0
    while i < order.size:
        key = (int(dt[order[i]]), int(st[order[i]]))
        j = i
        while j < order.size and (int(dt[order[j]]), int(st[order[j]])) == key:
            j += 1
        idx = order[i:j]
        rows = min(tile_size, n - key[0] * tile_size)
        cols = min(tile_size, n - key[1] * tile_size)
        tiles.append({
            'tile_index': len(tiles), 'dst_tile': key[0], 'src_tile': key[1],
            'edge_indices': [int(e) for e in idx], 'edge_count': int(idx.size),
            'unique_src': int(np.unique(s[idx]).size), 'unique_dst': int(np.unique(d[idx]).size),
            'utilization': float(idx.size / (rows * cols)),
        })
        i = j
    # mark the last tile of each destination block and its block-level destinations
    block_rows = {int(b): int(np.unique(d[dt == b]).size) for b in np.unique(dt)}
    for k, tile in enumerate(tiles):
        last = k + 1 == len(tiles) or tiles[k + 1]['dst_tile'] != tile['dst_tile']
        tile['writeback'] = bool(last)
        tile['writeback_rows'] = block_rows[tile['dst_tile']] if last else 0
    stage_ends = [0] * len(PIPELINE_STAGES)
    schedule: list[dict[str, Any]] = []
    sequential = 0
    for tile in tiles:
        e = tile['edge_count']
        cycles = [math.ceil(e / lanes), tile['unique_src'], math.ceil(e / lanes), tile['writeback_rows']]
        prev_end = 0
        entry = {'tile_index': tile['tile_index'], 'stages': []}
        for si, (stage, c) in enumerate(zip(PIPELINE_STAGES, cycles)):
            start = max(stage_ends[si], prev_end)
            end = start + c
            stage_ends[si] = end
            prev_end = end
            entry['stages'].append({'stage': stage, 'start': int(start), 'cycles': int(c), 'end': int(end)})
        sequential += sum(cycles)
        schedule.append(entry)
    pipelined = int(max(stage_ends)) if tiles else 0
    edge_order = [e for t in tiles for e in t['edge_indices']]
    return {
        'tile_size': tile_size, 'lanes': lanes, 'n': n, 'edge_count': int(s.size),
        'tile_count': len(tiles), 'tiles': tiles, 'schedule': schedule,
        'pipeline_stages': list(PIPELINE_STAGES), 'edge_order': edge_order,
        'cycles': {
            'kind': 'estimated',
            'pipelined': pipelined,
            'sequential': int(sequential),
            'speedup_vs_sequential': float(sequential / pipelined) if pipelined else 1.0,
        },
        'mean_utilization': float(np.mean([t['utilization'] for t in tiles])) if tiles else 0.0,
    }


# --------------------------------------------------------------------------- profile
def execution_profile(n: int, n_edges: int, *, steps: int = 1, wallclock_s: float | None = None,
                      tile_size: int = 16, weight_bytes: int = 1, state_bytes: int = 2,
                      index_bytes: int | None = None, cell_states: int = 4,
                      schedule: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Separate *measured* (caller-supplied wall-clock) from *estimated* costs.

    Estimated values are analytic: one MAC (2 ops) per edge per step plus
    ``cell_states`` element-wise ops per neuron per step, and bytes moved for
    weights, endpoint indices and neuron state. ``measured`` is populated only
    from ``wallclock_s`` (seconds over ``steps`` steps) supplied by the caller.
    """
    for name, val, lo in (('n', n, 0), ('n_edges', n_edges, 0), ('steps', steps, 1),
                          ('tile_size', tile_size, 1), ('weight_bytes', weight_bytes, 1),
                          ('state_bytes', state_bytes, 1), ('cell_states', cell_states, 0)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) < lo:
            raise ValueError(f'{name} must be an integer >= {lo}')
    n, n_edges, steps = int(n), int(n_edges), int(steps)
    if index_bytes is None:
        index_bytes = 1 if n <= 256 else 2 if n <= 65536 else 4
    mac_ops = 2 * n_edges * steps
    neuron_ops = int(cell_states) * n * steps
    weight_b = n_edges * int(weight_bytes)
    index_b = 2 * n_edges * int(index_bytes)
    state_b = n * int(state_bytes) * max(1, int(cell_states))
    bytes_per_step = weight_b + index_b + 2 * state_b  # read previous + write next buffer
    estimated: dict[str, Any] = {
        'kind': 'estimated',
        'ops_total': int(mac_ops + neuron_ops),
        'mac_ops': int(mac_ops),
        'neuron_ops': int(neuron_ops),
        'bytes_per_step': int(bytes_per_step),
        'bytes_total': int(bytes_per_step * steps),
        'weight_bytes': int(weight_b),
        'index_bytes': int(index_b),
        'state_bytes': int(state_b),
        'arithmetic_intensity': float((mac_ops + neuron_ops) / (bytes_per_step * steps)) if bytes_per_step else 0.0,
    }
    if schedule is not None:
        cyc = schedule.get('cycles') if isinstance(schedule, Mapping) else None
        if not isinstance(cyc, Mapping) or cyc.get('kind') != 'estimated':
            raise ValueError('schedule cycles must be labelled estimated')
        pipelined = cyc.get('pipelined', 0)
        if isinstance(pipelined, bool) or not isinstance(pipelined, (int, np.integer)) or int(pipelined) < 0:
            raise ValueError('schedule pipelined cycles must be a non-negative integer')
        estimated['cycles_per_step'] = int(pipelined)
        estimated['cycles_total'] = int(pipelined) * steps
    if wallclock_s is None:
        measured: dict[str, Any] = {'available': False, 'source': None}
    else:
        if isinstance(wallclock_s, bool) or not math.isfinite(float(wallclock_s)) or float(wallclock_s) < 0:
            raise ValueError('wallclock_s must be a finite non-negative number')
        w = float(wallclock_s)
        measured = {
            'available': True,
            'source': 'caller_supplied_wallclock',
            'wallclock_s': w,
            'latency_ms_per_step': 1000.0 * w / steps,
            'steps_per_s': (steps / w) if w > 0 else None,
            'achieved_ops_per_s_from_estimate': (estimated['ops_total'] / w) if w > 0 else None,
        }
    return {'n': n, 'n_edges': n_edges, 'steps': steps, 'tile_size': int(tile_size),
            'measured': measured, 'estimated': estimated, 'evidence_tier': EVIDENCE_TIER}


# --------------------------------------------------------------------------- manifest
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _brain_arrays(brain_state: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(brain_state, Mapping), 'brain_state must be a mapping')
    topo = brain_state.get('topology')
    _require(isinstance(topo, Mapping), 'brain_state.topology missing')
    n = topo.get('n')
    _require(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= MAX_POPULATION,
             f'topology.n must be an int in [1, {MAX_POPULATION}]')
    syn = brain_state.get('synapses')
    if isinstance(syn, Mapping) and 'src' in syn:
        src, dst = syn.get('src'), syn.get('dst')
        w = syn.get('w', syn.get('weight'))
        delay = syn.get('delay', topo.get('delay'))
    else:
        src, dst, w, delay = topo.get('src'), topo.get('dst'), topo.get('weight'), topo.get('delay')
    _require(all(isinstance(v, (list, tuple, np.ndarray)) for v in (src, dst, w)),
             'synapse src/dst/w must be lists')
    _require(len(src) == len(dst) == len(w), 'synapse arrays must have equal length')
    _require(len(src) <= MAX_SYNAPSES, f'synapse count exceeds {MAX_SYNAPSES}')
    delays_defaulted = delay is None
    if delays_defaulted:
        delay = [1] * len(src)
    _require(isinstance(delay, (list, tuple, np.ndarray)) and len(delay) == len(src),
             'synapse delay array must match the synapse count')
    sign = topo.get('sign')
    _require(isinstance(sign, (list, tuple, np.ndarray)) and len(sign) == n, 'topology.sign must have length n')
    region = topo.get('region')
    if not isinstance(region, (list, tuple, np.ndarray)) or len(region) != n:
        region = [0] * n
    return {'n': n, 'src': list(src), 'dst': list(dst), 'w': list(w), 'delay': list(delay),
            'sign': list(sign), 'region': list(region), 'delays_defaulted': delays_defaulted}


def _as_int_list(values: Iterable[Any]) -> list[int] | None:
    out = []
    for v in values:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
        elif isinstance(v, (float, np.floating)) and math.isfinite(float(v)) and float(v) == int(v):
            out.append(int(v))
        else:
            return None
    return out


def fpga_mapping_manifest(brain_state: Mapping[str, Any], *, tile_size: int | None = None,
                          lanes: int = 8, cell_hints: Sequence[str] = DEFAULT_CELL_HINTS,
                          preserve_nonzero: bool = True) -> dict[str, Any]:
    """Build an FPGA *mapping* manifest from a brain-state dict (duck-typed).

    Weights come from ``synapses`` (current, post-plasticity) when present,
    otherwise from ``topology``. Validation never raises for sign/endpoint
    problems: they are recorded in ``validation`` so a broken mapping is
    visible. Structural problems (missing arrays) raise ``ValueError``.
    """
    arrays = _brain_arrays(brain_state)
    n = arrays['n']
    errors: list[str] = []
    src = _as_int_list(arrays['src'])
    dst = _as_int_list(arrays['dst'])
    sign = _as_int_list(arrays['sign'])
    _require(src is not None and dst is not None, 'synapse endpoints must be integers')
    _require(sign is not None and all(s in (-1, 1) for s in sign), 'topology.sign must be +1/-1 integers')
    w = _finite_array(arrays['w'], 'synapse weights')
    delays = _as_int_list(arrays['delay'])
    _require(delays is not None, 'delays must be integers')
    region = _as_int_list(arrays['region']) or [0] * n
    if tile_size is None:
        spec = brain_state.get('spec') if isinstance(brain_state.get('spec'), Mapping) else {}
        tile_size = int(spec.get('tile_size', 16)) if isinstance(spec.get('tile_size', 16), int) else 16
    bad_endpoints = [i for i, (a, b) in enumerate(zip(src, dst)) if not (0 <= a < n and 0 <= b < n)]
    endpoints_ok = not bad_endpoints
    if bad_endpoints:
        errors.append(f'{len(bad_endpoints)} edge endpoint(s) out of range (first edge {bad_endpoints[0]})')
    q, scale = quantise_int8(w, preserve_nonzero=preserve_nonzero)
    stats = quantisation_stats(w, q, scale)
    sign_mismatch = []
    for i, (a, qv) in enumerate(zip(src, q.tolist())):
        if qv != 0 and 0 <= a < n and (1 if qv > 0 else -1) != sign[a]:
            sign_mismatch.append(i)
    signs_ok = not sign_mismatch
    if sign_mismatch:
        errors.append(f'{len(sign_mismatch)} quantised weight(s) disagree with source sign (first edge {sign_mismatch[0]})')
    delays_ok = all(1 <= dl <= 3 for dl in delays)
    if not delays_ok:
        errors.append('delay outside the 2-bit encodable range [1, 3]')
    if endpoints_ok:
        sched = tile_schedule(src, dst, n, tile_size, lanes=lanes)
    else:
        sched = {'tile_size': int(tile_size), 'lanes': int(lanes), 'n': n, 'edge_count': len(src),
                 'tile_count': 0, 'tiles': [], 'schedule': [], 'pipeline_stages': list(PIPELINE_STAGES),
                 'edge_order': [], 'cycles': {'kind': 'estimated', 'pipelined': 0, 'sequential': 0,
                                              'speedup_vs_sequential': 1.0}, 'mean_utilization': 0.0}
    tile_of_edge = [0] * len(src)
    for t in sched['tiles']:
        for e in t['edge_indices']:
            tile_of_edge[e] = t['tile_index']
    cells = brain_state.get('cells') if isinstance(brain_state.get('cells'), Mapping) else {}
    hints: dict[str, Any] = {}
    for name in cell_hints:
        vals = cells.get(name)
        if isinstance(vals, (list, tuple, np.ndarray)) and len(vals) == n:
            arr = _finite_array(vals, f'cells.{name}')
            hints[name] = {'q88': [int(v) for v in to_q88(arr)], **q88_stats(arr)}
    edges = [{'src': src[i], 'dst': dst[i], 'weight_int8': int(q[i]), 'source_sign': sign[src[i]] if 0 <= src[i] < n else 0,
              'delay': delays[i], 'tile_index': tile_of_edge[i]} for i in range(len(src))]
    nodes = [{'id': i, 'region': int(region[i]), 'source_sign': sign[i]} for i in range(n)]
    E = len(src)
    index_bytes = 1 if n <= 256 else 2
    weight_bits = 8 * E
    index_bits = 2 * 8 * index_bytes * E
    delay_bits = 2 * E
    state_bits = 16 * n * max(1, len(hints))
    total_bits = weight_bits + index_bits + delay_bits + 2 * state_bits
    resources = {
        'kind': 'estimated',
        'weight_bytes': E,
        'edge_index_bytes': 2 * index_bytes * E,
        'delay_bits': delay_bits,
        'neuron_state_bytes_double_buffered': 2 * state_bits // 8,
        'total_bits': int(total_bits),
        'bram18k_blocks': int(math.ceil(total_bits / 18432)) if total_bits else 0,
        'dsp_mac_units': int(lanes),
        'pipelined_cycles_per_step': sched['cycles']['pipelined'],
    }
    topo = brain_state['topology']
    return {
        'format': FPGA_FORMAT,
        'version': FPGA_VERSION,
        'evidence_tier': EVIDENCE_TIER,
        'source': {'topology': str(topo.get('name', 'unknown')), 'seed': topo.get('seed'),
                   'step': brain_state.get('step') if isinstance(brain_state.get('step'), int) else None},
        'graph': {'neuron_count': n, 'synapse_count': E, 'nodes': nodes, 'edges': edges},
        'numeric_format': {'weight': 'signed int8 symmetric', 'weight_levels': INT8_LEVELS,
                           'weight_range': [-INT8_LEVELS, INT8_LEVELS], 'neuron_state': 'Q8.8',
                           'q88_range': [Q88_MIN, Q88_MAX], 'delay_encoding': '2-bit integer [1,3]'},
        'scales': {'weight': scale},
        'quantisation': stats,
        'neuron_hints': hints,
        'tiles': sched['tiles'],
        'schedule': {k: sched[k] for k in ('tile_size', 'lanes', 'tile_count', 'pipeline_stages',
                                            'schedule', 'cycles', 'mean_utilization', 'edge_order')},
        'resource_estimate': resources,
        'validation': {'endpoints_ok': endpoints_ok, 'signs_ok': signs_ok, 'delays_ok': delays_ok,
                       'delays_defaulted': bool(arrays['delays_defaulted']),
                       'weights_bounded': bool(np.all(np.abs(q.astype(np.int64)) <= INT8_LEVELS)),
                       'errors': errors},
        'rtl_generated': False,
        'hardware_validated': False,
        'bitstream': None,
        'disclaimer': FPGA_DISCLAIMER,
    }


def validate_fpga_manifest(m: Any) -> dict[str, Any]:
    """Independently re-check a (possibly edited) FPGA mapping manifest."""
    errors: list[str] = []
    if not isinstance(m, Mapping):
        return {'valid': False, 'errors': ['manifest must be a JSON object']}
    if m.get('format') != FPGA_FORMAT:
        errors.append('unknown manifest format')
    if m.get('version') != FPGA_VERSION:
        errors.append('unsupported manifest version')
    for key in ('rtl_generated', 'hardware_validated'):
        if m.get(key) is not False:
            errors.append(f'{key} must be False (no hardware artefact is produced)')
    if m.get('bitstream') is not None:
        errors.append('bitstream must be None')
    graph = m.get('graph')
    if not isinstance(graph, Mapping) or not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('edges'), list):
        errors.append('graph nodes/edges missing')
        return {'valid': False, 'errors': errors}
    nodes, edges = graph['nodes'], graph['edges']
    n = len(nodes)
    if graph.get('neuron_count') != n:
        errors.append('declared neuron_count does not match nodes')
    if graph.get('synapse_count') != len(edges):
        errors.append('declared synapse_count does not match edges')
    signs = []
    for node in nodes:
        s = node.get('source_sign') if isinstance(node, Mapping) else None
        if s not in (-1, 1) or isinstance(s, bool):
            errors.append('node source_sign must be +1/-1')
            s = 0
        signs.append(s)
    endpoint_bad = sign_bad = weight_bad = delay_bad = 0
    for e in edges:
        if not isinstance(e, Mapping):
            endpoint_bad += 1
            continue
        a, b, q = e.get('src'), e.get('dst'), e.get('weight_int8')
        if not (type(a) is int and type(b) is int and 0 <= a < n and 0 <= b < n):
            endpoint_bad += 1
            continue
        if type(q) is not int or abs(q) > INT8_LEVELS:
            weight_bad += 1
            continue
        if q != 0 and (1 if q > 0 else -1) != signs[a]:
            sign_bad += 1
        if e.get('source_sign') != signs[a]:
            sign_bad += 1
        d = e.get('delay')
        if type(d) is not int or not 1 <= d <= 3:
            delay_bad += 1
    if endpoint_bad:
        errors.append(f'{endpoint_bad} edge endpoint(s) out of range')
    if weight_bad:
        errors.append(f'{weight_bad} weight(s) are not int8 in [-127, 127]')
    if sign_bad:
        errors.append(f'{sign_bad} sign violation(s): quantised weight sign differs from source sign')
    if delay_bad:
        errors.append(f'{delay_bad} delay(s) outside [1, 3]')
    tiles = m.get('tiles')
    sched = m.get('schedule') if isinstance(m.get('schedule'), Mapping) else {}
    tile_size = sched.get('tile_size')
    if not (type(tile_size) is int and tile_size >= 1):
        errors.append('schedule tile_size must be a positive integer')
        tile_size = None
    if isinstance(tiles, list):
        seen: list[int] = []
        structure_ok = True
        misplaced = 0
        for pos, t in enumerate(tiles):
            idx = t.get('edge_indices') if isinstance(t, Mapping) else None
            if not isinstance(idx, list) or any(type(e) is not int for e in idx):
                errors.append('tile edge_indices must be lists of integers')
                structure_ok = False
                break
            seen.extend(idx)
            if endpoint_bad or tile_size is None:
                continue
            dt_, st_ = t.get('dst_tile'), t.get('src_tile')
            for e in idx:
                if not 0 <= e < len(edges):
                    continue
                edge = edges[e]
                # each edge must sit in the (dst_tile, src_tile) block its endpoints imply
                if edge['dst'] // tile_size != dt_ or edge['src'] // tile_size != st_ or edge.get('tile_index') != pos:
                    misplaced += 1
        if structure_ok and sorted(seen) != list(range(len(edges))) and not endpoint_bad:
            errors.append('tiles do not cover every edge exactly once')
        if misplaced:
            errors.append(f'{misplaced} edge(s) assigned to a tile whose block does not contain them')
    else:
        errors.append('tiles missing')
    scale = (m.get('scales') or {}).get('weight') if isinstance(m.get('scales'), Mapping) else None
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        errors.append('weight scale must be finite and positive')
    cycles = (m.get('schedule') or {}).get('cycles') if isinstance(m.get('schedule'), Mapping) else None
    if not isinstance(cycles, Mapping) or cycles.get('kind') != 'estimated':
        errors.append('schedule cycles must be labelled estimated')
    return {'valid': not errors, 'errors': errors, 'neurons': n, 'synapses': len(edges)}


__all__ = [
    'FPGA_FORMAT', 'FPGA_VERSION', 'INT8_LEVELS', 'Q88_MIN', 'Q88_MAX', 'PIPELINE_STAGES',
    'quantise_int8', 'dequantise', 'dequantise_int8', 'quantisation_stats', 'to_q88', 'from_q88',
    'q88_round', 'q88_stats', 'tile_schedule', 'execution_profile', 'fpga_mapping_manifest',
    'validate_fpga_manifest',
]
