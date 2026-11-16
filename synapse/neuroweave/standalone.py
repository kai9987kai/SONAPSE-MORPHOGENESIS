"""Self-contained offline HTML runner for a frozen NEUROWEAVE package.

``standalone_html(package)`` turns a validated ``SONAPSE_NEUROWEAVE_PACKAGE``
into one HTML file with no external resources. The page embeds a compact
*frozen model* (JSON, script-safe encoded) and a pure-JavaScript mirror of

* ``NeuroWeaveBrain.frozen_step`` on a 24-value input ``[8 task | 16 sensory]``
  with neutral controls, ``uncertainty=0`` and the stored region route gain;
* ``NeuroWeaveBrain.predict`` (the raw readout probability);
* ``MicroCollective.predict`` on the 16 sensory values (when the package has a
  ``micro_collective`` component).

The JS follows the NumPy math operation by operation, including every
forward-path mechanism the brain consumes. Quantities that are derived from
NumPy random streams (input projection, FlyHash matrix) or are constant while
frozen (structural-encoder weights, sparse ``k``) are precomputed in Python and
embedded. Mechanisms the page cannot execute (learning, engine-level
subsystems) are never silently omitted: they are listed in a visible
"Not executed in standalone" section and in the package capability matrix
(``capabilities.frozen_inference.standalone``, see :func:`annotate_package`).

``frozen_reference`` is the Python oracle used by the parity tests.
Synthetic computational abstraction; no biological or hardware claim.
"""
from __future__ import annotations

import copy
import html
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .brain import HASH_CHANNELS, HASH_WINNERS, NeuroWeaveBrain, _KIND_INPUT_GAIN
from .cells import BASE_THRESHOLD, STATE_FIELDS, _BOUNDS
from .package import (CAPABILITIES, PACKAGE_FORMAT, package_fingerprint, validate_package)
from .spec import (EVIDENCE_TIER, MECHANISMS, SENSORY_CHANNEL_IDS, SENSORY_DIM, TASK_INPUT_DIM,
                   NeuroWeaveSpecV1)

__all__ = ['STANDALONE_SCHEMA', 'STANDALONE_INPUT_DIM', 'standalone_html', 'standalone_model',
           'standalone_capabilities', 'annotate_package', 'frozen_reference', 'script_safe_json',
           'STANDALONE_ENGINE_JS', 'SELECTIVE_LABEL', 'SELECTIVE_DEFAULT_THRESHOLD', 'selective_outcome',
           'default_selective_threshold']

STANDALONE_SCHEMA = 'NeuroWeaveStandaloneModelV1'
STANDALONE_INPUT_DIM = TASK_INPUT_DIM + SENSORY_DIM
SELECTIVE_LABEL = 'heuristic, not calibrated'
SELECTIVE_DEFAULT_THRESHOLD = 0.12


def default_selective_threshold(flags: Iterable[str]) -> float:
    """0.12 when ``uncertainty_gating`` is enabled (as in NEUROWEAVE-LAB 0.4.0), else 0 (never abstain)."""
    return SELECTIVE_DEFAULT_THRESHOLD if 'uncertainty_gating' in set(flags) else 0.0


def selective_outcome(probability: float, router_entropy: float | None, threshold: float) -> dict[str, Any]:
    """Abstain/review outcome from a *heuristic* selective score (not a calibrated confidence).

    ``margin = |2p - 1|`` (readout margin), ``risk = clip(router_entropy, 0, 1)`` (uncertainty proxy;
    0 when the package has no Micro-Collective), ``score = margin * (1 - risk)``; the runner abstains
    (``ABSTAIN · REVIEW``) when ``score < threshold``. The raw probability is never altered.
    """
    p = float(probability)
    thr = float(threshold)
    if not math.isfinite(p) or not math.isfinite(thr) or not 0.0 <= thr <= 1.0:
        raise ValueError('probability must be finite and threshold must be in [0, 1]')
    margin = abs(2.0 * p - 1.0)
    risk = 0.0 if router_entropy is None else min(max(float(router_entropy), 0.0), 1.0)
    score = margin * (1.0 - risk)
    abstain = score < thr
    return {'score': score, 'margin': margin, 'risk': risk,
            'risk_source': 'none' if router_entropy is None else 'router_entropy', 'threshold': thr,
            'abstain': abstain, 'decision': 'abstain_review' if abstain else 'predict',
            'raw_probability': p, 'label': SELECTIVE_LABEL}

# Forward-path mechanisms the JS mirror executes exactly as frozen_step does.
_EXECUTED: dict[str, str] = {
    'multi_soma': 'multi-soma fused update', 'extended_somas': 'primary/secondary/burst/gating somas',
    'dendritic_compartments': 'compartment gating', 'membrane_dynamics': 'membrane, threshold, refractory, calcium',
    'axonal_branches': 'branch/terminal release (cell myelin proxy evolves as in frozen_step)',
    'astrocyte_regulation': 'astrocyte gain', 'homeostatic_governor': 'homeostatic gain',
    'predictive_coding': 'per-unit predictive coding', 'hierarchical_temporal_memory': 'fast/mid/slow traces',
    'q88_state': 'Q8.8 rounding (round-half-even)', 'sparse_code': 'k-winners mask (k embedded)',
    'structural_encoder': 'graph fingerprint weights (embedded; constant while frozen)',
    'context_attention': 'top-8 incoming softmax attention', 'flyhash_projection': 'FlyHash code (matrix embedded)',
    'synaptic_delays': 'delay ring buffer', 'quantised_synapses': 'INT8 transmission',
    'adaptive_routing': 'destination-state routing', 'tiled_execution': 'tile-ordered accumulation',
    'adaptive_conduction': 'myelin-shortened delays and conduction scaling (myelin frozen)',
}
_PARTIAL: dict[str, str] = {
    'multi_region_routing': 'the stored region route-gain matrix is applied; the engine\'s neuromodulator-derived '
                            'route gain is not computed in the page',
    'uncertainty_gating': 'the gate is fixed at 1.0 (uncertainty=0) because the predictive ensemble that supplies '
                          'uncertainty is not executed',
}
_LEARNING_ONLY = ('reward_plasticity', 'stdp', 'meta_plasticity', 'structural_plasticity', 'neurogenesis',
                  'sleep_consolidation')
_ENGINE_ONLY: dict[str, str] = {
    'multimodal_sensory': 'engine sensory fusion (reliability weighting, imputation) is not run; sensory values '
                          'are fed to the brain as typed',
    'cross_modal_memory': 'cross-modal imputation belongs to the engine sensory hub',
    'neuromodulator_field': 'neuromodulator levels/controls are not simulated; controls are neutral',
    'affective_modulation': 'affect variables are not simulated; controls are neutral',
    'receptor_heterogeneity': 'receptor matrix belongs to the neuromodulator system',
    'predictive_ensemble': 'the predictive hierarchy is not executed',
    'hierarchical_prediction': 'the predictive hierarchy is not executed',
    'counterfactual_prediction': 'the predictive hierarchy is not executed',
    'efference_self_model': 'needs issued/observed actions from an environment',
    'specialist_graft': 'the specialist graft state is not part of the packaged Micro-Collective',
    'memory_ecology': 'memory recall is not composed into the drive',
    'predictive_replay': 'replay is a learning-time process',
    'representation_fusion': 'engine packet fusion of the 40-d drive is not run',
    'transmitter_inference': 'connectivity analysis only; not part of the forward pass',
    'associative_memory': 'not consumed by the NumPy runtime either',
    'fast_weights': 'not consumed by the NumPy runtime either',
}


# =========================================================================== helpers
def script_safe_json(obj: Any) -> str:
    """JSON safe to embed inside ``<script>``: escapes ``<``, ``>``, ``&`` and U+2028/U+2029."""
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return (text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def _require_package(package: Any) -> dict[str, Any]:
    if not isinstance(package, Mapping) or package.get('format') != PACKAGE_FORMAT:
        raise ValueError(f'standalone export needs a {PACKAGE_FORMAT} package')
    report = validate_package(package)
    if not report['valid']:
        raise ValueError('invalid package: ' + '; '.join(report['errors'][:6]))
    return copy.deepcopy(dict(package))


def _load_brain(package: Mapping[str, Any]) -> NeuroWeaveBrain:
    state = package['brain']
    try:
        spec = NeuroWeaveSpecV1.from_dict(state['spec'])
        brain = NeuroWeaveBrain(spec)
        brain.import_state(state)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f'package brain cannot be loaded for frozen inference: {exc}') from exc
    return brain


def _load_micro(package: Mapping[str, Any]):
    from .micro_collective import MicroCollective
    state = (package.get('components') or {}).get('micro_collective')
    if not isinstance(state, Mapping):
        return None
    cfg = state.get('config') or {}
    try:
        micro = MicroCollective(input_dim=cfg['input_dim'], output_dim=cfg['output_dim'], hidden=cfg['hidden'],
                                experts=1, top_k=1, seed=cfg['seed'])
        micro.import_state(state)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f'packaged Micro-Collective cannot be loaded: {exc}') from exc
    if micro.input_dim != SENSORY_DIM:
        return None
    return micro


def standalone_capabilities(mechanisms: Iterable[str], has_micro: bool) -> dict[str, Any]:
    """Which enabled mechanisms the standalone page executes, partially executes, or does not execute."""
    mech = [m for m in (x.id for x in MECHANISMS) if m in set(mechanisms)]
    executed, partial, not_executed = [], [], []
    for m in mech:
        if m in _EXECUTED:
            executed.append(m)
        elif m in _PARTIAL:
            partial.append({'id': m, 'reason': _PARTIAL[m]})
        elif m in _LEARNING_ONLY:
            not_executed.append({'id': m, 'reason': 'learning/structural change is disabled in frozen inference'})
        elif m in ('micro_collective', 'expert_routing'):
            if has_micro:
                executed.append(m)
            else:
                not_executed.append({'id': m, 'reason': 'no micro_collective component in the package'})
        else:
            not_executed.append({'id': m, 'reason': _ENGINE_ONLY.get(m, 'no standalone implementation')})
    if not has_micro:
        micro_note = 'not available (no micro_collective component)'
    else:
        micro_note = 'MicroCollective.predict on the 16 typed sensory values (learning disabled)'
    return {
        'runtime': 'javascript_standalone_html',
        'schema': STANDALONE_SCHEMA,
        'input_layout': {'task': TASK_INPUT_DIM, 'sensory': list(SENSORY_CHANNEL_IDS),
                         'brain_input': f'{STANDALONE_INPUT_DIM}-value [task | sensory] vector fed to frozen_step'},
        'controls': 'neutral (gain 1, threshold_shift 0, attention 1, plasticity_scale 1); uncertainty 0',
        'micro_collective': micro_note,
        'executed': executed,
        'partial': partial,
        'not_executed': not_executed,
        'runtime_capabilities': {
            'frozen_inference': {'status': 'supported',
                                 'explanation': 'JavaScript mirror of NeuroWeaveBrain.frozen_step and the raw readout.'},
            'continued_learning': {'status': 'not_available',
                                   'explanation': 'Plasticity runs only in the NumPy runtime (learn=True).'},
            'live_sensory_operation': {'status': 'not_available',
                                       'explanation': 'The page never requests camera or microphone access; '
                                                      'sensory channels are typed in. Raw media is never stored.'},
            'full_simulation': {'status': 'not_available',
                                'explanation': 'Ecology, agent, neuromodulation, prediction hierarchy and memory '
                                               'ecology are not executed in the page.'},
            'exported_memory': {'status': 'not_available',
                                'explanation': 'Memory ecology state is not embedded in the page.'},
            'persistent_model_state': {'status': 'not_available',
                                       'explanation': 'The page keeps state in memory only; Reset restores the '
                                                      'embedded state.'},
        },
        'evidence_tier': EVIDENCE_TIER,
    }


def annotate_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of ``package`` with ``capabilities.frozen_inference.standalone`` filled in and re-fingerprinted.

    Capability *statuses* are unchanged, so the package still validates."""
    pkg = _require_package(package)
    has_micro = _load_micro(pkg) is not None
    pkg['capabilities']['frozen_inference']['standalone'] = standalone_capabilities(pkg.get('mechanisms') or [],
                                                                                   has_micro)
    pkg.pop('integrity', None)
    pkg['integrity'] = {'algorithm': 'sha256', 'fingerprint': package_fingerprint(pkg)}
    report = validate_package(pkg)
    if not report['valid']:  # pragma: no cover - defensive
        raise ValueError('annotated package failed validation: ' + '; '.join(report['errors'][:4]))
    return pkg


# =========================================================================== frozen model
def standalone_model(package: Mapping[str, Any]) -> dict[str, Any]:
    """The compact JSON model the page embeds (everything the JS mirror needs)."""
    pkg = _require_package(package)
    brain = _load_brain(pkg)
    micro = _load_micro(pkg)
    spec, n, f = brain.spec, brain.n, brain.flags
    syn, cells = brain.synapses, brain.cells
    d = STANDALONE_INPUT_DIM
    caps = standalone_capabilities(spec.mechanisms, micro is not None)
    model: dict[str, Any] = {
        'schema': STANDALONE_SCHEMA, 'version': 1, 'evidence_tier': EVIDENCE_TIER,
        'n': int(n), 'input_dim': d, 'task_dim': TASK_INPUT_DIM, 'sensory_dim': SENSORY_DIM,
        'flags': sorted(f), 'sparse_k': max(1, int(round(n * spec.sparsity))), 'tile_size': int(spec.tile_size),
        'base_threshold': BASE_THRESHOLD, 'bounds': {k: list(_BOUNDS[k]) for k in STATE_FIELDS},
        'cells': {k: getattr(cells, k).tolist() for k in STATE_FIELDS},
        'region': cells.region.tolist(), 'n_regions': len(brain.regions),
        'kind_gain': [float(_KIND_INPUT_GAIN.get(brain.regions[r][1], 0.5)) for r in cells.region],
        'projection': brain._projection(d)[:n].tolist(),
        'hash': brain._hash_matrix(d).tolist() if 'flyhash_projection' in f else None,
        'hash_winners': HASH_WINNERS, 'hash_channels': HASH_CHANNELS,
        'structural_weight': brain._structural_weight().tolist() if 'structural_encoder' in f else None,
        'synapses': {'src': syn.src.tolist(), 'dst': syn.dst.tolist(), 'w': syn.w.tolist(),
                     'delay': syn.delay.tolist(), 'compartment': syn.compartment.tolist(),
                     'myelin': syn.myelin.tolist(), 'hist': syn.hist.tolist(), 'hist_count': int(syn.hist_count),
                     'route_gain': syn.route_gain.tolist()},
        'readout': {'w': brain.readout_w.tolist(), 'b': float(brain.readout_b)},
        'start_step': int(brain.step_count),
        'micro': None,
        'selective': {'threshold': default_selective_threshold(f), 'label': SELECTIVE_LABEL,
                      'score': 'margin*(1-clip(router_entropy,0,1)); margin=|2p-1|'},
        'meta': {'format': pkg.get('format'), 'version': pkg.get('version'),
                 'fingerprint': (pkg.get('integrity') or {}).get('fingerprint'),
                 'counts': pkg.get('counts'), 'mechanisms': list(spec.mechanisms),
                 'topology': spec.topology, 'seed': spec.seed,
                 'task_labels': [f'task_{i}' for i in range(TASK_INPUT_DIM)],
                 'sensory_labels': list(SENSORY_CHANNEL_IDS),
                 'package_capabilities': {k: {'status': v.get('status'), 'explanation': v.get('explanation')}
                                          for k, v in (pkg.get('capabilities') or {}).items() if k in CAPABILITIES},
                 'standalone': caps},
    }
    if micro is not None:
        model['micro'] = {'input_dim': micro.input_dim, 'output_dim': micro.output_dim, 'hidden': micro.hidden,
                          'experts': micro.n_experts, 'top_k': micro.top_k,
                          'enabled_in_spec': 'micro_collective' in f,
                          **{k: v.tolist() for k, v in micro._params().items()}}
    return model


def frozen_reference(package: Mapping[str, Any], inputs: Sequence[tuple[Any, Any]],
                     selective_threshold: float | None = None) -> list[dict[str, Any]]:
    """Python oracle: run ``frozen_step`` (+ readout, + Micro-Collective predict) for each (task, sensory).

    Each record also carries ``selective`` (see :func:`selective_outcome`); the threshold defaults to
    the one the standalone page embeds.
    """
    pkg = _require_package(package)
    brain = _load_brain(pkg)
    micro = _load_micro(pkg)
    thr = default_selective_threshold(brain.flags) if selective_threshold is None else float(selective_threshold)
    out: list[dict[str, Any]] = []
    for task, sens in inputs:
        t = np.asarray(task, dtype=np.float64).ravel()
        s = np.asarray(sens, dtype=np.float64).ravel()
        if t.shape != (TASK_INPUT_DIM,) or s.shape != (SENSORY_DIM,):
            raise ValueError(f'inputs must be {TASK_INPUT_DIM} task and {SENSORY_DIM} sensory values')
        y = brain.frozen_step(np.concatenate([t, s]))
        p = float(brain.predict())
        rec: dict[str, Any] = {'step': int(brain.step_count), 'output': y.tolist(), 'readout_probability': p,
                               'uncertainty': {'readout_ambiguity': 1.0 - abs(2.0 * p - 1.0), 'router_entropy': None},
                               'micro': None}
        if micro is not None:
            pred, routing = micro.predict(s)
            probs = np.asarray(routing['probabilities'])
            e = probs.size
            nz = probs[probs > 0]
            rec['uncertainty']['router_entropy'] = float(-(nz * np.log(nz)).sum() / math.log(e)) if e > 1 else 0.0
            rec['micro'] = {'prediction': pred.tolist(), 'experts': routing['experts'], 'weights': routing['weights'],
                            'probabilities': routing['probabilities']}
        rec['selective'] = selective_outcome(p, rec['uncertainty']['router_entropy'], thr)
        out.append(rec)
    return out


# =========================================================================== JavaScript
STANDALONE_ENGINE_JS = r"""
var NeuroWeaveStandalone = (function () {
  'use strict';
  var FIELDS = ['proximal','distal','apical','basal','primary','secondary','fast','slow','context','integrator',
    'burst','gating','membrane','threshold','refractory','spike','calcium','adaptation','trace','hillock',
    'branch_a','branch_b','terminal','release','homeo_gain','astrocyte','myelin','utility','consolidation',
    'pred_estimate','pred_error','surprise','tm_fast','tm_mid','tm_slow','output'];
  var Q88_FIELDS = ['fast','slow','context','membrane','integrator','proximal','distal','apical','basal'];
  function clip(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }
  function rint(x) { var r = Math.round(x); if (r - x === 0.5 && r % 2 !== 0) { r -= 1; } return r; }
  function q88(x) { return clip(rint(x * 256.0) / 256.0, -128.0, 32767.0 / 256.0); }
  function f64(a, n, name) {
    if (!Array.isArray(a) || (n !== undefined && a.length !== n)) { throw new Error('model field ' + name + ' malformed'); }
    var out = new Float64Array(a.length);
    for (var i = 0; i < a.length; i++) {
      if (typeof a[i] !== 'number' || !isFinite(a[i])) { throw new Error('model field ' + name + ' not finite'); }
      out[i] = a[i];
    }
    return out;
  }
  function i32(a, n, name) { var f = f64(a, n, name); return Int32Array.from(f); }
  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) { s += a[i]; } return s / a.length; }

  function create(model) {
    if (!model || model.schema !== 'NeuroWeaveStandaloneModelV1') { throw new Error('unsupported standalone model'); }
    var n = model.n, rt = { model: model, n: n, flags: {}, cells: {}, step: model.start_step | 0 };
    rt.selectiveThreshold = (model.selective && typeof model.selective.threshold === 'number') ? model.selective.threshold : 0.0;
    model.flags.forEach(function (f) { rt.flags[f] = true; });
    FIELDS.forEach(function (k) { rt.cells[k] = f64(model.cells[k], n, k); });
    rt.region = i32(model.region, n, 'region');
    var s = model.synapses, e = s.src.length;
    rt.src = i32(s.src, e, 'src'); rt.dst = i32(s.dst, e, 'dst'); rt.w = f64(s.w, e, 'w');
    rt.delay = i32(s.delay, e, 'delay'); rt.comp = i32(s.compartment, e, 'compartment');
    rt.emyelin = f64(s.myelin, e, 'myelin');
    rt.hist = s.hist.map(function (h, i) { return f64(h, n, 'hist' + i); });
    rt.histCount = s.hist_count | 0;
    rt.routeGain = s.route_gain.map(function (r, i) { return f64(r, model.n_regions, 'route_gain' + i); });
    for (var j = 0; j < e; j++) {
      if (rt.src[j] < 0 || rt.src[j] >= n || rt.dst[j] < 0 || rt.dst[j] >= n || rt.comp[j] < 0 || rt.comp[j] > 3) {
        throw new Error('synapse endpoint out of range');
      }
    }
    return rt;
  }

  function contextAttention(rt, prev) {
    var n = rt.n, src = rt.src, dst = rt.dst, w = rt.w, E = src.length, att = new Float64Array(n);
    if (!E) { return att; }
    var score = new Float64Array(E), order = new Array(E), k;
    for (k = 0; k < E; k++) { score[k] = Math.abs(w[k] * prev[src[k]]); order[k] = k; }
    order.sort(function (a, b) {
      if (dst[a] !== dst[b]) { return dst[a] - dst[b]; }
      var na = -score[a], nb = -score[b];
      if (na < nb) { return -1; } if (na > nb) { return 1; } return a - b;
    });
    var keep = [], cur = -1, rank = 0;
    for (k = 0; k < E; k++) {
      var e = order[k];
      if (dst[e] !== cur) { cur = dst[e]; rank = 0; }
      if (rank < 8) { keep.push(e); }
      rank++;
    }
    var qmax = new Float64Array(n).fill(-Infinity), den = new Float64Array(n), num = new Float64Array(n);
    keep.forEach(function (e) { var q = 2.2 * score[e]; if (q > qmax[dst[e]]) { qmax[dst[e]] = q; } });
    keep.forEach(function (e) {
      var ex = Math.exp(2.2 * score[e] - qmax[dst[e]]);
      den[dst[e]] += ex; num[dst[e]] += ex * prev[src[e]];
    });
    for (var i = 0; i < n; i++) { if (den[i] > 0) { att[i] = num[i] / den[i]; } }
    return att;
  }

  function argsortDesc(vals) {
    var idx = []; for (var i = 0; i < vals.length; i++) { idx.push(i); }
    idx.sort(function (a, b) { var na = -vals[a], nb = -vals[b]; if (na < nb) { return -1; } if (na > nb) { return 1; } return a - b; });
    return idx;
  }

  function readout(rt) {
    var m = rt.model, c = rt.cells, a = c.output, n = rt.n, i;
    var mu = mean(a), absA = new Float64Array(n), dev = new Float64Array(n), tf = new Float64Array(n),
        ts = new Float64Array(n), tc = new Float64Array(n), act = new Float64Array(n);
    for (i = 0; i < n; i++) {
      absA[i] = Math.abs(a[i]); dev[i] = Math.abs(a[i] - mu) * Math.abs(a[i] - mu);
      tf[i] = Math.tanh(c.fast[i]); ts[i] = Math.tanh(c.slow[i]); tc[i] = Math.tanh(c.context[i]);
      act[i] = Math.abs(a[i]) > 0.1 ? 1 : 0;
    }
    var feats = [mu, mean(absA), Math.sqrt(mean(dev)), mean(tf), mean(ts), mean(tc), mean(c.trace), mean(act)];
    var nr = m.n_regions, cnt = new Float64Array(nr), rm = new Float64Array(nr), ra = new Float64Array(nr);
    for (i = 0; i < n; i++) { cnt[rt.region[i]] += 1; rm[rt.region[i]] += a[i]; ra[rt.region[i]] += Math.abs(a[i]); }
    for (i = 0; i < nr; i++) { feats.push(rm[i] / Math.max(1, cnt[i])); }
    for (i = 0; i < nr; i++) { feats.push(ra[i] / Math.max(1, cnt[i])); }
    var dot = 0; for (i = 0; i < feats.length; i++) { dot += m.readout.w[i] * feats[i]; }
    var z = clip(m.readout.b + dot, -40.0, 40.0);
    return 1.0 / (1.0 + Math.exp(-z));
  }

  function microPredict(mc, sens) {
    var D = mc.input_dim, E = mc.experts, H = mc.hidden, O = mc.output_dim, x = new Float64Array(D), e, i, j;
    for (i = 0; i < D; i++) { x[i] = clip(sens[i], -10.0, 10.0); }
    var logits = new Float64Array(E), mx = -Infinity;
    for (e = 0; e < E; e++) {
      var dot = 0; for (j = 0; j < D; j++) { dot += x[j] * mc.router_w[e][j]; }
      logits[e] = dot + mc.router_b[e] + mc.balance_bias[e];
      if (logits[e] > mx) { mx = logits[e]; }
    }
    var ez = new Float64Array(E), tot = 0;
    for (e = 0; e < E; e++) { ez[e] = Math.exp(logits[e] - mx); tot += ez[e]; }
    var probs = []; for (e = 0; e < E; e++) { probs.push(ez[e] / tot); }
    var chosen = argsortDesc(probs).slice(0, mc.top_k), selSum = 0, gates = new Float64Array(E);
    chosen.forEach(function (k) { selSum += probs[k]; });
    chosen.forEach(function (k) { gates[k] = probs[k] / selSum; });
    var y = new Float64Array(O);
    for (e = 0; e < E; e++) {
      var hid = new Float64Array(H);
      for (i = 0; i < H; i++) { var s = 0; for (j = 0; j < D; j++) { s += mc.w1[e][i][j] * x[j]; } hid[i] = Math.tanh(s + mc.b1[e][i]); }
      for (i = 0; i < O; i++) { var o = 0; for (j = 0; j < H; j++) { o += mc.w2[e][i][j] * hid[j]; } y[i] += gates[e] * (o + mc.b2[e][i]); }
    }
    var pred = []; for (i = 0; i < O; i++) { pred.push(clip(y[i], -10.0, 10.0)); }
    var ent = 0; probs.forEach(function (p) { if (p > 0) { ent -= p * Math.log(p); } });
    return { prediction: pred, experts: chosen, weights: chosen.map(function (k) { return gates[k]; }),
             probabilities: probs, entropy: E > 1 ? ent / Math.log(E) : 0.0 };
  }

  function step(rt, task, sensory) {
    var m = rt.model, n = rt.n, f = rt.flags, c = rt.cells, d = m.input_dim, i, e, k;
    if (!task || !sensory || task.length !== m.task_dim || sensory.length !== m.sensory_dim) {
      throw new Error('expected ' + m.task_dim + ' task and ' + m.sensory_dim + ' sensory values');
    }
    var x = new Float64Array(d), raw = Array.prototype.concat.call([], Array.from(task), Array.from(sensory));
    for (i = 0; i < d; i++) {
      var v = Number(raw[i]);
      if (!isFinite(v)) { throw new Error('inputs must be finite numbers'); }
      x[i] = clip(v, -10.0, 10.0);
    }
    var prev = Float64Array.from(c.output);
    // ---- project_input
    var ext = new Float64Array(n);
    for (i = 0; i < n; i++) {
      var row = m.projection[i], dot = 0;
      for (k = 0; k < d; k++) { dot += row[k] * x[k]; }
      var pol = (i % 2 === 0) ? 1.0 : -0.75;
      ext[i] = m.kind_gain[i] * (pol * x[i % d] + dot);
    }
    if (f.flyhash_projection) {
      var hc = m.hash_channels, actv = new Float64Array(hc), code = new Float64Array(hc);
      for (k = 0; k < hc; k++) { var hs = 0; for (i = 0; i < d; i++) { hs += m.hash[k][i] * x[i]; } actv[k] = hs; }
      argsortDesc(actv).slice(0, m.hash_winners).forEach(function (j) { code[j] = 1.0; });
      for (i = 0; i < n; i++) { ext[i] = ext[i] + 0.16 * (code[i % hc] - m.hash_winners / hc); }
    }
    var gate = f.uncertainty_gating ? clip(1.0 / (1.0 + 2.2 * 0.0), 0.58, 1.0) : 1.0;
    // ---- transmit
    var src = rt.src, dst = rt.dst, E = src.length, pre = new Float64Array(E), signal = new Float64Array(E);
    for (e = 0; e < E; e++) { pre[e] = prev[src[e]]; }
    if (f.synaptic_delays) {
      for (e = 0; e < E; e++) {
        var de = rt.delay[e];
        if (f.adaptive_conduction) { de = Math.max(1, de - (rt.emyelin[e] > 0.78 ? 1 : 0)); }
        for (k = 2; k <= 3; k++) { if (rt.histCount >= k && de === k) { pre[e] = rt.hist[k - 1][src[e]]; } }
      }
    }
    var w = rt.w;
    if (f.quantised_synapses) {
      var scale = 1.0, wq = new Float64Array(E);
      if (E) { scale = 0; for (e = 0; e < E; e++) { scale = Math.max(scale, Math.abs(w[e])); } }
      if (scale > 0) {
        for (e = 0; e < E; e++) {
          var q = clip(rint(w[e] / scale * 127.0), -127, 127);
          if (q === 0 && w[e] !== 0) { q = w[e] > 0 ? 1 : -1; }
          wq[e] = q * (scale / 127.0);
        }
      }
      w = wq;
    }
    for (e = 0; e < E; e++) {
      var sg = pre[e] * w[e];
      if (f.adaptive_routing) {
        sg = sg * clip(0.86 + 0.24 * Math.tanh(Math.abs(c.gating[dst[e]]) + 0.3 * Math.abs(c.apical[dst[e]]) - 0.12), 0.62, 1.14);
      }
      if (f.uncertainty_gating) { sg = sg * clip(gate, 0.58, 1.0); }
      if (f.multi_region_routing) { sg = sg * rt.routeGain[rt.region[src[e]]][rt.region[dst[e]]]; }
      if (f.adaptive_conduction) { sg = sg * (0.85 + 0.25 * rt.emyelin[e]); }
      if (m.structural_weight && f.structural_encoder) { sg = sg * m.structural_weight[src[e]]; }
      signal[e] = sg;
    }
    var acc = new Float64Array(4 * n);
    if (f.tiled_execution && m.tile_size) {
      var tile = Math.max(1, m.tile_size | 0);
      for (var base = 0; base < E; base += tile) {
        var part = new Float64Array(4 * n), end = Math.min(E, base + tile);
        for (e = base; e < end; e++) { part[rt.comp[e] * n + dst[e]] += signal[e]; }
        for (i = 0; i < 4 * n; i++) { acc[i] += part[i]; }
      }
    } else {
      for (e = 0; e < E; e++) { acc[rt.comp[e] * n + dst[e]] += signal[e]; }
    }
    var dp = acc.subarray(0, n), dd = acc.subarray(n, 2 * n), da = Float64Array.from(acc.subarray(2 * n, 3 * n)),
        db = acc.subarray(3 * n, 4 * n);
    if (f.context_attention) {
      var att = contextAttention(rt, prev);
      for (i = 0; i < n; i++) { da[i] = da[i] + 0.3 * att[i]; }
    }
    // ---- cells.step with neutral controls (gain 1, threshold_shift 0, attention 1, plasticity_scale 1)
    var gain = 1.0, tshift = 0.0, attention = 1.0, pscale = 1.0, membrane = !!f.membrane_dynamics;
    var a = new Float64Array(n), spike = new Float64Array(n), B = m.base_threshold;
    for (i = 0; i < n; i++) {
      var u = ext[i], inc = dp[i] + dd[i] + da[i] + db[i], prevOut = c.output[i];
      var apicalIn = attention * da[i];
      c.proximal[i] = 0.58 * c.proximal[i] + 0.42 * (dp[i] + 0.34 * u);
      c.basal[i] = 0.70 * c.basal[i] + 0.30 * (db[i] + 0.20 * u);
      c.distal[i] = 0.94 * c.distal[i] + 0.06 * dd[i];
      c.apical[i] = 0.86 * c.apical[i] + 0.14 * apicalIn;
      var eff;
      if (f.dendritic_compartments) {
        var apicalGate = 1.0 + 0.5 * Math.tanh(2.0 * c.apical[i]);
        eff = c.proximal[i] + apicalGate * c.basal[i] + c.distal[i] + 0.25 * c.apical[i];
      } else { eff = inc; }
      c.fast[i] = 0.58 * c.fast[i] + u + 0.52 * eff;
      c.slow[i] = 0.94 * c.slow[i] + 0.14 * u + 0.18 * eff;
      c.context[i] = 0.69 * c.context[i] + attention * eff;
      var p;
      if (f.multi_soma) {
        var fused = 0.35 * c.fast[i] + 0.18 * c.slow[i] + 0.19 * c.context[i] + 0.13 * c.basal[i]
          + 0.08 * c.apical[i] + 0.07 * c.trace[i];
        c.integrator[i] = 0.91 * c.integrator[i] + 0.09 * fused;
        p = fused + 0.1 * c.integrator[i];
      } else { p = 0.84 * c.fast[i] + 0.16 * eff; }
      if (f.extended_somas) {
        c.primary[i] = 0.5 * c.primary[i] + 0.5 * (u + eff);
        c.secondary[i] = 0.97 * c.secondary[i] + 0.03 * Math.tanh(c.primary[i]);
        c.burst[i] = 0.72 * c.burst[i] + 0.28 * Math.max(0.0, c.primary[i] - (c.threshold[i] + tshift));
        c.gating[i] = clip(0.8 * c.gating[i] + 0.2 * (1.0 + 0.35 * Math.tanh(c.secondary[i]) - 0.1 * Math.tanh(c.burst[i])), 0.65, 1.35);
        p = c.gating[i] * (p + 0.25 * c.secondary[i] + 0.2 * c.burst[i]);
      }
      if (f.hierarchical_temporal_memory) {
        p = p + 0.045 * (c.tm_fast[i] - c.tm_mid[i]) + 0.035 * (c.tm_mid[i] - c.tm_slow[i]);
      }
      if (f.predictive_coding) {
        var prediction = 0.58 * c.pred_estimate[i] + 0.27 * c.fast[i] + 0.15 * c.slow[i];
        var err = p - prediction;
        c.pred_error[i] = err;
        c.surprise[i] = clip(0.94 * c.surprise[i] + 0.06 * Math.abs(err), 0.0, 2.0);
        p = p + clip(err, -1.5, 1.5) * 0.055 / (1.0 + c.surprise[i]);
        c.pred_estimate[i] = 0.91 * prediction + 0.09 * p;
      }
      var sp = 0.0;
      if (membrane) {
        c.adaptation[i] = clip(0.995 * c.adaptation[i] + 0.005 * Math.abs(prevOut), 0.0, 1.0);
        c.threshold[i] = clip(clip(B + 0.18 * c.adaptation[i], 0.42, 0.78) + tshift, 0.05, 1.6);
        c.membrane[i] = 0.78 * c.membrane[i] + 0.22 * p - 0.06 * c.adaptation[i];
        c.refractory[i] = Math.max(0.0, c.refractory[i] - 1.0);
        var fired = (c.membrane[i] > c.threshold[i]) && (c.refractory[i] === 0);
        if (fired) { c.membrane[i] = 0.26 * c.membrane[i]; }
        var hold = c.refractory[i] > 0;
        if (fired) { c.refractory[i] = 2.0; }
        sp = fired ? 1.0 : 0.0;
        c.calcium[i] = clip(0.965 * c.calcium[i] + 0.075 * sp + 0.008 * Math.abs(p), 0.0, 4.0);
        p = p + 0.15 * sp - 0.08 * c.calcium[i];
        if (hold) { p = 0.7 * p; }
      } else {
        c.threshold[i] = clip(B + tshift, 0.05, 1.6);
      }
      c.hillock[i] = 0.66 * c.hillock[i] + 0.34 * (sp > 0 ? 1.0 : Math.tanh(p));
      if (f.axonal_branches) {
        c.branch_a[i] = 0.72 * c.branch_a[i] + 0.28 * (c.hillock[i] + 0.2 * c.apical[i]);
        c.branch_b[i] = 0.84 * c.branch_b[i] + 0.16 * (c.hillock[i] - 0.15 * c.distal[i]);
        c.terminal[i] = 0.6 * c.branch_a[i] + 0.4 * c.branch_b[i];
        c.release[i] = Math.tanh(c.terminal[i]) * (0.4 + 0.6 * c.myelin[i]);
        p = p + 0.12 * c.release[i];
        c.myelin[i] = clip(c.myelin[i] + pscale * (0.0005 * Math.abs(c.hillock[i]) - 0.00008), 0.2, 0.98);
      }
      var ai = Math.tanh(gain * p - tshift);
      if (membrane && sp > 0) { ai = clip(ai + 0.12 * c.hillock[i], -1.0, 1.0); }
      a[i] = ai; spike[i] = sp;
    }
    var sk = m.sparse_k | 0;
    if (f.sparse_code && sk > 0 && sk < n) {
      var absA = new Float64Array(n); for (i = 0; i < n; i++) { absA[i] = Math.abs(a[i]); }
      var mask = new Uint8Array(n);
      argsortDesc(absA).slice(0, sk).forEach(function (j) { mask[j] = 1; });
      for (i = 0; i < n; i++) { if (!mask[i]) { a[i] = 0.08 * a[i]; } }
    }
    for (i = 0; i < n; i++) {
      var po = c.output[i];
      if (f.homeostatic_governor) {
        c.homeo_gain[i] = clip(c.homeo_gain[i] + 0.02 * pscale * (0.2 - Math.abs(po)), 0.5, 1.5);
        a[i] = a[i] * c.homeo_gain[i];
      }
      if (f.astrocyte_regulation) {
        c.astrocyte[i] = clip(0.97 * c.astrocyte[i] + 0.03 / (1.0 + 1.6 * Math.abs(po)), 0.3, 1.2);
        a[i] = a[i] * clip(c.astrocyte[i], 0.58, 1.12);
      }
      a[i] = clip(a[i], -1.0, 1.0);
    }
    if (f.q88_state) {
      for (i = 0; i < n; i++) { a[i] = q88(a[i]); }
      Q88_FIELDS.forEach(function (name) { var arr = c[name]; for (var j = 0; j < n; j++) { arr[j] = q88(arr[j]); } });
    }
    if (!membrane) { for (i = 0; i < n; i++) { spike[i] = a[i] > c.threshold[i] ? 1.0 : 0.0; } }
    var rate = 0.008 * Math.min(pscale, 4.0);
    for (i = 0; i < n; i++) {
      c.spike[i] = spike[i]; c.output[i] = a[i];
      c.trace[i] = clip((1.0 - rate) * c.trace[i] + rate * Math.abs(a[i]), 0.0, 1.0);
      if (f.hierarchical_temporal_memory) {
        c.tm_fast[i] = 0.62 * c.tm_fast[i] + 0.38 * Math.abs(a[i]);
        c.tm_mid[i] = 0.94 * c.tm_mid[i] + 0.06 * c.tm_fast[i];
        c.tm_slow[i] = 0.994 * c.tm_slow[i] + 0.006 * c.tm_mid[i];
      }
      c.utility[i] = 0.985 * c.utility[i] + 0.015 * Math.abs(a[i] * Math.tanh(c.fast[i]));
    }
    FIELDS.forEach(function (name) {
      var lo = m.bounds[name][0], hi = m.bounds[name][1], arr = c[name];
      for (var j = 0; j < n; j++) {
        var val = arr[j];
        if (val !== val) { val = 0.0; } else if (val === Infinity) { val = hi; } else if (val === -Infinity) { val = lo; }
        arr[j] = clip(val, lo, hi);
      }
    });
    // ---- delay ring buffer push
    rt.hist = [Float64Array.from(c.output), rt.hist[0], rt.hist[1]];
    rt.histCount = Math.min(3, rt.histCount + 1);
    rt.step += 1;
    var prob = readout(rt), res = {
      step: rt.step, output: Array.from(c.output), readout_probability: prob,
      uncertainty: { readout_ambiguity: 1.0 - Math.abs(2.0 * prob - 1.0), router_entropy: null }, micro: null
    };
    if (m.micro) {
      var mp = microPredict(m.micro, sensory);
      res.uncertainty.router_entropy = mp.entropy;
      res.micro = { prediction: mp.prediction, experts: mp.experts, weights: mp.weights, probabilities: mp.probabilities };
    }
    res.selective = selective(prob, res.uncertainty.router_entropy, rt.selectiveThreshold);
    return res;
  }

  // Heuristic selective score (NOT calibrated confidence); the raw probability is never altered.
  function selective(prob, entropy, threshold) {
    var thr = Number(threshold);
    if (!(prob === prob) || !isFinite(prob) || !(thr >= 0 && thr <= 1)) { throw new Error('threshold must be in [0, 1]'); }
    var margin = Math.abs(2.0 * prob - 1.0);
    var risk = (entropy === null || entropy === undefined) ? 0.0 : Math.min(Math.max(entropy, 0.0), 1.0);
    var score = margin * (1.0 - risk), abstain = score < thr;
    return { score: score, margin: margin, risk: risk, risk_source: (entropy === null || entropy === undefined) ? 'none' : 'router_entropy',
      threshold: thr, abstain: abstain, decision: abstain ? 'abstain_review' : 'predict', raw_probability: prob,
      label: 'heuristic, not calibrated' };
  }
  function setThreshold(rt, value) {
    var thr = Number(value);
    if (!(thr >= 0 && thr <= 1)) { throw new Error('threshold must be in [0, 1]'); }
    rt.selectiveThreshold = thr; return thr;
  }

  return { create: create, step: step, reset: function (rt) { var t = rt.selectiveThreshold, r = create(rt.model); r.selectiveThreshold = t; return r; },
    q88: q88, rint: rint, selective: selective, setThreshold: setThreshold };
})();
if (typeof module !== 'undefined' && module.exports) { module.exports = NeuroWeaveStandalone; }
"""

_UI_JS = r"""
(function () {
  'use strict';
  var model = JSON.parse(document.getElementById('nw-model').textContent);
  var rt = NeuroWeaveStandalone.create(model);
  var history = [];
  function inputs(prefix, count) {
    var out = [];
    for (var i = 0; i < count; i++) { out.push(Number(document.getElementById(prefix + i).value)); }
    return out;
  }
  function fmt(v) { return (typeof v === 'number' && isFinite(v)) ? v.toFixed(6) : 'n/a'; }
  function setText(id, text) { document.getElementById(id).textContent = text; }
  function draw() {
    var cv = document.getElementById('nw-history'), ctx = cv.getContext('2d');
    var w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent') || '#2563eb';
    ctx.lineWidth = 2; ctx.beginPath();
    var pts = history.slice(-120);
    pts.forEach(function (p, i) {
      var x = pts.length > 1 ? i * (w - 4) / (pts.length - 1) + 2 : w / 2, y = h - 2 - p * (h - 4);
      if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
    });
    ctx.stroke();
  }
  function show(res) {
    setText('nw-step', String(res.step));
    setText('nw-prob', fmt(res.readout_probability));
    var sel = res.selective;
    setText('nw-selective', fmt(sel.score) + '  (margin ' + fmt(sel.margin) + ', risk ' + fmt(sel.risk) + ' from ' + sel.risk_source + ')');
    setText('nw-decision', sel.abstain ? 'ABSTAIN \u00b7 REVIEW (score below threshold ' + sel.threshold + '; raw probability still shown above)' :
      'predict (score at or above threshold ' + sel.threshold + ')');
    setText('nw-ambiguity', fmt(res.uncertainty.readout_ambiguity));
    setText('nw-entropy', res.uncertainty.router_entropy === null ? 'n/a (no Micro-Collective)' : fmt(res.uncertainty.router_entropy));
    var out = res.output, mu = 0; out.forEach(function (v) { mu += v; }); mu /= out.length;
    setText('nw-mean', fmt(mu));
    var top = out.map(function (v, i) { return [Math.abs(v), i, v]; }).sort(function (a, b) { return b[0] - a[0] || a[1] - b[1]; }).slice(0, 8);
    setText('nw-top', top.map(function (t) { return '#' + t[1] + ': ' + t[2].toFixed(4); }).join('   '));
    setText('nw-micro', res.micro ? res.micro.prediction.map(function (v) { return v.toFixed(3); }).join(', ') +
      '  (experts ' + res.micro.experts.join(',') + ')' : 'not available');
    history.push(res.readout_probability); draw();
  }
  function doStep(times) {
    try {
      var t = inputs('nw-task-', model.task_dim), s = inputs('nw-sens-', model.sensory_dim), res;
      for (var k = 0; k < times; k++) { res = NeuroWeaveStandalone.step(rt, t, s); }
      setText('nw-error', ''); show(res);
    } catch (e) { setText('nw-error', String(e && e.message ? e.message : e)); }
  }
  var thrInput = document.getElementById('nw-abstain-threshold');
  thrInput.value = String(rt.selectiveThreshold);
  thrInput.addEventListener('change', function () {
    try { NeuroWeaveStandalone.setThreshold(rt, thrInput.value); setText('nw-error', ''); }
    catch (e) { setText('nw-error', String(e && e.message ? e.message : e)); thrInput.value = String(rt.selectiveThreshold); }
  });
  document.getElementById('nw-step-btn').addEventListener('click', function () { doStep(1); });
  document.getElementById('nw-run-btn').addEventListener('click', function () { doStep(10); });
  document.getElementById('nw-reset-btn').addEventListener('click', function () {
    rt = NeuroWeaveStandalone.reset(rt); history = []; draw();
    ['nw-prob', 'nw-ambiguity', 'nw-entropy', 'nw-mean', 'nw-top', 'nw-micro', 'nw-selective', 'nw-decision'].forEach(function (id) { setText(id, '-'); });
    setText('nw-step', String(rt.step)); setText('nw-error', '');
  });
  setText('nw-step', String(rt.step));
})();
"""

_CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#5d5d58;--card:#ffffff;--line:#d9d9d3;--accent:#2563eb;--warn:#9a3412;--ok:#166534}
@media (prefers-color-scheme: dark){:root{--bg:#141413;--fg:#ececea;--muted:#a3a39d;--card:#1f1f1d;--line:#3a3a36;--accent:#7aa2ff;--warn:#fdba74;--ok:#86efac}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 48px}h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.1rem;margin:28px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted)}input{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;margin-right:6px}
button.secondary{background:transparent;color:var(--accent)}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}code{font-size:12px;overflow-wrap:anywhere}
.status-supported{color:var(--ok)}.status-not_available,.status-unsupported{color:var(--warn)}
.kv{display:grid;grid-template-columns:200px 1fr;gap:4px 12px}.kv>div{min-width:0;overflow-wrap:anywhere}#nw-error{color:var(--warn)}canvas{width:100%;height:80px}
.table-wrap{overflow-x:auto}
@media (max-width:560px){.kv{grid-template-columns:1fr}.kv>div:nth-child(odd){color:var(--muted);margin-top:6px}}
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True).replace('\u2028', '&#8232;').replace('\u2029', '&#8233;')


def _cap_rows(caps: Mapping[str, Any]) -> str:
    rows = []
    for name, entry in caps.items():
        status = entry.get('status') if isinstance(entry, Mapping) else None
        expl = entry.get('explanation') if isinstance(entry, Mapping) else ''
        rows.append(f'<tr><td>{_e(name)}</td><td class="status-{_e(status)}">{_e(status)}</td><td>{_e(expl)}</td></tr>')
    return ''.join(rows)


def standalone_html(package: Mapping[str, Any]) -> str:
    """Render the self-contained offline HTML page for a validated package."""
    model = standalone_model(package)
    meta = model['meta']
    st = meta['standalone']
    task_inputs = ''.join(f'<label>{_e(lbl)}<input id="nw-task-{i}" type="number" step="any" value="0"></label>'
                          for i, lbl in enumerate(meta['task_labels']))
    sens_inputs = ''.join(f'<label>{_e(lbl)}<input id="nw-sens-{i}" type="number" step="any" value="0"></label>'
                          for i, lbl in enumerate(meta['sensory_labels']))
    executed = ', '.join(_e(m) for m in st['executed']) or 'none (all mechanisms off: v0.1 baseline update)'
    not_exec = ''.join(f'<tr><td><code>{_e(x["id"])}</code></td><td>partially executed</td><td>{_e(x["reason"])}</td></tr>'
                       for x in st['partial'])
    not_exec += ''.join(f'<tr><td><code>{_e(x["id"])}</code></td><td>not executed</td><td>{_e(x["reason"])}</td></tr>'
                        for x in st['not_executed'])
    if not not_exec:
        not_exec = '<tr><td colspan="3">Every enabled mechanism is executed by the standalone runner.</td></tr>'
    counts = meta.get('counts') or {}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<title>NEUROWEAVE Frozen Runner</title><style>{_CSS}</style></head>
<body><main>
<h1>NEUROWEAVE frozen runner</h1>
<p class="muted">Offline frozen inference for a {_e(meta['format'])} {_e(meta['version'])} package
(fingerprint <code>{_e(meta['fingerprint'])}</code>). Evidence tier: <b>{_e(model['evidence_tier'])}</b>.
Synthetic computational abstraction; no biological, hardware or performance claim. No network access, no camera, no microphone.</p>
<div class="card kv"><div>Neurons / synapses</div><div>{_e(counts.get('neurons'))} / {_e(counts.get('synapses'))}</div>
<div>Topology / seed</div><div>{_e(meta['topology'])} / {_e(meta['seed'])}</div>
<div>Brain input</div><div>{_e(st['input_layout']['brain_input'])}</div>
<div>Controls</div><div>{_e(st['controls'])}</div>
<div>Micro-Collective</div><div>{_e(st['micro_collective'])}</div>
<div>Executed mechanisms</div><div>{executed}</div></div>
<h2>Inputs</h2>
<div class="card"><div class="muted">Task (8)</div><div class="grid">{task_inputs}</div></div>
<div class="card"><div class="muted">Sensory (16, typed values; nothing is captured)</div><div class="grid">{sens_inputs}</div></div>
<div><button id="nw-step-btn" type="button">Step</button><button id="nw-run-btn" type="button">Run 10</button>
<button id="nw-reset-btn" type="button" class="secondary">Reset</button> <span id="nw-error" role="alert"></span></div>
<h2>Readout</h2>
<div class="card kv"><div>Step</div><div id="nw-step">-</div>
<div>Raw readout probability (always shown)</div><div id="nw-prob">-</div>
<div>Selective score (heuristic, not calibrated)</div><div id="nw-selective">-</div>
<div>Outcome</div><div id="nw-decision">-</div>
<div><label for="nw-abstain-threshold" style="color:inherit;font-size:inherit">Abstain threshold (0 = never abstain)</label></div><div><input id="nw-abstain-threshold" type="number" min="0" max="1" step="0.01" value="0"></div>
<div>Uncertainty proxy: readout ambiguity 1-|2p-1|</div><div id="nw-ambiguity">-</div>
<div>Uncertainty proxy: router entropy</div><div id="nw-entropy">-</div>
<div>Mean output</div><div id="nw-mean">-</div>
<div>Largest |output| units</div><div id="nw-top">-</div>
<div>Micro-Collective prediction</div><div id="nw-micro">-</div></div>
<div class="card"><div class="muted">Readout probability history</div><canvas id="nw-history" width="960" height="80"></canvas></div>
<h2>Capability matrix</h2>
<div class="card table-wrap"><div class="muted">This page (standalone runtime)</div>
<table><tr><th>Capability</th><th>Status</th><th>Reason</th></tr>{_cap_rows(st['runtime_capabilities'])}</table></div>
<div class="card table-wrap"><div class="muted">Package (as declared by the package)</div>
<table><tr><th>Capability</th><th>Status</th><th>Explanation</th></tr>{_cap_rows(meta['package_capabilities'])}</table></div>
<h2 id="nw-not-executed">Not executed in standalone</h2>
<div class="card table-wrap"><table><tr><th>Mechanism</th><th>Status</th><th>Reason</th></tr>{not_exec}</table></div>
</main>
<script type="application/json" id="nw-model">{script_safe_json(model)}</script>
<script id="nw-engine">{STANDALONE_ENGINE_JS}</script>
<script id="nw-ui">{_UI_JS}</script>
</body></html>
"""
