"""Cross-scale read-only synthesis across live Frontier Pack observers.

Fuses micro-scale representational coding (``sparse_dictionary``), meso-scale temporal
synchrony & phase binding (``synchrony``, ``kuramoto_binding``), and macro-scale
functional/effective connectivity & causal emergence (``connectomics``,
``causal_emergence``) into an interpretable, deterministic regime report.

Contract rules:
- Strictly read-only: never mutates the hub, observers, brain, or RNG.
- Never fabricates missing metrics: inactive or warming-up observers yield ``None``
  in their corresponding indices and are explicitly excluded from composite averages.
- Deterministic SHA-256 ``digest`` over the normalized JSON report.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

from .harness import _jsonable
from .spec import EVIDENCE_TIER

__all__ = [
    'SYNTHESIS_SCHEMA',
    'REGIME_LABELS',
    'synthesize_frontier_observers',
    'verify_synthesis_digest',
]

SYNTHESIS_SCHEMA = 'FrontierCrossScaleSynthesisV1'
REGIME_LABELS: tuple[str, ...] = (
    'disabled',
    'warmup',
    'hypersynchronous',
    'metastable_emergent',
    'modular_structured',
    'desynchronized_exploratory',
    'balanced_active',
)


def _finite(val: Any) -> float | None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _clip01(val: float) -> float:
    return min(1.0, max(0.0, float(val)))


def _synthesis_digest(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != 'digest'}
    encoded = json.dumps(_jsonable(body), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def verify_synthesis_digest(report: Mapping[str, Any]) -> bool:
    if not isinstance(report, Mapping) or report.get('schema') != SYNTHESIS_SCHEMA:
        return False
    recorded = report.get('digest')
    if not isinstance(recorded, str) or len(recorded) != 64:
        return False
    return recorded == _synthesis_digest(report)


def synthesize_frontier_observers(hub_or_summary: Any) -> dict[str, Any]:
    """Synthesize micro, meso, and macro observer summaries into a cross-scale report.

    Accepts either a :class:`~synapse.frontier.runtime.FrontierObserverHub` or a
    summary dict previously returned by ``hub.summary()`` / ``hub.telemetry()``.
    """
    if hasattr(hub_or_summary, 'summary') and callable(hub_or_summary.summary):
        summary = hub_or_summary.summary()
    elif isinstance(hub_or_summary, Mapping):
        summary = dict(hub_or_summary)
    else:
        raise ValueError('expected a FrontierObserverHub or observer summary mapping')

    enabled = bool(summary.get('enabled', False))
    observed_steps = int(summary.get('observed_steps', 0) or 0)
    observers = summary.get('observers')
    if not isinstance(observers, Mapping):
        observers = {}

    # ------------------------------------------------------------------ Micro: Sparse Dictionary
    sae = observers.get('sparse_dictionary') if isinstance(observers.get('sparse_dictionary'), Mapping) else None
    sae_ready = bool(sae and sae.get('ready'))
    fvu = _finite(sae.get('fvu_ema')) if sae_ready else None
    coding_fidelity = round(_clip01(1.0 - fvu), 6) if fvu is not None else None
    l0 = _finite(sae.get('l0_ema')) if sae_ready else None
    feats = _finite(sae.get('features')) if sae_ready else None
    active_feature_fraction = (
        round(_clip01(l0 / max(1.0, feats)), 6)
        if (l0 is not None and feats is not None and feats > 0.0) else None
    )
    dead_feature_fraction = (
        round(_clip01(_finite(sae.get('dead_fraction')) or 0.0), 6)
        if sae_ready and _finite(sae.get('dead_fraction')) is not None else None
    )

    # ------------------------------------------------------------------ Meso: Synchrony & Kuramoto
    sync = observers.get('synchrony') if isinstance(observers.get('synchrony'), Mapping) else None
    sync_steps = int(sync.get('observed_steps', 0) or 0) if sync else 0
    global_synchrony = (
        round(_clip01(_finite(sync.get('global_synchrony_index')) or 0.0), 6)
        if sync and sync_steps > 0 and _finite(sync.get('global_synchrony_index')) is not None else None
    )
    synchrony_entropy = (
        round(max(0.0, _finite(sync.get('synchrony_entropy')) or 0.0), 6)
        if sync and sync_steps > 0 and _finite(sync.get('synchrony_entropy')) is not None else None
    )

    kura = observers.get('kuramoto_binding') if isinstance(observers.get('kuramoto_binding'), Mapping) else None
    kura_steps = int(kura.get('observed_steps', 0) or 0) if kura else 0
    phase_R_last = _finite(kura.get('R_last')) if (kura and kura_steps > 0) else None
    phase_R_mean = _finite(kura.get('R_mean')) if (kura and kura_steps > 0) else None
    metastability = _finite(kura.get('metastability_index')) if (kura and kura_steps > 0) else None
    chimera = _finite(kura.get('chimera_index')) if (kura and kura_steps > 0) else None
    phase_clusters = int(kura['phase_clusters']) if (kura and kura_steps > 0 and isinstance(kura.get('phase_clusters'), int)) else None

    # ------------------------------------------------------------------ Macro: Connectomics & Causal Emergence
    conn = observers.get('connectomics') if isinstance(observers.get('connectomics'), Mapping) else None
    conn_ready = bool(conn and conn.get('ready'))
    mean_abs_fc = round(_clip01(_finite(conn.get('mean_abs_fc')) or 0.0), 6) if (conn_ready and _finite(conn.get('mean_abs_fc')) is not None) else None
    fc_modularity = round(_finite(conn.get('fc_modularity')), 6) if (conn_ready and _finite(conn.get('fc_modularity')) is not None) else None
    ec_density = round(_clip01(_finite(conn.get('ec_density')) or 0.0), 6) if (conn_ready and _finite(conn.get('ec_density')) is not None) else None
    dfc_silhouette = round(_finite(conn.get('dfc_silhouette')), 6) if (conn_ready and _finite(conn.get('dfc_silhouette')) is not None) else None

    ce_obs = observers.get('causal_emergence') if isinstance(observers.get('causal_emergence'), Mapping) else None
    ce_deferred = bool(ce_obs and ce_obs.get('deferred'))
    ce_ready = bool(ce_obs and not ce_deferred and ce_obs.get('null_ready'))
    ce_bits = round(_finite(ce_obs.get('ce')), 6) if (ce_ready and _finite(ce_obs.get('ce')) is not None) else None
    ce_excess = round(_finite(ce_obs.get('ce_excess')), 6) if (ce_ready and _finite(ce_obs.get('ce_excess')) is not None) else None
    ce_z = round(_finite(ce_obs.get('ce_z')), 6) if (ce_ready and _finite(ce_obs.get('ce_z')) is not None) else None
    ce_p = round(_finite(ce_obs.get('ce_p')), 6) if (ce_ready and _finite(ce_obs.get('ce_p')) is not None) else None

    # ------------------------------------------------------------------ Cross-Scale Composite Index
    components: dict[str, float] = {}
    if coding_fidelity is not None:
        # Reward non-dead, high-fidelity sparse representations
        alive_factor = 1.0 - (dead_feature_fraction or 0.0)
        components['micro_coding'] = round(_clip01(coding_fidelity * alive_factor), 6)
    if phase_R_mean is not None:
        # Peak balance around intermediate synchrony (R ~ 0.55) with positive metastability
        balance = _clip01(1.0 - abs(phase_R_mean - 0.55) / 0.55)
        meta_boost = _clip01((metastability or 0.0) / 0.05)
        components['meso_phase_balance'] = round(_clip01(0.7 * balance + 0.3 * meta_boost), 6)
    elif global_synchrony is not None:
        components['meso_synchrony'] = round(_clip01(1.0 - abs(global_synchrony - 0.5) / 0.5), 6)
    if fc_modularity is not None:
        components['macro_modularity'] = round(_clip01(max(0.0, fc_modularity) / 0.5), 6)
    if ce_z is not None:
        # Smoothly map positive z-score against block null into [0, 1]
        components['macro_emergence'] = round(_clip01(1.0 / (1.0 + math.exp(-(ce_z - 1.0)))), 6)

    multiscale_index = (
        round(sum(components.values()) / len(components), 6)
        if components else None
    )

    # ------------------------------------------------------------------ Transparent Regime Classification
    if not enabled or not observers:
        regime = 'disabled'
        regime_reason = 'Frontier observer hub is disabled or has no active observer modules.'
    elif observed_steps < 8 or not components:
        regime = 'warmup'
        regime_reason = f'Insufficient observer support ({observed_steps} observed steps; awaiting ready micro/meso/macro metrics).'
    elif phase_R_mean is not None and phase_R_mean >= 0.88 and (metastability is None or metastability < 0.02):
        regime = 'hypersynchronous'
        regime_reason = f'High phase locking (R_mean={phase_R_mean:.3f}) with low metastability.'
    elif (
        phase_R_mean is not None
        and 0.35 <= phase_R_mean < 0.88
        and ((metastability is not None and metastability >= 0.02) or (ce_z is not None and ce_z >= 2.0))
    ):
        regime = 'metastable_emergent'
        regime_reason = 'Intermediate phase coherence paired with elevated metastability or null-calibrated macro emergence.'
    elif fc_modularity is not None and fc_modularity >= 0.20 and (coding_fidelity is None or coding_fidelity >= 0.20):
        regime = 'modular_structured'
        regime_reason = f'Modular functional organization (modularity={fc_modularity:.3f}) without phase saturation.'
    elif phase_R_mean is not None and phase_R_mean < 0.35:
        regime = 'desynchronized_exploratory'
        regime_reason = f'Low global phase locking (R_mean={phase_R_mean:.3f}) indicating weakly coupled exploratory dynamics.'
    else:
        regime = 'balanced_active'
        regime_reason = 'Active multi-scale dynamics within nominal bounds across ready observers.'

    # ------------------------------------------------------------------ Read-Only Diagnostic Advisories
    advisories: list[dict[str, str]] = []
    if dead_feature_fraction is not None and dead_feature_fraction > 0.25:
        advisories.append({
            'scale': 'micro_coding',
            'level': 'warn',
            'code': 'high_dead_features',
            'message': f'Sparse dictionary dead_fraction={dead_feature_fraction:.3f} (>0.25); monitor representation rank or continual backprop.',
        })
    if coding_fidelity is not None and coding_fidelity < 0.15:
        advisories.append({
            'scale': 'micro_coding',
            'level': 'info',
            'code': 'low_reconstruction_fidelity',
            'message': f'Sparse dictionary coding_fidelity={coding_fidelity:.3f}; dictionary is still adapting or under-parameterized.',
        })
    if regime == 'hypersynchronous':
        advisories.append({
            'scale': 'meso_dynamics',
            'level': 'warn',
            'code': 'hypersynchronous_locking',
            'message': 'Elevated Kuramoto R with minimal metastability; inspect homeostatic regulation or recurrent gain.',
        })
    if ce_z is not None and ce_p is not None and ce_z >= 2.0 and ce_p <= 0.05:
        advisories.append({
            'scale': 'macro_structure',
            'level': 'positive',
            'code': 'macro_emergence_above_null',
            'message': f'Macro causal emergence exceeds block-shuffled null (ce_z={ce_z:.2f}, ce_p={ce_p:.3f}, ce_excess={ce_excess:.4f} bits).',
        })
    if ce_deferred:
        advisories.append({
            'scale': 'macro_structure',
            'level': 'info',
            'code': 'causal_emergence_deferred',
            'message': 'Causal emergence summary was deferred on this step; call hub.summary() for an unconditional full check.',
        })
    if not advisories:
        advisories.append({
            'scale': 'cross_scale',
            'level': 'ok',
            'code': 'nominal',
            'message': 'All active observer scales are within nominal synthetic operating bounds.',
        })

    scale_coverage = {
        'micro_coding': sae_ready,
        'meso_dynamics': bool((sync and sync_steps > 0) or (kura and kura_steps > 0)),
        'macro_structure': bool(conn_ready or ce_ready),
    }

    report: dict[str, Any] = {
        'schema': SYNTHESIS_SCHEMA,
        'schema_version': 1,
        'evidence_tier': EVIDENCE_TIER,
        'read_only': True,
        'enabled': enabled,
        'observed_steps': observed_steps,
        'regime': regime,
        'regime_reason': regime_reason,
        'scale_coverage': scale_coverage,
        'active_scales': sum(1 for v in scale_coverage.values() if v),
        'indices': {
            'multiscale_integration_index': multiscale_index,
            'coding_fidelity': coding_fidelity,
            'active_feature_fraction': active_feature_fraction,
            'dead_feature_fraction': dead_feature_fraction,
            'global_synchrony': global_synchrony,
            'synchrony_entropy': synchrony_entropy,
            'phase_coherence_R_last': None if phase_R_last is None else round(phase_R_last, 6),
            'phase_coherence_R_mean': None if phase_R_mean is None else round(phase_R_mean, 6),
            'metastability_index': None if metastability is None else round(metastability, 6),
            'chimera_index': None if chimera is None else round(chimera, 6),
            'phase_clusters': phase_clusters,
            'mean_abs_fc': mean_abs_fc,
            'fc_modularity': fc_modularity,
            'ec_density': ec_density,
            'dfc_silhouette': dfc_silhouette,
            'causal_emergence_bits': ce_bits,
            'causal_emergence_excess': ce_excess,
            'causal_emergence_z': ce_z,
            'causal_emergence_p': ce_p,
        },
        'index_components': components,
        'advisories': advisories,
        'non_claim': 'Synthetic cross-scale observer diagnostics over subsampled model activity; not biological validation.',
    }
    report = _jsonable(report)
    report['digest'] = _synthesis_digest(report)
    return report
