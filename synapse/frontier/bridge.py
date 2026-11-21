"""Sonapse host-integration bridge for the Frontier Pack (`SynapseConfig`, `IntegratedAgent`, `NeuralResidualGraft`).

Centralises the config, checkpoint, observer-stepping and graft-optimizer hooks so host wiring stays minimal,
additive and strictly trajectory-neutral when read-only observers are active.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .muon import make_graft_optimizer
from .runtime import FrontierObserverHub
from .spec import OBSERVER_MODULE_IDS, FrontierSpecV1

__all__ = [
    'CHECKPOINT_NAMESPACE',
    'coerce_frontier_spec',
    'frontier_audit_entries',
    'normalize_saved_frontier',
    'attach_frontier_to_agent',
    'step_agent_frontier',
    'export_agent_frontier',
    'import_agent_frontier',
]

CHECKPOINT_NAMESPACE = 'frontier_v1'


def coerce_frontier_spec(value: Any) -> FrontierSpecV1 | None:
    """Validate/convert a ``SynapseConfig.frontier`` value (``None``, ``FrontierSpecV1`` or mapping)."""
    if value is None:
        return None
    if isinstance(value, FrontierSpecV1):
        return value
    if isinstance(value, Mapping):
        return FrontierSpecV1.from_dict(value)
    raise ValueError('frontier must be None, a FrontierSpecV1 or a mapping')


def frontier_audit_entries(spec: FrontierSpecV1 | None) -> tuple[list[str], list[str]]:
    """Return ``(enabled, disabled)`` feature-audit entries for ``SynapseConfig.max_tandem_feature_audit``."""
    if spec is None or not spec.enabled:
        return [], ['frontier_v1']
    enabled = ['frontier_v1']
    disabled: list[str] = []
    for mod_id in OBSERVER_MODULE_IDS:
        key = f'frontier_v1.{mod_id}'
        if mod_id in spec.modules:
            enabled.append(key)
        else:
            disabled.append(key)
    return enabled, disabled


def normalize_saved_frontier(current_spec: FrontierSpecV1 | None,
                             normalized_saved: dict[str, Any]) -> bool:
    """Apply the additive legacy-checkpoint rule (same as DSAG and NEUROWEAVE) in ``SynapseConfig.matches_saved``.

    Returns ``False`` only when ``normalized_saved['frontier']`` is malformed; otherwise updates
    ``normalized_saved`` in place and returns ``True``."""
    normalized_saved.setdefault('frontier', None)
    saved_frontier = normalized_saved.get('frontier')
    saved_spec: FrontierSpecV1 | None = None
    if isinstance(saved_frontier, FrontierSpecV1):
        saved_spec = saved_frontier
    elif isinstance(saved_frontier, Mapping):
        try:
            saved_spec = FrontierSpecV1.from_dict(saved_frontier)
        except (TypeError, ValueError):
            return False
        normalized_saved['frontier'] = saved_spec
    elif saved_frontier is not None:
        return False

    # Read-only frontier observers can be additively enabled on a legacy checkpoint that had frontier disabled/None,
    # provided graft_optimizer stays at its default ('adam').
    if (current_spec is not None and current_spec.enabled
            and current_spec.graft_optimizer == 'adam'
            and (saved_spec is None or not saved_spec.enabled)):
        normalized_saved['frontier'] = current_spec
    return True


def attach_frontier_to_agent(agent: Any, spec: FrontierSpecV1 | None = None) -> FrontierObserverHub | None:
    """Attach a :class:`FrontierObserverHub` and optional Muon graft optimizer to an ``IntegratedAgent``."""
    active_spec = coerce_frontier_spec(getattr(getattr(agent, 'config', None), 'frontier', None)
                                       if spec is None else spec)
    if active_spec is None or not active_spec.enabled:
        agent.frontier_hub = None
        return None
    n_units = int(agent.brain.n)
    hub = FrontierObserverHub(active_spec, n_units=n_units)
    agent.frontier_hub = hub
    if active_spec.graft_optimizer == 'muon' and getattr(agent, 'neural_graft', None) is not None:
        graft = agent.neural_graft
        if graft.settings.get('optimizer_kind', 'adam') != 'muon':
            lr = float(graft.settings.get('learning_rate', 3e-4))
            graft.settings['optimizer_kind'] = 'muon'
            graft.optimizer = make_graft_optimizer(graft.network, kind='muon', lr=lr)
    return hub


def step_agent_frontier(agent: Any) -> dict[str, Any] | None:
    """Feed a read-only copy of ``agent.brain.r`` to ``agent.frontier_hub`` and update ``agent.last_telemetry``."""
    hub: FrontierObserverHub | None = getattr(agent, 'frontier_hub', None)
    if hub is None or not hub.enabled:
        return None
    hub.observe(agent.brain.r)
    summary = hub.telemetry()
    if hasattr(agent, 'last_telemetry') and isinstance(agent.last_telemetry, dict):
        agent.last_telemetry['frontier'] = summary
    return summary


def export_agent_frontier(agent: Any, state: dict[str, Any]) -> None:
    """Store ``state['frontier_v1']`` when an active ``frontier_hub`` is attached."""
    hub: FrontierObserverHub | None = getattr(agent, 'frontier_hub', None)
    if hub is not None and hub.enabled:
        state[CHECKPOINT_NAMESPACE] = hub.export_state()


def import_agent_frontier(agent: Any, state: Mapping[str, Any]) -> None:
    """Validate and restore ``state['frontier_v1']`` (allowing additive enable on legacy checkpoints)."""
    hub: FrontierObserverHub | None = getattr(agent, 'frontier_hub', None)
    saved_config = state.get('config')
    saved_frontier = saved_config.get('frontier') if isinstance(saved_config, Mapping) else None
    saved_enabled = (isinstance(saved_frontier, Mapping) and saved_frontier.get('enabled') is True) or (
        isinstance(saved_frontier, FrontierSpecV1) and saved_frontier.enabled
    )
    payload = state.get(CHECKPOINT_NAMESPACE)
    if payload is not None:
        if not saved_enabled:
            raise ValueError('checkpoint contains frontier_v1 state but its saved config does not enable frontier')
        if hub is None or not hub.enabled:
            raise ValueError('checkpoint contains frontier_v1 state but this agent has no enabled frontier hub')
        hub.import_state(payload)
    elif saved_enabled and hub is not None and hub.enabled:
        raise ValueError('frontier-enabled checkpoint is missing its frontier_v1 namespace')
    elif hub is not None and hub.enabled:
        # Legacy checkpoint without frontier_v1: reset observer hub to clean seeded initial state.
        hub.reset()
