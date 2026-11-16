"""Versioned NEUROWEAVE specification shared by every subsystem.

NEUROWEAVE is an *additive* Sonapse extension. Its default spec is disabled,
has zero residual gain, and enables no mechanism, so every existing Sonapse
computational path is unchanged unless a caller opts in explicitly.

Each mechanism is independently switchable by id. All mechanism ids, the
experiment registry and the sensory channel manifest are declared here so that
subsystems, experiments, persistence and the UI share one source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from typing import Any, Mapping

SPEC_SCHEMA = 'NeuroWeaveSpecV1'
SPEC_VERSION = 1
EVIDENCE_TIER = 'synthetic_abstraction'


@dataclass(frozen=True)
class MechanismInfo:
    id: str
    title: str
    category: str
    origin: str
    description: str


def _m(id: str, title: str, category: str, origin: str, description: str) -> MechanismInfo:
    return MechanismInfo(id, title, category, origin, description)


# The 31 NEUROWEAVE 0.4.0 switches (renamed to snake_case) followed by
# Sonapse-side extensions. Order is stable and part of the export contract.
MECHANISMS: tuple[MechanismInfo, ...] = (
    _m('multi_soma', 'Multi-soma integration', 'neuron', 'neuroweave', 'Coupled fast/slow/context/integrator soma channels replace the single baseline state update.'),
    _m('structural_encoder', 'Structural encoder', 'connectome', 'neuroweave', 'Graph fingerprints (degree, strength, reciprocity, two-hop, motif) weight local drive.'),
    _m('sparse_code', 'Sparse distributed code', 'representation', 'neuroweave', 'Global k-winners-take-all activity mask.'),
    _m('associative_memory', 'Associative trace memory', 'memory', 'neuroweave', 'Bounded similarity-retrieved episode store blended into drive.'),
    _m('reward_plasticity', 'Reward-gated plasticity', 'plasticity', 'neuroweave', 'Eligibility traces with bounded, sign-preserving reward-gated updates.'),
    _m('synaptic_delays', 'Synaptic delays', 'connectome', 'neuroweave', 'Per-edge 1-3 step delay queues.'),
    _m('homeostatic_governor', 'Homeostatic governor', 'neuron', 'neuroweave', 'Local homeostatic gain regulation toward a target rate.'),
    _m('tiled_execution', 'Tiled execution', 'hardware', 'neuroweave', 'Tile-grouped edge traversal with synchronous previous/next buffers (numerically identical).'),
    _m('context_attention', 'Connectivity-context attention', 'representation', 'neuroweave', 'Attention over the strongest incoming neighbours feeds the apical/context path.'),
    _m('flyhash_projection', 'FlyHash projection', 'representation', 'neuroweave', 'Fixed sparse random 16-channel projection of the task/sensory input.'),
    _m('neuromodulator_field', 'Neuromodulator field', 'modulation', 'neuroweave', 'DA/5-HT/OA/ACh/NE/eCB fields scale gain and plasticity.'),
    _m('structural_plasticity', 'Structural plasticity', 'plasticity', 'neuroweave', 'Utility-guided bounded pruning and regrowth with a reasoned event ledger.'),
    _m('sleep_consolidation', 'Sleep-like consolidation', 'memory', 'neuroweave', 'Periodic bounded replay, trace consolidation and synaptic downscaling.'),
    _m('predictive_ensemble', 'Predictive world model', 'prediction', 'neuroweave', 'Small prediction ensemble with disagreement as an uncertainty proxy.'),
    _m('adaptive_conduction', 'Adaptive conduction', 'connectome', 'neuroweave', 'Activity-dependent myelin proxy adjusts conduction delay.'),
    _m('quantised_synapses', 'Quantised synapses', 'hardware', 'neuroweave', 'INT8-like signed 127-level weight quantisation in transmission.'),
    _m('axonal_branches', 'Axonal branches', 'neuron', 'neuroweave', 'Two axon branch traces and terminal release shape the output.'),
    _m('fast_weights', 'Universal fast weights', 'memory', 'neuroweave', 'Reward-updated 8x8 key/value association matrix.'),
    _m('efference_self_model', 'Action-grounded self-model', 'prediction', 'neuroweave', 'Intended vs issued vs predicted vs observed action outcome comparison.'),
    _m('predictive_coding', 'Local predictive coding', 'prediction', 'neuroweave', 'Per-unit prediction estimate, bounded residual and surprise trace.'),
    _m('stdp', 'Pair-based STDP', 'plasticity', 'neuroweave', 'Pre/post spike traces produce sign-preserving potentiation/depression.'),
    _m('adaptive_routing', 'Context-adaptive routing', 'connectome', 'neuroweave', 'Destination soma/dendrite state scales transmission in [0.62, 1.14].'),
    _m('hierarchical_temporal_memory', 'Hierarchical temporal memory', 'memory', 'neuroweave', 'Fast/medium/slow per-unit traces feed a bounded temporal contrast.'),
    _m('uncertainty_gating', 'Uncertainty-gated propagation', 'prediction', 'neuroweave', 'Ensemble disagreement attenuates recurrent input in [0.58, 1].'),
    _m('multimodal_sensory', 'Multimodal sensory integration', 'sensory', 'neuroweave', 'Sixteen sensory channels are fused into the drive.'),
    _m('affective_modulation', 'Affective neuromodulation', 'modulation', 'neuroweave', 'Valence/arousal/curiosity/stress/motivation act as control variables.'),
    _m('micro_collective', 'Micro-Collective', 'learning', 'neuroweave', 'Local mixture-of-experts sensory-transition predictor.'),
    _m('expert_routing', 'Balanced expert routing', 'learning', 'neuroweave', 'Load-balancing router bias for the Micro-Collective.'),
    _m('meta_plasticity', 'Meta-plasticity', 'plasticity', 'neuroweave', 'Per-synapse learning-rate adaptation from update history.'),
    _m('predictive_replay', 'Predictive replay', 'memory', 'neuroweave', 'Novelty-prioritised replay of prediction-error transitions.'),
    _m('transmitter_inference', 'Transmitter inference', 'modulation', 'neuroweave', 'Connectivity-only six-class transmitter hypothesis; never relabels source sign.'),
    # ---- Sonapse-side extensions ------------------------------------------------
    _m('dendritic_compartments', 'Dendritic compartments', 'neuron', 'sonapse_extension', 'Proximal/distal/apical/basal compartments with branch-specific drive and gating.'),
    _m('extended_somas', 'Extended soma roles', 'neuron', 'sonapse_extension', 'Primary/secondary (DSAG-compatible), burst and gating soma roles.'),
    _m('membrane_dynamics', 'Membrane dynamics', 'neuron', 'sonapse_extension', 'Membrane state, adaptive threshold, refractory period, calcium and hillock.'),
    _m('astrocyte_regulation', 'Astrocyte-like regulation', 'neuron', 'sonapse_extension', 'Slow glial reservoir modulates local gain.'),
    _m('receptor_heterogeneity', 'Receptor heterogeneity', 'modulation', 'sonapse_extension', 'Per-unit receptor sensitivities make modulation heterogeneous.'),
    _m('multi_region_routing', 'Multi-region routing', 'connectome', 'sonapse_extension', 'Region-pair gains for sensory->association->memory->action and feedback paths.'),
    _m('neurogenesis', 'Bounded neurogenesis', 'plasticity', 'sonapse_extension', 'Capped neuron birth/repurposing with a reasoned ledger.'),
    _m('memory_ecology', 'Memory ecology', 'memory', 'sonapse_extension', 'Multi-timescale, provenance-tagged, sparse-indexed memory with interference tracking.'),
    _m('hierarchical_prediction', 'Hierarchical prediction horizons', 'prediction', 'sonapse_extension', 'Immediate/short/medium/long horizons with adaptive horizon selection.'),
    _m('counterfactual_prediction', 'Counterfactual prediction', 'prediction', 'sonapse_extension', 'Pure (non-mutating) alternative-action outcome forecasts.'),
    _m('cross_modal_memory', 'Cross-modal associative memory', 'sensory', 'sonapse_extension', 'Bounded association between modality pairs used to impute missing sensors.'),
    _m('specialist_graft', 'Specialist graft', 'learning', 'sonapse_extension', 'Independently trained Supermix-style specialist exchanging predictions; zero-initialised contribution.'),
    _m('representation_fusion', 'Representation fusion', 'representation', 'sonapse_extension', 'Typed multi-representation packets fused by configurable/learned weights.'),
    _m('q88_state', 'Q8.8 state representation', 'hardware', 'sonapse_extension', 'Fixed-point Q8.8 rounding of neuron state (hardware experiment).'),
)

MECHANISM_IDS: tuple[str, ...] = tuple(m.id for m in MECHANISMS)
NEUROWEAVE_ORIGINAL_IDS: tuple[str, ...] = tuple(m.id for m in MECHANISMS if m.origin == 'neuroweave')
assert len(NEUROWEAVE_ORIGINAL_IDS) == 31 and len(set(MECHANISM_IDS)) == len(MECHANISM_IDS)


@dataclass(frozen=True)
class ExperimentInfo:
    id: str
    title: str
    mechanisms: tuple[str, ...]
    kind: str = 'temporal_task'   # temporal_task | ecology | topology | lesion | graft | micro


EXPERIMENTS: tuple[ExperimentInfo, ...] = (
    ExperimentInfo('multimodal_integration', 'Multimodal sensory integration', ('multimodal_sensory', 'cross_modal_memory')),
    ExperimentInfo('affective_neuromodulation', 'Affective neuromodulation', ('neuromodulator_field', 'affective_modulation', 'receptor_heterogeneity')),
    ExperimentInfo('micro_collective_learning', 'Micro-Collective learning', ('multimodal_sensory', 'micro_collective'), 'micro'),
    ExperimentInfo('expert_routing', 'Expert routing', ('multimodal_sensory', 'micro_collective', 'expert_routing'), 'micro'),
    ExperimentInfo('meta_plasticity', 'Meta-plasticity', ('reward_plasticity', 'meta_plasticity')),
    ExperimentInfo('predictive_replay', 'Predictive replay', ('predictive_ensemble', 'predictive_replay', 'sleep_consolidation')),
    ExperimentInfo('context_attention', 'Connectivity-context attention', ('context_attention',)),
    ExperimentInfo('flyhash_projection', 'FlyHash-style sparse projection', ('flyhash_projection', 'sparse_code')),
    ExperimentInfo('stdp', 'STDP', ('membrane_dynamics', 'stdp')),
    ExperimentInfo('adaptive_conduction', 'Adaptive conduction', ('synaptic_delays', 'adaptive_conduction')),
    ExperimentInfo('quantised_synapses', 'Quantised synapses', ('quantised_synapses', 'q88_state')),
    ExperimentInfo('hierarchical_temporal_memory', 'Hierarchical temporal memory', ('hierarchical_temporal_memory',)),
    ExperimentInfo('uncertainty_gating', 'Uncertainty-gated propagation', ('predictive_ensemble', 'uncertainty_gating')),
    ExperimentInfo('predictive_coding', 'Predictive coding', ('predictive_coding', 'hierarchical_prediction')),
    ExperimentInfo('structural_plasticity', 'Structural plasticity', ('structural_plasticity', 'neurogenesis')),
    ExperimentInfo('associative_memory', 'Associative memory', ('associative_memory', 'fast_weights', 'memory_ecology')),
    ExperimentInfo('action_self_model', 'Action-grounded self-modelling', ('efference_self_model', 'counterfactual_prediction'), 'ecology'),
    ExperimentInfo('multi_soma_integration', 'Multi-soma integration', ('multi_soma', 'extended_somas', 'dendritic_compartments')),
    ExperimentInfo('multi_region_routing', 'Multi-region routing', ('multi_region_routing', 'adaptive_routing')),
    ExperimentInfo('lesion_robustness', 'Lesion robustness', (), 'lesion'),
    ExperimentInfo('topology_optimisation', 'Topology optimisation', (), 'topology'),
    ExperimentInfo('neural_grafting', 'Neural grafting', ('specialist_graft', 'representation_fusion'), 'graft'),
    ExperimentInfo('cross_modal_learning', 'Cross-modal learning', ('multimodal_sensory', 'cross_modal_memory', 'associative_memory')),
    ExperimentInfo('specialist_evolution', 'Specialist expert evolution', ('micro_collective', 'expert_routing', 'specialist_graft'), 'micro'),
)
EXPERIMENT_IDS: tuple[str, ...] = tuple(e.id for e in EXPERIMENTS)
assert len(EXPERIMENT_IDS) == 24

# Sixteen sensory channels (NEUROWEAVE contract) in a fixed order.
SENSORY_CHANNELS: tuple[tuple[str, str], ...] = (
    ('visual_luminance', 'visual'),
    ('visual_red_green', 'visual'),
    ('visual_blue_yellow', 'visual'),
    ('visual_edge_density', 'visual'),
    ('visual_motion', 'visual'),
    ('audio_low', 'auditory'),
    ('audio_mid', 'auditory'),
    ('audio_high', 'auditory'),
    ('olfactory', 'chemical'),
    ('gustatory', 'chemical'),
    ('thermal', 'somatic'),
    ('interoception', 'somatic'),
    ('proprioception', 'somatic'),
    ('mechanosensation', 'somatic'),
    ('environment_state', 'context'),
    ('task_context', 'context'),
)
SENSORY_CHANNEL_IDS: tuple[str, ...] = tuple(c for c, _ in SENSORY_CHANNELS)
SENSORY_MODES = ('disabled', 'simulated', 'external', 'live')

TOPOLOGIES: tuple[str, ...] = (
    'clustered', 'hub_skewed', 'uniform_random', 'degree_shuffled', 'erdos_renyi',
    'hub_capped', 'ring_lattice', 'hierarchical_modular', 'multi_scale_regional', 'hybrid',
    'multi_region', 'dense_multi_region',
)

REGIONS: tuple[tuple[str, str], ...] = (
    ('visual_sensory', 'sensory'),
    ('auditory_sensory', 'sensory'),
    ('chemical_somatic_sensory', 'sensory'),
    ('association_a', 'association'),
    ('association_b', 'association'),
    ('working_memory', 'memory'),
    ('episodic_memory', 'memory'),
    ('modulatory_hub', 'modulatory'),
    ('premotor', 'action'),
    ('motor', 'action'),
)

SOMA_ROLES = ('primary', 'secondary', 'fast', 'slow', 'context', 'integrator', 'burst', 'gating')
DENDRITES = ('proximal', 'distal', 'apical', 'basal')
NEUROMODULATORS = ('dopamine', 'serotonin', 'acetylcholine', 'noradrenaline', 'octopamine', 'endocannabinoid')
AFFECT_VARIABLES = ('valence', 'arousal', 'curiosity', 'stress', 'motivation', 'novelty', 'reward_prediction_error')
TASK_INPUT_DIM = 8
SENSORY_DIM = len(SENSORY_CHANNELS)

MAX_POPULATION = 4096      # validated import/runtime ceiling (not an architectural constant)
MAX_SYNAPSES = 250_000


def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


def _check_float(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    value = float(value)
    if not (low <= value <= high):
        raise ValueError(f'{name} must be in [{low}, {high}]')
    return value


@dataclass(frozen=True)
class NeuroWeaveSpecV1:
    """Immutable NEUROWEAVE configuration. Defaults are fully neutral."""

    enabled: bool = False
    observe_only: bool = False
    gain: float = 0.0               # residual current gain into the Sonapse graft bus
    max_gain: float = 0.1
    ramp_steps: int = 32
    seed: int = 7
    neurons: int = 48               # NEUROWEAVE default population
    topology: str = 'clustered'
    sparsity: float = 0.15          # k-winners active fraction
    tile_size: int = 16
    fan_in: int = 6                 # initial mean in-degree
    inhibitory_fraction: float = 0.2
    max_neurons: int = 512          # neurogenesis growth cap (safeguard)
    max_synapses: int = 8192        # structural growth cap (safeguard)
    structural_interval: int = 25
    sleep_interval: int = 100
    micro_experts: int = 4
    micro_top_k: int = 2
    ecology: str = 'default'
    mechanisms: tuple[str, ...] = ()
    sensory_modes: tuple[tuple[str, str], ...] = ()   # (channel_id, mode); absent channels -> 'simulated'

    def __post_init__(self) -> None:
        for name in ('enabled', 'observe_only'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        _check_float('gain', self.gain, 0.0, 1.0)
        _check_float('max_gain', self.max_gain, 0.0, 1.0)
        if self.gain > self.max_gain:
            raise ValueError('gain must not exceed max_gain')
        _check_int('ramp_steps', self.ramp_steps, 0, 1_000_000)
        _check_int('seed', self.seed, 0, 2**31 - 1)
        _check_int('neurons', self.neurons, 8, MAX_POPULATION)
        _check_int('max_neurons', self.max_neurons, self.neurons, MAX_POPULATION)
        _check_int('max_synapses', self.max_synapses, 16, MAX_SYNAPSES)
        _check_int('tile_size', self.tile_size, 1, 4096)
        _check_int('fan_in', self.fan_in, 1, 64)
        _check_int('structural_interval', self.structural_interval, 1, 100_000)
        _check_int('sleep_interval', self.sleep_interval, 1, 100_000)
        _check_int('micro_experts', self.micro_experts, 1, 64)
        _check_int('micro_top_k', self.micro_top_k, 1, self.micro_experts)
        _check_float('sparsity', self.sparsity, 0.01, 1.0)
        _check_float('inhibitory_fraction', self.inhibitory_fraction, 0.0, 0.9)
        if self.topology not in TOPOLOGIES:
            raise ValueError(f'topology must be one of {TOPOLOGIES}')
        if not isinstance(self.ecology, str) or not self.ecology:
            raise ValueError('ecology must be a non-empty string')
        mechs = tuple(self.mechanisms)
        unknown = [m for m in mechs if m not in MECHANISM_IDS]
        if unknown:
            raise ValueError(f'unknown NEUROWEAVE mechanisms: {unknown}')
        # canonical order, no duplicates
        object.__setattr__(self, 'mechanisms', tuple(m for m in MECHANISM_IDS if m in set(mechs)))
        modes = tuple((str(c), str(m)) for c, m in self.sensory_modes)
        for channel, mode in modes:
            if channel not in SENSORY_CHANNEL_IDS:
                raise ValueError(f'unknown sensory channel {channel!r}')
            if mode not in SENSORY_MODES:
                raise ValueError(f'sensory mode must be one of {SENSORY_MODES}')
        object.__setattr__(self, 'sensory_modes', tuple(sorted(dict(modes).items())))

    # ------------------------------------------------------------------ helpers
    def has(self, mechanism: str) -> bool:
        if mechanism not in MECHANISM_IDS:
            raise KeyError(mechanism)
        return mechanism in self.mechanisms

    def sensory_mode(self, channel: str) -> str:
        return dict(self.sensory_modes).get(channel, 'simulated')

    def with_mechanisms(self, mechanisms) -> 'NeuroWeaveSpecV1':
        return replace(self, mechanisms=tuple(mechanisms))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['mechanisms'] = list(self.mechanisms)
        data['sensory_modes'] = [list(item) for item in self.sensory_modes]
        data['schema'] = SPEC_SCHEMA
        data['schema_version'] = SPEC_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'NeuroWeaveSpecV1':
        if not isinstance(data, Mapping):
            raise ValueError('NEUROWEAVE spec must be a mapping')
        payload = dict(data)
        schema = payload.pop('schema', SPEC_SCHEMA)
        version = payload.pop('schema_version', SPEC_VERSION)
        if schema != SPEC_SCHEMA or version != SPEC_VERSION:
            raise ValueError('unsupported NEUROWEAVE spec schema')
        known = {f for f in cls.__dataclass_fields__}
        extra = set(payload) - known
        if extra:
            raise ValueError(f'unknown NEUROWEAVE spec fields: {sorted(extra)}')
        if 'mechanisms' in payload:
            payload['mechanisms'] = tuple(payload['mechanisms'])
        if 'sensory_modes' in payload:
            payload['sensory_modes'] = tuple(tuple(item) for item in payload['sensory_modes'])
        return cls(**payload)

    @property
    def spec_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')).hexdigest()

    @classmethod
    def all_mechanisms(cls, **kwargs) -> 'NeuroWeaveSpecV1':
        return cls(mechanisms=MECHANISM_IDS, **kwargs)

    @classmethod
    def max_preset(cls, seed: int = 7, neurons: int = 256) -> 'NeuroWeaveSpecV1':
        """Max Mode preset: every mechanism on, small bounded residual gain."""
        return cls(
            enabled=True, observe_only=False, gain=0.02, max_gain=0.1, ramp_steps=32,
            seed=seed, neurons=neurons, topology='dense_multi_region',
            max_neurons=max(neurons, min(MAX_POPULATION, neurons * 2)),
            max_synapses=min(MAX_SYNAPSES, max(8192, neurons * 48)),
            mechanisms=MECHANISM_IDS,
        )


def mechanism_catalog() -> list[dict[str, str]]:
    return [asdict(m) for m in MECHANISMS]


def experiment_catalog() -> list[dict[str, Any]]:
    return [{**asdict(e), 'mechanisms': list(e.mechanisms)} for e in EXPERIMENTS]
