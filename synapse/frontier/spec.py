"""Versioned specification and registry for the Sonapse frontier pack.

The frontier pack is an *additive, default-off* set of experimental modules
drawn from 2024-2026 research and from sibling ``kai9987kai`` projects. Nothing
here changes Sonapse or NEUROWEAVE behaviour unless a caller opts in:

* ``SynapseConfig().frontier`` is ``None`` and is omitted from ``to_dict()``.
* The live runtime hook is **observer-only**: it reads brain activity and never
  writes into the brain, the graft bus or the actor.
* Modules that can influence computation do so only as explicitly attached
  NEUROWEAVE engine extensions inside lab experiments, or through explicitly
  selected opt-in options.

Every module is a synthetic computational abstraction. Titles name the research
idea a module is inspired by; they are not claims of reproducing published
results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping

SPEC_SCHEMA = 'FrontierSpecV1'
SPEC_VERSION = 1
EVIDENCE_TIER = 'synthetic_abstraction'
LAB_SCHEMA = 'FrontierLabReceiptV1'


@dataclass(frozen=True)
class ModuleInfo:
    id: str
    title: str
    kind: str          # extension | analysis | model | methodology | discovery | tool | reference
    origin: str        # literature | portfolio
    source: str        # short human-readable provenance (full citations: docs/frontier/REFERENCES.md)
    description: str
    observer: bool = False   # can run as a read-only live observer


def _m(id: str, title: str, kind: str, origin: str, source: str, description: str, observer: bool = False) -> ModuleInfo:
    return ModuleInfo(id, title, kind, origin, source, description, observer)


MODULES: tuple[ModuleInfo, ...] = (
    _m('synchrony', 'Synchronisation representation and neuron-level models', 'extension', 'literature',
       'Continuous Thought Machines (Darlow et al., 2025)',
       'Pairwise decayed synchronisation of unit activity as a readout representation, plus private per-neuron history models and internal ticks.',
       observer=True),
    _m('continuum_memory', 'Continuum memory system', 'extension', 'literature',
       'Nested Learning / HOPE (Behrouz et al., 2025); Titans (Behrouz et al., 2025)',
       'A chain of associative memories updated at geometrically spaced frequencies with surprise-gated writes and decay.'),
    _m('btsp_memory', 'One-shot BTSP memory with binary synapses', 'extension', 'literature',
       'Behavioural time-scale synaptic plasticity model (Wu & Maass, 2025)',
       'Plateau-gated stochastic flips of binary synapses give one-shot content-addressable memory.'),
    _m('kuramoto_binding', 'Oscillatory binding units', 'extension', 'literature',
       'Artificial Kuramoto Oscillatory Neurons (Miyato et al., 2025)',
       'Unit-vector oscillators with tangent-space Kuramoto updates; synchrony clusters bind related inputs.',
       observer=True),
    _m('recursive_refiner', 'Recursive latent refinement', 'model', 'literature',
       'Tiny Recursive Model (Jolicoeur-Martineau, 2025); Hierarchical Reasoning Model (Wang et al., 2025)',
       'One tiny network repeatedly refines a latent and an answer with deep supervision and a halting head.'),
    _m('dendritic_layer', 'Structured dendritic layer', 'model', 'literature',
       'Dendritic ANNs (Chavlis & Poirazi, 2025)',
       'Dendrites sample restricted receptive fields and feed a soma; compared against dense layers at equal parameters.'),
    _m('muon_optimizer', 'Orthogonalised-momentum optimiser', 'tool', 'literature',
       'Muon (Jordan et al., 2024; Liu et al., 2025)',
       'Newton-Schulz orthogonalised momentum for matrix parameters with an AdamW fallback for the rest.'),
    _m('rigl_sparse', 'Dynamic sparse training', 'tool', 'literature',
       'RigL (Evci et al., 2020), as used in sonapse-evo',
       'Periodic magnitude pruning and gradient-magnitude regrowth at a fixed parameter budget.'),
    _m('sparse_dictionary', 'Sparse dictionary of brain activity', 'analysis', 'literature',
       'TopK sparse autoencoders (Gao et al., 2024)',
       'Online TopK sparse autoencoder over unit activity with dead-feature handling and causal feature ablation.',
       observer=True),
    _m('anytime_inference', 'Anytime-valid inference', 'methodology', 'literature',
       'Betting confidence sequences (Waudby-Smith & Ramdas, 2024); e-values (Ramdas et al., 2023)',
       'E-processes and confidence sequences that stay valid under optional stopping.'),
    _m('experiment_design', 'Decision-sensitive experiment selection', 'methodology', 'portfolio',
       'COUNTERFORM and Mnemorph (kai9987kai)',
       'Chooses the next experiment by expected reduction in downstream decision risk, with abstention.'),
    _m('quality_diversity', 'Quality-diversity architecture search', 'discovery', 'portfolio',
       'SONAPSE-ORIGIN and sonapse-evo (kai9987kai); MAP-Elites (Mouret & Clune, 2015); Darwin Godel Machine (Zhang et al., 2025)',
       'A MAP-Elites archive over NEUROWEAVE configurations with novelty-weighted parent selection and verifier-gated admission.'),
    _m('morphogenesis', 'Developmental growth and regeneration', 'discovery', 'portfolio',
       'morpheus, prometheus-alpha, GenesisEngine and MOLT (kai9987kai); growing neural cellular automata (Mordvintsev et al., 2020)',
       'Grows a connectome from a compact developmental program and regrows it after lesions with a stored pattern memory.'),
    _m('connectomics', 'Structural, functional and effective connectivity', 'analysis', 'portfolio',
       'SONAPSE-CONNECTOME (kai9987kai)',
       'Functional, dynamic-functional and lagged effective connectivity kept separate from structure, with structure-function coupling.',
       observer=True),
    _m('causal_emergence', 'Effective information and causal emergence', 'analysis', 'portfolio',
       'GhostInTheMachine and SONAPSE-ORIGIN (kai9987kai); effective information (Hoel et al., 2013)',
       'Effective information of micro and coarse-grained transition models with shuffle-null calibration.',
       observer=True),
    _m('hh_cable', 'Hodgkin-Huxley cable reference', 'reference', 'portfolio',
       'NeuroPulse-CableLab (kai9987kai); Hodgkin & Huxley (1952)',
       'Multi-compartment Hodgkin-Huxley cable used as a biophysical reference for the abstract cell model.'),
    _m('nir_export', 'Neuromorphic interchange export', 'tool', 'literature',
       'Neuromorphic Intermediate Representation (Pedersen et al., 2024)',
       'Exports spiking and NEUROWEAVE graphs as a NIR-style graph; writes real NIR files when the optional package is installed.'),
    _m('joint_failure', 'Joint-failure ensemble selection', 'methodology', 'portfolio',
       'FAULTLINE (kai9987kai)',
       'Selects voting panels from condition-indexed joint-failure observations instead of individual accuracy.'),
)
MODULE_IDS: tuple[str, ...] = tuple(m.id for m in MODULES)
OBSERVER_MODULE_IDS: tuple[str, ...] = tuple(m.id for m in MODULES if m.observer)
assert len(set(MODULE_IDS)) == len(MODULE_IDS)

GRAFT_OPTIMIZERS = ('adam', 'muon')


def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


@dataclass(frozen=True)
class FrontierSpecV1:
    """Immutable frontier configuration. Defaults are fully neutral (disabled, no modules)."""

    enabled: bool = False
    seed: int = 7
    modules: tuple[str, ...] = ()        # live read-only observers (subset of OBSERVER_MODULE_IDS)
    observe_every: int = 1               # observe every k-th executed agent step
    max_observed_units: int = 256        # deterministic unit subsample fed to observers (bounds cost)
    history: int = 256                   # bounded activity history kept by observers
    sae_features: int = 64
    sae_k: int = 8
    sync_pairs: int = 64
    fc_window: int = 128
    graft_optimizer: str = 'adam'        # opt-in optimiser for the live neural-graft fit ('muon' = frontier)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        _check_int('seed', self.seed, 0, 2**31 - 1)
        _check_int('observe_every', self.observe_every, 1, 10_000)
        _check_int('max_observed_units', self.max_observed_units, 8, 4096)
        _check_int('history', self.history, 16, 8192)
        _check_int('sae_features', self.sae_features, 4, 4096)
        _check_int('sae_k', self.sae_k, 1, self.sae_features)
        _check_int('sync_pairs', self.sync_pairs, 1, 4096)
        _check_int('fc_window', self.fc_window, 8, 8192)
        if self.graft_optimizer not in GRAFT_OPTIMIZERS:
            raise ValueError(f'graft_optimizer must be one of {GRAFT_OPTIMIZERS}')
        mods = tuple(self.modules)
        unknown = [m for m in mods if m not in MODULE_IDS]
        if unknown:
            raise ValueError(f'unknown frontier modules: {unknown}')
        not_observer = [m for m in mods if m not in OBSERVER_MODULE_IDS]
        if not_observer:
            raise ValueError(f'modules without a live observer: {not_observer} (observers: {OBSERVER_MODULE_IDS})')
        object.__setattr__(self, 'modules', tuple(m for m in MODULE_IDS if m in set(mods)))

    def has(self, module: str) -> bool:
        if module not in MODULE_IDS:
            raise KeyError(module)
        return module in self.modules

    def with_modules(self, modules) -> 'FrontierSpecV1':
        return replace(self, modules=tuple(modules))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['modules'] = list(self.modules)
        data['schema'] = SPEC_SCHEMA
        data['schema_version'] = SPEC_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'FrontierSpecV1':
        if not isinstance(data, Mapping):
            raise ValueError('frontier spec must be a mapping')
        payload = dict(data)
        if payload.pop('schema', SPEC_SCHEMA) != SPEC_SCHEMA or payload.pop('schema_version', SPEC_VERSION) != SPEC_VERSION:
            raise ValueError('unsupported frontier spec schema')
        extra = set(payload) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(f'unknown frontier spec fields: {sorted(extra)}')
        if 'modules' in payload:
            payload['modules'] = tuple(payload['modules'])
        return cls(**payload)

    @property
    def spec_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode('utf-8')).hexdigest()

    @classmethod
    def max_preset(cls, seed: int = 7) -> 'FrontierSpecV1':
        """Max Mode preset: every live observer on. Observers are read-only, so this
        does not change the actor's trajectory; the opt-in graft optimiser stays at its default."""
        return cls(enabled=True, seed=seed, modules=OBSERVER_MODULE_IDS)


def module_catalog() -> list[dict[str, Any]]:
    return [asdict(m) for m in MODULES]
