"""NEUROWEAVE: additive multi-compartment connectome extension for Sonapse (default off)."""
from .spec import (NeuroWeaveSpecV1, MECHANISMS, MECHANISM_IDS, EXPERIMENTS, EXPERIMENT_IDS,
                   SENSORY_CHANNELS, SENSORY_CHANNEL_IDS, TOPOLOGIES, REGIONS, mechanism_catalog, experiment_catalog)

__all__ = ["NeuroWeaveSpecV1", "MECHANISMS", "MECHANISM_IDS", "EXPERIMENTS", "EXPERIMENT_IDS",
           "SENSORY_CHANNELS", "SENSORY_CHANNEL_IDS", "TOPOLOGIES", "REGIONS", "mechanism_catalog", "experiment_catalog"]
