"""SONAPSE SPECTRA — Expanded Senses, Multimodal Signal Mixer & Internal Perception."""
from .internal_perception import InternalPerceptionSimulator, run_internal_perception_suite
from .mixer import (
    INJECTION_TARGETS,
    MIXING_OPERATORS,
    SIGNAL_ENCODINGS,
    encode_multimodal_stream,
    inject_signal_into_connectome,
    mix_neural_signals,
)
from .sensory_library import (
    ARTIFICIAL_SENSORY_MODALITIES,
    BIO_SENSORY_MODALITIES,
    ExpandedSensoryEncoder,
    benchmark_artificial_vs_raw_encoding,
)

__all__ = [
    "BIO_SENSORY_MODALITIES",
    "ARTIFICIAL_SENSORY_MODALITIES",
    "ExpandedSensoryEncoder",
    "benchmark_artificial_vs_raw_encoding",
    "SIGNAL_ENCODINGS",
    "MIXING_OPERATORS",
    "INJECTION_TARGETS",
    "encode_multimodal_stream",
    "mix_neural_signals",
    "inject_signal_into_connectome",
    "InternalPerceptionSimulator",
    "run_internal_perception_suite",
]
