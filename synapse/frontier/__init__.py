"""Sonapse frontier pack: additive, default-off experimental modules (see ``spec.py``)."""
from .spec import (EVIDENCE_TIER, LAB_SCHEMA, MODULE_IDS, MODULES, OBSERVER_MODULE_IDS, FrontierSpecV1,
                   ModuleInfo, module_catalog)
from .extension import EXTENSION_INTERFACE_VERSION, EngineExtension, RandomTraceControl

__all__ = ['EVIDENCE_TIER', 'LAB_SCHEMA', 'MODULE_IDS', 'MODULES', 'OBSERVER_MODULE_IDS', 'FrontierSpecV1',
           'ModuleInfo', 'module_catalog', 'EXTENSION_INTERFACE_VERSION', 'EngineExtension', 'RandomTraceControl']
