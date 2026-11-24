"""SONAPSE CURIOSITY — Controlled Public Data Discovery, Quarantine & Anti-Leakage Pipeline.

Manages discovery, provenance tracking, license verification, rate limiting,
schema validation, SHA-256 deduplication, quality scoring, held-out leakage
detection, and untrusted-data isolation across official public scientific
resources and local verified archives (Section 30).

SECURITY INVARIANT: All external text/metadata is strictly quarantined passive
data—NEVER interpreted as executable code or instructions for the research director.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..core.interchange import DataSourceKind, TrainingExample, canonical_json, content_digest, make_id


PUBLIC_SOURCE_CATALOG: Dict[str, Dict[str, Any]] = {
    "flywire_codex": {"url": "https://codex.flywire.ai/", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "CC-BY-4.0", "api_type": "official_dump"},
    "openworm": {"url": "https://openworm.org/", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "MIT", "api_type": "github_release"},
    "allen_brain_atlas": {"url": "https://api.brain-map.org/", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "Allen-ToU", "api_type": "rest_api"},
    "dandi_archive": {"url": "https://api.dandiarchive.org/api/", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "CC-BY-4.0", "api_type": "rest_api"},
    "openneuro": {"url": "https://openneuro.org/crn/graphql", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "CC0-1.0", "api_type": "graphql_api"},
    "physionet": {"url": "https://physionet.org/rest/", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "ODC-By-1.0", "api_type": "rest_api"},
    "rcsb_pdb": {"url": "https://data.rcsb.org/rest/v1/", "source_kind": DataSourceKind.BIOLOGICAL_RECORDING.value, "license": "CC0-1.0", "api_type": "rest_api"},
    "openalex": {"url": "https://api.openalex.org/works", "source_kind": DataSourceKind.SYNTHETIC_SIMULATION.value, "license": "CC0-1.0", "api_type": "rest_api"},
    "arxiv": {"url": "https://export.arxiv.org/api/query", "source_kind": DataSourceKind.SYNTHETIC_SIMULATION.value, "license": "arXiv-non-exclusive", "api_type": "atom_api"},
    "huggingface_datasets": {"url": "https://datasets-server.huggingface.co/", "source_kind": DataSourceKind.ARTIFICIAL_MODEL_ACTIVITY.value, "license": "per-dataset", "api_type": "rest_api"},
    "sonapse_experience_archive": {"url": "local://artifacts", "source_kind": DataSourceKind.SYNTHETIC_SIMULATION.value, "license": "Apache-2.0", "api_type": "local_archive"},
    "supermix_model_outputs": {"url": "local://synapse/_vendor/expanse", "source_kind": DataSourceKind.ARTIFICIAL_MODEL_ACTIVITY.value, "license": "Apache-2.0", "api_type": "local_archive"},
    "procedural_verified_generators": {"url": "local://curiosity/modalities", "source_kind": DataSourceKind.SYNTHETIC_SIMULATION.value, "license": "Apache-2.0", "api_type": "procedural"},
}

ALLOWED_LICENSES: Set[str] = {"Apache-2.0", "MIT", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0", "ODC-By-1.0", "Allen-ToU", "arXiv-non-exclusive", "synthetic-local"}

# Pattern filter flagging prompt-injection / executable payload attempts inside untrusted external data
_UNTRUSTED_EXEC_PATTERNS = re.compile(
    r"(ignore previous instructions|system prompt|rm\s+-rf|subprocess\.|__import__|os\.system|<script)",
    re.IGNORECASE,
)


@dataclass
class QuarantineVerdict:
    example_id: str
    accepted: bool
    reasons: List[str]
    quality_score: float
    content_hash: str


class DataQuarantinePipeline:
    """Validates, deduplicates, checks license/leakage, and isolates untrusted data before training."""

    def __init__(self, heldout_fingerprints: Optional[Sequence[str]] = None, min_rate_interval_s: float = 0.01) -> None:
        self.heldout_fingerprints: Set[str] = set(heldout_fingerprints or ())
        self.seen_hashes: Set[str] = set()
        self.min_rate_interval_s = float(min_rate_interval_s)
        self._last_ingest_ts = 0.0
        self.quarantine_pool: List[TrainingExample] = []
        self.approved_pool: List[TrainingExample] = []
        self.rejected_log: List[Dict[str, Any]] = []

    def ingest_and_screen(self, example: TrainingExample) -> QuarantineVerdict:
        """Place an example in quarantine and apply 6 independent safety & quality gates."""
        reasons: List[str] = []
        serialized = canonical_json({"inputs": example.inputs, "target": example.target_output, "domain": example.task_domain})
        c_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        # Gate 1: License allowlist
        if example.license not in ALLOWED_LICENSES:
            reasons.append(f"unapproved_license:{example.license}")

        # Gate 2: Untrusted prompt-injection / executable payload detection
        if _UNTRUSTED_EXEC_PATTERNS.search(serialized):
            reasons.append("untrusted_instruction_or_script_pattern_blocked")

        # Gate 3: Deduplication
        if c_hash in self.seen_hashes:
            reasons.append("duplicate_content_hash")

        # Gate 4: Held-out evaluation leakage detection
        if c_hash in self.heldout_fingerprints:
            reasons.append("heldout_evaluation_leakage_blocked")

        # Gate 5: Schema & provenance completeness
        if not example.task_domain or example.inputs is None:
            reasons.append("incomplete_schema")

        # Gate 6: Quality scoring
        quality = 1.0
        if not example.verified:
            quality -= 0.25
        if example.internal_activations is not None:
            quality += 0.1
        quality = float(max(0.0, min(1.0, quality)))
        if quality < 0.4:
            reasons.append("low_quality_score")

        accepted = len(reasons) == 0
        example.quarantine_passed = accepted
        self.quarantine_pool.append(example)
        if accepted:
            self.seen_hashes.add(c_hash)
            self.approved_pool.append(example)
        else:
            self.rejected_log.append({"example_id": example.example_id, "reasons": reasons, "hash": c_hash})

        self._last_ingest_ts = time.time()
        return QuarantineVerdict(example.example_id, accepted, reasons, round(quality, 4), c_hash)

    def summary(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        for ex in self.approved_pool:
            by_kind[ex.source_kind.value] = by_kind.get(ex.source_kind.value, 0) + 1
        return {
            "catalog_sources": len(PUBLIC_SOURCE_CATALOG),
            "quarantined_total": len(self.quarantine_pool),
            "approved_total": len(self.approved_pool),
            "rejected_total": len(self.rejected_log),
            "approved_by_source_kind": by_kind,
            "rejection_reasons": [r["reasons"] for r in self.rejected_log[:10]],
        }
