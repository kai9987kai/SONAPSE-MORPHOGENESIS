"""Persistent Experiment Registry, Developmental Lineage DAG & Transactional Checkpoints.

Stores experiment configurations, random seeds, code versions, dataset hashes,
parent/descendant checkpoint lineage, structural mutation records, neural
telemetry summaries, evaluation outcomes, discoveries, failed hypotheses, and
resource consumption in an atomic SQLite database + safe content-bound
``synapse.stateio`` checkpoint files.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from synapse.stateio import digest as state_digest, load_state, save_state_snapshot

from .interchange import (
    EvidenceState,
    ExperimentReceipt,
    ObjectiveCategory,
    canonical_json,
    content_digest,
    make_id,
)


class ExperimentRegistry:
    """SQLite + JSON persistent registry with transactional checkpoints and crash recovery."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.root_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root_dir / "morphogenesis_registry.sqlite3"
        self._lock = threading.RLock()
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    module TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    objective_category TEXT NOT NULL,
                    evidence_state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    seeds_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    baselines_json TEXT NOT NULL,
                    uncertainty_json TEXT NOT NULL,
                    resources_json TEXT NOT NULL,
                    failed_json TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    descendant_checkpoint_id TEXT,
                    code_version TEXT NOT NULL,
                    sha256_digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    label TEXT NOT NULL,
                    genome_digest TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    archived_only INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mutations (
                    mutation_id TEXT PRIMARY KEY,
                    parent_checkpoint_id TEXT,
                    child_checkpoint_id TEXT,
                    operator TEXT NOT NULL,
                    delta_json TEXT NOT NULL,
                    inverse_json TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS discoveries (
                    discovery_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    module TEXT NOT NULL,
                    evidence_state TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    report_path TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS failed_hypotheses (
                    failure_id TEXT PRIMARY KEY,
                    module TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def record_experiment(self, receipt: ExperimentReceipt) -> str:
        if not receipt.sha256_digest:
            receipt.seal()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, title, module, hypothesis, objective_category,
                    evidence_state, created_at, seeds_json, config_json, metrics_json,
                    baselines_json, uncertainty_json, resources_json, failed_json,
                    parent_checkpoint_id, descendant_checkpoint_id, code_version, sha256_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.experiment_id,
                    receipt.title,
                    receipt.module,
                    receipt.hypothesis,
                    receipt.objective_category.value,
                    receipt.evidence_state.value,
                    receipt.created_at_utc,
                    canonical_json(receipt.seeds),
                    canonical_json(receipt.config),
                    canonical_json(receipt.metrics),
                    canonical_json(receipt.baseline_comparisons),
                    canonical_json(receipt.uncertainty),
                    canonical_json(receipt.resource_usage),
                    canonical_json(receipt.failed_experiments),
                    receipt.parent_checkpoint_id,
                    receipt.descendant_checkpoint_id,
                    receipt.code_version,
                    receipt.sha256_digest,
                ),
            )
        return receipt.experiment_id

    def list_experiments(self, module: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if module:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE module = ? ORDER BY created_at DESC LIMIT ?",
                    (module, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        out = []
        for r in rows:
            out.append({
                "experiment_id": r["experiment_id"],
                "title": r["title"],
                "module": r["module"],
                "hypothesis": r["hypothesis"],
                "objective_category": r["objective_category"],
                "evidence_state": r["evidence_state"],
                "created_at": r["created_at"],
                "seeds": json.loads(r["seeds_json"]),
                "config": json.loads(r["config_json"]),
                "metrics": json.loads(r["metrics_json"]),
                "baseline_comparisons": json.loads(r["baselines_json"]),
                "uncertainty": json.loads(r["uncertainty_json"]),
                "resource_usage": json.loads(r["resources_json"]),
                "failed_experiments": json.loads(r["failed_json"]),
                "parent_checkpoint_id": r["parent_checkpoint_id"],
                "descendant_checkpoint_id": r["descendant_checkpoint_id"],
                "code_version": r["code_version"],
                "sha256_digest": r["sha256_digest"],
            })
        return out

    def save_checkpoint(
        self,
        payload: Mapping[str, Any],
        label: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save an immutable, content-bound checkpoint without overwriting parent checkpoints."""
        cid = checkpoint_id or make_id("ckpt")
        dest = self.checkpoints_dir / f"{cid}.pt"
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite existing immutable checkpoint {dest}")
        file_sha = save_state_snapshot(dest, dict(payload))
        g_digest = state_digest(dict(payload))
        now = time.time()
        meta = dict(metadata or {})
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, parent_id, label, genome_digest, file_path,
                    file_sha256, created_at, archived_only, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (cid, parent_id, label, g_digest, str(dest), file_sha, now, canonical_json(meta)),
            )
        return {
            "checkpoint_id": cid,
            "parent_id": parent_id,
            "label": label,
            "genome_digest": g_digest,
            "file_path": str(dest),
            "file_sha256": file_sha,
            "created_at": now,
        }

    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id}")
        payload = load_state(Path(row["file_path"]))
        meta = {
            "checkpoint_id": row["checkpoint_id"],
            "parent_id": row["parent_id"],
            "label": row["label"],
            "genome_digest": row["genome_digest"],
            "file_sha256": row["file_sha256"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
        }
        return payload, meta

    def list_checkpoints(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [
            {
                "checkpoint_id": r["checkpoint_id"],
                "parent_id": r["parent_id"],
                "label": r["label"],
                "genome_digest": r["genome_digest"],
                "file_sha256": r["file_sha256"],
                "created_at": r["created_at"],
                "metadata": json.loads(r["metadata_json"]),
            }
            for r in rows
        ]

    def record_mutation(
        self,
        operator: str,
        delta: Mapping[str, Any],
        inverse: Mapping[str, Any],
        accepted: bool,
        parent_checkpoint_id: Optional[str] = None,
        child_checkpoint_id: Optional[str] = None,
    ) -> str:
        mid = make_id("mut")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mutations (
                    mutation_id, parent_checkpoint_id, child_checkpoint_id,
                    operator, delta_json, inverse_json, accepted, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    parent_checkpoint_id,
                    child_checkpoint_id,
                    operator,
                    canonical_json(dict(delta)),
                    canonical_json(dict(inverse)),
                    1 if accepted else 0,
                    time.time(),
                ),
            )
        return mid

    def record_failed_hypothesis(
        self, module: str, hypothesis: str, reason: str, metrics: Optional[Mapping[str, Any]] = None
    ) -> str:
        fid = make_id("fail")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO failed_hypotheses (failure_id, module, hypothesis, reason, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fid, module, hypothesis, reason, canonical_json(dict(metrics or {})), time.time()),
            )
        return fid

    def record_discovery(
        self,
        title: str,
        module: str,
        evidence_state: EvidenceState,
        experiment_id: str,
        summary: Mapping[str, Any],
        report_path: Optional[str] = None,
        discovery_id: Optional[str] = None,
    ) -> str:
        did = discovery_id or make_id("disc")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO discoveries (
                    discovery_id, title, module, evidence_state, experiment_id,
                    summary_json, report_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (did, title, module, evidence_state.value, experiment_id, canonical_json(dict(summary)), report_path, now, now),
            )
        return did

    def list_discoveries(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM discoveries ORDER BY updated_at DESC").fetchall()
        return [
            {
                "discovery_id": r["discovery_id"],
                "title": r["title"],
                "module": r["module"],
                "evidence_state": r["evidence_state"],
                "experiment_id": r["experiment_id"],
                "summary": json.loads(r["summary_json"]),
                "report_path": r["report_path"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def get_lineage_graph(self) -> Dict[str, Any]:
        ckpts = self.list_checkpoints(limit=500)
        with self._lock, self._connect() as conn:
            muts = conn.execute("SELECT * FROM mutations ORDER BY created_at ASC LIMIT 500").fetchall()
        return {
            "nodes": ckpts,
            "edges": [
                {
                    "mutation_id": m["mutation_id"],
                    "source": m["parent_checkpoint_id"],
                    "target": m["child_checkpoint_id"],
                    "operator": m["operator"],
                    "accepted": bool(m["accepted"]),
                }
                for m in muts
            ],
        }
