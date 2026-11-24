"""Hardware Efficiency Profiler, Sparse Batch Engine & Progressive Selection Funnel.

Prioritizes CPU-first execution on standard Windows laptops / Surface Pro ARM64/x64,
with optional PyTorch accelerator probing and graceful CPU fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch


SCALE_TARGETS: Dict[str, Tuple[int, int]] = {
    "cellular_experiment": (1, 32),
    "micro_connectome": (32, 256),
    "small_connectome": (256, 2048),
    "medium_connectome": (2048, 16384),
    "large_connectome": (16384, 150000),
}


@dataclass
class ResourceBudget:
    max_wall_seconds: float = 30.0
    max_memory_mb: float = 1024.0
    max_cpu_threads: int = min(4, os.cpu_count() or 1)
    max_candidates: int = 2000
    allow_accelerator: bool = False


def probe_hardware() -> Dict[str, Any]:
    """Report available CPU/accelerator execution capabilities without side effects."""
    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return {
        "cpu_logical_cores": os.cpu_count() or 1,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": cuda_ok,
        "mps_available": mps_ok,
        "selected_device": "cuda" if cuda_ok else ("mps" if mps_ok else "cpu"),
        "scale_targets": {k: list(v) for k, v in SCALE_TARGETS.items()},
    }


def measure_callable(fn: Callable[[], Any]) -> Tuple[Any, Dict[str, Any]]:
    """Measure wall-clock time, CPU time, and peak Python allocation of a callable."""
    tracemalloc.start()
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        result = fn()
    finally:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        cpu_ms = (time.process_time() - c0) * 1000.0
        current_b, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, {
        "wall_ms": round(wall_ms, 3),
        "cpu_ms": round(cpu_ms, 3),
        "peak_memory_kb": round(peak_b / 1024.0, 2),
        "current_memory_kb": round(current_b / 1024.0, 2),
    }


class VectorizedMicroBatchSimulator:
    """High-throughput vectorized CPU engine for screening thousands of micro-connectomes.

    Runs B candidate recurrent connectomes of size N simultaneously in float32 using
    batched matrix multiplication (`np.einsum`), allowing 1,000+ candidates to be
    screened in milliseconds on CPU before promoting winners to full multi-compartment
    simulation.
    """

    def __init__(self, weights: np.ndarray, decays: np.ndarray, input_proj: np.ndarray) -> None:
        # weights: [B, N, N], decays: [B, N], input_proj: [B, N, D_in]
        self.weights = np.asarray(weights, dtype=np.float32)
        self.decays = np.asarray(decays, dtype=np.float32)
        self.input_proj = np.asarray(input_proj, dtype=np.float32)
        self.batch_size, self.n_neurons, _ = self.weights.shape

    def rollout(self, stimulus_seq: np.ndarray) -> np.ndarray:
        """Roll out all B candidates over stimulus sequence [T, D_in] -> [T, B, N]."""
        T = stimulus_seq.shape[0]
        stim = np.asarray(stimulus_seq, dtype=np.float32)
        states = np.zeros((T, self.batch_size, self.n_neurons), dtype=np.float32)
        h = np.zeros((self.batch_size, self.n_neurons), dtype=np.float32)
        one_minus = 1.0 - self.decays
        # Precompute external drive [T, B, N]
        ext = np.einsum("td,bnd->tbn", stim, self.input_proj)
        for t in range(T):
            rec = np.einsum("bij,bj->bi", self.weights, h)
            h = self.decays * h + one_minus * np.tanh(rec + ext[t])
            states[t] = h
        return states


@dataclass
class FunnelStageResult:
    stage_name: str
    candidates_in: int
    candidates_out: int
    wall_ms: float
    best_score: float
    median_score: float


class ProgressiveSelectionFunnel:
    """Multi-stage progressive screening funnel (cheap screening -> higher fidelity)."""

    def __init__(self, budget: Optional[ResourceBudget] = None) -> None:
        self.budget = budget or ResourceBudget()

    def run(
        self,
        candidates: Sequence[Any],
        stages: Sequence[Tuple[str, Callable[[Any], float], int]],
    ) -> Dict[str, Any]:
        """Filter candidates sequentially across (stage_name, scorer, top_k) stages."""
        t_start = time.perf_counter()
        active = list(candidates[: self.budget.max_candidates])
        stage_logs: List[Dict[str, Any]] = []
        scored: List[Tuple[float, Any]] = []

        for stage_name, scorer, keep_k in stages:
            if not active:
                break
            if (time.perf_counter() - t_start) > self.budget.max_wall_seconds:
                break
            st0 = time.perf_counter()
            scored = []
            for cand in active:
                if (time.perf_counter() - t_start) > self.budget.max_wall_seconds:
                    break
                val = float(scorer(cand))
                if not np.isfinite(val):
                    val = -1e9
                scored.append((val, cand))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            vals = [s[0] for s in scored] if scored else [0.0]
            kept = [c for _, c in scored[: max(1, min(keep_k, len(scored)))]]
            stage_logs.append({
                "stage_name": stage_name,
                "candidates_in": len(active),
                "candidates_out": len(kept),
                "wall_ms": round((time.perf_counter() - st0) * 1000.0, 3),
                "best_score": round(float(max(vals)), 6),
                "median_score": round(float(np.median(vals)), 6),
            })
            active = kept

        return {
            "winners": active,
            "final_scores": [s[0] for s in scored[: len(active)]],
            "stages": stage_logs,
            "total_wall_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }
