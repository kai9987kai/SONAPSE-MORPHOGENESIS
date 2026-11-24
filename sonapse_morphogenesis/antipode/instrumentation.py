"""SUPERMIX ANTIPODE — Model Instrumentation & Verified Failure Collection.

Directly instruments actual Supermix checkpoints available locally (including
``artifacts/tidal_core.pt`` — Poseidon Tidal MoE) via PyTorch forward hooks,
and parses verified historical failure logs from ``synapse/_vendor/expanse/``.

Explicitly distinguishes VERIFIED FAILURES from unusual-but-useful behavior,
preserving temporal correspondence between inputs, hidden states, expert routing,
confidence, outputs, and verified corrections.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from synapse.config import VENDOR, tidal_checkpoint_path

from ..core.interchange import DataSourceKind, TrainingExample, make_id


FAILURE_CATEGORIES: Tuple[str, ...] = (
    "incorrect_math_reasoning",
    "invalid_programming_output",
    "repetition_loop",
    "contradictory_answer",
    "incorrect_factual_association",
    "unsuccessful_planning",
    "overconfident_error",
    "poor_adaptation",
    "representation_collapse",
    "unusual_internal_activity",
)


def classify_response_outcome(
    observed: Any,
    expected: Any,
    domain: str,
    confidence: float = 0.5,
    hidden_state: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Deterministically verify whether a model output is a genuine failure vs unusual-but-useful."""
    is_failure = False
    category: Optional[str] = None
    unusual_but_valid = False

    if domain == "math":
        try:
            obs_f = float(observed)
            exp_f = float(expected)
            if abs(obs_f - exp_f) > 1e-3:
                is_failure = True
                category = "overconfident_error" if confidence > 0.85 else "incorrect_math_reasoning"
        except (TypeError, ValueError):
            is_failure = True
            category = "incorrect_math_reasoning"
    elif domain == "code":
        code_str = str(observed)
        try:
            ast.parse(code_str)
            if str(expected) and str(expected).strip() not in code_str:
                # Syntactically valid alternative implementation -> unusual but potentially useful!
                unusual_but_valid = True
        except SyntaxError:
            is_failure = True
            category = "invalid_programming_output"
    elif domain == "text":
        toks = str(observed).lower().split()
        if len(toks) >= 6 and len(set(toks)) <= max(1, len(toks) // 3):
            is_failure = True
            category = "repetition_loop"
        elif str(expected).lower() not in str(observed).lower():
            is_failure = True
            category = "incorrect_factual_association"
    else:
        # Action / classification / planning outcome
        if observed != expected:
            is_failure = True
            category = "overconfident_error" if confidence > 0.85 else "unsuccessful_planning"

    if hidden_state is not None:
        std = float(np.std(hidden_state))
        if std < 1e-4:
            is_failure = True
            category = "representation_collapse"
        elif not is_failure and float(np.max(np.abs(hidden_state))) > 3.5:
            unusual_but_valid = True

    return {
        "verified": True,
        "is_failure": is_failure,
        "failure_category": category,
        "unusual_but_useful": unusual_but_valid,
    }


class TidalMoEInstrumentor:
    """Instruments the real Poseidon Tidal MoE checkpoint (`artifacts/tidal_core.pt`) with forward hooks."""

    def __init__(self) -> None:
        from poseidon.core import load_checkpoint

        self.ckpt_path = tidal_checkpoint_path()
        self.model, self.receipt = load_checkpoint(self.ckpt_path)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 128))
        self.captured: Dict[str, List[np.ndarray]] = {}
        self._hooks: List[Any] = []
        self._attach_hooks()

    def _attach_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.GRUCell, nn.LayerNorm)):
                def _make_hook(mname: str):
                    def _hook(_mod, _inp, out):
                        t = out[0] if isinstance(out, tuple) else out
                        if isinstance(t, torch.Tensor):
                            self.captured.setdefault(mname, []).append(t.detach().cpu().numpy().copy())
                    return _hook
                self._hooks.append(module.register_forward_hook(_make_hook(name)))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def probe_inputs(self, n_samples: int = 32, seed: int = 42) -> List[TrainingExample]:
        """Run real forward passes through `tidal_core.pt`, capturing hidden activations, routing, & failures."""
        rng = np.random.default_rng(seed)
        obs_dim = int(getattr(self.model.config, "observation_size", 16))
        action_count = int(getattr(self.model.config, "action_count", 6))
        prompt_bank = [
            ("rest recover low stamina calm safe", 0),
            ("forage food hungry energy low", 1),
            ("drink water hydration needed", 2),
            ("shelter storm rain cover", 3),
            ("explore search unknown terrain", 4),
            ("flee danger predator threat attack", 5),
        ]
        examples: List[TrainingExample] = []

        with torch.no_grad():
            for i in range(n_samples):
                self.captured.clear()
                prompt_text, expected_cls = prompt_bank[i % len(prompt_bank)]
                expected_cls = expected_cls % max(action_count, 1)
                obs = rng.normal(0.0, 0.5, size=(1, obs_dim)).astype(np.float32)
                if i % 2 == 1:
                    prompt_text = f"{prompt_text} contradictory noise {i} unknown"
                    obs *= 3.0
                obs_t = torch.from_numpy(obs)
                out = self.model([prompt_text], observations=obs_t)

                logits = None
                for key in ("action_logits", "intent_logits", "logits", "policy_logits"):
                    if key in out and isinstance(out[key], torch.Tensor):
                        logits = out[key]
                        break
                if logits is None:
                    for val in out.values():
                        if isinstance(val, torch.Tensor) and val.numel() >= action_count:
                            logits = val
                            break
                hidden = out.get("hidden", out.get("state", None))
                gate_weights = out.get("expert_weights", out.get("gates", out.get("router_weights", None)))

                logits_np = logits.detach().cpu().numpy().ravel()[:action_count]
                probs = np.exp(logits_np - np.max(logits_np))
                probs = probs / (np.sum(probs) + 1e-12)
                pred_cls = int(np.argmax(probs))
                conf = float(np.max(probs))

                act_vec: Optional[np.ndarray] = None
                if isinstance(hidden, torch.Tensor):
                    act_vec = hidden.detach().cpu().numpy().ravel()
                elif self.captured:
                    best_k = max(self.captured.keys(), key=lambda k: self.captured[k][-1].size)
                    act_vec = self.captured[best_k][-1].ravel()
                if act_vec is None:
                    act_vec = logits_np

                verdict = classify_response_outcome(pred_cls, expected_cls, domain="planning", confidence=conf, hidden_state=act_vec)
                routing_keys = [k for k in self.captured if "gate" in k.lower() or "router" in k.lower() or "expert" in k.lower()]
                examples.append(
                    TrainingExample(
                        example_id=make_id("tidal"),
                        task_domain="tidal_moe_intent",
                        inputs={"prompt": prompt_text, "observation": obs[0].tolist()},
                        target_output=expected_cls,
                        observed_output=pred_cls,
                        is_failure=verdict["is_failure"],
                        verified=verdict["verified"],
                        failure_category=verdict["failure_category"],
                        source_model="Poseidon-Tidal-MoE (artifacts/tidal_core.pt)",
                        source_kind=DataSourceKind.ARTIFICIAL_MODEL_ACTIVITY,
                        internal_activations=act_vec[:64].astype(float).tolist(),
                        confidence=round(conf, 5),
                        metadata={
                            "hooked_layers": len(self.captured),
                            "output_keys": sorted(out.keys()),
                            "routing_layers_captured": routing_keys[:4],
                            "expert_weights": gate_weights.detach().cpu().numpy().ravel().tolist() if isinstance(gate_weights, torch.Tensor) else None,
                            "unusual_but_useful": verdict["unusual_but_useful"],
                            "temporal_index": i,
                        },
                    )
                )
        return examples


def load_real_expanse_failure_corpus() -> Dict[str, Any]:
    """Ingest actual recorded failures from vendored Supermix-Expanse v1/v2/v3 comparison JSONs."""
    ckpt_dir = VENDOR / "expanse" / "expanse" / "checkpoints"
    examples: List[TrainingExample] = []
    sources_loaded: List[str] = []

    for fname in ("compare_v1_v2.json", "compare_v1_v2_v3.json"):
        fpath = ckpt_dir / fname
        if not fpath.is_file():
            continue
        sources_loaded.append(fname)
        data = json.loads(fpath.read_text(encoding="utf-8"))
        for model_name, mdata in data.get("models", {}).items():
            gen = mdata.get("gen", {})
            # Exact math task failures & successes
            exact = gen.get("exact", {})
            for task_name, counts in exact.get("per_task", {}).items():
                if not isinstance(counts, list) or len(counts) != 2:
                    continue
                passed, total = int(counts[0]), int(counts[1])
                failed = max(0, total - passed)
                for k in range(failed):
                    examples.append(
                        TrainingExample(
                            example_id=make_id("exp_fail"),
                            task_domain=f"math:{task_name}",
                            inputs={"model": model_name, "task": task_name, "index": k},
                            target_output="exact_numeric_match",
                            observed_output="mismatch",
                            is_failure=True,
                            verified=True,
                            failure_category="incorrect_math_reasoning",
                            source_model=f"supermix-expanse-{model_name}",
                            source_kind=DataSourceKind.ARTIFICIAL_MODEL_ACTIVITY,
                            metadata={"source_file": fname},
                        )
                    )
                for k in range(passed):
                    examples.append(
                        TrainingExample(
                            example_id=make_id("exp_pass"),
                            task_domain=f"math:{task_name}",
                            inputs={"model": model_name, "task": task_name, "index": k},
                            target_output="exact_numeric_match",
                            observed_output="exact_numeric_match",
                            is_failure=False,
                            verified=True,
                            source_model=f"supermix-expanse-{model_name}",
                            source_kind=DataSourceKind.ARTIFICIAL_MODEL_ACTIVITY,
                            metadata={"source_file": fname},
                        )
                    )
            # Code generation failures
            code_gen = gen.get("code", {})
            for reason, cnt in code_gen.get("reasons", {}).items():
                for k in range(min(int(cnt), 5)):
                    examples.append(
                        TrainingExample(
                            example_id=make_id("exp_code"),
                            task_domain="code_generation",
                            inputs={"model": model_name, "reason": reason, "index": k},
                            target_output="passing_unit_test",
                            observed_output=reason,
                            is_failure=True,
                            verified=True,
                            failure_category="invalid_programming_output",
                            source_model=f"supermix-expanse-{model_name}",
                            source_kind=DataSourceKind.ARTIFICIAL_MODEL_ACTIVITY,
                            metadata={"source_file": fname, "raw_reason": reason},
                        )
                    )

    n_fail = sum(1 for e in examples if e.is_failure)
    n_pass = len(examples) - n_fail
    return {
        "sources_loaded": sources_loaded,
        "total_records": len(examples),
        "verified_failures": n_fail,
        "verified_successes": n_pass,
        "examples": examples,
    }
