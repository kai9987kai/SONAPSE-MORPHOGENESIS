"""Zero-Gain Backward-Compatible Integration Bridge with Sonapse-LAB.

Ensures:
1. ``SynapseConfig.morphogenesis`` defaults to ``None`` and is omitted from
   ``to_dict()`` when absent so pre-Morphogenesis configs/checkpoints remain
   byte-for-byte identical.
2. ``MorphogenesisBusChannel`` implements ``synapse.graft_bus.NeuralGraftProtocol``
   with strict zero-gain identity: when disabled or ``gain == 0.0``, it returns
   ``torch.zeros_like(r)`` bit-for-bit.
3. Read-only importers connect real Sonapse-LAB artifacts (``malecns_types.npz``,
   ``artifacts/tidal_core.pt``, ``synapse.neuroweave.topology.Topology``) into
   the Morphogenesis ``ConnectomeGraph`` interchange format without mutating them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
import torch

from synapse.config import ROOT, VENDOR, tidal_checkpoint_path
from synapse.graft_bus import NeuralGraftProtocol

from .compartment_engine import MultiFidelityConnectome
from .interchange import (
    CompartmentEdge,
    CompartmentKind,
    CompartmentNode,
    ConnectomeEdge,
    ConnectomeGraph,
    DataSourceKind,
    FidelityLevel,
    NeuronMorphologySpec,
    ObjectiveCategory,
)

CHECKPOINT_NAMESPACE = "morphogenesis_v1"


@dataclass(frozen=True)
class MorphogenesisSpecV1:
    """Additive, default-off configuration attached optionally to SynapseConfig."""
    enabled: bool = False
    gain: float = 0.0
    max_gain: float = 0.15
    fidelity: int = 2
    neurons: int = 16
    seed: int = 7
    observe_only: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if not 0.0 <= float(self.gain) <= float(self.max_gain):
            raise ValueError("gain must lie in [0, max_gain]")
        if int(self.fidelity) not in (1, 2, 3, 4, 5):
            raise ValueError("fidelity must be in 1..5")
        if not 1 <= int(self.neurons) <= 512:
            raise ValueError("neurons must be in 1..512")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MorphogenesisSpecV1":
        return cls(
            enabled=bool(d.get("enabled", False)),
            gain=float(d.get("gain", 0.0)),
            max_gain=float(d.get("max_gain", 0.15)),
            fidelity=int(d.get("fidelity", 2)),
            neurons=int(d.get("neurons", 16)),
            seed=int(d.get("seed", 7)),
            observe_only=bool(d.get("observe_only", True)),
        )

    @classmethod
    def max_preset(cls, seed: int = 7, neurons: int = 32) -> "MorphogenesisSpecV1":
        """Bounded, active Max Mode preset for Sonapse-LAB Tandem Max integration."""
        return cls(
            enabled=True,
            gain=0.02,
            max_gain=0.15,
            fidelity=2,
            neurons=max(8, min(128, int(neurons))),
            seed=int(seed),
            observe_only=False,
        )


def coerce_morphogenesis_spec(value: Any) -> Optional[MorphogenesisSpecV1]:
    if value is None:
        return None
    if isinstance(value, MorphogenesisSpecV1):
        return value
    if isinstance(value, Mapping):
        return MorphogenesisSpecV1.from_dict(value)
    raise ValueError("morphogenesis must be None, MorphogenesisSpecV1, or a mapping")


def normalize_saved_morphogenesis(
    current_spec: Optional[MorphogenesisSpecV1], normalized_saved: Dict[str, Any]
) -> bool:
    """Allow legacy checkpoints lacking morphogenesis_v1 to load cleanly (additive rule)."""
    normalized_saved.setdefault("morphogenesis", None)
    saved = normalized_saved.get("morphogenesis")
    saved_spec: Optional[MorphogenesisSpecV1] = None
    if isinstance(saved, MorphogenesisSpecV1):
        saved_spec = saved
    elif isinstance(saved, Mapping):
        try:
            saved_spec = MorphogenesisSpecV1.from_dict(saved)
        except (TypeError, ValueError):
            return False
        normalized_saved["morphogenesis"] = saved_spec
    elif saved is not None:
        return False

    if (
        current_spec is not None
        and current_spec.enabled
        and (saved_spec is None or not saved_spec.enabled)
    ):
        normalized_saved["morphogenesis"] = current_spec
    return True


def build_zero_impact_subconnectome(spec: MorphogenesisSpecV1) -> ConnectomeGraph:
    """Create a small structured connectome suitable for function-preserving grafting."""
    rng = np.random.default_rng(spec.seed)
    neurons: List[NeuronMorphologySpec] = []
    edges: List[ConnectomeEdge] = []
    n = spec.neurons
    for i in range(n):
        sign = 1 if i < int(0.8 * n) else -1
        c_soma = CompartmentNode(f"n{i}_soma", CompartmentKind.SOMA, length_um=20.0, radius_um=8.0, coords_3d=(float(i), 0.0, 0.0))
        c_dend = CompartmentNode(f"n{i}_dend", CompartmentKind.DENDRITE_APICAL, length_um=45.0, radius_um=2.0, coords_3d=(float(i), 1.0, 0.0))
        c_axon = CompartmentNode(f"n{i}_axon", CompartmentKind.AXON_SEGMENT, length_um=80.0, radius_um=1.0, myelin_factor=0.5, coords_3d=(float(i), -1.0, 0.0))
        nspec = NeuronMorphologySpec(
            morphology_id=f"morph_n{i}",
            family="zero_impact_graft_cell",
            name=f"MorphGraftCell-{i}",
            objective_category=ObjectiveCategory.COMPUTATIONAL_USEFULNESS,
            fidelity=FidelityLevel(spec.fidelity),
            compartments=[c_dend, c_soma, c_axon],
            edges=[
                CompartmentEdge(f"n{i}_dend", f"n{i}_soma", weight=0.6, coupling_type="electrical_axial"),
                CompartmentEdge(f"n{i}_soma", f"n{i}_axon", weight=0.8, coupling_type="electrical_axial"),
            ],
            input_compartments=[f"n{i}_dend"],
            output_compartments=[f"n{i}_axon"],
            dale_sign=sign,
        )
        neurons.append(nspec)
    for i in range(n):
        for _ in range(2):
            j = int(rng.integers(0, n))
            if j != i:
                w = float(rng.uniform(0.05, 0.25)) * neurons[i].dale_sign
                edges.append(ConnectomeEdge(pre_id=f"morph_n{i}", post_id=f"morph_n{j}", weight=w, delay_ms=1.0))
    return ConnectomeGraph(
        connectome_id=f"morph_sub_{spec.seed}",
        name="Zero-Impact Morphogenesis Sub-Connectome",
        neurons=neurons,
        edges=edges,
        sensory_ids=[f"morph_n{i}" for i in range(min(4, n))],
        motor_ids=[f"morph_n{i}" for i in range(max(0, n - 4), n)],
        source_kind=DataSourceKind.SYNTHETIC_SIMULATION,
    )


class MorphogenesisBusChannel(NeuralGraftProtocol):
    """Sonapse Graft Bus channel wrapping a MultiFidelityConnectome with zero-gain identity.

    Function-preserving initialization: ``out_proj`` starts at exact zero so attaching
    this graft produces 0.0 residual current until explicitly ramped or trained.
    """

    def __init__(self, host_dim: int, spec: Optional[MorphogenesisSpecV1] = None, name: str = "morphogenesis") -> None:
        self._name = name
        self.spec = spec or MorphogenesisSpecV1()
        self._enabled = bool(self.spec.enabled)
        self.host_dim = int(host_dim)
        self.graph = build_zero_impact_subconnectome(self.spec)
        self.connectome = MultiFidelityConnectome(self.graph, dt_ms=1.0, seed=self.spec.seed)

        rng = np.random.default_rng(self.spec.seed + 404)
        self.in_proj = (rng.standard_normal((self.spec.neurons, self.host_dim)) / max(np.sqrt(self.host_dim), 1.0)).astype(np.float32)
        # Strict function-preserving zero-output initialization!
        self.out_proj = np.zeros((self.host_dim, self.spec.neurons), dtype=np.float32)
        self._cached_proj: Optional[np.ndarray] = None
        self._last_telemetry: Dict[str, Any] = {"steps": 0, "gain": float(self.spec.gain), "observe_only": bool(self.spec.observe_only)}

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self.spec.enabled)

    def set_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError("enabled must be a bool")
        self._enabled = value

    def reset(self, batch_size: int = 1) -> None:
        """Episode boundary: clear microstep cache while preserving learned weights and state."""
        self._cached_proj = None

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(r)
        r_np = r.detach().cpu().numpy().astype(np.float32).ravel()
        if r_np.size != self.host_dim:
            padded = np.zeros(self.host_dim, dtype=np.float32)
            m = min(r_np.size, self.host_dim)
            padded[:m] = r_np[:m]
            r_np = padded
        if advance and (microstep == 0 or self._cached_proj is None):
            ext = self.in_proj @ r_np
            do_adapt = bool(adapt and not self.spec.observe_only)
            res = self.connectome.step(ext, adapt=do_adapt, collect_telemetry=False)
            last_out_f32 = self.connectome.last_outputs.astype(np.float32)
            proj = (self.out_proj @ last_out_f32) * float(self.spec.gain)
            self._cached_proj = proj.astype(np.float32)
            if do_adapt and self.spec.gain > 0.0:
                # Bounded Oja-stabilized Hebbian graft readout update (starts from zero)
                dw = 5e-4 * (np.outer(r_np, last_out_f32) - 0.1 * self.out_proj)
                self.out_proj = np.clip(self.out_proj + dw.astype(np.float32), np.float32(-0.25), np.float32(0.25)).astype(np.float32)
            self._last_telemetry = {
                "steps": int(res["step"]),
                "gain": float(self.spec.gain),
                "observe_only": bool(self.spec.observe_only),
                "step_energy": float(res["step_energy"]),
                "step_flops": int(res["step_flops"]),
                "mean_output": round(float(np.mean(res["outputs"])), 6),
                "out_proj_norm": round(float(np.linalg.norm(self.out_proj)), 6),
            }
        elif self._cached_proj is not None:
            proj = self._cached_proj
        else:
            proj = (self.out_proj @ self.connectome.last_outputs.astype(np.float32)) * float(self.spec.gain)

        if self.spec.observe_only or self.spec.gain == 0.0:
            return torch.zeros_like(r)
        out = torch.from_numpy(proj).to(device=r.device, dtype=r.dtype)
        if out.numel() != r.numel():
            out = out[: r.numel()] if out.numel() > r.numel() else torch.nn.functional.pad(out, (0, r.numel() - out.numel()))
        return out.view_as(r)

    def get_telemetry_dict(self, include_timing: bool = False) -> Dict[str, Any]:
        return dict(self._last_telemetry)

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema": "MorphogenesisCheckpointV1",
            "spec": self.spec.to_dict(),
            "host_dim": int(self.host_dim),
            "enabled": bool(self._enabled),
            "in_proj": self.in_proj.astype(np.float32).copy(),
            "out_proj": self.out_proj.astype(np.float32).copy(),
            "cached_proj": None if self._cached_proj is None else self._cached_proj.astype(np.float32).copy(),
            "connectome": self.connectome.export_state(),
            "last_outputs": self.connectome.last_outputs.copy(),
            "step_count": int(self.connectome.step_count),
            "last_telemetry": dict(self._last_telemetry),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if "enabled" in state:
            self._enabled = bool(state["enabled"])
        if "in_proj" in state:
            self.in_proj = np.asarray(state["in_proj"], dtype=np.float32).copy()
        if "out_proj" in state:
            self.out_proj = np.asarray(state["out_proj"], dtype=np.float32).copy()
        cached = state.get("cached_proj")
        self._cached_proj = None if cached is None else np.asarray(cached, dtype=np.float32).copy()
        if "connectome" in state and isinstance(state["connectome"], Mapping):
            self.connectome.import_state(state["connectome"])
        else:
            if "last_outputs" in state:
                self.connectome.last_outputs = np.asarray(state["last_outputs"], dtype=np.float64).copy()
            self.connectome.step_count = int(state.get("step_count", 0))
        if "last_telemetry" in state and isinstance(state["last_telemetry"], Mapping):
            self._last_telemetry = dict(state["last_telemetry"])


def load_malecns_subgraph(max_nodes: int = 32, seed: int = 42) -> ConnectomeGraph:
    """Import an induced subgraph from the vendored MaleCNS cell-type aggregate (read-only).

    Explicitly labels provenance as biological cell-type aggregate (not individual neuron EM).
    """
    npz_path = VENDOR / "expanse" / "expanse" / "data" / "malecns_types.npz"
    rng = np.random.default_rng(seed)
    if not npz_path.is_file():
        return build_zero_impact_subconnectome(MorphogenesisSpecV1(neurons=max_nodes, seed=seed))

    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.keys())
    n_total = 0
    edges_list: List[Tuple[int, int, float]] = []
    node_signs: Optional[np.ndarray] = None
    if "pre" in data and "post" in data and "weight" in data:
        pre_arr = data["pre"]
        post_arr = data["post"]
        w_arr = data["weight"]
        sign_arr = data["sign"] if "sign" in data else None
        n_total = int(sign_arr.shape[0]) if sign_arr is not None else int(max(pre_arr.max(), post_arr.max()) + 1)
        # Rank types by total outgoing synaptic count for a dense, realistic induced subgraph
        out_deg = np.bincount(pre_arr, minlength=n_total)
        top_idx = np.argsort(out_deg)[-max_nodes:]
        node_map = {int(orig): new_i for new_i, orig in enumerate(top_idx)}
        mask = np.isin(pre_arr, top_idx) & np.isin(post_arr, top_idx) & (pre_arr != post_arr)
        sub_pre = pre_arr[mask]
        sub_post = post_arr[mask]
        sub_w = w_arr[mask]
        for s_i, d_i, w_i in zip(sub_pre, sub_post, sub_w):
            edges_list.append((node_map[int(s_i)], node_map[int(d_i)], float(w_i)))
        if sign_arr is not None:
            node_signs = sign_arr[top_idx]
    else:
        return build_zero_impact_subconnectome(MorphogenesisSpecV1(neurons=max_nodes, seed=seed))

    n_nodes = max_nodes
    neurons: List[NeuronMorphologySpec] = []
    for i in range(n_nodes):
        dale = int(np.sign(node_signs[i])) if (node_signs is not None and node_signs[i] != 0) else (1 if i < int(0.8 * n_nodes) else -1)
        angle = 2.0 * np.pi * i / max(n_nodes, 1)
        c_soma = CompartmentNode(f"mcns_{i}_soma", CompartmentKind.SOMA, length_um=18.0, radius_um=6.5, coords_3d=(float(np.cos(angle) * 10), float(np.sin(angle) * 10), 0.0))
        c_dend = CompartmentNode(f"mcns_{i}_dend", CompartmentKind.DENDRITE_PROXIMAL, length_um=35.0, radius_um=1.8, coords_3d=(float(np.cos(angle) * 11), float(np.sin(angle) * 11), 1.0))
        neurons.append(
            NeuronMorphologySpec(
                morphology_id=f"malecns_{i}",
                family="malecns_cell_type_unit",
                name=f"MaleCNS-Type-{i}",
                objective_category=ObjectiveCategory.BIOLOGICAL_REALISM,
                fidelity=FidelityLevel.MULTI_COMPARTMENT_RECURRENT,
                compartments=[c_dend, c_soma],
                edges=[CompartmentEdge(f"mcns_{i}_dend", f"mcns_{i}_soma", weight=0.75, coupling_type="electrical_axial")],
                input_compartments=[f"mcns_{i}_dend"],
                output_compartments=[f"mcns_{i}_soma"],
                dale_sign=dale,
            )
        )

    # Normalize weights to stable recurrent scale
    max_w = max((abs(w) for _, _, w in edges_list), default=1.0) or 1.0
    conn_edges = [
        ConnectomeEdge(
            pre_id=f"malecns_{s}",
            post_id=f"malecns_{d}",
            weight=float(np.clip((abs(w) / max_w) * 0.65 * neurons[s].dale_sign, -2.0, 2.0)),
            delay_ms=1.0,
            provenance="malecns_v1_cell_type_aggregate",
        )
        for s, d, w in edges_list[: max_nodes * 8]
    ]
    return ConnectomeGraph(
        connectome_id=f"malecns_sub_{max_nodes}_s{seed}",
        name=f"MaleCNS v1.0 Induced Cell-Type Subgraph ({max_nodes} nodes)",
        neurons=neurons,
        edges=conn_edges,
        sensory_ids=[f"malecns_{i}" for i in range(min(4, n_nodes))],
        motor_ids=[f"malecns_{i}" for i in range(max(0, n_nodes - 4), n_nodes)],
        source_kind=DataSourceKind.BIOLOGICAL_RECORDING,
        provenance={
            "source_file": str(npz_path.relative_to(ROOT)),
            "representation": "cell_type_aggregate_induced_subgraph",
            "total_source_types": int(n_total),
            "npz_keys": keys,
            "note": "Janelia MaleCNS v1.0 cell-type aggregate (not single-neuron EM reconstruction).",
        },
    )


def get_tidal_core_path() -> Path:
    return tidal_checkpoint_path()


def run_morphogenesis_laboratory(
    spec: MorphogenesisSpecV1,
    channel: Optional[MorphogenesisBusChannel] = None,
    steps: int = 8,
    seed: int = 7,
) -> Dict[str, Any]:
    """Isolated, read-only bounded Morphogenesis laboratory receipt for Tandem Max Section 08.

    Executes on fresh isolated objects so the live ``IntegratedAgent`` trajectory and RNG
    are never perturbed prior to the Section 05 checkpoint commit.
    """
    from .compiled_engine import CompiledConnectomeEngine
    from ..emergence.causal_emergence import extract_maf_slow_subspace, compute_gaussian_mmi_pid

    rng = np.random.default_rng(int(seed) + 808)
    graph = build_zero_impact_subconnectome(spec)
    compiled = CompiledConnectomeEngine(graph, dt_ms=1.0, seed=int(seed))
    n_steps = max(6, min(32, int(steps)))
    traj = np.zeros((n_steps, len(graph.neurons)), dtype=np.float64)
    for t in range(n_steps):
        stim = rng.normal(0.0, 0.35, size=len(graph.neurons))
        res = compiled.step(stim, adapt=True)
        traj[t] = res["outputs"]

    maf = extract_maf_slow_subspace(traj, k_modes=min(3, max(1, len(graph.neurons) // 4)))
    half = max(1, len(graph.neurons) // 2)
    pid = compute_gaussian_mmi_pid(traj[:, :half], traj[:, half:])
    live_telemetry = channel.get_telemetry_dict(include_timing=False) if channel is not None else None
    return {
        "section": "08 · SONAPSE MORPHOGENESIS Laboratory",
        "status": "completed",
        "spec": spec.to_dict(),
        "compiled_engine_steps": n_steps,
        "neurons": len(graph.neurons),
        "compartments": compiled.n_comp_total,
        "maf_slow_autocorrelations": maf["autocorrelations"],
        "mmi_pid_synergy_nats": pid["synergy"],
        "live_channel_telemetry": live_telemetry,
    }
