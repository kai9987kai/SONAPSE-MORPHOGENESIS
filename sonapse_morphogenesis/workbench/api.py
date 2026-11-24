"""Interactive Scientific Workbench HTTP API for SONAPSE MORPHOGENESIS.

Strictly decouples scientific simulation state from browser rendering so that
closing or disabling the UI never alters or interrupts computation. Mounts
cleanly at ``/api/morphogenesis/*`` on Sonapse-LAB's ``Workbench`` OR runs
standalone via ``python -m sonapse_morphogenesis serve``.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse
import numpy as np

from ..affect import AFFECT_EPISTEMIC_TIERS, run_affective_emergence_investigation
from ..antipode import (
    FAILURE_CATEGORIES,
    NEGATIVE_TRAINING_OBJECTIVES,
    TidalMoEInstrumentor,
    load_real_expanse_failure_corpus,
)
from ..architect import (
    EXPERT_ROLES,
    LEARNING_ALGORITHMS,
    AutonomousResearchDirector,
)
from ..chimera import CHIMERA_MODES, merge_connectomes
from ..core import (
    DevelopmentalGenome,
    ExperimentRegistry,
    MultiFidelityConnectome,
    OrganelleAttachment,
    probe_hardware,
)
from ..curiosity import NOVEL_MODALITIES, PUBLIC_SOURCE_CATALOG
from ..foundry import (
    BIO_HYBRID_REGISTRY,
    CONVENTIONAL_NEURON_FAMILIES,
    GRAFT_NEURON_FAMILIES,
    ML_HYBRID_MECHANISMS,
    ORGANELLE_REGISTRY,
    build_conventional_neuron,
    build_graft_neuron,
    create_organelle,
    list_organelles,
    render_neuron_svg,
)
from ..genesis import (
    CANDIDATE_ORIGINS,
    MUTATION_OPERATORS,
    DevelopmentalArchive,
    apply_genome_mutation,
    generate_candidate_genome,
    screen_large_micro_population,
)
from ..hyperspace import HyperspaceEnvironment
from ..sentinel import GovernanceController
from ..spectra import (
    ARTIFICIAL_SENSORY_MODALITIES,
    BIO_SENSORY_MODALITIES,
    INJECTION_TARGETS,
    MIXING_OPERATORS,
    SIGNAL_ENCODINGS,
    encode_multimodal_stream,
    inject_signal_into_connectome,
    mix_neural_signals,
)
from ..symbiosis import COUPLING_MECHANISMS, run_symbiosis_cycle_experiment


STATIC_DIR = Path(__file__).resolve().parent / "static"


class MorphogenesisWorkbenchAPI:
    """Thread-safe scientific workbench backend with decoupled simulation & registry state."""

    def __init__(self, state_root: Path | str | None = None, seed: int = 7) -> None:
        self.root = Path(state_root or (Path.cwd() / "artifacts" / "morphogenesis_workbench")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.seed = int(seed)
        self.registry = ExperimentRegistry(self.root / "registry")
        self.governance = GovernanceController()
        self.archive = DevelopmentalArchive()
        self.director = AutonomousResearchDirector(registry=self.registry, seed=seed)
        self.hyperspace = HyperspaceEnvironment(dim=4, seed=seed)

        # Initialize active developmental connectome
        self.genome: DevelopmentalGenome = generate_candidate_genome("mixed_custom_standard", n_neurons=14, seed=seed)
        self.archive.store(self.genome)
        self.sim = MultiFidelityConnectome(self.genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
        self.paused = True
        self.activity_history: List[List[float]] = []
        self.chemical_history: List[Dict[str, float]] = []
        self.last_tx = None
        self.last_merge_receipt = None
        self.last_spectra_result: Optional[Dict[str, Any]] = None
        self.last_affect_result: Optional[Dict[str, Any]] = None
        self.last_architect_result: Optional[Dict[str, Any]] = None
        self.last_symbiosis_result: Optional[Dict[str, Any]] = None
        self.last_antipode_result: Optional[Dict[str, Any]] = None

        # Save initial baseline checkpoint so rollback is always ready
        self._save_current_checkpoint("initial_mixed_connectome")

    def _save_current_checkpoint(self, label: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "genome": self.genome.to_dict(),
            "step_count": self.sim.step_count,
            "last_outputs": self.sim.last_outputs.copy(),
        }
        return self.registry.save_checkpoint(payload, label=label, parent_id=parent_id)

    def _connectome_payload(self) -> Dict[str, Any]:
        g = self.genome.connectome
        nodes = []
        n = len(g.neurons)
        for i, u in enumerate(g.neurons):
            inst = self.sim.neurons.get(u.morphology_id)
            ang = 2.0 * np.pi * i / max(n, 1)
            layer_z = -1.2 if u.morphology_id in g.sensory_ids else (1.2 if u.morphology_id in g.motor_ids else 0.0)
            nodes.append({
                "id": u.morphology_id,
                "family": u.family,
                "name": u.name,
                "fidelity": int(u.fidelity),
                "dale_sign": u.dale_sign,
                "x": round(float(3.2 * np.cos(ang)), 3),
                "y": round(float(3.2 * np.sin(ang)), 3),
                "z": round(float(layer_z + 0.4 * np.sin(2.0 * ang)), 3),
                "activity": round(float(self.sim.last_outputs[i]) if i < len(self.sim.last_outputs) else 0.0, 4),
                "atp": round(float(inst.atp_pool), 2) if inst else 100.0,
                "compartments": [c.to_dict() for c in u.compartments],
                "organelles": [o.to_dict() for o in u.organelles],
                "formal_math": u.formal_math,
                "hypothesis": u.hypothesis,
                "role": "sensory" if u.morphology_id in g.sensory_ids else ("motor" if u.morphology_id in g.motor_ids else "association"),
            })
        edges = [
            {"pre": e.pre_id, "post": e.post_id, "weight": round(float(e.weight), 4), "delay_ms": e.delay_ms, "plasticity": e.plasticity_rule}
            for e in g.edges if e.active
        ]
        return {
            "genome_id": self.genome.genome_id,
            "generation": self.genome.generation,
            "connectome_id": g.connectome_id,
            "name": g.name,
            "source_kind": g.source_kind.value,
            "step_count": self.sim.step_count,
            "total_flops": self.sim.total_flops,
            "total_energy": round(self.sim.total_energy, 4),
            "nodes": nodes,
            "edges": edges,
            "neuromodulation": self.genome.neuromodulation_params,
            "activity_history": self.activity_history[-48:],
            "chemical_history": self.chemical_history[-48:],
            "mutation_history": self.genome.mutation_history[-20:],
        }

    def get(self, path: str) -> Dict[str, Any]:
        with self.lock:
            if path == "/api/morphogenesis/status":
                return {
                    "ok": True,
                    "version": "1.0.0",
                    "paused": self.paused,
                    "step_count": self.sim.step_count,
                    "genome_id": self.genome.genome_id,
                    "neuron_count": len(self.genome.connectome.neurons),
                    "synapse_count": self.genome.connectome.synapse_count,
                    "hardware": probe_hardware(),
                    "governance": {
                        "emergency_stopped": self.governance.emergency_stopped,
                        "stop_reason": self.governance.stop_reason,
                        "evaluator_integrity_hash": self.governance.evaluator_integrity_hash,
                    },
                }

            if path == "/api/morphogenesis/catalog":
                return {
                    "ok": True,
                    "graft_neurons": list(GRAFT_NEURON_FAMILIES),
                    "conventional_neurons": list(CONVENTIONAL_NEURON_FAMILIES),
                    "organelles": list_organelles(),
                    "bio_hybrids": sorted(BIO_HYBRID_REGISTRY.keys()),
                    "ml_hybrids": list(ML_HYBRID_MECHANISMS),
                    "candidate_origins": list(CANDIDATE_ORIGINS),
                    "mutation_operators": list(MUTATION_OPERATORS),
                    "chimera_modes": list(CHIMERA_MODES),
                    "spectra_encodings": list(SIGNAL_ENCODINGS),
                    "spectra_mixing_operators": list(MIXING_OPERATORS),
                    "spectra_injection_targets": list(INJECTION_TARGETS),
                    "bio_senses": list(BIO_SENSORY_MODALITIES),
                    "artificial_senses": list(ARTIFICIAL_SENSORY_MODALITIES),
                    "learning_algorithms": list(LEARNING_ALGORITHMS),
                    "research_experts": list(EXPERT_ROLES),
                    "coupling_mechanisms": list(COUPLING_MECHANISMS),
                    "negative_objectives": list(NEGATIVE_TRAINING_OBJECTIVES),
                    "novel_modalities": list(NOVEL_MODALITIES),
                    "public_data_sources": sorted(PUBLIC_SOURCE_CATALOG.keys()),
                    "affect_epistemic_tiers": list(AFFECT_EPISTEMIC_TIERS),
                }

            if path.startswith("/api/morphogenesis/neuron/"):
                fam = path.split("/api/morphogenesis/neuron/", 1)[1].strip("/")
                if fam in GRAFT_NEURON_FAMILIES:
                    spec = build_graft_neuron(fam)
                elif fam in CONVENTIONAL_NEURON_FAMILIES:
                    spec = build_conventional_neuron(fam)
                else:
                    return {"ok": False, "error": f"Unknown neuron family: {fam}"}
                return {
                    "ok": True,
                    "spec": spec.to_dict(),
                    "svg": render_neuron_svg(spec),
                    "physics_summary": [
                        {
                            "compartment_id": c.compartment_id,
                            "kind": c.kind.value,
                            "capacitance_nF": round(c.capacitance_nF, 5),
                            "axial_resistance_MOhm": round(c.axial_resistance_MOhm, 4),
                            "conduction_velocity_m_s": round(c.conduction_velocity_m_s, 3),
                            "transmission_delay_ms": round(c.transmission_delay_ms, 4),
                            "metabolic_cost": round(c.baseline_metabolic_cost, 4),
                        }
                        for c in spec.compartments
                    ],
                }

            if path == "/api/morphogenesis/connectome":
                return {"ok": True, **self._connectome_payload()}

            if path == "/api/morphogenesis/registry":
                return {
                    "ok": True,
                    "experiments": self.registry.list_experiments(limit=30),
                    "checkpoints": self.registry.list_checkpoints(limit=30),
                    "discoveries": self.registry.list_discoveries(),
                    "lineage": self.registry.get_lineage_graph(),
                }

            if path == "/api/morphogenesis/hyperspace":
                return {"ok": True, "slice": self.hyperspace.slice_3d_at_w(), "steps": self.hyperspace.steps}

            return {"ok": False, "error": f"Unknown Morphogenesis GET route: {path}"}

    def post(self, path: str, data: Mapping[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if path == "/api/morphogenesis/emergency_stop":
                if data.get("reset"):
                    self.governance.reset_emergency_stop("human_operator")
                else:
                    self.governance.trigger_emergency_stop(str(data.get("reason", "Manual operator stop")))
                    self.paused = True
                return {"ok": True, "emergency_stopped": self.governance.emergency_stopped, "reason": self.governance.stop_reason}

            permitted, reason = self.governance.verify_action("workbench_session", path)
            if not permitted:
                return {"ok": False, "error": reason}

            if path == "/api/morphogenesis/connectome/build":
                origin = str(data.get("origin", "mixed_custom_standard"))
                n_neurons = int(np.clip(int(data.get("n_neurons", 14)), 4, 48))
                seed = int(data.get("seed", self.seed))
                self.genome = generate_candidate_genome(origin, n_neurons=n_neurons, seed=seed)
                self.archive.store(self.genome)
                self.sim = MultiFidelityConnectome(self.genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=seed)
                self.activity_history.clear()
                self.chemical_history.clear()
                ck = self._save_current_checkpoint(f"build_{origin}")
                return {"ok": True, "checkpoint": ck, **self._connectome_payload()}

            if path == "/api/morphogenesis/step":
                steps = int(np.clip(int(data.get("steps", 1)), 1, 64))
                adapt = bool(data.get("adapt", False))
                ext_gain = float(data.get("stimulus_gain", 0.8))
                if data.get("reset"):
                    self.sim.reset()
                    self.activity_history.clear()
                    self.chemical_history.clear()
                last_out: Dict[str, Any] = {}
                for _ in range(steps):
                    t = self.sim.step_count
                    stim = np.array([
                        ext_gain * np.sin(0.28 * t),
                        ext_gain * np.cos(0.19 * t),
                        ext_gain * np.sin(0.45 * t),
                        ext_gain * 0.5,
                    ])
                    mods = {
                        "da": float(self.genome.neuromodulation_params.get("da_gain", 1.0) * 0.3),
                        "ach": float(self.genome.neuromodulation_params.get("ach_gain", 0.8) * 0.25),
                        "serotonin": float(self.genome.neuromodulation_params.get("serotonin_gain", 0.6) * 0.2),
                    }
                    last_out = self.sim.step(stim, modulators=mods, reward_signal=0.15 if adapt else 0.0, adapt=adapt)
                    self.activity_history.append([round(float(x), 4) for x in last_out["outputs"][:12]])
                    self.chemical_history.append({"paracrine": last_out["paracrine_field"], **mods})
                return {"ok": True, **self._connectome_payload()}

            if path == "/api/morphogenesis/mutate":
                if data.get("rollback") and self.last_tx is not None:
                    self.genome = self.last_tx.rollback()
                    self.sim = MultiFidelityConnectome(self.genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=self.seed)
                    self.last_tx = None
                    return {"ok": True, "rolled_back": True, **self._connectome_payload()}
                op = data.get("operator") or None
                seed = int(data.get("seed", int(time.time()) % 100000))
                child, tx = apply_genome_mutation(self.genome, operator=op, seed=seed)
                self.genome = child
                self.last_tx = tx
                self.archive.store(child)
                self.archive.record_tx(tx, accepted=True)
                self.sim = MultiFidelityConnectome(self.genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=self.seed)
                self.registry.record_mutation(tx.operator, tx.delta_summary, {"parent_digest": tx.parent_digest}, accepted=True)
                return {"ok": True, "transaction": {"id": tx.transaction_id, "operator": tx.operator, "delta": tx.delta_summary}, **self._connectome_payload()}

            if path == "/api/morphogenesis/neuron/edit":
                # Interactive compartment lesion/repair or organelle attachment on live connectome
                nid = str(data.get("neuron_id", ""))
                action = str(data.get("action", "attach_organelle"))
                if nid not in self.sim.neurons:
                    return {"ok": False, "error": f"Unknown neuron_id: {nid}"}
                inst = self.sim.neurons[nid]
                if action == "lesion" and inst.comp_ids:
                    cid = str(data.get("compartment_id", inst.comp_ids[0]))
                    inst.lesion_compartment(cid)
                elif action == "repair" and inst.comp_ids:
                    cid = str(data.get("compartment_id", inst.comp_ids[0]))
                    inst.repair_compartment(cid)
                elif action == "attach_organelle":
                    otype = str(data.get("organelle_type", "mnemosome"))
                    cid = str(data.get("compartment_id", inst.comp_ids[0]))
                    att = OrganelleAttachment(f"ui_{otype}_{len(inst.spec.organelles)}", otype, cid)
                    inst.spec.organelles.append(att)
                    inst.organelles[att.organelle_id] = create_organelle(att, seed=self.seed)
                    inst.organelle_comp_idx[att.organelle_id] = inst.id_to_idx[cid]
                return {"ok": True, **self._connectome_payload()}

            if path == "/api/morphogenesis/chimera/merge":
                orig_a = str(data.get("parent_a_origin", "mixed_custom_standard"))
                orig_b = str(data.get("parent_b_origin", "fractal_topology"))
                mode = str(data.get("mode", "synaptic_weight_interpolation"))
                ratio_a = float(data.get("ratio_a", 0.5))
                n = int(np.clip(int(data.get("n_neurons", 10)), 4, 24))
                ga = generate_candidate_genome(orig_a, n_neurons=n, seed=self.seed).connectome
                gb = generate_candidate_genome(orig_b, n_neurons=n, seed=self.seed + 19).connectome
                child, rec = merge_connectomes(ga, gb, mode=mode, ratio_a=ratio_a, seed=self.seed)
                self.last_merge_receipt = rec
                if data.get("load_into_active", True):
                    self.genome = DevelopmentalGenome(genome_id=child.connectome_id, connectome=child, generation=self.genome.generation + 1, parent_ids=[ga.connectome_id, gb.connectome_id])
                    self.sim = MultiFidelityConnectome(child, dt_ms=1.0, organelle_factory=create_organelle, seed=self.seed)
                return {
                    "ok": True,
                    "merge_id": rec.merge_id,
                    "mode": rec.mode,
                    "ratio_a": rec.ratio_a,
                    "alignment_score": rec.alignment_score,
                    "child_neurons": child.neuron_count,
                    "child_synapses": child.synapse_count,
                    "correspondence_sample": rec.correspondence_pairs[:8],
                    "connectome": self._connectome_payload(),
                }

            if path == "/api/morphogenesis/spectra/inject":
                mod_a = str(data.get("modality_a", "math_function"))
                mod_b = str(data.get("modality_b", "text"))
                op = str(data.get("operator", "phase_coupling"))
                enc = str(data.get("encoding", "rate"))
                target = str(data.get("target_scope", "dendrite"))
                strength = float(data.get("strength", 0.9))
                sa = encode_multimodal_stream(mod_a, data.get("payload_a", "chirp"), encoding=enc, length=32, channels=6, seed=self.seed)["signal"]
                sb = encode_multimodal_stream(mod_b, data.get("payload_b", "sonapse"), encoding=enc, length=32, channels=6, seed=self.seed + 1)["signal"]
                mixed = mix_neural_signals(sa, sb, operator=op, alpha=float(data.get("alpha", 0.5)), seed=self.seed)
                res = inject_signal_into_connectome(self.sim, mixed, target_scope=target, strength=strength, start_step=4, duration=20, seed=self.seed)
                self.last_spectra_result = res
                return {"ok": True, "result": res}

            if path == "/api/morphogenesis/affect/run":
                res = run_affective_emergence_investigation(seed=int(data.get("seed", self.seed)))
                self.last_affect_result = res
                return {"ok": True, "result": res}

            if path == "/api/morphogenesis/hyperspace/step":
                act = data.get("action", [0.6, 0.1, 0.0, 0.7])
                if data.get("reset"):
                    self.hyperspace.reset(self.seed)
                _, r, done, info = self.hyperspace.step(act)
                return {"ok": True, "reward": round(r, 4), "done": done, "info": info, "slice": self.hyperspace.slice_3d_at_w()}

            if path == "/api/morphogenesis/antipode/probe":
                inst = TidalMoEInstrumentor()
                exs = inst.probe_inputs(n_samples=12, seed=self.seed)
                inst.close()
                exp_c = load_real_expanse_failure_corpus()
                res = {
                    "tidal_probes": len(exs),
                    "tidal_failures": sum(1 for e in exs if e.is_failure),
                    "tidal_layers_hooked": exs[0].metadata.get("hooked_layers", 0) if exs else 0,
                    "tidal_sample": [
                        {"id": e.example_id, "target": e.target_output, "observed": e.observed_output, "is_failure": e.is_failure, "confidence": e.confidence, "activations_head": (e.internal_activations or [])[:8]}
                        for e in exs[:6]
                    ],
                    "expanse_sources": exp_c["sources_loaded"],
                    "expanse_verified_failures": exp_c["verified_failures"],
                    "expanse_verified_successes": exp_c["verified_successes"],
                }
                self.last_antipode_result = res
                return {"ok": True, "result": res}

            if path == "/api/morphogenesis/symbiosis/run":
                coupling = str(data.get("coupling", "learned_neural_adapters"))
                res = run_symbiosis_cycle_experiment(couplings=(coupling,), seed=self.seed)
                self.last_symbiosis_result = res
                return {"ok": True, "result": res}

            if path == "/api/morphogenesis/genesis/screen":
                pop_sz = int(np.clip(int(data.get("population_size", 512)), 32, 2048))
                res = screen_large_micro_population(population_size=pop_sz, n_neurons=12, steps=48, seed=self.seed)
                return {"ok": True, "result": res}

            if path == "/api/morphogenesis/architect/cycle":
                res = self.director.run_research_cycle(mini_neurons=8, scale_neurons=14)
                self.last_architect_result = res
                return {"ok": True, "result": res}

            if path == "/api/morphogenesis/checkpoint/save":
                label = str(data.get("label", f"manual_step_{self.sim.step_count}"))[:48]
                ck = self._save_current_checkpoint(label)
                return {"ok": True, "checkpoint": ck}

            if path == "/api/morphogenesis/checkpoint/rollback":
                cid = str(data.get("checkpoint_id", ""))
                payload, meta = self.registry.load_checkpoint(cid)
                self.genome = DevelopmentalGenome.from_dict(payload["genome"])
                self.sim = MultiFidelityConnectome(self.genome.connectome, dt_ms=1.0, organelle_factory=create_organelle, seed=self.seed)
                return {"ok": True, "restored": meta, **self._connectome_payload()}

            return {"ok": False, "error": f"Unknown Morphogenesis POST route: {path}"}


def serve_standalone(host: str = "127.0.0.1", port: int = 8766, state_root: Path | str | None = None) -> None:
    """Launch standalone loopback-only SONAPSE MORPHOGENESIS workbench server."""
    api = MorphogenesisWorkbenchAPI(state_root=state_root)
    html_path = STATIC_DIR / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path).path
            if parsed in ("/", "/morphogenesis", "/morphogenesis/"):
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.startswith("/api/morphogenesis/"):
                self._send_json(200, api.get(parsed))
                return
            self._send_json(404, {"ok": False, "error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(min(length, 1024 * 1024)) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                self._send_json(400, {"ok": False, "error": "Invalid JSON"})
                return
            if parsed.startswith("/api/morphogenesis/"):
                self._send_json(200, api.post(parsed, body))
                return
            self._send_json(404, {"ok": False, "error": "Not found"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SONAPSE MORPHOGENESIS Workbench running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
