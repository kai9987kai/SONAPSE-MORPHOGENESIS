# NEUROWEAVE → Sonapse integration contract (v1)

This file is the build contract for the `synapse/neuroweave/` package. It fixes
module boundaries, public APIs and invariants so subsystems can be implemented
independently and composed by `engine.py`.

Source studied: [NEUROWEAVE-LAB 0.4.0](https://github.com/kai9987kai/NEUROWEAVE-LAB)
(that repository's `index.html`, `docs/ARCHITECTURE.md`, `docs/MODEL_FORMAT.md`
and `tests/smoke-test.cjs`; they are not part of this repository). NEUROWEAVE is
a browser/JS prototype; this package re-implements its mechanisms in NumPy for
Sonapse. No NEUROWEAVE source is copied verbatim.

## Non-negotiable invariants

1. **Additive only.** No existing Sonapse module, default, CLI flag, route, DOM
   id, checkpoint namespace or test is removed or changed in behaviour.
   `scripts/neuroweave_compat_manifest.py --check` must pass.
2. **Neutral by default.** `NeuroWeaveSpecV1()` is disabled, gain 0, no
   mechanisms. `SynapseConfig().neuroweave is None` and is omitted from
   `to_dict()` so legacy checkpoint schemas stay byte-identical.
3. **All-off identity.** With zero mechanisms, `NeuroWeaveBrain.step` uses the
   NEUROWEAVE v0.1 baseline update. A candidate brain with all mechanisms off
   must match a baseline brain built from the same seed to `1e-12`.
4. **Zero-initialised grafts.** Any new contribution into an existing path
   (graft-bus current, specialist predictions, fusion weights, routing gains)
   starts neutral: gain 0 / identity / zero weights.
5. **Determinism.** Every random draw comes from `numpy.random.default_rng`
   seeded from the spec seed (plus fixed per-subsystem offsets). No global RNG,
   no wall-clock in computed values (wall-clock may appear only in clearly
   named `*_wallclock_*` / `latency_ms` telemetry fields).
6. **Bounded & finite.** All state is finite; weights, traces, gains, modulator
   levels and affect variables are clamped to documented bounds. Dale-style
   source sign is never flipped by plasticity.
7. **Honest evidence.** Every receipt carries
   `evidence_tier: 'synthetic_abstraction'`. Nothing claims biological fidelity,
   emotions, consciousness, FPGA execution or measured hardware speed.
8. **Privacy.** Raw camera frames / audio buffers are never stored, persisted
   or exported. Only derived scalar features are kept; `raw_media_stored` is
   always `False` in manifests.
9. NumPy only inside `synapse/neuroweave/` except `bridge.py` (torch adapter).
   Python 3.10+, type hints, no new third-party dependencies.

## Shared spec

`synapse/neuroweave/spec.py` defines (existing names are part of the contract):
`NeuroWeaveSpecV1`, `MECHANISM_IDS` (45 ids; first 31 are NEUROWEAVE's),
`EXPERIMENTS` (24), `SENSORY_CHANNELS` (16), `TOPOLOGIES` (12), `REGIONS` (10),
`SOMA_ROLES` (8), `DENDRITES` (4), `NEUROMODULATORS` (6), `AFFECT_VARIABLES` (7),
`TASK_INPUT_DIM = 8`, `SENSORY_DIM = 16`.

Every module exposes `export_state() -> dict` (JSON-serialisable: lists, floats,
ints, str, bool, None only) and `import_state(state: dict) -> None` that
validates shapes/finite values and raises `ValueError` on malformed input
without partially mutating (validate first, then assign).

## Module APIs

### topology.py
- `build_topology(name, n, *, seed, fan_in=6, inhibitory_fraction=0.2, regions=REGIONS) -> Topology`
- `Topology` dataclass: `n`, `src: np.ndarray[int64]`, `dst`, `weight: float64`
  (signed by source sign), `delay: int64 in [1,3]`, `sign: np.ndarray[int8]`
  per neuron (+1/-1), `region: np.ndarray[int64]` per neuron (index into REGIONS),
  `name`, `seed`. No self loops, no duplicate edges.
- All 12 names in `TOPOLOGIES`; `degree_shuffled` uses double-edge swaps from the
  clustered source graph and preserves every in/out degree exactly.
- `topology_metrics(topo) -> dict`: in/out degree histograms + mean/std/max/CV,
  density, mean clustering coefficient (directed→undirected), modularity of the
  region partition, mean shortest path + global efficiency (BFS, unweighted),
  hub concentration (top-10% in-degree share and Gini), per-region-pair edge
  counts and regional communication efficiency.

### cells.py
- `CellPopulation(n, *, seed, regions, signs)` with float64 arrays of length n:
  dendrites `proximal, distal, apical, basal`; somas `primary, secondary, fast,
  slow, context, integrator, burst, gating`; `membrane, threshold, refractory,
  spike, calcium, adaptation, trace, hillock, branch_a, branch_b, terminal,
  release, homeo_gain, astrocyte, myelin, utility, consolidation, pred_estimate,
  pred_error, surprise, tm_fast, tm_mid, tm_slow`, plus `output` (the value
  transmitted to synapses).
- `CellPopulation.step(drive: CompartmentDrive, flags: frozenset[str], controls: dict) -> None`
  where `CompartmentDrive` has arrays `proximal, distal, apical, basal`
  (recurrent/synaptic input per compartment) and `external` (feed-forward input).
  `controls` keys (all optional, neutral defaults): `gain` (1.0),
  `threshold_shift` (0.0), `attention` (1.0), `plasticity_scale` (1.0).
- Baseline path (no neuron flags): v0.1 update — `fast`, `slow`, `context`
  states, `output = tanh(...)`. Mechanism flags (`multi_soma`,
  `extended_somas`, `dendritic_compartments`, `membrane_dynamics`,
  `axonal_branches`, `astrocyte_regulation`, `homeostatic_governor`,
  `predictive_coding`, `hierarchical_temporal_memory`, `q88_state`) change the
  update only when present.
- `grow(k, region)` appends k neurons (neurogenesis); `reset_units(idx)` re-inits.

### connectome.py
- `SynapseTable.from_topology(topo, seed)`; arrays `src,dst,w,delay,trace,
  eligibility,utility,pre_spike,post_spike,routing_gain,myelin,last_signal,
  lr_scale,updates,created_step` and a ring-buffer delay queue.
- `transmit(output, flags, *, tile_size, uncertainty_gate=1.0, route_gain=None) -> CompartmentDrive`
  — exact same totals whether tiled or not (tiling only changes traversal order
  and is recorded in a profile). `quantised_synapses` uses signed INT8 grid.
- Learning: `reward_update(reward, lr, flags)`, `stdp_update(spikes, flags)`,
  `metaplastic_scale()`, `structural_step(step, rng, max_synapses) -> list[event]`,
  `downscale(factor)`, `consolidate()`. Events are dicts
  `{step, kind: 'prune'|'grow'|'rewire'|'neuron_birth'|'repurpose', src, dst, reason, utility}`.
- `lesion(mask_edges)` returns a lesioned *copy*; never mutates the original.

### brain.py
- `NeuroWeaveBrain(spec)` = topology + cells + synapses + readout.
- `step(external: np.ndarray, *, reward=0.0, controls=None, learn=True,
  uncertainty=0.0, route_gain=None) -> np.ndarray` (output vector length n).
- `frozen_step(...)` identical math with learning disabled (used by standalone runner parity).
- `export_state()/import_state()`, `clone()`.

### neuromod.py
- `NeuromodulatorSystem(n, *, seed, signs, src, dst)`: six levels in `[0,1]`,
  per-unit receptor matrix `(n, 6)` in `[0.5, 1.5]` (all 1.0 unless
  `receptor_heterogeneity`), local traces, `AffectState` with the 7 variables
  (valence/rpe in `[-1,1]`, others `[0,1]`).
- `update(reward, prediction_error, novelty, threat, energy, flags) -> None`.
- `controls(flags) -> dict` returning neutral values when flags are off:
  `learning_rate_scale, plasticity_scale, attention, exploration,
  consolidation_scale, prediction_precision, sensory_weight (16,),
  decision_temperature, persistence, threshold_shift, reward_scale, gain`.
  Each must be causally used by engine/agent code (no decorative values).
- `infer_transmitters(src, dst, w, sign) -> dict` six-class hypothesis + confidence
  (never changes `sign`).

### sensory.py
- `SensoryHub(spec)`; channel modes from `spec.sensory_mode()`.
- `visual_features(frame: np.ndarray HxWx3 uint8/float, previous=None) -> dict` with
  luminance, red_green, blue_yellow, edge_density, motion, change, centroid_x,
  centroid_y, spatial_spread, object_regions (count of connected bright regions,
  bounded), temporal_pattern. Frames are never stored (keep only a down-sampled
  luminance map for motion, never exported).
- `audio_features(samples: np.ndarray, sample_rate) -> dict` amplitude, low, mid,
  high band energies, change, onset/pattern score; buffers never stored.
- `simulate(ecology_obs: dict | None, step) -> np.ndarray[16]`.
- `fuse(raw: dict[channel->float|None]) -> SensoryFrame` with value (16,),
  reliability (16,), missing mask, predicted (16,), prediction_error (16,),
  uncertainty, cross-modal imputations; reliability-weighted; missing sensors
  imputed from cross-modal memory and flagged as `provenance='predicted'`.
- `manifest() -> dict` incl. `raw_media_stored: False`.

### micro_collective.py
- `MicroCollective(input_dim=16, output_dim=16, experts=4, top_k=2, hidden=24, seed)`:
  top-k router with load-balancing bias, per-expert tanh MLP, SGD self-supervised
  next-sensor prediction, bounded replay (novelty-prioritised), held-out
  validation split, best-state snapshot, rollback when validation worsens,
  SHA-256 training receipts, `expert_load` stats, `add_expert()` growth,
  `predict(x) -> (y, routing)`, `train_step(x, y_next) -> dict`.
- `SpecialistGraft(input_dim, context_dim, seed)`: separate small recurrent
  predictor trained only through its own interface; its contribution to the
  main system is multiplied by `coupling` which starts at `0.0`; `detach()`
  removes it with no residue. Receipts record its independent validation loss.

### memory.py
- `Provenance` enum: `SENSORY, PREDICTED, EXTERNAL, EXPERIENCE, SYNTHETIC`.
- `MemoryEcology(key_dim, value_dim, capacity, seed)`: multi-timescale traces,
  sparse (FlyHash) index, similarity + contextual retrieval, reward association,
  confidence, interference tracking, replay scheduler, forgetting/downscaling.
  `store(key, value, *, provenance, reward=0, context=None, confidence=1)`;
  `retrieve(key, *, context=None, k=4, verified_only=False)`; PREDICTED items are
  never returned as verified and can only be promoted to EXPERIENCE by
  `confirm(id, observed_value, tol)`. Only verified provenance (SENSORY, EXTERNAL,
  EXPERIENCE) counts as an observation: storing PREDICTED/SYNTHETIC items never
  moves the novelty traces or lowers verified items' confidence via interference.

### prediction.py
- `PredictiveHierarchy(dim, horizons=(1,4,16,64), ensemble=3, seed)`: ensemble
  linear/RLS predictors per horizon, disagreement, surprise, adaptive horizon
  selection, fast/medium/slow traces, `forecast(state)`, `update(state, target)`,
  `counterfactual(state, action_alternatives) -> list` (pure, non-mutating).
- `EfferenceSelfModel`: `record(intended, issued, predicted_outcome,
  observed_outcome) -> dict` with authorship in [0,1], disturbance, actuator
  slip, prediction error, control reliability (EMA).

### packets.py & events.py
- `RepresentationPacket` (frozen, schema `NeuroWeavePacketV1`, version 1): kind in
  `scalar, vector, spectral, phase, temporal, spatial, graph, sparse, vsa, event,
  population, context, predictive, uncertainty, compartment, hybrid`; payload
  tuple of finite floats, `timestamp`, `source` (region/soma/compartment ids),
  `confidence`, `uncertainty`, `provenance`. SHA-256 checksum. Adapter
  `to_neurowave_x2(packet)` / `from_neurowave_x2(np)` using
  `synapse.neurowave_x2.packets.NeuralPacket`.
- `RepresentationFusion(dim, kinds)`: configurable or learned fusion weights;
  with no packets (or fusion disabled) returns the scalar input unchanged.
- `EventBus`: typed topics `input, state, prediction, modulation, learning,
  action, memory, telemetry`; bounded history; subscribers; versioned envelope.

### ecology.py
- `NeuroWeaveEcology(config='default', seed, agents=1)`: 1200×660 arena, moving
  obstacles, food patches with replenishment, thermal zones, energy, reward,
  collisions, disturbances. Configs: `default, maze, scarce, shifting_rules,
  hidden_food (memory-dependent), long_horizon, multi_agent`.
- `reset()`, `step(actions) -> (obs list, rewards, done, info)`; `sensory_vector(i) -> np[16]`.
- `to_synapse_observation(obs) -> np.ndarray[155]` adapter for the existing
  Sonapse agent; `run_synapse_agent(agent, steps)` drives `agent.act/feedback`.

### hardware.py & package.py
- `quantise_int8`, `to_q88/from_q88`, `tile_schedule`, `execution_profile`,
  `fpga_mapping_manifest(brain)` with endpoint + sign validation and
  `rtl_generated: False, hardware_validated: False`.
- `package.py`: `SONAPSE_NEUROWEAVE_PACKAGE` schema `1.0.0` builder, validator
  (size limit 25 MB, neuron/synapse limits, finite checks, SHA-256 fingerprint),
  `inspect_package(text) -> report` read-only (never touches live state), legacy
  `NEUROWEAVE_MODEL_PACKAGE` 1.0.0/1.1.0/1.2.0 read-only import + FNV-1a check,
  capability matrix distinguishing frozen inference, continued learning, full
  simulation, live sensory operation, exported memory, persistent state.
