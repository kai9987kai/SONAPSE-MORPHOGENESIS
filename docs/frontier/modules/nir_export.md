# `nir_export`: NIR-style interchange export of NEUROWEAVE brains

Module file: `synapse/frontier/nir_export.py` · tests: `tests/test_frontier_biophysics.py` ·
experiment id: `nir_export` (`kind: 'validation'`) · no observer, no engine extension ·
evidence tier: `synthetic_abstraction`.

The module exports a NEUROWEAVE brain as a graph in the style of the **Neuromorphic Intermediate
Representation** (NIR, Pedersen et al., *Nature Communications* 2024). It uses NIR's node types and the
parameter names of the `nir` Python package. It is *inspired by* NIR and follows its primitives. It does not
claim equivalence with any NIR backend or hardware, and it has not been run on one.

The export is a **lossy approximation** of the abstract NEUROWEAVE cell. Every feature that cannot be
represented is listed in the graph metadata and in the receipt.

## Graph format

```
{'nodes':    {name: {'type': <NIR primitive>, 'params': {field: value}}},   # values are nested float lists
 'edges':    [[src, dst], ...],                                             # incoming edges are summed
 'metadata': {...}}                                                          # optional; ignored by equality
```

Handled node types and fields (same names as the `nir` dataclasses): `Input(shape)`, `Output(shape)`,
`Affine(weight, bias)`, `Linear(weight)`, `Scale(scale)`, `Delay(delay)`, `I(r)`,
`IF(r, v_threshold, v_reset)`, `LI(tau, r, v_leak)`, `LIF(tau, r, v_leak, v_threshold, v_reset)`,
`CubaLI(tau_syn, tau_mem, r, v_leak, w_in)` and
`CubaLIF(tau_syn, tau_mem, r, v_leak, v_threshold, v_reset, w_in)`. Weights are `(out, in)` matrices.

## What it computes

**Layout of `brain_to_nir(source, *, input_dim=None, step_ms=1.0)`.** `source` can be a `NeuroWeaveBrain`, a
`NeuroWeaveEngine`, or either one's `export_state()` dict. The exporter works on a validated private copy and
never mutates the source (tested).

```
input (Input, d) -> input_projection (Linear, n x d) -> cells (CubaLIF or LIF, n) -> output (Output, n)
for each synaptic delay class k (steps):
    cells -> delay_k (Delay, k * step_ms) -> recurrent_k (Linear, n x n) -> cells
```

* `input_projection` is the exact linear part of `brain.project_input`. It is computed column by column on a
  copy with FlyHash switched off, because FlyHash is nonlinear.
* `recurrent_k[dst, src]` holds `0.52 * w` summed over the synapses with delay `k`, where `0.52` is the
  recurrent gain of the fast channel. Static gains are folded into `w` when their mechanisms are on: int8
  quantisation, region-pair route gains, myelin gain `0.85 + 0.25 * myelin`, and structural source weights.
  The recurrent weights are a frozen snapshot of the synapse table.
* `delay_k` holds the delay in ms. Without `synaptic_delays` there is a single class `k = 1`, the one-step
  recurrence of the engine. That class also breaks the recurrent loop in the simulator.

**Cell mapping (`cell_parameters`).** The constants come from `CellPopulation.step` in
`synapse/neuroweave/cells.py`. A per-step decay maps to a time constant by the zero-order-hold relation
`tau = -dt / ln(decay)`:

| NEUROWEAVE (per step) | NIR parameter |
|---|---|
| `fast = 0.58 fast + u + 0.52 I_rec` | `CubaLIF` current: `tau_syn = -dt / ln 0.58`, `w_in = 1 / (1 - 0.58)` |
| `membrane = 0.78 membrane + 0.22 (0.84 fast) - 0.06 adaptation` | `tau_mem = -dt / ln 0.78`, `r = 0.84`, `v_leak = 0` |
| spike when `membrane > threshold`, `threshold = 0.53` at rest | `v_threshold = 0.53` |
| `membrane <- 0.26 membrane` at a spike | `v_reset = 0.26 * 0.53` (closest constant) |

Cells without `membrane_dynamics` become `LIF` on the fast channel: `tau = -dt / ln 0.58`,
`r = 1 / (1 - 0.58)`, and `v_threshold = atanh(0.53) / 0.84`, the fast level at which `tanh(0.84 fast)`
crosses 0.53. The source never resets the fast channel, so `v_reset = v_threshold`.

**Lossy features (`unrepresentable_features`).** Each entry gives `feature`, `kind` and `treatment`
(`approximated`, `dropped` or `frozen_snapshot`) with a reason. The entries are:

* Always: graded `tanh` output transmission (NIR edges from a spiking node carry 0/1 spikes), the direct 0.16
  recurrent feed-through, the readout head, input clipping, and runtime controls.
* With `membrane_dynamics`: the multiplicative reset, the refractory hold, the adaptive threshold and
  adaptation current, and calcium feedback.
* Without `membrane_dynamics`: the threshold on the graded output and the absence of reset in the source.
* Every enabled neuron mechanism (multi-soma, dendrites, sparse code, ...) and state-dependent connectome gain.
* Plasticity rules, exported as a `frozen_snapshot` of current weights.
* Engine-level mechanisms outside the brain graph.

**Validation (`validate_graph`).** The checks are:

* known node types with exactly the expected parameter fields, and finite parameters;
* consistent 1-D parameter vectors and 2-D weights, positive time constants and non-negative delays;
* edge endpoints that exist, and no duplicate edges;
* no edges into an `Input` or out of an `Output`;
* matching shapes along every edge, and at least one `Input` and one `Output`.

The function returns a report (`valid`, `errors`, `warnings`, counts and node types). It never raises on a bad
graph.

**Serialisation (`write_nir`, `read_nir`).** With `backend='auto'`, the module writes a real NIR file with
`nir.write` and reads it back with `nir.read`, but only if the optional `nir` package imports. Otherwise it
writes JSON tagged `'format': 'nir-json-fallback'`. `backend='json'` forces the fallback, and `'nir'` requires
the package. `read_nir` detects the format from the HDF5 signature. Either path reports whether the read-back
graph equals the written one: same node names and types, the same parameter values compared exactly, and the
same edge set.

**Simulator (`simulate_graph`).** A small NumPy discrete-time simulator for the node types above. It uses the
zero-order-hold forms of the NIR equations:

* `LI`/`LIF`: `v <- a v + (1 - a)(v_leak + r x)` with `a = exp(-dt / tau)`.
* `Cuba*`: `i <- a_s i + (1 - a_s) w_in x`, then `v` as above, driven by `r i`.
* Spiking types emit `v > v_threshold` and reset to `v_reset`. The simulator checks the membrane potential
  before reset so numerical overflow appears in the `finite` result even when the output spike is finite.
* `Delay` outputs its input from `round(delay / dt)` steps earlier. Delays of at least one step break
  recurrent loops; a loop without one raises `ValueError`.

**Experiment (`run_experiment`).** Each seed builds a brain with `make_spec`, by default 48 neurons on the
`clustered` topology with `membrane_dynamics`, `synaptic_delays`, `quantised_synapses` and `stdp`. It runs
warm-up steps on seeded inputs, then exports, validates, writes and reads back a file, and simulates the
exported graph. The checks, all of which must hold for `verdict: 'pass'`:

* `graph_valid`, `node_types_known`.
* `round_trip_equal`, `edge_and_node_counts_preserved`.
* `synapse_count_matches`: the non-zero recurrent entries equal the represented count, which is at most the
  number of unique `(src, dst, delay)` triples.
* `recurrent_drive_parity`: the exported Delay/Linear pairs, applied to NEUROWEAVE's own output history,
  reproduce `SynapseTable.transmit` within `1e-9`. This check does not apply, and is recorded as such, when
  state-dependent gains are on.
* `synaptic_current_parity`: isolated cells, with no recurrent input and the same random drive. The exported
  current equals the fast channel within `1e-9`.
* `membrane_parity_before_first_spike`: the exported membrane equals NEUROWEAVE's membrane minus its
  accumulated adaptation current (a dropped feature, corrected for explicitly) within `1e-9`, up to each
  cell's first spike.
* `simulation_finite`, `simulation_active` (at least one spike).
* `lossy_features_listed`.

The spike rates of the exported graph and of NEUROWEAVE on the same inputs, and the first-spike agreement of
isolated cells, are reported side by side as descriptive numbers. They are not expected to agree, because of
the listed approximations.

## Public API

```python
MODULE_ID = 'nir_export'; EXPERIMENTS = {'nir_export': run_experiment}
FALLBACK_FORMAT = 'nir-json-fallback'; NIR_FORMAT = 'nir-hdf5'
NODE_PARAMS; NIR_NODE_TYPES; NEURON_TYPES; SPIKING_TYPES; CELL_CONSTANTS
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict        # sealed validation receipt
brain_to_nir(source, *, input_dim=None, step_ms=1.0) -> dict
cell_parameters(model, *, dt=1.0) -> dict                                    # model in ('CubaLIF', 'LIF')
unrepresentable_features(source) -> list[dict]
validate_graph(graph) -> dict; node_io_shapes(name, node, errors=None)
canonical_graph(graph, *, include_metadata=True) -> dict; graphs_equal(a, b) -> bool
write_nir(graph, path, *, backend='auto') -> dict; read_nir(path) -> dict; nir_status() -> dict
simulate_graph(graph, inputs, *, dt=None, record_states=False, max_steps=100_000) -> dict
```

The `run_experiment` overrides are `neurons`, `topology`, `mechanisms`, `warmup_steps`, `simulate_steps`,
`input_dim`, `input_scale`, `step_ms`, `parity_cells`, `parity_steps`, `parity_drive_max`,
`parity_tolerance` and `backend`. `quick=True` sets 24 neurons, 30 warm-up steps, 80 simulated steps, and 8
parity cells for 80 steps.

All randomness comes from `numpy.random.default_rng(seed + offset)`. No global NumPy or torch RNG or thread
state is touched (tested). Bad input raises `ValueError` before anything is written. Graph and file sizes are
bounded (4096 nodes, 65 536 edges, 4 M parameter elements per field, 100 000 simulated steps).

## Connection to Sonapse

* An offline tool. It reads a brain, an engine or their exports and never writes back. Nothing in the live
  runtime calls it, and no Sonapse or NEUROWEAVE default, API or checkpoint format changes.
* `nir` is an optional dependency. It is imported lazily, and a broken install falls back to JSON.

## Not implemented

* **The real-NIR path is untested here.** The `nir` package is not installed in this build, so `nir.write`,
  `nir.read`, the conversion to `nir.NIRGraph` and NIR's own type checking of this recurrent graph have not
  been exercised. The test `test_real_nir_round_trip` is skipped. The conversion code follows the field names
  verified in the package source (see below).
* NIR's other primitives: `Conv1d`, `Conv2d`, `Flatten`, `SumPool2d`, `AvgPool2d` and `Threshold`.
  Multi-dimensional shapes are also not handled.
* Exact representation of the NEUROWEAVE cell. Graded transmission, multiplicative reset, refractoriness, the
  adaptive threshold and every listed mechanism are approximated or dropped. Plasticity is not representable
  in NIR and is exported as a frozen snapshot.
* Continuous-time semantics. The simulator is a zero-order-hold discretisation with the step `dt`, and NIR
  backends may integrate differently.
* Importing a NIR graph back into a NEUROWEAVE brain.

## Verified references

Checked on 2026-11-20.

1. Jens E. Pedersen, Steven Abreu, Matthias Jobst, Gregor Lenz, Vittorio Fra, Felix Christian Bauer, Dylan
   Richard Muir, Peng Zhou, Bernhard Vogginger, Kade Heckel, Gianvito Urgese, Sadasivan Shankar, Terrence C.
   Stewart, Sadique Sheik, Jason K. Eshraghian. **Neuromorphic intermediate representation: A unified
   instruction set for interoperable brain-inspired computing.** *Nature Communications* 15, 8122, published
   16 September 2024 (PubMed Central). doi:10.1038/s41467-024-52259-9,
   https://www.nature.com/articles/s41467-024-52259-9, https://pmc.ncbi.nlm.nih.gov/articles/PMC11405706/.
   Preprint: arXiv:2311.14641. Title, author order, volume, article number and date were checked on the PMC
   page. The base primitives (Input, Output, Affine, Convolution, Delay, Flatten, Integrator, Leaky
   integrator, Linear, Scale, Spike) and the higher-order IF, LIF and CuBa-LIF were checked against the
   paper's primitives table on the same page. (OSTI lists 15 September 2024.)
2. NIR Python package, https://github.com/neuromorphs/NIR, main branch, checked on 2026-11-20:
   * `nir/ir/neuron.py`: field names and order of `I`, `IF`, `LI`, `LIF`, `CubaLI`, `CubaLIF`, and the
     documented dynamics `tau dv/dt = (v_leak - v) + R I` and `tau_syn dI/dt = -I + w_in S`.
   * `nir/ir/linear.py`: `Affine(weight, bias)`, `Linear(weight)`, `Scale(scale)`, with the output size taken
     from `weight.shape[-2]`.
   * `nir/ir/delay.py`: `Delay(delay)` with `y(t) = x(t - tau)`.
   * `nir/ir/graph.py`: `Input(input_type)`, `Output(output_type)`, and `NIRGraph(nodes, edges)` with type
     checking on by default.
   * `nir/serialization.py`: `write(filename, graph, compression='gzip', ...)` and
     `read(filename, type_check=True)`, using HDF5 through `h5py`.
   * PyPI https://pypi.org/project/nir/ lists release 1.0.8 (6 July 2026) as the latest.

## Measured results

Measured on 2026-11-20 on Windows 11 on ARM64, with Python 3.12.10, NumPy 2.4 and one BLAS/OpenMP thread
(`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`), without the `nir` package, so the JSON fallback
was used. Seeds were 0, 1 and 2. Receipts are deterministic: two runs of the quick receipt gave the same digest
(tested). Wall-clock covers the whole call on a machine shared with other jobs.

| Run | Verdict | Key numbers | Wall-clock |
|---|---|---|---|
| `run_experiment(quick=True)` | `pass` (11/11 checks) | 24 neurons; 140 / 142 / 137 synapses, all represented; delay classes 1, 2, 3; graph of 10 nodes (Input, Linear ×4, CubaLIF, Delay ×3, Output) and 12 edges. Recurrent drive parity max abs diff 4.4e-16. Synaptic current diff 0. Membrane diff before first spike ≤ 1.1e-16. 10 lossy features listed. Exported-graph spike rate 0.0042 / 0.0031 / 0.0047 per cell-step vs NEUROWEAVE 0.240 / 0.0031 / 0.0 on the same inputs. Isolated-cell first-spike agreement 0.875 / 0.625 / 0.875 | 1.1–1.4 s |
| `run_experiment()` | `pass` (11/11 checks) | 48 neurons; 281 / 276 / 286 synapses, all represented. Recurrent parity ≤ 4.4e-16. Membrane diff ≤ 2.2e-16. Exported-graph spike rate 0.0044 / 0.0053 / 0.0056 vs NEUROWEAVE 0.051 / 0.0056 / 0.012. First-spike agreement 0.75 / 0.81 / 0.56 | 2.5 s |

Cell parameters at `dt = 1 ms`: `tau_syn = 1.836`, `tau_mem = 4.025`, `r = 0.84`, `w_in = 2.381`,
`v_threshold = 0.53`, `v_reset = 0.1378`.

The parity checks show that the parts called exact (linear input map, recurrent synapse table, synaptic
current, and the membrane up to the dropped adaptation current) are exact to rounding. The network-level spike
rates differ widely between the exported graph and NEUROWEAVE. This is expected: NIR edges carry binary spikes
where NEUROWEAVE transmits a graded `tanh` output, and refractoriness and the adaptive threshold are dropped.
The export is an interchange format with documented losses, not a faithful re-implementation.
