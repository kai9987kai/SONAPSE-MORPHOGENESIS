# `morphogenesis`: developmental growth and regeneration of a connectome

Module file: `synapse/frontier/morphogenesis.py` · tests: `tests/test_frontier_addons.py` ·
experiment id: `morphogenesis` · kind: discovery (no engine extension, no live observer) ·
evidence tier: `synthetic_abstraction`.

The module grows a 2-D multi-cell body from a single seed cell (**organizer**) under a compact five-parameter
developmental program, converts the grown tissue into a valid `synapse.neuroweave.topology.Topology` connectome,
and studies **regeneration after lesions** with and without a stored **pattern memory** (plus a capacity-matched
shuffled-memory control). It is inspired by the user's `morpheus`, `prometheus-alpha`, `GenesisEngine` and `MOLT`
repositories (`kai9987kai`), by Growing Neural Cellular Automata (Mordvintsev, Randazzo, Niklasson & Levin, 2020),
and by the bioelectric pattern-memory perspective on regeneration (Pezzulo & Levin, 2015).

This is a synthetic five-parameter growth-logit abstraction on a small grid, not a trained neural cellular
automaton or a biophysical tissue model.

## What it computes

**Developmental program (`DevelopmentalProgram`) and growth (`grow`).** A body lives on an $S \times S$ grid
($8 \le S \le 48$) starting from a single living cell at the organizer $(o_y, o_x) = (\lfloor S/2 \rfloor, \lfloor S/2 \rfloor)$.
At each developmental step ($2S$ steps by default):

1. For every grid site $(y, x)$, define scaled coordinates relative to the organizer:
   $$
   \Delta x = \frac{x - o_x}{S / 4}, \qquad \Delta y = \frac{y - o_y}{S / 4},
   $$
   and let $n(y, x) \in \{0, \dots, 8\}$ be the number of living 8-neighbours.
2. Evaluate the five-parameter growth logit $\theta = (\theta_0, \theta_1, \theta_2, \theta_3, \theta_4)$:
   $$
   \ell(y, x) = \theta_0 + \theta_1 \frac{n(y, x)}{8} + \theta_2 (\Delta x)^2 + \theta_3 (\Delta y)^2 + \theta_4 (\Delta x)^4
   + \mathbf{1}_{\text{memory}}\, \alpha \left(2 M(y, x) - 1\right),
   $$
   where $M(y, x) \in [0, 1]$ is the upsampled pattern-memory setpoint field (when present) and $\alpha \ge 0$ is
   `memory.strength` (default `3.0`).
3. Draw an asynchronous stochastic update mask $U(y, x) \sim \mathrm{Bernoulli}(p_{\text{update}})$ (`p_update = 0.5`),
   analogous to the per-cell stochastic update mask of Mordvintsev et al. (2020):
   * an empty site with $n(y, x) > 0$, $\ell(y, x) > 0$, and $U(y, x) = 1$ becomes alive;
   * a living non-organizer site with $\ell(y, x) < -\delta_{\text{death}}$ (`death_margin = 2.0`) and $U(y, x) = 1$
     dies (the organizer persists while alive).

**Positional differentiation (`_assign_types`).** Living cells differentiate into four functional types by position
relative to the organizer (French-flag style) and map to NEUROWEAVE region indices:

* `sensory` (`visual_sensory`, region `0`): $\Delta x < -1$;
* `action` (`premotor`, region `8`): $\Delta x > +1$;
* `memory` (`working_memory`, region `5`): $(\Delta x)^2 + (\Delta y)^2 \le 0.36$ (and $|\Delta x| \le 1$);
* `association` (`association_a`, region `3`): all remaining living sites.

**Connectome conversion (`body_to_topology`).** Living cells become neurons in row-major order ($N$ neurons):

* **Dale sign:** $\mathrm{sign}(y, x) = -1$ when $(7y + 13x) \bmod 5 = 0$ (~20 % inhibitory) and $+1$ otherwise,
  pinned to grid coordinates so a regenerated site keeps its sign.
* **Directed edges** connect distinct cells within Euclidean distance $r$ (`radius = 1.5` by default) when:
  * they have the **same** cell type (local recurrent),
  * the source has strictly **lower** processing rank (`KIND_RANK`: `sensory < association < memory < action`) than
    the target (feedforward), or
  * the source has strictly **higher** rank and is a direct 4-neighbour ($d \le 1$, local feedback).
* **Delays and weights:** integer delay $1 + \mathbf{1}_{d > 1.5} + \mathbf{1}_{d > 2.5} \in \{1, 2, 3\}$; signed
  weight $\mathrm{sign}_{\text{src}} \cdot \left(w_{\min} + w_{\text{span}} \cdot \mathrm{clip}(1 - (d - 1)/r, 0, 1)\right)$.
  Every generated `Topology` passes `topo.validate()` (no self-loops, no duplicate edges, delays in `[1, 3]`,
  weights matching source signs) and can be passed directly to `topology_metrics(topo)`.

**Evolution (`evolve_program`).** A seeded $(1 + \lambda)$ evolution strategy (`generations=30`, `offspring=8`,
`sigma=0.4`) mutates $\theta$ with isotropic Gaussian perturbations clipped to $[-50, 50]$ to maximise intersection-over-union
(`iou`) with a target binary mask (`two_lobes`, `ellipse`, or `cross`). Ties keep the parent so fitness is
monotonically non-decreasing.

**Lesion and regeneration (`lesion`, `make_memory`, `shuffled_memory`, `regrow`).**

* `lesion(body, box)` clears all cells in `[r0:r1, c0:c1)`.
* `make_memory(body, block=2, strength=3.0)` stores a coarse setpoint map of shape $(S/b) \times (S/b)$ holding the
  fractional occupancy of each $b \times b$ block (`0.5` is neutral inside $2M - 1$), plus the organizer coordinates
  $(o_y, o_x)$ (`(S // b)**2 + 2` numbers in total).
* **Regrowth arms (`regrow`):**
  1. **`pattern_memory` (candidate):** restores the organizer coordinates from the stored pattern memory and biases
     the growth logit by $\alpha (2M - 1)$.
  2. **`no_memory` (baseline):** runs the developmental program alone; if the lesion removed the organizer, the new
     organizer is re-derived as the surviving cell nearest the centroid of surviving tissue.
  3. **`shuffled_memory` (capacity-matched control):** stores the exact same number of numbers (`PatternMemory.numbers`)
     and total occupancy mass, with the block occupancies and organizer coordinates randomly permuted by a seeded
     generator (`default_rng(seed + 141421)`).

## Benchmark (`run_experiment`, `EXPERIMENTS = {'morphogenesis': run_experiment}`)

For each seed (`seeds=(0, 1, 2)` by default):

1. Evolve a `DevelopmentalProgram` towards `target_mask(target, size)` (`two_lobes` by default) and grow the intact
   body and its connectome.
2. Apply `lesions_per_seed` random square lesions covering `lesion_fraction = 0.4` of the body's bounding box
   (centred on randomly chosen living cells; some lesions remove the organizer, others spare it).
3. Regrow from each damaged state under `pattern_memory`, `no_memory`, and `shuffled_memory`, measuring:
   * **`iou`**: mask IoU with the pre-lesion body (headline metric, higher is better);
   * **`edge_jaccard`**: Jaccard similarity of position-keyed directed connectome edge sets;
   * **`type_agreement`**: fraction of overlapping living cells that recover their pre-lesion cell type;
   * **`metric_similarity`**: $1 - \text{mean relative gap}$ across connectome `density`, `clustering_mean`,
     `modularity`, and `global_efficiency`.
4. Headline verdict uses `harness.headline_verdict(vs_no_memory, vs_shuffled_memory)`: `improved` requires a paired
   95 % t-interval win over **both** `no_memory` and `shuffled_memory`.

| option | default | quick |
|---|---|---|
| grid size $S$ / target | 20 / `two_lobes` | 14 / `two_lobes` |
| ES generations / offspring $\lambda$ / $\sigma$ | 30 / 8 / 0.4 | 10 / 6 / 0.4 |
| lesions per seed / lesion fraction | 4 / 0.4 | 3 / 0.4 |
| memory block $b$ / strength $\alpha$ | 2 / 3.0 | 2 / 3.0 |
| pattern-memory numbers $(S/b)^2 + 2$ | 102 | 51 |

## Public API

```python
MODULE_ID = 'morphogenesis'
CELL_TYPES = ('sensory', 'association', 'memory', 'action')
TARGETS = ('two_lobes', 'ellipse', 'cross')
EXPERIMENTS = {'morphogenesis': run_experiment}

DevelopmentalProgram(theta=(2.0, 1.0, -1.2, -1.2, 0.0), p_update=0.5, death_margin=2.0)
PatternMemory(setpoint, block, organizer, strength=3.0)
    .field(size) -> np.ndarray (size, size)
    .numbers -> int
Body(alive, cell_type, organizer)
    .size -> int  .cells -> int  .copy() -> Body
target_mask(name, size) -> np.ndarray
grow(program, size, *, seed=0, steps=None) -> Body
lesion(body, box) -> Body
make_memory(body, *, block=2, strength=3.0) -> PatternMemory
shuffled_memory(memory, *, seed=0) -> PatternMemory
regrow(program, damaged, *, memory=None, seed=0, steps=None) -> Body
body_to_topology(body, *, radius=1.5, name='morphogenesis', seed=0) -> Topology
iou(a, b) -> float
edge_jaccard(body_a, body_b, *, radius=1.5) -> float
evolve_program(target, *, seed=0, generations=30, offspring=8, sigma=0.4,
               growth_seed=0, steps=None) -> dict
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict
```

## Connection to Sonapse

`body_to_topology` returns a validated `synapse.neuroweave.topology.Topology` with ten-region indexing and
Dale-signed weights, compatible with `topology_metrics` and with `NeuroWeaveEngine`. The module is default-off and
does not alter any existing `build_topology` generator.

## Not implemented

* **Differentiable multi-channel neural cellular automata** trained by backpropagation through time on RGBA images
  (Mordvintsev et al., 2020). This module evolves a 5-parameter radial/neighbour growth logit with a $(1+\lambda)$-ES.
* **Explicit voltage-gated ion channels or gap-junction diffusion equations** for bioelectric pattern memory
  (Pezzulo & Levin, 2015); the pattern memory is represented directly as a coarse spatial setpoint field plus the
  organizer coordinates.
* **Activity-dependent synaptic plasticity during development.** Connectome weights come from geometry and cell-type
  ranks after growth finishes.

## Verified references

* Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson and Michael Levin. **Growing Neural Cellular Automata.**
  *Distill* (published 11 February 2020). DOI [10.23915/distill.00023](https://doi.org/10.23915/distill.00023);
  URL <https://distill.pub/2020/growing-ca/>.
  * **Checked on 2026-11-19:** single-seed growth on a 2-D grid, local neighbourhood perception, asynchronous
    per-cell stochastic update mask, and regeneration experiments after rectangular lesions.
* Giovanni Pezzulo and Michael Levin. **Re-membering the body: applications of computational neuroscience to the
  top-down control of regeneration of limbs and other complex organs.** *Integrative Biology* 7(12), 1487–1517
  (December 2015). DOI [10.1039/c5ib00221d](https://doi.org/10.1039/c5ib00221d); PMC4700597 (PMID 26571072).
  * **Checked on 2026-11-19:** conceptualises regeneration as error-minimising remodelling guided by an encoded
    target-morphology "pattern memory" distinct from local growth rules alone.
* Portfolio provenance: `kai9987kai/morpheus`, `kai9987kai/prometheus-alpha`, `kai9987kai/GenesisEngine`, and
  `kai9987kai/MOLT`, inspected in the local read-only portfolio clones on 2026-11-19.

## Measured results

Measured on **2026-11-19** with `seeds=(0, 1, 2)` on one CPU thread (`OPENBLAS_NUM_THREADS=1`).

| mode | digest (prefix) | wall-clock | target IoU (initial $\to$ evolved) | `pattern_memory` IoU (edge Jaccard / type agree / metric sim) | `no_memory` IoU (edge Jaccard / type agree / metric sim) | `shuffled_memory` IoU (edge Jaccard / type agree / metric sim) | `pattern - no_memory` IoU (95 % CI) | `pattern - shuffled` IoU (95 % CI) | headline verdict (`verdict_basis`) |
|---|---|---|---|---|---|---|---|---|---|
| `quick=True` ($14\times 14$, 10 gen, 3 lesions/seed, 51 memory numbers) | `67f7c190ce564e66` | 3.39 s | $0.6351 \to 0.7437$ | 0.9593 (0.9580 / 1.0000 / 0.9770) | 0.9020 (0.8641 / 0.8526 / 0.9878) | 0.3294 (0.2393 / 0.4084 / 0.8252) | +0.0573 (`[+0.0347, +0.0799]`) | +0.6299 (`[+0.4159, +0.8439]`) | `improved` (`candidate_vs_baseline_and_control`) |
| `quick=False` ($20\times 20$, 30 gen, 4 lesions/seed, 102 memory numbers) | `18af03324af0663f` | 13.37 s | $0.6062 \to 0.8229$ | 0.9739 (0.9749 / 1.0000 / 0.9835) | 0.9358 (0.9131 / 0.9120 / 0.9873) | 0.1978 (0.1286 / 0.2724 / 0.5317) | +0.0381 (`[-0.2242, +0.3004]`) | +0.7761 (`[+0.3605, +1.1917]`) | `no_detectable_difference` (`candidate_vs_baseline`) |

In both modes `pattern_memory` restores 100 % cell-type agreement on overlapping tissue and beats the
capacity-matched `shuffled_memory` control with a wide margin (`improved`). At `quick=False` ($20 \times 20$),
several random lesions spare the organizer cell, allowing the developmental rule alone (`no_memory`) to regrow
most of the symmetrical shape (mean IoU 0.9358 vs 0.9739 with pattern memory), so across 3 seeds the paired 95 % t
interval against `no_memory` spans zero (`[-0.2242, +0.3004]`, `no_detectable_difference`). When a lesion removes
the organizer, `no_memory` shifts the organizer to the centroid of surviving tissue and systematically misplaces
cell-type poles (`test_regrowth_with_pattern_memory_recovers_better_after_losing_the_organizer`).
