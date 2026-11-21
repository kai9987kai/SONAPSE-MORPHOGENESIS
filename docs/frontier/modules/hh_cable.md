# `hh_cable`: Hodgkin-Huxley cable reference

Module file: `synapse/frontier/hh_cable.py` · tests: `tests/test_frontier_biophysics.py` ·
experiment id: `hh_cable` (`kind: 'validation'`) · no observer, no engine extension ·
evidence tier: `synthetic_abstraction`.

A multi-compartment Hodgkin-Huxley (HH) cable, used as a **biophysical reference** next to NEUROWEAVE's
abstract discrete-time cell. It is *inspired by* NeuroPulse-CableLab (kai9987kai), a single-file browser HH
cable workbench with Euler, RK4 and Rush-Larsen integrators and a dt versus dt/2 comparison. It is built on
the membrane equations of Hodgkin & Huxley (1952).

This is a numerical integration of the 1952 equations on a discretised cable. It does not reproduce the paper's
own procedure: HH solved a travelling-wave ODE (their eqn 30) by shooting for the velocity, using Hartree's
method. It is not a validated neuron simulator. The NEUROWEAVE comparison is descriptive only, and the
abstract cell is never "validated" against HH.

## What it computes

**Membrane and cable.** `v` is in mV with rest near −65 mV. HH's own variable is the displacement
`V = −(v + 65)`, with depolarisation negative.

```
Cm dv_i/dt = I_stim,i + g_ax (v_{i-1} - 2 v_i + v_{i+1}) - gNa m^3 h (v - ENa) - gK n^4 (v - EK) - gL (v - EL)
dx/dt      = phi (alpha_x(v) (1 - x) - beta_x(v) x),      x in {m, h, n},    phi = 3^((T - 6.3) / 10)
g_ax       = a / (2 Ra dx^2)       (HH eqn 28, I = (a / 2 R2) d2V/dx2, discretised by central differences)
```

* The rate functions are HH eqns 12, 13, 20, 21, 23 and 24 (p. 519), rewritten for `v`. For example,
  `alpha_m = 0.1 (v + 40) / (1 - exp(-(v + 40)/10))` and `beta_h = 1 / (1 + exp(-(v + 35)/10))`. The removable
  singularities at `v = −40` (alpha_m) and `v = −55` (alpha_n) take their limits, 1.0 and 0.1.
* Constants come from Table 3 (p. 520): `Cm = 1 uF/cm^2`, `gNa = 120`, `gK = 36`, `gL = 0.3 mS/cm^2`. The
  reversal potentials `V_Na = −115`, `V_K = +12`, `V_l = −10.613 mV` become `ENa = 50`, `EK = −77` and
  `EL = −54.387 mV`.
* Rates are for 6.3 °C, scaled by `phi` with a Q10 of 3 (pp. 519, 523).
* Both ends are sealed: no axial current leaves the first or last compartment. The stimulus is a current
  density (uA/cm^2) on compartment 0.
* Each simulation starts at the resting state, the most negative stable zero of the steady-state ionic current.

**Integrators.**

* `'rk4'`: classical Runge-Kutta. The constructor rejects time steps beyond a conservative linear stability
  bound.
* `'exp_euler'`: the gates are integrated exactly for frozen voltage (Rush-Larsen style). The voltage takes an
  exponential step with conductances and neighbour voltages frozen. It is first order and stable for any step.

Spike times are upward crossings of 0 mV, re-armed below −30 mV, located inside the step by cubic Hermite
interpolation. Everything is batched over amplitudes, intervals or diameters, so each search is a handful of
vectorised simulations.

**Measurements (methods of `HHCable`).**

* `resting_potential()`: bisection to 1e-10 mV.
* `threshold_current(duration_ms, ...)`: batched bisection for the smallest step amplitude that gives a spike
  at a site, by default the last compartment.
* `fi_curve(amplitudes, ...)`: steady rate `(k − 1) / (t_last − t_first)` over the spikes after a transient.
  A lone onset spike counts as 0 Hz.
* `conduction_velocity(...)`: distance over the difference in first-spike arrival at 1/3 and 2/3 of the
  cable, optionally batched over diameters.
* `refractory_period(...)`: paired pulses. A conditioning pulse (2 × the single-pulse threshold) is followed by
  a test pulse (4 × threshold). The result is the smallest onset-to-onset interval at which the test pulse
  evokes a second spike. This is an operational estimate of the absolute refractory period for that test
  strength.

**NEUROWEAVE reference.** `neuroweave_fi_curve` drives isolated `CellPopulation` cells from
`synapse/neuroweave/cells.py` with constant external drive and counts spikes. The cells have the
`membrane_dynamics` flag, no recurrent input and neutral controls. `neuroweave_threshold` finds the smallest
drive that fires. `compare_fi_curves` then compares the two f-I curves:

* Both curves are taken on the same threshold-normalised levels and normalised to their maximum.
* It reports the Spearman rank correlation over the suprathreshold levels, the level where repetitive firing
  starts, and the mean absolute difference.
* `adaptation_index` gives the mean of `(ISI_{k+1} − ISI_k) / (ISI_{k+1} + ISI_k)` at one level.

The comparison is descriptive: an abstract unit with one step as its time unit is compared in shape only.

## Lab experiment (`run_experiment`, `EXPERIMENTS = {'hh_cable': run_experiment}`)

Every named check must hold for `verdict: 'pass'`:

| Check | Known answer |
|---|---|
| `resting_within_0.5mV_of_documented_-65mV` | HH chose `V_l` so the ionic current is zero at `V = 0` (Table 3 footnote) |
| `rest_stationary_without_input` | zero current: no spike, voltage stays at rest to 1e-6 mV |
| `subthreshold_steps_silent` / `suprathreshold_steps_fire` | all-or-none response at 0.5 × and ≥ 1.1 × the long-step threshold |
| `spike_peak_above_0mV` | overshoot in the patch and at mid-cable for every diameter |
| `velocity_positive`, `velocity_increases_with_diameter` | |
| `velocity_scaling_exponent_within_tolerance_of_0.5` | HH eqn 34, `theta = sqrt(K a / (2 R2 CM))`: velocity grows as sqrt(diameter) on a uniform cable (tolerance 0.03 on the log-log slope) |
| `reference_velocity_within_tolerance_of_hh1952_computed` | HH p. 528: 18.8 m/s computed for their axon (a = 238 um, R2 = 35.4 ohm cm, CM = 1 uF/cm^2, 18.5 °C; observed 21.2 m/s); tolerance 2 %. Present only when the 476 um diameter, 18.5 °C and 35.4 ohm cm are used |
| `spike_time_converges_under_dt_halving` | patch RK4: the change in first-spike time at dt/2 at least halves or stays below 1e-6 ms, and is below 1e-3 ms |
| `exp_euler_first_order_convergence` | observed order between 0.7 and 1.5 |
| `velocity_changes_below_1pct_at_dt_half` | cable velocities at dt and dt/2 |
| `refractory_second_pulse_fails_then_recovers` | the test pulse fails at one pulse width and succeeds at the longest interval |

Settings:

* Patch at 6.3 °C, with `dt` 0.0125 ms and RK4.
* Cable at 18.5 °C: 40 mm long, 200 compartments, `dt` 0.0006 ms, diameters 119, 238 and 476 um, stimulated
  with a 0.2 ms pulse of 3000 uA/cm^2. The velocity is measured between 1/3 and 2/3 of the length, because
  both ends bias the local speed over several millimetres.
* `quick=True` uses 60 compartments and `dt` 0.005 ms for the cable, shorter windows, single-round searches and
  fewer f-I levels.
* The HH model is deterministic. `seeds` only seed the NEUROWEAVE `CellPopulation` constructors, and the
  receipt reports per-seed NEUROWEAVE curves with `seed_invariant`.
* All settings are overridable (`_DEFAULTS` keys). Unknown overrides raise `ValueError`.

## Public API

```python
MODULE_ID = 'hh_cable'; EXPERIMENTS = {'hh_cable': run_experiment}; METHODS = ('rk4', 'exp_euler')
HH_REST_MV = -65.0; HH1952_REFERENCE_AXON = {'diameter_um': 476.0, 'Ra_ohm_cm': 35.4, ...}
run_experiment(*, quick=False, seeds=(0, 1, 2), **overrides) -> dict           # sealed validation receipt
HHCable(compartments=1, length_um=100.0, diameter_um=476.0, Ra=35.4, Cm=1.0, gNa=120.0, gK=36.0, gL=0.3,
        ENa=50.0, EK=-77.0, EL=-54.387, dt_ms=0.01, method='rk4', temperature_c=6.3)
    .run(pulses, *, t_stop_ms, record=None, trace_site=None, trace_every=1)  # pulses: (P, 3) or (B, P, 3)
    .resting_potential(); .threshold_current(duration_ms, ...); .fi_curve(amplitudes, ...)
    .conduction_velocity(*, amplitude=3000.0, duration_ms=0.2, t_stop_ms=2.0, sites=None, diameters_um=None)
    .refractory_period(*, pulse_ms=0.5, threshold=None, ...); .replace(**changes); .config()
vtrap(x, y); hh_rates(v); steady_state_gates(v); temperature_factor(temperature_c)
neuroweave_fi_curve(drives, *, steps=400, transient=100, seed=0, flags=('membrane_dynamics',))
neuroweave_threshold(*, steps=400, seed=0, bracket=(0.0, 2.0), rounds=2, batch=64, flags=...)
adaptation_index(isis); average_ranks(x); spearman_rho(a, b); compare_fi_curves(levels, hh_rates, nw_rates)
```

* Units: um, ohm cm, uF/cm^2, mS/cm^2, mV, ms and uA/cm^2.
* Input is validated before anything runs, and bad input raises `ValueError`.
  The convergence study requires three successive time steps, each half its predecessor. The f-I study
  requires strictly increasing drive levels with at least three at or above threshold and one at or above
  1.1 times threshold.
* Simulations are bounded: at most 2 M steps, 4096 batch elements and 64 pulses. Non-finite states raise
  `FloatingPointError`.
* No randomness is used, and no global NumPy or torch RNG or thread state is touched (tested).

## Connection to Sonapse

* An offline reference. It neither reads nor writes any brain, engine or graft state, apart from
  constructing its own isolated `CellPopulation` cells for the comparison.
* No Sonapse or NEUROWEAVE default, API or checkpoint format changes.
* It gives the abstract NEUROWEAVE cell a fixed biophysical yardstick for the shape of its f-I curve and its
  adaptation. This is not a fidelity target.

## Not implemented

* Branched or tapered morphologies, non-uniform channel densities, synapses, extracellular potentials,
  myelination and other channel types. The cable is uniform and unbranched, with sealed ends.
* HH's own travelling-wave shooting method. Their observed 21.2 m/s is not a target; the check compares with
  their computed 18.8 m/s.
* Temperature dependence of `Cm` or of the conductances. HH also held these constants independent of
  temperature.
* Any claim that the NEUROWEAVE cell is biologically realistic.
* The refractory estimate depends on the test strength (4 × threshold) and the temperature (6.3 °C), and it
  is not compared with a published value.

## Verified references

Checked on 2026-11-20.

1. A. L. Hodgkin, A. F. Huxley. **A quantitative description of membrane current and its application to
   conduction and excitation in nerve.** *The Journal of Physiology* 117(4), 500–544, 28 August 1952.
   doi:10.1113/jphysiol.1952.sp004764, https://physoc.onlinelibrary.wiley.com/doi/10.1113/jphysiol.1952.sp004764,
   PubMed 12991237. Checked against page images of the paper's scanned PDF:
   * the rate equations (12), (13), (20), (21), (23), (24) and the 6.3 °C / Q10 = 3 statement (p. 519);
   * Table 3 constants and its footnote on `V_l` (p. 520);
   * eqns 27–30 for the propagated action potential (p. 522);
   * the factor `phi = 3^((T − 6.3)/10)` and the numerical method (p. 523);
   * `K = 10.47 /ms`, eqn 34, `a = 238 um`, `R2 = 35.4 ohm cm`, `CM = 1.0 uF/cm^2`, 18.5 °C, and the
     computed 18.8 m/s against the observed 21.2 m/s (p. 528).
2. NeuroPulse-CableLab (kai9987kai), a sibling repository. Its README and integrator list (Euler, RK4,
   Rush-Larsen gates, the dt versus dt/2 comparison) were read in a read-only clone. None of its code was
   executed or copied.

## Measured results

Measured on 2026-11-20 on Windows 11 on ARM64, with Python 3.12.10, NumPy 2.4 and one BLAS/OpenMP thread
(`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`). Seeds were 0, 1 and 2. Receipts are deterministic:
two runs of the quick receipt gave the same digest (tested). Wall-clock covers the whole call on a machine
shared with other jobs.

| Run | Verdict | Wall-clock |
|---|---|---|
| `run_experiment(quick=True)` | `pass` (13/13 checks) | 6.1–6.8 s |
| `run_experiment()` | `pass` (13/13 checks) | 36.4 s |

Key numbers (quick / default):

* **Rest:** −64.996 mV. Table 3 gives `V_l` to three decimals (−10.613 mV), which leaves the computed rest
  0.004 mV above −65.
* **Long-step threshold** (a spike within 15 / 40 ms of onset): 2.22–2.30 / 2.229–2.248 uA/cm^2.
* **Patch spike peak:** 41.8 / 42.7 mV.
* **Repetitive firing:**
  * With the default levels, the HH patch fires a single onset spike up to 2.5 × threshold, and repetitive
    firing starts between 2.5 × and 3 × (56.7 Hz at 3 ×, 112.8 Hz at 20 ×, 6.3 °C).
  * With the quick levels, it starts between 2 × and 3 × (57.1 Hz at 3 ×).
  * The HH adaptation index at 4 × threshold is −0.008 / −0.0007, so firing is essentially regular.
* **Convergence** (patch, 20 uA/cm^2 for 1 ms):
  * RK4 first-spike times at dt 0.0125, 0.00625 and 0.003125 ms agree to 2.1e-9 ms, which is the
    interpolation and rounding floor.
  * Exponential Euler changes by 1.9e-4 and then 8.6e-5 ms, an observed order of 1.15.
* **Velocity** (40 mm cable, 18.5 °C):
  * Quick (60 compartments, dt 0.005 ms): 9.23 / 13.15 / 18.65 m/s for 119 / 238 / 476 um. Log-log slope
    0.507. The 476 um axon is 0.79 % below HH's computed 18.8 m/s.
  * Default (200 compartments, dt 0.0006 ms): 9.35 / 13.24 / 18.72 m/s. Slope 0.501. 0.41 % below 18.8 m/s.
  * Relative change at dt/2 ≤ 4.7e-7.
* **Paired-pulse refractory estimate** (6.3 °C; 0.5 ms pulses; conditioning 2 ×, test 4 × the single-pulse
  threshold of 13.3 uA/cm^2): 9.45–9.63 / 9.46–9.49 ms.
* **NEUROWEAVE cell** (descriptive):
  * The threshold drive is 0.277 per step.
  * Repetitive firing starts at 1.1 × (quick) or 1.5 × (default) threshold, earlier than in HH.
  * The normalised rate saturates at 1.0 from 3 × (quick) or 3.5 × (default) threshold upwards.
  * Spearman rank correlation with the HH curve over the suprathreshold levels is 0.839 / 0.858, and the
    mean absolute difference of the normalised curves is 0.20 / 0.24.
  * The adaptation index is 0.0, so firing is regular.
  * Curves are identical across seeds (`seed_invariant: true`).

The cable reproduces the two quantitative anchors it is checked against: the √diameter scaling of eqn 34, and
HH's computed velocity for their axon, to under 1 %. The abstract NEUROWEAVE cell has a monotone f-I curve of
broadly similar rank order. It fires repetitively closer to threshold than HH, its normalised rate saturates
early, and it shows no adaptation. These are shape observations only.
