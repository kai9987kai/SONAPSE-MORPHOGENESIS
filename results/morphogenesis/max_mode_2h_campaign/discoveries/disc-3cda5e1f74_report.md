# Scientific Discovery Report: OMNI-MORPHOGENESIS Unified Max Mode Campaign

- **Discovery ID:** `disc-3cda5e1f74`
- **Experiment ID:** `exp-9ac0252680`
- **Module:** `max_mode`
- **Evidence State:** `larger_scale_validated`
- **Date (UTC):** `2026-11-26 01:13:34Z`
- **Source Branch:** `feat/sonapse-morphogenesis`

## 1. Research Hypothesis
A 3-tier nested oscillatory coordinator with OrganelleArbitrationCoordinator and FAULTLINE/Tether specialist weighting enables all 57 neuron families, 20 organelles, 33 learning rules, and coupled Sonapse-LAB Tandem Max simultaneously without destructive interference.

## 2. Actual Measured Results
```json
{
  "epochs_completed": 1956,
  "total_gamma_ticks": 46944,
  "total_theta_cycles": 7824,
  "early_quartile_reward": 0.59665,
  "late_quartile_reward": 0.59746,
  "learning_progression_delta": 0.0008,
  "paired_bootstrap_stats": {
    "mean_diff": 0.40097,
    "ci95_low": 0.377007,
    "ci95_high": 0.425196,
    "p_value": 0.00125,
    "cohens_d": 1.3962,
    "n_seeds": 512
  },
  "specialist_final_weights": {
    "compiled_pan_connectome": 0.1932,
    "focal_f1_f5_organelle_core": 0.1142,
    "ml_hybrid_18_bank": 0.2622,
    "bio_hybrid_16_collective": 0.2735,
    "host_sonapse_max_tandem": 0.0462,
    "antipode_repulsion_guard": 0.1108
  },
  "accepted_evolutions": 1001,
  "rejected_evolutions": 955,
  "organelle_arbitrations": 46944,
  "crosswire_abstentions": 0,
  "quarantine_accepted": 1319,
  "tidal_probe": {
    "probed": true,
    "count": 16,
    "failures": 12
  },
  "expanse_failures_loaded": 164
}
```

## 3. Matched Baseline Comparisons
```json
{
  "random_control_mean_reward": 0.19478
}
```

## 4. Statistical Uncertainty & Multiple-Testing Control
```json
{
  "bootstrap": {
    "mean_diff": 0.40097,
    "ci95_low": 0.377007,
    "ci95_high": 0.425196,
    "p_value": 0.00125,
    "cohens_d": 1.3962,
    "n_seeds": 512
  },
  "fdr": {
    "alpha": 0.05,
    "adjusted_p": [
      0.0025,
      0.03
    ],
    "rejected": [
      true,
      true
    ]
  }
}
```

## 5. Causal Intervention & Ablation Findings
```json
{
  "completeness_verified": true,
  "organelle_arbitrations": 46944
}
```

## 6. Relevant Failed / Negative Experiments
```json
[]
```

## 7. Alternative Explanations & Caveats
- Capacity differences (compartment count / state dimensionality) must be controlled alongside wall-clock FLOPs.
- Synthetic benchmarks do not establish biological validity or general intelligence.

## 8. Reproduction Instructions
```powershell
python -m sonapse_morphogenesis max-mode --seed 7 --epochs 1956
```