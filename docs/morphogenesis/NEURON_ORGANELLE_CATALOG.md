# SONAPSE FOUNDRY — Neuron & Synthetic Organelle Catalogue

All benchmark metrics below are **actual measured CPU runs** from `scripts/run_morphogenesis_experiments.py` (never fabricated).

## 1. Custom Synthetic Graft Neuron Families (21 Specified + 3 Original = 24)

| Family | Name | Objective | Comps | Orgs | Mean MSE (3 seeds) | Δ vs Rate Baseline | Energy | Strongest Task | Weakest Task |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `triarch` | Triarch Neuron | `speculative_synthetic_biology` | 6 | 3 | 0.86614 ± 0.03662 | -0.01569 | 115.27 | `anomaly_suppression` | `spectral_discrimination` |
| `ouroboros` | Ouroboros Neuron | `speculative_synthetic_biology` | 6 | 1 | 0.87658 ± 0.03364 | -0.02613 | 93.68 | `anomaly_suppression` | `spectral_discrimination` |
| `hydra` | Hydra Neuron | `computational_usefulness` | 7 | 5 | 0.94053 ± 0.12113 | -0.09008 | 141.24 | `anomaly_suppression` | `spectral_discrimination` |
| `mnemosome_neuron` | Mnemosome Neuron | `computational_usefulness` | 4 | 2 | 0.87441 ± 0.03495 | -0.02396 | 78.05 | `anomaly_suppression` | `spectral_discrimination` |
| `chameleon` | Chameleon Neuron | `speculative_synthetic_biology` | 4 | 2 | 0.80285 ± 0.05844 | +0.04760 | 73.75 | `anomaly_suppression` | `spectral_discrimination` |
| `janus` | Janus Neuron | `computational_usefulness` | 6 | 2 | 0.86592 ± 0.03742 | -0.01547 | 92.36 | `anomaly_suppression` | `spectral_discrimination` |
| `chronos` | Chronos Neuron | `computational_usefulness` | 6 | 2 | 0.87583 ± 0.03430 | -0.02538 | 117.89 | `anomaly_suppression` | `spectral_discrimination` |
| `nexus` | Nexus Neuron | `computational_usefulness` | 5 | 2 | 0.93818 ± 0.08105 | -0.08773 | 87.23 | `anomaly_suppression` | `spectral_discrimination` |
| `paracrine_broadcast` | Paracrine Broadcast Neuron | `speculative_synthetic_biology` | 3 | 1 | 0.91604 ± 0.05301 | -0.06560 | 55.38 | `anomaly_suppression` | `spectral_discrimination` |
| `mosaic` | Mosaic Neuron | `computational_usefulness` | 5 | 2 | 0.90269 ± 0.04034 | -0.05225 | 86.14 | `anomaly_suppression` | `spectral_discrimination` |
| `sentinel_neuron` | Sentinel Neuron | `computational_usefulness` | 4 | 2 | 0.86525 ± 0.03487 | -0.01480 | 70.39 | `anomaly_suppression` | `spectral_discrimination` |
| `frontier_neuron` | Frontier Neuron | `speculative_synthetic_biology` | 3 | 2 | 0.88451 ± 0.04039 | -0.03407 | 55.27 | `anomaly_suppression` | `spectral_discrimination` |
| `bifrost` | Bifrost Neuron | `computational_usefulness` | 5 | 1 | 1.00645 ± 0.12527 | -0.15601 | 81.16 | `anomaly_suppression` | `spectral_discrimination` |
| `resonator` | Resonator Neuron | `biological_realism` | 4 | 2 | 0.91502 ± 0.03830 | -0.06458 | 70.25 | `anomaly_suppression` | `spectral_discrimination` |
| `palimpsest` | Palimpsest Neuron | `computational_usefulness` | 5 | 2 | 0.87695 ± 0.03244 | -0.02650 | 95.49 | `anomaly_suppression` | `spectral_discrimination` |
| `counterpoint` | Counterpoint Neuron | `computational_usefulness` | 5 | 0 | 0.87569 ± 0.03300 | -0.02525 | 73.94 | `anomaly_suppression` | `spectral_discrimination` |
| `compass` | Compass Neuron | `biological_realism` | 7 | 0 | 0.84757 ± 0.04808 | +0.00288 | 101.44 | `anomaly_suppression` | `spectral_discrimination` |
| `prism` | Prism Neuron | `computational_usefulness` | 5 | 3 | 0.87410 ± 0.06155 | -0.02365 | 96.27 | `anomaly_suppression` | `spectral_discrimination` |
| `consensus` | Consensus Neuron | `computational_usefulness` | 5 | 1 | 0.86351 ± 0.08271 | -0.01306 | 74.58 | `anomaly_suppression` | `spectral_discrimination` |
| `veto` | Veto Neuron | `computational_usefulness` | 4 | 2 | 0.93001 ± 0.07638 | -0.07956 | 71.14 | `anomaly_suppression` | `spectral_discrimination` |
| `tessera` | Tessera Neuron | `computational_usefulness` | 5 | 1 | 0.86619 ± 0.03854 | -0.01574 | 78.92 | `anomaly_suppression` | `spectral_discrimination` |
| `symbiote_echo` | SymbioteEcho Neuron (Original) | `speculative_synthetic_biology` | 5 | 2 | 0.96070 ± 0.09124 | -0.11025 | 86.91 | `anomaly_suppression` | `spectral_discrimination` |
| `bifurcation_gate` | BifurcationGate Neuron (Original) | `computational_usefulness` | 4 | 0 | 0.89827 ± 0.04097 | -0.04783 | 58.30 | `anomaly_suppression` | `spectral_discrimination` |
| `holographic_dendrite` | HolographicDendrite Neuron (Original) | `speculative_synthetic_biology` | 4 | 3 | 0.85740 ± 0.04438 | -0.00695 | 79.46 | `anomaly_suppression` | `spectral_discrimination` |

### Formal Equations & Hypotheses

- **Triarch Neuron** (`triarch`): Disagreement between immediate, contextual, and predictive somas supplies a local self-supervised error signal.  
  Math: `$s_1=\phi_\text{fast}(x),\; s_2=\phi_\text{slow}(s_2, s_1),\; s_3=\hat{s}_1(s_2),\; e_{\text{local}}=s_1-s_3,\; y=\phi(s_1, s_2, e_{\text{local}})$`
- **Ouroboros Neuron** (`ouroboros`): Dual gated axonal-to-dendritic feedback loops with distinct physical delays sustain multi-scale temporal echoes.  
  Math: `$d_{\text{ret}}(t) = \phi(g_1 a(t-\tau_1) + g_2 a(t-\tau_2)),\; a(t)=\phi(W[d_{\text{in}}(t), d_{\text{ret}}(t)])$`
- **Hydra Neuron** (`hydra`): Independently adaptive dendritic branches implement cellular mixture-of-experts and branch-level attention.  
  Math: `$y = \phi\left(\sum_{k=1}^4 \alpha_k(x_k)\, f_k(x_k)\right),\; \alpha_k = \text{softmax}(q^\top h_k)$`
- **Mnemosome Neuron** (`mnemosome_neuron`): Intracellular associative memory organelle extends single-cell retention beyond exponential decay limits.  
  Math: `$m_t = \sum_j w_j(q_t) (0.55 v_j^{\text{fast}} + 0.45 v_j^{\text{slow}}),\; s_t = \phi(x_t + m_t)$`
- **Chameleon Neuron** (`chameleon`): Developmentally shifting computational phenotype adapts excitability from exploration to exploitation.  
  Math: `$\theta_{\text{pheno}}(t) = \text{Maturation}(\text{age}_t, \text{mod}_t);\; y_t = F_{\theta_{\text{pheno}}(t)}(x_t)$`
- **Janus Neuron** (`janus`): Dual independently gated axonal initiation zones decouple predictive reporting from overt behavioral firing.  
  Math: `$y_{\text{pred}} = \sigma(g_p) \hat{s}_t,\; y_{\text{beh}} = \sigma(g_b) s_t$`
- **Chronos Neuron** (`chronos`): Multi-timescale compartments paired with adjustable delay taps resolve temporal credit & rhythm tasks.  
  Math: `$h_k(t+1) = (1-\Delta t/\tau_k) h_k(t) + (\Delta t/\tau_k)\tanh(x(t-\delta_k)),\; \tau_k \in \{3,14,48\}\text{ ms}$`
- **Nexus Neuron** (`nexus`): Cross-coupled modality dendrites learn multiplicative cross-modal coincidences.  
  Math: `$z_{\text{bind}} = \phi\left(\sum_m W_m x_m + \sum_{m<n} \gamma_{mn} (x_m \odot x_n)\right)$`
- **Paracrine Broadcast Neuron** (`paracrine_broadcast`): Diffusible volume transmission coordinates neighboring cells independently of point-to-point synapses.  
  Math: `$\dot{C}_{\text{para}} = -\gamma C_{\text{para}} + \sum_i \beta_i s_{\text{sec},i}$`
- **Mosaic Neuron** (`mosaic`): Hot-swappable internal computational graft docks allow targeted evolution without cell replacement.  
  Math: `$y = \phi(W_a G_a(x) + W_b G_b(x))$`
- **Sentinel Neuron** (`sentinel_neuron`): Dedicated error-detection compartments flag anomalous or contradictory local neural patterns rapidly.  
  Math: `$a_{\text{err}} = |x_t - \hat{x}_t| + \lambda \hat{\sigma}_t$`
- **Frontier Neuron** (`frontier_neuron`): Simulated growth-cone compartment explores candidate targets based on correlation and novelty gradients.  
  Math: `$\Delta p_{\text{cone}} \propto \nabla \text{coherence}(s_{\text{cone}}, s_{\text{target}})$`
- **Bifrost Neuron** (`bifrost`): Dual native-connectome and ML-adapter interfaces enable clean cross-paradigm coupling with Supermix models.  
  Math: `$y_{\text{bio}} = \phi(W_b s),\; y_{\text{ML}} = W_{\text{adapt}} s$`
- **Resonator Neuron** (`resonator`): Coupled oscillatory compartments selectively amplify resonant frequencies and encode phase synchrony.  
  Math: `$\dot{x} = \omega y + I,\; \dot{y} = -\omega x - \gamma y$`
- **Palimpsest Neuron** (`palimpsest`): Gated separation of fast labile vs slow consolidated compartments mitigates catastrophic forgetting.  
  Math: `$m_{\text{fast}}(t) \text{ overwrites freely},\; m_{\text{slow}}(t) = (1-\epsilon g_t) m_{\text{slow}}(t-1) + \epsilon g_t m_{\text{fast}}(t)$`
- **Counterpoint Neuron** (`counterpoint`): Competing excitatory and inhibitory internal pathways sharpen local discrimination and temporal edge detection.  
  Math: `$d = \tanh(e_+ - e_-),\; e_+ \dashv\vdash e_-$`
- **Compass Neuron** (`compass`): Cyclic ring-attractor internal compartments maintain directional/heading phase representations.  
  Math: `$\dot{r}_k = -r_k + \phi\left(\sum_j W_{\text{ring}}(k-j) r_j + I_k\right)$`
- **Prism Neuron** (`prism`): Cable-geometry-tuned dendrites decompose mixed temporal inputs into distinct frequency bands.  
  Math: `$d_b(t) = H_{\tau_b, \omega_b}[x(t)],\; y = \phi(\sum_b w_b d_b)$`
- **Consensus Neuron** (`consensus`): Cross-gated evidence dendrites fire strongly only under multi-branch agreement, rejecting single-channel noise.  
  Math: `$y = \phi\left(\sum_{i=1}^3 \sigma(3 e_{(i+1)\bmod 3})\, e_i\right)$`
- **Veto Neuron** (`veto`): Fast shunting veto compartment suppresses unstable or contextually inappropriate outputs before axonal launch.  
  Math: `$y = \phi(I_{\text{drive}} - \beta_{\text{veto}} I_{\text{veto}}^+)$`
- **Tessera Neuron** (`tessera`): Independently gated output branches multiplex fast, delayed, and modulatory projections from one soma.  
  Math: `$y_k = g_k(c_t)\, \phi(s(t - \tau_k)),\; k \in \{\text{fast}, \text{mem}, \text{mod}\}$`
- **SymbioteEcho Neuron (Original)** (`symbiote_echo`): Dedicated mirror soma tracks peer-connectome discrepancy while CheckpointVesicle protects native identity.  
  Math: `$m_t = \phi(x_{\text{peer}} - \alpha s_t),\; s_t = \phi(x_{\text{self}} + \sigma(s_t) m_t)$`
- **BifurcationGate Neuron (Original)** (`bifurcation_gate`): Symmetric pitchfork-bistable coupled somas act as a single-neuron noise-tolerant 1-bit working memory latch.  
  Math: `$\dot{a} = -a + \tanh(0.95 a - 0.75 b + x),\; \dot{b} = -b + \tanh(0.95 b - 0.75 a - x)$`
- **HolographicDendrite Neuron (Original)** (`holographic_dendrite`): Quadrature phase-modulated dendrites superpose temporal sequences via phase interference inside one cell.  
  Math: `$z_t = \phi(d_{\sin}\cos\omega t + d_{\cos}\sin\omega t + m_t)$`

## 2. Improved Conventional Neuron Families (15)

| Family | Name | Fidelity | Objective | Hypothesis |
|---|---|---:|---|---|
| `lif_reference` | LIF Reference Neuron | F3 | `biological_realism` | Baseline spiking neuron benchmark for matching spike-rate efficiency. |
| `recurrent_rate` | Conventional Recurrent Rate Neuron | F1 | `computational_usefulness` | Lowest-cost continuous rate control baseline. |
| `izhikevich` | Izhikevich Adaptive Spiking Neuron | F3 | `biological_realism` | Supports tonic spiking, adaptation, and bursting at low CPU cost. |
| `conductance_hh` | Hodgkin-Huxley Conductance Neuron | F4 | `biological_realism` | Biophysical conductance reference quantifying ion-channel nonlinearity vs compute cost. |
| `adaptive_pyramidal` | Adaptive Pyramidal Neuron | F2 | `biological_realism` | Apical-basal NMDA coincidence detects top-down context paired with bottom-up input. |
| `predictive_interneuron` | Predictive Inhibitory Interneuron | F2 | `computational_usefulness` | Fast inhibitory cancellation of predictable signals improves surprise contrast. |
| `dynamic_threshold` | Dynamic-Threshold Neuron | F3 | `biological_realism` | Dynamic spike thresholds prevent saturation across wide stimulus amplitudes. |
| `homeostatic_regulated` | Homeostatically Regulated Neuron | F2 | `biological_realism` | Maintains target firing band under drifting input statistics. |
| `metaplastic_bcm` | Metaplastic BCM Neuron | F2 | `biological_realism` | Sliding modification threshold stabilizes Hebbian selectivity. |
| `sparse_burst` | Sparse Burst Neuron | F3 | `computational_usefulness` | Sparse high-confidence bursts reduce metabolic cost per transmitted bit. |
| `context_motor` | Context-Sensitive Motor Neuron | F2 | `computational_usefulness` | Separates motor drive from contextual execution permission. |
| `adaptive_conduction` | Adaptive Conduction Neuron | F2 | `biological_realism` | Myelin-modulated conduction delays synchronize arrival times across distant pathways. |
| `developmental_precursor` | Developmental Precursor Neuron | F2 | `speculative_synthetic_biology` | Undifferentiated progenitor cells specialize dynamically during early exposure. |
| `recovery_compensatory` | Recovery & Compensatory Plasticity Neuron | F2 | `computational_usefulness` | Redundant basal dendrites + Repairosome preserve signal throughput after single-branch lesions. |
| `reliability_sensitive` | Reliability-Sensitive Neuron | F2 | `computational_usefulness` | Down-weights volatile noisy inputs via local variance tracking. |

## 3. Modular Synthetic Cellular Organelles (14)

| Organelle | Class | Base Resource Cost | Description |
|---|---|---:|---|
| `mnemosome` | `MnemosomeOrganelle` | 0.45 | Internal fast/slow associative memory trace organelle. |
| `predicton` | `PredictonOrganelle` | 0.30 | Predicts next local compartment state from recent activity & drive. |
| `contraston` | `ContrastonOrganelle` | 0.20 | Computes signed contrast between expected and observed compartment activity. |
| `chronosome` | `ChronosomeOrganelle` | 0.30 | Programmable intracellular delay buffer for temporal coincidence detection. |
| `attentosome` | `AttentosomeOrganelle` | 0.25 | Context- and salience-gated selective input amplification organelle. |
| `routosome` | `RoutosomeOrganelle` | 0.25 | Computes soft routing weights selecting between feed-forward and recurrent paths. |
| `plasticity_nucleus` | `PlasticityNucleusOrganelle` | 0.30 | Regulates local synaptic plasticity gate based on surprise and dopamine. |
| `checkpoint_vesicle` | `CheckpointVesicleOrganelle` | 0.25 | Snapshots high-value compartment states and restores them on sharp error drops. |
| `repairosome` | `RepairosomeOrganelle` | 0.30 | Detects hypo/hyper-activity damage and injects homeostatic compensatory gain. |
| `uncertaintosome` | `UncertaintosomeOrganelle` | 0.25 | Tracks rolling squared residual variance to estimate local state uncertainty. |
| `translator_membrane` | `TranslatorMembraneOrganelle` | 0.20 | Bidirectional format converter bridging rate, spiking, and ML vector encodings. |
| `phase_organelle` | `PhaseOrganelle` | 0.30 | Internal Kuramoto/Hopf oscillator supporting phase-coherent gating. |
| `resource_regulator` | `ResourceRegulatorOrganelle` | 0.15 | Maintains local energy reserve and suppresses runaway firing under scarcity. |
| `developmental_genome` | `DevelopmentalGenomeOrganelle` | 0.20 | Tracks cellular maturation age and shifts excitability/phenotype cleanly. |

## 4. Non-Neural Biological Abstractions (12) & Software/ML Hybrid Mechanisms (13)

- **Bio-system abstractions (`SPECULATIVE_SYNTHETIC_BIOLOGY`):** `astrocytic_regulation`, `cardiac_pacemaker`, `cellular_repair`, `chemical_gradient_sensing`, `cochlear_mechanics`, `developmental_differentiation`, `endocrine_signalling`, `immune_recognition`, `intracellular_signalling_cascade`, `muscle_adaptation`, `retinal_processing`, `vascular_resource_distribution`
- **Software/ML hybrid mechanisms (with 3-variant exact/cellular/biological comparison):** `dendritic_attention`, `moe_branch_specialisation`, `state_space_compartment`, `reservoir_organelle`, `fast_weight_memory`, `recurrent_gating`, `memristive_synapse`, `error_correcting_pathway`, `wave_photonic_abstraction`, `local_predictive_coding`, `differentiable_routing`, `associative_hopfield_memory`, `external_ai_adapter`