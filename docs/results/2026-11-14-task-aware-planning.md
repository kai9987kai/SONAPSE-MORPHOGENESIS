# SYNAPSE task-aware delayed-reward planning study

## Frozen confirmation record

- **Decision:** Keep `task_value_mpc` opt-in. The predeclared promotion rule was not met.
- **Protocol:** [Frozen confirmation protocol](2026-11-14-task-aware-planning-protocol.json), digest `0d76689e4e39b75bfbf2eaf37f13b701522dbab3284ef325bb84cd21bbe3cf72`.
- **Raw receipt:** [Verified JSON receipt](2026-11-14-task-aware-planning-receipt.json), integrity `a4186d0f28c7ed0f63a345b4dacf20c1751862a21e5d733336e9c7451bb63194`. The verifier returned `ok: true` with no errors or warnings before this workbench update.
- **Source snapshot:** all 169 fingerprints match the frozen protocol digest `f555ac76ffd37902a126b7dde87892d0ef48e806669c2e7e877e7b962fbea8dd`.
- **Pilot:** seeds 23 and 29; all six query episodes were incomplete; task-aware completion difference was 0 and mean return difference was +0.3818. This was development evidence only.
- **Confirmation:** seeds 37, 41, 43, 47, 53, 59; 18 paired seed/task units; support 64 steps and query cap 256; one CPU thread. All four arms had zero successes, one failure and 17 incomplete episodes. Episodes without an environment terminal result stayed labeled incomplete.
- **Primary outcome:** task-aware pooled completion difference was 0.000 (descriptive 95% interval `[0.000, 0.000]`); forage completion difference was 0.000. This cannot satisfy the frozen requirement of at least +0.10 with a positive lower interval.
- **Return:** task-aware minus actor was −0.2283 (paired descriptive 95% interval `[−0.4863, −0.0274]`). On forage it was −0.6848 (`[−1.2927, −0.1649]`). These small local paired results do not establish broad capability, but provide no basis for promotion.
- **Costs:** 18,832 executed transitions of 32,768 allowed; 1,152 support and 17,680 query transitions; 369,872 model evaluations and 1,109,616 ensemble-member evaluations; no query updates; wall time 1,107.68 seconds.
- **Censoring limit:** 17 of 18 episodes did not reach a terminal state before the query cap. The native default timeout is beyond the cap, so this study leaves completion largely unresolved. The return comparison is reported descriptively and does not replace that primary outcome.

The generated report below retains the full per-task and per-arm summaries, paired intervals, costs, and all 169 source fingerprints. Receipt verification checks recorded hashes and arithmetic; it does not re-execute the environments or authenticate source history.


actual native and complex-world transitions; shared exploratory support trains both experience learners; one frozen paired query episode per seed/task/arm; heuristic short-horizon planner

**Decision: KEEP OPT-IN.** The feature remains off by default unless the frozen completion and forage criteria pass.

## Frozen protocol

- Seeds: `[37, 41, 43, 47, 53, 59]`; tasks: `['forage', 'delayed_match', 'button_sequence']`.
- Shared support: 64 executed transitions per seed/task.
- Query: one episode per arm, capped at 256 transitions.
- Arms: `['actor', 'horizon_1', 'horizon_3', 'task_value_h3']`; horizons: `[1, 3]`; task-aware horizon: 3.
- Promotion rule: task-aware pooled completion improves by at least 10 percentage points with a positive paired descriptive 95% bootstrap lower bound; forage completion is no worse than 5 percentage points; executed transitions stay within 32,768.
- Promotion comparison: +0.000; 95% descriptive interval `[0.0, 0.0]`; forage difference 0.0.

## Aggregate outcomes

| Arm | Completion | Mean return | Mean steps | Planner use | Abstentions | Model evaluations / step | Incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|
| actor | 0.000 | -3.0774 | 245.6 | 0.000 | 0.000 | 0.00 | 17 |
| horizon_1 | 0.000 | -3.1338 | 245.6 | 0.105 | 0.895 | 3.76 | 17 |
| horizon_3 | 0.000 | -3.3255 | 245.6 | 0.177 | 0.823 | 41.73 | 17 |
| task_value_h3 | 0.000 | -3.3057 | 245.6 | 0.137 | 0.863 | 38.19 | 17 |

## Task-level outcomes

### forage

| Arm | Completion | Mean return | Mean steps | Success / fail / timeout / incomplete |
|---|---:|---:|---:|---|
| actor | 0.000 | -2.9121 | 256.0 | 0 / 0 / 0 / 6 |
| horizon_1 | 0.000 | -3.0813 | 256.0 | 0 / 0 / 0 / 6 |
| horizon_3 | 0.000 | -3.6564 | 256.0 | 0 / 0 / 0 / 6 |
| task_value_h3 | 0.000 | -3.5969 | 256.0 | 0 / 0 / 0 / 6 |

### delayed_match

| Arm | Completion | Mean return | Mean steps | Success / fail / timeout / incomplete |
|---|---:|---:|---:|---|
| actor | 0.000 | -1.4558 | 224.7 | 0 / 1 / 0 / 5 |
| horizon_1 | 0.000 | -1.4558 | 224.7 | 0 / 1 / 0 / 5 |
| horizon_3 | 0.000 | -1.4558 | 224.7 | 0 / 1 / 0 / 5 |
| task_value_h3 | 0.000 | -1.4558 | 224.7 | 0 / 1 / 0 / 5 |

### button_sequence

| Arm | Completion | Mean return | Mean steps | Success / fail / timeout / incomplete |
|---|---:|---:|---:|---|
| actor | 0.000 | -4.8644 | 256.0 | 0 / 0 / 0 / 6 |
| horizon_1 | 0.000 | -4.8644 | 256.0 | 0 / 0 / 0 / 6 |
| horizon_3 | 0.000 | -4.8644 | 256.0 | 0 / 0 / 0 / 6 |
| task_value_h3 | 0.000 | -4.8644 | 256.0 | 0 / 0 / 0 / 6 |

## Paired comparisons with actor

| Arm | Completion difference | Completion 95% interval | Return difference | Return 95% interval |
|---|---:|---|---:|---|
| horizon_1 | +0.000 | [0.0, 0.0] | -0.0564 | [-0.14219275260101155, 0.0018236505638652052] |
| horizon_3 | +0.000 | [0.0, 0.0] | -0.2481 | [-0.5742830382086505, -0.03393403161298752] |
| task_value_h3 | +0.000 | [0.0, 0.0] | -0.2283 | [-0.48628924610403096, -0.0274146674985529] |

## Transition and model costs

| Cost | Value |
|---|---:|
| support_world_steps | 1152 |
| query_world_steps | 17680 |
| support_update_calls | 1152 |
| query_update_calls | 0 |
| model_evaluations | 369872 |
| member_evaluations | 1109616 |
| executed_transitions | 18832 |
| configured_transition_budget | 19584 |
| transition_limit | 32768 |

## Source fingerprints

The receipt contains the full per-file fingerprint map. This report includes every recorded fingerprint for direct inspection.

| Source | SHA-256 |
|---|---|
| `__init__.py` | `6270c2ab3c8d1e6e35fa54f748284d3c2f0a82dd167d8df295a64362bc2290e9` |
| `__main__.py` | `79ee6f134c5c5193377cf9b189cbdcc3e2095e49ba9cec310114441c15a917c4` |
| `_vendor/graftbrain/graftbrain/brain.py` | `161edc3283d9cd77bb5106a78aafc346d8538d1afdc47c417f066b5f181d24ea` |
| `_vendor/graftbrain/graftbrain/connectomes.py` | `329af282fafd26b0d0444291f4e53beae198d6f28f2bdfeb8eea6a62e73de3c1` |
| `_vendor/graftbrain/graftbrain/graft.py` | `8663492e63645cd88669503c1c7a1ec051be2d736fe416c8f9d7128617e5df85` |
| `_vendor/graftbrain/graftbrain/neuro_modules.py` | `d1237a929a9710ade3f7b7afd235bcea48d443a89f27c28617d89ad971466fb6` |
| `_vendor/graftbrain/graftbrain/plasticity.py` | `c26184bd8d38670aa7ef1011289a4e9c3f4309cc8f7e8670ec6ae9307c006167` |
| `_vendor/graftbrain/graftbrain/world.py` | `720a6a026e593adf37c280d1a3dffafa185899119f3eaf0f2a41cd478e833bf3` |
| `_vendor/poseidon/poseidon/core.py` | `a171fb9eb82baecd1fb142e598fe94ee80bdf229464673c9c237404734b2df8d` |
| `ablation_engine.py` | `1575063fe3a1bd4a8a6e12f5a94d9b716b5a7f585710bb0acfc1fdf1d39046d5` |
| `actions.py` | `f5f646748581c2f9c36c12d28ae4ef785eb3ef4ecbdeb39712c47be1b12ded48` |
| `active_inference.py` | `686351705ca1062f603c6dd5b5e4dcdd4832206f4046bb3e4a8184156705617d` |
| `active_training.py` | `7218ccccff5fa78143a01d0ac907ace2be27ddf1d63f4ab5e0cfd99c6f4f1697` |
| `activity_scheduler.py` | `765a835deb2daa3e61a809b97292c9e251dcda933f37de86697a780bafb59f1f` |
| `adaptation_coordinator.py` | `06d71c51f49edb0f3ee708d21111c7caf57fb095a6dca2d54dcb04fb31359dee` |
| `adaptation_study.py` | `9e705f9ced061357f5172e53dbca0007b1a6a8784ca55aaa1caa4b14f29f74dd` |
| `adaptive_compute.py` | `850a5f5a3f718e50d72a38bfe088879bb4b0d4bee131c6eb17fb320b1b5c2cfa` |
| `aether_graft_bridge.py` | `e340e5fc826da24ffc3c486c2fe2c5f728ea51fc03eeab0efcb91345087cad37` |
| `agent.py` | `8ae7ecfd668e070b3f2e4da03bdd124bec700e0bb63fc7e927f20527184d46aa` |
| `allostatic_load.py` | `a693de3a577da8eafe7047ce54b3d679a533b3e7dbd747bf034b8cd6b7fd30f8` |
| `apical_amplification.py` | `e2b0e605cd0647547ea1884860cb21471e0edb3ebbbd4825deb3a8ca290d788e` |
| `archimedes_buoyancy.py` | `d5de3088803cf5b0ab9db166c760507f5ba7420dbee18479595e77c79b7027d4` |
| `artifact:tidal_core.pt` | `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27` |
| `astrocyte.py` | `c9cc680483e8c8099bddf2d96d8e22242003e418a80b8a2cb0f60246c15835d1` |
| `auto_programmer.py` | `af1a42abce9148e3143d75a037217b9d34f4ff9e069396c4a3326494afd9d183` |
| `benchmark_suite.py` | `cb1d3866435915c806fcbd0b8f1263b43e1c3c8631ec35bd77b8551f3b627387` |
| `bioacoustic_soundscape.py` | `fd12f4b5c06b60642ff3a6414663afc5a59600ab20a54897d3ef2d895008a3c5` |
| `capabilities.py` | `9e542756f24f89844f08ef018cdb331b02479034dd3c96cba4f834e305d64010` |
| `causal_connectome_assay.py` | `57aff391964068a8c6e9d59a74e8bdf9a17bc773b7849424e6e2c9e1bdfdd905` |
| `causal_discovery.py` | `37b7a9ddf7dbacd2ae44f88471388d0f127bd59ad00ecb173e5418df4479e8dd` |
| `causal_experiment_assay.py` | `42ac614eb76dbcaf7789d39ecfadd9f590ab788754c67b84f668cc454020e05f` |
| `causal_growth.py` | `d7683a457f170a5cdd4fe67d59fe376484c803ee2d134206494dfb2f9b8eb619` |
| `causal_hypotheses.py` | `254faffc59849deed9451cddaad8711ff972d06834d97dcb58ae96016c98ad0f` |
| `causal_mediation.py` | `bdc72695855794f3be3cef80aa26e16274901c21382dcec62ddf26df81b76624` |
| `causal_world_model.py` | `d48b13a5e4484f3a033ff6ca4aea15c9aaaebb19ba20d1550c09fffcf77eee15` |
| `checkpoint_migration.py` | `00b2304525fee579b409ea869341b55160438e6f7808c6b48d9058767a39330c` |
| `circuit_atlas.py` | `86a754ae89294a5543e3d8098a4a2901d024079691c0bb79bb6604dd4014f9a2` |
| `cli.py` | `7aca0af71cc9ca112ac70acd351dae426e09b19f8f62f48948563ea56448c3c2` |
| `complex_world.py` | `8ac632486a2e7a1e91f0918be6b79e220ffefd9c833025d1b37e4c4589f947ea` |
| `compositional_retrieval.py` | `7c49288120cc362c917ae0d046b6702085384f5ff292cffc3eea94fa43d10555` |
| `compute.py` | `aaaeda77780ed44e7aa9263f12976c466ffe77a40955789d83894df85a4bd29b` |
| `config.py` | `f2636ef670ba0b287839627cf14fe652334b0136ce582ee3f9a4533a41037332` |
| `connectome_assets.py` | `7687deeddb4a845283e128af522b3b664272af0b29c12cc613a5f9697d7f5892` |
| `connectome_compiler.py` | `895e062bbf824d5c18cf51a60e358f6507ce032457ec7f2253cdf39c3af8a161` |
| `connectome_controls.py` | `532236ecffa3e3fdcbab85c855cc3a514cb3d74060dc27e02e6e9f21d3a8e4f7` |
| `connectome_datasets.py` | `b6bfce5633eeb3cf0a2df7cef360b99be4ff8e428c5d4591243a21bc84a9226e` |
| `connectome_interventions.py` | `cd82b3645665795eb6d16014fb2abd2648ee3ddd3b49cecf1d0e0c59aca12825` |
| `continual_backprop.py` | `c7607c82122096443b7ab71288ee077923c88252e5071f68028292c093400082` |
| `continual_probe.py` | `e2481b2413e11be0c90d02374f2d45d88201ed07a465d778363f536dc0e6af41` |
| `controlled_study.py` | `acfd707ac83097e6f226d13412fbe79d37e6481d4abc8aceaf7d1d7147801d64` |
| `criticality_vsa.py` | `8e1e8893e371f9500c1a4415f7aac1556f66af18767c178947289c21b702f998` |
| `cross_graft.py` | `92732fdcf76d5d3b2f17fa1bae75df92d75cacfa9232cdd7a434476f10a3f276` |
| `curriculum.py` | `e74cc89967cf4237f81739342a1a02c72a52a2307dcefacb85444235423f4ff3` |
| `dales_law.py` | `1243e1044cee9a5789a5e0940a59628c9f9957f5bb891c22ee8735448640b50b` |
| `darwinian_tournament.py` | `41b8fa8c2e648d5655b82bc313b9d089cee55e39df8cad552d2476ed833f3572` |
| `dendritic_ltc.py` | `b51413fc67e601f203e4859b7dae2fbbcfc16e02f9ed9bf44f722df1b49ec839` |
| `diffusion_repair.py` | `32d60705a482775b878144a285355f52a75a736aff92ed8d5dc82e60c4821282` |
| `digital_tool_arena.py` | `3412754bf37d0d11f57e14666288d41f6281940cd197fa7c46d0255e182b4aa6` |
| `dormant_lifecycle.py` | `7add199ed9c137907a9f6a1db48176442ed9700fb0a3732efcdf675668502c00` |
| `dvgem_tether.py` | `c79d3ea1145d2066c19492a87fccc771238d5fe96a6c554ef20bd79b4a5abdb5` |
| `efficiency.py` | `4546011dbcda07cb9247cd3d615af4bacf5a981a4cc2c0d5b2c22bf989c1e805` |
| `energy_efficiency_assay.py` | `3cab5830f7073cb0d349efbeff0d77dac560f84e62e4309a56ad3052670af316` |
| `epigenetics.py` | `665d0e35fb146f62ead4d931a18b65b4f633f045c1579b98963dce08565ab7cb` |
| `eprop.py` | `898476a3d9bf06b4159471c3c74c1458a23642d1d517aa07264eb44e4d306b13` |
| `event_driven_backend.py` | `3dba850816079075fc4d8de8e6b6fa614adf13c50145678d853f4c3b5b452c0d` |
| `evidence_engine.py` | `3273926cfda50171ab209c71cd3feaf427c0631849543df4fd9f1602738d4ec0` |
| `evolution.py` | `257a5b9835a4c953a12ee0838d07c5e8b85c1b598e270d3f0c247791c37574ae` |
| `experience.py` | `780af1c2e81c0eaf8423ddbc753a91acb04a00feecab6e152aada75b66833e9a` |
| `experience_planner.py` | `36e20911f8cbc9af61e902fd63f8e86b17a8b2eccf10c1e46337ed58538ed5ba` |
| `experiments.py` | `09974016f46949955863b944442f98b0869d93488b62e2ab12686c1874d5c27c` |
| `fast_weight_memory.py` | `f600544cf1463596cb5829d34a2f318e6139413cd67eabd1b09e643d82231cf6` |
| `flagship.py` | `b4890102fb3c0964fef45c749de0ff99ea6ab44c5a3443e571e75dbc36ca8703` |
| `flagship_benchmark.py` | `5a52a4675fd46c7db0c5acec99bc9da993ccd07e921e1c40d12e1aa84f813f6d` |
| `genome.py` | `74a83d48ef39377ba61d009e18705eb6dbed77926d591a212de99f22dbf9c0a6` |
| `glymphatic_clearance.py` | `d2592190c08eee6f7f25dd0230ddfd58fa8ae4a1ebf694d82b8e396c529a3a23` |
| `graft_bus.py` | `7b4e0ba67d0fda515960b05f0bb44054debe1b0f52e4f4b15e17d8733396a731` |
| `grafted_sections.py` | `1584cdc4882cac295ecb1d13a1ee852db528ab52904e81eb33f80e1402814865` |
| `graph_diffusion.py` | `2f7017f60983133c3ed9a5d456da4b49d353a95f4c933e5ee8474fa92627b639` |
| `homeostasis.py` | `ffb23c07f786e14f8d8a86101aefb950b8b8c841f6e5997a68ba8a3d04b3e119` |
| `hybrid_spiking.py` | `5b9b01ee1a0153854a39f3d285a82d16b38cb7dd75a3aac778d195c0380e62b9` |
| `hyperbolic_manifold.py` | `dce30f2ae6235f5dcba5394244b42b6299845a54b3f0b0a59f2ba684ed66c7ab` |
| `hypernetwork.py` | `da132a5544ad35479da9ce3e8491a6c343b023553391d49387b5275305800d0e` |
| `hypernetwork_trainer.py` | `b895cd02bfab62f6868843dcad4625db60505e9f35571f4e72ce59de92258bba` |
| `imagination.py` | `6471fcfd3ca362753cea1ad2dd7107273d94a9c43a8a0004d98f8980c79718d7` |
| `intervention_planner.py` | `933ac2700339e51ebb85e967df4ea701cfb9ab1baba8b1d58db5ec4ee3b4c427` |
| `latent_reasoner.py` | `15d8be1932393168983a26094bd786413ca658ec1f8605c6e43365cb3522c1d1` |
| `learning.py` | `72b4d9508e65345e02f0cb7e7aa909ea2ff67ddecb1ce91d9c497f2cc78bae14` |
| `live_programming.py` | `ddd15e864b08303cc13a7ebc9270ab550394b16f6caf5093f5b7b6ecd32977aa` |
| `manifold_alignment.py` | `30317c02729c3127e8efab35468dc8034a8761609d16a38e129ddc44394e6315` |
| `memory_carrier.py` | `bc47dd4e680781cadcd3093e5b9603232a43a721f2e2cf25307e0891280b2c70` |
| `memory_interference_assay.py` | `25fae8f764155f6bc47402958be34a5011d198acdb418b33a3f808102cedb669` |
| `meta_plasticity.py` | `80fc7258aea9e353228a11c06ccd534b63c37e41d8bed4b6c9f2c3d49ee89b94` |
| `metabolic.py` | `bd73f49caac8919308f2db11dd048975692a3aba308e83d30922be8e8ac8e0f7` |
| `metamorphic.py` | `412e928a0a7c1248d2f774431fc95b0754bac3fb65262baa58ce197ab7358dee` |
| `metaplasticity.py` | `0e64a220795d08f5b68b227fe7ac62bf2e729c1a05c7cbf2d4ec445763cd8621` |
| `metaplasticity_bcm.py` | `1f3b1be021efd39b6bba0ed28d2b43dbb846a1552f92abfef0c55359d4e80f18` |
| `mimomix_reasoner.py` | `c680548070732feaea772ff84c2ae611283408a6f77e444248c7c6a7929cc808` |
| `motif_discovery.py` | `897bae798268d27f74d05fe6bbd7b9c5ee8453bfbcecca418420131030602ca8` |
| `motif_transfer.py` | `905b025aff27ee992c1c3d6c9f9b0dd3a6db872a0e1e68a168f97f48b8a71113` |
| `motivation.py` | `69e883958c40f0744225fececefd07ea08fe4499e1720324a712e4c68076ede4` |
| `multi_connectome.py` | `5e34066d16abbc867663f0740e15d01578a8b132e4ae79af562086146315fdc3` |
| `multi_connectome_fusion.py` | `f873ea00ce83ad4ffee34c4a6abd9429e09b440c6f30f73296e7fabcdac06be0` |
| `multiscale_memory.py` | `7eebb018a5a8e653b9d692c5b9a0189468f631b633b8ba70efd040e51d4978fb` |
| `multistep_planner.py` | `0149a2cedc7e28554f2c2099195d2a750326a92b0fa372d663385238a9aa6e11` |
| `multitimescale_learning.py` | `e9065f4d42d9c84d4a20612d44286f696ea9d03527e81c2165b206892b9a97fe` |
| `neural_comm.py` | `5cffed1f49c1f14029b73f473ddfbde36bae0364b56ac563ef9b408201f596fe` |
| `neural_comm_bench.py` | `068b24880dcdc82c6a91356c37d4830b81ac1deab6d017cda25104611219e122` |
| `neural_comm_channels.py` | `612632c122ddd3639264b989b87f9033eef12597c1f818f0901bdf1244d77803` |
| `neural_comm_topology.py` | `1e1b98e155da27c30e3dee24d4bacc428fcf9013b97e1376655758b4ec808447` |
| `neural_comm_training.py` | `6f3f72ff39c84357d6fa801fb78256af2b0557f96f24d57a47da68a875136071` |
| `neural_graft.py` | `8944f8dc08bc360d9afd2eb497c9abd78a5dd8f04ecdc4c2da4dc09cb00fafb8` |
| `neuro_symbolic.py` | `5babd88df4dccdb78b01b6d78baabe3faf39e5b756d877b5db68410aeb6cd1f6` |
| `neurogenesis.py` | `ee8b3f8737642b0bcb4ea99300dbabe53e5b5cc6e4b1314201ff73beeae505c0` |
| `neuromodulation.py` | `17d216453b826a902a82cd73ccfdfd648d90b1b276f4ea4f9174eb949a2ad098` |
| `object_memory.py` | `6e82771ee1ba1cefd953c2d91fba8ec6187b26e73fa03a934bbac939e424facb` |
| `observatory.py` | `cfb9f5fac5a7e87df10da9575a37c7e497a53b2872eff5b2106f1e8150fbb238` |
| `omnibus_mode.py` | `1c4624fdc62e9fbb1f6cdcb553d15da36c42c4ce7ecc0f743fd7dfbea983723c` |
| `oscillatory_coherence.py` | `7f2cacb7dfdbabb68a300def29a6088f1e5cd8f2a83c13feb3414b3fc8e5934e` |
| `pac_waves.py` | `af819e246b5a70315809093359e119bf453a98d95fec4fc00bc8aee18a3c26bf` |
| `patches.py` | `db8bb9f84669b807b98ff64c78cd76acfb3389e42176928b3e43265c9fed080b` |
| `planning_study.py` | `05792c8e6ed5f91713a050ce9b36bfb2d3938f129edb194cd6b7cebf4aeb4454` |
| `plasticity_monitor.py` | `dfb34ba1951ec4bd80e2515d53d1146f263459d207df863733a61c92ca0b7cc1` |
| `plasticity_rule_learner.py` | `b556546d26a22b20f7f0733d45a683e7fbd182145fab9e86e0283a388ea5e24c` |
| `portable_export.py` | `d7dec8bd115ef52523077ee20b318f995d978f6d7c7f8c6e1cbbf4006d336a5e` |
| `predictive_coding.py` | `72fbbce24e670634a0d1999a8add32b65748c1882fbd6eb0834f80197660c48f` |
| `preference_dpo.py` | `71cfd89d301312c0bcf50b8a64045313e93f8d17aa3b0319cb034f3800c0e7b0` |
| `quantum_kuramoto.py` | `6dbdaed65924384503cae55bfc2ae96b9a192366ccc8d3b20b5829fbd4e8d151` |
| `reconsolidation.py` | `198383f1e0d9b8390699b54117fed140d24b0edd7a5e22acb93aca2cffd70ef5` |
| `recursive_learning.py` | `627a55d07e97ce695aaa5e13d7dad6ef2b157cf4ebd58df3131fc0a344c5090d` |
| `relational_memory.py` | `5948b1edf058be6fbf6f15665626bf7c4ce981efbe710bef3dd686612a08c71c` |
| `research_fabric.py` | `96262f4b65e25fb239a724e7225122066e3a74fc4aebecc3aa9a409e64f79a13` |
| `reusable_skills.py` | `f7afdfa6242f79d5230faaa1804a8b715991935ba1dd5b318dfb0b1d4b23e06c` |
| `rsi.py` | `2cf4a8ad8f035a92cf215d80ce583fd8d7892e0dec0434beb1cb194bb0a90285` |
| `rsi_verifier.py` | `8adbe7be7c889aa8daf8f95f363e71b01d99fd7d86f49607842da878b2f53e13` |
| `runtime.py` | `c483f2c42266b3864d77772a0518f66bc512a98d1120252e2d9e0741a5e1d80c` |
| `self_model.py` | `c25d99e388c71b5914ac50ba92c008fbc09ade60ceff85467c83d8e225eb0925` |
| `server.py` | `bc80270dac13220d44362955313f8d771cd09a39ee33cdd2bf408ae5a4010983` |
| `sim_worker.py` | `f13a2d1d2bca1651c940150d4143542e8e69f14913fe600c94591265a4088a8e` |
| `skill_hierarchy.py` | `c4d5f65c3fbcafb8c14e3c3e8ef05ee31756648333325f75e03b17c019adea66` |
| `sleep.py` | `c7c4063cf4a78d725ea8175b6b2686063bd5719aba315fb9431618b708e1f803` |
| `sources.py` | `df3dc3f1ad44adf12c2d285bd8e0dc5067f9bd7d5309bde2d56393e321484582` |
| `spatial_map.py` | `8da3e1679e8ea9a2d78b169f10ae7e1fa3e31a683e577e1eca36368e1449b62c` |
| `spectral.py` | `6f17a1376d6604a221eb3855b9c312a5ac8e5e2b2edfc77bf1c9b94a438b7f41` |
| `spectral_reservoir.py` | `263d1a8be5844692386e20aa36a636d334f98b5929740e328fd4b216bc56caca` |
| `spiking.py` | `125d6a7bbc22a0d18bd8acc354139189fdf1f87078a7ffc4166a8790682e8810` |
| `spiking_neuromorphic.py` | `34bd3defc2a14b1db117cbe317f90c04d501b3be08568d587b20a5ff26bcdccf` |
| `stateio.py` | `2eb7004b4fdd16358761c972a76bb0f1726f53286d0df035a8aedf5842517621` |
| `stdp.py` | `ee322ca1ebe3b0a50c1bc829418628561a963c9be83021a63149ba29fccfe692` |
| `structural_plasticity.py` | `1de641fa5ba2874e55cf0e8b514907627f4cd3cf80863a1772075a46c33dd669` |
| `supermix_adapters.py` | `d6dc6369c9518e8a21180bb2c315b58ae4a77f7c4a357b5e673c4630d55e1b64` |
| `surprise_memory.py` | `20669d2bc5ada0264aa5c3d43ada5719d5127a123c727c8f33578dbb541d6469` |
| `synesthetic_resonance.py` | `5a7b56e79a4b64db097904044acc4d1ccbc234927239fa6020a37c0d261732ea` |
| `tandem_max.py` | `ad766e50575f1019ec192d3f6f78f979b9617d607aca0a0769462a7416984da1` |
| `temporal_binding.py` | `cdffe8f59e5ee050974838f3bf67ea97f81b6aac2d94be77c39ffd4d4e2064d2` |
| `thermodynamics.py` | `4a552a9054f32f209fcd1bec8f91fb62f8798eb4d3d737cf2dd1b5c7ea6962a0` |
| `transfer.py` | `ad979ef1e6ee62cbee86ed102596a93c2f393347edf95477c83d5e19074bf7eb` |
| `transfer_entropy.py` | `f398d4858222647216cdae61bca8512748fb0e71c85c1bee7afb5005de84ad5b` |
| `transplantation.py` | `e73c93e6fb2e1278f61e175f9bacecc7e584071097af458fb8c391c006e13505` |
| `twin.py` | `d490a9ac49b332c65e5f0364ab4a27b46baa0af6c9779746cea4c26fe88ec7a1` |
| `uncertainty_calibration.py` | `d63b71371efa8490cddf8dbf66fb11535403de566358773f3ada4caeafb09b8a` |
| `uncertainty_guard.py` | `11ad7eb6ce14322c3bc6db45b14fd68a78b57732371976a940c241d926dc5b5e` |
| `unified_eval.py` | `14ae74b51bc1d0e0f51477955e528e2d60920e3b13b33bd484bf65ba38167035` |
| `verified_curriculum.py` | `6820ab7da2d4c3fab75f84d40f3b4bbc2370067d283657bbabf8a6bdab09ea42` |
| `verifier_rsi.py` | `782e050a4bbe624aa278914a7eaf67cd7634e3f5fa2082af3bf828c47c2a11fa` |
| `virtual_ecology.py` | `edca6c01479185710d59422862ab3b5430648fa92371742fd8f60968dc84e933` |
| `vision_encoder.py` | `ec3e94b93bb2cf19c5e108629648a5a5eb77ec8bd42a61107108145a472c50c4` |
| `world4d.py` | `6e01aa960df44cbfb09ce44ff04f02f912cb16800ce4c2f33aeadcac548c12a8` |
| `world_model.py` | `e7a417955c9cba247363f2e605c209512e420ace1d668f337735312d65f95d64` |
| `world_model_v2.py` | `1a913762d4fded22581d36356085aa72c6e5442eaae2e5e240fb5fdfed4987a5` |

## Evidence limits

finite local synthetic-world development study; no real-world reliability or general capability claim.
The unit is one paired seed/task episode. Bootstrap intervals are descriptive with a small local sample. Support is shared and fitted; each query begins from the same saved support state and query fitting is disabled. Task-aware planning is valid only for exact native task identities, uses value-guided short MPC with fixed task/graph/donor context, and may abstain. Receipts verify recorded hashes and arithmetic but do not re-execute worlds, authenticate source history, or support real-world reliability claims.

The confirmation had no separate task-context ablation; it estimates the
integrated task-aware route against the actor and does not isolate the
contribution of task identity, graph context, or donor state.
