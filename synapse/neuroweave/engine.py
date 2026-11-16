"""NeuroWeaveEngine: causal composition of every NEUROWEAVE subsystem.

The engine wires the reviewed subsystems (brain, neuromodulation, sensory hub,
memory ecology, predictive hierarchy, efference self-model, Micro-Collective,
specialist graft, representation fusion, event bus and the optional ecology)
into one step. Every coupling is causal and gated by its mechanism flag:

* sensory frame -> brain external input (``multimodal_sensory``), weighted by
  the neuromodulatory ``sensory_weight`` (``neuromodulator_field`` /
  ``affective_modulation``);
* neuromodulatory controls -> brain controls (gain, threshold shift, attention,
  plasticity and learning-rate scale), reward scale, readout temperature,
  memory consolidation / replay volume and prediction-packet precision;
* predictive-hierarchy disagreement -> brain uncertainty (``uncertainty_gating``);
* hierarchical surprise, sensory prediction error, Micro-Collective error and
  memory novelty -> neuromodulator update;
* the Micro-Collective trains on consecutive sensory frames
  (``micro_collective``) and its forecast feeds the fusion context and the
  PREDICTED memory items; the specialist graft contributes through its
  zero-initialised coupling (``specialist_graft``);
* memory ecology (``memory_ecology``) stores SENSORY observations and PREDICTED
  forecasts, confirms forecasts when the outcome arrives and feeds verified
  recall into the drive; predictive replay (``predictive_replay``) replays the
  memory schedule into the predictors every ``spec.sleep_interval`` steps;
* region route gains for ``multi_region_routing`` are modulated by attention
  and gain controls;
* structural plasticity / neurogenesis run inside the brain at
  ``spec.structural_interval`` within ``spec.max_synapses`` / ``spec.max_neurons``.

With no mechanisms enabled ``step(task_input=x)`` is exactly
``NeuroWeaveBrain.step(x)``. Everything is a synthetic computational
abstraction (``evidence_tier='synthetic_abstraction'``); raw media are never
stored.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from .brain import NeuroWeaveBrain
from .events import EventBus
from .memory import MemoryEcology, Provenance
from .micro_collective import MicroCollective, SpecialistGraft
from .neuromod import NeuromodulatorSystem, infer_transmitters
from .packets import RepresentationFusion, encode_vector
from .prediction import EfferenceSelfModel, PredictiveHierarchy
from .sensory import SensoryHub
from .spec import (AFFECT_VARIABLES, DENDRITES, EVIDENCE_TIER, NEUROMODULATORS, REGIONS, SENSORY_CHANNEL_IDS,
                   SENSORY_DIM, SOMA_ROLES, TASK_INPUT_DIM, NeuroWeaveSpecV1)

__all__ = ['NeuroWeaveEngine', 'ENGINE_SCHEMA', 'ACTION_DIM', 'DRIVE_DIM', 'ENGINE_FLAGS']

ENGINE_SCHEMA = 'NeuroWeaveEngineStateV1'
ACTION_DIM = 2                     # [thrust, turn]
DRIVE_DIM = TASK_INPUT_DIM + 2 * SENSORY_DIM   # task | sensory | memory recall
ENGINE_OFFSET = 7_919
MEMORY_CAPACITY = 256
CONFIRM_TOL = 0.25
EVENT_HISTORY = 32
ENGINE_FLAGS: tuple[str, ...] = (
    'multimodal_sensory', 'cross_modal_memory', 'neuromodulator_field', 'affective_modulation',
    'receptor_heterogeneity', 'predictive_ensemble', 'hierarchical_prediction', 'uncertainty_gating',
    'counterfactual_prediction', 'efference_self_model', 'micro_collective', 'expert_routing', 'meta_plasticity',
    'specialist_graft', 'memory_ecology', 'predictive_replay', 'representation_fusion', 'multi_region_routing',
    'transmitter_inference',
)
_PREDICTION_FLAGS = frozenset({'predictive_ensemble', 'hierarchical_prediction', 'uncertainty_gating',
                               'counterfactual_prediction', 'predictive_replay'})
_MOD_FLAGS = frozenset({'neuromodulator_field', 'affective_modulation'})
_DRIVE_FLAGS = frozenset({'multimodal_sensory', 'memory_ecology', 'representation_fusion'})
_THREAT_CH = tuple(SENSORY_CHANNEL_IDS.index(c) for c in
                   ('visual_motion', 'visual_edge_density', 'mechanosensation', 'thermal', 'audio_high'))
_INTERO = SENSORY_CHANNEL_IDS.index('interoception')
_SCALAR_CONTROLS = ('learning_rate_scale', 'plasticity_scale', 'attention', 'exploration', 'consolidation_scale',
                    'prediction_precision', 'decision_temperature', 'persistence', 'threshold_shift',
                    'reward_scale', 'gain')
_REGION_KIND = tuple(kind for _, kind in REGIONS)
_PREMOTOR = [i for i, (name, _) in enumerate(REGIONS) if name == 'premotor'][0]
_MOTOR = [i for i, (name, _) in enumerate(REGIONS) if name == 'motor'][0]


def _finite_vec(name: str, value: Any, *, max_len: int = 256) -> np.ndarray:
    try:
        v = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if v.size > max_len or not np.all(np.isfinite(v)):
        raise ValueError(f'{name} must be a finite vector of length <= {max_len}')
    return v


def _finite_scalar(name: str, value: Any) -> float:
    """A checkpoint scalar: a real int/float (not bool, not a string such as 'nan') and finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))             or not math.isfinite(float(value)):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _fit(v: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n)
    m = min(n, v.size)
    out[:m] = v[:m]
    return out


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(z, -40.0, 40.0))))


def _rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


class NeuroWeaveEngine:
    """One causal NEUROWEAVE system built from a :class:`NeuroWeaveSpecV1`."""

    def __init__(self, spec: NeuroWeaveSpecV1 | None = None) -> None:
        spec = spec if spec is not None else NeuroWeaveSpecV1()
        if not isinstance(spec, NeuroWeaveSpecV1):
            raise ValueError('spec must be a NeuroWeaveSpecV1')
        self.spec = spec
        self.flags = frozenset(spec.mechanisms)
        f = self.flags
        self.brain = NeuroWeaveBrain(spec)
        b = self.brain
        self.neuromod = NeuromodulatorSystem(b.n, seed=spec.seed, signs=b.cells.sign,
                                             src=b.synapses.src, dst=b.synapses.dst)
        self.sensory = SensoryHub(spec)
        self.memory = MemoryEcology(SENSORY_DIM, SENSORY_DIM, MEMORY_CAPACITY, spec.seed)
        horizons = (1, 4, 16, 64) if 'hierarchical_prediction' in f else (1,)
        # the ensemble (and so its disagreement-based uncertainty) exists only with predictive_ensemble
        ensemble = 3 if 'predictive_ensemble' in f else 1
        self.prediction = PredictiveHierarchy(SENSORY_DIM, horizons=horizons, ensemble=ensemble, seed=spec.seed,
                                              action_dim=ACTION_DIM if 'counterfactual_prediction' in f else 0)
        self.efference = EfferenceSelfModel(ACTION_DIM, seed=spec.seed)
        self.micro = MicroCollective.from_spec(spec)
        self.graft: SpecialistGraft | None = None
        if 'specialist_graft' in f:
            self.graft = SpecialistGraft(SENSORY_DIM, SENSORY_DIM, seed=spec.seed)
            self.micro.attach_specialist(self.graft)
        self.fusion = RepresentationFusion(DRIVE_DIM, kinds=('predictive', 'uncertainty'), mode='learned',
                                           seed=spec.seed)
        self.events = EventBus(EVENT_HISTORY)
        self.ecology = None
        self._rng = np.random.default_rng(spec.seed + ENGINE_OFFSET)
        self._reset_runtime()

    # ------------------------------------------------------------------ runtime state
    def _reset_runtime(self) -> None:
        self.step_count = 0
        self.prev_frame: np.ndarray | None = None
        self.prev_forecast: np.ndarray | None = None       # forecast of *this* step made last step
        self.prev_drive: np.ndarray | None = None
        self.prev_packet_state: dict[str, Any] | None = None
        self.pending_predicted_id: int | None = None
        self.last_action = np.zeros(ACTION_DIM)
        self.authorship = 1.0
        self.counters = {'micro_train_steps': 0, 'memory_sensory': 0, 'memory_predicted': 0, 'memory_confirmed': 0,
                         'memory_contradicted': 0, 'replayed': 0, 'replay_cycles': 0, 'fusion_updates': 0,
                         'efference_records': 0, 'ecology_steps': 0}
        self.last: dict[str, Any] = {}

    def end_episode(self) -> None:
        """Forget cross-step transients so no transition is learned across an episode boundary."""
        self.prev_frame = None
        self.prev_forecast = None
        self.prev_drive = None
        self.prev_packet_state = None
        self.pending_predicted_id = None
        self.last_action = np.zeros(ACTION_DIM)

    @property
    def n(self) -> int:
        return self.brain.n

    def has(self, mechanism: str) -> bool:
        return self.spec.has(mechanism)

    # ------------------------------------------------------------------ helpers
    @property
    def _mod_on(self) -> bool:
        return bool(self.flags & _MOD_FLAGS)

    @property
    def _prediction_on(self) -> bool:
        return bool(self.flags & _PREDICTION_FLAGS)

    def _raw_sensory(self, sensory: Any, ecology_obs: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if sensory is None:
            if ecology_obs is not None:
                self.sensory.simulate(ecology_obs, self.step_count)
            elif 'multimodal_sensory' in self.flags:
                self.sensory.simulate(None, self.step_count)
            return None
        if isinstance(sensory, Mapping):
            return dict(sensory)
        v = _finite_vec('sensory', sensory, max_len=SENSORY_DIM)
        if v.size != SENSORY_DIM:
            raise ValueError(f'sensory vector must have length {SENSORY_DIM}')
        return {cid: float(v[i]) for i, cid in enumerate(SENSORY_CHANNEL_IDS)}

    def _packets(self, forecast: np.ndarray | None, unc: list[float], precision: float) -> list:
        conf = float(np.clip(0.5 * precision, 0.0, 1.0))
        pk = [encode_vector(unc, kind='uncertainty', confidence=1.0, provenance='synthetic',
                            source=('engine', 'prediction'))]
        if forecast is not None:
            pk.append(encode_vector(np.clip(forecast, -1.0, 1.0), kind='predictive', confidence=conf,
                                    provenance='predicted', source=('engine', 'forecast')))
        return pk

    def route_gain_matrix(self, controls: Mapping[str, Any]) -> np.ndarray | None:
        """Region-pair gains: learned routes x attention on sensory sources x gain on action targets."""
        att = float(controls.get('attention', 1.0))
        gain = float(controls.get('gain', 1.0))
        if att == 1.0 and gain == 1.0:
            return None
        mod = np.ones((len(REGIONS), len(REGIONS)))
        for i, kind in enumerate(_REGION_KIND):
            if kind == 'sensory':
                mod[i, :] *= att
            if kind == 'action':
                mod[:, i] *= gain
        return np.clip(self.brain.synapses.route_gain * mod, 0.0, 2.0)

    def readout_probability(self, temperature: float = 1.0) -> float:
        b = self.brain
        z = b.readout_b + float(b.readout_w @ b.readout_features())
        return _sigmoid(z / max(1e-3, float(temperature)))

    # ------------------------------------------------------------------ step
    def step(self, task_input: Any = None, sensory: Any = None, *, reward: float = 0.0, learn: bool = True,
             ecology_obs: Mapping[str, Any] | None = None, intended_action: Any = None, issued_action: Any = None,
             observed_outcome: Any = None) -> dict[str, Any]:
        f = self.flags
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError('reward must be finite')
        if type(learn) is not bool:
            raise ValueError('learn must be a boolean')
        x_task = None if task_input is None else _finite_vec('task_input', task_input)
        issued = None if issued_action is None else _fit(_finite_vec('issued_action', issued_action), ACTION_DIM)
        intended = issued if intended_action is None else _fit(_finite_vec('intended_action', intended_action), ACTION_DIM)
        observed = None if observed_outcome is None else _fit(_finite_vec('observed_outcome', observed_outcome), ACTION_DIM)
        raw = self._raw_sensory(sensory, ecology_obs)
        bus, step = self.events, self.step_count

        # 1. sensory fusion with the previous step's modulatory attention
        prev_controls = self.neuromod.controls(f)
        frame = self.sensory.fuse(raw, f, sensory_weight=prev_controls['sensory_weight'] if self._mod_on else None,
                                  learn=learn)
        value = frame.value
        bus.publish('input', {'channels': value.tolist(), 'reliability': frame.reliability.tolist(),
                              'missing': [int(i) for i in np.flatnonzero(frame.missing)]}, step=step)

        # 2. efference self-model (agency-weighted credit)
        efference = None
        if 'efference_self_model' in f and issued is not None and observed is not None:
            efference = self.efference.record(intended, issued, self.efference.forward(issued), observed, learn=learn)
            self.authorship = float(efference['authorship'])
            self.counters['efference_records'] += 1
            bus.publish('action', {'action': issued.tolist(), 'intended': intended.tolist(),
                                   'authorship': self.authorship,
                                   'reliability': float(efference['control_reliability'])}, step=step)
        elif issued is not None:
            bus.publish('action', {'action': issued.tolist()}, step=step)

        # 3. Micro-Collective: train on the observed transition, then forecast the next frame
        micro_info: dict[str, Any] = {'active': False}
        micro_err = 0.0
        micro_forecast = None
        if 'micro_collective' in f:
            micro_info['active'] = True
            if self.prev_forecast is not None:
                micro_err = float(np.mean((self.prev_forecast - value) ** 2))
            if learn and self.prev_frame is not None:
                tr = self.micro.train_step(self.prev_frame, value)
                self.counters['micro_train_steps'] += 1
                micro_info['train'] = {'loss': tr['loss'], 'held_out': tr['held_out'],
                                       'prediction_error': tr['prediction_error']}
                bus.publish('learning', {'kind': 'micro_collective', 'magnitude': float(tr['prediction_error']),
                                         'updates': int(self.micro.updates),
                                         **({'loss': float(tr['loss'])} if tr['loss'] is not None else {})}, step=step)
            micro_forecast, routing = self.micro.predict(value)
            micro_forecast = np.clip(micro_forecast, -1.0, 1.0)
            micro_info.update(prediction=micro_forecast.tolist(), experts=routing['experts'],
                              specialist_coupling=routing['specialist_coupling'], error=micro_err)

        # 4. predictive hierarchy: surprise, uncertainty gate, adaptive horizon
        pred_info: dict[str, Any] = {'active': False, 'surprise': 0.0, 'gate': 1.0, 'horizon': 1, 'error': 0.0}
        hier_forecast = None
        if self._prediction_on:
            action = self.last_action if self.prediction.action_dim else None
            obs = self.prediction.observe(value, action, learn=learn)
            h0 = self.prediction.horizons[0]
            horizon = int(obs['horizon']) if 'hierarchical_prediction' in f else h0
            gate = float(self.prediction.uncertainty_gate(horizon))
            surprise = float(np.mean([obs['surprise'][h] for h in self.prediction.horizons]))
            err = float(obs['scored'].get(h0, 0.0))
            hier_forecast = np.clip(self.prediction.forecast(value, action)[h0]['mean'], -1.0, 1.0)
            pred_info = {'active': True, 'surprise': surprise, 'gate': gate, 'horizon': horizon, 'error': err,
                         'disagreement': float(self.prediction.disagreement[horizon])}
            bus.publish('prediction', {'horizon': horizon, 'error': err, 'surprise': surprise, 'gate': gate,
                                       'disagreement': pred_info['disagreement']}, step=step)
        counterfactual = None
        if 'counterfactual_prediction' in f and issued is not None:
            counterfactual = [{'action': c['action'], 'forecast_h1': c['forecast'][self.prediction.horizons[0]]['mean']}
                              for c in self.prediction.counterfactual(value, [issued, -issued, np.zeros(ACTION_DIM)])]

        # 5. memory ecology: confirm last forecast, retrieve verified recall
        mem_info: dict[str, Any] = {'active': False, 'hits': 0}
        recall = np.zeros(SENSORY_DIM)
        novelty = float(np.clip(2.0 * np.mean(frame.prediction_error), 0.0, 1.0))
        if 'memory_ecology' in f:
            mem = self.memory
            mem_info['active'] = True
            novelty = float(mem.default_surprise(value))
            if learn and self.pending_predicted_id is not None and self.pending_predicted_id in mem:
                item = mem.get(self.pending_predicted_id)
                if item['provenance'] == Provenance.PREDICTED.value:
                    res = mem.confirm(self.pending_predicted_id, value, CONFIRM_TOL)
                    self.counters['memory_confirmed' if res['confirmed'] else 'memory_contradicted'] += 1
                    mem_info['confirmation'] = {'confirmed': res['confirmed'], 'error': res['error']}
                    bus.publish('memory', {'operation': 'confirm', 'id': int(res['id']),
                                           'provenance': res['provenance']}, step=step)
            self.pending_predicted_id = None
            hits = mem.retrieve(value, k=4, verified_only=True)
            if hits:
                w = np.array([max(0.0, float(h['score'])) for h in hits])
                if w.sum() > 0:
                    recall = np.clip((w[:, None] * np.stack([np.asarray(h['value'], dtype=np.float64) for h in hits])).sum(0)
                                     / w.sum(), -1.0, 1.0)
            mem_info.update(hits=len(hits), recall_norm=float(np.linalg.norm(recall)),
                            ids=[int(h['id']) for h in hits])
            bus.publish('memory', {'operation': 'retrieve', 'hits': len(hits)}, step=step)

        # 6. neuromodulator appraisal from surprise / novelty / error, then controls for this step
        if self._mod_on:
            pe = float(np.clip(np.mean(frame.prediction_error) + min(1.0, pred_info['surprise'] / 3.0)
                               + micro_err, 0.0, 1.0))
            threat = float(np.clip(np.mean(np.maximum(0.0, value[list(_THREAT_CH)])), 0.0, 1.0))
            energy = 0.5 if frame.missing[_INTERO] else float(np.clip((value[_INTERO] + 1.0) / 2.0, 0.0, 1.0))
            out_prev = self.brain.cells.output
            self.neuromod.update(float(np.clip(reward, -1.0, 1.0)), pe, novelty, threat, energy, f,
                                 unit_activity=out_prev[:self.neuromod.n])
        controls = self.neuromod.controls(f)
        if self._mod_on:
            bus.publish('modulation', {'levels': self.neuromod.level_dict, 'affect': self.neuromod.affect.to_dict(),
                                       'controls': {k: float(controls[k]) for k in _SCALAR_CONTROLS}}, step=step)

        # 7. compose the external drive
        forecast_next = micro_forecast if micro_forecast is not None else hier_forecast
        if f & _DRIVE_FLAGS:
            drive = np.zeros(DRIVE_DIM)
            if x_task is not None:
                drive[:TASK_INPUT_DIM] = _fit(x_task, TASK_INPUT_DIM)
            if 'multimodal_sensory' in f:
                drive[TASK_INPUT_DIM:TASK_INPUT_DIM + SENSORY_DIM] = frame.weighted
            if 'memory_ecology' in f:
                drive[TASK_INPUT_DIM + SENSORY_DIM:] = 0.5 * recall
            precision = float(controls['prediction_precision'])
            unc = [float(frame.uncertainty), float(pred_info['gate'])]
            if 'representation_fusion' in f:
                if learn and self.prev_drive is not None and self.prev_packet_state is not None:
                    ps = self.prev_packet_state
                    prev_fc = None if ps['forecast'] is None else np.asarray(ps['forecast'])
                    self.fusion.learn(self.prev_drive, self._packets(prev_fc, ps['unc'], ps['precision']), drive)
                    self.counters['fusion_updates'] += 1
                self.prev_drive = drive.copy()
                self.prev_packet_state = {'forecast': None if forecast_next is None else forecast_next.tolist(),
                                          'unc': unc, 'precision': precision}
                external = self.fusion.fuse(drive, self._packets(forecast_next, unc, precision))
            else:
                external = drive
        else:
            external = x_task

        # 8. brain step with the applied controls
        uncertainty = 0.0
        if 'uncertainty_gating' in f and pred_info['active']:
            uncertainty = (1.0 / max(pred_info['gate'], 1e-6) - 1.0) / 2.2
        route_gain = self.route_gain_matrix(controls) if 'multi_region_routing' in f else None
        brain_controls = None
        if self._mod_on:
            brain_controls = {k: float(controls[k]) for k in ('gain', 'threshold_shift', 'attention',
                                                              'plasticity_scale', 'learning_rate_scale')}
        eff_reward = reward
        if self._mod_on:
            eff_reward *= float(controls['reward_scale'])
        if 'efference_self_model' in f:
            eff_reward *= 0.5 + 0.5 * self.authorship
        unit_gain = None
        if 'receptor_heterogeneity' in f and 'neuromodulator_field' in f:
            # receptor-weighted local traces -> per-unit excitability of the external drive
            g = self.neuromod.per_unit_gain(f)
            unit_gain = np.ones(self.brain.n)         # neurons born after construction keep unit gain
            m = min(unit_gain.size, g.size)
            unit_gain[:m] = g[:m]
        out = self.brain.step(external, reward=eff_reward, controls=brain_controls, learn=learn,
                              uncertainty=uncertainty, route_gain=route_gain, unit_gain=unit_gain)
        if learn and route_gain is not None and eff_reward != 0.0:
            # an explicit route gain bypasses the brain's own route learning; keep it learning
            lrs = float(np.clip(controls['learning_rate_scale'], 0.0, 4.0))
            self.brain.synapses.update_route_gain(eff_reward, self.brain.cells.output, lr=0.05 * lrs)
        if learn and reward != 0.0:
            self.brain.train_readout(1.0 if reward > 0 else 0.0)
        temperature = float(controls['decision_temperature']) if self._mod_on else 1.0
        prob = self.readout_probability(temperature)
        bus.publish('state', {'activity_mean': float(out.mean()), 'activity_std': float(out.std()), 'n': int(out.size),
                              'spike_rate': float(np.mean(self.brain.cells.spike != 0))}, step=step)

        # 9. memory writes (SENSORY observation, PREDICTED forecast)
        if 'memory_ecology' in f and learn:
            key = self.prev_frame if self.prev_frame is not None else value
            sid = self.memory.store(key, value, provenance=Provenance.SENSORY, reward=float(np.clip(reward, -1, 1)),
                                    confidence=float(np.clip(1.0 - frame.uncertainty, 0.05, 1.0)))
            self.counters['memory_sensory'] += 1
            mem_info['stored_sensory'] = int(sid)
            if forecast_next is not None:
                pid = self.memory.store(value, forecast_next, provenance=Provenance.PREDICTED, confidence=0.5)
                self.pending_predicted_id = int(pid)
                self.counters['memory_predicted'] += 1
                mem_info['stored_predicted'] = int(pid)
            self.memory.tick(1)

        # 10. bookkeeping, sleep-time predictive replay and consolidation
        self.prev_frame = value.copy()
        self.prev_forecast = None if micro_forecast is None else micro_forecast.copy()
        if issued is not None:
            self.last_action = issued.copy()
        self.step_count += 1
        replayed = 0
        if learn and self.step_count % self.spec.sleep_interval == 0:
            replayed = self._sleep(controls)

        result = {
            'step': step, 'output': out, 'n': int(out.size), 'readout_probability': prob,
            'sensory': {'value': value.tolist(), 'reliability': frame.reliability.tolist(),
                        'uncertainty': float(frame.uncertainty), 'missing': int(frame.missing.sum()),
                        'imputed': sorted(frame.imputations), 'mean_prediction_error': float(np.mean(frame.prediction_error)),
                        'fed_to_brain': 'multimodal_sensory' in f},
            'controls': {**{k: float(controls[k]) for k in _SCALAR_CONTROLS},
                         'sensory_weight': np.asarray(controls['sensory_weight'], dtype=float).tolist(),
                         'applied_to_brain': brain_controls is not None},
            'uncertainty': float(uncertainty), 'route_gain_applied': route_gain is not None,
            'prediction': pred_info, 'micro': micro_info, 'memory': mem_info, 'efference': efference,
            'counterfactual': counterfactual, 'replayed': replayed, 'effective_reward': float(eff_reward),
            'events': int(bus.sequence), 'evidence_tier': EVIDENCE_TIER,
        }
        self.last = {k: v for k, v in result.items() if k != 'output'}
        return result

    def _sleep(self, controls: Mapping[str, Any]) -> int:
        """Sleep-time memory work: predictive replay of the replay schedule and bounded downscaling."""
        f = self.flags
        if 'memory_ecology' not in f:
            return 0
        cons = float(controls['consolidation_scale'])
        replayed = 0
        if 'predictive_replay' in f:
            n = int(np.clip(round(4 * cons), 1, 16))
            for item in self.memory.replay_schedule(n, verified_only=True):
                key, val = np.asarray(item['key']), np.asarray(item['value'])
                if self._prediction_on and not self.prediction.action_dim:
                    self.prediction.update(key, val, horizon=self.prediction.horizons[0])
                if 'micro_collective' in f:
                    self.micro.train_step(key, val)
                replayed += 1
            self.counters['replayed'] += replayed
            self.counters['replay_cycles'] += 1
            self.events.publish('memory', {'operation': 'replay', 'hits': replayed}, step=self.step_count)
        if 'sleep_consolidation' in f:
            self.memory.downscale(float(np.clip(1.0 - 0.02 / max(cons, 0.25), 0.9, 1.0)))
        return replayed

    # ------------------------------------------------------------------ specialist / ecology
    def set_specialist_coupling(self, coupling: float) -> None:
        if self.graft is None:
            raise ValueError('no specialist graft attached')
        self.graft.set_coupling(coupling)

    def detach_specialist(self) -> dict[str, Any] | None:
        if self.graft is None:
            return None
        receipt = self.graft.detach()
        self.graft = None
        return receipt

    def attach_ecology(self, config: str = 'default') -> dict[str, Any]:
        from .ecology import NeuroWeaveEcology
        self.ecology = NeuroWeaveEcology(config, seed=self.spec.seed)
        return self.ecology.summary()

    def action_from_output(self, out: np.ndarray, controls: Mapping[str, Any] | None = None) -> np.ndarray:
        """[thrust, turn] from motor/premotor activity (pure)."""
        out = np.asarray(out, dtype=np.float64)
        region = self.brain.cells.region[:out.size]   # neurogenesis may append units after the step
        motor = float(out[region == _MOTOR].mean()) if np.any(region == _MOTOR) else 0.0
        premotor = float(out[region == _PREMOTOR].mean()) if np.any(region == _PREMOTOR) else 0.0
        return np.array([math.tanh(2.0 * motor), math.tanh(2.0 * (premotor - motor))])

    def run_ecology(self, steps: int, *, learn: bool = True, config: str | None = None) -> dict[str, Any]:
        """Closed loop: ecology sensory vector -> engine -> motor action -> ecology (bounded steps)."""
        if type(steps) is not int or not 1 <= steps <= 100_000:
            raise ValueError('steps must be an int in [1, 100000]')
        if self.ecology is None or (config is not None and config != self.ecology.config):
            name = config if config is not None else self.spec.ecology
            self.attach_ecology(name if name in _ecology_names() else 'default')
        eco = self.ecology
        reward, total = 0.0, 0.0
        intended = issued = observed = None
        prev = np.zeros(ACTION_DIM)
        probs = []
        for _ in range(steps):
            res = self.step(None, eco.sensory_vector(0), reward=reward, learn=learn,
                            intended_action=intended, issued_action=issued, observed_outcome=observed)
            probs.append(res['readout_probability'])
            c = res['controls']
            intended = self.action_from_output(res['output'])
            keep = float(np.clip(0.3 * c['persistence'], 0.0, 0.9))
            noise = self._rng.normal(0.0, 0.1 * c['exploration'], ACTION_DIM)
            issued = np.clip(keep * prev + (1.0 - keep) * intended + noise, -1.0, 1.0)
            prev = issued
            obs, rewards, done, _info = eco.step(issued[None, :])
            observed = np.array([obs[0]['speed'], obs[0]['turn']])
            reward = float(rewards[0])
            total += reward
            self.counters['ecology_steps'] += 1
            if done:
                eco.reset()
                self.end_episode()
                intended = issued = observed = None
                prev = np.zeros(ACTION_DIM)
        summary = eco.summary()
        return {'steps': steps, 'total_reward': total, 'mean_readout_probability': float(np.mean(probs)),
                'food_eaten': summary['food_eaten'], 'collisions': summary['collisions'],
                'ecology': summary, 'evidence_tier': EVIDENCE_TIER}

    # ------------------------------------------------------------------ inspection
    def _region_activity(self) -> list[dict[str, Any]]:
        c = self.brain.cells
        out = []
        for r, (name, kind) in enumerate(REGIONS):
            m = c.region == r
            out.append({'region': name, 'kind': kind, 'neurons': int(m.sum()),
                        'mean': float(c.output[m].mean()) if m.any() else 0.0,
                        'abs_mean': float(np.abs(c.output[m]).mean()) if m.any() else 0.0})
        return out

    def observatory(self) -> dict[str, Any]:
        b, c, syn, f = self.brain, self.brain.cells, self.brain.synapses, self.flags
        ledger = b.ledger
        neural = {
            'n': int(b.n), 'step': int(b.step_count), 'region_activity': self._region_activity(),
            'soma_means': {s: float(np.mean(getattr(c, s))) for s in SOMA_ROLES if hasattr(c, s)},
            'dendrite_means': {d: float(np.mean(getattr(c, d))) for d in DENDRITES if hasattr(c, d)},
            'axon': {k: float(np.mean(getattr(c, k))) for k in ('branch_a', 'branch_b', 'terminal', 'release', 'myelin')
                     if hasattr(c, k)},
            'synapses': {'count': int(syn.size), 'w_mean': float(syn.w.mean()) if syn.size else 0.0,
                         'w_abs_mean': float(np.abs(syn.w).mean()) if syn.size else 0.0,
                         'excitatory': int(np.sum(syn.w > 0)), 'inhibitory': int(np.sum(syn.w < 0)),
                         'delay_hist': np.bincount(syn.delay, minlength=4)[1:4].astype(int).tolist() if syn.size else [0, 0, 0],
                         'utility_mean': float(syn.utility.mean()) if syn.size else 0.0},
            'ledger': {'total': int(ledger.total), 'by_kind': dict(ledger.counts_by_kind),
                       'recent': [dict(e) for e in list(ledger.events)[-8:]]},
            'counters': dict(b.counters), 'profile_estimated': dict(b.estimated),
        }
        nm = self.neuromod
        rec = nm.receptor_matrix(f)
        controls = nm.controls(f)
        neuromodulation = {
            'active': self._mod_on, 'levels': nm.level_dict, 'affect': nm.affect.to_dict(),
            'receptors': {name: {'mean': float(rec[:, k].mean()), 'std': float(rec[:, k].std())}
                          for k, name in enumerate(NEUROMODULATORS)},
            'controls': {k: float(controls[k]) for k in _SCALAR_CONTROLS},
            'sensory_weight': np.asarray(controls['sensory_weight'], dtype=float).tolist(),
        }
        if 'transmitter_inference' in f and syn.size:
            ti = infer_transmitters(syn.src, syn.dst, syn.w, c.sign, n=b.n)
            preds = list(ti['predictions'])
            neuromodulation['transmitter_inference'] = {
                'labels': list(ti['labels']), 'counts': {lab: preds.count(lab) for lab in ti['labels']},
                'mean_confidence': float(np.mean(ti['confidence'])),
                'sign_consistent_fraction': float(np.mean(ti['sign_consistent'])), 'evidence': ti['evidence']}
        last_s = self.last.get('sensory', {})
        sensory = {'manifest': self.sensory.manifest(), 'last_values': last_s.get('value'),
                   'last_reliability': last_s.get('reliability'),
                   'last_mean_prediction_error': last_s.get('mean_prediction_error'),
                   'imputed': last_s.get('imputed', [])}
        micro = self.micro
        learning = {
            'micro': {'active': 'micro_collective' in f, 'experts': micro.n_experts, 'load': micro.expert_load,
                      'last_validation_loss': micro.last_validation_loss, 'best_validation_loss': micro.best_loss,
                      'lr': micro.lr, 'rollbacks': micro.rollbacks,
                      'receipts': [dict(r) for r in micro.receipts[-4:]]},
            'specialist': self.graft.receipt() if self.graft is not None else None,
            'plasticity': {'lr_scale_mean': float(syn.lr_scale.mean()) if syn.size else 0.0,
                           'updates_mean': float(syn.updates.mean()) if syn.size else 0.0,
                           'eligibility_abs_mean': float(np.abs(syn.eligibility).mean()) if syn.size else 0.0,
                           'route_gain_mean': float(syn.route_gain.mean())},
            'structural_changes': dict(ledger.counts_by_kind),
            'fusion_weights': dict(self.fusion.weights), 'counters': dict(self.counters),
        }
        out = {'schema': 'NeuroWeaveObservatoryV1', 'evidence_tier': EVIDENCE_TIER, 'step': self.step_count,
               'mechanisms': list(self.spec.mechanisms), 'neural': neural, 'neuromodulation': neuromodulation,
               'sensory': sensory, 'learning': learning, 'memory': self.memory.stats(),
               'prediction': {'active': self._prediction_on, 'horizons': list(self.prediction.horizons),
                              'metrics': self.prediction.metrics(),
                              'efference': {'control_reliability': self.efference.control_reliability,
                                            'records': self.efference.records}},
               'events': self.events.counters()}
        if self.ecology is not None:
            out['environment'] = self.ecology.summary()
        return out

    def telemetry(self, include_timing: bool = False) -> dict[str, Any]:
        b = self.brain
        out = b.cells.output
        t = {
            'schema': 'NeuroWeaveTelemetryV1', 'evidence_tier': EVIDENCE_TIER, 'step': self.step_count,
            'n': int(b.n), 'synapses': int(b.synapses.size), 'mechanisms': len(self.spec.mechanisms),
            'activity_mean': float(out.mean()), 'activity_std': float(out.std()),
            'readout_probability': float(self.last.get('readout_probability', 0.0)),
            'levels': self.neuromod.level_dict,
            'prediction_surprise': float(self.last.get('prediction', {}).get('surprise', 0.0)),
            'uncertainty_gate': float(self.last.get('prediction', {}).get('gate', 1.0)),
            'micro_validation_loss': self.micro.last_validation_loss,
            'memory_items': len(self.memory.items), 'events': int(self.events.sequence),
            'structural_events': int(b.ledger.total), 'counters': dict(self.counters),
            'ops_estimated_last': int(b.estimated['ops_last']), 'raw_media_stored': False,
        }
        if include_timing:
            t['wallclock_ms_last'] = float(b.measured['wallclock_ms_last'])
        return t

    # ------------------------------------------------------------------ hardware / package
    def fpga_manifest(self) -> dict[str, Any]:
        from .hardware import fpga_mapping_manifest, validate_fpga_manifest
        m = fpga_mapping_manifest(self.brain.export_state(), tile_size=self.spec.tile_size)
        m['independent_validation'] = validate_fpga_manifest(m)
        return m

    def component_states(self, components: Iterable[str] | None = None) -> dict[str, Any]:
        avail = {
            'neuromod': lambda: self.neuromod.export_state(),
            'sensory': lambda: self.sensory.export_state(),
            'micro_collective': lambda: self.micro.export_state(),
            'memory': lambda: self.memory.export_state(),
            'prediction': lambda: {'hierarchy': self.prediction.export_state(),
                                   'efference': self.efference.export_state()},
        }
        if self.ecology is not None:
            avail['environment'] = lambda: self.ecology.export_state()
        names = list(avail) if components is None else list(components)
        unknown = [c for c in names if c not in avail]
        if unknown:
            raise ValueError(f'unknown or unavailable components: {unknown}')
        return {c: avail[c]() for c in names}

    def export_package(self, *components: str) -> dict[str, Any]:
        from .package import build_package
        comps = self.component_states(components or None)
        return build_package(self.brain.export_state(), components=comps, spec=self.spec,
                             provenance={'engine_step': int(self.step_count)})

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        return {
            'schema': ENGINE_SCHEMA, 'schema_version': 1, 'evidence_tier': EVIDENCE_TIER,
            'spec': self.spec.to_dict(),
            'brain': self.brain.export_state(), 'neuromod': self.neuromod.export_state(),
            'sensory': self.sensory.export_state(), 'memory': self.memory.export_state(),
            'prediction': self.prediction.export_state(), 'efference': self.efference.export_state(),
            'micro': self.micro.export_state(),
            'graft': None if self.graft is None else self.graft.export_state(),
            'fusion': self.fusion.export_state(), 'events': self.events.export_state(),
            'ecology': None if self.ecology is None else self.ecology.export_state(),
            'runtime': {
                'step': int(self.step_count),
                'prev_frame': None if self.prev_frame is None else self.prev_frame.tolist(),
                'prev_forecast': None if self.prev_forecast is None else self.prev_forecast.tolist(),
                'prev_drive': None if self.prev_drive is None else self.prev_drive.tolist(),
                'prev_packet_state': None if self.prev_packet_state is None else dict(self.prev_packet_state),
                'pending_predicted_id': self.pending_predicted_id,
                'last_action': self.last_action.tolist(), 'authorship': float(self.authorship),
                'counters': dict(self.counters), 'rng': _rng_state(self._rng),
            },
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Rebuild every subsystem from ``state`` in a scratch engine, then swap (no partial mutation)."""
        if not isinstance(state, Mapping) or state.get('schema') != ENGINE_SCHEMA or state.get('schema_version') != 1:
            raise ValueError('not a NeuroWeaveEngineStateV1 state')
        try:
            spec = NeuroWeaveSpecV1.from_dict(state['spec'])
            if spec != self.spec:
                raise ValueError('engine state spec does not match this engine')
            fresh = NeuroWeaveEngine(spec)
            fresh.brain.import_state(state['brain'])
            fresh.neuromod.import_state(state['neuromod'])
            fresh.sensory.import_state(state['sensory'])
            fresh.memory.import_state(state['memory'])
            fresh.prediction.import_state(state['prediction'])
            fresh.efference.import_state(state['efference'])
            if fresh.graft is not None:
                fresh.micro.detach_specialist()
            fresh.micro.import_state(state['micro'])
            if state['graft'] is not None:
                if fresh.graft is None:
                    raise ValueError('state has a specialist graft but the spec does not enable it')
                fresh.graft.import_state(state['graft'])
                if fresh.graft.attached:
                    fresh.micro.attach_specialist(fresh.graft)
                else:
                    fresh.graft = None
            else:
                fresh.graft = None
            fresh.fusion.import_state(state['fusion'])
            fresh.events.import_state(state['events'])
            if state['ecology'] is not None:
                from .ecology import NeuroWeaveEcology
                eco_state = state['ecology']
                eco = NeuroWeaveEcology(eco_state.get('config', 'default') if isinstance(eco_state, Mapping) else 'default',
                                        seed=spec.seed)
                eco.import_state(eco_state)
                fresh.ecology = eco
            rt = state['runtime']
            if not isinstance(rt, Mapping):
                raise ValueError('runtime must be a mapping')

            def opt_vec(key: str, n: int) -> np.ndarray | None:
                v = rt.get(key)
                if v is None:
                    return None
                a = _finite_vec(key, v, max_len=n)
                if a.size != n:
                    raise ValueError(f'runtime.{key} must have length {n}')
                return a

            step = rt['step']
            if type(step) is not int or step < 0:
                raise ValueError('runtime.step must be a non-negative int')
            fresh.step_count = step
            fresh.prev_frame = opt_vec('prev_frame', SENSORY_DIM)
            fresh.prev_forecast = opt_vec('prev_forecast', SENSORY_DIM)
            fresh.prev_drive = opt_vec('prev_drive', DRIVE_DIM)
            ps = rt.get('prev_packet_state')
            if ps is not None:
                if not isinstance(ps, Mapping) or set(ps) != {'forecast', 'unc', 'precision'}:
                    raise ValueError('runtime.prev_packet_state malformed')
                fc = None if ps['forecast'] is None else _finite_vec('forecast', ps['forecast'], max_len=SENSORY_DIM).tolist()
                ps = {'forecast': fc, 'unc': _finite_vec('unc', ps['unc'], max_len=2).tolist(),
                      'precision': _finite_scalar('runtime.prev_packet_state.precision', ps['precision'])}
            fresh.prev_packet_state = ps
            pid = rt.get('pending_predicted_id')
            if pid is not None and (type(pid) is not int or pid < 0):
                raise ValueError('runtime.pending_predicted_id invalid')
            fresh.pending_predicted_id = pid
            la = opt_vec('last_action', ACTION_DIM)
            fresh.last_action = np.zeros(ACTION_DIM) if la is None else la
            auth = _finite_scalar('runtime.authorship', rt['authorship'])
            if not 0.0 <= auth <= 1.0:
                raise ValueError('runtime.authorship out of range')
            fresh.authorship = auth
            counters = rt['counters']
            if not isinstance(counters, Mapping) or set(counters) != set(fresh.counters) or any(
                    type(v) is not int or v < 0 for v in counters.values()):
                raise ValueError('runtime.counters malformed')
            fresh.counters = {k: int(v) for k, v in counters.items()}
            fresh._rng.bit_generator.state = rt['rng']
        except ValueError:
            raise
        except (TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
            raise ValueError(f'malformed engine state: {type(exc).__name__}: {exc}') from exc
        keep_subscribers = self.events
        self.__dict__.update(fresh.__dict__)
        # runtime subscribers are kept (EventBus.import_state semantics)
        keep_subscribers.import_state(state['events'])
        self.events = keep_subscribers
        self.last = {}


def _ecology_names() -> tuple[str, ...]:
    from .ecology import CONFIG_NAMES
    return CONFIG_NAMES
