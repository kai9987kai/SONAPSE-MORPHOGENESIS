"""Micro-Collective: a small top-k mixture-of-experts next-sensory predictor.

Re-implements the NEUROWEAVE 0.4.0 Micro-Collective mechanism in NumPy:

* a softmax router with top-k expert selection and an optional
  load-balancing bias that nudges under-used experts upward
  (``balanced=True``, mapped from the ``expert_routing`` mechanism flag);
* per-expert one-hidden-layer tanh MLPs trained online by manual backprop
  (plain SGD, global-norm gradient clipping, bounded weights) on a
  self-supervised next-sensory prediction target;
* a bounded replay buffer with novelty-prioritised sampling
  (``prioritized_replay``, mapped from ``predictive_replay``);
* a held-out validation split (every ``validation_every``-th sample is never
  trained on), a best-state snapshot and automatic rollback when validation
  loss worsens beyond a margin (``rollback``, mapped from ``meta_plasticity``);
* SHA-256 training receipts that report the model's validation loss next to
  the persistence baseline (predict ``x_{t+1} = x_t``) honestly.

``SpecialistGraft`` is a separate, independently trained GRU-like recurrent
predictor that can exchange predictions with the collective. Its contribution
is multiplied by ``coupling`` which starts at ``0.0`` (zero-initialised graft)
and is bounded to ``[0, 0.5]``.

This is a synthetic computational abstraction; it makes no claim of
biological fidelity.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .spec import EVIDENCE_TIER, NeuroWeaveSpecV1

MICRO_SCHEMA = 'NeuroWeaveMicroCollectiveV1'
GRAFT_SCHEMA = 'NeuroWeaveSpecialistGraftV1'
RECEIPT_SCHEMA = 'NeuroWeaveMicroReceiptV1'
GRAFT_RECEIPT_SCHEMA = 'NeuroWeaveSpecialistReceiptV1'
SCHEMA_VERSION = 1

MAX_EXPERTS = 64
MAX_DIM = 4096
INPUT_BOUND = 10.0
OUTPUT_BOUND = 10.0
WEIGHT_BOUND = 5.0
ROUTER_BIAS_BOUND = 2.0
MAX_COUPLING = 0.5
MAX_RECEIPTS = 32
MIN_LR = 1e-5

MICRO_SEED_OFFSET = 12031
GRAFT_SEED_OFFSET = 12071


# ----------------------------------------------------------------------------- helpers
def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return int(value)


def _check_float(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be a finite number in [{low}, {high}]')
    return value


def _check_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value


def _vector(value: Any, dim: int, name: str, bound: float = INPUT_BOUND) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if arr.shape != (dim,):
        raise ValueError(f'{name} must have shape ({dim},), got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    return np.clip(arr, -bound, bound)


def _array(state: Mapping[str, Any], key: str, shape: tuple[int, ...], bound: float | None = None) -> np.ndarray:
    if key not in state:
        raise ValueError(f'missing field {key!r}')
    try:
        arr = np.asarray(state[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{key} must be numeric') from exc
    if arr.shape != shape:
        raise ValueError(f'{key} must have shape {shape}, got {arr.shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{key} must be finite')
    if bound is not None and np.any(np.abs(arr) > bound + 1e-12):
        raise ValueError(f'{key} exceeds bound {bound}')
    return arr.copy()


def _opt_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _check_float(name, value, 0.0, 1e12)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _clip_global(grads: list[np.ndarray], max_norm: float) -> float:
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
    if norm > max_norm and norm > 0.0:
        scale = max_norm / norm
        for g in grads:
            g *= scale
    return norm


def _digest(arrays: list[np.ndarray], meta: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        h.update(np.ascontiguousarray(arr, dtype='<f8').tobytes())
    h.update(json.dumps(meta, sort_keys=True).encode('utf-8'))
    return h.hexdigest()


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    state = rng.bit_generator.state
    return {'bit_generator': str(state['bit_generator']),
            'state': {'state': int(state['state']['state']), 'inc': int(state['state']['inc'])},
            'has_uint32': int(state['has_uint32']), 'uinteger': int(state['uinteger'])}


def _rng_from_state(data: Any) -> np.random.Generator:
    if not isinstance(data, Mapping) or data.get('bit_generator') != 'PCG64':
        raise ValueError('rng state must be a PCG64 state mapping')
    rng = np.random.Generator(np.random.PCG64())
    try:
        rng.bit_generator.state = {'bit_generator': 'PCG64',
                                   'state': {'state': int(data['state']['state']), 'inc': int(data['state']['inc'])},
                                   'has_uint32': int(data['has_uint32']), 'uinteger': int(data['uinteger'])}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid rng state') from exc
    return rng


def _parse_items(raw: Any, in_dim: int, out_dim: int, capacity: int, name: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > capacity:
        raise ValueError(f'{name} must be a list of at most {capacity} items')
    items = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError(f'{name} items must be mappings')
        x = _array(item, 'x', (in_dim,), INPUT_BOUND)
        y = _array(item, 'y', (out_dim,), INPUT_BOUND)
        novelty = _check_float(f'{name}.novelty', item.get('novelty'), 0.0, 1e6)
        items.append({'x': x, 'y': y, 'novelty': novelty})
    return items


def _items_json(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{'x': it['x'].tolist(), 'y': it['y'].tolist(), 'novelty': float(it['novelty'])} for it in items]


# ============================================================================ MicroCollective
class MicroCollective:
    """Top-k mixture of tanh-MLP experts predicting the next sensory vector."""

    def __init__(self, input_dim: int = 16, output_dim: int = 16, experts: int = 4, top_k: int = 2,
                 hidden: int = 24, seed: int = 7, *, lr: float = 0.1, balanced: bool = True,
                 prioritized_replay: bool = True, rollback: bool = True, replay_capacity: int = 256,
                 validation_capacity: int = 64, validation_every: int = 5, replay_batch: int = 8,
                 eval_interval: int = 10, rollback_margin: float = 0.25, rollback_tolerance: float = 1e-3, grad_clip: float = 1.0,
                 balance_rate: float = 0.01, load_ema: float = 0.05, router_lr_scale: float = 0.5,
                 min_validation: int = 4) -> None:
        self.input_dim = _check_int('input_dim', input_dim, 1, MAX_DIM)
        self.output_dim = _check_int('output_dim', output_dim, 1, MAX_DIM)
        self.hidden = _check_int('hidden', hidden, 1, MAX_DIM)
        n_exp = _check_int('experts', experts, 1, MAX_EXPERTS)
        self.top_k = _check_int('top_k', top_k, 1, n_exp)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.lr = _check_float('lr', lr, MIN_LR, 1.0)
        self.base_lr = self.lr
        self.balanced = _check_bool('balanced', balanced)
        self.prioritized_replay = _check_bool('prioritized_replay', prioritized_replay)
        self.rollback_enabled = _check_bool('rollback', rollback)
        self.replay_capacity = _check_int('replay_capacity', replay_capacity, 1, 1_000_000)
        self.validation_capacity = _check_int('validation_capacity', validation_capacity, 1, 1_000_000)
        self.validation_every = _check_int('validation_every', validation_every, 2, 1_000_000)
        self.replay_batch = _check_int('replay_batch', replay_batch, 0, 4096)
        self.eval_interval = _check_int('eval_interval', eval_interval, 1, 1_000_000)
        self.rollback_margin = _check_float('rollback_margin', rollback_margin, 0.0, 100.0)
        self.rollback_tolerance = _check_float('rollback_tolerance', rollback_tolerance, 0.0, 1e6)
        self.grad_clip = _check_float('grad_clip', grad_clip, 1e-6, 1e6)
        self.balance_rate = _check_float('balance_rate', balance_rate, 0.0, 1.0)
        self.load_ema_rate = _check_float('load_ema', load_ema, 1e-6, 1.0)
        self.router_lr_scale = _check_float('router_lr_scale', router_lr_scale, 0.0, 10.0)
        self.min_validation = _check_int('min_validation', min_validation, 1, 1_000_000)

        self._init_rng = np.random.default_rng(self.seed + MICRO_SEED_OFFSET)
        self._rng = np.random.default_rng(self.seed + MICRO_SEED_OFFSET + 1)
        D, H, O = self.input_dim, self.hidden, self.output_dim
        self.w1 = np.zeros((0, H, D)); self.b1 = np.zeros((0, H))
        self.w2 = np.zeros((0, O, H)); self.b2 = np.zeros((0, O))
        self.router_w = np.zeros((0, D)); self.router_b = np.zeros(0)
        self.balance_bias = np.zeros(0)
        self.load_ema = np.zeros(0)
        self.load_counts = np.zeros(0, dtype=np.int64)
        for _ in range(n_exp):
            self._append_expert()
        self.load_ema[:] = self.top_k / self.n_experts

        self.replay: list[dict[str, Any]] = []
        self.validation: list[dict[str, Any]] = []
        self.samples = 0
        self.updates = 0
        self.sgd_steps = 0
        self.evaluations = 0
        self.rollbacks = 0
        self.experts_added = 0
        self.best: dict[str, np.ndarray] | None = None
        self.best_loss: float | None = None
        self.last_validation_loss: float | None = None
        self.receipts: list[dict[str, Any]] = []
        self._specialist: SpecialistGraft | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_spec(cls, spec: NeuroWeaveSpecV1, **kwargs: Any) -> 'MicroCollective':
        """Map spec flags: expert_routing->balanced, predictive_replay->prioritised replay,
        meta_plasticity->rollback. Absent flags give the neutral (off) behaviour."""
        params = dict(experts=spec.micro_experts, top_k=spec.micro_top_k, seed=spec.seed,
                      balanced=spec.has('expert_routing'),
                      prioritized_replay=spec.has('predictive_replay'),
                      rollback=spec.has('meta_plasticity'))
        params.update(kwargs)
        return cls(**params)

    @property
    def n_experts(self) -> int:
        return int(self.w1.shape[0])

    def _append_expert(self, grown: bool = False) -> int:
        rng = self._init_rng
        D, H, O = self.input_dim, self.hidden, self.output_dim
        self.w1 = np.concatenate([self.w1, rng.normal(0.0, 1.0 / math.sqrt(D), (1, H, D))])
        self.b1 = np.concatenate([self.b1, np.zeros((1, H))])
        self.w2 = np.concatenate([self.w2, rng.normal(0.0, 0.5 / math.sqrt(H), (1, O, H))])
        self.b2 = np.concatenate([self.b2, np.zeros((1, O))])
        self.router_w = np.concatenate([self.router_w, rng.normal(0.0, 0.3 / math.sqrt(D), (1, D))])
        # Initial experts get diverse random router preferences; a grown expert
        # starts at the mean preference so it competes on equal footing.
        new_bias = float(np.mean(self.router_b)) if grown else float(rng.normal(0.0, 0.75))
        self.router_b = np.concatenate([self.router_b, [np.clip(new_bias, -ROUTER_BIAS_BOUND, ROUTER_BIAS_BOUND)]])
        start_balance = float(np.mean(self.balance_bias)) if self.balance_bias.size else 0.0
        self.balance_bias = np.concatenate([self.balance_bias, [start_balance]])
        self.load_ema = np.concatenate([self.load_ema, [0.0]])
        self.load_counts = np.concatenate([self.load_counts, np.zeros(1, dtype=np.int64)])
        np.clip(self.w1, -WEIGHT_BOUND, WEIGHT_BOUND, out=self.w1)
        np.clip(self.w2, -WEIGHT_BOUND, WEIGHT_BOUND, out=self.w2)
        return self.n_experts - 1

    def add_expert(self) -> int:
        """Grow by one freshly initialised expert. Returns its index.

        The best-state snapshot has the old shape, so it is invalidated; the
        next validation establishes a new best."""
        if self.n_experts >= MAX_EXPERTS:
            raise ValueError(f'cannot exceed {MAX_EXPERTS} experts')
        idx = self._append_expert(grown=True)
        target = self.top_k / self.n_experts
        self.load_ema[:] = np.clip(self.load_ema * (self.n_experts - 1) / self.n_experts, 0.0, 1.0)
        self.load_ema[idx] = target
        self.best = None
        self.best_loss = None
        self.experts_added += 1
        return idx

    # ------------------------------------------------------------------ forward
    def _params(self) -> dict[str, np.ndarray]:
        return {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2,
                'router_w': self.router_w, 'router_b': self.router_b, 'balance_bias': self.balance_bias}

    def _forward_batch(self, X: np.ndarray, p: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
        """Batched forward pass. ``X`` has shape (B, input_dim).

        Every expert is evaluated (cheap at this scale) but only the top-k
        routed experts receive non-zero gates, so the output and all gradients
        are exactly those of sparse top-k routing."""
        p = self._params() if p is None else {**self._params(), **p}
        B = X.shape[0]
        logits = X @ p['router_w'].T + p['router_b'] + p['balance_bias']
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        probs = ez / ez.sum(axis=1, keepdims=True)
        chosen = np.argsort(-probs, axis=1, kind='stable')[:, : self.top_k]
        rows = np.arange(B)[:, None]
        gates = np.zeros_like(probs)
        sel = probs[rows, chosen]
        gates[rows, chosen] = sel / sel.sum(axis=1, keepdims=True)
        hid = np.tanh(np.einsum('ehd,bd->beh', p['w1'], X) + p['b1'][None])
        outs = np.einsum('eoh,beh->beo', p['w2'], hid) + p['b2'][None]
        y = np.einsum('be,beo->bo', gates, outs)
        return {'y': y, 'chosen': chosen, 'gates': gates, 'probs': probs, 'h': hid, 'o': outs}

    def _predict_batch(self, X: np.ndarray, p: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        return np.clip(self._forward_batch(X, p)['y'], -OUTPUT_BOUND, OUTPUT_BOUND)

    def _predict_main(self, x: np.ndarray, p: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        return self._predict_batch(x[None, :], p)[0]

    def predict(self, x: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict the next sensory vector. Pure: does not change any state."""
        xv = _vector(x, self.input_dim, 'x')
        f = self._forward_batch(xv[None, :])
        y = np.clip(f['y'][0], -OUTPUT_BOUND, OUTPUT_BOUND)
        chosen = f['chosen'][0]
        routing = {'experts': [int(e) for e in chosen],
                   'weights': [float(f['gates'][0, e]) for e in chosen],
                   'probabilities': [float(v) for v in f['probs'][0]],
                   'specialist_coupling': 0.0}
        if self._specialist is not None:
            contrib = self._specialist.contribute(xv)
            routing['specialist_coupling'] = float(self._specialist.coupling)
            y = np.clip(y + contrib, -OUTPUT_BOUND, OUTPUT_BOUND)
        return y, routing

    # ------------------------------------------------------------------ learning
    def _sgd(self, X: np.ndarray, T: np.ndarray, track_rows: int = 0) -> float:
        """One minibatch SGD step (manual backprop, mean-squared error).

        ``track_rows`` leading rows count toward the expert-load statistics."""
        B = X.shape[0]
        f = self._forward_batch(X)
        err = f['y'] - T                                     # (B, O)
        loss = float(np.mean(err * err))
        G, hid, outs = f['gates'], f['h'], f['o']
        d_o = G[:, :, None] * err[:, None, :] / B            # (B, E, O)
        dw2 = np.einsum('beo,beh->eoh', d_o, hid)
        db2 = d_o.sum(axis=0)
        dh = np.einsum('beo,eoh->beh', d_o, self.w2) * (1.0 - hid * hid)
        dw1 = np.einsum('beh,bd->ehd', dh, X)
        db1 = dh.sum(axis=0)
        dldg = np.einsum('bo,beo->be', err, outs)            # (B, E)
        dz = G * (dldg - np.sum(G * dldg, axis=1, keepdims=True)) / B   # zero for unrouted experts
        drw = dz.T @ X
        drb = dz.sum(axis=0)
        _clip_global([dw1, db1, dw2, db2, drw, drb], self.grad_clip)
        lr = self.lr
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        rlr = lr * self.router_lr_scale
        self.router_w -= rlr * drw
        self.router_b -= rlr * drb
        for arr in (self.w1, self.b1, self.w2, self.b2, self.router_w):
            np.clip(arr, -WEIGHT_BOUND, WEIGHT_BOUND, out=arr)
        np.clip(self.router_b, -ROUTER_BIAS_BOUND, ROUTER_BIAS_BOUND, out=self.router_b)
        for r in range(min(track_rows, B)):
            onehot = np.zeros(self.n_experts)
            onehot[f['chosen'][r]] = 1.0
            self.load_counts[f['chosen'][r]] += 1
            self.load_ema += self.load_ema_rate * (onehot - self.load_ema)
        self.sgd_steps += 1
        return loss

    def sample_replay(self, n: int, *, exclude_last: bool = False) -> list[int]:
        """Indices into ``self.replay`` (without replacement). Novelty-weighted when
        prioritised replay is on, uniform otherwise. Consumes the sampling RNG."""
        pool = len(self.replay) - (1 if exclude_last else 0)
        n = min(_check_int('n', n, 0, 1_000_000), max(pool, 0))
        if n == 0:
            return []
        if self.prioritized_replay:
            pr = np.array([it['novelty'] for it in self.replay[:pool]]) + 1e-6
            idx = self._rng.choice(pool, size=n, replace=False, p=pr / pr.sum())
        else:
            idx = self._rng.choice(pool, size=n, replace=False)
        return [int(i) for i in idx]

    def _apply_balance(self) -> None:
        if not self.balanced or self.balance_rate == 0.0:
            return
        target = self.top_k / self.n_experts
        self.balance_bias += self.balance_rate * (target - self.load_ema) / target
        np.clip(self.balance_bias, -ROUTER_BIAS_BOUND, ROUTER_BIAS_BOUND, out=self.balance_bias)

    def train_step(self, x: Any, y_next: Any) -> dict[str, Any]:
        """One self-supervised step on the transition ``x -> y_next``.

        Every ``validation_every``-th sample is held out (never trained on)."""
        xv = _vector(x, self.input_dim, 'x')
        tv = _vector(y_next, self.output_dim, 'y_next')
        self.samples += 1
        pred = self._predict_main(xv)
        novelty = float(np.mean((pred - tv) ** 2))
        item = {'x': xv, 'y': tv, 'novelty': novelty}
        out: dict[str, Any] = {'sample': self.samples, 'held_out': False, 'prediction_error': novelty,
                               'loss': None, 'replayed': 0, 'validation': None}
        if self.samples % self.validation_every == 0:
            self.validation.append(item)
            if len(self.validation) > self.validation_capacity:
                self.validation.pop(0)
            out['held_out'] = True
        else:
            self.replay.append(item)
            if len(self.replay) > self.replay_capacity:
                self.replay.pop(0)
            idx = self.sample_replay(self.replay_batch, exclude_last=True)
            X = np.stack([xv] + [self.replay[i]['x'] for i in idx])
            T = np.stack([tv] + [self.replay[i]['y'] for i in idx])
            out['loss'] = self._sgd(X, T, track_rows=1)
            if idx:
                # refresh replay priorities with the post-update error (prioritised replay)
                err = np.mean((self._predict_batch(X[1:]) - T[1:]) ** 2, axis=1)
                for i, e in zip(idx, err):
                    self.replay[i]['novelty'] = float(e)
            out['replayed'] = len(idx)
            self.updates += 1
            self._apply_balance()
            if self.updates % self.eval_interval == 0 and len(self.validation) >= self.min_validation:
                out['validation'] = self.validate()
        if self._specialist is not None:
            self._specialist.receive_main_prediction(pred)
            out['specialist'] = self._specialist.train_step(xv, tv)
        return out

    # ------------------------------------------------------------------ validation / rollback
    _SNAPSHOT_KEYS = ('w1', 'b1', 'w2', 'b2', 'router_w', 'router_b')

    def _copy_params(self) -> dict[str, np.ndarray]:
        # The load-balancing bias is a routing controller, not a learned weight:
        # it is excluded from snapshots so best/current comparisons share routing control.
        return {k: getattr(self, k).copy() for k in self._SNAPSHOT_KEYS}

    def evaluate(self, params: Mapping[str, np.ndarray] | None = None) -> float | None:
        """Mean squared error of the main predictor on the held-out buffer."""
        if not self.validation:
            return None
        X = np.stack([it['x'] for it in self.validation])
        T = np.stack([it['y'] for it in self.validation])
        return float(np.mean((self._predict_batch(X, params) - T) ** 2))

    def persistence_loss(self) -> float | None:
        """Held-out MSE of the persistence baseline ``y = x`` (None if dims differ)."""
        if not self.validation or self.input_dim != self.output_dim:
            return None
        return float(np.mean([np.mean((it['x'] - it['y']) ** 2) for it in self.validation]))

    def zero_baseline_loss(self) -> float | None:
        if not self.validation:
            return None
        return float(np.mean([np.mean(it['y'] ** 2) for it in self.validation]))

    def _restore(self, snap: Mapping[str, np.ndarray]) -> None:
        self.w1 = snap['w1'].copy(); self.b1 = snap['b1'].copy()
        self.w2 = snap['w2'].copy(); self.b2 = snap['b2'].copy()
        self.router_w = snap['router_w'].copy(); self.router_b = snap['router_b'].copy()

    def validate(self) -> dict[str, Any]:
        """Evaluate on held-out data; snapshot an improvement or roll back a regression.

        The best snapshot is re-scored on the *current* validation buffer so the
        comparison is like-for-like."""
        loss = self.evaluate()
        if loss is None:
            raise ValueError('no validation samples')
        self.evaluations += 1
        event = 'none'
        best_now = self.evaluate(self.best) if self.best is not None else None
        if best_now is None or loss < best_now:
            self.best = self._copy_params()
            self.best_loss = loss
            self.lr = min(self.base_lr, self.lr * 1.25)
            event = 'improved'
        else:
            self.best_loss = best_now
            if self.rollback_enabled and loss - best_now > max(self.rollback_margin * best_now, self.rollback_tolerance):
                self._restore(self.best)
                self.rollbacks += 1
                self.lr = max(self.base_lr / 16.0, self.lr * 0.5)
                event = 'rollback'
                loss = self.evaluate()
        self.last_validation_loss = loss
        receipt = self.receipt(event=event)
        self.receipts.append(receipt)
        if len(self.receipts) > MAX_RECEIPTS:
            self.receipts.pop(0)
        return {'validation_loss': loss, 'best_validation_loss': self.best_loss,
                'persistence_loss': receipt['persistence_loss'], 'event': event,
                'rolled_back': event == 'rollback'}

    # ------------------------------------------------------------------ stats / receipts
    @property
    def expert_load(self) -> dict[str, Any]:
        counts = self.load_counts.astype(np.float64)
        total = float(counts.sum())
        frac = counts / total if total > 0 else np.zeros_like(counts)
        mean = float(counts.mean()) if counts.size else 0.0
        cv = float(counts.std() / mean) if mean > 0 else 0.0
        return {'counts': [int(c) for c in self.load_counts], 'fraction': frac.tolist(),
                'ema': self.load_ema.tolist(), 'balance_bias': self.balance_bias.tolist(),
                'cv': cv, 'max_over_mean': float(counts.max() / mean) if mean > 0 else 0.0}

    def _counters(self) -> dict[str, Any]:
        return {'samples': self.samples, 'updates': self.updates, 'sgd_steps': self.sgd_steps,
                'evaluations': self.evaluations, 'rollbacks': self.rollbacks,
                'experts': self.n_experts, 'top_k': self.top_k, 'experts_added': self.experts_added,
                'lr': self.lr}

    def weights_digest(self) -> str:
        return _digest(list(self._params().values()), self._counters())

    def receipt(self, event: str = 'manual') -> dict[str, Any]:
        val = self.evaluate()
        pers = self.persistence_loss()
        return {'schema': RECEIPT_SCHEMA, 'evidence_tier': EVIDENCE_TIER, 'event': event,
                'sha256': self.weights_digest(), 'counters': self._counters(),
                'validation_loss': val, 'best_validation_loss': self.best_loss,
                'persistence_loss': pers, 'zero_baseline_loss': self.zero_baseline_loss(),
                'beats_persistence': (val is not None and pers is not None and val < pers),
                'validation_samples': len(self.validation), 'replay_size': len(self.replay),
                'expert_load': self.expert_load,
                'specialist': self._specialist.receipt() if self._specialist is not None else None}

    @property
    def parameter_count(self) -> int:
        return int(sum(v.size for k, v in self._params().items() if k != 'balance_bias'))

    # ------------------------------------------------------------------ specialist exchange
    def attach_specialist(self, graft: 'SpecialistGraft') -> None:
        if not isinstance(graft, SpecialistGraft):
            raise ValueError('graft must be a SpecialistGraft')
        if graft.input_dim != self.output_dim or graft.context_dim != self.output_dim:
            raise ValueError('specialist input_dim and context_dim must equal output_dim')
        if self.input_dim != self.output_dim:
            raise ValueError('specialist exchange requires input_dim == output_dim')
        if graft._host is not None and graft._host is not self:
            raise ValueError('specialist is attached elsewhere')
        if not graft.attached:
            raise ValueError('specialist has been detached')
        self._specialist = graft
        graft._host = self

    def detach_specialist(self) -> 'SpecialistGraft | None':
        graft = self._specialist
        self._specialist = None
        if graft is not None and graft._host is self:
            graft._host = None
        return graft

    @property
    def specialist(self) -> 'SpecialistGraft | None':
        return self._specialist

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        return {
            'schema': MICRO_SCHEMA, 'schema_version': SCHEMA_VERSION, 'evidence_tier': EVIDENCE_TIER,
            'config': {'input_dim': self.input_dim, 'output_dim': self.output_dim, 'hidden': self.hidden,
                       'experts': self.n_experts, 'top_k': self.top_k, 'seed': self.seed,
                       'base_lr': self.base_lr, 'balanced': self.balanced, 'prioritized_replay': self.prioritized_replay,
                       'rollback': self.rollback_enabled, 'replay_capacity': self.replay_capacity,
                       'validation_capacity': self.validation_capacity,
                       'validation_every': self.validation_every, 'replay_batch': self.replay_batch,
                       'eval_interval': self.eval_interval, 'rollback_margin': self.rollback_margin,
                       'rollback_tolerance': self.rollback_tolerance,
                       'grad_clip': self.grad_clip, 'balance_rate': self.balance_rate,
                       'load_ema': self.load_ema_rate, 'router_lr_scale': self.router_lr_scale,
                       'min_validation': self.min_validation},
            'weights': {k: v.tolist() for k, v in self._params().items()},
            'load': {'ema': self.load_ema.tolist(), 'counts': [int(c) for c in self.load_counts]},
            'counters': {'samples': self.samples, 'updates': self.updates, 'sgd_steps': self.sgd_steps,
                         'evaluations': self.evaluations, 'rollbacks': self.rollbacks,
                         'experts_added': self.experts_added},
            'lr': self.lr,
            'replay': _items_json(self.replay),
            'validation': _items_json(self.validation),
            'best': None if self.best is None else {k: v.tolist() for k, v in self.best.items()},
            'best_loss': self.best_loss,
            'last_validation_loss': self.last_validation_loss,
            'receipts': json.loads(json.dumps(self.receipts)),
            'rng': {'init': _rng_state(self._init_rng), 'sample': _rng_state(self._rng)},
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Validate fully, then replace the collective's state. Dimensions must match."""
        if not isinstance(state, Mapping):
            raise ValueError('state must be a mapping')
        if state.get('schema') != MICRO_SCHEMA or state.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported Micro-Collective schema')
        cfg = state.get('config')
        if not isinstance(cfg, Mapping):
            raise ValueError('config must be a mapping')
        for key in ('input_dim', 'output_dim', 'hidden'):
            if cfg.get(key) != getattr(self, key):
                raise ValueError(f'config {key} mismatch')
        E = _check_int('experts', cfg.get('experts'), 1, MAX_EXPERTS)
        new_cfg = {
            'top_k': _check_int('top_k', cfg.get('top_k'), 1, E),
            'seed': _check_int('seed', cfg.get('seed'), 0, 2**31 - 1),
            'base_lr': _check_float('base_lr', cfg.get('base_lr'), MIN_LR, 1.0),
            'balanced': _check_bool('balanced', cfg.get('balanced')),
            'prioritized_replay': _check_bool('prioritized_replay', cfg.get('prioritized_replay')),
            'rollback_enabled': _check_bool('rollback', cfg.get('rollback')),
            'replay_capacity': _check_int('replay_capacity', cfg.get('replay_capacity'), 1, 1_000_000),
            'validation_capacity': _check_int('validation_capacity', cfg.get('validation_capacity'), 1, 1_000_000),
            'validation_every': _check_int('validation_every', cfg.get('validation_every'), 2, 1_000_000),
            'replay_batch': _check_int('replay_batch', cfg.get('replay_batch'), 0, 4096),
            'eval_interval': _check_int('eval_interval', cfg.get('eval_interval'), 1, 1_000_000),
            'rollback_margin': _check_float('rollback_margin', cfg.get('rollback_margin'), 0.0, 100.0),
            'rollback_tolerance': _check_float('rollback_tolerance', cfg.get('rollback_tolerance'), 0.0, 1e6),
            'grad_clip': _check_float('grad_clip', cfg.get('grad_clip'), 1e-6, 1e6),
            'balance_rate': _check_float('balance_rate', cfg.get('balance_rate'), 0.0, 1.0),
            'load_ema_rate': _check_float('load_ema', cfg.get('load_ema'), 1e-6, 1.0),
            'router_lr_scale': _check_float('router_lr_scale', cfg.get('router_lr_scale'), 0.0, 10.0),
            'min_validation': _check_int('min_validation', cfg.get('min_validation'), 1, 1_000_000),
        }
        D, H, O = self.input_dim, self.hidden, self.output_dim
        shapes = {'w1': (E, H, D), 'b1': (E, H), 'w2': (E, O, H), 'b2': (E, O),
                  'router_w': (E, D), 'router_b': (E,), 'balance_bias': (E,)}
        bounds = {'w1': WEIGHT_BOUND, 'b1': WEIGHT_BOUND, 'w2': WEIGHT_BOUND, 'b2': WEIGHT_BOUND,
                  'router_w': WEIGHT_BOUND, 'router_b': ROUTER_BIAS_BOUND, 'balance_bias': ROUTER_BIAS_BOUND}
        weights = state.get('weights')
        if not isinstance(weights, Mapping):
            raise ValueError('weights must be a mapping')
        new_w = {k: _array(weights, k, s, bounds[k]) for k, s in shapes.items()}
        best_raw = state.get('best')
        if best_raw is None:
            new_best = None
        elif isinstance(best_raw, Mapping):
            new_best = {k: _array(best_raw, k, shapes[k], bounds[k]) for k in self._SNAPSHOT_KEYS}
        else:
            raise ValueError('best must be a mapping or None')
        load = state.get('load')
        if not isinstance(load, Mapping):
            raise ValueError('load must be a mapping')
        load_ema = _array(load, 'ema', (E,), 1.0)
        counts_raw = load.get('counts')
        if (not isinstance(counts_raw, list) or len(counts_raw) != E
                or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts_raw)):
            raise ValueError('load.counts must be a list of non-negative ints')
        counters = state.get('counters')
        if not isinstance(counters, Mapping):
            raise ValueError('counters must be a mapping')
        new_counters = {k: _check_int(k, counters.get(k), 0, 2**62) for k in
                        ('samples', 'updates', 'sgd_steps', 'evaluations', 'rollbacks', 'experts_added')}
        lr = _check_float('lr', state.get('lr'), MIN_LR, 1.0)
        replay = _parse_items(state.get('replay'), D, O, new_cfg['replay_capacity'], 'replay')
        validation = _parse_items(state.get('validation'), D, O, new_cfg['validation_capacity'], 'validation')
        best_loss = _opt_float(state.get('best_loss'), 'best_loss')
        last_val = _opt_float(state.get('last_validation_loss'), 'last_validation_loss')
        receipts = state.get('receipts')
        if not isinstance(receipts, list) or len(receipts) > MAX_RECEIPTS or not all(isinstance(r, Mapping) for r in receipts):
            raise ValueError('receipts must be a bounded list of mappings')
        try:
            receipts_copy = json.loads(json.dumps(receipts, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError('receipts must be JSON-serialisable') from exc
        rng = state.get('rng')
        if not isinstance(rng, Mapping):
            raise ValueError('rng must be a mapping')
        init_rng = _rng_from_state(rng.get('init'))
        sample_rng = _rng_from_state(rng.get('sample'))
        # ---- validated: assign
        for k, v in new_cfg.items():
            setattr(self, k, v)
        for k, v in new_w.items():
            setattr(self, k, v)
        self.best = new_best
        self.load_ema = load_ema
        self.load_counts = np.asarray(counts_raw, dtype=np.int64)
        for k, v in new_counters.items():
            setattr(self, k, v)
        self.lr = lr
        self.replay = replay
        self.validation = validation
        self.best_loss = best_loss
        self.last_validation_loss = last_val
        self.receipts = receipts_copy
        self._init_rng = init_rng
        self._rng = sample_rng


# ============================================================================ SpecialistGraft
class SpecialistGraft:
    """Independent GRU-like recurrent next-step predictor with zero-initialised coupling.

    Trained only through ``train_step`` (its own SGD, one-step truncated
    backprop through the gated update). ``receive_main_prediction`` supplies
    the main system's prediction as context input; ``emit_prediction`` returns
    the graft's latest prediction; ``contribute`` returns ``coupling * prediction``
    (exact zeros while ``coupling == 0``)."""

    def __init__(self, input_dim: int = 16, context_dim: int = 16, seed: int = 7, *, hidden: int = 12,
                 lr: float = 0.03, validation_every: int = 5, validation_capacity: int = 64,
                 grad_clip: float = 1.0, eval_interval: int = 10) -> None:
        self.input_dim = _check_int('input_dim', input_dim, 1, MAX_DIM)
        self.context_dim = _check_int('context_dim', context_dim, 1, MAX_DIM)
        self.hidden = _check_int('hidden', hidden, 1, MAX_DIM)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        self.lr = _check_float('lr', lr, MIN_LR, 1.0)
        self.validation_every = _check_int('validation_every', validation_every, 2, 1_000_000)
        self.validation_capacity = _check_int('validation_capacity', validation_capacity, 1, 1_000_000)
        self.grad_clip = _check_float('grad_clip', grad_clip, 1e-6, 1e6)
        self.eval_interval = _check_int('eval_interval', eval_interval, 1, 1_000_000)
        rng = np.random.default_rng(self.seed + GRAFT_SEED_OFFSET)
        D, C, H = self.input_dim, self.context_dim, self.hidden
        U = D + C + H
        self.wz = rng.normal(0.0, 1.0 / math.sqrt(U), (H, U))
        self.wr = rng.normal(0.0, 1.0 / math.sqrt(U), (H, U))
        self.wh = rng.normal(0.0, 1.0 / math.sqrt(U), (H, U))
        self.bz = np.zeros(H); self.br = np.zeros(H); self.bh = np.zeros(H)
        self.wo = rng.normal(0.0, 0.5 / math.sqrt(H), (D, H))
        self.bo = np.zeros(D)
        self.h = np.zeros(H)
        self.context = np.zeros(C)
        self.last_prediction = np.zeros(D)
        self.coupling = 0.0
        self.attached = True
        self._host: MicroCollective | None = None
        self.validation: list[dict[str, np.ndarray]] = []
        self.samples = 0
        self.updates = 0
        self.evaluations = 0
        self.best_validation_loss: float | None = None
        self.last_validation_loss: float | None = None

    # ------------------------------------------------------------------ core math
    def _weights(self) -> list[np.ndarray]:
        return [self.wz, self.bz, self.wr, self.br, self.wh, self.bh, self.wo, self.bo]

    def _forward(self, x: np.ndarray, c: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
        u = np.concatenate([x, c, h])
        z = _sigmoid(self.wz @ u + self.bz)
        r = _sigmoid(self.wr @ u + self.br)
        u2 = np.concatenate([x, c, r * h])
        hc = np.tanh(self.wh @ u2 + self.bh)
        hn = (1.0 - z) * h + z * hc
        y = self.wo @ hn + self.bo
        return {'u': u, 'z': z, 'r': r, 'u2': u2, 'hc': hc, 'hn': hn, 'y': y, 'h': h}

    def _sgd(self, f: Mapping[str, np.ndarray], t: np.ndarray) -> float:
        H = self.hidden
        err = f['y'] - t
        loss = float(np.mean(err * err))
        dwo = np.outer(err, f['hn']); dbo = err.copy()
        dhn = self.wo.T @ err
        z, r, hc, h = f['z'], f['r'], f['hc'], f['h']
        dz = dhn * (hc - h) * z * (1.0 - z)
        dhc = dhn * z * (1.0 - hc * hc)
        dwh = np.outer(dhc, f['u2']); dbh = dhc.copy()
        d_rh = (self.wh.T @ dhc)[-H:]
        dr = d_rh * h * r * (1.0 - r)
        dwr = np.outer(dr, f['u']); dbr = dr.copy()
        dwz = np.outer(dz, f['u']); dbz = dz.copy()
        grads = [dwz, dbz, dwr, dbr, dwh, dbh, dwo, dbo]
        _clip_global(grads, self.grad_clip)
        for w, g in zip(self._weights(), grads):
            w -= self.lr * g
            np.clip(w, -WEIGHT_BOUND, WEIGHT_BOUND, out=w)
        return loss

    def _require_attached(self) -> None:
        if not self.attached:
            raise RuntimeError('specialist graft has been detached')

    # ------------------------------------------------------------------ exchange interface
    def receive_main_prediction(self, vec: Any) -> None:
        self._require_attached()
        self.context = _vector(vec, self.context_dim, 'main prediction')

    def emit_prediction(self) -> np.ndarray:
        return self.last_prediction.copy()

    def set_coupling(self, coupling: float) -> None:
        self._require_attached()
        self.coupling = _check_float('coupling', coupling, 0.0, MAX_COUPLING)

    def contribute(self, x: Any) -> np.ndarray:
        """``coupling * prediction(x | context, hidden)``; pure (does not advance state)."""
        xv = _vector(x, self.input_dim, 'x')
        if not self.attached or self.coupling == 0.0:
            return np.zeros(self.input_dim)
        y = np.clip(self._forward(xv, self.context, self.h)['y'], -OUTPUT_BOUND, OUTPUT_BOUND)
        return self.coupling * y

    def step(self, x: Any) -> np.ndarray:
        """Inference step: advance the hidden state and return the prediction."""
        self._require_attached()
        xv = _vector(x, self.input_dim, 'x')
        f = self._forward(xv, self.context, self.h)
        self.h = np.clip(f['hn'], -1.0, 1.0)
        self.last_prediction = np.clip(f['y'], -OUTPUT_BOUND, OUTPUT_BOUND)
        return self.last_prediction.copy()

    def train_step(self, x: Any, y_next: Any) -> dict[str, Any]:
        self._require_attached()
        xv = _vector(x, self.input_dim, 'x')
        tv = _vector(y_next, self.input_dim, 'y_next')
        self.samples += 1
        f = self._forward(xv, self.context, self.h)
        out: dict[str, Any] = {'sample': self.samples, 'held_out': False, 'loss': None, 'validation_loss': None}
        if self.samples % self.validation_every == 0:
            self.validation.append({'x': xv, 'c': self.context.copy(), 'h': self.h.copy(), 'y': tv})
            if len(self.validation) > self.validation_capacity:
                self.validation.pop(0)
            out['held_out'] = True
        else:
            out['loss'] = self._sgd(f, tv)
            self.updates += 1
            if self.updates % self.eval_interval == 0 and self.validation:
                out['validation_loss'] = self.validate()
        self.h = np.clip(f['hn'], -1.0, 1.0)
        self.last_prediction = np.clip(f['y'], -OUTPUT_BOUND, OUTPUT_BOUND)
        return out

    def evaluate(self) -> float | None:
        if not self.validation:
            return None
        errs = [np.mean((np.clip(self._forward(it['x'], it['c'], it['h'])['y'], -OUTPUT_BOUND, OUTPUT_BOUND)
                         - it['y']) ** 2) for it in self.validation]
        return float(np.mean(errs))

    def persistence_loss(self) -> float | None:
        if not self.validation:
            return None
        return float(np.mean([np.mean((it['x'] - it['y']) ** 2) for it in self.validation]))

    def validate(self) -> float:
        loss = self.evaluate()
        if loss is None:
            raise ValueError('no validation samples')
        self.evaluations += 1
        self.last_validation_loss = loss
        if self.best_validation_loss is None or loss < self.best_validation_loss:
            self.best_validation_loss = loss
        return loss

    def weights_digest(self) -> str:
        return _digest(self._weights(), {'samples': self.samples, 'updates': self.updates,
                                         'evaluations': self.evaluations, 'coupling': self.coupling})

    def receipt(self) -> dict[str, Any]:
        val = self.evaluate()
        pers = self.persistence_loss()
        return {'schema': GRAFT_RECEIPT_SCHEMA, 'evidence_tier': EVIDENCE_TIER,
                'sha256': self.weights_digest(), 'validation_loss': val,
                'best_validation_loss': self.best_validation_loss, 'persistence_loss': pers,
                'beats_persistence': (val is not None and pers is not None and val < pers),
                'samples': self.samples, 'updates': self.updates, 'evaluations': self.evaluations,
                'coupling': self.coupling, 'attached': self.attached,
                'validation_samples': len(self.validation)}

    def detach(self) -> dict[str, Any]:
        """Remove the graft: unlink from any host, zero the coupling, return a final receipt.

        After detaching, ``contribute`` returns zeros and the exchange methods raise."""
        host = self._host
        if host is not None and host._specialist is self:
            host.detach_specialist()
        self._host = None
        self.coupling = 0.0
        self.context = np.zeros(self.context_dim)
        receipt = self.receipt()
        self.attached = False
        receipt['attached'] = False
        return receipt

    # ------------------------------------------------------------------ persistence
    def export_state(self) -> dict[str, Any]:
        return {
            'schema': GRAFT_SCHEMA, 'schema_version': SCHEMA_VERSION, 'evidence_tier': EVIDENCE_TIER,
            'config': {'input_dim': self.input_dim, 'context_dim': self.context_dim, 'hidden': self.hidden,
                       'seed': self.seed, 'validation_every': self.validation_every,
                       'validation_capacity': self.validation_capacity, 'grad_clip': self.grad_clip,
                       'eval_interval': self.eval_interval},
            'weights': {n: w.tolist() for n, w in zip(('wz', 'bz', 'wr', 'br', 'wh', 'bh', 'wo', 'bo'), self._weights())},
            'h': self.h.tolist(), 'context': self.context.tolist(),
            'last_prediction': self.last_prediction.tolist(),
            'coupling': self.coupling, 'attached': self.attached, 'lr': self.lr,
            'validation': [{k: v.tolist() for k, v in it.items()} for it in self.validation],
            'counters': {'samples': self.samples, 'updates': self.updates, 'evaluations': self.evaluations},
            'best_validation_loss': self.best_validation_loss,
            'last_validation_loss': self.last_validation_loss,
        }

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError('state must be a mapping')
        if state.get('schema') != GRAFT_SCHEMA or state.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported specialist graft schema')
        cfg = state.get('config')
        if not isinstance(cfg, Mapping):
            raise ValueError('config must be a mapping')
        for key in ('input_dim', 'context_dim', 'hidden'):
            if cfg.get(key) != getattr(self, key):
                raise ValueError(f'config {key} mismatch')
        new_cfg = {
            'seed': _check_int('seed', cfg.get('seed'), 0, 2**31 - 1),
            'validation_every': _check_int('validation_every', cfg.get('validation_every'), 2, 1_000_000),
            'validation_capacity': _check_int('validation_capacity', cfg.get('validation_capacity'), 1, 1_000_000),
            'grad_clip': _check_float('grad_clip', cfg.get('grad_clip'), 1e-6, 1e6),
            'eval_interval': _check_int('eval_interval', cfg.get('eval_interval'), 1, 1_000_000),
        }
        D, C, H = self.input_dim, self.context_dim, self.hidden
        U = D + C + H
        shapes = {'wz': (H, U), 'bz': (H,), 'wr': (H, U), 'br': (H,), 'wh': (H, U), 'bh': (H,),
                  'wo': (D, H), 'bo': (D,)}
        weights = state.get('weights')
        if not isinstance(weights, Mapping):
            raise ValueError('weights must be a mapping')
        new_w = {k: _array(weights, k, s, WEIGHT_BOUND) for k, s in shapes.items()}
        h = _array(state, 'h', (H,), 1.0)
        context = _array(state, 'context', (C,), INPUT_BOUND)
        last = _array(state, 'last_prediction', (D,), OUTPUT_BOUND)
        coupling = _check_float('coupling', state.get('coupling'), 0.0, MAX_COUPLING)
        attached = _check_bool('attached', state.get('attached'))
        if not attached and coupling != 0.0:
            raise ValueError('a detached graft must have zero coupling')
        lr = _check_float('lr', state.get('lr'), MIN_LR, 1.0)
        raw_val = state.get('validation')
        if not isinstance(raw_val, list) or len(raw_val) > new_cfg['validation_capacity']:
            raise ValueError('validation must be a bounded list')
        val = []
        for it in raw_val:
            if not isinstance(it, Mapping):
                raise ValueError('validation items must be mappings')
            val.append({'x': _array(it, 'x', (D,), INPUT_BOUND), 'c': _array(it, 'c', (C,), INPUT_BOUND),
                        'h': _array(it, 'h', (H,), 1.0), 'y': _array(it, 'y', (D,), INPUT_BOUND)})
        counters = state.get('counters')
        if not isinstance(counters, Mapping):
            raise ValueError('counters must be a mapping')
        new_counters = {k: _check_int(k, counters.get(k), 0, 2**62) for k in ('samples', 'updates', 'evaluations')}
        best = _opt_float(state.get('best_validation_loss'), 'best_validation_loss')
        last_val = _opt_float(state.get('last_validation_loss'), 'last_validation_loss')
        # ---- validated: assign
        for k, v in new_cfg.items():
            setattr(self, k, v)
        for k, v in new_w.items():
            setattr(self, k, v)
        self.h, self.context, self.last_prediction = h, context, last
        self.coupling, self.attached, self.lr = coupling, attached, lr
        self.validation = val
        for k, v in new_counters.items():
            setattr(self, k, v)
        self.best_validation_loss = best
        self.last_validation_loss = last_val
        if not attached and self._host is not None:
            # importing a detached state removes the graft from its host (no residue)
            host = self._host
            self._host = None
            if host._specialist is self:
                host._specialist = None


__all__ = ['MicroCollective', 'SpecialistGraft', 'MICRO_SCHEMA', 'GRAFT_SCHEMA', 'MAX_COUPLING']
