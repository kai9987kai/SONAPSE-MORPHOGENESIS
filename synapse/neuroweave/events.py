"""Typed, versioned in-process event bus for NEUROWEAVE subsystems.

Topics are fixed (``input, state, prediction, modulation, learning, action,
memory, telemetry``). Each publish creates a validated, JSON-safe
``Envelope`` (schema ``NeuroWeaveEventV1``). Subscriber failures are isolated:
exceptions are counted and recorded, never propagated to the publisher.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

EVENT_SCHEMA = 'NeuroWeaveEventV1'
EVENT_VERSION = 1
TOPICS: tuple[str, ...] = ('input', 'state', 'prediction', 'modulation', 'learning', 'action', 'memory', 'telemetry')
WILDCARD = '*'
MAX_ERROR_LOG = 64
MAX_DEPTH = 16

# Typed payload interface per topic: required/optional keys -> type tag.
INTERFACES: dict[str, dict[str, Any]] = {
    'input': {'description': 'Derived sensory/task input (never raw media).',
              'required': {'channels': 'list[float]'},
              'optional': {'provenance': 'str', 'missing': 'list[int]', 'reliability': 'list[float]'}},
    'state': {'description': 'Population state summary.',
              'required': {'activity_mean': 'float', 'activity_std': 'float'},
              'optional': {'n': 'int', 'spike_rate': 'float'}},
    'prediction': {'description': 'Prediction error / uncertainty at a horizon.',
                   'required': {'horizon': 'int', 'error': 'float'},
                   'optional': {'disagreement': 'float', 'surprise': 'float', 'gate': 'float'}},
    'modulation': {'description': 'Neuromodulator levels and affect control variables.',
                   'required': {'levels': 'dict[str,float]'},
                   'optional': {'affect': 'dict[str,float]', 'controls': 'dict[str,float]'}},
    'learning': {'description': 'Plasticity / training update record.',
                 'required': {'kind': 'str', 'magnitude': 'float'},
                 'optional': {'updates': 'int', 'loss': 'float'}},
    'action': {'description': 'Issued action and efference comparison.',
               'required': {'action': 'list[float]'},
               'optional': {'intended': 'list[float]', 'authorship': 'float', 'reliability': 'float'}},
    'memory': {'description': 'Memory operation.',
               'required': {'operation': 'str'},
               'optional': {'id': 'int|str', 'provenance': 'str', 'hits': 'int'}},
    'telemetry': {'description': 'Named scalar metric.',
                  'required': {'name': 'str', 'value': 'float'},
                  'optional': {'unit': 'str', 'latency_ms': 'float'}},
}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_type(tag: str, v: Any) -> bool:
    if tag == 'float':
        return _is_num(v)
    if tag == 'int':
        return isinstance(v, int) and not isinstance(v, bool)
    if tag == 'str':
        return isinstance(v, str)
    if tag == 'bool':
        return isinstance(v, bool)
    if tag == 'int|str':
        return isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool))
    if tag == 'list[float]':
        return isinstance(v, list) and all(_is_num(x) for x in v)
    if tag == 'list[int]':
        return isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    if tag == 'dict[str,float]':
        return isinstance(v, dict) and all(isinstance(k, str) and _is_num(x) for k, x in v.items())
    raise ValueError(f'unknown type tag {tag!r}')  # pragma: no cover - registry is static


def _json_safe_fast(value: Any, depth: int) -> Any:
    """Fast recursive JSON-safety copy; raises ValueError on invalid values."""
    if depth > MAX_DEPTH:
        raise ValueError('nesting too deep')
    t = type(value)
    if value is None or t is bool or t is str or t is int:
        return value
    if t is float:
        if not math.isfinite(value):
            raise ValueError('non-finite float')
        return value
    if t is list or t is tuple:
        next_depth = depth + 1
        out_list = []
        append = out_list.append
        for v in value:
            tv = type(v)
            if v is None or tv is bool or tv is str or tv is int:
                append(v)
            elif tv is float:
                if not math.isfinite(v):
                    raise ValueError('non-finite float')
                append(v)
            else:
                append(_json_safe_fast(v, next_depth))
        return out_list
    if t is dict:
        next_depth = depth + 1
        out_dict = {}
        for k, v in value.items():
            if type(k) is not str and not isinstance(k, str):
                raise ValueError('dict keys must be str')
            tv = type(v)
            if v is None or tv is bool or tv is str or tv is int:
                out_dict[k] = v
            elif tv is float:
                if not math.isfinite(v):
                    raise ValueError('non-finite float')
                out_dict[k] = v
            else:
                out_dict[k] = _json_safe_fast(v, next_depth)
        return out_dict
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('non-finite float')
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_fast(v, depth + 1) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError('dict keys must be str')
            out[k] = _json_safe_fast(v, depth + 1)
        return out
    raise ValueError(f'{t.__name__} is not JSON-safe')


def _json_safe_with_path(value: Any, path: str, depth: int) -> Any:
    """Error-reporting variant that preserves exact diagnostic path strings."""
    if depth > MAX_DEPTH:
        raise ValueError(f'{path}: nesting too deep')
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'{path}: non-finite float')
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_with_path(v, f'{path}[{i}]', depth + 1) for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f'{path}: dict keys must be str')
            out[k] = _json_safe_with_path(v, f'{path}.{k}', depth + 1)
        return out
    raise ValueError(f'{path}: {type(value).__name__} is not JSON-safe')


def json_safe(value: Any, path: str = 'payload', depth: int = 0) -> Any:
    """Return a deep copy of ``value`` if it is JSON-safe, else raise ValueError."""
    try:
        return _json_safe_fast(value, depth)
    except ValueError:
        return _json_safe_with_path(value, path, depth)


def _validate_payload_schema(topic: str, payload: dict[str, Any]) -> None:
    spec = INTERFACES[topic]
    for key, tag in spec['required'].items():
        if key not in payload:
            raise ValueError(f'{topic} payload missing required key {key!r}')
        if not _check_type(tag, payload[key]):
            raise ValueError(f'{topic}.{key} must be {tag}')
    for key, tag in spec['optional'].items():
        if key in payload and not _check_type(tag, payload[key]):
            raise ValueError(f'{topic}.{key} must be {tag}')


def validate_payload(topic: str, payload: Mapping[str, Any]) -> None:
    """Check a payload against ``INTERFACES[topic]`` (extra keys allowed if JSON-safe)."""
    if topic not in INTERFACES:
        raise ValueError(f'unknown topic {topic!r}')
    if not isinstance(payload, dict):
        raise ValueError('payload must be a dict')
    json_safe(payload)
    _validate_payload_schema(topic, payload)


def _clone_json(value: Any) -> Any:
    t = type(value)
    if value is None or t is bool or t is str or t is int or t is float:
        return value
    if t is list:
        return [_clone_json(v) for v in value]
    if t is dict:
        return {k: _clone_json(v) for k, v in value.items()}
    return copy.deepcopy(value)


@dataclass(frozen=True)
class Envelope:
    topic: str
    step: int
    source: str
    payload: dict
    sequence: int
    schema: str = EVENT_SCHEMA
    version: int = EVENT_VERSION

    def __post_init__(self) -> None:
        if self.schema != EVENT_SCHEMA or self.version != EVENT_VERSION:
            raise ValueError('unsupported envelope schema/version')
        if self.topic not in TOPICS:
            raise ValueError(f'unknown topic {self.topic!r}')
        for name in ('step', 'sequence'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f'{name} must be a non-negative int')
        if not isinstance(self.source, str) or not self.source:
            raise ValueError('source must be a non-empty string')
        if not isinstance(self.payload, dict):
            raise ValueError('payload must be a dict')
        object.__setattr__(self, 'payload', json_safe(self.payload))

    @classmethod
    def _from_validated(cls, topic: str, step: int, source: str, payload: dict[str, Any], sequence: int) -> 'Envelope':
        if topic not in TOPICS:
            raise ValueError(f'unknown topic {topic!r}')
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('step must be a non-negative int')
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError('sequence must be a non-negative int')
        if not isinstance(source, str) or not source:
            raise ValueError('source must be a non-empty string')
        obj = object.__new__(cls)
        object.__setattr__(obj, 'topic', topic)
        object.__setattr__(obj, 'step', step)
        object.__setattr__(obj, 'source', source)
        object.__setattr__(obj, 'payload', payload)
        object.__setattr__(obj, 'sequence', sequence)
        object.__setattr__(obj, 'schema', EVENT_SCHEMA)
        object.__setattr__(obj, 'version', EVENT_VERSION)
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {'schema': self.schema, 'version': self.version, 'topic': self.topic, 'step': self.step,
                'source': self.source, 'payload': _clone_json(self.payload), 'sequence': self.sequence}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'Envelope':
        if not isinstance(d, Mapping):
            raise ValueError('envelope dict required')
        try:
            return cls(topic=d['topic'], step=d['step'], source=d['source'], payload=d['payload'],
                       sequence=d['sequence'], schema=d['schema'], version=d['version'])
        except KeyError as exc:
            raise ValueError(f'envelope missing {exc}') from exc


class EventBus:
    def __init__(self, history_limit: int = 256, *, strict: bool = True) -> None:
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or not 1 <= history_limit <= 100000:
            raise ValueError('history_limit must be an int in [1, 100000]')
        self.history_limit = history_limit
        self.strict = bool(strict)
        self.sequence = 0
        self._subs: dict[int, tuple[str, Callable[[Envelope], Any]]] = {}
        self._next_token = 1
        self._history: dict[str, deque[Envelope]] = {t: deque(maxlen=history_limit) for t in TOPICS}
        self.published = {t: 0 for t in TOPICS}
        self.delivered = {t: 0 for t in TOPICS}
        self.errors = {t: 0 for t in TOPICS}
        self.error_log: deque[dict[str, Any]] = deque(maxlen=MAX_ERROR_LOG)

    def subscribe(self, topic: str, callback: Callable[[Envelope], Any]) -> int:
        if topic != WILDCARD and topic not in TOPICS:
            raise ValueError(f'unknown topic {topic!r}')
        if not callable(callback):
            raise ValueError('callback must be callable')
        token = self._next_token
        self._next_token += 1
        self._subs[token] = (topic, callback)
        return token

    def unsubscribe(self, token: int) -> bool:
        return self._subs.pop(token, None) is not None

    def subscriber_count(self, topic: str | None = None) -> int:
        return sum(1 for t, _ in self._subs.values() if topic is None or t == topic)

    def publish(self, topic: str, payload: Mapping[str, Any], *, step: int = 0, source: str = 'neuroweave',
                strict: bool | None = None) -> Envelope:
        if topic not in TOPICS:
            raise ValueError(f'unknown topic {topic!r}')
        if not isinstance(payload, Mapping):
            raise ValueError('payload must be a mapping')
        raw_payload = payload if type(payload) is dict else dict(payload)
        safe_payload = json_safe(raw_payload)
        if (self.strict if strict is None else strict):
            if topic not in INTERFACES:
                raise ValueError(f'unknown topic {topic!r}')
            _validate_payload_schema(topic, safe_payload)
        env = Envelope._from_validated(topic=topic, step=step, source=source, payload=safe_payload, sequence=self.sequence)
        self.sequence += 1
        self._history[topic].append(env)
        self.published[topic] += 1
        if self._subs:
            for token, (t, cb) in list(self._subs.items()):
                if t != topic and t != WILDCARD:
                    continue
                try:
                    # Each subscriber gets its own deep copy of the payload so a subscriber
                    # mutating it cannot corrupt history or what later subscribers see.
                    cb(Envelope._from_validated(topic=env.topic, step=env.step, source=env.source,
                                                payload=_clone_json(env.payload), sequence=env.sequence))
                    self.delivered[topic] += 1
                except Exception as exc:  # noqa: BLE001 - isolation is the point
                    self.errors[topic] += 1
                    self.error_log.append({'token': token, 'topic': topic, 'sequence': env.sequence,
                                           'error': f'{type(exc).__name__}: {exc}'[:500]})
        return env

    def history(self, topic: str, limit: int | None = None) -> list[Envelope]:
        if topic not in TOPICS:
            raise ValueError(f'unknown topic {topic!r}')
        items = list(self._history[topic])
        return items if limit is None else items[-limit:] if limit > 0 else []

    def counters(self) -> dict[str, Any]:
        return {'published': dict(self.published), 'delivered': dict(self.delivered), 'errors': dict(self.errors),
                'sequence': self.sequence, 'subscribers': self.subscriber_count()}

    def export_state(self) -> dict[str, Any]:
        return {'schema': 'NeuroWeaveEventBusV1', 'history_limit': self.history_limit, 'strict': self.strict,
                'sequence': self.sequence,
                'history': {t: [e.to_dict() for e in self._history[t]] for t in TOPICS},
                'published': dict(self.published), 'delivered': dict(self.delivered), 'errors': dict(self.errors),
                'error_log': [dict(e) for e in self.error_log]}

    def import_state(self, state: Mapping[str, Any]) -> None:
        """Restore history/counters (subscribers are runtime-only and kept)."""
        if not isinstance(state, Mapping) or state.get('schema') != 'NeuroWeaveEventBusV1':
            raise ValueError('not a NeuroWeaveEventBusV1 state')
        if state.get('history_limit') != self.history_limit:
            raise ValueError('history_limit mismatch')
        seq = state.get('sequence')
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError('sequence invalid')
        hist = state.get('history')
        if not isinstance(hist, Mapping) or set(hist) != set(TOPICS):
            raise ValueError('history must cover every topic')
        new_hist = {}
        for t in TOPICS:
            items = hist[t]
            if not isinstance(items, list) or len(items) > self.history_limit:
                raise ValueError(f'history[{t}] malformed')
            envs = [Envelope.from_dict(d) for d in items]
            if any(e.topic != t or e.sequence >= seq for e in envs):
                raise ValueError(f'history[{t}] contains inconsistent envelopes')
            if [e.sequence for e in envs] != sorted({e.sequence for e in envs}):
                raise ValueError(f'history[{t}] sequences must be unique and increasing')
            new_hist[t] = deque(envs, maxlen=self.history_limit)
        all_seq = [e.sequence for q in new_hist.values() for e in q]
        if len(all_seq) != len(set(all_seq)):
            raise ValueError('history sequence numbers must be unique across topics')
        counts = {}
        for key in ('published', 'delivered', 'errors'):
            c = state.get(key)
            if not isinstance(c, Mapping) or set(c) != set(TOPICS) or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in c.values()):
                raise ValueError(f'{key} counters malformed')
            counts[key] = {t: int(c[t]) for t in TOPICS}
        strict_flag = state.get('strict', self.strict)
        if not isinstance(strict_flag, bool):
            raise ValueError('strict must be a bool')
        log = state.get('error_log', [])
        if not isinstance(log, list) or len(log) > MAX_ERROR_LOG:
            raise ValueError('error_log malformed')
        log = [json_safe(e) for e in log]
        if any(not isinstance(e, dict) for e in log):
            raise ValueError('error_log entries must be dicts')
        self.sequence = seq
        self.strict = strict_flag
        self._history = new_hist
        self.published, self.delivered, self.errors = counts['published'], counts['delivered'], counts['errors']
        self.error_log = deque(log, maxlen=MAX_ERROR_LOG)


__all__ = ['EVENT_SCHEMA', 'EVENT_VERSION', 'TOPICS', 'INTERFACES', 'Envelope', 'EventBus', 'validate_payload',
           'json_safe']
