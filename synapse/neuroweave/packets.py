"""Typed representation packets, encoders and a zero-initialised fusion layer.

A ``RepresentationPacket`` carries one typed representation (scalar, spectral,
phase, sparse, VSA, ...) with provenance, confidence and uncertainty, and a
SHA-256 checksum over its canonical JSON form. ``RepresentationFusion`` mixes
packets into a scalar drive vector through fixed seeded projections and
per-kind weights that start at exactly zero, so fusion is an identity on the
scalar drive until weights are configured or learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PACKET_SCHEMA = 'NeuroWeavePacketV1'
PACKET_VERSION = 1
PACKET_KINDS: tuple[str, ...] = (
    'scalar', 'vector', 'spectral', 'phase', 'temporal', 'spatial', 'graph', 'sparse', 'vsa', 'event',
    'population', 'context', 'predictive', 'uncertainty', 'compartment', 'hybrid',
)
PROVENANCES: tuple[str, ...] = ('sensory', 'predicted', 'external', 'experience', 'synthetic')
FORMS: tuple[str, ...] = ('dense', 'binary', 'index')
MAX_PAYLOAD = 65536
PAYLOAD_BOUND = 1.0e9
NW_X2_PREFIX = 'nwp1:'
FUSION_SEED_OFFSET = 4303
# Projection-cache budget (float64 elements, ~64 MB): oldest projections are evicted first.
PROJECTION_CACHE_ELEMENTS = 8_000_000


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _json_scalar(name: str, v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f'{name} must be finite')
        return float(v)
    raise ValueError(f'{name} must be a JSON scalar, got {type(v).__name__}')


@dataclass(frozen=True)
class RepresentationPacket:
    """Immutable typed representation (schema ``NeuroWeavePacketV1``)."""

    kind: str
    payload: tuple[float, ...]
    timestamp: float = 0.0
    source: tuple[str, ...] = ('unspecified',)
    confidence: float = 1.0
    uncertainty: float = 0.0
    provenance: str = 'synthetic'
    form: str = 'dense'
    dim: int = -1
    metadata: tuple[tuple[str, Any], ...] = ()
    schema: str = PACKET_SCHEMA
    version: int = PACKET_VERSION
    checksum: str = field(init=False, default='')

    def __post_init__(self) -> None:
        if self.schema != PACKET_SCHEMA or self.version != PACKET_VERSION:
            raise ValueError(f'unsupported packet schema {self.schema!r} v{self.version!r}')
        if self.kind not in PACKET_KINDS:
            raise ValueError(f'unknown packet kind {self.kind!r}')
        if self.provenance not in PROVENANCES:
            raise ValueError(f'unknown provenance {self.provenance!r}')
        if self.form not in FORMS:
            raise ValueError(f'unknown form {self.form!r}')
        if self.form != 'dense' and self.kind != 'sparse':
            raise ValueError('binary/index forms are only valid for sparse packets')
        if self.kind == 'sparse' and self.form == 'dense':
            raise ValueError("sparse packets must use form 'binary' or 'index'")
        if isinstance(self.payload, (str, bytes)):
            raise ValueError('payload must be a sequence of numbers, not a string')
        try:
            payload = tuple(float(v) for v in self.payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'payload must be numeric: {exc}') from exc
        if any(isinstance(v, (bool, np.bool_)) for v in self.payload):
            raise ValueError('payload must not contain bools')
        if not payload or len(payload) > MAX_PAYLOAD:
            raise ValueError(f'payload length must be in [1, {MAX_PAYLOAD}]')
        if not all(math.isfinite(v) and abs(v) <= PAYLOAD_BOUND for v in payload):
            raise ValueError('payload must be finite and bounded')
        dim = len(payload) if self.dim == -1 and self.form != 'index' else self.dim
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 or dim > MAX_PAYLOAD:
            raise ValueError('dim must be a positive int (required for index form)')
        if self.form in ('dense', 'binary') and len(payload) != dim:
            raise ValueError(f'payload length {len(payload)} != dim {dim}')
        if self.form == 'binary' and any(v not in (0.0, 1.0) for v in payload):
            raise ValueError('binary sparse payload must contain only 0/1')
        if self.form == 'index':
            if any(v != int(v) or v < 0 or v >= dim for v in payload):
                raise ValueError('index sparse payload must hold integer indices in [0, dim)')
            if list(payload) != sorted(set(payload)):
                raise ValueError('index sparse payload must be strictly increasing (unique, sorted)')
        if self.kind == 'phase' and any(v < -math.pi - 1e-12 or v > math.pi + 1e-12 for v in payload):
            raise ValueError('phase payload must lie in [-pi, pi]')
        ts = self.timestamp
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts < 0:
            raise ValueError('timestamp must be a finite non-negative number')
        for name in ('confidence', 'uncertainty'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(f'{name} must be in [0, 1]')
        src = (self.source,) if isinstance(self.source, str) else tuple(self.source)
        if not src or not all(isinstance(s, str) and s for s in src):
            raise ValueError('source must be one or more non-empty strings')
        meta_in = self.metadata.items() if isinstance(self.metadata, Mapping) else self.metadata
        meta = []
        for item in meta_in:
            if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError('metadata must be (str key, JSON scalar) pairs')
            meta.append((item[0], _json_scalar(f'metadata[{item[0]}]', item[1])))
        meta.sort(key=lambda kv: kv[0])
        if len({k for k, _ in meta}) != len(meta):
            raise ValueError('duplicate metadata keys')
        object.__setattr__(self, 'payload', payload)
        object.__setattr__(self, 'dim', int(dim))
        object.__setattr__(self, 'timestamp', float(ts))
        object.__setattr__(self, 'confidence', float(self.confidence))
        object.__setattr__(self, 'uncertainty', float(self.uncertainty))
        object.__setattr__(self, 'source', src)
        object.__setattr__(self, 'metadata', tuple(meta))
        object.__setattr__(self, 'checksum', hashlib.sha256(_canonical(self._body()).encode('utf-8')).hexdigest())

    def _body(self) -> dict[str, Any]:
        return {'schema': self.schema, 'version': self.version, 'kind': self.kind, 'form': self.form,
                'dim': self.dim, 'payload': list(self.payload), 'timestamp': self.timestamp,
                'source': list(self.source), 'confidence': self.confidence, 'uncertainty': self.uncertainty,
                'provenance': self.provenance, 'metadata': [list(kv) for kv in self.metadata]}

    @property
    def reliability(self) -> float:
        """Down-weighting factor used by fusion: confidence * (1 - uncertainty)."""
        return self.confidence * (1.0 - self.uncertainty)

    def dense(self) -> np.ndarray:
        """Logical dense vector (index-form sparse packets are expanded to 0/1)."""
        if self.form == 'index':
            out = np.zeros(self.dim)
            out[np.asarray(self.payload, dtype=np.int64)] = 1.0
            return out
        return np.asarray(self.payload, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        d = self._body()
        d['checksum'] = self.checksum
        return d

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'RepresentationPacket':
        if not isinstance(data, Mapping):
            raise ValueError('packet dict required')
        for key in ('payload', 'source'):
            if isinstance(data.get(key), (str, bytes, Mapping)):
                raise ValueError(f'{key} must be a list, not {type(data.get(key)).__name__}')
        meta_raw = data.get('metadata', ())
        if isinstance(meta_raw, (str, bytes, Mapping)) or any(
                not isinstance(kv, (list, tuple)) for kv in meta_raw):
            raise ValueError('metadata must be a list of [key, value] pairs')
        try:
            pkt = cls(kind=data['kind'], payload=tuple(data['payload']), timestamp=data['timestamp'],
                      source=tuple(data['source']), confidence=data['confidence'],
                      uncertainty=data['uncertainty'], provenance=data['provenance'], form=data['form'],
                      dim=data['dim'], metadata=tuple(tuple(kv) for kv in data.get('metadata', ())),
                      schema=data.get('schema', PACKET_SCHEMA), version=data.get('version', PACKET_VERSION))
        except (KeyError, TypeError) as exc:
            raise ValueError(f'malformed packet dict: {exc}') from exc
        if 'checksum' in data and data['checksum'] != pkt.checksum:
            raise ValueError('packet checksum mismatch')
        return pkt


# --------------------------------------------------------------------------- NeuroWave-X2 adapter
def to_neurowave_x2(packet: RepresentationPacket, *, sender_region: int = 0, receiver_region: int = 0,
                    sequence_id: int = 0, topology_version: int = 0, precision_bits: int = 32):
    """Map a packet onto ``synapse.neurowave_x2.packets.NeuralPacket``.

    kind/form/dim/provenance/source/metadata/checksum travel in ``protocol_id``
    (prefixed canonical JSON); timestamp, confidence and uncertainty travel in
    ``phase_metadata``.
    """
    from synapse.neurowave_x2.packets import NeuralPacket
    if not isinstance(packet, RepresentationPacket):
        raise ValueError('RepresentationPacket required')
    header = {'kind': packet.kind, 'form': packet.form, 'dim': packet.dim, 'provenance': packet.provenance,
              'source': list(packet.source), 'metadata': [list(kv) for kv in packet.metadata],
              'checksum': packet.checksum}
    return NeuralPacket(sender_region=sender_region, receiver_region=receiver_region,
                        emission_step=int(packet.timestamp), sequence_id=sequence_id,
                        topology_version=topology_version, protocol_id=NW_X2_PREFIX + _canonical(header),
                        payload=tuple(packet.payload), precision_bits=precision_bits,
                        phase_metadata=(packet.timestamp, packet.confidence, packet.uncertainty))


def from_neurowave_x2(np_packet: Any) -> RepresentationPacket:
    """Inverse of :func:`to_neurowave_x2`; verifies the embedded SHA-256 checksum."""
    proto = getattr(np_packet, 'protocol_id', None)
    if not isinstance(proto, str) or not proto.startswith(NW_X2_PREFIX):
        raise ValueError('not a NeuroWeave packet (protocol_id prefix missing)')
    try:
        header = json.loads(proto[len(NW_X2_PREFIX):])
        ts, conf, unc = (float(v) for v in np_packet.phase_metadata)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'malformed NeuroWave-X2 packet: {exc}') from exc
    if not isinstance(header, dict):
        raise ValueError('malformed protocol header')
    return RepresentationPacket.from_dict({
        'kind': header.get('kind'), 'form': header.get('form'), 'dim': header.get('dim'),
        'provenance': header.get('provenance'), 'source': header.get('source', []),
        'metadata': header.get('metadata', []), 'payload': list(np_packet.payload),
        'timestamp': ts, 'confidence': conf, 'uncertainty': unc, 'checksum': header.get('checksum')})


# --------------------------------------------------------------------------- encoders
def _vec(name: str, x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be a non-empty finite array')
    return arr


def _pkt(kind: str, payload: Any, **kw: Any) -> RepresentationPacket:
    return RepresentationPacket(kind=kind, payload=tuple(float(v) for v in np.asarray(payload).ravel()), **kw)


def encode_scalar(value: float, **kw: Any) -> RepresentationPacket:
    return _pkt('scalar', _vec('value', [value]), **kw)


def encode_vector(values: Any, *, kind: str = 'vector', **kw: Any) -> RepresentationPacket:
    """Dense vector; ``kind`` may be any dense kind (context, predictive, compartment, ...)."""
    return _pkt(kind, _vec('values', values), **kw)


def encode_spectral(signal: Any, *, n_bins: int | None = None, **kw: Any) -> RepresentationPacket:
    """Normalised rFFT magnitudes of a 1-D signal (DC removed)."""
    x = _vec('signal', signal)
    mag = np.abs(np.fft.rfft(x - x.mean())) / x.size
    if n_bins is not None:
        mag = mag[:max(1, int(n_bins))]
    return _pkt('spectral', mag, **kw)


def encode_phase(signal: Any, *, n_bins: int | None = None, **kw: Any) -> RepresentationPacket:
    """Phase angle of rFFT bins in [-pi, pi]; near-zero-magnitude bins get phase 0."""
    x = _vec('signal', signal)
    spec = np.fft.rfft(x)
    ph = np.where(np.abs(spec) > 1e-12 * max(1.0, float(np.abs(x).max())), np.angle(spec), 0.0)
    if n_bins is not None:
        ph = ph[:max(1, int(n_bins))]
    return _pkt('phase', np.clip(ph, -math.pi, math.pi), **kw)


def encode_population(value: float, *, n: int = 16, low: float = -1.0, high: float = 1.0,
                      width: float | None = None, **kw: Any) -> RepresentationPacket:
    """Gaussian tuning-curve population code over ``n`` evenly spaced preferred values."""
    v = float(_vec('value', [value])[0])
    if n < 2 or not high > low:
        raise ValueError('population needs n >= 2 and high > low')
    centers = np.linspace(low, high, n)
    sigma = width if width is not None else (high - low) / (n - 1)
    return _pkt('population', np.exp(-0.5 * ((v - centers) / sigma) ** 2), **kw)


def decode_population(packet: RepresentationPacket, *, low: float = -1.0, high: float = 1.0) -> float:
    act = packet.dense()
    centers = np.linspace(low, high, act.size)
    return float((act * centers).sum() / max(act.sum(), 1e-12))


def encode_sparse(values: Any, k: int, *, form: str = 'index', **kw: Any) -> RepresentationPacket:
    """k-winners-take-all (largest values; ties broken by lower index)."""
    x = _vec('values', values)
    if not 1 <= k <= x.size:
        raise ValueError('k must be in [1, len(values)]')
    winners = np.sort(np.argsort(-x, kind='stable')[:k])
    if form == 'index':
        return RepresentationPacket(kind='sparse', payload=tuple(float(i) for i in winners), form='index',
                                    dim=int(x.size), **kw)
    if form == 'binary':
        b = np.zeros(x.size)
        b[winners] = 1.0
        return _pkt('sparse', b, form='binary', **kw)
    raise ValueError("form must be 'index' or 'binary'")


def vsa_vector(name: str, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Deterministic bipolar (+1/-1) hypervector for a symbol name."""
    h = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode('utf-8')).digest()[:8], 'big')
    rng = np.random.default_rng(h)
    return np.where(rng.random(dim) < 0.5, -1.0, 1.0)


def vsa_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR binding by circular convolution."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def vsa_unbind(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Approximate inverse binding by circular correlation."""
    return np.real(np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(a)))) / a.size


def encode_vsa(bindings: Mapping[str, str] | Sequence[tuple[str, str]], *, dim: int = 64, seed: int = 0,
               bipolar: bool = True, **kw: Any) -> RepresentationPacket:
    """Bundle of role (*) filler circular-convolution bindings; bipolarised by sign."""
    pairs = list(bindings.items()) if isinstance(bindings, Mapping) else list(bindings)
    if not pairs:
        raise ValueError('at least one binding required')
    acc = np.zeros(dim)
    for role, filler in pairs:
        acc += vsa_bind(vsa_vector(str(role), dim, seed), vsa_vector(str(filler), dim, seed))
    out = np.where(acc >= 0, 1.0, -1.0) if bipolar else acc
    raw_meta = kw.pop('metadata', ())
    meta = tuple(raw_meta.items() if isinstance(raw_meta, Mapping) else raw_meta) + (('vsa_seed', int(seed)),)
    return _pkt('vsa', out, metadata=meta, **kw)


def encode_event(magnitude: float, *, onset: bool = True, duration: float = 1.0, label: str = '',
                 code_dim: int = 8, **kw: Any) -> RepresentationPacket:
    """[onset, magnitude, log1p(duration), hashed label one-hot over code_dim]."""
    mag = float(_vec('magnitude', [magnitude])[0])
    if duration < 0 or code_dim < 1:
        raise ValueError('duration must be >= 0 and code_dim >= 1')
    code = np.zeros(code_dim)
    if label:
        code[int(hashlib.sha256(label.encode('utf-8')).hexdigest(), 16) % code_dim] = 1.0
    return _pkt('event', np.concatenate([[1.0 if onset else 0.0, mag, math.log1p(duration)], code]), **kw)


def encode_spatial(x: float, y: float, *, grid: int = 4, sigma: float | None = None,
                   **kw: Any) -> RepresentationPacket:
    """Place-field grid activations for normalised coordinates (x, y) in [0, 1]."""
    px, py = (float(np.clip(v, 0.0, 1.0)) for v in _vec('xy', [x, y]))
    if grid < 1:
        raise ValueError('grid must be >= 1')
    c = (np.arange(grid) + 0.5) / grid
    s = sigma if sigma is not None else 1.0 / grid
    gx, gy = np.meshgrid(c, c, indexing='xy')
    act = np.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2 * s * s))
    return _pkt('spatial', act.ravel(), **kw)


def encode_graph(adjacency: Any, *, k_eig: int = 4, **kw: Any) -> RepresentationPacket:
    """[n, density, mean/std/max degree, k smallest normalised-Laplacian eigenvalues]."""
    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError('adjacency must be a finite square matrix')
    S = (np.abs(A) + np.abs(A.T)) * 0.5
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    deg = (S > 0).sum(axis=1).astype(np.float64)
    density = float(deg.sum() / max(n * (n - 1), 1))
    w = S.sum(axis=1)
    inv = np.where(w > 0, 1.0 / np.sqrt(np.where(w > 0, w, 1.0)), 0.0)
    L = np.diag((w > 0).astype(np.float64)) - inv[:, None] * S * inv[None, :]
    eig = np.sort(np.linalg.eigvalsh(L))
    feats = np.zeros(k_eig)
    feats[:min(k_eig, n)] = eig[:k_eig]
    return _pkt('graph', np.concatenate([[n, density, deg.mean(), deg.std(), deg.max()], feats]), **kw)


def encode_temporal(series: Any, *, taus: Sequence[float] = (1.0, 4.0, 16.0), **kw: Any) -> RepresentationPacket:
    """Final multi-timescale EMA traces of a series (rows = time; per-tau, per-channel)."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        raise ValueError('series must be a non-empty finite 1-D or 2-D array')
    if not taus or any(t < 1 for t in taus):
        raise ValueError('taus must be >= 1')
    traces = []
    for tau in taus:
        a = 1.0 / float(tau)
        tr = np.zeros(arr.shape[1])
        for row in arr:
            tr += a * (row - tr)
        traces.append(tr)
    return _pkt('temporal', np.concatenate(traces), **kw)


# --------------------------------------------------------------------------- fusion
class RepresentationFusion:
    """Mix typed packets into a ``dim``-length drive with zero-initialised weights.

    ``fuse(drive, packets) = drive + sum_k w_k * sum_{p in kind k} rel(p) * Proj(k, len(p)) @ dense(p)``
    where ``rel(p) = confidence * (1 - uncertainty)`` and ``Proj`` is a fixed
    seeded Gaussian matrix scaled by ``1/sqrt(len)``. With every weight at its
    initial 0.0 the output equals ``drive`` exactly.
    """

    MODES = ('configurable', 'learned')

    def __init__(self, dim: int, kinds: Iterable[str] = PACKET_KINDS, mode: str = 'configurable', *,
                 seed: int = 7, lr: float = 0.05, weight_bound: float = 2.0) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 65536:
            raise ValueError('dim must be a positive int')
        ks = tuple(kinds)
        if not ks or any(k not in PACKET_KINDS for k in ks) or len(set(ks)) != len(ks):
            raise ValueError('kinds must be unique packet kinds')
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}')
        if not (0 < lr <= 1.0) or not (0 < weight_bound <= 100.0):
            raise ValueError('lr in (0,1] and weight_bound in (0,100] required')
        self.dim, self.kinds, self.mode, self.seed = dim, ks, mode, int(seed)
        self.lr, self.weight_bound = float(lr), float(weight_bound)
        self.weights: dict[str, float] = {k: 0.0 for k in ks}
        self.updates = 0
        self.ignored = 0
        self._proj_cache: dict[tuple[str, int], np.ndarray] = {}
        self._proj_elements = 0

    def projection(self, kind: str, length: int) -> np.ndarray:
        if kind not in PACKET_KINDS or isinstance(length, bool) or not 1 <= int(length) <= MAX_PAYLOAD:
            raise ValueError('projection needs a packet kind and a length in [1, MAX_PAYLOAD]')
        key = (kind, int(length))
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached
        k_idx = PACKET_KINDS.index(kind)
        rng = np.random.default_rng([self.seed + FUSION_SEED_OFFSET, k_idx, int(length)])
        mat = rng.normal(0.0, 1.0 / math.sqrt(length), size=(self.dim, int(length)))
        # Bounded cache: evict oldest entries (projections are regenerated identically from the seed).
        while self._proj_cache and self._proj_elements + mat.size > PROJECTION_CACHE_ELEMENTS:
            old = next(iter(self._proj_cache))
            self._proj_elements -= self._proj_cache.pop(old).size
        if mat.size <= PROJECTION_CACHE_ELEMENTS:
            self._proj_cache[key] = mat
            self._proj_elements += mat.size
        return mat

    def set_weight(self, kind: str, weight: float) -> None:
        if kind not in self.weights:
            raise ValueError(f'kind {kind!r} not managed by this fusion layer')
        w = float(weight)
        if not math.isfinite(w) or abs(w) > self.weight_bound:
            raise ValueError(f'weight must be finite and within +/-{self.weight_bound}')
        self.weights[kind] = w

    def contributions(self, packets: Iterable[RepresentationPacket], *,
                      only_nonzero: bool = False) -> dict[str, np.ndarray]:
        """Per-kind reliability-weighted projected sums (before kind weights).

        With ``only_nonzero`` kinds whose weight is exactly 0 are skipped without
        projecting (they cannot change the output). All packets are type-checked
        before any counter is touched.
        """
        pkts = list(packets)
        if any(not isinstance(p, RepresentationPacket) for p in pkts):
            raise ValueError('fuse() accepts RepresentationPacket instances only')
        out: dict[str, np.ndarray] = {}
        for p in pkts:
            if p.kind not in self.weights:
                self.ignored += 1
                continue
            if only_nonzero and self.weights[p.kind] == 0.0:
                continue
            rel = p.reliability
            if rel <= 0.0:
                continue
            proj = self.projection(p.kind, p.dim)
            if p.form == 'index':  # sum of selected columns == Proj @ dense 0/1 vector
                c = rel * proj[:, np.asarray(p.payload, dtype=np.int64)].sum(axis=1)
            else:
                c = rel * (proj @ p.dense())
            out[p.kind] = out[p.kind] + c if p.kind in out else c
        return out

    def _check_drive(self, scalar_drive: Any) -> np.ndarray:
        d = np.asarray(scalar_drive, dtype=np.float64)
        if d.shape != (self.dim,) or not np.all(np.isfinite(d)):
            raise ValueError(f'scalar_drive must be a finite vector of length {self.dim}')
        return d

    def fuse(self, scalar_drive: Any, packets: Iterable[RepresentationPacket] = ()) -> np.ndarray:
        d = self._check_drive(scalar_drive)
        out = d.copy()
        for kind, c in self.contributions(packets, only_nonzero=True).items():
            w = self.weights[kind]
            if w != 0.0:
                out = out + w * c
        return out

    def learn(self, scalar_drive: Any, packets: Sequence[RepresentationPacket], target: Any) -> dict[str, float]:
        """Local delta rule ``w_k += lr * <target - fused, c_k> / (|c_k|^2 + dim)`` (learned mode)."""
        if self.mode != 'learned':
            raise ValueError("learn() requires mode='learned'")
        d = self._check_drive(scalar_drive)
        t = self._check_drive(target)
        contrib = self.contributions(packets)
        fused = d.copy()
        for kind, c in contrib.items():
            fused = fused + self.weights[kind] * c
        err = t - fused
        for kind, c in contrib.items():
            step = self.lr * float(err @ c) / (float(c @ c) + 1e-3 * self.dim)
            self.weights[kind] = float(np.clip(self.weights[kind] + step, -self.weight_bound, self.weight_bound))
        self.updates += 1
        return {'loss': float(np.mean(err ** 2))}

    def export_state(self) -> dict[str, Any]:
        return {'schema': 'NeuroWeaveRepresentationFusionV1', 'dim': self.dim, 'kinds': list(self.kinds),
                'mode': self.mode, 'seed': self.seed, 'lr': self.lr, 'weight_bound': self.weight_bound,
                'weights': {k: self.weights[k] for k in self.kinds}, 'updates': self.updates, 'ignored': self.ignored}

    def import_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get('schema') != 'NeuroWeaveRepresentationFusionV1':
            raise ValueError('not a NeuroWeaveRepresentationFusionV1 state')
        if state.get('dim') != self.dim or list(state.get('kinds', [])) != list(self.kinds):
            raise ValueError('dim/kinds mismatch')
        if state.get('mode') not in self.MODES or state.get('seed') != self.seed:
            raise ValueError('mode/seed mismatch')
        w = state.get('weights')
        if not isinstance(w, Mapping) or set(w) != set(self.kinds):
            raise ValueError('weights must cover exactly the managed kinds')
        bound = state.get('weight_bound')
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not (0 < bound <= 100):
            raise ValueError('weight_bound invalid')
        new_w = {}
        for k in self.kinds:
            v = w[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > bound:
                raise ValueError(f'weight {k} invalid')
            new_w[k] = float(v)
        lr = state.get('lr')
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not (0 < lr <= 1):
            raise ValueError('lr invalid')
        counts = []
        for key in ('updates', 'ignored'):
            v = state.get(key)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f'{key} invalid')
            counts.append(v)
        self.mode, self.lr, self.weight_bound, self.weights = state['mode'], float(lr), float(bound), new_w
        self.updates, self.ignored = counts


__all__ = [
    'PACKET_SCHEMA', 'PACKET_VERSION', 'PACKET_KINDS', 'PROVENANCES', 'FORMS', 'RepresentationPacket',
    'to_neurowave_x2', 'from_neurowave_x2', 'encode_scalar', 'encode_vector', 'encode_spectral', 'encode_phase',
    'encode_population', 'decode_population', 'encode_sparse', 'vsa_vector', 'vsa_bind', 'vsa_unbind',
    'encode_vsa', 'encode_event', 'encode_spatial', 'encode_graph', 'encode_temporal', 'RepresentationFusion',
]
