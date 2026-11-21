"""Structured dendritic layer (dANN) compared with a dense MLP at matched trainable parameters.

Inspired by (not a reproduction of): S. Chavlis and P. Poirazi, "Dendrites endow artificial
neural networks with accurate, robust and parameter-efficient learning", Nature
Communications 16, 943 (22 January 2025), https://doi.org/10.1038/s41467-025-56297-9
(preprint arXiv:2404.03708).

What is implemented
-------------------
``DendriticLayer`` is a two-stage sparse layer: ``somas * dendrites_per_soma`` dendrites each
receive exactly ``synapses_per_dendrite`` inputs through a *fixed* binary synaptic mask
(``W <- W * M``, applied in the forward pass so masked weights get no gradient, and re-applied
after every optimiser step as in the paper's Eq. 1/3); each dendrite applies a leaky ReLU
(slope 0.1, as in the paper); each soma sums its own dendrites through trainable cable weights
(every dendrite belongs to exactly one soma) and applies a second leaky ReLU::

    d_k = lrelu(sum_i M_ki W_ki x_i + b_k)              (dendrite k)
    s_j = lrelu(sum_{k in dendrites(j)} c_jk d_k + a_j)  (soma j)

Input sampling (fixed at construction from a seeded NumPy generator):

* ``random``: each dendrite samples ``synapses_per_dendrite`` distinct inputs uniformly;
* ``local``: each dendrite gets a uniformly drawn centre pixel and samples the nearest pixels
  (its 3x3 block, subsampled, or expanded ring by ring with random fill), following the
  authors' released ``receptive_fields.py``;
* ``global``: each soma gets a centre pixel; its dendrites' centres are drawn from the
  neighbourhood of that centre by the same ring rule, and each dendrite then samples
  locally around its own centre.

``parameter_count()`` counts only unmasked synapses plus biases and cable weights.

Benchmark (``run_experiment``)
------------------------------
Procedurally generated 12x12 glyphs (10 stroke-drawn classes, no downloads) with random
translation, scale, small rotation, stroke-thickness jitter and Gaussian pixel noise. Arms:
one dANN per sampling mode and a dense two-hidden-layer MLP whose width is chosen so its
trainable parameters match the dANN's (within ``param_tolerance``). All arms share data,
minibatch order, optimiser (Adam) and number of updates. Metrics: held-out accuracy and
held-out accuracy under extra pixel noise. The best dANN is chosen by mean *validation*
accuracy; the verdict is ``harness.compare`` (paired 95 % t interval over seeds) of that dANN
against the dense MLP on held-out accuracy; robustness gets its own comparison.

Synthetic abstraction only: the paper's datasets (MNIST variants, CIFAR-10), sizes and
efficiency scores are not reproduced.
"""
from __future__ import annotations

import contextlib
import functools
import math
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import harness

__all__ = ['MODULE_ID', 'EXPERIMENTS', 'SAMPLING_MODES', 'GLYPHS', 'GLYPH_NAMES', 'GLYPH_SIDE', 'DEFAULTS', 'QUICK',
           'sample_receptive_fields', 'DendriticLayer', 'DendriticNet', 'DenseMLP', 'dense_parameter_count',
           'dann_parameter_count', 'matched_dense_hidden', 'render_glyph', 'make_glyph_dataset', 'train_classifier',
           'accuracy', 'run_experiment']

MODULE_ID = 'dendritic_layer'
SAMPLING_MODES = ('random', 'local', 'global')
NEGATIVE_SLOPE = 0.1                   # leaky ReLU slope used for dendrites and somata in the paper
GLYPH_SIDE = 12

_MASK_OFFSET = 41_011
_INIT_OFFSET = 41_113
_DATA_OFFSET = 41_201
_NOISE_OFFSET = 41_303
_STREAM_OFFSET = 41_407

# Ten stroke glyphs as polylines in the unit square (x to the right, y downwards).
GLYPHS: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {
    'box': (((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)),),
    'bar': (((0.5, 0), (0.5, 1)),),
    'plus': (((0.5, 0), (0.5, 1)), ((0, 0.5), (1, 0.5))),
    'cross': (((0, 0), (1, 1)), ((1, 0), (0, 1))),
    'ell': (((0, 0), (0, 1), (1, 1)),),
    'tee': (((0, 0), (1, 0)), ((0.5, 0), (0.5, 1))),
    'zed': (((0, 0), (1, 0), (0, 1), (1, 1)),),
    'caret': (((0, 1), (0.5, 0), (1, 1)),),
    'aitch': (((0, 0), (0, 1)), ((1, 0), (1, 1)), ((0, 0.5), (1, 0.5))),
    'cup': (((0, 0), (0, 1), (1, 1), (1, 0)),),
}
GLYPH_NAMES: tuple[str, ...] = tuple(GLYPHS)


def _segments(name: str) -> np.ndarray:
    segs = [(a, b) for line in GLYPHS[name] for a, b in zip(line[:-1], line[1:])]
    return np.asarray(segs, dtype=np.float64)                  # (K, 2 endpoints, 2 coords)


_SEGMENTS = {name: _segments(name) for name in GLYPH_NAMES}


def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


# ====================================================================== receptive fields
def _ring(center: tuple[int, int], radius: int, shape: tuple[int, int]) -> list[int]:
    """Flat indices at Chebyshev distance exactly ``radius`` (0 = the centre) inside the image."""
    r0, c0 = center
    h, w = shape
    out = []
    for r in range(max(0, r0 - radius), min(h, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(w, c0 + radius + 1)):
            if max(abs(r - r0), abs(c - c0)) == radius:
                out.append(r * w + c)
    return out


def _neighbourhood(center: tuple[int, int], count: int, shape: tuple[int, int], rng: np.random.Generator) -> list[int]:
    """``count`` distinct pixels nearest ``center``: its 3x3 block (subsampled when larger than needed),
    then rings at Chebyshev radius 2, 3, ... with a random subset of the last ring needed."""
    h, w = shape
    if count > h * w:
        raise ValueError('neighbourhood larger than the image')
    block = _ring(center, 0, shape) + _ring(center, 1, shape)
    if len(block) >= count:
        return sorted(int(i) for i in rng.choice(block, size=count, replace=False))
    chosen = list(block)
    radius = 2
    while len(chosen) < count:
        ring = _ring(center, radius, shape)
        need = count - len(chosen)
        chosen.extend(ring if len(ring) <= need else (int(i) for i in rng.choice(ring, size=need, replace=False)))
        radius += 1
    return sorted(chosen)


def sample_receptive_fields(in_features: int, somas: int, dendrites_per_soma: int, synapses_per_dendrite: int,
                            sampling: str, *, image_shape: tuple[int, int] | None = None,
                            rng: np.random.Generator) -> np.ndarray:
    """Input indices per dendrite, shape ``(somas * dendrites_per_soma, synapses_per_dendrite)``, sorted per row.

    Dendrites ``j * dendrites_per_soma ... (j + 1) * dendrites_per_soma - 1`` belong to soma ``j``."""
    _check_int('in_features', in_features, 1, 1 << 20)
    _check_int('somas', somas, 1, 1 << 16)
    _check_int('dendrites_per_soma', dendrites_per_soma, 1, 1 << 12)
    _check_int('synapses_per_dendrite', synapses_per_dendrite, 1, in_features)
    if sampling not in SAMPLING_MODES:
        raise ValueError(f'sampling must be one of {SAMPLING_MODES}')
    shape = _image_shape(in_features, image_shape)
    if sampling == 'global' and dendrites_per_soma > in_features:
        raise ValueError('global sampling needs dendrites_per_soma <= in_features (distinct dendrite centres)')
    n_dend = somas * dendrites_per_soma
    fields = np.zeros((n_dend, synapses_per_dendrite), dtype=np.int64)
    h, w = shape
    if sampling == 'random':
        for k in range(n_dend):
            fields[k] = np.sort(rng.choice(in_features, size=synapses_per_dendrite, replace=False))
    elif sampling == 'local':
        for k in range(n_dend):
            c = int(rng.integers(in_features))
            fields[k] = _neighbourhood((c // w, c % w), synapses_per_dendrite, shape, rng)
    else:
        for j in range(somas):
            c = int(rng.integers(in_features))
            centres = _neighbourhood((c // w, c % w), dendrites_per_soma, shape, rng)
            order = rng.permutation(len(centres))
            for i, ci in enumerate(order):
                cc = centres[int(ci)]
                fields[j * dendrites_per_soma + i] = _neighbourhood((cc // w, cc % w), synapses_per_dendrite, shape, rng)
    return fields


def _image_shape(in_features: int, image_shape: tuple[int, int] | None) -> tuple[int, int]:
    if image_shape is None:
        side = math.isqrt(in_features)
        return (side, side) if side * side == in_features else (1, in_features)
    if (not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2
            or any(type(v) is not int or v < 1 for v in image_shape) or image_shape[0] * image_shape[1] != in_features):
        raise ValueError('image_shape must be two positive ints whose product is in_features')
    return int(image_shape[0]), int(image_shape[1])


# ====================================================================== models
class _Linear(nn.Module):
    """Affine map initialised like ``nn.Linear`` but from an explicit generator (never the global RNG)."""

    def __init__(self, in_features: int, out_features: int, generator: torch.Generator) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound, generator=generator))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class DendriticLayer(nn.Module):
    """Dendrites with fixed sparse receptive fields feeding somata that sum their own dendrites.

    ``synaptic_mask`` (buffer, 0/1, shape ``(somas * dendrites_per_soma, in_features)``) has exactly
    ``synapses_per_dendrite`` ones per row and never changes; ``cable_weight`` has shape
    ``(somas, dendrites_per_soma)`` (each dendrite connects to one soma)."""

    def __init__(self, in_features: int, somas: int, dendrites_per_soma: int, synapses_per_dendrite: int,
                 sampling: str = 'random', *, image_shape: tuple[int, int] | None = None, seed: int = 0,
                 negative_slope: float = NEGATIVE_SLOPE) -> None:
        super().__init__()
        _check_int('seed', seed, 0, 2**31 - 1)
        if isinstance(negative_slope, bool) or not isinstance(negative_slope, (int, float)) or not 0.0 <= negative_slope < 1.0:
            raise ValueError('negative_slope must be in [0, 1)')
        fields = sample_receptive_fields(in_features, somas, dendrites_per_soma, synapses_per_dendrite, sampling,
                                         image_shape=image_shape, rng=np.random.default_rng(seed + _MASK_OFFSET))
        self.in_features, self.somas = in_features, somas
        self.dendrites_per_soma, self.synapses_per_dendrite = dendrites_per_soma, synapses_per_dendrite
        self.sampling, self.seed, self.negative_slope = sampling, seed, float(negative_slope)
        self.image_shape = _image_shape(in_features, image_shape)
        n_dend = somas * dendrites_per_soma
        mask = torch.zeros(n_dend, in_features)
        mask[torch.arange(n_dend).unsqueeze(1), torch.from_numpy(fields)] = 1.0
        self.register_buffer('synaptic_mask', mask)
        self.register_buffer('fields', torch.from_numpy(fields))
        g = torch.Generator().manual_seed(seed + _INIT_OFFSET)
        bs, bc = 1.0 / math.sqrt(synapses_per_dendrite), 1.0 / math.sqrt(dendrites_per_soma)
        self.synaptic_weight = nn.Parameter(torch.empty(n_dend, in_features).uniform_(-bs, bs, generator=g) * mask)
        self.dendritic_bias = nn.Parameter(torch.empty(n_dend).uniform_(-bs, bs, generator=g))
        self.cable_weight = nn.Parameter(torch.empty(somas, dendrites_per_soma).uniform_(-bc, bc, generator=g))
        self.soma_bias = nn.Parameter(torch.empty(somas).uniform_(-bc, bc, generator=g))

    def dendritic_activity(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f'x must be a tensor of shape (batch, {self.in_features})')
        return F.leaky_relu(F.linear(x, self.synaptic_weight * self.synaptic_mask, self.dendritic_bias),
                            self.negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dendritic_activity(x).view(-1, self.somas, self.dendrites_per_soma)
        return F.leaky_relu((d * self.cable_weight).sum(-1) + self.soma_bias, self.negative_slope)

    @torch.no_grad()
    def enforce_mask(self) -> None:
        """Re-apply ``W <- W * M`` (a no-op unless something wrote into masked weights)."""
        self.synaptic_weight.mul_(self.synaptic_mask)

    def cable_mask(self) -> torch.Tensor:
        """Dense 0/1 soma-by-dendrite connectivity implied by the tree (each dendrite -> one soma)."""
        return torch.repeat_interleave(torch.eye(self.somas), self.dendrites_per_soma, dim=1)

    def parameter_count(self) -> int:
        """Trainable parameters that exist in the model: unmasked synapses, biases, cable weights."""
        return int(self.synaptic_mask.sum().item()) + self.dendritic_bias.numel() + self.cable_weight.numel() \
            + self.soma_bias.numel()

    def mask_density(self) -> float:
        return float(self.synaptic_mask.mean().item())

    def macs(self) -> int:
        """Multiply-accumulates per example that a sparse implementation would need (masked entries excluded)."""
        return int(self.synaptic_mask.sum().item()) + self.cable_weight.numel()


class DendriticNet(nn.Module):
    """``DendriticLayer`` followed by a dense linear read-out over the somata."""

    def __init__(self, in_features: int, classes: int, somas: int, dendrites_per_soma: int, synapses_per_dendrite: int,
                 sampling: str = 'random', *, image_shape: tuple[int, int] | None = None, seed: int = 0) -> None:
        super().__init__()
        _check_int('classes', classes, 2, 1 << 16)
        self.dendritic = DendriticLayer(in_features, somas, dendrites_per_soma, synapses_per_dendrite, sampling,
                                        image_shape=image_shape, seed=seed)
        g = torch.Generator().manual_seed(seed + _INIT_OFFSET + 1)
        self.readout = _Linear(somas, classes, g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.dendritic(x))

    def enforce_mask(self) -> None:
        self.dendritic.enforce_mask()

    def parameter_count(self) -> int:
        return self.dendritic.parameter_count() + self.readout.weight.numel() + self.readout.bias.numel()

    def macs(self) -> int:
        return self.dendritic.macs() + self.readout.weight.numel()


def dense_parameter_count(in_features: int, classes: int, hidden: int) -> int:
    """Trainable parameters of ``DenseMLP(in_features, classes, hidden)``."""
    return in_features * hidden + hidden + hidden * hidden + hidden + hidden * classes + classes


def dann_parameter_count(somas: int, dendrites_per_soma: int, synapses_per_dendrite: int, classes: int) -> int:
    """Trainable parameters of ``DendriticNet`` (existing synapses, dendritic biases, cable weights, somatic
    biases, read-out); independent of the input size and of the sampling mode."""
    dendrites = somas * dendrites_per_soma
    return dendrites * synapses_per_dendrite + 2 * dendrites + somas + somas * classes + classes


class DenseMLP(nn.Module):
    """Dense control: two hidden layers of width ``hidden`` with the same leaky ReLU, then a linear read-out."""

    def __init__(self, in_features: int, classes: int, hidden: int, *, seed: int = 0) -> None:
        super().__init__()
        self.in_features = _check_int('in_features', in_features, 1, 1 << 20)
        _check_int('classes', classes, 2, 1 << 16)
        self.hidden = _check_int('hidden', hidden, 1, 1 << 14)
        _check_int('seed', seed, 0, 2**31 - 1)
        g = torch.Generator().manual_seed(seed + _INIT_OFFSET + 2)
        self.fc1, self.fc2, self.readout = _Linear(in_features, hidden, g), _Linear(hidden, hidden, g), _Linear(hidden, classes, g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f'x must be a tensor of shape (batch, {self.in_features})')
        h = F.leaky_relu(self.fc1(x), NEGATIVE_SLOPE)
        return self.readout(F.leaky_relu(self.fc2(h), NEGATIVE_SLOPE))

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def macs(self) -> int:
        return int(self.fc1.weight.numel() + self.fc2.weight.numel() + self.readout.weight.numel())


def matched_dense_hidden(target_params: int, in_features: int, classes: int) -> int:
    """Width whose ``DenseMLP`` parameter count is closest to ``target_params``."""
    _check_int('target_params', target_params, 1, 1 << 40)
    best, gap = 1, None
    for h in range(1, 1 << 14):
        n = dense_parameter_count(in_features, classes, h)
        if gap is None or abs(n - target_params) < gap:
            best, gap = h, abs(n - target_params)
        if n > target_params:
            break
    return best


# ====================================================================== glyph data
def _segment_distance(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance from each point (P, 2) to the nearest segment (K, 2, 2)."""
    a, b = segs[:, 0][None], segs[:, 1][None]                     # (1, K, 2)
    p = points[:, None, :]                                          # (P, 1, 2)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.sqrt(((p - proj) ** 2).sum(-1)).min(axis=1)


_PIXELS = np.stack(np.meshgrid(np.arange(GLYPH_SIDE) + 0.5, np.arange(GLYPH_SIDE) + 0.5, indexing='xy'), -1).reshape(-1, 2)


def render_glyph(label: int, rng: np.random.Generator, *, noise: float = 0.0, shift: float = 2.0,
                 size: tuple[float, float] = (6.0, 8.5), thickness: tuple[float, float] = (0.35, 0.9),
                 rotation: float = 0.25) -> np.ndarray:
    """One anti-aliased 12x12 glyph (flattened, float64) with random placement, scale, rotation (radians),
    half stroke width and additive Gaussian pixel noise."""
    _check_int('label', label, 0, len(GLYPH_NAMES) - 1)
    segs = _SEGMENTS[GLYPH_NAMES[label]]
    s = rng.uniform(*size)
    theta = rng.uniform(-rotation, rotation)
    cx, cy = GLYPH_SIDE / 2 + rng.uniform(-shift, shift, size=2)
    half = rng.uniform(*thickness)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    placed = ((segs - 0.5) * s) @ rot.T + np.array([cx, cy])
    img = np.clip(half + 0.5 - _segment_distance(_PIXELS, placed), 0.0, 1.0)
    if noise > 0:
        img = img + rng.normal(0.0, noise, size=img.shape)
    return img


def make_glyph_dataset(n_per_class: int, *, seed: int, noise: float = 0.2, **render_kw: Any
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Balanced, shuffled glyph set: ``x`` float32 ``(10 * n_per_class, 144)``, ``y`` int64 labels."""
    _check_int('n_per_class', n_per_class, 1, 1 << 20)
    _check_int('seed', seed, 0, 2**40)
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not 0.0 <= noise <= 10.0:
        raise ValueError('noise must be in [0, 10]')
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(GLYPH_NAMES)), n_per_class)
    labels = labels[rng.permutation(labels.size)]
    x = np.stack([render_glyph(int(c), rng, noise=float(noise), **render_kw) for c in labels])
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))


# ====================================================================== training / evaluation
@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(x).argmax(-1) == y).float().mean())


@functools.lru_cache(maxsize=1)
def _fused_adam_available() -> bool:
    """Whether this torch build runs the fused CPU Adam kernel (several times faster for tiny models)."""
    try:
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.ones(2)
        torch.optim.Adam([p], lr=1e-3, fused=True).step()
        return bool(torch.isfinite(p).all())
    except (RuntimeError, TypeError, ValueError):
        return False


def train_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, updates: int, batch_size: int, lr: float,
                     generator: torch.Generator) -> dict[str, Any]:
    """Adam on cross-entropy over a deterministic shuffled minibatch stream; masks re-applied after each step."""
    _check_int('updates', updates, 1, 1 << 20)
    _check_int('batch_size', batch_size, 1, 1 << 16)
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not 0.0 < float(lr) <= 1.0:
        raise ValueError('lr must be a number in (0, 1]')
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] < 1:
        raise ValueError('x must be (n, features) and y (n,) with n >= 1')
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), **({'fused': True} if _fused_adam_available() else {}))
    enforce = getattr(model, 'enforce_mask', None)
    order = torch.empty(0, dtype=torch.long)
    losses = []
    model.train()
    for _ in range(updates):
        if order.numel() < batch_size:
            order = torch.cat([order, torch.randperm(x.shape[0], generator=generator)])
        idx, order = order[:batch_size], order[batch_size:]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if enforce is not None:
            enforce()
        losses.append(float(loss.detach()))
    return {'updates': updates, 'final_loss': float(np.mean(losses[-max(1, len(losses) // 10):]))}


# ====================================================================== experiment
DEFAULTS: dict[str, Any] = {
    'train_per_class': 60, 'val_per_class': 50, 'test_per_class': 200, 'noise': 0.2, 'extra_noise': 0.5,
    'somas': 16, 'dendrites_per_soma': 8, 'synapses_per_dendrite': 9,
    'updates': 1000, 'batch_size': 64, 'lr': 3e-3, 'param_tolerance': 0.10,
}
QUICK: dict[str, Any] = {**DEFAULTS, 'train_per_class': 40, 'val_per_class': 20, 'test_per_class': 50, 'updates': 250}
_INT_KEYS = {'train_per_class': (2, 100_000), 'val_per_class': (2, 100_000), 'test_per_class': (2, 100_000),
             'somas': (1, 4096), 'dendrites_per_soma': (1, GLYPH_SIDE ** 2),
             'synapses_per_dendrite': (1, GLYPH_SIDE ** 2), 'updates': (1, 1_000_000), 'batch_size': (1, 65_536)}
_FLOAT_KEYS = {'noise': (0.0, 5.0), 'extra_noise': (0.0, 5.0), 'lr': (1e-6, 1.0), 'param_tolerance': (0.0, 1.0)}


def _config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(QUICK if quick else DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f'unknown dendritic_layer options: {sorted(unknown)}')
    cfg.update(overrides)
    for k, (lo, hi) in _INT_KEYS.items():
        _check_int(k, cfg[k], lo, hi)
    for k, (lo, hi) in _FLOAT_KEYS.items():
        v = cfg[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= v <= hi:
            raise ValueError(f'{k} must be a number in [{lo}, {hi}]')
        cfg[k] = float(v)
    return cfg


@contextlib.contextmanager
def _single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    try:
        if previous != 1:
            torch.set_num_threads(1)
        yield
    finally:
        if torch.get_num_threads() != previous:
            torch.set_num_threads(previous)


def _seed_data(seed: int, cfg: Mapping[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    base = seed * 10 + _DATA_OFFSET
    data = {'train': make_glyph_dataset(cfg['train_per_class'], seed=base, noise=cfg['noise']),
            'val': make_glyph_dataset(cfg['val_per_class'], seed=base + 1, noise=cfg['noise']),
            'test': make_glyph_dataset(cfg['test_per_class'], seed=base + 2, noise=cfg['noise'])}
    xt, yt = data['test']
    g = torch.Generator().manual_seed(seed + _NOISE_OFFSET)
    data['test_noisy'] = (xt + cfg['extra_noise'] * torch.randn(xt.shape, generator=g), yt)
    return data


def _budget(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Parameter budget of every arm, computed before any training; ``ValueError`` when the dense control
    cannot be matched within ``param_tolerance`` (e.g. a dANN smaller than the narrowest dense MLP)."""
    in_features, classes = GLYPH_SIDE ** 2, len(GLYPH_NAMES)
    target = dann_parameter_count(cfg['somas'], cfg['dendrites_per_soma'], cfg['synapses_per_dendrite'], classes)
    hidden = matched_dense_hidden(target, in_features, classes)
    dense = dense_parameter_count(in_features, classes, hidden)
    gap = abs(dense - target) / target
    if gap > cfg['param_tolerance']:
        raise ValueError(f'the dense MLP cannot match the dANN budget: {dense} vs {target} parameters '
                         f'(gap {gap:.3f} > param_tolerance {cfg["param_tolerance"]})')
    params = {f'dann_{m}': target for m in SAMPLING_MODES}
    params['dense_mlp'] = dense
    return {'trainable_parameters': params, 'dense_hidden': hidden, 'max_relative_gap': gap}


def _run_seed(seed: int, cfg: dict[str, Any], hidden: int) -> dict[str, Any]:
    data = _seed_data(seed, cfg)
    x_train, y_train = data['train']
    in_features, classes = x_train.shape[1], len(GLYPH_NAMES)
    shape = dict(somas=cfg['somas'], dendrites_per_soma=cfg['dendrites_per_soma'],
                 synapses_per_dendrite=cfg['synapses_per_dendrite'])
    models: dict[str, nn.Module] = {f'dann_{m}': DendriticNet(in_features, classes, sampling=m, seed=seed, **shape)
                                    for m in SAMPLING_MODES}
    models['dense_mlp'] = DenseMLP(in_features, classes, hidden, seed=seed)
    arms = {}
    for name, model in models.items():
        gen = torch.Generator().manual_seed(seed + _STREAM_OFFSET)          # same minibatch order for every arm
        t0 = time.perf_counter()
        log = train_classifier(model, x_train, y_train, updates=cfg['updates'], batch_size=cfg['batch_size'],
                               lr=cfg['lr'], generator=gen)
        t1 = time.perf_counter()
        metrics = {split: accuracy(model, *data[split]) for split in ('train', 'val', 'test', 'test_noisy')}
        arm = {'seed': int(seed), 'parameters': model.parameter_count(), 'estimated_macs_per_example': model.macs(),
               'training': log, 'train_accuracy': metrics['train'], 'val_accuracy': metrics['val'],
               'heldout_accuracy': metrics['test'], 'heldout_noisy_accuracy': metrics['test_noisy'],
               'measured_wallclock_s': {'train': t1 - t0, 'eval': time.perf_counter() - t1}}
        if isinstance(model, DendriticNet):
            layer = model.dendritic
            arm['mask_density'] = layer.mask_density()
            arm['masked_weights_nonzero'] = int((layer.synaptic_weight.detach() * (1 - layer.synaptic_mask)).ne(0).sum())
        arms[name] = arm
    return {'arms': arms}


def _summary(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    keys = ('train_accuracy', 'val_accuracy', 'heldout_accuracy', 'heldout_noisy_accuracy')
    return {k: float(np.mean([a[k] for a in per_seed])) for k in keys}


def run_experiment(*, quick: bool = False, seeds: Iterable[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """dANN (random / local / global sampling) vs a parameter-matched dense MLP on procedural glyphs.

    Returns a sealed frontier receipt. ``overrides`` may change any key of ``DEFAULTS``."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique non-negative ints')
    cfg = _config(quick, overrides)
    budget = _budget(cfg)                                  # ValueError before any training
    params, gap = budget['trainable_parameters'], budget['max_relative_gap']
    t_start = time.perf_counter()
    with _single_thread():
        runs = [_run_seed(s, cfg, budget['dense_hidden']) for s in seeds]
    names = [f'dann_{m}' for m in SAMPLING_MODES] + ['dense_mlp']
    per_arm = {n: [r['arms'][n] for r in runs] for n in names}
    if any(a['parameters'] != params[n] for n in names for a in per_arm[n]):
        raise RuntimeError('built models do not match the computed parameter budget')
    summaries = {n: _summary(per_arm[n]) for n in names}
    dann_names = names[:-1]
    best = max(dann_names, key=lambda n: (summaries[n]['val_accuracy'], -dann_names.index(n)))
    acc = {n: [a['heldout_accuracy'] for a in per_arm[n]] for n in names}
    noisy = {n: [a['heldout_noisy_accuracy'] for a in per_arm[n]] for n in names}
    comparisons = {'best_dann_vs_dense': harness.compare(acc[best], acc['dense_mlp']),
                   'best_dann_vs_dense_noisy': harness.compare(noisy[best], noisy['dense_mlp'])}
    for n in dann_names:
        comparisons[f'{n}_vs_dense'] = harness.compare(acc[n], acc['dense_mlp'])
        comparisons[f'{n}_vs_dense_noisy'] = harness.compare(noisy[n], noisy['dense_mlp'])
    receipt = harness.base_receipt(MODULE_ID, 'model_comparison', {
        **cfg, 'quick': quick, 'seeds': list(seeds), 'task': 'procedural_glyphs_12x12_10_classes',
        'glyphs': list(GLYPH_NAMES), 'metric': 'heldout_accuracy', 'robustness_metric': 'heldout_noisy_accuracy',
        'render': {'shift_px': 2.0, 'size_px': [6.0, 8.5], 'half_thickness_px': [0.35, 0.9], 'rotation_rad': 0.25},
        'optimizer': 'Adam', 'nonlinearity': f'leaky_relu({NEGATIVE_SLOPE})', 'torch_threads': 1})
    receipt['parameter_budget'] = {'trainable_parameters': params, 'dense_hidden': budget['dense_hidden'],
                                   'max_relative_gap': gap, 'tolerance': cfg['param_tolerance'],
                                   'within_tolerance': bool(gap <= cfg['param_tolerance']),
                                   'dann_mask_density': per_arm['dann_random'][0]['mask_density'],
                                   'counting': 'dANN counts only unmasked synapses + biases + cable + read-out'}
    receipt['arms'] = {n: {'per_seed': per_arm[n], 'summary': summaries[n]} for n in names}
    receipt['best_dann'] = {'arm': best, 'selected_by': 'highest mean validation accuracy (held-out set unused)'}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['best_dann_vs_dense']['verdict']
    receipt['verdict_basis'] = f'best_dann({best})_vs_dense_mlp: paired 95% t interval on held-out accuracy over seeds'
    receipt['robustness_verdict'] = comparisons['best_dann_vs_dense_noisy']['verdict']
    receipt['estimated_cost'] = {'macs_per_example': {n: per_arm[n][0]['estimated_macs_per_example'] for n in names},
                                 'optimizer_updates_per_arm': cfg['updates'],
                                 'implementation': 'dense masked matmul; sparsity is not exploited for speed'}
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t_start}
    receipt['notes'] = (
        'Synthetic glyph benchmark inspired by dendritic ANNs (Chavlis & Poirazi, Nat. Commun. 2025); not a '
        'reproduction (no MNIST/CIFAR, no efficiency scores, no sparse-ANN control). All arms: same data, same '
        'minibatch order, same Adam updates; the dense MLP width is chosen to match trainable parameters. The best '
        'dANN is picked on a separate validation set before the held-out comparison. With few seeds the paired t '
        'interval is wide.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
