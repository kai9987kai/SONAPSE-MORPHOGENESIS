"""Recursive latent refinement: a tiny recursive model with deep supervision and a halting head.

Inspired by (not a reproduction of):

* A. Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks",
  arXiv:2510.04871 (October 2025), https://arxiv.org/abs/2510.04871 -- the Tiny
  Recursive Model (TRM).
* G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, S. Song, Y. Abbasi Yadkori,
  "Hierarchical Reasoning Model", arXiv:2506.21734 (June 2025),
  https://arxiv.org/abs/2506.21734 -- deep supervision with detached segments and a
  Q-learning halting head (ACT).

What is implemented
-------------------
One tiny network ``f`` refines a latent ``z`` and an answer state ``y`` for an embedded
question ``e = E x``. States are ``tokens`` vectors of size ``latent_dim`` (one per cell for
Sudoku; ``tokens=1`` gives a plain vector model). ``f`` is a residual block with a
parameter-free norm: a token-mixing MLP across positions (MLP-Mixer style, as TRM uses
for Sudoku instead of self-attention; skipped when ``tokens == 1``) followed by a
per-position channel MLP.

* latent recursion: ``n`` times ``z <- f(z + y + e)``, then once ``y <- f(y + z)``. The
  presence or absence of ``e`` in the sum is what tells the shared network which of the
  two updates it is doing (as in TRM);
* one supervision step: ``T - 1`` latent recursions without gradient, then one with
  gradient; the answer logits ``O y`` (per token) and a halting logit
  ``q = w . mean_tokens(y) + b`` are read from ``y``; ``(y, z)`` are detached before the
  next supervision step;
* deep supervision: every supervision step takes its own loss (cross-entropy on the
  answer plus binary cross-entropy of ``sigmoid(q)`` against "the whole answer is
  correct") and its own optimiser step; up to ``N_sup`` steps per example;
* halting (ACT, TRM variant without the Q-continue head): during training an example
  leaves the batch when ``q > 0`` (after a random minimum number of steps with
  probability ``halt_exploration``, as in HRM) or after ``N_sup`` steps and its slot is
  refilled with a fresh example; evaluation runs all ``N_sup`` steps and also reports
  the answer at the first step whose halting probability exceeds 0.5.

Benchmark (``run_experiment``)
------------------------------
4x4 Sudoku completion. All 288 valid 4x4 grids are enumerated, split by *grid* into
disjoint train / validation / held-out sets, and cells are blanked (keeping a unique
solution among all 288 grids) to make puzzles. Four arms at matched trainable
parameters (within ``param_tolerance``):

* ``trm``: the recursive model trained with deep supervision and halting;
* ``unrolled``: the *same* network and initial weights run for the same inference depth,
  trained with one loss at the end and full backpropagation through the unroll (no deep
  supervision);
* ``shallow``: the same network and initial weights with a single latent recursion
  (``n + 1`` applications of ``f``) and one loss (no recursion beyond one pass);
* ``mlp``: a single-pass flat MLP whose width matches the trainable parameters.

All arms share the data, the minibatch stream and the number of optimiser updates.
Metrics: held-out exact-solve rate and cell accuracy on blank cells. The verdict compares
``trm`` with the baseline that has the best mean *validation* exact-solve
(``harness.compare``, paired 95 % t interval over seeds).

Everything here is a synthetic computational benchmark; it does not reproduce the
papers' results (their Sudoku is 9x9 "Sudoku-Extreme" with 2-layer SwiGLU/RMSNorm mixer
blocks, EMA weights and heavy augmentation, none of which is used here).
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

__all__ = ['MODULE_ID', 'EXPERIMENTS', 'ARMS', 'GRID_COUNT', 'CELLS', 'DIGITS', 'INPUT_DIM', 'ANSWER_DIM', 'UNITS',
           'DEFAULTS', 'QUICK', 'enumerate_sudoku4', 'is_valid_sudoku4', 'is_consistent_puzzle', 'solution_count',
           'make_puzzles', 'encode_puzzles', 'TinyRecursiveModel', 'SinglePassMLP', 'matched_mlp_hidden',
           'train_recursive', 'train_unrolled', 'train_single_pass', 'evaluate_recursive', 'evaluate_single_pass',
           'run_experiment']

MODULE_ID = 'recursive_refiner'
ARMS = ('trm', 'unrolled', 'shallow', 'mlp')
SIDE = 4
CELLS = 16
DIGITS = 4
GRID_COUNT = 288                       # number of valid 4x4 Sudoku grids (verified by enumeration in the tests)
INPUT_DIM = CELLS * (DIGITS + 1)      # one-hot per cell over {blank, 1, 2, 3, 4}
ANSWER_DIM = CELLS * DIGITS           # logits per cell over {1, 2, 3, 4}

_SPLIT_OFFSET = 31_001
_PUZZLE_OFFSET = 31_103
_INIT_OFFSET = 31_207
_MLP_INIT_OFFSET = 31_259
_STREAM_OFFSET = 31_301
_HALT_OFFSET = 31_337


def _units() -> tuple[tuple[int, ...], ...]:
    rows = [tuple(r * SIDE + c for c in range(SIDE)) for r in range(SIDE)]
    cols = [tuple(r * SIDE + c for r in range(SIDE)) for c in range(SIDE)]
    boxes = [tuple((2 * br + i) * SIDE + 2 * bc + j for i in range(2) for j in range(2))
             for br in range(2) for bc in range(2)]
    return tuple(rows + cols + boxes)


UNITS: tuple[tuple[int, ...], ...] = _units()     # 4 rows, 4 columns, 4 boxes (cell indices, row-major)
_PEERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(sorted({p for u in UNITS if c in u for p in u} - {c})) for c in range(CELLS))


# ====================================================================== Sudoku utilities
@functools.lru_cache(maxsize=1)
def _grid_table() -> np.ndarray:
    grid = [0] * CELLS
    out: list[tuple[int, ...]] = []

    def fill(i: int) -> None:
        if i == CELLS:
            out.append(tuple(grid))
            return
        used = {grid[p] for p in _PEERS[i] if p < i}
        for d in range(1, DIGITS + 1):
            if d not in used:
                grid[i] = d
                fill(i + 1)
        grid[i] = 0

    fill(0)
    table = np.asarray(out, dtype=np.int8)
    table.flags.writeable = False
    return table


def enumerate_sudoku4() -> np.ndarray:
    """All valid 4x4 Sudoku grids by backtracking, shape ``(288, 16)``, digits 1..4, lexicographic order."""
    return _grid_table().copy()


def _as_cells(grid: Any, name: str = 'grid') -> np.ndarray:
    g = np.asarray(grid)
    if g.size != CELLS:
        raise ValueError(f'a 4x4 Sudoku {name} has 16 cells')
    if g.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integers')
    return g.reshape(CELLS).astype(np.int64)


def is_valid_sudoku4(grid: Any) -> bool:
    """True when ``grid`` (16 ints, row-major, or 4x4) is a complete valid 4x4 Sudoku."""
    g = _as_cells(grid)
    if g.min() < 1 or g.max() > DIGITS:
        return False
    return all(sorted(g[list(u)].tolist()) == [1, 2, 3, 4] for u in UNITS)


def is_consistent_puzzle(puzzle: Any) -> bool:
    """True when a partial grid (0 = blank, digits 1..4) has no repeated digit in any row, column or box."""
    p = _as_cells(puzzle, 'puzzle')
    if p.min() < 0 or p.max() > DIGITS:
        return False
    for u in UNITS:
        vals = [int(v) for v in p[list(u)] if v]
        if len(vals) != len(set(vals)):
            return False
    return True


@functools.lru_cache(maxsize=1)
def _bitsets() -> tuple[tuple[int, ...], ...]:
    """``bits[c][d]``: bitset over the 288 grids of those with digit ``d`` in cell ``c`` (``d=0`` = all grids)."""
    table = _grid_table()
    full = (1 << len(table)) - 1
    bits = [[full] + [0] * DIGITS for _ in range(CELLS)]
    for gi, g in enumerate(table):
        for c in range(CELLS):
            bits[c][int(g[c])] |= 1 << gi
    return tuple(tuple(b) for b in bits)


def _count(p: np.ndarray) -> int:
    bits = _bitsets()
    mask = bits[0][0]
    for c in range(CELLS):
        d = int(p[c])
        if d:
            mask &= bits[c][d]
    return mask.bit_count()


def solution_count(puzzle: Any) -> int:
    """Number of valid 4x4 grids that agree with the givens of ``puzzle`` (0 = blank)."""
    p = _as_cells(puzzle, 'puzzle')
    if p.min() < 0 or p.max() > DIGITS:
        raise ValueError('puzzle cells must be in 0..4')
    return _count(p)


def make_puzzles(grids: Any, count: int, *, rng: np.random.Generator, min_blanks: int, max_blanks: int,
                 max_attempts: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Blank cells of randomly chosen ``grids`` while the solution stays unique among all 288 grids.

    Each puzzle draws a target number of blanks uniformly from ``[min_blanks, max_blanks]`` and removes
    cells in random order, skipping removals that would admit a second solution. Returns
    ``(puzzles, solutions)``, both ``(count, 16)`` int8 (puzzles use 0 for blank)."""
    g = np.asarray(grids)
    if g.ndim != 2 or g.shape[1] != CELLS or g.shape[0] < 1 or g.dtype.kind not in 'iu':
        raise ValueError('grids must be an integer array of shape (n, 16)')
    if not all(is_valid_sudoku4(row) for row in g):
        raise ValueError('grids must be valid 4x4 Sudoku grids')
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive int')
    if type(min_blanks) is not int or type(max_blanks) is not int or not 1 <= min_blanks <= max_blanks <= 12:
        raise ValueError('need 1 <= min_blanks <= max_blanks <= 12 (a unique 4x4 puzzle keeps at least 4 givens)')
    puzzles = np.zeros((count, CELLS), dtype=np.int8)
    solutions = np.zeros((count, CELLS), dtype=np.int8)
    for k in range(count):
        for _ in range(max_attempts):
            sol = g[int(rng.integers(len(g)))].astype(np.int64)
            target = int(rng.integers(min_blanks, max_blanks + 1))
            p = sol.copy()
            blanks = 0
            for c in rng.permutation(CELLS):
                if blanks == target:
                    break
                keep = p[c]
                p[c] = 0
                if _count(p) == 1:
                    blanks += 1
                else:
                    p[c] = keep
            if blanks >= min_blanks:
                break
        else:
            raise RuntimeError('could not make a unique puzzle with the requested number of blanks')
        puzzles[k] = p
        solutions[k] = sol
    return puzzles, solutions


def encode_puzzles(puzzles: Any) -> torch.Tensor:
    """One-hot encoding over {blank, 1, 2, 3, 4} per cell: float32 tensor of shape ``(n, 80)`` (cell-major)."""
    p = np.asarray(puzzles)
    if p.ndim != 2 or p.shape[1] != CELLS or p.dtype.kind not in 'iu' or p.size == 0:
        raise ValueError('puzzles must be a non-empty integer array of shape (n, 16)')
    if p.min() < 0 or p.max() > DIGITS:
        raise ValueError('puzzle cells must be in 0..4')
    n = p.shape[0]
    onehot = np.zeros((n, CELLS, DIGITS + 1), dtype=np.float32)
    onehot[np.arange(n)[:, None], np.arange(CELLS)[None, :], p.astype(np.int64)] = 1.0
    return torch.from_numpy(onehot.reshape(n, INPUT_DIM))


def _targets(solutions: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(solutions, dtype=np.int64) - 1)


# ====================================================================== models
def _check_int(name: str, value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an int in [{low}, {high}]')
    return value


class _Linear(nn.Module):
    """Affine map initialised like ``nn.Linear`` but from an explicit generator (never the global RNG)."""

    def __init__(self, in_features: int, out_features: int, generator: torch.Generator) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound, generator=generator))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def _norm(h: torch.Tensor) -> torch.Tensor:
    """Parameter-free post-norm (layer norm without affine terms) that keeps recursed states bounded."""
    return F.layer_norm(h, h.shape[-1:])


def _truncated_normal(n: int, generator: torch.Generator, limit: float = 2.0) -> torch.Tensor:
    out = torch.empty(n).normal_(generator=generator)
    for _ in range(64):                      # rejection resampling of the (rare) out-of-range entries
        bad = out.abs() > limit
        if not bool(bad.any()):
            break
        out[bad] = torch.empty(int(bad.sum())).normal_(generator=generator)
    return out.clamp_(-limit, limit)


class TinyRecursiveModel(nn.Module):
    """One tiny network that recursively refines a latent ``z`` and an answer state ``y``.

    Inputs are ``(batch, tokens * input_dim)`` (token-major) and answers ``(batch, tokens * answer_dim)``;
    internally ``x``, ``y`` and ``z`` are ``(batch, tokens, latent_dim)``. The network is
    ``f(s) = norm(t + W2 relu(W1 t + b1) + b2)`` with ``t = norm(s + token_mlp(s))``, where the
    token-mixing MLP (hidden width ``token_hidden``) acts across positions and is skipped when
    ``tokens == 1``; ``norm`` is a layer norm without affine terms (TRM uses RMSNorm and SwiGLU, swapped
    here for speed on CPU). A latent recursion is ``n_latent_steps`` updates ``z <- f(z + y + E x)``
    followed by ``y <- f(y + z)``; a supervision step is ``recursions - 1`` latent recursions without
    gradient plus one with gradient, then the per-token answer head ``O y`` and the halting head
    ``q = w . mean_tokens(y) + b`` (probability ``sigmoid(q)``). ``(y, z)`` are detached between
    supervision steps. ``y_init`` / ``z_init`` are fixed (non-trainable) truncated-normal buffers shared by
    all positions and the halting head starts at ``w = 0, b = -5`` (as in the HRM/TRM code), so early
    training rarely halts. ``net_calls`` counts applications of ``f``."""

    def __init__(self, input_dim: int, answer_dim: int, latent_dim: int, hidden: int, n_latent_steps: int,
                 supervision_steps: int, *, recursions: int = 2, tokens: int = 1, token_hidden: int | None = None,
                 seed: int = 0) -> None:
        super().__init__()
        self.input_dim = _check_int('input_dim', input_dim, 1, 1 << 16)
        self.answer_dim = _check_int('answer_dim', answer_dim, 1, 1 << 16)
        self.latent_dim = _check_int('latent_dim', latent_dim, 2, 4096)
        self.hidden = _check_int('hidden', hidden, 1, 8192)
        self.n_latent_steps = _check_int('n_latent_steps', n_latent_steps, 1, 64)
        self.supervision_steps = _check_int('supervision_steps', supervision_steps, 1, 64)
        self.recursions = _check_int('recursions', recursions, 1, 64)
        self.tokens = _check_int('tokens', tokens, 1, 4096)
        if tokens == 1:
            if token_hidden not in (None, 0):
                raise ValueError('token_hidden needs tokens > 1')
            self.token_hidden = 0
        else:
            self.token_hidden = _check_int('token_hidden', tokens if token_hidden is None else token_hidden, 1, 8192)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        g = torch.Generator().manual_seed(self.seed + _INIT_OFFSET)
        self.input_embedding = _Linear(input_dim, latent_dim, g)
        self.net_in = _Linear(latent_dim, hidden, g)
        self.net_out = _Linear(hidden, latent_dim, g)
        self.output_head = _Linear(latent_dim, answer_dim, g)
        self.q_head = _Linear(latent_dim, 1, g)
        self.token_in = _Linear(tokens, self.token_hidden, g) if tokens > 1 else None
        self.token_out = _Linear(self.token_hidden, tokens, g) if tokens > 1 else None
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
        self.register_buffer('y_init', _truncated_normal(latent_dim, g))
        self.register_buffer('z_init', _truncated_normal(latent_dim, g))
        self.net_calls = 0

    # ------------------------------------------------------------------ building blocks
    def net(self, s: torch.Tensor) -> torch.Tensor:
        """The single shared tiny network ``f`` on states of shape ``(batch, tokens, latent_dim)``."""
        self.net_calls += 1
        if self.token_in is not None:
            mixed = self.token_out(F.relu(self.token_in(s.transpose(1, 2)))).transpose(1, 2)
            s = _norm(s + mixed)
        return _norm(s + self.net_out(F.relu(self.net_in(s))))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        width = self.tokens * self.input_dim
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != width or not x.is_floating_point():
            raise ValueError(f'x must be a float tensor of shape (batch, {width})')
        return self.input_embedding(x.reshape(x.shape[0], self.tokens, self.input_dim))

    def initial_state(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, self.tokens, self.latent_dim)
        return self.y_init.expand(shape).clone(), self.z_init.expand(shape).clone()

    def latent_recursion(self, e: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.n_latent_steps):
            z = self.net(z + y + e)
        y = self.net(y + z)
        return y, z

    def read_out(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Answer logits ``(batch, tokens * answer_dim)`` and halting logits ``(batch,)`` from ``y``."""
        return self.output_head(y).reshape(y.shape[0], -1), self.q_head(y.mean(1)).squeeze(-1)

    def supervision_step(self, e: torch.Tensor, y: torch.Tensor, z: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``recursions - 1`` gradient-free recursions, one with gradient; returns detached ``(y, z)``,
        answer logits and the halting logit."""
        with torch.no_grad():
            for _ in range(self.recursions - 1):
                y, z = self.latent_recursion(e, y, z)
        y, z = self.latent_recursion(e, y, z)
        logits, q = self.read_out(y)
        return y.detach(), z.detach(), logits, q

    def halt_probability(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.read_out(y)[1])

    # ------------------------------------------------------------------ full passes
    def forward(self, x: torch.Tensor, *, steps: int | None = None) -> dict[str, torch.Tensor]:
        """Run ``steps`` (default ``supervision_steps``) supervision steps; returns the final logits, the
        per-step logits ``(steps, batch, tokens * answer_dim)`` and halting probabilities ``(steps, batch)``."""
        steps = self.supervision_steps if steps is None else _check_int('steps', steps, 1, 1024)
        e = self.embed(x)
        y, z = self.initial_state(x.shape[0])
        logits, halts = [], []
        for _ in range(steps):
            y, z, lg, q = self.supervision_step(e, y, z)
            logits.append(lg)
            halts.append(torch.sigmoid(q))
        return {'logits': logits[-1], 'step_logits': torch.stack(logits), 'halt_prob': torch.stack(halts)}

    def unrolled(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The same computation as ``forward`` (``supervision_steps * recursions`` latent recursions) with
        gradient through every recursion and no intermediate read-out; returns final logits and halting logit."""
        e = self.embed(x)
        y, z = self.initial_state(x.shape[0])
        for _ in range(self.supervision_steps * self.recursions):
            y, z = self.latent_recursion(e, y, z)
        return self.read_out(y)

    # ------------------------------------------------------------------ accounting
    def calls_per_supervision_step(self) -> int:
        return self.recursions * (self.n_latent_steps + 1)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def inference_macs(self) -> int:
        """Estimated multiply-accumulates per example for a full ``forward`` (all supervision steps)."""
        d, h, t = self.latent_dim, self.hidden, self.tokens
        net = t * 2 * d * h + (2 * d * t * self.token_hidden if t > 1 else 0)
        per_step = self.calls_per_supervision_step() * net + t * d * self.answer_dim + d
        return int(t * self.input_dim * d + self.supervision_steps * per_step)


class SinglePassMLP(nn.Module):
    """Single-pass baseline: ``depth`` ReLU hidden layers of width ``hidden``, then a linear answer head."""

    def __init__(self, input_dim: int, answer_dim: int, hidden: int, *, depth: int = 2, seed: int = 0) -> None:
        super().__init__()
        self.input_dim = _check_int('input_dim', input_dim, 1, 1 << 16)
        self.answer_dim = _check_int('answer_dim', answer_dim, 1, 1 << 16)
        self.hidden = _check_int('hidden', hidden, 1, 8192)
        self.depth = _check_int('depth', depth, 1, 16)
        self.seed = _check_int('seed', seed, 0, 2**31 - 1)
        g = torch.Generator().manual_seed(self.seed + _MLP_INIT_OFFSET)
        dims = [input_dim] + [hidden] * depth
        self.layers = nn.ModuleList(_Linear(a, b, g) for a, b in zip(dims[:-1], dims[1:]))
        self.head = _Linear(hidden, answer_dim, g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.input_dim or not x.is_floating_point():
            raise ValueError(f'x must be a float tensor of shape (batch, {self.input_dim})')
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.head(h)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def inference_macs(self) -> int:
        return int(self.input_dim * self.hidden + (self.depth - 1) * self.hidden ** 2 + self.hidden * self.answer_dim)


def _mlp_params(hidden: int, input_dim: int, answer_dim: int, depth: int) -> int:
    return input_dim * hidden + hidden + (depth - 1) * (hidden * hidden + hidden) + hidden * answer_dim + answer_dim


def matched_mlp_hidden(target_params: int, input_dim: int, answer_dim: int, depth: int = 2) -> int:
    """Hidden width whose ``SinglePassMLP`` parameter count is closest to ``target_params``."""
    _check_int('target_params', target_params, 1, 1 << 40)
    _check_int('input_dim', input_dim, 1, 1 << 16)
    _check_int('answer_dim', answer_dim, 1, 1 << 16)
    _check_int('depth', depth, 1, 16)
    best, gap = 1, None
    for h in range(1, 8193):
        n = _mlp_params(h, input_dim, answer_dim, depth)
        if gap is None or abs(n - target_params) < gap:
            best, gap = h, abs(n - target_params)
        if n > target_params:
            break
    return best


# ====================================================================== training
class _IndexStream:
    """Deterministic endless stream of shuffled example indices (one permutation per epoch)."""

    def __init__(self, n: int, generator: torch.Generator) -> None:
        self.n = int(n)
        self.g = generator
        self.buf = torch.empty(0, dtype=torch.long)
        self.drawn = 0

    def take(self, k: int) -> torch.Tensor:
        parts = []
        while k > 0:
            if self.buf.numel() == 0:
                self.buf = torch.randperm(self.n, generator=self.g)
            part, self.buf = self.buf[:k], self.buf[k:]
            parts.append(part)
            k -= part.numel()
        out = torch.cat(parts)
        self.drawn += int(out.numel())
        return out


@functools.lru_cache(maxsize=1)
def _fused_adamw_available() -> bool:
    """Whether this torch build runs the fused CPU AdamW kernel (several times faster for tiny models)."""
    try:
        p = nn.Parameter(torch.zeros(2))
        p.grad = torch.ones(2)
        torch.optim.AdamW([p], lr=1e-3, fused=True).step()
        return bool(torch.isfinite(p).all())
    except (RuntimeError, TypeError, ValueError):
        return False


def _optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    kw: dict[str, Any] = dict(lr=float(cfg['lr']), betas=(0.9, 0.95), weight_decay=float(cfg['weight_decay']))
    if _fused_adamw_available():
        kw['fused'] = True
    return torch.optim.AdamW(model.parameters(), **kw)


def _answer_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean cell cross-entropy and the per-example "every cell correct" indicator."""
    cells = logits.view(-1, CELLS, DIGITS)
    ce = F.cross_entropy(cells.reshape(-1, DIGITS), target.reshape(-1))
    exact = (cells.argmax(-1) == target).all(-1)
    return ce, exact


def _step(model: nn.Module, opt: torch.optim.Optimizer, loss: torch.Tensor, clip: float) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()


def _tail_loss(losses: list[float]) -> float:
    return float(np.mean(losses[-max(1, len(losses) // 10):]))


def train_recursive(model: TinyRecursiveModel, x: torch.Tensor, target: torch.Tensor, cfg: Mapping[str, Any],
                    generator: torch.Generator) -> dict[str, Any]:
    """Deep supervision with ACT halting (TRM style).

    A batch of ``batch_size`` slots is carried across optimiser updates. Each update is one supervision
    step for every slot; a slot halts when its halting logit is positive (and its random minimum step
    count is reached) or after ``supervision_steps`` steps, and is then refilled with a fresh example
    from the index stream with re-initialised ``(y, z)``."""
    updates, batch = int(cfg['updates']), int(cfg['batch_size'])
    n_sup, explore, clip = model.supervision_steps, float(cfg['halt_exploration']), float(cfg['grad_clip'])
    opt = _optimizer(model, cfg)
    stream = _IndexStream(x.shape[0], generator)
    halt_gen = torch.Generator().manual_seed(int(cfg['_seed']) + _HALT_OFFSET)

    def min_steps(k: int) -> torch.Tensor:
        explore_mask = torch.rand(k, generator=halt_gen) < explore
        drawn = torch.randint(2, max(3, n_sup + 1), (k,), generator=halt_gen)
        return torch.where(explore_mask, drawn.clamp(max=n_sup), torch.ones(k, dtype=torch.long))

    idx = stream.take(batch)
    xb, tb = x[idx], target[idx]
    y, z = model.initial_state(batch)
    steps = torch.zeros(batch, dtype=torch.long)
    mins = min_steps(batch)
    finished_steps: list[int] = []
    losses: list[float] = []
    model.train()
    for _ in range(updates):
        e = model.embed(xb)
        y, z, logits, q = model.supervision_step(e, y, z)
        ce, exact = _answer_loss(logits, tb)
        loss = ce + F.binary_cross_entropy_with_logits(q, exact.float())
        _step(model, opt, loss, clip)
        losses.append(float(loss.detach()))
        steps += 1
        halted = ((q.detach() > 0) & (steps >= mins)) | (steps >= n_sup)
        k = int(halted.sum())
        if k:
            finished_steps.extend(steps[halted].tolist())
            fresh = stream.take(k)
            xb, tb = xb.clone(), tb.clone()
            xb[halted], tb[halted] = x[fresh], target[fresh]
            reset = halted.view(-1, *([1] * (y.ndim - 1)))
            y = torch.where(reset, model.y_init.expand_as(y), y)
            z = torch.where(reset, model.z_init.expand_as(z), z)
            steps[halted] = 0
            mins[halted] = min_steps(k)
    return {'updates': updates, 'examples_drawn': stream.drawn, 'final_loss': _tail_loss(losses),
            'mean_supervision_steps_per_finished_example': float(np.mean(finished_steps)) if finished_steps else None}


def train_unrolled(model: TinyRecursiveModel, x: torch.Tensor, target: torch.Tensor, cfg: Mapping[str, Any],
                   generator: torch.Generator) -> dict[str, Any]:
    """Same network, no deep supervision: one loss at the end of the full unroll, backpropagated through
    every recursion; one fresh minibatch per optimiser update."""
    updates, batch, clip = int(cfg['updates']), int(cfg['batch_size']), float(cfg['grad_clip'])
    opt = _optimizer(model, cfg)
    stream = _IndexStream(x.shape[0], generator)
    losses: list[float] = []
    model.train()
    for _ in range(updates):
        idx = stream.take(batch)
        logits, q = model.unrolled(x[idx])
        ce, exact = _answer_loss(logits, target[idx])
        loss = ce + F.binary_cross_entropy_with_logits(q, exact.float())
        _step(model, opt, loss, clip)
        losses.append(float(loss.detach()))
    return {'updates': updates, 'examples_drawn': stream.drawn, 'final_loss': _tail_loss(losses)}


def train_single_pass(model: SinglePassMLP, x: torch.Tensor, target: torch.Tensor, cfg: Mapping[str, Any],
                      generator: torch.Generator) -> dict[str, Any]:
    updates, batch, clip = int(cfg['updates']), int(cfg['batch_size']), float(cfg['grad_clip'])
    opt = _optimizer(model, cfg)
    stream = _IndexStream(x.shape[0], generator)
    losses: list[float] = []
    model.train()
    for _ in range(updates):
        idx = stream.take(batch)
        ce, _ = _answer_loss(model(x[idx]), target[idx])
        _step(model, opt, ce, clip)
        losses.append(float(ce.detach()))
    return {'updates': updates, 'examples_drawn': stream.drawn, 'final_loss': _tail_loss(losses)}


# ====================================================================== evaluation
def _cell_metrics(pred: torch.Tensor, target: torch.Tensor, blank: torch.Tensor) -> tuple[float, float]:
    correct = pred == target
    cell_acc = float((correct & blank).sum()) / max(1, int(blank.sum()))
    exact = float(correct.all(-1).float().mean())
    return cell_acc, exact


@torch.no_grad()
def evaluate_recursive(model: TinyRecursiveModel, x: torch.Tensor, target: torch.Tensor,
                       puzzles: np.ndarray) -> dict[str, Any]:
    """Final-step and per-step metrics, halting-head calibration and halting-based (adaptive) answers."""
    model.eval()
    out = model(x)
    steps = out['step_logits'].shape[0]
    pred = out['step_logits'].view(steps, -1, CELLS, DIGITS).argmax(-1)          # (S, N, 16)
    blank = torch.from_numpy(np.asarray(puzzles) == 0)
    per_step_exact = (pred == target).all(-1)                                       # (S, N)
    cell_acc, exact = _cell_metrics(pred[-1], target, blank)
    halt = out['halt_prob']                                                         # (S, N)
    above = halt > 0.5
    first = torch.where(above.any(0), above.float().argmax(0), torch.full_like(above[0], steps - 1, dtype=torch.long))
    chosen = pred.gather(0, first.view(1, -1, 1).expand(1, -1, CELLS)).squeeze(0)
    a_cell, a_exact = _cell_metrics(chosen, target, blank)
    brier = float(((halt - per_step_exact.float()) ** 2).mean())
    decided = float(((halt > 0.5) == per_step_exact).float().mean())
    return {'cell_accuracy': cell_acc, 'exact_solve': exact,
            'per_step_exact_solve': per_step_exact.float().mean(-1).tolist(),
            'adaptive_cell_accuracy': a_cell, 'adaptive_exact_solve': a_exact,
            'adaptive_mean_steps': float((first + 1).float().mean()),
            'halting_brier': brier, 'halting_decision_accuracy': decided,
            'halt_prob_range': [float(halt.min()), float(halt.max())]}


@torch.no_grad()
def evaluate_single_pass(model: SinglePassMLP, x: torch.Tensor, target: torch.Tensor,
                         puzzles: np.ndarray) -> dict[str, Any]:
    model.eval()
    pred = model(x).view(-1, CELLS, DIGITS).argmax(-1)
    blank = torch.from_numpy(np.asarray(puzzles) == 0)
    cell_acc, exact = _cell_metrics(pred, target, blank)
    return {'cell_accuracy': cell_acc, 'exact_solve': exact}


# ====================================================================== experiment
DEFAULTS: dict[str, Any] = {
    'train_grids': 160, 'val_grids': 32, 'test_grids': 96,
    'train_puzzles': 4096, 'val_puzzles': 512, 'test_puzzles': 1024, 'train_eval_puzzles': 512,
    'min_blanks': 8, 'max_blanks': 12,
    'latent_dim': 24, 'hidden': 48, 'token_hidden': 24, 'n_latent_steps': 2, 'recursions': 2,
    'supervision_steps': 4, 'mlp_depth': 2, 'updates': 600, 'batch_size': 32, 'lr': 2e-3, 'weight_decay': 0.01,
    'grad_clip': 1.0, 'halt_exploration': 0.1, 'param_tolerance': 0.10,
}
QUICK: dict[str, Any] = {**DEFAULTS, 'train_puzzles': 1024, 'val_puzzles': 128, 'test_puzzles': 256,
                         'train_eval_puzzles': 128, 'latent_dim': 16, 'hidden': 32, 'token_hidden': 16,
                         'n_latent_steps': 1, 'supervision_steps': 2, 'updates': 100, 'batch_size': 64, 'lr': 4e-3}
_INT_KEYS = {'train_grids': (1, 286), 'val_grids': (1, 286), 'test_grids': (1, 286), 'train_puzzles': (8, 1 << 20),
             'val_puzzles': (8, 1 << 20), 'test_puzzles': (8, 1 << 20), 'train_eval_puzzles': (8, 1 << 20),
             'min_blanks': (1, 12), 'max_blanks': (1, 12), 'latent_dim': (2, 1024), 'hidden': (1, 4096),
             'token_hidden': (1, 1024), 'n_latent_steps': (1, 16), 'recursions': (1, 16), 'supervision_steps': (1, 32),
             'mlp_depth': (1, 8), 'updates': (1, 100_000), 'batch_size': (1, 4096)}
_FLOAT_KEYS = {'lr': (1e-6, 1.0), 'weight_decay': (0.0, 10.0), 'grad_clip': (0.0, 1e6), 'halt_exploration': (0.0, 1.0),
               'param_tolerance': (0.0, 1.0)}


def _config(quick: bool, overrides: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(QUICK if quick else DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f'unknown recursive_refiner options: {sorted(unknown)}')
    cfg.update(overrides)
    for k, (lo, hi) in _INT_KEYS.items():
        _check_int(k, cfg[k], lo, hi)
    for k, (lo, hi) in _FLOAT_KEYS.items():
        v = cfg[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= v <= hi:
            raise ValueError(f'{k} must be a number in [{lo}, {hi}]')
        cfg[k] = float(v)
    if cfg['train_grids'] + cfg['val_grids'] + cfg['test_grids'] > GRID_COUNT:
        raise ValueError('train_grids + val_grids + test_grids must not exceed 288')
    if cfg['min_blanks'] > cfg['max_blanks']:
        raise ValueError('min_blanks must not exceed max_blanks')
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


def _seed_data(seed: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    table = _grid_table()
    order = np.random.default_rng(seed + _SPLIT_OFFSET).permutation(len(table))
    a, b = cfg['train_grids'], cfg['train_grids'] + cfg['val_grids']
    split = {'train': order[:a], 'val': order[a:b], 'test': order[b:b + cfg['test_grids']]}
    rng = np.random.default_rng(seed + _PUZZLE_OFFSET)
    data: dict[str, Any] = {'split': split}
    sizes = {'train': cfg['train_puzzles'], 'val': cfg['val_puzzles'], 'test': cfg['test_puzzles']}
    for name in ('train', 'val', 'test'):
        p, s = make_puzzles(table[split[name]], sizes[name], rng=rng, min_blanks=cfg['min_blanks'],
                            max_blanks=cfg['max_blanks'])
        data[name] = (p, s, encode_puzzles(p), _targets(s))
    k = min(cfg['train_eval_puzzles'], cfg['train_puzzles'])
    p, s, xt, tt = data['train']
    data['train_eval'] = (p[:k], s[:k], xt[:k], tt[:k])
    return data


def _models(seed: int, cfg: Mapping[str, Any]) -> tuple[dict[str, nn.Module], int]:
    """The four arms for one seed: trm / unrolled / shallow share the network and initial weights."""
    net = dict(latent_dim=cfg['latent_dim'], hidden=cfg['hidden'], n_latent_steps=cfg['n_latent_steps'],
               tokens=CELLS, token_hidden=cfg['token_hidden'], seed=seed)
    depth = dict(supervision_steps=cfg['supervision_steps'], recursions=cfg['recursions'])
    models: dict[str, nn.Module] = {
        'trm': TinyRecursiveModel(DIGITS + 1, DIGITS, **net, **depth),
        'unrolled': TinyRecursiveModel(DIGITS + 1, DIGITS, **net, **depth),
        'shallow': TinyRecursiveModel(DIGITS + 1, DIGITS, **net, supervision_steps=1, recursions=1),
    }
    mlp_hidden = matched_mlp_hidden(models['trm'].parameter_count(), INPUT_DIM, ANSWER_DIM, cfg['mlp_depth'])
    models['mlp'] = SinglePassMLP(INPUT_DIM, ANSWER_DIM, mlp_hidden, depth=cfg['mlp_depth'], seed=seed)
    return models, mlp_hidden


def _run_seed(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    data = _seed_data(seed, cfg)
    _, _, x_train, t_train = data['train']
    models, mlp_hidden = _models(seed, cfg)
    plan = {'trm': (train_recursive, evaluate_recursive), 'unrolled': (train_unrolled, evaluate_recursive),
            'shallow': (train_unrolled, evaluate_recursive), 'mlp': (train_single_pass, evaluate_single_pass)}
    tcfg = {**cfg, '_seed': seed}
    arms: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        model = models[name]
        trainer, evaluator = plan[name]
        gen = torch.Generator().manual_seed(seed + _STREAM_OFFSET)      # same example stream for every arm
        t0 = time.perf_counter()
        log = trainer(model, x_train, t_train, tcfg, gen)
        t1 = time.perf_counter()
        evals = {split: evaluator(model, data[split][2], data[split][3], data[split][0])
                 for split in ('train_eval', 'val', 'test')}
        t2 = time.perf_counter()
        arms[name] = {'seed': int(seed), 'parameters': model.parameter_count(), 'training': log,
                      'train': evals['train_eval'], 'val': evals['val'], 'heldout': evals['test'],
                      'estimated_inference_macs_per_puzzle': model.inference_macs(),
                      'measured_wallclock_s': {'train': t1 - t0, 'eval': t2 - t1}}
    blanks = {name: float(np.mean(np.sum(data[name][0] == 0, axis=1))) for name in ('train', 'val', 'test')}
    split = data['split']
    overlap = (len(set(split['train'].tolist()) & set(split['test'].tolist()))
               + len(set(split['train'].tolist()) & set(split['val'].tolist()))
               + len(set(split['val'].tolist()) & set(split['test'].tolist())))
    return {'arms': arms, 'mlp_hidden': mlp_hidden, 'mean_blanks': blanks, 'grid_overlap': int(overlap)}


def _arm_summary(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(split: str, metric: str) -> float | None:
        vals = [r[split].get(metric) for r in per_seed]
        vals = [float(v) for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    out = {f'{split}_{metric}': mean(split, metric) for split in ('train', 'val', 'heldout')
           for metric in ('cell_accuracy', 'exact_solve')}
    if 'adaptive_exact_solve' in per_seed[0]['heldout']:
        out['heldout_adaptive_exact_solve'] = mean('heldout', 'adaptive_exact_solve')
        out['heldout_adaptive_mean_steps'] = mean('heldout', 'adaptive_mean_steps')
        out['heldout_halting_brier'] = mean('heldout', 'halting_brier')
        out['heldout_halting_decision_accuracy'] = mean('heldout', 'halting_decision_accuracy')
        steps = np.mean([r['heldout']['per_step_exact_solve'] for r in per_seed], axis=0)
        out['heldout_per_step_exact_solve'] = [float(v) for v in steps]
    return out


def run_experiment(*, quick: bool = False, seeds: Iterable[int] = (0, 1, 2), **overrides: Any) -> dict[str, Any]:
    """TRM vs the same network unrolled / shallow vs a single-pass MLP on 4x4 Sudoku (held-out grids).

    Returns a sealed frontier receipt. ``overrides`` may change any key of ``DEFAULTS``."""
    if type(quick) is not bool:
        raise ValueError('quick must be a boolean')
    seeds = tuple(seeds)
    if not seeds or any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique non-negative ints')
    cfg = _config(quick, overrides)
    probe, _ = _models(seeds[0], cfg)                      # budgets do not depend on the seed: check before training
    params = {n: probe[n].parameter_count() for n in ARMS}
    gap = max(abs(params[n] - params['trm']) / params['trm'] for n in ARMS)
    if gap > cfg['param_tolerance']:
        raise ValueError(f'the single-pass MLP cannot match the recursive model budget: {params} '
                         f'(gap {gap:.3f} > param_tolerance {cfg["param_tolerance"]})')
    t_start = time.perf_counter()
    with _single_thread():
        runs = [_run_seed(s, cfg) for s in seeds]
    per_arm = {n: [r['arms'][n] for r in runs] for n in ARMS}
    if any(a['parameters'] != params[n] for n in ARMS for a in per_arm[n]):
        raise RuntimeError('built models do not match the computed parameter budget')
    summaries = {n: _arm_summary(per_arm[n]) for n in ARMS}
    baselines = ARMS[1:]
    best = max(baselines, key=lambda n: (summaries[n]['val_exact_solve'], -baselines.index(n)))
    exact = {n: [a['heldout']['exact_solve'] for a in per_arm[n]] for n in ARMS}
    cells = {n: [a['heldout']['cell_accuracy'] for a in per_arm[n]] for n in ARMS}
    comparisons = {'trm_vs_best_baseline': harness.compare(exact['trm'], exact[best])}
    for n in baselines:
        comparisons[f'trm_vs_{n}'] = harness.compare(exact['trm'], exact[n])
    comparisons['trm_vs_best_baseline_cell_accuracy'] = harness.compare(cells['trm'], cells[best])
    receipt = harness.base_receipt(MODULE_ID, 'model_comparison', {
        **cfg, 'quick': quick, 'seeds': list(seeds), 'task': 'sudoku_4x4_completion', 'arms': list(ARMS),
        'metric': 'heldout_exact_solve', 'input_encoding': 'one_hot_blank_plus_4_digits_per_cell',
        'trm_tokens': CELLS, 'optimizer': 'AdamW(betas=(0.9, 0.95))', 'torch_threads': 1})
    receipt['data'] = {'grid_count': GRID_COUNT, 'split': 'disjoint grids per seed (train / val / held-out)',
                       'grid_overlap_between_splits': [r['grid_overlap'] for r in runs],
                       'mean_blanks': [r['mean_blanks'] for r in runs],
                       'uniqueness': 'every puzzle has exactly one solution among the 288 grids'}
    receipt['parameter_budget'] = {'trainable_parameters': params, 'mlp_hidden': runs[0]['mlp_hidden'],
                                   'max_relative_gap_vs_trm': gap, 'tolerance': cfg['param_tolerance'],
                                   'within_tolerance': bool(gap <= cfg['param_tolerance'])}
    receipt['arms'] = {n: {'per_seed': per_arm[n], 'summary': summaries[n]} for n in ARMS}
    receipt['best_baseline'] = {'arm': best, 'selected_by': 'highest mean validation exact-solve (held-out grids unused)'}
    receipt['comparisons'] = comparisons
    receipt['verdict'] = comparisons['trm_vs_best_baseline']['verdict']
    receipt['verdict_basis'] = f'trm_vs_best_baseline({best}): paired 95% t interval on held-out exact-solve over seeds'
    n_sup, rec, n_lat = cfg['supervision_steps'], cfg['recursions'], cfg['n_latent_steps']
    deep = n_sup * rec * (n_lat + 1)
    receipt['estimated_cost'] = {
        'inference_macs_per_puzzle': {n: per_arm[n][0]['estimated_inference_macs_per_puzzle'] for n in ARMS},
        'network_applications_per_puzzle': {'trm': deep, 'unrolled': deep, 'shallow': n_lat + 1, 'mlp': 1},
        'backpropagated_applications_per_update': {'trm': n_lat + 1, 'unrolled': deep, 'shallow': n_lat + 1,
                                                   'mlp': 1},
        'optimizer_updates_per_arm': cfg['updates'],
        'trm_examples_drawn': [a['training']['examples_drawn'] for a in per_arm['trm']],
    }
    receipt['measured_wallclock_s'] = {'total': time.perf_counter() - t_start,
                                       'train_per_arm': {n: float(np.sum([a['measured_wallclock_s']['train']
                                                                          for a in per_arm[n]])) for n in ARMS}}
    receipt['notes'] = (
        'Synthetic 4x4 Sudoku benchmark inspired by TRM (Jolicoeur-Martineau, 2025) and HRM (Wang et al., 2025); '
        'not a reproduction. Tiny 1-block mixer net (token-mixing MLP + channel MLP, ReLU, affine-free layer norm) '
        'on 16 cell tokens; no EMA, no augmentation (it would map training grids onto held-out grids). All arms: '
        'same data, same example stream, same number of AdamW updates; trm, unrolled and shallow share the network '
        'and initial weights; mlp is a flat MLP width-matched on trainable parameters, so trm_vs_mlp mixes the '
        'recursion effect with the per-cell weight sharing, while trm_vs_unrolled isolates deep supervision and '
        'trm_vs_shallow isolates recursion depth. TRM draws fewer distinct examples per update (each example gets '
        'up to supervision_steps updates); unrolled backpropagates through the whole unroll (more compute per '
        'update). Halting is trained (ACT, no Q-continue head); headline metrics use all supervision steps, '
        'adaptive_* use the first step with halting probability > 0.5. With few seeds the paired t interval is wide.')
    return harness.seal(receipt)


EXPERIMENTS: dict[str, Callable[..., dict[str, Any]]] = {MODULE_ID: run_experiment}
