# -*- coding: utf-8 -*-
"""stock_corr v2 fully-valid returns panel generator (deterministic).

The stock_corr v2 FAST path (de-mean Gram, 1 accumulator) only applies to
fully-valid panels (every column count == T), and the realistic stock_corr
use case is clean, all-finite returns. The gate re-baseline (2026-08-05,
user decision "same-panel re-baseline") therefore measures BOTH the GPU fast
path and the CuPy reference on these exact .bin panels.

Distribution: continuous returns matching the corpus_synth_v1 returns shape
(normal(0, 0.02), std ~0.023, no exact ties -- the corpus has ZERO exact-zero
returns, so artificial zeros/tie ints would create near-zero-variance pairs
that falsely trigger the cancellation fall-back and inflate the CuPy gate).
All cells finite, so the panel is fully valid and the GPU fast path applies.

One 5000-column panel is generated; N=500/2000/5000 run on its first-N
column prefix (same slicing as the corpus gate returns[:, :actual]).

Determinism: numpy default_rng(seed), seed = 20260805 + N_cols.
"""
from __future__ import annotations

import hashlib
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
T = 1218
N_COLS = 5000
SIZES = [500, 2000, 5000]
SEED = 20260805


def make_panel(T: int, N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 0.02, size=(T, N))  # continuous returns, no exact ties
    return v.astype(np.float64)


def main() -> None:
    X = make_panel(T, N_COLS, SEED + N_COLS)
    if not np.isfinite(X).all():
        raise SystemExit("panel not fully finite -- fast path would not apply")
    bin_path = HERE / f"stock_corr_panel_v1_{N_COLS}.bin"
    X.tofile(bin_path)
    h = hashlib.sha256(X.tobytes()).hexdigest()
    print(f"wrote {bin_path}")
    print(f"  shape (T={T}, N={N_COLS}), sha256={h[:16]}..., bytes={X.nbytes}")
    # per-column stats the fast-path dispatch relies on (all count==T, low bias)
    mean = X.mean(axis=0)
    sigma = X.std(axis=0)
    print(f"  |mean|/sigma: max {np.abs(mean/sigma).max():.3f} "
          f"(fast path is correctness-agnostic to bias; reported for reference)")
    for N in SIZES:
        sub = X[:, :N]
        print(f"  slice N={N}: fully finite={np.isfinite(sub).all()}, "
              f"per-col count==T: {(sub.shape[0]==T and bool(np.isfinite(sub).all()))}")


if __name__ == "__main__":
    main()
