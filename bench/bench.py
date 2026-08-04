"""Measured slicing comparison against upstream anndata on identical inputs."""

from __future__ import annotations

import os
import platform
import sys
import time

import anndata as upstream
import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from mojoanndata import AnnData  # noqa: E402


def timed(fn, reps=5):
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
        _ = result.shape
    return best


def report(name, mojo, reference):
    speedup = reference / mojo
    print(f"| {name} | {mojo * 1e3:.2f} ms | {reference * 1e3:.2f} ms | {speedup:.2f}x |", flush=True)


def main():
    rng = np.random.default_rng(42)
    dense = rng.normal(size=(2_000, 400)).astype(np.float64)
    rows = rng.choice(dense.shape[0], 1_500, replace=False)
    cols = rng.choice(dense.shape[1], 300, replace=False)
    fast_dense, ref_dense = AnnData(dense), upstream.AnnData(dense)

    csr = sparse.random(2_000, 300, density=0.03, format="csr", random_state=42,
                        dtype=np.float64)
    csr_rows = rng.choice(csr.shape[0], 1_500, replace=False)
    csr_cols = np.sort(rng.choice(csr.shape[1], 180, replace=False))
    fast_csr, ref_csr = AnnData(csr), upstream.AnnData(csr)

    large_dense = rng.normal(size=(5_500, 1_024)).astype(np.float64)
    large_rows = rng.choice(large_dense.shape[0], 4_500, replace=False)
    large_cols = rng.choice(large_dense.shape[1], 900, replace=False)
    fast_large_dense, ref_large_dense = AnnData(large_dense), upstream.AnnData(large_dense)

    print(f"Machine: {platform.platform()} ({platform.processor() or 'processor unavailable'})", flush=True)
    print("| kernel | mojo-anndata | anndata | speedup |")
    print("| --- | ---: | ---: | ---: |")
    report("dense f64 2000x400 -> 1500x300", timed(lambda: fast_dense[rows, cols]),
           timed(lambda: ref_dense[rows, cols].copy()))
    report("dense f64 5500x1024 -> 4500x900", timed(lambda: fast_large_dense[large_rows, large_cols]),
           timed(lambda: ref_large_dense[large_rows, large_cols].copy()))
    report("CSR f64 2000x300 -> 1500x180", timed(lambda: fast_csr[csr_rows, csr_cols]),
           timed(lambda: ref_csr[csr_rows, csr_cols].copy()))


if __name__ == "__main__":
    main()
