# mojo-anndata

`mojo-anndata` is a standalone, Mojo-accelerated subset of
[`anndata`](https://anndata.readthedocs.io/) for one of the operations that
dominates interactive single-cell workflows: slicing an annotated matrix.
It provides an eager in-memory `mojoanndata.AnnData` with the covered upstream
names and signatures, so code using basic construction and `adata[obs, var]`
can move over without changing its slicing calls.

## Covered subset

- Dense `float64` two-axis gather (`slice`, integer/fancy indices, boolean
  masks, negative indices, and observation/variable names) is performed in
  Mojo.
- CSR `float64` row-and-column slicing is performed in Mojo, preserving a CSR
  result. Repeated column selections use SciPy's correct fallback.
- `obs`, `var`, `layers`, `obsm`, `varm`, `obsp`, `varp`, and `uns` remain
  aligned and are eagerly copied during a slice. `uns` alone is shared with the
  slice's parent, as in upstream anndata views; `copy()` deep-copies it.
- `shape`, `n_obs`, `n_vars`, `obs_names`, `var_names`, `copy`, `to_df`,
  `obs_vector`, `var_vector`, and the `*_keys` helpers are available.

This is intentionally not a full replacement for anndata. Backed `.h5ad`
files, lazy views, `raw`, concatenation, I/O, sparse formats other than the
CSR fast path, and the broader ecosystem API are not covered. Non-`float64`
dense arrays and unsupported sparse forms use a NumPy/SciPy fallback.

## Install and use

```bash
pixi install
pixi run build
```

`pixi` exports `PYTHONPATH=python`, so this example runs directly with
`pixi run python example.py`:

```python
import numpy as np
import pandas as pd
from mojoanndata import AnnData

adata = AnnData(
    np.arange(20, dtype=np.float64).reshape(5, 4),
    obs=pd.DataFrame(index=["cell0", "cell1", "cell2", "cell3", "cell4"]),
    var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    layers={"counts": np.arange(20, dtype=np.float64).reshape(5, 4)},
)
subset = adata[["cell4", "cell1"], ["g2", "g0"]]
assert subset.shape == (2, 2)
assert subset.X.tolist() == [[18.0, 16.0], [6.0, 4.0]]
```

Run the parity suite and the machine-locked benchmark with:

```bash
pixi run test
pixi run bench
```

## Benchmark

Measured with `pixi run bench` on `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
(`x86_64`), best of five identical eager slices. The comparison is against the
installed upstream `anndata` (0.13.2) on the same arrays. "before" is `master`
at `4fe6c1a`, "after" is this revision; both were measured with the same harness
on a shared 72-CPU host. Each cell is the fastest of several runs, which is what
`bench/bench.py` reports; the large dense row ranged 11.23-13.46 ms across the
after-runs and 17.41-17.88 ms across the before-runs.

| kernel | before | after | anndata | speedup (after) |
| --- | ---: | ---: | ---: | ---: |
| dense f64 2000x400 -> 1500x300 | 1.42 ms | 1.19 ms | 5.95 ms | 5.01x |
| dense f64 5500x1024 -> 4500x900 | 17.41 ms | 11.23 ms | 60.04 ms | 5.34x |
| CSR f64 2000x300 -> 1500x180 | 1.03 ms | 1.02 ms | 2.41 ms | 2.36x |

The gain is in the Python glue, not the kernels: the two larger rows were
dominated by work outside Mojo (`bench/analysis.md` §1). The dense kernels run
no floating-point arithmetic at all, so nothing was reordered: results are
bit-identical and every existing tolerance is unchanged.

The workloads where that glue dominates most (`bench/analysis.md` §1), same
shapes (5500x1024 -> 4500x900), alternating before/after in one session:

| workload | before | after |
| --- | ---: | ---: |
| `__getitem__`, 2 layers + 2 `obsm`, integer positions | 45.3 ms | 33.7 ms |
| same, string label selectors | 48.6 ms | 35.1 ms |
| `__getitem__`, no layers | 17.5 ms | 11.1 ms |
| `copy()`, 2 layers + `obsm` | 134 ms | 70 ms |

Numbers are workload- and machine-specific; rerun the benchmark before making
performance decisions for a different matrix shape or sparsity pattern.

Dense and CSR slicing are memory-bound gathers/scatters with effectively zero
arithmetic intensity, so there is no GPU path: transfer and launch overhead
would lose to the CPU implementation.

## How it works

Python resolves labels and NumPy-style selectors, allocates the result, and
keeps all annotations aligned. The ctypes boundary passes contiguous buffer
addresses as signed 64-bit integers, which the single Mojo compilation unit
rebuilds as pointers. Dense matrices are row-major `float64`; the CSR kernel
walks `indptr`, `indices`, and `data` once for each selected source row while a
column map translates source columns into the destination layout. Dense slices
with at least two million output elements take the `_parallel` symbol, which
walks the same eight row chunks in order (there is no threading yet). Canonical
CSR inputs are borrowed without a copy, and native int32 and int64 index kernels
avoid dtype conversion at the FFI boundary. Python owns every allocation, so the
native library has no allocation or lifetime policy. Slicing passes only arrays
it just allocated to a private constructor flag, so the result holds them
instead of copying each one a second time; a public constructor still copies its
inputs.
