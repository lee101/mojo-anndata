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

## GPU path: not taken

**CPU only: memory-bound.** Every kernel in this port is a gather or scatter
with no floating-point arithmetic: `src/capi.mojo` copies values and maps
indices, and nothing multiplies or sums. `bench/analysis.md` (counting
conventions, and §7) scores them at 0.042-0.129 flops/byte under a deliberately
generous convention (one integer address computation per output element), i.e.
**~16x below the ~2 flops/byte candidate bar at its highest** and ~48x at its
lowest. Nothing in the ranked table
qualifies, so no kernel was ported and no device code was written.

The device itself is available, so this is not a "no GPU on the host" result:
`nvidia-smi --query-gpu=memory.free` reported 5560-6352 MiB free out of
32607 MiB (NVIDIA GeForce RTX 5090) during this pass, above the 4000 MiB floor.
`DeviceContext` / `enqueue_create_buffer` / `enqueue_copy` / `enqueue_function`
do compile on the pinned `mojo ==1.2.0.dev2026092605` (`max.gpu.host`) — they
were reused unchanged by earlier ports — but the gate is arithmetic intensity,
not API availability.

The measured cost is already at DRAM speed, with no arithmetic to hide a
transfer behind. Same-session `pixi run bench` on this host, each call already
touching 8.4 MB, 69 MB and 0.18 MB (rate = bytes touched ÷ CPU time, so it
includes the Python glue each call carries):

| kernel | flops/byte | CPU time (this pass) | bytes touched | effective rate |
| --- | ---: | ---: | ---: | ---: |
| dense f64 2000x400 -> 1500x300 | 0.053 | 1.46 ms | 8.4 MB | 5.8 GB/s |
| dense f64 5500x1024 -> 4500x900 | 0.058 | 12.17 ms | 69.3 MB | 5.7 GB/s |
| CSR f64 2000x300 -> 1500x180 | 0.075 | 1.15 ms | 0.18 MB | 0.16 GB/s (glue-dominated, kernel 0.094 ms) |

A GPU version would pay, per call, one host-to-device copy of the source, one
device-to-host copy of the result and one launch before doing the same byte
movement. At 12.17 ms for the largest case the transfer alone is a few ms over
PCIe for 69 MB, for a kernel that has zero flops to hide it behind — so it can
only lose. The remaining headroom in these rows is Python glue
(`bench/analysis.md` §1, §3), which is an `unroll` problem on the CPU, not a
GPU one.

Consequences for this phase, stated explicitly:

- No GPU code was added, so **no test here requires a device and none were
  skipped**; `pixi run test` runs its full set on this CPU-only-for-kernels
  basis. There is nothing for a GPU-less host to trip over.
- The CPU path remains the only path, which is what a machine without a device
  would use anyway.
- There are no before/after kernel numbers to record, because no kernel
  changed: the table above is before = after, measured this pass. The three
  `pixi run bench` rows (1.46 / 12.17 / 1.15 ms against upstream 7.02 / 60.61 /
  2.59 ms) are unchanged from the `parallel` phase.

### Parallel dense gather

Dense slicing is the one kernel split across cores. The `_parallel` symbol used
to walk eight row chunks **in one thread** — it was never parallel, and the 2M
element gate sent large slices to it anyway. It now runs the same per-row gather
through `max.algorithm.parallelize` (`src/capi.mojo`); the `std.algorithm` and
`std.threading` forms do not compile on this toolchain. Each work item writes one
output row, so the result is **bit-identical** to the serial kernel whatever the
row-to-worker assignment; `tests/test_parallel.py` asserts that directly for both
symbols on the same input, including a slice with fewer rows than workers.

Measurements below are kernel-only (`_take_dense` on pre-built int64 index
arrays, full ctypes path minus `np.empty`), median of 21 interleaved serial /
parallel pairs on one process each, under `flock`, on the shared 72-CPU host
above. Worker count is compiled in, so the 1/2/4 sweep is a rebuild per row.
`×` = parallel_worker ÷ serial, below 1.00 means parallel wins:

| output elements | serial | 1 worker | 2 workers | 4 workers | 8 workers | 16 workers |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.25M (1500x1500) | 5.25 ms | 1.00 | 4.84 | 4.22 | 3.98 | 4.37 |
| 3.375M (2250x1500) | 8.01 ms | 1.05 | 2.10 | 1.85 | 1.72 | 2.14 |
| 4.5M (3000x1500) | 10.85 ms | 1.00 | 1.76 | 0.96 | — | — |
| 5.625M (3750x1500) | 14.05 ms | 1.05 | 1.47 | 0.81 | — | — |
| 9.0M (6000x1500) | 22.01 ms | 1.06 | 1.16 | 0.63 | — | — |

Read as factors against serial: one worker is the row-at-a-time kernel with zero
thread cost, so it is the fair "not parallel" baseline; any worker count above 1
starts the thread machinery. The crossover to a win is **~4.5M output elements**
(≈36 MB of result); at 2.25M four workers is 4.2x *slower*. Above the crossover
the gain grows with size (0.81 at 5.6M, 0.63 at 9M) and the marginal return is
real but small — 2→4 workers is about 1.3x, and 8/16 were measured at 2.25M and
3.375M where the rows were *already* near break-even, i.e. those columns test
over-subscription against the launch cost, not scaling at scale.

**Four workers is the cap and the shipping value.** The host is shared and the
gather is DRAM-bound, the gain flattened from 4 to 8 in every row where both were
measured (0.96→1.72 is dominated by the fixed cost, not bandwidth), and 16 was
never better than 8. 2 workers lost at every size tested, so it is not used.

**Gate: 8,000,000 output elements**, a margin of ~1.75x above the measured 4.5M
crossover, plus `rows >= workers`. Real callers therefore stay serial well past
the point where threading starts to win: the 5.06M element slice
(`6000x1500 -> 4500x1125`, measured 19.95 ms serial vs 12.12 ms four-worker
kernel) still routes to the serial kernel, because above that size the ctor and
allocation around the gather dominate anyway. Through the real
`AnnData.__getitem__`, turning the compiled worker count down to 1 (i.e. serial
routing everywhere) to a 4-worker build:

| slice | 1 worker build | 4 workers, shipped | anndata |
| --- | ---: | ---: | ---: |
| 8000x2000 -> 6000x1500 (9.0M, above gate) | 47.85 ms | **23.27 ms** | 202.78 ms |
| 2000x400 -> 1500x300 (0.45M, below gate) | 1.35 ms | 1.16 ms | 5.99 ms |
| 5500x1024 -> 4500x900 (4.05M, below gate) | 11.35 ms | 11.55 ms | 59.02 ms |

Only the row above the gate moves; the two below it are unchanged within the
run-to-run spread, which is the point of the threshold. The `pixi run bench`
rows from the top of this section are unchanged by this phase (1.16 / 11.55 /
1.04 ms against 1.25 / 11.93 / 1.08 ms before it): all three are below the gate
and still take the serial kernel, as intended. Nothing here was reverted — the
parallel path wins at exactly the sizes it now runs at — but it is a small,
late-arriving win: below ~4.5M elements the honest parallel path *loses* to
serial, and the code routes those sizes to serial rather than shipping a
parallelism that would be worse than none. The cost of being wrong is the
per-call thread cost above (~4.5-5 ms of worker startup at any n) times the
number of calls, which is why the gate is not merely at the crossover.

## How it works

Python resolves labels and NumPy-style selectors, allocates the result, and
keeps all annotations aligned. The ctypes boundary passes contiguous buffer
addresses as signed 64-bit integers, which the single Mojo compilation unit
rebuilds as pointers. Dense matrices are row-major `float64`; the CSR kernel
walks `indptr`, `indices`, and `data` once for each selected source row while a
column map translates source columns into the destination layout. Dense slices
with at least eight million output elements take the `_parallel` symbol, which
runs the same row gather through `max.algorithm.parallelize` with one output row
per work item and four workers; everything smaller, including the 4.5 million
element point where threading first breaks even, stays on the serial kernel
(see *Parallel dense gather* below). Canonical
CSR inputs are borrowed without a copy, and native int32 and int64 index kernels
avoid dtype conversion at the FFI boundary. Python owns every allocation, so the
native library has no allocation or lifetime policy. Slicing passes only arrays
it just allocated to a private constructor flag, so the result holds them
instead of copying each one a second time; a public constructor still copies its
inputs.
