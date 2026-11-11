# ANALYZE phase — where the time in mojo-anndata actually goes

Session: 2026-11-11, `master` @ `4fe6c1a`, clean worktree before and after.
Nothing in `src/` or `tests/` was touched; this file is the only addition.

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical CPUs, 2x 45 MiB L3,
`Linux-6.8.0-147-generic-x86_64-with-glibc2.39`. Python 3.13.14, numpy 2.5.1,
scipy 1.18.0, pandas 3.0.5, upstream anndata 0.13.2 (importable, so every
comparison below is real upstream, not a remembered number).
Toolchain: `mojo ==1.2.0.dev2026092605` (from `pixi.toml`).

## How every number was produced

| source | what it gives |
| --- | --- |
| `pixi run bench` (exit 0) | the three published rows, under `flock /tmp/mojo-bench.lock`, run alone before any other measurement |
| `/tmp/final_table.py` (not committed) | per-row kernel-only, end-to-end, and upstream timings in one process, `min` of 7-21 reps, all arrays from `default_rng(42)` |
| targeted follow-ups | serial-vs-parallel kernel at both sizes, `__getitem__` phase split, ctor profile, CSR i32-vs-i64 at one shape, 1-row ctypes round-trip |

`best of N` (`time.perf_counter`) throughout, which is what `bench/bench.py`
does. Kernel-only numbers call the exported symbol directly with pre-built
int64 index arrays, so they exclude `_positions`, `np.empty`, and `_lib.lib()`
lookup; the glue column is `end-to-end minus kernel-only` for the same call on
the same data. Glue therefore covers `_positions`, the `np.empty` result buffer,
the ctypes marshalling, `_take_matrix` dispatch, and — for `__getitem__` rows —
`obs`/`var` frame copies and the `AnnData.__init__` of the result.

`pixi run bench` on this session:

| kernel | mojo-anndata | anndata | speedup |
| --- | ---: | ---: | ---: |
| dense f64 2000x400 -> 1500x300 | 1.65 ms | 5.87 ms | 3.55x |
| dense f64 5500x1024 -> 4500x900 | 16.74 ms | 60.50 ms | 3.61x |
| CSR f64 2000x300 -> 1500x180 | 1.06 ms | 2.41 ms | 2.27x |

These are the same three calls as rows H10 (serial dense), H7/H4 (large dense)
and H9 (CSR) below and land inside the run-to-run spread of those rows (see
*Variance*). A second `pixi run bench` at the end of the session, after all
probing, reported 1.74 / 20.98 / 1.10 ms against 6.36 / 70.98 / 2.47 ms, i.e.
the large dense row alone ranges 16.74-20.98 ms here. The README quotes 1.22 /
10.56 / 0.84 ms from an older kernel; the current code is slower on all three.

## Counting conventions (needed to read the flops/byte column)

- `bytes touched` = minimum source read + result write for one pass of the
  call, from the actual shapes and dtypes: dense reads every element of the
  selected source rows plus the `rows`/`cols` index vectors; CSR reads the
  touched `nnz` of `data`/`indices` plus `rows` and `col_map`, and writes the
  compacted output.
- `flops` is a **convention**, not real arithmetic: 1 per gathered output
  element (one int address computation) and 1 per stored CSR nonzero (one
  bounds compare plus one store). There is **no floating-point arithmetic
  anywhere in this port** — the kernels only move bytes. Column D of the table
  is therefore "integer ops the kernel must issue", and the ratio is an upper
  bound on arithmetic intensity.
- `memory-bound` is assigned when the measured runtime is at or above the
  streaming time for `bytes touched` at this machine's single-core figure
  (~1 ns per 8-byte element, taken from the in-cache dense kernel at
  1.34 ns/element). Nothing here is compute-bound; the useful question below is
  *bound by Python leaf-call overhead* vs *bound by memory traffic*.
- `kernel` vs `glue` is the split defined above. "Kernel" rows are Go-side
  (Mojo) time; "glue" is Python/ctypes/NumPy time.

## Ranked hotspot table

Ranked by measured end-to-end time. Input sizes are the exact shapes passed.
`A` = action phase from the last column of this phase's brief.

| # | function | input size | time (end-to-end) | bytes touched | flops (conv.) | flops/byte | bound by | time sits in | A |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| H1 | `AnnData.__getitem__`, 2 layers + 2 `obsm`, annot. | 5500x1024 -> 4500x900, int positions | **130.06 ms** | 94.2 MB | 12.15M | 0.129 | Python leaf calls | glue 130.06 / kernel n/a (`_take_dense` measured 11.14 ms of it) | unroll |
| H2 | same, string label selectors | 5500x1024 -> 4500x900, 4500 + 900 str names | **127.37 ms** | 94.2 MB | 12.15M | 0.129 | Python leaf calls | glue; `_positions` on str rows costs 0.74 ms of it | unroll |
| H3 | `AnnData.copy` | 5500x1024, 2 layers + `obsm` | **182.03 ms** | 292.4 MB | 16.90M | 0.058 | Python leaf calls | glue (16 `_take_dense`/`_take_axis` calls + 4 ctor copies) | unroll |
| H4 | `mad_dense_take_f64_parallel` -> `__getitem__` no layers | 5500x1024 -> 4500x900 | **55.94 ms** | 69.3 MB | 4.05M | 0.058 | memory | kernel 9.26 ms, **glue 46.7 ms** | unroll |
| H5 | `AnnData.__init__` | 5500x1024 f64 | **35.33 ms** (22.11 ms second run) | 90.1 MB | 0 | 0.0 | memory copy | glue: 19.33 ms is `np.ndarray.copy` | none |
| H6 | `mad_csr_take_f64_i32` + `_take_csr` glue | 20000x2000 CSR, nnz=2.0M -> 15000x1000 | **12.69 ms** | 30.0 MB | 1.50M | 0.050 | memory | kernel 10.34 ms, glue 2.35 ms | none |
| H7 | `mad_dense_take_f64_parallel` | 5500x1024 -> 4500x900 | **10.69 ms** (serial 10.53) | 69.3 MB | 4.05M | 0.058 | memory | kernel (2.59 ns/element) | parallel |
| H8 | bool-mask rows, `__getitem__` | 5500x1024, 2772/5500 rows kept | **9.99 ms** | 45.5 MB | 2.84M | 0.062 | memory | kernel gather; `_positions` mask->indices 0.011 ms | simd |
| H9 | `mad_csr_take_f64_i32` | 2000x300 CSR, nnz=18000 -> 1500x180 | **0.269 ms** (kernel 0.094) | 0.18 MB | 13543 | 0.075 | Python leaf calls | glue 0.175 ms = 65% | unroll |
| H10 | `mad_dense_take_f64` (serial) | 2000x400 -> 1500x300 | **0.607 ms** | 8.4 MB | 450K | 0.053 | memory | kernel (in-cache, 1.34 ns/el) | simd |
| H11 | full `__getitem__`, 1 layer path, small | 2000x400 -> 1000x200, str names | **1.344 ms** | 4.8 MB | 200K | 0.042 | Python leaf calls | glue 1.34 minus 0.33 ms positions | unroll |
| H12 | `mad_csr_take_f64_i32` | 500x100 CSR, nnz=5000 -> 300x50 | **0.150 ms** (kernel 0.017) | 0.041 MB | 3014 | 0.073 | Python leaf calls | glue 0.133 ms = 89% | unroll |
| H13 | `take_dense` float32 fallback | 2000x400 -> 1500x300 f32 | **2.86 ms** | 4.2 MB | 450K | 0.107 | memory (`np.ix_`) | glue (NumPy fallback, no Mojo) | simd |
| H14 | `obs_vector` | 5500x1024, column gather | **0.0270 ms** | 44 KB | 5500 | 0.125 | Python leaf calls | glue (NumPy fancy-index copy) | none |
| H15 | `var_vector` | 5500x1024, row copy | **0.0044 ms** | 8 KB | 1024 | 0.125 | Python leaf calls | glue (`np.asarray(...).ravel`) | none |
| H16 | ctypes round-trip, dense 1-row call | 1x4 -> 1x4 | **1.94 us** | 32 B | 4 | 0.125 | Python leaf calls | glue: ctypes dispatch | none |
| H17 | ctypes round-trip, CSR 1-row call | 1-row, 1 nnz | **~2 us** (same call shape as H16) | 16 B | 1 | 0.063 | Python leaf calls | glue: ctypes dispatch | none |

Upstream comparison for every row that has one (`—` = no upstream equivalent;
slower-than-upstream rows are marked **SLOWER**):

| # | mojo-anndata | upstream anndata | port / upstream | verdict |
| --- | ---: | ---: | ---: | --- |
| H1 | 130.06 ms | 292.06 ms | 0.45x | faster |
| H2 | 127.37 ms | 298.83 ms | 0.43x | faster |
| H3 | 182.03 ms | 104.92 ms | **1.74x** | **SLOWER** |
| H4 | 55.94 ms | 110.77 ms | 0.51x | faster |
| H5 | 35.33 ms (22.11 ms) | 2.59 ms | **13.7x** (8.5x) | **SLOWER** |
| H6 | 12.69 ms | 44.66 ms | 0.28x | faster |
| H7 | 10.69 ms | 75.96 ms (full slice+copy) | — | kernel only |
| H9 | 0.269 ms | 2.30 ms | 0.12x | faster |
| H10 | 0.607 ms | 5.91 ms (full slice+copy) | — | kernel only |
| H11 | 1.344 ms | 3.86 ms | 0.35x | faster |
| H12 | 0.150 ms | 1.85 ms | 0.08x | faster |
| H13 | 2.86 ms | 5.46 ms | 0.52x | faster |
| H14 | 0.0270 ms | 0.532 ms | 0.05x | faster |
| H15 | 0.0044 ms | 0.463 ms | 0.01x | faster |

Non-kernel baselines, same session, for scale:

| baseline | time |
| --- | ---: |
| `np.take(np.take(d, rows), cols)` 5500x1024 -> 4500x900 | 26.59 ms |
| `d[np.ix_(rows, cols)].copy()` same shape | 40.05 ms |
| `np.take(np.take(...))` 2000x400 -> 1500x300 | 1.009 ms |
| `d[np.ix_(rows, cols)].copy()` same shape | 2.234 ms |
| `X.copy()` 5500x1024 f64 | 21.18 ms |
| CSR full-shape `x[rows, :][:, cols].copy()` 20000x2000 | 44.66 ms |
| upstream `ra[rows, cols]` **view, no `.copy()`** 5500x1024 | 0.79 ms |

## 1. The dominant cost is Python leaf calls, not kernels

The two biggest end-to-end rows, H1 (130 ms) and H2 (127 ms), decompose as:

| phase of `__getitem__` | measured |
| --- | ---: |
| `_take_dense` on `X` | 11.14 ms |
| `_take_dense` on layer `counts` | 10.77 ms |
| `_take_dense` on layer `lognorm` | 11.21 ms |
| `obs.iloc[rows].copy()` | 0.23 ms |
| `var.iloc[cols].copy()` | 0.14 ms |
| `AnnData(Xsub, obs, var, layers=dict)` results ctor | 39.20 ms |
| `AnnData(Xsub)` results ctor alone (same shape) | 17.46 ms |
| sum of the parts | ~90 ms |
| measured full call | 110-130 ms |

So of the ~130 ms, roughly 33 ms is the three Mojo gathers (25%) and ~97 ms is
Python: the result `AnnData.__init__` re-copying the freshly-gathered blocks,
plus frame copies, plus `_take_matrix` dispatch per container. With no layers
(H4) the same shape still spends 46.7 ms outside the kernel for a single
25 MB block — the ctor copies it again after `_take_dense` already allocated it.

The result is that `__getitem__` at annotation-realistic size is **3.7x slower
than the kernel inside it**, and this single fact dominates the ranking. It is
also why the port loses to upstream on `copy()` (H3): upstream copies arrays in
a handful of C-level calls, while this port drives ~16 separate
`_take_dense`/`_take_axis` calls and then reconstructs four containers, each of
which copies again.

**Verdict: `unroll` phase.** The actionable work is batching the per-container
Python loops and removing the double copy in the results constructor. A kernel
optimisation in this section moves at most 25% of the wall time.

## 2. Kernel-only performance, and the parallel path that is not parallel

Dense gather, same kernel body, both symbols, same data, 21 reps:

| shape | `mad_dense_take_f64` | `mad_dense_take_f64_parallel` | ns/element |
| --- | ---: | ---: | ---: |
| 2000x400 -> 1500x300 (in L2) | **0.607 ms** | 0.643 ms (0.639 serial vs 0.643 parallel) | 1.34 / 1.43 |
| 5500x1024 -> 4500x900 (36.9 MB read) | **10.53 / 10.92 ms** | 10.69 / 11.18 ms | 2.76 / 2.75 |

The "parallel" symbol is **not faster than the serial one at either size** —
differences are inside run-to-run variance. It splits by raw `nrows_out` into 8
static chunks with no `max.algorithm.parallelize` anywhere in `src/`
(`grep -rn "parallelize" src/` returns 0 matches; `DENSE_PARALLEL_TASKS = 8`
at `src/capi.mojo:12` is only a loop bound). So the 2M-element threshold in
`_core.py:16` routes large slices to a function that does the same work serially
with a worse access pattern.

Per element the large slice costs **2.59 ns vs 1.34 ns in-cache**, i.e. it is
at DRAM speed for a 36.9 MB stream — the gather itself is already close to
memory-bound, and the remaining headroom is prefetch-friendly addressing, not
arithmetic.

**Verdict: `parallel`** (actually use `max.algorithm.parallelize` under the
existing 2M-element gate, and drop the gate if it loses), then **`simd`** for
the in-cache case. `unroll` first only for the CSR element loop, see §4.

## 3. Small-input rows are ctypes-dominated

ctypes round-trip on a 1-row dense call: **1.94 us**; the same call shape on
the CSR symbol is ~2 us. At 2000x300 CSR (H9) the kernel is 0.094 ms against
0.175 ms of glue, and at 500x100 (H12) the kernel is 0.017 ms against 0.133 ms
of glue — 89% glue. The glue is *not* a per-element ctypes loop: there is no
Python loop over matrix elements anywhere. It is fixed per-call work —
`np.unique(cols)` dedupe 0.043 ms, capacity `np.sum` 0.025 ms, 3 `np.empty`
0.0025 ms, `csr_matrix` construction 0.039 ms, `sort_indices`, plus the ~2 us
ctypes entry and `_lib.lib()` global lookup. At small nnz that fixed cost is
the whole call.

**Verdict: `unroll`** — one call into a fused prelude is unnecessary; the win
is removing the `np.unique`/`np.sum` round trips from Python, which is an
`unroll`-phase batching change, not a kernel change. No GPU, no SIMD: the
kernel is 17 us and the fixed cost is the problem.

## 4. CSR scales linearly but pays i64 at the big shape

| shape | indices dtype | kernel | end-to-end | upstream | ns / touched nnz |
| --- | --- | ---: | ---: | ---: | ---: |
| 500x100, nnz=5000, touched=3014 | int32 | 0.017 ms | 0.150 ms | 1.85 ms | 5.6 |
| 2000x300, nnz=18000, touched=13543 | int32 | 0.094 ms | 0.269 ms | 2.30 ms | 6.9 |
| 20000x2000, nnz=2.0M, touched=1.50M | int32 | 10.34 ms | 12.69 ms | 44.66 ms | 7.85 |
| 20000x2000, same data cast | int64 | — | 13.98 ms | — | — |

The `mad_csr_take_f64_i32` loop is a scalar per-nonzero branch-and-store
(`src/capi.mojo:102-113`): ~7.9 ns per nonzero, i.e. it is memory-latency bound
on the random `indices -> col_map` load, not compute bound. int64 indices cost
a measured 10% more end-to-end at the same nnz. No upstream row is lost here;
all four are faster than SciPy.

**Verdict: `none` for the kernel** (0.05 flops/byte, already at memory speed,
10-36x faster than SciPy); the `unroll` action from §3 covers the small-shape
glue.

## 5. The port loses to upstream in two places

- **H3 `copy()`: 182.03 ms vs 104.92 ms (SLOWER, 1.74x).** Upstream copies
  each array once; this port re-gathers every container through `_take_matrix`
  with freshly built `arange` positions and then the ctor copies again.
- **H5 `__init__`: 35.33 ms vs 2.59 ms (SLOWER, 13.7x)** on the first run and
  22.11 vs 2.59 ms on a re-run (8.5x) — `cProfile` over 3 ctors attributes
  124 ms of 138 ms to `numpy.ndarray.copy`, i.e. `_matrix`'s
  `array.copy()`, which upstream avoids by holding the caller's array. The
  port copies the full 45 MB input on construction by design (`_core.py:233`).

Both rows stay in the table and are marked. Neither is fixable by `simd`,
`parallel`, or `gpu`; they are `unroll`-phase call-structure changes, and the
`__init__` policy change in particular is a semantic decision for a later
phase, not an analysis-phase change.

## 6. Per-call allocations and copies found

From reading `python/mojoanndata/_core.py` and confirmed by the numbers above.

| site | allocation / copy | measured share |
| --- | --- | --- |
| `_matrix` (`_core.py:233`) | full `array.copy()` of `X` and of every layer in the ctor | 19.33 ms of a 22.11 ms ctor at 5500x1024 |
| `_take_dense` results ctor | `AnnData(Xsub)` copies the already-fresh `Xsub` again | 17.46 ms of the 55.94 ms no-layer slice |
| `_take_dense` per layer in `__getitem__` | one Mojo gather per layer, then one ctor copy each | 3 x ~11 ms gather + copy in H1 |
| `_take_csr` (`_core.py:168-175`) | `col_map = np.full`, 3 `np.empty`, `np.ascontiguousarray` x2 | 0.010 ms + 0.025 ms capacity + 0.0025 ms allocs per call |
| `_take_csr` (`_core.py:155`) | `np.unique(cols)` dedupe probe every call | 0.043 ms per call |
| `_take_csr` (`_core.py:189`) | `result.sort_indices()` after the kernel already emits sorted output | inside the 0.039 ms post step |
| `_positions` (`_core.py:112,116,120,128,136`) | `flatnonzero`, `astype(int64)`, `np.where`, then `np.ascontiguousarray` — up to 4 passes over the index array | 0.330 ms for 1000+200 str names; 0.74 ms for 4500 str rows |
| `copy()` (`_core.py:354-363`) | `arange` x2 + `_take_*` for every container + ctor re-copy | whole 182 ms row |
| `_frame(None, n)` (`_core.py:24`) | `[str(i) for i in range(n)]` Python comprehension for a default index | 0.923 ms at n=5500 |

No Python loop re-enters ctypes per element. The loops that do exist are per
*container* (per layer, per `obsm` key, per axis) and per *call*.

## 7. GPU verdict

Every kernel in this port is a gather/scatter: dense is 0.053-0.058 flops per
byte, CSR is 0.050-0.075, and even the optimistic counting convention in
§"Counting conventions" already inflates the numerator. `MOJO_NOTES.md` §5 sets
the CPU-only bar at roughly 2 flops/byte; these kernels are **25-40x below
it**, and the measured runtime at the largest shape (2.59 ns/element, 45 MB/s
effective per core, 10.7 ms for 69 MB touched) is already DRAM-limited on the
CPU. A GPU path would pay one copy-in of the source and one copy-out of the
result plus a launch for work that has no arithmetic to hide them behind.

**Verdict: `none` — no GPU phase for any row in this table.** The same flops/
byte figures rule out `simd` helping the *large* dense cases (memory-bound) and
keep `simd` relevant only for the in-cache 2000x400 case and the float32
fallback.

## 8. Where each hotspot is sent

| hotspot | next phase | reason (measured) |
| --- | --- | --- |
| H1, H2, H4, H11 annotated `__getitem__` | unroll | 75-84% of time is per-container Python leaf calls and the result ctor's second copy |
| H3 `copy()` | unroll | 1.74x slower than upstream; 16 gather calls + 4 ctor copies |
| H5 `__init__` | unroll (policy question) | 8.5-13.7x slower than upstream; 87% of ctor is one deliberate `X.copy()` |
| H6, H9, H12 CSR kernel | none (kernel); unroll (glue) | 7.9 ns/touched-nnz, memory-latency bound; small shapes are 65-89% fixed Python cost |
| H7 dense parallel | parallel | `*_parallel` is measured no faster than serial; `max.algorithm.parallelize` unused |
| H8 bool mask | simd | kernel-bound on the gather; effective index list is already dense |
| H10 dense serial in-cache | simd | 1.34 ns/element at 8.4 MB; cache-resident so addressing is the cost |
| H13 float32 fallback | simd | no Mojo path at all today; NumPy `ix_` at 2.86 ms |
| H14, H15 vector accessors | none | already 11-75x faster than upstream |
| H16, H17 ctypes entry | none | ~1.9 us fixed, negligible except at H9/H12 sizes, handled by the unroll row |

## 9. Variance and caveats

- `__getitem__` rows move with cache state and with the (serial) "parallel"
  kernel: the no-layer slice measured 55.94 ms here and 46.7 ms of that was
   glue; an earlier run of the same call measured 88.68 ms. The undecorated
  gather underneath (`pixi run bench`) measured 16.74 ms and 20.98 ms on two
  runs, because it carries no layers or `obsm`. The table's H1/H2/H4 figures
  are from the consolidated run; the other values are quoted only to show
  spread. The ordering that drives the ranking (H1 > H4 > H7, and glue >
  kernel for every annotated call) holds in every run.
- H5 has two measured values (35.33 and 22.11 ms) because `_matrix`'s copy of a
  45 MB array depends on whether it comes from page cache; both are shown.
- H17 is the same call shape as H16 on the other symbol and is reported at the
  same ~2 us resolution rather than claimed to more digits.
- `obs_vector`/`var_vector` are deprecated upstream (FutureWarning); the
  upstream column includes its deprecation wrapper, which is part of why the
  ratios are so large.
- `pixi run bench` was run first and alone under `flock`; no other benchmark ran
  concurrently.
