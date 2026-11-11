# Verification Report — mojo-anndata VERIFY phase

**Date:** 2026-11-11  
**Machine:** Linux-6.8.0-147-generic-x86_64-with-glibc2.39 (x86_64)  
**Toolchain:** mojo ==1.2.0.dev2026092605, Python 3.13, numpy 2.5.1, scipy 1.18.0, pandas 3.0.5, anndata 0.13.2

---

## 1. Build and Test

```
pixi run build && pixi run test
```

- **Build:** ✅ Passed (0.9s)
- **Tests:** ✅ 112 passed, 22 warnings (all expected deprecation/uniqueness warnings)

No test failures. All parity assertions against upstream anndata 0.13.2 hold with `RTOL=1e-12, ATOL=0.0`.

---

## 2. Benchmark Reproduction

Ran `pixi run bench` 15 times (5 initial + 5 + 5) under `flock /tmp/mojo-bench.lock`.  
Each run reports best-of-5 `time.perf_counter` on identical RNG-seeded inputs.

### README Table 1 (Main Benchmark)

| kernel | README `after` | README `anndata` | Measured `mojo` (best) | Measured `anndata` (best) | Within 20%? |
|--------|---------------|-----------------|------------------------|--------------------------|-------------|
| dense f64 2000x400 → 1500x300 | 1.19 ms | 5.95 ms | **1.23 ms** | **5.89 ms** | ✅ (3% / 1%) |
| dense f64 5500x1024 → 4500x900 | 11.23 ms | 60.04 ms | **10.94 ms** | **58.20 ms** | ✅ (3% / 3%) |
| CSR f64 2000x300 → 1500x180 | 1.02 ms | 2.41 ms | **1.04 ms** | **2.34 ms** | ✅ (2% / 3%) |

**All three rows reproduce within 20%.** No README changes needed.

### README Table 2 (Workload Breakdown, `bench/analysis.md` §1)

| workload | README `before` | README `after` |
|----------|----------------|---------------|
| `__getitem__`, 2 layers + 2 `obsm`, integer positions | 45.3 ms | 33.7 ms |
| same, string label selectors | 48.6 ms | 35.1 ms |
| `__getitem__`, no layers | 17.5 ms | 11.1 ms |
| `copy()`, 2 layers + `obsm` | 134 ms | 70 ms |

**Not remeasured in this session.** These numbers come from the dedicated profiling session documented in `bench/analysis.md` (single-process, min-of-7-to-21, kernel/glue split, different harness). They are workload- and machine-specific; the README already notes this. No action needed.

### README Table 3 (GPU / Parallel Analysis)

The flops/byte, CPU time, bytes touched, and effective rate table (lines 119–123 in README) and the parallel scaling table (lines 162–168) are from `bench/analysis.md` profiling runs, not from `pixi run bench`. They were not remeasured; they remain as documented.

---

## 3. Same-Input Verification

The benchmark harness (`bench/bench.py`) constructs identical arrays for both sides using `np.random.default_rng(42)` and the same index selections:

```python
rng = np.random.default_rng(42)
dense = rng.normal(size=(2_000, 400)).astype(np.float64)
rows = rng.choice(dense.shape[0], 1_500, replace=False)
cols = rng.choice(dense.shape[1], 300, replace=False)
fast_dense, ref_dense = AnnData(dense), upstream.AnnData(dense)
```

Both mojo-anndata and upstream anndata receive **bit-identical** input matrices and index arrays. No work is skipped on either side; both return a full slice result whose `.shape` is touched to prevent dead-code elimination.

---

## 4. Test Parity Check

Inspected `git log -p -- tests/` across all commits:

- **No assertions were removed or loosened.**  
- All parity tests (`test_parity.py`, `test_slicing.py`, `test_parallel.py`) retain `RTOL=1e-12, ATOL=0.0` and full structure/value comparison against upstream anndata 0.13.2.
- Degenerate-input tests still assert the exact exception class upstream raises.
- The parallel dense tests assert bit-identical output between serial and parallel kernels (`np.array_equal(serial.view(np.int64), parallel.view(np.int64))`).
- The gate test (`test_take_dense_routes_by_size`) asserts the 8M-element threshold exactly as documented.

**No regressions found.** All 112 tests pass.

---

## 5. Changes Made

**None.** The README benchmark numbers reproduce within the 20% tolerance. The test suite is green with full parity assertions intact. No code changes were required.

---

## 6. Summary

| Check | Result |
|-------|--------|
| `pixi run build` | ✅ Pass |
| `pixi run test` | ✅ 112 passed |
| README Table 1 reproduced | ✅ All 3 rows within 20% |
| Same inputs for both sides | ✅ Verified in `bench/bench.py` |
| No work skipped | ✅ Both sides return full slice, `.shape` touched |
| Test parity assertions intact | ✅ No removed/loosened assertions found in git history |
| Final `pixi run build && pixi run test` | ✅ Both green |

The VERIFY phase is complete. The published numbers are reproducible on this machine.