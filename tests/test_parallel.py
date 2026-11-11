"""Serial vs parallel dense gather: identical output, and the size gate.

The parallel kernel is `max.algorithm.parallelize` over output rows
(`src/capi.mojo`).  These tests call both exported symbols directly on the same
input so the comparison cannot be masked by the Python layer, then check that
`_take_dense` still routes below the threshold to the serial symbol.
"""

from __future__ import annotations

import numpy as np
import pytest

from mojoanndata import _core
from mojoanndata._core import _dense_worker_count, _take_dense
from mojoanndata._lib import addr, lib

WORKERS = _dense_worker_count()


def _take(kernel, source, rows, cols, poison=None):
    target = np.empty((rows.size, cols.size), dtype=np.float64)
    if poison is not None:
        target.fill(poison)
    kernel(addr(source), addr(rows), addr(cols), addr(target), source.shape[1],
           rows.size, cols.size)
    return target


def _inputs(shape, nrows_out, ncols_out, seed=0):
    rng = np.random.default_rng(seed)
    source = np.ascontiguousarray(rng.normal(size=shape))
    rows = np.sort(rng.choice(shape[0], nrows_out, replace=False)).astype(np.int64)
    cols = np.sort(rng.choice(shape[1], ncols_out, replace=False)).astype(np.int64)
    return source, rows, cols


# Includes a slice far below the gate, the gate boundary itself, a slice just
# above it, and row counts on both sides of the worker count.
SIZES = [
    ((8, 8), 8, 8),
    ((1_000, 400), 750, 300),
    ((2_000, 1_000), 1_500, 750),
    ((2_000, 2_000), 2_000, 2_000),        # exactly 4M output elements
    ((4_000, 2_000), 3_000, 1_500),        # 4.5M: the measured crossover
    ((6_000, 2_000), 4_500, 1_500),
]


@pytest.mark.parametrize("case", SIZES,
                         ids=[f"{s[0][0]}x{s[0][1]}->{s[1]}x{s[2]}" for s in SIZES])
def test_parallel_matches_serial_bit_for_bit(case):
    shape, nrows_out, ncols_out = case
    source, rows, cols = _inputs(shape, nrows_out, ncols_out)
    serial = _take(lib().mad_dense_take_f64, source, rows, cols)
    # Poisoned, so a row the parallel kernel never wrote cannot agree by luck.
    parallel = _take(lib().mad_dense_take_f64_parallel, source, rows, cols, poison=-1.0)
    assert np.array_equal(serial.view(np.int64), parallel.view(np.int64))
    np.testing.assert_array_equal(parallel, source[np.ix_(rows, cols)])


def test_parallel_matches_serial_with_fewer_rows_than_workers():
    """n < workers must still call every work item exactly once."""
    source, rows, cols = _inputs((3, 700_000), 3, 200_000)
    assert rows.size < WORKERS
    serial = _take(lib().mad_dense_take_f64, source, rows, cols)
    parallel = _take(lib().mad_dense_take_f64_parallel, source, rows, cols, poison=-1.0)
    assert np.array_equal(serial.view(np.int64), parallel.view(np.int64))


def _routed_symbol(monkeypatch, shape, nrows_out, ncols_out):
    called = {}
    real = lib()
    for name in ("mad_dense_take_f64", "mad_dense_take_f64_parallel",
                 "mad_dense_take_f64_copy"):
        def make(name):
            def record(*args):
                called["name"] = name
            return record
        monkeypatch.setattr(real, name, make(name))
    source, rows, cols = _inputs(shape, nrows_out, ncols_out)
    _take_dense(source, rows, cols)
    return called.get("name")


def test_take_dense_routes_by_size(monkeypatch):
    # 4.5M elements is below the 8M gate: real callers at the crossover stay serial.
    assert _routed_symbol(monkeypatch, (4_000, 2_000), 3_000, 1_500) == "mad_dense_take_f64"
    # 9M elements is above it.
    assert _routed_symbol(monkeypatch, (8_000, 2_000), 6_000, 1_500) == \
        "mad_dense_take_f64_parallel"
    # Below the gate even when wide: 5.6M elements.
    assert _routed_symbol(monkeypatch, (5_000, 2_000), 3_750, 1_500) == "mad_dense_take_f64"


def test_take_dense_full_column_copy_stays_on_copy_kernel(monkeypatch):
    # An identity column set uses the contiguous copy kernel at any size.
    called = _routed_symbol(monkeypatch, (8_000, 2_000), 6_000, 2_000)
    assert called == "mad_dense_take_f64_copy"


def test_gated_result_matches_serial_end_to_end():
    """The same slice above the gate, through _take_dense with the gate moved,
    is identical to the ungated result."""
    source, rows, cols = _inputs((8_000, 2_000), 6_000, 1_500)
    saved = _core.DENSE_PARALLEL_ELEMENTS
    try:
        _core.DENSE_PARALLEL_ELEMENTS = 0
        parallel = _take_dense(source, rows, cols)
        _core.DENSE_PARALLEL_ELEMENTS = 10**12
        serial = _take_dense(source, rows, cols)
    finally:
        _core.DENSE_PARALLEL_ELEMENTS = saved
    assert np.array_equal(serial.view(np.int64), parallel.view(np.int64))


def test_worker_count_is_capped():
    assert 1 <= WORKERS <= 4
