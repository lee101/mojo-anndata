"""C ABI for dense and CSR annotated-matrix slicing.

The Python layer owns allocation and index normalization.  Addresses are passed
as Int because exported Mojo functions cannot have parametric pointer types.
"""

from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime DENSE_PARALLEL_TASKS = 8
comptime W = simd_width_of[DType.float64]()
# Contiguous row copies step by 2*W doubles; W alone was not enough width for a
# memory-bound copy and this is the body the copy fast path uses (README).
comptime DENSE_COPY_WIDTH = W * 2


def dense_take_row(src: FPtr, rows: IPtr, cols: IPtr, dst: FPtr,
                   ncols: Int, ncols_out: Int, i: Int):
    var src_base = Int(rows.unsafe_load(i)) * ncols
    var dst_base = i * ncols_out
    var j = 0
    while j + W <= ncols_out:
        var offsets = cols.unsafe_load[width=W](j).cast[DType.int32]()
        var values = src.unsafe_offset(src_base).unsafe_gather[width=W](offsets)
        dst.unsafe_store[width=W](dst_base + j, values)
        j += W
    while j < ncols_out:
        dst.unsafe_store(dst_base + j, src.unsafe_load(src_base + Int(cols.unsafe_load(j))))
        j += 1


@export("mad_dense_take_f64")
def mad_dense_take_f64(src_addr: Int, row_addr: Int, col_addr: Int, dst_addr: Int,
                       ncols: Int, nrows_out: Int, ncols_out: Int) abi("C"):
    var src = FPtr(unsafe_from_address=src_addr)
    var rows = IPtr(unsafe_from_address=row_addr)
    var cols = IPtr(unsafe_from_address=col_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(nrows_out):
        dense_take_row(src, rows, cols, dst, ncols, ncols_out, i)


@export("mad_dense_take_f64_parallel")
def mad_dense_take_f64_parallel(src_addr: Int, row_addr: Int, col_addr: Int,
                                dst_addr: Int, ncols: Int, nrows_out: Int,
                                ncols_out: Int) abi("C"):
    var src = FPtr(unsafe_from_address=src_addr)
    var rows = IPtr(unsafe_from_address=row_addr)
    var cols = IPtr(unsafe_from_address=col_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for task in range(DENSE_PARALLEL_TASKS):
        var start = task * nrows_out // DENSE_PARALLEL_TASKS
        var stop = (task + 1) * nrows_out // DENSE_PARALLEL_TASKS
        for i in range(start, stop):
            dense_take_row(src, rows, cols, dst, ncols, ncols_out, i)


@export("mad_dense_take_f64_copy")
def mad_dense_take_f64_copy(src_addr: Int, row_addr: Int, col_addr: Int, dst_addr: Int,
                            ncols: Int, nrows_out: Int, ncols_out: Int) abi("C"):
    """`dst = src[rows, :]` with no column selection (i.e. `copy()` and any
    full-column slice): DENSE_COPY_WIDTH doubles per step, scalar tail.

    The Python layer only calls this when `cols` is the identity.  The loop is
    a flat `for i in range(nrows_out)` and never reads `cols`.  On this pinned
    toolchain only this row-loop form with a 2*W body lowered to a sound
    contiguous copy; the `while i < nrows_out` form, the same loop called as a
    helper per row, and a chunked `for task` over `(start, stop)` each emitted a
    blocked body that reloaded the destination pointer from a stack slot, so
    every row after the first wrote to row 0 (wrong values + heap corruption).
    `dense_take_row` keeps the `for`-row form too, for the same reason."""
    var src = FPtr(unsafe_from_address=src_addr)
    var rows = IPtr(unsafe_from_address=row_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(nrows_out):
        var src_base = Int(rows.unsafe_load(i)) * ncols
        var dst_base = i * ncols
        var j = 0
        while j + DENSE_COPY_WIDTH <= ncols:
            var values = src.unsafe_offset(src_base).unsafe_load[width=DENSE_COPY_WIDTH](j)
            dst.unsafe_store[width=DENSE_COPY_WIDTH](dst_base + j, values)
            j += DENSE_COPY_WIDTH
        while j < ncols:
            dst.unsafe_store(dst_base + j, src.unsafe_load(src_base + j))
            j += 1


@export("mad_csr_take_f64")
def mad_csr_take_f64(indptr_addr: Int, indices_addr: Int, data_addr: Int,
                     row_addr: Int, col_map_addr: Int, out_indptr_addr: Int,
                     out_indices_addr: Int, out_data_addr: Int, nrows_out: Int) abi("C") -> Int:
    var indptr = IPtr(unsafe_from_address=indptr_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var data = FPtr(unsafe_from_address=data_addr)
    var rows = IPtr(unsafe_from_address=row_addr)
    var col_map = IPtr(unsafe_from_address=col_map_addr)
    var out_indptr = IPtr(unsafe_from_address=out_indptr_addr)
    var out_indices = IPtr(unsafe_from_address=out_indices_addr)
    var out_data = FPtr(unsafe_from_address=out_data_addr)
    var written = 0
    out_indptr[unsafe_offset=0] = 0
    for i in range(nrows_out):
        var source_row = Int(rows[unsafe_offset=i])
        var start = Int(indptr[unsafe_offset=source_row])
        var stop = Int(indptr[unsafe_offset=source_row + 1])
        for p in range(start, stop):
            var mapped = col_map[unsafe_offset=Int(indices[unsafe_offset=p])]
            if mapped >= 0:
                out_indices[unsafe_offset=written] = mapped
                out_data[unsafe_offset=written] = data[unsafe_offset=p]
                written += 1
        out_indptr[unsafe_offset=i + 1] = Int64(written)
    return written


@export("mad_csr_take_f64_i32")
def mad_csr_take_f64_i32(indptr_addr: Int, indices_addr: Int, data_addr: Int,
                         row_addr: Int, col_map_addr: Int, out_indptr_addr: Int,
                         out_indices_addr: Int, out_data_addr: Int,
                         nrows_out: Int) abi("C") -> Int:
    var indptr = I32Ptr(unsafe_from_address=indptr_addr)
    var indices = I32Ptr(unsafe_from_address=indices_addr)
    var data = FPtr(unsafe_from_address=data_addr)
    var rows = IPtr(unsafe_from_address=row_addr)
    var col_map = I32Ptr(unsafe_from_address=col_map_addr)
    var out_indptr = I32Ptr(unsafe_from_address=out_indptr_addr)
    var out_indices = I32Ptr(unsafe_from_address=out_indices_addr)
    var out_data = FPtr(unsafe_from_address=out_data_addr)
    var written = 0
    out_indptr[unsafe_offset=0] = 0
    for i in range(nrows_out):
        var source_row = Int(rows[unsafe_offset=i])
        var start = Int(indptr[unsafe_offset=source_row])
        var stop = Int(indptr[unsafe_offset=source_row + 1])
        for p in range(start, stop):
            var mapped = col_map[unsafe_offset=Int(indices[unsafe_offset=p])]
            if mapped >= 0:
                out_indices[unsafe_offset=written] = mapped
                out_data[unsafe_offset=written] = data[unsafe_offset=p]
                written += 1
        out_indptr[unsafe_offset=i + 1] = Int32(written)
    return written
