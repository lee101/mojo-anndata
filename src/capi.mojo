"""C ABI for dense and CSR annotated-matrix slicing.

The Python layer owns allocation and index normalization.  Addresses are passed
as Int because exported Mojo functions cannot have parametric pointer types.
"""

from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def dense_take_row(src: FPtr, rows: IPtr, cols: IPtr, dst: FPtr,
                   ncols: Int, ncols_out: Int, i: Int):
    comptime W = simd_width_of[DType.float64]()
    var src_base = Int(rows[i]) * ncols
    var dst_base = i * ncols_out
    var j = 0
    while j + W <= ncols_out:
        var offsets = cols.load[width=W](j)
        var values = (src + src_base).gather[width=W](offsets)
        dst.store(dst_base + j, values)
        j += W
    while j < ncols_out:
        dst[dst_base + j] = src[src_base + Int(cols[j])]
        j += 1


@export("mad_dense_take_f64")
def mad_dense_take_f64(src_addr: Int, row_addr: Int, col_addr: Int, dst_addr: Int,
                       ncols: Int, nrows_out: Int, ncols_out: Int) abi("C"):
    var src = FPtr(unsafe_from_address=src_addr)
    var rows = IPtr(unsafe_from_address=row_addr)
    var cols = IPtr(unsafe_from_address=col_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    @parameter
    def take_row(i: Int):
        dense_take_row(src, rows, cols, dst, ncols, ncols_out, i)
    if nrows_out * ncols_out >= 4_000_000:
        parallelize[take_row](nrows_out, 8)
    else:
        for i in range(nrows_out):
            take_row(i)


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
    out_indptr[0] = 0
    for i in range(nrows_out):
        var source_row = Int(rows[i])
        var start = Int(indptr[source_row])
        var stop = Int(indptr[source_row + 1])
        for p in range(start, stop):
            var mapped = col_map[Int(indices[p])]
            if mapped >= 0:
                out_indices[written] = mapped
                out_data[written] = data[p]
                written += 1
        out_indptr[i + 1] = Int64(written)
    return written
