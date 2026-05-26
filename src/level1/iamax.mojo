# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Index of Maximum Absolute Value (`level1.iamax`)
============================================
Provides index of maximum absolute value operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of
from ._tuning import IAMAX_N_THREADS, IAMAX_PAR_THRESHOLD


def _iamax_serial[
    dtype: DType, simd_width: Int
](x: BLASPtr[dtype, _], start: Int, length: Int,) -> Tuple[Scalar[dtype], Int]:
    """Find max abs value and its absolute index in x[start:start+length]."""
    var best_val = Scalar[dtype](0)
    var best_idx = start
    var i = 0

    while i + simd_width <= length:
        var av = abs(x.load[width=simd_width](start + i))
        var chunk_max = av.reduce_max()
        if chunk_max > best_val:
            best_val = chunk_max
            for j in range(simd_width):
                if abs(x[start + i + j]) == chunk_max:
                    best_idx = start + i + j
                    break
        i += simd_width

    while i < length:
        var cur = abs(x[start + i])
        if cur > best_val:
            best_val = cur
            best_idx = start + i
        i += 1

    return best_val, best_idx


def iamax[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
    *,
    n_threads: Int = IAMAX_N_THREADS,
    par_threshold: Int = IAMAX_PAR_THRESHOLD,
](n: Int, x: BLASPtr[dtype, origin], incx: Int) -> Int:
    """
    Find the index of the element with maximum absolute value in vector X.

    Parameters:
        mut: Indicates whether the pointer is mutable (True) or immutable (False).
        origin: Memory origin of the pointer x.
        dtype: Data type of the elements in vector X.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The index (0-based) of the element with maximum absolute value.
        Returns 0 if n <= 0.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 0

    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1 and n > par_threshold:
        var local_max = alloc[Scalar[dtype]](n_threads)
        var local_idx = alloc[Int](n_threads)
        var chunk_size = (n + n_threads - 1) // n_threads

        @parameter
        def worker(tid: Int):
            var start = tid * chunk_size
            var end = min(start + chunk_size, n)
            if end <= start:
                local_max[tid] = Scalar[dtype](0)
                local_idx[tid] = start
                return
            var result = _iamax_serial[dtype, simd_width](x, start, end - start)
            local_max[tid] = result[0]
            local_idx[tid] = result[1]

        parallelize[worker](n_threads)

        var imax = local_idx[0]
        var max_val = local_max[0]
        for t in range(1, n_threads):
            if local_max[t] > max_val:
                max_val = local_max[t]
                imax = local_idx[t]

        local_max.free()
        local_idx.free()
        return imax

    if incx == 1:
        var result = _iamax_serial[dtype, simd_width](x, 0, n)
        return result[1]

    var ix: Int = 0
    var imax: Int = 0
    var max_val: Scalar[dtype] = abs(x[0])
    if incx < 0:
        ix = (-n + 1) * incx
    ix += incx
    for i in range(1, n):
        var current_abs = abs(x[ix])
        if current_abs > max_val:
            max_val = current_abs
            imax = i
        ix += incx
    return imax
