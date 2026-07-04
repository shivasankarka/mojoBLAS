# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Dot Product Operations (`level1.dot`)
============================================
Provides dot product operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of
from ._tuning import DOT_N_THREADS, DOT_PAR_THRESHOLD, DOT_N_ACC


def _dot_serial[
    dtype: DType, simd_width: Int, n_acc: Int
](xc: BLASPtr[dtype, _], yc: BLASPtr[dtype, _], length: Int,) -> Scalar[dtype]:
    """Inner dot kernel with n_acc independent SIMD accumulators.

    Each accumulator holds a simd_width-wide lane of partial sums. They are
    independent with no data dependency between acc0, acc1, ..., so the CPU
    can issue n_acc * simd_width FMAs back to back.
    """
    comptime stride: Int = simd_width * n_acc

    var acc0 = SIMD[dtype, simd_width](0)
    var acc1 = SIMD[dtype, simd_width](0)
    var acc2 = SIMD[dtype, simd_width](0)
    var acc3 = SIMD[dtype, simd_width](0)

    var i = 0
    while i + stride <= length:
        acc0 += xc.load[width=simd_width](i + 0 * simd_width) * yc.load[
            width=simd_width
        ](i + 0 * simd_width)
        acc1 += xc.load[width=simd_width](i + 1 * simd_width) * yc.load[
            width=simd_width
        ](i + 1 * simd_width)
        acc2 += xc.load[width=simd_width](i + 2 * simd_width) * yc.load[
            width=simd_width
        ](i + 2 * simd_width)
        acc3 += xc.load[width=simd_width](i + 3 * simd_width) * yc.load[
            width=simd_width
        ](i + 3 * simd_width)
        i += stride

    while i + simd_width <= length:
        acc0 += xc.load[width=simd_width](i) * yc.load[width=simd_width](i)
        i += simd_width

    var result = (acc0 + acc1 + acc2 + acc3).reduce_add()

    while i < length:
        result += xc[i] * yc[i]
        i += 1

    return result


def dot[
    mut_x: Bool,
    mut_y: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: Origin[mut=mut_y],
    //,
    dtype: DType,
    *,
    n_threads: Int = DOT_N_THREADS,
    par_threshold: Int = DOT_PAR_THRESHOLD,
    n_acc: Int = DOT_N_ACC,
](
    n: Int,
    dx: BLASPtr[dtype, origin_x],
    incx: Int,
    dy: BLASPtr[dtype, origin_y],
    incy: Int,
) -> Scalar[dtype]:
    """
    Compute the dot product of two vectors X and Y.

    Parameters:
        mut_x: Indicates whether the pointer to vector X is mutable (True) or immutable (False).
        mut_y: Indicates whether the pointer to vector Y is mutable (True) or immutable (False).
        origin_x: Memory origin of the pointer to vector X.
        origin_y: Memory origin of the pointer to vector Y.
        dtype: Data type of the elements in vectors X and Y.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.
        n_acc: Number of independent SIMD accumulators in the inner kernel.

    Args:
        n: Number of elements in vectors X and Y.
        dx: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        dy: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.

    Returns:
        The dot product as a scalar value.
    """
    var result: Scalar[dtype] = 0
    # TODO: not sure if returning 0 is the best way to handle n <= 0 case. Check with BLAS spec.
    if n <= 0:
        return result

    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1 and incy == 1:
        if n > par_threshold:
            var partials = alloc[Scalar[dtype]](n_threads)
            for i in range(n_threads):
                partials[i] = 0
            var chunk_size = (n + n_threads - 1) // n_threads

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, n)
                var length = end - start
                if length <= 0:
                    partials[tid] = 0
                    return
                partials[tid] = _dot_serial[dtype, simd_width, n_acc](
                    dx + start, dy + start, length
                )

            parallelize[worker](n_threads)
            for i in range(n_threads):
                result += partials[i]
            partials.free()
            return result

        return _dot_serial[dtype, simd_width, n_acc](dx, dy, n)

    var ix: Int = 0
    var iy: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for _ in range(n):
        result += dx[ix] * dy[iy]
        ix += incx
        iy += incy

    return result
