# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Euclidean Norm Operations (`level1.nrm2`)
============================================
Provides Euclidean norm operations as defined in the BLAS library standard.
"""

from std.math import sqrt
from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of
from ._tuning import NRM2_N_THREADS, NRM2_PAR_THRESHOLD, NRM2_N_ACC


def _nrm2_serial[
    dtype: DType, simd_width: Int, n_acc: Int
](xc: BLASPtr[dtype, _], length: Int,) -> Scalar[dtype]:
    """Inner nrm2 kernel with n_acc (4) independent SIMD accumulators."""
    comptime stride: Int = simd_width * n_acc

    var acc0 = SIMD[dtype, simd_width](0)
    var acc1 = SIMD[dtype, simd_width](0)
    var acc2 = SIMD[dtype, simd_width](0)
    var acc3 = SIMD[dtype, simd_width](0)

    var i = 0
    while i + stride <= length:
        var v0 = xc.load[width=simd_width](i + 0 * simd_width)
        var v1 = xc.load[width=simd_width](i + 1 * simd_width)
        var v2 = xc.load[width=simd_width](i + 2 * simd_width)
        var v3 = xc.load[width=simd_width](i + 3 * simd_width)
        acc0 += v0 * v0
        acc1 += v1 * v1
        acc2 += v2 * v2
        acc3 += v3 * v3
        i += stride

    while i + simd_width <= length:
        var v = xc.load[width=simd_width](i)
        acc0 += v * v
        i += simd_width

    var result = (acc0 + acc1 + acc2 + acc3).reduce_add()

    while i < length:
        result += xc[i] * xc[i]
        i += 1

    return result


def nrm2[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
    *,
    n_threads: Int = NRM2_N_THREADS,
    par_threshold: Int = NRM2_PAR_THRESHOLD,
    n_acc: Int = NRM2_N_ACC,
](n: Int, x: BLASPtr[dtype, origin], incx: Int,) -> Scalar[dtype]:
    """
    Compute the Euclidean norm (2-norm) of a vector X.

    Parameters:
        mut: Indicates whether the pointer is mutable (True) or immutable (False).
        origin: Memory origin of the pointer x.
        dtype: Data type of the elements in vector X.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.
        n_acc: Number of independent SIMD accumulators in the inner kernel.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The Euclidean norm as a scalar value.
    """
    result: Scalar[dtype] = 0
    if n <= 0:
        return result

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1:
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
                partials[tid] = _nrm2_serial[dtype, simd_width, n_acc](
                    x + start, length
                )

            parallelize[worker](n_threads)
            for i in range(n_threads):
                result += partials[i]
            partials.free()
            return sqrt(result)

        return sqrt(_nrm2_serial[dtype, simd_width, n_acc](x, n))

    var ix: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    for _ in range(n):
        result += x[ix] * x[ix]
        ix += incx
    return sqrt(result)
