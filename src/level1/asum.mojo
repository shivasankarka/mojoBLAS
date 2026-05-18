# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Absolute Sum Operations (`level1.asum`)
============================================
Provides absolute sum operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of
from ._tuning import ASUM_N_THREADS, ASUM_PAR_THRESHOLD, ASUM_N_ACC


def _asum_serial[
    dtype: DType, simd_width: Int, n_acc: Int
](xc: BLASPtr[dtype, _], length: Int,) -> Scalar[dtype]:
    """Inner asum kernel with n_acc independent SIMD accumulators."""
    comptime stride: Int = simd_width * n_acc

    # I wonder if we can create n_acc accumulators at comptime without adding it manually, stack alloc?
    var acc0 = SIMD[dtype, simd_width](0)
    var acc1 = SIMD[dtype, simd_width](0)
    var acc2 = SIMD[dtype, simd_width](0)
    var acc3 = SIMD[dtype, simd_width](0)

    var i = 0
    while i + stride <= length:
        acc0 += abs(xc.load[width=simd_width](i + 0 * simd_width))
        acc1 += abs(xc.load[width=simd_width](i + 1 * simd_width))
        acc2 += abs(xc.load[width=simd_width](i + 2 * simd_width))
        acc3 += abs(xc.load[width=simd_width](i + 3 * simd_width))
        i += stride

    while i + simd_width <= length:
        acc0 += abs(xc.load[width=simd_width](i))
        i += simd_width

    var result = (acc0 + acc1 + acc2 + acc3).reduce_add()

    while i < length:
        result += abs(xc[i])
        i += 1

    return result


def asum[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
    *,
    n_threads: Int = ASUM_N_THREADS,
    par_threshold: Int = ASUM_PAR_THRESHOLD,
    n_acc: Int = ASUM_N_ACC,
](n: Int, dx: BLASPtr[dtype, origin], incx: Int,) -> Scalar[dtype]:
    """
    Compute the sum of absolute values of elements in vector X.

    Parameters:
        mut: Indicates whether the pointer is mutable (True) or immutable (False).
        origin: Memory origin of the pointer dx.
        dtype: Data type of the elements in vector X.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.
        n_acc: Number of independent SIMD accumulators in the inner kernel.

    Args:
        n: Number of elements in vector X.
        dx: Pointer to the first element of vector X. Dimension should be at least (1 + (n - 1) * abs(incx)).
        incx: Increment for the elements of X.

    Returns:
        The sum of absolute values as a scalar value.
    """
    var result: Scalar[dtype] = 0.0

    if n <= 0 or incx == 0:
        return result

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1:
        if n > par_threshold:
            # TODO: make partials stack allocated/SIMD since it's always small?
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
                partials[tid] = _asum_serial[dtype, simd_width, n_acc](
                    dx + start, length
                )

            parallelize[worker](n_threads)
            for i in range(n_threads):
                result += partials[i]
            partials.free()
            return result

        return _asum_serial[dtype, simd_width, n_acc](dx, n)

    var ix: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    for _ in range(n):
        result += abs(dx[ix])
        ix += incx
    return result


# NOTE: Using internal complex scalar type. not sure if this is the best way because we can't do vectorization on this UnsafePointer[ComplexScalar[dtype]].
# def cdasum[dtype: DType](
#     out dasum: Scalar[dtype],
#     n: Int,
#     dx: BLASPtr[ComplexScalar[dtype]],
#     incx: Int
# ):
#     """
#     Compute the sum of absolute values of elements in complex vector X.

#     Args:
#         n: Number of elements in vector X.
#         dx: Pointer to the first element of vector X. Dimension should be at least (1 + (n - 1) * abs(incx)).
#         incx: Increment for the elements of X.

#     Returns:
#         The sum of absolute values as a scalar value.
#     """
#     comptime simd_width: Int = simd_width_of[dtype]()
#     dasum: Scalar[dtype] = 0.0

#     if n <= 0 or incx <= 0:
#         return

#     if incx == 1:
#         for i in range(n):
#             dasum += cmplx_abs(dx[i])
#         return

#     var nincx: Int = n * incx
#     for i in range(0, nincx, incx):
#         dasum += cmplx_abs(dx[i])
#     return
