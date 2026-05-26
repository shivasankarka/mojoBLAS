# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Vector Scale Operations (`level1.scal`)
============================================
Provides vector scale operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize, parallelize
from std.sys.info import simd_width_of
from ._tuning import SCAL_N_THREADS, SCAL_PAR_THRESHOLD, SCAL_UNROLL


def scal[
    origin: MutOrigin,
    //,
    dtype: DType,
    *,
    n_threads: Int = SCAL_N_THREADS,
    par_threshold: Int = SCAL_PAR_THRESHOLD,
    unroll_factor: Int = SCAL_UNROLL,
](n: Int, alpha: Scalar[dtype], dx: BLASPtr[dtype, origin], incx: Int,):
    """
    Scale a vector by a scalar: X := alpha * X.

    Parameters:
        origin: Memory origin of the pointer dx.
        dtype: Data type of the elements in vector X.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.
        unroll_factor: Unroll factor for vectorized loops.

    Args:
        n: Number of elements in vector X.
        alpha: Scalar multiplier.
        dx: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
    """
    if n <= 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1:
        if n > par_threshold:
            var chunk_size = (n + n_threads - 1) // n_threads

            @parameter
            def worker(tid: Int):
                var start = tid * chunk_size
                var end = min(start + chunk_size, n)
                var length = end - start
                if length <= 0:
                    return
                var xc = dx + start

                def closure_p[width: Int](i: Int) {xc, alpha}:
                    xc.store[width=width](i, alpha * xc.load[width=width](i))

                vectorize[simd_width, unroll_factor=unroll_factor](
                    length, closure_p
                )

            parallelize[worker](n_threads)
            return

        def closure[width: Int](i: Int) {dx, alpha}:
            dx.store[width=width](i, alpha * dx.load[width=width](i))

        vectorize[simd_width, unroll_factor=unroll_factor](n, closure)
        return

    var ix: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx

    for _ in range(n):
        dx[ix] = alpha * dx[ix]
        ix += incx
