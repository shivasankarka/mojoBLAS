# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Vector Swap Operations (`level1.swap`)
============================================
Provides vector swap operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize, parallelize
from std.sys.info import simd_width_of
from ._tuning import SWAP_N_THREADS, SWAP_PAR_THRESHOLD, SWAP_UNROLL


# Named `vswap` to avoid conflict with `swap` in std library.
def vswap[
    origin_x: MutOrigin,
    origin_y: MutOrigin,
    //,
    dtype: DType,
    *,
    n_threads: Int = SWAP_N_THREADS,
    par_threshold: Int = SWAP_PAR_THRESHOLD,
    unroll_factor: Int = SWAP_UNROLL,
](
    n: Int,
    dx: BLASPtr[dtype, origin_x],
    incx: Int,
    dy: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Swap the elements of two vectors X and Y: X <-> Y.

    Parameters:
        origin_x: Memory origin of the pointer dx.
        origin_y: Memory origin of the pointer dy.
        dtype: Data type of the elements in vectors X and Y.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.
        unroll_factor: Unroll factor for vectorized loops.

    Args:
        n: Number of elements in vectors X and Y.
        dx: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        dy: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.
    """
    if n <= 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1 and incy == 1:
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
                var yc = dy + start

                def closure_p[width: Int](i: Int) {xc, yc}:
                    var temp = xc.load[width=width](i)
                    xc.store[width=width](i, yc.load[width=width](i))
                    yc.store[width=width](i, temp)

                vectorize[simd_width, unroll_factor=unroll_factor](
                    length, closure_p
                )

            parallelize[worker](n_threads)
            return

        def closure[width: Int](i: Int) {dx, dy}:
            var temp = dx.load[width=width](i)
            dx.store[width=width](i, dy.load[width=width](i))
            dy.store[width=width](i, temp)

        vectorize[simd_width, unroll_factor=unroll_factor](n, closure)
        return

    var ix: Int = 0
    var iy: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for _ in range(n):
        var temp = dx[ix]
        dx[ix] = dy[iy]
        dy[iy] = temp
        ix += incx
        iy += incy
