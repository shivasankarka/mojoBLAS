# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Plane Rotation Operations (`level1.rot`)
============================================
Provides plane rotation operations as defined in the BLAS library standard.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of

# ===----------------------------------------------------------------------=== #
# Max
# ===----------------------------------------------------------------------=== #
from max.algorithm.backend.cpu import parallelize

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.level1._tuning import (
    ROT_N_THREADS,
    ROT_PAR_THRESHOLD,
    ROT_UNROLL,
)
from mojoblas.type_aliases import BLASPtr


def rot[
    origin_x: MutOrigin,
    origin_y: MutOrigin,
    //,
    dtype: DType,
    *,
    n_threads: Int = ROT_N_THREADS,
    par_threshold: Int = ROT_PAR_THRESHOLD,
    unroll_factor: Int = ROT_UNROLL,
](
    n: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    y: BLASPtr[dtype, origin_y],
    incy: Int,
    c: Scalar[dtype],
    s: Scalar[dtype],
) -> None:
    """
    Apply Givens rotation to vectors X and Y.

    Performs the transformation:
    [ x[i] ]   [  c  s ] [ x[i] ]
    [ y[i] ] = [ -s  c ] [ y[i] ]

    Parameters:
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y.
        dtype: Data type of the elements in vectors X and Y.
        n_threads: Number of threads for parallel execution.
        par_threshold: Minimum n to switch to parallel execution.
        unroll_factor: Unroll factor for vectorized loops.

    Args:
        n: Number of elements in vectors X and Y.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        y: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.
        c: Cosine component of the rotation.
        s: Sine component of the rotation.
    """
    if n <= 0:
        return

    var ix: Int = 0
    var iy: Int = 0
    comptime simd_width: Int = simd_width_of[dtype]()

    # TODO: test if I can implement a accumulator alg here.
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
                var xc = x.unsafe_offset(start)
                var yc = y.unsafe_offset(start)

                def closure_p[width: Int](i: Int) {yc, xc, c, s}:
                    var xv = xc.unsafe_load[width=width](i)
                    var yv = yc.unsafe_load[width=width](i)
                    xc.unsafe_store[width=width](i, c * xv + s * yv)
                    yc.unsafe_store[width=width](i, -s * xv + c * yv)

                vectorize[simd_width, unroll_factor=unroll_factor](
                    length, closure_p
                )

            parallelize[worker](n_threads)
            return

        def closure[width: Int](i: Int) {y, x, c, s}:
            var xv = x.unsafe_load[width=width](i)
            var yv = y.unsafe_load[width=width](i)
            x.unsafe_store[width=width](i, c * xv + s * yv)
            y.unsafe_store[width=width](i, -s * xv + c * yv)

        vectorize[simd_width, unroll_factor=unroll_factor](n, closure)
        return

    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for _ in range(n):
        var temp_x = c * x[unsafe_offset=ix] + s * y[unsafe_offset=iy]
        var temp_y = -s * x[unsafe_offset=ix] + c * y[unsafe_offset=iy]
        x[unsafe_offset=ix] = temp_x
        y[unsafe_offset=iy] = temp_y
        ix += incx
        iy += incy
