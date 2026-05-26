# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Modified Givens Rotation Operations (`level1.rotm`)
============================================
Provides modified Givens rotation operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize, parallelize
from std.sys.info import simd_width_of
from ._tuning import ROTM_N_THREADS, ROTM_PAR_THRESHOLD, ROTM_UNROLL


def rotm[
    origin_x: MutOrigin,
    origin_y: MutOrigin,
    mut_param: Bool,
    origin_param: Origin[mut=mut_param],
    //,
    dtype: DType,
    *,
    n_threads: Int = ROTM_N_THREADS,
    par_threshold: Int = ROTM_PAR_THRESHOLD,
    unroll_factor: Int = ROTM_UNROLL,
](
    n: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    y: BLASPtr[dtype, origin_y],
    incy: Int,
    param: BLASPtr[dtype, origin_param],
) -> None:
    """
    Apply a modified Givens rotation to vectors X and Y.

    The rotation is defined by the 5-element parameter vector param, where
    param[0] is the flag (-2, -1, 0, or 1) controlling which H matrix elements
    are used, and param[1..4] hold h11, h21, h12, h22 as needed.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vectors X and Y.
        x: Pointer to the first element of vector X (input/output).
        incx: Increment for the elements of X.
        y: Pointer to the first element of vector Y (input/output).
        incy: Increment for the elements of Y.
        param: 5-element parameter vector defining the rotation.
    """
    if n <= 0:
        return

    var flag = param[0]
    if flag == -2:
        return

    var h11: Scalar[dtype] = 0
    var h12: Scalar[dtype] = 0
    var h21: Scalar[dtype] = 0
    var h22: Scalar[dtype] = 0

    if flag < 0:
        h11 = param[1]
        h21 = param[2]
        h12 = param[3]
        h22 = param[4]
    elif flag == 0:
        h12 = param[3]
        h21 = param[2]
    else:
        h11 = param[1]
        h22 = param[4]

    if incx == 1 and incy == 1:
        comptime simd_width: Int = simd_width_of[dtype]()

        if flag < 0:
            # x[i] = h11*x[i] + h12*y[i]
            # y[i] = h21*x[i] + h22*y[i]
            if n > par_threshold:
                var chunk_size = (n + n_threads - 1) // n_threads

                @parameter
                def worker_neg(tid: Int):
                    var start = tid * chunk_size
                    var end = min(start + chunk_size, n)
                    var length = end - start
                    if length <= 0:
                        return
                    var xc = x + start
                    var yc = y + start

                    def closure[
                        width: Int
                    ](i: Int) {xc, yc, h11, h12, h21, h22}:
                        var xv = xc.load[width=width](i)
                        var yv = yc.load[width=width](i)
                        xc.store[width=width](i, h11 * xv + h12 * yv)
                        yc.store[width=width](i, h21 * xv + h22 * yv)

                    vectorize[simd_width, unroll_factor=unroll_factor](
                        length, closure
                    )

                parallelize[worker_neg](n_threads)
            else:

                def closure_neg[width: Int](i: Int) {x, y, h11, h12, h21, h22}:
                    var xv = x.load[width=width](i)
                    var yv = y.load[width=width](i)
                    x.store[width=width](i, h11 * xv + h12 * yv)
                    y.store[width=width](i, h21 * xv + h22 * yv)

                vectorize[simd_width, unroll_factor=unroll_factor](
                    n, closure_neg
                )

        elif flag == 0:
            # x[i] = x[i] + h12*y[i], y[i] = h21*x[i] + y[i]
            if n > par_threshold:
                var chunk_size = (n + n_threads - 1) // n_threads

                @parameter
                def worker_zero(tid: Int):
                    var start = tid * chunk_size
                    var end = min(start + chunk_size, n)
                    var length = end - start
                    if length <= 0:
                        return
                    var xc = x + start
                    var yc = y + start

                    def closure[width: Int](i: Int) {xc, yc, h12, h21}:
                        var xv = xc.load[width=width](i)
                        var yv = yc.load[width=width](i)
                        xc.store[width=width](i, xv + h12 * yv)
                        yc.store[width=width](i, h21 * xv + yv)

                    vectorize[simd_width, unroll_factor=unroll_factor](
                        length, closure
                    )

                parallelize[worker_zero](n_threads)
            else:

                def closure_zero[width: Int](i: Int) {x, y, h12, h21}:
                    var xv = x.load[width=width](i)
                    var yv = y.load[width=width](i)
                    x.store[width=width](i, xv + h12 * yv)
                    y.store[width=width](i, h21 * xv + yv)

                vectorize[simd_width, unroll_factor=unroll_factor](
                    n, closure_zero
                )

        else:
            # flag > 0: h12 = h21 = implied (-1/+1); x[i] = h11*x[i] + y[i], y[i] = -x[i] + h22*y[i]
            if n > par_threshold:
                var chunk_size = (n + n_threads - 1) // n_threads

                @parameter
                def worker_pos(tid: Int):
                    var start = tid * chunk_size
                    var end = min(start + chunk_size, n)
                    var length = end - start
                    if length <= 0:
                        return
                    var xc = x + start
                    var yc = y + start

                    def closure[width: Int](i: Int) {xc, yc, h11, h22}:
                        var xv = xc.load[width=width](i)
                        var yv = yc.load[width=width](i)
                        xc.store[width=width](i, h11 * xv + yv)
                        yc.store[width=width](i, -xv + h22 * yv)

                    vectorize[simd_width, unroll_factor=unroll_factor](
                        length, closure
                    )

                parallelize[worker_pos](n_threads)
            else:

                def closure_pos[width: Int](i: Int) {x, y, h11, h22}:
                    var xv = x.load[width=width](i)
                    var yv = y.load[width=width](i)
                    x.store[width=width](i, h11 * xv + yv)
                    y.store[width=width](i, -xv + h22 * yv)

                vectorize[simd_width, unroll_factor=unroll_factor](
                    n, closure_pos
                )

        return

    var kx = 0 if incx > 0 else (1 - n) * incx
    var ky = 0 if incy > 0 else (1 - n) * incy
    if flag < 0:
        for _ in range(n):
            var w = x[kx]
            var z = y[ky]
            x[kx] = w * h11 + z * h12
            y[ky] = w * h21 + z * h22
            kx += incx
            ky += incy
    elif flag == 0:
        for _ in range(n):
            var w = x[kx]
            var z = y[ky]
            x[kx] = w + z * h12
            y[ky] = w * h21 + z
            kx += incx
            ky += incy
    else:
        for _ in range(n):
            var w = x[kx]
            var z = y[ky]
            x[kx] = w * h11 + z
            y[ky] = -w + h22 * z
            kx += incx
            ky += incy
