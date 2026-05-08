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


def rotm[
    origin_x: MutOrigin,
    origin_y: MutOrigin,
    mut_param: Bool,
    origin_param: Origin[mut=mut_param],
    //,
    dtype: DType,
](
    n: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    y: BLASPtr[dtype, origin_y],
    incy: Int,
    param: BLASPtr[dtype, origin_param],
) -> None:
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

    if incx == incy and incx > 0:
        var nsteps = n * incx
        for i in range(0, nsteps, incx):
            var w = x[i]
            var z = y[i]
            if flag < 0:
                x[i] = w * h11 + z * h12
                y[i] = w * h21 + z * h22
            elif flag == 0:
                x[i] = w + z * h12
                y[i] = w * h21 + z
            else:
                x[i] = w * h11 + z
                y[i] = -w + h22 * z
        return

    var kx = 0 if incx > 0 else (1 - n) * incx
    var ky = 0 if incy > 0 else (1 - n) * incy
    for _ in range(n):
        var w = x[kx]
        var z = y[ky]
        if flag < 0:
            x[kx] = w * h11 + z * h12
            y[ky] = w * h21 + z * h22
        elif flag == 0:
            x[kx] = w + z * h12
            y[ky] = w * h21 + z
        else:
            x[kx] = w * h11 + z
            y[ky] = -w + h22 * z

        kx += incx
        ky += incy
