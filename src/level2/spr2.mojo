# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Packed Rank-2 Operations (`level2.spr2`)
=============================================
Provides symmetric packed rank-2 operations as defined in the BLAS library standard.
"""


def spr2[
    mut_x: Bool,
    mut_y: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: Origin[mut=mut_y],
    origin_ap: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    y: BLASPtr[dtype, origin_y],
    incy: Int,
    ap: BLASPtr[dtype, origin_ap],
):
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif n < 0:
        info = 2
    elif incx == 0:
        info = 5
    elif incy == 0:
        info = 7

    if info != 0:
        print("spr2: Info", info)
        return

    if n == 0 or alpha == 0:
        return

    var upper = uplo == "U" or uplo == "u"
    var kx: Int = 0 if incx > 0 else (1 - n) * incx
    var ky: Int = 0 if incy > 0 else (1 - n) * incy
    var kk: Int = 0

    if upper:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0 or y[j] != 0:
                    var temp1: Scalar[dtype] = alpha * y[j]
                    var temp2: Scalar[dtype] = alpha * x[j]
                    for i in range(j + 1):
                        ap[kk + i] = ap[kk + i] + x[i] * temp1 + y[i] * temp2
                kk += j + 1
        else:
            var jx: Int = kx
            var jy: Int = ky
            for j in range(n):
                if x[jx] != 0 or y[jy] != 0:
                    var temp1: Scalar[dtype] = alpha * y[jy]
                    var temp2: Scalar[dtype] = alpha * x[jx]
                    var ix: Int = kx
                    var iy: Int = ky
                    for k in range(kk, kk + j + 1):
                        ap[k] = ap[k] + x[ix] * temp1 + y[iy] * temp2
                        ix += incx
                        iy += incy
                jx += incx
                jy += incy
                kk += j + 1
    else:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0 or y[j] != 0:
                    var temp1: Scalar[dtype] = alpha * y[j]
                    var temp2: Scalar[dtype] = alpha * x[j]
                    for i in range(j, n):
                        ap[kk + i - j] = (
                            ap[kk + i - j] + x[i] * temp1 + y[i] * temp2
                        )
                kk += n - j
        else:
            var jx: Int = kx
            var jy: Int = ky
            for j in range(n):
                if x[jx] != 0 or y[jy] != 0:
                    var temp1: Scalar[dtype] = alpha * y[jy]
                    var temp2: Scalar[dtype] = alpha * x[jx]
                    var ix: Int = jx
                    var iy: Int = jy
                    for k in range(kk, kk + n - j):
                        ap[k] = ap[k] + x[ix] * temp1 + y[iy] * temp2
                        ix += incx
                        iy += incy
                jx += incx
                jy += incy
                kk += n - j

    return
