# ===----------------------------------------------------------------------=== #
# mojoBLAS: Symmetric Packed Rank-1 Operations
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #
"""
Symmetric Packed Rank-1 Operations (mojoblas.level2.spr).
=========================================================
Provides symmetric packed rank-1 operations as defined in the BLAS library standard.
"""

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.type_aliases import BLASPtr


def spr[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    origin_ap: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    ap: BLASPtr[dtype, origin_ap],
):
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif n < 0:
        info = 2
    elif incx == 0:
        info = 5

    if info != 0:
        print("spr: Info", info)
        return

    if n == 0 or alpha == 0:
        return

    var upper = uplo == "U" or uplo == "u"
    var kx: Int = 0 if incx > 0 else (1 - n) * incx
    var kk: Int = 0

    if upper:
        if incx == 1:
            for j in range(n):
                if x[unsafe_offset=j] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=j]
                    for i in range(j + 1):
                        ap[unsafe_offset=kk + i] = (
                            ap[unsafe_offset=kk + i] + x[unsafe_offset=i] * temp
                        )
                kk += j + 1
        else:
            var jx: Int = kx
            for j in range(n):
                if x[unsafe_offset=jx] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=jx]
                    var ix: Int = kx
                    for k in range(kk, kk + j + 1):
                        ap[unsafe_offset=k] = (
                            ap[unsafe_offset=k] + x[unsafe_offset=ix] * temp
                        )
                        ix += incx
                jx += incx
                kk += j + 1
    else:
        if incx == 1:
            for j in range(n):
                if x[unsafe_offset=j] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=j]
                    for i in range(j, n):
                        ap[unsafe_offset=kk + i - j] = (
                            ap[unsafe_offset=kk + i - j]
                            + x[unsafe_offset=i] * temp
                        )
                kk += n - j
        else:
            var jx: Int = kx
            for j in range(n):
                if x[unsafe_offset=jx] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=jx]
                    var ix: Int = jx
                    for k in range(kk, kk + n - j):
                        ap[unsafe_offset=k] = (
                            ap[unsafe_offset=k] + x[unsafe_offset=ix] * temp
                        )
                        ix += incx
                jx += incx
                kk += n - j

    return
