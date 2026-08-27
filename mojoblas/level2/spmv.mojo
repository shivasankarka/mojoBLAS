# ===----------------------------------------------------------------------=== #
# mojoBLAS: Symmetric Packed Matrix-Vector Operations
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #
"""
Symmetric Packed Matrix-Vector Operations (mojoblas.level2.spmv).
=================================================================
Provides symmetric packed matrix-vector operations as defined in the BLAS library standard.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory.alloc import unsafe_alloc

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.type_aliases import BLASPtr


def spmv[
    mut_ap: Bool,
    mut_x: Bool,
    origin_ap: Origin[mut=mut_ap],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    ap: BLASPtr[dtype, origin_ap],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    beta: Scalar[dtype],
    y: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Performs the matrix-vector operation y := alpha*A*x + beta*y,
    where A is an n by n symmetric matrix stored in packed format.

    Parameters:
        mut_ap: Indicates whether the pointer ap is mutable (True) or immutable (False).
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        origin_ap: Memory origin of the pointer ap.
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        alpha: The scalar multiplier for the matrix-vector product.
        ap: A pointer to the packed symmetric matrix A.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
        beta: The scalar multiplier for the vector y.
        y: A pointer to the first element of the vector y (input/output).
        incy: The increment for the elements of y.
    """
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif n < 0:
        info = 2
    elif incx == 0:
        info = 5
    elif incy == 0:
        info = 8

    if info != 0:
        print("spmv: Info", info)
        return

    if n == 0 or (alpha == 0 and beta == 1):
        return

    var upper = uplo == "U" or uplo == "u"

    if incx == 1 and incy == 1:
        if beta == 0:
            for i in range(n):
                y[unsafe_offset=i] = 0
        elif beta != 1:
            for i in range(n):
                y[unsafe_offset=i] = beta * y[unsafe_offset=i]

        if alpha == 0:
            return

        if upper:
            var kk: Int = 0
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[unsafe_offset=j]
                var temp2: Scalar[dtype] = 0
                for i in range(j):
                    var aij = ap[unsafe_offset=kk + i]
                    y[unsafe_offset=i] = y[unsafe_offset=i] + temp1 * aij
                    temp2 = temp2 + aij * x[unsafe_offset=i]
                y[unsafe_offset=j] = (
                    y[unsafe_offset=j]
                    + temp1 * ap[unsafe_offset=kk + j]
                    + alpha * temp2
                )
                kk += j + 1
        else:
            var kk: Int = 0
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[unsafe_offset=j]
                var temp2: Scalar[dtype] = 0
                y[unsafe_offset=j] = (
                    y[unsafe_offset=j] + temp1 * ap[unsafe_offset=kk]
                )
                for i in range(j + 1, n):
                    var aij = ap[unsafe_offset=kk + i - j]
                    y[unsafe_offset=i] = y[unsafe_offset=i] + temp1 * aij
                    temp2 = temp2 + aij * x[unsafe_offset=i]
                y[unsafe_offset=j] = y[unsafe_offset=j] + alpha * temp2
                kk += n - j
        return

    var xbuf = unsafe_alloc[Scalar[dtype]](n)
    var ybuf = unsafe_alloc[Scalar[dtype]](n)

    var kx: Int = 0 if incx > 0 else (1 - n) * incx
    var ky: Int = 0 if incy > 0 else (1 - n) * incy

    var ix: Int = kx
    var iy: Int = ky
    for i in range(n):
        xbuf[unsafe_offset=i] = x[unsafe_offset=ix]
        ybuf[unsafe_offset=i] = y[unsafe_offset=iy]
        ix += incx
        iy += incy

    for i in range(n):
        var sum: Scalar[dtype] = 0
        for j in range(n):
            var ii = i
            var jj = j
            if upper:
                if ii > jj:
                    ii = j
                    jj = i
                var idx = (jj * (jj + 1)) // 2 + ii
                sum = sum + ap[unsafe_offset=idx] * xbuf[unsafe_offset=j]
            else:
                if ii < jj:
                    ii = j
                    jj = i
                var start = jj * n - (jj * (jj - 1)) // 2
                var idx = start + (ii - jj)
                sum = sum + ap[unsafe_offset=idx] * xbuf[unsafe_offset=j]
        ybuf[unsafe_offset=i] = alpha * sum + beta * ybuf[unsafe_offset=i]

    iy = ky
    for i in range(n):
        y[unsafe_offset=iy] = ybuf[unsafe_offset=i]
        iy += incy

    xbuf.unsafe_free()
    ybuf.unsafe_free()
