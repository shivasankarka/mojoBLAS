# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Packed Matrix-Vector Operations (`level2.spmv`)
=============================================
Provides symmetric packed matrix-vector operations as defined in the BLAS library standard.
"""


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

    # Fast path for contiguous vectors using direct packed indexing.
    if incx == 1 and incy == 1:
        if beta == 0:
            for i in range(n):
                y[i] = 0
        elif beta != 1:
            for i in range(n):
                y[i] = beta * y[i]

        if alpha == 0:
            return

        if upper:
            var kk: Int = 0
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[j]
                var temp2: Scalar[dtype] = 0
                for i in range(j):
                    var aij = ap[kk + i]
                    y[i] = y[i] + temp1 * aij
                    temp2 = temp2 + aij * x[i]
                y[j] = y[j] + temp1 * ap[kk + j] + alpha * temp2
                kk += j + 1
        else:
            var kk: Int = 0
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[j]
                var temp2: Scalar[dtype] = 0
                y[j] = y[j] + temp1 * ap[kk]
                for i in range(j + 1, n):
                    var aij = ap[kk + i - j]
                    y[i] = y[i] + temp1 * aij
                    temp2 = temp2 + aij * x[i]
                y[j] = y[j] + alpha * temp2
                kk += n - j
        return

    var xbuf = alloc[Scalar[dtype]](n)
    var ybuf = alloc[Scalar[dtype]](n)

    var kx: Int = 0 if incx > 0 else (1 - n) * incx
    var ky: Int = 0 if incy > 0 else (1 - n) * incy

    var ix: Int = kx
    var iy: Int = ky
    for i in range(n):
        xbuf[i] = x[ix]
        ybuf[i] = y[iy]
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
                sum = sum + ap[idx] * xbuf[j]
            else:
                if ii < jj:
                    ii = j
                    jj = i
                var start = jj * n - (jj * (jj - 1)) // 2
                var idx = start + (ii - jj)
                sum = sum + ap[idx] * xbuf[j]
        ybuf[i] = alpha * sum + beta * ybuf[i]

    iy = ky
    for i in range(n):
        y[iy] = ybuf[i]
        iy += incy

    xbuf.free()
    ybuf.free()
