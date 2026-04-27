# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Triangular Band Matrix-Vector Operations (`level2.tbmv`)
=============================================

Provides triangular band matrix-vector operations as defined in the BLAS library standard.
"""


def tbmv[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    trans: String,
    diag: String,
    n: Int,
    k: Int,
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
):
    """
    Performs the matrix-vector operation x := A*x or x := A^T*x,
    where A is an n by n triangular band matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_x: Memory origin of the pointer x (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for x := A*x, 'T' or 'C' for x := A^T*x.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        k: The number of super-diagonals (if upper) or sub-diagonals (if lower).
        a: A pointer to the first element of the matrix A (stored in band format).
        lda: The leading dimension of the matrix A.
        x: A pointer to the first element of the vector x (input/output).
        incx: The increment for the elements of x.
    """
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif (
        trans != "N"
        and trans != "n"
        and trans != "T"
        and trans != "t"
        and trans != "C"
        and trans != "c"
    ):
        info = 2
    elif diag != "U" and diag != "u" and diag != "N" and diag != "n":
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < k + 1:
        info = 7
    elif incx == 0:
        info = 9

    if info != 0:
        print("tbmv: Info", info)
        return

    if n == 0:
        return

    var no_unit = diag == "N" or diag == "n"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans == "N" or trans == "n"

    var kx: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    if no_trans:
        if upper:
            if incx == 1:
                for j in range(n):
                    if x[j] != 0:
                        var temp: Scalar[dtype] = x[j]
                        var i_start: Int = max(0, j - k)
                        for i in range(i_start, j):
                            x[i] = x[i] + temp * a[k - j + i + j * lda]
                        if no_unit:
                            x[j] = x[j] * a[k + j * lda]
            else:
                var jx: Int = kx
                for j in range(n):
                    if x[jx - 1] != 0:
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = max(kx, kx + (j - k) * incx)
                        for i in range(max(0, j - k), j):
                            x[ix - 1] = (
                                x[ix - 1] + temp * a[k - j + i + j * lda]
                            )
                            ix += incx
                        if no_unit:
                            x[jx - 1] = x[jx - 1] * a[k + j * lda]
                    jx += incx
        else:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    if x[j] != 0:
                        var temp: Scalar[dtype] = x[j]
                        var i_end: Int = min(n, j + k + 1)
                        for i in range(j + 1, i_end):
                            x[i] = x[i] + temp * a[k - j + i + j * lda]
                        if no_unit:
                            x[j] = x[j] * a[k + j * lda]
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    if x[jx - 1] != 0:
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = kx_plus
                        var i_end: Int = min(n, j + k + 1)
                        for i in range(j + 1, i_end):
                            x[ix - 1] = (
                                x[ix - 1] + temp * a[k - j + i + j * lda]
                            )
                            ix += incx
                        if no_unit:
                            x[jx - 1] = x[jx - 1] * a[k + j * lda]
                    jx -= incx
    else:
        if upper:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    var temp: Scalar[dtype] = x[j]
                    if no_unit:
                        temp = temp * a[k + j * lda]
                    var i_start: Int = max(0, j - k)
                    for i in range(j - 1, i_start - 1, -1):
                        temp = temp + a[k - j + i + j * lda] * x[i]
                    x[j] = temp
            else:
                var jx: Int = kx + (n - 1) * incx
                for j in range(n - 1, -1, -1):
                    var ix: Int = jx
                    var temp: Scalar[dtype] = x[jx - 1]
                    if no_unit:
                        temp = temp * a[k + j * lda]
                    var i_start: Int = max(0, j - k)
                    for i in range(j - 1, i_start - 1, -1):
                        ix -= incx
                        temp = temp + a[k - j + i + j * lda] * x[ix - 1]
                    x[jx - 1] = temp
                    jx -= incx
        else:
            if incx == 1:
                for j in range(n):
                    var temp: Scalar[dtype] = x[j]
                    if no_unit:
                        temp = temp * a[k + j * lda]
                    var i_end: Int = min(n, j + k + 1)
                    for i in range(j + 1, i_end):
                        temp = temp + a[k - j + i + j * lda] * x[i]
                    x[j] = temp
            else:
                var jx: Int = kx
                for j in range(n):
                    var ix: Int = jx
                    var temp: Scalar[dtype] = x[jx - 1]
                    if no_unit:
                        temp = temp * a[k + j * lda]
                    var i_end: Int = min(n, j + k + 1)
                    for i in range(j + 1, i_end):
                        ix += incx
                        temp = temp + a[k - j + i + j * lda] * x[ix - 1]
                    x[jx - 1] = temp
                    jx += incx

    return
