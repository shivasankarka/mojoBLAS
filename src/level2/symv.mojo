# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Matrix-Vector Operations (`level2.symv`)
=============================================

Provides symmetric matrix-vector operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def symv[
    mut_a: Bool,
    mut_x: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    beta: Scalar[dtype],
    y: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Performs the matrix-vector operation y := alpha*A*x + beta*y,
    where A is an n by n symmetric matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        alpha: The scalar multiplier for the matrix-vector product.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
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
    elif lda < max(1, n):
        info = 5
    elif incx == 0:
        info = 7
    elif incy == 0:
        info = 10

    if info != 0:
        print("symv: Info", info)
        return

    if n == 0 or (alpha == 0 and beta == 1):
        return

    var leny: Int = n
    var ky: Int = 1
    if incy < 0:
        ky = 1 - (leny - 1) * incy

    comptime simd_width: Int = simd_width_of[dtype]()

    if beta != 1:
        if incy == 1:
            if beta == 0:
                for i in range(leny):
                    y[i] = 0
            else:

                def scale_y[width: Int](i: Int) {y, beta}:
                    y.store[width=width](i, beta * y.load[width=width](i))

                vectorize[simd_width](leny, scale_y)
        else:
            var iy: Int = ky
            if beta == 0:
                for _ in range(leny):
                    y[iy - 1] = 0
                    iy += incy
            else:
                for _ in range(leny):
                    y[iy - 1] = beta * y[iy - 1]
                    iy += incy

    if alpha == 0:
        return

    var kx: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    var upper = uplo == "U" or uplo == "u"

    if upper:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0:
                    var temp1: Scalar[dtype] = alpha * x[j]
                    var temp2: Scalar[dtype] = 0
                    var aj = a + j * lda

                    # Single vectorized pass: update y[0..j) and accumulate dot(a[:,j], x[0..j))
                    def fused_upper[
                        width: Int
                    ](i: Int) {y, mut temp2, aj, x, temp1}:
                        var av = aj.load[width=width](i)
                        y.store[width=width](
                            i, y.load[width=width](i) + temp1 * av
                        )
                        temp2 += (av * x.load[width=width](i)).reduce_add()

                    vectorize[simd_width](j, fused_upper)
                    y[j] = y[j] + temp1 * a[j + j * lda] + alpha * temp2
        elif incx == 1:
            for j in range(n):
                if x[j] != 0:
                    var temp1: Scalar[dtype] = alpha * x[j]
                    var temp2: Scalar[dtype] = 0
                    var aj = a + j * lda
                    var iy: Int = ky

                    def dot_upper_sx[width: Int](i: Int) {mut temp2, aj, x}:
                        temp2 += (
                            aj.load[width=width](i) * x.load[width=width](i)
                        ).reduce_add()

                    vectorize[simd_width](j, dot_upper_sx)
                    for i in range(j):
                        y[iy - 1] = y[iy - 1] + temp1 * a[i + j * lda]
                        iy += incy
                    y[j] = y[j] + temp1 * a[j + j * lda] + alpha * temp2
        else:
            var jx: Int = kx
            for j in range(n):
                if x[jx - 1] != 0:
                    var temp1: Scalar[dtype] = alpha * x[jx - 1]
                    var temp2: Scalar[dtype] = 0
                    var ix: Int = kx
                    for i in range(j):
                        y[i] = y[i] + temp1 * a[i + j * lda]
                        temp2 = temp2 + a[i + j * lda] * x[ix - 1]
                        ix += incx
                    y[j] = y[j] + temp1 * a[j + j * lda] + alpha * temp2
                jx += incx
    else:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0:
                    var temp1: Scalar[dtype] = alpha * x[j]
                    var temp2: Scalar[dtype] = 0
                    var aj = a + j * lda

                    def fused_lower[
                        width: Int
                    ](i: Int) {y, mut temp2, aj, x, temp1, j}:
                        var ii = j + 1 + i
                        var av = aj.load[width=width](ii)
                        y.store[width=width](
                            ii, y.load[width=width](ii) + temp1 * av
                        )
                        temp2 += (av * x.load[width=width](ii)).reduce_add()

                    vectorize[simd_width](n - j - 1, fused_lower)
                    y[j] = y[j] + temp1 * a[j + j * lda] + alpha * temp2
        elif incx == 1:
            for j in range(n):
                if x[j] != 0:
                    var temp1: Scalar[dtype] = alpha * x[j]
                    var temp2: Scalar[dtype] = 0
                    var aj = a + j * lda
                    var iy: Int = ky + (j + 1) * incy

                    def dot_lower_sx[width: Int](i: Int) {mut temp2, aj, x, j}:
                        var ii = j + 1 + i
                        temp2 += (
                            aj.load[width=width](ii) * x.load[width=width](ii)
                        ).reduce_add()

                    vectorize[simd_width](n - j - 1, dot_lower_sx)
                    for i in range(j + 1, n):
                        y[iy - 1] = y[iy - 1] + temp1 * a[i + j * lda]
                        iy += incy
                    y[j] = y[j] + temp1 * a[j + j * lda] + alpha * temp2
        else:
            var jx: Int = kx
            for j in range(n):
                if x[jx - 1] != 0:
                    var temp1: Scalar[dtype] = alpha * x[jx - 1]
                    var temp2: Scalar[dtype] = 0
                    var ix: Int = kx
                    for i in range(j + 1, n):
                        y[i] = y[i] + temp1 * a[i + j * lda]
                        temp2 = temp2 + a[i + j * lda] * x[ix - 1]
                        ix += incx
                    y[j] = y[j] + temp1 * a[j + j * lda] + alpha * temp2
                jx += incx

    return
