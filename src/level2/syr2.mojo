# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Rank-2 Operations (`level2.syr2`)
=============================================
Provides symmetric rank-2 operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def syr2[
    mut_x: Bool,
    mut_y: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: Origin[mut=mut_y],
    origin_a: MutOrigin,
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
    a: BLASPtr[dtype, origin_a],
    lda: Int,
):
    """
    Performs the symmetric rank 2 operation A := alpha*x*y^T + alpha*y*x^T + A,
    where A is an n by n symmetric matrix.

    Parameters:
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        mut_y: Indicates whether the pointer y is mutable (True) or immutable (False).
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y.
        origin_a: Memory origin of the pointer a (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        alpha: The scalar multiplier for the rank-2 update.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
        y: A pointer to the first element of the vector y.
        incy: The increment for the elements of y.
        a: A pointer to the first element of the matrix A (input/output).
        lda: The leading dimension of the matrix A.
    """
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif n < 0:
        info = 2
    elif incx == 0:
        info = 5
    elif incy == 0:
        info = 7
    elif lda < max(1, n):
        info = 9

    if info != 0:
        print("syr2: Info", info)
        return

    if n == 0 or alpha == 0:
        return

    var kx: Int = 1
    var ky: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx
    if incy < 0:
        ky = 1 - (n - 1) * incy

    var upper = uplo == "U" or uplo == "u"

    comptime simd_width: Int = simd_width_of[dtype]()

    if upper:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0 or y[j] != 0:
                    var temp1: Scalar[dtype] = alpha * y[j]
                    var temp2: Scalar[dtype] = alpha * x[j]
                    var aj = a + j * lda

                    def rank2_upper[
                        width: Int
                    ](i: Int) {aj, x, y, temp1, temp2}:
                        aj.store[width=width](
                            i,
                            aj.load[width=width](i)
                            + x.load[width=width](i) * temp1
                            + y.load[width=width](i) * temp2,
                        )

                    vectorize[simd_width](j + 1, rank2_upper)
        else:
            var jx: Int = kx
            var jy: Int = ky
            for j in range(n):
                if x[jx - 1] != 0 or y[jy - 1] != 0:
                    var temp1: Scalar[dtype] = alpha * y[jy - 1]
                    var temp2: Scalar[dtype] = alpha * x[jx - 1]
                    var ix: Int = kx
                    var iy: Int = ky
                    for i in range(j + 1):
                        a[i + j * lda] = (
                            a[i + j * lda]
                            + x[ix - 1] * temp1
                            + y[iy - 1] * temp2
                        )
                        ix += incx
                        iy += incy
                jx += incx
                jy += incy
    else:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0 or y[j] != 0:
                    var temp1: Scalar[dtype] = alpha * y[j]
                    var temp2: Scalar[dtype] = alpha * x[j]
                    var aj = a + j * lda

                    def rank2_lower[
                        width: Int
                    ](i: Int) {aj, x, y, temp1, temp2, j}:
                        var ii = j + i
                        aj.store[width=width](
                            ii,
                            aj.load[width=width](ii)
                            + x.load[width=width](ii) * temp1
                            + y.load[width=width](ii) * temp2,
                        )

                    vectorize[simd_width](n - j, rank2_lower)
        else:
            var jx: Int = kx
            var jy: Int = ky
            for j in range(n):
                if x[jx - 1] != 0 or y[jy - 1] != 0:
                    var temp1: Scalar[dtype] = alpha * y[jy - 1]
                    var temp2: Scalar[dtype] = alpha * x[jx - 1]
                    var ix: Int = jx
                    var iy: Int = jy
                    for i in range(j, n):
                        a[i + j * lda] = (
                            a[i + j * lda]
                            + x[ix - 1] * temp1
                            + y[iy - 1] * temp2
                        )
                        ix += incx
                        iy += incy
                jx += incx
                jy += incy

    return
