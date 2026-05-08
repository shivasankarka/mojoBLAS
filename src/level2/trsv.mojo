# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Triangular Solve Operations (`level2.trsv`)
=============================================
Provides triangular solve operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def trsv[
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
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
):
    """
    Solves a system of linear equations A*x = b or A^T*x = b,
    where A is an n by n triangular matrix.

    Optimized with SIMD vectorization and parallelization.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_x: Memory origin of the pointer x (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for A*x = b, 'T' or 'C' for A^T*x = b.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
        x: On entry, the right-hand side vector b. On exit, the solution vector x.
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
    elif lda < max(1, n):
        info = 6
    elif incx == 0:
        info = 8

    if info != 0:
        print("trsv: Info", info)
        return

    if n == 0:
        return

    var no_unit = diag == "N" or diag == "n"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans == "N" or trans == "n"

    var kx: Int = 1
    # var ky: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    comptime simd_width: Int = simd_width_of[dtype]()

    if no_trans:
        if upper:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    if x[j] != 0:
                        if no_unit:
                            x[j] = x[j] / a[j + j * lda]
                        var temp: Scalar[dtype] = x[j]
                        var aj = a + j * lda

                        def axpy_upper[width: Int](i: Int) {x, aj, temp}:
                            x.store[width=width](
                                i,
                                x.load[width=width](i)
                                - temp * aj.load[width=width](i),
                            )

                        vectorize[simd_width](j, axpy_upper)
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    jx -= incx
                    if x[jx - 1] != 0:
                        if no_unit:
                            x[jx - 1] = x[jx - 1] / a[j + j * lda]
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = jx
                        for i in range(j - 1, -1, -1):
                            ix -= incx
                            x[ix - 1] = x[ix - 1] - temp * a[i + j * lda]
        else:
            if incx == 1:
                for j in range(n):
                    if x[j] != 0:
                        if no_unit:
                            x[j] = x[j] / a[j + j * lda]
                        var temp: Scalar[dtype] = x[j]
                        var aj = a + j * lda

                        def axpy_lower[width: Int](i: Int) {x, aj, temp, j}:
                            var ii = j + 1 + i
                            x.store[width=width](
                                ii,
                                x.load[width=width](ii)
                                - temp * aj.load[width=width](ii),
                            )

                        vectorize[simd_width](n - j - 1, axpy_lower)
            else:
                var jx: Int = kx
                for j in range(n):
                    if x[jx - 1] != 0:
                        if no_unit:
                            x[jx - 1] = x[jx - 1] / a[j + j * lda]
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = jx
                        for i in range(j + 1, n):
                            ix += incx
                            x[ix - 1] = x[ix - 1] - temp * a[i + j * lda]
                    jx += incx
    else:
        # Trans paths: sequential dot reduction — scalar (each j depends on prior)
        if upper:
            if incx == 1:
                for j in range(n):
                    var temp: Scalar[dtype] = x[j]
                    var aj = a + j * lda

                    def dot_upper[width: Int](i: Int) {mut temp, aj, x}:
                        temp -= (
                            aj.load[width=width](i) * x.load[width=width](i)
                        ).reduce_add()

                    vectorize[simd_width](j, dot_upper)
                    if no_unit:
                        temp = temp / a[j + j * lda]
                    x[j] = temp
            else:
                var jx: Int = kx
                for j in range(n):
                    var ix: Int = kx
                    var temp: Scalar[dtype] = x[jx - 1]
                    for i in range(j):
                        temp = temp - a[i + j * lda] * x[ix - 1]
                        ix += incx
                    if no_unit:
                        temp = temp / a[j + j * lda]
                    x[jx - 1] = temp
                    jx += incx
        else:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    var temp: Scalar[dtype] = x[j]
                    var aj = a + j * lda

                    def dot_lower[width: Int](i: Int) {mut temp, aj, x, j}:
                        var ii = j + 1 + i
                        temp -= (
                            aj.load[width=width](ii) * x.load[width=width](ii)
                        ).reduce_add()

                    vectorize[simd_width](n - j - 1, dot_lower)
                    if no_unit:
                        temp = temp / a[j + j * lda]
                    x[j] = temp
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    var ix: Int = kx_plus
                    var temp: Scalar[dtype] = x[jx - 1]
                    for i in range(n - 1, j, -1):
                        temp = temp - a[i + j * lda] * x[ix - 1]
                        ix -= incx
                    if no_unit:
                        temp = temp / a[j + j * lda]
                    x[jx - 1] = temp
                    jx -= incx

    return
