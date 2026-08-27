# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Rank-1 Operations (`level2.syr`)
=============================================
Provides symmetric rank-1 operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of
from mojoblas.type_aliases import BLASPtr


def syr[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    origin_a: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    a: BLASPtr[dtype, origin_a],
    lda: Int,
):
    """
    Performs the symmetric rank 1 operation A := alpha*x*x^T + A,
    where A is an n by n symmetric matrix.

    Parameters:
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        origin_x: Memory origin of the pointer x.
        origin_a: Memory origin of the pointer a (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        alpha: The scalar multiplier for the rank-1 update.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
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
    elif lda < max(1, n):
        info = 7

    if info != 0:
        print("syr: Info", info)
        return

    if n == 0 or alpha == 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    var kx: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    var upper = uplo == "U" or uplo == "u"

    if upper:
        if incx == 1:
            for j in range(n):
                if x[unsafe_offset=j] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=j]
                    var aj = a.unsafe_offset(j * lda)

                    def axpy_upper[width: Int](i: Int) {aj, x, temp}:
                        aj.unsafe_store[width=width](
                            i,
                            aj.unsafe_load[width=width](i)
                            + x.unsafe_load[width=width](i) * temp,
                        )

                    vectorize[simd_width](j + 1, axpy_upper)
        else:
            var jx: Int = kx
            for j in range(n):
                if x[unsafe_offset=jx - 1] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=jx - 1]
                    var ix: Int = kx
                    for i in range(j + 1):
                        a[unsafe_offset=i + j * lda] = a[unsafe_offset=i + j * lda] + x[unsafe_offset=ix - 1] * temp
                        ix += incx
                jx += incx
    else:
        if incx == 1:
            for j in range(n):
                if x[unsafe_offset=j] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=j]
                    var aj = a.unsafe_offset(j * lda)

                    def axpy_lower[width: Int](i: Int) {aj, x, temp, j}:
                        var ii = j + i
                        aj.unsafe_store[width=width](
                            ii,
                            aj.unsafe_load[width=width](ii)
                            + x.unsafe_load[width=width](ii) * temp,
                        )

                    vectorize[simd_width](n - j, axpy_lower)
        else:
            var jx: Int = kx
            for j in range(n):
                if x[unsafe_offset=jx - 1] != 0:
                    var temp: Scalar[dtype] = alpha * x[unsafe_offset=jx - 1]
                    var ix: Int = jx
                    for i in range(j, n):
                        a[unsafe_offset=i + j * lda] = a[unsafe_offset=i + j * lda] + x[unsafe_offset=ix - 1] * temp
                        ix += incx
                jx += incx

    return
