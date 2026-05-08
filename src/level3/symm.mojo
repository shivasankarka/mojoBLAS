# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Matrix-Matrix Operations (`level3.symm`)
=============================================
Provides symmetric matrix-matrix operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize, parallelize
from std.sys.info import simd_width_of


def symm[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    side: String,
    uplo: String,
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[dtype, origin_c],
    ldc: Int,
):
    """
    Performs the matrix-matrix operation C := alpha*A*B + beta*C,
    where A is a symmetric matrix and B and C are m by n matrices.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_b: Indicates whether the pointer b is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b.
        origin_c: Memory origin of the pointer c (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        side: Specifies whether A is on the left ('L') or right ('R') of B.
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        m: The number of rows of the matrices B and C.
        n: The number of columns of the matrices B and C.
        alpha: The scalar multiplier for the matrix product A*B.
        a: A pointer to the first element of the symmetric matrix A.
        lda: The leading dimension of the matrix A.
        b: A pointer to the first element of the matrix B.
        ldb: The leading dimension of the matrix B.
        beta: The scalar multiplier for the matrix C.
        c: A pointer to the first element of the matrix C (input/output).
        ldc: The leading dimension of the matrix C.
    """
    var info: Int = 0
    if side != "L" and side != "l" and side != "R" and side != "r":
        info = 1
    elif uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif lda < max(1, m if (side == "L" or side == "l") else n):
        info = 7
    elif ldb < max(1, m):
        info = 9
    elif ldc < max(1, m):
        info = 12

    if info != 0:
        print("symm: Info", info)
        return

    if m == 0 or n == 0 or (alpha == 0 and beta == 1):
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    comptime PAR_THRESHOLD: Int = 64
    var left_side = side == "L" or side == "l"
    var upper = uplo == "U" or uplo == "u"

    if left_side:

        @parameter
        def symm_left_col(j: Int):
            var cj = c + j * ldc
            var bj = b + j * ldb
            if beta == 0:
                for i in range(m):
                    cj[i] = 0
            elif beta != 1:

                def scale_cj[width: Int](i: Int) {cj, beta}:
                    cj.store[width=width](i, beta * cj.load[width=width](i))

                vectorize[simd_width](m, scale_cj)

            if upper:
                for l in range(m - 1, -1, -1):
                    if bj[l] != 0:
                        var temp1: Scalar[dtype] = alpha * bj[l]
                        var temp2: Scalar[dtype] = 0
                        var al = a + l * lda

                        def fused_upper[
                            width: Int
                        ](i: Int) {cj, mut temp2, al, bj, temp1}:
                            var av = al.load[width=width](i)
                            cj.store[width=width](
                                i, cj.load[width=width](i) + temp1 * av
                            )
                            temp2 += (av * bj.load[width=width](i)).reduce_add()

                        vectorize[simd_width](l, fused_upper)
                        cj[l] = cj[l] + temp1 * a[l + l * lda] + alpha * temp2
            else:
                for l in range(m):
                    if bj[l] != 0:
                        var temp1: Scalar[dtype] = alpha * bj[l]
                        var temp2: Scalar[dtype] = 0
                        var al = a + l * lda

                        def fused_lower[
                            width: Int
                        ](i: Int) {cj, mut temp2, al, bj, temp1, l}:
                            var ii = l + 1 + i
                            var av = al.load[width=width](ii)
                            cj.store[width=width](
                                ii, cj.load[width=width](ii) + temp1 * av
                            )
                            temp2 += (
                                av * bj.load[width=width](ii)
                            ).reduce_add()

                        vectorize[simd_width](m - l - 1, fused_lower)
                        cj[l] = cj[l] + temp1 * a[l + l * lda] + alpha * temp2

        if n >= PAR_THRESHOLD:
            parallelize[symm_left_col](n)
        else:
            for j in range(n):
                symm_left_col(j)
    else:

        @parameter
        def symm_right_col(j: Int):
            var cj = c + j * ldc
            var bj = b + j * ldb
            if beta == 0:
                for i in range(m):
                    cj[i] = 0
            elif beta != 1:

                def scale_cj_r[width: Int](i: Int) {cj, beta}:
                    cj.store[width=width](i, beta * cj.load[width=width](i))

                vectorize[simd_width](m, scale_cj_r)

            var temp_diag: Scalar[dtype] = alpha * a[j + j * lda]

            def axpy_diag[width: Int](i: Int) {cj, bj, temp_diag}:
                cj.store[width=width](
                    i,
                    cj.load[width=width](i)
                    + temp_diag * bj.load[width=width](i),
                )

            vectorize[simd_width](m, axpy_diag)

            if upper:
                for k in range(j):
                    var temp_k: Scalar[dtype] = alpha * a[k + j * lda]
                    var bk = b + k * ldb

                    def axpy_upper_r[width: Int](i: Int) {cj, bk, temp_k}:
                        cj.store[width=width](
                            i,
                            cj.load[width=width](i)
                            + temp_k * bk.load[width=width](i),
                        )

                    vectorize[simd_width](m, axpy_upper_r)
                for k in range(j + 1, n):
                    var temp_k: Scalar[dtype] = alpha * a[j + k * lda]
                    var bk = b + k * ldb

                    def axpy_upper_r2[width: Int](i: Int) {cj, bk, temp_k}:
                        cj.store[width=width](
                            i,
                            cj.load[width=width](i)
                            + temp_k * bk.load[width=width](i),
                        )

                    vectorize[simd_width](m, axpy_upper_r2)
            else:
                for k in range(j):
                    var temp_k: Scalar[dtype] = alpha * a[j + k * lda]
                    var bk = b + k * ldb

                    def axpy_lower_r[width: Int](i: Int) {cj, bk, temp_k}:
                        cj.store[width=width](
                            i,
                            cj.load[width=width](i)
                            + temp_k * bk.load[width=width](i),
                        )

                    vectorize[simd_width](m, axpy_lower_r)
                for k in range(j + 1, n):
                    var temp_k: Scalar[dtype] = alpha * a[k + j * lda]
                    var bk = b + k * ldb

                    def axpy_lower_r2[width: Int](i: Int) {cj, bk, temp_k}:
                        cj.store[width=width](
                            i,
                            cj.load[width=width](i)
                            + temp_k * bk.load[width=width](i),
                        )

                    vectorize[simd_width](m, axpy_lower_r2)

        if n >= PAR_THRESHOLD:
            parallelize[symm_right_col](n)
        else:
            for j in range(n):
                symm_right_col(j)

    return
