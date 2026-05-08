# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Triangular Solve Operations (`level3.trsm`)
=============================================

Provides triangular solve operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def trsm[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: MutOrigin,
    //,
    dtype: DType,
](
    side: String,
    uplo: String,
    trans_a: String,
    diag: String,
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
):
    """
    Solves a system of matrix equations A*X = alpha*B or X*A = alpha*B,
    where A is a triangular matrix and B is an m by n matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        side: Specifies whether A is on the left ('L') or right ('R') of B.
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans_a: Specifies whether A is transposed ('N' for no, 'T' or 'C' for yes).
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        m: The number of rows of the matrix B.
        n: The number of columns of the matrix B.
        alpha: The scalar multiplier.
        a: A pointer to the first element of the triangular matrix A.
        lda: The leading dimension of the matrix A.
        b: On entry, the right-hand side matrix B. On exit, the solution matrix X.
        ldb: The leading dimension of the matrix B.
    """
    var info: Int = 0
    if side != "L" and side != "l" and side != "R" and side != "r":
        info = 1
    elif uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 2
    elif (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 3
    elif diag != "U" and diag != "u" and diag != "N" and diag != "n":
        info = 4
    elif m < 0:
        info = 5
    elif n < 0:
        info = 6
    elif lda < max(1, m if (side == "L" or side == "l") else n):
        info = 9
    elif ldb < max(1, m):
        info = 11

    if info != 0:
        print("trsm: Info", info)
        return

    if m == 0 or n == 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()

    if alpha != 1:
        for j in range(n):
            var bj = b + j * ldb

            def scale_init[width: Int](i: Int) {bj, alpha}:
                bj.store[width=width](i, alpha * bj.load[width=width](i))

            vectorize[simd_width](m, scale_init)

    var left_side = side == "L" or side == "l"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans_a == "N" or trans_a == "n"
    var no_unit = diag == "N" or diag == "n"

    if left_side:
        if no_trans:
            if upper:
                for j in range(n):
                    var bj = b + j * ldb
                    for l in range(m - 1, -1, -1):
                        if bj[l] != 0:
                            if no_unit:
                                bj[l] = bj[l] / a[l + l * lda]
                            var pivot: Scalar[dtype] = bj[l]
                            var al = a + l * lda

                            def axpy_lu[width: Int](i: Int) {bj, al, pivot}:
                                bj.store[width=width](
                                    i,
                                    bj.load[width=width](i)
                                    - pivot * al.load[width=width](i),
                                )

                            vectorize[simd_width](l, axpy_lu)
            else:
                for j in range(n):
                    var bj = b + j * ldb
                    for l in range(m):
                        if bj[l] != 0:
                            if no_unit:
                                bj[l] = bj[l] / a[l + l * lda]
                            var pivot: Scalar[dtype] = bj[l]
                            var al = a + l * lda

                            def axpy_ll[width: Int](i: Int) {bj, al, pivot, l}:
                                var ii = l + 1 + i
                                bj.store[width=width](
                                    ii,
                                    bj.load[width=width](ii)
                                    - pivot * al.load[width=width](ii),
                                )

                            vectorize[simd_width](m - l - 1, axpy_ll)
        else:
            # Trans left: U^T*X=B forward sub, L^T*X=B backward sub
            if upper:
                # U^T[i,l] = U[l,i] = a[l + i*lda]; forward i=0..m-1, deps on l<i
                for j in range(n):
                    var bj = b + j * ldb
                    for i in range(m):
                        for l in range(i):
                            bj[i] = bj[i] - a[l + i * lda] * bj[l]
                        if no_unit:
                            bj[i] = bj[i] / a[i + i * lda]
            else:
                # L^T[i,l] = L[l,i] = a[l + i*lda]; backward i=m-1..0, deps on l>i
                for j in range(n):
                    var bj = b + j * ldb
                    for i in range(m - 1, -1, -1):
                        for l in range(i + 1, m):
                            bj[i] = bj[i] - a[l + i * lda] * bj[l]
                        if no_unit:
                            bj[i] = bj[i] / a[i + i * lda]
    else:
        if no_trans:
            if upper:
                # X*U=B: x_j = (b_j - sum_{l<j} X[:,l]*U[l,j]) / U[j,j], forward j
                # U[l,j] = a[l + j*lda]
                for j in range(n):
                    var bj = b + j * ldb
                    for l in range(j):
                        var bl = b + l * ldb
                        var alj: Scalar[dtype] = a[l + j * lda]

                        def axpy_ru[width: Int](i: Int) {bj, bl, alj}:
                            bj.store[width=width](
                                i,
                                bj.load[width=width](i)
                                - alj * bl.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_ru)
                    if no_unit:
                        var inv_diag: Scalar[dtype] = 1.0 / a[j + j * lda]

                        def scale_u[width: Int](i: Int) {bj, inv_diag}:
                            bj.store[width=width](
                                i, bj.load[width=width](i) * inv_diag
                            )

                        vectorize[simd_width](m, scale_u)
            else:
                # X*L=B: x_j = (b_j - sum_{l>j} X[:,l]*L[l,j]) / L[j,j], backward j
                # L[l,j] = a[l + j*lda]
                for j in range(n - 1, -1, -1):
                    var bj = b + j * ldb
                    for l in range(j + 1, n):
                        var bl = b + l * ldb
                        var alj: Scalar[dtype] = a[l + j * lda]

                        def axpy_rl[width: Int](i: Int) {bj, bl, alj}:
                            bj.store[width=width](
                                i,
                                bj.load[width=width](i)
                                - alj * bl.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_rl)
                    if no_unit:
                        var inv_diag: Scalar[dtype] = 1.0 / a[j + j * lda]

                        def scale_l[width: Int](i: Int) {bj, inv_diag}:
                            bj.store[width=width](
                                i, bj.load[width=width](i) * inv_diag
                            )

                        vectorize[simd_width](m, scale_l)
        else:
            if upper:
                # X*U^T=B: x_j depends on l<j (left-to-right)
                for j in range(n):
                    var bj = b + j * ldb
                    for l in range(j):
                        var bl = b + l * ldb
                        var alj: Scalar[dtype] = a[l + j * lda]

                        def axpy_rtu[width: Int](i: Int) {bj, bl, alj}:
                            bj.store[width=width](
                                i,
                                bj.load[width=width](i)
                                - alj * bl.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_rtu)
                    if no_unit:
                        var inv_diag: Scalar[dtype] = 1.0 / a[j + j * lda]

                        def scale_rtu[width: Int](i: Int) {bj, inv_diag}:
                            bj.store[width=width](
                                i, bj.load[width=width](i) * inv_diag
                            )

                        vectorize[simd_width](m, scale_rtu)
            else:
                # X*L^T=B: x_j depends on l>j (right-to-left)
                for j in range(n - 1, -1, -1):
                    var bj = b + j * ldb
                    for l in range(j + 1, n):
                        var bl = b + l * ldb
                        var alj: Scalar[dtype] = a[l + j * lda]

                        def axpy_rtl[width: Int](i: Int) {bj, bl, alj}:
                            bj.store[width=width](
                                i,
                                bj.load[width=width](i)
                                - alj * bl.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_rtl)
                    if no_unit:
                        var inv_diag: Scalar[dtype] = 1.0 / a[j + j * lda]

                        def scale_rtl[width: Int](i: Int) {bj, inv_diag}:
                            bj.store[width=width](
                                i, bj.load[width=width](i) * inv_diag
                            )

                        vectorize[simd_width](m, scale_rtl)

    return
