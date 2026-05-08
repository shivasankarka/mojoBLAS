# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
General Matrix-Matrix Operations (`level3.gemm`)
=============================================
Provides general matrix-matrix operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize, parallelize
from std.sys.info import simd_width_of


def gemm[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    trans_a: String,
    trans_b: String,
    m: Int,
    n: Int,
    k: Int,
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
    where A, B, and C are matrices with appropriate dimensions.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_b: Indicates whether the pointer b is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b.
        origin_c: Memory origin of the pointer c (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        trans_a: Specifies whether A is transposed ('N' for no, 'T' or 'C' for yes).
        trans_b: Specifies whether B is transposed ('N' for no, 'T' or 'C' for yes).
        m: The number of rows of the matrix C and of the matrix A if not transposed.
        n: The number of columns of the matrix C and of the matrix B if not transposed.
        k: The number of columns of A if not transposed, or rows if transposed.
        alpha: The scalar multiplier for the matrix product A*B.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
        b: A pointer to the first element of the matrix B.
        ldb: The leading dimension of the matrix B.
        beta: The scalar multiplier for the matrix C.
        c: A pointer to the first element of the matrix C (input/output).
        ldc: The leading dimension of the matrix C.
    """
    var info: Int = 0
    if (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 1
    elif (
        trans_b != "N"
        and trans_b != "n"
        and trans_b != "T"
        and trans_b != "t"
        and trans_b != "C"
        and trans_b != "c"
    ):
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < max(1, m if (trans_a == "N" or trans_a == "n") else k):
        info = 8
    elif ldb < max(1, k if (trans_b == "N" or trans_b == "n") else n):
        info = 10
    elif ldc < max(1, m):
        info = 13

    if info != 0:
        print("gemm: Info", info)
        return

    if m == 0 or n == 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()

    if alpha == 0 or k == 0:
        if beta == 0:
            for j in range(n):
                var cj = c + j * ldc

                def zero_col[width: Int](i: Int) {cj}:
                    cj.store[width=width](i, SIMD[dtype, width](0))

                vectorize[simd_width](m, zero_col)
        elif beta != 1:
            for j in range(n):
                var cj = c + j * ldc

                def scale_col[width: Int](i: Int) {cj, beta}:
                    cj.store[width=width](i, beta * cj.load[width=width](i))

                vectorize[simd_width](m, scale_col)
        return

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"

    # Threshold: only parallelize outer-j if n is large enough to amortize thread cost
    comptime PAR_THRESHOLD: Int = 64

    if no_trans_a:
        if no_trans_b:
            # C += alpha * A * B  (both col-major, each column j independent)
            @parameter
            def gemm_nn_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_nn[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_nn)
                elif beta != 1:

                    def scale_nn[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_nn)
                for l in range(k):
                    if b[l + j * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[l + j * ldb]
                        var al = a + l * lda

                        def axpy_nn[width: Int](i: Int) {cj, al, temp}:
                            cj.store[width=width](
                                i,
                                cj.load[width=width](i)
                                + temp * al.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_nn)

            if n >= PAR_THRESHOLD:
                parallelize[gemm_nn_col](n)
            else:
                for j in range(n):
                    gemm_nn_col(j)
        else:
            # C += alpha * A * B^T
            @parameter
            def gemm_nt_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_nt[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_nt)
                elif beta != 1:

                    def scale_nt[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_nt)
                for l in range(k):
                    if b[j + l * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[j + l * ldb]
                        var al = a + l * lda

                        def axpy_nt[width: Int](i: Int) {cj, al, temp}:
                            cj.store[width=width](
                                i,
                                cj.load[width=width](i)
                                + temp * al.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_nt)

            if n >= PAR_THRESHOLD:
                parallelize[gemm_nt_col](n)
            else:
                for j in range(n):
                    gemm_nt_col(j)
    else:
        if no_trans_b:
            # C += alpha * A^T * B  — A rows non-contiguous (stride lda), scalar inner
            @parameter
            def gemm_tn_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_tn[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_tn)
                elif beta != 1:

                    def scale_tn[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_tn)
                for l in range(k):
                    if b[l + j * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[l + j * ldb]
                        var al = a + l
                        for i in range(m):
                            cj[i] = cj[i] + temp * al[i * lda]

            if n >= PAR_THRESHOLD:
                parallelize[gemm_tn_col](n)
            else:
                for j in range(n):
                    gemm_tn_col(j)
        else:
            # C += alpha * A^T * B^T
            @parameter
            def gemm_tt_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_tt[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_tt)
                elif beta != 1:

                    def scale_tt[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_tt)
                for l in range(k):
                    if b[j + l * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[j + l * ldb]
                        var al = a + l
                        for i in range(m):
                            cj[i] = cj[i] + temp * al[i * lda]

            if n >= PAR_THRESHOLD:
                parallelize[gemm_tt_col](n)
            else:
                for j in range(n):
                    gemm_tt_col(j)

    return
