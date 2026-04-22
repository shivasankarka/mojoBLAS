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

    if alpha == 0 or k == 0:
        if beta == 0:
            for j in range(n):
                for i in range(m):
                    c[i + j * ldc] = 0
        elif beta != 1:
            for j in range(n):
                for i in range(m):
                    c[i + j * ldc] = beta * c[i + j * ldc]
        return

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"

    if no_trans_a:
        if no_trans_b:
            for j in range(n):
                if beta == 0:
                    for i in range(m):
                        c[i + j * ldc] = 0
                elif beta != 1:
                    for i in range(m):
                        c[i + j * ldc] = beta * c[i + j * ldc]
                for l in range(k):
                    if b[l + j * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[l + j * ldb]
                        for i in range(m):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp * a[i + l * lda]
                            )
        else:
            for j in range(n):
                if beta == 0:
                    for i in range(m):
                        c[i + j * ldc] = 0
                elif beta != 1:
                    for i in range(m):
                        c[i + j * ldc] = beta * c[i + j * ldc]
                for l in range(k):
                    if b[j + l * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[j + l * ldb]
                        for i in range(m):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp * a[i + l * lda]
                            )
    else:
        if no_trans_b:
            for j in range(n):
                if beta == 0:
                    for i in range(m):
                        c[i + j * ldc] = 0
                elif beta != 1:
                    for i in range(m):
                        c[i + j * ldc] = beta * c[i + j * ldc]
                for l in range(k):
                    if b[l + j * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[l + j * ldb]
                        for i in range(m):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp * a[l + i * lda]
                            )
        else:
            for j in range(n):
                if beta == 0:
                    for i in range(m):
                        c[i + j * ldc] = 0
                elif beta != 1:
                    for i in range(m):
                        c[i + j * ldc] = beta * c[i + j * ldc]
                for l in range(k):
                    if b[j + l * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[j + l * ldb]
                        for i in range(m):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp * a[l + i * lda]
                            )

    return
