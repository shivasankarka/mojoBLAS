# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Symmetric Rank-2k Operations (`level3.syr2k`)
=============================================

Provides symmetric rank-2k operations as defined in the BLAS library standard.
"""

def syr2k[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    trans: String,
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
    Performs the symmetric rank 2k operation C := alpha*A*B^T + alpha*B*A^T + beta*C
    or C := alpha*A^T*B + alpha*B^T*A + beta*C.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_b: Indicates whether the pointer b is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b.
        origin_c: Memory origin of the pointer c (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether C is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for C := alpha*A*B^T + alpha*B*A^T + beta*C, 'T' or 'C' for C := alpha*A^T*B + alpha*B^T*A + beta*C.
        n: The order of the matrix C.
        k: The number of columns of A and B if not transposed.
        alpha: The scalar multiplier for the matrix product.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
        b: A pointer to the first element of the matrix B.
        ldb: The leading dimension of the matrix B.
        beta: The scalar multiplier for the matrix C.
        c: A pointer to the first element of the matrix C (input/output).
        ldc: The leading dimension of the matrix C.
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
    elif n < 0:
        info = 3
    elif k < 0:
        info = 4
    elif lda < max(1, n if (trans == "N" or trans == "n") else k):
        info = 7
    elif ldb < max(1, n if (trans == "N" or trans == "n") else k):
        info = 9
    elif ldc < max(1, n):
        info = 12

    if info != 0:
        print("syr2k: Info", info)
        return

    if n == 0 or (alpha == 0 and beta == 1):
        return

    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans == "N" or trans == "n"

    if beta == 0:
        if upper:
            for j in range(n):
                for i in range(j + 1):
                    c[i + j * ldc] = 0
        else:
            for j in range(n):
                for i in range(j, n):
                    c[i + j * ldc] = 0
    elif beta != 1:
        if upper:
            for j in range(n):
                for i in range(j + 1):
                    c[i + j * ldc] = beta * c[i + j * ldc]
        else:
            for j in range(n):
                for i in range(j, n):
                    c[i + j * ldc] = beta * c[i + j * ldc]

    if alpha == 0:
        return

    if no_trans:
        if upper:
            for j in range(n):
                for l in range(k):
                    if a[j + l * lda] != 0 or b[j + l * ldb] != 0:
                        var temp1: Scalar[dtype] = alpha * a[j + l * lda]
                        var temp2: Scalar[dtype] = alpha * b[j + l * ldb]
                        for i in range(j + 1):
                            c[i + j * ldc] = (
                                c[i + j * ldc]
                                + temp1 * b[i + l * ldb]
                                + temp2 * a[i + l * lda]
                            )
        else:
            for j in range(n):
                for l in range(k):
                    if a[j + l * lda] != 0 or b[j + l * ldb] != 0:
                        var temp1: Scalar[dtype] = alpha * a[j + l * lda]
                        var temp2: Scalar[dtype] = alpha * b[j + l * ldb]
                        for i in range(j, n):
                            c[i + j * ldc] = (
                                c[i + j * ldc]
                                + temp1 * b[i + l * ldb]
                                + temp2 * a[i + l * lda]
                            )
    else:
        if upper:
            for j in range(n):
                for i in range(j + 1):
                    var temp1: Scalar[dtype] = 0
                    var temp2: Scalar[dtype] = 0
                    for l in range(k):
                        temp1 = temp1 + a[l + i * lda] * b[l + j * ldb]
                        temp2 = temp2 + b[l + i * ldb] * a[l + j * lda]
                    c[i + j * ldc] = c[i + j * ldc] + alpha * (temp1 + temp2)
        else:
            for j in range(n):
                for i in range(j, n):
                    var temp1: Scalar[dtype] = 0
                    var temp2: Scalar[dtype] = 0
                    for l in range(k):
                        temp1 = temp1 + a[l + i * lda] * b[l + j * ldb]
                        temp2 = temp2 + b[l + i * ldb] * a[l + j * lda]
                    c[i + j * ldc] = c[i + j * ldc] + alpha * (temp1 + temp2)

    return
