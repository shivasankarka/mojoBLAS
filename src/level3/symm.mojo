def symm[
    dtype: DType
](
    side: String,
    uplo: String,
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[Scalar[dtype]],
    lda: Int,
    b: BLASPtr[Scalar[dtype]],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[Scalar[dtype]],
    ldc: Int,
):
    """
    Performs the matrix-matrix operation C := alpha*A*B + beta*C,
    where A is a symmetric matrix and B and C are m by n matrices.

    Parameters:
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

    var left_side = side == "L" or side == "l"
    var upper = uplo == "U" or uplo == "u"

    if left_side:
        for j in range(n):
            if beta == 0:
                for i in range(m):
                    c[i + j * ldc] = 0
            elif beta != 1:
                for i in range(m):
                    c[i + j * ldc] = beta * c[i + j * ldc]

            if upper:
                for l in range(m - 1, -1, -1):
                    if b[l + j * ldb] != 0:
                        var temp1: Scalar[dtype] = alpha * b[l + j * ldb]
                        var temp2: Scalar[dtype] = 0
                        for i in range(l):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp1 * a[i + l * lda]
                            )
                            temp2 = temp2 + a[i + l * lda] * b[i + j * ldb]
                        c[l + j * ldc] = (
                            c[l + j * ldc]
                            + temp1 * a[l + l * lda]
                            + alpha * temp2
                        )
            else:
                for l in range(m):
                    if b[l + j * ldb] != 0:
                        var temp1: Scalar[dtype] = alpha * b[l + j * ldb]
                        var temp2: Scalar[dtype] = 0
                        for i in range(l + 1, m):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp1 * a[i + l * lda]
                            )
                            temp2 = temp2 + a[i + l * lda] * b[i + j * ldb]
                        c[l + j * ldc] = (
                            c[l + j * ldc]
                            + temp1 * a[l + l * lda]
                            + alpha * temp2
                        )
    else:
        for j in range(n):
            if beta == 0:
                for i in range(m):
                    c[i + j * ldc] = 0
            elif beta != 1:
                for i in range(m):
                    c[i + j * ldc] = beta * c[i + j * ldc]

            if upper:
                for l in range(n - 1, -1, -1):
                    if b[l + j * ldb] != 0:
                        var temp1: Scalar[dtype] = alpha * b[l + j * ldb]
                        var temp2: Scalar[dtype] = 0
                        for i in range(l):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp1 * a[i + l * lda]
                            )
                            temp2 = temp2 + a[i + l * lda] * b[i + j * ldb]
                        c[l + j * ldc] = (
                            c[l + j * ldc]
                            + temp1 * a[l + l * lda]
                            + alpha * temp2
                        )
            else:
                for l in range(n):
                    if b[l + j * ldb] != 0:
                        var temp1: Scalar[dtype] = alpha * b[l + j * ldb]
                        var temp2: Scalar[dtype] = 0
                        for i in range(l + 1, n):
                            c[i + j * ldc] = (
                                c[i + j * ldc] + temp1 * a[i + l * lda]
                            )
                            temp2 = temp2 + a[i + l * lda] * b[i + j * ldb]
                        c[l + j * ldc] = (
                            c[l + j * ldc]
                            + temp1 * a[l + l * lda]
                            + alpha * temp2
                        )

    return
