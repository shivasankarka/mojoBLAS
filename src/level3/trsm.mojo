def trsm[
    dtype: DType
](
    side: String,
    uplo: String,
    trans_a: String,
    diag: String,
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[Scalar[dtype]],
    lda: Int,
    b: BLASPtr[Scalar[dtype]],
    ldb: Int,
):
    """
    Solves a system of matrix equations A*X = alpha*B or X*A = alpha*B,
    where A is a triangular matrix and B is an m by n matrix.

    Parameters:
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

    if alpha != 1:
        for j in range(n):
            for i in range(m):
                b[i + j * ldb] = alpha * b[i + j * ldb]

    var left_side = side == "L" or side == "l"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans_a == "N" or trans_a == "n"
    var no_unit = diag == "N" or diag == "n"

    if left_side:
        if no_trans:
            if upper:
                for j in range(n):
                    for l in range(m - 1, -1, -1):
                        if b[l + j * ldb] != 0:
                            if no_unit:
                                b[l + j * ldb] = b[l + j * ldb] / a[l + l * lda]
                            for i in range(l):
                                b[i + j * ldb] = (
                                    b[i + j * ldb]
                                    - b[l + j * ldb] * a[i + l * lda]
                                )
            else:
                for j in range(n):
                    for l in range(m):
                        if b[l + j * ldb] != 0:
                            if no_unit:
                                b[l + j * ldb] = b[l + j * ldb] / a[l + l * lda]
                            for i in range(l + 1, m):
                                b[i + j * ldb] = (
                                    b[i + j * ldb]
                                    - b[l + j * ldb] * a[i + l * lda]
                                )
        else:
            if upper:
                for j in range(n):
                    for i in range(m):
                        if no_unit:
                            b[i + j * ldb] = b[i + j * ldb] / a[i + i * lda]
                        for l in range(i + 1, m):
                            b[i + j * ldb] = (
                                b[i + j * ldb] - a[i + l * lda] * b[l + j * ldb]
                            )
            else:
                for j in range(n):
                    for i in range(m - 1, -1, -1):
                        if no_unit:
                            b[i + j * ldb] = b[i + j * ldb] / a[i + i * lda]
                        for l in range(i):
                            b[i + j * ldb] = (
                                b[i + j * ldb] - a[i + l * lda] * b[l + j * ldb]
                            )
    else:
        if no_trans:
            if upper:
                for j in range(n):
                    if no_unit:
                        for i in range(m):
                            b[i + j * ldb] = b[i + j * ldb] / a[j + j * lda]
                    for l in range(j + 1, n):
                        for i in range(m):
                            b[i + j * ldb] = (
                                b[i + j * ldb] - a[j + l * lda] * b[i + l * ldb]
                            )
            else:
                for j in range(n - 1, -1, -1):
                    if no_unit:
                        for i in range(m):
                            b[i + j * ldb] = b[i + j * ldb] / a[j + j * lda]
                    for l in range(j):
                        for i in range(m):
                            b[i + j * ldb] = (
                                b[i + j * ldb] - a[j + l * lda] * b[i + l * ldb]
                            )
        else:
            if upper:
                for j in range(n - 1, -1, -1):
                    for l in range(j):
                        for i in range(m):
                            b[i + j * ldb] = (
                                b[i + j * ldb] - a[l + j * lda] * b[i + l * ldb]
                            )
                    if no_unit:
                        for i in range(m):
                            b[i + j * ldb] = b[i + j * ldb] / a[j + j * lda]
            else:
                for j in range(n):
                    for l in range(j + 1, n):
                        for i in range(m):
                            b[i + j * ldb] = (
                                b[i + j * ldb] - a[l + j * lda] * b[i + l * ldb]
                            )
                    if no_unit:
                        for i in range(m):
                            b[i + j * ldb] = b[i + j * ldb] / a[j + j * lda]

    return
