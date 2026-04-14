def trmm[
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
    Performs the matrix-matrix operation B := alpha*A*B or B := alpha*B*A,
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
        alpha: The scalar multiplier for the matrix product.
        a: A pointer to the first element of the triangular matrix A.
        lda: The leading dimension of the matrix A.
        b: A pointer to the first element of the matrix B (input/output).
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
        print("trmm: Info", info)
        return

    if m == 0 or n == 0:
        return

    if alpha == 0:
        for j in range(n):
            for i in range(m):
                b[i + j * ldb] = 0
        return

    var left_side = side == "L" or side == "l"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans_a == "N" or trans_a == "n"
    var no_unit = diag == "N" or diag == "n"

    if left_side:
        if no_trans:
            if upper:
                for j in range(n):
                    for k in range(m):
                        if b[k + j * ldb] != 0:
                            var temp: Scalar[dtype] = alpha * b[k + j * ldb]
                            for i in range(k):
                                b[i + j * ldb] = (
                                    b[i + j * ldb] + temp * a[i + k * lda]
                                )
                            if no_unit:
                                temp = temp * a[k + k * lda]
                            b[k + j * ldb] = temp
            else:
                for j in range(n):
                    for k in range(m - 1, -1, -1):
                        if b[k + j * ldb] != 0:
                            var temp: Scalar[dtype] = alpha * b[k + j * ldb]
                            b[k + j * ldb] = temp
                            if no_unit:
                                b[k + j * ldb] = b[k + j * ldb] * a[k + k * lda]
                            for i in range(k + 1, m):
                                b[i + j * ldb] = (
                                    b[i + j * ldb] + temp * a[i + k * lda]
                                )
        else:
            if upper:
                for j in range(n):
                    for i in range(m - 1, -1, -1):
                        var temp: Scalar[dtype] = b[i + j * ldb]
                        if no_unit:
                            temp = temp * a[i + i * lda]
                        for k in range(i):
                            temp = temp + a[k + i * lda] * b[k + j * ldb]
                        b[i + j * ldb] = alpha * temp
            else:
                for j in range(n):
                    for i in range(m):
                        var temp: Scalar[dtype] = b[i + j * ldb]
                        if no_unit:
                            temp = temp * a[i + i * lda]
                        for k in range(i + 1, m):
                            temp = temp + a[k + i * lda] * b[k + j * ldb]
                        b[i + j * ldb] = alpha * temp
    else:
        if no_trans:
            if upper:
                for j in range(n - 1, -1, -1):
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[j + j * lda]
                    for i in range(m):
                        b[i + j * ldb] = temp * b[i + j * ldb]
                    for k in range(j):
                        if a[k + j * lda] != 0:
                            temp = alpha * a[k + j * lda]
                            for i in range(m):
                                b[i + j * ldb] = (
                                    b[i + j * ldb] + temp * b[i + k * ldb]
                                )
            else:
                for j in range(n):
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[j + j * lda]
                    for i in range(m):
                        b[i + j * ldb] = temp * b[i + j * ldb]
                    for k in range(j + 1, n):
                        if a[k + j * lda] != 0:
                            temp = alpha * a[k + j * lda]
                            for i in range(m):
                                b[i + j * ldb] = (
                                    b[i + j * ldb] + temp * b[i + k * ldb]
                                )
        else:
            if upper:
                for k in range(n):
                    for j in range(k):
                        if a[j + k * lda] != 0:
                            var temp: Scalar[dtype] = alpha * a[j + k * lda]
                            for i in range(m):
                                b[i + j * ldb] = (
                                    b[i + j * ldb] + temp * b[i + k * ldb]
                                )
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[k + k * lda]
                    if temp != 1:
                        for i in range(m):
                            b[i + k * ldb] = temp * b[i + k * ldb]
            else:
                for k in range(n - 1, -1, -1):
                    for j in range(k + 1, n):
                        if a[j + k * lda] != 0:
                            var temp: Scalar[dtype] = alpha * a[j + k * lda]
                            for i in range(m):
                                b[i + j * ldb] = (
                                    b[i + j * ldb] + temp * b[i + k * ldb]
                                )
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[k + k * lda]
                    if temp != 1:
                        for i in range(m):
                            b[i + k * ldb] = temp * b[i + k * ldb]

    return
