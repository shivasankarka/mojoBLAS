def tbsv[
    dtype: DType
](
    uplo: String,
    trans: String,
    diag: String,
    n: Int,
    k: Int,
    a: BLASPtr[Scalar[dtype]],
    lda: Int,
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
):
    """
    Solves a system of linear equations A*x = b or A^T*x = b,
    where A is an n by n triangular band matrix.

    Parameters:
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for A*x = b, 'T' or 'C' for A^T*x = b.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        k: The number of super-diagonals (if upper) or sub-diagonals (if lower).
        a: A pointer to the first element of the matrix A (stored in band format).
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
    elif k < 0:
        info = 5
    elif lda < k + 1:
        info = 7
    elif incx == 0:
        info = 9

    if info != 0:
        print("tbsv: Info", info)
        return

    if n == 0:
        return

    var no_unit = diag == "N" or diag == "n"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans == "N" or trans == "n"

    var kx: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    if no_trans:
        if upper:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    if x[j] != 0:
                        if no_unit:
                            x[j] = x[j] / a[k + j * lda]
                        var temp: Scalar[dtype] = x[j]
                        var i_start: Int = max(0, j - k)
                        for i in range(j - 1, i_start - 1, -1):
                            x[i] = x[i] - temp * a[k - j + i + j * lda]
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    jx -= incx
                    if x[jx - 1] != 0:
                        if no_unit:
                            x[jx - 1] = x[jx - 1] / a[k + j * lda]
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = jx
                        var i_start: Int = max(0, j - k)
                        for i in range(j - 1, i_start - 1, -1):
                            ix -= incx
                            x[ix - 1] = (
                                x[ix - 1] - temp * a[k - j + i + j * lda]
                            )
        else:
            if incx == 1:
                for j in range(n):
                    if x[j] != 0:
                        if no_unit:
                            x[j] = x[j] / a[k + j * lda]
                        var temp: Scalar[dtype] = x[j]
                        var i_end: Int = min(n, j + k + 1)
                        for i in range(j + 1, i_end):
                            x[i] = x[i] - temp * a[k - j + i + j * lda]
            else:
                var jx: Int = kx
                for j in range(n):
                    if x[jx - 1] != 0:
                        if no_unit:
                            x[jx - 1] = x[jx - 1] / a[k + j * lda]
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = jx
                        var i_end: Int = min(n, j + k + 1)
                        for i in range(j + 1, i_end):
                            ix += incx
                            x[ix - 1] = (
                                x[ix - 1] - temp * a[k - j + i + j * lda]
                            )
                    jx += incx
    else:
        if upper:
            if incx == 1:
                for j in range(n):
                    var temp: Scalar[dtype] = x[j]
                    var i_start: Int = max(0, j - k)
                    for i in range(i_start, j):
                        temp = temp - a[k - j + i + j * lda] * x[i]
                    if no_unit:
                        temp = temp / a[k + j * lda]
                    x[j] = temp
            else:
                var jx: Int = kx
                for j in range(n):
                    var ix: Int = kx
                    var temp: Scalar[dtype] = x[jx - 1]
                    var i_start: Int = max(0, j - k)
                    for i in range(i_start, j):
                        temp = temp - a[k - j + i + j * lda] * x[ix - 1]
                        ix += incx
                    if no_unit:
                        temp = temp / a[k + j * lda]
                    x[jx - 1] = temp
                    jx += incx
        else:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    var temp: Scalar[dtype] = x[j]
                    var i_end: Int = min(n, j + k + 1)
                    for i in range(j + 1, i_end):
                        temp = temp - a[k - j + i + j * lda] * x[i]
                    if no_unit:
                        temp = temp / a[k + j * lda]
                    x[j] = temp
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    var ix: Int = kx_plus
                    var temp: Scalar[dtype] = x[jx - 1]
                    var i_end: Int = min(n, j + k + 1)
                    for i in range(j + 1, i_end):
                        temp = temp - a[k - j + i + j * lda] * x[ix - 1]
                        ix -= incx
                    if no_unit:
                        temp = temp / a[k + j * lda]
                    x[jx - 1] = temp
                    jx -= incx

    return
