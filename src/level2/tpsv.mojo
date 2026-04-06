def tpsv[
    dtype: DType
](
    uplo: String,
    trans: String,
    diag: String,
    n: Int,
    ap: BLASPtr[Scalar[dtype]],
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
):
    """
    Solves a system of linear equations A*x = b or A^T*x = b,
    where A is an n by n triangular matrix stored in packed format.

    Parameters:
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for A*x = b, 'T' or 'C' for A^T*x = b.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        ap: A pointer to the packed triangular matrix A.
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
    elif incx == 0:
        info = 7

    if info != 0:
        print("tpsv: Info", info)
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
            var kk: Int = (n * (n + 1)) // 2
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    kk -= n - j
                    if x[j] != 0:
                        if no_unit:
                            x[j] = x[j] / ap[kk + j]
                        var temp: Scalar[dtype] = x[j]
                        var k: Int = kk
                        for i in range(j - 1, -1, -1):
                            k += 1
                            x[i] = x[i] - temp * ap[k]
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    kk -= n - j
                    jx -= incx
                    if x[jx - 1] != 0:
                        if no_unit:
                            x[jx - 1] = x[jx - 1] / ap[kk + j]
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = jx
                        for k in range(kk + n - j - 1, kk, -1):
                            ix -= incx
                            x[ix - 1] = x[ix - 1] - temp * ap[k]
        else:
            var kk: Int = 1
            if incx == 1:
                for j in range(n):
                    if x[j] != 0:
                        if no_unit:
                            x[j] = x[j] / ap[kk]
                        var temp: Scalar[dtype] = x[j]
                        for k in range(kk + n - j - 1, kk, -1):
                            x[k - kk] = x[k - kk] - temp * ap[k]
                    kk += n - j
            else:
                var jx: Int = kx
                for j in range(n):
                    if x[jx - 1] != 0:
                        if no_unit:
                            x[jx - 1] = x[jx - 1] / ap[kk]
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = jx
                        for k in range(kk + n - j - 1, kk, -1):
                            ix += incx
                            x[ix - 1] = x[ix - 1] - temp * ap[k]
                    jx += incx
                    kk += n - j
    else:
        if upper:
            var kk: Int = 1
            if incx == 1:
                for j in range(n):
                    var temp: Scalar[dtype] = x[j]
                    var k: Int = kk
                    for i in range(j):
                        k += 1
                        temp = temp - ap[k] * x[i]
                    if no_unit:
                        temp = temp / ap[kk + j]
                    x[j] = temp
                    kk += n - j
            else:
                var jx: Int = kx
                for j in range(n):
                    var ix: Int = kx
                    var temp: Scalar[dtype] = x[jx - 1]
                    for k in range(kk, kk + j):
                        temp = temp - ap[k] * x[ix - 1]
                        ix += incx
                    if no_unit:
                        temp = temp / ap[kk + j]
                    x[jx - 1] = temp
                    jx += incx
                    kk += n - j
        else:
            var kk: Int = (n * (n + 1)) // 2
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    var temp: Scalar[dtype] = x[j]
                    var k: Int = kk
                    for i in range(j + 1, n):
                        k -= 1
                        temp = temp - ap[k] * x[i]
                    if no_unit:
                        temp = temp / ap[kk]
                    x[j] = temp
                    kk -= n - j
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    var ix: Int = kx_plus
                    var temp: Scalar[dtype] = x[jx - 1]
                    for k in range(kk + n - j - 1, kk, -1):
                        temp = temp - ap[k] * x[ix - 1]
                        ix -= incx
                    if no_unit:
                        temp = temp / ap[kk]
                    x[jx - 1] = temp
                    jx -= incx
                    kk -= n - j

    return
