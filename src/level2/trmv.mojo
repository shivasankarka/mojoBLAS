def trmv[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    trans: String,
    diag: String,
    n: Int,
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
):
    """
    Performs the matrix-vector operation x := A*x or x := A^T*x,
    where A is an n by n triangular matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_x: Memory origin of the pointer x (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for x := A*x, 'T' or 'C' for x := A^T*x.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
        x: A pointer to the first element of the vector x (input/output).
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
    elif lda < max(1, n):
        info = 6
    elif incx == 0:
        info = 8

    if info != 0:
        print("trmv: Info", info)
        return

    if n == 0:
        return

    var no_unit = diag == "N" or diag == "n"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans == "N" or trans == "n"

    var kx: Int = 1
    var ky: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    if no_trans:
        if upper:
            if incx == 1:
                for j in range(n):
                    if x[j] != 0:
                        var temp: Scalar[dtype] = x[j]
                        for i in range(j):
                            x[i] = x[i] + temp * a[i + j * lda]
                        if no_unit:
                            x[j] = x[j] * a[j + j * lda]
            else:
                var jx: Int = kx
                for j in range(n):
                    if x[jx - 1] != 0:
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = kx
                        for i in range(j):
                            x[ix - 1] = x[ix - 1] + temp * a[i + j * lda]
                            ix += incx
                        if no_unit:
                            x[jx - 1] = x[jx - 1] * a[j + j * lda]
                    jx += incx
        else:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    if x[j] != 0:
                        var temp: Scalar[dtype] = x[j]
                        for i in range(n - 1, j, -1):
                            x[i] = x[i] + temp * a[i + j * lda]
                        if no_unit:
                            x[j] = x[j] * a[j + j * lda]
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    if x[jx - 1] != 0:
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = kx_plus
                        for i in range(n - 1, j, -1):
                            x[ix - 1] = x[ix - 1] + temp * a[i + j * lda]
                            ix -= incx
                        if no_unit:
                            x[jx - 1] = x[jx - 1] * a[j + j * lda]
                    jx -= incx
    else:
        if upper:
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    var temp: Scalar[dtype] = x[j]
                    if no_unit:
                        temp = temp * a[j + j * lda]
                    for i in range(j - 1, -1, -1):
                        temp = temp + a[i + j * lda] * x[i]
                    x[j] = temp
            else:
                var jx: Int = kx + (n - 1) * incx
                for j in range(n - 1, -1, -1):
                    var ix: Int = jx
                    var temp: Scalar[dtype] = x[jx - 1]
                    if no_unit:
                        temp = temp * a[j + j * lda]
                    for i in range(j - 1, -1, -1):
                        ix -= incx
                        temp = temp + a[i + j * lda] * x[ix - 1]
                    x[jx - 1] = temp
                    jx -= incx
        else:
            if incx == 1:
                for j in range(n):
                    var temp: Scalar[dtype] = x[j]
                    if no_unit:
                        temp = temp * a[j + j * lda]
                    for i in range(j + 1, n):
                        temp = temp + a[i + j * lda] * x[i]
                    x[j] = temp
            else:
                var jx: Int = kx
                for j in range(n):
                    var ix: Int = jx
                    var temp: Scalar[dtype] = x[jx - 1]
                    if no_unit:
                        temp = temp * a[j + j * lda]
                    for i in range(j + 1, n):
                        ix += incx
                        temp = temp + a[i + j * lda] * x[ix - 1]
                    x[jx - 1] = temp
                    jx += incx

    return
