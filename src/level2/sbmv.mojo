def sbmv[
    dtype: DType
](
    uplo: String,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[Scalar[dtype]],
    lda: Int,
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
    beta: Scalar[dtype],
    y: BLASPtr[Scalar[dtype]],
    incy: Int,
):
    """
    Performs the matrix-vector operation y := alpha*A*x + beta*y,
    where A is an n by n symmetric band matrix.

    Parameters:
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        k: The number of super-diagonals of the matrix A.
        alpha: The scalar multiplier for the matrix-vector product.
        a: A pointer to the first element of the matrix A (stored in band format).
        lda: The leading dimension of the matrix A.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
        beta: The scalar multiplier for the vector y.
        y: A pointer to the first element of the vector y.
        incy: The increment for the elements of y.
    """
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif n < 0:
        info = 2
    elif k < 0:
        info = 3
    elif lda < k + 1:
        info = 6
    elif incx == 0:
        info = 8
    elif incy == 0:
        info = 11

    if info != 0:
        print("sbmv: Info", info)
        return

    if n == 0 or (alpha == 0 and beta == 1):
        return

    var leny: Int = n
    var ky: Int = 1
    if incy < 0:
        ky = 1 - (leny - 1) * incy

    if beta != 1:
        if incy == 1:
            if beta == 0:
                for i in range(leny):
                    y[i] = 0
            else:
                for i in range(leny):
                    y[i] = beta * y[i]
        else:
            var iy: Int = ky
            if beta == 0:
                for i in range(leny):
                    y[iy - 1] = 0
                    iy += incy
            else:
                for i in range(leny):
                    y[iy - 1] = beta * y[iy - 1]
                    iy += incy

    if alpha == 0:
        return

    var kx: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx

    var upper = uplo == "U" or uplo == "u"

    if upper:
        if incx == 1:
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[j]
                var temp2: Scalar[dtype] = 0
                var j_offset: Int = k - j
                for i in range(max(0, j - k), j):
                    y[i] = y[i] + temp1 * a[j_offset + i + j * lda]
                    temp2 = temp2 + a[j_offset + i + j * lda] * x[i]
                y[j] = y[j] + temp1 * a[k + j * lda] + alpha * temp2
        else:
            var jx: Int = kx
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[jx - 1]
                var temp2: Scalar[dtype] = 0
                var ix: Int = kx + max(0, j - k) * incx
                var j_offset: Int = k - j
                for i in range(max(0, j - k), j):
                    y[i] = y[i] + temp1 * a[j_offset + i + j * lda]
                    temp2 = temp2 + a[j_offset + i + j * lda] * x[ix - 1]
                    ix += incx
                y[j] = y[j] + temp1 * a[k + j * lda] + alpha * temp2
                jx += incx
    else:
        if incx == 1:
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[j]
                var temp2: Scalar[dtype] = 0
                for i in range(j + 1, min(n, j + k + 1)):
                    y[i] = y[i] + temp1 * a[i - j + j * lda]
                    temp2 = temp2 + a[i - j + j * lda] * x[i]
                y[j] = y[j] + temp1 * a[j * lda] + alpha * temp2
        else:
            var jx: Int = kx
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[jx - 1]
                var temp2: Scalar[dtype] = 0
                var ix: Int = jx + incx
                for i in range(j + 1, min(n, j + k + 1)):
                    y[i] = y[i] + temp1 * a[i - j + j * lda]
                    temp2 = temp2 + a[i - j + j * lda] * x[ix - 1]
                    ix += incx
                y[j] = y[j] + temp1 * a[j * lda] + alpha * temp2
                jx += incx

    return
