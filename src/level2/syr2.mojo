def syr2[
    dtype: DType
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
    y: BLASPtr[Scalar[dtype]],
    incy: Int,
    a: BLASPtr[Scalar[dtype]],
    lda: Int,
):
    """
    Performs the symmetric rank 2 operation A := alpha*x*y^T + alpha*y*x^T + A,
    where A is an n by n symmetric matrix.

    Parameters:
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        alpha: The scalar multiplier for the rank-2 update.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
        y: A pointer to the first element of the vector y.
        incy: The increment for the elements of y.
        a: A pointer to the first element of the matrix A (input/output).
        lda: The leading dimension of the matrix A.
    """
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif n < 0:
        info = 2
    elif incx == 0:
        info = 5
    elif incy == 0:
        info = 7
    elif lda < max(1, n):
        info = 9

    if info != 0:
        print("syr2: Info", info)
        return

    if n == 0 or alpha == 0:
        return

    var kx: Int = 1
    var ky: Int = 1
    if incx < 0:
        kx = 1 - (n - 1) * incx
    if incy < 0:
        ky = 1 - (n - 1) * incy

    var upper = uplo == "U" or uplo == "u"

    if upper:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0 or y[j] != 0:
                    var temp1: Scalar[dtype] = alpha * y[j]
                    var temp2: Scalar[dtype] = alpha * x[j]
                    for i in range(j + 1):
                        a[i + j * lda] = (
                            a[i + j * lda] + x[i] * temp1 + y[i] * temp2
                        )
        else:
            var jx: Int = kx
            var jy: Int = ky
            for j in range(n):
                if x[jx - 1] != 0 or y[jy - 1] != 0:
                    var temp1: Scalar[dtype] = alpha * y[jy - 1]
                    var temp2: Scalar[dtype] = alpha * x[jx - 1]
                    var ix: Int = kx
                    var iy: Int = ky
                    for i in range(j + 1):
                        a[i + j * lda] = (
                            a[i + j * lda]
                            + x[ix - 1] * temp1
                            + y[iy - 1] * temp2
                        )
                        ix += incx
                        iy += incy
                jx += incx
                jy += incy
    else:
        if incx == 1 and incy == 1:
            for j in range(n):
                if x[j] != 0 or y[j] != 0:
                    var temp1: Scalar[dtype] = alpha * y[j]
                    var temp2: Scalar[dtype] = alpha * x[j]
                    for i in range(j, n):
                        a[i + j * lda] = (
                            a[i + j * lda] + x[i] * temp1 + y[i] * temp2
                        )
        else:
            var jx: Int = kx
            var jy: Int = ky
            for j in range(n):
                if x[jx - 1] != 0 or y[jy - 1] != 0:
                    var temp1: Scalar[dtype] = alpha * y[jy - 1]
                    var temp2: Scalar[dtype] = alpha * x[jx - 1]
                    var ix: Int = jx
                    var iy: Int = jy
                    for i in range(j, n):
                        a[i + j * lda] = (
                            a[i + j * lda]
                            + x[ix - 1] * temp1
                            + y[iy - 1] * temp2
                        )
                        ix += incx
                        iy += incy
                jx += incx
                jy += incy

    return
