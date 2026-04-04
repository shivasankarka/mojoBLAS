def spmv[
    dtype: DType
](
    uplo: String,
    n: Int,
    alpha: Scalar[dtype],
    ap: BLASPtr[Scalar[dtype]],
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
    beta: Scalar[dtype],
    y: BLASPtr[Scalar[dtype]],
    incy: Int,
):
    """
    Performs the matrix-vector operation y := alpha*A*x + beta*y,
    where A is an n by n symmetric matrix stored in packed format.

    Parameters:
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        n: The order of the matrix A.
        alpha: The scalar multiplier for the matrix-vector product.
        ap: A pointer to the packed symmetric matrix A.
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
    elif incx == 0:
        info = 5
    elif incy == 0:
        info = 8

    if info != 0:
        print("spmv: Info", info)
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
        var kk: Int = 1
        if incx == 1:
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[j]
                var temp2: Scalar[dtype] = 0
                var k: Int = kk
                for i in range(j):
                    y[i] = y[i] + temp1 * ap[k]
                    temp2 = temp2 + ap[k] * x[i]
                    k += 1
                y[j] = y[j] + temp1 * ap[kk + j] + alpha * temp2
                kk += n - j
        else:
            var jx: Int = kx
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[jx - 1]
                var temp2: Scalar[dtype] = 0
                var ix: Int = kx
                var i: Int = 0
                for k in range(kk, kk + j):
                    y[i] = y[i] + temp1 * ap[k]
                    temp2 = temp2 + ap[k] * x[ix - 1]
                    ix += incx
                    i += 1
                y[j] = y[j] + temp1 * ap[kk + j] + alpha * temp2
                jx += incx
                kk += n - j
    else:
        var kk: Int = 1
        if incx == 1:
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[j]
                var temp2: Scalar[dtype] = 0
                var k: Int = kk
                for i in range(j, n):
                    y[i] = y[i] + temp1 * ap[k]
                    temp2 = temp2 + ap[k] * x[i]
                    k += 1
                y[j] = y[j] + temp1 * ap[kk] + alpha * temp2
                kk += n - j
        else:
            var jx: Int = kx
            for j in range(n):
                var temp1: Scalar[dtype] = alpha * x[jx - 1]
                var temp2: Scalar[dtype] = 0
                var ix: Int = jx
                var i: Int = j
                for k in range(kk, kk + n - j):
                    y[i] = y[i] + temp1 * ap[k]
                    temp2 = temp2 + ap[k] * x[ix - 1]
                    ix += incx
                    i += 1
                y[j] = y[j] + temp1 * ap[kk] + alpha * temp2
                jx += incx
                kk += n - j

    return
