def gbmv[
    mut_a: Bool,
    mut_x: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](
    trans: String,
    m: Int,
    n: Int,
    kl: Int,
    ku: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    beta: Scalar[dtype],
    y: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Performs the matrix-vector operation y := alpha*A*x + beta*y,
    where A is an m by n band matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y.
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        trans: Specifies the operation: 'N' for y := alpha*A*x + beta*y, 'T' or 'C' for y := alpha*A^T*x + beta*y.
        m: The number of rows of the matrix A.
        n: The number of columns of the matrix A.
        kl: The number of sub-diagonals of the matrix A.
        ku: The number of super-diagonals of the matrix A.
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
    if (
        trans != "N"
        and trans != "n"
        and trans != "T"
        and trans != "t"
        and trans != "C"
        and trans != "c"
    ):
        info = 1
    elif m < 0:
        info = 2
    elif n < 0:
        info = 3
    elif kl < 0:
        info = 4
    elif ku < 0:
        info = 5
    elif lda < kl + ku + 1:
        info = 7
    elif incx == 0:
        info = 9
    elif incy == 0:
        info = 12

    if info != 0:
        print("gbmv: Info", info)
        return

    if m == 0 or n == 0 or (alpha == 0 and beta == 1):
        return

    var lenx: Int
    var leny: Int

    if trans == "N" or trans == "n":
        lenx = n
        leny = m
    else:
        lenx = m
        leny = n

    var kx: Int = 1
    var ky: Int = 1
    if incx < 0:
        kx = 1 - (lenx - 1) * incx
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

    var no_trans = trans == "N" or trans == "n"

    if no_trans:
        var jx: Int = kx
        if incy == 1:
            for j in range(n):
                if x[jx - 1] != 0:
                    var temp: Scalar[dtype] = alpha * x[jx - 1]
                    var i_start: Int = max(0, j - ku)
                    var i_end: Int = min(m, j + kl + 1)
                    for i in range(i_start, i_end):
                        y[i] = y[i] + temp * a[ku - j + i + j * lda]
                jx += incx
        else:
            for j in range(n):
                if x[jx - 1] != 0:
                    var temp: Scalar[dtype] = alpha * x[jx - 1]
                    var iy: Int = ky
                    var i_start: Int = max(0, j - ku)
                    var i_end: Int = min(m, j + kl + 1)
                    for i in range(i_start, i_end):
                        y[iy - 1] = y[iy - 1] + temp * a[ku - j + i + j * lda]
                        iy += incy
                jx += incx
    else:
        var jy: Int = ky
        if incx == 1:
            for j in range(n):
                var temp: Scalar[dtype] = 0
                var i_start: Int = max(0, j - ku)
                var i_end: Int = min(m, j + kl + 1)
                for i in range(i_start, i_end):
                    temp = temp + a[ku - j + i + j * lda] * x[i]
                y[jy - 1] = y[jy - 1] + alpha * temp
                jy += incy
        else:
            for j in range(n):
                var temp: Scalar[dtype] = 0
                var ix: Int = kx
                var i_start: Int = max(0, j - ku)
                var i_end: Int = min(m, j + kl + 1)
                for i in range(i_start, i_end):
                    temp = temp + a[ku - j + i + j * lda] * x[ix - 1]
                    ix += incx
                y[jy - 1] = y[jy - 1] + alpha * temp
                jy += incy

    return
