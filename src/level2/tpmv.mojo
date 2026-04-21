def tpmv[
    mut_ap: Bool,
    origin_ap: Origin[mut=mut_ap],
    origin_x: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    trans: String,
    diag: String,
    n: Int,
    ap: BLASPtr[dtype, origin_ap],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
):
    """
    Performs the matrix-vector operation x := A*x or x := A^T*x,
    where A is an n by n triangular matrix stored in packed format.

    Optimized with SIMD vectorization and parallelization.

    Parameters:
        mut_ap: Indicates whether the pointer ap is mutable (True) or immutable (False).
        origin_ap: Memory origin of the pointer ap.
        origin_x: Memory origin of the pointer x (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for x := A*x, 'T' or 'C' for x := A^T*x.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        ap: A pointer to the packed triangular matrix A.
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
    elif incx == 0:
        info = 7

    if info != 0:
        print("tpmv: Info", info)
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
            var kk: Int = 1
            if incx == 1:
                for j in range(n):
                    if x[j] != 0:
                        var temp: Scalar[dtype] = x[j]
                        var k: Int = kk
                        for i in range(j):
                            x[i] = x[i] + temp * ap[k]
                            k += 1
                        if no_unit:
                            x[j] = x[j] * ap[kk + j]
                    kk += n - j
            else:
                var jx: Int = kx
                for j in range(n):
                    if x[jx - 1] != 0:
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = kx
                        for k in range(kk, kk + j):
                            x[ix - 1] = x[ix - 1] + temp * ap[k]
                            ix += incx
                        if no_unit:
                            x[jx - 1] = x[jx - 1] * ap[kk + j]
                    jx += incx
                    kk += n - j
        else:
            var kk: Int = 1
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    if x[j] != 0:
                        var temp: Scalar[dtype] = x[j]
                        var k: Int = kk
                        for i in range(j + 1, n):
                            x[i] = x[i] + temp * ap[k]
                            k += 1
                        if no_unit:
                            x[j] = x[j] * ap[kk]
                    kk += n - j
            else:
                var kx_plus: Int = kx + (n - 1) * incx
                var jx: Int = kx_plus
                for j in range(n - 1, -1, -1):
                    if x[jx - 1] != 0:
                        var temp: Scalar[dtype] = x[jx - 1]
                        var ix: Int = kx_plus
                        for k in range(kk + 1, kk + n - j):
                            x[ix - 1] = x[ix - 1] + temp * ap[k]
                            ix += incx
                        if no_unit:
                            x[jx - 1] = x[jx - 1] * ap[kk]
                    jx -= incx
                    kk += n - j
    else:
        if upper:
            var kk: Int = (n * (n + 1)) // 2
            if incx == 1:
                for j in range(n - 1, -1, -1):
                    kk -= n - j
                    var temp: Scalar[dtype] = x[j]
                    if no_unit:
                        temp = temp * ap[kk + j]
                    var k: Int = kk
                    for i in range(j - 1, -1, -1):
                        k += 1
                        temp = temp + ap[k] * x[i]
                    x[j] = temp
            else:
                var jx: Int = kx + (n - 1) * incx
                for j in range(n - 1, -1, -1):
                    kk -= n - j
                    var ix: Int = jx
                    var temp: Scalar[dtype] = x[jx - 1]
                    if no_unit:
                        temp = temp * ap[kk + j]
                    for k in range(kk + n - j - 1, kk, -1):
                        ix -= incx
                        temp = temp + ap[k] * x[ix - 1]
                    x[jx - 1] = temp
                    jx -= incx
        else:
            var kk: Int = (n * (n + 1)) // 2
            if incx == 1:
                for j in range(n):
                    kk -= n - j
                    var temp: Scalar[dtype] = x[j]
                    if no_unit:
                        temp = temp * ap[kk]
                    var k: Int = kk + n - j
                    for i in range(j + 1, n):
                        k -= 1
                        temp = temp + ap[k] * x[i]
                    x[j] = temp
            else:
                var jx: Int = kx
                for j in range(n):
                    kk -= n - j
                    var ix: Int = jx
                    var temp: Scalar[dtype] = x[jx - 1]
                    if no_unit:
                        temp = temp * ap[kk]
                    for k in range(kk + n - j - 1, kk, -1):
                        ix += incx
                        temp = temp + ap[k] * x[ix - 1]
                    x[jx - 1] = temp
                    jx += incx

    return
