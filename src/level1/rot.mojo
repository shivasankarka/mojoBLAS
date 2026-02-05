from src import BLASPtr


fn drot[
    dtype: DType
](
    n: Int32,
    x: BLASPtr[Scalar[dtype]],
    incx: Int32,
    y: BLASPtr[Scalar[dtype]],
    incy: Int32,
    c: Scalar[dtype],
    s: Scalar[dtype],
) -> None:
    """
    Apply Givens rotation to vectors X and Y.

    Performs the transformation:
    [ x[i] ]   [  c  s ] [ x[i] ]
    [ y[i] ] = [ -s  c ] [ y[i] ]

    Args:
        n: Number of elements in vectors X and Y.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        y: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.
        c: Cosine component of the rotation.
        s: Sine component of the rotation.
    """
    if n <= 0:
        return

    var ix: Int32 = 0
    var iy: Int32 = 0

    # Handle negative increments
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    # Apply Givens rotation to each element pair
    for i in range(n):
        var temp_x = c * x[ix] + s * y[iy]
        var temp_y = -s * x[ix] + c * y[iy]
        x[ix] = temp_x
        y[iy] = temp_y
        ix += incx
        iy += incy
