from src import BLASPtr


fn dcopy[
    dtype: DType
](
    n: Int32,
    x: BLASPtr[Scalar[dtype]],
    incx: Int32,
    y: BLASPtr[Scalar[dtype]],
    incy: Int32,
) raises -> None:
    """
    Copy a vector X to vector Y: Y := X.

    Args:
        n: Number of elements in vectors X and Y.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        y: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.
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

    # Perform the copy operation: y[i] = x[i]
    for i in range(n):
        y[iy] = x[ix]
        ix += incx
        iy += incy
