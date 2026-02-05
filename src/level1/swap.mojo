from src import BLASPtr


fn dswap[
    dtype: DType
](
    n: Int32,
    x: BLASPtr[Scalar[dtype]],
    incx: Int32,
    y: BLASPtr[Scalar[dtype]],
    incy: Int32,
) -> None:
    """
    Swap the elements of two vectors X and Y: X <-> Y.

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

    # Perform the swap operation: swap x[i] and y[i]
    for i in range(n):
        temp: Scalar[dtype] = x[ix]
        x[ix] = y[iy]
        y[iy] = temp
        ix += incx
        iy += incy
