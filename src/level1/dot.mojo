from src import BLASPtr


fn ddot[
    dtype: DType
](
    n: Int32,
    x: BLASPtr[Scalar[dtype]],
    incx: Int32,
    y: BLASPtr[Scalar[dtype]],
    incy: Int32,
) -> Scalar[dtype]:
    """
    Compute the dot product of two vectors X and Y.

    Args:
        n: Number of elements in vectors X and Y.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        y: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.

    Returns:
        The dot product as a scalar value.
    """
    result: Scalar[dtype] = 0

    if n <= 0:
        return result

    var ix: Int32 = 0
    var iy: Int32 = 0

    # Handle negative increments
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    # Compute the dot product
    for i in range(n):
        result += x[ix] * y[iy]
        ix += incx
        iy += incy

    return result
