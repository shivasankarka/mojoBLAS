from src import BLASPtr


fn daxpy[
    dtype: DType
](
    n: Int32,
    alpha: Scalar[dtype],
    x: BLASPtr[Scalar[dtype]],
    incx: Int32,
    y: BLASPtr[Scalar[dtype]],
    incy: Int32,
) -> None:
    """
    Perform the AXPY operation: Y := alpha * X + Y.

    Args:
        n: Number of elements in vectors X and Y.
        alpha: Scalar multiplier for vector X.
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

    # Perform the AXPY operation: y[i] = alpha * x[i] + y[i]
    for i in range(n):
        y[iy] = alpha * x[ix] + y[iy]
        ix += incx
        iy += incy
