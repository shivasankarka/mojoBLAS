from src import BLASPtr


fn dscal[
    dtype: DType
](
    n: Int32,
    alpha: Scalar[dtype],
    x: BLASPtr[Scalar[dtype]],
    incx: Int32,
) -> None:
    """
    Scale a vector by a scalar: X := alpha * X.

    Args:
        n: Number of elements in vector X.
        alpha: Scalar multiplier.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
    """
    if n <= 0:
        return

    var ix: Int32 = 0

    # Handle negative increments
    if incx < 0:
        ix = (-n + 1) * incx

    # Perform the scaling operation: x[i] = alpha * x[i]
    for i in range(n):
        x[ix] = alpha * x[ix]
        ix += incx
