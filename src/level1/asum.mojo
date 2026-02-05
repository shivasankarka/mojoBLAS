from src import BLASPtr


fn dasum[
    dtype: DType
](n: Int32, x: BLASPtr[Scalar[dtype]], incx: Int32) -> Scalar[dtype]:
    """
    Compute the sum of absolute values of elements in vector X.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The sum of absolute values as a scalar value.
    """
    if n <= 0 or incx == 0:
        return 0

    var sum: Scalar[dtype] = 0.0
    var ix: Int32 = 0

    # Handle negative increments
    if incx < 0:
        ix = (-n + 1) * incx

    # Compute the sum of absolute values
    for i in range(n):
        sum += abs(x[ix])
        ix += incx

    return sum
