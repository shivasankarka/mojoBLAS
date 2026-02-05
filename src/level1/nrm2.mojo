from math import sqrt

from src import BLASPtr


fn dnrm2[
    dtype: DType
](n: Int32, x: BLASPtr[Scalar[dtype]], incx: Int32,) -> Scalar[dtype]:
    """
    Compute the Euclidean norm (2-norm) of a vector X.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The Euclidean norm as a scalar value.
    """
    result: Scalar[dtype] = 0
    if n <= 0:
        return result
    var ix: Int32 = 0
    # Handle negative increments
    if incx < 0:
        ix = (-n + 1) * incx
    # Compute the sum of squares
    for i in range(n):
        result += x[ix] * x[ix]
        ix += incx
    return sqrt(result)
