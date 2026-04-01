from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def iamax[dtype: DType](n: Int, x: BLASPtr[Scalar[dtype]], incx: Int) -> Int:
    """
    Find the index of the element with maximum absolute value in vector X.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The index (1-based) of the element with maximum absolute value.
        Returns 0 if n <= 0.
    """
    if n <= 0:
        return 0

    if n == 1:
        return 1

    var ix: Int = 0
    var imax: Int = 1
    var max_val: Scalar[dtype] = abs(x[0])

    if incx < 0:
        ix = (-n + 1) * incx

    ix += incx
    for i in range(1, n):
        var current_abs = abs(x[ix])
        if current_abs > max_val:
            max_val = current_abs
            imax = i + 1
        ix += incx

    return imax
