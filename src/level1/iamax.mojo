from src import BLASPtr


fn di_amax[
    dtype: DType
](n: Int32, x: BLASPtr[Scalar[dtype]], incx: Int32) -> Int32:
    """
    Find the index of the element with maximum absolute value in vector X.

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

    var ix: Int32 = 0
    var imax: Int32 = 1
    var max_val: Scalar[dtype] = abs(x[0])

    # Handle negative increments
    if incx < 0:
        ix = (-n + 1) * incx

    # Start from the second element
    ix += incx
    for i in range(1, n):
        var current_abs = abs(x[ix])
        if current_abs > max_val:
            max_val = current_abs
            imax = i + 1  # 1-based indexing
        ix += incx

    return imax
