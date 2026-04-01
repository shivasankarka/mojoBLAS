from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def dot[
    dtype: DType
](
    n: Int,
    dx: BLASPtr[Scalar[dtype]],
    incx: Int,
    dy: BLASPtr[Scalar[dtype]],
    incy: Int,
) -> Scalar[dtype]:
    """
    Compute the dot product of two vectors X and Y.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vectors X and Y.
        dx: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        dy: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.

    Returns:
        The dot product as a scalar value.
    """
    var result: Scalar[dtype] = 0
    # TODO: not sure if returning 0 is the best way to handle n <= 0 case. Check with BLAS spec.
    if n <= 0:
        return result

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1 and incy == 1:

        @parameter
        def closure[width: Int](i: Int) unified {mut result, read dx, read dy}:
            result += (
                dx.load[width=width](i) * dy.load[width=width](i)
            ).reduce_add()

        vectorize[simd_width](n, closure)
        return result

    var ix: Int = 0
    var iy: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for i in range(n):
        result += dx[ix] * dy[iy]
        ix += incx
        iy += incy

    return result


def dot[
    dtype: DType,
    n: Int,
    incx: Int,
    incy: Int,
](dx: BLASPtr[Scalar[dtype]], dy: BLASPtr[Scalar[dtype]],) -> Scalar[dtype]:
    """
    Compute the dot product of two vectors X and Y.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.
        n: Number of elements in vectors X and Y.
        incx: Increment for the elements of X.
        incy: Increment for the elements of Y.

    Args:
        dx: Pointer to the first element of vector X.
        dy: Pointer to the first element of vector Y.

    Returns:
        The dot product as a scalar value.
    """
    return dot[dtype](n, dx, incx, dy, incy)
