from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def scal[
    dtype: DType
](n: Int, alpha: Scalar[dtype], dx: BLASPtr[Scalar[dtype]], incx: Int,):
    """
    Scale a vector by a scalar: X := alpha * X.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vector X.
        alpha: Scalar multiplier.
        dx: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
    """
    if n <= 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1:

        @parameter
        def closure[width: Int](i: Int) unified {mut dx, read alpha}:
            dx.store[width=width](i, alpha * dx.load[width=width](i))

        vectorize[simd_width](n, closure)
        return

    var ix: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx

    for i in range(n):
        dx[ix] = alpha * dx[ix]
        ix += incx


def scal[
    dtype: DType, n: Int, alpha: Scalar[dtype], incx: Int
](dx: BLASPtr[Scalar[dtype]]):
    """
    Scale a vector by a scalar: X := alpha * X.

    Parameters:
        dtype: Data type of the elements in vector X.
        n: Number of elements in vector X.
        alpha: Scalar multiplier.
        incx: Increment for the elements of X.

    Args:
        dx: Pointer to the first element of vector X.
    """
    scal[dtype](n, alpha, dx, incx)
