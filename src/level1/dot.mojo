from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of

fn ddot[
    dtype: DType
](
    n: Int,
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
    y: BLASPtr[Scalar[dtype]],
    incy: Int,
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
    var result: Scalar[dtype] = 0

    if n <= 0:
        return result

    var ix: Int = 0
    var iy: Int = 0
    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1 and incy == 1:
        @parameter
        fn closure[width: Int](i: Int) unified {mut result, read x, read y}:
            result += (x.load[width=width](i) * y.load[width=width](i)).reduce_add()
        vectorize[simd_width](n, closure)
        return result

    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for i in range(n):
        result += x[ix] * y[iy]
        ix += incx
        iy += incy

    return result
