from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of

fn dscal[
    dtype: DType
](
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
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

    var ix: Int = 0
    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1:
        @parameter
        fn closure[width: Int](i: Int) unified {mut x, read alpha}:
            x.store[width=width](i, alpha * x.load[width=width](i))
        vectorize[simd_width](n, closure)
        return

    if incx < 0:
        ix = (-n + 1) * incx

    for i in range(n):
        x[ix] = alpha * x[ix]
        ix += incx
