from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of


fn daxpy[
    dtype: DType
](
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[Scalar[dtype]],
    # x: UnsafePointer[Scalar[dtype], **_],
    incx: Int,
    y: BLASPtr[Scalar[dtype]],
    # mut y: UnsafePointer[Scalar[dtype], **_],
    incy: Int,
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

    var ix: Int = 0
    var iy: Int = 0

    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1 and incy == 1:
        @parameter
        fn closure[width: Int](i: Int) unified {mut y, read x, read alpha}:
            y.store[width=width](i, alpha * x.load[width=width](i) + y.load[width=width](i))
        vectorize[simd_width](n, closure)
        return

    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    # Perform the AXPY operation: y[i] = alpha * x[i] + y[i]
    for i in range(n):
        y[iy] = alpha * x[ix] + y[iy]
        ix += incx
        iy += incy
