from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of

fn dswap[
    dtype: DType
](
    n: Int,
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
    y: BLASPtr[Scalar[dtype]],
    incy: Int,
) -> None:
    """
    Swap the elements of two vectors X and Y: X <-> Y.

    Args:
        n: Number of elements in vectors X and Y.
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
        fn closure[width: Int](i: Int) unified {mut x, mut y}:
            var temp = x.load[width=width](i)
            x.store[width=width](i, y.load[width=width](i))
            y.store[width=width](i, temp)
        vectorize[simd_width](n, closure)
        return

    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for i in range(n):
        temp: Scalar[dtype] = x[ix]
        x[ix] = y[iy]
        y[iy] = temp
        ix += incx
        iy += incy
