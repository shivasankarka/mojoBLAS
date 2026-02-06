from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of

fn drot[
    dtype: DType
](
    n: Int,
    x: BLASPtr[Scalar[dtype]],
    incx: Int,
    y: BLASPtr[Scalar[dtype]],
    incy: Int,
    c: Scalar[dtype],
    s: Scalar[dtype],
) -> None:
    """
    Apply Givens rotation to vectors X and Y.

    Performs the transformation:
    [ x[i] ]   [  c  s ] [ x[i] ]
    [ y[i] ] = [ -s  c ] [ y[i] ]

    Args:
        n: Number of elements in vectors X and Y.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        y: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.
        c: Cosine component of the rotation.
        s: Sine component of the rotation.
    """
    if n <= 0:
        return

    var ix: Int32 = 0
    var iy: Int32 = 0
    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1 and incy == 1:
        @parameter
        fn closure[width: Int](i: Int) unified {mut y, read x, read c, read s}:
            var temp_x = c * x.load[width=width](i) + s * y.load[width=width](i)
            var temp_y = -s * x.load[width=width](i) + c * y.load[width=width](i)
            y.store[width=width](i, temp_y)
            x.store[width=width](i, temp_x)
        vectorize[simd_width](n, closure)
        return

    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for i in range(n):
        var temp_x = c * x[ix] + s * y[iy]
        var temp_y = -s * x[ix] + c * y[iy]
        x[ix] = temp_x
        y[iy] = temp_y
        ix += incx
        iy += incy
