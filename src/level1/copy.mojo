from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of
from std.memory import memcpy


def copy[
    mut: Bool,
    origin_x: Origin[mut=mut],
    origin_y: MutOrigin,
    //,
    dtype: DType
](
    n: Int,
    dx: BLASPtr[dtype, origin_x],
    incx: Int,
    dy: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Copy a vector X to vector Y: Y := X.

    Parameters:
        mut: Indicates whether the pointer to vector X is mutable (True) or immutable (False).
        origin_x: Memory origin of the pointer to vector X.
        origin_y: Memory origin of the pointer to vector Y.
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vectors X and Y.
        dx: Pointer to the first element of vector X. dimension should be at least (1 + (n - 1) * abs(incx)).
        incx: Increment for the elements of X.
        dy: Pointer to the first element of vector Y. dimension should be at least (1 + (n - 1) * abs(incy)).
        incy: Increment for the elements of Y.
    """
    if n <= 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1 and incy == 1:
        memcpy(dest=dy, src=dx, count=n)
        # @parameter
        # def closure[width: Int](i: Int) unified {mut y, read x}:
        #     y.store[width=width](i, x.load[width=width](i))

        # vectorize[simd_width](n, closure)
        # return

    var ix: Int = 0
    var iy: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for i in range(n):
        dy[iy] = dx[ix]
        ix += incx
        iy += incy
