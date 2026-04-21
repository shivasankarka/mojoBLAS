from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


# Named `vwap` to avoid conflict with `swap` in std library.
def vswap[
    origin_x: MutOrigin, origin_y: MutOrigin, //, dtype: DType
](
    n: Int,
    dx: BLASPtr[dtype, origin_x],
    incx: Int,
    dy: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Swap the elements of two vectors X and Y: X <-> Y.

    Parameters:
        origin_x: Memory origin of the pointer dx.
        origin_y: Memory origin of the pointer dy.
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vectors X and Y.
        dx: Pointer to the first element of vector X.
        incx: Increment for the elements of X.
        dy: Pointer to the first element of vector Y.
        incy: Increment for the elements of Y.
    """
    if n <= 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1 and incy == 1:

        @parameter
        def closure[width: Int](i: Int) unified {mut dx, mut dy}:
            var temp = dx.load[width=width](i)
            dx.store[width=width](i, dy.load[width=width](i))
            dy.store[width=width](i, temp)

        vectorize[simd_width](n, closure)
        return

    var ix: Int = 0
    var iy: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for _ in range(n):
        temp = dx[ix]
        dx[ix] = dy[iy]
        dy[iy] = temp
        ix += incx
        iy += incy
