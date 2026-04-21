from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def dot[
    mut_x: Bool,
    mut_y: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: Origin[mut=mut_y],
    //,
    dtype: DType,
](
    n: Int,
    dx: BLASPtr[dtype, origin_x],
    incx: Int,
    dy: BLASPtr[dtype, origin_y],
    incy: Int,
) -> Scalar[dtype]:
    """
    Compute the dot product of two vectors X and Y.

    Parameters:
        mut_x: Indicates whether the pointer to vector X is mutable (True) or immutable (False).
        mut_y: Indicates whether the pointer to vector Y is mutable (True) or immutable (False).
        origin_x: Memory origin of the pointer to vector X.
        origin_y: Memory origin of the pointer to vector Y.
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

    for _ in range(n):
        result += dx[ix] * dy[iy]
        ix += incx
        iy += incy

    return result
