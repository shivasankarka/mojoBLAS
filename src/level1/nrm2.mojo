from std.math import sqrt
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def nrm2[
    dtype: DType
](n: Int, x: BLASPtr[Scalar[dtype]], incx: Int,) -> Scalar[dtype]:
    """
    Compute the Euclidean norm (2-norm) of a vector X.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The Euclidean norm as a scalar value.
    """
    result: Scalar[dtype] = 0
    if n <= 0:
        return result

    var ix: Int = 0
    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1:

        @parameter
        def closure[width: Int](i: Int) unified {mut result, read x}:
            result += (
                x.load[width=width](i) * x.load[width=width](i)
            ).reduce_add()

        vectorize[simd_width](n, closure)
        return sqrt(result)

    if incx < 0:
        ix = (-n + 1) * incx
    for i in range(n):
        result += x[ix] * x[ix]
        ix += incx
    return sqrt(result)
