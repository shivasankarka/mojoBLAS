from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of

# need to fix this for complex types. Currently only works for real types.
fn dasum[
    dtype: DType
](n: Int, x: BLASPtr[Scalar[dtype]], incx: Int) -> Scalar[dtype]:
    """
    Compute the sum of absolute values of elements in vector X.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The sum of absolute values as a scalar value.
    """
    if n <= 0 or incx == 0:
        return 0

    var sum: Scalar[dtype] = 0.0
    var ix: Int = 0
    comptime simd_width: Int = simd_width_of[dtype]()

    if incx == 1:
        @parameter
        fn closure[width: Int](i: Int) unified {mut sum, read x}:
            sum += abs(x.load[width=width](i)).reduce_add()
        vectorize[simd_width](n, closure)
        return sum

    if incx < 0:
        ix = (-n + 1) * incx

    for i in range(n):
        sum += abs(x[ix])
        ix += incx

    return sum
