from src import BLASPtr

from algorithm.functional import vectorize
from sys.info import simd_width_of


fn dasum[
    dtype: DType
](out dasum: Scalar[dtype], n: Int, dx: BLASPtr[Scalar[dtype]], incx: Int):
    """
    Compute the sum of absolute values of elements in vector X.

    Args:
        n: Number of elements in vector X.
        dx: Pointer to the first element of vector X. Dimension should be at least (1 + (n - 1) * abs(incx)).
        incx: Increment for the elements of X.

    Returns:
        The sum of absolute values as a scalar value.
    """
    comptime simd_width: Int = simd_width_of[dtype]()
    dasum: Scalar[dtype] = 0.0

    if n <= 0 or incx <= 0:
        return

    if incx == 1:

        @parameter
        fn closure[width: Int](i: Int) unified {mut dasum, read dx}:
            dasum += abs(dx.load[width=width](i)).reduce_add()

        vectorize[simd_width](n, closure)
        return

    var nincx: Int = n * incx
    for i in range(0, nincx, incx):
        dasum += abs(dx[i])
    return


# NOTE: Using internal complex scalar type. not sure if this is the best way because we can't do vectorization on this UnsafePointer[ComplexScalar[dtype]].
# fn cdasum[dtype: DType](
#     out dasum: Scalar[dtype],
#     n: Int,
#     dx: BLASPtr[ComplexScalar[dtype]],
#     incx: Int
# ):
#     """
#     Compute the sum of absolute values of elements in complex vector X.

#     Args:
#         n: Number of elements in vector X.
#         dx: Pointer to the first element of vector X. Dimension should be at least (1 + (n - 1) * abs(incx)).
#         incx: Increment for the elements of X.

#     Returns:
#         The sum of absolute values as a scalar value.
#     """
#     comptime simd_width: Int = simd_width_of[dtype]()
#     dasum: Scalar[dtype] = 0.0

#     if n <= 0 or incx <= 0:
#         return

#     if incx == 1:
#         for i in range(n):
#             dasum += cmplx_abs(dx[i])
#         return

#     var nincx: Int = n * incx
#     for i in range(0, nincx, incx):
#         dasum += cmplx_abs(dx[i])
#     return
