# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Absolute Sum Operations (`level1.asum`)
============================================
Provides absolute sum operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def asum[
    mut: Bool,
    origin: Origin[mut=mut],
    //,
    dtype: DType,
](n: Int, dx: BLASPtr[dtype, origin], incx: Int) -> Scalar[dtype]:
    """
    Compute the sum of absolute values of elements in vector X.

    Parameters:
        mut: Indicates whether the pointer is mutable (True) or immutable (False).
        origin: Memory origin of the pointer dx.
        dtype: Data type of the elements in vector X.

    Args:
        n: Number of elements in vector X.
        dx: Pointer to the first element of vector X. Dimension should be at least (1 + (n - 1) * abs(incx)).
        incx: Increment for the elements of X.

    Returns:
        The sum of absolute values as a scalar value.
    """
    var result: Scalar[dtype] = 0.0

    if n <= 0 or incx <= 0:
        return result

    if incx == 1:
        comptime simd_width: Int = simd_width_of[dtype]()

        def closure[width: Int](i: Int) {mut result, read dx}:
            result += abs(dx.load[width=width](i)).reduce_add()

        vectorize[simd_width](n, closure)
        return result

    var nincx: Int = n * incx
    for i in range(0, nincx, incx):
        result += abs(dx[i])
    return result


# NOTE: Using internal complex scalar type. not sure if this is the best way because we can't do vectorization on this UnsafePointer[ComplexScalar[dtype]].
# def cdasum[dtype: DType](
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
