# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Index of Maximum Absolute Value (`level1.iamax`)
============================================

 Provides index of maximum absolute value operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of


def iamax[
    mut: Bool, origin: Origin[mut=mut], //, dtype: DType
](n: Int, x: BLASPtr[dtype, origin], incx: Int) -> Int:
    """
    Find the index of the element with maximum absolute value in vector X.

    Parameters:
        mut: Indicates whether the pointer is mutable (True) or immutable (False).
        origin: Memory origin of the pointer x.
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vector X.
        x: Pointer to the first element of vector X.
        incx: Increment for the elements of X.

    Returns:
        The index (1-based) of the element with maximum absolute value.
        Returns 0 if n <= 0.
    """
    if n <= 0:
        return 0

    if n == 1:
        return 1

    var ix: Int = 0
    var imax: Int = 1
    var max_val: Scalar[dtype] = abs(x[0])

    if incx < 0:
        ix = (-n + 1) * incx

    ix += incx
    for i in range(1, n):
        var current_abs = abs(x[ix])
        if current_abs > max_val:
            max_val = current_abs
            imax = i
        ix += incx

    return imax
