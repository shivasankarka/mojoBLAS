# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
General Rank-1 Update (`level2.ger`)
===========================================

Provides general rank-1 update operations as defined in the BLAS library standard.
"""


def ger[
    mut_x: Bool,
    mut_y: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: Origin[mut=mut_y],
    origin_a: MutOrigin,
    //,
    dtype: DType,
](
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    y: BLASPtr[dtype, origin_y],
    incy: Int,
    a: BLASPtr[dtype, origin_a],
    lda: Int,
):
    """
    Performs the rank-1 operation A := alpha*x*y^T + A,
    where A is an m by n matrix, x is an m-element vector,
    and y is an n-element vector.

    Parameters:
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        mut_y: Indicates whether the pointer y is mutable (True) or immutable (False).
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y.
        origin_a: Memory origin of the pointer a (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        m: The number of rows of the matrix A.
        n: The number of columns of the matrix A.
        alpha: The scalar multiplier.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
        y: A pointer to the first element of the vector y.
        incy: The increment for the elements of y.
        a: A pointer to the first element of the matrix A (input/output).
        lda: The leading dimension of the matrix A.
    """
    var info: Int = 0
    if m < 0:
        info = 2
    elif n < 0:
        info = 3
    elif lda < max(1, m):
        info = 6
    elif incx == 0:
        info = 7
    elif incy == 0:
        info = 9

    if info != 0:
        print("ger: Info", info)
        return

    if m == 0 or n == 0 or alpha == 0:
        return

    var kx: Int = 1
    var ky: Int = 1
    if incx < 0:
        kx = 1 - (m - 1) * incx
    if incy < 0:
        ky = 1 - (n - 1) * incy

    if incy == 1:
        for j in range(n):
            if y[j] != 0:
                var temp: Scalar[dtype] = alpha * y[j]
                for i in range(m):
                    a[i + j * lda] = a[i + j * lda] + x[i] * temp
    else:
        var jy: Int = ky
        for j in range(n):
            if y[jy - 1] != 0:
                var temp: Scalar[dtype] = alpha * y[jy - 1]
                var ix: Int = kx
                for i in range(m):
                    a[i + j * lda] = a[i + j * lda] + x[ix - 1] * temp
                    ix += incx
            jy += incy

    return
