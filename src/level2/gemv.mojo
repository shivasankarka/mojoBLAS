# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
General Matrix-Vector Operations (`level2.gemv`)
=============================================

Provides general matrix-vector operations as defined in the BLAS library standard.

This module implements the gemv operation for matrix-vector multiplication
with support for various transpositions and scaling factors.
"""

from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of

# NOTE: I think we could so much optimization here if we promote a lot of these arguments into
# compile time params. Especially will help with all these if branches.


def gemv[
    mut_a: Bool,
    mut_x: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](
    trans: String,
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    x: BLASPtr[dtype, origin_x],
    incx: Int,
    beta: Scalar[dtype],
    y: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Performs the matrix-vector operation y := alpha*A*x + beta*y,
    where alpha and beta are scalars, x and y are vectors and A is an m by n matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_x: Indicates whether the pointer x is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_x: Memory origin of the pointer x.
        origin_y: Memory origin of the pointer y (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        trans: A string indicating whether to transpose the matrix A ('N' for no transpose, 'T' for transpose).
        m: The number of rows of the matrix A.
        n: The number of columns of the matrix A.
        alpha: The scalar multiplier for the matrix-vector product.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
        x: A pointer to the first element of the vector x.
        incx: The increment for the elements of x.
        beta: The scalar multiplier for the vector y.
        y: A pointer to the first element of the vector y (input/output).
        incy: The increment for the elements of y.
    """
    var info: Int = 0
    if trans != "N" and trans != "T" and trans != "C":
        info = 1
    elif m < 0:
        info = 2
    elif n < 0:
        info = 3
    elif lda < max(1, m):
        info = 6
    elif incx == 0:
        info = 8
    elif incy == 0:
        info = 11
    if info != 0:
        print("gemv: Info", info)
        return

    if m == 0 or n == 0 or (alpha == 0 and beta == 1):
        return

    var lenx: Int
    var leny: Int

    if trans == "N":
        lenx = n
        leny = m
    else:
        lenx = m
        leny = n

    var kx: Int = 1
    var ky: Int = 1
    if incx < 0:
        kx = 1 - (lenx - 1) * incx
    if incy < 0:
        ky = 1 - (leny - 1) * incy

    comptime simd_width: Int = simd_width_of[dtype]()
    if beta != 1:
        if incy == 1:
            if beta == 0:
                for i in range(leny):
                    y[i] = 0
            else:

                @parameter
                def closure[width: Int](i: Int) unified {mut y, read beta}:
                    y.store[width=width](i, beta * y.load[width=width](i))

                vectorize[simd_width](leny, closure)
        else:
            var iy: Int = ky
            if beta == 0:
                for _ in range(leny):
                    y[iy - 1] = 0
                    iy += incy
            else:
                for _ in range(leny):
                    y[iy - 1] = beta * y[iy - 1]
                    iy += incy
    if alpha == 0:
        return

    # NOTE: might be parallizable
    if trans == "N":
        var jx: Int = kx
        if incy == 1:
            for j in range(n):
                var temp: Scalar[dtype] = alpha * x[jx - 1]
                for i in range(m):
                    y[i] = y[i] + temp * a[i + j * lda]
                jx += incx
        else:
            for j in range(n):
                var temp: Scalar[dtype] = alpha * x[jx - 1]
                var iy: Int = ky
                for i in range(m):
                    y[iy - 1] = y[iy - 1] + temp * a[i + j * lda]
                    iy += incy
                jx += incx
    else:
        var jy: Int = ky
        if incx == 1:
            for j in range(n):
                var temp: Scalar[dtype] = 0
                for i in range(m):
                    temp = temp + a[i + j * lda] * x[i]
                y[jy - 1] = y[jy - 1] + alpha * temp
                jy += incy
        else:
            for j in range(n):
                var temp: Scalar[dtype] = 0
                var ix: Int = kx
                for i in range(m):
                    temp = temp + a[i + j * lda] * x[ix - 1]
                    ix += incx
                y[jy - 1] = y[jy - 1] + alpha * temp
                jy += incy

    return
