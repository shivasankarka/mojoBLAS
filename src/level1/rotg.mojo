# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Givens Rotation Operations (`level1.rotg`)
============================================

Provides Givens rotation operations as defined in the BLAS library standard.
"""

from std.math import sqrt, copysign


def rotg[
    origin_a: MutOrigin,
    origin_b: MutOrigin,
    origin_c: MutOrigin,
    origin_s: MutOrigin,
    //,
    dtype: DType,
](
    a: BLASPtr[dtype, origin_a],
    b: BLASPtr[dtype, origin_b],
    c: BLASPtr[dtype, origin_c],
    s: BLASPtr[dtype, origin_s],
) -> None:
    """
    Construct Givens rotation that eliminates the second component of a 2-vector.

    Given scalars a and b, constructs a Givens rotation such that:
    [  c  s ] [ a ]   [ r ]
    [ -s  c ] [ b ] = [ 0 ]

    Where r = sqrt(a² + b²)

    Parameters:
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b.
        origin_c: Memory origin of the pointer c.
        origin_s: Memory origin of the pointer s.
        dtype: Data type of the scalars a, b, c, and s.

    Args:
        a: Pointer to first scalar (input/output - becomes r on output).
        b: Pointer to second scalar (input/output - becomes z on output).
        c: Pointer to cosine component of rotation (output).
        s: Pointer to sine component of rotation (output).
    """
    var a_val = a[0]
    var b_val = b[0]

    if b_val == 0:
        c[0] = 1.0
        s[0] = 0.0
        a[0] = a_val
        b[0] = 0.0
        return

    if a_val == 0:
        c[0] = 0.0
        s[0] = 1.0
        a[0] = abs(b_val)
        b[0] = 1.0
        return

    var abs_a = abs(a_val)
    var abs_b = abs(b_val)

    if abs_a > abs_b:
        var t = b_val / a_val
        var u = copysign(sqrt(1.0 + t * t), a_val)
        c[0] = 1.0 / u
        s[0] = c[0] * t
        a[0] = a_val * u
        b[0] = s[0]
    else:
        var t = a_val / b_val
        var u = copysign(sqrt(1.0 + t * t), b_val)
        s[0] = 1.0 / u
        c[0] = s[0] * t
        a[0] = b_val * u
        b[0] = 1.0 if a_val != 0 else Scalar[dtype](0.0)
