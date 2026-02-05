from src import BLASPtr
from math import sqrt, copysign


fn drotg[
    dtype: DType
](
    a: BLASPtr[Scalar[dtype]],
    b: BLASPtr[Scalar[dtype]],
    c: BLASPtr[Scalar[dtype]],
    s: BLASPtr[Scalar[dtype]],
) -> None:
    """
    Construct Givens rotation that eliminates the second component of a 2-vector.

    Given scalars a and b, constructs a Givens rotation such that:
    [  c  s ] [ a ]   [ r ]
    [ -s  c ] [ b ] = [ 0 ]

    Where r = sqrt(a² + b²)

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

    # Use the larger absolute value for numerical stability
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
