from src.level1 import *
from memory import UnsafePointer

from testing import TestSuite


def test_copy():
    print("Testing copy...")
    var x = alloc[Scalar[DType.float32]](3)
    var y = alloc[Scalar[DType.float32]](3)

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    y[0] = 0.0
    y[1] = 0.0
    y[2] = 0.0

    try:
        dcopy(3, x, 1, y, 1)
    except:
        print("Error in copy")

    print("x:", x[0], x[1], x[2])
    print("y:", y[0], y[1], y[2])

    x.free()
    y.free()


def test_scal():
    print("\nTesting scal...")
    var x = alloc[Scalar[DType.float32]](3)

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    print("Before scaling:", x[0], x[1], x[2])
    dscal(3, Float32(2.0), x, 1)
    print("After scaling by 2:", x[0], x[1], x[2])

    x.free()


def test_axpy():
    print("\nTesting axpy...")
    var x = alloc[Scalar[DType.float32]](3)
    var y = alloc[Scalar[DType.float32]](3)

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    y[0] = 4.0
    y[1] = 5.0
    y[2] = 6.0

    print("Before axpy - x:", x[0], x[1], x[2])
    print("Before axpy - y:", y[0], y[1], y[2])

    daxpy(3, Float32(2.0), x, 1, y, 1)

    print("After y := 2*x + y - y:", y[0], y[1], y[2])

    x.free()
    y.free()


def test_dot():
    print("\nTesting dot...")
    var x = alloc[Scalar[DType.float32]](3)
    var y = alloc[Scalar[DType.float32]](3)

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    y[0] = 4.0
    y[1] = 5.0
    y[2] = 6.0

    var result = ddot(3, x, 1, y, 1)
    print("Dot product:", result, "(expected: 32)")

    x.free()
    y.free()


def test_nrm2():
    print("\nTesting nrm2...")
    var x = alloc[Scalar[DType.float32]](3)

    x[0] = 3.0
    x[1] = 4.0
    x[2] = 0.0

    var result = dnrm2(3, x, 1)
    print("Euclidean norm:", result, "(expected: 5)")

    x.free()


def test_asum():
    print("\nTesting asum...")
    var x = alloc[Scalar[DType.float32]](4)

    x[0] = 1.0
    x[1] = -2.0
    x[2] = 3.0
    x[3] = -4.0

    var result = dasum(4, x, 1)
    print("Sum of absolute values:", result, "(expected: 10)")

    x.free()


def test_swap():
    print("\nTesting swap...")
    var x = alloc[Scalar[DType.float32]](3)
    var y = alloc[Scalar[DType.float32]](3)

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    y[0] = 4.0
    y[1] = 5.0
    y[2] = 6.0

    print("Before swap - x:", x[0], x[1], x[2])
    print("Before swap - y:", y[0], y[1], y[2])

    dswap(3, x, 1, y, 1)

    print("After swap - x:", x[0], x[1], x[2])
    print("After swap - y:", y[0], y[1], y[2])

    x.free()
    y.free()


def test_iamax():
    print("\nTesting iamax...")
    var x = alloc[Scalar[DType.float32]](5)

    x[0] = 1.0
    x[1] = -5.0
    x[2] = 3.0
    x[3] = 2.0
    x[4] = -4.0

    var result = di_amax(5, x, 1)
    print(
        "Index of max absolute value:", result, "(expected: 2 for value -5.0)"
    )

    x.free()


def test_rotg():
    print("\nTesting rotg...")
    var a = alloc[Scalar[DType.float32]](1)
    var b = alloc[Scalar[DType.float32]](1)
    var c = alloc[Scalar[DType.float32]](1)
    var s = alloc[Scalar[DType.float32]](1)

    a[0] = 3.0
    b[0] = 4.0

    print("Before rotg - a:", a[0], "b:", b[0])

    drotg(a, b, c, s)

    print("After rotg - r:", a[0], "z:", b[0])
    print("cos:", c[0], "sin:", s[0])

    a.free()
    b.free()
    c.free()
    s.free()


def test_rot():
    print("\nTesting rot...")
    var x = alloc[Scalar[DType.float32]](2)
    var y = alloc[Scalar[DType.float32]](2)

    x[0] = 1.0
    x[1] = 2.0

    y[0] = 3.0
    y[1] = 4.0

    print("Before rotation - x:", x[0], x[1])
    print("Before rotation - y:", y[0], y[1])

    # Apply 90 degree rotation (c=0, s=1)
    drot(2, x, 1, y, 1, Float32(0.0), Float32(1.0))

    print("After 90° rotation - x:", x[0], x[1])
    print("After 90° rotation - y:", y[0], y[1])

    x.free()
    y.free()


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
