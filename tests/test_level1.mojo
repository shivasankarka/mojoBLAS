# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory.alloc import unsafe_alloc
from std.testing import (
    assert_almost_equal,
    assert_true,
    TestSuite,
)

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.level1 import (
    asum,
    axpy,
    copy,
    dot,
    iamax,
    nrm2,
    rot,
    rotg,
    rotm,
    rotmg,
    scal,
    vswap,
)


def test_copy() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 0.0, 0.0, 0.0

    copy(3, x, 1, y, 1)

    assert_true(y[unsafe_offset=0] == 1.0, "copy failed at index 0")
    assert_true(y[unsafe_offset=1] == 2.0, "copy failed at index 1")
    assert_true(y[unsafe_offset=2] == 3.0, "copy failed at index 2")

    x.unsafe_free()
    y.unsafe_free()


def test_copy_with_increment() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](6)
    var y = unsafe_alloc[Scalar[DType.float32]](6)

    x[unsafe_offset=0], x[unsafe_offset=2], x[unsafe_offset=4] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 0.0, 0.0, 0.0

    copy(3, x, 2, y, 2)

    assert_true(y[unsafe_offset=0] == 1.0, "copy with inc failed at index 0")
    assert_true(y[unsafe_offset=2] == 2.0, "copy with inc failed at index 2")
    assert_true(y[unsafe_offset=4] == 3.0, "copy with inc failed at index 4")

    x.unsafe_free()
    y.unsafe_free()


def test_copy_float64() raises:
    var x = unsafe_alloc[Scalar[DType.float64]](3)
    var y = unsafe_alloc[Scalar[DType.float64]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.5, 2.5, 3.5
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 0.0, 0.0, 0.0

    copy(3, x, 1, y, 1)

    assert_almost_equal(y[unsafe_offset=0], Float64(1.5))
    assert_almost_equal(y[unsafe_offset=1], Float64(2.5))
    assert_almost_equal(y[unsafe_offset=2], Float64(3.5))

    x.unsafe_free()
    y.unsafe_free()


def test_scal() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0

    scal(3, Float32(2.0), x, 1)

    assert_true(x[unsafe_offset=0] == 2.0, "scal failed at index 0")
    assert_true(x[unsafe_offset=1] == 4.0, "scal failed at index 1")
    assert_true(x[unsafe_offset=2] == 6.0, "scal failed at index 2")

    x.unsafe_free()


def test_axpy() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 4.0, 5.0, 6.0

    axpy(3, Float32(2.0), x, 1, y, 1)
    # y = 2*x + y → [6, 9, 12]
    assert_true(y[unsafe_offset=0] == 6.0, "axpy failed at index 0")
    assert_true(y[unsafe_offset=1] == 9.0, "axpy failed at index 1")
    assert_true(y[unsafe_offset=2] == 12.0, "axpy failed at index 2")

    x.unsafe_free()
    y.unsafe_free()


def test_dot() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 4.0, 5.0, 6.0

    var result = dot(3, x, 1, y, 1)

    assert_true(result == 32.0, "dot product incorrect")

    x.unsafe_free()
    y.unsafe_free()


def test_dot_with_increment() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](6)
    var y = unsafe_alloc[Scalar[DType.float32]](6)

    x[unsafe_offset=0], x[unsafe_offset=2], x[unsafe_offset=4] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=2], y[unsafe_offset=4] = 4.0, 5.0, 6.0

    var result = dot(3, x, 2, y, 2)

    assert_true(result == 32.0, "dot with increment incorrect")

    x.unsafe_free()
    y.unsafe_free()


def test_dot_float64() raises:
    var x = unsafe_alloc[Scalar[DType.float64]](3)
    var y = unsafe_alloc[Scalar[DType.float64]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 4.0, 5.0, 6.0

    var result = dot(3, x, 1, y, 1)

    assert_almost_equal(result, Float64(32.0))

    x.unsafe_free()
    y.unsafe_free()


def test_dot_orthogonal() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 0.0, 0.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 0.0, 1.0, 0.0

    var result = dot(3, x, 1, y, 1)

    assert_true(result == 0.0, "dot product of orthogonal vectors should be 0")

    x.unsafe_free()
    y.unsafe_free()


def test_nrm2() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 3.0, 4.0, 0.0

    var result = nrm2(3, x, 1)

    assert_true(result == 5.0, "nrm2 incorrect")

    x.unsafe_free()


def test_asum() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](4)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2], x[
        unsafe_offset=3
    ] = (1.0, -2.0, 3.0, -4.0)

    var result = asum(4, x, 1)

    assert_true(result == 10.0, "asum incorrect")

    x.unsafe_free()


def test_asum_negative_increment() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](4)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2], x[
        unsafe_offset=3
    ] = (1.0, -2.0, 3.0, -4.0)

    var result = asum(4, x, -1)

    assert_true(result == 10.0, "asum with negative increment incorrect")

    x.unsafe_free()


def test_axpy_negative_increment() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = (
        10.0,
        20.0,
        30.0,
    )

    # Negative increments are handled from base pointer via internal offset.
    axpy(3, Float32(2.0), x, -1, y, -1)

    assert_true(
        y[unsafe_offset=0] == 12.0
        and y[unsafe_offset=1] == 24.0
        and y[unsafe_offset=2] == 36.0,
        "axpy negative increment incorrect",
    )

    x.unsafe_free()
    y.unsafe_free()


def test_dot_negative_increment() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 4.0, 5.0, 6.0

    var result = dot(3, x, -1, y, -1)
    assert_true(result == 32.0, "dot negative increment incorrect")

    x.unsafe_free()
    y.unsafe_free()


def test_n_le_zero_noop_paths() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](2)
    var y = unsafe_alloc[Scalar[DType.float32]](2)
    x[unsafe_offset=0], x[unsafe_offset=1] = 1.0, 2.0
    y[unsafe_offset=0], y[unsafe_offset=1] = 3.0, 4.0

    copy(0, x, 1, y, 1)
    scal(0, Float32(2.0), x, 1)
    axpy(0, Float32(2.0), x, 1, y, 1)
    vswap(0, x, 1, y, 1)
    rot(0, x, 1, y, 1, Float32(1.0), Float32(0.0))

    assert_true(
        x[unsafe_offset=0] == 1.0 and x[unsafe_offset=1] == 2.0,
        "n<=0 should be no-op for x",
    )
    assert_true(
        y[unsafe_offset=0] == 3.0 and y[unsafe_offset=1] == 4.0,
        "n<=0 should be no-op for y",
    )
    assert_true(dot(0, x, 1, y, 1) == 0.0, "dot n<=0 should return 0")
    assert_true(nrm2(0, x, 1) == 0.0, "nrm2 n<=0 should return 0")
    assert_true(asum(0, x, 1) == 0.0, "asum n<=0 should return 0")
    assert_true(iamax(0, x, 1) == 0, "iamax n<=0 should return 0")

    x.unsafe_free()
    y.unsafe_free()


def test_swap() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](3)
    var y = unsafe_alloc[Scalar[DType.float32]](3)

    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2] = 1.0, 2.0, 3.0
    y[unsafe_offset=0], y[unsafe_offset=1], y[unsafe_offset=2] = 4.0, 5.0, 6.0

    vswap(3, x, 1, y, 1)

    assert_true(
        x[unsafe_offset=0] == 4.0 and y[unsafe_offset=0] == 1.0,
        "swap failed at index 0",
    )
    assert_true(
        x[unsafe_offset=1] == 5.0 and y[unsafe_offset=1] == 2.0,
        "swap failed at index 1",
    )
    assert_true(
        x[unsafe_offset=2] == 6.0 and y[unsafe_offset=2] == 3.0,
        "swap failed at index 2",
    )

    x.unsafe_free()
    y.unsafe_free()


def test_iamax() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](5)
    x[unsafe_offset=0], x[unsafe_offset=1], x[unsafe_offset=2], x[
        unsafe_offset=3
    ], x[unsafe_offset=4] = (1.0, -5.0, 3.0, 2.0, -4.0)

    var result = iamax(5, x, 1)

    print("iamax result:", result)
    assert_true(result == 1, "iamax incorrect index")

    x.unsafe_free()


def test_rotg() raises:
    var a = unsafe_alloc[Scalar[DType.float32]](1)
    var b = unsafe_alloc[Scalar[DType.float32]](1)
    var c = unsafe_alloc[Scalar[DType.float32]](1)
    var s = unsafe_alloc[Scalar[DType.float32]](1)

    a[unsafe_offset=0], b[unsafe_offset=0] = 3.0, 4.0

    rotg(a, b, c, s)

    # r should be 5 (hypotenuse)
    assert_true(a[unsafe_offset=0] == 5.0, "rotg r incorrect")

    # c^2 + s^2 = 1
    var norm = (
        c[unsafe_offset=0] * c[unsafe_offset=0]
        + s[unsafe_offset=0] * s[unsafe_offset=0]
    )
    assert_true(norm == 1.0, "rotg normalization failed")

    a.unsafe_free()
    b.unsafe_free()
    c.unsafe_free()
    s.unsafe_free()


def test_rot() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](2)
    var y = unsafe_alloc[Scalar[DType.float32]](2)

    x[unsafe_offset=0], x[unsafe_offset=1] = 1.0, 2.0
    y[unsafe_offset=0], y[unsafe_offset=1] = 3.0, 4.0

    # 90° rotation: (x, y) -> (y, -x)
    rot(2, x, 1, y, 1, Float32(0.0), Float32(1.0))

    assert_true(
        x[unsafe_offset=0] == 3.0 and y[unsafe_offset=0] == -1.0,
        "rot failed at index 0",
    )
    assert_true(
        x[unsafe_offset=1] == 4.0 and y[unsafe_offset=1] == -2.0,
        "rot failed at index 1",
    )

    x.unsafe_free()
    y.unsafe_free()


def test_rotm() raises:
    var x = unsafe_alloc[Scalar[DType.float32]](2)
    var y = unsafe_alloc[Scalar[DType.float32]](2)
    var p = unsafe_alloc[Scalar[DType.float32]](5)

    x[unsafe_offset=0], x[unsafe_offset=1] = 1.0, 2.0
    y[unsafe_offset=0], y[unsafe_offset=1] = 3.0, 4.0
    p[unsafe_offset=0], p[unsafe_offset=1], p[unsafe_offset=2], p[
        unsafe_offset=3
    ], p[unsafe_offset=4] = (-1.0, 1.0, 2.0, 3.0, 4.0)

    rotm(2, x, 1, y, 1, p)

    assert_almost_equal(x[unsafe_offset=0], Float32(10.0))
    assert_almost_equal(x[unsafe_offset=1], Float32(14.0))
    assert_almost_equal(y[unsafe_offset=0], Float32(14.0))
    assert_almost_equal(y[unsafe_offset=1], Float32(20.0))

    x.unsafe_free()
    y.unsafe_free()
    p.unsafe_free()


def test_rotmg() raises:
    var d1 = unsafe_alloc[Scalar[DType.float32]](1)
    var d2 = unsafe_alloc[Scalar[DType.float32]](1)
    var x1 = unsafe_alloc[Scalar[DType.float32]](1)
    var y1 = unsafe_alloc[Scalar[DType.float32]](1)
    var p = unsafe_alloc[Scalar[DType.float32]](5)

    d1[unsafe_offset=0], d2[unsafe_offset=0], x1[unsafe_offset=0], y1[
        unsafe_offset=0
    ] = (2.0, 3.0, 4.0, 5.0)
    for i in range(5):
        p[unsafe_offset=i] = 0.0

    rotmg(d1, d2, x1, y1, p)

    assert_almost_equal(d1[unsafe_offset=0], Float32(2.1028037))
    assert_almost_equal(d2[unsafe_offset=0], Float32(1.4018692))
    assert_almost_equal(x1[unsafe_offset=0], Float32(7.1333333))
    assert_almost_equal(p[unsafe_offset=0], Float32(1.0))
    assert_almost_equal(p[unsafe_offset=1], Float32(0.5333333))
    assert_almost_equal(p[unsafe_offset=2], Float32(0.0))
    assert_almost_equal(p[unsafe_offset=3], Float32(0.0))
    assert_almost_equal(p[unsafe_offset=4], Float32(0.8))

    d1.unsafe_free()
    d2.unsafe_free()
    x1.unsafe_free()
    y1.unsafe_free()
    p.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# def test_copy() raises:
#     print("Testing copy...")
#     var x = alloc[Scalar[DType.float32]](3)
#     var y = alloc[Scalar[DType.float32]](3)

#     x[0] = 1.0
#     x[1] = 2.0
#     x[2] = 3.0

#     y[0] = 0.0
#     y[1] = 0.0
#     y[2] = 0.0

#     copy(3, x, 1, y, 1)

#     print("x:", x[0], x[1], x[2])
#     print("y:", y[0], y[1], y[2])

#     x.free()
#     y.free()


# def test_scal() raises:
#     print("\nTesting scal...")
#     var x = alloc[Scalar[DType.float32]](3)

#     x[0] = 1.0
#     x[1] = 2.0
#     x[2] = 3.0

#     print("Before scaling:", x[0], x[1], x[2])
#     scal(3, Float32(2.0), x, 1)
#     print("After scaling by 2:", x[0], x[1], x[2])

#     x.free()


# def test_axpy() raises:
#     print("\nTesting axpy...")
#     var x = alloc[Scalar[DType.float32]](3)
#     var y = alloc[Scalar[DType.float32]](3)

#     x[0] = 1.0
#     x[1] = 2.0
#     x[2] = 3.0

#     y[0] = 4.0
#     y[1] = 5.0
#     y[2] = 6.0

#     print("Before axpy - x:", x[0], x[1], x[2])
#     print("Before axpy - y:", y[0], y[1], y[2])

#     axpy(3, Float32(2.0), x, 1, y, 1)

#     print("After y := 2*x + y - y:", y[0], y[1], y[2])

#     x.free()
#     y.free()


# def test_dot() raises:
#     print("\nTesting dot...")
#     var x = alloc[Scalar[DType.float32]](3)
#     var y = alloc[Scalar[DType.float32]](3)

#     x[0] = 1.0
#     x[1] = 2.0
#     x[2] = 3.0

#     y[0] = 4.0
#     y[1] = 5.0
#     y[2] = 6.0

#     var result = dot(3, x, 1, y, 1)
#     print("Dot product:", result, "(expected: 32)")

#     x.free()
#     y.free()


# def test_nrm2() raises:
#     print("\nTesting nrm2...")
#     var x = alloc[Scalar[DType.float32]](3)

#     x[0] = 3.0
#     x[1] = 4.0
#     x[2] = 0.0

#     var result = nrm2(3, x, 1)
#     print("Euclidean norm:", result, "(expected: 5)")

#     x.free()


# def test_asum() raises:
#     print("\nTesting asum...")
#     var x = alloc[Scalar[DType.float32]](4)

#     x[0] = 1.0
#     x[1] = -2.0
#     x[2] = 3.0
#     x[3] = -4.0

#     var result = asum(4, x, 1)
#     print("Sum of absolute values:", result, "(expected: 10)")

#     x.free()


# def test_swap() raises:
#     print("\nTesting swap...")
#     var x = alloc[Scalar[DType.float32]](3)
#     var y = alloc[Scalar[DType.float32]](3)

#     x[0] = 1.0
#     x[1] = 2.0
#     x[2] = 3.0

#     y[0] = 4.0
#     y[1] = 5.0
#     y[2] = 6.0

#     print("Before swap - x:", x[0], x[1], x[2])
#     print("Before swap - y:", y[0], y[1], y[2])

#     vswap(3, x, 1, y, 1)

#     print("After swap - x:", x[0], x[1], x[2])
#     print("After swap - y:", y[0], y[1], y[2])

#     x.free()
#     y.free()


# def test_iamax() raises:
#     print("\nTesting iamax...")
#     var x = alloc[Scalar[DType.float32]](5)

#     x[0] = 1.0
#     x[1] = -5.0
#     x[2] = 3.0
#     x[3] = 2.0
#     x[4] = -4.0

#     var result = iamax(5, x, 1)
#     print(
#         "Index of max absolute value:", result, "(expected: 2 for value -5.0)"
#     )

#     x.free()


# def test_rotg() raises:
#     print("\nTesting rotg...")
#     var a = alloc[Scalar[DType.float32]](1)
#     var b = alloc[Scalar[DType.float32]](1)
#     var c = alloc[Scalar[DType.float32]](1)
#     var s = alloc[Scalar[DType.float32]](1)

#     a[0] = 3.0
#     b[0] = 4.0

#     print("Before rotg - a:", a[0], "b:", b[0])

#     rotg(a, b, c, s)

#     print("After rotg - r:", a[0], "z:", b[0])
#     print("cos:", c[0], "sin:", s[0])

#     a.free()
#     b.free()
#     c.free()
#     s.free()


# def test_rot() raises:
#     print("\nTesting rot...")
#     var x = alloc[Scalar[DType.float32]](2)
#     var y = alloc[Scalar[DType.float32]](2)

#     x[0] = 1.0
#     x[1] = 2.0

#     y[0] = 3.0
#     y[1] = 4.0

#     print("Before rotation - x:", x[0], x[1])
#     print("Before rotation - y:", y[0], y[1])

#     # Apply 90 degree rotation (c=0, s=1)
#     rot(2, x, 1, y, 1, Float32(0.0), Float32(1.0))

#     print("After 90° rotation - x:", x[0], x[1])
#     print("After 90° rotation - y:", y[0], y[1])

#     x.free()
#     y.free()


# def main() raises:
#     TestSuite.discover_tests[__functions_in_module()]().run()
