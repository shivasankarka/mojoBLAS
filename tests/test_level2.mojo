# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory.alloc import unsafe_alloc
from std.testing import (
    assert_almost_equal,
    TestSuite,
)

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.level2 import (
    gemv,
    symv,
    trmv,
)


def test_gemv_no_transpose() raises:
    print("Testing gemv (no transpose)...")
    var m = 2
    var n = 3
    var lda = m

    var a = unsafe_alloc[Scalar[DType.float32]](m * n)
    var x = unsafe_alloc[Scalar[DType.float32]](n)
    var y = unsafe_alloc[Scalar[DType.float32]](m)

    a[unsafe_offset=0] = 1.0
    a[unsafe_offset=1] = 4.0
    a[unsafe_offset=2] = 2.0
    a[unsafe_offset=3] = 5.0
    a[unsafe_offset=4] = 3.0
    a[unsafe_offset=5] = 6.0

    x[unsafe_offset=0] = 1.0
    x[unsafe_offset=1] = 1.0
    x[unsafe_offset=2] = 1.0

    y[unsafe_offset=0] = 0.0
    y[unsafe_offset=1] = 0.0

    gemv("N", m, n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[unsafe_offset=0], Float32(6.0))
    assert_almost_equal(y[unsafe_offset=1], Float32(15.0))

    a.unsafe_free()
    x.unsafe_free()
    y.unsafe_free()


def test_gemv_transpose() raises:
    print("Testing gemv (transpose)...")
    var m = 2
    var n = 3
    var lda = m

    var a = unsafe_alloc[Scalar[DType.float32]](m * n)
    var x = unsafe_alloc[Scalar[DType.float32]](m)
    var y = unsafe_alloc[Scalar[DType.float32]](n)

    a[unsafe_offset=0] = 1.0
    a[unsafe_offset=1] = 4.0
    a[unsafe_offset=2] = 2.0
    a[unsafe_offset=3] = 5.0
    a[unsafe_offset=4] = 3.0
    a[unsafe_offset=5] = 6.0

    x[unsafe_offset=0] = 1.0
    x[unsafe_offset=1] = 2.0

    y[unsafe_offset=0] = 0.0
    y[unsafe_offset=1] = 0.0
    y[unsafe_offset=2] = 0.0

    gemv("T", m, n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[unsafe_offset=0], Float32(9.0))
    assert_almost_equal(y[unsafe_offset=1], Float32(12.0))
    assert_almost_equal(y[unsafe_offset=2], Float32(15.0))

    a.unsafe_free()
    x.unsafe_free()
    y.unsafe_free()


def test_gemv_with_beta() raises:
    print("Testing gemv (beta accumulation)...")
    var m = 2
    var n = 2
    var lda = m

    var a = unsafe_alloc[Scalar[DType.float32]](m * n)
    var x = unsafe_alloc[Scalar[DType.float32]](n)
    var y = unsafe_alloc[Scalar[DType.float32]](m)

    a[unsafe_offset=0] = 1.0
    a[unsafe_offset=1] = 3.0
    a[unsafe_offset=2] = 2.0
    a[unsafe_offset=3] = 4.0

    x[unsafe_offset=0] = 1.0
    x[unsafe_offset=1] = 1.0

    y[unsafe_offset=0] = 1.0
    y[unsafe_offset=1] = 1.0

    gemv("N", m, n, Float32(1.0), a, lda, x, 1, Float32(1.0), y, 1)

    assert_almost_equal(y[unsafe_offset=0], Float32(4.0))
    assert_almost_equal(y[unsafe_offset=1], Float32(8.0))

    a.unsafe_free()
    x.unsafe_free()
    y.unsafe_free()


def test_trmv_upper() raises:
    print("Testing trmv (upper)...")
    var n = 3
    var lda = n

    var a = unsafe_alloc[Scalar[DType.float32]](n * n)
    var x = unsafe_alloc[Scalar[DType.float32]](n)

    a[unsafe_offset=0] = 1.0
    a[unsafe_offset=1] = 0.0
    a[unsafe_offset=2] = 0.0
    a[unsafe_offset=3] = 2.0
    a[unsafe_offset=4] = 4.0
    a[unsafe_offset=5] = 0.0
    a[unsafe_offset=6] = 3.0
    a[unsafe_offset=7] = 5.0
    a[unsafe_offset=8] = 6.0

    x[unsafe_offset=0] = 1.0
    x[unsafe_offset=1] = 1.0
    x[unsafe_offset=2] = 1.0

    trmv("U", "N", "N", n, a, lda, x, 1)

    assert_almost_equal(x[unsafe_offset=0], Float32(6.0))
    assert_almost_equal(x[unsafe_offset=1], Float32(9.0))
    assert_almost_equal(x[unsafe_offset=2], Float32(6.0))

    a.unsafe_free()
    x.unsafe_free()


def test_symv_lower() raises:
    print("Testing symv (lower)...")
    var n = 2
    var lda = n

    var a = unsafe_alloc[Scalar[DType.float32]](n * n)
    var x = unsafe_alloc[Scalar[DType.float32]](n)
    var y = unsafe_alloc[Scalar[DType.float32]](n)

    a[unsafe_offset=0] = 1.0
    a[unsafe_offset=1] = 2.0
    a[unsafe_offset=2] = 2.0
    a[unsafe_offset=3] = 3.0

    x[unsafe_offset=0] = 1.0
    x[unsafe_offset=1] = 1.0

    y[unsafe_offset=0] = 0.0
    y[unsafe_offset=1] = 0.0

    symv("L", n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[unsafe_offset=0], Float32(3.0))
    assert_almost_equal(y[unsafe_offset=1], Float32(5.0))

    a.unsafe_free()
    x.unsafe_free()
    y.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
