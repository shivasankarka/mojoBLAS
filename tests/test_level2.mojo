from src.level2 import *
from testing import assert_almost_equal, assert_equal, TestSuite

# TODO: Improve the testing framework to better handle memory allocation and deallocation.

def test_gemv_no_transpose():
    print("Testing gemv (no transpose)...")
    var m = 2
    var n = 3
    var lda = m

    var a = alloc[Scalar[DType.float32]](m * n)
    var x = alloc[Scalar[DType.float32]](n)
    var y = alloc[Scalar[DType.float32]](m)

    a[0] = 1.0
    a[1] = 4.0
    a[2] = 2.0
    a[3] = 5.0
    a[4] = 3.0
    a[5] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0

    y[0] = 0.0
    y[1] = 0.0

    dgemv("N", m, n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(6.0))
    assert_almost_equal(y[1], Float32(15.0))

    a.free()
    x.free()
    y.free()


def test_gemv_transpose():
    print("Testing gemv (transpose)...")
    var m = 2
    var n = 3
    var lda = m

    var a = alloc[Scalar[DType.float32]](m * n)
    var x = alloc[Scalar[DType.float32]](m)
    var y = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 4.0
    a[2] = 2.0
    a[3] = 5.0
    a[4] = 3.0
    a[5] = 6.0

    x[0] = 1.0
    x[1] = 2.0

    y[0] = 0.0
    y[1] = 0.0
    y[2] = 0.0

    dgemv("T", m, n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(9.0))
    assert_almost_equal(y[1], Float32(12.0))
    assert_almost_equal(y[2], Float32(15.0))

    a.free()
    x.free()
    y.free()


def test_gemv_with_beta():
    print("Testing gemv (beta accumulation)...")
    var m = 2
    var n = 2
    var lda = m

    var a = alloc[Scalar[DType.float32]](m * n)
    var x = alloc[Scalar[DType.float32]](n)
    var y = alloc[Scalar[DType.float32]](m)

    a[0] = 1.0
    a[1] = 3.0
    a[2] = 2.0
    a[3] = 4.0

    x[0] = 1.0
    x[1] = 1.0

    y[0] = 1.0
    y[1] = 1.0

    dgemv("N", m, n, Float32(1.0), a, lda, x, 1, Float32(1.0), y, 1)

    assert_almost_equal(y[0], Float32(4.0))
    assert_almost_equal(y[1], Float32(8.0))

    a.free()
    x.free()
    y.free()


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
