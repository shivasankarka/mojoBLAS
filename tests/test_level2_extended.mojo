from std.testing import assert_almost_equal, assert_equal, TestSuite

from src.level2 import gemv, trmv, trsv, symv, syr, syr2


def test_trmv_upper_no_transpose() raises:
    print("Testing trmv (upper, no transpose)...")
    var n = 3
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 0.0
    a[3] = 2.0
    a[4] = 4.0
    a[5] = 0.0
    a[6] = 3.0
    a[7] = 5.0
    a[8] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0

    trmv("U", "N", "N", n, a, lda, x, 1)

    assert_almost_equal(x[0], Float32(6.0))
    assert_almost_equal(x[1], Float32(9.0))
    assert_almost_equal(x[2], Float32(6.0))

    a.free()
    x.free()


def test_trmv_lower_no_transpose() raises:
    print("Testing trmv (lower, no transpose)...")
    var n = 3
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 4.0
    a[3] = 0.0
    a[4] = 3.0
    a[5] = 5.0
    a[6] = 0.0
    a[7] = 0.0
    a[8] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0

    trmv("L", "N", "N", n, a, lda, x, 1)

    assert_almost_equal(x[0], Float32(1.0))
    assert_almost_equal(x[1], Float32(5.0))
    assert_almost_equal(x[2], Float32(15.0))

    a.free()
    x.free()


def test_trsv_upper_no_transpose() raises:
    print("Testing trsv (upper, no transpose)...")
    var n = 3
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 0.0
    a[3] = 2.0
    a[4] = 4.0
    a[5] = 0.0
    a[6] = 3.0
    a[7] = 5.0
    a[8] = 6.0

    x[0] = 11.0
    x[1] = 18.0
    x[2] = 12.0

    trsv("U", "N", "N", n, a, lda, x, 1)

    assert_almost_equal(x[0], Float32(1.0))
    assert_almost_equal(x[1], Float32(2.0))
    assert_almost_equal(x[2], Float32(2.0))

    a.free()
    x.free()


def test_symv_upper() raises:
    print("Testing symv (upper)...")
    var n = 3
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)
    var y = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 2.0
    a[4] = 4.0
    a[5] = 5.0
    a[6] = 3.0
    a[7] = 5.0
    a[8] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0
    y[0] = 0.0
    y[1] = 0.0
    y[2] = 0.0

    symv("U", n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(6.0))
    assert_almost_equal(y[1], Float32(11.0))
    assert_almost_equal(y[2], Float32(14.0))

    a.free()
    x.free()
    y.free()


def test_syr_upper() raises:
    print("Testing syr (upper)...")
    var n = 3
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 0.0
    a[3] = 2.0
    a[4] = 4.0
    a[5] = 0.0
    a[6] = 3.0
    a[7] = 5.0
    a[8] = 6.0

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    syr("U", n, Float32(1.0), x, 1, a, lda)

    assert_almost_equal(a[0], Float32(2.0))
    assert_almost_equal(a[3], Float32(4.0))
    assert_almost_equal(a[6], Float32(6.0))
    assert_almost_equal(a[4], Float32(8.0))
    assert_almost_equal(a[7], Float32(11.0))
    assert_almost_equal(a[8], Float32(15.0))

    a.free()
    x.free()


def test_syr2_upper() raises:
    print("Testing syr2 (upper)...")
    var n = 2
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)
    var y = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 2.0
    a[3] = 3.0

    x[0] = 1.0
    x[1] = 2.0
    y[0] = 3.0
    y[1] = 4.0

    syr2("U", n, Float32(1.0), x, 1, y, 1, a, lda)

    assert_almost_equal(a[0], Float32(7.0))
    assert_almost_equal(a[2], Float32(12.0))
    assert_almost_equal(a[3], Float32(19.0))

    a.free()
    x.free()
    y.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
