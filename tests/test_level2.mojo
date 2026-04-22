from std.testing import assert_almost_equal, assert_equal, TestSuite

from src.level2 import gemv, trmv, trsv, symv, syr, syr2, gbmv, sbmv, spmv, tbmv, tbsv, tpmv, tpsv


def test_gemv_no_transpose() raises:
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

    gemv("N", m, n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(6.0))
    assert_almost_equal(y[1], Float32(15.0))

    a.free()
    x.free()
    y.free()


def test_gemv_transpose() raises:
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

    gemv("T", m, n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(9.0))
    assert_almost_equal(y[1], Float32(12.0))
    assert_almost_equal(y[2], Float32(15.0))

    a.free()
    x.free()
    y.free()


def test_gemv_with_beta() raises:
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

    gemv("N", m, n, Float32(1.0), a, lda, x, 1, Float32(1.0), y, 1)

    assert_almost_equal(y[0], Float32(4.0))
    assert_almost_equal(y[1], Float32(8.0))

    a.free()
    x.free()
    y.free()


def test_trmv_upper() raises:
    print("Testing trmv (upper)...")
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


def test_symv_lower() raises:
    print("Testing symv (lower)...")
    var n = 2
    var lda = n

    var a = alloc[Scalar[DType.float32]](n * n)
    var x = alloc[Scalar[DType.float32]](n)
    var y = alloc[Scalar[DType.float32]](n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 2.0
    a[3] = 3.0

    x[0] = 1.0
    x[1] = 1.0

    y[0] = 0.0
    y[1] = 0.0

    symv("L", n, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(3.0))
    assert_almost_equal(y[1], Float32(5.0))

    a.free()
    x.free()
    y.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
