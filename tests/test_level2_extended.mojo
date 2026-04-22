from std.testing import assert_almost_equal, assert_equal, TestSuite

from src.level2 import (
    gemv,
    trmv,
    trsv,
    symv,
    syr,
    syr2,
    spr,
    spr2,
    ger,
    gbmv,
    sbmv,
    spmv,
    tbmv,
    tbsv,
    tpmv,
    tpsv,
)


def test_ger() raises:
    print("Testing ger...")
    var m = 3
    var n = 4
    var lda = m

    var a = alloc[Scalar[DType.float32]](m * n)
    var x = alloc[Scalar[DType.float32]](m)
    var y = alloc[Scalar[DType.float32]](n)

    # A is zero-initialized
    for i in range(m * n):
        a[i] = 0.0

    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    y[0] = 1.0
    y[1] = 2.0
    y[2] = 3.0
    y[3] = 4.0

    ger(m, n, Float32(2.0), x, 1, y, 1, a, lda)

    # Expected: A = alpha * x * y^T
    # A[i,j] = 2.0 * x[i] * y[j]
    # Column-major storage
    # Column 0: [2, 4, 6]
    # Column 1: [4, 8, 12]
    # Column 2: [6, 12, 18]
    # Column 3: [8, 16, 24]
    assert_almost_equal(a[0], Float32(2.0))
    assert_almost_equal(a[1], Float32(4.0))
    assert_almost_equal(a[2], Float32(6.0))
    assert_almost_equal(a[3], Float32(4.0))
    assert_almost_equal(a[4], Float32(8.0))
    assert_almost_equal(a[5], Float32(12.0))
    assert_almost_equal(a[6], Float32(6.0))
    assert_almost_equal(a[7], Float32(12.0))
    assert_almost_equal(a[8], Float32(18.0))
    assert_almost_equal(a[9], Float32(8.0))
    assert_almost_equal(a[10], Float32(16.0))
    assert_almost_equal(a[11], Float32(24.0))

    a.free()
    x.free()
    y.free()


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


def test_spr_upper() raises:
    print("Testing spr (upper packed)...")
    var n = 2
    var ap = alloc[Scalar[DType.float32]](3)
    var x = alloc[Scalar[DType.float32]](2)

    ap[0] = 0.0
    ap[1] = 0.0
    ap[2] = 0.0
    x[0] = 1.0
    x[1] = 2.0

    spr("U", n, Float32(1.0), x, 1, ap)

    assert_almost_equal(ap[0], Float32(1.0))
    assert_almost_equal(ap[1], Float32(2.0))
    assert_almost_equal(ap[2], Float32(4.0))

    ap.free()
    x.free()


def test_spr2_upper() raises:
    print("Testing spr2 (upper packed)...")
    var n = 2
    var ap = alloc[Scalar[DType.float32]](3)
    var x = alloc[Scalar[DType.float32]](2)
    var y = alloc[Scalar[DType.float32]](2)

    ap[0] = 0.0
    ap[1] = 0.0
    ap[2] = 0.0
    x[0] = 1.0
    x[1] = 2.0
    y[0] = 3.0
    y[1] = 4.0

    spr2("U", n, Float32(1.0), x, 1, y, 1, ap)

    assert_almost_equal(ap[0], Float32(6.0))
    assert_almost_equal(ap[1], Float32(10.0))
    assert_almost_equal(ap[2], Float32(16.0))

    ap.free()
    x.free()
    y.free()


def test_gbmv() raises:
    print("Testing gbmv...")
    var m = 4
    var n = 4
    var kl = 1
    var ku = 1
    var lda = 4
    var a = alloc[Scalar[DType.float32]](16)
    var x = alloc[Scalar[DType.float32]](4)
    var y = alloc[Scalar[DType.float32]](4)

    for i in range(16):
        a[i] = 0.0
    a[0] = 1.0
    a[1] = 2.0
    a[4] = 3.0
    a[5] = 4.0
    a[6] = 5.0
    a[9] = 6.0
    a[10] = 7.0
    a[14] = 8.0

    for i in range(4):
        x[i] = 1.0
        y[i] = 0.0

    gbmv("N", m, n, kl, ku, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(5.0))
    assert_almost_equal(y[1], Float32(4.0))
    assert_almost_equal(y[2], Float32(11.0))
    assert_almost_equal(y[3], Float32(7.0))

    a.free()
    x.free()
    y.free()


def test_sbmv_lower() raises:
    print("Testing sbmv (lower)...")
    var n = 3
    var k = 1
    var lda = 3
    var a = alloc[Scalar[DType.float32]](12)
    var x = alloc[Scalar[DType.float32]](3)
    var y = alloc[Scalar[DType.float32]](3)

    for i in range(12):
        a[i] = 0.0
    a[0] = 1.0
    a[2] = 2.0
    a[3] = 3.0
    a[5] = 4.0
    a[6] = 5.0
    a[8] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0
    y[0] = 0.0
    y[1] = 0.0
    y[2] = 0.0

    sbmv("L", n, k, Float32(1.0), a, lda, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(1.0))
    assert_almost_equal(y[1], Float32(3.0))
    assert_almost_equal(y[2], Float32(5.0))

    a.free()
    x.free()
    y.free()


def test_spmv_lower() raises:
    print("Testing spmv (lower)...")
    var n = 3
    var ap = alloc[Scalar[DType.float32]](6)
    var x = alloc[Scalar[DType.float32]](3)
    var y = alloc[Scalar[DType.float32]](3)

    ap[0] = 1.0
    ap[1] = 2.0
    ap[2] = 3.0
    ap[3] = 4.0
    ap[4] = 5.0
    ap[5] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0
    y[0] = 0.0
    y[1] = 0.0
    y[2] = 0.0

    spmv("L", n, Float32(1.0), ap, x, 1, Float32(0.0), y, 1)

    assert_almost_equal(y[0], Float32(6.0))
    assert_almost_equal(y[1], Float32(11.0))
    assert_almost_equal(y[2], Float32(14.0))

    ap.free()
    x.free()
    y.free()


def test_tbmv_upper() raises:
    print("Testing tbmv (upper)...")
    var n = 3
    var k = 1
    var lda = 2
    var a = alloc[Scalar[DType.float32]](6)
    var x = alloc[Scalar[DType.float32]](3)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 4.0
    a[3] = 3.0
    a[4] = 5.0
    a[5] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0

    tbmv("U", "N", "N", n, k, a, lda, x, 1)

    assert_almost_equal(x[0], Float32(6.0))
    assert_almost_equal(x[1], Float32(8.0))
    assert_almost_equal(x[2], Float32(6.0))

    a.free()
    x.free()


def test_tbsv_upper() raises:
    print("Testing tbsv (upper)...")
    var n = 3
    var k = 1
    var lda = 2
    var a = alloc[Scalar[DType.float32]](6)
    var x = alloc[Scalar[DType.float32]](3)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 4.0
    a[3] = 3.0
    a[4] = 5.0
    a[5] = 6.0

    x[0] = 3.0
    x[1] = 9.0
    x[2] = 6.0

    tbsv("U", "N", "N", n, k, a, lda, x, 1)

    assert_almost_equal(x[0], Float32(-1.1666667))
    assert_almost_equal(x[1], Float32(1.3333333))
    assert_almost_equal(x[2], Float32(1.0))

    a.free()
    x.free()


def test_tpmv_upper() raises:
    print("Testing tpmv (upper)...")
    var n = 3
    var ap = alloc[Scalar[DType.float32]](6)
    var x = alloc[Scalar[DType.float32]](3)

    ap[0] = 1.0
    ap[1] = 2.0
    ap[2] = 4.0
    ap[3] = 3.0
    ap[4] = 5.0
    ap[5] = 6.0

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 1.0

    tpmv("U", "N", "N", n, ap, x, 1)

    assert_almost_equal(x[0], Float32(6.0))
    assert_almost_equal(x[1], Float32(9.0))
    assert_almost_equal(x[2], Float32(6.0))

    ap.free()
    x.free()


def test_tpsv_upper() raises:
    print("Testing tpsv (upper)...")
    var n = 3
    var ap = alloc[Scalar[DType.float32]](6)
    var x = alloc[Scalar[DType.float32]](3)

    ap[0] = 1.0
    ap[1] = 2.0
    ap[2] = 4.0
    ap[3] = 3.0
    ap[4] = 5.0
    ap[5] = 6.0

    x[0] = 6.0
    x[1] = 9.0
    x[2] = 6.0

    tpsv("U", "N", "N", n, ap, x, 1)

    assert_almost_equal(x[0], Float32(1.0))
    assert_almost_equal(x[1], Float32(1.0))
    assert_almost_equal(x[2], Float32(1.0))

    ap.free()
    x.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
