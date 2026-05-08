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


# ── gemv new cases ────────────────────────────────────────────────────────────


def test_gemv_no_trans_larger_x() raises:
    # m=2, n=3, A col-major=[1,4,2,5,3,6], x=[1,2,3] -> y=[14,32]
    var a = alloc[Scalar[DType.float64]](6)
    var x = alloc[Scalar[DType.float64]](3)
    var y = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 4
    a[2] = 2
    a[3] = 5
    a[4] = 3
    a[5] = 6
    x[0] = 1
    x[1] = 2
    x[2] = 3
    y[0] = 0
    y[1] = 0
    gemv[DType.float64]("N", 2, 3, Float64(1.0), a, 2, x, 1, Float64(0.0), y, 1)
    assert_almost_equal(y[0], Float64(14.0))
    assert_almost_equal(y[1], Float64(32.0))
    a.free()
    x.free()
    y.free()


def test_gemv_trans_larger() raises:
    # m=3, n=2, A col-major=[1,2,3,4,5,6] (3x2), x=[1,1,1] -> y=[6,15]
    var a = alloc[Scalar[DType.float64]](6)
    var x = alloc[Scalar[DType.float64]](3)
    var y = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 2
    a[2] = 3
    a[3] = 4
    a[4] = 5
    a[5] = 6
    x[0] = 1
    x[1] = 1
    x[2] = 1
    y[0] = 0
    y[1] = 0
    gemv[DType.float64]("T", 3, 2, Float64(1.0), a, 3, x, 1, Float64(0.0), y, 1)
    assert_almost_equal(y[0], Float64(6.0))
    assert_almost_equal(y[1], Float64(15.0))
    a.free()
    x.free()
    y.free()


def test_gemv_beta_scaling() raises:
    # A=[[1,2],[3,4]], x=[1,1], y=[1,1], alpha=1, beta=2 -> y=[1+2+2, 3+4+2]=[5,9]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    var y = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 3
    a[2] = 2
    a[3] = 4
    x[0] = 1
    x[1] = 1
    y[0] = 1
    y[1] = 1
    gemv[DType.float64]("N", 2, 2, Float64(1.0), a, 2, x, 1, Float64(2.0), y, 1)
    assert_almost_equal(y[0], Float64(5.0))
    assert_almost_equal(y[1], Float64(9.0))
    a.free()
    x.free()
    y.free()


def test_gemv_alpha_scaling() raises:
    # A=[[1,2],[3,4]], x=[1,1], y=[0,0], alpha=3 -> y=[9,21]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    var y = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 3
    a[2] = 2
    a[3] = 4
    x[0] = 1
    x[1] = 1
    y[0] = 0
    y[1] = 0
    gemv[DType.float64]("N", 2, 2, Float64(3.0), a, 2, x, 1, Float64(0.0), y, 1)
    assert_almost_equal(y[0], Float64(9.0))
    assert_almost_equal(y[1], Float64(21.0))
    a.free()
    x.free()
    y.free()


def test_gemv_trans_beta() raises:
    # A=[[1,3],[2,4]], x=[1,2], y=[1,1], alpha=1, beta=1
    # y := A^T*x + y = [1+6, 2+8] + [1,1] = [8,11]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    var y = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 3
    a[2] = 2
    a[3] = 4
    x[0] = 1
    x[1] = 2
    y[0] = 1
    y[1] = 1
    gemv[DType.float64]("T", 2, 2, Float64(1.0), a, 2, x, 1, Float64(1.0), y, 1)
    assert_almost_equal(y[0], Float64(8.0))
    assert_almost_equal(y[1], Float64(11.0))
    a.free()
    x.free()
    y.free()


# ── trmv new cases ────────────────────────────────────────────────────────────


def test_trmv_lower_no_trans() raises:
    # L col-major n=3: [1,2,4,0,3,5,0,0,6], x=[1,2,3] -> L*x=[1,8,32]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    a[0] = 1
    a[1] = 2
    a[2] = 4
    a[3] = 0
    a[4] = 3
    a[5] = 5
    a[6] = 0
    a[7] = 0
    a[8] = 6
    x[0] = 1
    x[1] = 2
    x[2] = 3
    trmv[DType.float64]("L", "N", "N", 3, a, 3, x, 1)
    assert_almost_equal(x[0], Float64(1.0))
    assert_almost_equal(x[1], Float64(8.0))
    assert_almost_equal(x[2], Float64(32.0))
    a.free()
    x.free()


def test_trmv_upper_trans() raises:
    # U^T*x: U col-major=[1,0,0,2,4,0,3,5,6], x=[1,1,1] -> [1,6,14]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 2
    a[4] = 4
    a[5] = 0
    a[6] = 3
    a[7] = 5
    a[8] = 6
    x[0] = 1
    x[1] = 1
    x[2] = 1
    trmv[DType.float64]("U", "T", "N", 3, a, 3, x, 1)
    assert_almost_equal(x[0], Float64(1.0))
    assert_almost_equal(x[1], Float64(6.0))
    assert_almost_equal(x[2], Float64(14.0))
    a.free()
    x.free()


def test_trmv_lower_trans() raises:
    # L^T*x: L=[[1,0,0],[2,3,0],[4,5,6]], x=[1,1,1] -> L^T*x=[7,8,6]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    a[0] = 1
    a[1] = 2
    a[2] = 4
    a[3] = 0
    a[4] = 3
    a[5] = 5
    a[6] = 0
    a[7] = 0
    a[8] = 6
    x[0] = 1
    x[1] = 1
    x[2] = 1
    trmv[DType.float64]("L", "T", "N", 3, a, 3, x, 1)
    assert_almost_equal(x[0], Float64(7.0))
    assert_almost_equal(x[1], Float64(8.0))
    assert_almost_equal(x[2], Float64(6.0))
    a.free()
    x.free()


def test_trmv_unit_diagonal() raises:
    # unit upper n=2: diag treated as 1, off-diag=2, x=[3,4] -> [3+8,4]=[11,4]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    a[0] = 99
    a[1] = 0
    a[2] = 2
    a[3] = 99
    x[0] = 3
    x[1] = 4
    trmv[DType.float64]("U", "N", "U", 2, a, 2, x, 1)
    assert_almost_equal(x[0], Float64(11.0))
    assert_almost_equal(x[1], Float64(4.0))
    a.free()
    x.free()


# ── trsv new cases ────────────────────────────────────────────────────────────


def test_trsv_lower_no_trans() raises:
    # L=[[2,0,0],[1,3,0],[4,2,5]], b=[4,5,6] -> x=[2,1,-0.8]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    a[0] = 2
    a[1] = 1
    a[2] = 4
    a[3] = 0
    a[4] = 3
    a[5] = 2
    a[6] = 0
    a[7] = 0
    a[8] = 5
    x[0] = 4
    x[1] = 5
    x[2] = 6
    trsv[DType.float64]("L", "N", "N", 3, a, 3, x, 1)
    assert_almost_equal(x[0], Float64(2.0), atol=1e-6)
    assert_almost_equal(x[1], Float64(1.0), atol=1e-6)
    assert_almost_equal(x[2], Float64(-0.8), atol=1e-6)
    a.free()
    x.free()


def test_trsv_upper_trans() raises:
    # U^T*x=b: U=[[1,2,3],[0,4,5],[0,0,6]], b=[1,6,18] -> x=[1,1,5/3]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 2
    a[4] = 4
    a[5] = 0
    a[6] = 3
    a[7] = 5
    a[8] = 6
    x[0] = 1
    x[1] = 6
    x[2] = 18
    trsv[DType.float64]("U", "T", "N", 3, a, 3, x, 1)
    assert_almost_equal(x[0], Float64(1.0), atol=1e-6)
    assert_almost_equal(x[1], Float64(1.0), atol=1e-6)
    assert_almost_equal(x[2], Float64(1.6666667), atol=1e-5)
    a.free()
    x.free()


def test_trsv_lower_trans() raises:
    # L^T*x=b: L=[[1,0],[2,3]], b=[5,6] -> L^T=[[1,2],[0,3]], x=[1,2]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 2
    a[2] = 0
    a[3] = 3
    x[0] = 5
    x[1] = 6
    trsv[DType.float64]("L", "T", "N", 2, a, 2, x, 1)
    assert_almost_equal(x[0], Float64(1.0), atol=1e-6)
    assert_almost_equal(x[1], Float64(2.0), atol=1e-6)
    a.free()
    x.free()


def test_trsv_unit_diagonal() raises:
    # unit upper n=2: [[1,2],[0,1]]*x=[5,3] -> x=[-1,3]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    a[0] = 99
    a[1] = 0
    a[2] = 2
    a[3] = 99
    x[0] = 5
    x[1] = 3
    trsv[DType.float64]("U", "N", "U", 2, a, 2, x, 1)
    assert_almost_equal(x[0], Float64(-1.0), atol=1e-6)
    assert_almost_equal(x[1], Float64(3.0), atol=1e-6)
    a.free()
    x.free()


# ── symv new cases ────────────────────────────────────────────────────────────


def test_symv_upper_3x3() raises:
    # A upper col-major=[1,0,0,2,4,0,3,5,6], x=[1,2,3], alpha=1, beta=0
    # A=[[1,2,3],[2,4,5],[3,5,6]], A*x=[14,25,31]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    var y = alloc[Scalar[DType.float64]](3)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 2
    a[4] = 4
    a[5] = 0
    a[6] = 3
    a[7] = 5
    a[8] = 6
    x[0] = 1
    x[1] = 2
    x[2] = 3
    y[0] = 0
    y[1] = 0
    y[2] = 0
    symv[DType.float64]("U", 3, Float64(1.0), a, 3, x, 1, Float64(0.0), y, 1)
    assert_almost_equal(y[0], Float64(14.0))
    assert_almost_equal(y[1], Float64(25.0))
    assert_almost_equal(y[2], Float64(31.0))
    a.free()
    x.free()
    y.free()


def test_symv_upper_alpha_beta() raises:
    # same A and x, alpha=2, beta=3, y=[1,1,1] -> 2*[14,25,31]+3*[1,1,1]=[31,53,65]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    var y = alloc[Scalar[DType.float64]](3)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 2
    a[4] = 4
    a[5] = 0
    a[6] = 3
    a[7] = 5
    a[8] = 6
    x[0] = 1
    x[1] = 2
    x[2] = 3
    y[0] = 1
    y[1] = 1
    y[2] = 1
    symv[DType.float64]("U", 3, Float64(2.0), a, 3, x, 1, Float64(3.0), y, 1)
    assert_almost_equal(y[0], Float64(31.0))
    assert_almost_equal(y[1], Float64(53.0))
    assert_almost_equal(y[2], Float64(65.0))
    a.free()
    x.free()
    y.free()


# ── syr new cases ─────────────────────────────────────────────────────────────


def test_syr_upper_3x3() raises:
    # A=0 n=3, x=[1,2,3], alpha=2 -> upper tri of 2*x*x^T
    # col-major: [0,0]=2,[0,1]=4,[0,2]=6,[1,1]=8,[1,2]=12,[2,2]=18
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    for i in range(9):
        a[i] = 0
    x[0] = 1
    x[1] = 2
    x[2] = 3
    syr[DType.float64]("U", 3, Float64(2.0), x, 1, a, 3)
    assert_almost_equal(a[0], Float64(2.0))
    assert_almost_equal(a[3], Float64(4.0))
    assert_almost_equal(a[4], Float64(8.0))
    assert_almost_equal(a[6], Float64(6.0))
    assert_almost_equal(a[7], Float64(12.0))
    assert_almost_equal(a[8], Float64(18.0))
    a.free()
    x.free()


def test_syr_lower_2x2() raises:
    # A=0, x=[1,2], alpha=1 -> lower tri: [0,0]=1,[1,0]=2,[1,1]=4
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    a[0] = 0
    a[1] = 0
    a[2] = 0
    a[3] = 0
    x[0] = 1
    x[1] = 2
    syr[DType.float64]("L", 2, Float64(1.0), x, 1, a, 2)
    assert_almost_equal(a[0], Float64(1.0))
    assert_almost_equal(a[1], Float64(2.0))
    assert_almost_equal(a[3], Float64(4.0))
    a.free()
    x.free()


# ── syr2 new cases ────────────────────────────────────────────────────────────


def test_syr2_lower_3x3() raises:
    # A=0, x=[1,2,3], y=[4,5,6], alpha=1
    # lower tri (col-major): [0,0]=8,[1,0]=13,[2,0]=20,[1,1]=20,[2,1]=27,[2,2]=36
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    var y = alloc[Scalar[DType.float64]](3)
    for i in range(9):
        a[i] = 0
    x[0] = 1
    x[1] = 2
    x[2] = 3
    y[0] = 4
    y[1] = 5
    y[2] = 6
    syr2[DType.float64]("L", 3, Float64(1.0), x, 1, y, 1, a, 3)
    assert_almost_equal(a[0], Float64(8.0))
    assert_almost_equal(a[1], Float64(13.0))
    assert_almost_equal(a[2], Float64(18.0))
    assert_almost_equal(a[4], Float64(20.0))
    assert_almost_equal(a[5], Float64(27.0))
    assert_almost_equal(a[8], Float64(36.0))
    a.free()
    x.free()
    y.free()


# ── ger new cases ─────────────────────────────────────────────────────────────


def test_ger_3x3() raises:
    # A=0, m=3, n=3, x=[1,2,3], y=[4,5,6], alpha=1
    # col-major: [4,8,12, 5,10,15, 6,12,18]
    var a = alloc[Scalar[DType.float64]](9)
    var x = alloc[Scalar[DType.float64]](3)
    var y = alloc[Scalar[DType.float64]](3)
    for i in range(9):
        a[i] = 0
    x[0] = 1
    x[1] = 2
    x[2] = 3
    y[0] = 4
    y[1] = 5
    y[2] = 6
    ger[DType.float64](3, 3, Float64(1.0), x, 1, y, 1, a, 3)
    assert_almost_equal(a[0], Float64(4.0))
    assert_almost_equal(a[1], Float64(8.0))
    assert_almost_equal(a[2], Float64(12.0))
    assert_almost_equal(a[3], Float64(5.0))
    assert_almost_equal(a[4], Float64(10.0))
    assert_almost_equal(a[5], Float64(15.0))
    assert_almost_equal(a[6], Float64(6.0))
    assert_almost_equal(a[7], Float64(12.0))
    assert_almost_equal(a[8], Float64(18.0))
    a.free()
    x.free()
    y.free()


def test_ger_alpha_nonunit() raises:
    # A=[[1,2],[3,4]], x=[1,1], y=[1,1], alpha=2 -> A += 2*ones = [[3,4],[5,6]]
    var a = alloc[Scalar[DType.float64]](4)
    var x = alloc[Scalar[DType.float64]](2)
    var y = alloc[Scalar[DType.float64]](2)
    a[0] = 1
    a[1] = 3
    a[2] = 2
    a[3] = 4
    x[0] = 1
    x[1] = 1
    y[0] = 1
    y[1] = 1
    ger[DType.float64](2, 2, Float64(2.0), x, 1, y, 1, a, 2)
    assert_almost_equal(a[0], Float64(3.0))
    assert_almost_equal(a[1], Float64(5.0))
    assert_almost_equal(a[2], Float64(4.0))
    assert_almost_equal(a[3], Float64(6.0))
    a.free()
    x.free()
    y.free()


# ── spmv new case ─────────────────────────────────────────────────────────────


def test_spmv_upper_2x2() raises:
    # upper packed [a00,a01,a11]=[1,2,3], x=[1,1], y=[0,0]
    # A=[[1,2],[2,3]], A*x=[3,5]
    var ap = alloc[Scalar[DType.float64]](3)
    var x = alloc[Scalar[DType.float64]](2)
    var y = alloc[Scalar[DType.float64]](2)
    ap[0] = 1
    ap[1] = 2
    ap[2] = 3
    x[0] = 1
    x[1] = 1
    y[0] = 0
    y[1] = 0
    spmv[DType.float64]("U", 2, Float64(1.0), ap, x, 1, Float64(0.0), y, 1)
    assert_almost_equal(y[0], Float64(3.0))
    assert_almost_equal(y[1], Float64(5.0))
    ap.free()
    x.free()
    y.free()


# ── tpmv new case ─────────────────────────────────────────────────────────────


def test_tpmv_lower_no_trans() raises:
    # AP lower packed [1,2,3,4,5,6], x=[1,1,1]
    # L=[[1,0,0],[2,3,0],[4,5,6]], L*x=[1,5,15]
    var ap = alloc[Scalar[DType.float64]](6)
    var x = alloc[Scalar[DType.float64]](3)
    ap[0] = 1
    ap[1] = 2
    ap[2] = 3
    ap[3] = 4
    ap[4] = 5
    ap[5] = 6
    x[0] = 1
    x[1] = 1
    x[2] = 1
    tpmv[DType.float64]("L", "N", "N", 3, ap, x, 1)
    assert_almost_equal(x[0], Float64(1.0))
    assert_almost_equal(x[1], Float64(6.0))
    assert_almost_equal(x[2], Float64(14.0))
    ap.free()
    x.free()


# ── tpsv new case ─────────────────────────────────────────────────────────────


def test_tpsv_lower_no_trans() raises:
    # AP lower packed =[1,2,3,4,5,6], x=[1,6,14] (tpmv lower result) -> x=[1,1,1]
    var ap = alloc[Scalar[DType.float64]](6)
    var x = alloc[Scalar[DType.float64]](3)
    ap[0] = 1
    ap[1] = 2
    ap[2] = 3
    ap[3] = 4
    ap[4] = 5
    ap[5] = 6
    x[0] = 1
    x[1] = 6
    x[2] = 14
    tpsv[DType.float64]("L", "N", "N", 3, ap, x, 1)
    assert_almost_equal(x[0], Float64(1.0), atol=1e-6)
    assert_almost_equal(x[1], Float64(1.0), atol=1e-6)
    assert_almost_equal(x[2], Float64(1.0), atol=1e-6)
    ap.free()
    x.free()


# ── spr new cases ─────────────────────────────────────────────────────────────


def test_spr_lower_3x3() raises:
    # AP lower packed =0, x=[1,2,3], alpha=1
    # lower packed [a00,a10,a20,a11,a21,a22] = [1,2,3,4,6,9]
    var ap = alloc[Scalar[DType.float64]](6)
    var x = alloc[Scalar[DType.float64]](3)
    for i in range(6):
        ap[i] = 0
    x[0] = 1
    x[1] = 2
    x[2] = 3
    spr[DType.float64]("L", 3, Float64(1.0), x, 1, ap)
    assert_almost_equal(ap[0], Float64(1.0))
    assert_almost_equal(ap[1], Float64(2.0))
    assert_almost_equal(ap[2], Float64(3.0))
    assert_almost_equal(ap[3], Float64(4.0))
    assert_almost_equal(ap[4], Float64(6.0))
    assert_almost_equal(ap[5], Float64(9.0))
    ap.free()
    x.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
