from std.testing import assert_almost_equal, TestSuite

from src.level3 import gemm, syrk, trmm, trsm, symm, syr2k


def test_gemm_no_transpose() raises:
    print("Testing gemm (no transpose)...")
    var m = 2
    var n = 2
    var k = 2
    var lda = m
    var ldb = k
    var ldc = m

    var a = alloc[Scalar[DType.float32]](lda * k)
    var b = alloc[Scalar[DType.float32]](ldb * n)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 4.0

    b[0] = 5.0
    b[1] = 6.0
    b[2] = 7.0
    b[3] = 8.0

    for i in range(ldc * n):
        c[i] = 0.0

    gemm("N", "N", m, n, k, Float32(1.0), a, lda, b, ldb, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(23.0))
    assert_almost_equal(c[1], Float32(34.0))
    assert_almost_equal(c[2], Float32(31.0))
    assert_almost_equal(c[3], Float32(46.0))

    a.free()
    b.free()
    c.free()


def test_gemm_with_beta() raises:
    print("Testing gemm (with beta)...")
    var m = 2
    var n = 2
    var k = 2
    var lda = m
    var ldb = k
    var ldc = m

    var a = alloc[Scalar[DType.float32]](lda * k)
    var b = alloc[Scalar[DType.float32]](ldb * n)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 4.0

    b[0] = 5.0
    b[1] = 6.0
    b[2] = 7.0
    b[3] = 8.0

    c[0] = 1.0
    c[1] = 1.0
    c[2] = 1.0
    c[3] = 1.0

    gemm("N", "N", m, n, k, Float32(1.0), a, lda, b, ldb, Float32(1.0), c, ldc)

    assert_almost_equal(c[0], Float32(24.0))
    assert_almost_equal(c[1], Float32(35.0))
    assert_almost_equal(c[2], Float32(32.0))
    assert_almost_equal(c[3], Float32(47.0))

    a.free()
    b.free()
    c.free()


def test_syrk_upper() raises:
    print("Testing syrk (upper)...")
    var n = 2
    var k = 3
    var lda = n
    var ldc = n

    var a = alloc[Scalar[DType.float32]](lda * k)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 4.0
    a[4] = 5.0
    a[5] = 6.0

    for i in range(ldc * n):
        c[i] = 0.0

    syrk("U", "N", n, k, Float32(1.0), a, lda, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(35.0))
    assert_almost_equal(c[2], Float32(44.0))
    assert_almost_equal(c[3], Float32(56.0))

    a.free()
    c.free()


def test_trmm_left_upper() raises:
    print("Testing trmm (left, upper)...")
    var m = 2
    var n = 2
    var lda = m
    var ldb = m

    var a = alloc[Scalar[DType.float32]](lda * m)
    var b = alloc[Scalar[DType.float32]](ldb * n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 2.0
    a[3] = 3.0

    b[0] = 1.0
    b[1] = 2.0
    b[2] = 3.0
    b[3] = 4.0

    trmm("L", "U", "N", "N", m, n, Float32(1.0), a, lda, b, ldb)

    assert_almost_equal(b[0], Float32(5.0))
    assert_almost_equal(b[1], Float32(6.0))
    assert_almost_equal(b[2], Float32(11.0))
    assert_almost_equal(b[3], Float32(12.0))

    a.free()
    b.free()


def test_trsm_left_upper() raises:
    print("Testing trsm (left, upper)...")
    var m = 2
    var n = 2
    var lda = m
    var ldb = m

    var a = alloc[Scalar[DType.float32]](lda * m)
    var b = alloc[Scalar[DType.float32]](ldb * n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 2.0
    a[3] = 3.0

    b[0] = 5.0
    b[1] = 6.0
    b[2] = 11.0
    b[3] = 12.0

    trsm("L", "U", "N", "N", m, n, Float32(1.0), a, lda, b, ldb)

    assert_almost_equal(b[0], Float32(1.0))
    assert_almost_equal(b[1], Float32(2.0))
    assert_almost_equal(b[2], Float32(3.0))
    assert_almost_equal(b[3], Float32(4.0))

    a.free()
    b.free()


def test_symm_left_upper() raises:
    print("Testing symm (left, upper)...")
    var m = 2
    var n = 2
    var lda = m
    var ldb = m
    var ldc = m

    var a = alloc[Scalar[DType.float32]](lda * m)
    var b = alloc[Scalar[DType.float32]](ldb * n)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 2.0
    a[3] = 3.0

    b[0] = 1.0
    b[1] = 2.0
    b[2] = 3.0
    b[3] = 4.0

    for i in range(ldc * n):
        c[i] = 0.0

    symm("L", "U", m, n, Float32(1.0), a, lda, b, ldb, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(5.0))
    assert_almost_equal(c[1], Float32(8.0))
    assert_almost_equal(c[2], Float32(11.0))
    assert_almost_equal(c[3], Float32(18.0))

    a.free()
    b.free()
    c.free()


def test_syr2k_upper() raises:
    print("Testing syr2k (upper)...")
    var n = 2
    var k = 3
    var lda = n
    var ldb = n
    var ldc = n

    var a = alloc[Scalar[DType.float32]](lda * k)
    var b = alloc[Scalar[DType.float32]](ldb * k)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 4.0
    a[2] = 2.0
    a[3] = 5.0
    a[4] = 3.0
    a[5] = 6.0

    b[0] = 7.0
    b[1] = 10.0
    b[2] = 8.0
    b[3] = 11.0
    b[4] = 9.0
    b[5] = 12.0

    for i in range(ldc * n):
        c[i] = 0.0

    syr2k("U", "N", n, k, Float32(1.0), a, lda, b, ldb, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(100.0))
    assert_almost_equal(c[2], Float32(190.0))
    assert_almost_equal(c[3], Float32(334.0))

    a.free()
    b.free()
    c.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
