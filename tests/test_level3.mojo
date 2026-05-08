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


def test_gemm_transpose_b() raises:
    print("Testing gemm (transpose B)...")
    var m = 2
    var n = 2
    var k = 3
    var lda = m
    var ldb = n
    var ldc = m

    var a = alloc[Scalar[DType.float32]](lda * k)
    var b = alloc[Scalar[DType.float32]](ldb * k)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 4.0
    a[4] = 5.0
    a[5] = 6.0

    b[0] = 1.0
    b[1] = 3.0
    b[2] = 5.0
    b[3] = 2.0
    b[4] = 4.0
    b[5] = 6.0

    for i in range(ldc * n):
        c[i] = 0.0

    gemm("N", "T", m, n, k, Float32(1.0), a, lda, b, ldb, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(36.0))
    assert_almost_equal(c[1], Float32(46.0))
    assert_almost_equal(c[2], Float32(39.0))
    assert_almost_equal(c[3], Float32(50.0))

    a.free()
    b.free()
    c.free()


def test_gemm_transpose_both() raises:
    print("Testing gemm (transpose both)...")
    var m = 2
    var n = 2
    var k = 3
    var lda = k
    var ldb = n
    var ldc = m

    var a = alloc[Scalar[DType.float32]](lda * m)
    var b = alloc[Scalar[DType.float32]](ldb * k)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 4.0
    a[4] = 5.0
    a[5] = 6.0

    b[0] = 1.0
    b[1] = 3.0
    b[2] = 5.0
    b[3] = 2.0
    b[4] = 4.0
    b[5] = 6.0

    for i in range(ldc * n):
        c[i] = 0.0

    gemm("T", "T", m, n, k, Float32(1.0), a, lda, b, ldb, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(23.0))
    assert_almost_equal(c[1], Float32(53.0))
    assert_almost_equal(c[2], Float32(25.0))
    assert_almost_equal(c[3], Float32(58.0))

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


def test_syrk_lower() raises:
    print("Testing syrk (lower)...")
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

    syrk("L", "N", n, k, Float32(1.0), a, lda, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(35.0))
    assert_almost_equal(c[1], Float32(44.0))
    assert_almost_equal(c[3], Float32(56.0))

    a.free()
    c.free()


def test_syrk_transpose() raises:
    print("Testing syrk (transpose)...")
    var n = 2
    var k = 3
    var lda = k
    var ldc = n

    var a = alloc[Scalar[DType.float32]](lda * n)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 3.0
    a[2] = 5.0
    a[3] = 2.0
    a[4] = 4.0
    a[5] = 6.0

    for i in range(ldc * n):
        c[i] = 0.0

    syrk("U", "T", n, k, Float32(1.0), a, lda, Float32(0.0), c, ldc)

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


def test_gemm_alpha_zero() raises:
    print("Testing gemm (alpha=0)...")
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
    c[1] = 2.0
    c[2] = 3.0
    c[3] = 4.0

    gemm("N", "N", m, n, k, Float32(0.0), a, lda, b, ldb, Float32(1.0), c, ldc)

    assert_almost_equal(c[0], Float32(1.0))
    assert_almost_equal(c[1], Float32(2.0))
    assert_almost_equal(c[2], Float32(3.0))
    assert_almost_equal(c[3], Float32(4.0))

    a.free()
    b.free()
    c.free()


def test_trmm_no_unit_diagonal() raises:
    print("Testing trmm (no unit diagonal)...")
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


def test_trsm_no_unit_diagonal() raises:
    print("Testing trsm (no unit diagonal)...")
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


def test_symm_right_upper() raises:
    print("Testing symm (right, upper)...")
    var m = 2
    var n = 2
    var lda = n
    var ldb = m
    var ldc = m

    var a = alloc[Scalar[DType.float32]](lda * n)
    var b = alloc[Scalar[DType.float32]](ldb * n)
    var c = alloc[Scalar[DType.float32]](ldc * n)

    a[0] = 1.0
    a[1] = 0.0
    a[2] = 2.0
    a[3] = 3.0

    b[0] = 1.0
    b[1] = 2.0
    b[2] = 3.0
    b[3] = 4.0

    for i in range(ldc * n):
        c[i] = 0.0

    symm("R", "U", m, n, Float32(1.0), a, lda, b, ldb, Float32(0.0), c, ldc)

    assert_almost_equal(c[0], Float32(7.0))
    assert_almost_equal(c[1], Float32(10.0))
    assert_almost_equal(c[2], Float32(11.0))
    assert_almost_equal(c[3], Float32(16.0))

    a.free()
    b.free()
    c.free()


# ── gemm new cases ────────────────────────────────────────────────────────────


def test_gemm_transpose_a() raises:
    # A col-major lda=3: A=[[1,4],[2,5],[3,6]] (3x2), same for B
    # A^T (2x3) * B (3x2) = [[14,32],[32,77]] col-major=[14,32,32,77]
    var a = alloc[Scalar[DType.float64]](6)
    var b = alloc[Scalar[DType.float64]](6)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 2
    a[2] = 3
    a[3] = 4
    a[4] = 5
    a[5] = 6
    b[0] = 1
    b[1] = 2
    b[2] = 3
    b[3] = 4
    b[4] = 5
    b[5] = 6
    c[0] = 0
    c[1] = 0
    c[2] = 0
    c[3] = 0
    gemm[DType.float64](
        "T", "N", 2, 2, 3, Float64(1.0), a, 3, b, 3, Float64(0.0), c, 2
    )
    assert_almost_equal(c[0], Float64(14.0))
    assert_almost_equal(c[1], Float64(32.0))
    assert_almost_equal(c[2], Float64(32.0))
    assert_almost_equal(c[3], Float64(77.0))
    a.free()
    b.free()
    c.free()


def test_gemm_alpha_scaling() raises:
    # A=[[1,2],[3,4]], B=I, alpha=3, beta=0 -> C=3*A col-major=[3,9,6,12]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 3
    a[2] = 2
    a[3] = 4
    b[0] = 1
    b[1] = 0
    b[2] = 0
    b[3] = 1
    c[0] = 0
    c[1] = 0
    c[2] = 0
    c[3] = 0
    gemm[DType.float64](
        "N", "N", 2, 2, 2, Float64(3.0), a, 2, b, 2, Float64(0.0), c, 2
    )
    assert_almost_equal(c[0], Float64(3.0))
    assert_almost_equal(c[1], Float64(9.0))
    assert_almost_equal(c[2], Float64(6.0))
    assert_almost_equal(c[3], Float64(12.0))
    a.free()
    b.free()
    c.free()


def test_gemm_beta_accumulate() raises:
    # A=I, B=I, alpha=1, beta=2, C=[[1,2],[3,4]] -> C = I + 2*C col-major=[3,7,5,9] wait...
    # col-major C: c[0]=1,c[1]=3,c[2]=2,c[3]=4 -> alpha*I*I + 2*C = I+2C
    # c[0]=1+2=3, c[1]=0+6=6, c[2]=0+4=4, c[3]=1+8=9
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 1
    b[0] = 1
    b[1] = 0
    b[2] = 0
    b[3] = 1
    c[0] = 1
    c[1] = 3
    c[2] = 2
    c[3] = 4
    gemm[DType.float64](
        "N", "N", 2, 2, 2, Float64(1.0), a, 2, b, 2, Float64(2.0), c, 2
    )
    assert_almost_equal(c[0], Float64(3.0))
    assert_almost_equal(c[1], Float64(6.0))
    assert_almost_equal(c[2], Float64(4.0))
    assert_almost_equal(c[3], Float64(9.0))
    a.free()
    b.free()
    c.free()


# ── syrk new cases ────────────────────────────────────────────────────────────


def test_syrk_upper_with_beta() raises:
    # A=[[1,2,3],[4,5,6]] (2x3), C=I (2x2), alpha=1, beta=2
    # C = A*A^T + 2*I col-major=[16,32,32,79] wait let me recheck
    # A*A^T = [[14,32],[32,77]], 2*I=[[2,0],[0,2]], sum=[[16,32],[32,79]]
    # col-major: [16,32,32,79]
    var a = alloc[Scalar[DType.float64]](6)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 4
    a[2] = 2
    a[3] = 5
    a[4] = 3
    a[5] = 6
    c[0] = 1
    c[1] = 0
    c[2] = 0
    c[3] = 1
    syrk[DType.float64]("U", "N", 2, 3, Float64(1.0), a, 2, Float64(2.0), c, 2)
    assert_almost_equal(c[0], Float64(16.0))
    assert_almost_equal(c[2], Float64(32.0))
    assert_almost_equal(c[3], Float64(79.0))
    a.free()
    c.free()


def test_syrk_lower_with_beta() raises:
    # same A and C, "L" uplo: lower triangle updated
    var a = alloc[Scalar[DType.float64]](6)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 4
    a[2] = 2
    a[3] = 5
    a[4] = 3
    a[5] = 6
    c[0] = 1
    c[1] = 0
    c[2] = 0
    c[3] = 1
    syrk[DType.float64]("L", "N", 2, 3, Float64(1.0), a, 2, Float64(2.0), c, 2)
    assert_almost_equal(c[0], Float64(16.0))
    assert_almost_equal(c[1], Float64(32.0))
    assert_almost_equal(c[3], Float64(79.0))
    a.free()
    c.free()


# ── syr2k new cases ───────────────────────────────────────────────────────────


def test_syr2k_lower() raises:
    # from numpy: A=[[1,2],[3,4],[5,6]]^T (2x3), B=[[7,8],[9,10],[11,12]]^T (2x3)
    # lower tri of A*B^T + B*A^T at n=2, k=3
    # result lower: [178,214,256] -> c[0,0]=178, c[1,0]=214, c[1,1]=256
    var a = alloc[Scalar[DType.float64]](6)
    var b = alloc[Scalar[DType.float64]](6)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 3
    a[2] = 5
    a[3] = 2
    a[4] = 4
    a[5] = 6
    b[0] = 7
    b[1] = 9
    b[2] = 11
    b[3] = 8
    b[4] = 10
    b[5] = 12
    c[0] = 0
    c[1] = 0
    c[2] = 0
    c[3] = 0
    syr2k[DType.float64](
        "L", "T", 2, 3, Float64(1.0), a, 3, b, 3, Float64(0.0), c, 2
    )
    assert_almost_equal(c[0], Float64(178.0))
    assert_almost_equal(c[1], Float64(214.0))
    assert_almost_equal(c[3], Float64(256.0))
    a.free()
    b.free()
    c.free()


# ── symm new cases ────────────────────────────────────────────────────────────


def test_symm_left_lower() raises:
    # a col-major=[1,2,2,3]: A=[[1,2],[2,3]] (symmetric)
    # b col-major=[1,2,3,4]: B[:,0]=[1,2], B[:,1]=[3,4] -> B=[[1,3],[2,4]]
    # C = A*B = [[1*1+2*2, 1*3+2*4],[2*1+3*2, 2*3+3*4]] = [[5,11],[8,18]] col-major=[5,8,11,18]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 2
    a[2] = 2
    a[3] = 3
    b[0] = 1
    b[1] = 2
    b[2] = 3
    b[3] = 4
    c[0] = 0
    c[1] = 0
    c[2] = 0
    c[3] = 0
    symm[DType.float64](
        "L", "L", 2, 2, Float64(1.0), a, 2, b, 2, Float64(0.0), c, 2
    )
    assert_almost_equal(c[0], Float64(5.0))
    assert_almost_equal(c[1], Float64(8.0))
    assert_almost_equal(c[2], Float64(11.0))
    assert_almost_equal(c[3], Float64(18.0))
    a.free()
    b.free()
    c.free()


def test_symm_beta_nonzero() raises:
    # A=I (symmetric), B=[[1,2],[3,4]], alpha=1, beta=2, C=[[1,0],[0,1]]
    # C = B + 2*C col-major: [1+2,3+0,2+0,4+2]=[3,3,2,6]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    var c = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 1
    b[0] = 1
    b[1] = 3
    b[2] = 2
    b[3] = 4
    c[0] = 1
    c[1] = 0
    c[2] = 0
    c[3] = 1
    symm[DType.float64](
        "L", "U", 2, 2, Float64(1.0), a, 2, b, 2, Float64(2.0), c, 2
    )
    assert_almost_equal(c[0], Float64(3.0))
    assert_almost_equal(c[1], Float64(3.0))
    assert_almost_equal(c[2], Float64(2.0))
    assert_almost_equal(c[3], Float64(6.0))
    a.free()
    b.free()
    c.free()


# ── trmm new cases ────────────────────────────────────────────────────────────


def test_trmm_right_upper() raises:
    # B=[[1,2],[3,4]], A=[[1,2],[0,3]] upper right: B=B*A col-major=[1,3,8,18]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 2
    a[3] = 3
    b[0] = 1
    b[1] = 3
    b[2] = 2
    b[3] = 4
    trmm[DType.float64]("R", "U", "N", "N", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(1.0))
    assert_almost_equal(b[1], Float64(3.0))
    assert_almost_equal(b[2], Float64(8.0))
    assert_almost_equal(b[3], Float64(18.0))
    a.free()
    b.free()


def test_trmm_left_lower() raises:
    # B=[[1,2],[3,4]], A=[[1,0],[2,3]] lower left: B=A*B col-major=[1,11,2,16]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 2
    a[2] = 0
    a[3] = 3
    b[0] = 1
    b[1] = 3
    b[2] = 2
    b[3] = 4
    trmm[DType.float64]("L", "L", "N", "N", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(1.0))
    assert_almost_equal(b[1], Float64(11.0))
    assert_almost_equal(b[2], Float64(2.0))
    assert_almost_equal(b[3], Float64(16.0))
    a.free()
    b.free()


def test_trmm_left_upper_trans() raises:
    # B=[[1,2],[3,4]], A=[[1,2],[0,3]] upper trans left: B=A^T*B
    # A^T=[[1,0],[2,3]], A^T*B=[[1,2],[2*1+3*3,2*2+3*4]]= [[1,2],[11,16]] col-major=[1,11,2,16]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 2
    a[3] = 3
    b[0] = 1
    b[1] = 3
    b[2] = 2
    b[3] = 4
    trmm[DType.float64]("L", "U", "T", "N", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(1.0))
    assert_almost_equal(b[1], Float64(11.0))
    assert_almost_equal(b[2], Float64(2.0))
    assert_almost_equal(b[3], Float64(16.0))
    a.free()
    b.free()


def test_trmm_alpha_scaling() raises:
    # A=I upper, B=[[1,2],[3,4]], alpha=3 -> B=3*B col-major=[3,9,6,12]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 0
    a[3] = 1
    b[0] = 1
    b[1] = 3
    b[2] = 2
    b[3] = 4
    trmm[DType.float64]("L", "U", "N", "N", 2, 2, Float64(3.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(3.0))
    assert_almost_equal(b[1], Float64(9.0))
    assert_almost_equal(b[2], Float64(6.0))
    assert_almost_equal(b[3], Float64(12.0))
    a.free()
    b.free()


# ── trsm new cases ────────────────────────────────────────────────────────────


def test_trsm_right_upper() raises:
    # X*U=B: U=[[1,2],[0,3]], B=[[7,8],[5,6]] col-major=[7,5,8,6]
    # X=B*U^-1 col-major=[7,5,-2,-4/3] wait numpy: [7,5,-2,-1.333]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 2
    a[3] = 3
    b[0] = 7
    b[1] = 5
    b[2] = 8
    b[3] = 6
    trsm[DType.float64]("R", "U", "N", "N", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(7.0), atol=1e-6)
    assert_almost_equal(b[1], Float64(5.0), atol=1e-6)
    assert_almost_equal(b[2], Float64(-2.0), atol=1e-6)
    assert_almost_equal(b[3], Float64(-1.3333333), atol=1e-5)
    a.free()
    b.free()


def test_trsm_left_lower() raises:
    # L*X=B: L=[[1,0],[2,3]], B=[[1,2],[8,12]] col-major=[1,8,2,12]
    # X=L^-1*B col-major=[1,2,2,2.667]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 2
    a[2] = 0
    a[3] = 3
    b[0] = 1
    b[1] = 8
    b[2] = 2
    b[3] = 12
    trsm[DType.float64]("L", "L", "N", "N", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(1.0), atol=1e-6)
    assert_almost_equal(b[1], Float64(2.0), atol=1e-6)
    assert_almost_equal(b[2], Float64(2.0), atol=1e-6)
    assert_almost_equal(b[3], Float64(2.6666667), atol=1e-5)
    a.free()
    b.free()


def test_trsm_left_upper_trans() raises:
    # U^T*X=B: U=[[1,2],[0,3]], B=[[1,2],[6,12]] col-major=[1,6,2,12]
    # X col-major=[1,1.333,2,2.667] -- from numpy: [1,4/3,2,8/3]
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 1
    a[1] = 0
    a[2] = 2
    a[3] = 3
    b[0] = 1
    b[1] = 6
    b[2] = 2
    b[3] = 12
    trsm[DType.float64]("L", "U", "T", "N", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(1.0), atol=1e-6)
    assert_almost_equal(b[1], Float64(1.3333333), atol=1e-5)
    assert_almost_equal(b[2], Float64(2.0), atol=1e-6)
    assert_almost_equal(b[3], Float64(2.6666667), atol=1e-5)
    a.free()
    b.free()


def test_trsm_unit_diagonal() raises:
    # unit upper left: [[1,2],[0,1]]*X=[[5,6],[3,4]] -> X=[[5-6,6-8],[3,4]]=wait
    # col-major B=[5,3,6,4], unit upper: b[0]/=1, b[0]-=2*b[1] etc
    # after: b[0]=5-2*3=-1, b[1]=3, b[2]=6-2*4=-2, b[3]=4
    var a = alloc[Scalar[DType.float64]](4)
    var b = alloc[Scalar[DType.float64]](4)
    a[0] = 99
    a[1] = 0
    a[2] = 2
    a[3] = 99
    b[0] = 5
    b[1] = 3
    b[2] = 6
    b[3] = 4
    trsm[DType.float64]("L", "U", "N", "U", 2, 2, Float64(1.0), a, 2, b, 2)
    assert_almost_equal(b[0], Float64(-1.0), atol=1e-6)
    assert_almost_equal(b[1], Float64(3.0), atol=1e-6)
    assert_almost_equal(b[2], Float64(-2.0), atol=1e-6)
    assert_almost_equal(b[3], Float64(4.0), atol=1e-6)
    a.free()
    b.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
