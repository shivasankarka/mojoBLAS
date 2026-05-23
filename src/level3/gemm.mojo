# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
General Matrix-Matrix Operations (`level3.gemm`)
=============================================
Provides general matrix-matrix operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import vectorize, parallelize
from std.sys.info import simd_width_of
from std.memory import stack_allocation
from ._tuning import (
    GEMM_V6_NR,
    GEMM_V6_TK,
    GEMM_V6_MC,
    GEMM_V6_PAR_THRESHOLD,
    GEMM_V7_NR,
    GEMM_V7_TK,
    GEMM_V7_MC,
    GEMM_V7_PAR_THRESHOLD,
    GEMM_DISPATCH_THRESHOLD,
)


def gemm[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    trans_a: String,
    trans_b: String,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[dtype, origin_c],
    ldc: Int,
):
    """
    Performs the matrix-matrix operation C := alpha*A*B + beta*C.

    Parameters:
        mut_a: Mutability of pointer a.
        mut_b: Mutability of pointer b.
        origin_a: Memory origin of a.
        origin_b: Memory origin of b.
        origin_c: Memory origin of c (mutable, input/output).
        dtype: Element data type.

    Args:
        trans_a: 'N'/'n' for no transpose, 'T'/'t' or 'C'/'c' for transpose.
        trans_b: 'N'/'n' for no transpose, 'T'/'t' or 'C'/'c' for transpose.
        m: Rows of C and of A (if not transposed).
        n: Columns of C and of B (if not transposed).
        k: Columns of A (if not transposed) / rows of B (if not transposed).
        alpha: Scalar multiplier for A*B.
        a: Pointer to A.
        lda: Leading dimension of A.
        b: Pointer to B.
        ldb: Leading dimension of B.
        beta: Scalar multiplier for C on input.
        c: Pointer to C (input/output).
        ldc: Leading dimension of C.
    """
    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if no_trans_a and no_trans_b:
        if n < GEMM_DISPATCH_THRESHOLD:
            gemm_v3[dtype](trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        else:
            gemm_v7[dtype](trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        return
    _gemm_naive[dtype](trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)


def _gemm_naive[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    trans_a: String,
    trans_b: String,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[dtype, origin_c],
    ldc: Int,
):
    """
    Reference GEMM: C := alpha*A*B + beta*C (all transpose variants).

    Column-major, BLAS-compatible reference implementation used as the fallback
    for transposed variants not handled by the optimized kernels.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        mut_b: Indicates whether the pointer b is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b.
        origin_c: Memory origin of the pointer c (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        trans_a: Specifies whether A is transposed ('N' for no, 'T' or 'C' for yes).
        trans_b: Specifies whether B is transposed ('N' for no, 'T' or 'C' for yes).
        m: The number of rows of the matrix C and of the matrix A if not transposed.
        n: The number of columns of the matrix C and of the matrix B if not transposed.
        k: The number of columns of A if not transposed, or rows if transposed.
        alpha: The scalar multiplier for the matrix product A*B.
        a: A pointer to the first element of the matrix A.
        lda: The leading dimension of the matrix A.
        b: A pointer to the first element of the matrix B.
        ldb: The leading dimension of the matrix B.
        beta: The scalar multiplier for the matrix C.
        c: A pointer to the first element of the matrix C (input/output).
        ldc: The leading dimension of the matrix C.
    """
    var info: Int = 0
    if (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 1
    elif (
        trans_b != "N"
        and trans_b != "n"
        and trans_b != "T"
        and trans_b != "t"
        and trans_b != "C"
        and trans_b != "c"
    ):
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < max(1, m if (trans_a == "N" or trans_a == "n") else k):
        info = 8
    elif ldb < max(1, k if (trans_b == "N" or trans_b == "n") else n):
        info = 10
    elif ldc < max(1, m):
        info = 13

    if info != 0:
        print("gemm: Info", info)
        return

    if m == 0 or n == 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()

    if alpha == 0 or k == 0:
        if beta == 0:
            for j in range(n):
                var cj = c + j * ldc

                def zero_col[width: Int](i: Int) {cj}:
                    cj.store[width=width](i, SIMD[dtype, width](0))

                vectorize[simd_width](m, zero_col)
        elif beta != 1:
            for j in range(n):
                var cj = c + j * ldc

                def scale_col[width: Int](i: Int) {cj, beta}:
                    cj.store[width=width](i, beta * cj.load[width=width](i))

                vectorize[simd_width](m, scale_col)
        return

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"

    comptime PAR_THRESHOLD: Int = 64

    if no_trans_a:
        if no_trans_b:
            @parameter
            def gemm_nn_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_nn[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_nn)
                elif beta != 1:

                    def scale_nn[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_nn)
                for l in range(k):
                    if b[l + j * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[l + j * ldb]
                        var al = a + l * lda

                        def axpy_nn[width: Int](i: Int) {cj, al, temp}:
                            cj.store[width=width](
                                i,
                                cj.load[width=width](i)
                                + temp * al.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_nn)

            if n >= PAR_THRESHOLD:
                parallelize[gemm_nn_col](n)
            else:
                for j in range(n):
                    gemm_nn_col(j)
        else:
            @parameter
            def gemm_nt_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_nt[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_nt)
                elif beta != 1:

                    def scale_nt[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_nt)
                for l in range(k):
                    if b[j + l * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[j + l * ldb]
                        var al = a + l * lda

                        def axpy_nt[width: Int](i: Int) {cj, al, temp}:
                            cj.store[width=width](
                                i,
                                cj.load[width=width](i)
                                + temp * al.load[width=width](i),
                            )

                        vectorize[simd_width](m, axpy_nt)

            if n >= PAR_THRESHOLD:
                parallelize[gemm_nt_col](n)
            else:
                for j in range(n):
                    gemm_nt_col(j)
    else:
        if no_trans_b:
            @parameter
            def gemm_tn_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_tn[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_tn)
                elif beta != 1:

                    def scale_tn[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_tn)
                for l in range(k):
                    if b[l + j * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[l + j * ldb]
                        var al = a + l
                        for i in range(m):
                            cj[i] = cj[i] + temp * al[i * lda]

            if n >= PAR_THRESHOLD:
                parallelize[gemm_tn_col](n)
            else:
                for j in range(n):
                    gemm_tn_col(j)
        else:
            @parameter
            def gemm_tt_col(j: Int):
                var cj = c + j * ldc
                if beta == 0:

                    def zero_tt[width: Int](i: Int) {cj}:
                        cj.store[width=width](i, SIMD[dtype, width](0))

                    vectorize[simd_width](m, zero_tt)
                elif beta != 1:

                    def scale_tt[width: Int](i: Int) {cj, beta}:
                        cj.store[width=width](i, beta * cj.load[width=width](i))

                    vectorize[simd_width](m, scale_tt)
                for l in range(k):
                    if b[j + l * ldb] != 0:
                        var temp: Scalar[dtype] = alpha * b[j + l * ldb]
                        var al = a + l
                        for i in range(m):
                            cj[i] = cj[i] + temp * al[i * lda]

            if n >= PAR_THRESHOLD:
                parallelize[gemm_tt_col](n)
            else:
                for j in range(n):
                    gemm_tt_col(j)

    return


comptime _V3_MR: Int = simd_width_of[DType.float32]()
comptime _V3_NR: Int = 8
comptime _V3_TK: Int = 256


def gemm_v3[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    trans_a: String,
    trans_b: String,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[dtype, origin_c],
    ldc: Int,
):
    """
    Packed-B register micro-kernel GEMM: C := alpha*A*B + beta*C.
    """
    var info: Int = 0
    if (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 1
    elif (
        trans_b != "N"
        and trans_b != "n"
        and trans_b != "T"
        and trans_b != "t"
        and trans_b != "C"
        and trans_b != "c"
    ):
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < max(1, m if (trans_a == "N" or trans_a == "n") else k):
        info = 8
    elif ldb < max(1, k if (trans_b == "N" or trans_b == "n") else n):
        info = 10
    elif ldc < max(1, m):
        info = 13
    if info != 0:
        print("gemm_v3: Info", info)
        return

    if m == 0 or n == 0:
        return

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        gemm[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    comptime MR = _V3_MR
    comptime NR = _V3_NR
    comptime TK = _V3_TK
    comptime PAR_THRESHOLD: Int = 8

    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v3[width: Int](i: Int) {cj}:
                cj.store[width=width](i, SIMD[dtype, width](0))

            vectorize[simd_width_of[dtype]()](m, zero_v3)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v3[width: Int](i: Int) {cj, beta}:
                cj.store[width=width](i, beta * cj.load[width=width](i))

            vectorize[simd_width_of[dtype]()](m, scale_v3)

    if alpha == 0 or k == 0:
        return

    @parameter
    def col_group(jr_block: Int):
        var j0 = jr_block * NR
        if j0 >= n:
            return
        var jlen = min(NR, n - j0)

        var b_pack = stack_allocation[TK * NR, dtype]()

        for k0 in range(0, k, TK):
            var klen = min(TK, k - k0)

            for l in range(klen):
                comptime for r in range(NR):
                    b_pack[l * NR + r] = (
                        b[(k0 + l) + (j0 + r) * ldb] if r < jlen else 0
                    )

            var ir = 0
            while ir + MR <= m:
                var acc0 = SIMD[dtype, MR](0)
                var acc1 = SIMD[dtype, MR](0)
                var acc2 = SIMD[dtype, MR](0)
                var acc3 = SIMD[dtype, MR](0)
                var acc4 = SIMD[dtype, MR](0)
                var acc5 = SIMD[dtype, MR](0)
                var acc6 = SIMD[dtype, MR](0)
                var acc7 = SIMD[dtype, MR](0)

                for l in range(klen):
                    var av = alpha * (a + (k0 + l) * lda + ir).load[width=MR]()
                    acc0 = av * b_pack[l * NR + 0] + acc0
                    acc1 = av * b_pack[l * NR + 1] + acc1
                    acc2 = av * b_pack[l * NR + 2] + acc2
                    acc3 = av * b_pack[l * NR + 3] + acc3
                    acc4 = av * b_pack[l * NR + 4] + acc4
                    acc5 = av * b_pack[l * NR + 5] + acc5
                    acc6 = av * b_pack[l * NR + 6] + acc6
                    acc7 = av * b_pack[l * NR + 7] + acc7

                var cp0 = c + (j0 + 0) * ldc + ir
                cp0.store[width=MR](acc0 + cp0.load[width=MR]())
                if jlen > 1:
                    var cp1 = c + (j0 + 1) * ldc + ir
                    cp1.store[width=MR](acc1 + cp1.load[width=MR]())
                if jlen > 2:
                    var cp2 = c + (j0 + 2) * ldc + ir
                    cp2.store[width=MR](acc2 + cp2.load[width=MR]())
                if jlen > 3:
                    var cp3 = c + (j0 + 3) * ldc + ir
                    cp3.store[width=MR](acc3 + cp3.load[width=MR]())
                if jlen > 4:
                    var cp4 = c + (j0 + 4) * ldc + ir
                    cp4.store[width=MR](acc4 + cp4.load[width=MR]())
                if jlen > 5:
                    var cp5 = c + (j0 + 5) * ldc + ir
                    cp5.store[width=MR](acc5 + cp5.load[width=MR]())
                if jlen > 6:
                    var cp6 = c + (j0 + 6) * ldc + ir
                    cp6.store[width=MR](acc6 + cp6.load[width=MR]())
                if jlen > 7:
                    var cp7 = c + (j0 + 7) * ldc + ir
                    cp7.store[width=MR](acc7 + cp7.load[width=MR]())
                ir += MR

            while ir < m:
                for jc in range(jlen):
                    var sum: Scalar[dtype] = 0
                    for l in range(klen):
                        sum += a[ir + (k0 + l) * lda] * b_pack[l * NR + jc]
                    c[ir + (j0 + jc) * ldc] += alpha * sum
                ir += 1

    var n_groups_v3 = (n + NR - 1) // NR
    if n_groups_v3 >= PAR_THRESHOLD:
        parallelize[col_group](n_groups_v3)
    else:
        for jg in range(n_groups_v3):
            col_group(jg)


comptime _V6_NR: Int = GEMM_V6_NR
comptime _V6_TK: Int = GEMM_V6_TK
comptime _V6_MC: Int = GEMM_V6_MC


def gemm_v6[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    trans_a: String,
    trans_b: String,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[dtype, origin_c],
    ldc: Int,
):
    """
    Shared-A-pack + MC-blocked GEMM: C := alpha*A*B + beta*C (NN path, generic dtype).
    """
    var info: Int = 0
    if (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 1
    elif (
        trans_b != "N"
        and trans_b != "n"
        and trans_b != "T"
        and trans_b != "t"
        and trans_b != "C"
        and trans_b != "c"
    ):
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < max(1, m if (trans_a == "N" or trans_a == "n") else k):
        info = 8
    elif ldb < max(1, k if (trans_b == "N" or trans_b == "n") else n):
        info = 10
    elif ldc < max(1, m):
        info = 13
    if info != 0:
        print("gemm_v6: Info", info)
        return

    if m == 0 or n == 0:
        return

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        gemm[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    comptime MR: Int = simd_width_of[dtype]()
    comptime NR: Int = _V6_NR
    comptime TK: Int = _V6_TK
    comptime MC: Int = _V6_MC
    comptime PAR_THRESHOLD: Int = GEMM_V6_PAR_THRESHOLD

    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v6[width: Int](i: Int) {cj}:
                cj.store[width=width](i, SIMD[dtype, width](0))

            vectorize[MR](m, zero_v6)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v6[width: Int](i: Int) {cj, beta}:
                cj.store[width=width](i, beta * cj.load[width=width](i))

            vectorize[MR](m, scale_v6)

    if alpha == 0 or k == 0:
        return

    var a_pack = alloc[Scalar[dtype]](MC * TK)

    for mc0 in range(0, m, MC):
        var mc_len = min(MC, m - mc0)

        for k0 in range(0, k, TK):
            var klen = min(TK, k - k0)

            for ir in range(0, mc_len, MR):
                var row_count = min(MR, mc_len - ir)
                for l in range(klen):
                    if row_count == MR:
                        (a_pack + ir * TK + l * MR).store[width=MR](
                            alpha
                            * (a + (k0 + l) * lda + mc0 + ir).load[width=MR]()
                        )
                    else:
                        for rr in range(row_count):
                            a_pack[ir * TK + l * MR + rr] = (
                                alpha * a[mc0 + ir + rr + (k0 + l) * lda]
                            )
                        for rr in range(row_count, MR):
                            a_pack[ir * TK + l * MR + rr] = 0

            @parameter
            def col_group_v6(jr_block: Int):
                var j0 = jr_block * NR
                if j0 >= n:
                    return
                var jlen = min(NR, n - j0)

                var b_pack = stack_allocation[TK * NR, dtype]()
                for l in range(klen):
                    comptime for r in range(NR):
                        b_pack[l * NR + r] = (
                            b[(k0 + l) + (j0 + r) * ldb] if r < jlen else 0
                        )

                var ir = 0
                while ir + MR <= mc_len:
                    var acc0 = SIMD[dtype, MR](0)
                    var acc1 = SIMD[dtype, MR](0)
                    var acc2 = SIMD[dtype, MR](0)
                    var acc3 = SIMD[dtype, MR](0)
                    var acc4 = SIMD[dtype, MR](0)
                    var acc5 = SIMD[dtype, MR](0)
                    var acc6 = SIMD[dtype, MR](0)
                    var acc7 = SIMD[dtype, MR](0)
                    var acc8 = SIMD[dtype, MR](0)
                    var acc9 = SIMD[dtype, MR](0)
                    var acc10 = SIMD[dtype, MR](0)
                    var acc11 = SIMD[dtype, MR](0)
                    var acc12 = SIMD[dtype, MR](0)
                    var acc13 = SIMD[dtype, MR](0)
                    var acc14 = SIMD[dtype, MR](0)
                    var acc15 = SIMD[dtype, MR](0)

                    for l in range(klen):
                        var av = (
                            alpha * (a_pack + ir * TK + l * MR).load[width=MR]()
                        )
                        acc0 = av * b_pack[l * NR + 0] + acc0
                        acc1 = av * b_pack[l * NR + 1] + acc1
                        acc2 = av * b_pack[l * NR + 2] + acc2
                        acc3 = av * b_pack[l * NR + 3] + acc3
                        acc4 = av * b_pack[l * NR + 4] + acc4
                        acc5 = av * b_pack[l * NR + 5] + acc5
                        acc6 = av * b_pack[l * NR + 6] + acc6
                        acc7 = av * b_pack[l * NR + 7] + acc7
                        acc8 = av * b_pack[l * NR + 8] + acc8
                        acc9 = av * b_pack[l * NR + 9] + acc9
                        acc10 = av * b_pack[l * NR + 10] + acc10
                        acc11 = av * b_pack[l * NR + 11] + acc11
                        acc12 = av * b_pack[l * NR + 12] + acc12
                        acc13 = av * b_pack[l * NR + 13] + acc13
                        acc14 = av * b_pack[l * NR + 14] + acc14
                        acc15 = av * b_pack[l * NR + 15] + acc15

                    var row = mc0 + ir
                    var cp0 = c + (j0 + 0) * ldc + row
                    cp0.store[width=MR](acc0 + cp0.load[width=MR]())
                    if jlen > 1:
                        var p = c + (j0 + 1) * ldc + row
                        p.store[width=MR](acc1 + p.load[width=MR]())
                    if jlen > 2:
                        var p = c + (j0 + 2) * ldc + row
                        p.store[width=MR](acc2 + p.load[width=MR]())
                    if jlen > 3:
                        var p = c + (j0 + 3) * ldc + row
                        p.store[width=MR](acc3 + p.load[width=MR]())
                    if jlen > 4:
                        var p = c + (j0 + 4) * ldc + row
                        p.store[width=MR](acc4 + p.load[width=MR]())
                    if jlen > 5:
                        var p = c + (j0 + 5) * ldc + row
                        p.store[width=MR](acc5 + p.load[width=MR]())
                    if jlen > 6:
                        var p = c + (j0 + 6) * ldc + row
                        p.store[width=MR](acc6 + p.load[width=MR]())
                    if jlen > 7:
                        var p = c + (j0 + 7) * ldc + row
                        p.store[width=MR](acc7 + p.load[width=MR]())
                    if jlen > 8:
                        var p = c + (j0 + 8) * ldc + row
                        p.store[width=MR](acc8 + p.load[width=MR]())
                    if jlen > 9:
                        var p = c + (j0 + 9) * ldc + row
                        p.store[width=MR](acc9 + p.load[width=MR]())
                    if jlen > 10:
                        var p = c + (j0 + 10) * ldc + row
                        p.store[width=MR](acc10 + p.load[width=MR]())
                    if jlen > 11:
                        var p = c + (j0 + 11) * ldc + row
                        p.store[width=MR](acc11 + p.load[width=MR]())
                    if jlen > 12:
                        var p = c + (j0 + 12) * ldc + row
                        p.store[width=MR](acc12 + p.load[width=MR]())
                    if jlen > 13:
                        var p = c + (j0 + 13) * ldc + row
                        p.store[width=MR](acc13 + p.load[width=MR]())
                    if jlen > 14:
                        var p = c + (j0 + 14) * ldc + row
                        p.store[width=MR](acc14 + p.load[width=MR]())
                    if jlen > 15:
                        var p = c + (j0 + 15) * ldc + row
                        p.store[width=MR](acc15 + p.load[width=MR]())
                    ir += MR

                while ir < mc_len:
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += a_pack[ir * TK + l * MR] * b_pack[l * NR + jc]
                        c[(mc0 + ir) + (j0 + jc) * ldc] += alpha * s
                    ir += 1

            var n_groups_v6 = (n + NR - 1) // NR
            if n_groups_v6 >= PAR_THRESHOLD:
                parallelize[col_group_v6](n_groups_v6)
            else:
                for jg in range(n_groups_v6):
                    col_group_v6(jg)

    a_pack.free()


comptime _V7_NR: Int = GEMM_V7_NR
comptime _V7_TK: Int = GEMM_V7_TK
comptime _V7_MC: Int = GEMM_V7_MC


def gemm_v7[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    trans_a: String,
    trans_b: String,
    m: Int,
    n: Int,
    k: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
    beta: Scalar[dtype],
    c: BLASPtr[dtype, origin_c],
    ldc: Int,
):
    """
    blocked shared-A-pack GEMM: C := alpha*A*B + beta*C.
    """
    var info: Int = 0
    if (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 1
    elif (
        trans_b != "N"
        and trans_b != "n"
        and trans_b != "T"
        and trans_b != "t"
        and trans_b != "C"
        and trans_b != "c"
    ):
        info = 2
    elif m < 0:
        info = 3
    elif n < 0:
        info = 4
    elif k < 0:
        info = 5
    elif lda < max(1, m if (trans_a == "N" or trans_a == "n") else k):
        info = 8
    elif ldb < max(1, k if (trans_b == "N" or trans_b == "n") else n):
        info = 10
    elif ldc < max(1, m):
        info = 13
    if info != 0:
        print("gemm_v7: Info", info)
        return

    if m == 0 or n == 0:
        return

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        gemm[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    comptime MR: Int = simd_width_of[dtype]()
    comptime NR: Int = _V7_NR
    comptime TK: Int = _V7_TK
    comptime MC: Int = _V7_MC
    comptime PAR_THRESHOLD: Int = GEMM_V7_PAR_THRESHOLD

    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v7[width: Int](i: Int) {cj}:
                cj.store[width=width](i, SIMD[dtype, width](0))

            vectorize[MR](m, zero_v7)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v7[width: Int](i: Int) {cj, beta}:
                cj.store[width=width](i, beta * cj.load[width=width](i))

            vectorize[MR](m, scale_v7)

    if alpha == 0 or k == 0:
        return

    var a_pack = alloc[Scalar[dtype]](MC * TK)

    for mc0 in range(0, m, MC):
        var mc_len = min(MC, m - mc0)

        for k0 in range(0, k, TK):
            var klen = min(TK, k - k0)

            for ir in range(0, mc_len, MR):
                var row_count = min(MR, mc_len - ir)
                for l in range(klen):
                    if row_count == MR:
                        (a_pack + ir * TK + l * MR).store[width=MR](
                            (a + (k0 + l) * lda + mc0 + ir).load[width=MR]()
                        )
                    else:
                        for rr in range(row_count):
                            a_pack[ir * TK + l * MR + rr] = a[
                                mc0 + ir + rr + (k0 + l) * lda
                            ]
                        for rr in range(row_count, MR):
                            a_pack[ir * TK + l * MR + rr] = 0

            @parameter
            def col_group_v7(jr_block: Int):
                var j0 = jr_block * NR
                if j0 >= n:
                    return
                var jlen = min(NR, n - j0)
                var b_pack = stack_allocation[TK * NR, dtype]()

                for l in range(klen):
                    comptime for r in range(NR):
                        b_pack[l * NR + r] = (
                            b[(k0 + l) + (j0 + r) * ldb] if r < jlen else 0
                        )

                var ir = 0
                if jlen == NR:
                    while ir + MR <= mc_len:
                        var acc0 = SIMD[dtype, MR](0)
                        var acc1 = SIMD[dtype, MR](0)
                        var acc2 = SIMD[dtype, MR](0)
                        var acc3 = SIMD[dtype, MR](0)
                        var acc4 = SIMD[dtype, MR](0)
                        var acc5 = SIMD[dtype, MR](0)
                        var acc6 = SIMD[dtype, MR](0)
                        var acc7 = SIMD[dtype, MR](0)
                        var acc8 = SIMD[dtype, MR](0)
                        var acc9 = SIMD[dtype, MR](0)
                        var acc10 = SIMD[dtype, MR](0)
                        var acc11 = SIMD[dtype, MR](0)
                        var acc12 = SIMD[dtype, MR](0)
                        var acc13 = SIMD[dtype, MR](0)
                        var acc14 = SIMD[dtype, MR](0)
                        var acc15 = SIMD[dtype, MR](0)

                        for l in range(klen):
                            var av = (a_pack + ir * TK + l * MR).load[width=MR]()
                            acc0 = av * b_pack[l * NR + 0] + acc0
                            acc1 = av * b_pack[l * NR + 1] + acc1
                            acc2 = av * b_pack[l * NR + 2] + acc2
                            acc3 = av * b_pack[l * NR + 3] + acc3
                            acc4 = av * b_pack[l * NR + 4] + acc4
                            acc5 = av * b_pack[l * NR + 5] + acc5
                            acc6 = av * b_pack[l * NR + 6] + acc6
                            acc7 = av * b_pack[l * NR + 7] + acc7
                            acc8 = av * b_pack[l * NR + 8] + acc8
                            acc9 = av * b_pack[l * NR + 9] + acc9
                            acc10 = av * b_pack[l * NR + 10] + acc10
                            acc11 = av * b_pack[l * NR + 11] + acc11
                            acc12 = av * b_pack[l * NR + 12] + acc12
                            acc13 = av * b_pack[l * NR + 13] + acc13
                            acc14 = av * b_pack[l * NR + 14] + acc14
                            acc15 = av * b_pack[l * NR + 15] + acc15

                        var row = mc0 + ir
                        var p0 = c + (j0 + 0) * ldc + row
                        p0.store[width=MR](acc0 + p0.load[width=MR]())
                        var p1 = c + (j0 + 1) * ldc + row
                        p1.store[width=MR](acc1 + p1.load[width=MR]())
                        var p2 = c + (j0 + 2) * ldc + row
                        p2.store[width=MR](acc2 + p2.load[width=MR]())
                        var p3 = c + (j0 + 3) * ldc + row
                        p3.store[width=MR](acc3 + p3.load[width=MR]())
                        var p4 = c + (j0 + 4) * ldc + row
                        p4.store[width=MR](acc4 + p4.load[width=MR]())
                        var p5 = c + (j0 + 5) * ldc + row
                        p5.store[width=MR](acc5 + p5.load[width=MR]())
                        var p6 = c + (j0 + 6) * ldc + row
                        p6.store[width=MR](acc6 + p6.load[width=MR]())
                        var p7 = c + (j0 + 7) * ldc + row
                        p7.store[width=MR](acc7 + p7.load[width=MR]())
                        var p8 = c + (j0 + 8) * ldc + row
                        p8.store[width=MR](acc8 + p8.load[width=MR]())
                        var p9 = c + (j0 + 9) * ldc + row
                        p9.store[width=MR](acc9 + p9.load[width=MR]())
                        var p10 = c + (j0 + 10) * ldc + row
                        p10.store[width=MR](acc10 + p10.load[width=MR]())
                        var p11 = c + (j0 + 11) * ldc + row
                        p11.store[width=MR](acc11 + p11.load[width=MR]())
                        var p12 = c + (j0 + 12) * ldc + row
                        p12.store[width=MR](acc12 + p12.load[width=MR]())
                        var p13 = c + (j0 + 13) * ldc + row
                        p13.store[width=MR](acc13 + p13.load[width=MR]())
                        var p14 = c + (j0 + 14) * ldc + row
                        p14.store[width=MR](acc14 + p14.load[width=MR]())
                        var p15 = c + (j0 + 15) * ldc + row
                        p15.store[width=MR](acc15 + p15.load[width=MR]())
                        ir += MR
                else:
                    while ir + MR <= mc_len:
                        var acc0 = SIMD[dtype, MR](0)
                        var acc1 = SIMD[dtype, MR](0)
                        var acc2 = SIMD[dtype, MR](0)
                        var acc3 = SIMD[dtype, MR](0)
                        var acc4 = SIMD[dtype, MR](0)
                        var acc5 = SIMD[dtype, MR](0)
                        var acc6 = SIMD[dtype, MR](0)
                        var acc7 = SIMD[dtype, MR](0)
                        var acc8 = SIMD[dtype, MR](0)
                        var acc9 = SIMD[dtype, MR](0)
                        var acc10 = SIMD[dtype, MR](0)
                        var acc11 = SIMD[dtype, MR](0)
                        var acc12 = SIMD[dtype, MR](0)
                        var acc13 = SIMD[dtype, MR](0)
                        var acc14 = SIMD[dtype, MR](0)
                        var acc15 = SIMD[dtype, MR](0)

                        for l in range(klen):
                            var av = (a_pack + ir * TK + l * MR).load[width=MR]()
                            acc0 = av * b_pack[l * NR + 0] + acc0
                            acc1 = av * b_pack[l * NR + 1] + acc1
                            acc2 = av * b_pack[l * NR + 2] + acc2
                            acc3 = av * b_pack[l * NR + 3] + acc3
                            acc4 = av * b_pack[l * NR + 4] + acc4
                            acc5 = av * b_pack[l * NR + 5] + acc5
                            acc6 = av * b_pack[l * NR + 6] + acc6
                            acc7 = av * b_pack[l * NR + 7] + acc7
                            acc8 = av * b_pack[l * NR + 8] + acc8
                            acc9 = av * b_pack[l * NR + 9] + acc9
                            acc10 = av * b_pack[l * NR + 10] + acc10
                            acc11 = av * b_pack[l * NR + 11] + acc11
                            acc12 = av * b_pack[l * NR + 12] + acc12
                            acc13 = av * b_pack[l * NR + 13] + acc13
                            acc14 = av * b_pack[l * NR + 14] + acc14
                            acc15 = av * b_pack[l * NR + 15] + acc15

                        var row = mc0 + ir
                        var cp0 = c + (j0 + 0) * ldc + row
                        cp0.store[width=MR](acc0 + cp0.load[width=MR]())
                        if jlen > 1:
                            var p = c + (j0 + 1) * ldc + row
                            p.store[width=MR](acc1 + p.load[width=MR]())
                        if jlen > 2:
                            var p = c + (j0 + 2) * ldc + row
                            p.store[width=MR](acc2 + p.load[width=MR]())
                        if jlen > 3:
                            var p = c + (j0 + 3) * ldc + row
                            p.store[width=MR](acc3 + p.load[width=MR]())
                        if jlen > 4:
                            var p = c + (j0 + 4) * ldc + row
                            p.store[width=MR](acc4 + p.load[width=MR]())
                        if jlen > 5:
                            var p = c + (j0 + 5) * ldc + row
                            p.store[width=MR](acc5 + p.load[width=MR]())
                        if jlen > 6:
                            var p = c + (j0 + 6) * ldc + row
                            p.store[width=MR](acc6 + p.load[width=MR]())
                        if jlen > 7:
                            var p = c + (j0 + 7) * ldc + row
                            p.store[width=MR](acc7 + p.load[width=MR]())
                        if jlen > 8:
                            var p = c + (j0 + 8) * ldc + row
                            p.store[width=MR](acc8 + p.load[width=MR]())
                        if jlen > 9:
                            var p = c + (j0 + 9) * ldc + row
                            p.store[width=MR](acc9 + p.load[width=MR]())
                        if jlen > 10:
                            var p = c + (j0 + 10) * ldc + row
                            p.store[width=MR](acc10 + p.load[width=MR]())
                        if jlen > 11:
                            var p = c + (j0 + 11) * ldc + row
                            p.store[width=MR](acc11 + p.load[width=MR]())
                        if jlen > 12:
                            var p = c + (j0 + 12) * ldc + row
                            p.store[width=MR](acc12 + p.load[width=MR]())
                        if jlen > 13:
                            var p = c + (j0 + 13) * ldc + row
                            p.store[width=MR](acc13 + p.load[width=MR]())
                        if jlen > 14:
                            var p = c + (j0 + 14) * ldc + row
                            p.store[width=MR](acc14 + p.load[width=MR]())
                        if jlen > 15:
                            var p = c + (j0 + 15) * ldc + row
                            p.store[width=MR](acc15 + p.load[width=MR]())
                        ir += MR

                while ir < mc_len:
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += a_pack[ir * TK + l * MR] * b_pack[l * NR + jc]
                        c[(mc0 + ir) + (j0 + jc) * ldc] += s
                    ir += 1

            var n_groups_v7 = (n + NR - 1) // NR
            if n_groups_v7 >= PAR_THRESHOLD:
                parallelize[col_group_v7](n_groups_v7)
            else:
                for jg in range(n_groups_v7):
                    col_group_v7(jg)

    a_pack.free()
