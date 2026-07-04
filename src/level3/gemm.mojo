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
from std.memory import stack_allocation, memset_zero
from std.sys._assembly import inlined_assembly
from ._tuning import (
    GEMM_V6_NR,
    GEMM_V6_TK,
    GEMM_V6_MC,
    GEMM_V6_PAR_THRESHOLD,
    GEMM_V7_NR,
    GEMM_V7_TK,
    GEMM_V7_MC,
    GEMM_V7_PAR_THRESHOLD,
    GEMM_V8_MC,
    GEMM_V8_TK,
    GEMM_V8_PAR_THRESHOLD,
    GEMM_V9_TILE,
    GEMM_V9_NZ,
    GEMM_V9_MC,
    GEMM_V9_TK,
    GEMM_V9_PAR_THRESHOLD,
    GEMM_V10_TILE,
    GEMM_V10_NZ,
    GEMM_V10_MC,
    GEMM_V10_TK,
    GEMM_V10_PAR_THRESHOLD,
    GEMM_V11_NZ,
    GEMM_V11_UK,
    GEMM_V11_MC,
    GEMM_V11_TK,
    GEMM_V11_ROW_PAR_THRESHOLD,
    GEMM_V11_SMALL_MC,
    GEMM_V11_PAR_THRESHOLD,
    GEMM_V12_MC,
    GEMM_V12_NC,
    GEMM_V12_TK,
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
        comptime if dtype == DType.float32:
            gemm_v11[dtype](
                trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
            )
            return
        if n < GEMM_DISPATCH_THRESHOLD:
            gemm_v3[dtype](
                trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
            )
        else:
            gemm_v7[dtype](
                trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
            )
        return
    _gemm_naive[dtype](
        trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
    )


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
    Blocked shared-A-pack GEMM: C := alpha*A*B + beta*C.
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
                            var av = (a_pack + ir * TK + l * MR).load[
                                width=MR
                            ]()
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
                            var av = (a_pack + ir * TK + l * MR).load[
                                width=MR
                            ]()
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


# ── AMX helpers (inline assembly, Apple M1/M2 only) ──────────────────────────


@always_inline
def _amx_set():
    inlined_assembly[
        "nop\nnop\nnop\n.word (0x201000 + (17 << 5) + 0)",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def _amx_clr():
    inlined_assembly[
        "nop\nnop\nnop\n.word (0x201000 + (17 << 5) + 1)",
        NoneType,
        constraints="~{memory}",
        has_side_effect=True,
    ]()


@always_inline
def _amx_ldx(gpr: Int):
    inlined_assembly[
        ".word (0x201000 + (0 << 5) + 0$0 - ((0$0 >> 4) * 6))",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](gpr)


@always_inline
def _amx_ldy(gpr: Int):
    inlined_assembly[
        ".word (0x201000 + (1 << 5) + 0$0 - ((0$0 >> 4) * 6))",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](gpr)


@always_inline
def _amx_stz(gpr: Int):
    inlined_assembly[
        ".word (0x201000 + (5 << 5) + 0$0 - ((0$0 >> 4) * 6))",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](gpr)


@always_inline
def _amx_fma32(gpr: Int):
    inlined_assembly[
        ".word (0x201000 + (12 << 5) + 0$0 - ((0$0 >> 4) * 6))",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](gpr)


# Tile mode fma32: Z[z*4::4][:] += outer(Y[y][:], X[x][:])
# clear_z=True zeroes the Z tile before accumulating.
@always_inline
def _amx_fma32_tile(z: Int, x: Int, y: Int, clear_z: Bool):
    var operand = (y << 6) | (x << 16) | (z << 20) | (1 << 27 if clear_z else 0)
    _amx_fma32(operand)


# Load 16 f32 elements from ptr into X[row] (row in 0..7).
@always_inline
def _amx_load_x_row[dtype: DType](ptr: BLASPtr[dtype, _], row: Int):
    _amx_ldx((row << 56) | Int(ptr))


@always_inline
def _amx_load_x_2rows[dtype: DType](ptr: BLASPtr[dtype, _], row: Int):
    _amx_ldx((row << 56) | (1 << 62) | Int(ptr))


# Load 16 f32 elements from ptr into Y[row].
@always_inline
def _amx_load_y_row[dtype: DType](ptr: BLASPtr[dtype, _], row: Int):
    _amx_ldy((row << 56) | Int(ptr))


@always_inline
def _amx_load_y_2rows[dtype: DType](ptr: BLASPtr[dtype, _], row: Int):
    _amx_ldy((row << 56) | (1 << 62) | Int(ptr))


# Store Z[z_offset] (16 f32 elements) to ptr.
@always_inline
def _amx_store_z_row[dtype: DType](ptr: BLASPtr[dtype, _], z_offset: Int):
    _amx_stz((z_offset << 56) | Int(ptr))


# ── gemm_v8: AMX-backed f32 GEMM, MC-blocked ─────────────────────────────────
#
# AMX register file (f32 mode):
#   X[8][16], Y[8][16] — input tiles, 16 f32 per row
#   Z[64][16]          — accumulator, 64 rows of 16 f32
#
# Micro-kernel: 16×16 output tile of C per AMX Z-tile.
#   For each k-step l:
#     load B[l, j0:j0+16] → X[0]    (one row of B panel)
#     load A[i0:i0+16, l] is column-major so A[:,l] is contiguous →
#       since A is packed row-major in a_pack, load a_pack row → Y[0]
#     fma32 tile mode: Z[0::4][:] += outer(Y[0], X[0])
#       (this computes a 16×16 outer product in one instruction)
#   After all k: store Z rows → C tile.
#
# Blocking: same MC/TK structure as v7.
#   a_pack[MC×TK] in heap, shared across all column groups.
#   b_pack[TK×16] stack-allocated per column group.
#   PAR_THRESHOLD controls when column-group loop parallelizes.

comptime _V8_TILE: Int = 16  # AMX f32 tile width = 16 elements
comptime _V8_MC: Int = GEMM_V8_MC
comptime _V8_TK: Int = GEMM_V8_TK


def gemm_v8[
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
    AMX-backed GEMM for f32 on Apple M1/M2: C := alpha*A*B + beta*C (NN path).

    Parameters:
        mut_a: Mutability of pointer a.
        mut_b: Mutability of pointer b.
        origin_a: Memory origin of a.
        origin_b: Memory origin of b.
        origin_c: Memory origin of c (mutable, input/output).
        dtype: Element data type (must be float32).

    Args:
        trans_a: 'N'/'n' for no transpose, 'T'/'t' or 'C'/'c' for transpose.
        trans_b: 'N'/'n' for no transpose, 'T'/'t' or 'C'/'c' for transpose.
        m: Rows of C and A.
        n: Columns of C and B.
        k: Columns of A / rows of B.
        alpha: Scalar multiplier for A*B.
        a: Pointer to A.
        lda: Leading dimension of A.
        b: Pointer to B.
        ldb: Leading dimension of B.
        beta: Scalar multiplier for C on input.
        c: Pointer to C (input/output).
        ldc: Leading dimension of C.
    """
    comptime assert dtype == DType.float32, "gemm_v8 requires float32"

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        _gemm_naive[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    if m == 0 or n == 0:
        return

    comptime TILE: Int = _V8_TILE  # 16 — AMX f32 row width
    comptime MC: Int = _V8_MC
    comptime TK: Int = _V8_TK
    comptime PAR_THRESHOLD: Int = GEMM_V8_PAR_THRESHOLD
    comptime MR: Int = simd_width_of[dtype]()  # for scalar tail (NEON width)

    # Apply beta to C
    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v8[width: Int](i: Int) {cj}:
                cj.store[width=width](i, SIMD[dtype, width](0))

            vectorize[MR](m, zero_v8)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v8[width: Int](i: Int) {cj, beta}:
                cj.store[width=width](i, beta * cj.load[width=width](i))

            vectorize[MR](m, scale_v8)

    if alpha == 0 or k == 0:
        return

    # a_pack layout: a_pack[ir_tile * TK * TILE + l * TILE + lane]
    # where ir_tile = ir // TILE, lane = 0..TILE-1
    # Each TILE-row of A packed as TILE contiguous f32 values per k-step.
    var a_pack = alloc[Scalar[dtype]](MC * TK)

    # Temporary aligned buffer for Z readback (16 f32 = one AMX Z row)
    var z_buf = stack_allocation[TILE * TILE, dtype, alignment=128]()

    for mc0 in range(0, m, MC):
        var mc_len = min(MC, m - mc0)

        for k0 in range(0, k, TK):
            var klen = min(TK, k - k0)

            # Pack A[mc0:mc0+mc_len, k0:k0+klen] into a_pack.
            # Layout: a_pack[(ir//TILE)*TK*TILE + l*TILE + (ir%TILE)]
            # so that for a given tile row and k-step, 16 elements are contiguous.
            for ir_tile in range(0, mc_len, TILE):
                var row_count = min(TILE, mc_len - ir_tile)
                var base = ir_tile * TK
                for l in range(klen):
                    for r in range(row_count):
                        a_pack[base + l * TILE + r] = (
                            alpha * a[mc0 + ir_tile + r + (k0 + l) * lda]
                        )
                    for r in range(row_count, TILE):
                        a_pack[base + l * TILE + r] = 0

            @parameter
            def col_group_v8(jr_block: Int):
                var j0 = jr_block * TILE
                if j0 >= n:
                    return
                var jlen = min(TILE, n - j0)

                # Pack B[k0:k0+klen, j0:j0+jlen] into b_pack (TILE wide, zero-padded).
                var b_pack = stack_allocation[TK * TILE, dtype, alignment=128]()
                for l in range(klen):
                    for r in range(jlen):
                        b_pack[l * TILE + r] = b[(k0 + l) + (j0 + r) * ldb]
                    for r in range(jlen, TILE):
                        b_pack[l * TILE + r] = 0

                # AMX micro-kernel: 16×16 tile of C at a time.
                var ir_tile = 0
                while ir_tile + TILE <= mc_len:
                    var a_base = ir_tile * TK
                    _amx_set()  # enable AMX, clear Z

                    # Accumulate over k panel
                    for l in range(klen):
                        # Load row of packed A (16 f32) into Y[0]
                        _amx_load_y_row[dtype](a_pack + a_base + l * TILE, 0)
                        # Load row of packed B (16 f32) into X[0]
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        # Tile FMA: Z[0::4][:] += outer(Y[0], X[0]) — 16×16 outer product
                        _amx_fma32_tile(0, 0, 0, False)

                    # Store Z rows to z_buf then scatter to C (column-major)
                    # Z tile mode lays out result as: Z[row*4][col] for row in 0..15, col in 0..15
                    # i.e. Z[0], Z[4], Z[8], ... Z[60] hold rows 0..15 of the 16×16 result.
                    comptime for row in range(TILE):
                        _amx_store_z_row[dtype](z_buf + row * TILE, row * 4)

                    _amx_clr()

                    # Scatter z_buf to C: z_buf[row*TILE + col] → C[mc0+ir_tile+row, j0+col]
                    for col in range(jlen):
                        for row in range(TILE):
                            c[
                                (mc0 + ir_tile + row) + (j0 + col) * ldc
                            ] += z_buf[row * TILE + col]

                    ir_tile += TILE

                # Scalar tail for remaining rows (< TILE).
                # a_pack layout: [(ir//TILE)*TK*TILE + l*TILE + (ir%TILE)]
                while ir_tile < mc_len:
                    var tile_base = (ir_tile // TILE) * TK * TILE
                    var lane = ir_tile % TILE
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += (
                                a_pack[tile_base + l * TILE + lane]
                                * b_pack[l * TILE + jc]
                            )
                        c[(mc0 + ir_tile) + (j0 + jc) * ldc] += s
                    ir_tile += 1

            var n_groups_v8 = (n + TILE - 1) // TILE
            if n_groups_v8 >= PAR_THRESHOLD:
                parallelize[col_group_v8](n_groups_v8)
            else:
                for jg in range(n_groups_v8):
                    col_group_v8(jg)

    a_pack.free()


# ── gemm_v9: AMX f32, 4-Z-tile micro-kernel, direct C writeback ───────────────
#
# Improvements over v8:
#   1. 4 Z tiles in flight per column group (Z[0..3] → 4 groups of 16 A-rows).
#      Per k-step: load B row → X[0], load 4 A rows → Y[0..3],
#      issue 4 fma32_tile instructions. 4× more compute per set/clr pair.
#   2. One _amx_set/_amx_clr per NZ*TILE=64-row block, not per 16-row block.
#   3. Z rows stored directly into a row-major c_tile buffer, then each
#      C column written with SIMD loads+stores — no element-wise scatter.

comptime _V9_TILE: Int = GEMM_V9_TILE  # 16
comptime _V9_NZ: Int = GEMM_V9_NZ  # 4  — Z tiles per column group
comptime _V9_MR: Int = _V9_NZ * _V9_TILE  # 64 — rows per AMX block
comptime _V9_MC: Int = GEMM_V9_MC  # 256
comptime _V9_TK: Int = GEMM_V9_TK  # 256


def gemm_v9[
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
    AMX f32 GEMM v9 for Apple M1/M2: C := alpha*A*B + beta*C (NN path).

    Uses 4 simultaneous AMX Z tiles (64 rows × 16 cols of C per block),
    one _amx_set/_amx_clr per 64-row block, and direct SIMD writeback.

    Parameters:
        mut_a: Mutability of pointer a.
        mut_b: Mutability of pointer b.
        origin_a: Memory origin of a.
        origin_b: Memory origin of b.
        origin_c: Memory origin of c (mutable).
        dtype: Element data type (must be float32).

    Args:
        trans_a: 'N'/'n' — no transpose only (others fall back to naive).
        trans_b: 'N'/'n' — no transpose only.
        m: Rows of C and A.
        n: Columns of C and B.
        k: Inner dimension.
        alpha: Scalar multiplier for A*B.
        a: Pointer to A (column-major).
        lda: Leading dimension of A.
        b: Pointer to B (column-major).
        ldb: Leading dimension of B.
        beta: Scalar multiplier for C on input.
        c: Pointer to C (column-major, input/output).
        ldc: Leading dimension of C.
    """
    comptime assert dtype == DType.float32, "gemm_v9 requires float32"

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        _gemm_naive[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    if m == 0 or n == 0:
        return

    comptime TILE: Int = _V9_TILE
    comptime NZ: Int = _V9_NZ
    comptime MR: Int = _V9_MR  # 64
    comptime MC: Int = _V9_MC
    comptime TK: Int = _V9_TK
    comptime PAR_THRESHOLD: Int = GEMM_V9_PAR_THRESHOLD
    comptime SIMD_W: Int = simd_width_of[dtype]()  # 4 on M2 (128-bit NEON)

    # Apply beta to C.
    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v9[w: Int](i: Int) {cj}:
                cj.store[width=w](i, SIMD[dtype, w](0))

            vectorize[SIMD_W](m, zero_v9)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v9[w: Int](i: Int) {cj, beta}:
                cj.store[width=w](i, beta * cj.load[width=w](i))

            vectorize[SIMD_W](m, scale_v9)

    if alpha == 0 or k == 0:
        return

    # a_pack layout: a_pack[(iz + ir_block*NZ)*TK*TILE + l*TILE + lane]
    # Simplified: each group of TILE rows (index iz within MR-block) occupies
    # a contiguous TK*TILE block, so:
    #   a_pack[global_tile_idx * TK * TILE + l * TILE + lane]
    # where global_tile_idx = mc_row // TILE.
    # Total size: (MC // TILE) * TK * TILE = MC * TK floats.
    var a_pack = alloc[Scalar[dtype]](MC * TK)

    for mc0 in range(0, m, MC):
        var mc_len = min(MC, m - mc0)

        for k0 in range(0, k, TK):
            var klen = min(TK, k - k0)

            # Pack A panel: MC rows × klen cols → a_pack.
            for tile_idx in range(0, mc_len, TILE):
                var row_count = min(TILE, mc_len - tile_idx)
                var base = tile_idx * TK
                for l in range(klen):
                    for r in range(row_count):
                        a_pack[base + l * TILE + r] = (
                            alpha * a[mc0 + tile_idx + r + (k0 + l) * lda]
                        )
                    for r in range(row_count, TILE):
                        a_pack[base + l * TILE + r] = 0

            @parameter
            def col_group_v9(jr_block: Int):
                var j0 = jr_block * TILE
                if j0 >= n:
                    return
                var jlen = min(TILE, n - j0)

                # Pack B: klen × TILE → b_pack[l*TILE+r], zero-padded.
                var b_pack = stack_allocation[TK * TILE, dtype, alignment=128]()
                for l in range(klen):
                    for r in range(jlen):
                        b_pack[l * TILE + r] = b[(k0 + l) + (j0 + r) * ldb]
                    for r in range(jlen, TILE):
                        b_pack[l * TILE + r] = 0

                # c_tile[row * TILE + col]: row-major MR×TILE output buffer.
                # row in [0, MR), col in [0, TILE). Aligned for SIMD.
                var c_tile = stack_allocation[MR * TILE, dtype, alignment=128]()

                # ── Main loop: NZ*TILE = 64 rows per AMX block ───────────────
                var ir = 0
                while ir + MR <= mc_len:
                    # a_pack base offsets for the NZ Z-tiles.
                    var ab0 = ir * TK
                    var ab1 = ab0 + TILE * TK
                    var ab2 = ab1 + TILE * TK
                    var ab3 = ab2 + TILE * TK

                    _amx_set()

                    for l in range(klen):
                        # One B row into X[0] (shared across all 4 FMAs).
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        # Four A rows into Y[0..3].
                        _amx_load_y_row[dtype](a_pack + ab0 + l * TILE, 0)
                        _amx_load_y_row[dtype](a_pack + ab1 + l * TILE, 1)
                        _amx_load_y_row[dtype](a_pack + ab2 + l * TILE, 2)
                        _amx_load_y_row[dtype](a_pack + ab3 + l * TILE, 3)
                        # Z[z][:,:] += outer(Y[z][:], X[0][:])  for z in 0..3
                        _amx_fma32_tile(0, 0, 0, False)
                        _amx_fma32_tile(1, 0, 1, False)
                        _amx_fma32_tile(2, 0, 2, False)
                        _amx_fma32_tile(3, 0, 3, False)

                    # Store all NZ×TILE Z rows into c_tile (row-major).
                    # Tile-mode Z layout: Z tile z holds output rows z*TILE..(z+1)*TILE-1.
                    # Z row for tile z, output row r is z*4+r (stride-4 in Z's 64 rows).
                    # We store to c_tile row (z*TILE+r), address c_tile+(z*TILE+r)*TILE.
                    comptime for z in range(NZ):
                        comptime for r in range(TILE):
                            _amx_store_z_row[dtype](
                                c_tile + (z * TILE + r) * TILE, z + r * 4
                            )

                    _amx_clr()

                    # Accumulate c_tile into C.
                    # c_tile is row-major: c_tile[row*TILE + col].
                    # C is column-major: C[row, col] at c[col*ldc + row].
                    # For each output column col, gather the MR values from c_tile
                    # (strided by TILE) and add to the contiguous C column.
                    for col in range(jlen):
                        var c_col_ptr = c + (j0 + col) * ldc + mc0 + ir
                        comptime for blk in range(MR // SIMD_W):
                            var base_row = blk * SIMD_W
                            var v = SIMD[dtype, SIMD_W](0)
                            comptime for lane in range(SIMD_W):
                                v[lane] = c_tile[(base_row + lane) * TILE + col]
                            c_col_ptr.store[width=SIMD_W](
                                base_row,
                                c_col_ptr.load[width=SIMD_W](base_row) + v,
                            )

                    ir += MR

                # ── Single-tile AMX tail (TILE ≤ remaining rows < MR) ────────
                while ir + TILE <= mc_len:
                    var ab0 = ir * TK
                    _amx_set()
                    for l in range(klen):
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        _amx_load_y_row[dtype](a_pack + ab0 + l * TILE, 0)
                        _amx_fma32_tile(0, 0, 0, False)
                    comptime for r in range(TILE):
                        _amx_store_z_row[dtype](c_tile + r * TILE, r * 4)
                    _amx_clr()

                    for col in range(jlen):
                        var c_col_ptr = c + (j0 + col) * ldc + mc0 + ir
                        comptime for blk in range(TILE // SIMD_W):
                            var base_row = blk * SIMD_W
                            var v = SIMD[dtype, SIMD_W](0)
                            comptime for lane in range(SIMD_W):
                                v[lane] = c_tile[(base_row + lane) * TILE + col]
                            c_col_ptr.store[width=SIMD_W](
                                base_row,
                                c_col_ptr.load[width=SIMD_W](base_row) + v,
                            )

                    ir += TILE

                # ── Scalar tail (< TILE rows remaining) ──────────────────────
                while ir < mc_len:
                    var tile_base = (ir // TILE) * TK * TILE
                    var lane = ir % TILE
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += (
                                a_pack[tile_base + l * TILE + lane]
                                * b_pack[l * TILE + jc]
                            )
                        c[(mc0 + ir) + (j0 + jc) * ldc] += s
                    ir += 1

            var n_groups = (n + TILE - 1) // TILE
            if n_groups >= PAR_THRESHOLD:
                parallelize[col_group_v9](n_groups)
            else:
                for jg in range(n_groups):
                    col_group_v9(jg)

    a_pack.free()


# ── gemm_v10: AMX f32, 4-Z-tile + SIMD-transpose writeback ───────────────────
#
# Improvements over v9:
#   1. 4 Z tiles in flight: 64 rows × 16 cols per block.
#      Per k-step: 1 ldx (B into X[0]) + 4 ldy (A into Y[0..3]) + 4 fma32_tile.
#      2× more FMAs per k-step vs v9, 8× fewer set/clr vs v8.
#   2. SIMD 4×4 in-register transpose of the Z output before writing to C.
#      Z is stored row-major (each Z row = one output row, 16 floats).
#      We process the 16×16 output tile in 4×4 SIMD blocks: load 4 rows as
#      four SIMD[f32,4] registers, transpose with interleave, then store 4
#      contiguous output columns directly — no scalar gather at all.

comptime _V10_TILE: Int = GEMM_V10_TILE  # 16
comptime _V10_NZ: Int = GEMM_V10_NZ  # 4
comptime _V10_MR: Int = _V10_NZ * _V10_TILE  # 128
comptime _V10_MC: Int = GEMM_V10_MC  # 256
comptime _V10_TK: Int = GEMM_V10_TK  # 256


@always_inline
def _transpose_store_tile[
    origin_c: MutOrigin,
    //,
    dtype: DType,
    TILE: Int,
](
    z_buf: BLASPtr[dtype, _],
    c_col_base: BLASPtr[dtype, origin_c],
    ldc: Int,
    jlen: Int,
):
    """
    Transpose a row-major TILE×TILE z_buf into column-major C in-register.

    z_buf[row*TILE + col] → C[row, col]  (C: c_col_base + col*ldc + row).
    Uses 4×4 SIMD blocks: load 4 rows × 4 cols, transpose in-register,
    SIMD-store 4 contiguous rows per column. No scalar gather.
    jlen guards partial columns (runtime bound on which columns to write).
    """
    comptime SW: Int = 4  # NEON f32 lane count for the transpose block

    comptime for col_blk in range(TILE // SW):
        comptime c0 = col_blk * SW
        comptime for row_blk in range(TILE // SW):
            comptime r0 = row_blk * SW
            var r0v = z_buf.load[width=TILE](r0 * TILE).slice[SW, offset=c0]()
            var r1v = z_buf.load[width=TILE]((r0 + 1) * TILE).slice[
                SW, offset=c0
            ]()
            var r2v = z_buf.load[width=TILE]((r0 + 2) * TILE).slice[
                SW, offset=c0
            ]()
            var r3v = z_buf.load[width=TILE]((r0 + 3) * TILE).slice[
                SW, offset=c0
            ]()
            var t0 = SIMD[dtype, SW](r0v[0], r1v[0], r2v[0], r3v[0])
            var t1 = SIMD[dtype, SW](r0v[1], r1v[1], r2v[1], r3v[1])
            var t2 = SIMD[dtype, SW](r0v[2], r1v[2], r2v[2], r3v[2])
            var t3 = SIMD[dtype, SW](r0v[3], r1v[3], r2v[3], r3v[3])
            if c0 < jlen:
                var cp = c_col_base + c0 * ldc + r0
                cp.store[width=SW](0, cp.load[width=SW](0) + t0)
            if c0 + 1 < jlen:
                var cp = c_col_base + (c0 + 1) * ldc + r0
                cp.store[width=SW](0, cp.load[width=SW](0) + t1)
            if c0 + 2 < jlen:
                var cp = c_col_base + (c0 + 2) * ldc + r0
                cp.store[width=SW](0, cp.load[width=SW](0) + t2)
            if c0 + 3 < jlen:
                var cp = c_col_base + (c0 + 3) * ldc + r0
                cp.store[width=SW](0, cp.load[width=SW](0) + t3)


@always_inline
def _transpose_store_tile_set[
    origin_c: MutOrigin,
    //,
    dtype: DType,
    TILE: Int,
](
    z_buf: BLASPtr[dtype, _],
    c_col_base: BLASPtr[dtype, origin_c],
    ldc: Int,
    jlen: Int,
):
    """Transpose a row-major TILE×TILE z_buf into column-major C without accumulating.
    """
    comptime SW: Int = 4

    comptime for col_blk in range(TILE // SW):
        comptime c0 = col_blk * SW
        comptime for row_blk in range(TILE // SW):
            comptime r0 = row_blk * SW
            var r0v = z_buf.load[width=TILE](r0 * TILE).slice[SW, offset=c0]()
            var r1v = z_buf.load[width=TILE]((r0 + 1) * TILE).slice[
                SW, offset=c0
            ]()
            var r2v = z_buf.load[width=TILE]((r0 + 2) * TILE).slice[
                SW, offset=c0
            ]()
            var r3v = z_buf.load[width=TILE]((r0 + 3) * TILE).slice[
                SW, offset=c0
            ]()
            var t0 = SIMD[dtype, SW](r0v[0], r1v[0], r2v[0], r3v[0])
            var t1 = SIMD[dtype, SW](r0v[1], r1v[1], r2v[1], r3v[1])
            var t2 = SIMD[dtype, SW](r0v[2], r1v[2], r2v[2], r3v[2])
            var t3 = SIMD[dtype, SW](r0v[3], r1v[3], r2v[3], r3v[3])
            if c0 < jlen:
                (c_col_base + c0 * ldc + r0).store[width=SW](0, t0)
            if c0 + 1 < jlen:
                (c_col_base + (c0 + 1) * ldc + r0).store[width=SW](0, t1)
            if c0 + 2 < jlen:
                (c_col_base + (c0 + 2) * ldc + r0).store[width=SW](0, t2)
            if c0 + 3 < jlen:
                (c_col_base + (c0 + 3) * ldc + r0).store[width=SW](0, t3)


def gemm_v10[
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
    AMX f32 GEMM v10 for Apple M1/M2: C := alpha*A*B + beta*C (NN path).

    4 AMX Z tiles in flight (64 rows × 16 cols per block), plus SIMD 4×4
    in-register transpose for zero-scatter writeback from Z to column-major C.

    Parameters:
        mut_a: Mutability of pointer a.
        mut_b: Mutability of pointer b.
        origin_a: Memory origin of a.
        origin_b: Memory origin of b.
        origin_c: Memory origin of c (mutable).
        dtype: Element data type (must be float32).

    Args:
        trans_a: 'N'/'n' — no transpose only (others fall back to naive).
        trans_b: 'N'/'n' — no transpose only.
        m: Rows of C and A.
        n: Columns of C and B.
        k: Inner dimension.
        alpha: Scalar multiplier for A*B.
        a: Pointer to A (column-major).
        lda: Leading dimension of A.
        b: Pointer to B (column-major).
        ldb: Leading dimension of B.
        beta: Scalar multiplier for C on input.
        c: Pointer to C (column-major, input/output).
        ldc: Leading dimension of C.
    """
    comptime assert dtype == DType.float32, "gemm_v10 requires float32"

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        _gemm_naive[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    if m == 0 or n == 0:
        return

    comptime TILE: Int = _V10_TILE  # 16
    comptime NZ: Int = _V10_NZ  # 4
    comptime MR: Int = _V10_MR  # 64
    comptime MC: Int = _V10_MC  # 256
    comptime TK: Int = _V10_TK  # 256
    comptime PAR_THRESHOLD: Int = GEMM_V10_PAR_THRESHOLD
    comptime SIMD_W: Int = simd_width_of[dtype]()  # 4 on M2

    # Apply beta to C.
    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v10[w: Int](i: Int) {cj}:
                cj.store[width=w](i, SIMD[dtype, w](0))

            vectorize[SIMD_W](m, zero_v10)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v10[w: Int](i: Int) {cj, beta}:
                cj.store[width=w](i, beta * cj.load[width=w](i))

            vectorize[SIMD_W](m, scale_v10)

    if alpha == 0 or k == 0:
        return

    # a_pack[tile_idx * TK * TILE + l * TILE + lane]
    # tile_idx = mc_row // TILE, l = k-step, lane = row within tile.
    var a_pack = alloc[Scalar[dtype]](MC * TK)

    for mc0 in range(0, m, MC):
        var mc_len = min(MC, m - mc0)

        for k0 in range(0, k, TK):
            var klen = min(TK, k - k0)

            # Pack A: MC rows × klen cols → a_pack.
            for tile_idx in range(0, mc_len, TILE):
                var row_count = min(TILE, mc_len - tile_idx)
                var base = tile_idx * TK
                for l in range(klen):
                    for r in range(row_count):
                        a_pack[base + l * TILE + r] = (
                            alpha * a[mc0 + tile_idx + r + (k0 + l) * lda]
                        )
                    for r in range(row_count, TILE):
                        a_pack[base + l * TILE + r] = 0

            @parameter
            def col_group_v10(jr_block: Int):
                var j0 = jr_block * TILE
                if j0 >= n:
                    return
                var jlen = min(TILE, n - j0)

                # Pack B: klen × TILE → b_pack[l*TILE+r], zero-padded.
                var b_pack = stack_allocation[TK * TILE, dtype, alignment=128]()
                for l in range(klen):
                    for r in range(jlen):
                        b_pack[l * TILE + r] = b[(k0 + l) + (j0 + r) * ldb]
                    for r in range(jlen, TILE):
                        b_pack[l * TILE + r] = 0

                # z_buf is per column group; sharing it across parallel groups races.
                var z_buf = stack_allocation[MR * TILE, dtype, alignment=128]()

                var c_base = c + mc0  # base of the mc-block in C

                # ── Main loop: NZ*TILE = 64 rows per AMX block ───────────────
                var ir = 0
                while ir + MR <= mc_len:
                    _amx_set()

                    for l in range(klen):
                        # B row → X[0], shared by all 8 FMAs.
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        # 4 A rows → Y[0..3].
                        comptime for z in range(NZ):
                            _amx_load_y_row[dtype](
                                a_pack + (ir + z * TILE) * TK + l * TILE, z
                            )
                        # 4 tile-mode FMAs.
                        comptime for z in range(NZ):
                            _amx_fma32_tile(z, 0, z, False)

                    # Store all NZ*TILE Z rows into z_buf (row-major).
                    # Z tile z, output row r → z_buf[(z*TILE+r)*TILE].
                    comptime for z in range(NZ):
                        comptime for r in range(TILE):
                            _amx_store_z_row[dtype](
                                z_buf + (z * TILE + r) * TILE, z + r * 4
                            )

                    _amx_clr()

                    # SIMD-transpose z_buf (MR×TILE, row-major) → C (col-major).
                    # Process in NZ sub-tiles of TILE×TILE each.
                    comptime for z in range(NZ):
                        var z_sub = z_buf + z * TILE * TILE
                        var c_row_base = c_base + ir + z * TILE + j0 * ldc
                        _transpose_store_tile[dtype, TILE](
                            z_sub, c_row_base, ldc, jlen
                        )

                    ir += MR

                # ── 4-Z-tile block (64 rows, same as v9) ────────────────────
                comptime NZ4: Int = 4
                comptime MR4: Int = NZ4 * TILE
                while ir + MR4 <= mc_len:
                    _amx_set()
                    for l in range(klen):
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        comptime for z in range(NZ4):
                            _amx_load_y_row[dtype](
                                a_pack + (ir + z * TILE) * TK + l * TILE, z
                            )
                        comptime for z in range(NZ4):
                            _amx_fma32_tile(z, 0, z, False)
                    comptime for z in range(NZ4):
                        comptime for r in range(TILE):
                            _amx_store_z_row[dtype](
                                z_buf + (z * TILE + r) * TILE, z + r * 4
                            )
                    _amx_clr()
                    comptime for z in range(NZ4):
                        var z_sub = z_buf + z * TILE * TILE
                        var c_row_base = c_base + ir + z * TILE + j0 * ldc
                        _transpose_store_tile[dtype, TILE](
                            z_sub, c_row_base, ldc, jlen
                        )
                    ir += MR4

                # ── Single-tile AMX tail ─────────────────────────────────────
                while ir + TILE <= mc_len:
                    _amx_set()
                    for l in range(klen):
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        _amx_load_y_row[dtype](a_pack + ir * TK + l * TILE, 0)
                        _amx_fma32_tile(0, 0, 0, False)
                    comptime for r in range(TILE):
                        _amx_store_z_row[dtype](z_buf + r * TILE, r * 4)
                    _amx_clr()
                    var c_row_base = c_base + ir + j0 * ldc
                    _transpose_store_tile[dtype, TILE](
                        z_buf, c_row_base, ldc, jlen
                    )
                    ir += TILE

                # ── Scalar tail (<TILE rows) ─────────────────────────────────
                while ir < mc_len:
                    var tile_base = (ir // TILE) * TK * TILE
                    var lane = ir % TILE
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += (
                                a_pack[tile_base + l * TILE + lane]
                                * b_pack[l * TILE + jc]
                            )
                        c[(mc0 + ir) + (j0 + jc) * ldc] += s
                    ir += 1

            var n_groups = (n + TILE - 1) // TILE
            if n_groups >= PAR_THRESHOLD:
                parallelize[col_group_v10](n_groups)
            else:
                for jg in range(n_groups):
                    col_group_v10(jg)

    a_pack.free()


# ── gemm_v11: AMX f32, 4-Z-tile with 4x k-unroll (X[0..3] × Y[0..3]) ──────────
#
# Root cause of v9 gap vs Accelerate: 1 ldx + 4 ldy + 4 fma per k-step means
# the out-of-order engine sees one B-row load at a time, stalling on load-use.
#
# Fix: unroll k by UK=4. Each unrolled group:
#   1. Load X[0..3] = b_pack[l+0..3] (4 independent B-row loads, all issued early).
#   2. For each Z tile z=0..3:
#        load Y[0..3] = a_pack[tile-z, l+0..3]
#        fma32_tile(z, 0, 0), fma32_tile(z, 1, 1), fma32_tile(z, 2, 2), fma32_tile(z, 3, 3)
#
# Per unrolled step: 4 ldx + 4*(4 ldy + 4 fma) = 36 AMX ops computing 4×4×16×16
# = 4096 multiply-adds.  The OoO engine overlaps the 4 X-loads (fully independent,
# different X slots) with the Y-load + FMA pipeline of the previous tile.
# Writeback: same 4x4 SIMD transpose path as v10.

comptime _V11_TILE: Int = 16
comptime _V11_NZ: Int = GEMM_V11_NZ  # 4
comptime _V11_UK: Int = GEMM_V11_UK  # 4 — k-unroll factor
comptime _V11_MR: Int = _V11_NZ * _V11_TILE  # 64
comptime _V11_MC: Int = GEMM_V11_MC  # 256
comptime _V11_TK: Int = GEMM_V11_TK  # 256


def gemm_v11[
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
    AMX f32 GEMM v11 for Apple M1/M2: 4-Z-tile kernel with 4x k-loop unrolling.

    Uses X[0..3] for 4 B rows and Y[0..3] for 4 A rows per Z tile per unrolled
    group.  The 4 independent X loads allow the OoO engine to overlap B-fetch
    with the Y+FMA pipeline, closing the instruction-level parallelism gap vs
    Accelerate.  Writeback: row-major Z buffer + SIMD transpose stores (same as v10).

    Parameters:
        mut_a: Mutability of pointer a.
        mut_b: Mutability of pointer b.
        origin_a: Memory origin of a.
        origin_b: Memory origin of b.
        origin_c: Memory origin of c (mutable).
        dtype: Element data type (must be float32).

    Args:
        trans_a: 'N'/'n' only (others fall back to naive).
        trans_b: 'N'/'n' only.
        m: Rows of C and A.
        n: Columns of C and B.
        k: Inner dimension.
        alpha: Scalar multiplier for A*B.
        a: Pointer to A (column-major).
        lda: Leading dimension of A.
        b: Pointer to B (column-major).
        ldb: Leading dimension of B.
        beta: Scalar multiplier for C on input.
        c: Pointer to C (column-major, input/output).
        ldc: Leading dimension of C.
    """
    comptime assert dtype == DType.float32, "gemm_v11 requires float32"

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        _gemm_naive[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    if m == 0 or n == 0:
        return

    comptime TILE: Int = _V11_TILE
    comptime NZ: Int = _V11_NZ  # 4
    comptime UK: Int = _V11_UK  # 4
    comptime MR: Int = _V11_MR  # 64
    comptime MC: Int = _V11_MC  # 256
    comptime SMALL_MC: Int = GEMM_V11_SMALL_MC
    comptime TK: Int = _V11_TK  # 256
    comptime ROW_PAR_THRESHOLD: Int = GEMM_V11_ROW_PAR_THRESHOLD
    comptime PAR_THRESHOLD: Int = GEMM_V11_PAR_THRESHOLD
    comptime SIMD_W: Int = simd_width_of[dtype]()  # 4 on M2

    if alpha == 0 or k == 0:
        if beta == 0:
            for j in range(n):
                var cj = c + j * ldc

                def zero_v11_empty[w: Int](i: Int) {cj}:
                    cj.store[width=w](i, SIMD[dtype, w](0))

                vectorize[SIMD_W](m, zero_v11_empty)
        elif beta != 1:
            for j in range(n):
                var cj = c + j * ldc

                def scale_v11_empty[w: Int](i: Int) {cj, beta}:
                    cj.store[width=w](i, beta * cj.load[width=w](i))

                vectorize[SIMD_W](m, scale_v11_empty)
        return

    var direct_store_first_panel = beta == 0 and n <= ROW_PAR_THRESHOLD

    # Apply beta to C. For small/medium beta=0, the first k-panel overwrites C directly.
    if beta == 0 and not direct_store_first_panel:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v11[w: Int](i: Int) {cj}:
                cj.store[width=w](i, SIMD[dtype, w](0))

            vectorize[SIMD_W](m, zero_v11)
    elif beta != 0 and beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v11[w: Int](i: Int) {cj, beta}:
                cj.store[width=w](i, beta * cj.load[width=w](i))

            vectorize[SIMD_W](m, scale_v11)

    var n_groups = (n + TILE - 1) // TILE
    var mc_stride = MC
    if n <= ROW_PAR_THRESHOLD:
        mc_stride = SMALL_MC
    var use_pair_loads = n <= 512

    for k0 in range(0, k, TK):
        var klen = min(TK, k - k0)
        var b_stride = TK
        if n <= ROW_PAR_THRESHOLD:
            b_stride = klen
        var a_stride = klen
        var mc_groups = (m + mc_stride - 1) // mc_stride

        # b_panel[j_group * b_stride * TILE + l * TILE + lane]
        # Packed once per k-panel and reused for every MC row block.
        var b_panel = alloc[Scalar[dtype]](n_groups * b_stride * TILE)
        # One A-pack arena per k-panel; each row task owns a disjoint slice.
        var a_panels = alloc[Scalar[dtype]](mc_groups * mc_stride * a_stride)

        # Pack B panel for all output column groups, then reuse it for each MC block.
        for jr_block in range(n_groups):
            var j0 = jr_block * TILE
            var jlen = min(TILE, n - j0)
            var b_base = jr_block * b_stride * TILE
            for l in range(klen):
                for r in range(jlen):
                    b_panel[b_base + l * TILE + r] = b[
                        (k0 + l) + (j0 + r) * ldb
                    ]
                for r in range(jlen, TILE):
                    b_panel[b_base + l * TILE + r] = 0

        @parameter
        def row_block_v11(mc_block: Int):
            var mc0 = mc_block * mc_stride
            if mc0 >= m:
                return
            var mc_len = min(mc_stride, m - mc0)
            var a_pack = a_panels + mc_block * mc_stride * a_stride

            # Pack A panel for this row block.
            for tile_idx in range(0, mc_len, TILE):
                var row_count = min(TILE, mc_len - tile_idx)
                var base = tile_idx * a_stride
                for l in range(klen):
                    for r in range(row_count):
                        a_pack[base + l * TILE + r] = (
                            alpha * a[mc0 + tile_idx + r + (k0 + l) * lda]
                        )
                    for r in range(row_count, TILE):
                        a_pack[base + l * TILE + r] = 0

            @parameter
            def col_group_v11(jr_block: Int):
                var j0 = jr_block * TILE
                if j0 >= n:
                    return
                var jlen = min(TILE, n - j0)
                var b_pack = b_panel + jr_block * b_stride * TILE

                # z_buf[row * TILE + col]: row-major MR×TILE output buffer.
                var z_buf = stack_allocation[MR * TILE, dtype, alignment=128]()

                # ── Main loop: NZ*TILE = 64 rows per AMX block ───────────────
                var ir = 0
                while ir + MR <= mc_len:
                    var ab0 = ir * a_stride

                    _amx_set()

                    # 4x unrolled k-loop: process UK k-steps at a time.
                    # Per unrolled group:
                    #   Step 1: load X[0..UK-1] = b_pack rows l..l+UK-1 (all B loads upfront).
                    #   Step 2: for each Z tile z, load Y[0..UK-1] = a-tile-z rows, issue UK FMAs.
                    # The OoO engine can schedule all UK ldx instructions ahead of the ldy+fma chain.
                    var l = 0
                    while l + UK <= klen:
                        # Load UK B rows into X[0..UK-1] (independent, fills X register file).
                        if use_pair_loads:
                            comptime for u in range(0, UK, 2):
                                _amx_load_x_2rows[dtype](
                                    b_pack + (l + u) * TILE, u
                                )
                        else:
                            comptime for u in range(UK):
                                _amx_load_x_row[dtype](
                                    b_pack + (l + u) * TILE, u
                                )

                        # For each Z tile, load UK A rows into Y[0..UK-1] then issue UK FMAs.
                        # Y[0..UK-1] is reused per Z tile — fine since FMAs for z0 complete
                        # before we overwrite Y for z1.
                        comptime for z in range(NZ):
                            if use_pair_loads:
                                comptime for u in range(0, UK, 2):
                                    _amx_load_y_2rows[dtype](
                                        a_pack
                                        + (ab0 + z * TILE * a_stride)
                                        + (l + u) * TILE,
                                        u,
                                    )
                            else:
                                comptime for u in range(UK):
                                    _amx_load_y_row[dtype](
                                        a_pack
                                        + (ab0 + z * TILE * a_stride)
                                        + (l + u) * TILE,
                                        u,
                                    )
                            comptime for u in range(UK):
                                _amx_fma32_tile(z, u, u, False)

                        l += UK

                    # Scalar tail for remaining k-steps (< UK).
                    while l < klen:
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        comptime for z in range(NZ):
                            _amx_load_y_row[dtype](
                                a_pack + (ab0 + z * TILE * a_stride) + l * TILE,
                                z,
                            )
                        comptime for z in range(NZ):
                            _amx_fma32_tile(z, 0, z, False)
                        l += 1

                    # Store all NZ×TILE Z rows into z_buf (row-major).
                    comptime for z in range(NZ):
                        comptime for r in range(TILE):
                            _amx_store_z_row[dtype](
                                z_buf + (z * TILE + r) * TILE, z + r * 4
                            )
                    _amx_clr()
                    comptime for z in range(NZ):
                        var z_sub = z_buf + z * TILE * TILE
                        var c_row_base = c + (j0 * ldc) + mc0 + ir + z * TILE
                        if direct_store_first_panel and k0 == 0:
                            _transpose_store_tile_set[dtype, TILE](
                                z_sub, c_row_base, ldc, jlen
                            )
                        else:
                            _transpose_store_tile[dtype, TILE](
                                z_sub, c_row_base, ldc, jlen
                            )

                    ir += MR

                # ── Single-tile AMX tail (TILE ≤ remaining rows < MR) ────────
                while ir + TILE <= mc_len:
                    var ab0 = ir * a_stride
                    _amx_set()
                    var l = 0
                    while l + UK <= klen:
                        if use_pair_loads:
                            comptime for u in range(0, UK, 2):
                                _amx_load_x_2rows[dtype](
                                    b_pack + (l + u) * TILE, u
                                )
                            comptime for u in range(0, UK, 2):
                                _amx_load_y_2rows[dtype](
                                    a_pack + ab0 + (l + u) * TILE, u
                                )
                        else:
                            comptime for u in range(UK):
                                _amx_load_x_row[dtype](
                                    b_pack + (l + u) * TILE, u
                                )
                            comptime for u in range(UK):
                                _amx_load_y_row[dtype](
                                    a_pack + ab0 + (l + u) * TILE, u
                                )
                        comptime for u in range(UK):
                            _amx_fma32_tile(0, u, u, False)
                        l += UK
                    while l < klen:
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        _amx_load_y_row[dtype](a_pack + ab0 + l * TILE, 0)
                        _amx_fma32_tile(0, 0, 0, False)
                        l += 1
                    comptime for r in range(TILE):
                        _amx_store_z_row[dtype](z_buf + r * TILE, r * 4)
                    _amx_clr()

                    var c_row_base = c + (j0 * ldc) + mc0 + ir
                    if direct_store_first_panel and k0 == 0:
                        _transpose_store_tile_set[dtype, TILE](
                            z_buf, c_row_base, ldc, jlen
                        )
                    else:
                        _transpose_store_tile[dtype, TILE](
                            z_buf, c_row_base, ldc, jlen
                        )

                    ir += TILE

                # ── Scalar tail (< TILE rows) ─────────────────────────────────
                while ir < mc_len:
                    var tile_base = (ir // TILE) * a_stride * TILE
                    var lane = ir % TILE
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += (
                                a_pack[tile_base + l * TILE + lane]
                                * b_pack[l * TILE + jc]
                            )
                        if direct_store_first_panel and k0 == 0:
                            c[(mc0 + ir) + (j0 + jc) * ldc] = s
                        else:
                            c[(mc0 + ir) + (j0 + jc) * ldc] += s
                    ir += 1

            if n_groups >= PAR_THRESHOLD:
                for jg in range(n_groups):
                    col_group_v11(jg)
            else:
                for jg in range(n_groups):
                    col_group_v11(jg)

        if mc_groups >= 2:
            parallelize[row_block_v11](mc_groups)
        else:
            row_block_v11(0)

        a_panels.free()
        b_panel.free()


# ── gemm_v12: AMX f32, v11 large-kernel with NC-blocked B panel ───────────────

comptime _V12_TILE: Int = 16
comptime _V12_NZ: Int = 4
comptime _V12_UK: Int = 4
comptime _V12_MR: Int = _V12_NZ * _V12_TILE
comptime _V12_MC: Int = GEMM_V12_MC
comptime _V12_NC: Int = GEMM_V12_NC
comptime _V12_TK: Int = GEMM_V12_TK


def gemm_v12[
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
    Large-N AMX f32 GEMM experiment.

    v12 keeps v11's corrected 4-Z, UK=4 micro-kernel and row-block parallelism,
    but packs B in NC-column chunks instead of one full-width B panel.
    """
    comptime assert dtype == DType.float32, "gemm_v12 requires float32"

    var no_trans_a = trans_a == "N" or trans_a == "n"
    var no_trans_b = trans_b == "N" or trans_b == "n"
    if not no_trans_a or not no_trans_b:
        _gemm_naive[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    if n <= GEMM_V11_ROW_PAR_THRESHOLD:
        gemm_v11[dtype](
            trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
        )
        return

    if m == 0 or n == 0:
        return

    comptime TILE: Int = _V12_TILE
    comptime NZ: Int = _V12_NZ
    comptime UK: Int = _V12_UK
    comptime MR: Int = _V12_MR
    comptime MC: Int = _V12_MC
    comptime NC: Int = _V12_NC
    comptime TK: Int = _V12_TK
    comptime SIMD_W: Int = simd_width_of[dtype]()

    if alpha == 0 or k == 0:
        if beta == 0:
            for j in range(n):
                var cj = c + j * ldc

                def zero_v12_empty[w: Int](i: Int) {cj}:
                    cj.store[width=w](i, SIMD[dtype, w](0))

                vectorize[SIMD_W](m, zero_v12_empty)
        elif beta != 1:
            for j in range(n):
                var cj = c + j * ldc

                def scale_v12_empty[w: Int](i: Int) {cj, beta}:
                    cj.store[width=w](i, beta * cj.load[width=w](i))

                vectorize[SIMD_W](m, scale_v12_empty)
        return

    if beta == 0:
        for j in range(n):
            var cj = c + j * ldc

            def zero_v12[w: Int](i: Int) {cj}:
                cj.store[width=w](i, SIMD[dtype, w](0))

            vectorize[SIMD_W](m, zero_v12)
    elif beta != 1:
        for j in range(n):
            var cj = c + j * ldc

            def scale_v12[w: Int](i: Int) {cj, beta}:
                cj.store[width=w](i, beta * cj.load[width=w](i))

            vectorize[SIMD_W](m, scale_v12)

    for k0 in range(0, k, TK):
        var klen = min(TK, k - k0)
        var n_groups = (n + TILE - 1) // TILE
        var b_panel = alloc[Scalar[dtype]](n_groups * TK * TILE)

        for jr_block in range(n_groups):
            var j0 = jr_block * TILE
            var jlen = min(TILE, n - j0)
            var b_base = jr_block * TK * TILE
            for l in range(klen):
                for r in range(jlen):
                    b_panel[b_base + l * TILE + r] = b[
                        (k0 + l) + (j0 + r) * ldb
                    ]
                for r in range(jlen, TILE):
                    b_panel[b_base + l * TILE + r] = 0

        var mc_groups = (m + MC - 1) // MC
        var a_stride = klen
        var a_panels = alloc[Scalar[dtype]](mc_groups * MC * a_stride)

        @parameter
        def row_block_v12(mc_block: Int):
            var mc0 = mc_block * MC
            if mc0 >= m:
                return
            var mc_len = min(MC, m - mc0)
            var a_pack = a_panels + mc_block * MC * a_stride

            for tile_idx in range(0, mc_len, TILE):
                var row_count = min(TILE, mc_len - tile_idx)
                var base = tile_idx * a_stride
                for l in range(klen):
                    for r in range(row_count):
                        a_pack[base + l * TILE + r] = (
                            alpha * a[mc0 + tile_idx + r + (k0 + l) * lda]
                        )
                    for r in range(row_count, TILE):
                        a_pack[base + l * TILE + r] = 0

            @parameter
            def col_group_v12(jr_block: Int):
                var j0 = jr_block * TILE
                if j0 >= n:
                    return
                var jlen = min(TILE, n - j0)
                var b_pack = b_panel + jr_block * TK * TILE
                var z_buf = stack_allocation[MR * TILE, dtype, alignment=128]()

                var ir = 0
                while ir + MR <= mc_len:
                    var ab0 = ir * a_stride
                    _amx_set()
                    var l = 0
                    while l + UK <= klen:
                        comptime for u in range(UK):
                            _amx_load_x_row[dtype](b_pack + (l + u) * TILE, u)
                        comptime for z in range(NZ):
                            comptime for u in range(UK):
                                _amx_load_y_row[dtype](
                                    a_pack
                                    + (ab0 + z * TILE * a_stride)
                                    + (l + u) * TILE,
                                    u,
                                )
                            comptime for u in range(UK):
                                _amx_fma32_tile(z, u, u, False)
                        l += UK
                    while l < klen:
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        comptime for z in range(NZ):
                            _amx_load_y_row[dtype](
                                a_pack + (ab0 + z * TILE * a_stride) + l * TILE,
                                z,
                            )
                        comptime for z in range(NZ):
                            _amx_fma32_tile(z, 0, z, False)
                        l += 1

                    comptime for z in range(NZ):
                        comptime for r in range(TILE):
                            _amx_store_z_row[dtype](
                                z_buf + (z * TILE + r) * TILE, z + r * 4
                            )
                    _amx_clr()
                    comptime for z in range(NZ):
                        var z_sub = z_buf + z * TILE * TILE
                        var c_row_base = c + (j0 * ldc) + mc0 + ir + z * TILE
                        _transpose_store_tile[dtype, TILE](
                            z_sub, c_row_base, ldc, jlen
                        )
                    ir += MR

                while ir + TILE <= mc_len:
                    var ab0 = ir * a_stride
                    _amx_set()
                    var l = 0
                    while l + UK <= klen:
                        comptime for u in range(UK):
                            _amx_load_x_row[dtype](b_pack + (l + u) * TILE, u)
                        comptime for u in range(UK):
                            _amx_load_y_row[dtype](
                                a_pack + ab0 + (l + u) * TILE, u
                            )
                        comptime for u in range(UK):
                            _amx_fma32_tile(0, u, u, False)
                        l += UK
                    while l < klen:
                        _amx_load_x_row[dtype](b_pack + l * TILE, 0)
                        _amx_load_y_row[dtype](a_pack + ab0 + l * TILE, 0)
                        _amx_fma32_tile(0, 0, 0, False)
                        l += 1
                    comptime for r in range(TILE):
                        _amx_store_z_row[dtype](z_buf + r * TILE, r * 4)
                    _amx_clr()
                    var c_row_base = c + (j0 * ldc) + mc0 + ir
                    _transpose_store_tile[dtype, TILE](
                        z_buf, c_row_base, ldc, jlen
                    )
                    ir += TILE

                while ir < mc_len:
                    var tile_base = (ir // TILE) * a_stride * TILE
                    var lane = ir % TILE
                    for jc in range(jlen):
                        var s: Scalar[dtype] = 0
                        for l in range(klen):
                            s += (
                                a_pack[tile_base + l * TILE + lane]
                                * b_pack[l * TILE + jc]
                            )
                        c[(mc0 + ir) + (j0 + jc) * ldc] += s
                    ir += 1

            for nc0 in range(0, n, NC):
                var nc_len = min(NC, n - nc0)
                var nc_groups = (nc_len + TILE - 1) // TILE
                var nc_group0 = nc0 // TILE
                for jg in range(nc_groups):
                    col_group_v12(nc_group0 + jg)

        if mc_groups >= 2:
            parallelize[row_block_v12](mc_groups)
        else:
            row_block_v12(0)

        a_panels.free()
        b_panel.free()
