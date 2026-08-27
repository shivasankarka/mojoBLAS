# ===----------------------------------------------------------------------=== #
# mojoBLAS: Triangular Matrix-Matrix Operations
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #
"""
Triangular Matrix-Matrix Operations (mojoblas.level3.trmm).
===========================================================
Provides triangular matrix-matrix operations as defined in the BLAS library standard.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of

# ===----------------------------------------------------------------------=== #
# Max
# ===----------------------------------------------------------------------=== #
from max.algorithm.backend.cpu import parallelize

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.type_aliases import BLASPtr


def trmm[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: MutOrigin,
    //,
    dtype: DType,
](
    side: String,
    uplo: String,
    trans_a: String,
    diag: String,
    m: Int,
    n: Int,
    alpha: Scalar[dtype],
    a: BLASPtr[dtype, origin_a],
    lda: Int,
    b: BLASPtr[dtype, origin_b],
    ldb: Int,
):
    """
    Performs the matrix-matrix operation B := alpha*A*B or B := alpha*B*A,
    where A is a triangular matrix and B is an m by n matrix.

    Parameters:
        mut_a: Indicates whether the pointer a is mutable (True) or immutable (False).
        origin_a: Memory origin of the pointer a.
        origin_b: Memory origin of the pointer b (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        side: Specifies whether A is on the left ('L') or right ('R') of B.
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans_a: Specifies whether A is transposed ('N' for no, 'T' or 'C' for yes).
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        m: The number of rows of the matrix B.
        n: The number of columns of the matrix B.
        alpha: The scalar multiplier for the matrix product.
        a: A pointer to the first element of the triangular matrix A.
        lda: The leading dimension of the matrix A.
        b: A pointer to the first element of the matrix B (input/output).
        ldb: The leading dimension of the matrix B.
    """
    var info: Int = 0
    if side != "L" and side != "l" and side != "R" and side != "r":
        info = 1
    elif uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 2
    elif (
        trans_a != "N"
        and trans_a != "n"
        and trans_a != "T"
        and trans_a != "t"
        and trans_a != "C"
        and trans_a != "c"
    ):
        info = 3
    elif diag != "U" and diag != "u" and diag != "N" and diag != "n":
        info = 4
    elif m < 0:
        info = 5
    elif n < 0:
        info = 6
    elif lda < max(1, m if (side == "L" or side == "l") else n):
        info = 9
    elif ldb < max(1, m):
        info = 11

    if info != 0:
        print("trmm: Info", info)
        return

    if m == 0 or n == 0:
        return

    if alpha == 0:
        for j in range(n):
            for i in range(m):
                b[unsafe_offset=i + j * ldb] = 0
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    comptime PAR_THRESHOLD: Int = 64
    var left_side = side == "L" or side == "l"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans_a == "N" or trans_a == "n"
    var no_unit = diag == "N" or diag == "n"

    if left_side:
        if no_trans:
            if upper:
                for j in range(n):
                    var bj = b.unsafe_offset(j * ldb)
                    for k in range(m):
                        if bj[unsafe_offset=k] != 0:
                            var temp: Scalar[dtype] = (
                                alpha * bj[unsafe_offset=k]
                            )
                            var ak = a.unsafe_offset(k * lda)

                            def axpy_lu[width: Int](i: Int) {bj, ak, temp}:
                                bj.unsafe_store[width=width](
                                    i,
                                    bj.unsafe_load[width=width](i)
                                    + temp * ak.unsafe_load[width=width](i),
                                )

                            vectorize[simd_width](k, axpy_lu)
                            if no_unit:
                                temp = temp * a[unsafe_offset=k + k * lda]
                            bj[unsafe_offset=k] = temp
            else:
                for j in range(n):
                    var bj = b.unsafe_offset(j * ldb)
                    for k in range(m - 1, -1, -1):
                        if bj[unsafe_offset=k] != 0:
                            var temp: Scalar[dtype] = (
                                alpha * bj[unsafe_offset=k]
                            )
                            bj[unsafe_offset=k] = temp
                            if no_unit:
                                bj[unsafe_offset=k] = (
                                    bj[unsafe_offset=k]
                                    * a[unsafe_offset=k + k * lda]
                                )
                            var ak = a.unsafe_offset(k * lda)

                            def axpy_ll[width: Int](i: Int) {bj, ak, temp, k}:
                                var ii = k + 1 + i
                                bj.unsafe_store[width=width](
                                    ii,
                                    bj.unsafe_load[width=width](ii)
                                    + temp * ak.unsafe_load[width=width](ii),
                                )

                            vectorize[simd_width](m - k - 1, axpy_ll)
        else:
            # Trans left: each j column of B is independent — safe to parallelize
            if upper:

                @parameter
                def trmm_lt_upper(j: Int):
                    var bj = b.unsafe_offset(j * ldb)
                    for i in range(m - 1, -1, -1):
                        var temp: Scalar[dtype] = bj[unsafe_offset=i]
                        if no_unit:
                            temp = temp * a[unsafe_offset=i + i * lda]
                        for kk in range(i):
                            temp = (
                                temp
                                + a[unsafe_offset=kk + i * lda]
                                * bj[unsafe_offset=kk]
                            )
                        bj[unsafe_offset=i] = alpha * temp

                if n >= PAR_THRESHOLD:
                    parallelize[trmm_lt_upper](n)
                else:
                    for j in range(n):
                        trmm_lt_upper(j)
            else:

                @parameter
                def trmm_lt_lower(j: Int):
                    var bj = b.unsafe_offset(j * ldb)
                    for i in range(m):
                        var temp: Scalar[dtype] = bj[unsafe_offset=i]
                        if no_unit:
                            temp = temp * a[unsafe_offset=i + i * lda]
                        for kk in range(i + 1, m):
                            temp = (
                                temp
                                + a[unsafe_offset=kk + i * lda]
                                * bj[unsafe_offset=kk]
                            )
                        bj[unsafe_offset=i] = alpha * temp

                if n >= PAR_THRESHOLD:
                    parallelize[trmm_lt_lower](n)
                else:
                    for j in range(n):
                        trmm_lt_lower(j)
    else:
        if no_trans:
            # Right no-trans: each j only reads from columns k<j or k>j, writes to j — safe
            if upper:

                @parameter
                def trmm_rn_upper(j: Int):
                    var bj = b.unsafe_offset(j * ldb)
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[unsafe_offset=j + j * lda]

                    def scale_bj_u[width: Int](i: Int) {bj, temp}:
                        bj.unsafe_store[width=width](
                            i, temp * bj.unsafe_load[width=width](i)
                        )

                    vectorize[simd_width](m, scale_bj_u)
                    for kk in range(j):
                        if a[unsafe_offset=kk + j * lda] != 0:
                            temp = alpha * a[unsafe_offset=kk + j * lda]
                            var off_j = j * ldb
                            var off_k = kk * ldb

                            def axpy_ru[
                                width: Int
                            ](i: Int) {b, off_j, off_k, temp}:
                                b.unsafe_store[width=width](
                                    off_j + i,
                                    b.unsafe_load[width=width](off_j + i)
                                    + temp
                                    * b.unsafe_load[width=width](off_k + i),
                                )

                            vectorize[simd_width](m, axpy_ru)

                # upper no-trans: j iterates n-1 down to 0 (sequential dep) — skip parallelize
                for j in range(n - 1, -1, -1):
                    trmm_rn_upper(j)
            else:

                @parameter
                def trmm_rn_lower(j: Int):
                    var bj = b.unsafe_offset(j * ldb)
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[unsafe_offset=j + j * lda]

                    def scale_bj_l[width: Int](i: Int) {bj, temp}:
                        bj.unsafe_store[width=width](
                            i, temp * bj.unsafe_load[width=width](i)
                        )

                    vectorize[simd_width](m, scale_bj_l)
                    for kk in range(j + 1, n):
                        if a[unsafe_offset=kk + j * lda] != 0:
                            temp = alpha * a[unsafe_offset=kk + j * lda]
                            var off_j = j * ldb
                            var off_k = kk * ldb

                            def axpy_rl[
                                width: Int
                            ](i: Int) {b, off_j, off_k, temp}:
                                b.unsafe_store[width=width](
                                    off_j + i,
                                    b.unsafe_load[width=width](off_j + i)
                                    + temp
                                    * b.unsafe_load[width=width](off_k + i),
                                )

                            vectorize[simd_width](m, axpy_rl)

                # lower no-trans: j iterates 0..n-1 forward (sequential dep) — skip parallelize
                for j in range(n):
                    trmm_rn_lower(j)
        else:
            if upper:
                for k in range(n):
                    var bk = b.unsafe_offset(k * ldb)
                    var off_k = k * ldb
                    for j in range(k):
                        if a[unsafe_offset=j + k * lda] != 0:
                            var temp: Scalar[dtype] = (
                                alpha * a[unsafe_offset=j + k * lda]
                            )
                            var off_j = j * ldb

                            def axpy_rtu[
                                width: Int
                            ](i: Int) {b, off_j, off_k, temp}:
                                b.unsafe_store[width=width](
                                    off_j + i,
                                    b.unsafe_load[width=width](off_j + i)
                                    + temp
                                    * b.unsafe_load[width=width](off_k + i),
                                )

                            vectorize[simd_width](m, axpy_rtu)
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[unsafe_offset=k + k * lda]
                    if temp != 1:

                        def scale_bk_u[width: Int](i: Int) {bk, temp}:
                            bk.unsafe_store[width=width](
                                i, temp * bk.unsafe_load[width=width](i)
                            )

                        vectorize[simd_width](m, scale_bk_u)
            else:
                for k in range(n - 1, -1, -1):
                    var bk = b.unsafe_offset(k * ldb)
                    var off_k = k * ldb
                    for j in range(k + 1, n):
                        if a[unsafe_offset=j + k * lda] != 0:
                            var temp: Scalar[dtype] = (
                                alpha * a[unsafe_offset=j + k * lda]
                            )
                            var off_j = j * ldb

                            def axpy_rtl[
                                width: Int
                            ](i: Int) {b, off_j, off_k, temp}:
                                b.unsafe_store[width=width](
                                    off_j + i,
                                    b.unsafe_load[width=width](off_j + i)
                                    + temp
                                    * b.unsafe_load[width=width](off_k + i),
                                )

                            vectorize[simd_width](m, axpy_rtl)
                    var temp: Scalar[dtype] = alpha
                    if no_unit:
                        temp = temp * a[unsafe_offset=k + k * lda]
                    if temp != 1:

                        def scale_bk_l[width: Int](i: Int) {bk, temp}:
                            bk.unsafe_store[width=width](
                                i, temp * bk.unsafe_load[width=width](i)
                            )

                        vectorize[simd_width](m, scale_bk_l)

    return
