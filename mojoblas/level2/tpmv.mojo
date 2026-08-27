# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Triangular Packed Matrix-Vector Operations (`level2.tpmv`)
=============================================
Provides triangular packed matrix-vector operations as defined in the BLAS library standard.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory.alloc import unsafe_alloc

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.type_aliases import BLASPtr


def tpmv[
    mut_ap: Bool,
    origin_ap: Origin[mut=mut_ap],
    origin_x: MutOrigin,
    //,
    dtype: DType,
](
    uplo: String,
    trans: String,
    diag: String,
    n: Int,
    ap: BLASPtr[dtype, origin_ap],
    x: BLASPtr[dtype, origin_x],
    incx: Int,
):
    """
    Performs the matrix-vector operation x := A*x or x := A^T*x,
    where A is an n by n triangular matrix stored in packed format.

    Parameters:
        mut_ap: Indicates whether the pointer ap is mutable (True) or immutable (False).
        origin_ap: Memory origin of the pointer ap.
        origin_x: Memory origin of the pointer x (mutable, input/output).
        dtype: The data type of the elements (e.g., Float32, Float64).

    Args:
        uplo: Specifies whether A is upper ('U') or lower ('L') triangular.
        trans: Specifies the operation: 'N' for x := A*x, 'T' or 'C' for x := A^T*x.
        diag: Specifies whether A is unit triangular ('U') or not ('N').
        n: The order of the matrix A.
        ap: A pointer to the packed triangular matrix A.
        x: A pointer to the first element of the vector x (input/output).
        incx: The increment for the elements of x.
    """
    var info: Int = 0
    if uplo != "U" and uplo != "u" and uplo != "L" and uplo != "l":
        info = 1
    elif (
        trans != "N"
        and trans != "n"
        and trans != "T"
        and trans != "t"
        and trans != "C"
        and trans != "c"
    ):
        info = 2
    elif diag != "U" and diag != "u" and diag != "N" and diag != "n":
        info = 3
    elif n < 0:
        info = 4
    elif incx == 0:
        info = 7

    if info != 0:
        print("tpmv: Info", info)
        return

    if n == 0:
        return

    var no_unit = diag == "N" or diag == "n"
    var upper = uplo == "U" or uplo == "u"
    var no_trans = trans == "N" or trans == "n"

    if incx == 1:
        var x_in = unsafe_alloc[Scalar[dtype]](n)
        var x_out = unsafe_alloc[Scalar[dtype]](n)
        for i in range(n):
            x_in[unsafe_offset=i] = x[unsafe_offset=i]

        if no_trans:
            for i in range(n):
                var sum: Scalar[dtype] = 0
                if upper:
                    for j in range(i, n):
                        var aij: Scalar[dtype] = 1 if (
                            i == j and not no_unit
                        ) else ap[unsafe_offset=(j * (j + 1)) // 2 + i]
                        sum = sum + aij * x_in[unsafe_offset=j]
                else:
                    for j in range(i + 1):
                        var start = j * n - (j * (j - 1)) // 2
                        var aij: Scalar[dtype] = 1 if (
                            i == j and not no_unit
                        ) else ap[unsafe_offset=start + (i - j)]
                        sum = sum + aij * x_in[unsafe_offset=j]
                x_out[unsafe_offset=i] = sum
        else:
            for i in range(n):
                var sum: Scalar[dtype] = 0
                if upper:
                    for j in range(i + 1):
                        var aji: Scalar[dtype] = 1 if (
                            i == j and not no_unit
                        ) else ap[unsafe_offset=(i * (i + 1)) // 2 + j]
                        sum = sum + aji * x_in[unsafe_offset=j]
                else:
                    for j in range(i, n):
                        var start = i * n - (i * (i - 1)) // 2
                        var aji: Scalar[dtype] = 1 if (
                            i == j and not no_unit
                        ) else ap[unsafe_offset=start + (j - i)]
                        sum = sum + aji * x_in[unsafe_offset=j]
                x_out[unsafe_offset=i] = sum

        for i in range(n):
            x[unsafe_offset=i] = x_out[unsafe_offset=i]
        x_in.unsafe_free()
        x_out.unsafe_free()
        return

    var x_in = unsafe_alloc[Scalar[dtype]](n)
    var x_out = unsafe_alloc[Scalar[dtype]](n)

    var kx: Int = 0 if incx > 0 else (1 - n) * incx
    var ix: Int = kx
    for i in range(n):
        x_in[unsafe_offset=i] = x[unsafe_offset=ix]
        ix += incx

    def a_at(
        i: Int, j: Int
    ) {imm no_unit, imm upper, imm ap, imm n} -> Scalar[dtype]:
        if i == j and not no_unit:
            return 1
        if upper:
            if i > j:
                return 0
            var idx = (j * (j + 1)) // 2 + i
            return ap[unsafe_offset=idx]
        if i < j:
            return 0
        var start = j * n - (j * (j - 1)) // 2
        return ap[unsafe_offset=start + (i - j)]

    for i in range(n):
        var sum: Scalar[dtype] = 0
        for j in range(n):
            if no_trans:
                sum = sum + a_at(i, j) * x_in[unsafe_offset=j]
            else:
                sum = sum + a_at(j, i) * x_in[unsafe_offset=j]
        x_out[unsafe_offset=i] = sum

    ix = kx
    for i in range(n):
        x[unsafe_offset=ix] = x_out[unsafe_offset=i]
        ix += incx

    x_in.unsafe_free()
    x_out.unsafe_free()
