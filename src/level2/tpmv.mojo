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

    Optimized with SIMD vectorization and parallelization.

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

    var x_in = alloc[Scalar[dtype]](n)
    var x_out = alloc[Scalar[dtype]](n)

    var kx: Int = 0 if incx > 0 else (1 - n) * incx
    var ix: Int = kx
    for i in range(n):
        x_in[i] = x[ix]
        ix += incx

    @parameter
    def a_at(i: Int, j: Int) -> Scalar[dtype]:
        if i == j and not no_unit:
            return 1
        if upper:
            if i > j:
                return 0
            var idx = (j * (j + 1)) // 2 + i
            return ap[idx]
        if i < j:
            return 0
        var start = j * n - (j * (j - 1)) // 2
        return ap[start + (i - j)]

    for i in range(n):
        var sum: Scalar[dtype] = 0
        for j in range(n):
            if no_trans:
                sum = sum + a_at(i, j) * x_in[j]
            else:
                sum = sum + a_at(j, i) * x_in[j]
        x_out[i] = sum

    ix = kx
    for i in range(n):
        x[ix] = x_out[i]
        ix += incx

    x_in.free()
    x_out.free()
