# Function Reference

Complete list of available BLAS operations in mojoBLAS.

## Available Functions

### Level 1 BLAS Operations (Vector-Vector)

| Function | Description | Formula |
|----------|-------------|---------|
| `copy` | Copy vector X to vector Y | `Y := X` |
| `scal` | Scale vector by scalar | `X := α*X` |
| `axpy` | Vector plus scaled vector | `Y := α*X + Y` |
| `dot` | Dot product | `result = X · Y` |
| `nrm2` | Euclidean norm | `result = √(Σx²)` |
| `asum` | Sum of absolute values | `result = Σ|x|` |
| `swap` | Swap vectors X and Y | `X ↔ Y` |
| `iamax` | Index of max absolute value | `result = argmax(|x|)` |
| `rotg` | Generate Given rotation | Construct rotation matrix |
| `rot` | Apply Givens rotation | Apply rotation to vectors |

### Level 2 BLAS Operations (Matrix-Vector)

| Function | Description | Formula |
|----------|-------------|---------|
| `gemv` | General matrix-vector multiply | `Y := α*A*X + β*Y` |
| `gbmv` | General banded matrix-vector multiply | `Y := α*A*X + β*Y` |
| `sbmv` | Symmetric banded matrix-vector multiply | `Y := α*A*X + β*Y` |
| `spmv` | Symmetric packed matrix-vector multiply | `Y := α*A*X + β*Y` |
| `symv` | Symmetric matrix-vector multiply | `Y := α*A*X + β*Y` |
| `syr` | Symmetric rank-1 update | `A := α*X*X^T + A` |
| `syr2` | Symmetric rank-2 update | `A := α*X*Y^T + α*Y*X^T + A` |
| `tbmv` | Triangular banded matrix-vector multiply | `X := A*X` |
| `tbsv` | Triangular banded solve | `X := A^-1*X` |
| `tpmv` | Triangular packed matrix-vector multiply | `X := A*X` |
| `tpsv` | Triangular packed solve | `X := A^-1*X` |
| `trmv` | Triangular matrix-vector multiply | `X := A*X` |
| `trsv` | Triangular solve | `X := A^-1*X` |

### Level 3 BLAS Operations (Matrix-Matrix)

| Function | Description | Formula |
|----------|-------------|---------|
| `gemm` | General matrix multiply | `C := α*A*B + β*C` |
| `symm` | Symmetric matrix multiply | `C := α*A*B + β*C` |
| `syrk` | Symmetric rank-k update | `C := α*A*A^T + β*C` |
| `syr2k` | Symmetric rank-2k update | `C := α*A*B^T + α*B*A^T + β*C` |
| `trmm` | Triangular matrix multiply | `B := α*A*B` |
| `trsm` | Triangular solve (multiple RHS) | `B := α*A^-1*B` |

## Function Signatures

### Level 1 BLAS

```mojo
# Vector copy
fn copy[
    mut: Bool,
    origin_x: Origin[mut=mut],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](n: Int, dx: BLASPtr[dtype, origin_x], incx: Int, dy: BLASPtr[dtype, origin_y], incy: Int) -> None

# Vector scaling
fn scal[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    //,
    dtype: DType,
](n: Int, alpha: Scalar[dtype], x: BLASPtr[dtype, origin_x], incx: Int) -> None

# AXPY operation
fn axpy[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,    //,
    dtype: DType,
](n: Int, alpha: Scalar[dtype], x: BLASPtr[dtype, origin_x], incx: Int, y: BLASPtr[dtype, origin_y], incy: Int) -> None

# Dot product
fn dot[
    mut_x: Bool,
    mut_y: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: Origin[mut=mut_y],
    //,
    dtype: DType,
](n: Int, x: BLASPtr[dtype, origin_x], incx: Int, y: BLASPtr[dtype, origin_y], incy: Int) -> Scalar[dtype]

# Euclidean norm
fn nrm2[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    //,
    dtype: DType,
](n: Int, x: BLASPtr[dtype, origin_x], incx: Int) -> Scalar[dtype]

# Absolute sum
fn asum[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    //,
    dtype: DType,
](n: Int, x: BLASPtr[dtype, origin_x], incx: Int) -> Scalar[dtype]

# Vector swap
fn vswap[
    mut_x: Bool,
    origin_x: MutOrigin,
    origin_y: MutOrigin,
    //,
    dtype: DType,
](n: Int, x: BLASPtr[dtype, origin_x], incx: Int, y: BLASPtr[dtype, origin_y], incy: Int) -> None

# Index of max absolute value
fn iamax[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    //,
    dtype: DType,
](n: Int, x: BLASPtr[dtype, origin_x], incx: Int) -> Int

# Generate Givens rotation
fn rotg[
    origin_a: MutOrigin,
    origin_b: MutOrigin,
    //,
    dtype: DType,
](a: Scalar[dtype], b: Scalar[dtype]) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype], Scalar[dtype]]

# Apply Givens rotation
fn rot[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](n: Int, x: BLASPtr[dtype, origin_x], incx: Int, y: BLASPtr[dtype, origin_y], incy: Int, c: Scalar[dtype], s: Scalar[dtype]) -> None
```

### Level 2 BLAS

```mojo
# General matrix-vector multiply
fn gemv[
    mut_a: Bool,
    mut_x: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](trans: String, m: Int, n: Int, alpha: Scalar[dtype], a: BLASPtr[dtype, origin_a], lda: Int,
 x: BLASPtr[dtype, origin_x], incx: Int, beta: Scalar[dtype], y: BLASPtr[dtype, origin_y], incy: Int) -> None

# General banded matrix-vector multiply
fn gbmv[
    mut_a: Bool,
    mut_x: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](trans: String, m: Int, n: Int, kl: Int, ku: Int, alpha: Scalar[dtype],
 a: BLASPtr[dtype, origin_a], lda: Int, x: BLASPtr[dtype, origin_x], incx: Int,
 beta: Scalar[dtype], y: BLASPtr[dtype, origin_y], incy: Int) -> None

# Symmetric matrix-vector multiply
fn symv[
    mut_a: Bool,
    mut_x: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: Origin[mut=mut_x],
    origin_y: MutOrigin,
    //,
    dtype: DType,
](uplo: String, n: Int, alpha: Scalar[dtype], a: BLASPtr[dtype, origin_a], lda: Int,
 x: BLASPtr[dtype, origin_x], incx: Int, beta: Scalar[dtype], y: BLASPtr[dtype, origin_y], incy: Int) -> None

# Triangular matrix-vector multiply
fn trmv[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: MutOrigin,
    //,
    dtype: DType,
](uplo: String, trans: String, diag: String, n: Int, a: BLASPtr[dtype, origin_a], lda: Int,
 x: BLASPtr[dtype, origin_x], incx: Int) -> None

# Triangular solve
fn trsv[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_x: MutOrigin,
    //,
    dtype: DType,
](uplo: String, trans: String, diag: String, n: Int, a: BLASPtr[dtype, origin_a], lda: Int,
 x: BLASPtr[dtype, origin_x], incx: Int) -> None

# Symmetric rank-1 update
fn syr[
    mut_x: Bool,
    origin_x: Origin[mut=mut_x],
    origin_a: MutOrigin,
    //,
    dtype: DType,
](uplo: String, n: Int, alpha: Scalar[dtype], x: BLASPtr[dtype, origin_x], incx: Int,
 a: BLASPtr[dtype, origin_a], lda: Int) -> None
```

### Level 3 BLAS

```mojo
# General matrix-matrix multiply
fn gemm[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](trans_a: String, trans_b: String, m: Int, n: Int, k: Int, alpha: Scalar[dtype],
 a: BLASPtr[dtype, origin_a], lda: Int, b: BLASPtr[dtype, origin_b], ldb: Int,
 beta: Scalar[dtype], c: BLASPtr[dtype, origin_c], ldc: Int) -> None

# Symmetric matrix-matrix multiply
fn symm[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](side: String, uplo: String, m: Int, n: Int, alpha: Scalar[dtype],
 a: BLASPtr[dtype, origin_a], lda: Int, b: BLASPtr[dtype, origin_b], ldb: Int,
 beta: Scalar[dtype], c: BLASPtr[dtype, origin_c], ldc: Int) -> None

# Symmetric rank-k update
fn syrk[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](uplo: String, trans: String, n: Int, k: Int, alpha: Scalar[dtype],
 a: BLASPtr[dtype, origin_a], lda: Int, beta: Scalar[dtype],
 c: BLASPtr[dtype, origin_c], ldc: Int) -> None

# Triangular matrix-matrix multiply
fn trmm[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: MutOrigin,
    //,
    dtype: DType,
](side: String, uplo: String, trans_a: String, diag: String, m: Int, n: Int,
 alpha: Scalar[dtype], a: BLASPtr[dtype, origin_a], lda: Int,
 b: BLASPtr[dtype, origin_b], ldb: Int) -> None

# Triangular solve (multiple RHS)
fn trsm[
    mut_a: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: MutOrigin,
    //,
    dtype: DType,
](side: String, uplo: String, trans_a: String, diag: String, m: Int, n: Int,
 alpha: Scalar[dtype], a: BLASPtr[dtype, origin_a], lda: Int,
 b: BLASPtr[dtype, origin_b], ldb: Int) -> None
```

## Memory Model

mojoBLAS uses a memory origin tracking system to support flexible memory management:

- **Mutability parameters** (`mut_X: Bool`): Controls whether a pointer can be mutated
- **Origin types**:
  - `Origin[mut=mut_X]`: Memory origin tied to mutability parameter
  - `MutOrigin`: Always-mutable origin for input/output parameters
- **BLASPtr[dtype, origin]**: Typed pointer alias for BLAS operations

### Parameter Order Convention

For each BLAS function:

1. **Compile-time parameters**: First, using pattern `mut_X: Bool -> origin_X: Origin[mut=mut_X]`
2. **Separator**: `//,`
3. **Runtime parameters**: Including `dtype: DType` last
4. **Function arguments**: Follow standard BLAS parameter ordering