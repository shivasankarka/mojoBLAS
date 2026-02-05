# mojoBLAS

A high-performance **BLAS (Basic Linear Algebra Subprograms)** implementation written in [Mojo](https://modular.com/mojo), leveraging Mojo's powerful systems programming capabilities and zero-cost abstractions for maximum performance.

## Motivation

This project is just a try at implementing BLAS operations in hopes of using it as a backend for [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo). Also, it's just fun to code these operations and get into the rabbit hole of optimizations. What I have so far are,

- **Level 1 BLAS Operations**: Complete implementation of all standard Level 1 BLAS routines.
- **Generic Implementation**: Supports all DType in existing Level 1 BLAS routines. 
- **Standard Compliant**: Follows the standard BLAS API conventions.

## 📦 Installation

Not complete yet, Will be updated, please write some codes while I update this :) 

### Prerequisites

- Currently works on **Mojo** nightly version `>=0.26.2.0.dev2026020505,<0.27` (see [Mojo installation guide](https://docs.modular.com/mojo/manual/get-started/))

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mojoBLAS.git
   cd mojoBLAS
   ```

2. **Install dependencies:**
   ```bash
   pixi install
   ```

3. **Format code (optional):**
   ```bash
   pixi run format
   ```

## 🔧 Usage

### Basic Example

```mojo
from mojoBLAS.src.level1 import dot, axpy, nrm2
from memory import UnsafePointer

fn main():
    # Create vectors
    var x = UnsafePointer[Float32].alloc(3)
    var y = UnsafePointer[Float32].alloc(3)
    
    # Initialize data
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0
    y[0] = 4.0; y[1] = 5.0; y[2] = 6.0
    
    # Compute dot product: x · y
    var result = dot[DType.float32](3, x, 1, y, 1)
    print("Dot product:", result)  # Output: 32.0
    
    # Perform AXPY: y = α*x + y
    axpy[DType.float32](3, 2.0, x, 1, y, 1)
    print("After AXPY:", y[0], y[1], y[2])  # Output: 6.0, 9.0, 12.0
    
    # Compute Euclidean norm
    var norm = nrm2[DType.float32](3, x, 1)
    print("Euclidean norm:", norm)
    
    # Clean up
    x.free()
    y.free()
```

### Available Functions

#### Level 1 BLAS Operations

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
| `rotg` | Generate Givens rotation | Construct rotation matrix |
| `rot` | Apply Givens rotation | Apply rotation to vectors |

### Function Signatures

```mojo
# Vector copy
fn copy[dtype: DType](n: Int32, x: BLASPtr[Scalar[dtype]], incx: Int32, 
                      y: BLASPtr[Scalar[dtype]], incy: Int32) raises -> None

# Vector scaling
fn scal[dtype: DType](n: Int32, alpha: Scalar[dtype], 
                      x: BLASPtr[Scalar[dtype]], incx: Int32) -> None

# AXPY operation
fn axpy[dtype: DType](n: Int32, alpha: Scalar[dtype], 
                      x: BLASPtr[Scalar[dtype]], incx: Int32,
                      y: BLASPtr[Scalar[dtype]], incy: Int32) -> None

# Dot product
fn dot[dtype: DType](n: Int32, x: BLASPtr[Scalar[dtype]], incx: Int32,
                     y: BLASPtr[Scalar[dtype]], incy: Int32) -> Scalar[dtype]

# And more...
```

## Testing

Run the test suite to verify all implementations:

```bash
pixi run test_level1
```

## Project Structure

```
mojoBLAS/
├── src/
│   ├── __init__.mojo          # Main package initialization
│   ├── type_aliases.mojo      # Type definitions and aliases
│   └── level1/               # Level 1 BLAS implementations
│       ├── __init__.mojo     # Level 1 exports
│       ├── copy.mojo         # Vector copy
│       ├── scal.mojo         # Vector scaling
│       ├── axpy.mojo         # AXPY operation
│       ├── dot.mojo          # dot product
│       ├── nrm2.mojo         # euclidean norm
│       ├── asum.mojo         # sum of absolute values
│       ├── swap.mojo         # vector swap
│       ├── iamax.mojo        # index of max element
│       ├── rotg.mojo         # generate givens rotation
│       └── rot.mojo          # apply givens rotation
├── tests/
│   └── test_level1.mojo      # level 1 blas tests
├── pixi.toml                 # project configuration
└── readme.md                 # this file
```

## Roadmap

### Short term goals:
- [x] **Level 1 BLAS**: Completed.
- [ ] **Level 2 BLAS**: Matrix-vector operations (GEMV, GER, etc.)
- [ ] **Level 3 BLAS**: Matrix-matrix operations (GEMM, TRMM, etc.)

# Long terms goals:
- [ ] **LAPACK Subset**: Selected linear algebra routines
- [ ] **GPU Acceleration**: CUDA/ROCm backend support

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests. Any help with benchmarking mojoBLAS with BLAS will be appreciated too :)

### Guidelines

- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Modular Team**: For creating the amazing Mojo language.
- **BLAS Community**: For establishing the standard linear algebra interface
- **Contributors**: Everyone who helps make this project better

## 📚 References

- [BLAS (Basic Linear Algebra Subprograms)](https://netlib.org/blas/)
- [Mojo Programming Language](https://docs.modular.com/mojo/)
- [Linear Algebra PACKage (LAPACK)](https://netlib.org/lapack/)

---
