# mojoBLAS

A high-performance **BLAS (Basic Linear Algebra Subprograms)** implementation written in [Mojo](https://modular.com/mojo), leveraging Mojo's powerful systems programming capabilities and zero-cost abstractions for maximum performance.

## Motivation

This project started as an attempt to implement BLAS backend for math operations in [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo). Then I thought why not try implementing the full BLAS in pure Mojo (We have all been there 😉). It's just fun to dive into coding these operations and go down the rabbit hole of optimizations (I've barely scratched the surface xD). Here's what I have so far:

- **Complete Real BLAS Implementation**: All 34 standard real BLAS routines (Level 1/2/3) implemented.
- **Generic Implementation**: Supports all real datatypes (through DType) in existing BLAS routines.
- **Comprehensive Testing**: Almost full test coverage for all three BLAS levels (with openblas reference values).

## 📦 Installation

### Prerequisites

- Currently works on latest **Mojo** version `>=0.26.2.0,<0.27` (see [Mojo installation guide](https://docs.modular.com/mojo/manual/get-started/))

### Setup

#### Method 1: As a dependency

1) Add to pixi.toml
```toml
[workspace]
preview = ["pixi-build"]

[dependencies]
mojo = ">=0.26.2.0,<0.27"
mojoblas = { git = "https://github.com/shivasankarka/mojoBLAS.git", branch = "main"}
```

#### Method 2: Clone and develop

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivasankarka/mojoBLAS.git
   cd mojoBLAS
   ```

2. **Install dependencies:**
   ```bash
   pixi install
   ```

## 🔧 Usage

### Basic Example

```mojo
from mojoblas.src.level1 import dot, axpy, nrm2

fn main():
    # Create vectors
    var x = alloc[Float32](3)
    var y = alloc[Float32](3)

    # Initialize data
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    y[0] = 4.0
    y[1] = 5.0
    y[2] = 6.0

    # Compute dot product: x · y
    var result = dot(3, x, 1, y, 1)
    print("Dot product:", result)  # Output: 32.0

    # Perform AXPY: y = α*x + y
    axpy(3, 2.0, x, 1, y, 1)
    print("After AXPY:", y[0], y[1], y[2])  # Output: 6.0, 9.0, 12.0

    # Compute Euclidean norm
    var norm = nrm2(3, x, 1)
    print("Euclidean norm:", norm)

    # Clean up
    x.free()
    y.free()
```

### Available Functions

See [docs/reference.md](docs/reference.md) for the complete list of available functions and their signatures.

## Testing

Run the test suite to verify all implementations:

```bash
# Level 1 tests
pixi run test_level1

# Level 2 tests
pixi run test_level2

# Level 3 tests
pixi run test_level3
```

## Benchmarking

The project includes a comprehensive benchmarking suite that compares mojoBLAS against system BLAS (OpenBLAS, Accelerate on macOS, OpenBLAS on Linux) across all three BLAS levels.

### Running Benchmarks

The benchmark environment is isolated from the main dependencies to keep the core environment lightweight. Install and run benchmarks using:

```bash
# Build and run C benchmarks (Accelerate + OpenBLAS)
pixi run -e bench bench_c_build
pixi run -e bench bench_c_run

# Run Mojo benchmarks for each level
pixi run -e bench bench_mojo_l1
pixi run -e bench bench_mojo_l2
pixi run -e bench bench_mojo_l3

# Generate comparison plots
pixi run -e bench bench_plot

# Run everything at once (C benchmarks, Mojo benchmarks, and plotting)
pixi run -e bench bench_all
```

**Benchmark outputs:**
- **JSON results**: `benchmarks/mojo_l{1,2,3}_results.json` and `benchmarks/c_bench_results.json`
- **Comparison plots**: `benchmarks/bench_plot_level{1,2,3}.png`

## Project Structure

```
mojoBLAS/
├── src/
│   ├── __init__.mojo              # Main package initialization
│   ├── type_aliases.mojo          # Type definitions and aliases
│   ├── level1/                    # Level 1 BLAS implementations
│   │   ├── __init__.mojo
│   │   ├── axpy.mojo
│   │   ├── asum.mojo
│   │   ├── copy.mojo
│   │   ├── dot.mojo
│   │   ├── iamax.mojo
│   │   ├── nrm2.mojo
│   │   ├── rot.mojo
│   │   ├── rotg.mojo
│   │   ├── rotm.mojo
│   │   ├── rotmg.mojo
│   │   ├── scal.mojo
│   │   └── swap.mojo
│   ├── level2/                    # Level 2 BLAS implementations
│   │   ├── __init__.mojo
│   │   ├── gemv.mojo
│   │   ├── ger.mojo
│   │   ├── gbmv.mojo
│   │   ├── sbmv.mojo
│   │   ├── spmv.mojo
│   │   ├── spr.mojo
│   │   ├── spr2.mojo
│   │   ├── symv.mojo
│   │   ├── syr.mojo
│   │   ├── syr2.mojo
│   │   ├── tbmv.mojo
│   │   ├── tbsv.mojo
│   │   ├── tpmv.mojo
│   │   ├── tpsv.mojo
│   │   ├── trmv.mojo
│   │   └── trsv.mojo
│   └── level3/                    # Level 3 BLAS implementations
│       ├── __init__.mojo
│       ├── gemm.mojo
│       ├── symm.mojo
│       ├── syrk.mojo
│       ├── syr2k.mojo
│       ├── trmm.mojo
│       └── trsm.mojo
├── tests/
│   ├── reference.c
│   ├── test_level1.mojo
│   ├── test_level2.mojo
│   ├── test_level2_extended.mojo
│   └── test_level3.mojo
├── benchmarks/
│   ├── benchmark_level1.mojo      # Level 1 Mojo benchmarks
│   ├── benchmark_level2.mojo      # Level 2 Mojo benchmarks
│   ├── benchmark_level3.mojo      # Level 3 Mojo benchmarks
│   ├── bench.c                    # C benchmark (Accelerate + OpenBLAS)
│   ├── Makefile                   # Build C benchmark
│   └── plot_bench.py              # Generate comparison plots
├── pixi.toml                      # Project configuration
└── README.md                      # This file
```

## Roadmap

### Completed:
- [x] **Level 1 BLAS**
- [x] **Level 2 BLAS**
- [x] **Level 3 BLAS**
- [x] **Benchmarking Suite**: Comparison against Accelerate and OpenBLAS.

### Future goals (In the order):
- [ ] **Performance Optimizations**: SIMD vectorization, parallel execution.
- [ ] **Complex Number Support**: Complex BLAS operations
- [ ] **GPU Acceleration**: CUDA/ROCm backend support

## Contributing

Contributions are welcome! Any help would be appreciated :)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

It is inspired by and based on the Netlib BLAS reference implementation:
http://www.netlib.org/blas/

Original authors:
Lawson, Hanson, Kincaid, Krogh, Dongarra, et al.

This is an independent reimplementation. Any errors or differences are our own.

- **Modular Team**: For creating the amazing Mojo language.
- **BLAS Community**: For establishing the standard linear algebra interface.

## 📚 References

- [BLAS (Basic Linear Algebra Subprograms)](https://netlib.org/blas/)
- [Mojo Programming Language](https://docs.modular.com/mojo/)

---
