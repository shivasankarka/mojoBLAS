# Changelog

All notable changes to mojoBLAS are documented here.

## [0.2.0] - 2026-09-04

0.1.0 was published to the modular-community prefix.dev channel but never
tagged/released on GitHub; this release supersedes it there.

### Changed

- Migrated off deprecated pointer APIs (`.load`/`.store`/`.free`/`alloc`/
  positional `__getitem__`/pointer `__add__`) to their `unsafe_*`
  equivalents in the test suites and benchmark/tuning scripts, matching
  the library code already migrated for 0.1.0 — `pixi run test_level1/2/3`
  builds and runs with zero deprecation warnings.
- Filled in missing `Args`/`Parameters` docstring sections on `gemm_v3`,
  `gemm_v6`, `gemm_v7`, `gemm_v12`, `spr`, and `spr2` to match their sibling
  routines, per `scripts/check_mojo_standards.py`.
- Fixed stale `from src.*` imports (leftover from the `src/` → `mojoblas/`
  rename) in `benchmark_gemm_variants.mojo`, `benchmark_gemm_versions.mojo`,
  `check_gemm_v11_correctness.mojo`, `tune_level1_routine.mojo`, and
  `tune_level3_gemm.mojo`.
- Moved `CHANGELOG.md` to `docs/CHANGELOG.md`.

## [0.1.0] - 2026-08-27

First tagged release.

### Added

- **Level 1 BLAS**: `asum`, `axpy`, `copy`, `dot`, `iamax`, `nrm2`, `rot`,
  `rotg`, `rotm`, `rotmg`, `scal`, `swap` — SIMD-vectorized unit-stride
  paths with `parallelize` above per-routine thresholds.
- **Level 2 BLAS**: `gbmv`, `gemv`, `ger`, `sbmv`, `spmv`, `spr`, `spr2`,
  `symv`, `syr`, `syr2`, `tbmv`, `tbsv`, `tpmv`, `tpsv`, `trmv`, `trsv`.
- **Level 3 BLAS**: `gemm`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`. GEMM
  includes a hand-tuned AMX-backed f32 kernel (`gemm_v11`/`v12`, ~960
  GFLOPS on Apple M2).
- Benchmarking suite comparing mojoBLAS against OpenBLAS and Apple
  Accelerate.
- Test suites for all three levels (92 tests total).

### Changed

- Package layout renamed from `src/` to `mojoblas/` to match the installed
  package name; all internal imports now use absolute `mojoblas.*` paths.
- Migrated off deprecated pointer APIs (`.load`/`.store`/`.free`/`alloc`/
  positional `__getitem__`/pointer `__add__`) to their `unsafe_*`
  equivalents, and off `std.algorithm.functional.parallelize` to
  `max.algorithm.backend.cpu.parallelize`, for compatibility with Mojo
  1.0.0.
- Pinned `mojo` to `==1.0.0` for package build/host/run dependencies and
  `>=1.0.0,<2` for the dev environment; pinned `max-core` to
  `>=26.5.0,<27`.

### Known limitations

- Strided paths (`incx`/`incy` != 1) across Level 1 are scalar and not
  parallelized; only the unit-stride fast paths are SIMD-vectorized.
- No complex number support.
- No GPU acceleration (exploratory code exists under `gpu/` but is not
  part of the package).
