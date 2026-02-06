from src.level1 import *
from memory import UnsafePointer
from time import sleep
import benchmark
from benchmark import keep

comptime SIZE: Int = 100

fn test_axpy():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var y = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)
        y[i] = Float32((i + 1) * 10)

    daxpy[DType.float32](SIZE, Float32(2.0), x, 1, y, 1)

    keep(x)
    keep(y)
    x.free()
    y.free()


fn test_copy() raises:
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var y = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)

    dcopy[DType.float32](SIZE, x, 1, y, 1)

    keep(x)
    keep(y)
    x.free()
    y.free()


fn test_scal():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)

    dscal[DType.float32](SIZE, Float32(2.5), x, 1)

    keep(x)
    x.free()


fn test_dot():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var y = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)
        y[i] = Float32(i + 2)

    var result = ddot[DType.float32](SIZE, x, 1, y, 1)

    keep(result)
    keep(x)
    keep(y)
    x.free()
    y.free()


fn test_asum():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1) if i % 2 == 0 else Float32(-(i + 1))

    var result = dasum[DType.float32](SIZE, x, 1)

    keep(result)
    keep(x)
    x.free()


fn test_nrm2():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)

    var result = dnrm2[DType.float32](SIZE, x, 1)

    keep(result)
    keep(x)
    x.free()


fn test_swap():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var y = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)
        y[i] = Float32((i + 1) * 10)

    dswap[DType.float32](SIZE, x, 1, y, 1)

    keep(x)
    keep(y)
    x.free()
    y.free()


fn test_iamax():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1) if i != SIZE // 2 else Float32(SIZE * 2)

    var result = di_amax[DType.float32](SIZE, x, 1)

    keep(result)
    keep(x)
    x.free()


fn test_rot():
    var x = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    var y = alloc[Scalar[DType.float32]](SIZE).unsafe_origin_cast[
        MutAnyOrigin
    ]()
    for i in range(SIZE):
        x[i] = Float32(i + 1)
        y[i] = Float32((i + 1) * 2)

    drot[DType.float32](
        SIZE, x, 1, y, 1, Float32(0.6), Float32(0.8)
    )

    keep(x)
    keep(y)
    x.free()
    y.free()


fn test_rotg():
    var a = alloc[Scalar[DType.float32]](1).unsafe_origin_cast[MutAnyOrigin]()
    var b = alloc[Scalar[DType.float32]](1).unsafe_origin_cast[MutAnyOrigin]()
    var c = alloc[Scalar[DType.float32]](1).unsafe_origin_cast[MutAnyOrigin]()
    var s = alloc[Scalar[DType.float32]](1).unsafe_origin_cast[MutAnyOrigin]()

    a[0] = Float32(3.0)
    b[0] = Float32(4.0)

    drotg[DType.float32](a, b, c, s)

    keep(a)
    keep(b)
    keep(c)
    keep(s)
    a.free()
    b.free()
    c.free()
    s.free()


fn benchmark_axpy() raises:
    print("DAXPY Benchmark (Y := alpha * X + Y)")
    var report = benchmark.run[func1=test_axpy](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_copy() raises:
    print("DCOPY Benchmark (Y := X)")
    var report = benchmark.run[func1=test_copy](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_scal() raises:
    print("DSCAL Benchmark (X := alpha * X)")
    var report = benchmark.run[func1=test_scal](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_dot() raises:
    print("DDOT Benchmark (dot product X • Y)")
    var report = benchmark.run[func1=test_dot](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_asum() raises:
    print("DASUM Benchmark (sum of absolute values)")
    var report = benchmark.run[func1=test_asum](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_nrm2() raises:
    print("DNRM2 Benchmark (Euclidean norm)")
    var report = benchmark.run[func1=test_nrm2](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_swap() raises:
    print("DSWAP Benchmark (swap X <-> Y)")
    var report = benchmark.run[func1=test_swap](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_iamax() raises:
    print("DIAMAX Benchmark (index of max absolute value)")
    var report = benchmark.run[func1=test_iamax](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_rot() raises:
    print("DROT Benchmark (Givens rotation)")
    var report = benchmark.run[func1=test_rot](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn benchmark_rotg() raises:
    print("DROTG Benchmark (construct Givens rotation)")
    var report = benchmark.run[func1=test_rotg](max_runtime_secs=1)
    report.print()
    print("Mean time (ns):", report.mean("ns"))
    print()


fn main() raises:
    print("mojoBLAS Level 1 BLAS Functions Benchmark")
    print()

    benchmark_axpy()
    benchmark_copy()
    benchmark_scal()
    benchmark_dot()
    benchmark_asum()
    benchmark_nrm2()
    benchmark_swap()
    benchmark_iamax()
    benchmark_rot()
    benchmark_rotg()
