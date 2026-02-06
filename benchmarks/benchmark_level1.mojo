from src.level1 import *
from memory import UnsafePointer
from memory import memset_zero
from time import sleep
import benchmark
from benchmark import keep

fn bench_daxpy[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    var y = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)
        y[i] = Float32((i + 1) * 10)

    @parameter
    fn daxpy_only() -> None:
        daxpy[DType.float32](current_size, Float32(2.0), x, 1, y, 1)
        keep(x)
        keep(y)

    var report = benchmark.run[daxpy_only](max_runtime_secs=1)


    x.free()
    y.free()

    return report.mean("ns")

fn bench_dcopy[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    var y = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)

    @parameter
    fn dcopy_only() -> None:
        dcopy[DType.float32](current_size, x, 1, y, 1)
        keep(x)
        keep(y)

    var report = benchmark.run[dcopy_only](max_runtime_secs=1)


    x.free()
    y.free()

    return report.mean("ns")


fn bench_dscal[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)

    @parameter
    fn dscal_only() -> None:
        dscal[DType.float32](current_size, Float32(2.5), x, 1)
        keep(x)

    var report = benchmark.run[dscal_only](max_runtime_secs=1)


    x.free()

    return report.mean("ns")


fn bench_ddot[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    var y = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)
        y[i] = Float32(i + 2)

    @parameter
    fn ddot_only() -> None:
        var result = ddot[DType.float32](current_size, x, 1, y, 1)
        keep(result)
        keep(x)
        keep(y)

    var report = benchmark.run[ddot_only](max_runtime_secs=1)


    x.free()
    y.free()

    return report.mean("ns")


fn bench_dasum[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1) if i % 2 == 0 else Float32(-(i + 1))

    @parameter
    fn dasum_only() -> None:
        var result = dasum[DType.float32](current_size, x, 1)
        keep(result)
        keep(x)

    var report = benchmark.run[dasum_only](max_runtime_secs=1)


    x.free()

    return report.mean("ns")


fn bench_dnrm2[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)

    @parameter
    fn dnrm2_only() -> None:
        var result = dnrm2[DType.float32](current_size, x, 1)
        keep(result)
        keep(x)

    var report = benchmark.run[dnrm2_only](max_runtime_secs=1)


    x.free()

    return report.mean("ns")


fn bench_dswap[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    var y = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)
        y[i] = Float32((i + 1) * 10)

    @parameter
    fn dswap_only() -> None:
        dswap[DType.float32](current_size, x, 1, y, 1)
        keep(x)
        keep(y)

    var report = benchmark.run[dswap_only](max_runtime_secs=1)


    x.free()
    y.free()

    return report.mean("ns")


fn bench_diamax[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1) if i != current_size // 2 else Float32(current_size * 2)

    @parameter
    fn diamax_only() -> None:
        var result = di_amax[DType.float32](current_size, x, 1)
        keep(result)
        keep(x)

    var report = benchmark.run[diamax_only](max_runtime_secs=1)


    x.free()

    return report.mean("ns")


fn bench_drot[current_size: Int]() raises -> Float64:
    var x = alloc[Scalar[DType.float32]](current_size)
    var y = alloc[Scalar[DType.float32]](current_size)
    for i in range(current_size):
        x[i] = Float32(i + 1)
        y[i] = Float32((i + 1) * 2)

    @parameter
    fn drot_only() -> None:
        drot[DType.float32](current_size, x, 1, y, 1, Float32(0.6), Float32(0.8))
        keep(x)
        keep(y)

    var report = benchmark.run[drot_only](max_runtime_secs=1)


    x.free()
    y.free()

    return report.mean("ns")


fn bench_drotg[current_size: Int]() raises -> Float64:
    var a = alloc[Scalar[DType.float32]](1)
    var b = alloc[Scalar[DType.float32]](1)
    var c = alloc[Scalar[DType.float32]](1)
    var s = alloc[Scalar[DType.float32]](1)

    a[0] = Float32(3.0)
    b[0] = Float32(4.0)

    @parameter
    fn drotg_only() -> None:
        drotg[DType.float32](a, b, c, s)
        keep(a)
        keep(b)
        keep(c)
        keep(s)

    var report = benchmark.run[drotg_only](max_runtime_secs=1)


    a.free()
    b.free()
    c.free()
    s.free()

    return report.mean("ns")


comptime sizes: List[Int] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

fn benchmark_axpy() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_daxpy[size]())
    return times^

fn benchmark_copy() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_dcopy[size]())
    return times^


fn benchmark_scal() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_dscal[size]())
    return times^


fn benchmark_dot() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_ddot[size]())
    return times^


fn benchmark_asum() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_dasum[size]())
    return times^


fn benchmark_nrm2() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_dnrm2[size]())
    return times^


fn benchmark_swap() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_dswap[size]())
    return times^


fn benchmark_iamax() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_diamax[size]())
    return times^


fn benchmark_rot() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_drot[size]())
    return times^


fn benchmark_rotg() raises -> List[Float64]:
    var times: List[Float64] = []
    @parameter
    for size in materialize[sizes]():
        times.append(bench_drotg[size]())
    return times^


fn main() raises:
    var min_n: Int = 256
    var max_n: Int = 262144
    var step: Int = 2
    var first = True
    var axpy_ns = benchmark_axpy()
    var scal_ns = benchmark_scal()
    var dot_ns = benchmark_dot()
    var nrm2_ns = benchmark_nrm2()
    var sum_ns = benchmark_asum()
    var idx = 0

    print("{")
    print("  \"metadata\": {")
    print("    \"min_n\": ", min_n, ",")
    print("    \"max_n\": ", max_n, ",")
    print("    \"step\": ", step, ",")
    print("    \"sizes\": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]")
    print("  },")
    print("  \"results\": [")

    @parameter
    for size in materialize[sizes]():
        var axpy_ns_value = axpy_ns[idx]
        var scal_ns_value = scal_ns[idx]
        var dot_ns_value = dot_ns[idx]
        var nrm2_ns_value = nrm2_ns[idx]
        var sum_ns_value = sum_ns[idx]
        var axpy_s = axpy_ns_value * 1e-9
        var scal_s = scal_ns_value * 1e-9
        var dot_s = dot_ns_value * 1e-9
        var nrm2_s = nrm2_ns_value * 1e-9
        var sum_s = sum_ns_value * 1e-9

        if not first:
            print(",")
        first = False
        print("    {\"lib\":\"mojo\",\"op\":\"axpy\",\"n\":", size, ",\"avg_ns\":", axpy_ns_value, ",\"avg_seconds\":", axpy_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"scal\",\"n\":", size, ",\"avg_ns\":", scal_ns_value, ",\"avg_seconds\":", scal_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"dot\",\"n\":", size, ",\"avg_ns\":", dot_ns_value, ",\"avg_seconds\":", dot_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"nrm2\",\"n\":", size, ",\"avg_ns\":", nrm2_ns_value, ",\"avg_seconds\":", nrm2_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"sum\",\"n\":", size, ",\"avg_ns\":", sum_ns_value, ",\"avg_seconds\":", sum_s, "}")

        idx += 1

    print("  ]")
    print("}")
