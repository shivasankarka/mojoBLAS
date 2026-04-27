from std.memory import UnsafePointer, memset_zero
from std.time import sleep
import std.benchmark as benchmark
from std.benchmark import keep

from mojoblas.level2 import *

comptime f64 = DType.float64

def bench_dgemv[current_size: Int]() raises -> Float64:
    var m = current_size
    var n = current_size
    var lda = m
    var x = alloc[Scalar[f64]](n)
    var y = alloc[Scalar[f64]](m)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(m * n):
        a[i] = Float64(i + 1)
    for i in range(n):
        x[i] = Float64(i + 1)
    for i in range(m):
        y[i] = Float64(i + 1)

    @parameter
    def dgemv_only() -> None:
        gemv[f64]("N", m, n, Float64(1.0), a, lda, x, 1, Float64(0.0), y, 1)

    keep(a)
    keep(x)
    keep(y)

    var report = benchmark.run[dgemv_only](max_runtime_secs=1)

    a.free()
    x.free()
    y.free()

    return report.mean("ns")

def bench_dgemv_trans[current_size: Int]() raises -> Float64:
    var m = current_size
    var n = current_size
    var lda = m
    var x = alloc[Scalar[f64]](m)
    var y = alloc[Scalar[f64]](n)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(m * n):
        a[i] = Float64(i + 1)
    for i in range(m):
        x[i] = Float64(i + 1)
    for i in range(n):
        y[i] = Float64(i + 1)

    @parameter
    def dgemv_trans_only() -> None:
        gemv[f64]("T", m, n, Float64(1.0), a, lda, x, 1, Float64(0.0), y, 1)

    keep(a)
    keep(x)
    keep(y)

    var report = benchmark.run[dgemv_trans_only](max_runtime_secs=1)

    a.free()
    x.free()
    y.free()

    return report.mean("ns")

def bench_dtrmv[current_size: Int]() raises -> Float64:
    var n = current_size
    var lda = n
    var x = alloc[Scalar[f64]](n)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(n * n):
        a[i] = Float64(i + 1)
    for i in range(n):
        x[i] = Float64(i + 1)

    @parameter
    def dtrmv_only() -> None:
        trmv[f64]("U", "N", "N", n, a, lda, x, 1)

    keep(a)
    keep(x)

    var report = benchmark.run[dtrmv_only](max_runtime_secs=1)

    a.free()
    x.free()

    return report.mean("ns")

def bench_dtrsv[current_size: Int]() raises -> Float64:
    var n = current_size
    var lda = n
    var x = alloc[Scalar[f64]](n)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(n * n):
        a[i] = Float64(i + 1)
    for i in range(n):
        x[i] = Float64(i + 1)

    @parameter
    def dtrsv_only() -> None:
        trsv[f64]("U", "N", "N", n, a, lda, x, 1)

    keep(a)
    keep(x)

    var report = benchmark.run[dtrsv_only](max_runtime_secs=1)

    a.free()
    x.free()

    return report.mean("ns")

def bench_dsymv[current_size: Int]() raises -> Float64:
    var n = current_size
    var lda = n
    var x = alloc[Scalar[f64]](n)
    var y = alloc[Scalar[f64]](n)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(n * n):
        a[i] = Float64(i + 1)
    for i in range(n):
        x[i] = Float64(i + 1)
        y[i] = Float64(i + 1)

    @parameter
    def dsymv_only() -> None:
        symv[f64]("U", n, Float64(1.0), a, lda, x, 1, Float64(0.0), y, 1)

    keep(a)
    keep(x)
    keep(y)

    var report = benchmark.run[dsymv_only](max_runtime_secs=1)

    a.free()
    x.free()
    y.free()

    return report.mean("ns")

def bench_dsyr[current_size: Int]() raises -> Float64:
    var n = current_size
    var lda = n
    var x = alloc[Scalar[f64]](n)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(n * n):
        a[i] = Float64(i + 1)
    for i in range(n):
        x[i] = Float64(i + 1)

    @parameter
    def dsyr_only() -> None:
        syr[f64]("U", n, Float64(1.0), x, 1, a, lda)

    keep(a)
    keep(x)

    var report = benchmark.run[dsyr_only](max_runtime_secs=1)

    a.free()
    x.free()

    return report.mean("ns")

def bench_dsyr2[current_size: Int]() raises -> Float64:
    var n = current_size
    var lda = n
    var x = alloc[Scalar[f64]](n)
    var y = alloc[Scalar[f64]](n)
    var a = alloc[Scalar[f64]](lda * n)
    for i in range(n * n):
        a[i] = Float64(i + 1)
    for i in range(n):
        x[i] = Float64(i + 1)
        y[i] = Float64(i + 1)

    @parameter
    def dsyr2_only() -> None:
        syr2[f64]("U", n, Float64(1.0), x, 1, y, 1, a, lda)

    keep(a)
    keep(x)
    keep(y)

    var report = benchmark.run[dsyr2_only](max_runtime_secs=1)

    a.free()
    x.free()
    y.free()

    return report.mean("ns")

comptime sizes: List[Int] = [32, 64, 128, 256, 512]

def benchmark_gemv() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dgemv[size]())
    return times^

def benchmark_gemv_trans() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dgemv_trans[size]())
    return times^

def benchmark_trmv() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dtrmv[size]())
    return times^

def benchmark_trsv() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dtrsv[size]())
    return times^

def benchmark_symv() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dsymv[size]())
    return times^

def benchmark_syr() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dsyr[size]())
    return times^

def benchmark_syr2() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dsyr2[size]())
    return times^

def main() raises:
    var min_n: Int = 32
    var max_n: Int = 512
    var step: Int = 2
    var first = True
    var gemv_ns = benchmark_gemv()
    var gemv_trans_ns = benchmark_gemv_trans()
    var trmv_ns = benchmark_trmv()
    var trsv_ns = benchmark_trsv()
    var symv_ns = benchmark_symv()
    var syr_ns = benchmark_syr()
    var syr2_ns = benchmark_syr2()
    var idx = 0

    print("{")
    print("  \"metadata\": {")
    print("    \"min_n\": ", min_n, ",")
    print("    \"max_n\": ", max_n, ",")
    print("    \"step\": ", step, ",")
    print("    \"sizes\": [32, 64, 128, 256, 512]")
    print("  },")
    print("  \"results\": [")

    comptime for size in materialize[sizes]():
        var gemv_ns_value = gemv_ns[idx]
        var gemv_trans_ns_value = gemv_trans_ns[idx]
        var trmv_ns_value = trmv_ns[idx]
        var trsv_ns_value = trsv_ns[idx]
        var symv_ns_value = symv_ns[idx]
        var syr_ns_value = syr_ns[idx]
        var syr2_ns_value = syr2_ns[idx]
        var gemv_s = gemv_ns_value * 1e-9
        var gemv_trans_s = gemv_trans_ns_value * 1e-9
        var trmv_s = trmv_ns_value * 1e-9
        var trsv_s = trsv_ns_value * 1e-9
        var symv_s = symv_ns_value * 1e-9
        var syr_s = syr_ns_value * 1e-9
        var syr2_s = syr2_ns_value * 1e-9

        if not first:
            print(",")
        first = False
        print("    {\"lib\":\"mojo\",\"op\":\"gemv\",\"n\":", size, ",\"avg_ns\":", gemv_ns_value, ",\"avg_seconds\":", gemv_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"gemv_trans\",\"n\":", size, ",\"avg_ns\":", gemv_trans_ns_value, ",\"avg_seconds\":", gemv_trans_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"trmv\",\"n\":", size, ",\"avg_ns\":", trmv_ns_value, ",\"avg_seconds\":", trmv_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"trsv\",\"n\":", size, ",\"avg_ns\":", trsv_ns_value, ",\"avg_seconds\":", trsv_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"symv\",\"n\":", size, ",\"avg_ns\":", symv_ns_value, ",\"avg_seconds\":", symv_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"syr\",\"n\":", size, ",\"avg_ns\":", syr_ns_value, ",\"avg_seconds\":", syr_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"syr2\",\"n\":", size, ",\"avg_ns\":", syr2_ns_value, ",\"avg_seconds\":", syr2_s, "}")

        idx += 1

    print("  ]")
    print("}")
