from std.memory import UnsafePointer, memset_zero
from std.time import sleep
import std.benchmark as benchmark
from std.benchmark import keep

from mojoblas.level3 import *

comptime f64 = DType.float64

def bench_dgemm[current_size: Int]() raises -> Float64:
    var m = current_size
    var n = current_size
    var k = current_size
    var lda = m
    var ldb = k
    var ldc = m
    var a = alloc[Scalar[f64]](lda * k)
    var b = alloc[Scalar[f64]](ldb * n)
    var c = alloc[Scalar[f64]](ldc * n)
    for i in range(lda * k):
        a[i] = Float64(i + 1)
    for i in range(ldb * n):
        b[i] = Float64(i + 1)
    for i in range(ldc * n):
        c[i] = Float64(i + 1)

    @parameter
    def dgemm_only()  -> None:
        gemm[f64]("N", "N", m, n, k, Float64(1.0), a, lda, b, ldb, Float64(1.0), c, ldc)

    keep(a)
    keep(b)
    keep(c)

    var report = benchmark.run[dgemm_only](max_runtime_secs=1)

    a.free()
    b.free()
    c.free()

    return report.mean("ns")

def bench_dsyrk[current_size: Int]() raises -> Float64:
    var n = current_size
    var k = current_size
    var lda = n
    var ldc = n
    var a = alloc[Scalar[f64]](lda * k)
    var c = alloc[Scalar[f64]](ldc * n)
    for i in range(lda * k):
        a[i] = Float64(i + 1)
    for i in range(ldc * n):
        c[i] = Float64(i + 1)

    @parameter
    def dsyrk_only() -> None:
        syrk[f64]("U", "N", n, k, Float64(1.0), a, lda, Float64(1.0), c, ldc)

    keep(a)
    keep(c)

    var report = benchmark.run[dsyrk_only](max_runtime_secs=1)

    a.free()
    c.free()

    return report.mean("ns")

def bench_dsyr2k[current_size: Int]() raises -> Float64:
    var n = current_size
    var k = current_size
    var lda = n
    var ldb = n
    var ldc = n
    var a = alloc[Scalar[f64]](lda * k)
    var b = alloc[Scalar[f64]](ldb * k)
    var c = alloc[Scalar[f64]](ldc * n)
    for i in range(lda * k):
        a[i] = Float64(i + 1)
    for i in range(ldb * k):
        b[i] = Float64(i + 1)
    for i in range(ldc * n):
        c[i] = Float64(i + 1)

    @parameter
    def dsyr2k_only() -> None:
        syr2k[f64]("U", "N", n, k, Float64(1.0), a, lda, b, ldb, Float64(1.0), c, ldc)

    keep(a)
    keep(b)
    keep(c)

    var report = benchmark.run[dsyr2k_only](max_runtime_secs=1)

    a.free()
    b.free()
    c.free()

    return report.mean("ns")

def bench_dsymm[current_size: Int]() raises -> Float64:
    var m = current_size
    var n = current_size
    var lda = m
    var ldb = m
    var ldc = m
    var a = alloc[Scalar[f64]](lda * m)
    var b = alloc[Scalar[f64]](ldb * n)
    var c = alloc[Scalar[f64]](ldc * n)
    for i in range(lda * m):
        a[i] = Float64(i + 1)
    for i in range(ldb * n):
        b[i] = Float64(i + 1)
    for i in range(ldc * n):
        c[i] = Float64(i + 1)

    @parameter
    def dsymm_only() -> None:
        symm[f64]("L", "U", m, n, Float64(1.0), a, lda, b, ldb, Float64(1.0), c, ldc)

    keep(a)
    keep(b)
    keep(c)

    var report = benchmark.run[dsymm_only](max_runtime_secs=1)

    a.free()
    b.free()
    c.free()

    return report.mean("ns")

def bench_dtrmm[current_size: Int]() raises -> Float64:
    var m = current_size
    var n = current_size
    var lda = m
    var ldb = m
    var a = alloc[Scalar[f64]](lda * m)
    var b = alloc[Scalar[f64]](ldb * n)
    for i in range(lda * m):
        a[i] = Float64(i + 1)
    for i in range(ldb * n):
        b[i] = Float64(i + 1)

    @parameter
    def dtrmm_only() -> None:
        trmm[f64]("L", "U", "N", "N", m, n, Float64(1.0), a, lda, b, ldb)

    keep(a)
    keep(b)

    var report = benchmark.run[dtrmm_only](max_runtime_secs=1)

    a.free()
    b.free()

    return report.mean("ns")

def bench_dtrsm[current_size: Int]() raises -> Float64:
    var m = current_size
    var n = current_size
    var lda = m
    var ldb = m
    var a = alloc[Scalar[f64]](lda * m)
    var b = alloc[Scalar[f64]](ldb * n)
    for j in range(m):
        for i in range(m):
            if i > j:
                a[i + j * lda] = Float64(0.0)
            elif i == j:
                a[i + j * lda] = Float64(1.0) + Float64((i % 9) + 1) * Float64(1e-3)
            else:
                a[i + j * lda] = Float64(((i + j) % 17) + 1) * Float64(1e-4)
    for i in range(ldb * n):
        b[i] = Float64(1.0) + Float64((i % 23) + 1) * Float64(1e-3)

    @parameter
    def dtrsm_only() -> None:
        trsm[f64]("L", "U", "N", "N", m, n, Float64(1.0), a, lda, b, ldb)

    keep(a)
    keep(b)

    var report = benchmark.run[dtrsm_only](max_runtime_secs=1)

    a.free()
    b.free()

    return report.mean("ns")

comptime sizes: List[Int] = [32, 64, 128, 256, 512]

def benchmark_gemm() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dgemm[size]())
    return times^

def benchmark_syrk() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dsyrk[size]())
    return times^

def benchmark_syr2k() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dsyr2k[size]())
    return times^

def benchmark_symm() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dsymm[size]())
    return times^

def benchmark_trmm() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dtrmm[size]())
    return times^

def benchmark_trsm() raises -> List[Float64]:
    var times: List[Float64] = []
    comptime for size in materialize[sizes]():
        times.append(bench_dtrsm[size]())
    return times^

def main() raises:
    var min_n: Int = 32
    var max_n: Int = 512
    var step: Int = 2
    var first = True
    var gemm_ns = benchmark_gemm()
    var syrk_ns = benchmark_syrk()
    var syr2k_ns = benchmark_syr2k()
    var symm_ns = benchmark_symm()
    var trmm_ns = benchmark_trmm()
    var trsm_ns = benchmark_trsm()
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
        var gemm_ns_value = gemm_ns[idx]
        var syrk_ns_value = syrk_ns[idx]
        var syr2k_ns_value = syr2k_ns[idx]
        var symm_ns_value = symm_ns[idx]
        var trmm_ns_value = trmm_ns[idx]
        var trsm_ns_value = trsm_ns[idx]
        var gemm_s = gemm_ns_value * 1e-9
        var syrk_s = syrk_ns_value * 1e-9
        var syr2k_s = syr2k_ns_value * 1e-9
        var symm_s = symm_ns_value * 1e-9
        var trmm_s = trmm_ns_value * 1e-9
        var trsm_s = trsm_ns_value * 1e-9

        if not first:
            print(",")
        first = False
        print("    {\"lib\":\"mojo\",\"op\":\"gemm\",\"n\":", size, ",\"avg_ns\":", gemm_ns_value, ",\"avg_seconds\":", gemm_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"syrk\",\"n\":", size, ",\"avg_ns\":", syrk_ns_value, ",\"avg_seconds\":", syrk_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"syr2k\",\"n\":", size, ",\"avg_ns\":", syr2k_ns_value, ",\"avg_seconds\":", syr2k_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"symm\",\"n\":", size, ",\"avg_ns\":", symm_ns_value, ",\"avg_seconds\":", symm_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"trmm\",\"n\":", size, ",\"avg_ns\":", trmm_ns_value, ",\"avg_seconds\":", trmm_s, "}")
        print(",")
        print("    {\"lib\":\"mojo\",\"op\":\"trsm\",\"n\":", size, ",\"avg_ns\":", trsm_ns_value, ",\"avg_seconds\":", trsm_s, "}")

        idx += 1

    print("  ]")
    print("}")
