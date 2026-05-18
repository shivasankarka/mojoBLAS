# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Vector Alpha Plus Operations (`level1.axpy`)
============================================
Provides vector alpha plus operations as defined in the BLAS library standard.
"""

from std.algorithm.functional import parallelize
from std.sys.info import simd_width_of
from ._tuning import (
    AXPY_N_THREADS,
    AXPY_PAR_THRESHOLD,
    AXPY_MIN_CHUNK_PER_THREAD,
    AXPY_UNROLL,
    AXPY_N_ACC,
)


def _axpy_serial[
    mut: Bool,
    origin_xc: Origin[mut=mut],
    origin_yc: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    n_acc: Int,
](
    xc: BLASPtr[dtype, origin_xc],
    yc: BLASPtr[dtype, origin_yc],
    da: Scalar[dtype],
    length: Int,
):
    """Inner kernel: yc[i] = da * xc[i] + yc[i] with n_acc independent SIMD streams."""
    comptime stride: Int = simd_width * n_acc
    var i = 0
    while i + stride <= length:
        yc.store[width=simd_width](i + 0 * simd_width, da * xc.load[width=simd_width](i + 0 * simd_width) + yc.load[width=simd_width](i + 0 * simd_width))
        yc.store[width=simd_width](i + 1 * simd_width, da * xc.load[width=simd_width](i + 1 * simd_width) + yc.load[width=simd_width](i + 1 * simd_width))
        yc.store[width=simd_width](i + 2 * simd_width, da * xc.load[width=simd_width](i + 2 * simd_width) + yc.load[width=simd_width](i + 2 * simd_width))
        yc.store[width=simd_width](i + 3 * simd_width, da * xc.load[width=simd_width](i + 3 * simd_width) + yc.load[width=simd_width](i + 3 * simd_width))
        i += stride
    while i + simd_width <= length:
        yc.store[width=simd_width](i, da * xc.load[width=simd_width](i) + yc.load[width=simd_width](i))
        i += simd_width
    while i < length:
        yc[i] = da * xc[i] + yc[i]
        i += 1


def _axpy_add_serial[
    mut: Bool,
    origin_xc: Origin[mut=mut],
    origin_yc: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    n_acc: Int,
](xc: BLASPtr[dtype, origin_xc], yc: BLASPtr[dtype, origin_yc], length: Int):
    """Inner kernel for da==1: yc[i] += xc[i]."""
    comptime stride: Int = simd_width * n_acc
    var i = 0
    while i + stride <= length:
        yc.store[width=simd_width](i + 0 * simd_width, xc.load[width=simd_width](i + 0 * simd_width) + yc.load[width=simd_width](i + 0 * simd_width))
        yc.store[width=simd_width](i + 1 * simd_width, xc.load[width=simd_width](i + 1 * simd_width) + yc.load[width=simd_width](i + 1 * simd_width))
        yc.store[width=simd_width](i + 2 * simd_width, xc.load[width=simd_width](i + 2 * simd_width) + yc.load[width=simd_width](i + 2 * simd_width))
        yc.store[width=simd_width](i + 3 * simd_width, xc.load[width=simd_width](i + 3 * simd_width) + yc.load[width=simd_width](i + 3 * simd_width))
        i += stride
    while i + simd_width <= length:
        yc.store[width=simd_width](i, xc.load[width=simd_width](i) + yc.load[width=simd_width](i))
        i += simd_width
    while i < length:
        yc[i] += xc[i]
        i += 1


def _axpy_sub_serial[
    mut: Bool,
    origin_xc: Origin[mut=mut],
    origin_yc: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    n_acc: Int,
](xc: BLASPtr[dtype, origin_xc], yc: BLASPtr[dtype, origin_yc], length: Int):
    """Inner kernel for da==-1: yc[i] -= xc[i]."""
    comptime stride: Int = simd_width * n_acc
    var i = 0
    while i + stride <= length:
        yc.store[width=simd_width](i + 0 * simd_width, yc.load[width=simd_width](i + 0 * simd_width) - xc.load[width=simd_width](i + 0 * simd_width))
        yc.store[width=simd_width](i + 1 * simd_width, yc.load[width=simd_width](i + 1 * simd_width) - xc.load[width=simd_width](i + 1 * simd_width))
        yc.store[width=simd_width](i + 2 * simd_width, yc.load[width=simd_width](i + 2 * simd_width) - xc.load[width=simd_width](i + 2 * simd_width))
        yc.store[width=simd_width](i + 3 * simd_width, yc.load[width=simd_width](i + 3 * simd_width) - xc.load[width=simd_width](i + 3 * simd_width))
        i += stride
    while i + simd_width <= length:
        yc.store[width=simd_width](i, yc.load[width=simd_width](i) - xc.load[width=simd_width](i))
        i += simd_width
    while i < length:
        yc[i] -= xc[i]
        i += 1


def axpy[
    mut: Bool,
    origin_x: Origin[mut=mut],
    origin_y: MutOrigin,
    //,
    dtype: DType,
    *,
    n_threads: Int = AXPY_N_THREADS,
    par_threshold: Int = AXPY_PAR_THRESHOLD,
    min_chunk: Int = AXPY_MIN_CHUNK_PER_THREAD,
    unroll_factor: Int = AXPY_UNROLL,
    n_acc: Int = AXPY_N_ACC,
](
    n: Int,
    da: Scalar[dtype],
    dx: BLASPtr[dtype, origin_x],
    incx: Int,
    dy: BLASPtr[dtype, origin_y],
    incy: Int,
):
    """
    Perform the AXPY operation: Y := alpha * X + Y.

    Contiguous path (incx == incy == 1):
        Uses n_acc independent SIMD streams of width simd_width_of[dtype]() to
        hide FMA latency without a serial accumulator dependency chain. Parallelizes
        across n_threads when n exceeds par_threshold and each thread would receive
        at least min_chunk elements (guarded by nt >= 2 to avoid spawn overhead).
        Special-cased for da == 1 (add) and da == -1 (subtract) to skip the multiply.

    Strided path (incx != 1 or incy != 1):
        Scalar loop with da == 1 / da == -1 fast paths. Same-stride positive
        case (incx == incy > 0) uses a single index to avoid dual-pointer tracking.

    Parameters:
        mut: Mutability of the pointer to X.
        origin_x: Memory origin of X.
        origin_y: Memory origin of Y.
        dtype: Element data type.
        n_threads: Max threads for parallel execution.
        par_threshold: Minimum n to consider parallelism.
        min_chunk: Minimum elements per thread; caps active thread count so
                   nt = min(n_threads, n // min_chunk). Parallel path is skipped
                   entirely when nt < 2.
        unroll_factor: Reserved for future use.
        n_acc: Number of independent SIMD streams in the inner kernel (default 4).

    Args:
        n: Number of elements. No-op if n <= 0 or da == 0.
        da: Scalar multiplier for X.
        dx: Pointer to X; must span at least 1 + (n-1)*abs(incx) elements.
        incx: Stride for X. Negative strides traverse X in reverse.
        dy: Pointer to Y; must span at least 1 + (n-1)*abs(incy) elements.
        incy: Stride for Y. Negative strides traverse Y in reverse.
    """
    if n <= 0 or da == 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()

    if incx != 1 or incy != 1:
        if incx == incy and incx > 0:
            var nsteps = n * incx
            if da == 1:
                for i in range(0, nsteps, incx):
                    dy[i] += dx[i]
            elif da == -1:
                for i in range(0, nsteps, incx):
                    dy[i] -= dx[i]
            else:
                for i in range(0, nsteps, incx):
                    dy[i] = da * dx[i] + dy[i]
            return

        var ix: Int = 0
        var iy: Int = 0
        if incx < 0:
            ix = (-n + 1) * incx
        if incy < 0:
            iy = (-n + 1) * incy
        if da == 1:
            for _ in range(n):
                dy[iy] += dx[ix]
                ix += incx
                iy += incy
        elif da == -1:
            for _ in range(n):
                dy[iy] -= dx[ix]
                ix += incx
                iy += incy
        else:
            for _ in range(n):
                dy[iy] = da * dx[ix] + dy[iy]
                ix += incx
                iy += incy
        return

    if n > par_threshold:
        var nt = min(n_threads, max(1, n // min_chunk))
        if nt >= 2:
            var chunk_size = (n + nt - 1) // nt

            if da == 1:
                @parameter
                def worker_add(tid: Int):
                    var start = tid * chunk_size
                    var end = min(start + chunk_size, n)
                    if end <= start:
                        return
                    _axpy_add_serial[dtype, simd_width, n_acc](
                        dx + start, dy + start, end - start
                    )

                parallelize[worker_add](nt)
            elif da == -1:
                @parameter
                def worker_sub(tid: Int):
                    var start = tid * chunk_size
                    var end = min(start + chunk_size, n)
                    if end <= start:
                        return
                    _axpy_sub_serial[dtype, simd_width, n_acc](
                        dx + start, dy + start, end - start
                    )

                parallelize[worker_sub](nt)
            else:
                @parameter
                def worker(tid: Int):
                    var start = tid * chunk_size
                    var end = min(start + chunk_size, n)
                    if end <= start:
                        return
                    _axpy_serial[dtype, simd_width, n_acc](
                        dx + start, dy + start, da, end - start
                    )

                parallelize[worker](nt)
            return

    if da == 1:
        _axpy_add_serial[dtype, simd_width, n_acc](dx, dy, n)
    elif da == -1:
        _axpy_sub_serial[dtype, simd_width, n_acc](dx, dy, n)
    else:
        _axpy_serial[dtype, simd_width, n_acc](dx, dy, da, n)


# NOTE: GPU implementation are commented out for now. Undergoing development :)

# def gdaxpy_backend[
#     dtype: DType, SIMD_WIDTH: Int
# ](
#     n: Int,
#     da: Scalar[dtype],
#     dx: UnsafePointer[Scalar[dtype], MutAnyOrigin],
#     dy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
# ):
#     var tid: Int = Int((block_idx.x * block_dim.x) + thread_idx.x)
#     if tid < n:
#         dy[tid] = da * dx[tid] + dy[tid]


# def gdaxpy[
#     dtype: DType
# ](
#     ctx: DeviceContext,
#     n: Int,
#     da: Scalar[dtype],
#     dx: DeviceBuffer[dtype],
#     incx: Int,
#     dy: DeviceBuffer[dtype],
#     incy: Int,
# ) raises -> None:
#     if n <= 0 or da == 0:
#         return
#     if incx == 0 or incy == 0:
#         return

#     comptime THREADS_PER_BLOCK = (256, 1)

#     if incx == 1 and incy == 1:
#         comptime SIMD_WIDTH = simd_width_of[dtype]() // 2
#         var work_per_block = THREADS_PER_BLOCK[0] * SIMD_WIDTH
#         var blocks_x = (n + work_per_block - 1) // work_per_block
#         var BLOCKS_PER_GRID = (blocks_x, 1)

#         ctx.enqueue_function[
#             gdaxpy_backend[dtype, SIMD_WIDTH],
#             gdaxpy_backend[dtype, SIMD_WIDTH],
#         ](
#             n,
#             da,
#             dx.unsafe_ptr(),
#             dy.unsafe_ptr(),
#             grid_dim=BLOCKS_PER_GRID,
#             block_dim=THREADS_PER_BLOCK,
#         )
#         return

#     var blocks_x = (n + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
#     var BLOCKS_PER_GRID = (blocks_x, 1)

#     ctx.enqueue_function[
#         gdaxpy_backend_strided[dtype],
#         gdaxpy_backend_strided[dtype],
#     ](
#         n,
#         da,
#         dx.unsafe_ptr(),
#         incx,
#         dy.unsafe_ptr(),
#         incy,
#         grid_dim=BLOCKS_PER_GRID,
#         block_dim=THREADS_PER_BLOCK,
#     )

#     ctx.synchronize()

# def zaxpy[
#     dtype: DType
# ](
#     n: Int,
#     ca: ComplexScalar[dtype],
#     cx: BLASPtr[ComplexScalar[dtype]],
#     incx: Int,
#     cy: BLASPtr[ComplexScalar[dtype]],
#     incy: Int,
# ) -> None where dtype.is_floating_point():
#     """
#     Perform the AXPY operation: Y := alpha * X + Y.

#     Args:
#         n: Number of elements in vectors X and Y.
#         ca: Scalar multiplier for vector X.
#         cx: Pointer to the first element of vector X.
#         incx: Increment for the elements of X.
#         cy: Pointer to the first element of vector Y.
#         incy: Increment for the elements of Y.
#     """
#     if n <= 0 or ca.squared_norm() == 0:
#         return

#     var ix: Int = 0
#     var iy: Int = 0
#     comptime simd_width: Int = simd_width_of[dtype]()

#     if incx == 1 and incy == 1:
#         for i in range(n):
#             cy[i] = ca * cx[i] + cy[i]
#         return

#     if incx < 0:
#         ix = (-n + 1) * incx
#     if incy < 0:
#         iy = (-n + 1) * incy

#     for i in range(n):
#         cy[iy] = ca * cx[ix] + cy[iy]
#         ix += incx
#         iy += incy
