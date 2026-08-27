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

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.sys.info import simd_width_of

# ===----------------------------------------------------------------------=== #
# Max
# ===----------------------------------------------------------------------=== #
from max.algorithm.backend.cpu import parallelize

# ===----------------------------------------------------------------------=== #
# mojoBLAS
# ===----------------------------------------------------------------------=== #
from mojoblas.level1._tuning import (
    AXPY_MIN_CHUNK_PER_THREAD,
    AXPY_N_ACC,
    AXPY_N_THREADS,
    AXPY_PAR_THRESHOLD,
)
from mojoblas.type_aliases import BLASPtr


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
    """Inner kernel: yc[i] = da * xc[i] + yc[i] with n_acc independent SIMD streams.
    """
    comptime stride: Int = simd_width * n_acc
    var i = 0
    while i + stride <= length:
        yc.unsafe_store[width=simd_width](
            i + 0 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 0 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 0 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 1 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 1 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 1 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 2 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 2 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 2 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 3 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 3 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 3 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 4 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 4 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 4 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 5 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 5 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 5 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 6 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 6 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 6 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 7 * simd_width,
            da * xc.unsafe_load[width=simd_width](i + 7 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 7 * simd_width),
        )
        i += stride
    while i + simd_width <= length:
        yc.unsafe_store[width=simd_width](
            i,
            da * xc.unsafe_load[width=simd_width](i)
            + yc.unsafe_load[width=simd_width](i),
        )
        i += simd_width
    while i < length:
        yc[unsafe_offset=i] = da * xc[unsafe_offset=i] + yc[unsafe_offset=i]
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
        yc.unsafe_store[width=simd_width](
            i + 0 * simd_width,
            xc.unsafe_load[width=simd_width](i + 0 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 0 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 1 * simd_width,
            xc.unsafe_load[width=simd_width](i + 1 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 1 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 2 * simd_width,
            xc.unsafe_load[width=simd_width](i + 2 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 2 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 3 * simd_width,
            xc.unsafe_load[width=simd_width](i + 3 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 3 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 4 * simd_width,
            xc.unsafe_load[width=simd_width](i + 4 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 4 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 5 * simd_width,
            xc.unsafe_load[width=simd_width](i + 5 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 5 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 6 * simd_width,
            xc.unsafe_load[width=simd_width](i + 6 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 6 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 7 * simd_width,
            xc.unsafe_load[width=simd_width](i + 7 * simd_width)
            + yc.unsafe_load[width=simd_width](i + 7 * simd_width),
        )
        i += stride
    while i + simd_width <= length:
        yc.unsafe_store[width=simd_width](
            i,
            xc.unsafe_load[width=simd_width](i)
            + yc.unsafe_load[width=simd_width](i),
        )
        i += simd_width
    while i < length:
        yc[unsafe_offset=i] += xc[unsafe_offset=i]
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
        yc.unsafe_store[width=simd_width](
            i + 0 * simd_width,
            yc.unsafe_load[width=simd_width](i + 0 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 0 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 1 * simd_width,
            yc.unsafe_load[width=simd_width](i + 1 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 1 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 2 * simd_width,
            yc.unsafe_load[width=simd_width](i + 2 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 2 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 3 * simd_width,
            yc.unsafe_load[width=simd_width](i + 3 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 3 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 4 * simd_width,
            yc.unsafe_load[width=simd_width](i + 4 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 4 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 5 * simd_width,
            yc.unsafe_load[width=simd_width](i + 5 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 5 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 6 * simd_width,
            yc.unsafe_load[width=simd_width](i + 6 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 6 * simd_width),
        )
        yc.unsafe_store[width=simd_width](
            i + 7 * simd_width,
            yc.unsafe_load[width=simd_width](i + 7 * simd_width)
            - xc.unsafe_load[width=simd_width](i + 7 * simd_width),
        )
        i += stride
    while i + simd_width <= length:
        yc.unsafe_store[width=simd_width](
            i,
            yc.unsafe_load[width=simd_width](i)
            - xc.unsafe_load[width=simd_width](i),
        )
        i += simd_width
    while i < length:
        yc[unsafe_offset=i] -= xc[unsafe_offset=i]
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

    Parameters:
        mut: Mutability of the pointer to X.
        origin_x: Memory origin of X.
        origin_y: Memory origin of Y.
        dtype: Element data type.
        n_threads: Max threads for parallel execution.
        par_threshold: Minimum n to consider parallelism.
        min_chunk: Minimum elements per thread.
        n_acc: Number of independent SIMD streams in the inner kernel.

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
                    dy[unsafe_offset=i] += dx[unsafe_offset=i]
            elif da == -1:
                for i in range(0, nsteps, incx):
                    dy[unsafe_offset=i] -= dx[unsafe_offset=i]
            else:
                for i in range(0, nsteps, incx):
                    dy[unsafe_offset=i] = (
                        da * dx[unsafe_offset=i] + dy[unsafe_offset=i]
                    )
            return

        var ix: Int = 0
        var iy: Int = 0
        if incx < 0:
            ix = (-n + 1) * incx
        if incy < 0:
            iy = (-n + 1) * incy
        if da == 1:
            for _ in range(n):
                dy[unsafe_offset=iy] += dx[unsafe_offset=ix]
                ix += incx
                iy += incy
        elif da == -1:
            for _ in range(n):
                dy[unsafe_offset=iy] -= dx[unsafe_offset=ix]
                ix += incx
                iy += incy
        else:
            for _ in range(n):
                dy[unsafe_offset=iy] = (
                    da * dx[unsafe_offset=ix] + dy[unsafe_offset=iy]
                )
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
                        dx.unsafe_offset(start),
                        dy.unsafe_offset(start),
                        end - start,
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
                        dx.unsafe_offset(start),
                        dy.unsafe_offset(start),
                        end - start,
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
                        dx.unsafe_offset(start),
                        dy.unsafe_offset(start),
                        da,
                        end - start,
                    )

                parallelize[worker](nt)
            return

    if da == 1:
        _axpy_add_serial[dtype, simd_width, n_acc](dx, dy, n)
    elif da == -1:
        _axpy_sub_serial[dtype, simd_width, n_acc](dx, dy, n)
    else:
        _axpy_serial[dtype, simd_width, n_acc](dx, dy, da, n)
