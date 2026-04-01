from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of
from std.memory import memcpy


def axpy[
    dtype: DType
](
    n: Int,
    da: Scalar[dtype],
    dx: BLASPtr[Scalar[dtype]],
    incx: Int,
    dy: BLASPtr[Scalar[dtype]],
    incy: Int,
):
    """
    Perform the AXPY operation: Y := alpha * X + Y.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.

    Args:
        n: Number of elements in vectors X and Y.
        da: Scalar multiplier for vector X.
        dx: Pointer to the first element of vector X. dimension should be at least (1 + (n - 1) * abs(incx)).
        incx: Increment for the elements of X.
        dy: Pointer to the first element of vector Y. dimension should be at least (1 + (n - 1) * abs(incy)).
        incy: Increment for the elements of Y.
    """
    if n <= 0 or da == 0:
        return

    comptime simd_width: Int = simd_width_of[dtype]()
    if incx == 1 and incy == 1:

        @parameter
        def closure[width: Int](i: Int) unified {mut dy, read dx, read da}:
            dy.store[width=width](
                i, da * dx.load[width=width](i) + dy.load[width=width](i)
            )

        vectorize[simd_width](n, closure)
        return

    # TODO: I think we could speed this up with strided_load and strided_store.
    var ix: Int = 0
    var iy: Int = 0
    if incx < 0:
        ix = (-n + 1) * incx
    if incy < 0:
        iy = (-n + 1) * incy

    for i in range(n):
        dy[iy] = da * dx[ix] + dy[iy]
        ix += incx
        iy += incy


def axpy[
    dtype: DType, n: Int, da: Scalar[dtype], incx: Int, incy: Int
](
    dx: BLASPtr[Scalar[dtype]],
    dy: BLASPtr[Scalar[dtype]],
) where dtype.is_floating_point():
    """
    AXPY operation for compile-time known parameters.

    Parameters:
        dtype: Data type of the elements in vectors X and Y.
        n: Number of elements in vectors X and Y.
        da: Scalar multiplier for vector X.
        incx: Increment for the elements of X.
        incy: Increment for the elements of Y.

    Args:
        dx: Pointer to the first element of vector X.
        dy: Pointer to the first element of vector Y.
    """
    axpy[dtype](n, da, dx, incx, dy, incy)


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


# def gdaxpy[dtype: DType](
#     n: Int,
#     da: Scalar[dtype],
#     dx: BLASPtr[Scalar[dtype]],
#     incx: Int,
#     dy: BLASPtr[Scalar[dtype]],
#     incy: Int,
# ) raises -> None where dtype.is_floating_point():

#     comptime THREADS_PER_BLOCK = (256, 1)
#     var BLOCKS_PER_GRID = ((n + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0], 1)

#     with DeviceContext() as ctx:
#         res = ctx.enqueue_create_buffer[dtype](n)
#         res.enqueue_fill(0)
#         x = ctx.enqueue_create_buffer[dtype](n)
#         x.enqueue_fill(0)
#         y = ctx.enqueue_create_buffer[dtype](n)
#         y.enqueue_fill(0)

#         with x.map_to_host() as a_host:
#             memcpy(dest=a_host.unsafe_ptr(), src=dx, count=n)
#         with y.map_to_host() as b_host:
#             memcpy(dest=b_host.unsafe_ptr(), src=dy, count=n)

#         ctx.enqueue_function[gdaxpy_backend[dtype], gdaxpy_backend[dtype]](
#             n, da, x, incx, y, incy,
#             grid_dim=BLOCKS_PER_GRID,
#             block_dim=THREADS_PER_BLOCK,
#         )

#         ctx.synchronize()

#         with y.map_to_host() as b_host:
#             memcpy(dest=dy, src=b_host.unsafe_ptr(), count=n)

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
