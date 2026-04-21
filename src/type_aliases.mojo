comptime BLASPtr[dtype: DType, origin: Origin] = UnsafePointer[
    Scalar[dtype], origin
]
"""An unsafe pointer type for BLAS operations."""
