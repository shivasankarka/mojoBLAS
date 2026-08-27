# ===----------------------------------------------------------------------=== #
# mojoBLAS: Type Aliases
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #
"""
Type Aliases (mojoblas.type_aliases).
=====================================
Provides type aliases for BLAS operations with memory origin tracking.
"""

comptime BLASPtr[dtype: DType, origin: Origin] = Pointer[Scalar[dtype], origin]
"""An unsafe pointer type for BLAS operations."""
