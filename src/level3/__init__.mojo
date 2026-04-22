# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Level 3 BLAS Operations (`level3`)
============================================

This module provides Level 3 BLAS (Basic Linear Algebra Subprograms) operations
implemented in Mojo. Level 3 BLAS includes matrix-matrix operations
such as general matrix multiplication, symmetric multiplication, and triangular solves.

The implementations are optimized for performance while maintaining compatibility
with the standard BLAS interface.

Exports:
    gemm: General matrix-matrix multiplication
    symm: Symmetric matrix-matrix multiplication
    syrk: Symmetric rank-k update
    syr2k: Symmetric rank-2k update
    trmm: Triangular matrix-matrix multiplication
    trsm: Triangular matrix solve
"""

from .gemm import gemm
from .symm import symm
from .syrk import syrk
from .syr2k import syr2k
from .trmm import trmm
from .trsm import trsm
