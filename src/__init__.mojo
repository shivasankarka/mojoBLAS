# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
mojoBLAS: Mojo implementation of BLAS
=====================================

A high-performance Mojo implementation of BLAS (Basic Linear Algebra Subprograms)
library operations. This library provides Level 1, Level 2, and Level 3 BLAS
routines implemented in the Mojo programming language.

Level 1: Vector-vector operations
Level 2: Matrix-vector operations
Level 3: Matrix-matrix operations

For more information about BLAS, see: http://www.netlib.org/blas/
"""

from .type_aliases import BLASPtr
