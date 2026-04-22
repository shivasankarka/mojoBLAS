# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Level 1 BLAS Operations (`level1`)
============================================

This module provides Level 1 BLAS (Basic Linear Algebra Subprograms) operations
implemented in Mojo. Level 1 BLAS includes vector-vector operations
such as dot products, norms, and rotations.
"""
# TODO: Add vectorized/parallelized operations.
# Add benchmark against BLAS operations.
# Add support for Complex data types.

from .copy import copy
from .scal import scal
from .axpy import axpy
from .asum import asum
from .dot import dot
from .nrm2 import nrm2
from .swap import vswap
from .iamax import iamax
from .rotg import rotg
from .rot import rot
