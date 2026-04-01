"""
Provides Level 1 BLAS (Basic Linear Algebra Subprograms) operations.
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
