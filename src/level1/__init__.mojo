"""
Provides Level 1 BLAS (Basic Linear Algebra Subprograms) operations.
"""
# TODO: Add vectorized/parallelized operations.
# Add benchmark against BLAS operations.

from .copy import dcopy
from .scal import dscal
from .axpy import daxpy
from .asum import dasum
from .dot import ddot
from .nrm2 import dnrm2
from .swap import dswap
from .iamax import di_amax
from .rotg import drotg
from .rot import drot
