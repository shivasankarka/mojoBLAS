# Level 3 GEMM tuning constants.
# Target: Apple M2

comptime GEMM_DISPATCH_THRESHOLD: Int = 512

# gemm_v6
comptime GEMM_V6_NR: Int = 16
comptime GEMM_V6_TK: Int = 128
comptime GEMM_V6_MC: Int = 256
comptime GEMM_V6_PAR_THRESHOLD: Int = 4

# gemm_v7
comptime GEMM_V7_NR: Int = 16
comptime GEMM_V7_TK: Int = 128
comptime GEMM_V7_MC: Int = 256
comptime GEMM_V7_PAR_THRESHOLD: Int = 12

# gemm_v8 (AMX f32)
# TILE is fixed at 16 (AMX f32 row width), not a tuning constant.
# MC*TK*4 bytes = a_pack size; target M2 L2 (4 MB per core).
comptime GEMM_V8_MC: Int = 256
comptime GEMM_V8_TK: Int = 256
comptime GEMM_V8_PAR_THRESHOLD: Int = 8

# gemm_v9 (AMX f32, 4-tile Z, direct C writeback)
# NZ=4: use all 4 AMX Z tile slots simultaneously (Z[0..3] → 4×16 rows of C).
# MC must be a multiple of NZ*TILE = 64.
# TK: k-panel depth; MC*TK floats in L2 per core (4 MB on M2).
comptime GEMM_V9_TILE: Int = 16  # AMX f32 row width (fixed by hardware)
comptime GEMM_V9_NZ: Int = 4  # Z tiles in flight per column group
comptime GEMM_V9_MC: Int = 256  # MC rows of A packed at once (must be % (NZ*TILE)==0)
comptime GEMM_V9_TK: Int = 256  # k-panel depth
comptime GEMM_V9_PAR_THRESHOLD: Int = 4  # min column groups to parallelize

# gemm_v10 (AMX f32, 4-tile Z, transposed Z writeback)
# NZ=4: tile-mode f32 exposes 4 independent Z tile offsets.
# Writeback: store Z rows into row-major buf, then 4×4 SIMD-transpose 16×16
# sub-tiles to produce column-major output, store contiguous SIMD columns to C.
comptime GEMM_V10_TILE: Int = 16  # AMX f32 row width (fixed)
comptime GEMM_V10_NZ: Int = 4  # Z tile offsets in flight
comptime GEMM_V10_MC: Int = 256  # must be % (NZ*TILE==64) == 0
comptime GEMM_V10_TK: Int = 256  # k-panel depth
comptime GEMM_V10_PAR_THRESHOLD: Int = 4

# gemm_v11 (AMX f32, 4-tile Z, 4x k-unroll using X[0..3] × Y[0..3])
# Key change vs v9: unroll k-loop by 4, loading X[0..3] = 4 B rows, then for each
# Z tile z=0..3 load Y[0..3] = A-tile-z[l..l+3] and issue 4 fma32_tile(z, u, u).
# This gives the out-of-order engine 4 independent X loads to overlap with FMAs.
# Writeback follows v10 (row-major Z buffer + 4x4 SIMD transpose stores).
# MC*TK must fit in L2 (256*256*4 = 256 KB per core on M2).
comptime GEMM_V11_NZ: Int = 4  # Z tile offsets per block
comptime GEMM_V11_UK: Int = 4  # k-unroll factor (must divide TK evenly)
comptime GEMM_V11_MC: Int = 256
comptime GEMM_V11_TK: Int = 1024
comptime GEMM_V11_ROW_PAR_THRESHOLD: Int = 1024
comptime GEMM_V11_SMALL_MC: Int = 64
comptime GEMM_V11_PAR_THRESHOLD: Int = 4

# gemm_v12 (AMX f32, v11 large-kernel with NC-blocked B packing)
comptime GEMM_V12_MC: Int = 256
comptime GEMM_V12_NC: Int = 2048
comptime GEMM_V12_TK: Int = 1024
