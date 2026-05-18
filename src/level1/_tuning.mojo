# Level 1 per-routine tuning constants (M2)

# AXPY
comptime AXPY_N_THREADS: Int = 4
comptime AXPY_PAR_THRESHOLD: Int = 4096
comptime AXPY_MIN_CHUNK_PER_THREAD: Int = 32768
comptime AXPY_N_ACC: Int = 4

# COPY
comptime COPY_N_THREADS: Int = 8
comptime COPY_PAR_THRESHOLD: Int = 8192
comptime COPY_UNROLL: Int = 16

# SCAL
comptime SCAL_N_THREADS: Int = 4
comptime SCAL_PAR_THRESHOLD: Int = 32768
comptime SCAL_UNROLL: Int = 4

# DOT
comptime DOT_N_THREADS: Int = 4
comptime DOT_PAR_THRESHOLD: Int = 16384
comptime DOT_N_ACC: Int = 4

# ASUM
comptime ASUM_N_THREADS: Int = 8
comptime ASUM_PAR_THRESHOLD: Int = 16384
comptime ASUM_N_ACC: Int = 4

# NRM2
comptime NRM2_N_THREADS: Int = 8
comptime NRM2_PAR_THRESHOLD: Int = 16384
comptime NRM2_N_ACC: Int = 4

# SWAP
comptime SWAP_N_THREADS: Int = 8
comptime SWAP_PAR_THRESHOLD: Int = 16384
comptime SWAP_UNROLL: Int = 16

# IAMAX
comptime IAMAX_N_THREADS: Int = 4
comptime IAMAX_PAR_THRESHOLD: Int = 8192

# ROT
comptime ROT_N_THREADS: Int = 8
comptime ROT_PAR_THRESHOLD: Int = 16384
comptime ROT_UNROLL: Int = 16

# ROTM
comptime ROTM_N_THREADS: Int = 4
comptime ROTM_PAR_THRESHOLD: Int = 16384
comptime ROTM_UNROLL: Int = 16
