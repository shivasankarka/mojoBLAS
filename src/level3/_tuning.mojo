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
