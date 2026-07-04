CC := cc
CFLAGS := -O3 -std=c11 -Wall -Wextra -pedantic -DACCELERATE_NEW_LAPACK
# -framework Accelerate links Apple's AMX-accelerated BLAS directly (macOS)
LDFLAGS := -framework Accelerate

BIN := c_bench
SRC := bench.c

all: $(BIN) c_bench_single c_bench_multi c_bench_l3_f32

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

c_bench_single: bench_single_thread.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

c_bench_multi: bench_multi_thread.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

c_bench_l3_f32: bench_level3_f32.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BIN) c_bench_single c_bench_multi c_bench_l3_f32

.PHONY: all clean
