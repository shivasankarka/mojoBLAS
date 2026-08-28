CC := cc
CFLAGS := -O3 -std=c11 -Wall -Wextra -pedantic -DACCELERATE_NEW_LAPACK
# -framework Accelerate links Apple's AMX-accelerated BLAS directly (macOS)
LDFLAGS := -framework Accelerate

BIN := c_bench
SRC := bench.c

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BIN)

.PHONY: all clean
