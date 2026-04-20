CC := cc
CFLAGS := -O3 -std=c11 -Wall -Wextra -pedantic
LDFLAGS :=

BIN := c_bench
SRC := bench.c

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BIN)

.PHONY: all clean
