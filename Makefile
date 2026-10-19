# OpenBench builds with:  make EXE=<name> [EVALFILE=<path to .pzn>]
# The first target is the default.

EXE      := peazero
EVALFILE ?= networks/default.pzn
export EVALFILE

# OpenBench passes the engine's compiler (cargo) as CC. Don't hand that to
# cargo: build scripts (tract-linalg's assembly kernels, via the cc crate)
# would try to run "cargo" as their C compiler. Without CC they find the
# system C compiler themselves.
unexport CC CXX

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
else
	NAME := $(EXE)
endif

.PHONY: openbench release test perft-full clean

openbench:
	cargo rustc --release -p peazero --bin peazero -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo build --release

test:
	cargo test --release

perft-full:
	cargo test --release --test perft standard_suite_full -- --ignored --nocapture

clean:
	cargo clean