# OpenBench builds with:  make EXE=<name>
# The first target is the default.

EXE := peazero

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