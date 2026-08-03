# hetero-sim top-level entry points

PY := .venv/bin/python
OP ?= Tests/Kernels/FP32/GEMM/Regular

.PHONY: run gvsoc smoke clean

# Run the full pipeline on one op:  make run OP=Tests/Kernels/FP32/GEMM/Regular
run:
	$(PY) pipeline/run.py $(OP)

# (Re)build the GVSoC targets
gvsoc:
	cd deps/gvsoc && . ../../.venv/bin/activate && \
	  make all TARGETS="cva6 snitch spatz" CMAKE_FLAGS="-j $$(nproc)"

# Boot-level sanity checks for the three cores
smoke:
	$(PY) pipeline/run.py Tests/Kernels/FP32/GEMM/Regular --timeout 300

clean:
	rm -rf work/*
