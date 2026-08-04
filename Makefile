# hetero-sim top-level entry points

PY := .venv/bin/python
OP ?= Tests/Kernels/FP32/GEMM/Regular
# DEBUG=1 traces every command the pipeline runs and the files it generated
DBG := $(if $(DEBUG),--debug)

.PHONY: run gvsoc smoke clean

# Run the full pipeline on one op:  make run OP=Tests/Kernels/FP32/GEMM/Regular
run:
	$(PY) pipeline/run.py $(OP) $(DBG)

# (Re)build the GVSoC targets
gvsoc:
	cd deps/gvsoc && . ../../.venv/bin/activate && \
	  make all TARGETS="cva6 snitch spatz" CMAKE_FLAGS="-j $$(nproc)"

# Boot-level sanity checks for the three cores
smoke:
	$(PY) pipeline/run.py Tests/Kernels/FP32/GEMM/Regular --timeout 300 $(DBG)

clean:
	rm -rf work/*
