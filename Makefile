# hetero-sim top-level entry points

PY := .venv/bin/python
OP ?= Tests/Kernels/FP32/GEMM/Regular
ROOT := $(CURDIR)
# MEM=ideal runs the zero-latency targets instead of the modelled memory system
MEM ?= real
# DEBUG=1 traces every command the pipeline runs and the files it generated
DBG := $(if $(DEBUG),--debug)
TARGETS := cva6 snitch spatz cva6_real snitch_real spatz_real

.PHONY: run gvsoc smoke clean

# Run the full pipeline on one op:  make run OP=Tests/Kernels/FP32/GEMM/Regular
run:
	$(PY) pipeline/run.py $(OP) --memory $(MEM) $(DBG)

# (Re)build the GVSoC targets. MODULES puts targets/ on the module path so the
# build picks up the *_real targets and their timing-cache model.
gvsoc:
	cd deps/gvsoc && . ../../.venv/bin/activate && \
	  make all TARGETS="$(TARGETS)" MODULES="$(ROOT)/targets" CMAKE_FLAGS="-j $$(nproc)"

# Boot-level sanity checks for the three cores
smoke:
	$(PY) pipeline/run.py Tests/Kernels/FP32/GEMM/Regular --memory $(MEM) --timeout 300 $(DBG)

clean:
	rm -rf work/*
