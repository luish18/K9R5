# hetero-sim top-level entry points

PY := .venv/bin/python
OP ?= Tests/Kernels/FP32/GEMM/Regular
ROOT := $(CURDIR)
# MEM=ideal runs the zero-latency targets instead of the modelled memory system
MEM ?= real
# DEBUG=1 traces every command the pipeline runs and the files it generated
DBG := $(if $(DEBUG),--debug)
TARGETS := cva6 snitch spatz cva6_real snitch_real spatz_real hetero_soc

.PHONY: run gvsoc smoke ssr-test mesh-probe clean

# Snitch bare-metal test build (the pipeline's snitch flags, minus the
# generated network) used by the ssr-test target below.
SNITCH_CC := toolchains/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-gcc
SNITCH_CFLAGS := -march=rv32imafd_zicsr_zifencei -mabi=ilp32d -mcmodel=medany \
  -nostdlib -nostartfiles -O3 -DDEEPLOY_GENERIC_PLATFORM \
  -Ideps/deeploy/TargetLibraries/Generic/inc -Iruntime/common -Iruntime/snitch \
  -Truntime/snitch/link.ld
SNITCH_GLUE := runtime/common/crt0.S runtime/common/syscalls.c
SNITCH_LIBS := -Wl,--gc-sections -lc -lm -lgcc
GVSOC := deps/gvsoc/install/bin/gvsoc

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

# Snitch Xssr/Xfrep checks: that the model streams and repeats at all
# (ssr_probe), and that the kernels are right on the shapes the benchmark ops
# do not reach — leftover columns, transposes, leftover filters (ssr_kernels).
ssr-test:
	@mkdir -p work/ssr-test
	$(SNITCH_CC) $(SNITCH_CFLAGS) $(SNITCH_GLUE) runtime/tests/ssr_probe.c \
	  $(SNITCH_LIBS) -o work/ssr-test/ssr_probe.elf
	$(SNITCH_CC) $(SNITCH_CFLAGS) $(SNITCH_GLUE) runtime/tests/ssr_kernels.c \
	  runtime/snitch/kernels/*.c $(SNITCH_LIBS) -o work/ssr-test/ssr_kernels.elf
	@for t in ssr_probe ssr_kernels; do \
	  (cd work/ssr-test && PATH="$(ROOT)/.venv/bin:$$PATH" $(ROOT)/$(GVSOC) \
	    --target-dir=$(ROOT)/targets --target=snitch_real --binary=$$t.elf run \
	    2>/dev/null | grep -v '^WARNING'); \
	done

# hetero_soc board check: do the three cores boot in one simulation, and does
# each level of the memory system answer at the cost hetero/system.py says?
# Every cycle count the SoC produces later rests on this.
mesh-probe:
	$(PY) pipeline/gen_system_header.py --check
	$(PY) pipeline/build_mesh.py --test mesh_probe
	@cd work/mesh_probe && mkdir -p run && cd run && \
	  HES_ELF_SNITCH=$(ROOT)/work/mesh_probe/snitch/snitch.elf \
	  HES_ELF_SPATZ=$(ROOT)/work/mesh_probe/spatz/spatz.elf \
	  PATH="$(ROOT)/.venv/bin:$$PATH" $(ROOT)/$(GVSOC) \
	    --target-dir=$(ROOT)/targets --target=hetero_soc \
	    --binary=$(ROOT)/work/mesh_probe/host/host.elf run 2>/dev/null | grep -v '^WARNING'

clean:
	rm -rf work/*
