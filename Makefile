# hetero-sim top-level entry points

PY := .venv/bin/python
OP ?= Tests/Kernels/FP32/GEMM/Regular
ROOT := $(CURDIR)
# MEM=ideal runs the zero-latency targets instead of the modelled memory system
MEM ?= real
# DEBUG=1 traces every command the pipeline runs and the files it generated
DBG := $(if $(DEBUG),--debug)
TARGETS := cva6 snitch spatz cva6_real snitch_real spatz_real hetero_soc ara_v2 ara_host hetero_ara

.PHONY: run gvsoc smoke ssr-test ara-test mesh-probe mesh-test hetero mnist kws clean

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
# (ssr_probe), that the GEMM/Conv kernels are right on the shapes the benchmark
# ops do not reach -- leftover columns, transposes, leftover filters
# (ssr_kernels) -- and that the KWS front-end's streamed filterbank and DCT are
# right on the ragged filter widths and cepstra counts the application never
# reaches (ssr_mfcc).
ssr-test:
	@mkdir -p work/ssr-test
	$(SNITCH_CC) $(SNITCH_CFLAGS) $(SNITCH_GLUE) runtime/tests/ssr_probe.c \
	  $(SNITCH_LIBS) -o work/ssr-test/ssr_probe.elf
	$(SNITCH_CC) $(SNITCH_CFLAGS) $(SNITCH_GLUE) runtime/tests/ssr_kernels.c \
	  runtime/snitch/kernels/gemm_fp32_ssr.c runtime/snitch/kernels/conv2d_fp32_ssr.c \
	  $(SNITCH_LIBS) -o work/ssr-test/ssr_kernels.elf
	$(SNITCH_CC) $(SNITCH_CFLAGS) $(SNITCH_GLUE) runtime/tests/ssr_mfcc.c \
	  runtime/snitch/kernels/mfcc_fp32_ssr.c $(SNITCH_LIBS) -o work/ssr-test/ssr_mfcc.elf
	@for t in ssr_probe ssr_kernels ssr_mfcc; do \
	  (cd work/ssr-test && PATH="$(ROOT)/.venv/bin:$$PATH" $(ROOT)/$(GVSOC) \
	    --target-dir=$(ROOT)/targets --target=snitch_real --binary=$$t.elf run \
	    2>/dev/null | grep -v '^WARNING'); \
	done

# CVA6 + Ara vector host checks on ara_host. ara_probe: vector loads and stores
# over the lengths, alignments and scalar interleavings compiled programs reach,
# the strided and indexed accesses, vid.v and vsetvli zero, zero that the model
# used to get wrong, and the instructions GCC's dense kernels are built from.
# ara_kernels: Deeploy's integer GEMM vectorized with the host's kernel flags
# against the same source built with no vector extension, element by element.
# Same toolchain, glue and libraries as ssr-test. Each run is time-limited so a
# hang fails instead of blocking; ARA_GVSOC_FLAGS passes extra options such as
# traces through.
ARA_ARCH := -march=rv64imafdcv_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -nostdlib -nostartfiles
ARA_CFLAGS := $(ARA_ARCH) -O2 -fno-tree-vectorize -fno-tree-loop-distribute-patterns \
  -Iruntime/common -Truntime/common/link.ld
ARA_GENERIC := -Ideps/deeploy/TargetLibraries/Generic/inc -DDEEPLOY_GENERIC_PLATFORM
ARA_GEMM_S8 := deps/deeploy/TargetLibraries/Generic/src/Gemm_s8.c
ARA_TIMEOUT ?= 300
ARA_GVSOC_FLAGS ?=
ara-test:
	@mkdir -p work/ara-test
	$(SNITCH_CC) $(ARA_CFLAGS) $(SNITCH_GLUE) runtime/tests/ara_probe.c \
	  $(SNITCH_LIBS) -o work/ara-test/ara_probe.elf
	$(SNITCH_CC) $(ARA_ARCH) $(ARA_GENERIC) -O3 -ffast-math \
	  -DGemm_s8_s8_s32_s32=Gemm_s8_vec -c $(ARA_GEMM_S8) -o work/ara-test/gemm_s8_vec.o
	$(SNITCH_CC) -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany -O2 \
	  $(ARA_GENERIC) -DGemm_s8_s8_s32_s32=Gemm_s8_ref -c $(ARA_GEMM_S8) -o work/ara-test/gemm_s8_ref.o
	$(SNITCH_CC) $(ARA_CFLAGS) $(SNITCH_GLUE) runtime/tests/ara_kernels.c \
	  work/ara-test/gemm_s8_vec.o work/ara-test/gemm_s8_ref.o $(SNITCH_LIBS) \
	  -o work/ara-test/ara_kernels.elf
	@for t in ara_probe ara_kernels; do \
	  (cd work/ara-test && PATH="$(ROOT)/.venv/bin:$$PATH" timeout $(ARA_TIMEOUT) \
	    $(ROOT)/$(GVSOC) --target-dir=$(ROOT)/targets --target=ara_host \
	    --binary=$$t.elf $(ARA_GVSOC_FLAGS) run 2>/dev/null | grep -v '^WARNING'); \
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

# hetero_soc dispatch check: the host hands a GEMM, a MatMul and a Conv2d to
# both clusters, in main memory and staged into TCDM, and compares every
# output element against the scalar kernel run on its own core.
mesh-test:
	$(PY) pipeline/gen_system_header.py --check
	$(PY) pipeline/build_mesh.py --test mesh_offload --cluster cluster_main.c \
	  --host-extra hes_host.c
	@cd work/mesh_offload && mkdir -p run && cd run && \
	  HES_ELF_SNITCH=$(ROOT)/work/mesh_offload/snitch/snitch.elf \
	  HES_ELF_SPATZ=$(ROOT)/work/mesh_offload/spatz/spatz.elf \
	  PATH="$(ROOT)/.venv/bin:$$PATH" $(ROOT)/$(GVSOC) \
	    --target-dir=$(ROOT)/targets --target=hetero_soc \
	    --binary=$(ROOT)/work/mesh_offload/host/host.elf run 2>/dev/null | grep -v '^WARNING'

# Run one op on the whole SoC, with Deeploy mapping each node to an engine.
#   make hetero OP=Tests/Kernels/FP32/GEMM/Regular
#   make hetero OP=... PIN=snitch      force one engine, for the comparison
#   make hetero OP=... HOST=ara        the SoC whose CVA6 carries an Ara vector unit
HOST ?= cva6
hetero:
	$(PY) pipeline/run_hetero.py $(OP) --host $(HOST) $(if $(PIN),--pin $(PIN)) $(DBG)

# Train the MNIST CNN, export it, and classify the embedded test images on the
# whole SoC.  make mnist IMAGES=16  runs fewer of them.
# REUSE=1 keeps the committed network.onnx and only rebuilds the evaluation
# set, which is what you want when changing IMAGES.
IMAGES ?= 64
mnist:
	$(PY) pipeline/mnist.py --images $(IMAGES) $(if $(REUSE),--reuse)
	$(PY) pipeline/run_hetero.py ops/mnist --host $(HOST) $(if $(PIN),--pin $(PIN)) $(DBG)

# Keyword spotting: the application that gives both clusters work at once.
# The MFCC front-end runs on one cluster while the classifier runs on the
# other, so unlike mnist this one is not a sum of engines but a max.
#   make kws                       front-end on snitch, pipelined
#   make kws SERIAL=1              the same work with the clusters taking turns
#   make kws FE=spatz              the other placement
#   make kws PIN=snitch            pin the classifier's nodes, as with mnist
# REUSE=1 keeps the committed network.onnx and only rebuilds the clips.
CLIPS ?= 16
FE ?= snitch
kws:
	$(PY) pipeline/kws.py --clips $(CLIPS) $(if $(REUSE),--reuse)
	$(PY) pipeline/run_hetero.py ops/kws --frontend $(FE) --host $(HOST) \
	  $(if $(SERIAL),--serial) $(if $(PIN),--pin $(PIN)) $(DBG)

clean:
	rm -rf work/*
