# hetero-sim: GVSoC simulation of CVA6/Snitch/Spatz, driven from ONNX via Deeploy.
#
# setup.sh fetches GVSoC/Deeploy at pinned commits and the RISC-V toolchain,
# then builds all six GVSoC targets — that build is what this image bakes in,
# so the container starts ready to run the pipeline on any host with Docker.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# Packages GVSoC's own build instructions call for (Ubuntu 22.04 list), plus
# git/curl/cmake/a C++ toolchain for the rest of setup.sh.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      doxygen \
      python3-pip \
      libsdl2-dev \
      libsdl2-ttf-dev \
      curl \
      ca-certificates \
      cmake \
      ninja-build \
      gtkwave \
      libsndfile1-dev \
      rsync \
      autoconf \
      automake \
      texinfo \
      libtool \
      pkg-config \
      wget \
      python3-sphinx \
    && rm -rf /var/lib/apt/lists/*

# uv drives the Python env (and fetches its own Python 3.11 if the OS doesn't have one).
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
COPY . .

# Fetches deps/gvsoc, deps/deeploy and the toolchain at the pinned commits in
# setup.sh, applies the local GVSoC patches, and builds all six targets.
RUN ./setup.sh

ENV PATH="/workspace/.venv/bin:/workspace/toolchains/xpack-riscv-none-elf-gcc-15.2.0-1/bin:/workspace/deps/gvsoc/install/bin:${PATH}"

# Default: run the bundled GEMM benchmark op; override with e.g.
#   docker run --rm hetero-sim python pipeline/run.py Tests/Kernels/FP32/MatMul
#   docker run --rm -it hetero-sim bash
CMD ["python", "pipeline/run.py", "Tests/Kernels/FP32/GEMM/Regular"]
