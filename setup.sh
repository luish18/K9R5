#!/usr/bin/env bash
# Bootstrap hetero-sim: fetch dependencies at pinned commits, patch the GVSoC
# vector model, build the simulator, and install the Python environment.
#
# Safe to re-run: each step is skipped if it is already done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Pinned dependency versions. GVSoC submodules follow from its superproject
# commit, so only the top-level SHAs are pinned here.
GVSOC_URL=https://github.com/gvsoc/gvsoc.git
GVSOC_COMMIT=d66f7587c4b7709ffcc805c960acc18f994614be
DEEPLOY_URL=https://github.com/pulp-platform/Deeploy.git
DEEPLOY_COMMIT=aa268252165d96788fbe01f518f90f0d89f63c13
TOOLCHAIN_VERSION=15.2.0-1
TOOLCHAIN_DIR="toolchains/xpack-riscv-none-elf-gcc-${TOOLCHAIN_VERSION}"
TOOLCHAIN_URL="https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v${TOOLCHAIN_VERSION}/xpack-riscv-none-elf-gcc-${TOOLCHAIN_VERSION}-linux-x64.tar.gz"

# GVSoC targets to build: the three stock ones (--memory ideal) and the three
# in targets/ that model the memory system (--memory real, the default).
TARGETS="cva6 snitch spatz cva6_real snitch_real spatz_real hetero_soc"

log() { printf '\n=== %s\n' "$1"; }

command -v uv >/dev/null || {
  echo "error: 'uv' is required (https://astral.sh/uv). Install it and re-run." >&2
  exit 1
}

log "RISC-V toolchain"
if [ -d "$TOOLCHAIN_DIR" ]; then
  echo "already present: $TOOLCHAIN_DIR"
else
  mkdir -p toolchains
  curl -fsSL "$TOOLCHAIN_URL" | tar xz -C toolchains
fi

log "Deeploy @ ${DEEPLOY_COMMIT:0:8}"
if [ -d deps/deeploy ]; then
  echo "already cloned: deps/deeploy"
else
  mkdir -p deps
  git clone "$DEEPLOY_URL" deps/deeploy
  git -C deps/deeploy checkout --detach "$DEEPLOY_COMMIT"
fi

log "GVSoC @ ${GVSOC_COMMIT:0:8}"
if [ -d deps/gvsoc ]; then
  echo "already cloned: deps/gvsoc"
else
  git clone "$GVSOC_URL" deps/gvsoc
  git -C deps/gvsoc checkout --detach "$GVSOC_COMMIT"
  git -C deps/gvsoc submodule update --init --recursive --jobs=8
fi

log "GVSoC model patches"
for patch in "$ROOT"/deps/patches/gvsoc-core-*.patch; do
  name="$(basename "$patch")"
  if git -C deps/gvsoc/core apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: $name"
  else
    git -C deps/gvsoc/core apply "$patch"
    echo "applied: $name"
  fi
done

log "Python environment"
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install -p .venv -e deps/deeploy
uv pip install -p .venv \
  -r deps/gvsoc/requirements.txt \
  -r deps/gvsoc/core/requirements.txt \
  -r deps/gvsoc/gapy/requirements.txt \
  ninja

log "Building GVSoC (6 targets) — this takes a few minutes"
# The build drives gapy from the venv, so run it with the venv activated.
# MODULES adds targets/ to the module roots, which is what lets the build see
# the *_real targets and compile the timing-cache model that lives with them.
( cd deps/gvsoc && . "$ROOT/.venv/bin/activate" \
  && make all TARGETS="$TARGETS" MODULES="$ROOT/targets" CMAKE_FLAGS="-j $(nproc)" )

log "Done. Try:"
echo "  .venv/bin/python pipeline/run.py Tests/Kernels/FP32/GEMM/Regular"
