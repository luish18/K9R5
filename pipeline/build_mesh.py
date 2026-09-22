#!/usr/bin/env python3
"""Build the three ELFs of a hetero_soc run: host, snitch cluster, spatz cluster.

The clusters and the host are different ISAs and different linker scripts, so
a run is three compiles rather than one. This module holds that knowledge so
both the tests and the pipeline driver use the same flags.

  python pipeline/build_mesh.py --test mesh_probe
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import pathlib  # noqa: E402

from common import TC, sh, set_debug  # noqa: E402

RUNTIME = ROOT / "runtime"
MESH = RUNTIME / "mesh"
DEEPLOY = ROOT / "deps" / "deeploy"
GENERIC_LIB = DEEPLOY / "TargetLibraries" / "Generic"

COMMON_FLAGS = ["-mcmodel=medany", "-nostdlib", "-nostartfiles", "-ffunction-sections",
                "-DDEEPLOY_GENERIC_PLATFORM"]
# Glue is control code -- the job loop, the runtime, the test harness -- and has
# nothing to gain from vectorization. It must not be vectorized on the Spatz
# cluster: its -march carries `v`, so GCC will happily emit RVV for ordinary
# scalar loops, and the GVSoC Spatz model has gaps there (see the four fixes in
# deps/patches). pipeline/run.py draws the same line for the same reason.
GLUE_FLAGS = ["-fno-tree-vectorize"]
LINK_LIBS = ["-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lc", "-lm", "-lgcc"]

# Shared include path for every image in a run.
INCS = [RUNTIME / "common", MESH, GENERIC_LIB / "inc"]


def set_mesh_dir(path):
    """Build against a different copy of runtime/mesh.

    A design-space sweep gives every design point its own copy, so that
    concurrent cells do not regenerate each other's hes_system.h while a
    compile is reading it.

    It has to be a *copy of the whole directory*, not just the generated files
    with an -I in front of it. runtime/mesh/hes_host.h, hes_cluster.h and
    hes_mailbox.h all `#include "hes_system.h"` with quotes, and GCC resolves a
    quoted include relative to the including file's own directory before it
    looks at any -I. So a cell that only overrode the include path would link
    the cell's linker script against the *shared* header -- no error, just
    wrong addresses and plausible wrong cycle counts. sweep/cell.py copies the
    directory; this function points the builder at the copy.

    The Image dataclasses capture their linker and crt0 paths at
    class-definition time, so those have to be rebound too, not just MESH.
    """
    global MESH, INCS
    MESH = pathlib.Path(path)
    INCS = [RUNTIME / "common", MESH, GENERIC_LIB / "inc"]
    for image, linker, crt0 in ((HOST, "host.ld", "crt0_host.S"),
                                (HOST_ARA, "host.ld", "crt0_host.S"),
                                (SNITCH, "snitch.ld", "crt0_cluster.S"),
                                (SPATZ, "spatz.ld", "crt0_cluster.S")):
        image.linker = MESH / linker
        image.crt0 = MESH / crt0

# First-party kernels every image gets, on the same terms as the Deeploy
# Generic library: compiled with the image's kernel flags (so a vector core
# autovectorizes them) and subject to the same link-time rename, so a core with
# a hand-written version takes the name and this one stays reachable as
# <name>_generic. The MFCC front-end lives here because all three engines have
# to be able to run it -- that is the whole point of measuring where it belongs.
COMMON_KERNELS = sorted((RUNTIME / "common" / "kernels").glob("*.c"))


class Image:
    """One of the three binaries a hetero_soc run needs.

    `kernel_srcs` / `kernel_overrides` carry the per-core kernel story the
    per-core pipeline already has: the Snitch cluster replaces three Deeploy
    Generic kernels with its Xssr/Xfrep versions, and the originals stay
    linked in as <name>_generic for the shapes the rewrite does not cover.
    """

    def __init__(self, name, march, mabi, linker, crt0, defines=(), extra_incs=(),
                 kernel_flags=("-O3",), kernel_srcs=(), kernel_overrides=()):
        self.name = name
        self.march = march
        self.mabi = mabi
        self.linker = linker
        self.crt0 = crt0
        self.defines = list(defines)
        self.extra_incs = list(extra_incs)
        self.kernel_flags = list(kernel_flags)
        self.kernel_srcs = list(kernel_srcs)
        self.kernel_overrides = list(kernel_overrides)

    def rename_flags(self):
        return [f"-D{sym}={sym}_generic" for sym in self.kernel_overrides]

    def arch_flags(self):
        return [f"-march={self.march}", f"-mabi={self.mabi}"]

    def include_flags(self):
        return [f"-I{p}" for p in INCS + self.extra_incs]


# The scalar orchestrator, and the same orchestrator with an Ara vector unit.
# The vector one adds `v` to the march and vectorizes its kernels; glue stays
# scalar through GLUE_FLAGS either way.
HOST = Image(
    name="host",
    # The host keeps the C extension: unlike the clusters it has no decoupled
    # FP subsystem for a compressed FP load to diverge from.
    march="rv64imafdc_zicsr_zifencei",
    mabi="lp64d",
    linker=MESH / "host.ld",
    crt0=MESH / "crt0_host.S",
    defines=["-DHES_HOST"],
)

SNITCH = Image(
    name="snitch",
    # No C extension: the GVSoC Snitch model executes compressed FP loads on
    # the integer core, diverging from the decoupled FP subsystem.
    march="rv32imafd_zicsr_zifencei",
    mabi="ilp32d",
    linker=MESH / "snitch.ld",
    crt0=MESH / "crt0_cluster.S",
    defines=["-DHES_CLUSTER_SNITCH"],
    extra_incs=[RUNTIME / "snitch"],
    kernel_srcs=sorted((RUNTIME / "snitch" / "kernels").glob("*.c")),
    kernel_overrides=["MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32",
                      "Conv2d_fp32_fp32_fp32_NCHW", "Mfcc_fp32_fp32"],
)

SPATZ = Image(
    name="spatz",
    march="rv32imafd_zicsr_zifencei_v",
    mabi="ilp32d",
    linker=MESH / "spatz.ld",
    crt0=MESH / "crt0_cluster.S",
    defines=["-DHES_CLUSTER_SPATZ"],
    # -ffast-math is what lets GCC vectorize the FP reductions in the Generic
    # kernels it still uses; the hand-written ones below do not need it.
    kernel_flags=["-O3", "-ffast-math"],
    # GEMM and MatMul are written against RVV directly: left to the
    # autovectorizer GCC picks the reduction axis, which costs a strided load
    # and a horizontal reduction per output element.
    kernel_srcs=sorted((RUNTIME / "spatz" / "kernels").glob("*.c")),
    kernel_overrides=["MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32"],
)

HOST_ARA = Image(
    name="host",
    march="rv64imafdc_zicsr_zifencei_v",
    mabi="lp64d",
    linker=MESH / "host.ld",
    crt0=MESH / "crt0_host.S",
    defines=["-DHES_HOST"],
    kernel_flags=["-O3", "-ffast-math"],
)

IMAGES = {img.name: img for img in (HOST, SNITCH, SPATZ)}

# Which orchestrator a run uses, and the board that matches it.
HOSTS = {"cva6": (HOST, "hetero_soc"), "ara": (HOST_ARA, "hetero_ara")}


def build(image: Image, sources, out_dir: Path, opt="-O2", extra_flags=(),
          with_kernels=False) -> Path:
    """Compile and link one image. Returns the ELF path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / f"{image.name}.elf"

    objs = []

    def compile_(srcs, flags, tag):
        for src in srcs:
            obj = out_dir / f"{tag}_{Path(src).stem}.o"
            r = sh([str(TC), *image.arch_flags(), *COMMON_FLAGS, *flags,
                    *image.defines, *image.include_flags(), "-c", str(src),
                    "-o", str(obj)], produces=[obj])
            if r.returncode != 0:
                sys.exit(f"[{image.name}] compile failed for {src}:\n{r.stderr}")
            objs.append(obj)

    compile_([image.crt0] + list(sources), [opt, *GLUE_FLAGS, *extra_flags], "glue")

    if with_kernels:
        extra_flags = list(extra_flags)
        # The rename applies to the Generic library alone, so the cluster's own
        # kernels and the code calling them keep the original names.
        compile_(sorted((GENERIC_LIB / "src").glob("*.c")),
                 image.kernel_flags + image.rename_flags(), "k")
        compile_(COMMON_KERNELS,
                 image.kernel_flags + image.rename_flags(), "common")
        compile_(image.kernel_srcs, image.kernel_flags, "core")

    r = sh([str(TC), *image.arch_flags(), *COMMON_FLAGS, f"-T{image.linker}",
            *[str(o) for o in objs], *LINK_LIBS, "-o", str(elf)], produces=[elf])
    if r.returncode != 0:
        sys.exit(f"[{image.name}] link failed:\n{r.stderr}")
    return elf


def build_test(test: str, work: Path, cluster_src=None, host_extra=()) -> dict:
    """Build a standalone bare-metal check: one host program, plus the cluster
    program both clusters run."""
    host_src = RUNTIME / "tests" / f"{test}.c"
    syscalls = RUNTIME / "common" / "syscalls.c"
    if cluster_src is None:
        cluster_src = MESH / "cluster_probe.c"
    with_kernels = cluster_src.name == "cluster_main.c"

    return {
        "host": build(HOST, [syscalls, host_src, *host_extra], work / "host",
                      with_kernels=with_kernels),
        "snitch": build(SNITCH, [cluster_src], work / "snitch",
                        with_kernels=with_kernels),
        "spatz": build(SPATZ, [cluster_src], work / "spatz",
                       with_kernels=with_kernels),
    }


def build_network(gen_dir: Path, work: Path, samples: int = 1,
                  host_main: Path = None, extra_incs = (), host: str = "cva6",
                  extra_defines = ()) -> dict:
    """Build the three ELFs for a Deeploy-generated network.

    The host links the generated Network.c, the host runtime and the Generic
    kernel library -- it runs the nodes the mapper left on it. Each cluster
    links the job loop and its own kernels.

    `host_main` selects the host program: the default runs the graph once and
    diffs it against the ONNX reference, while an op that ships its own
    evaluation set (MNIST, KWS) supplies one that loops over it and scores.

    `extra_defines` reaches only the host, which is where an application's
    build-time choices live -- KWS uses it for the front-end's cluster and for
    the serial baseline.
    """
    if host_main is None:
        host_main = MESH / "host_main.c"
    host_sources = [
        RUNTIME / "common" / "syscalls.c",
        MESH / "hes_host.c",
        host_main,
        gen_dir / "Network.c",
    ]
    cluster_src = MESH / "cluster_main.c"
    incs = [f"-I{gen_dir}", *[f"-I{p}" for p in extra_incs]]

    host_image = HOSTS[host][0]
    return {
        "host": build(host_image, host_sources, work / "host", with_kernels = True,
                      extra_flags = [*incs, f"-DHES_SAMPLES={samples}",
                                     *extra_defines]),
        "snitch": build(SNITCH, [cluster_src], work / "snitch", with_kernels = True),
        "spatz": build(SPATZ, [cluster_src], work / "spatz", with_kernels = True),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="name under runtime/tests")
    ap.add_argument("--cluster", default="cluster_probe.c",
                    help="cluster program under runtime/mesh (default: %(default)s)")
    ap.add_argument("--host-extra", default=[], action="append",
                    help="extra host source, relative to runtime/mesh")
    ap.add_argument("--mesh-dir", default=None,
                    help="build against this copy of runtime/mesh (a sweep gives "
                         "each design point its own)")
    ap.add_argument("--work-dir", default=None,
                    help="build into this directory instead of work/<test>")
    ap.add_argument("-d", "--debug", action="store_true")
    args = ap.parse_args()

    set_debug(args.debug)
    if args.mesh_dir:
        set_mesh_dir(args.mesh_dir)
    work = pathlib.Path(args.work_dir) if args.work_dir else ROOT / "work" / args.test
    elfs = build_test(args.test, work,
                      cluster_src=MESH / args.cluster,
                      host_extra=[MESH / s for s in args.host_extra])
    for name, elf in elfs.items():
        # A build directory outside the repository root is legitimate -- a sweep
        # cell is one -- so fall back to the full path rather than raising from
        # a print statement.
        try:
            shown = elf.relative_to(ROOT)
        except ValueError:
            shown = elf
        print(f"{name:7} {shown}")


if __name__ == "__main__":
    main()
