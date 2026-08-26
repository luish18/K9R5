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
                      "Conv2d_fp32_fp32_fp32_NCHW"],
)

SPATZ = Image(
    name="spatz",
    march="rv32imafd_zicsr_zifencei_v",
    mabi="ilp32d",
    linker=MESH / "spatz.ld",
    crt0=MESH / "crt0_cluster.S",
    defines=["-DHES_CLUSTER_SPATZ"],
    # -ffast-math is what lets GCC vectorize the FP reductions in the Generic
    # kernels to RVV; without it the Spatz vector unit sits idle.
    kernel_flags=["-O3", "-ffast-math"],
)

IMAGES = {img.name: img for img in (HOST, SNITCH, SPATZ)}


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


def build_network(gen_dir: Path, work: Path, samples: int = 1) -> dict:
    """Build the three ELFs for a Deeploy-generated network.

    The host links the generated Network.c, the host runtime and the Generic
    kernel library -- it runs the nodes the mapper left on it. Each cluster
    links the job loop and its own kernels.
    """
    host_sources = [
        RUNTIME / "common" / "syscalls.c",
        MESH / "hes_host.c",
        MESH / "host_main.c",
        gen_dir / "Network.c",
    ]
    cluster_src = MESH / "cluster_main.c"
    incs = [f"-I{gen_dir}"]

    return {
        "host": build(HOST, host_sources, work / "host", with_kernels = True,
                      extra_flags = [*incs, f"-DHES_SAMPLES={samples}"]),
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
    ap.add_argument("-d", "--debug", action="store_true")
    args = ap.parse_args()

    set_debug(args.debug)
    elfs = build_test(args.test, ROOT / "work" / args.test,
                      cluster_src=MESH / args.cluster,
                      host_extra=[MESH / s for s in args.host_extra])
    for name, elf in elfs.items():
        print(f"{name:7} {elf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
