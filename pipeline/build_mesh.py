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
LINK_LIBS = ["-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lc", "-lm", "-lgcc"]

# Shared include path for every image in a run.
INCS = [RUNTIME / "common", MESH, GENERIC_LIB / "inc"]


class Image:
    """One of the three binaries a hetero_soc run needs."""

    def __init__(self, name, march, mabi, linker, crt0, defines=(), extra_incs=()):
        self.name = name
        self.march = march
        self.mabi = mabi
        self.linker = linker
        self.crt0 = crt0
        self.defines = list(defines)
        self.extra_incs = list(extra_incs)

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
)

SPATZ = Image(
    name="spatz",
    march="rv32imafd_zicsr_zifencei_v",
    mabi="ilp32d",
    linker=MESH / "spatz.ld",
    crt0=MESH / "crt0_cluster.S",
    defines=["-DHES_CLUSTER_SPATZ"],
)

IMAGES = {img.name: img for img in (HOST, SNITCH, SPATZ)}


def build(image: Image, sources, out_dir: Path, opt="-O2", extra_flags=()) -> Path:
    """Compile and link one image. Returns the ELF path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / f"{image.name}.elf"

    objs = []
    for src in [image.crt0] + list(sources):
        obj = out_dir / f"{Path(src).stem}.o"
        r = sh([str(TC), *image.arch_flags(), *COMMON_FLAGS, opt, *extra_flags,
                *image.defines, *image.include_flags(), "-c", str(src), "-o", str(obj)],
               produces=[obj])
        if r.returncode != 0:
            sys.exit(f"[{image.name}] compile failed for {src}:\n{r.stderr}")
        objs.append(obj)

    r = sh([str(TC), *image.arch_flags(), *COMMON_FLAGS, f"-T{image.linker}",
            *[str(o) for o in objs], *LINK_LIBS, "-o", str(elf)], produces=[elf])
    if r.returncode != 0:
        sys.exit(f"[{image.name}] link failed:\n{r.stderr}")
    return elf


def build_test(test: str, work: Path) -> dict:
    """Build a standalone bare-metal check: one host program, one cluster
    program that both clusters run."""
    host_src = RUNTIME / "tests" / f"{test}.c"
    cluster_src = MESH / "cluster_probe.c"
    syscalls = RUNTIME / "common" / "syscalls.c"

    return {
        "host": build(HOST, [syscalls, host_src], work / "host"),
        "snitch": build(SNITCH, [cluster_src], work / "snitch"),
        "spatz": build(SPATZ, [cluster_src], work / "spatz"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="name under runtime/tests")
    ap.add_argument("-d", "--debug", action="store_true")
    args = ap.parse_args()

    set_debug(args.debug)
    elfs = build_test(args.test, ROOT / "work" / args.test)
    for name, elf in elfs.items():
        print(f"{name:7} {elf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
