#!/usr/bin/env python3
"""hetero-sim pipeline: ONNX op -> Deeploy C -> per-core GVSoC runs -> metrics.

Usage:
  python pipeline/run.py Tests/Kernels/FP32/GEMM/Regular            # Deeploy test name
  python pipeline/run.py /path/to/dir                               # dir with network.onnx + inputs.npz + outputs.npz
  python pipeline/run.py <op> --cores cva6,snitch,spatz
  python pipeline/run.py <op> --memory ideal                        # zero-latency memory instead of the modelled one
  python pipeline/run.py <op> --spatz-kernels autovec               # GCC's RVV instead of the hand-written kernels
  python pipeline/run.py <op> --debug                               # trace every command and the files it produced
                                                                    # (same as HES_DEBUG=1; trace goes to stderr)

Each simulated core type:
  cva6   : rv64gc host core, L1 I$/D$ + L2 + DRAM             (target cva6_real)
  snitch : rv32imafd Snitch core, data in cluster TCDM        (target snitch_real)
  spatz  : Snitch + Spatz VPU, RVV kernels, data in TCDM      (target spatz_real)

--memory selects how main memory is simulated. `real` (the default) runs the
targets above, which model cache misses, DRAM latency and refill bandwidth.
`ideal` runs the zero-latency targets the first version of the pipeline used
(cva6_ideal, snitch, spatz), where every access completes in a single cycle and
cycle counts measure compute alone.
"""

import argparse
import json
import os
import re
import subprocess
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DEEPLOY_TEST, GENERIC_LIB, GVSOC, PYTHON, RESULTS, ROOT,  # noqa: E402
                    RUNTIME, TARGETS, TC, WORK, dbg, note_file, set_debug, sh)


@dataclass
class Core:
    name: str
    march: str
    mabi: str
    linker: Path
    # GVSoC target per memory model, keyed as in MEMORY_MODELS.
    targets: dict[str, str]
    kernel_flags: list[str] = field(default_factory=list)
    # Core-specific kernels compiled in addition to the Deeploy Generic ones,
    # and the Generic functions they replace. Each replaced function is renamed
    # <name>_generic across the Generic library, which frees the name for the
    # core-specific definition and keeps the original reachable as a fallback
    # for the shapes the optimized version does not cover.
    kernel_srcs: list[Path] = field(default_factory=list)
    kernel_overrides: list[str] = field(default_factory=list)
    kernel_incs: list[Path] = field(default_factory=list)

    def target(self, memory: str) -> str:
        return self.targets[memory]

    def rename_flags(self) -> list[str]:
        return [f"-D{sym}={sym}_generic" for sym in self.kernel_overrides]


# How main memory is simulated. `real` uses the targets in targets/, which
# model the memory system; `ideal` uses zero-latency memory, where a cycle
# count is a pure compute cost.
MEMORY_MODELS = {
    "real": "modelled memory system",
    "ideal": "ideal memory / infinite cache",
}
DEFAULT_MEMORY = "real"


CORES = {
    "cva6": Core(
        name="cva6",
        targets={"real": "cva6_real", "ideal": "cva6_ideal"},
        march="rv64imafdc_zicsr_zifencei",
        mabi="lp64d",
        linker=RUNTIME / "common" / "link.ld",
        kernel_flags=["-O3"],
    ),
    # The same CVA6 host with the ara_v2 vector unit attached (targets/
    # ara_host.py). Opt in with --cores cva6,ara to compare the orchestrator
    # with and without vector hardware on identical source; it is not in the
    # default set, so the committed baselines are unaffected.
    #
    # There is no zero-latency variant of this board, so --memory ideal runs
    # the same modelled-memory target and its cycles are not comparable with
    # the other cores' ideal numbers.
    "ara": Core(
        name="ara",
        targets={"real": "ara_host", "ideal": "ara_host"},
        march="rv64imafdc_zicsr_zifencei_v",
        mabi="lp64d",
        linker=RUNTIME / "common" / "link.ld",
        # -ffast-math is what lets GCC vectorize the FP reductions in the
        # Generic kernels. Glue stays scalar via GLUE_FLAGS: with `v` in the
        # march GCC will otherwise emit RVV for ordinary control loops, which
        # is how the cluster staging plan was corrupted during M2.
        kernel_flags=["-O3", "-ffast-math"],
    ),
    "snitch": Core(
        name="snitch",
        targets={"real": "snitch_real", "ideal": "snitch"},
        march="rv32imafd_zicsr_zifencei",
        mabi="ilp32d",
        linker=RUNTIME / "snitch" / "link.ld",
        kernel_flags=["-O3"],
        # Snitch's FP subsystem is only worth its area with Xssr and Xfrep, so
        # the FP kernels that dominate these ops are hand-written against them
        # (runtime/snitch/snitch_ssr.h). GCC knows neither extension; both are
        # emitted as raw encodings.
        kernel_srcs=sorted((RUNTIME / "snitch" / "kernels").glob("*.c")),
        kernel_overrides=["MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32",
                          "Conv2d_fp32_fp32_fp32_NCHW"],
        kernel_incs=[RUNTIME / "snitch"],
    ),
    "spatz": Core(
        name="spatz",
        targets={"real": "spatz_real", "ideal": "spatz"},
        march="rv32imafd_zicsr_zifencei_v",
        mabi="ilp32d",
        linker=RUNTIME / "spatz" / "link.ld",
        # Flags for the kernels GCC still compiles: -ffast-math is what lets it
        # vectorize FP reductions at all. The MatMul/GEMM kernels that dominate
        # these ops are hand-written against RVV instead (SPATZ_TUNED_KERNELS,
        # applied in main), because autovectorizing them puts the reduction
        # inside the vector unit and costs about 5x.
        kernel_flags=["-O3", "-ffast-math"],
        # MatMul and GEMM are written against RVV by hand. Left to the
        # autovectorizer, GCC picks the reduction axis of the dot product,
        # which costs a strided load of B and a horizontal reduction for every
        # output element; the microkernel vectorizes the output columns
        # instead (runtime/spatz/kernels/gemm_fp32_rvv.c).
        kernel_srcs=sorted((RUNTIME / "spatz" / "kernels").glob("*.c")),
        kernel_overrides=["MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32"],
    ),
}

# Hand-written RVV kernels for Spatz, applied unless --spatz-kernels autovec
# asks for whatever GCC makes of the Deeploy Generic sources. Autovectorized
# RVV puts the reduction inside the vector unit -- a strided gather down a
# column of B plus a vfredusum per output element -- which costs about 5x.
SPATZ_TUNED_KERNELS = {
    "kernel_srcs": sorted((RUNTIME / "spatz" / "kernels").glob("*.c")),
    "kernel_overrides": ["MatMul_fp32_fp32_fp32", "Gemm_fp32_fp32_fp32_fp32"],
    "kernel_incs": [RUNTIME / "spatz"],
}

# Clearing these on a Core drops its hand-written kernels and leaves the
# generated network calling the Deeploy ones. Spatz declares none of them
# statically -- they are applied below -- so clearing is what keeps
# --spatz-kernels autovec honest if any are ever added to CORES.
KERNEL_FIELDS = ("kernel_srcs", "kernel_overrides", "kernel_incs")

GLUE_FLAGS = ["-O2", "-fno-tree-vectorize"]
COMMON_FLAGS = ["-mcmodel=medany", "-nostdlib", "-nostartfiles", "-ffunction-sections",
                "-DDEEPLOY_GENERIC_PLATFORM"]
# --allow-multiple-definition: upstream Deeploy duplicates _plp_sqrt_q32 in two kernels
LINK_LIBS = ["-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lc", "-lm", "-lgcc"]


def resolve_test_dir(op: str) -> Path:
    for cand in (Path(op), DEEPLOY_TEST / op, DEEPLOY_TEST / "Tests" / op):
        if (cand / "network.onnx").is_file():
            return cand.resolve()
    sys.exit(f"error: cannot find network.onnx under '{op}' "
             f"(tried absolute path and relative to {DEEPLOY_TEST})")


def generate_c(test_dir: Path, gen_dir: Path) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    r = sh([str(PYTHON), "generateNetwork.py", "-t", str(test_dir),
            "-p", "Generic", "-d", str(gen_dir)], cwd=DEEPLOY_TEST,
           produces=[gen_dir])
    if r.returncode != 0:
        sys.exit(f"Deeploy codegen failed:\n{r.stdout}\n{r.stderr}")


def build(core: Core, gen_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / "net.elf"

    incs = [f"-I{gen_dir}", f"-I{GENERIC_LIB / 'inc'}", f"-I{RUNTIME / 'common'}"]
    incs += [f"-I{p}" for p in core.kernel_incs]
    arch = ["-march=" + core.march, "-mabi=" + core.mabi]

    objs = []

    def compile_(srcs, flags, tag):
        for src in srcs:
            obj = out_dir / f"{tag}_{Path(src).stem}.o"
            r = sh([str(TC), *arch, *COMMON_FLAGS, *flags, *incs,
                    f"-DCORE_NAME=\"{core.name}\"", "-c", str(src), "-o", str(obj)],
                   produces=[obj])
            if r.returncode != 0:
                sys.exit(f"[{core.name}] compile failed for {src}:\n{r.stderr}")
            objs.append(obj)

    glue = [RUNTIME / "common" / "crt0.S",
            RUNTIME / "common" / "syscalls.c",
            RUNTIME / "common" / "bench_main.c",
            gen_dir / "Network.c"]
    kernels = sorted((GENERIC_LIB / "src").glob("*.c"))

    compile_(glue, GLUE_FLAGS, "glue")
    # The rename only applies to the Generic library: the generated network and
    # the core's own kernels keep calling the overridden names.
    compile_(kernels, core.kernel_flags + core.rename_flags(), "k")
    compile_(core.kernel_srcs, core.kernel_flags, "core")

    r = sh([str(TC), *arch, *COMMON_FLAGS, f"-T{core.linker}",
            *[str(o) for o in objs], *LINK_LIBS, "-o", str(elf)],
           produces=[elf])
    if r.returncode != 0:
        sys.exit(f"[{core.name}] link failed:\n{r.stderr}")
    return elf


def parse_caches(out: str) -> list[dict]:
    """Per-cache counters, printed by the timing caches at the end of a run."""
    caches = []
    for m in re.finditer(r"\[HES-MEM\] cache=(\S+) accesses=(\d+) reads=(\d+) writes=(\d+) "
                         r"hits=(\d+) misses=(\d+) latency_cycles=(\d+)", out):
        # Hits and misses are counted per line looked up, so an access that
        # straddles two lines contributes two of them.
        lookups = int(m.group(5)) + int(m.group(6))
        caches.append({
            "cache": m.group(1).split("/")[-1],
            "accesses": int(m.group(2)),
            "reads": int(m.group(3)),
            "writes": int(m.group(4)),
            "hits": int(m.group(5)),
            "misses": int(m.group(6)),
            "hit_rate": round(int(m.group(5)) / lookups, 4) if lookups else None,
            "latency_cycles": int(m.group(7)),
        })
    return caches


def simulate(core: Core, memory: str, elf: Path, run_dir: Path, timeout_s: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(GVSOC), f"--target-dir={TARGETS}",
           f"--target={core.target(memory)}", f"--binary={elf}", "run"]

    env = dict(os.environ)
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env['PATH']}"

    t0 = time.time()
    try:
        r = sh(cmd, cwd=run_dir, timeout=timeout_s, env=env, produces=[run_dir])
    except subprocess.TimeoutExpired:
        return {"core": core.name, "status": "timeout"}
    wall = time.time() - t0

    out = r.stdout + r.stderr
    log = run_dir / "sim.log"
    log.write_text(out)
    note_file(log, "simulator stdout+stderr")

    m = re.search(r"\[HES\] core=(\S+) cycles=(\d+) instret=(\d+) errors=(\d+) "
                  r"total=(\d+) maxdiff_e6=(\d+)", out)
    if not m:
        return {"core": core.name, "status": "no-metrics",
                "log_tail": out.strip().splitlines()[-6:]}

    result = {
        "core": core.name,
        "target": core.target(memory),
        "status": "ok" if m.group(4) == "0" else "wrong-result",
        "cycles": int(m.group(2)),
        "instret": int(m.group(3)) or None,
        "errors": int(m.group(4)),
        "outputs": int(m.group(5)),
        "maxdiff": int(m.group(6)) / 1e6,
        "sim_wall_s": round(wall, 1),
    }
    caches = parse_caches(out)
    if caches:
        result["caches"] = caches
    return result


def report(op_name: str, memory: str, results: list[dict],
           spatz_kernels: str = "tuned") -> None:
    ok = {r["core"]: r for r in results if "cycles" in r}
    base = ok.get("cva6")

    kern = "" if spatz_kernels == "tuned" else f", spatz kernels: {spatz_kernels}"
    print(f"\n=== {op_name} — per-core performance "
          f"({MEMORY_MODELS[memory]}{kern}) ===\n")
    hdr = f"{'core':8} {'status':13} {'cycles':>12} {'instret':>10} {'speedup':>9}  {'maxdiff':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        cyc = r.get("cycles")
        speed = f"{base['cycles'] / cyc:.2f}x" if base and cyc else "-"
        print(f"{r['core']:8} {r['status']:13} "
              f"{cyc if cyc is not None else '-':>12} "
              f"{r.get('instret') or '-':>10} {speed:>9}  "
              f"{r.get('maxdiff', '-'):>10}")
    print()

    if any(r.get("caches") for r in results):
        # The caches count the whole program — startup, the timed op, and the
        # output check — while `cycles` above covers the timed op alone.
        print("cache counters (whole run, not only the timed op):\n")
        hdr = (f"{'core':8} {'cache':8} {'accesses':>10} {'misses':>9} "
               f"{'hit rate':>9} {'latency':>10}")
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            for c in r.get("caches", []):
                rate = f"{100 * c['hit_rate']:.2f}%" if c["hit_rate"] is not None else "-"
                print(f"{r['core']:8} {c['cache']:8} {c['accesses']:>10} {c['misses']:>9} "
                      f"{rate:>9} {c['latency_cycles']:>10}")
        print()

    RESULTS.mkdir(exist_ok=True)
    suffix = "" if memory == DEFAULT_MEMORY else f"-{memory}"
    suffix += "" if spatz_kernels == "tuned" else f"-spatz-{spatz_kernels}"
    out = RESULTS / f"{op_name.replace('/', '_')}{suffix}.json"
    out.write_text(json.dumps({"op": op_name, "memory": memory,
                               "spatz_kernels": spatz_kernels,
                               "results": results}, indent=2))
    note_file(out, "per-core metrics")
    print(f"results written to {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", help="Deeploy test dir (network.onnx + inputs.npz + outputs.npz)")
    ap.add_argument("--cores", default="cva6,snitch,spatz")
    ap.add_argument("--memory", choices=list(MEMORY_MODELS), default=DEFAULT_MEMORY,
                    help="how main memory is simulated (default: %(default)s)")
    ap.add_argument("--spatz-kernels", choices=["tuned", "autovec"], default="tuned",
                    help="MatMul/GEMM kernels for spatz: the hand-written RVV ones "
                         "in runtime/spatz/kernels (default), or autovectorized "
                         "from the Deeploy Generic sources")
    ap.add_argument("--timeout", type=int, default=600, help="per-sim timeout [s]")
    ap.add_argument("-d", "--debug", action="store_true",
                    help="trace every command run and the files it generated (on stderr)")
    args = ap.parse_args()

    set_debug(args.debug or os.environ.get("HES_DEBUG", "") not in ("", "0"))

    if args.spatz_kernels == "tuned":
        for key, value in SPATZ_TUNED_KERNELS.items():
            setattr(CORES["spatz"], key, value)
    else:
        for field in KERNEL_FIELDS:
            setattr(CORES["spatz"], field, [])

    test_dir = resolve_test_dir(args.op)
    op_name = test_dir.name if test_dir.name != "." else "op"
    try:
        # resolve() both sides: test_dir is resolved, so a symlinked deps/
        # checkout would otherwise not compare equal and the op would lose its
        # Tests/-relative name.
        op_name = str(test_dir.relative_to((DEEPLOY_TEST / "Tests").resolve())).replace("/", "_")
    except ValueError:
        pass

    work = WORK / op_name
    gen_dir = work / "gen"

    print(f"[1/3] Deeploy: {test_dir.name}/network.onnx -> C  ({gen_dir.relative_to(ROOT)})")
    generate_c(test_dir, gen_dir)

    results = []
    cores = [c.strip() for c in args.cores.split(",")]
    for i, cname in enumerate(cores):
        core = CORES[cname]
        print(f"[2/3] build + [3/3] simulate: {cname} "
              f"({i + 1}/{len(cores)})", flush=True)
        elf = build(core, gen_dir, work / cname)
        # The binary does not depend on the memory model, only the target does,
        # so runs of the two models keep their logs side by side.
        results.append(simulate(core, args.memory, elf,
                                work / cname / f"run-{args.memory}", args.timeout))

    report(op_name, args.memory, results, args.spatz_kernels)


if __name__ == "__main__":
    main()
