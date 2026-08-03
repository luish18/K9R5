#!/usr/bin/env python3
"""hetero-sim pipeline: ONNX op -> Deeploy C -> per-core GVSoC runs -> metrics.

Usage:
  python pipeline/run.py Tests/Kernels/FP32/GEMM/Regular            # Deeploy test name
  python pipeline/run.py /path/to/dir                               # dir with network.onnx + inputs.npz + outputs.npz
  python pipeline/run.py <op> --cores cva6,snitch,spatz

Each simulated core type:
  cva6   : rv64gc host core, ideal zero-latency memory        (target cva6_ideal)
  snitch : rv32imafd Snitch core, data in single-cycle TCDM   (target snitch)
  spatz  : Snitch + Spatz VPU, RVV kernels, data in TCDM      (target spatz)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEEPLOY = ROOT / "deps" / "deeploy"
DEEPLOY_TEST = DEEPLOY / "DeeployTest"
GENERIC_LIB = DEEPLOY / "TargetLibraries" / "Generic"
GVSOC = ROOT / "deps" / "gvsoc" / "install" / "bin" / "gvsoc"
TC = ROOT / "toolchains" / "xpack-riscv-none-elf-gcc-15.2.0-1" / "bin" / "riscv-none-elf-gcc"
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNTIME = ROOT / "runtime"
WORK = ROOT / "work"
RESULTS = ROOT / "results"


@dataclass
class Core:
    name: str
    target: str
    march: str
    mabi: str
    linker: Path
    target_dir: Path | None = None
    kernel_flags: list[str] = field(default_factory=list)


CORES = {
    "cva6": Core(
        name="cva6",
        target="cva6_ideal",
        target_dir=ROOT / "targets",
        march="rv64imafdc_zicsr_zifencei",
        mabi="lp64d",
        linker=RUNTIME / "common" / "link.ld",
        kernel_flags=["-O3"],
    ),
    "snitch": Core(
        name="snitch",
        target="snitch",
        march="rv32imafd_zicsr_zifencei",
        mabi="ilp32d",
        linker=RUNTIME / "snitch" / "link.ld",
        kernel_flags=["-O3"],
    ),
    "spatz": Core(
        name="spatz",
        target="spatz",
        march="rv32imafd_zicsr_zifencei_v",
        mabi="ilp32d",
        linker=RUNTIME / "spatz" / "link.ld",
        # Autovectorize kernels to RVV. -ffast-math is required for GCC to
        # vectorize FP reductions (dot products / GEMM inner loops).
        kernel_flags=["-O3", "-ffast-math"],
    ),
}

GLUE_FLAGS = ["-O2", "-fno-tree-vectorize"]
COMMON_FLAGS = ["-mcmodel=medany", "-nostdlib", "-nostartfiles", "-ffunction-sections",
                "-DDEEPLOY_GENERIC_PLATFORM"]
# --allow-multiple-definition: upstream Deeploy duplicates _plp_sqrt_q32 in two kernels
LINK_LIBS = ["-Wl,--gc-sections", "-Wl,--allow-multiple-definition", "-lc", "-lm", "-lgcc"]


def sh(cmd, cwd=None, timeout=None, env=None):
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, env=env,
                          capture_output=True, text=True)


def resolve_test_dir(op: str) -> Path:
    for cand in (Path(op), DEEPLOY_TEST / op, DEEPLOY_TEST / "Tests" / op):
        if (cand / "network.onnx").is_file():
            return cand.resolve()
    sys.exit(f"error: cannot find network.onnx under '{op}' "
             f"(tried absolute path and relative to {DEEPLOY_TEST})")


def generate_c(test_dir: Path, gen_dir: Path) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    r = sh([str(PYTHON), "generateNetwork.py", "-t", str(test_dir),
            "-p", "Generic", "-d", str(gen_dir)], cwd=DEEPLOY_TEST)
    if r.returncode != 0:
        sys.exit(f"Deeploy codegen failed:\n{r.stdout}\n{r.stderr}")


def build(core: Core, gen_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / "net.elf"

    incs = [f"-I{gen_dir}", f"-I{GENERIC_LIB / 'inc'}", f"-I{RUNTIME / 'common'}"]
    arch = ["-march=" + core.march, "-mabi=" + core.mabi]

    objs = []

    def compile_(srcs, flags, tag):
        for src in srcs:
            obj = out_dir / f"{tag}_{Path(src).stem}.o"
            r = sh([str(TC), *arch, *COMMON_FLAGS, *flags, *incs,
                    f"-DCORE_NAME=\"{core.name}\"", "-c", str(src), "-o", str(obj)])
            if r.returncode != 0:
                sys.exit(f"[{core.name}] compile failed for {src}:\n{r.stderr}")
            objs.append(obj)

    glue = [RUNTIME / "common" / "crt0.S",
            RUNTIME / "common" / "syscalls.c",
            RUNTIME / "common" / "bench_main.c",
            gen_dir / "Network.c"]
    kernels = sorted((GENERIC_LIB / "src").glob("*.c"))

    compile_(glue, GLUE_FLAGS, "glue")
    compile_(kernels, core.kernel_flags, "k")

    r = sh([str(TC), *arch, *COMMON_FLAGS, f"-T{core.linker}",
            *[str(o) for o in objs], *LINK_LIBS, "-o", str(elf)])
    if r.returncode != 0:
        sys.exit(f"[{core.name}] link failed:\n{r.stderr}")
    return elf


def simulate(core: Core, elf: Path, run_dir: Path, timeout_s: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(GVSOC)]
    if core.target_dir:
        cmd += [f"--target-dir={core.target_dir}"]
    cmd += [f"--target={core.target}", f"--binary={elf}", "run"]

    import os
    env = dict(os.environ)
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env['PATH']}"

    t0 = time.time()
    try:
        r = sh(cmd, cwd=run_dir, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"core": core.name, "status": "timeout"}
    wall = time.time() - t0

    out = r.stdout + r.stderr
    (run_dir / "sim.log").write_text(out)

    m = re.search(r"\[HES\] core=(\S+) cycles=(\d+) instret=(\d+) errors=(\d+) "
                  r"total=(\d+) maxdiff_e6=(\d+)", out)
    if not m:
        return {"core": core.name, "status": "no-metrics",
                "log_tail": out.strip().splitlines()[-6:]}

    return {
        "core": core.name,
        "status": "ok" if m.group(4) == "0" else "wrong-result",
        "cycles": int(m.group(2)),
        "instret": int(m.group(3)) or None,
        "errors": int(m.group(4)),
        "outputs": int(m.group(5)),
        "maxdiff": int(m.group(6)) / 1e6,
        "sim_wall_s": round(wall, 1),
    }


def report(op_name: str, results: list[dict]) -> None:
    ok = {r["core"]: r for r in results if "cycles" in r}
    base = ok.get("cva6")

    print(f"\n=== {op_name} — per-core performance (ideal memory / infinite cache) ===\n")
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

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{op_name.replace('/', '_')}.json"
    out.write_text(json.dumps({"op": op_name, "results": results}, indent=2))
    print(f"results written to {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", help="Deeploy test dir (network.onnx + inputs.npz + outputs.npz)")
    ap.add_argument("--cores", default="cva6,snitch,spatz")
    ap.add_argument("--timeout", type=int, default=600, help="per-sim timeout [s]")
    args = ap.parse_args()

    test_dir = resolve_test_dir(args.op)
    op_name = test_dir.name if test_dir.name != "." else "op"
    try:
        op_name = str(test_dir.relative_to(DEEPLOY_TEST / "Tests")).replace("/", "_")
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
        results.append(simulate(core, elf, work / cname / "run", args.timeout))

    report(op_name, results)


if __name__ == "__main__":
    main()
