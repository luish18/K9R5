#!/usr/bin/env python3
"""Sweep a group of ONNX models across a design space.

    python pipeline/sweep/run.py --models ops/mnist ops/kws --grid ofat -o results/sweep/s1

For each (design point, model) it writes one JSONL row carrying the resolved
design, the cycles, the cache counters and energy, and the modelled area -- so a
row says what was run, not what was asked for.

Structure, and why:

  * Designs are grouped by build key. Everything except VLEN is a property
    GVSoC reads from its generated config, so a whole grid usually runs against
    one existing build; the driver refuses up front rather than silently
    running the wrong VLEN if a build is missing.
  * Calibration is per design, not per cell, and cached: the engine cost table
    depends on the hardware, not on which network is running.
  * Every cell gets its own copy of runtime/mesh. That is not fastidiousness --
    hes_host.h and friends include "hes_system.h" with quotes, which GCC
    resolves next to the including file before any -I, so a cell that shared
    the directory would link its own linker script against another cell's
    addresses and report plausible wrong numbers.
  * A cell whose status is not ok is recorded as infeasible with the reason,
    never dropped. A design too small to hold a layer must show up as
    infeasible rather than as fast.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import area as area_model          # noqa: E402
import design as design_mod        # noqa: E402

PYTHON = ROOT / ".venv" / "bin" / "python"
MESH = ROOT / "runtime" / "mesh"


# --- grids ------------------------------------------------------------------

# One factor at a time: the baseline, then each knob moved alone. This is the
# screening pass -- it says which knobs move cycles at all, and a full factorial
# over everything would be 10^5 designs for no extra information.
OFAT = {
    "HOST_NB_LANES": [2, 8],
    "SPATZ_NB_LANES": [2, 8],
    "SNITCH_NB_CORE": [5, 17],
    "SPATZ_NB_CORE": [5, 17],
    "DCACHE_SIZE": [16 * 1024, 64 * 1024, 128 * 1024],
    "ICACHE_SIZE": [8 * 1024, 32 * 1024],
    "L2_SIZE": [128 * 1024, 2048 * 1024],
    "L2_WAYS": [4, 16],
    "LINE_SIZE": [32, 128],
    "TCDM_SIZE": [0x10000, 0x40000],
    "WIDE_AXI_WIDTH": [32, 128],
}


def expand(grid_name: str):
    """The design points of a named grid, baseline first."""
    if grid_name != "ofat":
        raise SystemExit(f"unknown grid: {grid_name}")
    designs = [{}]
    for knob, values in OFAT.items():
        for v in values:
            designs.append({knob: v})
    return designs


# --- running one cell -------------------------------------------------------

def sh(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def prepare(design, cell_dir, env):
    """The per-design work: its own mesh copy and generated header."""
    mesh = cell_dir / "mesh"
    if not mesh.exists():
        shutil.copytree(MESH, mesh)
    r = sh([PYTHON, ROOT / "pipeline" / "gen_system_header.py", "--out-dir", mesh], env=env)
    if r.returncode:
        raise RuntimeError(f"header generation failed:\n{r.stdout}\n{r.stderr}")
    return mesh


def calibrate(design_dir, mesh, env, host):
    """Re-measure the engine cost table for this design. Cached per design."""
    rates = design_dir / "rates.json"
    if rates.exists():
        return rates

    build = design_dir / "calib"
    r = sh([PYTHON, ROOT / "pipeline" / "build_mesh.py", "--test", "mesh_calib",
            "--cluster", "cluster_main.c", "--host-extra", "hes_host.c",
            "--mesh-dir", mesh, "--work-dir", build], env=env)
    if r.returncode:
        raise RuntimeError(f"calibration build failed:\n{r.stdout}\n{r.stderr}")

    run_dir = build / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cenv = dict(env)
    cenv["HES_ELF_SNITCH"] = str(build / "snitch" / "snitch.elf")
    cenv["HES_ELF_SPATZ"] = str(build / "spatz" / "spatz.elf")
    cenv["PATH"] = f"{ROOT / '.venv' / 'bin'}:{cenv['PATH']}"
    target = "hetero_ara" if host == "ara" else "hetero_soc"
    r = sh([ROOT / "deps/gvsoc/install/bin/gvsoc", f"--target-dir={ROOT / 'targets'}",
            f"--target={target}", f"--binary={build / 'host' / 'host.elf'}", "run"],
           cwd=run_dir, env=cenv)
    log = run_dir / "calib.log"
    log.write_text(r.stdout + r.stderr)

    r = sh([PYTHON, HERE / "calibrate.py", log, "-o", rates, "--host", host], env=env)
    if r.returncode:
        raise RuntimeError(f"calibration failed for this design:\n{r.stdout}\n{r.stderr}")
    return rates


def run_cell(design, model, out_dir, host, power, images):
    """One (design, model) measurement."""
    slug = design_mod.slug(design)
    design_dir = out_dir / "designs" / slug
    design_dir.mkdir(parents=True, exist_ok=True)
    path = design_mod.write(design, design_dir / "design.json")

    env = dict(os.environ)
    env["HES_DESIGN"] = str(path)

    row = {"design_slug": slug, "design": design_mod.resolve(design),
           "model": Path(model).name, "host": host}

    bad = design_mod.validate(design)
    if bad:
        row.update(status="invalid", reasons=bad)
        return row

    t0 = time.time()
    try:
        mesh = prepare(design, design_dir, env)
        rates = calibrate(design_dir, mesh, env, host)
        env["HES_RATES"] = str(rates)

        cell = out_dir / "cells" / f"{slug}__{Path(model).name}"
        cell.mkdir(parents=True, exist_ok=True)
        result = cell / "result.json"
        cmd = [PYTHON, ROOT / "pipeline" / "run_hetero.py", model,
               "--host", host, "--mesh-dir", mesh, "--tag", slug,
               "--out", result, "--images", images, "-q"]
        if power:
            cmd.append("--power")
        r = sh(cmd, env=env)
        if not result.exists():
            row.update(status="failed", reasons=[(r.stdout + r.stderr)[-1500:]])
            return row
        res = json.load(result.open())["result"]
        row["status"] = res.get("status", "unknown")
        for k in ("cycles", "cycles_per_image", "cycles_per_clip", "accuracy",
                  "offload_failures", "maxdiff", "caches", "per_engine_cycles"):
            if k in res:
                row[k] = res[k]
        if row["status"] == "ok" and res.get("offload_failures"):
            row["status"] = "wrong-result"
    except Exception as exc:                        # noqa: BLE001
        row.update(status="error", reasons=[str(exc)])
        return row
    finally:
        row["wall_s"] = round(time.time() - t0, 1)

    # Area is a pure function of the design, so it costs nothing to attach and
    # makes every row self-contained for the Pareto pass.
    try:
        parts = area_model.breakdown({"_path": str(path)})
        row["area_au"] = sum(parts.values())
        row["area_breakdown"] = parts
        row["area_coefficients_sourced"] = all(
            c["sourced"] for c in area_model.COEFFS.values())
    except Exception as exc:                        # noqa: BLE001
        row["area_error"] = str(exc)

    if "caches" in row and row["caches"]:
        pj = [c.get("dynamic_pj") for c in row["caches"]]
        row["cache_dynamic_pj"] = sum(p for p in pj if p) if any(pj) else None

    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="op directories")
    ap.add_argument("--grid", default="ofat")
    ap.add_argument("-o", "--out", required=True, help="sweep output directory")
    ap.add_argument("--host", default="cva6", choices=("cva6", "ara"))
    ap.add_argument("--power", action="store_true", help="measure energy too (slower)")
    ap.add_argument("--images", default="16", help="samples per model run")
    ap.add_argument("--limit", type=int, default=None, help="stop after N designs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    designs = expand(args.grid)
    if args.limit:
        designs = designs[:args.limit]

    # Designs needing a build we do not have would otherwise run against
    # whatever VLEN happens to be compiled in, and report it under the label
    # they asked for. Refuse instead.
    have = design_mod.build_key({})
    needs_build = sorted({design_mod.build_key(d) for d in designs} - {have})
    if needs_build:
        raise SystemExit(
            "error: these design points need GVSoC builds that do not exist: "
            + ", ".join(needs_build)
            + "\n  VLEN is compiled into the model, so running them now would "
              "measure the built VLEN and label it as the requested one.\n"
              "  Build them first, or drop the VLEN axis from the grid.")

    jsonl = out / "sweep.jsonl"
    n = 0
    with jsonl.open("w") as fh:
        for d in designs:
            for model in args.models:
                row = run_cell(d, model, out, args.host, args.power, args.images)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                cyc = row.get("cycles_per_image") or row.get("cycles_per_clip") or row.get("cycles")
                print(f"[{n:3}] {row['design_slug'][:34]:34} {row['model']:8} "
                      f"{row['status']:12} cycles={cyc} wall={row.get('wall_s')}s")
    print(f"\n{n} rows -> {jsonl}")


if __name__ == "__main__":
    main()
