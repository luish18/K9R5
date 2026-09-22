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



class Progress:
    """One updating line for the whole sweep, in the style run_hetero.py uses.

    A screening grid is two dozen cells of half a minute each, so the useful
    question during a run is "how far in, and how much longer" -- which a line
    per finished cell answers only in arrears. The bar also names the phase the
    current cell is in, because the phases have very different lengths and a
    stall in one of them looks nothing like a stall in another.

    Falls back to one line per cell when stdout is not a terminal, so piping to
    a file or through `docker exec` without -t stays readable instead of filling
    with carriage returns.
    """

    WIDTH = 24

    def __init__(self, total: int, mode: str):
        self.total = total
        self.done = 0
        self.t0 = time.time()
        self.durations = []
        self.cell = ""
        self.phase = ""
        if mode == "auto":
            self.bar = sys.stdout.isatty()
        else:
            self.bar = mode == "bar"

    @staticmethod
    def _clock(seconds) -> str:
        seconds = int(max(0, seconds))
        return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"

    def _eta(self):
        if not self.durations:
            return None
        # Median, not mean: one rebuilt or pathologically slow cell should not
        # drag the estimate for the rest.
        ordered = sorted(self.durations)
        typical = ordered[len(ordered) // 2]
        return typical * (self.total - self.done)

    def start(self, cell: str) -> None:
        self.cell = cell
        self.phase = "starting"
        self._render()

    def step(self, phase: str) -> None:
        self.phase = phase
        self._render()

    def finish(self, row: dict) -> None:
        self.done += 1
        if row.get("wall_s"):
            self.durations.append(row["wall_s"])
        if not self.bar:
            cyc = (row.get("cycles_per_image") or row.get("cycles_per_clip")
                   or row.get("cycles"))
            print(f"[{self.done:3}/{self.total}] {row['design_slug'][:34]:34} "
                  f"{row['model']:8} {row['status']:12} cycles={cyc} "
                  f"wall={row.get('wall_s')}s", flush=True)
        elif row["status"] != "ok":
            # A failure must not scroll past behind the bar.
            self._clear()
            why = "; ".join(row.get("reasons", []))[:90]
            print(f"  {row['status']:10} {row['design_slug'][:34]:34} {why}", flush=True)
        self._render()

    def _clear(self) -> None:
        if self.bar:
            sys.stdout.write("\r" + " " * 110 + "\r")

    def _render(self) -> None:
        if not self.bar:
            return
        frac = self.done / self.total if self.total else 0
        filled = int(round(frac * self.WIDTH))
        bar = "\u2588" * filled + "\u2591" * (self.WIDTH - filled)
        eta = self._eta()
        tail = f"  eta ~{self._clock(eta)}" if eta else ""
        # Truncate the whole label, not the cell name: cutting the cell alone
        # left a dangling separator ("design \u00b7  \u00b7 phase").
        label = f"{self.cell} \u00b7 {self.phase}"
        if len(label) > 44:
            label = label[:43] + "\u2026"
        sys.stdout.write(f"\r  [{bar}] {self.done:2}/{self.total}  {frac * 100:3.0f}%  "
                         f"{label:44} {self._clock(time.time() - self.t0)}{tail}   ")
        sys.stdout.flush()

    def done_all(self) -> None:
        self._clear()


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


def run_cell(design, model, out_dir, host, power, images, progress=None):
    """One (design, model) measurement."""
    def phase(name):
        if progress is not None:
            progress.step(name)

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
        phase("mesh + header")
        mesh = prepare(design, design_dir, env)
        # Cached per design, so this is free for every model after the first.
        phase("calibrating" if not (design_dir / "rates.json").exists() else "calibration cached")
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
        phase("codegen + build + simulate")
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
    phase("scoring")
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
    ap.add_argument("--progress", choices=("auto", "bar", "lines"), default="auto",
                    help="auto uses a bar on a terminal and one line per cell "
                         "otherwise (default: %(default)s)")
    args = ap.parse_args()

    # Resolved, not as given: every cell path derives from this, and a relative
    # spelling made build_mesh.py fail when it tried to report an ELF path
    # relative to the repository root.
    out = Path(args.out).resolve()
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
    total = len(designs) * len(args.models)
    prog = Progress(total, args.progress)
    n = 0
    with jsonl.open("w") as fh:
        for d in designs:
            for model in args.models:
                # The slug's trailing hash disambiguates directories, not
                # humans; drop it for the display only.
                shown = design_mod.slug(d)
                if shown != "baseline":
                    shown = shown.rsplit("-", 1)[0]
                prog.start(f"{shown} \u00b7 {Path(model).name}")
                row = run_cell(d, model, out, args.host, args.power, args.images,
                               progress=prog)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                prog.finish(row)
    prog.done_all()

    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{n} rows -> {jsonl}")
    print(f"{ok} ok, {n - ok} not, in {Progress._clock(time.time() - prog.t0)}")


if __name__ == "__main__":
    main()
