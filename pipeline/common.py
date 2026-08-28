#!/usr/bin/env python3
"""Helpers shared by the pipeline drivers: paths, the debug trace, and the
command runner that reports what each command produced.

Split out of run.py unchanged so the per-core benchmark and the hetero_soc
driver trace their runs the same way.
"""

import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEEPLOY = ROOT / "deps" / "deeploy"
DEEPLOY_TEST = DEEPLOY / "DeeployTest"
GENERIC_LIB = DEEPLOY / "TargetLibraries" / "Generic"
GVSOC = ROOT / "deps" / "gvsoc" / "install" / "bin" / "gvsoc"
TC = ROOT / "toolchains" / "xpack-riscv-none-elf-gcc-15.2.0-1" / "bin" / "riscv-none-elf-gcc"
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNTIME = ROOT / "runtime"
TARGETS = ROOT / "targets"
WORK = ROOT / "work"
RESULTS = ROOT / "results"


DEBUG = False


def rel(p) -> str:
    """Path relative to the repo root when it is inside it, else absolute."""
    p = Path(p)
    for cand in (p, p.resolve()):
        try:
            return str(cand.relative_to(ROOT))
        except ValueError:
            pass
    return str(p)


def human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def dbg(msg: str) -> None:
    """One line of --debug trace, on stderr so stdout stays the report."""
    if DEBUG:
        print(f"[dbg] {msg}", file=sys.stderr, flush=True)


def note_file(path: Path, why: str) -> None:
    """Trace a file written by the driver itself rather than by a command."""
    if DEBUG:
        size = human_size(path.stat().st_size) if path.is_file() else "not created"
        dbg(f"  {rel(path)}  ({size}, {why})")


def snapshot(paths) -> dict:
    """Files under each directory in `paths`, with mtimes, before a command runs."""
    seen = {}
    for p in paths:
        p = Path(p)
        seen[p] = ({f: f.stat().st_mtime_ns for f in p.rglob("*") if f.is_file()}
                   if p.is_dir() else {})
    return seen


def report_produced(produces, before: dict) -> None:
    """Trace what a command generated: named files, plus files new or rewritten
    in the directories it writes into (so a re-run still lists its outputs)."""
    if not DEBUG:
        return
    for p in produces:
        p = Path(p)
        if p.is_dir():
            old = before.get(p, {})
            touched = sorted(f for f in p.rglob("*")
                             if f.is_file() and f.stat().st_mtime_ns != old.get(f))
            for f in touched:
                dbg(f"  -> {rel(f)}  ({human_size(f.stat().st_size)})")
            if not touched:
                dbg(f"  -> {rel(p)}/  (nothing written)")
        elif p.is_file():
            dbg(f"  -> {rel(p)}  ({human_size(p.stat().st_size)})")
        else:
            dbg(f"  -> {rel(p)}  (not created)")


def sh(cmd, cwd=None, timeout=None, env=None, produces=()):
    """Run a command, capturing its output.

    `produces` names the files (or directories) the command is expected to
    generate; with --debug the command line and those files are traced.
    """
    before = {}
    if DEBUG:
        before = snapshot(produces)
        dbg(f"$ {shlex.join(str(c) for c in cmd)}")
        if cwd:
            dbg(f"  (cwd: {rel(cwd)})")

    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, timeout=timeout, env=env,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        dbg(f"  timed out after {timeout}s")
        report_produced(produces, before)
        raise

    dbg(f"  exit={r.returncode} in {time.time() - t0:.1f}s")
    report_produced(produces, before)
    return r



def set_debug(on: bool) -> None:
    """Turn the command trace on or off for this process."""
    global DEBUG
    DEBUG = on
