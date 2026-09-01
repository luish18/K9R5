#!/usr/bin/env python3
"""Run an ONNX graph on the hetero_soc board: one chip, three kinds of core.

  python pipeline/run_hetero.py Tests/Kernels/FP32/GEMM/Regular
  python pipeline/run_hetero.py ops/mnist --pin snitch

Deeploy compiles the graph and decides which core runs each node
(pipeline/hetero_platform); this drives that, builds the three binaries a run
needs, starts the simulation, and reports.

Unlike pipeline/run.py -- which measures three separate boards and can wait for
each to finish -- a whole network on one board runs for minutes, so the
simulator's output is streamed rather than captured. Every node the guest
completes prints a beacon, and this turns those into a live progress line:

  [ 5/11] Conv@snitch   sim 4.21 Mcyc  wall 108s  39 kcyc/s

If no beacon arrives for --stall-timeout seconds the run is declared stalled
and the last output is printed, so a wedged simulation is visible instead of
being mistaken for a slow one.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mesh import HOSTS, build_network  # noqa: E402
from common import GVSOC, PYTHON, RESULTS, ROOT, TARGETS, WORK, note_file, set_debug  # noqa: E402

DEEPLOY_TEST = ROOT / "deps" / "deeploy" / "DeeployTest"

PROG_RE = re.compile(r"\[HES-PROG\] sample=(\d+)/(\d+) node=(\d+) op=(\S+) "
                     r"engine_id=(\d+) cycles=(\d+)")

# Engine ids as hetero_platform/progress.py emits them into the generated code.
ENGINE_NAMES = {0: "cva6", 1: "snitch", 2: "spatz"}
WAIT_RE = re.compile(r"\[HES-WAIT\] engine=(\S+) job=(\d+) kernel=(\d+) polls=(\d+)")
DONE_RE = re.compile(r"\[HES\] core=(\S+) cycles=(\d+) instret=(\d+) errors=(\d+) "
                     r"total=(\d+) maxdiff_e6=(\d+) offload_failures=(\d+)")
MNIST_RE = re.compile(r"\[HES-MNIST\] images=(\d+) correct=(\d+) agree_with_onnx=(\d+) "
                      r"cycles_total=(\d+) cycles_per_image=(\d+) offload_failures=(\d+)")
KWS_RE = re.compile(
    r"\[HES-KWS\] clips=(\d+) correct=(\d+) agree_with_onnx=(\d+) "
    r"mfcc_maxdiff_e6=(\d+) cycles_total=(\d+) cycles_per_clip=(\d+) "
    r"frontend_engine=(\d+) frontend_busy=(\d+) frontend_wait=(\d+) hidden=(\d+) "
    r"snitch_busy=(\d+) spatz_busy=(\d+) pipelined=(\d+) offload_failures=(\d+)")
MEM_RE = re.compile(r"\[HES-MEM\] cache=(\S+) accesses=(\d+) reads=(\d+) writes=(\d+) "
                    r"hits=(\d+) misses=(\d+) latency_cycles=(\d+)")


# An op that ships its own evaluation set brings its own host program too: one
# that loops over the set and scores it, instead of running a single inference
# and diffing it. An op is recognised by the data header it carries.
#
#   header          the generated header that identifies the application
#   main            the host program under runtime/mesh
#   beacon          the regex for the single metrics line it prints
#   samples_flag    what --images caps, for the message
APPS = {
    "mnist": {"header": "mnist_data.h", "main": "mnist_main.c", "unit": "image"},
    "kws": {"header": "kws_data.h", "main": "kws_main.c", "unit": "clip"},
}


def detect_app(test_dir: Path):
    """Which application this op directory is, if any."""
    for name, app in APPS.items():
        if (test_dir / app["header"]).is_file():
            return name, app
    return None, None


def resolve_op(op: str) -> Path:
    for cand in (Path(op), DEEPLOY_TEST / op, DEEPLOY_TEST / "Tests" / op):
        if (cand / "network.onnx").is_file():
            return cand.resolve()
    sys.exit(f"error: cannot find network.onnx under '{op}'")


def generate(test_dir: Path, gen_dir: Path, pin, debug) -> dict:
    cmd = [str(PYTHON), str(ROOT / "pipeline" / "hetero_platform" / "generate.py"),
           "-t", str(test_dir), "-d", str(gen_dir)]
    if pin:
        cmd += ["--pin", pin]
    r = subprocess.run(cmd, capture_output = not debug, text = True)
    if r.returncode != 0:
        sys.exit(f"Deeploy codegen failed:\n{r.stdout or ''}\n{r.stderr or ''}")
    if not debug and r.stdout:
        for line in r.stdout.splitlines():
            if line.startswith("  "):
                print(line)
    return json.loads((gen_dir / "mapping.json").read_text())


class Progress:
    """One updating status line, driven by the guest's own beacons."""

    def __init__(self, total_nodes: int, quiet: bool):
        self.total = total_nodes
        self.quiet = quiet
        self.t0 = time.time()
        self.last_beacon = self.t0
        self.node = 0
        self.sample = (0, 0)
        self.sim_cycles = 0
        self.per_engine = {}
        self.per_node = []
        self.waits = []

    def _render(self, label: str) -> None:
        if self.quiet:
            return
        wall = time.time() - self.t0
        rate = self.sim_cycles / wall / 1000 if wall > 0 else 0
        pos = f"[{self.node:2}/{self.total:2}]" if self.total else f"[{self.node:2}]"
        if self.sample[1] > 1:
            pos = f"[{self.sample[0]}/{self.sample[1]}]" + pos
        sys.stdout.write(f"\r  {pos} {label:22} sim {self.sim_cycles / 1e6:7.2f} Mcyc  "
                         f"wall {wall:5.0f}s  {rate:6.1f} kcyc/s   ")
        sys.stdout.flush()

    def beacon(self, m) -> None:
        sample, samples, node, op, engine_id, cycles = m.groups()
        engine = ENGINE_NAMES.get(int(engine_id), f"engine{engine_id}")
        self.last_beacon = time.time()
        self.sample = (int(sample), int(samples))
        self.node = int(node) + 1
        self.sim_cycles += int(cycles)
        self.per_engine[engine] = self.per_engine.get(engine, 0) + int(cycles)
        self.per_node.append({"node": int(node), "op": op, "engine": engine,
                              "cycles": int(cycles)})
        self._render(f"{op}@{engine}")

    def wait(self, m) -> None:
        engine, job, kernel, polls = m.groups()
        self.last_beacon = time.time()
        self.waits.append({"engine": engine, "job": int(job), "polls": int(polls)})
        self._render(f"waiting on {engine}")

    def finish(self) -> None:
        if not self.quiet:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()


def simulate(elfs: dict, run_dir: Path, total_nodes: int, timeout_s: int,
             stall_s: int, quiet: bool, target: str = "hetero_soc") -> dict:
    run_dir.mkdir(parents = True, exist_ok = True)
    env = dict(os.environ)
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env['PATH']}"
    env["HES_ELF_SNITCH"] = str(elfs["snitch"])
    env["HES_ELF_SPATZ"] = str(elfs["spatz"])

    cmd = [str(GVSOC), f"--target-dir={TARGETS}", f"--target={target}",
           f"--binary={elfs['host']}", "run"]

    proc = subprocess.Popen(cmd, cwd = run_dir, env = env, text = True,
                            stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
                            bufsize = 1)

    prog = Progress(total_nodes, quiet)
    lines, result, caches = [], None, []
    mnist_result = None
    kws_result = None
    stalled = False
    t_start = time.time()

    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("WARNING"):
                continue
            lines.append(line)

            m = PROG_RE.search(line)
            if m:
                prog.beacon(m)
                continue
            m = WAIT_RE.search(line)
            if m:
                prog.wait(m)
                continue
            m = DONE_RE.search(line)
            if m:
                result = m
                continue
            m = MNIST_RE.search(line)
            if m:
                mnist_result = m
                continue
            m = KWS_RE.search(line)
            if m:
                kws_result = m
                continue
            m = MEM_RE.search(line)
            if m:
                caches.append(m)
                continue
            if line.strip():
                if not quiet:
                    sys.stdout.write("\r" + " " * 100 + "\r")
                print(line)

            if time.time() - prog.last_beacon > stall_s:
                stalled = True
                break
            if time.time() - t_start > timeout_s:
                break
    finally:
        prog.finish()
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    (run_dir / "sim.log").write_text("\n".join(lines) + "\n")
    note_file(run_dir / "sim.log", "simulator stdout+stderr")

    out = {
        "status": "ok",
        "wall_s": round(time.time() - t_start, 1),
        "nodes": prog.per_node,
        "per_engine_cycles": prog.per_engine,
        "waits": prog.waits,
    }
    if stalled:
        out["status"] = "stalled"
        out["log_tail"] = lines[-10:]
    elif result is None:
        out["status"] = "no-metrics"
        out["log_tail"] = lines[-10:]
    else:
        out.update({
            "cycles": int(result.group(2)),
            "errors": int(result.group(4)),
            "outputs": int(result.group(5)),
            "maxdiff": int(result.group(6)) / 1e6,
            "offload_failures": int(result.group(7)),
        })
        if out["errors"] or out["offload_failures"]:
            out["status"] = "wrong-result"
    if mnist_result is not None:
        images, correct, agree, total, per_image, fails = (
            int(g) for g in mnist_result.groups())
        out.update({
            "images": images,
            "correct": correct,
            "agree_with_onnx": agree,
            "cycles": total,
            "cycles_per_image": per_image,
            "accuracy": round(correct / images, 4) if images else None,
            "offload_failures": fails,
        })
        # Disagreeing with the reference model is the failure; a mispredicted
        # digit is not.
        out["status"] = "ok" if (agree == images and fails == 0) else "wrong-result"
    if kws_result is not None:
        (clips, correct, agree, maxdiff_e6, total, per_clip, fe_engine, fe_busy,
         fe_wait, hidden, snitch_busy, spatz_busy, pipelined, fails) = (
            int(g) for g in kws_result.groups())
        out.update({
            "clips": clips,
            "correct": correct,
            "agree_with_onnx": agree,
            "mfcc_maxdiff": maxdiff_e6 / 1e6,
            "cycles": total,
            "cycles_per_clip": per_clip,
            "accuracy": round(correct / clips, 4) if clips else None,
            "frontend_engine": ENGINE_NAMES.get(fe_engine, str(fe_engine)),
            "frontend_busy": fe_busy,
            "frontend_wait": fe_wait,
            # Front-end cycles that cost the host nothing because they ran while
            # the classifier did. Zero by construction without --serial.
            "hidden_cycles": hidden,
            "cluster_busy": {"snitch": snitch_busy, "spatz": spatz_busy},
            "pipelined": bool(pipelined),
            "offload_failures": fails,
        })
        # Three ways this run can be wrong, and a misheard keyword is none of
        # them: disagreeing with onnxruntime, a front-end that drifted from the
        # numpy reference, or a failed offload.
        out["status"] = ("ok" if (agree == clips and fails == 0
                                  and out["mfcc_maxdiff"] <= 1e-3)
                         else "wrong-result")
    if caches:
        out["caches"] = [{
            "cache": m.group(1).split("/")[-1],
            "accesses": int(m.group(2)),
            "misses": int(m.group(6)),
            "latency_cycles": int(m.group(7)),
        } for m in caches]
    return out


def report(op_name: str, mapping: dict, res: dict) -> None:
    print(f"\n=== {op_name} on hetero_soc ===\n")

    print(f"{'node':>4}  {'op':16} {'engine':8} {'cycles':>12}")
    print("-" * 44)
    for n in res["nodes"]:
        print(f"{n['node']:>4}  {n['op']:16} {n['engine']:8} {n['cycles']:>12}")

    if res["per_engine_cycles"]:
        total = sum(res["per_engine_cycles"].values())
        print(f"\n{'engine':8} {'cycles':>12} {'share':>8}")
        print("-" * 30)
        for engine, cycles in sorted(res["per_engine_cycles"].items(),
                                     key = lambda kv: -kv[1]):
            share = 100 * cycles / total if total else 0
            print(f"{engine:8} {cycles:>12} {share:>7.1f}%")

    print()
    if "images" in res:
        print(f"images   {res['images']}   correct {res['correct']} "
              f"({100 * res['accuracy']:.1f}%)   agreeing with onnxruntime "
              f"{res['agree_with_onnx']}/{res['images']}")
        print(f"cycles   {res['cycles']} total, {res['cycles_per_image']} per image")
    if "clips" in res:
        print(f"clips    {res['clips']}   correct {res['correct']} "
              f"({100 * res['accuracy']:.1f}%)   agreeing with onnxruntime "
              f"{res['agree_with_onnx']}/{res['clips']}")
        print(f"cycles   {res['cycles']} total, {res['cycles_per_clip']} per clip")
        print(f"mfcc     max |chip - numpy| = {res['mfcc_maxdiff']:.2e}")
        busy = res["cluster_busy"]
        print(f"\nfront-end on {res['frontend_engine']}, "
              f"{'pipelined' if res['pipelined'] else 'serial'}")
        # The table above counts only graph nodes, and the MFCC front-end is
        # not one -- it is dispatched by the host program, so it emits no node
        # beacon and its cluster shows up there with nothing to do. This is the
        # whole application: the clusters as they counted themselves, the host
        # from its own node beacons.
        whole = dict(busy)
        whole["cva6"] = res["per_engine_cycles"].get("cva6", 0)
        grand = sum(whole.values())
        print(f"\n{'engine':8} {'cycles':>12} {'share':>8}   whole application")
        print("-" * 48)
        for name, cycles in sorted(whole.items(), key = lambda kv: -kv[1]):
            share = 100 * cycles / grand if grand else 0
            note = "  <- front-end" if name == res["frontend_engine"] else ""
            print(f"{name:8} {cycles:>12} {share:>7.1f}%{note}")
        # The number the whole application exists to produce: front-end cycles
        # that the classifier's own runtime paid for.
        pct = 100 * res["hidden_cycles"] / res["frontend_busy"] if res["frontend_busy"] else 0
        print(f"\nfront-end {res['frontend_busy']} cycles, of which "
              f"{res['hidden_cycles']} ({pct:.1f}%) overlapped the classifier "
              f"and cost no wall time")
    if res["status"] == "ok" and "images" not in res and "clips" not in res:
        print(f"status   ok       cycles={res['cycles']}  maxdiff={res.get('maxdiff')}")
    elif res["status"] == "ok":
        print("\nstatus   ok")
    elif res["status"] == "stalled":
        print("status   STALLED  no progress beacon before the stall timeout")
        for line in res.get("log_tail", []):
            print(f"  | {line}")
    else:
        print(f"status   {res['status']}")
        for line in res.get("log_tail", []):
            print(f"  | {line}")
    print(f"wall     {res['wall_s']}s"
          + (f"   (pinned to {mapping['pin']})" if mapping.get("pin") else ""))

    RESULTS.mkdir(exist_ok = True)
    suffix = f"-pin-{mapping['pin']}" if mapping.get("pin") else ""
    suffix += "" if mapping.get("host", "cva6") == "cva6" else f"-{mapping['host']}"
    if mapping.get("frontend") and mapping["frontend"] != "snitch":
        suffix += f"-fe-{mapping['frontend']}"
    if mapping.get("serial"):
        suffix += "-serial"
    out = RESULTS / f"{op_name.replace('/', '_')}-hetero{suffix}.json"
    out.write_text(json.dumps({"op": op_name, "mapping": mapping, "result": res},
                              indent = 2))
    print(f"\nresults written to {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", help = "op dir (network.onnx + inputs.npz + outputs.npz)")
    ap.add_argument("--pin", choices = ["cva6", "snitch", "spatz"],
                    help = "force every node the engine can run onto it")
    ap.add_argument("--host", choices = list(HOSTS), default = "cva6",
                    help = "orchestrator: the scalar CVA6, or the same core with "
                           "an Ara vector unit (default: %(default)s)")
    ap.add_argument("--images", type = int, default = 1000000,
                    help = "cap the samples an evaluation-set op classifies")
    ap.add_argument("--frontend", choices = ["snitch", "spatz"], default = "snitch",
                    help = "cluster that runs the KWS MFCC front-end; the "
                           "classifier's nodes go wherever the mapper put them "
                           "(default: %(default)s)")
    ap.add_argument("--serial", action = "store_true",
                    help = "KWS: run the front-end after the classifier instead "
                           "of overlapping it -- the same work with the two "
                           "clusters taking turns, which is the baseline the "
                           "pipelined run is measured against")
    ap.add_argument("--timeout", type = int, default = 3600, help = "wall limit [s]")
    ap.add_argument("--stall-timeout", type = int, default = 180,
                    help = "declare a stall after this long with no beacon [s]")
    ap.add_argument("-q", "--quiet", action = "store_true",
                    help = "no live progress line")
    ap.add_argument("-d", "--debug", action = "store_true")
    args = ap.parse_args()

    set_debug(args.debug or os.environ.get("HES_DEBUG", "") not in ("", "0"))

    test_dir = resolve_op(args.op)
    try:
        op_name = str(test_dir.relative_to((DEEPLOY_TEST / "Tests").resolve())).replace("/", "_")
    except ValueError:
        op_name = test_dir.name

    variant = f"{args.host}-{args.pin or 'mapped'}"
    if (test_dir / "kws_data.h").is_file():
        variant += f"-fe{args.frontend}" + ("-serial" if args.serial else "")
    work = WORK / f"hetero_{op_name}" / variant
    gen_dir = work / "gen"

    print(f"[1/3] Deeploy: {test_dir.name}/network.onnx -> C, mapped across engines")
    mapping = generate(test_dir, gen_dir, args.pin, args.debug)
    mapping["host"] = args.host

    print(f"[2/3] build: {args.host} host + snitch cluster + spatz cluster")
    app_name, app = detect_app(test_dir)
    defines = []
    if app_name == "kws":
        # The front-end and the classifier have to be on *different* clusters.
        # A cluster's mailbox holds one job descriptor, so posting the
        # front-end for clip n+1 to the same cluster the classifier for clip n
        # is using would overwrite a job in flight -- which the host runtime
        # refuses rather than corrupts, turning the run into a wall of failed
        # offloads. Catch it here instead, where it can be explained.
        clash = [n["node"] for n in mapping["nodes"]
                 if n["engine"] == args.frontend]
        if clash:
            sys.exit(
                f"error: the front-end is on {args.frontend}, but the mapper "
                f"also put {len(clash)} classifier node(s) there "
                f"({', '.join(clash[:3])}{'...' if len(clash) > 3 else ''}).\n"
                f"       The two stages have to run on different clusters -- "
                f"that is what makes them overlap.\n"
                f"       Pin the classifier to the other one, e.g. "
                f"--frontend {args.frontend} --pin "
                f"{'snitch' if args.frontend == 'spatz' else 'spatz'}")
        engine_id = {v: k for k, v in ENGINE_NAMES.items()}[args.frontend]
        defines = [f"-DKWS_FRONTEND_ENGINE={engine_id}",
                   f"-DKWS_PIPELINED={0 if args.serial else 1}"]
        mapping["frontend"] = args.frontend
        mapping["serial"] = args.serial
    elfs = build_network(
        gen_dir, work,
        host_main = (ROOT / "runtime" / "mesh" / app["main"]) if app else None,
        extra_incs = [test_dir] if app else (),
        samples = args.images if app else 1,
        host = args.host,
        extra_defines = defines)

    print(f"[3/3] simulate on {HOSTS[args.host][1]}")
    res = simulate(elfs, work / "run", len(mapping["nodes"]), args.timeout,
                   args.stall_timeout, args.quiet, target = HOSTS[args.host][1])

    report(op_name, mapping, res)
    sys.exit(0 if res["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
