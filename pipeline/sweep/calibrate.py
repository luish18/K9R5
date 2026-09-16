#!/usr/bin/env python3
"""Turn one mesh_calib run into the engine cost table for that design point.

pipeline/hetero_platform/mapper.py prices a node as

    cost = OFFLOAD_FIXED[e] + OFFLOAD_PER_BYTE[e] * bytes + macs / RATES[e][op]

and its committed tables describe one specific SoC. A sweep changes the very
things they measure, so every design point re-derives its own table from
runtime/tests/mesh_calib.c -- one build and one simulation -- and the mapper
loads it instead. Without this a 16-lane Spatz would keep being handed work at
a 4-lane Spatz's prices, and the sweep would be measuring the mapper's staleness
rather than the hardware.

Nothing is fitted in the bare-metal program: it prints the raw cycle counts and
the arithmetic happens here, where it can be read and checked.

    python sweep/calibrate.py <mesh_calib output> -o rates.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"\[HES-CAL\] engine=(?P<engine>\S+) op=(?P<op>\S+) macs=(?P<macs>\d+) "
    r"bytes=(?P<bytes>\d+) staged=(?P<staged>\d+) cycles=(?P<cycles>\d+) "
    r"host_cycles=(?P<host>\d+) pass=(?P<pass>\d+) ok=(?P<ok>\d+)")

FAILURES = re.compile(r"\[HES-CAL\] failures=(\d+)")

# The op names mapper.py keys on. MatMul_tiny is the offload intercept probe,
# not an operator the mapper ever prices.
OPS = ("Gemm", "MatMul", "Conv")


def parse(text):
    """Every [HES-CAL] measurement in `text`, plus the failure count."""
    rows = [m.groupdict() for m in LINE.finditer(text)]
    for r in rows:
        for k in ("macs", "bytes", "staged", "cycles", "host", "pass", "ok"):
            r[k] = int(r[k])
    fail = FAILURES.search(text)
    if fail is None:
        raise SystemExit("error: no [HES-CAL] failures= line -- the calibration "
                         "run did not finish (check for a hang or a trap)")
    return rows, int(fail.group(1))


def build_table(rows, failures, host = "cva6"):
    """The RATES / OFFLOAD_FIXED / OFFLOAD_PER_BYTE tables these rows imply."""
    if failures:
        raise SystemExit(
            f"error: calibration reported {failures} failures -- this design "
            "computes wrong results, so any rate read off it would be the rate "
            "of a wrong answer. The design point is infeasible, not slow.")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        raise SystemExit("error: calibration measurements marked not-ok: "
                         + ", ".join(f"{r['engine']}/{r['op']}" for r in bad))

    engines = sorted({r["engine"] for r in rows})
    rates, fixed, per_byte = {}, {}, {}

    for engine in engines:
        mine = [r for r in rows if r["engine"] == engine]

        # Rates come from the warm pass where there is one. A network runs each
        # kind of node many times, so the warm cost is the one that dominates;
        # the cold pass is what prices the first offload below. The host has no
        # warm/cold distinction (it never stages), so it falls back to pass 0.
        def pick(op):
            warm = [r for r in mine if r["op"] == op and r["pass"] == 1]
            cold = [r for r in mine if r["op"] == op and r["pass"] == 0]
            chosen = warm or cold
            return chosen[-1] if chosen else None

        row_for = {op: pick(op) for op in OPS}
        if any(v is None for v in row_for.values()):
            missing = [op for op, v in row_for.items() if v is None]
            raise SystemExit(f"error: engine {engine} has no measurement for "
                             f"{', '.join(missing)}")

        rates[engine] = {op: r["macs"] / r["cycles"] for op, r in row_for.items()}
        # mapper.py falls back to _default for every operator it has no row for.
        # Conv is the most memory-bound of the three and so the most pessimistic
        # stand-in, which is the safe direction: it keeps the mapper from
        # shipping an unmeasured operator to a cluster on optimism alone. This
        # mirrors the committed table, where _default equals the Conv rate.
        rates[engine]["_default"] = rates[engine]["Conv"]

        # The offload overhead is what the host waited beyond what the engine
        # spent computing. The host engine offloads nothing, so both are zero.
        if engine == "cva6":
            fixed[engine], per_byte[engine] = 0, 0.0
            continue

        tiny = [r for r in mine if r["op"] == "MatMul_tiny"]
        if not tiny:
            raise SystemExit(f"error: engine {engine} has no MatMul_tiny probe, "
                             "so the fixed offload cost cannot be separated "
                             "from the per-byte one")

        # OFFLOAD_FIXED is what shipping a node costs when there is essentially
        # no arithmetic to do: the mailbox write, the doorbell, waking the
        # control core, the completion handshake. That is the tiny probe's whole
        # round trip, less the little arithmetic it does carry -- not merely the
        # host-side residual over the cluster's own count, which would be a few
        # hundred cycles and would have the mapper shipping ten-MAC nodes to a
        # cluster. This term is the one that matters: it is what keeps a small
        # node on the host.
        t = tiny[-1]
        tiny_compute = t["macs"] / rates[engine]["MatMul"]
        fixed[engine] = max(0, int(round(t["host"] - tiny_compute)))

        # OFFLOAD_PER_BYTE is the part of the cost that scales with how much
        # data has to reach the cluster. The cold pass pays for pulling the
        # operands in; the warm pass does not. Their difference over the same
        # working set is therefore the data-movement cost per byte, and it is
        # measured rather than inferred from a two-point fit against rates that
        # already absorb staging.
        slopes = []
        for op in OPS + ("MatMul_tiny",):
            cold = [r for r in mine if r["op"] == op and r["pass"] == 0]
            warm = [r for r in mine if r["op"] == op and r["pass"] == 1]
            if cold and warm and cold[-1]["bytes"] > 0:
                delta = cold[-1]["cycles"] - warm[-1]["cycles"]
                if delta > 0:
                    slopes.append(delta / cold[-1]["bytes"])
        # Median, so one operator with an unusual access pattern (Conv gathers
        # its window and refills far more than it stages) does not set the rate
        # for all of them.
        slopes.sort()
        per_byte[engine] = slopes[len(slopes) // 2] if slopes else 0.0

    # The probe names the orchestrator "cva6" on both boards, because that is
    # what hes_engine_name() returns. But mapper.py prices the host by which
    # board it is -- RATES[self.host], "cva6" or "ara" -- so a calibration run
    # on the vector host has to be filed under "ara" or the mapper would keep
    # using the committed scalar row and never see what Ara actually does here.
    if host != "cva6":
        for table in (rates, fixed, per_byte):
            if "cva6" in table:
                table[host] = table.pop("cva6")

    return {"RATES": rates, "OFFLOAD_FIXED": fixed, "OFFLOAD_PER_BYTE": per_byte}


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs = "?", help = "mesh_calib output (default: stdin)")
    ap.add_argument("-o", "--out", help = "write the table here as JSON")
    ap.add_argument("--host", default = "cva6", choices = ("cva6", "ara"),
                    help = "which orchestrator this calibration ran on; the host "
                           "row is filed under this name because mapper.py prices "
                           "the host by which board it is (default: %(default)s)")
    args = ap.parse_args()

    text = Path(args.log).read_text() if args.log else sys.stdin.read()
    rows, failures = parse(text)
    table = build_table(rows, failures, host = args.host)

    blob = json.dumps(table, indent = 2, sort_keys = True)
    if args.out:
        Path(args.out).write_text(blob + "\n")
        print(f"wrote {args.out}")
    else:
        print(blob)


if __name__ == "__main__":
    main()
