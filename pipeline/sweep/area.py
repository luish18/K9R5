#!/usr/bin/env python3
"""A first-order area model for one design point.

    python pipeline/sweep/area.py <design.json>

--- Read this before using a number out of here ----------------------------

The energy model (targets/hetero/power.py) descends from a characterized SRAM
macro that ships with GVSoC. **There is no equivalent for area.** Nothing in
the GVSoC tree, and nothing in this repository, carries a silicon area figure.
So every coefficient below is a PLACEHOLDER, marked as such in the table, and
the absolute numbers this module produces are not silicon areas.

What it does give, and what the sweep actually needs, is a *structural* model:
it counts the things that occupy area -- SRAM bits, FPU datapaths, scalar
cores, crossbar ports -- as explicit functions of the design point. Filling in
real coefficients later changes the weights, not the structure.

Two rules follow, and pipeline/sweep/report.py enforces both:

  1. Report area in the placeholder units it is computed in ("au"), never in
     mm^2, until the coefficients are sourced.
  2. Never present an area-derived ranking without the sensitivity band. A
     design that is Pareto-optimal only for one particular guess at
     SRAM_AU_PER_BIT is not a finding, and the sweep must say so.

To source them properly, the published areas for the very IP being simulated
are the right place to start -- Ara, Spatz, Snitch and CVA6 all have papers
reporting area at a stated node. Record the node with each number: the
Siracusa-derived energy coefficients are at an unrecorded node, so mixing the
two into one figure of merit is not currently defensible either.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "targets"))

# --- Coefficients -----------------------------------------------------------
#
# Every one of these is a placeholder. The `sourced` flag is what report.py
# reads to decide whether it may print an absolute figure; flip it only
# alongside a real citation in the comment.

COEFFS = {
    # Area of one bit of SRAM. Dominates the caches, the TCDM and the vector
    # register files, so it is the single most important number here.
    "SRAM_AU_PER_BIT": {"value": 1.0, "sourced": False,
                        "note": "placeholder; needs an SRAM macro area at a stated node"},
    # One fp32 FMA datapath -- the unit a vector lane is built from.
    "FP32_FMA_AU": {"value": 40000.0, "sourced": False,
                    "note": "placeholder; needs an FPnew/lane area figure"},
    # A scalar core, without its caches or vector unit.
    "SNITCH_CORE_AU": {"value": 150000.0, "sourced": False,
                       "note": "placeholder; needs the Snitch core area"},
    "CVA6_CORE_AU": {"value": 900000.0, "sourced": False,
                     "note": "placeholder; needs the CVA6 core area"},
    # Control overhead of a vector unit beyond its lanes: the sequencer, the
    # VLSU, the mask unit.
    "VECTOR_CTRL_AU": {"value": 200000.0, "sourced": False,
                       "note": "placeholder; needs an Ara/Spatz control-logic area"},
    # One master port on the TCDM crossbar. The crossbar grows faster than
    # linearly in ports, which is why nb_masters is tracked explicitly.
    "XBAR_AU_PER_PORT": {"value": 8000.0, "sourced": False,
                         "note": "placeholder; needs an interconnect area figure"},
    # One byte per cycle of AXI width.
    "AXI_AU_PER_BYTE": {"value": 5000.0, "sourced": False,
                        "note": "placeholder; needs an AXI crossbar area figure"},
}


def _c(name):
    return COEFFS[name]["value"]


def sram_au(size_bytes: int) -> float:
    """Area of an SRAM of `size_bytes`, counted in bits."""
    return size_bytes * 8 * _c("SRAM_AU_PER_BIT")


def vector_unit_au(vlen_bits: int, nb_lanes: int, lane_width_bytes: int,
                   nb_vregs: int = 32) -> float:
    """A vector unit: its register file, its lanes, and its control.

    The register file is the part that scales with VLEN, and it is usually the
    larger half -- which is the whole reason VLEN is worth sweeping separately
    from lane count rather than treating "wider" as one knob.
    """
    regfile = vlen_bits * nb_vregs * _c("SRAM_AU_PER_BIT")
    # A lane is lane_width bytes of datapath, i.e. that many fp32 FMAs wide.
    fmas = nb_lanes * max(1, lane_width_bytes // 4)
    return regfile + fmas * _c("FP32_FMA_AU") + _c("VECTOR_CTRL_AU")


def breakdown(design: dict) -> dict:
    """Area of every block of the SoC this design describes, in placeholder units."""
    import importlib
    import os

    # The design reaches system.py/memsys.py the same way it reaches the
    # simulator: through the environment. Re-import so repeated calls in one
    # process see their own design rather than the first one's.
    os.environ["HES_DESIGN"] = design["_path"]
    for mod in ("hetero.design", "hetero.memsys", "hetero.system"):
        if mod in sys.modules:
            del sys.modules[mod]
    system = importlib.import_module("hetero.system")
    memsys = importlib.import_module("hetero.memsys")

    out = {}

    # Host caches.
    for name, geom in (("icache", memsys.ICACHE), ("dcache", memsys.DCACHE),
                       ("l2", memsys.L2)):
        out[name] = sram_au(geom["size"])

    # The orchestrator, and its vector unit if the board carries one.
    out["cva6"] = _c("CVA6_CORE_AU")
    out["ara"] = vector_unit_au(system.HOST_VLEN, system.HOST_NB_LANES,
                                system.HOST_LANE_WIDTH)

    # The clusters.
    for cluster in system.CLUSTERS:
        tag = cluster.name
        out[f"{tag}_tcdm"] = sram_au(system.TCDM_SIZE)
        out[f"{tag}_cores"] = cluster.nb_core * _c("SNITCH_CORE_AU")
        if cluster.use_spatz:
            out[f"{tag}_vector"] = cluster.nb_core * vector_unit_au(
                cluster.spatz_vlen, cluster.spatz_nb_lanes,
                cluster.spatz_lane_width)
        # The TCDM crossbar. nb_masters is one port per core plus, on a Spatz
        # cluster, one per lane per core -- which is why lane count shows up in
        # the interconnect and not only in the datapath.
        nb_masters = cluster.nb_core
        if cluster.use_spatz:
            nb_masters += cluster.nb_core * cluster.spatz_nb_lanes
        out[f"{tag}_xbar"] = nb_masters * _c("XBAR_AU_PER_PORT")

    out["axi"] = (system.NARROW_AXI_WIDTH + system.WIDE_AXI_WIDTH) * _c("AXI_AU_PER_BYTE")

    return out


def total(design: dict) -> float:
    return sum(breakdown(design).values())


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", help = "a design JSON (may be {} for the defaults)")
    ap.add_argument("--json", action = "store_true", help = "emit machine-readable output")
    args = ap.parse_args()

    design = {"_path": str(Path(args.design).resolve())}
    parts = breakdown(design)
    tot = sum(parts.values())

    if args.json:
        print(json.dumps({"total_au": tot, "breakdown": parts,
                          "coefficients_sourced": all(c["sourced"] for c in COEFFS.values())},
                         indent = 2, sort_keys = True))
        return

    print("area breakdown (PLACEHOLDER UNITS -- see the module docstring)\n")
    for name, value in sorted(parts.items(), key = lambda kv: -kv[1]):
        print(f"  {name:16} {value:14,.0f} au  {100 * value / tot:5.1f}%")
    print(f"  {'TOTAL':16} {tot:14,.0f} au")
    unsourced = [k for k, c in COEFFS.items() if not c["sourced"]]
    if unsourced:
        print(f"\n  {len(unsourced)} of {len(COEFFS)} coefficients are placeholders: "
              f"{', '.join(unsourced)}")
        print("  These are NOT silicon areas. Ranking designs on them requires the "
              "sensitivity pass.")


if __name__ == "__main__":
    main()
