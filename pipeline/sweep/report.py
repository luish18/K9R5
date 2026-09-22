#!/usr/bin/env python3
"""Read a sweep's JSONL and say what it found.

    python pipeline/sweep/report.py results/sweep/s1/sweep.jsonl

Three sections, in the order they should be trusted:

  1. Sensitivity -- which knobs moved cycles at all, against the baseline.
     This is the screening result, and it rests on nothing but measured cycles.
  2. Pareto front over (cycles, area, energy).
  3. Robustness -- how much of that front survives the area coefficients being
     wrong, which they are.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# How far an area coefficient is assumed to be able to move. The coefficients
# in area.py are placeholders, so this is not a confidence interval -- it is a
# test of whether a conclusion depends on them at all.
PERTURB = 0.30


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def cycles_of(row):
    for k in ("cycles_per_image", "cycles_per_clip", "cycles"):
        if row.get(k):
            return row[k]
    return None


def sensitivity(rows, model):
    """Per-knob effect on cycles, against the baseline of the same model.

    Returns the baseline cycles, the per-knob effects, and the baseline design
    itself -- the last so the table can show what each knob was moved *from*.
    A row reading "TCDM_SIZE 65536 +222%" does not say whether that is half the
    baseline or twice it.
    """
    mine = [r for r in rows if r["model"] == model and r["status"] == "ok"]
    base = next((r for r in mine if r["design_slug"] == "baseline"), None)
    if base is None:
        return None, [], {}
    b = cycles_of(base)

    effects = defaultdict(list)
    for r in mine:
        if r["design_slug"] == "baseline":
            continue
        diff = {k: v for k, v in r["design"].items()
                if v != base["design"].get(k)}
        if len(diff) != 1:
            continue                      # factorial point, not an OFAT one
        knob, value = next(iter(diff.items()))
        c = cycles_of(r)
        effects[knob].append((value, c, 100.0 * (c - b) / b))
    return (b, sorted(effects.items(), key=lambda kv: -max(abs(p[2]) for p in kv[1])),
            base["design"])


def pareto(points):
    """Non-dominated points. Each is (key, [objectives]), all minimised."""
    out = []
    for key, obj in points:
        dominated = any(
            all(o2 <= o1 for o1, o2 in zip(obj, other))
            and any(o2 < o1 for o1, o2 in zip(obj, other))
            for k2, other in points if k2 != key)
        if not dominated:
            out.append((key, obj))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    args = ap.parse_args()
    rows = load(args.jsonl)

    ok = [r for r in rows if r["status"] == "ok"]
    other = [r for r in rows if r["status"] != "ok"]
    print(f"{len(rows)} cells: {len(ok)} ok, {len(other)} not\n")
    for r in other:
        why = "; ".join(r.get("reasons", []))[:120]
        print(f"  {r['status']:10} {r['design_slug'][:36]:36} {r['model']:8} {why}")
    if other:
        print()

    for model in sorted({r["model"] for r in ok}):
        base, eff, base_design = sensitivity(rows, model)
        if base is None:
            continue
        print(f"=== {model}: sensitivity (baseline {base:,} cycles) ===\n")
        # Base cycles repeat down the column, but keeping them on the row makes
        # a line self-contained: grepped, pasted or compared across models, it
        # still says what it was measured against.
        print(f"  {'knob':20} {'base':>10} {'value':>10} "
              f"{'base cycles':>13} {'cycles':>12} {'vs base':>9}")
        for knob, pts in eff:
            was = base_design.get(knob, "?")
            for value, c, pct in sorted(pts):
                print(f"  {knob:20} {was:>10} {value:>10} "
                      f"{base:>13,} {c:>12,} {pct:>+8.1f}%")
        flat = [k for k, pts in eff if all(abs(p[2]) < 0.05 for p in pts)]
        if flat:
            print(f"\n  No measurable effect: {', '.join(flat)}")
            print("  (A host vector knob does nothing on the scalar host -- those "
                  "need --host ara.)")
        print()

    # --- Pareto -------------------------------------------------------------
    have_area = [r for r in ok if r.get("area_au")]
    have_energy = [r for r in have_area if r.get("cache_dynamic_pj")]
    if not have_area:
        print("No area figures in these rows; skipping the Pareto pass.")
        return

    sourced = all(r.get("area_coefficients_sourced") for r in have_area)
    objectives = ("cycles", "area")
    pts = []
    for r in have_area:
        obj = [cycles_of(r), r["area_au"]]
        if have_energy and r.get("cache_dynamic_pj"):
            obj.append(r["cache_dynamic_pj"])
        pts.append((f"{r['design_slug']}|{r['model']}", obj))
    if have_energy and len(have_energy) == len(have_area):
        objectives = ("cycles", "area", "energy")

    front = pareto(pts)
    print(f"=== Pareto front over {', '.join(objectives)} "
          f"({len(front)} of {len(pts)} non-dominated) ===\n")
    for key, obj in sorted(front, key=lambda kv: kv[1][0]):
        vals = "  ".join(f"{o:,.0f}" for o in obj)
        print(f"  {key[:46]:46} {vals}")

    # --- Robustness ---------------------------------------------------------
    print(f"\n=== Robustness: area coefficients +/-{int(PERTURB*100)}% ===\n")
    if sourced:
        print("  All area coefficients are sourced; the front above stands on "
              "measured figures.")
        return
    # Scale the SRAM-heavy and logic-heavy halves in opposite directions: that
    # is the perturbation the front is most exposed to, because it changes the
    # balance between "more cache" and "more compute" rather than the total.
    sram_keys = ("icache", "dcache", "l2", "snitch_tcdm", "spatz_tcdm")
    stable = set(k for k, _ in front)
    for direction in (1, -1):
        moved = []
        for r in have_area:
            parts = r.get("area_breakdown") or {}
            adj = sum(v * (1 + direction * PERTURB) if k in sram_keys else v
                      for k, v in parts.items())
            obj = [cycles_of(r), adj]
            if len(objectives) == 3:
                obj.append(r["cache_dynamic_pj"])
            moved.append((f"{r['design_slug']}|{r['model']}", obj))
        stable &= set(k for k, _ in pareto(moved))

    lost = sorted(set(k for k, _ in front) - stable)
    print(f"  {len(stable)} of {len(front)} front members survive both perturbations.")
    if lost:
        print("  Depend on the placeholder coefficients: "
              + ", ".join(k[:40] for k in lost))
    print("\n  Area coefficients are PLACEHOLDERS (see pipeline/sweep/area.py).")
    print("  Treat the cycles column as the only measured objective here.")


if __name__ == "__main__":
    main()
