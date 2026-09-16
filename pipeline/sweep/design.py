#!/usr/bin/env python3
"""A design point, and the rules that say whether it could be built.

The simulator will happily run configurations that no one could tape out, and
report them as fast. The predicates here are what keep those out of the
results -- see `validate` for the one that actually bites.
"""

import hashlib
import json
from pathlib import Path

# Every knob the sweep may set, with the value the committed SoC uses. Keeping
# the defaults here as well as in targets/hetero/*.py is deliberate: it lets a
# design point be written out fully resolved, so a result row records what was
# run rather than what was asked for, and a later change to a default cannot
# silently re-label an old row.
DEFAULTS = {
    # Ara, on the vector host
    "HOST_VLEN": 4096,
    "HOST_NB_LANES": 4,
    "HOST_LANE_WIDTH": 8,
    # Spatz, per core of the Spatz cluster
    "SPATZ_VLEN": 512,
    "SPATZ_NB_LANES": 4,
    "SPATZ_LANE_WIDTH": 8,
    # Cluster shape
    "SNITCH_NB_CORE": 9,
    "SPATZ_NB_CORE": 9,
    "TCDM_SIZE": 0x20000,
    # Host memory hierarchy
    "ICACHE_SIZE": 16 * 1024, "ICACHE_WAYS": 4,
    "DCACHE_SIZE": 32 * 1024, "DCACHE_WAYS": 8,
    "L2_SIZE": 512 * 1024, "L2_WAYS": 8,
    "LINE_SIZE": 64,
    # Interconnect
    "NARROW_AXI_WIDTH": 8,
    "WIDE_AXI_WIDTH": 64,
}

# Knobs that are compiled into the model rather than read from its config, so
# changing one needs a GVSoC rebuild and a target of its own. Everything else
# costs nothing but a re-run.
BUILD_TIME = ("HOST_VLEN", "SPATZ_VLEN")


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def validate(design: dict):
    """Reasons this design could not be built. Empty means it is buildable.

    The important one is the vector geometry. GVSoC's vector model computes
    nb_elem_per_cycle = nb_lanes * lane_width / sewb with no upper bound
    (vector_unit_compute.cpp), so a design whose per-cycle chunk exceeds a
    vector register retires a whole vector operation per cycle and looks
    superb. Real Ara requires the register to hold at least one chunk. Nothing
    in the simulator will complain -- which is exactly why this is here.
    """
    d = dict(DEFAULTS, **design)
    bad = []

    for tag, vlen, lanes, width in (
            ("host", d["HOST_VLEN"], d["HOST_NB_LANES"], d["HOST_LANE_WIDTH"]),
            ("spatz", d["SPATZ_VLEN"], d["SPATZ_NB_LANES"], d["SPATZ_LANE_WIDTH"])):
        chunk_bits = lanes * width * 8
        if chunk_bits > vlen:
            bad.append(
                f"{tag}: {lanes} lanes x {width} B = {chunk_bits} bits per cycle "
                f"exceeds VLEN {vlen}; the model would retire a whole vector op "
                f"per cycle and overstate this design")
        if not _pow2(vlen):
            bad.append(f"{tag}: VLEN {vlen} is not a power of two")
        if not _pow2(lanes):
            bad.append(f"{tag}: {lanes} lanes is not a power of two")

    # The cache model fatals on a bad geometry rather than mismodelling it, so
    # this only saves a wasted cell -- but a wasted cell is a wasted build too.
    for tag in ("ICACHE", "DCACHE", "L2"):
        size, ways = d[f"{tag}_SIZE"], d[f"{tag}_WAYS"]
        line = d["LINE_SIZE"]
        if size % (ways * line):
            bad.append(f"{tag}: {size} B does not divide into {ways} ways x {line} B lines")
        else:
            sets = size // (ways * line)
            if not _pow2(sets):
                bad.append(f"{tag}: {sets} sets is not a power of two")
    if not _pow2(d["LINE_SIZE"]):
        bad.append(f"LINE_SIZE {d['LINE_SIZE']} is not a power of two")

    # TCDM: 32 banks, and the interleaver masks with size-1.
    if not _pow2(d["TCDM_SIZE"]):
        bad.append(f"TCDM_SIZE {d['TCDM_SIZE']} is not a power of two")
    if d["TCDM_SIZE"] % 32:
        bad.append(f"TCDM_SIZE {d['TCDM_SIZE']} does not divide into 32 banks")

    # Each cluster keeps a 4 KiB stack per core plus the mailbox in its TCDM,
    # and a job needs room for its operands on top of that.
    for tag in ("SNITCH", "SPATZ"):
        n = d[f"{tag}_NB_CORE"]
        if n < 2:
            bad.append(f"{tag}_NB_CORE {n}: a cluster needs a DMA core plus at "
                       "least one compute core")
        stacks = n * 0x1000
        if stacks + 0x200 >= d["TCDM_SIZE"]:
            bad.append(f"{tag}: {n} cores x 4 KiB of stack does not fit a "
                       f"{d['TCDM_SIZE']} B TCDM")

    return bad


def resolve(design: dict) -> dict:
    """The design, with every default filled in."""
    unknown = set(design) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown design keys: {sorted(unknown)} -- a typo here "
                         "would silently run the default and label it as the "
                         "value you asked for")
    return dict(DEFAULTS, **design)


def slug(design: dict) -> str:
    """A short stable name for a design point.

    Built from the *resolved* design, so two spellings of the same machine get
    the same slug and two different machines never collide.
    """
    full = resolve(design)
    diff = {k: v for k, v in full.items() if v != DEFAULTS[k]}
    if not diff:
        return "baseline"
    digest = hashlib.sha1(
        json.dumps(full, sort_keys=True).encode()).hexdigest()[:8]
    # A readable prefix helps when reading a directory listing; the digest is
    # what actually guarantees uniqueness.
    lead = "-".join(f"{k.lower()}{v}" for k, v in sorted(diff.items())[:2])
    return f"{lead}-{digest}"[:60]


def build_key(design: dict) -> str:
    """Identifies the GVSoC build a design needs.

    Designs sharing this key share one target and one build; the rest of the
    sweep costs only a re-run.
    """
    full = resolve(design)
    return "-".join(f"{k.lower()}{full[k]}" for k in BUILD_TIME)


def write(design: dict, path: Path) -> Path:
    """Write the fully-resolved design where the pipeline can pick it up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolve(design), indent=2, sort_keys=True) + "\n")
    return path
