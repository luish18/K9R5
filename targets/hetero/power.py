#
# Energy model for the memories of the hetero_soc boards.
#
# GVSoC accumulates energy during the simulation from activity the models
# report (vp::PowerSource), so this file supplies coefficients rather than
# doing any accounting itself: the numbers below become properties on the cache
# and memory components, and `gvsoc --power` turns the accounting on.
#
# --- Where the numbers come from -------------------------------------------
#
# The anchor is the only characterized power model in the GVSoC tree:
# pulp/chips/siracusa/power_models/l1/, which describes the L1 of Siracusa, a
# taped-out PULP chip. Its cluster.json gives l1/mapping/size = 0x40000 over
# nb_l1_banks = 1 << int(log2(nb_pe * banking_factor)) = 16 banks, so those
# values describe a **16 KiB SRAM bank**:
#
#     read/write dynamic   2.124280487 pJ per access
#     leakage              1.707210625e-5 W
#     background dynamic   5.2e-6 W
#
# at the operating point its tables are indexed by, 25 C and 1.2 V.
#
# --- What is NOT known ------------------------------------------------------
#
# GVSoC does not record the technology node these were characterized at, and
# there is no README beside them. **Do not combine energies derived from here
# with an area model quoted at a different node without saying so.** The
# operating point (25 C, 1.2 V) is recorded; the node is not. Until someone
# confirms it from the Siracusa publication, treat absolute joules as
# indicative and rely on *ratios between design points*, which is what the
# sweep actually compares. sweep/ reports a sensitivity band for exactly this
# reason.
#
# --- The scaling rule -------------------------------------------------------
#
# One characterized size has to cover caches from 8 KiB to 2 MiB and TCDM banks
# of a few KiB, so the anchor is scaled by geometry:
#
#   dynamic energy per access ~ sqrt(size)
#       A square SRAM array's bitlines and wordlines each grow as the square
#       root of its capacity, and the energy of an access is dominated by
#       driving them. This is the standard first-order approximation and is
#       what CACTI-style models reduce to when everything else is held fixed.
#       It is an approximation: it ignores banking, port count and the sense
#       amplifiers, all of which a real macro would change together with size.
#
#   leakage ~ size
#       Leakage is per-cell, so it scales with the number of cells.
#
# Both rules are deliberately simple and deliberately visible. A sweep that
# concludes "a bigger cache wins" on the strength of an energy model should be
# able to show which rule carried the conclusion.
#

import math

# --- The anchor, from pulp/chips/siracusa/power_models/l1/ ------------------
REF_SIZE = 16 * 1024          # bytes, one Siracusa L1 bank
REF_ACCESS_PJ = 2.124280487   # pJ, read or write
REF_LEAKAGE_W = 1.707210625e-5
REF_BACKGROUND_W = 5.2e-6

# Operating point the anchor's tables are indexed by. Emitted into every model
# so the numbers carry their conditions rather than floating free.
TEMPERATURE = "25"
VOLTAGE = "1.2"

# A refill moves a whole line between two levels rather than performing one
# array access, so it is charged as the accesses that line's worth of data
# implies. The line is written once into this cache and read once out of the
# level below; the factor is per byte of line over the width of one access.
ACCESS_BYTES = 4


def _pj(size: int) -> float:
    """Dynamic energy of one access to an SRAM of `size` bytes, in pJ."""
    return REF_ACCESS_PJ * math.sqrt(size / REF_SIZE)


def _leak_w(size: int) -> float:
    """Leakage of an SRAM of `size` bytes, in W."""
    return REF_LEAKAGE_W * (size / REF_SIZE)


def _value(unit: str, value: float) -> dict:
    """One GVSoC power table: a constant, indexed by temperature and voltage."""
    return {
        "type": "linear",
        "unit": unit,
        "values": {TEMPERATURE: {VOLTAGE: {"any": f"{value:.12g}"}}},
    }


def cache_model(size: int, line_size: int) -> dict:
    """Power properties for a TimingCache of this geometry.

    Keys match the sources targets/hetero/timing_cache.cpp registers: an access
    charges `read` or `write`, and a line actually transferred charges `refill`.
    """
    access = _pj(size)
    # A refill writes line_size bytes into the array, which is line_size /
    # ACCESS_BYTES array accesses' worth of energy.
    refill = access * (line_size / ACCESS_BYTES)
    return {
        "read": {"dynamic": _value("pJ", access)},
        "write": {"dynamic": _value("pJ", access)},
        "refill": {"dynamic": _value("pJ", refill)},
        "background": {
            "dynamic": _value("W", REF_BACKGROUND_W * (size / REF_SIZE)),
            "leakage": _value("W", _leak_w(size)),
        },
    }


def memory_model(size: int) -> dict:
    """Power properties for a memory.Memory of `size` bytes.

    Keys are the ones core/models/memory/memory.cpp registers. The three widths
    are charged the same, as in the Siracusa anchor -- an SRAM access drives the
    full wordline whatever slice of it the requester wanted.
    """
    access = _pj(size)
    return {
        "read_8": {"dynamic": _value("pJ", access)},
        "read_16": {"dynamic": _value("pJ", access)},
        "read_32": {"dynamic": _value("pJ", access)},
        "write_8": {"dynamic": _value("pJ", access)},
        "write_16": {"dynamic": _value("pJ", access)},
        "write_32": {"dynamic": _value("pJ", access)},
        "background": {
            "dynamic": _value("W", REF_BACKGROUND_W * (size / REF_SIZE)),
            "leakage": _value("W", _leak_w(size)),
        },
    }


# --- Main memory ------------------------------------------------------------
#
# ESTIMATE, NOT AN ANCHOR. Everything above descends from a characterized SRAM
# macro in the GVSoC tree. There is no such model for main memory, and the
# sqrt(size) SRAM rule must NOT be stretched to it: applying it to a 2 GiB
# window would claim ~770 pJ per access from a rule about bitline lengths in an
# on-chip array, which is not the physics of an off-chip DRAM.
#
# DRAM access energy is dominated by the I/O and the row activation, not by
# capacity, so it is modelled as a constant per access. The value below is an
# order-of-magnitude placeholder standing in for the tens-of-pJ-per-access
# region that DRAM and HBM are usually quoted in.
#
# This is the single least defensible number in the energy model, and it
# matters: a sweep over cache sizes trades on-chip accesses against main-memory
# accesses, so this constant sets the exchange rate. Two consequences:
#
#   1. It must be replaced with a figure for the memory actually being modelled
#      (and the memory actually being modelled must be decided -- memsys.py
#      describes it only as "100 cycles, 8 B/cycle").
#   2. Until then, sweep/ must report main-memory energy as its own line rather
#      than folded into a total, and must include this coefficient in the
#      sensitivity band. A design ranking that flips when this moves within its
#      plausible range is not a result.
#
DRAM_ACCESS_PJ_ESTIMATE = 20.0
DRAM_ACCESS_PJ_IS_ESTIMATE = True


def dram_model() -> dict:
    """Power properties for the main-memory window. See the caveat above."""
    e = DRAM_ACCESS_PJ_ESTIMATE
    return {
        "read_8": {"dynamic": _value("pJ", e)},
        "read_16": {"dynamic": _value("pJ", e)},
        "read_32": {"dynamic": _value("pJ", e)},
        "write_8": {"dynamic": _value("pJ", e)},
        "write_16": {"dynamic": _value("pJ", e)},
        "write_32": {"dynamic": _value("pJ", e)},
        # Left out deliberately: a background/leakage figure for off-chip DRAM
        # would be a second invented number on top of the first, and refresh
        # power does not belong to any one design point in this sweep.
    }


def attach_cluster_tcdm(cluster_comp, arch) -> int:
    """Give every TCDM bank of a built cluster its energy coefficients.

    The banks are memory.Memory instances created inside GVSoC's SnitchCluster,
    so they are reached through the built tree rather than at construction --
    the same approach hetero/snitch_memsys.py takes to the HBM mapping.
    Returns how many banks were found, so the caller can fail loudly on zero
    rather than silently reporting a cluster whose scratchpad costs nothing.
    """
    nb_banks = arch.tcdm.nb_superbanks * arch.tcdm.nb_banks_per_superbank
    bank_size = int(arch.tcdm.area.size // nb_banks)
    model = memory_model(bank_size)
    found = 0
    for i in range(nb_banks):
        try:
            bank = cluster_comp.get_component(f'tcdm/bank_{i}')
        except KeyError:
            continue
        bank.add_properties(model)
        found += 1
    if found != nb_banks:
        raise RuntimeError(
            f"expected {nb_banks} TCDM banks to give power models to, found "
            f"{found} -- the GVSoC Snitch cluster layout changed")
    return found
