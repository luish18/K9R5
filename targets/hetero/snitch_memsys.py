#
# Retuning of the Snitch/Spatz board memory system.
#
# GVSoC builds both boards from the same Soc class, which hardcodes
#
#     hbm_latency = 0 if arch.use_spatz else 100
#
# The zero is deliberate upstream — the Spatz RTL benchmarks it ships with run
# against a zero-latency memory — but it means instruction fetches that miss
# the cluster instruction cache refill from HBM for free, which is the one
# place the Spatz simulation has no memory delay at all.
#
# Rather than fork the ~90-line Soc to change one argument, the mapping is
# retuned on the built tree, before the configuration is generated.
#

from pulp.chips.snitch.snitch import SnitchBoard

from hetero import memsys

# Base address of the HBM window in the Snitch/Spatz address map
# (SnitchArch.Chip.Soc.hbm).
HBM_BASE = 0x8000_0000


def retune_hbm(board, latency: int):
    """Set the latency of the HBM mapping of `board`'s wide AXI router.

    Raises if the board does not look the way this code expects, so that a
    GVSoC update that moves the mapping is a loud failure rather than a
    simulation that silently keeps the old latency.
    """
    axi = board.get_component('chip/soc/wide_axi')
    mappings = axi.get_property('mappings')

    hbm = [mapping for mapping in mappings.values() if mapping['base'] == HBM_BASE]
    if len(hbm) != 1:
        raise RuntimeError(
            f"expected exactly one mapping at 0x{HBM_BASE:x} on chip/soc/wide_axi, "
            f"found {len(hbm)} — the GVSoC Snitch board layout changed")

    hbm[0]['latency'] = latency


class SnitchRealBoard(SnitchBoard):
    """Snitch board whose HBM answers at the memory system's DRAM latency."""

    def __init__(self, parent, name: str, parser, options, spatz=False):
        super().__init__(parent, name, parser, options, spatz=spatz)

        retune_hbm(self, memsys.DRAM_LATENCY)


class SpatzRealBoard(SnitchRealBoard):
    """Spatz board whose HBM answers at the memory system's DRAM latency."""

    def __init__(self, parent, name: str, parser, options):
        super().__init__(parent, name, parser, options, spatz=True)
