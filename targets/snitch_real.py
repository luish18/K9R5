#
# Snitch virtual board with the same main-memory latency as the other
# hetero-sim targets.
#
# The Snitch cluster GVSoC ships already models its memory system: instruction
# fetches go through the two-level cluster instruction cache and refill from
# HBM, and data accesses go through the banked TCDM interleaver, which charges
# bank conflicts. What this target pins down is the HBM latency, so that the
# same main memory sits behind CVA6, Snitch and Spatz (see hetero/memsys.py).
#
# GVSoC's stock Snitch board already charges 100 cycles for HBM, so this
# target only differs from `snitch` if DRAM_LATENCY is retuned. The Spatz
# board is the one that was getting its instruction fetches for free — see
# spatz_real.py.
#

import gvsoc.runner as gvsoc

from hetero.snitch_memsys import SnitchRealBoard


class Target(gvsoc.Target):

    gapy_description = "Snitch virtual board (main memory at the shared DRAM latency)"
    name = "snitch_real"
    model = SnitchRealBoard

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=SnitchRealBoard, name=name)
