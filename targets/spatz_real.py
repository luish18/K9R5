#
# Spatz virtual board with a realistic main-memory latency.
#
# GVSoC's stock Spatz board maps HBM with latency 0 (see
# hetero/snitch_memsys.py), so instruction fetches that miss the cluster
# instruction cache refill for free. Autovectorized kernels are large and the
# shared cluster cache is 8 KiB, so those refills are not rare: this is the
# memory delay the Spatz numbers were missing.
#
# Data accesses are unchanged — they go to the cluster TCDM, which is
# single-cycle in the real hardware too, through the interleaver that charges
# bank conflicts.
#

import gvsoc.runner as gvsoc

from hetero.snitch_memsys import SpatzRealBoard


class Target(gvsoc.Target):

    gapy_description = "Spatz virtual board (main memory at the shared DRAM latency)"
    name = "spatz_real"
    model = SpatzRealBoard

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=SpatzRealBoard, name=name)
