#
# hetero_soc: the whole chip in one simulation -- a CVA6 orchestrator, a
# 9-core Snitch cluster with the Xssr/Xfrep sequencers, and a Snitch+Spatz
# pair, sharing one main memory.
#
# The board is described in hetero/soc.py and its parameters in
# hetero/system.py. Unlike the per-core targets, a run needs three binaries:
# --binary is the host ELF, and the two cluster ELFs come from the
# HES_ELF_SNITCH / HES_ELF_SPATZ environment variables.
#

import gvsoc.runner as gvsoc

from hetero.soc import HeteroBoard


class Target(gvsoc.Target):

    gapy_description = ("Heterogeneous SoC: CVA6 host + Snitch cluster (Xssr/Xfrep) "
                        "+ Snitch/Spatz pair, shared main memory")
    name = "hetero_soc"
    model = HeteroBoard

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=HeteroBoard, name=name)
