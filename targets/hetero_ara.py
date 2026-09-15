#
# hetero_ara: the hetero_soc chip with a vector orchestrator.
#
# Identical to hetero_soc -- same Snitch cluster, same Snitch/Spatz pair, same
# memory system, same addresses -- except that the CVA6 carries an Ara vector
# unit (ISS v2, see targets/ara_host.py).
#
# It exists alongside hetero_soc rather than replacing it so the two can be run
# against each other: the question this board answers is what a vector
# orchestrator is worth on the elementwise and pooling work that the clusters
# have no kernels for, which the MNIST run showed to be about half the time.
#

import gvsoc.runner as gvsoc

from hetero.soc import HeteroAraBoard


class Target(gvsoc.Target):

    gapy_description = ("Heterogeneous SoC with a vector orchestrator: CVA6 + Ara "
                        "+ Snitch cluster (Xssr/Xfrep) + Snitch/Spatz pair")
    name = "hetero_ara"
    model = HeteroAraBoard

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=HeteroAraBoard, name=name)
