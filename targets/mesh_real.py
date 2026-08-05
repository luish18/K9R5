#
# CVA6 manager + Snitch cluster mesh, with the memory hierarchy of
# targets/hetero/system.py (L1 TCDM -> L2 -> L3 -> HyperRAM).
#
# Phase 1a of docs/hetero-mesh-plan.md: host, one cluster, shared memory
# levels. The cluster is present and addressable but halted — releasing it is
# the dispatch runtime of phase 2.
#

import gvsoc.runner as gvsoc

from hetero.mesh import MeshBoard


class Target(gvsoc.Target):

    gapy_description = "CVA6 manager + Snitch cluster mesh (L1/L2/L3/HyperRAM)"
    name = "mesh_real"
    model = MeshBoard

    def __init__(self, parser, options=None, name=None):
        super().__init__(parser, options, model=MeshBoard, name=name)
