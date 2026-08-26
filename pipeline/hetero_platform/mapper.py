# SPDX-License-Identifier: Apache-2.0
"""Choosing which core runs each node.

Deeploy's default EngineMapper takes the first engine that says it can execute
a node, which makes the answer depend on the order the engines happen to be
listed in. This one picks the cheapest, from a cost model seeded with the
per-core measurements this repository already has in results/.

The model is deliberately crude, because it only has to get the *ordering*
right, and the ordering is what the measurements establish:

    cost = offload_overhead(engine) + macs(node) / rate(engine, op)

`rate` is MACs per cycle, derived from results/: for each measured operator,
the MAC count of that benchmark divided by the cycles each core took. The
Snitch cluster's rate is the single-core Xssr/Xfrep rate multiplied by the
eight compute cores it splits the work over. `offload_overhead` is what a job
costs before any arithmetic happens -- mailbox write, doorbell, DMA staging,
completion -- measured by make mesh-test.

The overhead term is the part that matters: it is what stops a small node
being shipped to a cluster that would finish the arithmetic quickly and spend
ten times longer getting the data there and back.

Every number is a measurement or an arithmetic consequence of one; nothing here
is a guess about hardware. --pin overrides the whole thing, which is how the
mapper's choice gets checked against the alternatives rather than assumed.
"""

from typing import Dict, Optional

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import DeploymentEngine
from Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses import EngineMapper

from .engines import ClusterEngine, working_set_bytes

# MACs per cycle, per engine and operator class.
#
# cva6/spatz come straight from results/: e.g. GEMM/Regular is 32x32x32 =
# 32768 MACs in 489.0k cycles on the CVA6 (0.067 MAC/cycle) and 60.9k on Spatz
# (0.538). snitch is the single-core Xssr/Xfrep rate (32768 / 43.5k = 0.753)
# times the 8 compute cores of the cluster; make mesh-test measures 7.3k cycles
# for the same shape, i.e. 4.5 MAC/cycle, which is what is used here.
RATES: Dict[str, Dict[str, float]] = {
    "cva6": {"Gemm": 0.067, "MatMul": 0.075, "Conv": 0.070, "_default": 0.070},
    "snitch": {"Gemm": 4.50, "MatMul": 4.07, "Conv": 1.60, "_default": 1.60},
    "spatz": {"Gemm": 0.54, "MatMul": 0.57, "Conv": 0.16, "_default": 0.16},
}

# Cycles a job costs before any arithmetic. Measured with make mesh-test: the
# Snitch cluster reports ~7.3k cycles for a 32x32x32 GEMM whose arithmetic
# alone is ~7.3k at the rate above, so the fixed part is dominated by staging
# and is charged per byte below; this is the part that does not scale.
OFFLOAD_FIXED = {"cva6": 0, "snitch": 1200, "spatz": 1200}

# Cycles per byte staged into TCDM and back, from the same measurements
# (Conv2d 4x16x16 stages ~11.5 KiB and the difference between its staged and
# unstaged runs is dominated by the transfer).
OFFLOAD_PER_BYTE = {"cva6": 0.0, "snitch": 0.10, "spatz": 0.10}


def node_macs(node: gs.Node) -> float:
    """Multiply-accumulates the node performs, from its shapes.

    Used only to compare engines against each other, so an operator whose cost
    is not dominated by MACs just needs a consistent number rather than an
    exact one.
    """

    def shape_of(tensor):
        shape = getattr(tensor, "shape", None)
        if not shape or any(not isinstance(d, int) for d in shape):
            return None
        return list(shape)

    out = shape_of(node.outputs[0]) if node.outputs else None
    out_elems = 1
    for d in (out or [1]):
        out_elems *= d

    if node.op in ("Gemm", "MatMul") and len(node.inputs) >= 2:
        a = shape_of(node.inputs[0])
        if a:
            return out_elems * a[-1]
    elif node.op == "Conv" and len(node.inputs) >= 2:
        w = shape_of(node.inputs[1])
        if w:
            taps = 1
            for d in w[1:]:
                taps *= d
            return out_elems * taps
    return float(out_elems)


class CostEngineMapper(EngineMapper):
    """Pick the engine whose modelled cost for the node is lowest.

    `pin` forces every node the named engine can execute onto it, which is what
    the mapped-versus-pinned comparison uses.
    """

    def __init__(self, engineDict: Dict[str, DeploymentEngine],
                 pin: Optional[str] = None) -> None:
        super().__init__(engineDict)
        self.pin = pin
        self.decisions = []  #: (node name, op, engine, cost) for the report

    def cost(self, engine: DeploymentEngine, node: gs.Node) -> float:
        rates = RATES.get(engine.name, RATES["cva6"])
        rate = rates.get(node.op, rates["_default"])
        compute = node_macs(node) / rate
        overhead = OFFLOAD_FIXED.get(engine.name, 0)
        if isinstance(engine, ClusterEngine):
            overhead += OFFLOAD_PER_BYTE.get(engine.name, 0.0) * working_set_bytes(node)
        return compute + overhead

    def mapNodeToEngine(self, node: gs.Node, graph: gs.Graph) -> Optional[DeploymentEngine]:
        _ = graph
        candidates = [e for e in self.engineDict.values() if e.canExecute(node)]
        if not candidates:
            return None

        if self.pin is not None:
            pinned = self.engineDict.get(self.pin)
            if pinned is not None and pinned in candidates:
                self.decisions.append((node.name, node.op, pinned.name, None))
                return pinned

        best = min(candidates, key = lambda e: self.cost(e, node))
        self.decisions.append((node.name, node.op, best.name, self.cost(best, node)))
        return best


def make_mapper(pin: Optional[str] = None):
    """A CostEngineMapper factory the deployer can instantiate."""

    class _Mapper(CostEngineMapper):

        def __init__(self, engineDict):
            super().__init__(engineDict, pin = pin)

    return _Mapper
