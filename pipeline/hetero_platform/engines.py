# SPDX-License-Identifier: Apache-2.0
"""The three engines of the hetero_soc board.

  cva6    the orchestrator. Runs the whole Deeploy Generic mapping, so it is
          the engine that can execute anything, and the one every node falls
          back to.
  snitch  the 9-core Snitch cluster with the Xssr/Xfrep sequencers. Offloads
          MatMul, Gemm and Conv2d; those are the kernels runtime/snitch/kernels
          implements against the two extensions.
  spatz   the Snitch/Spatz pair. Same three operators, autovectorized to RVV.

A cluster engine only claims a node it can really run, which is narrower than
"the operator appears in my mapping":

  * fp32 only -- the cluster kernels are the fp32 ones. This cannot be decided
    from the ONNX dtypes alone: Deeploy infers the real element type from the
    input data, so a graph whose tensors are declared float32 but hold integer
    values is deployed as an integer network. The generator therefore tells the
    cluster engines whether the network came out fp32, and they decline
    everything if it did not -- an integer network runs entirely on the host.
  * the working set has to fit the cluster scratchpad. The Spatz pair cannot
    compute on anything outside its TCDM at all (its vector load/store unit is
    wired to the scratchpad), and the Snitch cluster would fall back to running
    against main memory, which the measurements say is 5-8x slower.

Getting this wrong is not a performance bug, it is a correctness one, so
canExecute is where it is enforced rather than in the cost model.
"""

import sys
from pathlib import Path

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t
from Deeploy.DeeployTypes import DeploymentEngine, NodeBinding, NodeMapper
from Deeploy.Targets.Generic.Bindings import BasicTransformer
from Deeploy.Targets.Generic.Layers import ConvLayer, GEMMLayer
from Deeploy.Targets.Generic.Parsers import GenericConv2DParser, GenericGEMMParser, MatMulParser
from Deeploy.Targets.Generic.Platform import GenericMapping
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker, GEMMChecker, MatMulChecker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "targets"))
from hetero import system  # noqa: E402

from . import templates  # noqa: E402

# What the host needs on top of the Deeploy runtime.
HOST_INCLUDES = ["DeeployBasicMath.h", "hes_host.h"]

_FP32 = PointerClass(float32_t)

# Bytes of a cluster's TCDM a job may use for its operands. The scratchpad also
# holds the cluster's own data, its per-core stacks and the mailbox, so this is
# deliberately below the raw size; cluster_main.c refuses a job that does not
# fit and the host reports it rather than computing on main memory by accident.
TCDM_BUDGET = system.TCDM_SIZE - system.MAILBOX_SIZE \
    - system.SNITCH_CLUSTER.nb_core * system.CLUSTER_STACK_SIZE \
    - 8 * 1024


def _bindings(checker, template):
    return [NodeBinding(checker, template, BasicTransformer)]


def _cluster_mapping(engine_macro: str):
    """MatMul / Gemm / Conv2d, emitted as offloads to one engine."""
    matmul = NodeMapper(
        MatMulParser(),
        _bindings(MatMulChecker([_FP32, _FP32], [_FP32]), templates.matmul(engine_macro)))
    gemm = NodeMapper(
        GenericGEMMParser(),
        _bindings(GEMMChecker([_FP32, _FP32, _FP32], [_FP32]), templates.gemm(engine_macro)))
    conv = NodeMapper(
        GenericConv2DParser(),
        _bindings(ConvChecker([_FP32, _FP32, _FP32], [_FP32]), templates.conv2d(engine_macro)))
    return {
        "MatMul": GEMMLayer([matmul]),
        "Gemm": GEMMLayer([gemm]),
        "Conv": ConvLayer([conv]),
    }


def _tensor_bytes(tensor) -> int:
    if not hasattr(tensor, "shape") or tensor.shape is None:
        return 0
    n = 1
    for dim in tensor.shape:
        if not isinstance(dim, int):
            return 0  # dynamic: cannot size it, treat as free
        n *= dim
    dtype = getattr(tensor, "dtype", None)
    width = np.dtype(dtype).itemsize if dtype is not None else 4
    return n * width


def _all_fp32(node: gs.Node) -> bool:
    for tensor in list(node.inputs) + list(node.outputs):
        dtype = getattr(tensor, "dtype", None)
        if dtype is None:
            continue
        if np.dtype(dtype) != np.float32:
            return False
    return True


def working_set_bytes(node: gs.Node) -> int:
    """Bytes a cluster would have to stage to run this node."""
    return sum(_tensor_bytes(t) for t in list(node.inputs) + list(node.outputs))


class Cva6HostEngine(DeploymentEngine):
    """The orchestrator. Executes anything Deeploy's Generic platform can."""

    def __init__(self, name: str = "cva6", Mapping = None,
                 includeList = HOST_INCLUDES) -> None:
        super().__init__(name, Mapping if Mapping is not None else GenericMapping,
                         "", includeList)

    def canExecute(self, node: gs.Node) -> bool:
        return node.op in self.Mapping


class ClusterEngine(DeploymentEngine):
    """A Snitch-family cluster reached by offload."""

    def __init__(self, name: str, engine_macro: str, includeList = HOST_INCLUDES,
                 tcdm_budget: int = TCDM_BUDGET, enabled: bool = True) -> None:
        super().__init__(name, _cluster_mapping(engine_macro), "", includeList)
        self.engine_macro = engine_macro
        self.tcdm_budget = tcdm_budget
        # False when the network is not fp32: the clusters have no integer
        # kernels, so they must not claim a node they cannot run.
        self.enabled = enabled

    def canExecute(self, node: gs.Node) -> bool:
        if not self.enabled:
            return False
        if node.op not in self.Mapping:
            return False
        if not _all_fp32(node):
            return False
        return working_set_bytes(node) <= self.tcdm_budget


class SnitchClusterEngine(ClusterEngine):
    """8 compute cores plus a DMA core, each an integer core in front of a
    decoupled FP subsystem with the SSR data movers and the FREP sequencer."""

    def __init__(self, name: str = "snitch", enabled: bool = True) -> None:
        super().__init__(name, "HES_ENGINE_SNITCH", enabled = enabled)


class SpatzClusterEngine(ClusterEngine):
    """One Snitch core with a 4-lane Spatz vector unit, plus a DMA core."""

    def __init__(self, name: str = "spatz", enabled: bool = True) -> None:
        super().__init__(name, "HES_ENGINE_SPATZ", enabled = enabled)
