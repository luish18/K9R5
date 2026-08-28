# SPDX-License-Identifier: Apache-2.0
"""The hetero_soc deployment platform.

Buffers are the Generic ones: every tensor lives in the host's main memory,
which is the only memory all three engines can reach. A cluster stages what it
needs into its own scratchpad for the duration of a node and writes the result
back, so there is no separate memory level for Deeploy to allocate into and no
tiler to satisfy -- the operators this targets fit a cluster scratchpad whole.
"""

from typing import List, Optional

from Deeploy.DeeployTypes import DeploymentPlatform
from Deeploy.Targets.Generic.Platform import GenericConstantBuffer, GenericStructBuffer, GenericTransientBuffer, \
    GenericVariableBuffer

from .engines import Cva6HostEngine, SnitchClusterEngine, SpatzClusterEngine

INCLUDE_LIST = ["DeeployBasicMath.h", "hes_host.h"]


class HeteroPlatform(DeploymentPlatform):
    """CVA6 orchestrator, Snitch cluster, Snitch/Spatz pair.

    The host engine is listed last on purpose: it can execute every operator,
    so with Deeploy's default first-match mapper it would take everything. The
    cost mapper in mapper.py does not depend on the order, but the fallback
    behaviour should still be sensible.
    """

    def __init__(self,
                 engines: Optional[List] = None,
                 fp32_network: bool = True,
                 variableBuffer = GenericVariableBuffer,
                 constantBuffer = GenericConstantBuffer,
                 structBuffer = GenericStructBuffer,
                 transientBuffer = GenericTransientBuffer) -> None:
        if engines is None:
            # The clusters only have fp32 kernels; on an integer network they
            # decline every node and everything runs on the host.
            engines = [SnitchClusterEngine(enabled = fp32_network),
                       SpatzClusterEngine(enabled = fp32_network),
                       Cva6HostEngine()]
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer,
                         transientBuffer)
