# SPDX-License-Identifier: Apache-2.0
"""Bracketing every node with a progress beacon.

A simulation of a whole network runs for minutes, and until it finishes there
is otherwise nothing to distinguish "still working" from "wedged". This pass
wraps each node's execution block so the host prints

    [HES-PROG] sample=3/64 node=5 op=Conv engine=snitch cycles=173912

as soon as that node completes. pipeline/run_hetero.py turns the stream into a
live progress line and uses the gap between beacons as a stall watchdog.

The pass is attached by rewriting each binding's CodeTransformation rather than
by rebuilding Deeploy's bindings: `instrument()` gives every binding in a
mapping a fresh CodeTransformation with this pass in front of the existing
ones, leaving the shared BasicTransformer that Deeploy's own platforms use
untouched.
"""

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformation, CodeTransformationPass, ExecutionBlock, \
    NetworkContext, NodeTemplate, _NoVerbosity


# Deeploy copies bindings, and a copy of the pass would carry a copy of any
# state held on the instance -- which showed up as every node being numbered 0.
# The numbering and the "already wrapped" set therefore live here, at module
# scope, where copying a pass cannot split them.
_node_index = {}
_wrapped = set()


def reset() -> None:
    """Forget the numbering, so a second network in one process starts at 0."""
    _node_index.clear()
    _wrapped.clear()


def index_of(name: str) -> int:
    """A stable number per node, in the order the nodes are first seen."""
    if name not in _node_index:
        _node_index[name] = len(_node_index)
    return _node_index[name]


class HeteroProgressPass(CodeTransformationPass):
    """Print a beacon after each node, with the engine that ran it."""

    def __init__(self, engine_of_node):
        super().__init__()
        self.engine_of_node = engine_of_node

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        # The same block can be handed to the transformation more than once;
        # wrapping it twice would nest two beacons around one kernel call and
        # report the node as having run twice.
        key = id(executionBlock)
        if key in _wrapped:
            return ctxt, executionBlock
        _wrapped.add(key)

        idx = index_of(name)
        engine, op = self.engine_of_node(name)
        # The node name goes in as a literal so the beacon says what ran, not
        # just where in the schedule it was.
        executionBlock.addLeft(
            NodeTemplate(f'uint32_t hes_t_{idx} = hes_node_begin({idx}, "{op}", {engine});\n'), {})
        executionBlock.addRight(
            NodeTemplate(f'hes_node_end({idx}, "{op}", {engine}, hes_t_{idx});\n'), {})
        return ctxt, executionBlock


ENGINE_MACRO = {
    "cva6": "HES_ENGINE_CVA6",
    "snitch": "HES_ENGINE_SNITCH",
    "spatz": "HES_ENGINE_SPATZ",
}


def instrument(mapping: dict, progress_pass: CodeTransformationPass) -> None:
    """Put `progress_pass` in front of every binding's transformations.

    Each binding gets its own CodeTransformation so the instances Deeploy
    shares between platforms are not modified.
    """
    for layer in mapping.values():
        for mapper in getattr(layer, "maps", []):
            for binding in getattr(mapper, "bindings", []):
                existing = binding.codeTransformer
                binding.codeTransformer = CodeTransformation(
                    [progress_pass, *existing.passes])
