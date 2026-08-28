#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn an ONNX graph into C for the hetero_soc board.

The same job DeeployTest/generateNetwork.py does for a single-engine platform,
with three engines and a cost mapper instead. Deeploy is imported as a library
and its Generic parsers, layers, type checkers and code generator are reused
verbatim; nothing in deps/ is patched.

Run from the repository root:

  python pipeline/hetero_platform/generate.py -t <op dir> -d <out dir> [--pin snitch]

It writes the same four files the existing pipeline already builds against --
Network.c, Network.h, testinputs.h, testoutputs.h -- plus mapping.json, which
records the engine each node was given and why.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

# Captured before the chdir below: Deeploy's code generator has to run with
# DeeployTest as the working directory, so any path the caller gave has to be
# resolved against where they actually were.
CWD = Path.cwd()
ROOT = Path(__file__).resolve().parent.parent.parent
DEEPLOY_TEST = ROOT / "deps" / "deeploy" / "DeeployTest"
sys.path.insert(0, str(DEEPLOY_TEST))
sys.path.insert(0, str(ROOT / "pipeline"))

from testUtils.codeGenerate import generateTestNetwork  # noqa: E402
from testUtils.typeMapping import inferTypeAndOffset  # noqa: E402

from Deeploy.DeeployTypes import _NoVerbosity  # noqa: E402
from Deeploy.EngineExtension.NetworkDeployers.EngineColoringDeployer import \
    EngineColoringDeployerWrapper  # noqa: E402
from Deeploy.Targets.Generic.Deployer import GenericDeployer  # noqa: E402
from Deeploy.Targets.Generic.Platform import GenericOptimizer  # noqa: E402

from hetero_platform import progress  # noqa: E402
from hetero_platform.mapper import make_mapper  # noqa: E402
from hetero_platform.deployment import HeteroPlatform  # noqa: E402


def is_fp32_network(input_types) -> bool:
    """Whether Deeploy inferred every network input as float32.

    The ONNX dtype is not enough: Deeploy picks the element type from the data,
    so an integer test whose tensors are declared float32 still deploys as an
    integer network, and the clusters have no integer kernels for it.
    """
    for ptr in input_types.values():
        referenced = getattr(ptr, "referencedType", None)
        if getattr(referenced, "typeName", None) != "float32_t":
            return False
    return bool(input_types)


def build_deployer(graph, input_types, input_offsets, state_dir, pin=None):
    """A Generic deployer made engine-colour-aware, with the cost mapper.

    EngineColoringDeployerWrapper is Deeploy's own mechanism for this: it
    inserts the colouring pass around every lowering pass, and asserts at the
    end that no node was left uncoloured.
    """
    platform = HeteroPlatform(fp32_network = is_fp32_network(input_types))
    deployer = GenericDeployer(graph,
                               platform,
                               input_types,
                               GenericOptimizer,
                               lambda g: list(g.nodes),
                               name = "DeeployNetwork",
                               default_channels_first = True,
                               deeployStateDir = state_dir,
                               inputOffsets = input_offsets)
    return EngineColoringDeployerWrapper(deployer, make_mapper(pin))


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--test-dir", required = True,
                    help = "directory holding network.onnx, inputs.npz, outputs.npz")
    ap.add_argument("-d", "--dump-dir", required = True, help = "where to write the C")
    ap.add_argument("--pin", default = None, choices = ["cva6", "snitch", "spatz"],
                    help = "force every node the engine can run onto it, instead "
                           "of letting the cost model choose")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = (CWD / test_dir).resolve()
    dump_dir = Path(args.dump_dir)
    if not dump_dir.is_absolute():
        dump_dir = (CWD / dump_dir).resolve()
    dump_dir.mkdir(parents = True, exist_ok = True)

    graph = gs.import_onnx(onnx.load_model(str(test_dir / "network.onnx")))
    inputs = np.load(test_dir / "inputs.npz")
    outputs = np.load(test_dir / "outputs.npz")

    test_inputs = [inputs[k].reshape(-1).astype(np.float64) for k in inputs.files]
    test_outputs = [outputs[k].reshape(-1).astype(np.float64) for k in outputs.files]

    # signProp is off: the fp32 path this board targets has no offsets to
    # propagate, and Deeploy's Generic platform is not a signProp platform.
    input_types, input_offsets = {}, {}
    for i, values in enumerate(test_inputs):
        if np.prod(values.shape) == 0:
            continue
        _type, offset = inferTypeAndOffset(values, False)
        input_types[f"input_{i}"] = _type
        input_offsets[f"input_{i}"] = offset

    deployer = build_deployer(graph, input_types, input_offsets,
                              str(dump_dir / "deeployStates"), args.pin)

    # The code transformations run inside prepare(), after lowering and
    # colouring but before code generation, so the progress pass has to read
    # the engine from the live graph rather than from anything collected
    # afterwards. By the time it runs, layers are bound and the colouring pass
    # has written attrs["engine"] on every node -- EngineColoringDeployer
    # asserts as much.
    decided = {}

    def engine_of_node(name):
        layer = deployer.layerBinding.get(name) if deployer.layerBinding else None
        if layer is None:
            return "HES_ENGINE_CVA6", "?"
        engine = layer.node.attrs.get("engine", "cva6")
        decided[name] = (engine, layer.node.op)
        return progress.ENGINE_MACRO.get(engine, "HES_ENGINE_CVA6"), layer.node.op

    progress.reset()
    progress_pass = progress.HeteroProgressPass(engine_of_node)
    for engine in deployer.Platform.engines:
        progress.instrument(engine.Mapping, progress_pass)

    deployer.prepare(_NoVerbosity)

    generateTestNetwork(deployer, test_inputs, test_outputs, str(dump_dir), _NoVerbosity)

    # Report the placement next to the generated code.
    placement = []
    for name, (engine, op) in decided.items():
        placement.append({"node": name, "op": op, "engine": engine})
    (dump_dir / "mapping.json").write_text(json.dumps({
        "pin": args.pin,
        "nodes": placement,
    }, indent = 2))

    by_engine = {}
    for entry in placement:
        by_engine.setdefault(entry["engine"], []).append(entry["op"])
    for engine, ops in sorted(by_engine.items()):
        print(f"  {engine:7} {len(ops):3} nodes  {', '.join(sorted(set(ops)))}")
    print(f"  -> {dump_dir.relative_to(ROOT) if dump_dir.is_relative_to(ROOT) else dump_dir}")


if __name__ == "__main__":
    os.chdir(DEEPLOY_TEST)
    main()
