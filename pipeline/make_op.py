#!/usr/bin/env python3
"""Create a pipeline-ready op directory from an ONNX model + input tensors.

The pipeline consumes a directory holding network.onnx / inputs.npz /
outputs.npz (the Deeploy test-case format). This helper takes any
single-op (or small) ONNX model plus its inputs and produces the expected
outputs with onnxruntime.

Usage:
  python pipeline/make_op.py model.onnx inputs.npz -o ops/myop
  python pipeline/make_op.py model.onnx --random -o ops/myop   # random inputs
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="ONNX model (single op or small graph)")
    ap.add_argument("inputs", nargs="?", help="inputs.npz (arrays in graph input order)")
    ap.add_argument("--random", action="store_true", help="generate random inputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True, help="output op directory")
    args = ap.parse_args()

    model = onnx.load(args.model)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    graph_inputs = sess.get_inputs()

    if args.random:
        rng = np.random.default_rng(args.seed)
        feed = {}
        for gi in graph_inputs:
            shape = [d if isinstance(d, int) else 1 for d in gi.shape]
            if "float" in gi.type:
                feed[gi.name] = rng.random(shape, dtype=np.float32)
            else:
                feed[gi.name] = rng.integers(-128, 127, shape, dtype=np.int8).astype(np.int64
                    if "int64" in gi.type else np.int8)
    elif args.inputs:
        data = np.load(args.inputs)
        names = list(data.files)
        feed = {gi.name: data[names[i]] for i, gi in enumerate(graph_inputs)}
    else:
        ap.error("provide inputs.npz or --random")

    outputs = sess.run(None, feed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.model, out / "network.onnx")
    np.savez(out / "inputs.npz", **{f"input_{i}": v for i, v in enumerate(feed.values())})
    np.savez(out / "outputs.npz", **{f"output_{i}": v for i, v in enumerate(outputs)})

    print(f"op directory ready: {out}")
    print(f"  inputs : {[v.shape for v in feed.values()]}")
    print(f"  outputs: {[o.shape for o in outputs]}")
    print(f"run it with:\n  .venv/bin/python pipeline/run.py {out}")


if __name__ == "__main__":
    main()
