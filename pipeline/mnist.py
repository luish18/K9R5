#!/usr/bin/env python3
"""Train a small MNIST CNN, export it to ONNX, and wrap it as a pipeline op.

  python pipeline/mnist.py --train --images 64

Writes ops/mnist/:

  network.onnx   the graph Deeploy compiles
  inputs.npz     the first test image, for Deeploy's single-inference codegen
  outputs.npz    what onnxruntime produces for it
  mnist_data.h   the images, their true labels and onnxruntime's prediction
                 for each, which runtime/mesh/mnist_main.c loops over

Everything is fp32. The shape is chosen so that every node's working set fits
a cluster scratchpad (85,504 bytes usable, see pipeline/hetero_platform
/engines.py) -- the widest is the first fully-connected weight at 400x32.

    28x28x1  --Conv 3x3, 8 --> 26x26x8 --ReLU--> --MaxPool 2--> 13x13x8
             --Conv 3x3,16 --> 11x11x16 --ReLU--> --MaxPool 2-->  5x5x16
             --Flatten--> 400 --Gemm--> 32 --ReLU--> --Gemm--> 10 --Softmax

Training is plain numpy, so this needs nothing beyond what the pipeline already
installs. It is self-checking in two ways: the numpy forward pass is compared
against onnxruntime on the exported graph, which catches an export that does
not match what was trained, and the reported test accuracy catches training
that did not work.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
OP_DIR = ROOT / "ops" / "mnist"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

C1, C2, FC = 8, 16, 32      # conv1 filters, conv2 filters, hidden units
K = 3                       # kernel size, both convs


# --- data ------------------------------------------------------------------

def load_mnist(cache: Path):
    """MNIST as float32 in [0,1], NCHW. Downloaded once and cached."""
    if not cache.exists():
        cache.parent.mkdir(parents = True, exist_ok = True)
        print(f"downloading MNIST -> {cache.relative_to(ROOT)}")
        urllib.request.urlretrieve(MNIST_URL, cache)
    with np.load(cache) as d:
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    prep = lambda x: (x.astype(np.float32) / 255.0)[:, None, :, :]
    return prep(xtr), ytr.astype(np.int64), prep(xte), yte.astype(np.int64)


# --- layers ----------------------------------------------------------------

def im2col(x, k):
    """(N,C,H,W) -> (N, C*k*k, OH*OW) for a valid k x k convolution."""
    n, c, h, w = x.shape
    oh, ow = h - k + 1, w - k + 1
    # A sliding window view costs no copy; the reshape below materializes it.
    windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis = (2, 3))
    # (N, C, OH, OW, k, k) -> (N, C, k, k, OH, OW) -> (N, C*k*k, OH*OW)
    return windows.transpose(0, 1, 4, 5, 2, 3).reshape(n, c * k * k, oh * ow)


def conv_forward(x, w, b, k):
    n, _, h, w_in = x.shape
    oh, ow = h - k + 1, w_in - k + 1
    cols = im2col(x, k)                                  # (N, C*k*k, OH*OW)
    out = np.einsum("fj,njp->nfp", w.reshape(w.shape[0], -1), cols)
    return (out + b[None, :, None]).reshape(n, w.shape[0], oh, ow), cols


def conv_backward(dout, cols, x_shape, w, k):
    n, c, h, w_in = x_shape
    f = w.shape[0]
    do = dout.reshape(n, f, -1)                          # (N,F,P)
    dw = np.einsum("nfp,njp->fj", do, cols).reshape(w.shape)
    db = do.sum(axis = (0, 2))
    dcols = np.einsum("fj,nfp->njp", w.reshape(f, -1), do)
    # Scatter the columns back onto the input.
    oh, ow = h - k + 1, w_in - k + 1
    dx = np.zeros(x_shape, dtype = np.float32)
    dcols = dcols.reshape(n, c, k, k, oh, ow)
    for i in range(k):
        for j in range(k):
            dx[:, :, i:i + oh, j:j + ow] += dcols[:, :, i, j, :, :]
    return dx, dw, db


def maxpool_forward(x):
    n, c, h, w = x.shape
    oh, ow = h // 2, w // 2
    # Drop an odd trailing row/column, exactly as ONNX MaxPool with no padding.
    xc = x[:, :, :oh * 2, :ow * 2].reshape(n, c, oh, 2, ow, 2)
    flat = xc.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, oh, ow, 4)
    idx = flat.argmax(axis = -1)
    return flat.max(axis = -1), (idx, x.shape, oh, ow)


def maxpool_backward(dout, cache):
    idx, xshape, oh, ow = cache
    n, c, h, w = xshape
    dflat = np.zeros((n, c, oh, ow, 4), dtype = np.float32)
    np.put_along_axis(dflat, idx[..., None], dout[..., None], axis = -1)
    dx = np.zeros(xshape, dtype = np.float32)
    back = dflat.reshape(n, c, oh, ow, 2, 2).transpose(0, 1, 2, 4, 3, 5)
    dx[:, :, :oh * 2, :ow * 2] = back.reshape(n, c, oh * 2, ow * 2)
    return dx


# --- the network -----------------------------------------------------------

class Net:

    def __init__(self, rng):
        # He initialization: the scale is set by the fan-in, which is every
        # element of a conv filter but the *first* axis of a dense weight,
        # since those are stored (in, out) for ONNX Gemm.
        def he(shape, fan_in):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)

        self.w1 = he((C1, 1, K, K), 1 * K * K)
        self.b1 = np.zeros(C1, dtype = np.float32)
        self.w2 = he((C2, C1, K, K), C1 * K * K)
        self.b2 = np.zeros(C2, dtype = np.float32)
        self.flat = C2 * 5 * 5
        self.w3 = he((self.flat, FC), self.flat)
        self.b3 = np.zeros(FC, dtype = np.float32)
        self.w4 = he((FC, 10), FC)
        self.b4 = np.zeros(10, dtype = np.float32)

    def forward(self, x, train = True):
        c = {}
        a1, c["cols1"] = conv_forward(x, self.w1, self.b1, K)
        r1 = np.maximum(a1, 0.0)
        p1, c["pool1"] = maxpool_forward(r1)
        a2, c["cols2"] = conv_forward(p1, self.w2, self.b2, K)
        r2 = np.maximum(a2, 0.0)
        p2, c["pool2"] = maxpool_forward(r2)
        fl = p2.reshape(x.shape[0], -1)
        z3 = fl @ self.w3 + self.b3
        r3 = np.maximum(z3, 0.0)
        z4 = r3 @ self.w4 + self.b4
        if not train:
            return z4
        c.update(x = x, a1 = a1, p1 = p1, a2 = a2, p2shape = p2.shape,
                 fl = fl, z3 = z3, r3 = r3)
        return z4, c

    def backward(self, c, dz4, lr):
        dw4 = c["r3"].T @ dz4
        db4 = dz4.sum(0)
        dr3 = dz4 @ self.w4.T
        dz3 = dr3 * (c["z3"] > 0)
        dw3 = c["fl"].T @ dz3
        db3 = dz3.sum(0)
        dfl = dz3 @ self.w3.T
        dp2 = dfl.reshape(c["p2shape"])
        dr2 = maxpool_backward(dp2, c["pool2"])
        da2 = dr2 * (c["a2"] > 0)
        dp1, dw2, db2 = conv_backward(da2, c["cols2"], c["p1"].shape, self.w2, K)
        dr1 = maxpool_backward(dp1, c["pool1"])
        da1 = dr1 * (c["a1"] > 0)
        _, dw1, db1 = conv_backward(da1, c["cols1"], c["x"].shape, self.w1, K)

        for p, g in ((self.w1, dw1), (self.b1, db1), (self.w2, dw2),
                     (self.b2, db2), (self.w3, dw3), (self.b3, db3),
                     (self.w4, dw4), (self.b4, db4)):
            p -= lr * g.astype(np.float32)


def train(net, x, y, xte, yte, epochs, batch, lr, rng):
    n = len(x)
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n - batch + 1, batch):
            idx = order[s:s + batch]
            xb, yb = x[idx], y[idx]
            logits, cache = net.forward(xb)
            # Softmax cross-entropy, and its gradient.
            shifted = logits - logits.max(1, keepdims = True)
            exp = np.exp(shifted)
            prob = exp / exp.sum(1, keepdims = True)
            total += -np.log(prob[np.arange(len(yb)), yb] + 1e-12).mean()
            dz = prob
            dz[np.arange(len(yb)), yb] -= 1.0
            net.backward(cache, (dz / len(yb)).astype(np.float32), lr)
        acc = accuracy(net, xte, yte)
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / (n // batch):.4f}  "
              f"test acc {100 * acc:.2f}%")
    return accuracy(net, xte, yte)


def accuracy(net, x, y, chunk = 1000):
    hits = 0
    for s in range(0, len(x), chunk):
        hits += (net.forward(x[s:s + chunk], train = False).argmax(1)
                 == y[s:s + chunk]).sum()
    return hits / len(x)


# --- ONNX ------------------------------------------------------------------

def export_onnx(net, path: Path):
    """The trained weights as a graph Deeploy can compile, batch size 1."""
    init, nodes = [], []

    def add_init(name, arr):
        init.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))

    add_init("w1", net.w1)
    add_init("b1", net.b1)
    add_init("w2", net.w2)
    add_init("b2", net.b2)
    # ONNX Gemm computes A @ B, so the weights go in as (in, out) directly.
    add_init("w3", net.w3)
    # Deeploy's GEMM kernel takes C as a full M x O tensor; M is 1 here.
    add_init("b3", net.b3.reshape(1, FC))
    add_init("w4", net.w4)
    add_init("b4", net.b4.reshape(1, 10))
    add_init("flat_shape", np.array([1, net.flat], dtype = np.int64))

    # dilations is not optional as far as Deeploy's Conv parser is concerned:
    # it reads the attribute rather than defaulting it, and a node without one
    # fails to bind.
    conv = lambda n, i, w, b, o: helper.make_node(
        "Conv", [i, w, b], [o], name = n, kernel_shape = [K, K],
        strides = [1, 1], pads = [0, 0, 0, 0], group = 1, dilations = [1, 1])
    # ceil_mode likewise: the parser reads it rather than defaulting it.
    pool = lambda n, i, o: helper.make_node(
        "MaxPool", [i], [o], name = n, kernel_shape = [2, 2], strides = [2, 2],
        pads = [0, 0, 0, 0], ceil_mode = 0)

    nodes += [
        conv("conv1", "input", "w1", "b1", "c1"),
        helper.make_node("Relu", ["c1"], ["r1"], name = "relu1"),
        pool("pool1", "r1", "p1"),
        conv("conv2", "p1", "w2", "b2", "c2"),
        helper.make_node("Relu", ["c2"], ["r2"], name = "relu2"),
        pool("pool2", "r2", "p2"),
        helper.make_node("Reshape", ["p2", "flat_shape"], ["fl"], name = "flatten"),
        helper.make_node("Gemm", ["fl", "w3", "b3"], ["z3"], name = "fc1",
                         alpha = 1.0, beta = 1.0, transA = 0, transB = 0),
        helper.make_node("Relu", ["z3"], ["r3"], name = "relu3"),
        helper.make_node("Gemm", ["r3", "w4", "b4"], ["z4"], name = "fc2",
                         alpha = 1.0, beta = 1.0, transA = 0, transB = 0),
        helper.make_node("Softmax", ["z4"], ["output"], name = "softmax", axis = 1),
    ]

    graph = helper.make_graph(
        nodes, "mnist_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 28, 28])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        initializer = init)

    model = helper.make_model(
        graph, producer_name = "hetero-sim/pipeline/mnist.py",
        opset_imports = [helper.make_opsetid("", 13)])
    model.ir_version = 8
    # Deeploy's lowering passes read the shape of every tensor, including the
    # intermediates -- its NCHW-to-NHWC pass indexes them directly. Only the
    # graph input and output carry a declared shape, so infer the rest.
    model = onnx.shape_inference.infer_shapes(model)
    missing = [v.name for v in model.graph.value_info
               if not v.type.tensor_type.HasField("shape")]
    if missing:
        sys.exit(f"error: shape inference left {missing} without a shape")
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return model


# --- the C data header -----------------------------------------------------

def write_data_header(path: Path, images, labels, reference):
    """The evaluation set, as C arrays the host program loops over."""
    n, elems = images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
    flat = images.reshape(n, elems)

    out = [
        "/* Generated by pipeline/mnist.py. Do not edit.",
        " *",
        " * The images runtime/mesh/mnist_main.c classifies, their true labels,",
        " * and what onnxruntime predicts for each on the same graph -- so the",
        " * simulation can be checked against the reference model and not only",
        " * against the ground truth.",
        " */",
        "#ifndef MNIST_DATA_H",
        "#define MNIST_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define MNIST_NUM_IMAGES {n}",
        f"#define MNIST_IMAGE_ELEMS {elems}",
        "",
        f"static const float mnist_images[{n}][{elems}] = {{",
    ]
    for row in flat:
        vals = ", ".join(f"{v:.6f}f" for v in row)
        out.append(f"  {{{vals}}},")
    out += ["};", "",
            f"static const uint8_t mnist_labels[{n}] = {{"
            + ", ".join(str(int(v)) for v in labels) + "};", "",
            "/* argmax of the ONNX model on the same images, from onnxruntime. */",
            f"static const uint8_t mnist_reference[{n}] = {{"
            + ", ".join(str(int(v)) for v in reference) + "};", "",
            "#endif /* MNIST_DATA_H */", ""]
    path.write_text("\n".join(out))


# --- driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type = int, default = 64,
                    help = "test images to embed for the simulation (default: %(default)s)")
    ap.add_argument("--epochs", type = int, default = 4)
    ap.add_argument("--batch", type = int, default = 128)
    ap.add_argument("--lr", type = float, default = 0.05)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--reuse", action = "store_true",
                    help = "keep the network.onnx already in the op directory and "
                           "only rebuild the evaluation set from it, instead of "
                           "training again. Use this to change --images without "
                           "silently replacing the tracked model with new weights.")
    ap.add_argument("-o", "--out", default = str(OP_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    rng = np.random.default_rng(args.seed)

    xtr, ytr, xte, yte = load_mnist(out / "mnist.npz")
    print(f"train {xtr.shape}  test {xte.shape}")

    onnx_path = out / "network.onnx"
    net = None

    if args.reuse:
        if not onnx_path.exists():
            sys.exit(f"error: --reuse needs an existing {onnx_path}")
        print(f"reusing {onnx_path.name}, not training")
    else:
        net = Net(rng)
        print(f"training {args.epochs} epochs, batch {args.batch}, lr {args.lr}")
        acc = train(net, xtr, ytr, xte, yte, args.epochs, args.batch, args.lr, rng)
        print(f"final test accuracy over all {len(xte)} images: {100 * acc:.2f}%")
        export_onnx(net, onnx_path)

    sess = ort.InferenceSession(str(onnx_path), providers = ["CPUExecutionProvider"])
    sample = xte[:args.images]
    ort_out = np.concatenate(
        [sess.run(None, {"input": sample[i:i + 1]})[0] for i in range(len(sample))])

    if net is not None:
        # The export has to agree with what was trained, element for element.
        np_logits = net.forward(sample, train = False)
        np_prob = np.exp(np_logits - np_logits.max(1, keepdims = True))
        np_prob /= np_prob.sum(1, keepdims = True)
        drift = np.abs(ort_out - np_prob).max()
        print(f"numpy vs onnxruntime, max |difference|: {drift:.3e}")
        if drift > 1e-4:
            sys.exit("error: the exported graph does not match the trained network")

    ref = ort_out.argmax(1)
    labels = yte[:args.images]
    print(f"onnxruntime accuracy on the {len(sample)} embedded images: "
          f"{100 * (ref == labels).mean():.1f}%")

    # inputs.npz / outputs.npz drive Deeploy's single-inference codegen.
    np.savez(out / "inputs.npz", input_0 = sample[:1])
    np.savez(out / "outputs.npz", output_0 = ort_out[:1])
    write_data_header(out / "mnist_data.h", sample, labels, ref)

    print(f"\nop directory ready: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print(f"run it with:\n  python pipeline/run_hetero.py ops/mnist")


if __name__ == "__main__":
    main()
