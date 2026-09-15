#!/usr/bin/env python3
"""Keyword spotting: the application that gives both clusters work of their own.

  python pipeline/kws.py --clips 16

MNIST put every dense node on Spatz and everything else on the host; the Snitch
cluster ran zero cycles (results/mnist-hetero.json). Nothing in that network has
the shape Snitch's Xssr/Xfrep sequencers are good at, so the cost model was right
to ignore them, and the chip's second accelerator never did anything.

This application is built so that it does. It is a two-stage streaming pipeline:

  audio --> [MFCC front-end]  --> features --> [small CNN] --> keyword
             Snitch cluster                     Spatz cluster + host

The front-end is the part that suits Snitch. Its cost is dominated by a radix-2
FFT (butterflies at a stride that halves every stage -- an affine walk SSR does
natively) and a mel filterbank (40 triangular filters, each a *reduction* over a
*ragged 10-30 bin run*, which is RVV's worst case: short vl and a horizontal
vfredusum per filter). The classifier behind it is Conv/Gemm with long
unit-stride inner loops, which is exactly what Spatz already wins.

And because frames keep arriving, the front-end for clip N+1 can run while the
classifier for clip N is still going -- so both clusters are busy at once, which
is the part of "heterogeneous" the project had not yet demonstrated.

Writes ops/kws/:

  network.onnx   the classifier graph Deeploy compiles
  inputs.npz     the first test clip's features, for single-inference codegen
  outputs.npz    what onnxruntime produces for it
  kws_data.h     the raw audio, the reference MFCCs, the true labels and
                 onnxruntime's prediction -- what runtime/mesh/kws_main.c loops
                 over.  The reference MFCCs are what makes the *front-end*
                 checkable on its own, rather than only visible as a
                 misclassification.

The audio is synthesized here, procedurally, from formant templates: no
download, no new dependency, same constraint pipeline/mnist.py already respects.
That means the accuracy number says the network trained, not that the model
competes with Speech Commands literature -- what is being measured is the
architecture.  Everything is fp32, and the training core (conv/pool forward and
backward, the SGD loop) is imported from pipeline/mnist.py rather than copied.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mnist import Net, accuracy, train  # noqa: E402
from mnist import C1, C2, FC, K  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OP_DIR = ROOT / "ops" / "kws"

# --- front-end geometry ----------------------------------------------------
#
# Chosen so the front-end costs roughly what the classifier costs: if either
# side dominates, overlapping them has nothing to show.  FFT_LEN is the knob
# that moves the balance -- it is the front-end's dominant term.
SAMPLE_RATE = 16000
FRAME_LEN = 256           # 16 ms
FFT_LEN = 256             # radix-2, so a power of two
HOP = 128                 # 8 ms, 50% overlap
NB_FRAMES = 32
NB_BINS = FFT_LEN // 2 + 1
NB_MEL = 40
NB_CEP = 13
MEL_FLOOR = 1e-6          # log floor: bounds the relative error the log can
                          # amplify out of a near-zero mel energy
NB_SAMPLES = FRAME_LEN + (NB_FRAMES - 1) * HOP

# The feature map handed to the classifier: (channels, frames, cepstra).
# Frames are the row axis so the kernel writes each frame's cepstra
# contiguously -- one DMA-friendly run per frame, and the natural unit to
# slice across the cluster's compute cores.
FEAT_SHAPE = (1, NB_FRAMES, NB_CEP)
FEAT_ELEMS = NB_FRAMES * NB_CEP

# --- the keywords ----------------------------------------------------------
#
# Each keyword is a sequence of formant "phones".  A phone is (F1, F2, F3,
# duration weight, voiced).  Voiced phones get a harmonic stack shaped by the
# formant resonances; unvoiced ones get noise shaped by the same envelope.
# The point is not realism, it is that the classes differ in ways an MFCC
# actually sees: formant positions and their movement over time.
KEYWORDS = {
    "yes":   [(500, 2000, 2800, 0.35, True), (300, 2300, 3000, 0.30, True),
              (600, 1700, 2500, 0.35, False)],
    "no":    [(450, 1100, 2500, 0.45, True), (400,  800, 2400, 0.55, True)],
    "up":    [(350,  900, 2400, 0.55, True), (700, 1200, 2300, 0.45, False)],
    "down":  [(600, 1200, 2500, 0.30, True), (450,  900, 2300, 0.40, True),
              (300, 1600, 2600, 0.30, True)],
    "left":  [(500, 1800, 2700, 0.30, True), (400, 2100, 2900, 0.35, True),
              (600, 1500, 2400, 0.35, False)],
    "right": [(600, 1300, 2200, 0.30, False), (400, 2000, 2800, 0.40, True),
              (500, 1600, 2500, 0.30, False)],
    "on":    [(650, 1100, 2400, 0.50, True), (350,  950, 2300, 0.50, True)],
    "off":   [(700, 1150, 2500, 0.45, True), (500, 1400, 2600, 0.55, False)],
    "stop":  [(550, 1500, 2600, 0.30, False), (450,  850, 2200, 0.35, True),
              (600, 1300, 2400, 0.35, False)],
    "go":    [(400, 1700, 2500, 0.35, True), (500,  950, 2300, 0.65, True)],
}
LABELS = sorted(KEYWORDS)
NB_CLASSES = len(LABELS)


def _formant_gain(freqs, formants, bandwidth = 110.0):
    """Spectral envelope of a set of resonances, evaluated at `freqs`.

    A Lorentzian per formant rather than a real filter: it gives the harmonic
    stack the peaks an MFCC keys on, without a filter design step.
    """
    gain = np.full_like(freqs, 0.02)
    for f0 in formants:
        gain = gain + 1.0 / (1.0 + ((freqs - f0) / bandwidth) ** 2)
    return gain


def synth_clip(phones, rng):
    """One utterance of a keyword, with everything a speaker would vary."""
    f0 = rng.uniform(85.0, 200.0)
    onset = int(rng.integers(0, NB_SAMPLES // 6))
    gain = rng.uniform(0.6, 1.0)

    weights = np.array([p[3] for p in phones], dtype = np.float64)
    weights = weights * rng.uniform(0.85, 1.15, size = len(phones))
    weights /= weights.sum()
    span = NB_SAMPLES - onset - int(rng.integers(0, NB_SAMPLES // 8))
    lengths = np.maximum((weights * span).astype(int), 8)

    clip = np.zeros(NB_SAMPLES, dtype = np.float64)
    pos = onset
    for (f1, f2, f3, _, voiced), n in zip(phones, lengths):
        if pos >= NB_SAMPLES:
            break
        n = min(n, NB_SAMPLES - pos)
        t = np.arange(n) / SAMPLE_RATE
        # Formants jitter per utterance, the way a speaker's vocal tract differs.
        formants = [f * rng.uniform(0.92, 1.08) for f in (f1, f2, f3)]

        if voiced:
            harmonics = np.arange(1, int(SAMPLE_RATE / 2 / f0))
            freqs = harmonics * f0
            amps = _formant_gain(freqs, formants) / harmonics
            phase = rng.uniform(0, 2 * np.pi, size = len(harmonics))
            seg = (amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * t[None, :]
                                          + phase[:, None])).sum(0)
        else:
            # Shape noise with the same envelope, in the frequency domain.
            noise = rng.standard_normal(n)
            spec = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
            seg = np.fft.irfft(spec * _formant_gain(freqs, formants), n = n)

        # Raised-cosine attack and decay, so phone boundaries are not clicks.
        ramp = max(1, n // 8)
        env = np.ones(n)
        env[:ramp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
        env[-ramp:] = env[:ramp][::-1]
        peak = np.abs(seg).max()
        clip[pos:pos + n] += gain * env * seg / (peak if peak > 0 else 1.0)
        pos += n

    snr = rng.uniform(10.0, 30.0)
    power = np.mean(clip ** 2)
    if power > 0:
        clip = clip + rng.standard_normal(NB_SAMPLES) * np.sqrt(
            power / (10.0 ** (snr / 10.0)))
    peak = np.abs(clip).max()
    return (clip / peak if peak > 0 else clip).astype(np.float32)


def make_dataset(n_per_class, rng):
    x = np.empty((n_per_class * NB_CLASSES, NB_SAMPLES), dtype = np.float32)
    y = np.empty(n_per_class * NB_CLASSES, dtype = np.int64)
    i = 0
    for label, name in enumerate(LABELS):
        for _ in range(n_per_class):
            x[i] = synth_clip(KEYWORDS[name], rng)
            y[i] = label
            i += 1
    order = rng.permutation(len(x))
    return x[order], y[order]


# --- the MFCC front-end, in numpy ------------------------------------------
#
# This is the golden reference the Snitch kernel is checked against, so it is
# written to be *reproducible in C*, not to be clever: the same window, the
# same power spectrum, the same banded filterbank, the same log floor, the
# same orthonormal DCT-II.

def hann_window():
    """Periodic Hann -- the one that satisfies COLA at 50% overlap."""
    n = np.arange(FRAME_LEN)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / FRAME_LEN)).astype(np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(low_hz = 20.0, high_hz = None):
    """Triangular mel filters as a dense (NB_MEL, NB_BINS) matrix."""
    high_hz = high_hz if high_hz is not None else SAMPLE_RATE / 2
    edges = _mel_to_hz(np.linspace(_hz_to_mel(low_hz), _hz_to_mel(high_hz),
                                   NB_MEL + 2))
    bins = np.fft.rfftfreq(FFT_LEN, 1.0 / SAMPLE_RATE)
    fb = np.zeros((NB_MEL, NB_BINS), dtype = np.float32)
    for m in range(NB_MEL):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        left = (bins - lo) / max(mid - lo, 1e-9)
        right = (hi - bins) / max(hi - mid, 1e-9)
        fb[m] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def band_filterbank(fb):
    """The same filters, stored banded: (coeffs, start, length) per filter.

    Dense would be 40x129 floats; banded is ~3 KB and, more to the point, it
    is what lets the Snitch kernel walk each filter as one SSR stream instead
    of multiplying by zeros.
    """
    coeff, start, length = [], [], []
    for row in fb:
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            start.append(0)
            length.append(0)
            continue
        s, e = int(nz[0]), int(nz[-1]) + 1
        start.append(s)
        length.append(e - s)
        coeff.extend(row[s:e].tolist())
    return (np.array(coeff, dtype = np.float32),
            np.array(start, dtype = np.uint16),
            np.array(length, dtype = np.uint16))


def dct_matrix():
    """Orthonormal DCT-II, (NB_CEP, NB_MEL), applied to the log-mel vector."""
    m = np.arange(NB_MEL)
    d = np.empty((NB_CEP, NB_MEL), dtype = np.float64)
    for c in range(NB_CEP):
        d[c] = np.cos(np.pi * c * (2 * m + 1) / (2 * NB_MEL))
    d *= np.sqrt(2.0 / NB_MEL)
    d[0] *= np.sqrt(0.5)
    return d.astype(np.float32)


def mfcc(audio, window, fb, dct, chunk = 256):
    """(N, NB_SAMPLES) audio -> (N, 1, NB_FRAMES, NB_CEP) features.

    Batched, because the intermediate spectra for a whole training set do not
    fit comfortably in memory.
    """
    out = np.empty((len(audio),) + FEAT_SHAPE, dtype = np.float32)
    offsets = np.arange(NB_FRAMES) * HOP
    idx = offsets[:, None] + np.arange(FRAME_LEN)[None, :]
    for s in range(0, len(audio), chunk):
        block = audio[s:s + chunk]
        frames = block[:, idx] * window                     # (B, T, FRAME_LEN)
        spec = np.fft.rfft(frames, n = FFT_LEN, axis = -1)  # (B, T, NB_BINS)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = power @ fb.T                                  # (B, T, NB_MEL)
        logmel = np.log(np.maximum(mel, MEL_FLOOR))
        out[s:s + chunk] = (logmel @ dct.T).reshape(-1, *FEAT_SHAPE)
    return out


# --- the classifier --------------------------------------------------------
#
# The same topology as the MNIST network, so it reuses that training code
# unchanged and every node's working set stays far inside the cluster
# scratchpad budget (85,504 B, pipeline/hetero_platform/engines.py).
#
#   1x32x13 --Conv 3x3, 8--> 8x30x11 --ReLU--> --MaxPool 2--> 8x15x5
#           --Conv 3x3,16--> 16x13x3 --ReLU--> --MaxPool 2--> 16x6x1
#           --Flatten--> 96 --Gemm--> 32 --ReLU--> --Gemm--> 10 --Softmax

def _shape_after(h, w):
    h, w = h - K + 1, w - K + 1          # conv1
    h, w = h // 2, w // 2                # pool1
    h, w = h - K + 1, w - K + 1          # conv2
    return h // 2, w // 2                # pool2


POOL_H, POOL_W = _shape_after(NB_FRAMES, NB_CEP)
FLAT = C2 * POOL_H * POOL_W


class KwsNet(Net):
    """mnist.Net with this application's input shape and class count.

    Only __init__ differs: forward, backward and the SGD loop are shape-generic
    and are used exactly as MNIST uses them.
    """

    def __init__(self, rng):
        def he(shape, fan_in):
            return (rng.standard_normal(shape)
                    * np.sqrt(2.0 / fan_in)).astype(np.float32)

        self.w1 = he((C1, 1, K, K), 1 * K * K)
        self.b1 = np.zeros(C1, dtype = np.float32)
        self.w2 = he((C2, C1, K, K), C1 * K * K)
        self.b2 = np.zeros(C2, dtype = np.float32)
        self.flat = FLAT
        self.w3 = he((self.flat, FC), self.flat)
        self.b3 = np.zeros(FC, dtype = np.float32)
        self.w4 = he((FC, NB_CLASSES), FC)
        self.b4 = np.zeros(NB_CLASSES, dtype = np.float32)


def export_onnx(net, path: Path):
    """The trained classifier as a graph Deeploy can compile, batch size 1."""
    init, nodes = [], []

    def add_init(name, arr):
        init.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))

    add_init("w1", net.w1)
    add_init("b1", net.b1)
    add_init("w2", net.w2)
    add_init("b2", net.b2)
    add_init("w3", net.w3)
    # Deeploy's GEMM kernel takes C as a full M x O tensor; M is 1 here.
    add_init("b3", net.b3.reshape(1, FC))
    add_init("w4", net.w4)
    add_init("b4", net.b4.reshape(1, NB_CLASSES))
    add_init("flat_shape", np.array([1, net.flat], dtype = np.int64))

    # dilations and ceil_mode are read by Deeploy's parsers rather than
    # defaulted; a node without them fails to bind (see pipeline/mnist.py).
    conv = lambda n, i, w, b, o: helper.make_node(
        "Conv", [i, w, b], [o], name = n, kernel_shape = [K, K],
        strides = [1, 1], pads = [0, 0, 0, 0], group = 1, dilations = [1, 1])
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
        nodes, "kws_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [1, 1, NB_FRAMES, NB_CEP])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, NB_CLASSES])],
        initializer = init)

    model = helper.make_model(
        graph, producer_name = "hetero-sim/pipeline/kws.py",
        opset_imports = [helper.make_opsetid("", 13)])
    model.ir_version = 8
    # Deeploy's lowering passes read every intermediate tensor's shape.
    model = onnx.shape_inference.infer_shapes(model)
    missing = [v.name for v in model.graph.value_info
               if not v.type.tensor_type.HasField("shape")]
    if missing:
        sys.exit(f"error: shape inference left {missing} without a shape")
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return model


# --- the C data header -----------------------------------------------------

def _c_floats(values, per_line = 8):
    """Float literals, wrapped -- one 4000-element array on one line makes for
    a header no compiler and no reader enjoys."""
    lines, row = [], []
    for i, v in enumerate(values):
        row.append(f"{v:.8e}f")
        if len(row) == per_line or i == len(values) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    return lines


def write_data_header(path: Path, audio, features, labels, reference,
                      window, band, dct):
    """Everything runtime/mesh/kws_main.c needs, as C arrays.

    Beyond the evaluation set this also carries the front-end's *constants* --
    window, banded filterbank, twiddles, DCT -- so the numpy reference and the
    on-chip kernel provably use the same ones rather than two independently
    derived tables that happen to agree.
    """
    coeff, start, length = band
    n = len(audio)

    # Twiddle factors for a decimation-in-time radix-2 FFT: cos/sin of
    # -2*pi*k/FFT_LEN for k < FFT_LEN/2, interleaved, which is the order the
    # kernel walks them in.
    k = np.arange(FFT_LEN // 2)
    ang = -2.0 * np.pi * k / FFT_LEN
    twiddles = np.empty(FFT_LEN, dtype = np.float32)
    twiddles[0::2] = np.cos(ang)
    twiddles[1::2] = np.sin(ang)

    out = [
        "/* Generated by pipeline/kws.py. Do not edit.",
        " *",
        " * The clips runtime/mesh/kws_main.c classifies, the MFCC features the",
        " * numpy reference computes for them, their true labels, and what",
        " * onnxruntime predicts on the same graph.",
        " *",
        " * kws_mfcc_ref is what makes the front-end checkable on its own: a",
        " * wrong FFT shows up as a feature mismatch, not merely as a digit the",
        " * network got wrong.",
        " *",
        " * The front-end constants below are the same tables the numpy",
        " * reference used, emitted rather than recomputed, so the two cannot",
        " * drift apart.",
        " */",
        "#ifndef KWS_DATA_H",
        "#define KWS_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define KWS_NUM_CLIPS      {n}",
        f"#define KWS_NUM_CLASSES    {NB_CLASSES}",
        f"#define KWS_SAMPLE_RATE    {SAMPLE_RATE}",
        f"#define KWS_NB_SAMPLES     {NB_SAMPLES}",
        f"#define KWS_FRAME_LEN      {FRAME_LEN}",
        f"#define KWS_FFT_LEN        {FFT_LEN}",
        f"#define KWS_HOP            {HOP}",
        f"#define KWS_NB_FRAMES      {NB_FRAMES}",
        f"#define KWS_NB_BINS        {NB_BINS}",
        f"#define KWS_NB_MEL         {NB_MEL}",
        f"#define KWS_NB_CEP         {NB_CEP}",
        f"#define KWS_FEAT_ELEMS     {FEAT_ELEMS}",
        f"#define KWS_MEL_COEFFS     {len(coeff)}",
        "",
        "/* --- front-end constants ------------------------------------------ */",
        "",
        f"static const float kws_window[{FRAME_LEN}] = {{",
    ]
    out += _c_floats(window)
    out += ["};", "",
            "/* cos/sin of -2*pi*k/FFT_LEN, interleaved, k < FFT_LEN/2. */",
            f"static const float kws_twiddles[{FFT_LEN}] = {{"]
    out += _c_floats(twiddles)
    out += ["};", "",
            "/* The triangular mel filters, stored banded: filter m covers bins",
            "   [start[m], start[m]+len[m]) with the coefficients at the running",
            "   offset sum(len[0..m-1]). */",
            f"static const float kws_mel_coeff[{len(coeff)}] = {{"]
    out += _c_floats(coeff)
    out += ["};", "",
            f"static const uint16_t kws_mel_start[{NB_MEL}] = {{"
            + ", ".join(str(int(v)) for v in start) + "};",
            f"static const uint16_t kws_mel_len[{NB_MEL}] = {{"
            + ", ".join(str(int(v)) for v in length) + "};", "",
            "/* Orthonormal DCT-II, row-major (NB_CEP x NB_MEL). */",
            f"static const float kws_dct[{NB_CEP * NB_MEL}] = {{"]
    out += _c_floats(dct.reshape(-1))
    out += ["};", "",
            "/* --- the evaluation set ------------------------------------------ */",
            "",
            f"static const float kws_audio[{n}][{NB_SAMPLES}] = {{"]
    for clip in audio:
        out.append("  {")
        out += ["  " + line for line in _c_floats(clip)]
        out.append("  },")
    out += ["};", "",
            "/* The numpy front-end's output for the same clips. */",
            f"static const float kws_mfcc_ref[{n}][{FEAT_ELEMS}] = {{"]
    for feat in features.reshape(n, FEAT_ELEMS):
        out.append("  {")
        out += ["  " + line for line in _c_floats(feat)]
        out.append("  },")
    out += ["};", "",
            f"static const uint8_t kws_labels[{n}] = {{"
            + ", ".join(str(int(v)) for v in labels) + "};", "",
            "/* argmax of the ONNX model on the same features, from onnxruntime. */",
            f"static const uint8_t kws_reference[{n}] = {{"
            + ", ".join(str(int(v)) for v in reference) + "};", "",
            "static const char *const kws_class_names[" + str(NB_CLASSES) + "] = {"
            + ", ".join(f'"{name}"' for name in LABELS) + "};", "",
            "#endif /* KWS_DATA_H */", ""]
    path.write_text("\n".join(out))


# --- driver ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type = int, default = 16,
                    help = "test clips to embed for the simulation (default: %(default)s)")
    ap.add_argument("--train-per-class", type = int, default = 300)
    ap.add_argument("--test-per-class", type = int, default = 60)
    ap.add_argument("--epochs", type = int, default = 12)
    ap.add_argument("--batch", type = int, default = 64)
    ap.add_argument("--lr", type = float, default = 0.02)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--reuse", action = "store_true",
                    help = "keep the network.onnx already in the op directory and "
                           "only rebuild the evaluation set from it, instead of "
                           "training again")
    ap.add_argument("-o", "--out", default = str(OP_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)
    rng = np.random.default_rng(args.seed)

    window, fb, dct = hann_window(), mel_filterbank(), dct_matrix()
    band = band_filterbank(fb)
    print(f"front-end: {NB_FRAMES} frames of {FRAME_LEN} @ hop {HOP}, "
          f"FFT {FFT_LEN}, {NB_MEL} mel -> {NB_CEP} cepstra "
          f"({len(band[0])} banded coefficients vs {NB_MEL * NB_BINS} dense)")

    print(f"synthesizing {NB_CLASSES} keywords: {', '.join(LABELS)}")
    xtr_a, ytr = make_dataset(args.train_per_class, rng)
    xte_a, yte = make_dataset(args.test_per_class, rng)
    xtr = mfcc(xtr_a, window, fb, dct)
    xte = mfcc(xte_a, window, fb, dct)
    print(f"train {xtr.shape}  test {xte.shape}")

    onnx_path = out / "network.onnx"
    net = None

    if args.reuse:
        if not onnx_path.exists():
            sys.exit(f"error: --reuse needs an existing {onnx_path}")
        print(f"reusing {onnx_path.name}, not training")
    else:
        net = KwsNet(rng)
        print(f"training {args.epochs} epochs, batch {args.batch}, lr {args.lr}")
        acc = train(net, xtr, ytr, xte, yte, args.epochs, args.batch, args.lr, rng)
        print(f"final test accuracy over all {len(xte)} clips: {100 * acc:.2f}%")
        export_onnx(net, onnx_path)

    sess = ort.InferenceSession(str(onnx_path), providers = ["CPUExecutionProvider"])
    sample_a = xte_a[:args.clips]
    sample = xte[:args.clips]
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
    labels = yte[:args.clips]
    print(f"onnxruntime accuracy on the {len(sample)} embedded clips: "
          f"{100 * (ref == labels).mean():.1f}%")

    np.savez(out / "inputs.npz", input_0 = sample[:1])
    np.savez(out / "outputs.npz", output_0 = ort_out[:1])
    write_data_header(out / "kws_data.h", sample_a, sample, labels, ref,
                      window, band, dct)

    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nop directory ready: {rel}")
    print("run it with:\n  make kws")


if __name__ == "__main__":
    main()
