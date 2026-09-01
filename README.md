# hetero-sim — CVA6 / Snitch / Spatz ONNX-op benchmark pipeline

GVSoC simulation of three heterogeneous RISC-V core types, driven from ONNX via Deeploy:

```
ONNX op + inputs  ──Deeploy──►  C code  ──riscv-gcc──►  3 ELFs  ──GVSoC──►  per-core metrics
```

| core   | what is simulated                                    | GVSoC target | ISA               |
|--------|------------------------------------------------------|--------------|-------------------|
| cva6   | CVA6 64-bit host core, L1 I$/D$ + L2 + DRAM          | `cva6_real` (in `targets/`) | rv64imafdc |
| snitch | Snitch integer core + FP subsystem, SSR streamers and FREP sequencer, data in cluster TCDM | `snitch_real` (in `targets/`) | rv32imafd + Xssr/Xfrep |
| spatz  | Snitch + Spatz vector unit (4 lanes), RVV kernels    | `spatz_real` (in `targets/`) | rv32imafd + V |

Each core runs the best code the pipeline can give it, not the same code: CVA6
scalar `-O3`, and both accelerator cores' FP kernels hand-written against what
they actually have — Snitch against its two custom extensions, Spatz against
RVV. See [The Snitch FP extensions](#the-snitch-fp-extensions-xssr--xfrep) and
[The Spatz RVV kernels](#the-spatz-rvv-kernels).

**The memory system is simulated,** not assumed away: cache misses, DRAM latency and
refill bandwidth all land in the cycle counts. `--memory ideal` switches back to the
zero-latency targets (`cva6_ideal`, `snitch`, `spatz`), where every access completes
in a single cycle and a cycle count is a pure compute cost. See
[The memory system](#the-memory-system).

## Setup

Dependencies (GVSoC, Deeploy, the RISC-V toolchain) are large and are not tracked in
git. One command fetches them at pinned commits, applies the local GVSoC patches, and
builds the simulator (including the targets in `targets/` and their cache model):

```bash
./setup.sh
```

It needs `uv`, `cmake`, `git`, `curl`, and a C++ toolchain; re-running it is safe.

### Docker

`Dockerfile` runs the same `./setup.sh` inside an Ubuntu 22.04 image, so it works the
same on any platform Docker runs on:

```bash
docker build -t hetero-sim .
docker run --rm hetero-sim                                                    # default op
docker run --rm hetero-sim python pipeline/run.py Tests/Kernels/FP32/MatMul   # any op
docker run --rm -it hetero-sim bash                                           # shell
```

The image bakes in the built GVSoC targets, so no fetching happens at container start;
only the `docker build` needs network access.

## Usage

Run a bundled Deeploy single-op test (`network.onnx` + `inputs.npz` + `outputs.npz`):

```bash
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/GEMM/Regular
```

Bring your own ONNX op (expected outputs computed with onnxruntime):

```bash
.venv/bin/python pipeline/make_op.py myop.onnx myinputs.npz -o ops/myop
.venv/bin/python pipeline/run.py ops/myop
```

Output: a per-core table (cycles, speedup vs CVA6, numerical error vs the ONNX
reference), the cache counters behind it, and the same numbers as JSON in `results/`.

Swap Spatz's hand-written kernels for whatever GCC autovectorizes:

```bash
.venv/bin/python pipeline/run.py <op> --spatz-kernels autovec
```

Results for the non-default choice land in `results/<op>-spatz-autovec.json`.

Compare the two memory models on the same op:

```bash
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/MatMul                  # modelled memory
.venv/bin/python pipeline/run.py Tests/Kernels/FP32/MatMul --memory ideal   # zero-latency
```

Results land in `results/<op>.json` and `results/<op>-ideal.json`.

### Debugging a run

`--debug` (or `-d`, or `HES_DEBUG=1`, or `make run DEBUG=1`) traces every command
the pipeline runs — Deeploy codegen, each compile, the link, GVSoC — followed by
the files that command generated, so a failure can be reproduced by hand:

```bash
.venv/bin/python pipeline/run.py ops/mymatmul --debug
```

```
[dbg] $ .venv/bin/python generateNetwork.py -t ops/mymatmul -p Generic -d work/mymatmul/gen
[dbg]   (cwd: deps/deeploy/DeeployTest)
[dbg]   exit=0 in 1.0s
[dbg]   -> work/mymatmul/gen/Network.c  (25.0 KiB)
[dbg]   -> work/mymatmul/gen/testinputs.h  (23.6 KiB)
...
[dbg] $ toolchains/.../riscv-none-elf-gcc -march=rv64imafdc_zicsr_zifencei ... -o work/mymatmul/cva6/net.elf
[dbg]   exit=0 in 0.0s
[dbg]   -> work/mymatmul/cva6/net.elf  (27.2 KiB)
```

The trace goes to stderr, so stdout still carries only the report:
`run.py <op> --debug 2>trace.log`. Files the driver itself writes (`sim.log`,
the results JSON) are traced too. Commands that produce nothing are shown as
`(not created)`, which is what a failed compile or a timed-out simulation looks
like.

## Results

Cycles for the timed op, one core of each type, on the default modelled-memory
targets. `×` is against CVA6; the fastest core in each row is in bold. Every
row is verified against the ONNX reference and reproduced by
`results/<op>.json`.

| operator | shape | cva6 | snitch | spatz |
|----------|-------|------|--------|-------|
| Add                | 64 × fp32, elementwise                       | **863**  | 1053 (0.8×)        | 1035 (0.8×)      |
| MatMul             | 2 × (16×32 · 32×8) fp32                      | 119.0k   | 12.6k (9.4×)       | **6.8k (17.5×)** |
| MatMul (custom op) | 32×32×32 fp32                                | 477.5k   | 43.1k (11.1×)      | **7.5k (63.3×)** |
| GEMM               | 32×32×32 fp32 + bias                         | 489.0k   | 43.5k (11.2×)      | **8.4k (58.5×)** |
| GEMM (int8)        | 32×32×32, s8·s8 → s32                        | 575.9k   | 451.7k (1.3×)      | **84.1k (6.8×)** |
| Conv2D + bias      | 2×64×32 fp32, 4 filters 2×8×8, stride 2×4    | 1.86M    | **173.9k (10.7×)** | 457.0k (4.1×)    |
| Softmax            | 512 fp32, 32 rows of 16                      | 36.5k    | 51.0k (0.7×)       | **32.5k (1.1×)** |

Reading it:

- **Spatz takes every fp32 matmul-shaped op**, by 17-63× over CVA6 and 2-6×
  over Snitch, once its kernels are hand-written against RVV rather than
  autovectorized — see [The Spatz RVV kernels](#the-spatz-rvv-kernels). While
  Spatz ran compiler output this table had Snitch ahead on all of them.
- **Snitch still takes Conv2D**, and its extensions are what put it 10.7× over
  CVA6 everywhere: Xssr removes the loads and the address arithmetic and Xfrep
  removes the loop, which leaves an FMA rate the stream bandwidth sets — see
  [The Snitch FP extensions](#the-snitch-fp-extensions-xssr--xfrep). Spatz has
  no hand-written convolution yet, which is the obvious next kernel.
- **Spatz takes int8 GEMM**, by 5.4× over Snitch: SSR feeds the FP regfile and
  FREP replays FP instructions, so neither helps an integer reduction, while
  RVV vectorizes it directly.
- **Softmax is `expf`-bound** on all three, so they land within 1.6× of each
  other and no amount of streaming or vectorizing moves it. Snitch is last
  because the libm code is scalar work on its small integer core, and FREP
  replays FP instructions from a 16-entry buffer, not a libm call.
- **Add is too small to say anything** — 64 elements, where the result is
  dominated by call and loop overhead rather than the 64 additions.

The dividing line between the two accelerated cores is not integer versus
float. It is whether the work reduces to a dense FP multiply-accumulate loop
that Snitch's sequencer can replay: if it does, Snitch is competitive; if it
does not — an integer reduction, a libm call — Spatz is the better cluster
whatever the datatype.

`--memory ideal` (zero-latency memory, `results/<op>-ideal.json`) keeps the same
ordering everywhere except Add, where Spatz's 635 cycles edge out CVA6's 715
once its instruction fetches stop paying DRAM. The cost of the modelled memory
system per core is broken down under
[What it costs](#what-it-costs).

## Keyword spotting: using both clusters at once

The table above, and the MNIST network built on it, measure one thing: which
core is fastest at a given operator. That is operator selection, and it leaves
two things on the table. MNIST gives the Snitch cluster **zero** cycles — the
cost model correctly sends every dense node to Spatz — and `hes_offload()`
blocks, so even when two clusters could work at once they take turns and the
total is a sum rather than a maximum.

`ops/kws` is an application built to need both. It is keyword spotting on
synthetic speech, and it has two stages that want different hardware and are
available at the same time:

```
 clip N+1 ─┐
           ▼
   [SNITCH cluster]  window → FFT → |·|² → mel → log → DCT      one job, 8 cores,
           │         (HES_K_MFCC_FP32)                          sliced by frame
           ▼  features 32×13   (main memory, double-buffered)
   [SPATZ cluster]   Conv 3×3 → Conv 3×3 → Gemm → Gemm          Deeploy-generated,
           │                                                    existing kernels
           ▼  logits
   [CVA6]            ReLU / MaxPool / Softmax / argmax / control

 t0:  snitch(clip 0)
 t1:  snitch(clip 1)  ||  spatz+cva6(clip 0)      ← both clusters live
 t2:  snitch(clip 2)  ||  spatz+cva6(clip 1)
```

Because clips keep arriving, the front-end for clip N+1 is independent of the
classifier for clip N. `runtime/mesh/kws_main.c` posts the one before running
the other, and collects it after. `RunNetwork()` offloads its Conv and Gemm
nodes to a *different* cluster's mailbox, so the overlap needs nothing from the
generated code — only that `hes_offload()` be split into `hes_post()` +
`hes_wait()`, which is all `runtime/mesh/hes_host.c` changed. Each cluster
already had its own `seq`/`done_seq` pair; one job per cluster in flight was
always expressible, `hes_offload()` just never used it.

Run it with `make kws`. 16 clips, all numbers from the modelled-memory board:

| | cycles/clip | vs. pipelined |
|---|---|---|
| front-end on snitch, classifier on spatz | **256,652** | — |
| the same work, serial (`SERIAL=1`) | 468,232 | 1.82× slower |
| front-end on spatz, classifier on snitch (`FE=spatz PIN=snitch`) | 269,434 | 1.05× slower |
| classifier on the host (`PIN=cva6`, 8 clips) | 1,539,740 | 6.0× slower |

and, for the first time in this repository, all three engines do real work:

| engine | cycles (16 clips) | share | |
|---|---|---|---|
| snitch | 3,615,971 | 50.2% | the MFCC front-end |
| spatz | 1,831,164 | 25.4% | Conv and Gemm |
| cva6 | 1,755,342 | 24.4% | ReLU, MaxPool, Softmax, control |

against MNIST's `spatz 53% / cva6 47% / snitch 0%`. 93.4% of the front-end's
cycles overlap the classifier and cost no wall time at all.

### What the placement comparison says

The two rows in the middle of the first table are the same work with the two
stages swapped between the clusters, and the interesting part is that Spatz is
faster at **both** stages and still loses:

| stage | on snitch | on spatz | |
|---|---|---|---|
| MFCC front-end | 226.0k/clip | **185.7k/clip** | spatz, 1.22× |
| classifier (cluster share) | 129.8k/clip | **114.4k/clip** | spatz, 1.13× |

The two stages have to be on different clusters — a cluster's mailbox holds one
descriptor, so posting the next front-end to the cluster the classifier is using
would overwrite a job in flight, which `hes_post()` refuses and
`pipeline/run_hetero.py` catches before the build. So the choice is which stage
gets Spatz, and a pipeline's period is the *maximum* of its stages, not their
sum. The classifier is the slower stage under either assignment, so Spatz
belongs on the classifier and the front-end goes to Snitch by elimination.
"Put each stage on the core that runs it fastest" is not the rule; "give the
better core to the stage that sets the period" is.

### What the SSR front-end kernel does and does not buy

`runtime/snitch/kernels/mfcc_fp32_ssr.c` streams two of the four stages, and
they are the two the access-pattern argument rests on:

- the **mel filterbank** — 40 reductions, each over its own ragged 10–30 bin
  run. Short vectors *and* a horizontal reduction per filter is the worst case
  for RVV; for SSR the data-dependent trip count is just a register, and the
  accumulators stay live across the whole filter.
- the **DCT-II** — a small dense matrix-vector product, the same block shape as
  the GEMM kernel's inner loop.

It is worth 8.2% of the front-end (992.5k → 911.6k cycles over 4 clips), and no
more, because the FFT dominates and is **not** streamed. Its butterfly wants
four reads and four writes per iteration against three data movers, so streaming
it means splitting it into passes through a scratch buffer, and whether that
costs less than the loads it removes is a real question rather than a rhetorical
one. It is left open. That is also why Spatz wins the front-end above: the
measurement does not support the a-priori claim that the front-end is
Snitch-shaped work, and the claim is reported as refuted rather than quietly
dropped. A hand-written RVV front-end for Spatz — which does not exist either —
would likely widen its lead further.

### How it is checked

Three independent checks, and a misheard keyword is not one of them:

- every clip's prediction against **onnxruntime** on the same graph — 16/16, the
  same discipline `mnist_main.c` uses;
- every clip's **features** against the numpy front-end in `pipeline/kws.py`,
  embedded in `ops/kws/kws_data.h`. Max |chip − numpy| = 5×10⁻⁶. Without this a
  wrong FFT would only ever surface as a misclassification;
- `make ssr-test` runs `runtime/tests/ssr_mfcc.c`, which compares the streamed
  filterbank and DCT against a scalar reference on the ragged filter widths and
  cepstra counts the application never reaches — bit-identical on all five.

The audio is synthesized procedurally from formant templates (no download, no
new dependency), so the 98.8% test accuracy says the network trained, not that
the model competes with Speech Commands. What is being measured is the chip.

## Layout

```
setup.sh               fetch + patch + build all dependencies (pinned commits)
pipeline/run.py        the pipeline driver (codegen → build → simulate → report)
pipeline/make_op.py    wrap an ONNX model + inputs into a pipeline op directory
runtime/               bare-metal glue: crt0, semihosting, linker scripts, bench main
runtime/snitch/snitch_ssr.h    Xssr/Xfrep intrinsics as raw instruction encodings
runtime/snitch/kernels/        the FP kernels Snitch overrides (MatMul/GEMM/Conv2d/MFCC)
runtime/spatz/kernels/         the FP kernels Spatz overrides, hand-written RVV
runtime/common/kernels/        first-party kernels every core gets (the MFCC front-end)
runtime/mesh/kws_main.c        keyword spotting: both clusters running at once
pipeline/kws.py                synthesize the audio, train the classifier, export ops/kws
runtime/tests/         standalone bare-metal checks (`make ssr-test` runs the SSR ones)
targets/*_real.py      GVSoC targets with the memory system modelled (the default)
targets/cva6_ideal.py  GVSoC target with zero-latency memory (--memory ideal)
targets/hetero/        memory-system parameters + the timing-cache model (C++ and generator)
deps/patches/          local fixes to GVSoC models (tracked; applied by setup.sh)
deps/gvsoc             GVSoC checkout + build           (untracked)
deps/deeploy           Deeploy, installed editable      (untracked)
toolchains/            xPack riscv-none-elf-gcc 15.2    (untracked)
work/                  per-op build + simulation artifacts, disposable (untracked)
ops/                   input op directories (onnx + npz)
results/               metrics JSON per op — committed as the verified baseline
                       (<op>.json modelled memory, <op>-ideal.json zero-latency)
```

## How it works

1. **Deeploy codegen** — `generateNetwork.py -p Generic` turns the ONNX op into
   `Network.c` (kernel calls + weights) plus `testinputs.h` / `testoutputs.h`.
2. **Per-core build** — the same generated C is cross-compiled three times.
   Snitch replaces three of the Deeploy kernels with SSR/FREP versions of its
   own and Spatz replaces two with RVV versions (both below); the Deeploy ones
   stay linked in as `<name>_generic` and still run the shapes the rewrites do
   not cover. Spatz's remaining kernels are compiled `-O3 -ffast-math` so GCC
   autovectorizes them; glue code stays scalar on both, because the Spatz
   `-march` carries `v` and GCC will otherwise emit RVV for ordinary control
   loops.
   Snitch/Spatz builds avoid the C extension because
   the GVSoC Snitch model executes compressed FP loads on the integer core,
   diverging from the decoupled FP subsystem.
3. **Simulation** — each ELF runs on its GVSoC board; `bench_main.c` reads
   `mcycle` around `RunNetwork()`, verifies outputs against the ONNX reference,
   and prints a `[HES] ...` metrics line via semihosting. Which board depends on
   `--memory`; the binary does not, so both models run the same code.
4. **Report** — `run.py` parses the metrics and the `[HES-MEM]` cache counters and
   emits the comparison table + JSON.

## The Snitch FP extensions (Xssr / Xfrep)

A Snitch core is a small integer core in front of a decoupled FP subsystem, and
two vendor extensions are what make that split pay off:

- **Xssr** — stream semantic registers. `ft0`/`ft1`/`ft2` stop being registers:
  once a data mover is configured with a loop nest (up to four levels of bounds
  and strides, plus a repeat count), every read of `ft0` pops the next element
  of the stream out of memory and every write pushes one back. Loads, address
  arithmetic and pointer bumps leave the loop.
- **Xfrep** — FP repetition. `frep.o rs1, len, 0, 0` hands the next `len`
  instructions to the FPU sequencer, which replays them `rs1`+1 times from its
  16-entry buffer. The integer core issues the body once and is then done.

An inner loop that was load / load / fmadd / bump / branch becomes one `frep.o`
over a handful of `fmadd.s`, running at whatever rate the streams can feed.

### Emitting them from GCC

`riscv-none-elf-gcc` supports neither extension (there is no upstream binutils
support for either), so [`runtime/snitch/snitch_ssr.h`](runtime/snitch/snitch_ssr.h) emits
each one as a raw encoding via `.insn`, which needs nothing from the assembler:

| instruction | emitted as |
|-------------|------------|
| `scfgwi rs1, reg<<5\|ssr` — write an SSR config register | `.insn r 0x2b, 2, reg, x0, rs1, x<ssr>` |
| `scfgri rd, reg<<5\|ssr` — read one back                 | `.insn r 0x2b, 1, reg, rd, x0, x<ssr>` |
| `frep.o rs1, len, 0, 0` — repeat the next `len` insns    | `.insn i 0x0b, 0, x1, rs1, len-1` |
| `frep.i rs1, len, 0, 0` — repeat each of them in place   | `.insn i 0x0b, 0, x0, rs1, len-1` |

The `x1`/`x0` in the frep encodings is not a register: that field carries
`stagger_mask<<1 | is_outer`, and register staggering is unused here. Enabling
the streams is an ordinary CSR write (`csrsi 0x7c0, 1`), which GCC can already
assemble.

On top of those, the header provides the same calls the Snitch runtime does —
`ssr_loop_1d`..`4d`, `ssr_repeat`, `ssr_read`, `ssr_write` — including its
convention that a bound register holds count-1 and that stride *i*+1 is the
*delta* applied when loop *i* wraps, i.e. the next step minus the distance the
inner loop already travelled.

While SSR is enabled every FP instruction touching ft0-ft2 consumes stream
elements, including anything the compiler decides to emit there. The enable and
disable therefore sit inside the same `asm volatile` block as the loop body, so
the enabled window holds exactly the instructions written by hand, and ft0-ft2
are clobbered so the register allocator stays away from them.

### The kernels

`pipeline/run.py` compiles [`runtime/snitch/kernels/`](runtime/snitch/kernels)
for the snitch core and renames the Deeploy Generic definitions it replaces to
`<name>_generic` (a `-D` applied to the Generic library alone). The generated
network keeps calling the original names and so gets the streamed version; the
streamed version calls `<name>_generic` for the shapes it does not cover.

| kernel | ft0 | ft1 | FREP body |
|--------|-----|-----|-----------|
| `MatMul_fp32_fp32_fp32`, `Gemm_fp32_fp32_fp32_fp32` | `A[i][k]`, held for 8 FMAs | 8 consecutive `B[k][j]` | 8 `fmadd.s`, replayed N times |
| `Conv2d_fp32_fp32_fp32_NCHW` | input window element, held for 4 FMAs | tap *k* of 4 filters | 4 `fmadd.s`, replayed C·P·Q times |

Both put the *reused* operand on ft0 with an SSR repeat count, so the two
streams issue 1 + unroll accesses per unroll FMAs; both unroll wide enough that
the independent accumulators cover the 3-cycle FMA latency. That ratio is what
sets the speed: the model funnels the three SSR ports through one per-core
router, which `make ssr-test` measures at about one element per cycle, so
GEMM's 9 accesses per 8 FMAs puts the floor near 1.1 cycles/FMA. It measures
1.33, the difference being the per-block C loads, result stores and FREP setup
outside the streamed body.

Reductions keep the generic order, so MatMul and Conv2d come out bit-identical
to the scalar kernels; GEMM differs in the last bit only because it starts its
accumulator at C instead of adding C at the end.

Columns past the last full block, transposed GEMM operands and filters past the
last group of four run scalar. `make ssr-test` exercises exactly those paths
(plus a streaming probe) on `snitch_real` — the benchmark ops only reach the
shapes that divide evenly.

### What it buys

Snitch cycles for the timed op, everything else unchanged:

| op                     | scalar FP | Xssr + Xfrep |        |
|------------------------|-----------|--------------|--------|
| Conv/Regular_2D_Bias   | 1.14M     | 174k         | 6.5x   |
| GEMM/Regular           | 214k      | 43.5k        | 4.9x   |
| MatMul                 | 52.7k     | 12.6k        | 4.2x   |
| mymatmul (32×32×32)    | 206k      | 43.1k        | 4.8x   |

which was enough to put one Snitch core ahead of the 4-lane Spatz on all four
ops while Spatz ran compiler output. It now holds only on Conv2D (174k vs
457k), the one op Spatz has no hand-written kernel for; on the MatMul and GEMM
shapes, where both cores' kernels are hand-written, Spatz is 5-8x ahead — see
[The Spatz RVV kernels](#the-spatz-rvv-kernels).

Three ops in the baseline are untouched, and none of them can use the
extensions:

- **Softmax** spends its cycles inside `expf`; FREP replays FP instructions from
  a 16-entry buffer, not a libm call, and the surrounding max/sum/scale passes
  are a small part of the total.
- **Integer GEMM** reduces in integer registers. SSR feeds the FP regfile and
  FREP replays FP instructions, so neither applies.
- **Add** is emitted inline into the generated `Network.c` by Deeploy instead of
  being called as a kernel, so there is no function to override.

## The Spatz RVV kernels

Spatz used to run whatever GCC autovectorized out of the Deeploy sources, and
that cost about 5x. GCC vectorizes the innermost loop, which for a matmul is
the dot product, and that shape is wrong for this machine twice over: it loads
`B` with `vlse32.v` down a column — a strided gather whose stride is a multiple
of the TCDM bank interleave for every power-of-two row length, so the elements
serialize onto one bank — and it runs a `vfredusum` reduction per output
element, the one vector operation whose cost does not amortize over the vector
length.

[`runtime/spatz/kernels/`](runtime/spatz/kernels) keeps the accumulator in a
vector register across `k` and broadcasts the scalar `A` element instead:

```c
acc[r] = vfmacc_vf(acc[r], A[i+r][k], B[k][j..j+vl], vl);
```

Every load is then unit-stride, no reduction is executed at all, and four rows
of `A` share one `B` load. It runs at `LMUL=4`, which is worth 1.8x on its own:
a back-to-back `vfmacc` loop with every operand already in the vector register
file sustains 0.180 cycles/element at `LMUL=1` against 0.128 at `LMUL=4`, so a
third of the peak goes to issue overhead before the kernel touches memory.

### What it buys

Spatz cycles for the timed op, everything else unchanged:

| op                     | autovectorized | hand-written RVV |        |
|------------------------|----------------|------------------|--------|
| MatMul                 | 15.8k          | 6.8k             | 2.3x   |
| mymatmul (32x32x32)    | 57.6k          | 7.5k             | 7.6x   |
| GEMM/Regular           | 60.9k          | 8.4k             | 7.3x   |

Transposed operands break the unit-stride `B` load and fall back to the Deeploy
kernel, which stays reachable as `<name>_generic` exactly as it does for
Snitch. Conv2D is untouched and still autovectorized, which is why Snitch keeps
that row.

## The memory system

Each core keeps the memory architecture it actually has — CVA6 caches, Snitch and
Spatz a cluster scratchpad — and all three sit behind the same main memory, so a
cycle difference between them is a difference in how the core uses memory and not in
how the memory was configured. Every number lives in
[`targets/hetero/memsys.py`](targets/hetero/memsys.py).

```
cva6_real     fetch ── L1 I$ 16K/4-way ─┐
                                        ├─ L2 512K/8-way ── DRAM (100 cycles, 8 B/cycle)
              data  ── L1 D$ 32K/8-way ─┘
                       write-through, 8-entry store buffer
                       (peripherals are mapped uncached)

snitch_real   fetch ── cluster I$ (L0 per core + 8K shared) ── HBM (100 cycles)
              data  ── banked TCDM, single cycle, bank conflicts modelled

spatz_real    same as snitch_real, plus the Spatz VLSU ports into the TCDM
```

Cache geometry follows the CVA6 defaults (16 KiB 4-way instruction cache, 32 KiB
8-way write-through data cache) with one deliberate deviation: 64-byte lines instead
of the 128-bit lines of the RTL, because the GVSoC ISS fetches instructions in
64-byte granules and a shorter line would turn one fetch into several refills the
real front-end never issues.

### The cache model

`targets/hetero/timing_cache.cpp` is a cache that models timing only. It holds no
data: every access is forwarded to the next level, so a store can never be lost in a
line that gets evicted and a read can never return something main memory does not
hold. What it owns is the latency — it keeps a tag array, decides hit or miss, and
rewrites what the master sees:

- **hit** → the hit latency. Read hits are forwarded to the next level marked as debug
  accesses, so the level below serves the bytes without counting them: its tag array
  and its counters only ever see this cache's miss stream.
- **miss** → the next level's latency plus the line transfer, with refills serialized
  through the port to that level, so a burst of misses pays bandwidth and not only
  latency.
- **store** → write-through into the store buffer, which drains to the next level.
  The core stalls only once the buffer is full. A store that misses does not pull the
  line in, matching CVA6's write-through L1.

Replacement is LRU. The `[HES-MEM]` line each cache prints at the end of a run is what
the report's cache table shows; those counters cover the whole program, while `cycles`
covers the timed op alone.

### What it costs

Same binaries, same ops, `--memory real` against `--memory ideal`:

| op                     | cva6                | snitch            | spatz              |
|------------------------|---------------------|-------------------|--------------------|
| Add/Regular            | 715 → 863 (+21%)    | 1053 (unchanged)  | 635 → 1035 (+63%)  |
| Conv/Regular_2D_Bias   | 1.74M → 1.86M (+7%) | 174k (unchanged)  | 452k → 457k (+1%)  |
| GEMM/Regular           | 473k → 489k (+3%)   | 43.5k (unchanged) | 6.36k → 8.36k (+31%)|
| MatMul                 | 118k → 119k (+1%)   | 12.6k (unchanged) | 5.02k → 6.82k (+36%)|
| Softmax/Regular        | 34.1k → 36.5k (+7%) | 51.0k (unchanged) | 30.4k → 32.5k (+7%)|
| Integer GEMM/Regular   | 575k → 576k (+0.2%) | 452k (unchanged)  | 82.4k → 84.1k (+2%)|

Snitch does not move because GVSoC's Snitch board already charged 100 cycles for HBM,
which is the DRAM latency the three targets now share; its instruction fetches do pay
it, and raising `DRAM_LATENCY` to 1000 lengthens the Conv run by 25% (it was 4% before
the SSR/FREP kernels: the compute shrank 6.5x, the instruction refills did not, so the
same memory system now costs proportionally more). Spatz moves
because its board mapped HBM with *zero* latency, so instruction fetches that missed
the 8 KiB cluster cache used to refill for free. CVA6 moves because it had no memory
system at all.

Spatz's MatMul and GEMM rows moved from +2% to +31-36% when its kernels were
hand-written: the compute shrank ~7x while the instruction refills did not, so
the same memory system now costs proportionally far more. That is the same
effect the Conv/`DRAM_LATENCY` note above describes for Snitch, and it is the
general shape of the thing — the better the kernel, the larger the share of
what remains that is memory.

The single-op kernels stay compute-bound: their working sets are at most 128 KiB by
construction (they have to fit the Snitch TCDM), and the operands are staged into
local memory before the timed region on every core — `bench_main` copies the inputs
into the network buffers, which warms the CVA6 caches the same way `crt0` fills the
TCDM. What the modelled memory system charges for is therefore the steady-state cost:
capacity misses, store traffic and instruction refills, not the cold start.

## Caveats

- Cache counters in the report cover the whole program (startup, timed op, output
  check); the cycle counts next to them cover the timed op only.
- Ops must fit in 128 KiB TCDM (data + heap + 8 KiB stack) for Snitch/Spatz.
- `minstret` is not implemented by these GVSoC core models; instruction counts
  are reported when available.
- Spatz executes whatever RVV reaches it; instructions the GVSoC model does not
  implement trap and are reported as a failed run rather than a wrong number.
  Four gaps hit by autovectorized code are fixed in `deps/patches/` — see
  `## GVSoC model fixes` below.
- The cores do not run identical code, on purpose: the numbers are meant to be
  each core at its best, not a controlled experiment on one source. Snitch uses
  Xssr/Xfrep, Spatz uses hand-written RVV, CVA6 runs the scalar kernels.
  `--spatz-kernels autovec` swaps Spatz's back for the compiler's output, which
  is how the comparison below was measured.
- Multi-core parallelization (8-core Snitch cluster, multi-CC Spatz) is still
  not used: every number is one core of each type. That is the largest
  remaining lever on the Snitch and Spatz numbers, and it needs a cluster
  runtime (barriers, DMA, per-core tiling) rather than a kernel rewrite.

## GVSoC model fixes

`setup.sh` applies every `deps/patches/gvsoc-core-*.patch` to `deps/gvsoc/core`.

### CVA6 load latency

`gvsoc-core-cva6-load-latency.patch`. The CVA6 model runs each instruction's handler
at commit and then marks its destination registers ready in that same cycle — which
overwrote the timestamp the LSU had just written for a load, discarding the latency of
every data access. Memory delays reached the core's fetch path but never its loads: a
10-cycle L1 hit latency changed a run by exactly zero cycles. The commit stage now
keeps whichever is later, so a dependent instruction waits for the memory system.

This also affects `--memory ideal`, where a load takes one cycle to come back rather
than none. It moved exactly one number in the committed baseline: Conv on CVA6, by 5
cycles out of 1.74M.

### Spatz vector model

Autovectorized kernels exercise the Spatz model harder than the hand-written
benchmarks it ships with. `gvsoc-core-rvv-extensions-and-fixes.patch` carries four
fixes:

1. **Missing instructions** — added `vsext`/`vzext.vf*`, `vwadd`/`vwsub[u].w{v,x}`
   and `vmv<nr>r.v`; implemented `vl<nr>r.v` / `vs<nr>r.v`, which were `abort()` stubs.
2. **Stale `vtype` on deferred execution** — the VPU runs an instruction's functional
   handler at *completion* but read the live `vl`/`vtype`, which a later `vsetvli` on
   the scalar core had already changed. The configuration is now captured at dispatch
   and restored around the handler and in the VLSU.
3. **`vmv.x.s` writeback race** — declared with a vector-output format, so its integer
   destination was never scoreboarded and the scalar core read a stale register.
4. **TCDM bank-crossing bursts** — the Spatz VLSU issued full-width bursts from
   misaligned addresses, crossing a bank boundary and silently corrupting data (this
   was the Conv2D failure). Bursts are now clamped at the interleave boundary.
