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
scalar `-O3`, Spatz autovectorized to RVV, Snitch's FP kernels hand-written
against its two custom extensions — see
[The Snitch FP extensions](#the-snitch-fp-extensions-xssr--xfrep).

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
| MatMul             | 2 × (16×32 · 32×8) fp32                      | 119.0k   | **12.6k (9.4×)**   | 15.8k (7.5×)     |
| MatMul (custom op) | 32×32×32 fp32                                | 477.5k   | **43.1k (11.1×)**  | 57.6k (8.3×)     |
| GEMM               | 32×32×32 fp32 + bias                         | 489.0k   | **43.5k (11.2×)**  | 60.9k (8.0×)     |
| GEMM (int8)        | 32×32×32, s8·s8 → s32                        | 575.9k   | 451.7k (1.3×)      | **84.1k (6.8×)** |
| Conv2D + bias      | 2×64×32 fp32, 4 filters 2×8×8, stride 2×4    | 1.86M    | **173.9k (10.7×)** | 457.0k (4.1×)    |
| Softmax            | 512 fp32, 32 rows of 16                      | 36.5k    | 51.0k (0.7×)       | **32.5k (1.1×)** |

Reading it:

- **One Snitch core takes every fp32 reduction**, including from the 4-lane
  Spatz. Xssr removes the loads and the address arithmetic and Xfrep removes
  the loop, which leaves an FMA rate the stream bandwidth sets — see
  [The Snitch FP extensions](#the-snitch-fp-extensions-xssr--xfrep).
- **Spatz takes int8 GEMM**, by 5.4× over Snitch: SSR feeds the FP regfile and
  FREP replays FP instructions, so neither helps an integer reduction, while
  RVV vectorizes it directly.
- **Softmax is `expf`-bound** on all three, so they land within 1.6× of each
  other and no amount of streaming or vectorizing moves it. Snitch is last
  because the libm code is scalar work on its small integer core.
- **Add is too small to say anything** — 64 elements, where the result is
  dominated by call and loop overhead rather than the 64 additions.

`--memory ideal` (zero-latency memory, `results/<op>-ideal.json`) keeps the same
ordering everywhere except Add, where Spatz's 635 cycles edge out CVA6's 715
once its instruction fetches stop paying DRAM. The cost of the modelled memory
system per core is broken down under
[What it costs](#what-it-costs).

## Layout

```
setup.sh               fetch + patch + build all dependencies (pinned commits)
pipeline/run.py        the pipeline driver (codegen → build → simulate → report)
pipeline/make_op.py    wrap an ONNX model + inputs into a pipeline op directory
runtime/               bare-metal glue: crt0, semihosting, linker scripts, bench main
runtime/snitch/snitch_ssr.h    Xssr/Xfrep intrinsics as raw instruction encodings
runtime/snitch/kernels/        the FP kernels Snitch overrides (MatMul/GEMM/Conv2d)
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
   Kernels for Spatz are compiled `-O3 -ffast-math` so GCC autovectorizes them to
   RVV; glue code stays scalar. Snitch replaces three of the Deeploy kernels
   with SSR/FREP versions of its own (below); the Deeploy ones stay linked in as
   `<name>_generic` and still run the shapes the rewrite does not cover.
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

which is enough to put one Snitch core ahead of the 4-lane Spatz on all four ops
(Conv 174k vs 457k, GEMM 43.5k vs 60.9k, MatMul 12.6k vs 15.8k, mymatmul 43.1k
vs 57.6k).

Three ops in the baseline are untouched, and none of them can use the
extensions:

- **Softmax** spends its cycles inside `expf`; FREP replays FP instructions from
  a 16-entry buffer, not a libm call, and the surrounding max/sum/scale passes
  are a small part of the total.
- **Integer GEMM** reduces in integer registers. SSR feeds the FP regfile and
  FREP replays FP instructions, so neither applies.
- **Add** is emitted inline into the generated `Network.c` by Deeploy instead of
  being called as a kernel, so there is no function to override.

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
| GEMM/Regular           | 473k → 489k (+3%)   | 43.5k (unchanged) | 59.8k → 60.9k (+2%)|
| MatMul                 | 118k → 119k (+1%)   | 12.6k (unchanged) | 14.6k → 15.8k (+8%)|
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
- Spatz executes whatever RVV GCC emits; instructions the GVSoC model does not
  implement trap and are reported as a failed run rather than a wrong number.
  Four gaps hit by autovectorized code are fixed in `deps/patches/` — see
  `## GVSoC model fixes` below.
- The cores do not run identical code, on purpose: the numbers are meant to be
  each core at its best, not a controlled experiment on one source. Snitch uses
  Xssr/Xfrep, Spatz uses RVV, CVA6 runs the scalar kernels.
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
