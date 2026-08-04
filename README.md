# hetero-sim — CVA6 / Snitch / Spatz ONNX-op benchmark pipeline

GVSoC simulation of three heterogeneous RISC-V core types, driven from ONNX via Deeploy:

```
ONNX op + inputs  ──Deeploy──►  C code  ──riscv-gcc──►  3 ELFs  ──GVSoC──►  per-core metrics
```

| core   | what is simulated                                    | GVSoC target | ISA               |
|--------|------------------------------------------------------|--------------|-------------------|
| cva6   | CVA6 64-bit host core, L1 I$/D$ + L2 + DRAM          | `cva6_real` (in `targets/`) | rv64imafdc |
| snitch | Snitch cluster core, data in cluster TCDM            | `snitch_real` (in `targets/`) | rv32imafd |
| spatz  | Snitch + Spatz vector unit (4 lanes), RVV kernels    | `spatz_real` (in `targets/`) | rv32imafd + V |

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

## Layout

```
setup.sh               fetch + patch + build all dependencies (pinned commits)
pipeline/run.py        the pipeline driver (codegen → build → simulate → report)
pipeline/make_op.py    wrap an ONNX model + inputs into a pipeline op directory
runtime/               bare-metal glue: crt0, semihosting, linker scripts, bench main
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
   RVV; glue code stays scalar. Snitch/Spatz builds avoid the C extension because
   the GVSoC Snitch model executes compressed FP loads on the integer core,
   diverging from the decoupled FP subsystem.
3. **Simulation** — each ELF runs on its GVSoC board; `bench_main.c` reads
   `mcycle` around `RunNetwork()`, verifies outputs against the ONNX reference,
   and prints a `[HES] ...` metrics line via semihosting. Which board depends on
   `--memory`; the binary does not, so both models run the same code.
4. **Report** — `run.py` parses the metrics and the `[HES-MEM]` cache counters and
   emits the comparison table + JSON.

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
| Conv/Regular_2D_Bias   | 1.74M → 1.86M (+7%) | 1.14M (unchanged) | 452k → 457k (+1%)  |
| GEMM/Regular           | 473k → 489k (+3%)   | 214k (unchanged)  | 59.8k → 60.9k (+2%)|
| MatMul                 | 118k → 119k (+1%)   | 52.7k (unchanged) | 14.6k → 15.8k (+8%)|
| Softmax/Regular        | 34.1k → 36.5k (+7%) | 51.0k (unchanged) | 30.4k → 32.5k (+7%)|
| Integer GEMM/Regular   | 575k → 576k (+0.2%) | 452k (unchanged)  | 82.4k → 84.1k (+2%)|

Snitch does not move because GVSoC's Snitch board already charged 100 cycles for HBM,
which is the DRAM latency the three targets now share; its instruction fetches do pay
it, and raising `DRAM_LATENCY` to 1000 lengthens the Conv run by 4%. Spatz moves
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
- Multi-core parallelization (8-core Snitch cluster, multi-CC Spatz) is not used:
  each run measures one core of each type, apples-to-apples.

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
