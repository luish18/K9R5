# hetero-sim — CVA6 / Snitch / Spatz ONNX-op benchmark pipeline

GVSoC simulation of three heterogeneous RISC-V core types, driven from ONNX via Deeploy:

```
ONNX op + inputs  ──Deeploy──►  C code  ──riscv-gcc──►  3 ELFs  ──GVSoC──►  per-core metrics
```

| core   | what is simulated                                    | GVSoC target | ISA               |
|--------|------------------------------------------------------|--------------|-------------------|
| cva6   | CVA6 64-bit host core, ideal zero-latency memory     | `cva6_ideal` (in `targets/`) | rv64imafdc |
| snitch | Snitch cluster core, data in single-cycle TCDM       | `snitch`     | rv32imafd         |
| spatz  | Snitch + Spatz vector unit (4 lanes), RVV kernels    | `spatz`      | rv32imafd + V     |

**Infinite-cache assumption:** every binary keeps all data in single-cycle memory
(zero-latency `mem` for CVA6, cluster TCDM for Snitch/Spatz), so cycle counts measure
compute, not the memory system.

## Setup

Dependencies (GVSoC, Deeploy, the RISC-V toolchain) are large and are not tracked in
git. One command fetches them at pinned commits, applies the local GVSoC patch, and
builds the simulator:

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
reference) printed to stdout and saved as JSON in `results/`.

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
targets/cva6_ideal.py  custom GVSoC target (CVA6 with zero-latency memory)
deps/patches/          local fixes to the GVSoC vector model (tracked; applied by setup.sh)
deps/gvsoc             GVSoC checkout + build           (untracked)
deps/deeploy           Deeploy, installed editable      (untracked)
toolchains/            xPack riscv-none-elf-gcc 15.2    (untracked)
work/                  per-op build + simulation artifacts, disposable (untracked)
ops/                   input op directories (onnx + npz)
results/               metrics JSON per op — committed as the verified baseline
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
   and prints a `[HES] ...` metrics line via semihosting.
4. **Report** — `run.py` parses the metrics and emits the comparison table + JSON.

## Caveats

- Ops must fit in 128 KiB TCDM (data + heap + 8 KiB stack) for Snitch/Spatz.
- `minstret` is not implemented by these GVSoC core models; instruction counts
  are reported when available.
- Spatz executes whatever RVV GCC emits; instructions the GVSoC model does not
  implement trap and are reported as a failed run rather than a wrong number.
  Four gaps hit by autovectorized code are fixed in `deps/patches/` — see
  `## GVSoC vector-model fixes` below.
- Multi-core parallelization (8-core Snitch cluster, multi-CC Spatz) is not used:
  each run measures one core of each type, apples-to-apples.

## GVSoC vector-model fixes

Autovectorized kernels exercise the Spatz model harder than the hand-written
benchmarks it ships with. `deps/patches/gvsoc-core-rvv-extensions-and-fixes.patch`
(applied to `deps/gvsoc/core` by `setup.sh`) carries four fixes:

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
