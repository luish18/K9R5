# Plan — CVA6 manager + Snitch/Spatz cluster mesh, 2.5D/3D memory

Status: proposal, nothing implemented.
Scope: extend hetero-sim from *three single cores measured separately* to *one
system where an ONNX model runs across all three, each kernel on the core that
is fastest for it*.

The measurements that motivate the mapping are already in `results/` and the
per-core table in the README.

## 1. The target architecture

| level | element | as drawn | **as modelled first** |
|---|---|---|---|
| 2.5D die | CVA6 manager | 1 per 8 clusters | 1 per **2** clusters |
| | cluster | 1× Snitch ctrl/DMA + 8× Snitch+Spatz, L1 TCDM | same shape, **homogeneous cores** — core 0 acts as ctrl/DMA |
| | cluster ↔ cluster | D2D over interposer | same, **DMA-only, no coherence** |
| | L2 | shared, on the compute die | same |
| 3D stack | L3 | scratchpad on stacked dies (TSV) | same, **one die** |
| external | HyperRAM | reached by DMA, no DDR | same |

Memory hierarchy: **L1 (cluster TCDM) → L2 (compute die) → L3 (3D) → HyperRAM**.

Every number above lives in one file (§4, phase 0). Growing to 8 clusters, a
mixed-core cluster or a multi-die L3 is a parameter change plus the work called
out in §6 — the plan is written so those are *later*, not *never*.

## 2. What already exists

Most of this system is composition, not invention.

**GVSoC**

| piece | where | note |
|---|---|---|
| multi-cluster Snitch board | `pulp/chips/snitch/snitch.py` | `nb_cluster` builds a tile mesh; `noc_type='floonoc'` for a real NoC |
| 9-core cluster | same, `nb_core_per_cluster` | 9 = the snRuntime convention of 8 compute + 1 DMA core, which is the drawn cluster |
| Snitch+Spatz cluster | `pulp/snitch/snitch_cluster/spatz_cluster_v3.py` | accurate model: 16 × 8 KiB TCDM banks, deny/retry crossbar |
| **D2D link** | `pulp/chips/soft_hier_old/c2c_platform/d2dlink.{py,cpp}` | `link_latency_ns`, `link_bandwidth_GBps`, flit granularity, credit FIFOs |
| **HyperRAM device** | `core/models/devices/hyperbus/hyperram.{py,_impl.cpp}` | controller side in `pulp/udma/hyper` |
| DMA from C | Snitch ISA `Xdma` (`dmsrc`/`dmdst`/`dmcpy`/`dmstat`) | already decoded by the ISS we run |
| hierarchy precedent | `pulp/chips/occamy/` | 6 quadrants × 4 clusters, CVA6 hartid reserved |

**Deeploy**

| piece | where | note |
|---|---|---|
| **engine coloring** | `Deeploy/EngineExtension/…/EngineColoringPasses.py` | `EngineMapper.mapNodeToEngine(node, graph)`; the deployer asserts every node is colored |
| working 2-engine platform | `Deeploy/Targets/Neureka/Platform.py` | `engines=[NeurekaEngine, PULPClusterEngine]` — the pattern to copy |
| memory hierarchy | `Deeploy/MemoryLevelExtension/MemoryLevels.py` | `MemoryLevel(name, neighbourNames, size)` + `MemoryHierarchy` |
| Snitch cluster platform | `Deeploy/Targets/Snitch/` | 8-core kernels, `SnitchDma`, GEMM/Softmax tile constraints |
| tiled L3→L2→L1 flow | `DeeployTest/deeployRunner_tiled_snitch.py` | `tiling_enabled=True`, default simulator `gvsoc` |

**This repo:** the pipeline (codegen → build → simulate → report), the per-core
kernel-override mechanism in `pipeline/run.py`, the Xssr/Xfrep kernels in
`runtime/snitch/`, and `targets/hetero/` — a parameter file, a timing-cache
model, and the "retune the built tree instead of forking the class" pattern in
`snitch_memsys.py`.

**Trap:** Deeploy's `Chimera` platform looks like this target (heterogeneous
host + clusters) but is a stub — one engine, one `Add` mapping, and
`TargetLibraries/Chimera/src/Add.c` contains nothing but an include. Take its
shape, not its code.

## 3. What is missing

1. No board has CVA6 *and* Snitch clusters in one address space — `pulp/cva6.py`
   is standalone, the Snitch board has no host.
2. A cluster's cores are all built from one class, so 1 plain Snitch + 8
   Snitch+Spatz in one cluster needs a new cluster class.
3. The Snitch board goes TCDM → HBM: no on-die L2, no 3D L3, no HyperRAM.
4. `D2DLink` sits in an `_old` chip directory, wired to nothing we build.
5. No host↔cluster dispatch runtime; our `crt0.S` parks every non-zero hart.
6. Deeploy has no CVA6+Snitch+Spatz platform, and neither its 4-level hierarchy
   nor multi-engine tiling has been exercised.

## 4. Phases

Each phase ends in something that runs and something that can be falsified.

### Phase 0 — the machine model, one file

`targets/hetero/system.py`: cluster count, cores per cluster, TCDM/L2/L3 sizes,
per-level latency and width, D2D `latency_ns` / `GBps`, HyperRAM timings. Same
role `memsys.py` plays for the current targets, so every later phase reads its
parameters from one place instead of hardcoding them.

*Deliverable:* the table in §1 as ~40 lines of named constants with sources.

### Phase 1 — the GVSoC board

**1a — smallest thing that boots.** `targets/mesh_real.py`: CVA6 host + **one**
cluster + shared L2, no D2D. Reuse `targets/hetero/timing_cache.cpp` for L2 and
the post-build tree edit from `snitch_memsys.py` rather than forking upstream
classes.

**1b — the mesh.** Second cluster, `D2DLink` between tiles (lifted out of
`soft_hier_old` into `targets/hetero/`), L3 behind L2, HyperRAM behind a DMA.

*Validation:* a per-level latency probe in the style of `runtime/tests/ssr_probe.c`
that measures a cluster core's round trip to L1/L2/L3/HyperRAM and asserts it
matches `system.py`. **Without this, every cycle count downstream is
unfalsifiable.**

*Effort:* 1a ~2 days, 1b ~3 days.

### Phase 2 — the cluster runtime

`runtime/mesh/`: separate host and cluster `crt0`; a job descriptor in L2
(kernel id, tile pointers, dims, core count); cluster cores wait on a mailbox and
barrier through the cluster registers GVSoC already models; transfers via the
`Xdma` instructions the ISS already decodes.

*Deliverable:* a hand-written two-kernel demo — GEMM on the cluster, Softmax on
the host — with verified outputs and a cycle breakdown.

**Do not start phase 3 until this works.** It is the critical path: everything
after it assumes a cluster can be booted, fed and waited on.

*Effort:* ~4 days.

### Phase 3 — the Deeploy platform

`HeteroPlatform(engines=[CVA6HostEngine, SnitchClusterEngine, SpatzClusterEngine])`
plus a cost-model `EngineMapper` subclass. This is the "each kernel to its most
optimized core" requirement, and the cost table is seeded from measurements we
already have:

| node | engine | measured reason |
|---|---|---|
| MatMul / GEMM / Conv fp32 | Snitch cluster | 4.2–6.5× over scalar with Xssr/Xfrep; also beats Spatz |
| GEMM int8, quantized ops | Spatz cluster | 5.4× over Snitch — neither Snitch extension helps an integer reduction |
| Softmax, elementwise, control | CVA6 host | `expf`-bound; offload costs more than it saves |

Per-engine kernel libraries reuse what exists: Generic for the host, the SSR/FREP
kernels for Snitch, autovectorized RVV for Spatz — the same override mechanism
`pipeline/run.py` already has, generalized from per-core to per-engine.

`MemoryHierarchy` = L1 → L2 → L3 → HyperRAM, default level HyperRAM. Tiling
reuses `Deeploy/Targets/Snitch/TileConstraints`.

*Deliverable:* `generateNetwork` emits a `Network.c` whose nodes dispatch to the
engine the mapper chose.

*Effort:* ~1 week.

### Phase 4 — pipeline integration

`pipeline/run.py --platform hetero` builds the host and cluster ELFs, runs the
one board, and the report grows a per-node table: engine, cycles, DMA bytes per
level.

*Effort:* ~2 days.

### Phase 5 — calibration

A real ONNX model end to end with numerical verification, then the experiment
that justifies the whole thing: **mapped vs. forced placement**, showing the
mapper's choice beats pinning every node to one engine.

*Effort:* ~3 days.

## 5. Simplifications taken, and why

Deliberate cuts, listed so a reviewer can object to them individually:

- **2 clusters, not 8.** 8 clusters × 9 cores is 73 ISS instances; today's
  single-core Conv already takes ~1 min of wall time for 1.8M cycles. Scale after
  phase 5. Note that `core_type='fast'` buys speed but loses the accurate FP
  subsystem — which is exactly where the Xssr/Xfrep numbers come from — so it is
  not a free switch.
- **Homogeneous clusters.** Use the Spatz cluster with core 0 as ctrl/DMA rather
  than building a mixed-core cluster class. Revisit only if core 0's vector unit
  measurably distorts results.
- **D2D is DMA-only, no coherence.** Matches the hardware and matches what
  Deeploy's tiler already assumes.
- **Tile L3→L2→L1 only.** HyperRAM is staged by explicit prologue/epilogue DMA.
  A 4-level ILP across multiple engines is where the tiler will blow up.
- **One L3 die.** The stack in the drawing is depth, not a different model.

## 6. Risks

- **HyperRAM dominates** any model whose weights do not fit L3 — that is the
  architectural point of the L3 scratchpad, so the report must show the split
  instead of burying it in a total.
- **Tiler solve time** grows with levels × engines; the phase-3 mitigation is the
  three-level restriction above.
- **Simulation throughput** is the practical limit on model size, not
  correctness. Budget small models.
- **Phase 2 is the risk concentrate.** If cluster dispatch does not work, phases
  3–5 have nothing to stand on. It is deliberately scheduled before any Deeploy
  work.

## 7. Open questions

1. Cluster count and cores per cluster to publish results at — 2×8 for
   iteration, 8×8 for the headline number?
2. Is the CVA6 manager expected to *compute*, or only to schedule and move data?
   The mapping table assumes it computes the ops that suit it (Softmax, control).
3. HyperRAM part and clock — the timings decide whether the L3 scratchpad is
   sized for weights or only for activations.
4. Does the D2D link need to carry cluster-to-cluster traffic in the first
   model, or is all inter-cluster data staged through L2?
