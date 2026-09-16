# Chip design optimization parameters

Inventory of every parameter of the simulated `hetero_soc` chip (CVA6 host +
Snitch cluster + Spatz cluster, described in
[relatorio-desenvolvimento.md](relatorio-desenvolvimento.md)) that can be
changed to explore a different point in the design space. The chip is not
synthesized RTL — it is a cycle-level GVSoC model driven entirely by Python
board files, so every parameter below is a named constant in this repo, not a
register in a hardware description language. That is a feature for a thesis:
each row is a knob you can turn and re-measure with `pipeline/run_hetero.py`
without touching a hardware description.

Parameters are grouped by the part of the chip they describe, then by whether
they are already exposed as a constant (**tunable now**) or would require new
modelling work first (**not modelled yet** — from
[hetero-mesh-plan.md](hetero-mesh-plan.md), the design's own roadmap).

There is no vendored RTL in this repo (`deps/` holds only three GVSoC patches,
no submodules, no SystemVerilog). GVSoC itself — the simulator that supplies
the CVA6/Ara/Snitch/Spatz component models — is fetched by `setup.sh` at a
pinned commit, not checked into this worktree, so any microarchitectural
detail *inside* those upstream models that isn't overridden by a constant
below (e.g. CVA6 pipeline depth, branch predictor) is fixed by that pinned
version and out of scope for a parameter sweep from this repo alone.

## 1. CVA6 host (orchestrator)

| Parameter | Current value | Where | What it controls |
|---|---|---|---|
| `isa` (march) | `rv64imafdc` (scalar) / `rv64imafdcv` (vector) | [soc.py:145-158](../targets/hetero/soc.py) | ISA extensions the host core decodes; `v` switches in the Ara vector unit below |
| `HOST_VLEN` | 4096 bits | [system.py:66](../targets/hetero/system.py:66) | Vector register length of the Ara unit attached to the host — 8× a Spatz core's 512 b, i.e. 128 fp32 per register |
| `HOST_NB_LANES` | 4 | [system.py:67](../targets/hetero/system.py:67) | Number of parallel Ara vector lanes |
| `HOST_LANE_WIDTH` | 8 bytes | [system.py:68](../targets/hetero/system.py:68) | Bytes processed per lane per cycle |
| `HOST_HARTID` | 16 | [system.py:72](../targets/hetero/system.py:72) | Hart id of the host, kept above every cluster hartid |
| `ICACHE.size` | 16 KiB | [memsys.py:38](../targets/hetero/memsys.py:38) | L1 instruction cache capacity |
| `ICACHE.ways` | 4 | [memsys.py:39](../targets/hetero/memsys.py:39) | I-cache associativity |
| `ICACHE.hit_latency` | 0 cycles | [memsys.py:44](../targets/hetero/memsys.py:44) | Front-end stall on an I-cache hit |
| `DCACHE.size` | 32 KiB | [memsys.py:49](../targets/hetero/memsys.py:49) | L1 data cache capacity |
| `DCACHE.ways` | 8 | [memsys.py:50](../targets/hetero/memsys.py:50) | D-cache associativity |
| `DCACHE.hit_latency` | 1 cycle | [memsys.py:55](../targets/hetero/memsys.py:55) | Extra cycles on a D-cache hit (on top of the ISS's 2-cycle load-to-use) |
| `DCACHE.write_allocate` | `False` (write-through) | [memsys.py:60](../targets/hetero/memsys.py:60) | Whether a store miss pulls in the line |
| `DCACHE.store_buffer_size` | 8 entries | [memsys.py:61](../targets/hetero/memsys.py:61) | Store buffer depth draining to L2 |
| `LINE_SIZE` (I$/D$/L2) | 64 bytes | [memsys.py:30](../targets/hetero/memsys.py:30) | Cache line size shared by all three levels |
| `L2.size` | 512 KiB | [memsys.py:66](../targets/hetero/memsys.py:66) | Shared L2 capacity |
| `L2.ways` | 8 | [memsys.py:67](../targets/hetero/memsys.py:67) | L2 associativity |
| `L2_LATENCY` | 10 cycles | [memsys.py:33](../targets/hetero/memsys.py:33) | L2 hit latency |
| `L2_WIDTH` | 16 bytes/cycle | [memsys.py:35](../targets/hetero/memsys.py:35) | L1↔L2 bus width (sets refill duration for I$/D$) |
| `HOST_HEAP_SIZE` | 8 MiB | [soc.py:285](../targets/hetero/soc.py:285) | Heap Deeploy's `InitNetwork` allocates activations from |
| `HOST_STACK_SIZE` | 256 KiB | [soc.py:286](../targets/hetero/soc.py:286) | Host stack size |
| `HOST_LOAD_SIZE` | 64 MiB | [soc.py:35](../targets/hetero/soc.py:35) | Address-space window reserved for the host's text/rodata/data image |

Every cache above (I$, D$, L2) is one instance of the same generic
`TimingCache` model, so the full set of knobs any cache level exposes is
`size`, `line_size`, `ways`, `hit_latency`, `miss_latency`, `refill_cycles`,
`write_cycles`, `write_allocate`, `store_buffer_size` — see
[timing_cache.py:44-48](../targets/hetero/timing_cache.py:44). Only the ones
each level actually sets non-default are listed in the table; e.g.
`miss_latency` is left at 0 everywhere today (the "next level" latency is
charged by that level's own mapping instead), so it is itself an unused knob
worth trying.

## 2. Snitch cluster (integer core + Xssr/Xfrep FP subsystem)

| Parameter | Current value | Where | What it controls |
|---|---|---|---|
| `nb_core` | 9 (8 compute + 1 DMA/ctrl) | [soc.py:238](../targets/hetero/soc.py:238) | Cores in the cluster — snRuntime convention is compute cores + 1 dedicated DMA core |
| `isa` | `rv32imfdca` | [soc.py:241](../targets/hetero/soc.py:241) | Scalar ISA (no vector extension — that's what makes it "Snitch" and not "Spatz") |
| `core_type` | `accurate` | [soc.py:240](../targets/hetero/soc.py:240) | Accurate vs. `fast` ISS model — `fast` is quicker to simulate but drops the decoupled FP subsystem that Xssr/Xfrep numbers depend on (noted as a real trade-off, not a free switch, in [hetero-mesh-plan.md:169-171](../docs/hetero-mesh-plan.md)) |
| `nb_perf_counters` | 16 | [soc.py:242](../targets/hetero/soc.py:242) | Hardware perf-counter registers exposed per cluster |
| `TCDM_SIZE` | 128 KiB | [system.py:79](../targets/hetero/system.py:79) | Scratchpad capacity per cluster (banked, single-cycle) |
| `CLUSTER_STACK_SIZE` | 4 KiB/core | [system.py:297](../targets/hetero/system.py:297) | Stack budget carved out of TCDM per core |
| `PERIPHERAL_SIZE` | 64 KiB | [system.py:81](../targets/hetero/system.py:81) | Address window for the cluster's peripheral regmap |
| `base` (cluster address) | `0x1000_0000` | [system.py:237](../targets/hetero/system.py:237) | Where the Snitch cluster sits in the shared address space |
| `CLUSTER_IRQ` | 19 | [system.py:102](../targets/hetero/system.py:102) | Interrupt line the host's mailbox doorbell raises on the cluster's control core |
| TCDM banking (32 banks × 8 B, 256 B interleave) | fixed | GVSoC's `ClusterArch`/`SnitchCluster` model (not a constant in this repo — see [relatorio-desenvolvimento.md §6.4](relatorio-desenvolvimento.md)) | Bank-conflict behavior of strided TCDM accesses; changing it means forking the upstream GVSoC class, not editing a constant here |
| Xssr / Xfrep | fixed feature, not sized | — | Stream Semantic Registers / FP-repeat sequencer that removes load/address-generation and loop overhead — the reason a scalar Snitch core beats naive RVV in dense FMA loops |

## 3. Spatz cluster (Snitch core + RVV vector unit)

| Parameter | Current value | Where | What it controls |
|---|---|---|---|
| `nb_core` | 9 (8 compute + 1 DMA/ctrl) | [soc.py:259](../targets/hetero/soc.py:259) | Cores in the cluster (raised from GVSoC's stock 2-core Spatz board to match the Snitch cluster's compute count) |
| `isa` | `rv32imfdcav` | [soc.py:262](../targets/hetero/soc.py:262) | Scalar ISA + `v` (RVV) |
| `spatz_nb_lanes` | 4 | [soc.py:263](../targets/hetero/soc.py:263) | Vector lanes per Spatz core |
| `nb_perf_counters` | 2 | [soc.py:264](../targets/hetero/soc.py:264) | Hardware perf-counter registers exposed per cluster |
| `core_type` | ignored | [soc.py:261](../targets/hetero/soc.py:261) | A Spatz cluster always runs the `SnitchFast` model regardless of this field |
| `base` (cluster address) | `0x0010_0000` | [system.py:258](../targets/hetero/system.py:258) | Where the Spatz cluster sits in the shared address space |
| TCDM / stack / peripheral sizing | shared with Snitch cluster | §2 above | Same `Cluster` class, same constants — a Spatz cluster's vector load/store unit is wired to TCDM only, so its working set *must* fit there (enforced in [engines.py:53-56](../pipeline/hetero_platform/engines.py:53)) |

## 4. Interconnect and shared memory system

| Parameter | Current value | Where | What it controls |
|---|---|---|---|
| `FREQUENCY` | 10 MHz | [system.py:21](../targets/hetero/system.py:21) | Single clock domain for host, both clusters and memory |
| `narrow_axi` bandwidth | 8 bytes/cycle | [soc.py:106](../targets/hetero/soc.py:106) | Bus width for core data accesses and host→cluster traffic |
| `wide_axi` bandwidth | 64 bytes/cycle | [soc.py:107](../targets/hetero/soc.py:107) | Bus width for cluster DMA and instruction-cache refills |
| `DRAM_LATENCY` | 100 cycles | [memsys.py:16](../targets/hetero/memsys.py:16) | Latency of an access reaching main memory (shared by all three cores since [snitch_memsys.py](../targets/hetero/snitch_memsys.py), which retunes GVSoC's stock 0-cycle Spatz HBM to this same value) |
| `DRAM_WIDTH` | 8 bytes/cycle | [memsys.py:21](../targets/hetero/memsys.py:21) | Port bandwidth to main memory, sets how long a line refill occupies the port |
| `HBM_SIZE` | 2 GiB | [system.py:28](../targets/hetero/system.py:28) | Shared main memory capacity, one pool for host + both clusters |
| `MAILBOX_SIZE` | 512 B | [system.py:279](../targets/hetero/system.py:279) | Job-descriptor size the host can post to a cluster |
| `MAILBOX_MAX_ARGS` | 16 | [system.py:280](../targets/hetero/system.py:280) | Argument slots in a job descriptor |
| Address-map windows (`HOST_LOAD_SIZE`, `SNITCH_LOAD_SIZE`, `SPATZ_LOAD_SIZE`, `DRAM_SIZE`, …) | 64 MiB each | [system.py:35-50](../targets/hetero/system.py:35) | How the shared address space is partitioned between the three ELF images and peripherals |

## 5. Memory-model switch (evaluation knob, not a chip parameter)

| Parameter | Values | Where | What it controls |
|---|---|---|---|
| `--memory` | `real` (default) / `ideal` | [run.py:69-73](../pipeline/run.py:69), CLI flag in `pipeline/run.py` | Swaps the modelled cache/DRAM timing above for zero-latency memory, isolating compute cost from memory-system cost — useful to attribute a design change to the memory system vs. the core |

## 6. Cost model (software layer that evaluates the hardware design)

These don't change the simulated chip, but they change how a design's numbers
get turned into a mapping decision — worth listing because tuning them changes
which core an operator lands on, i.e. they are part of the same optimization
loop.

| Parameter | Current value | Where | What it controls |
|---|---|---|---|
| `RATES` | per-engine, per-op MACs/cycle table | [mapper.py:44-50](../pipeline/hetero_platform/mapper.py:44) | Throughput used to estimate a node's cost on each engine |
| `OFFLOAD_FIXED` | cva6: 0, snitch/spatz: 1200 cycles | [mapper.py:63](../pipeline/hetero_platform/mapper.py:63) | Fixed cost of a mailbox round trip, charged per offload |
| `OFFLOAD_PER_BYTE` | cva6: 0.0, snitch/spatz: 0.10 cycles/byte | [mapper.py:66](../pipeline/hetero_platform/mapper.py:66) | Cost of staging operands into/out of cluster TCDM |
| `TCDM_BUDGET` | `TCDM_SIZE` minus mailbox and per-core stacks | [engines.py:53-56](../pipeline/hetero_platform/engines.py:53) | Bytes of TCDM a job's operands may actually use |

## 7. Not modelled yet (open design space, from `hetero-mesh-plan.md`)

The project's own roadmap ([hetero-mesh-plan.md](hetero-mesh-plan.md)) lists
further axes that would extend the design space but need new modelling work
before they become constants like the ones above:

| Parameter | Planned value / range | Why it's not a constant yet |
|---|---|---|
| Cluster count | 2 (current) vs. 8 (target architecture) | 8 clusters × 9 cores is 73 ISS instances — a simulation-throughput cost, not a code change ([hetero-mesh-plan.md §5](hetero-mesh-plan.md)) |
| Cores per cluster | 9 (fixed at 8 compute + 1 DMA) | Changeable via `nb_core`, but snRuntime and the address-map constants above assume this split |
| Mixed-core clusters | homogeneous only | Would need a new cluster class — GVSoC currently builds every core in a cluster from one class |
| D2D link `link_latency_ns` / `link_bandwidth_GBps` | unset | `D2DLink` exists in GVSoC (`pulp/chips/soft_hier_old/c2c_platform/`) but is wired to nothing in this board |
| L3 (die-stacked scratchpad) size/latency | unset | Memory hierarchy stops at L2 → HBM today; no L3 level modelled |
| HyperRAM part/clock | unset | No HyperRAM device attached to this board yet |
| `core_type='fast'` | not used | Faster ISS but drops the decoupled FP subsystem Xssr/Xfrep numbers depend on — a real accuracy/speed trade-off, not a free switch |

## Notes for the thesis

- Every "tunable now" parameter is a plain Python constant read by the GVSoC
  board (`targets/hetero/soc.py`) and by the C runtime through a generated
  header (`pipeline/gen_system_header.py` → `runtime/mesh/hes_system.h`), so
  changing one and re-running `pipeline/run_hetero.py` is the entire
  experiment loop — no RTL, no re-synthesis.
- The **TCDM banking** of both clusters (32 banks × 8 B, 256 B interleave
  period) is the one hardware detail in this design that is *not* a constant
  in this repo — it's fixed inside GVSoC's own `SnitchCluster`/`ClusterArch`
  model. It matters for the report because it explains why a strided vector
  load collapses onto 1-2 banks for any power-of-two row width (see
  [relatorio-desenvolvimento.md §6.4](relatorio-desenvolvimento.md)) — a
  parameter worth calling out as "fixed by the platform" rather than omitting.
- The clock frequency (`FREQUENCY`) is a single domain for the whole chip by
  design ([system.py:19-21](../targets/hetero/system.py:19)): a cycle
  difference between engines in the results is a difference in work per
  cycle, never a difference in clocking. Splitting it into per-engine domains
  would be a design change with its own trade-off (cross-domain
  synchronization cost) and is not currently modelled.
