#
# Parameters of the CVA6 + Snitch/Spatz cluster mesh modelled by the mesh_real
# target — phase 0 of docs/hetero-mesh-plan.md.
#
# One file for the whole machine, the way memsys.py is one file for the
# single-core memory system: the board (hetero/mesh.py), the linker scripts and
# the latency probe all read their numbers from here, so a parameter is changed
# in one place and the probe immediately says whether the model followed.
#
# Where a number is an assumption rather than a measurement it says so. The
# probe validates that the *model* matches these numbers; it cannot validate
# that the numbers match silicon.
#

from hetero import memsys

# --- Clock ------------------------------------------------------------------
#
# 1 GHz, chosen so that a nanosecond is a cycle and the D2D link parameters
# (specified in ns and GB/s, see D2D below) convert without a fudge factor.
CLOCK_FREQ = 1_000_000_000


def ns_to_cycles(ns: float) -> int:
    return int(ns * CLOCK_FREQ / 1e9)


# --- Topology ---------------------------------------------------------------
#
# The drawing has one CVA6 manager per 8 clusters, each cluster 1 control/DMA
# core + 8 compute cores. Modelled first with 2 clusters (see "Simplifications"
# in the plan): 8 clusters is 73 cores of ISS, which is a simulation-time
# problem before it is a modelling problem.
NB_CLUSTERS = 2

# Cores per cluster, including the DMA/control core. GVSoC's Snitch cluster
# puts the DMA core last (index nb_core-1) and its Spatz cluster puts it first,
# which is the 1 + 8 split of the drawing either way.
NB_CORES_PER_CLUSTER = 9

# Compute cores carry the vector unit; the control core does not need it. Until
# a mixed-core cluster class exists (plan §6), a cluster is homogeneous and this
# selects which kind: False = Snitch with Xssr/Xfrep, True = Snitch + Spatz.
CLUSTER_USE_SPATZ = False
SPATZ_NB_LANES = 4

# 'accurate' keeps the decoupled FP subsystem, the SSR data movers and the FREP
# sequencer — i.e. everything the Snitch kernels in runtime/snitch/ depend on.
# 'fast' simulates quicker and would silently change what is being measured.
CLUSTER_CORE_TYPE = 'accurate'


# --- Address map ------------------------------------------------------------
#
# Kept compatible with GVSoC's Snitch cluster, which hardcodes its own layout
# relative to the cluster base (TCDM at +0, peripherals at +0x20000, zero-memory
# at +0x30000), so cluster bases stay where that model expects them.
BOOTROM = 0x0000_1000, 0x0001_0000  # cluster boot rom
CLUSTER_BASE = 0x1000_0000  # cluster i at CLUSTER_BASE + i * CLUSTER_STRIDE
CLUSTER_STRIDE = 0x0004_0000
L2_BASE = 0x7000_0000
L3_BASE = 0x8000_0000
HYPERRAM_BASE = 0x9000_0000


# --- L1: cluster TCDM -------------------------------------------------------
#
# Fixed by the GVSoC cluster model: 128 KiB over 4 superbanks x 8 banks of
# 8 bytes, single cycle, bank conflicts modelled by the interleaver. Repeated
# here because the linker scripts and the probe need the size.
L1_SIZE = 0x0002_0000
L1_LATENCY = 0  # on top of the cycle the ISS charges anyway

# --- L2: shared SRAM on the compute die -------------------------------------
#
# ASSUMPTION. 4 MiB is the order of magnitude of a shared on-die scratchpad for
# a cluster group of this size (Occamy: 8 MiB across 6 quadrants). The latency
# is a hop across the die's AXI crossbar into SRAM — twice the 10 cycles
# memsys.py charges for CVA6's private L2, since this one is further away and
# shared. Width is the wide AXI, 64 B/cycle.
L2_SIZE = 4 * 1024 * 1024
L2_LATENCY = 20
L2_WIDTH = 64

# --- L3: scratchpad on the stacked die (TSV) --------------------------------
#
# ASSUMPTION. One die (the drawing stacks several; depth is capacity, not a
# different model — plan §5). TSVs are wide and short, so the die crossing costs
# latency rather than bandwidth: same 64 B/cycle as L2, but a further ~30 cycles
# on top of it.
L3_SIZE = 64 * 1024 * 1024
L3_LATENCY = 50
L3_WIDTH = 64

# --- HyperRAM: external memory, no DDR --------------------------------------
#
# ASSUMPTION, and the one most worth replacing with a real part number (plan §7,
# open question 3). Placeholder is a 200 MHz x8 HyperBus device: DDR on 8 lines
# is 400 MB/s, which at 1 GHz core clock is 0.4 B/cycle — rounded to 1 here
# because the router models bandwidth in whole bytes per cycle, so this model is
# optimistic by 2.5x and must not be quoted as a HyperRAM result until the part
# is pinned down. Initial latency is the device's read latency plus controller
# and bus turnaround.
HYPERRAM_SIZE = 32 * 1024 * 1024
HYPERRAM_LATENCY = 150
HYPERRAM_WIDTH = 1

# --- Interconnect -----------------------------------------------------------
#
# Two AXI paths, as in GVSoC's Snitch SoC: a narrow one for the cores' scalar
# accesses and the host's control traffic, a wide one for DMA and cluster
# refills.
NARROW_WIDTH = 8
WIDE_WIDTH = 64

# --- D2D: cluster-to-cluster over the interposer ----------------------------
#
# ASSUMPTION. GVSoC's D2DLink model defaults to 256 ns / 256 GB/s, which is a
# chip-to-chip link on a board; an interposer D2D is short. 20 ns and 64 GB/s
# is the order of magnitude of a die-to-die link on a silicon interposer.
# Flit granularity is the model's default and matches the wide AXI beat.
D2D_LATENCY_NS = 20
D2D_BANDWIDTH_GBPS = 64
D2D_FLIT_BYTES = 64

# The DRAM latency the single-core targets already share, reused so a cycle
# count from mesh_real is comparable with one from cva6_real/snitch_real.
DRAM_LATENCY = memsys.DRAM_LATENCY


def cluster_base(cluster_id: int) -> int:
    return CLUSTER_BASE + cluster_id * CLUSTER_STRIDE


def check() -> None:
    """Fail loudly on a parameter set that cannot be built.

    Called by the board before it instantiates anything, so a bad edit here is
    a clear error rather than a simulation that quietly does something else.
    """
    assert NB_CLUSTERS >= 1, "need at least one cluster"
    assert NB_CORES_PER_CLUSTER >= 2, "a cluster needs a DMA core plus a compute core"
    assert CLUSTER_STRIDE >= 0x0004_0000, "cluster stride must cover TCDM + peripherals + zero memory"
    assert L2_BASE + L2_SIZE <= L3_BASE, "L2 window overlaps L3"
    assert L3_BASE + L3_SIZE <= HYPERRAM_BASE, "L3 window overlaps HyperRAM"
    assert cluster_base(NB_CLUSTERS - 1) + CLUSTER_STRIDE <= L2_BASE, "cluster windows overlap L2"
    assert CLUSTER_CORE_TYPE in ('accurate', 'fast'), CLUSTER_CORE_TYPE
