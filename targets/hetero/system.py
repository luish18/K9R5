#
# The hetero_soc machine model: one CVA6 orchestrator, one Snitch cluster with
# the Xssr/Xfrep sequencers, one Snitch+Spatz pair, one shared main memory.
#
# Every address, size and core count of that SoC lives here. The GVSoC board
# (targets/hetero/soc.py), the linker scripts and the bare-metal runtime all
# read these numbers -- the runtime through runtime/mesh/hes_system.h, which
# pipeline/gen_system_header.py generates from this file, so that C and Python
# cannot disagree about where anything is.
#
# Cache geometry and main-memory timing are shared with the per-core targets
# and stay in memsys.py.
#

from hetero import memsys  # noqa: F401  (re-exported for the header generator)
from hetero.design import get as _d

# --- Clock -----------------------------------------------------------------
#
# One clock domain for the whole SoC: a cycle difference between two engines
# is a difference in work done per cycle, not in how they were clocked.
FREQUENCY = _d('FREQUENCY', 10_000_000)

# --- Main memory -----------------------------------------------------------
#
# The Snitch/Spatz boards call this HBM and map it at 0x8000_0000; the CVA6
# board calls the same window 'mem'. One memory, one base, all three cores.
HBM_BASE = 0x8000_0000
HBM_SIZE = 0x8000_0000

# Where each of the three ELFs is linked. They share one address space, so
# their text/rodata must not overlap. The host owns the first quarter, which
# is also where every network tensor lives; each cluster gets its own window
# for text and for the load image of the data it copies into TCDM.
HOST_LOAD_BASE = 0x8000_0000
HOST_LOAD_SIZE = 0x0400_0000
SNITCH_LOAD_BASE = 0x8400_0000
SNITCH_LOAD_SIZE = 0x0400_0000
SPATZ_LOAD_BASE = 0x8800_0000
SPATZ_LOAD_SIZE = 0x0400_0000

# --- Host peripherals ------------------------------------------------------
#
# Kept at the addresses the CVA6 boards already use, so runtime/common and the
# existing per-core targets need no change.
DRAM_BASE = 0xB000_0000
DRAM_SIZE = 0x1000_0000
STDOUT_BASE = 0xC000_0000
STDOUT_SIZE = 0x1000_0000
CONTROL_REGS_BASE = 0xD000_0000
CONTROL_REGS_SIZE = 0x1000_0000

# Boot ROM of the Snitch board. Both clusters boot straight into their own
# ELF entry (see Cluster.boot_addr), so nothing fetches from here; it is
# mapped only because the cluster models expect the window to exist.
BOOTROM_BASE = 0x0000_1000
BOOTROM_SIZE = 0x0001_0000

# --- Host vector unit -------------------------------------------------------
#
# Geometry of the Ara unit attached to the CVA6 orchestrator. vlen is the
# vector register length in bits; the stock GVSoC ara targets use 4096, which
# is eight times the 512 a Spatz core carries, so one host vector register
# holds 128 fp32 against a Spatz core's 16. Lanes and lane width are kept the
# same as Spatz so that a comparison between them isolates the register file
# and the memory side rather than confounding all three.
HOST_VLEN = _d('HOST_VLEN', 4096)
HOST_NB_LANES = _d('HOST_NB_LANES', 4)
HOST_LANE_WIDTH = _d('HOST_LANE_WIDTH', 8)

# --- Cluster address map ---------------------------------------------------
#
# Fixed by ClusterArch in GVSoC (pulp/snitch/snitch_cluster/snitch_cluster.py):
# TCDM at the cluster base, then the peripherals, then the zero memory.
# TCDM_SIZE is the swept value; everything after it follows, so the windows
# cannot disagree with each other the way four independent literals could.
# GVSoC's own ClusterArch hardcodes the peripheral at base+0x20000 and the zero
# memory at base+0x30000, so a non-default TCDM size also needs the retune in
# hetero/soc.py -- which is what enforces the two stay consistent.
TCDM_OFFSET = 0x0000_0000
TCDM_SIZE = _d('TCDM_SIZE', 0x0002_0000)   # 128 KiB, banked, single cycle
PERIPHERAL_SIZE = 0x0001_0000
ZERO_MEM_SIZE = 0x0001_0000
PERIPHERAL_OFFSET = TCDM_OFFSET + TCDM_SIZE
ZERO_MEM_OFFSET = PERIPHERAL_OFFSET + PERIPHERAL_SIZE
CLUSTER_WINDOW = ZERO_MEM_OFFSET + ZERO_MEM_SIZE  # per-cluster slice of the map

# --- On-chip interconnect --------------------------------------------------
#
# Bytes per cycle of the two AXI crossbars in the SoC. The narrow one carries
# the host's accesses to cluster address space; the wide one carries cluster
# DMA and instruction refills to main memory. Both were literals in soc.py.
NARROW_AXI_WIDTH = _d('NARROW_AXI_WIDTH', 8)
WIDE_AXI_WIDTH = _d('WIDE_AXI_WIDTH', 64)

# --- Cluster peripheral registers ------------------------------------------
#
# The peripheral regmap has regwidth 64 and opens with three multiregs
# (PERF_COUNTER_ENABLE, HART_SELECT, PERF_COUNTER), each holding
# NumPerfCounters entries, followed by the scalar registers below. The two
# cluster variants declare a *different* NumPerfCounters -- 16 for the Snitch
# cluster, 2 for the Spatz one -- so these offsets are per cluster, not
# global, and Cluster derives them from its own counter count.
# pipeline/gen_system_header.py checks every one of them against the regmap
# header GVSoC generates for that variant.
REGWIDTH_BYTES = 8
NB_MULTIREGS = 3                 # PERF_COUNTER_ENABLE, HART_SELECT, PERF_COUNTER

# The cluster-local interrupt CL_CLINT_SET raises on a core. ClusterArch fixes
# it at 19 (barrier_irq), and the cluster wires cluster_registers'
# external_irq_<n> to that IRQ line of core <n>.
CLUSTER_IRQ = 19

# --- The clusters ----------------------------------------------------------


class Cluster:
    """One compute cluster of the SoC.

    The attribute names of `props` are the five fields GVSoC's ClusterArch
    reads off the properties object it is given, which is what lets two
    clusters of different types live in one board without forking the
    SnitchCluster model.
    """

    def __init__(self, name, base, nb_core, use_spatz, isa, load_base, load_size,
                 nb_perf_counters, core_type='accurate', spatz_nb_lanes=4,
                 spatz_lane_width=8, spatz_vlen=512, first_hartid=0):
        # A Spatz cluster's vector load/store unit is wired to the cluster TCDM
        # only (SnitchCluster binds o_VLSU straight to the TCDM interleaver), so
        # a vectorized kernel reading main memory returns garbage rather than
        # faulting. Operands therefore have to be staged into TCDM before a
        # kernel runs -- on this cluster that is a correctness requirement, not
        # a performance choice.
        self.requires_staging = use_spatz
        self.name = name
        self.base = base
        self.nb_core = nb_core
        self.use_spatz = use_spatz
        self.isa = isa
        self.core_type = core_type
        self.spatz_nb_lanes = spatz_nb_lanes
        # GVSoC reaches neither of these: ClusterArch copies five fields and
        # SnitchCluster constructs its cores with a fixed argument list that has
        # no vlen, so snitch_core.py takes its own defaults. hetero/soc.py
        # applies both to the built cluster instead -- see retune_spatz_vu().
        self.spatz_lane_width = spatz_lane_width
        self.spatz_vlen = spatz_vlen
        self.first_hartid = first_hartid
        self.load_base = load_base
        self.load_size = load_size
        self.nb_perf_counters = nb_perf_counters

    # The core the cluster's iDMA is offloaded from. GVSoC hardcodes this as
    # core 0 for a Spatz cluster and the last core otherwise.
    @property
    def dma_core(self):
        return 0 if self.use_spatz else self.nb_core - 1

    # The core that takes the job descriptor, stages data and fans the work
    # out. It has to be the DMA core: GVSoC binds the cluster's iDMA offload
    # port to that core alone, so no other core can issue the Xdma
    # instructions. This is also the snRuntime convention -- the Snitch
    # cluster is 8 compute cores plus a DMA core, not 9 equal ones.
    @property
    def ctrl_core(self):
        return self.dma_core

    # Cores that run kernel work: every core except the DMA core.
    @property
    def nb_compute(self):
        return self.nb_core - 1

    @property
    def tcdm_base(self):
        return self.base + TCDM_OFFSET

    @property
    def peripheral_base(self):
        return self.base + PERIPHERAL_OFFSET

    @property
    def zero_mem_base(self):
        return self.base + ZERO_MEM_OFFSET

    # --- peripheral register offsets, relative to peripheral_base ----------
    #
    # Three NumPerfCounters-deep multiregs come first, then the scalars in
    # declaration order.

    @property
    def _scalar_base(self):
        return NB_MULTIREGS * self.nb_perf_counters * REGWIDTH_BYTES

    @property
    def perf_counter_enable_offset(self):
        return 0

    @property
    def hart_select_offset(self):
        return self.nb_perf_counters * REGWIDTH_BYTES

    @property
    def perf_counter_offset(self):
        return 2 * self.nb_perf_counters * REGWIDTH_BYTES

    @property
    def cl_clint_set_offset(self):
        return self._scalar_base

    @property
    def cl_clint_clear_offset(self):
        return self._scalar_base + REGWIDTH_BYTES

    @property
    def hw_barrier_offset(self):
        return self._scalar_base + 2 * REGWIDTH_BYTES

    # Upper half of the cluster's main-memory window: free space past its code
    # and .data load image, used for staging and for the memory probe.
    @property
    def scratch_base(self):
        return self.load_base + self.load_size // 2

    @property
    def scratch_size(self):
        return self.load_size // 2

    @property
    def props(self):
        """The duck-typed properties object ClusterArch expects."""
        return _ClusterProps(self)


class _ClusterProps:
    """Exactly the five fields ClusterArch.__init__ reads, per cluster.

    spatz_lane_width and spatz_vlen are deliberately *not* here: ClusterArch
    drops every field it does not name, so adding them would be silently
    ignored. hetero/soc.py applies them to the built cluster instead.
    """

    def __init__(self, cluster):
        self.nb_core_per_cluster = cluster.nb_core
        self.use_spatz = cluster.use_spatz
        self.spatz_nb_lanes = cluster.spatz_nb_lanes
        self.core_type = cluster.core_type
        self.isa = cluster.isa


# The Snitch cluster: 8 compute cores plus a DMA core, each an integer core in
# front of a decoupled FP subsystem with the SSR data movers and the FREP
# sequencer. This is the cluster the fp32 GEMM/MatMul/Conv kernels in
# runtime/snitch/kernels/ were written for. Kept at the base GVSoC's own
# Snitch board uses, which is also what runtime/snitch/link.ld already assumes.
SNITCH_CLUSTER = Cluster(
    name='snitch',
    base=0x1000_0000,
    nb_core=_d('SNITCH_NB_CORE', 9),
    use_spatz=False,
    core_type='accurate',
    isa='rv32imfdca',
    nb_perf_counters=16,
    first_hartid=0,
    load_base=SNITCH_LOAD_BASE,
    load_size=SNITCH_LOAD_SIZE,
)

# The Spatz cluster: core 0 drives the DMA, cores 1..8 are Snitch cores each
# with a 4-lane Spatz vector unit and its VLSU ports into the TCDM.
#
# GVSoC's own Spatz board builds two cores, which made this cluster one compute
# core against the Snitch cluster's eight -- an 8x handicap that had nothing to
# do with either core's throughput. Both clusters now have eight compute cores
# plus a DMA core, so a difference between them is a difference in what a core
# does per cycle. Base as on GVSoC's Spatz board, matching runtime/spatz/link.ld.
SPATZ_CLUSTER = Cluster(
    name='spatz',
    base=0x0010_0000,
    nb_core=_d('SPATZ_NB_CORE', 9),
    use_spatz=True,
    core_type='accurate',   # ignored: a Spatz cluster always uses SnitchFast
    isa='rv32imfdcav',
    spatz_nb_lanes=_d('SPATZ_NB_LANES', 4),
    spatz_lane_width=_d('SPATZ_LANE_WIDTH', 8),
    spatz_vlen=_d('SPATZ_VLEN', 512),
    nb_perf_counters=2,
    first_hartid=SNITCH_CLUSTER.nb_core,
    load_base=SPATZ_LOAD_BASE,
    load_size=SPATZ_LOAD_SIZE,
)

CLUSTERS = [SNITCH_CLUSTER, SPATZ_CLUSTER]

# The CVA6 does not sit in a cluster, so its hartid is picked above every
# cluster hartid rather than colliding with one of them. It has to be *derived*
# rather than a literal: it used to be a hardcoded 16, which the Spatz cluster
# had already grown into (first_hartid 9 + 9 cores = 9..17), so the host shared
# core 7's id. Nothing keyed on it -- crt0_host.S wants it only so traces can
# tell the cores apart -- but it made trace attribution ambiguous, and any
# increase in a cluster's core count made it worse.
HOST_HARTID = max(c.first_hartid + c.nb_core for c in CLUSTERS)

# Two engines must never share a hartid. Cheap to assert, and the assertion is
# what lets nb_core be swept without re-deriving this by hand each time.
_hartids = [h for c in CLUSTERS for h in range(c.first_hartid, c.first_hartid + c.nb_core)]
if len(set(_hartids)) != len(_hartids) or HOST_HARTID in _hartids:
    raise RuntimeError(
        f"hartid collision: clusters occupy {sorted(set(_hartids))} and the host "
        f"takes {HOST_HARTID}")

# --- The job mailbox -------------------------------------------------------
#
# The host hands a cluster work by writing a descriptor into the first bytes
# of that cluster's TCDM and then setting the cluster-local interrupt of its
# control core. The cluster linker scripts keep .data/.bss/stacks above this
# window, so the descriptor is never overwritten by the cluster's own data.
MAILBOX_OFFSET = 0x0000_0000
MAILBOX_SIZE = 0x0000_0200       # 512 B
MAILBOX_MAX_ARGS = 16

# The host's heap and stack. The heap is what Deeploy's generated InitNetwork
# allocates its input/output buffers from; every other tensor is static. Both
# are laid out as symbols rather than sections -- see host_linker() for why.
HOST_HEAP_SIZE = 0x0080_0000     # 8 MiB
HOST_STACK_SIZE = 0x0004_0000    # 256 KiB

# Upper half of the host's window: free main memory past its image, heap and
# stack. The memory probe walks a span of it larger than the L2 so that every
# step has to miss all the way down.
HOST_SCRATCH_BASE = HOST_LOAD_BASE + HOST_LOAD_SIZE // 2
HOST_SCRATCH_SIZE = HOST_LOAD_SIZE // 2

# Bytes of TCDM given to each core as a stack. Kernels here are non-recursive
# and keep their working set in the streams, so 4 KiB is ample; 9 cores of it
# is 36 KiB of the 128 KiB TCDM.
CLUSTER_STACK_SIZE = 0x1000

# --- Engine ids ------------------------------------------------------------
#
# Shared by the host runtime, the cluster runtime and the Deeploy engine
# mapper, so a node's colour and the descriptor it produces are the same
# number end to end.
ENGINE_HOST = 0
ENGINE_SNITCH = 1
ENGINE_SPATZ = 2

ENGINE_NAMES = {
    ENGINE_HOST: 'cva6',
    ENGINE_SNITCH: 'snitch',
    ENGINE_SPATZ: 'spatz',
}

ENGINE_OF_CLUSTER = {
    'snitch': ENGINE_SNITCH,
    'spatz': ENGINE_SPATZ,
}
