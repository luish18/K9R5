#
# The hetero_soc board: one CVA6 orchestrator, one 9-core Snitch cluster with
# the Xssr/Xfrep sequencers, one Snitch+Spatz pair, one shared main memory.
#
#   CVA6 host (rv64imafdc)
#     fetch ── L1 I$ ─┐
#     data  ── dico ──┼── L2 ── HBM 0x8000_0000
#                │    │
#                │  L1 D$ (write-through)
#                └── uncached: cluster windows, stdout, control regs, dram
#                                     │
#          ┌──────────────────────────┴───────────────────────┐
#       narrow AXI (8 B)                          wide AXI (64 B)
#          │                                              │
#     snitch cluster @ 0x1000_0000            spatz cluster @ 0x0010_0000
#     9 cores, Snitch + FP subsystem          2 cores, Snitch + Spatz VPU
#     (SSR data movers, FREP sequencer)       (4 lanes, VLSU into TCDM)
#
# There is no upstream GVSoC board that puts a CVA6 and Snitch clusters in one
# address space, so this composes stock components: the CVA6 cache hierarchy is
# the one targets/cva6_real.py already models, and the clusters are unmodified
# SnitchCluster instances. What makes two *different* cluster types possible in
# one board is that ClusterArch reads its shape off a properties object, so
# each cluster gets its own (see system.Cluster.props) instead of the whole
# board sharing one.
#
# Every address here comes from hetero/system.py, which also generates the
# runtime's hes_system.h, so the board and the software cannot disagree.
#

import os

import gvsoc.systree as st
import interco.router as router
import memory.memory as memory
import pulp.cva6.control_regs
import pulp.cva6.cva6
from pulp.cpu.iss.cva6 import Cva6
from pulp.cpu.iss.cva6_config import Cva6Config
import utils.loader.loader
from elftools.elf.elffile import ELFFile
from pulp.snitch.snitch_cluster.snitch_cluster import ClusterArch, SnitchCluster
from pulp.stdout.stdout_v3 import Stdout
from vp.clock_domain import Clock_domain

from hetero import memsys, power, system
from hetero.timing_cache import TimingCache

# The three ELFs of a run. gapy carries a single --binary, which the host
# takes; the two cluster binaries arrive by environment because they are
# per-board configuration rather than "the program" gapy knows about.
ELF_ENV = {
    'snitch': 'HES_ELF_SNITCH',
    'spatz': 'HES_ELF_SPATZ',
}


def elf_entry(path: str) -> int:
    with open(path, 'rb') as f:
        return ELFFile(f)['e_entry']


def retune_tcdm(arch, cluster):
    """Resize the cluster TCDM and move the windows that sit above it.

    GVSoC's ClusterArch hardcodes a 128 KiB TCDM at the cluster base, the
    peripherals at base+0x20000 and the zero memory at base+0x30000
    (snitch_cluster.py, class Tcdm and the two Area() calls beside it), so a
    different TCDM size has to be applied to the arch object before the cluster
    is built from it. Everything downstream -- the bank sizes, the interleaver
    offset mask, the DMA's local window, the router mappings -- reads these
    fields off arch rather than re-deriving them, so this is the only place
    that needs to change.

    The Area objects are mutated rather than replaced: Area is defined inside a
    `if os.environ.get('USE_GVRUN') is None:` block upstream, so constructing
    one here would bind us to that branch.
    """
    # Raise rather than silently model something else, the same way
    # snitch_memsys.retune_hbm does.
    if arch.tcdm.nb_superbanks != 4 or arch.tcdm.nb_banks_per_superbank != 8:
        raise RuntimeError(
            f"expected a 4x8 TCDM on cluster {cluster.name}, found "
            f"{arch.tcdm.nb_superbanks}x{arch.tcdm.nb_banks_per_superbank} "
            "-- the GVSoC Snitch cluster layout changed")
    if arch.tcdm.area.base != cluster.base:
        raise RuntimeError(
            f"expected the TCDM of cluster {cluster.name} at its base "
            f"0x{cluster.base:x}, found 0x{arch.tcdm.area.base:x}")

    nb_banks = arch.tcdm.nb_superbanks * arch.tcdm.nb_banks_per_superbank
    if system.TCDM_SIZE % nb_banks != 0:
        raise RuntimeError(
            f"TCDM_SIZE 0x{system.TCDM_SIZE:x} does not divide evenly into "
            f"{nb_banks} banks")
    if system.TCDM_SIZE & (system.TCDM_SIZE - 1):
        # The interleaver masks with area.size - 1 (snitch_cluster.py), so a
        # non-power-of-two TCDM would alias rather than fail.
        raise RuntimeError(f"TCDM_SIZE 0x{system.TCDM_SIZE:x} must be a power of two")

    arch.tcdm.area.size = system.TCDM_SIZE
    arch.tcdm.bank_size = system.TCDM_SIZE // nb_banks

    arch.peripheral.base = cluster.base + system.PERIPHERAL_OFFSET
    arch.peripheral.size = system.PERIPHERAL_SIZE
    arch.zero_mem.base = cluster.base + system.ZERO_MEM_OFFSET
    arch.zero_mem.size = system.ZERO_MEM_SIZE


def retune_spatz_vu(comp, cluster):
    """Apply the Spatz vector geometry GVSoC's cluster will not carry.

    ClusterArch copies five fields off the properties object, and SnitchCluster
    builds its cores with a fixed argument list that has no vlen, so
    snitch_core.py always takes its own defaults (vlen=512, lane_width=8).

    vlen is a compile-time define, so it has to be rewritten in the core's
    c_flags -- which works because c_flags are hashed into the generated
    component name lazily, inside Component.__build(), long after this
    constructor has returned. lane_width is a plain JSON property, so setting it
    costs nothing at all.
    """
    tag = '-DCONFIG_ISS_VLEN='
    for core in comp.cores:
        hits = [f for f in core.c_flags if f.startswith(tag)]
        if len(hits) != 1:
            raise RuntimeError(
                f"expected exactly one {tag} flag on {core.name}, found {len(hits)} "
                "-- the GVSoC Spatz core stopped setting VLEN through add_c_flags")
        # Replace in place, never move. get_generated_component() hashes
        # ''.join(sources + cflags + libs), so the flag's *position* is part of
        # the component name: appending it instead would rename the model even
        # when the value is unchanged, and ask for a .so that was never built.
        core.c_flags = [f'{tag}{int(cluster.spatz_vlen)}' if f.startswith(tag) else f
                        for f in core.c_flags]
        core.add_property('vu/lane_width', cluster.spatz_lane_width)


class HeteroSoc(st.Component):

    def __init__(self, parent, name, parser, vector_host=False):
        super().__init__(parent, name)

        [args, __] = parser.parse_known_args()
        host_binary = args.binary

        cluster_binaries = {}
        for cluster in system.CLUSTERS:
            path = os.environ.get(ELF_ENV[cluster.name])
            if path is not None and not os.path.isfile(path):
                raise RuntimeError(
                    f"{ELF_ENV[cluster.name]}={path} does not exist")
            cluster_binaries[cluster.name] = path

        #
        # Components
        #

        # Main memory, shared by the host and both clusters. Purely functional:
        # all of the timing lives in the caches and in the mapping latencies,
        # because every access -- hit or miss -- is forwarded down to it.
        mem = memory.Memory(self, 'mem', size=system.HBM_SIZE, atomics=True,
                            width_log2=-1)
        # Main memory is off-chip: it gets the DRAM estimate, NOT the SRAM
        # scaling rule the caches and TCDM use. hetero/power.py explains why the
        # two must not be mixed, and flags this as the weakest coefficient in
        # the model.
        mem.add_properties(power.dram_model())

        # Boot ROM of the Snitch cluster model. Nothing fetches from it (each
        # cluster boots straight into its own ELF entry), but the window has to
        # exist because the cluster models map it.
        rom = memory.Memory(self, 'rom', size=system.BOOTROM_SIZE,
                                   stim_file=self.get_file_path('pulp/snitch/bootrom.bin'))

        # Host peripherals, at the addresses the CVA6 boards already use so the
        # same host binary runs on cva6_real and here.
        dram = memory.Memory(self, 'dram', size=system.DRAM_SIZE, atomics=True,
                             width_log2=-1)
        regs = pulp.cva6.control_regs.ControlRegs(self, 'control_regs',
                                                  dram_end=0xc0000000)
        stdout = Stdout(self, 'stdout')

        # SoC interconnect. Narrow carries core data accesses and everything
        # the host issues at a cluster; wide carries cluster DMA and
        # instruction-cache refills.
        narrow_axi = router.Router(self, 'narrow_axi', bandwidth=system.NARROW_AXI_WIDTH)
        wide_axi = router.Router(self, 'wide_axi', bandwidth=system.WIDE_AXI_WIDTH)

        # Clusters. Each gets its own properties object, which is what lets one
        # board hold a 9-core Snitch cluster and a 2-core Spatz pair.
        clusters = {}
        for cluster in system.CLUSTERS:
            binary = cluster_binaries[cluster.name]
            # Boot the cores directly at the ELF entry, bypassing the bootrom:
            # the two clusters run different binaries, and the bootrom reads
            # one shared entry word.
            boot_addr = elf_entry(binary) if binary else cluster.base
            arch = ClusterArch(cluster.props, cluster.base, cluster.first_hartid,
                               auto_fetch=False, boot_addr=boot_addr)
            # parser=None on purpose: SnitchCluster would otherwise hand the
            # host's rv64 ELF to its rv32 cores as debug info.
            retune_tcdm(arch, cluster)
            comp = SnitchCluster(self, f'cluster_{cluster.name}',
                                 arch, parser=None, entry=boot_addr)
            if cluster.use_spatz:
                retune_spatz_vu(comp, cluster)
            power.attach_cluster_tcdm(comp, arch)
            clusters[cluster.name] = comp

        # Host cache hierarchy, geometry and latencies from hetero/memsys.py.
        icache = TimingCache(self, 'icache', stats=True, **memsys.ICACHE)
        dcache = TimingCache(self, 'dcache', stats=True, **memsys.DCACHE)
        l2 = TimingCache(self, 'l2', stats=True, **memsys.L2)

        # Energy coefficients, scaled to each cache's own geometry. Without
        # these every cache reports zero energy under --power, which reads as
        # "the memory hierarchy is free" -- the opposite of what a sweep over
        # cache sizes needs to see. See hetero/power.py for the anchor and the
        # scaling rule.
        for cache, geom in ((icache, memsys.ICACHE), (dcache, memsys.DCACHE),
                            (l2, memsys.L2)):
            cache.add_properties(power.cache_model(geom['size'], geom['line_size']))

        # Data-side router: splits cacheable main memory from everything a
        # cache must never hold -- the peripherals, and the cluster windows,
        # which the clusters write behind the host's back.
        dico = router.Router(self, 'dico')
        # Shared port into the L2, fed by both L1 caches.
        l2_ico = router.Router(self, 'l2_ico')
        # Main-memory port, also used by the loaders so they bypass the caches.
        mem_ico = router.Router(self, 'mem_ico')

        # The orchestrator, with or without a vector unit. The scalar variant
        # is the ISS v1 core the board has always used; the vector one is the
        # ISS v2 core with ara_v2 attached, as in targets/ara_host.py. Both
        # expose data, fetch and fetchen identically, so only the vector
        # variant's extra o_VLSU binding differs below.
        if vector_host:
            host = Cva6(self, 'host', config=Cva6Config(
                isa='rv64imafdcv',
                boot_addr=system.HOST_LOAD_BASE,
                hart_id=system.HOST_HARTID,
                fetch_enable=False,
                htif=True,
                vlen=system.HOST_VLEN,
                nb_lanes=system.HOST_NB_LANES,
                lane_width=system.HOST_LANE_WIDTH,
            ))
        else:
            host = pulp.cva6.cva6.CVA6(self, 'host', isa='rv64imafdc',
                                       boot_addr=system.HOST_LOAD_BASE,
                                       core_id=system.HOST_HARTID)

        #
        # Bindings
        #

        # --- host data side ---
        dico.o_MAP(dcache.i_INPUT(), name='mem', base=system.HBM_BASE,
                   size=system.HBM_SIZE, rm_base=False)
        dico.o_MAP(stdout.i_INPUT(), name='stdout', base=system.STDOUT_BASE,
                   size=system.STDOUT_SIZE)
        dico.o_MAP(dram.i_INPUT(), name='dram', base=system.DRAM_BASE,
                   size=system.DRAM_SIZE, latency=memsys.DRAM_LATENCY)
        dico.o_MAP(regs.i_INPUT(), name='control_regs',
                   base=system.CONTROL_REGS_BASE, size=system.CONTROL_REGS_SIZE)
        # Cluster TCDM and peripherals: uncached, straight onto the narrow AXI.
        for cluster in system.CLUSTERS:
            dico.o_MAP(narrow_axi.i_INPUT(), name=f'cluster_{cluster.name}',
                       base=cluster.base, size=system.CLUSTER_WINDOW,
                       rm_base=False)

        # --- host cache hierarchy ---
        icache.o_OUTPUT(l2_ico.i_INPUT())
        dcache.o_OUTPUT(l2_ico.i_INPUT(1))
        l2_ico.o_MAP(l2.i_INPUT(), name='l2', base=system.HBM_BASE,
                     size=system.HBM_SIZE, rm_base=False)
        l2.o_OUTPUT(mem_ico.i_INPUT())
        # Main memory is reached through the wide AXI, the same port the
        # clusters use: there is one main memory and one port into it, so host
        # and cluster traffic meet where they would on the real SoC.
        mem_ico.o_MAP(wide_axi.i_INPUT(), name='hbm', base=system.HBM_BASE,
                      size=system.HBM_SIZE, rm_base=False)
        # The loaders share this router, so it has to reach every region a
        # program segment can be linked into.
        mem_ico.o_MAP(dram.i_INPUT(), name='dram', base=system.DRAM_BASE,
                      size=system.DRAM_SIZE, latency=memsys.DRAM_LATENCY)

        self.bind(host, 'data', dico, 'input')
        self.bind(host, 'fetch', icache, 'input')
        if vector_host:
            # Vector accesses take the same route as scalar data, so they see
            # the same cacheable/uncacheable split and the same hierarchy.
            host.o_VLSU(dico.i_INPUT())

        # --- SoC interconnect ---
        # Main memory hangs off the wide router; the narrow one forwards to it,
        # as on GVSoC's own Snitch board.
        wide_axi.o_MAP(mem.i_INPUT(), name='hbm', base=system.HBM_BASE,
                       size=system.HBM_SIZE, rm_base=True,
                       latency=memsys.DRAM_LATENCY)
        narrow_axi.o_MAP(wide_axi.i_INPUT(), name='hbm', base=system.HBM_BASE,
                         size=system.HBM_SIZE, rm_base=False)
        wide_axi.o_MAP(rom.i_INPUT(), name='rom', base=system.BOOTROM_BASE,
                       size=system.BOOTROM_SIZE, rm_base=True)
        narrow_axi.o_MAP(rom.i_INPUT(), name='rom', base=system.BOOTROM_BASE,
                         size=system.BOOTROM_SIZE, rm_base=True)

        for cluster in system.CLUSTERS:
            comp = clusters[cluster.name]
            comp.o_NARROW_SOC(narrow_axi.i_INPUT())
            comp.o_WIDE_SOC(wide_axi.i_INPUT())
            narrow_axi.o_MAP(comp.i_NARROW_INPUT(), name=f'cluster_{cluster.name}',
                             base=cluster.base, size=system.CLUSTER_WINDOW,
                             rm_base=False)

        # --- binary loading ---
        # One loader per ELF. All three write only into main memory, so they
        # all sit on mem_ico and none of them warms the host caches: the core
        # starts with everything cold, as it would after reset.
        host_loader = utils.loader.loader.ElfLoader(self, 'loader_host',
                                                   binary=host_binary)
        host_loader.o_OUT(mem_ico.i_INPUT())
        self.bind(host_loader, 'start', host, 'fetchen')

        for cluster in system.CLUSTERS:
            binary = cluster_binaries[cluster.name]
            if binary is None:
                # No ELF for this cluster: its cores were built with
                # fetch_enable off and nothing releases them, so they stay
                # halted instead of fetching whatever happens to be in TCDM.
                continue
            loader = utils.loader.loader.ElfLoader(self, f'loader_{cluster.name}',
                                                   binary=binary)
            loader.o_OUT(mem_ico.i_INPUT())
            # Release the cluster's cores once its image is in memory.
            loader.o_START(clusters[cluster.name].i_FETCHEN())

        self.clusters = clusters


class HeteroBoard(st.Component):

    def __init__(self, parent, name, parser, options, vector_host=False):
        super().__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=system.FREQUENCY)
        soc = HeteroSoc(self, 'soc', parser, vector_host=vector_host)
        self.bind(clock, 'out', soc, 'clock')


class HeteroAraBoard(HeteroBoard):
    """The same SoC with a vector orchestrator instead of a scalar one."""

    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, parser, options, vector_host=True)
