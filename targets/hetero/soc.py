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
import utils.loader.loader
from elftools.elf.elffile import ELFFile
from pulp.snitch.snitch_cluster.snitch_cluster import ClusterArch, SnitchCluster
from pulp.stdout.stdout_v3 import Stdout
from vp.clock_domain import Clock_domain

from hetero import memsys, system
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


class HeteroSoc(st.Component):

    def __init__(self, parent, name, parser):
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
        narrow_axi = router.Router(self, 'narrow_axi', bandwidth=8)
        wide_axi = router.Router(self, 'wide_axi', bandwidth=64)

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
            clusters[cluster.name] = SnitchCluster(self, f'cluster_{cluster.name}',
                                                   arch, parser=None, entry=boot_addr)

        # Host cache hierarchy, geometry and latencies from hetero/memsys.py.
        icache = TimingCache(self, 'icache', stats=True, **memsys.ICACHE)
        dcache = TimingCache(self, 'dcache', stats=True, **memsys.DCACHE)
        l2 = TimingCache(self, 'l2', stats=True, **memsys.L2)

        # Data-side router: splits cacheable main memory from everything a
        # cache must never hold -- the peripherals, and the cluster windows,
        # which the clusters write behind the host's back.
        dico = router.Router(self, 'dico')
        # Shared port into the L2, fed by both L1 caches.
        l2_ico = router.Router(self, 'l2_ico')
        # Main-memory port, also used by the loaders so they bypass the caches.
        mem_ico = router.Router(self, 'mem_ico')

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

    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=system.FREQUENCY)
        soc = HeteroSoc(self, 'soc', parser)
        self.bind(clock, 'out', soc, 'clock')
