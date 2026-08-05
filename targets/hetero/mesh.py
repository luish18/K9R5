#
# CVA6 manager + Snitch cluster(s) + shared L2 — phase 1a of
# docs/hetero-mesh-plan.md.
#
# GVSoC has a CVA6 board and a Snitch cluster board, but nothing with both in
# one address space, so this composes them: the cluster comes from
# pulp.snitch.snitch_cluster unchanged, the host from pulp.cva6.cva6 unchanged,
# and what is new here is the interconnect between them and the memory levels
# they share. Every number comes from hetero/system.py.
#
#   host.fetch ── L1 I$ ─┐
#                        ├── narrow AXI ─┬─ L2 ── L3 ── HyperRAM
#   host.data ── dico ─┬─ L1 D$ ─────────┤   (each mapping charges its latency)
#                      │                 ├─ cluster i (TCDM, peripherals)
#                      └── uncached ─────┘
#   cluster.narrow ────────────────────────┘   wide AXI: same memories, DMA path
#
# The host keeps its private L1 caches (geometry from memsys.py, the same ones
# cva6_real models) because without them every instruction fetch would go to the
# shared L2 and the fetch path, not the kernel, would set the cycle count. The
# shared on-die L2 is the next level down and is a scratchpad, not a cache, so
# the host has no private L2 here.
#
# Cluster windows are deliberately *not* cacheable on the host: TCDM is shared
# with the cluster's own cores, and a host-side copy of it could go stale.
#
# Phase 1a builds the host, one cluster and the memory levels; the cluster is
# instantiated halted (auto_fetch=False) because nothing releases it until the
# dispatch runtime of phase 2. D2D links between clusters are phase 1b.
#

import gvsoc.systree
import interco.router as router
import memory.memory as memory
import utils.loader.loader
from pulp.cva6.cva6 import CVA6
from pulp.snitch.snitch_cluster.snitch_cluster import ClusterArch, SnitchCluster
from vp.clock_domain import Clock_domain

from hetero import memsys, system
from hetero.timing_cache import TimingCache


class _ClusterProperties:
    """The subset of SnitchArchProperties that ClusterArch actually reads.

    GVSoC's own properties class is bound to its board's command-line parameter
    declarations, so this passes the same fields without dragging that in.
    """

    def __init__(self):
        self.nb_core_per_cluster = system.NB_CORES_PER_CLUSTER
        self.use_spatz = system.CLUSTER_USE_SPATZ
        self.spatz_nb_lanes = system.SPATZ_NB_LANES
        self.core_type = system.CLUSTER_CORE_TYPE
        self.isa = 'rv32imfdcav' if system.CLUSTER_USE_SPATZ else 'rv32imfdca'


class MeshSoc(gvsoc.systree.Component):

    def __init__(self, parent, name, parser, nb_cluster: int):
        super().__init__(parent, name)

        system.check()

        binary = None
        if parser is not None:
            [args, __] = parser.parse_known_args()
            binary = args.binary

        #
        # Interconnect
        #
        narrow_axi = router.Router(self, 'narrow_axi', bandwidth=system.NARROW_WIDTH)
        wide_axi = router.Router(self, 'wide_axi', bandwidth=system.WIDE_WIDTH)

        #
        # Memory levels. Latency is charged by the mapping, bandwidth by the
        # memory's own width, so a level is described by exactly the two
        # numbers system.py gives it.
        #
        rom_base, rom_size = system.BOOTROM
        rom = memory.Memory(self, 'bootrom', size=rom_size, atomics=True, width_log2=-1)

        l2 = memory.Memory(self, 'l2', size=system.L2_SIZE, atomics=True,
                           width_log2=system.L2_WIDTH.bit_length() - 1)
        l3 = memory.Memory(self, 'l3', size=system.L3_SIZE, atomics=True,
                           width_log2=system.L3_WIDTH.bit_length() - 1)
        hyperram = memory.Memory(self, 'hyperram', size=system.HYPERRAM_SIZE, atomics=True,
                                 width_log2=system.HYPERRAM_WIDTH.bit_length() - 1)

        for ico in (narrow_axi, wide_axi):
            ico.o_MAP(rom.i_INPUT(), name='bootrom', base=rom_base, size=rom_size,
                      rm_base=True)
            ico.o_MAP(l2.i_INPUT(), name='l2', base=system.L2_BASE, size=system.L2_SIZE,
                      rm_base=True, latency=system.L2_LATENCY)
            ico.o_MAP(l3.i_INPUT(), name='l3', base=system.L3_BASE, size=system.L3_SIZE,
                      rm_base=True, latency=system.L3_LATENCY)
            ico.o_MAP(hyperram.i_INPUT(), name='hyperram', base=system.HYPERRAM_BASE,
                      size=system.HYPERRAM_SIZE, rm_base=True,
                      latency=system.HYPERRAM_LATENCY)

        #
        # Clusters. hartid 0 is the host, so cluster harts start at 1.
        #
        properties = _ClusterProperties()
        self.clusters = []
        for cluster_id in range(nb_cluster):
            arch = ClusterArch(properties, system.cluster_base(cluster_id),
                               first_hartid=1 + cluster_id * system.NB_CORES_PER_CLUSTER,
                               auto_fetch=False, boot_addr=rom_base)
            cluster = SnitchCluster(self, f'cluster_{cluster_id}', arch, parser,
                                    entry=rom_base, auto_fetch=False)
            self.clusters.append(cluster)

            cluster.o_NARROW_SOC(narrow_axi.i_INPUT())
            cluster.o_WIDE_SOC(wide_axi.i_INPUT())
            for ico in (narrow_axi, wide_axi):
                ico.o_MAP(cluster.i_NARROW_INPUT(), name=f'cluster_{cluster_id}',
                          base=system.cluster_base(cluster_id),
                          size=system.CLUSTER_STRIDE, rm_base=False)

        #
        # Host. Boots straight out of L2, where the loader puts its binary:
        # there is no host bootrom because the manager is the thing the
        # simulation starts, not something a ROM brings up.
        #
        host = CVA6(self, 'host', isa='rv64imafdc', boot_addr=system.L2_BASE)

        icache = TimingCache(self, 'host_icache', stats=True, **memsys.ICACHE)
        dcache = TimingCache(self, 'host_dcache', stats=True, **memsys.DCACHE)

        # Data-side split: the memory levels are cacheable, everything the
        # cluster owns is not.
        host_dico = router.Router(self, 'host_dico', bandwidth=system.NARROW_WIDTH)
        for base, size in ((system.L2_BASE, system.L2_SIZE),
                           (system.L3_BASE, system.L3_SIZE),
                           (system.HYPERRAM_BASE, system.HYPERRAM_SIZE)):
            host_dico.o_MAP(dcache.i_INPUT(), name=f'cached_{base:x}', base=base,
                            size=size, rm_base=False)
        host_dico.o_MAP(narrow_axi.i_INPUT(), name='uncached_clusters',
                        base=system.CLUSTER_BASE,
                        size=nb_cluster * system.CLUSTER_STRIDE, rm_base=False)
        host_dico.o_MAP(narrow_axi.i_INPUT(), name='uncached_bootrom', base=rom_base,
                        size=rom_size, rm_base=False)

        icache.o_OUTPUT(narrow_axi.i_INPUT())
        dcache.o_OUTPUT(narrow_axi.i_INPUT())

        self.bind(host, 'fetch', icache, 'input')
        self.bind(host, 'data', host_dico, 'input')

        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary)
        loader.o_OUT(wide_axi.i_INPUT())
        self.bind(loader, 'start', host, 'fetchen')


class MeshBoard(gvsoc.systree.Component):

    def __init__(self, parent, name, parser, options, nb_cluster: int = 1):
        super().__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=system.CLOCK_FREQ)
        soc = MeshSoc(self, 'soc', parser, nb_cluster=nb_cluster)
        self.bind(clock, 'out', soc, 'clock')
