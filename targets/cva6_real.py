#
# CVA6 virtual board with a modelled memory system.
#
# Same board as targets/cva6_ideal.py, but the zero-latency flat memory is
# replaced by the hierarchy a CVA6 SoC actually has:
#
#     host.fetch ── L1 I$ ─┐
#                          ├─ l2_ico ── L2 ── mem_ico ── DRAM
#     host.data ── dico ───┘                              (DRAM_LATENCY)
#                    ├── stdout / control_regs / dram      uncached
#                    └── (only 0x8000_0000 is cacheable)
#
# The loader is bound behind the caches so the binary image does not warm
# them: the core starts with everything cold, as it would after reset.
#
# Cache geometry and latencies come from hetero/memsys.py.
#

import gvsoc.runner
import pulp.cva6.cva6
import memory.memory as memory
import pulp.cva6.control_regs
from vp.clock_domain import Clock_domain
import interco.router as router
import utils.loader.loader
import gvsoc.systree as st
from elftools.elf.elffile import *
import gvsoc.runner as gvsoc
from pulp.stdout.stdout_v3 import Stdout

from hetero.timing_cache import TimingCache
from hetero import memsys

MEM_BASE = 0x80000000
MEM_SIZE = 0x80000000


class Soc(st.Component):

    def __init__(self, parent, name, parser):
        super().__init__(parent, name)

        [args, __] = parser.parse_known_args()
        binary = args.binary

        dram = memory.Memory(self, 'dram', size=0x10000000, atomics=True, width_log2=-1)

        # Main memory. It is a purely functional model: all of the timing of
        # the memory system lives in the caches and in the mapping latency
        # below, because every access — hit or miss — is forwarded to it and
        # bandwidth applied here would also be charged to hits.
        mem = memory.Memory(self, 'mem', size=MEM_SIZE, atomics=True, width_log2=-1)
        regs = pulp.cva6.control_regs.ControlRegs(self, 'control_regs', dram_end=0xc0000000)

        stdout = Stdout(self, 'stdout')

        # Data-side router: splits the cacheable main memory from the
        # uncacheable peripherals, which a cache must never hold.
        dico = router.Router(self, 'dico')
        # Shared port into the L2, fed by both L1 caches.
        l2_ico = router.Router(self, 'l2_ico')
        # Main memory port, also used by the loader so that it bypasses the
        # caches.
        mem_ico = router.Router(self, 'mem_ico')

        icache = TimingCache(self, 'icache', stats=True, **memsys.ICACHE)
        dcache = TimingCache(self, 'dcache', stats=True, **memsys.DCACHE)
        l2 = TimingCache(self, 'l2', stats=True, **memsys.L2)

        host = pulp.cva6.cva6.CVA6(self, 'host', isa='rv64imafdc', boot_addr=MEM_BASE)

        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary)

        #
        # Bindings
        #

        # Data side: main memory through the L1 data cache, everything else
        # straight to the peripheral.
        dico.o_MAP(dcache.i_INPUT(), name='mem', base=MEM_BASE, size=MEM_SIZE, rm_base=False)
        dico.o_MAP(stdout.i_INPUT(), name='stdout', base=0xC0000000, size=0x10000000)
        dico.o_MAP(dram.i_INPUT(), name='dram', base=0xB0000000, size=0x10000000,
            latency=memsys.DRAM_LATENCY)
        dico.o_MAP(regs.i_INPUT(), name='control_regs', base=0xD0000000, size=0x10000000)

        # Both L1 caches refill from the L2.
        icache.o_OUTPUT(l2_ico.i_INPUT())
        dcache.o_OUTPUT(l2_ico.i_INPUT(1))
        l2_ico.o_MAP(l2.i_INPUT(), name='l2', base=MEM_BASE, size=MEM_SIZE, rm_base=False)

        # The L2 refills from main memory. The mapping latency is what a miss
        # all the way down to DRAM costs; the L2 charges it only on a miss.
        l2.o_OUTPUT(mem_ico.i_INPUT())
        mem_ico.o_MAP(mem.i_INPUT(), name='mem', base=MEM_BASE, size=MEM_SIZE, rm_base=True,
            latency=memsys.DRAM_LATENCY)
        # The loader shares this router, so it needs to reach every region a
        # program segment can be linked into.
        mem_ico.o_MAP(dram.i_INPUT(), name='dram', base=0xB0000000, size=0x10000000,
            latency=memsys.DRAM_LATENCY)

        self.bind(host, 'data', dico, 'input')
        self.bind(host, 'fetch', icache, 'input')
        self.bind(loader, 'out', mem_ico, 'input')
        self.bind(loader, 'start', host, 'fetchen')


class Cva6RealChip(st.Component):

    def __init__(self, parent, name, parser, options):

        super(Cva6RealChip, self).__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=10000000)

        soc = Soc(self, 'soc', parser)

        self.bind(clock, 'out', soc, 'clock')


class Target(gvsoc.Target):

    gapy_description = "CVA6 virtual board (modelled memory system: L1 I$/D$, L2, DRAM)"

    def __init__(self, parser, options):
        super(Target, self).__init__(parser, options,
            model=Cva6RealChip)
