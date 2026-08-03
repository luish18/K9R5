#
# CVA6 virtual board with idealized (zero-latency) memory.
# Copy of gvsoc's pulp/cva6.py target with `latency=0` on the memory
# mappings, so the simulation measures compute, not memory ("infinite
# cache" assumption of the hetero-sim pipeline).
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


class Soc(st.Component):

    def __init__(self, parent, name, parser):
        super().__init__(parent, name)

        [args, __] = parser.parse_known_args()

        binary = None
        if parser is not None:
            [args, otherArgs] = parser.parse_known_args()
            binary = args.binary

        dram = memory.Memory(self, 'dram', size=0x10000000, atomics=True, width_log2=-1)

        mem = memory.Memory(self, 'mem', size=0x80000000, atomics=True, width_log2=-1)
        regs = pulp.cva6.control_regs.ControlRegs(self, 'control_regs', dram_end=0xc0000000)

        stdout = Stdout(self, 'stdout')

        ico = router.Router(self, 'ico')

        # latency=0: ideal single-cycle memory
        ico.add_mapping('mem', base=0x80000000, remove_offset=0x80000000, size=0x80000000, latency=0)
        self.bind(ico, 'mem', mem, 'input')

        ico.o_MAP(stdout.i_INPUT(), name='stdout', base=0xC0000000, size=0x10000000)
        ico.o_MAP(dram.i_INPUT(), name='dram', base=0xB0000000, size=0x10000000, latency=0)
        ico.o_MAP(regs.i_INPUT(), name='control_regs', base=0xD0000000, size=0x10000000)

        host = pulp.cva6.cva6.CVA6(self, 'host', isa='rv64imafdc', boot_addr=0x80000000)

        loader = utils.loader.loader.ElfLoader(self, 'loader', binary=binary)

        self.bind(host, 'data', ico, 'input')
        self.bind(host, 'fetch', ico, 'input')
        self.bind(loader, 'out', ico, 'input')
        self.bind(loader, 'start', host, 'fetchen')


class Cva6IdealChip(st.Component):

    def __init__(self, parent, name, parser, options):

        super(Cva6IdealChip, self).__init__(parent, name, options=options)

        clock = Clock_domain(self, 'clock', frequency=10000000)

        soc = Soc(self, 'soc', parser)

        self.bind(clock, 'out', soc, 'clock')


class Target(gvsoc.Target):

    gapy_description = "CVA6 virtual board (ideal zero-latency memory)"

    def __init__(self, parser, options):
        super(Target, self).__init__(parser, options,
            model=Cva6IdealChip)
