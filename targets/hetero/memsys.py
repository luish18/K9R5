#
# Parameters of the memory system modelled by the cva6_real / snitch_real /
# spatz_real targets.
#
# The three cores keep their native memory architecture — CVA6 caches, Snitch
# and Spatz a cluster scratchpad — but they all sit behind the same main
# memory, so a cycle count difference between them is a difference in how the
# core uses memory and not in how the memory was configured.
#
# Every number here is a design(...) default, so a sweep can override it from a
# JSON design file -- see hetero/design.py. The defaults are the values this
# repository committed, and `gen_system_header.py --check` with no HES_DESIGN
# set is what keeps them that way.
#
# The cache *geometry* (sizes, ways, line size) is what a designer picks when
# generating RTL and is what the sweep moves. The latencies and port widths
# below are modelling assumptions about the environment rather than choices
# about this chip, so they stay at their measured/realistic values in a sweep
# even though they are overridable here for sensitivity studies.
#

from hetero.design import get as _d

# --- Main memory -----------------------------------------------------------
#
# Latency of an access that reaches main memory, in core cycles. 100 cycles is
# what GVSoC's own Snitch board already charges for its HBM; using it for CVA6
# too puts the same DRAM behind all three cores.
DRAM_LATENCY = _d('DRAM_LATENCY', 100)

# Bytes per cycle the port to main memory sustains. Used to derive how long a
# line transfer occupies that port, so a stream of misses pays bandwidth and
# not only latency.
DRAM_WIDTH = _d('DRAM_WIDTH', 8)

# --- CVA6 cache hierarchy --------------------------------------------------
#
# Geometry follows the CVA6 defaults (16 KiB 4-way instruction cache, 32 KiB
# 8-way write-through data cache), with one deliberate deviation: the line
# size is 64 bytes rather than the 128 bits of the RTL, because the GVSoC ISS
# fetches instructions in 64-byte granules and a shorter line would turn one
# fetch into several refills that the real front-end never issues.
LINE_SIZE = _d('LINE_SIZE', 64)

# Cycles the L2 needs to answer a hit. On-chip SRAM a bus crossing away.
L2_LATENCY = _d('L2_LATENCY', 10)
# Bytes per cycle between L1 and L2.
L2_WIDTH = _d('L2_WIDTH', 16)

ICACHE = {
    'size': _d('ICACHE_SIZE', 16 * 1024),
    'ways': _d('ICACHE_WAYS', 4),
    'line_size': LINE_SIZE,
    # A hit costs nothing: the ISS accounts every cycle of fetch latency as a
    # full front-end stall, while the real CVA6 front-end runs ahead of the
    # pipeline and hides the access. Only refills stall the core here.
    'hit_latency': _d('ICACHE_HIT_LATENCY', 0),
    'refill_cycles': LINE_SIZE // L2_WIDTH,
}

DCACHE = {
    'size': _d('DCACHE_SIZE', 32 * 1024),
    'ways': _d('DCACHE_WAYS', 8),
    'line_size': LINE_SIZE,
    # One cycle on top of the cycle the ISS charges anyway, i.e. the two-cycle
    # load-to-use of a CVA6 L1 hit. The scoreboard turns it into a stall only
    # when the consuming instruction is right behind the load.
    'hit_latency': _d('DCACHE_HIT_LATENCY', 1),
    'refill_cycles': LINE_SIZE // L2_WIDTH,
    # CVA6's L1 data cache is write-through: a store updates the line and is
    # handed to the write buffer, which drains it to the next level. A store
    # that misses does not pull the line in.
    'write_allocate': False,
    'store_buffer_size': _d('DCACHE_STORE_BUFFER_SIZE', 8),
    'write_cycles': 1,
}

L2 = {
    'size': _d('L2_SIZE', 512 * 1024),
    'ways': _d('L2_WAYS', 8),
    'line_size': LINE_SIZE,
    'hit_latency': L2_LATENCY,
    'refill_cycles': LINE_SIZE // DRAM_WIDTH,
    # Unified and inclusive of the write traffic the L1 sends through it.
    'write_allocate': True,
    'store_buffer_size': _d('L2_STORE_BUFFER_SIZE', 8),
    'write_cycles': 1,
}
