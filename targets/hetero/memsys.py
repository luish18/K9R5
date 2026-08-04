#
# Parameters of the memory system modelled by the cva6_real / snitch_real /
# spatz_real targets.
#
# The three cores keep their native memory architecture — CVA6 caches, Snitch
# and Spatz a cluster scratchpad — but they all sit behind the same main
# memory, so a cycle count difference between them is a difference in how the
# core uses memory and not in how the memory was configured.
#

# --- Main memory -----------------------------------------------------------
#
# Latency of an access that reaches main memory, in core cycles. 100 cycles is
# what GVSoC's own Snitch board already charges for its HBM; using it for CVA6
# too puts the same DRAM behind all three cores.
DRAM_LATENCY = 100

# Bytes per cycle the port to main memory sustains. Used to derive how long a
# line transfer occupies that port, so a stream of misses pays bandwidth and
# not only latency.
DRAM_WIDTH = 8

# --- CVA6 cache hierarchy --------------------------------------------------
#
# Geometry follows the CVA6 defaults (16 KiB 4-way instruction cache, 32 KiB
# 8-way write-through data cache), with one deliberate deviation: the line
# size is 64 bytes rather than the 128 bits of the RTL, because the GVSoC ISS
# fetches instructions in 64-byte granules and a shorter line would turn one
# fetch into several refills that the real front-end never issues.
LINE_SIZE = 64

# Cycles the L2 needs to answer a hit. On-chip SRAM a bus crossing away.
L2_LATENCY = 10
# Bytes per cycle between L1 and L2.
L2_WIDTH = 16

ICACHE = {
    'size': 16 * 1024,
    'ways': 4,
    'line_size': LINE_SIZE,
    # A hit costs nothing: the ISS accounts every cycle of fetch latency as a
    # full front-end stall, while the real CVA6 front-end runs ahead of the
    # pipeline and hides the access. Only refills stall the core here.
    'hit_latency': 0,
    'refill_cycles': LINE_SIZE // L2_WIDTH,
}

DCACHE = {
    'size': 32 * 1024,
    'ways': 8,
    'line_size': LINE_SIZE,
    # One cycle on top of the cycle the ISS charges anyway, i.e. the two-cycle
    # load-to-use of a CVA6 L1 hit. The scoreboard turns it into a stall only
    # when the consuming instruction is right behind the load.
    'hit_latency': 1,
    'refill_cycles': LINE_SIZE // L2_WIDTH,
    # CVA6's L1 data cache is write-through: a store updates the line and is
    # handed to the write buffer, which drains it to the next level. A store
    # that misses does not pull the line in.
    'write_allocate': False,
    'store_buffer_size': 8,
    'write_cycles': 1,
}

L2 = {
    'size': 512 * 1024,
    'ways': 8,
    'line_size': LINE_SIZE,
    'hit_latency': L2_LATENCY,
    'refill_cycles': LINE_SIZE // DRAM_WIDTH,
    # Unified and inclusive of the write traffic the L1 sends through it.
    'write_allocate': True,
    'store_buffer_size': 8,
    'write_cycles': 1,
}
