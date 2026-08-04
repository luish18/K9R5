#
# Generator for the timing-only cache used by the hetero-sim memory hierarchy.
# See timing_cache.cpp for what the model does and does not simulate.
#

import gvsoc.systree


class TimingCache(gvsoc.systree.Component):
    """A cache that models timing only: every access is forwarded to the next
    memory level, and the tag array is used to rewrite the latency the master
    observes.

    Attributes
    ----------
    size: int
        Total capacity in bytes. Must be ways * line_size * a power-of-2
        number of sets.
    line_size: int
        Line size in bytes, and the granularity of a refill.
    ways: int
        Associativity. Replacement within a set is LRU.
    hit_latency: int
        Cycles a hit costs the master.
    miss_latency: int
        Cycles the next memory level takes to answer a refill, on top of any
        latency that level reports itself.
    refill_cycles: int
        Cycles a line transfer occupies the port to the next level, i.e.
        line_size divided by the width of that port. Refills queue on it, so a
        stream of misses pays bandwidth as well as latency.
    write_cycles: int
        Cycles one drained store occupies the port to the next level.
    write_allocate: bool
        True if a store that misses refills the line. A write-through L1
        normally does not (the store goes straight to the store buffer).
    store_buffer_size: int
        Number of stores the write buffer holds. The master stalls on a store
        only once the buffer is full.
    stats: bool
        True to print a [HES-MEM] counters line at the end of the simulation.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, size: int,
            line_size: int = 64, ways: int = 4, hit_latency: int = 1,
            miss_latency: int = 0, refill_cycles: int = 1, write_cycles: int = 1,
            write_allocate: bool = False, store_buffer_size: int = 0,
            stats: bool = False):

        super().__init__(parent, name)

        self.add_sources(['hetero/timing_cache.cpp'])

        self.add_properties({
            'size': size,
            'line_size': line_size,
            'ways': ways,
            'hit_latency': hit_latency,
            'miss_latency': miss_latency,
            'refill_cycles': refill_cycles,
            'write_cycles': write_cycles,
            'write_allocate': write_allocate,
            'store_buffer_size': store_buffer_size,
            'stats': stats,
        })

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        """Master-facing port. Requests to be looked up are sent here."""
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        """Binds the port to the next memory level. Every request goes through
        it, hit or miss."""
        self.itf_bind('output', itf, signature='io')
