/*
 * Timing-only cache model for the hetero-sim memory hierarchy.
 *
 * The model sits between a master (a core fetch/data port, or another cache)
 * and the next memory level. Every request is forwarded verbatim to the next
 * level, so the data seen by the core is always the data in memory: there is
 * no cache storage, no dirty state, no write-back, and therefore no way for
 * the model to lose a store or return a stale line. What the model does own is
 * the *timing*: it keeps a tag array, decides hit or miss, and rewrites the
 * latency the master observes.
 *
 * Timing per request:
 *   hit   -> hit_latency
 *   miss  -> next-level latency + miss_latency + refill_cycles, with refills
 *            serialized against each other through the next-level port so a
 *            burst of misses pays the port bandwidth as well as its latency.
 *   write -> write-through. The store drains through a store buffer of
 *            store_buffer_size entries; the master only stalls once the buffer
 *            is full, which is the behaviour of the write buffer in front of a
 *            write-through L1 (CVA6's data cache is write-through).
 *
 * Because every access is forwarded, the latency the next level reports is
 * counted only on a miss: on a hit the real hardware would not have gone
 * downstream at all.
 *
 * A forwarded request keeps the size the master asked for — the cache never
 * turns a miss into a line-sized read, since the bytes it needs are exactly the
 * ones the master asked for. The cost of moving the whole line is refill_cycles,
 * and the next level indexes the same line either way, so a chain of these
 * caches agrees on what is resident even though no line-sized request is ever
 * issued.
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/debug_mem.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

// Tag value for a line that holds nothing.
#define LINE_INVALID ((uint64_t)-1)

class TimingCache : public vp::Component, public vp::DebugMemIf
{

public:
    TimingCache(vp::ComponentConf &config);

    void reset(bool active) override;
    void stop() override;

    // --- Debug-memory backdoor -------------------------------------------
    //
    // The cache holds no data, so it has nothing of its own to serve here: it
    // is transparent to a backdoor access exactly as it is to a request, and
    // both simply forward to the next level. Being transparent matters beyond
    // tidiness. The ISS v2 semi-hosting path reads its string arguments
    // through this interface rather than through the data port
    // (cpu/iss_v2/src/syscalls.cpp), resolving it by walking the LSU data port
    // to the first component that offers one. With a cache in that path and no
    // implementation here, that walk finds nothing, and every semi-hosted
    // write is silently dropped -- a program that runs correctly and prints
    // nothing.
    vp::DebugMemIf *debug_mem_if() override { return this; }
    int debug_mem_access(uint64_t addr, uint8_t *data, uint64_t size,
        bool is_write) override;
    void debug_mem_regions(std::vector<vp::DebugMemRegion> &regions,
        uint64_t local_base, uint64_t window_size, uint64_t entry_base,
        int depth) override;

private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);

    // True if every line the access touches is already resident. Changes no
    // state: it runs before the request is forwarded, to decide whether the
    // next level is being accessed for real or only as the backing store.
    bool probe(uint64_t addr, uint64_t size);
    // Returns the latency, in cycles, the master should see for this access,
    // updating the tag array and the next-level port on the way.
    int64_t access(uint64_t addr, uint64_t size, bool is_write, int64_t next_level_latency);
    // Looks up one line, allocating it (LRU victim) when it is missing. Returns
    // the cycle at which the line is readable, or -1 if the line was missing.
    int64_t lookup(uint64_t line, bool allocate, int64_t ready_time);

    // Resolved lazily on first use and then remembered; see the definition.
    vp::DebugMemIf *next_level_backdoor();
    vp::DebugMemIf *backdoor = NULL;
    bool backdoor_resolved = false;

    vp::Trace trace;
    vp::IoSlave input_itf;
    vp::IoMaster output_itf;

    // Geometry
    int line_bits;
    int nb_sets;
    int set_mask;
    int nb_ways;

    // Timing
    int64_t hit_latency;
    int64_t miss_latency;
    int64_t refill_cycles;
    int64_t write_cycles;
    int64_t store_buffer_cycles;  // depth of the store buffer, as drain cycles
    bool write_allocate;
    bool stats;

    // Tag array. Every vector is indexed by set * nb_ways + way.
    std::vector<uint64_t> tags;
    std::vector<uint64_t> lru;        // last use, for the replacement policy
    std::vector<int64_t> line_ready;  // cycle the line becomes readable
    uint64_t lru_counter;

    // Cycle at which the port to the next level is free again. Refills and
    // drained stores both queue on it, which is what makes a stream of misses
    // pay bandwidth and not just latency.
    int64_t next_level_free;

    // Power. GVSoC accumulates energy during the simulation from the activity
    // the model reports, so these are charged on the same paths as the counters
    // below rather than estimated afterwards from them. Every other level of
    // this hierarchy already has power sources (memory.cpp declares them, and
    // the ISS charges per instruction group); this cache was the one component
    // that contributed nothing, which would have read as "the caches are free".
    //
    // A refill is separate from an access on purpose: a hit touches the tag and
    // data arrays once, while a miss moves a whole line across the port to the
    // next level, and the two differ by far more than a scale factor.
    vp::PowerSource read_power;
    vp::PowerSource write_power;
    vp::PowerSource refill_power;
    vp::PowerSource background_power;

    // Statistics, reported on stop() when the 'stats' property is set.
    uint64_t nb_read;
    uint64_t nb_write;
    uint64_t nb_hit;
    uint64_t nb_miss;
    uint64_t nb_latency_cycles;
};

// The component behind the output port, if it offers a backdoor. The cache
// applies no address translation, so nothing needs adjusting on the way.
//
// Resolved once and remembered. The walk is recursive over the binding graph,
// and semi-hosting reads its strings a byte at a time, so resolving per access
// makes every character printed cost a graph traversal -- which does not show
// up on a short run and then dominates a long one.
vp::DebugMemIf *TimingCache::next_level_backdoor()
{
    if (!this->backdoor_resolved)
    {
        this->backdoor_resolved = true;
        std::vector<vp::SlavePort *> finals = this->output_itf.get_final_ports();
        if (!finals.empty() && finals[0]->get_owner() != nullptr)
        {
            this->backdoor = finals[0]->get_owner()->debug_mem_if();
        }
    }
    return this->backdoor;
}

int TimingCache::debug_mem_access(uint64_t addr, uint8_t *data, uint64_t size,
    bool is_write)
{
    vp::DebugMemIf *next = this->next_level_backdoor();
    if (next == NULL)
    {
        return -1;
    }
    return next->debug_mem_access(addr, data, size, is_write);
}

void TimingCache::debug_mem_regions(std::vector<vp::DebugMemRegion> &regions,
    uint64_t local_base, uint64_t window_size, uint64_t entry_base, int depth)
{
    // Recurse rather than take the default, which would advertise the cache
    // itself as a terminal region holding the data. It holds none.
    vp::DebugMemIf *next = this->next_level_backdoor();
    if (next != NULL)
    {
        next->debug_mem_regions(regions, local_base, window_size, entry_base,
            depth + 1);
    }
}

TimingCache::TimingCache(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    this->input_itf.set_req_meth(&TimingCache::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("output", &this->output_itf);

    int size = this->get_js_config()->get_child_int("size");
    int line_size = this->get_js_config()->get_child_int("line_size");
    this->nb_ways = this->get_js_config()->get_child_int("ways");

    this->hit_latency = this->get_js_config()->get_child_int("hit_latency");
    this->miss_latency = this->get_js_config()->get_child_int("miss_latency");
    this->refill_cycles = this->get_js_config()->get_child_int("refill_cycles");
    this->write_cycles = this->get_js_config()->get_child_int("write_cycles");
    this->write_allocate = this->get_js_config()->get_child_bool("write_allocate");
    this->stats = this->get_js_config()->get_child_bool("stats");

    // Unset in the config, new_power_source leaves the source contributing
    // nothing, so a board with no power model behaves exactly as before.
    this->power.new_power_source("background", &this->background_power,
                                 this->get_js_config()->get("**/background"));
    this->power.new_power_source("read", &this->read_power,
                                 this->get_js_config()->get("**/read"));
    this->power.new_power_source("write", &this->write_power,
                                 this->get_js_config()->get("**/write"));
    this->power.new_power_source("refill", &this->refill_power,
                                 this->get_js_config()->get("**/refill"));

    int store_buffer_size = this->get_js_config()->get_child_int("store_buffer_size");
    this->store_buffer_cycles = (int64_t)store_buffer_size * this->write_cycles;

    this->nb_sets = size / line_size / this->nb_ways;

    this->line_bits = 0;
    while ((1 << this->line_bits) < line_size)
    {
        this->line_bits++;
    }
    this->set_mask = this->nb_sets - 1;

    // The set index is extracted with a mask, and the geometry must describe
    // the size that was asked for. Both are cheap to get wrong from the
    // generator, so refuse to run rather than silently model something else.
    if ((1 << this->line_bits) != line_size || (this->nb_sets & this->set_mask) != 0 ||
        this->nb_sets * this->nb_ways * line_size != size)
    {
        this->trace.fatal("Invalid cache geometry (size: %d, line_size: %d, ways: %d): "
                          "size must be ways * line_size * a power-of-2 number of sets, "
                          "and line_size must be a power of 2\n",
                          size, line_size, this->nb_ways);
    }

    this->tags.resize(this->nb_sets * this->nb_ways);
    this->lru.resize(this->nb_sets * this->nb_ways);
    this->line_ready.resize(this->nb_sets * this->nb_ways);
}

void TimingCache::reset(bool active)
{
    if (active)
    {
        std::fill(this->tags.begin(), this->tags.end(), LINE_INVALID);
        std::fill(this->lru.begin(), this->lru.end(), 0);
        std::fill(this->line_ready.begin(), this->line_ready.end(), 0);
        this->lru_counter = 0;
        this->next_level_free = 0;
        this->nb_read = 0;
        this->nb_write = 0;
        this->nb_hit = 0;
        this->nb_miss = 0;
        this->nb_latency_cycles = 0;

        this->background_power.leakage_power_start();
        this->background_power.dynamic_power_start();
    }
    else
    {
        this->background_power.leakage_power_stop();
        this->background_power.dynamic_power_stop();
    }
}

int64_t TimingCache::lookup(uint64_t line, bool allocate, int64_t ready_time)
{
    int set = (int)(line & this->set_mask);
    uint64_t *set_tags = &this->tags[set * this->nb_ways];

    for (int way = 0; way < this->nb_ways; way++)
    {
        if (set_tags[way] == line)
        {
            this->lru[set * this->nb_ways + way] = ++this->lru_counter;
            return this->line_ready[set * this->nb_ways + way];
        }
    }

    if (!allocate)
    {
        return -1;
    }

    // Evict the least recently used way. Invalid ways have lru 0 and are
    // therefore picked first.
    int victim = 0;
    uint64_t oldest = this->lru[set * this->nb_ways];
    for (int way = 1; way < this->nb_ways; way++)
    {
        if (this->lru[set * this->nb_ways + way] < oldest)
        {
            oldest = this->lru[set * this->nb_ways + way];
            victim = way;
        }
    }

    set_tags[victim] = line;
    this->lru[set * this->nb_ways + victim] = ++this->lru_counter;
    this->line_ready[set * this->nb_ways + victim] = ready_time;

    return -1;
}

bool TimingCache::probe(uint64_t addr, uint64_t size)
{
    if (size == 0)
    {
        size = 1;
    }

    uint64_t first_line = addr >> this->line_bits;
    uint64_t last_line = (addr + size - 1) >> this->line_bits;

    for (uint64_t line = first_line; line <= last_line; line++)
    {
        int set = (int)(line & this->set_mask);
        uint64_t *set_tags = &this->tags[set * this->nb_ways];
        bool resident = false;

        for (int way = 0; way < this->nb_ways; way++)
        {
            if (set_tags[way] == line)
            {
                resident = true;
                break;
            }
        }

        if (!resident)
        {
            return false;
        }
    }

    return true;
}

int64_t TimingCache::access(uint64_t addr, uint64_t size, bool is_write,
                            int64_t next_level_latency)
{
    int64_t now = this->clock.get_cycles();

    if (size == 0)
    {
        size = 1;
    }

    if (is_write)
    {
        this->nb_write++;
        if (this->power.is_enabled())
        {
            this->write_power.account_energy_quantum();
        }
    }
    else
    {
        this->nb_read++;
        if (this->power.is_enabled())
        {
            this->read_power.account_energy_quantum();
        }
    }

    // The tag lookup itself always costs the hit latency.
    int64_t ready = now + this->hit_latency;

    uint64_t first_line = addr >> this->line_bits;
    uint64_t last_line = (addr + size - 1) >> this->line_bits;

    for (uint64_t line = first_line; line <= last_line; line++)
    {
        // A store that misses only allocates when the cache is write-allocate;
        // otherwise it goes straight to the store buffer below.
        bool allocate = !is_write || this->write_allocate;

        // Refills queue on the port to the next level: the line transfer
        // occupies it for refill_cycles, so back-to-back misses serialize.
        int64_t start = std::max(now, this->next_level_free);
        int64_t refill_done = start + next_level_latency + this->miss_latency + this->refill_cycles;

        int64_t line_ready = this->lookup(line, allocate, refill_done);

        if (line_ready >= 0)
        {
            this->nb_hit++;
            // A hit on a line whose refill has not landed yet still waits for it.
            ready = std::max(ready, line_ready);
            continue;
        }

        this->nb_miss++;

        if (!allocate)
        {
            continue;
        }

        // Past this point the line really is transferred, so this is where the
        // line-transfer energy belongs -- not on every miss, since a
        // write-through store that misses never pulls the line in.
        if (this->power.is_enabled())
        {
            this->refill_power.account_energy_quantum();
        }

        this->next_level_free = start + this->refill_cycles;
        ready = std::max(ready, refill_done);
    }

    if (is_write)
    {
        // Write-through: the store is handed to the store buffer, which drains
        // it to the next level. The master stalls only for the part of the
        // drain queue that no longer fits in the buffer.
        int64_t start = std::max(now, this->next_level_free);
        this->next_level_free = start + this->write_cycles;
        ready = std::max(ready, this->next_level_free - this->store_buffer_cycles);
    }

    int64_t latency = std::max((int64_t)0, ready - now);
    this->nb_latency_cycles += latency;

    this->trace.msg(vp::Trace::LEVEL_TRACE,
                    "Access (addr: 0x%llx, size: 0x%llx, is_write: %d, latency: %lld)\n",
                    (unsigned long long)addr, (unsigned long long)size, is_write,
                    (long long)latency);

    return latency;
}

vp::IoReqStatus TimingCache::req(vp::Block *__this, vp::IoReq *req)
{
    TimingCache *_this = (TimingCache *)__this;

    uint64_t addr = req->get_addr();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();
    bool debug = req->is_debug();

    // A read that hits would never leave this cache in real hardware. It still
    // has to, because this level holds no data, but it is marked as a debug
    // access on the way down so that the next level serves the bytes without
    // seeing it as a reference: its tag array, its port occupancy and its
    // counters must only ever see this cache's miss stream. Stores are always
    // real accesses downstream — the cache is write-through.
    bool served_here = !debug && !is_write && _this->probe(addr, size);
    if (served_here)
    {
        req->set_debug(true);
    }

    // The data always comes from the next level, so the cache can never hand
    // back something the memory does not hold.
    vp::IoReqStatus status = _this->output_itf.req_forward(req);

    if (served_here)
    {
        req->set_debug(false);
    }

    // Debug accesses (loader, gdb) must not be timed, and must not disturb the
    // tag array either: they are not accesses the core performed.
    if (debug || status == vp::IO_REQ_INVALID)
    {
        return status;
    }

    int64_t next_level_latency = (int64_t)req->get_latency();
    int64_t latency = _this->access(addr, size, is_write, next_level_latency);

    if (status == vp::IO_REQ_OK)
    {
        req->set_exact_latency(latency);
    }
    else
    {
        // The next level answered asynchronously and owns the response. Its own
        // timing already applies; add what the lookup costs on top of it.
        req->inc_latency(latency);
    }

    return status;
}

void TimingCache::stop()
{
    if (this->stats)
    {
        uint64_t nb_access = this->nb_read + this->nb_write;
        // Energy is on the same line as the counters that produced it, so a
        // result can never carry one without the other. Both are zero unless
        // the run enabled --power.
        //
        // The unit is PICOJOULES, not joules: GVSoC accumulates in whatever
        // unit the power table declares, and hetero/power.py declares pJ (as
        // the Siracusa model it is anchored on does). Verified numerically --
        // the reported figure equals accesses * per-access pJ + refills *
        // refill pJ to ten significant figures.
        //
        // KNOWN GAP: leakage reports zero. The source is registered and
        // leakage_power_start() is called on reset, the same way
        // core/models/memory/memory.cpp does it, but nothing accumulates. The
        // likely cause is that the power engine needs a supply state and
        // voltage set on the block before static power is integrated
        // (BlockPower::power_supply_set_all / voltage_set_all), which no board
        // in this repository calls. Until that is resolved, treat reported
        // energy as DYNAMIC ONLY -- which is the activity-driven part a sweep
        // over geometry is mostly comparing anyway, but it means a design
        // cannot be charged for the static cost of a larger array.
        double dynamic_pj = 0, leakage_pj = 0;
        if (this->power.is_enabled())
        {
            this->power.get_total_energy(&dynamic_pj, &leakage_pj);
        }
        printf("[HES-MEM] cache=%s accesses=%llu reads=%llu writes=%llu hits=%llu misses=%llu "
               "latency_cycles=%llu dynamic_pj=%.9g leakage_pj=%.9g\n",
               this->get_path().c_str(), (unsigned long long)nb_access,
               (unsigned long long)this->nb_read, (unsigned long long)this->nb_write,
               (unsigned long long)this->nb_hit, (unsigned long long)this->nb_miss,
               (unsigned long long)this->nb_latency_cycles, dynamic_pj, leakage_pj);
        fflush(stdout);
    }
}

extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new TimingCache(config);
}
