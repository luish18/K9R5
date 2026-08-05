/* Phase-1a validation for the mesh_real board: does each memory level answer
 * at the latency targets/hetero/system.py configured for it?
 *
 * Runs on the CVA6 manager. Builds a pointer chain in the level under test and
 * chases it, so every load depends on the one before it and the loop measures
 * latency rather than throughput.
 *
 * Two things make the measurement mean what it says:
 *
 *   - the chain is twice the host data cache, chased cyclically, so LRU never
 *     has the next hop and every access reaches the level;
 *   - the comparison is on *deltas* between levels, where the loop's own work
 *     and the cache's fixed overhead cancel exactly.
 *
 * A miss costs the level's latency plus one line at the level's width, so the
 * expected delta carries both. Getting this wrong is how a probe ends up
 * confirming whatever the model does.
 */
#include "../common/bench.h"
#include "../common/miniio.h"
#include "../mesh/system.h"

/* One hop per cache line, working set = 2x the data cache. */
#define STEP MESH_LINE_SIZE
#define HOPS (2 * MESH_DCACHE_SIZE / STEP)

static int failures = 0;

static void build_chain(volatile uint64_t *base) {
  for (unsigned i = 0; i < HOPS; i++) {
    volatile uint64_t *slot = (volatile uint64_t *)((uintptr_t)base + i * STEP);
    *slot = (uint64_t)((uintptr_t)base + ((i + 1) % HOPS) * STEP);
  }
}

/* Cycles per dependent load, chasing the whole chain twice. */
static uint64_t chase(volatile uint64_t *base) {
  uintptr_t p = (uintptr_t)base;
  uint64_t t0 = read_mcycle();
  for (unsigned i = 0; i < 2 * HOPS; i++)
    p = (uintptr_t) * (volatile uint64_t *)p;
  uint64_t t1 = read_mcycle();
  __asm__ volatile("" ::"r"(p));
  return (t1 - t0) / (2 * HOPS);
}

/* What a miss to this level costs relative to a miss to L2. */
static uint64_t expected_delta(uint64_t latency, uint64_t width) {
  uint64_t here = latency + MESH_LINE_SIZE / width;
  uint64_t l2 = MESH_L2_LATENCY + MESH_LINE_SIZE / MESH_L2_WIDTH;
  return here - l2;
}

static void report(const char *name, uint64_t cycles) {
  print_str("  ");
  print_str(name);
  print_str(": ");
  print_u64(cycles);
  print_str(" cycles/access\n");
}

static void check_delta(const char *name, uint64_t measured, uint64_t base,
                        uint64_t expected) {
  int64_t delta = (int64_t)measured - (int64_t)base;
  int64_t err = delta - (int64_t)expected;
  if (err < 0)
    err = -err;
  /* 5% of the expected step, floor 2 cycles, for rounding in the per-access
     division and the cache's own bookkeeping. */
  int64_t slack = (int64_t)expected / 20;
  if (slack < 2)
    slack = 2;
  int ok = err <= slack;
  failures += !ok;
  print_str(ok ? "  ok   " : "  FAIL ");
  print_str(name);
  print_str(": measured +");
  print_i64(delta);
  print_str(", expected +");
  print_u64(expected);
  print_str("\n");
}

int main(void) {
  print_str("mesh_probe: memory levels seen from the CVA6 manager\n");

  /* Past the manager's image, heap and stack. */
  volatile uint64_t *l2 = (volatile uint64_t *)(MESH_L2_BASE + MESH_L2_SIZE / 2);
  volatile uint64_t *l3 = (volatile uint64_t *)MESH_L3_BASE;
  volatile uint64_t *hyper = (volatile uint64_t *)MESH_HYPERRAM_BASE;
  volatile uint64_t *tcdm = mesh_cluster_tcdm(0);

  build_chain(l2);
  build_chain(l3);
  build_chain(hyper);

  uint64_t c_l2 = chase(l2);
  uint64_t c_l3 = chase(l3);
  uint64_t c_hyper = chase(hyper);

  report("L2      ", c_l2);
  report("L3      ", c_l3);
  report("HyperRAM", c_hyper);

  print_str("\n");
  check_delta("L3 vs L2      ", c_l3, c_l2,
              expected_delta(MESH_L3_LATENCY, MESH_L3_WIDTH));
  check_delta("HyperRAM vs L2", c_hyper, c_l2,
              expected_delta(MESH_HYPERRAM_LATENCY, MESH_HYPERRAM_WIDTH));

  /* The cluster TCDM is single-cycle memory one router hop away, and the host
     maps it uncached, so it must come back faster than a cached miss to L2.
     The chain only needs to fit TCDM here, not beat a cache. */
  for (unsigned i = 0; i < MESH_L1_SIZE / STEP; i++) {
    volatile uint64_t *slot = (volatile uint64_t *)((uintptr_t)tcdm + i * STEP);
    *slot = (uint64_t)((uintptr_t)tcdm + ((i + 1) % (MESH_L1_SIZE / STEP)) * STEP);
  }
  uintptr_t p = (uintptr_t)tcdm;
  uint64_t t0 = read_mcycle();
  for (unsigned i = 0; i < 256; i++)
    p = (uintptr_t) * (volatile uint64_t *)p;
  uint64_t c_tcdm = (read_mcycle() - t0) / 256;
  __asm__ volatile("" ::"r"(p));
  report("cluster 0 TCDM (uncached)", c_tcdm);

  int tcdm_ok = c_tcdm < c_l2;
  failures += !tcdm_ok;
  print_str(tcdm_ok ? "  ok   " : "  FAIL ");
  print_str("cluster TCDM is nearer than a cached L2 miss\n");

  /* Every level answering in the same handful of cycles is not a board bug: it
   * is the GVSoC CVA6 model not charging load-to-use latency at all, which the
   * pre-existing cva6_real target does too. Say so, rather than leaving three
   * bare FAILs that look like a mistake in the address map. */
  if (c_l2 == c_l3 && c_l3 == c_hyper) {
    print_str("\n  note: all levels measured identical. The CVA6 model commits a\n"
              "  dependent instruction without waiting for the load's data, so no\n"
              "  data-side latency reaches mcycle. See docs/hetero-mesh-plan.md,\n"
              "  \"Findings\" — this blocks phase 1 validation on the host core.\n");
  }

  print_str(failures ? "mesh_probe: FAILED\n" : "mesh_probe: OK\n");
  return failures != 0;
}
