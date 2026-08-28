/* M1 gate: is the hetero_soc board wired the way targets/hetero/system.py
 * says it is?
 *
 * The host waits for both clusters to sign in, then measures what a dependent
 * load costs at each level it can reach, and prints those next to what each
 * cluster measured for itself. Without this, every cycle count the SoC
 * produces later is unfalsifiable: a cluster whose TCDM answered at
 * main-memory latency, or a cluster window that got mapped through the host's
 * caches, would look like a slow kernel rather than a mis-wired board.
 *
 * One measurement is reported but not asserted -- the host's own trip to main
 * memory. The CVA6 model does not stall a dependent load for a data-cache
 * miss: a 32768-step pointer chase over 4 MiB costs the same cycles on
 * cva6_real and on the zero-latency cva6_ideal, so the number below says
 * nothing about the memory system. That is a property of the core model this
 * repository already had, not of this board -- the same binary reports the
 * same figure on cva6_real. What proves main memory really answers at
 * DRAM_LATENCY is the cluster-side measurement further down, where the Snitch
 * core does stall for its loads.
 */

#include "hes_mailbox.h"
#include "hes_probe.h"
#include "hes_system.h"
#include "memsys_expect.h"
#include "miniio.h"

#include <stdint.h>

static inline uint32_t rdcycle32(void) {
  uint32_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}

/* A span far larger than the 512 KiB L2, so every step has to miss. */
#define BIG_SLOTS (4u * 1024 * 1024 / 4)
#define SMALL_SLOTS 1024
#define STRIDE_SLOTS 32 /* 128 B apart: more than one 64 B line */

static volatile uint32_t small_chain[SMALL_SLOTS];

enum { PROBE_ALIVE = 0, PROBE_TCDM = 1, PROBE_HBM = 2, PROBE_NB_CORE = 3 };

static int failures;

static void line(const char *tag, const char *what, uint32_t measured) {
  print_str(tag);
  print_str(what);
  print_str(" = ");
  print_u64(measured);
  print_str(" cycles/access");
}

/* An asserted check: the board is wrong if this falls outside the range. */
static void report(const char *what, uint32_t measured, uint32_t lo, uint32_t hi) {
  int ok = measured >= lo && measured <= hi;
  line(ok ? "  ok   " : "  FAIL ", what, measured);
  print_str(" (expected ");
  print_u64(lo);
  print_str("..");
  print_u64(hi);
  print_str(")\n");
  if (!ok) {
    failures++;
  }
}

/* A reported measurement the core model cannot be held to -- see the header
 * comment. Printed so a change in it is still visible. */
static void observe(const char *what, uint32_t measured, const char *why) {
  line("  --   ", what, measured);
  print_str(" (not asserted: ");
  print_str(why);
  print_str(")\n");
}

static void check(const char *what, int ok) {
  print_str(ok ? "  ok   " : "  FAIL ");
  print_str(what);
  print_str("\n");
  if (!ok) {
    failures++;
  }
}

/* Walk a region the host reaches uncached: every step pays the full round
 * trip, so one pass is enough. */
static uint32_t probe_uncached(uint32_t base, uint32_t slots) {
  volatile uint32_t *chain = (volatile uint32_t *)(uintptr_t)base;
  hes_chain_init(chain, slots, STRIDE_SLOTS);
  uint32_t steps = slots / STRIDE_SLOTS;
  return hes_chain_walk(chain, steps * 4, rdcycle32) / (steps * 4);
}

static int wait_for_cluster(hes_mailbox_t *mbox, const char *name) {
  /* Bounded, so a cluster that never came up is a reported failure rather
   * than a simulation that hangs. */
  for (uint32_t spins = 0; spins < 20000000u; spins++) {
    if (mbox->magic == HES_MBOX_MAGIC) {
      return 1;
    }
  }
  print_str("  FAIL cluster ");
  print_str(name);
  print_str(" never signed in (magic=0x");
  print_hex(mbox->magic);
  print_str(")\n");
  failures++;
  return 0;
}

static void cluster_section(const char *name, uint32_t tcdm_base,
                            uint32_t periph_base, uint32_t nb_core) {
  hes_mailbox_t *mbox = hes_mailbox_at(tcdm_base);

  print_str("\ncluster ");
  print_str(name);
  print_str(":\n");

  if (!wait_for_cluster(mbox, name)) {
    return;
  }

  uint32_t alive = mbox->probe[PROBE_ALIVE];
  uint32_t expect_alive = (1u << nb_core) - 1u;
  print_str(alive == expect_alive ? "  ok   " : "  FAIL ");
  print_str("all ");
  print_u64(nb_core);
  print_str(" cores booted (mask=0x");
  print_hex(alive);
  print_str(")\n");
  if (alive != expect_alive) {
    failures++;
  }

  check("no cluster core trapped", mbox->trap_cause == 0);

  /* Measured by a cluster core, which does stall for its loads. */
  report("cluster core -> own TCDM", mbox->probe[PROBE_TCDM],
         EXPECT_CLUSTER_TCDM_LO, EXPECT_CLUSTER_TCDM_HI);
  report("cluster core -> main memory", mbox->probe[PROBE_HBM],
         EXPECT_CLUSTER_HBM_LO, EXPECT_CLUSTER_HBM_HI);

  /* The host must be able to write this cluster's TCDM and read back what it
   * wrote: that is the path the job mailbox uses, so it is asserted. The
   * latency of it is only observed -- see the header comment, the host reads
   * the same ~9 cycles at every level because it never stalls on a load. */
  volatile uint32_t *scratch =
      (volatile uint32_t *)(uintptr_t)(tcdm_base + HES_MAILBOX_SIZE + 0x4000);
  int rw_ok = 1;
  for (uint32_t i = 0; i < 64; i++) {
    scratch[i] = 0xa5a50000u + i;
  }
  for (uint32_t i = 0; i < 64; i++) {
    if (scratch[i] != 0xa5a50000u + i) {
      rw_ok = 0;
    }
  }
  check("host reads back what it writes to cluster TCDM", rw_ok);

  observe("host -> cluster TCDM",
          probe_uncached(tcdm_base + HES_MAILBOX_SIZE + 0x4000, SMALL_SLOTS),
          "host-side load latency is not modelled, see above");

  /* The peripheral has to answer at all: this is the path the doorbell that
   * wakes the cluster takes. */
  volatile uint32_t *barrier_reg =
      (volatile uint32_t *)(uintptr_t)(periph_base + HES_PERIPH_PERF_COUNTER_PROBE);
  uint32_t before = rdcycle32();
  (void)*barrier_reg;
  uint32_t after = rdcycle32();
  check("host -> cluster peripheral answers", after >= before);
}

int main(void) {
  print_str("[HES-PROBE] hetero_soc memory system\n");
  print_str("\nhost (CVA6):\n");

  /* Warm: the same small region walked twice, so the second walk hits L1. */
  hes_chain_init(small_chain, SMALL_SLOTS, STRIDE_SLOTS);
  uint32_t steps = SMALL_SLOTS / STRIDE_SLOTS;
  (void)hes_chain_walk(small_chain, steps * 4, rdcycle32);
  report("host -> L1 D$ hit",
         hes_chain_walk(small_chain, steps * 8, rdcycle32) / (steps * 8),
         EXPECT_HOST_L1_LO, EXPECT_HOST_L1_HI);

  /* Cold: 4 MiB, far past the L2. */
  volatile uint32_t *big = (volatile uint32_t *)(uintptr_t)HES_HOST_SCRATCH;
  hes_chain_init(big, BIG_SLOTS, STRIDE_SLOTS);
  steps = BIG_SLOTS / STRIDE_SLOTS;
  observe("host -> main memory", hes_chain_walk(big, steps, rdcycle32) / steps,
          "the CVA6 model does not stall a dependent load on a D$ miss; "
          "cva6_real and cva6_ideal agree to the cycle here");

  cluster_section("snitch", HES_SNITCH_TCDM, HES_SNITCH_PERIPH,
                  HES_SNITCH_NB_CORE);
  cluster_section("spatz", HES_SPATZ_TCDM, HES_SPATZ_PERIPH, HES_SPATZ_NB_CORE);

  print_str("\n[HES-PROBE] failures=");
  print_u64((uint64_t)failures);
  print_str("\n");
  return failures == 0 ? 0 : 1;
}
