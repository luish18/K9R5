/* Cluster side of the M1 board check.
 *
 * Every core reports that it booted, then the control core measures what a
 * dependent load costs it from its own TCDM and from main memory, and leaves
 * both numbers in the mailbox for the host to print. The cores then park:
 * this image has no job loop yet, it exists to prove the board is wired the
 * way targets/hetero/system.py says it is.
 */

#include "hes_cluster.h"
#include "hes_mailbox.h"
#include "hes_probe.h"

#include <stdint.h>

static inline uint32_t rdcycle32(void) {
  uint32_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}

/* Walked out of TCDM: a cluster core's own scratchpad. */
#define TCDM_SLOTS 1024
static volatile uint32_t tcdm_chain[TCDM_SLOTS];

/* Walked out of main memory. Placed in the cluster's .data, which lives in
 * TCDM, so it is pointed at main memory explicitly below instead. */
#define HBM_SLOTS 1024

enum { PROBE_ALIVE = 0, PROBE_TCDM = 1, PROBE_HBM = 2, PROBE_NB_CORE = 3 };

/* Line size of the cluster's path to main memory, in 4-byte slots: stride by
 * more than that so consecutive steps cannot share a refill. */
#define STRIDE_SLOTS 32

int main(void) {
  hes_mailbox_t *mbox = hes_mailbox_at(HES_MY_TCDM);
  uint32_t core = hes_core_idx();

  /* Each core signs in. A plain OR is safe here only because the barrier
   * below separates it from any reader. */
  if (hes_is_ctrl_core()) {
    mbox->alive_mask = 0;
    mbox->trap_cause = 0;
  }
  hes_barrier();

  for (uint32_t i = 0; i < HES_MY_NB_CORE; i++) {
    if (i == core) {
      mbox->alive_mask |= (1u << core);
    }
    hes_barrier();
  }

  if (hes_is_ctrl_core()) {
    /* TCDM: single-cycle banked scratchpad, one interleaver crossing away. */
    hes_chain_init(tcdm_chain, TCDM_SLOTS, STRIDE_SLOTS);
    uint32_t steps = TCDM_SLOTS / STRIDE_SLOTS;
    mbox->probe[PROBE_TCDM] =
        hes_chain_walk(tcdm_chain, steps * 8, rdcycle32) / (steps * 8);

    /* Main memory, reached over the cluster's narrow AXI. The chain is built
     * in the cluster's own window of main memory, well past its code. */
    volatile uint32_t *hbm_chain =
        (volatile uint32_t *)(uintptr_t)(HES_MY_LOAD_SCRATCH);
    hes_chain_init(hbm_chain, HBM_SLOTS, STRIDE_SLOTS);
    steps = HBM_SLOTS / STRIDE_SLOTS;
    mbox->probe[PROBE_HBM] =
        hes_chain_walk(hbm_chain, steps * 4, rdcycle32) / (steps * 4);

    mbox->probe[PROBE_ALIVE] = mbox->alive_mask;
    mbox->probe[PROBE_NB_CORE] = HES_MY_NB_CORE;
    mbox->nb_cores = HES_MY_NB_CORE;

    /* Published last: the host waits on this. */
    __asm__ volatile("" ::: "memory");
    mbox->magic = HES_MBOX_MAGIC;
  }

  hes_barrier();

  /* No job loop in this image. Park without exiting: only the host ends the
   * simulation. */
  for (;;) {
    __asm__ volatile("wfi");
  }
  return 0;
}
