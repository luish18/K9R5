/* A dependent-load latency probe, shared by the host and the cluster sides.
 *
 * Every iteration's address depends on the value the previous load returned,
 * so the loads cannot overlap and the measured cycles per access is the real
 * round trip to whichever level of the memory system holds the line.
 *
 * The chain is built in place: chain[i] holds the index of the next element,
 * so walking it is one load per step. With `stride` larger than a cache line
 * each step touches a new line, and with a span larger than a cache the walk
 * misses in it -- which is what separates a hit measurement from a miss one.
 */
#ifndef HES_PROBE_H
#define HES_PROBE_H

#include <stdint.h>

/* Slots of `chain` to touch, and how far apart. Kept small enough that the
 * pointer chain itself fits comfortably in a cluster TCDM. */
static inline void hes_chain_init(volatile uint32_t *chain, uint32_t nslots,
                                  uint32_t stride_slots) {
  /* A single cycle through every slot we intend to visit, so the walk never
   * leaves the region and never repeats a line before it has to. */
  uint32_t steps = nslots / stride_slots;
  for (uint32_t i = 0; i < steps; i++) {
    chain[i * stride_slots] = ((i + 1) % steps) * stride_slots;
  }
}

/* Walk the chain `steps` times and return the total cycles. The caller
 * divides by `steps` to get cycles per dependent access. */
static inline uint32_t hes_chain_walk(volatile uint32_t *chain, uint32_t steps,
                                      uint32_t (*read_cycle)(void)) {
  uint32_t idx = 0;
  uint32_t t0 = read_cycle();
  for (uint32_t i = 0; i < steps; i++) {
    idx = chain[idx];
  }
  uint32_t t1 = read_cycle();
  /* Keep the walk from being optimized away. */
  __asm__ volatile("" ::"r"(idx));
  return t1 - t0;
}

#endif /* HES_PROBE_H */
