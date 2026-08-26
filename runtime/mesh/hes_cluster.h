/* Which cluster this translation unit is being built for, and the cluster
 * primitives the runtime needs from it.
 *
 * The build selects a cluster with -DHES_CLUSTER_SNITCH or -DHES_CLUSTER_SPATZ;
 * everything below resolves to that cluster's numbers from hes_system.h, so
 * the same source builds for both.
 *
 * Safe to include from .S: the C helpers are hidden from the assembler.
 */
#ifndef HES_CLUSTER_H
#define HES_CLUSTER_H

#include "hes_system.h"

#if defined(HES_CLUSTER_SNITCH)
#define HES_MY_NAME "snitch"
#define HES_MY_ENGINE HES_ENGINE_SNITCH
#define HES_MY_BASE HES_SNITCH_BASE
#define HES_MY_TCDM HES_SNITCH_TCDM
#define HES_MY_PERIPH HES_SNITCH_PERIPH
#define HES_MY_NB_CORE HES_SNITCH_NB_CORE
#define HES_MY_DMA_CORE HES_SNITCH_DMA_CORE
#define HES_MY_CTRL_CORE HES_SNITCH_CTRL_CORE
#define HES_MY_FIRST_HARTID HES_SNITCH_FIRST_HARTID
#define HES_MY_CL_CLINT_SET HES_SNITCH_CL_CLINT_SET
#define HES_MY_CL_CLINT_CLEAR HES_SNITCH_CL_CLINT_CLEAR
#define HES_MY_HW_BARRIER HES_SNITCH_HW_BARRIER
#define HES_MY_PERF_COUNTER HES_SNITCH_PERF_COUNTER
#define HES_MY_PERF_COUNTER_ENABLE HES_SNITCH_PERF_COUNTER_ENABLE
#define HES_MY_NB_PERF_COUNTERS HES_SNITCH_NB_PERF_COUNTERS
#define HES_MY_LOAD_SCRATCH HES_SNITCH_LOAD_SCRATCH
#define HES_MY_LOAD_SCRATCH_SIZE HES_SNITCH_LOAD_SCRATCH_SIZE
#elif defined(HES_CLUSTER_SPATZ)
#define HES_MY_NAME "spatz"
#define HES_MY_ENGINE HES_ENGINE_SPATZ
#define HES_MY_BASE HES_SPATZ_BASE
#define HES_MY_TCDM HES_SPATZ_TCDM
#define HES_MY_PERIPH HES_SPATZ_PERIPH
#define HES_MY_NB_CORE HES_SPATZ_NB_CORE
#define HES_MY_DMA_CORE HES_SPATZ_DMA_CORE
#define HES_MY_CTRL_CORE HES_SPATZ_CTRL_CORE
#define HES_MY_FIRST_HARTID HES_SPATZ_FIRST_HARTID
#define HES_MY_CL_CLINT_SET HES_SPATZ_CL_CLINT_SET
#define HES_MY_CL_CLINT_CLEAR HES_SPATZ_CL_CLINT_CLEAR
#define HES_MY_HW_BARRIER HES_SPATZ_HW_BARRIER
#define HES_MY_PERF_COUNTER HES_SPATZ_PERF_COUNTER
#define HES_MY_PERF_COUNTER_ENABLE HES_SPATZ_PERF_COUNTER_ENABLE
#define HES_MY_NB_PERF_COUNTERS HES_SPATZ_NB_PERF_COUNTERS
#define HES_MY_LOAD_SCRATCH HES_SPATZ_LOAD_SCRATCH
#define HES_MY_LOAD_SCRATCH_SIZE HES_SPATZ_LOAD_SCRATCH_SIZE
#else
#error "build a cluster source with -DHES_CLUSTER_SNITCH or -DHES_CLUSTER_SPATZ"
#endif

/* The barrier CSR of the Snitch core models. Reading it requests the cluster
 * barrier and stalls the core until every core in the cluster has done the
 * same -- a hardware stall, so a core waiting here costs no cycles and no
 * simulation time. */
#define HES_CSR_BARRIER 0x7C2

#ifndef __ASSEMBLER__

#include <stdint.h>

static inline volatile uint32_t *hes_periph(uint32_t offset) {
  return (volatile uint32_t *)(uintptr_t)(HES_MY_PERIPH + offset);
}

/* Cluster-local index of the core running this code, 0 .. HES_MY_NB_CORE-1. */
static inline uint32_t hes_core_idx(void) {
  uint32_t hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  return hartid - HES_MY_FIRST_HARTID;
}

static inline int hes_is_ctrl_core(void) {
  return hes_core_idx() == HES_MY_CTRL_CORE;
}

static inline int hes_is_dma_core(void) {
  return hes_core_idx() == HES_MY_DMA_CORE;
}

/* Barrier across every core of the cluster. All of them must call it --
 * including the DMA core -- because the cluster's barrier register only
 * releases once each one has requested it. */
static inline void hes_barrier(void) {
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("csrr x0, %0" ::"i"(HES_CSR_BARRIER) : "memory");
  __asm__ volatile("" ::: "memory");
}

/* Raise / lower the cluster-local interrupt of the cores in `mask`. Writing
 * CL_CLINT_SET drives IRQ HES_CLUSTER_IRQ of every core whose bit is set. */
static inline void hes_clint_set(uint32_t mask) {
  *hes_periph(HES_MY_CL_CLINT_SET) = mask;
}

static inline void hes_clint_clear(uint32_t mask) {
  *hes_periph(HES_MY_CL_CLINT_CLEAR) = mask;
}

#endif /* __ASSEMBLER__ */

#endif /* HES_CLUSTER_H */
