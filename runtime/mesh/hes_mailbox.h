/* The job mailbox: how the CVA6 host hands work to a cluster.
 *
 * One descriptor lives in the first HES_MAILBOX_SIZE bytes of each cluster's
 * TCDM, below everything the cluster links there (the generated cluster
 * linker scripts start TCDM above it). The host writes it over the narrow
 * AXI; the cluster reads it out of its own single-cycle scratchpad.
 *
 * Protocol, per job:
 *
 *   host                                cluster control core   other cores
 *   ------------------------------------------------------------------------
 *   write args, kernel_id               wfi                    barrier
 *   seq = seq + 1
 *   CL_CLINT_SET = 1 << ctrl_core  -->  wake, clear own bit
 *   poll done_seq                       barrier -------------> released
 *                                       run slice              run slice
 *                                       barrier <------------- barrier
 *                                       cycles, done_seq = seq
 *   done_seq == seq: job finished  <--  wfi                    barrier
 *
 * `seq` rather than a flag so a stale `done` from the previous job can never
 * be read as this one finishing. Cores waiting in the barrier or in wfi are
 * stalled in hardware and cost no simulated cycles.
 *
 * Every field is 32 bits: the clusters are rv32, and every address the host
 * hands them (main memory at 0x8000_0000, TCDM, peripherals) fits in 32 bits.
 */
#ifndef HES_MAILBOX_H
#define HES_MAILBOX_H

#include "hes_system.h"

/* Written by the control core once it has booted and initialized TCDM. The
 * host waits for it before dispatching anything, which is also what proves a
 * cluster came up at all. */
#define HES_MBOX_MAGIC 0x48455331u /* "HES1" */

/* crt0_cluster.S records a trap here without needing the C layout. */
#define HES_MBOX_TRAP_OFFSET 28

#ifndef __ASSEMBLER__

#include <stdint.h>

typedef struct {
  volatile uint32_t magic;        /* HES_MBOX_MAGIC once the cluster is up   */
  volatile uint32_t alive_mask;   /* bit i set by core i at boot             */
  volatile uint32_t seq;          /* host: bumped for every new job          */
  volatile uint32_t kernel_id;    /* host: which kernel to run               */
  volatile uint32_t nargs;        /* host: how many of args[] are used       */
  volatile uint32_t done_seq;     /* cluster: seq of the last finished job   */
  volatile uint32_t cycles;       /* cluster: cycles the last job took       */
  volatile uint32_t trap_cause;   /* cluster: mcause, if a core trapped      */
  volatile uint32_t nb_cores;     /* cluster: cores that ran the last job    */
  volatile uint32_t flags;        /* host: HES_JOB_* for this job            */
  volatile uint32_t staged;       /* cluster: 1 if it staged into TCDM       */
  volatile uint32_t error;        /* cluster: HES_ERR_* if it refused the job */
  volatile uint32_t args[HES_MAILBOX_MAX_ARGS];
  volatile uint32_t probe[16];    /* scratch for the latency probe           */
} hes_mailbox_t;

_Static_assert(sizeof(hes_mailbox_t) <= HES_MAILBOX_SIZE,
               "mailbox descriptor does not fit its reserved TCDM window");
_Static_assert(__builtin_offsetof(hes_mailbox_t, trap_cause) == HES_MBOX_TRAP_OFFSET,
               "HES_MBOX_TRAP_OFFSET disagrees with the struct layout");

static inline hes_mailbox_t *hes_mailbox_at(uint32_t tcdm_base) {
  return (hes_mailbox_t *)(uintptr_t)(tcdm_base + HES_MAILBOX_OFFSET);
}

/* Why a cluster refused a job. */
enum {
  HES_ERR_NONE = 0,
  /* The kernels on this cluster only reach cluster-local memory, and the
     operands could not be staged into it. */
  HES_ERR_NEEDS_STAGING = 1,
};

/* Kernel ids. 0 is reserved so a zeroed mailbox never looks like a job. */
enum {
  HES_K_NONE = 0,
  HES_K_PROBE = 1,          /* measure memory latencies, fill probe[]        */
  HES_K_MATMUL_FP32 = 2,    /* MatMul_fp32_fp32_fp32                         */
  HES_K_GEMM_FP32 = 3,      /* Gemm_fp32_fp32_fp32_fp32                      */
  HES_K_CONV2D_FP32 = 4,    /* Conv2d_fp32_fp32_fp32_NCHW                    */
};

#endif /* __ASSEMBLER__ */

#endif /* HES_MAILBOX_H */
