/* Handing work from the CVA6 orchestrator to a cluster.
 *
 * A cluster is addressed by engine id (HES_ENGINE_SNITCH / HES_ENGINE_SPATZ).
 * hes_offload() writes the job descriptor into that cluster's mailbox, rings
 * its doorbell, and waits for it to finish; the cluster's own view of the
 * protocol is in cluster_main.c.
 *
 * While waiting, the host prints a heartbeat every so often. A cluster kernel
 * that hangs therefore still produces output naming the engine and the job it
 * hung on, instead of a silent simulation that has to be guessed at.
 */
#ifndef HES_HOST_H
#define HES_HOST_H

#include "hes_job.h"
#include "hes_mailbox.h"
#include "hes_system.h"

#include <stdint.h>

/* Result of one offload. */
typedef struct {
  uint32_t cycles;   /* cycles the cluster spent on the job              */
  uint32_t nb_cores; /* compute cores it spread the job over             */
  uint32_t staged;   /* 1 if it staged operands into TCDM first          */
  uint32_t ok;       /* 0 if the cluster trapped, refused or never answered */
  uint32_t error;    /* HES_ERR_* if the cluster refused the job            */
} hes_result_t;

void hes_engine_init(void);
int hes_engine_ready(uint32_t engine);
/* 1 if this engine's kernels only reach cluster-local memory, so every job
   for it has to be staged. hes_offload() applies this itself. */
int hes_engine_requires_staging(uint32_t engine);
const char *hes_engine_name(uint32_t engine);

/* Run `kernel` on `engine` with `nargs` arguments. Blocks until it finishes. */
hes_result_t hes_offload(uint32_t engine, uint32_t kernel,
                         const uint32_t *args, uint32_t nargs, uint32_t flags);

/* --- What the generated network calls -------------------------------------
 *
 * One wrapper per kernel the clusters implement. The argument order is the
 * Deeploy Generic kernel signature, so a node that Deeploy would have called
 * directly becomes the same call with an engine in front of it.
 *
 * Each returns 0 on success. A failure is also counted in hes_failures() and
 * reported on the console, so a run that silently produced wrong numbers is
 * not possible.
 */

int hes_offload_matmul(uint32_t engine, const void *A, const void *B, void *Y,
                       uint32_t M, uint32_t N, uint32_t O);

int hes_offload_gemm(uint32_t engine, const void *A, const void *B,
                     const void *C, void *Y, uint32_t M, uint32_t N, uint32_t O,
                     uint32_t transA, uint32_t transB);

int hes_offload_conv2d(uint32_t engine, const void *A, uint32_t C, uint32_t H,
                       uint32_t W, const void *weights, uint32_t F, uint32_t P,
                       uint32_t Q, uint32_t SP, uint32_t SQ, const void *bias,
                       uint32_t has_bias, void *Y);

/* Offloads that failed so far. Non-zero means the reported numbers are not
 * trustworthy. */
uint32_t hes_failures(void);

/* --- Progress ------------------------------------------------------------
 *
 * The generated network brackets every node with these, so a run reports what
 * it is doing while it runs rather than only at the end. hes_node_end prints
 *
 *   [HES-PROG] node=<i> op=<Op> engine=<name> cycles=<n>
 *
 * which is what pipeline/run_hetero.py turns into a live progress line and
 * what a stall watchdog keys off.
 */
uint32_t hes_node_begin(uint32_t idx, const char *op, uint32_t engine);
void hes_node_end(uint32_t idx, const char *op, uint32_t engine, uint32_t t0);

/* Counts inferences, so progress can be reported as image i of N. */
void hes_set_sample(uint32_t idx, uint32_t total);

/* Polls between heartbeats, and the cap after which a job is declared hung. */
#define HES_HEARTBEAT_POLLS (1u << 16)
#define HES_MAX_HEARTBEATS 200

#endif /* HES_HOST_H */
