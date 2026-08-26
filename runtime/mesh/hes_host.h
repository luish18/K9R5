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

/* Polls between heartbeats, and the cap after which a job is declared hung. */
#define HES_HEARTBEAT_POLLS (1u << 16)
#define HES_MAX_HEARTBEATS 200

#endif /* HES_HOST_H */
