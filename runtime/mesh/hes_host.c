#include "hes_host.h"

#include "miniio.h"

typedef struct {
  const char *name;
  uint32_t tcdm;
  uint32_t periph;
  uint32_t clint_set;
  uint32_t ctrl_core;
  uint32_t requires_staging;
} engine_t;

static const engine_t engines[HES_NB_ENGINES] = {
    [HES_ENGINE_CVA6] = {"cva6", 0, 0, 0, 0, 0},
    [HES_ENGINE_SNITCH] = {"snitch", HES_SNITCH_TCDM, HES_SNITCH_PERIPH,
                           HES_SNITCH_CL_CLINT_SET, HES_SNITCH_CTRL_CORE,
                           HES_SNITCH_REQUIRES_STAGING},
    [HES_ENGINE_SPATZ] = {"spatz", HES_SPATZ_TCDM, HES_SPATZ_PERIPH,
                          HES_SPATZ_CL_CLINT_SET, HES_SPATZ_CTRL_CORE,
                          HES_SPATZ_REQUIRES_STAGING},
};

/* Job numbers are handed out by the host, one sequence per cluster, so a
 * stale done_seq from the previous job can never be mistaken for this one. */
static uint32_t next_seq[HES_NB_ENGINES];

int hes_engine_requires_staging(uint32_t engine) {
  return engine < HES_NB_ENGINES ? (int)engines[engine].requires_staging : 0;
}

const char *hes_engine_name(uint32_t engine) {
  return engine < HES_NB_ENGINES ? engines[engine].name : "?";
}

int hes_engine_ready(uint32_t engine) {
  if (engine == HES_ENGINE_CVA6 || engine >= HES_NB_ENGINES) {
    return 0;
  }
  return hes_mailbox_at(engines[engine].tcdm)->magic == HES_MBOX_MAGIC;
}

/* Wait for both clusters to finish booting. They start at the same time as the
 * host, so this normally returns immediately. */
void hes_engine_init(void) {
  for (uint32_t e = 1; e < HES_NB_ENGINES; e++) {
    for (uint32_t spins = 0; spins < 20000000u && !hes_engine_ready(e); spins++) {
    }
    next_seq[e] = hes_mailbox_at(engines[e].tcdm)->done_seq;
  }
}

static void ring_doorbell(const engine_t *e) {
  volatile uint32_t *clint =
      (volatile uint32_t *)(uintptr_t)(e->periph + e->clint_set);
  *clint = 1u << e->ctrl_core;
}

hes_result_t hes_offload(uint32_t engine, uint32_t kernel, const uint32_t *args,
                         uint32_t nargs, uint32_t flags) {
  hes_result_t res = {0, 0, 0, 0};
  if (engine == HES_ENGINE_CVA6 || engine >= HES_NB_ENGINES) {
    return res;
  }

  const engine_t *e = &engines[engine];
  hes_mailbox_t *mbox = hes_mailbox_at(e->tcdm);

  /* This engine's kernels cannot see main memory, so staging is not optional. */
  if (e->requires_staging) {
    flags |= HES_JOB_STAGE;
  }

  for (uint32_t i = 0; i < nargs && i < HES_MAILBOX_MAX_ARGS; i++) {
    mbox->args[i] = args[i];
  }
  mbox->kernel_id = kernel;
  mbox->nargs = nargs;
  mbox->flags = flags;

  /* The descriptor has to be in place before the job number that publishes
   * it; the cluster reads them in the opposite order. */
  __asm__ volatile("" ::: "memory");
  uint32_t seq = ++next_seq[engine];
  mbox->seq = seq;
  __asm__ volatile("" ::: "memory");

  ring_doorbell(e);

  for (uint32_t beat = 0; beat < HES_MAX_HEARTBEATS; beat++) {
    for (uint32_t i = 0; i < HES_HEARTBEAT_POLLS; i++) {
      if (mbox->done_seq == seq) {
        res.cycles = mbox->cycles;
        res.nb_cores = mbox->nb_cores;
        res.staged = mbox->staged;
        res.error = mbox->error;
        res.ok = mbox->trap_cause == 0 && mbox->error == 0;
        return res;
      }
    }
    /* Still running. Say so, so a hung kernel is visible instead of silent. */
    print_str("[HES-WAIT] engine=");
    print_str(e->name);
    print_str(" job=");
    print_u64(seq);
    print_str(" kernel=");
    print_u64(kernel);
    print_str(" polls=");
    print_u64((uint64_t)(beat + 1) * HES_HEARTBEAT_POLLS);
    print_str("\n");
  }

  print_str("[HES-WAIT] engine=");
  print_str(e->name);
  print_str(" job=");
  print_u64(seq);
  print_str(" GAVE UP\n");
  res.ok = 0;
  return res;
}
