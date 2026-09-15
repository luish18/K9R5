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

/* Offloads that went wrong. Non-zero means the reported numbers are not
 * trustworthy; hes_check() below is the only thing that raises it. */
static uint32_t failure_count;

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

/* Jobs posted but not yet waited for, one per engine. The mailbox holds a
 * single descriptor, so a second post before the wait would overwrite a job
 * the cluster may still be reading -- refuse that rather than corrupt it. */
static uint32_t in_flight[HES_NB_ENGINES];
/* Cycles each engine has spent on jobs, as the engine itself counted them.
 * With two clusters running at once the host's own elapsed time no longer
 * says how busy either was, so they report it and this keeps the running
 * total. */
static uint32_t busy_cycles[HES_NB_ENGINES];

uint32_t hes_engine_busy(uint32_t engine) {
  return engine < HES_NB_ENGINES ? busy_cycles[engine] : 0;
}

int hes_engine_in_flight(uint32_t engine) {
  return engine < HES_NB_ENGINES ? (int)in_flight[engine] : 0;
}

int hes_post(uint32_t engine, uint32_t kernel, const uint32_t *args,
             uint32_t nargs, uint32_t flags) {
  if (engine == HES_ENGINE_CVA6 || engine >= HES_NB_ENGINES) {
    return 1;
  }
  if (in_flight[engine]) {
    /* Not counted here: every post is matched by a hes_wait(), which returns a
     * zeroed result when nothing is in flight and so fails in hes_check()
     * exactly once. Counting it in both places would double the number the run
     * reports. */
    print_str("[HES-ERR] post to busy engine ");
    print_str(engines[engine].name);
    print_str("\n");
    return 1;
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
  mbox->seq = ++next_seq[engine];
  __asm__ volatile("" ::: "memory");

  ring_doorbell(e);
  in_flight[engine] = 1;
  return 0;
}

hes_result_t hes_wait(uint32_t engine) {
  hes_result_t res = {0, 0, 0, 0, 0};
  if (engine == HES_ENGINE_CVA6 || engine >= HES_NB_ENGINES ||
      !in_flight[engine]) {
    return res;
  }

  const engine_t *e = &engines[engine];
  hes_mailbox_t *mbox = hes_mailbox_at(e->tcdm);
  const uint32_t seq = next_seq[engine];
  const uint32_t kernel = mbox->kernel_id;
  in_flight[engine] = 0;

  for (uint32_t beat = 0; beat < HES_MAX_HEARTBEATS; beat++) {
    for (uint32_t i = 0; i < HES_HEARTBEAT_POLLS; i++) {
      if (mbox->done_seq == seq) {
        res.cycles = mbox->cycles;
        res.nb_cores = mbox->nb_cores;
        res.staged = mbox->staged;
        res.error = mbox->error;
        res.ok = mbox->trap_cause == 0 && mbox->error == 0;
        busy_cycles[engine] += res.cycles;
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

hes_result_t hes_offload(uint32_t engine, uint32_t kernel, const uint32_t *args,
                         uint32_t nargs, uint32_t flags) {
  hes_result_t res = {0, 0, 0, 0, 0};
  if (hes_post(engine, kernel, args, nargs, flags) != 0) {
    return res;
  }
  return hes_wait(engine);
}

/* --- Progress ------------------------------------------------------------ */

static uint32_t sample_idx, sample_total;

uint32_t hes_failures(void) { return failure_count; }

void hes_set_sample(uint32_t idx, uint32_t total) {
  sample_idx = idx;
  sample_total = total;
}

static inline uint32_t rdcycle_lo(void) {
  uint32_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}

uint32_t hes_node_begin(uint32_t idx, const char *op, uint32_t engine) {
  (void)idx;
  (void)op;
  (void)engine;
  return rdcycle_lo();
}

void hes_node_end(uint32_t idx, const char *op, uint32_t engine, uint32_t t0) {
  uint32_t cycles = rdcycle_lo() - t0;
  print_str("[HES-PROG] sample=");
  print_u64(sample_idx);
  print_str("/");
  print_u64(sample_total);
  print_str(" node=");
  print_u64(idx);
  print_str(" op=");
  print_str(op);
  /* The engine goes out as a number, not a name. Printing the name would
   * mean another semi-hosted string write, and ISS v2's semi-hosting has been
   * seen to drop one and repeat the previous one in its place -- harmless for
   * a literal, but it silently misattributed a node's cycles when the driver
   * had to parse the name back. The number cannot be misread, and the driver
   * maps it with the same table that generated the code. */
  print_str(" engine_id=");
  print_u64(engine);
  print_str(" cycles=");
  print_u64(cycles);
  print_str("\n");
}

/* --- Kernel wrappers ------------------------------------------------------ */

int hes_check(uint32_t engine, const char *what, hes_result_t r) {
  if (r.ok) {
    return 0;
  }
  failure_count++;
  print_str("[HES-ERR] ");
  print_str(what);
  print_str(" failed on ");
  print_str(hes_engine_name(engine));
  if (r.error == HES_ERR_NEEDS_STAGING) {
    print_str(": operands do not fit the cluster scratchpad");
  } else if (r.error == HES_ERR_NO_SCRATCH) {
    print_str(": no room in the cluster scratchpad for the kernel's working "
              "storage");
  }
  print_str("\n");
  return 1;
}


int hes_offload_matmul(uint32_t engine, const void *A, const void *B, void *Y,
                       uint32_t M, uint32_t N, uint32_t O) {
  uint32_t args[HES_MM_NARGS];
  args[HES_MM_A] = (uint32_t)(uintptr_t)A;
  args[HES_MM_B] = (uint32_t)(uintptr_t)B;
  args[HES_MM_Y] = (uint32_t)(uintptr_t)Y;
  args[HES_MM_M] = M;
  args[HES_MM_N] = N;
  args[HES_MM_O] = O;
  return hes_check(engine, "MatMul",
                hes_offload(engine, HES_K_MATMUL_FP32, args, HES_MM_NARGS,
                            HES_JOB_STAGE));
}

int hes_offload_gemm(uint32_t engine, const void *A, const void *B,
                     const void *C, void *Y, uint32_t M, uint32_t N, uint32_t O,
                     uint32_t transA, uint32_t transB) {
  uint32_t args[HES_GEMM_NARGS];
  args[HES_GEMM_A] = (uint32_t)(uintptr_t)A;
  args[HES_GEMM_B] = (uint32_t)(uintptr_t)B;
  args[HES_GEMM_C] = (uint32_t)(uintptr_t)C;
  args[HES_GEMM_Y] = (uint32_t)(uintptr_t)Y;
  args[HES_GEMM_M] = M;
  args[HES_GEMM_N] = N;
  args[HES_GEMM_O] = O;
  args[HES_GEMM_TRANSA] = transA;
  args[HES_GEMM_TRANSB] = transB;
  return hes_check(engine, "Gemm",
                hes_offload(engine, HES_K_GEMM_FP32, args, HES_GEMM_NARGS,
                            HES_JOB_STAGE));
}

int hes_offload_conv2d(uint32_t engine, const void *A, uint32_t C, uint32_t H,
                       uint32_t W, const void *weights, uint32_t F, uint32_t P,
                       uint32_t Q, uint32_t SP, uint32_t SQ, const void *bias,
                       uint32_t has_bias, void *Y) {
  uint32_t args[HES_CONV_NARGS];
  args[HES_CONV_A] = (uint32_t)(uintptr_t)A;
  args[HES_CONV_C] = C;
  args[HES_CONV_H] = H;
  args[HES_CONV_W] = W;
  args[HES_CONV_B] = (uint32_t)(uintptr_t)weights;
  args[HES_CONV_F] = F;
  args[HES_CONV_P] = P;
  args[HES_CONV_Q] = Q;
  args[HES_CONV_SP] = SP;
  args[HES_CONV_SQ] = SQ;
  args[HES_CONV_BIAS] = (uint32_t)(uintptr_t)bias;
  args[HES_CONV_HAS_BIAS] = has_bias;
  args[HES_CONV_Y] = (uint32_t)(uintptr_t)Y;
  return hes_check(engine, "Conv2d",
                hes_offload(engine, HES_K_CONV2D_FP32, args, HES_CONV_NARGS,
                            HES_JOB_STAGE));
}

/* --- The MFCC front-end --------------------------------------------------- */

static void mfcc_args(const hes_mfcc_job_t *job, uint32_t *args) {
  args[HES_MFCC_AUDIO] = (uint32_t)(uintptr_t)job->audio;
  args[HES_MFCC_OUT] = (uint32_t)(uintptr_t)job->out;
  args[HES_MFCC_NB_FRAMES] = job->nb_frames;
  args[HES_MFCC_FRAME_LEN] = job->frame_len;
  args[HES_MFCC_HOP] = job->hop;
  args[HES_MFCC_FFT_LEN] = job->fft_len;
  args[HES_MFCC_WINDOW] = (uint32_t)(uintptr_t)job->window;
  args[HES_MFCC_TWIDDLES] = (uint32_t)(uintptr_t)job->twiddles;
  args[HES_MFCC_MEL_COEFF] = (uint32_t)(uintptr_t)job->mel_coeff;
  args[HES_MFCC_MEL_START] = (uint32_t)(uintptr_t)job->mel_start;
  args[HES_MFCC_MEL_LEN] = (uint32_t)(uintptr_t)job->mel_len;
  args[HES_MFCC_NB_MEL] = job->nb_mel;
  args[HES_MFCC_DCT] = (uint32_t)(uintptr_t)job->dct;
  args[HES_MFCC_NB_CEP] = job->nb_cep;
  args[HES_MFCC_MEL_COEFFS] = job->mel_coeffs;
}

int hes_offload_mfcc(uint32_t engine, const hes_mfcc_job_t *job) {
  uint32_t args[HES_MFCC_NARGS];
  mfcc_args(job, args);
  return hes_check(engine, "Mfcc",
                   hes_offload(engine, HES_K_MFCC_FP32, args, HES_MFCC_NARGS,
                               HES_JOB_STAGE));
}

int hes_post_mfcc(uint32_t engine, const hes_mfcc_job_t *job) {
  uint32_t args[HES_MFCC_NARGS];
  mfcc_args(job, args);
  return hes_post(engine, HES_K_MFCC_FP32, args, HES_MFCC_NARGS,
                  HES_JOB_STAGE);
}
