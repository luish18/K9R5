/* The job loop every core of a cluster runs.
 *
 * The DMA core owns the conversation with the host: it sleeps in wfi until the
 * host rings the cluster-local doorbell, stages the operands into TCDM if they
 * fit, and joins the barrier. Joining the barrier is what releases the compute
 * cores -- the cluster's barrier register only fires once every core has
 * requested it, so the compute cores parked in it are woken by the DMA core
 * arriving, with no interrupt and no polling.
 *
 * Both waits are hardware stalls in the model: a core in wfi or in the barrier
 * consumes no simulated cycles and costs no wall-clock time, which is what
 * makes eleven cores in one simulation affordable.
 *
 *   DMA core                          compute cores
 *   ------------------------------------------------------------------
 *   wfi                               barrier   (parked, free)
 *   wake on doorbell, clear it
 *   read descriptor, stage inputs
 *   barrier  ---------------------->  released
 *   (idle)                            run own slice
 *   barrier  <----------------------  barrier
 *   copy results out, publish done
 *   wfi                               barrier
 */

#include "hes_cluster.h"
#include "hes_dma.h"
#include "hes_job.h"
#include "hes_mailbox.h"

#include <stdint.h>

#include "DeeployBasicMath.h"

/* TCDM the linker left between .bss and the per-core stacks. */
extern char __heap_start[];
extern char __heap_end[];

/* The arguments the kernels actually run with. Different from the mailbox
 * copy when the DMA core has staged operands into TCDM and rewritten the
 * pointers. Lives in TCDM (.bss), so every core reads it at full speed. */
static volatile uint32_t active_args[HES_MAILBOX_MAX_ARGS];
static volatile uint32_t active_kernel;

/* The pointer arguments of a job, as parallel arrays: which argument slot,
 * whether the kernel writes it, and how many bytes it covers. */
#define MAX_OPERANDS 4

typedef struct {
  uint32_t n;
  uint32_t arg[MAX_OPERANDS];
  uint32_t is_output[MAX_OPERANDS];
  uint32_t bytes[MAX_OPERANDS];
  uint32_t local[MAX_OPERANDS];
} operands_t;

static void op_add(operands_t *o, uint32_t arg, uint32_t is_output,
                   uint32_t bytes) {
  uint32_t i = o->n++;
  o->arg[i] = arg;
  o->is_output[i] = is_output;
  o->bytes[i] = bytes;
}

static inline uint32_t rdcycle32(void) {
  uint32_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}

/* Split `n` items across the compute cores. Core `idx` takes [*first, *last). */
static void slice(uint32_t n, uint32_t idx, uint32_t nb, uint32_t *first,
                  uint32_t *last) {
  uint32_t base = n / nb;
  uint32_t rem = n % nb;
  /* The first `rem` cores take one extra item, so the split stays even. */
  *first = idx * base + (idx < rem ? idx : rem);
  *last = *first + base + (idx < rem ? 1 : 0);
}

/*
 * Which buffers a job reads and writes, so the DMA core can stage them.
 * Returns the operand count, or 0 for a kernel that is not staged.
 */
static void plan_operands(uint32_t kernel, volatile uint32_t *a, operands_t *o) {
  const uint32_t f = sizeof(float32_t);
  o->n = 0;
  switch (kernel) {
  case HES_K_MATMUL_FP32: {
    uint32_t M = a[HES_MM_M], N = a[HES_MM_N], O = a[HES_MM_O];
    op_add(o, HES_MM_A, 0, M * N * f);
    op_add(o, HES_MM_B, 0, N * O * f);
    op_add(o, HES_MM_Y, 1, M * O * f);
    break;
  }
  case HES_K_GEMM_FP32: {
    uint32_t M = a[HES_GEMM_M], N = a[HES_GEMM_N], O = a[HES_GEMM_O];
    op_add(o, HES_GEMM_A, 0, M * N * f);
    op_add(o, HES_GEMM_B, 0, N * O * f);
    op_add(o, HES_GEMM_C, 0, M * O * f);
    op_add(o, HES_GEMM_Y, 1, M * O * f);
    break;
  }
  case HES_K_CONV2D_FP32: {
    uint32_t C = a[HES_CONV_C], H = a[HES_CONV_H], W = a[HES_CONV_W];
    uint32_t F = a[HES_CONV_F], P = a[HES_CONV_P], Q = a[HES_CONV_Q];
    uint32_t H_out = (H - P) / a[HES_CONV_SP] + 1;
    uint32_t W_out = (W - Q) / a[HES_CONV_SQ] + 1;
    op_add(o, HES_CONV_A, 0, C * H * W * f);
    op_add(o, HES_CONV_B, 0, F * C * P * Q * f);
    op_add(o, HES_CONV_Y, 1, F * H_out * W_out * f);
    if (a[HES_CONV_HAS_BIAS]) {
      op_add(o, HES_CONV_BIAS, 0, F * f);
    }
    break;
  }
  default:
    break;
  }
}

/* Copy every operand into TCDM scratch and point the active arguments at the
 * copies. Returns 0 and leaves the arguments alone if they do not all fit, in
 * which case the kernel runs against main memory instead. Only the DMA core
 * may call this: it alone has the offload port to the cluster iDMA. */
static uint32_t stage_in(uint32_t kernel, volatile uint32_t *args,
                         operands_t *o) {
  plan_operands(kernel, args, o);
  if (o->n == 0) {
    return 0;
  }

  uint32_t cursor = (uint32_t)(uintptr_t)__heap_start;
  uint32_t limit = (uint32_t)(uintptr_t)__heap_end;

  for (uint32_t i = 0; i < o->n; i++) {
    cursor = (cursor + 63u) & ~63u;
    if (cursor + o->bytes[i] > limit) {
      return 0; /* does not fit: run against main memory */
    }
    o->local[i] = cursor;
    cursor += o->bytes[i];
  }

  for (uint32_t i = 0; i < o->n; i++) {
    if (!o->is_output[i]) {
      hes_dma_copy(o->local[i], args[o->arg[i]], o->bytes[i]);
    }
    active_args[o->arg[i]] = o->local[i];
  }
  hes_dma_wait();
  return 1;
}

static void stage_out(volatile uint32_t *args, operands_t *o) {
  for (uint32_t i = 0; i < o->n; i++) {
    if (o->is_output[i]) {
      hes_dma_copy(args[o->arg[i]], o->local[i], o->bytes[i]);
    }
  }
  hes_dma_wait();
}

/*
 * Run this core's share of the job. Each kernel is sliced so that a core's
 * share is a pointer offset plus a smaller extent -- no kernel is modified and
 * no results are combined afterwards, because the slices write disjoint output.
 */
static void run_slice(uint32_t kernel, volatile uint32_t *a, uint32_t idx,
                      uint32_t nb) {
  switch (kernel) {
  case HES_K_MATMUL_FP32: {
    uint32_t M = a[HES_MM_M], N = a[HES_MM_N], O = a[HES_MM_O];
    uint32_t first, last;
    /* Rows of A and of Y: disjoint per core. */
    slice(M, idx, nb, &first, &last);
    if (first >= last) {
      return;
    }
    MatMul_fp32_fp32_fp32(
        (const float32_t *)(uintptr_t)(a[HES_MM_A] + first * N * sizeof(float32_t)),
        (const float32_t *)(uintptr_t)a[HES_MM_B],
        (float32_t *)(uintptr_t)(a[HES_MM_Y] + first * O * sizeof(float32_t)),
        last - first, N, O);
    break;
  }
  case HES_K_GEMM_FP32: {
    uint32_t M = a[HES_GEMM_M], N = a[HES_GEMM_N], O = a[HES_GEMM_O];
    uint32_t transA = a[HES_GEMM_TRANSA];
    uint32_t first, last;
    /* With transA the A operand is stored [N][M], so a row slice is not a
     * contiguous offset; leave those to one core rather than get it subtly
     * wrong. */
    if (transA) {
      first = 0;
      last = (idx == 0) ? M : 0;
    } else {
      slice(M, idx, nb, &first, &last);
    }
    if (first >= last) {
      return;
    }
    Gemm_fp32_fp32_fp32_fp32(
        (const float32_t *)(uintptr_t)(a[HES_GEMM_A] + first * N * sizeof(float32_t)),
        (const float32_t *)(uintptr_t)a[HES_GEMM_B],
        (const float32_t *)(uintptr_t)(a[HES_GEMM_C] + first * O * sizeof(float32_t)),
        (float32_t *)(uintptr_t)(a[HES_GEMM_Y] + first * O * sizeof(float32_t)),
        last - first, N, O, (int32_t)transA, (int32_t)a[HES_GEMM_TRANSB]);
    break;
  }
  case HES_K_CONV2D_FP32: {
    uint32_t C = a[HES_CONV_C], H = a[HES_CONV_H], W = a[HES_CONV_W];
    uint32_t F = a[HES_CONV_F], P = a[HES_CONV_P], Q = a[HES_CONV_Q];
    uint32_t H_out = (H - P) / a[HES_CONV_SP] + 1;
    uint32_t W_out = (W - Q) / a[HES_CONV_SQ] + 1;
    const uint32_t f = sizeof(float32_t);
    uint32_t first, last;
    /* Filters: the output is [F][H_out][W_out] and the weights [F][C][P][Q],
     * so slicing on F is a contiguous offset into both. */
    slice(F, idx, nb, &first, &last);
    if (first >= last) {
      return;
    }
    Conv2d_fp32_fp32_fp32_NCHW(
        (const float32_t *)(uintptr_t)a[HES_CONV_A], C, H, W,
        (const float32_t *)(uintptr_t)(a[HES_CONV_B] + first * C * P * Q * f),
        last - first, P, Q, a[HES_CONV_SP], a[HES_CONV_SQ],
        (const float32_t *)(uintptr_t)(a[HES_CONV_BIAS] + first * f),
        (bool)a[HES_CONV_HAS_BIAS],
        (float32_t *)(uintptr_t)(a[HES_CONV_Y] + first * H_out * W_out * f));
    break;
  }
  default:
    break;
  }
}

int main(void) {
  hes_mailbox_t *mbox = hes_mailbox_at(HES_MY_TCDM);
  const uint32_t core = hes_core_idx();
  const uint32_t nb_compute = HES_MY_NB_COMPUTE;

  /* Sign in, so the host can tell the cluster came up and with how many
   * cores. The barrier around it keeps the OR race-free. */
  if (hes_is_ctrl_core()) {
    mbox->alive_mask = 0;
    mbox->trap_cause = 0;
    mbox->done_seq = 0;
    mbox->seq = 0;
  }
  hes_barrier();
  for (uint32_t i = 0; i < HES_MY_NB_CORE; i++) {
    if (i == core) {
      mbox->alive_mask |= (1u << core);
    }
    hes_barrier();
  }
  if (hes_is_ctrl_core()) {
    mbox->nb_cores = HES_MY_NB_CORE;
    __asm__ volatile("" ::: "memory");
    mbox->magic = HES_MBOX_MAGIC;
  }
  hes_barrier();

  uint32_t served = 0;
  static operands_t ops;
  uint32_t staged = 0;
  uint32_t t0 = 0;

  for (;;) {
    if (hes_is_ctrl_core()) {
      /* Sleep until the host rings the doorbell. wfi returns on any
       * interrupt, so the condition is re-checked rather than assumed. */
      while (mbox->seq == served) {
        __asm__ volatile("wfi");
        hes_clint_clear(1u << HES_MY_CTRL_CORE);
      }
      served = mbox->seq;

      t0 = rdcycle32();
      active_kernel = mbox->kernel_id;
      for (uint32_t i = 0; i < HES_MAILBOX_MAX_ARGS; i++) {
        active_args[i] = mbox->args[i];
      }
      staged = 0;
      ops.n = 0;
      if (mbox->flags & HES_JOB_STAGE) {
        staged = stage_in(active_kernel, mbox->args, &ops);
        if (!staged) {
          /* Did not fit: put the main-memory pointers back. */
          for (uint32_t i = 0; i < HES_MAILBOX_MAX_ARGS; i++) {
            active_args[i] = mbox->args[i];
          }
        }
      }
      /* On a cluster whose kernels can only reach cluster-local memory,
       * running unstaged would compute silently wrong numbers. Refuse the
       * job instead, and say so. */
      if (HES_MY_REQUIRES_STAGING && !staged) {
        mbox->error = HES_ERR_NEEDS_STAGING;
        active_kernel = HES_K_NONE;
      } else {
        mbox->error = 0;
      }
    }

    /* The DMA core arriving here is what releases the compute cores. */
    hes_barrier();

    if (hes_is_compute_core()) {
      run_slice(active_kernel, active_args, hes_compute_idx(), nb_compute);
    }

    hes_barrier();

    if (hes_is_ctrl_core()) {
      if (staged) {
        stage_out(mbox->args, &ops);
      }
      mbox->cycles = rdcycle32() - t0;
      mbox->staged = staged;
      mbox->nb_cores = nb_compute;
      __asm__ volatile("" ::: "memory");
      mbox->done_seq = served;
    }
  }
  return 0;
}
