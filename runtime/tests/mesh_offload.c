/* M2 gate: can the host actually hand work to both clusters and get the right
 * answer back?
 *
 * For each kernel the host computes a reference with the scalar Deeploy
 * kernel on its own core, then offloads the same problem to the Snitch
 * cluster and to the Snitch/Spatz pair, twice each -- once asking for the
 * operands to be left in main memory, once asking the cluster DMA to stage
 * them into TCDM -- and compares every output element.
 *
 * The Spatz pair reports "staged" both times: its vector load/store unit is
 * wired to the cluster TCDM only, so the runtime stages its operands whatever
 * the caller asked for. Running it against main memory would not fault, it
 * would quietly compute the wrong numbers.
 *
 * MatMul and Conv2d must come out bit-identical: the Xssr/Xfrep kernels keep
 * the generic reduction order. GEMM is allowed a last-bit difference because
 * it starts its accumulator at C instead of adding C at the end.
 *
 * The cycle counts are printed alongside, which is what shows the 8-core
 * split and the DMA staging doing anything.
 */

#include "hes_host.h"
#include "hes_job.h"
#include "hes_system.h"
#include "miniio.h"

#include "DeeployBasicMath.h"

#include <stdint.h>

/* Problem sizes: big enough that eight cores have something to divide, small
 * enough to stage into a 128 KiB TCDM. */
#define M 32
#define N 32
#define O 32

/* Conv: 4 input channels of 16x16, 8 filters of 4x3x3, stride 1. */
#define C_IN 4
#define H_IN 16
#define W_IN 16
#define F_OUT 8
#define KP 3
#define KQ 3
#define H_OUT (H_IN - KP + 1)
#define W_OUT (W_IN - KQ + 1)

/* Operands live in main memory, where the clusters' DMA can reach them. */
#define ALIGNED __attribute__((aligned(64)))
static ALIGNED float32_t A[M * N], B[N * O], Cm[M * O];
static ALIGNED float32_t Y_ref[M * O], Y_dut[M * O];
static ALIGNED float32_t conv_in[C_IN * H_IN * W_IN];
static ALIGNED float32_t conv_w[F_OUT * C_IN * KP * KQ];
static ALIGNED float32_t conv_b[F_OUT];
static ALIGNED float32_t conv_ref[F_OUT * H_OUT * W_OUT];
static ALIGNED float32_t conv_dut[F_OUT * H_OUT * W_OUT];

static int failures;



/* A cheap deterministic spread of values, so a wrong slice shows up as a
 * mismatch rather than as zeros that happen to agree. */
static float32_t sample(uint32_t i) {
  return (float32_t)((int32_t)((i * 1103515245u + 12345u) >> 20) % 97 - 48) *
         0.03125f;
}

static void fill(float32_t *p, uint32_t n, uint32_t salt) {
  for (uint32_t i = 0; i < n; i++) {
    p[i] = sample(i + salt);
  }
}

static void zero(float32_t *p, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    p[i] = 0.0f;
  }
}

static void verdict(const char *what, uint32_t engine, hes_result_t r,
                    uint32_t mismatches, uint32_t worst_ulp) {
  int ok = r.ok && mismatches == 0;
  print_str(ok ? "  ok   " : "  FAIL ");
  print_str(what);
  print_str(" on ");
  print_str(hes_engine_name(engine));
  print_str(r.staged ? " staged  " : " in dram ");
  print_str(" cycles=");
  print_u64(r.cycles);
  print_str(" cores=");
  print_u64(r.nb_cores);
  if (mismatches) {
    print_str(" MISMATCHES=");
    print_u64(mismatches);
    print_str(" worst_ulp=");
    print_u64(worst_ulp);
  }
  if (r.error == HES_ERR_NEEDS_STAGING) {
    print_str(" (refused: this engine only reaches cluster-local memory)");
  } else if (!r.ok) {
    print_str(" (cluster reported failure)");
  }
  print_str("\n");
  if (!ok) {
    failures++;
  }
}


/* Compare bitwise, reporting how far apart the two results are in ulps so a
 * near-miss is distinguishable from a wrong answer. */
static uint32_t compare(const float32_t *ref, const float32_t *dut, uint32_t n,
                        uint32_t allowed_ulp, uint32_t *worst) {
  uint32_t bad = 0;
  *worst = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t a, b;
    __builtin_memcpy(&a, &ref[i], 4);
    __builtin_memcpy(&b, &dut[i], 4);
    uint32_t d = a > b ? a - b : b - a;
    if (d > *worst) {
      *worst = d;
    }
    if (d > allowed_ulp) {
      bad++;
    }
  }
  return bad;
}


static void run_matmul(uint32_t engine, uint32_t flags) {
  zero(Y_dut, M * O);
  uint32_t args[HES_MM_NARGS];
  args[HES_MM_A] = (uint32_t)(uintptr_t)A;
  args[HES_MM_B] = (uint32_t)(uintptr_t)B;
  args[HES_MM_Y] = (uint32_t)(uintptr_t)Y_dut;
  args[HES_MM_M] = M;
  args[HES_MM_N] = N;
  args[HES_MM_O] = O;

  hes_result_t r = hes_offload(engine, HES_K_MATMUL_FP32, args, HES_MM_NARGS, flags);
  uint32_t worst;
  /* Same reduction order as the scalar kernel: must be exact. */
  uint32_t bad = compare(Y_ref, Y_dut, M * O, 0, &worst);
  verdict("MatMul 32x32x32", engine, r, bad, worst);
}

static void run_gemm(uint32_t engine, uint32_t flags) {
  zero(Y_dut, M * O);
  uint32_t args[HES_GEMM_NARGS];
  args[HES_GEMM_A] = (uint32_t)(uintptr_t)A;
  args[HES_GEMM_B] = (uint32_t)(uintptr_t)B;
  args[HES_GEMM_C] = (uint32_t)(uintptr_t)Cm;
  args[HES_GEMM_Y] = (uint32_t)(uintptr_t)Y_dut;
  args[HES_GEMM_M] = M;
  args[HES_GEMM_N] = N;
  args[HES_GEMM_O] = O;
  args[HES_GEMM_TRANSA] = 0;
  args[HES_GEMM_TRANSB] = 0;

  hes_result_t r = hes_offload(engine, HES_K_GEMM_FP32, args, HES_GEMM_NARGS, flags);
  uint32_t worst;
  /* Accumulator starts at C rather than adding it at the end, so the last
   * bit may differ. */
  uint32_t bad = compare(Y_ref, Y_dut, M * O, 2, &worst);
  verdict("GEMM 32x32x32", engine, r, bad, worst);
}


/* Compare what the cluster staged into TCDM against the originals. */

static void run_conv(uint32_t engine, uint32_t flags) {
  zero(conv_dut, F_OUT * H_OUT * W_OUT);
  uint32_t args[HES_CONV_NARGS];
  args[HES_CONV_A] = (uint32_t)(uintptr_t)conv_in;
  args[HES_CONV_C] = C_IN;
  args[HES_CONV_H] = H_IN;
  args[HES_CONV_W] = W_IN;
  args[HES_CONV_B] = (uint32_t)(uintptr_t)conv_w;
  args[HES_CONV_F] = F_OUT;
  args[HES_CONV_P] = KP;
  args[HES_CONV_Q] = KQ;
  args[HES_CONV_SP] = 1;
  args[HES_CONV_SQ] = 1;
  args[HES_CONV_BIAS] = (uint32_t)(uintptr_t)conv_b;
  args[HES_CONV_HAS_BIAS] = 1;
  args[HES_CONV_Y] = (uint32_t)(uintptr_t)conv_dut;

  hes_result_t r = hes_offload(engine, HES_K_CONV2D_FP32, args, HES_CONV_NARGS, flags);
  uint32_t worst;
  uint32_t bad = compare(conv_ref, conv_dut, F_OUT * H_OUT * W_OUT, 0, &worst);
  verdict("Conv2d 4x16x16 -> 8 filters", engine, r, bad, worst);
}

int main(void) {
  print_str("[HES-OFFLOAD] host -> cluster dispatch\n");

  hes_engine_init();
  for (uint32_t e = HES_ENGINE_SNITCH; e < HES_NB_ENGINES; e++) {
    if (!hes_engine_ready(e)) {
      print_str("  FAIL cluster ");
      print_str(hes_engine_name(e));
      print_str(" never signed in\n");
      failures++;
    }
  }

  fill(A, M * N, 1);
  fill(B, N * O, 7919);
  fill(Cm, M * O, 104729);
  fill(conv_in, C_IN * H_IN * W_IN, 15485863);
  fill(conv_w, F_OUT * C_IN * KP * KQ, 32452843);
  fill(conv_b, F_OUT, 49979687);

  /* References, computed by the host with the scalar Generic kernels. */
  uint32_t t0 = 0;
  __asm__ volatile("csrr %0, mcycle" : "=r"(t0));
  MatMul_fp32_fp32_fp32(A, B, Y_ref, M, N, O);
  uint32_t t1 = 0;
  __asm__ volatile("csrr %0, mcycle" : "=r"(t1));
  print_str("  --   MatMul reference on cva6: cycles=");
  print_u64(t1 - t0);
  print_str("\n");

  for (uint32_t stage = 0; stage < 2; stage++) {
    uint32_t flags = stage ? HES_JOB_STAGE : 0;
    run_matmul(HES_ENGINE_SNITCH, flags);
    run_matmul(HES_ENGINE_SPATZ, flags);
  }

  Gemm_fp32_fp32_fp32_fp32(A, B, Cm, Y_ref, M, N, O, 0, 0);
  for (uint32_t stage = 0; stage < 2; stage++) {
    uint32_t flags = stage ? HES_JOB_STAGE : 0;
    run_gemm(HES_ENGINE_SNITCH, flags);
    run_gemm(HES_ENGINE_SPATZ, flags);
  }

  Conv2d_fp32_fp32_fp32_NCHW(conv_in, C_IN, H_IN, W_IN, conv_w, F_OUT, KP, KQ, 1,
                             1, conv_b, true, conv_ref);
  for (uint32_t stage = 0; stage < 2; stage++) {
    uint32_t flags = stage ? HES_JOB_STAGE : 0;
    run_conv(HES_ENGINE_SNITCH, flags);
    run_conv(HES_ENGINE_SPATZ, flags);
  }

  for (uint32_t e = HES_ENGINE_SNITCH; e < HES_NB_ENGINES; e++) {
    uint32_t tcdm = (e == HES_ENGINE_SNITCH) ? HES_SNITCH_TCDM : HES_SPATZ_TCDM;
    uint32_t cause = hes_mailbox_at(tcdm)->trap_cause;
    if (cause) {
      print_str("  FAIL cluster ");
      print_str(hes_engine_name(e));
      print_str(" trapped, mcause=");
      print_u64(cause);
      print_str("\n");
      failures++;
    }
  }

  print_str("\n[HES-OFFLOAD] failures=");
  print_u64((uint64_t)failures);
  print_str("\n");
  return failures == 0 ? 0 : 1;
}
