/* Per-design calibration of the node-to-engine cost model.
 *
 * pipeline/hetero_platform/mapper.py prices a node as
 *
 *     cost = OFFLOAD_FIXED[e] + OFFLOAD_PER_BYTE[e] * bytes + macs / RATES[e][op]
 *
 * and its tables are measurements of *one* SoC -- the 9-core clusters, 4-lane
 * Spatz, 4096-bit Ara this repository committed. A design-space sweep changes
 * exactly the things those numbers depend on, so a swept design mapped with
 * them would be judged by another machine's arithmetic: a 16-lane Spatz would
 * keep getting work its own rates say belongs elsewhere.
 *
 * This program re-measures the whole table for whatever SoC it is built and
 * run against, in one build and one simulation. Driving the same calibration
 * through run_hetero.py --pin would cost nine codegen+build+simulate cycles
 * per design point for the same information.
 *
 * It emits one machine-readable line per measurement:
 *
 *   [HES-CAL] engine=<name> op=<Op> macs=<n> bytes=<n> staged=<0|1>
 *             cycles=<n> host_cycles=<n> ok=<0|1>
 *
 * `cycles` is what the engine spent on the kernel (for a cluster, what it
 * counted itself; for the host, the whole call). `host_cycles` is the round
 * trip the host saw, so the difference is the offload overhead. Nothing is
 * fitted here -- sweep/calibrate.py does the arithmetic, because a rate is a
 * ratio of two numbers this program already prints and a regression is easier
 * to check in Python than in bare-metal C.
 *
 * Correctness is still checked against the host's own scalar kernels. A design
 * whose kernels compute the wrong answer must fail calibration rather than be
 * credited with whatever rate the wrong answer was produced at.
 */

#include "hes_host.h"
#include "hes_job.h"
#include "hes_system.h"
#include "miniio.h"

#include "DeeployBasicMath.h"

#include <stdint.h>

/* The shapes mapper.py documents its rates at, so a recalibrated table is
 * directly comparable with the committed one. */
#define M 32
#define N 32
#define O 32

#define C_IN 4
#define H_IN 16
#define W_IN 16
#define F_OUT 8
#define KP 3
#define KQ 3
#define H_OUT (H_IN - KP + 1)
#define W_OUT (W_IN - KQ + 1)

/* A deliberately tiny job: almost no arithmetic, so its round trip is very
 * nearly the fixed cost of offloading anything at all. That is the intercept
 * OFFLOAD_FIXED wants, and without it the per-byte slope would absorb it. */
#define MS 4

#define ALIGNED __attribute__((aligned(64)))
static ALIGNED float32_t A[M * N], B[N * O], Cm[M * O];
static ALIGNED float32_t Y_ref[M * O], Y_dut[M * O];
static ALIGNED float32_t conv_in[C_IN * H_IN * W_IN];
static ALIGNED float32_t conv_w[F_OUT * C_IN * KP * KQ];
static ALIGNED float32_t conv_b[F_OUT];
static ALIGNED float32_t conv_ref[F_OUT * H_OUT * W_OUT];
static ALIGNED float32_t conv_dut[F_OUT * H_OUT * W_OUT];

static int failures;

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

static uint32_t compare(const float32_t *ref, const float32_t *dut, uint32_t n,
                        uint32_t allowed_ulp) {
  uint32_t bad = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t a, b;
    __builtin_memcpy(&a, &ref[i], 4);
    __builtin_memcpy(&b, &dut[i], 4);
    uint32_t d = a > b ? a - b : b - a;
    if (d > allowed_ulp) {
      bad++;
    }
  }
  return bad;
}

static uint32_t cal_pass;

static void emit(const char *engine, const char *op, uint32_t macs,
                 uint32_t bytes, int staged, uint32_t cycles,
                 uint32_t host_cycles, int ok) {
  print_str("[HES-CAL] engine=");
  print_str(engine);
  print_str(" op=");
  print_str(op);
  print_str(" macs=");
  print_u64(macs);
  print_str(" bytes=");
  print_u64(bytes);
  print_str(" staged=");
  print_u64((uint32_t)staged);
  print_str(" cycles=");
  print_u64(cycles);
  print_str(" host_cycles=");
  print_u64(host_cycles);
  print_str(" pass=");
  print_u64(cal_pass);
  print_str(" ok=");
  print_u64((uint32_t)ok);
  print_str("\n");
  if (!ok) {
    failures++;
  }
}

static inline uint32_t now(void) {
  uint32_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}

/* --- the three kernels, offloaded ---------------------------------------- */

static void cal_matmul(uint32_t engine, uint32_t m, uint32_t n, uint32_t o,
                       const char *op) {
  zero(Y_dut, m * o);
  uint32_t args[HES_MM_NARGS];
  args[HES_MM_A] = (uint32_t)(uintptr_t)A;
  args[HES_MM_B] = (uint32_t)(uintptr_t)B;
  args[HES_MM_Y] = (uint32_t)(uintptr_t)Y_dut;
  args[HES_MM_M] = m;
  args[HES_MM_N] = n;
  args[HES_MM_O] = o;

  uint32_t t0 = now();
  hes_result_t r = hes_offload(engine, HES_K_MATMUL_FP32, args, HES_MM_NARGS,
                               HES_JOB_STAGE);
  uint32_t t1 = now();

  MatMul_fp32_fp32_fp32(A, B, Y_ref, m, n, o);
  uint32_t bad = compare(Y_ref, Y_dut, m * o, 0);
  emit(hes_engine_name(engine), op, m * n * o,
       4u * (m * n + n * o + m * o), r.staged, r.cycles, t1 - t0,
       r.ok && bad == 0);
}

static void cal_gemm(uint32_t engine) {
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

  uint32_t t0 = now();
  hes_result_t r = hes_offload(engine, HES_K_GEMM_FP32, args, HES_GEMM_NARGS,
                               HES_JOB_STAGE);
  uint32_t t1 = now();

  uint32_t bad = compare(Y_ref, Y_dut, M * O, 2);
  emit(hes_engine_name(engine), "Gemm", M * N * O,
       4u * (M * N + N * O + 2 * M * O), r.staged, r.cycles, t1 - t0,
       r.ok && bad == 0);
}

static void cal_conv(uint32_t engine) {
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

  uint32_t t0 = now();
  hes_result_t r = hes_offload(engine, HES_K_CONV2D_FP32, args, HES_CONV_NARGS,
                               HES_JOB_STAGE);
  uint32_t t1 = now();

  uint32_t bad = compare(conv_ref, conv_dut, F_OUT * H_OUT * W_OUT, 0);
  emit(hes_engine_name(engine), "Conv",
       (uint32_t)F_OUT * H_OUT * W_OUT * C_IN * KP * KQ,
       4u * (C_IN * H_IN * W_IN + F_OUT * C_IN * KP * KQ + F_OUT +
             F_OUT * H_OUT * W_OUT),
       r.staged, r.cycles, t1 - t0, r.ok && bad == 0);
}

/* --- the host, running the same kernels in place -------------------------- */

static void cal_host(void) {
  const char *host = hes_engine_name(HES_ENGINE_CVA6);

  uint32_t t0 = now();
  MatMul_fp32_fp32_fp32(A, B, Y_ref, M, N, O);
  uint32_t t1 = now();
  emit(host, "MatMul", M * N * O, 4u * (M * N + N * O + M * O), 0,
       t1 - t0, t1 - t0, 1);

  t0 = now();
  Gemm_fp32_fp32_fp32_fp32(A, B, Cm, Y_ref, M, N, O, 0, 0);
  t1 = now();
  emit(host, "Gemm", M * N * O, 4u * (M * N + N * O + 2 * M * O), 0,
       t1 - t0, t1 - t0, 1);

  t0 = now();
  Conv2d_fp32_fp32_fp32_NCHW(conv_in, C_IN, H_IN, W_IN, conv_w, F_OUT, KP, KQ,
                             1, 1, conv_b, 1, conv_ref);
  t1 = now();
  emit(host, "Conv", (uint32_t)F_OUT * H_OUT * W_OUT * C_IN * KP * KQ,
       4u * (C_IN * H_IN * W_IN + F_OUT * C_IN * KP * KQ + F_OUT +
             F_OUT * H_OUT * W_OUT),
       0, t1 - t0, t1 - t0, 1);
}

int main(void) {
  print_str("[HES-CAL] calibrating the engine cost model for this SoC\n");

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

  /* The host first: it computes the references the clusters are checked
   * against, and its own rates in the same pass. */
  cal_pass = 0;
  cal_host();

  /* Twice, because the two passes measure different things and a network
   * contains both. Pass 0 is cold -- the cluster's instruction cache has not
   * seen the kernel and its TCDM holds none of the operands, which is what the
   * first node of a given kind in a graph pays. Pass 1 is warm, which is what
   * every later node of that kind pays. The committed RATES table was read off
   * a warm pass (make mesh-test runs its staged loop twice), so sweep's
   * calibrate.py uses pass 1 for the rates and keeps pass 0 to price the first
   * offload. Measuring only one of them would misprice one half of every
   * network. */
  for (cal_pass = 0; cal_pass < 2; cal_pass++) {
    for (uint32_t e = HES_ENGINE_SNITCH; e < HES_NB_ENGINES; e++) {
      if (!hes_engine_ready(e)) {
        continue;
      }
      cal_matmul(e, MS, MS, MS, "MatMul_tiny");
      cal_matmul(e, M, N, O, "MatMul");
      Gemm_fp32_fp32_fp32_fp32(A, B, Cm, Y_ref, M, N, O, 0, 0);
      cal_gemm(e);
      cal_conv(e);
    }
  }

  print_str("[HES-CAL] failures=");
  print_u64((uint32_t)failures);
  print_str("\n");
  return failures ? 1 : 0;
}
