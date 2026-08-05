/* Correctness test for the SSR/FREP kernels in runtime/snitch/kernels/.
 *
 * The benchmark ops only hit the shapes that divide evenly (MatMul O=8,
 * GEMM O=32 untransposed, Conv F=4). This runs the cases they miss — columns
 * past the last block, transposed operands, filters past the last group — and
 * compares every output against a scalar reference computed here.
 *
 * Build: the kernels plus this file, with the Deeploy Generic includes on the
 * path; the *_generic fallbacks the kernels call are stubbed below with the
 * same scalar code the reference uses.
 */
#include "../common/bench.h"
#include "../common/miniio.h"

#include "DeeployBasicMath.h"

void MatMul_fp32_fp32_fp32(const float32_t *__restrict__ A,
                           const float32_t *__restrict__ B,
                           float32_t *__restrict__ Y, uint32_t M, uint32_t N,
                           uint32_t O);
void Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ A,
                              const float32_t *__restrict__ B,
                              const float32_t *__restrict__ C,
                              float32_t *__restrict__ Y, uint32_t M, uint32_t N,
                              uint32_t O, int32_t transA, int32_t transB);
void Conv2d_fp32_fp32_fp32_NCHW(const float32_t *__restrict__ A, uint32_t C,
                                uint32_t H_padded, uint32_t W_padded,
                                const float32_t *__restrict__ W, uint32_t F,
                                uint32_t P, uint32_t Q, uint32_t SP,
                                uint32_t SQ,
                                const float32_t *__restrict__ bias,
                                const bool has_bias,
                                float32_t *__restrict__ Y);

/* --- scalar reference, also used as the kernels' _generic fallback --- */

void Gemm_fp32_fp32_fp32_fp32_generic(const float32_t *__restrict__ A,
                                      const float32_t *__restrict__ B,
                                      const float32_t *__restrict__ C,
                                      float32_t *__restrict__ Y, uint32_t M,
                                      uint32_t N, uint32_t O, int32_t transA,
                                      int32_t transB) {
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < O; ++j) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; ++k) {
        sum += A[transA ? (k * M + i) : (i * N + k)] *
               B[transB ? (j * N + k) : (k * O + j)];
      }
      Y[i * O + j] = C != NULL ? sum + C[i * O + j] : sum;
    }
  }
}

void MatMul_fp32_fp32_fp32_generic(const float32_t *__restrict__ A,
                                   const float32_t *__restrict__ B,
                                   float32_t *__restrict__ Y, uint32_t M,
                                   uint32_t N, uint32_t O) {
  Gemm_fp32_fp32_fp32_fp32_generic(A, B, NULL, Y, M, N, O, 0, 0);
}

void Conv2d_fp32_fp32_fp32_NCHW_generic(
    const float32_t *__restrict__ A, uint32_t C, uint32_t H_padded,
    uint32_t W_padded, const float32_t *__restrict__ W, uint32_t F, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ, const float32_t *__restrict__ bias,
    const bool has_bias, float32_t *__restrict__ Y) {
  uint32_t H_out = (H_padded - P) / SP + 1;
  uint32_t W_out = (W_padded - Q) / SQ + 1;
  for (uint32_t f = 0; f < F; ++f) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        float32_t sum = 0.0f;
        for (uint32_t c = 0; c < C; ++c)
          for (uint32_t p = 0; p < P; ++p)
            for (uint32_t q = 0; q < Q; ++q)
              sum += A[c * H_padded * W_padded + (h * SP + p) * W_padded +
                       (w * SQ + q)] *
                     W[f * C * P * Q + c * P * Q + p * Q + q];
        Y[f * H_out * W_out + h * W_out + w] = has_bias ? sum + bias[f] : sum;
      }
    }
  }
}

/* --- harness --- */

static int failures = 0;

static void check(const char *name, const float32_t *got,
                  const float32_t *want, uint32_t n) {
  float32_t worst = 0.0f;
  for (uint32_t i = 0; i < n; i++) {
    float32_t d = got[i] - want[i];
    if (d < 0)
      d = -d;
    if (d > worst)
      worst = d;
  }
  int ok = worst < 1e-4f;
  failures += !ok;
  print_str(ok ? "  ok   " : "  FAIL ");
  print_str(name);
  print_str("  maxdiff_e6=");
  print_u64((uint64_t)(worst * 1e6f));
  print_str("\n");
}

#define M 5
#define N 9
#define O 11 /* not a multiple of the 8-column unroll */

static float32_t A[M * N], B[N * O], Cm[M * O], Y[M * O], REF[M * O];
static float32_t At[N * M], Bt[O * N];

/* Conv: 3 filters, so the last one falls past the 4-filter group. */
#define CONV_C 2
#define CONV_H 9
#define CONV_W 8
#define CONV_F 3
#define CONV_P 3
#define CONV_Q 2
#define CONV_SP 2
#define CONV_SQ 3
#define CONV_H_OUT ((CONV_H - CONV_P) / CONV_SP + 1)
#define CONV_W_OUT ((CONV_W - CONV_Q) / CONV_SQ + 1)

static float32_t IN[CONV_C * CONV_H * CONV_W];
static float32_t WT[CONV_F * CONV_C * CONV_P * CONV_Q];
static float32_t BIAS[CONV_F];
static float32_t COUT[CONV_F * CONV_H_OUT * CONV_W_OUT];
static float32_t CREF[CONV_F * CONV_H_OUT * CONV_W_OUT];

static float32_t noise(uint32_t i) { return (float32_t)(i % 13) * 0.25f - 1.5f; }

int main(void) {
  for (uint32_t i = 0; i < M * N; i++)
    A[i] = noise(i);
  for (uint32_t i = 0; i < N * O; i++)
    B[i] = noise(i + 3);
  for (uint32_t i = 0; i < M * O; i++)
    Cm[i] = noise(i + 7);
  for (uint32_t i = 0; i < M; i++)
    for (uint32_t k = 0; k < N; k++)
      At[k * M + i] = A[i * N + k];
  for (uint32_t k = 0; k < N; k++)
    for (uint32_t j = 0; j < O; j++)
      Bt[j * N + k] = B[k * O + j];

  print_str("ssr_kernels: leftover columns and transposes\n");

  MatMul_fp32_fp32_fp32_generic(A, B, REF, M, N, O);
  MatMul_fp32_fp32_fp32(A, B, Y, M, N, O);
  check("MatMul  O%unroll!=0", Y, REF, M * O);

  Gemm_fp32_fp32_fp32_fp32_generic(A, B, Cm, REF, M, N, O, 0, 0);
  Gemm_fp32_fp32_fp32_fp32(A, B, Cm, Y, M, N, O, 0, 0);
  check("Gemm    transA=0 transB=0", Y, REF, M * O);

  Gemm_fp32_fp32_fp32_fp32(At, B, Cm, Y, M, N, O, 1, 0);
  check("Gemm    transA=1 transB=0", Y, REF, M * O);

  Gemm_fp32_fp32_fp32_fp32(A, Bt, Cm, Y, M, N, O, 0, 1);
  check("Gemm    transA=0 transB=1", Y, REF, M * O);

  Gemm_fp32_fp32_fp32_fp32(At, Bt, Cm, Y, M, N, O, 1, 1);
  check("Gemm    transA=1 transB=1", Y, REF, M * O);

  /* Fewer columns than the unroll: the kernel must hand over to the
   * generic implementation instead of configuring an empty stream. */
  MatMul_fp32_fp32_fp32_generic(A, B, REF, M, N, 4);
  MatMul_fp32_fp32_fp32(A, B, Y, M, N, 4);
  check("MatMul  O<unroll", Y, REF, M * 4);

  for (uint32_t i = 0; i < CONV_C * CONV_H * CONV_W; i++)
    IN[i] = noise(i + 1);
  for (uint32_t i = 0; i < CONV_F * CONV_C * CONV_P * CONV_Q; i++)
    WT[i] = noise(i + 5);
  for (uint32_t i = 0; i < CONV_F; i++)
    BIAS[i] = noise(i + 11);

  Conv2d_fp32_fp32_fp32_NCHW_generic(IN, CONV_C, CONV_H, CONV_W, WT, CONV_F,
                                     CONV_P, CONV_Q, CONV_SP, CONV_SQ, BIAS,
                                     true, CREF);
  Conv2d_fp32_fp32_fp32_NCHW(IN, CONV_C, CONV_H, CONV_W, WT, CONV_F, CONV_P,
                             CONV_Q, CONV_SP, CONV_SQ, BIAS, true, COUT);
  check("Conv2d  F%unroll!=0, bias", COUT, CREF,
        CONV_F * CONV_H_OUT * CONV_W_OUT);

  Conv2d_fp32_fp32_fp32_NCHW_generic(IN, CONV_C, CONV_H, CONV_W, WT, CONV_F,
                                     CONV_P, CONV_Q, CONV_SP, CONV_SQ, BIAS,
                                     false, CREF);
  Conv2d_fp32_fp32_fp32_NCHW(IN, CONV_C, CONV_H, CONV_W, WT, CONV_F, CONV_P,
                             CONV_Q, CONV_SP, CONV_SQ, BIAS, false, COUT);
  check("Conv2d  F%unroll!=0, no bias", COUT, CREF,
        CONV_F * CONV_H_OUT * CONV_W_OUT);

  print_str(failures ? "ssr_kernels: FAILED\n" : "ssr_kernels: OK\n");
  return failures != 0;
}
