/* MatMul / GEMM for Spatz, as an RVV microkernel.
 *
 * Replaces the Deeploy Generic fp32 kernels of the same name (which stay
 * available as <name>_generic, see pipeline/build_mesh.py).
 *
 * The point of writing these by hand is the loop order. Left to autovectorize
 * the generic C, GCC vectorizes the *reduction* axis -- the k loop of the dot
 * product -- which costs, for every single output element:
 *
 *   vlse32.v      a strided load of B, one element per row, scattered
 *                 across the TCDM banks instead of a unit-stride burst
 *   vfredusum.vs  a horizontal tree reduction, serializing the lanes that
 *                 were just used in parallel
 *   vfmv.f.s      a vector-to-scalar extract, and back to scalar code
 *
 * which measured 0.54 MAC/cycle on a 32x32x32 GEMM, roughly an eighth of what
 * four lanes can do.
 *
 * This vectorizes the *output columns* instead. B is read unit-stride, A comes
 * in as a scalar broadcast through vfmacc.vf, the accumulator stays in a vector
 * register for the whole k loop, and there is no reduction at all -- the
 * accumulator is the answer and gets stored once.
 *
 * Rows are unrolled by four so each B vector load feeds four FMAs rather than
 * one, which is what moves the kernel off being load-bound. The four
 * accumulators are independent, which also covers the FPU latency.
 *
 * The k order is unchanged from the generic kernel, and the accumulator starts
 * at zero for MatMul, so MatMul stays bit-identical to the scalar version.
 * GEMM starts its accumulator at C rather than adding C at the end, the same
 * deviation the Snitch kernel makes, which can move the last bit.
 */

#include "DeeployBasicMath.h"

#include <riscv_vector.h>

/* Output rows computed together, sharing each load of B. */
#define UNROLL_M 4

/* The Deeploy implementations, renamed by the build. They cover what the
 * microkernel does not: the transposed operand layouts. */
void MatMul_fp32_fp32_fp32_generic(const float32_t *__restrict__ pSrcA,
                                   const float32_t *__restrict__ pSrcB,
                                   float32_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O);

void Gemm_fp32_fp32_fp32_fp32_generic(const float32_t *__restrict__ pSrcA,
                                      const float32_t *__restrict__ pSrcB,
                                      const float32_t *__restrict__ pDstC,
                                      float32_t *__restrict__ pDstY, uint32_t M,
                                      uint32_t N, uint32_t O, int32_t transA,
                                      int32_t transB);

/* Four output rows at once, over all of O.
 *
 * RVV vector types are sizeless and cannot live in an array, so the four
 * accumulators are named rather than indexed, and the strip width is fixed at
 * UNROLL_M.
 *
 * pSrcC may be NULL, in which case the accumulators start at zero and this is
 * a MatMul.
 */
static void gemm_rows4(const float32_t *__restrict__ pSrcA,
                       const float32_t *__restrict__ pSrcB,
                       const float32_t *__restrict__ pSrcC,
                       float32_t *__restrict__ pDstY, uint32_t i, uint32_t N,
                       uint32_t O) {
  const float32_t *a0 = &pSrcA[(i + 0) * N];
  const float32_t *a1 = &pSrcA[(i + 1) * N];
  const float32_t *a2 = &pSrcA[(i + 2) * N];
  const float32_t *a3 = &pSrcA[(i + 3) * N];

  for (uint32_t j = 0; j < O;) {
    size_t vl = __riscv_vsetvl_e32m1(O - j);

    vfloat32m1_t acc0, acc1, acc2, acc3;
    if (pSrcC) {
      acc0 = __riscv_vle32_v_f32m1(&pSrcC[(i + 0) * O + j], vl);
      acc1 = __riscv_vle32_v_f32m1(&pSrcC[(i + 1) * O + j], vl);
      acc2 = __riscv_vle32_v_f32m1(&pSrcC[(i + 2) * O + j], vl);
      acc3 = __riscv_vle32_v_f32m1(&pSrcC[(i + 3) * O + j], vl);
    } else {
      acc0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      acc1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      acc2 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      acc3 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    }

    /* The whole point: one unit-stride load of B per k, reused by four FMAs,
     * and nothing else touching memory inside the loop. */
    for (uint32_t k = 0; k < N; k++) {
      vfloat32m1_t vb = __riscv_vle32_v_f32m1(&pSrcB[k * O + j], vl);
      acc0 = __riscv_vfmacc_vf_f32m1(acc0, a0[k], vb, vl);
      acc1 = __riscv_vfmacc_vf_f32m1(acc1, a1[k], vb, vl);
      acc2 = __riscv_vfmacc_vf_f32m1(acc2, a2[k], vb, vl);
      acc3 = __riscv_vfmacc_vf_f32m1(acc3, a3[k], vb, vl);
    }

    __riscv_vse32_v_f32m1(&pDstY[(i + 0) * O + j], acc0, vl);
    __riscv_vse32_v_f32m1(&pDstY[(i + 1) * O + j], acc1, vl);
    __riscv_vse32_v_f32m1(&pDstY[(i + 2) * O + j], acc2, vl);
    __riscv_vse32_v_f32m1(&pDstY[(i + 3) * O + j], acc3, vl);
    j += vl;
  }
}

/* One output row, for the rows past the last full strip of four. */
static void gemm_row1(const float32_t *__restrict__ pSrcA,
                      const float32_t *__restrict__ pSrcB,
                      const float32_t *__restrict__ pSrcC,
                      float32_t *__restrict__ pDstY, uint32_t i, uint32_t N,
                      uint32_t O) {
  const float32_t *a0 = &pSrcA[i * N];

  for (uint32_t j = 0; j < O;) {
    size_t vl = __riscv_vsetvl_e32m1(O - j);
    vfloat32m1_t acc = pSrcC ? __riscv_vle32_v_f32m1(&pSrcC[i * O + j], vl)
                             : __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (uint32_t k = 0; k < N; k++) {
      vfloat32m1_t vb = __riscv_vle32_v_f32m1(&pSrcB[k * O + j], vl);
      acc = __riscv_vfmacc_vf_f32m1(acc, a0[k], vb, vl);
    }
    __riscv_vse32_v_f32m1(&pDstY[i * O + j], acc, vl);
    j += vl;
  }
}

static void gemm_rvv(const float32_t *__restrict__ pSrcA,
                     const float32_t *__restrict__ pSrcB,
                     const float32_t *__restrict__ pSrcC,
                     float32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                     uint32_t O) {
  uint32_t i = 0;
  for (; i + UNROLL_M <= M; i += UNROLL_M) {
    gemm_rows4(pSrcA, pSrcB, pSrcC, pDstY, i, N, O);
  }
  for (; i < M; i++) {
    gemm_row1(pSrcA, pSrcB, pSrcC, pDstY, i, N, O);
  }
}

void MatMul_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                           const float32_t *__restrict__ pSrcB,
                           float32_t *__restrict__ pDstY, uint32_t M,
                           uint32_t N, uint32_t O) {
  if (M == 0 || N == 0 || O == 0) {
    MatMul_fp32_fp32_fp32_generic(pSrcA, pSrcB, pDstY, M, N, O);
    return;
  }
  gemm_rvv(pSrcA, pSrcB, NULL, pDstY, M, N, O);
}

void Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                              const float32_t *__restrict__ pSrcB,
                              const float32_t *__restrict__ pDstC,
                              float32_t *__restrict__ pDstY, uint32_t M,
                              uint32_t N, uint32_t O, int32_t transA,
                              int32_t transB) {
  /* A transposed operand makes B's row stride stop being O, which is exactly
   * the unit-stride property this kernel is built on. Leave those to the
   * generic version rather than get them subtly wrong. */
  if (transA || transB || M == 0 || N == 0 || O == 0) {
    Gemm_fp32_fp32_fp32_fp32_generic(pSrcA, pSrcB, pDstC, pDstY, M, N, O,
                                     transA, transB);
    return;
  }
  gemm_rvv(pSrcA, pSrcB, pDstC, pDstY, M, N, O);
}
