/* MatMul / GEMM for Spatz, hand-written against RVV.
 *
 * The default for Spatz; --spatz-kernels autovec runs whatever GCC makes of
 * the Deeploy Generic sources instead, which is the comparison this file
 * exists to answer. GCC vectorizes the innermost k loop, i.e. the dot product,
 * and that shape is wrong for this machine twice over:
 *
 *   vlse32.v  B[k][j] down a column — a strided gather whose stride is a
 *             multiple of the TCDM bank interleave for every power-of-two
 *             row length, so the elements serialize onto one bank
 *   vfredusum a full vector reduction per output element, plus the vsetvli
 *             and vfmv.f.s around it
 *
 * The standard fix is to move the reduction out of the vector unit: keep the
 * accumulator in a vector register across k and broadcast the A element
 * instead. Every load is then unit-stride, and no reduction is executed at all
 *
 *   acc[r] += A[i+r][k] * B[k][j..j+vl]      (vfmacc.vf)
 *
 * ROWS rows of A share one B vector load, which is what puts the kernel on the
 * compute side of the load/FMA balance: one load feeds ROWS x vl FMAs.
 *
 * LMUL=4 rather than 1, which is worth 1.8x on its own: a back-to-back vfmacc
 * loop with every operand already in the vector register file sustains 0.180
 * cycles/element at LMUL=1 against 0.128 at LMUL=4, so a third of the peak
 * goes to issue overhead before the kernel touches memory. Four accumulators
 * at LMUL=4 occupy 16 of the 32 vector registers, and the B vector four more.
 *
 * Replaces the Deeploy Generic kernels of the same name, which stay reachable
 * as <name>_generic and take the shapes below that this path does not cover
 * (transposed operands, where B would no longer be contiguous along j).
 */

#include <riscv_vector.h>

#include "DeeployBasicMath.h"

/* Rows of A per block: the number of independent vector accumulators, and the
 * reuse factor on each B load. */
#define ROWS 4

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

/* One block: rows [i, i+rows) of Y, columns [j, j+vl). */
static inline void gemm_block(const float32_t *__restrict__ pSrcA,
                              const float32_t *__restrict__ pSrcB,
                              const float32_t *__restrict__ pSrcC,
                              float32_t *__restrict__ pDstY, uint32_t N,
                              uint32_t O, uint32_t i, uint32_t j, uint32_t rows,
                              size_t vl) {
  vfloat32m4_t acc0, acc1, acc2, acc3;

  if (pSrcC != NULL) {
    acc0 = __riscv_vle32_v_f32m4(&pSrcC[(i + 0) * O + j], vl);
    acc1 = rows > 1 ? __riscv_vle32_v_f32m4(&pSrcC[(i + 1) * O + j], vl) : acc0;
    acc2 = rows > 2 ? __riscv_vle32_v_f32m4(&pSrcC[(i + 2) * O + j], vl) : acc0;
    acc3 = rows > 3 ? __riscv_vle32_v_f32m4(&pSrcC[(i + 3) * O + j], vl) : acc0;
  } else {
    acc0 = acc1 = acc2 = acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  }

  const float32_t *a = &pSrcA[i * N];
  const float32_t *b = &pSrcB[j];

  for (uint32_t k = 0; k < N; ++k) {
    vfloat32m4_t bv = __riscv_vle32_v_f32m4(b, vl);
    acc0 = __riscv_vfmacc_vf_f32m4(acc0, a[0 * N], bv, vl);
    if (rows > 1)
      acc1 = __riscv_vfmacc_vf_f32m4(acc1, a[1 * N], bv, vl);
    if (rows > 2)
      acc2 = __riscv_vfmacc_vf_f32m4(acc2, a[2 * N], bv, vl);
    if (rows > 3)
      acc3 = __riscv_vfmacc_vf_f32m4(acc3, a[3 * N], bv, vl);
    a += 1;
    b += O;
  }

  __riscv_vse32_v_f32m4(&pDstY[(i + 0) * O + j], acc0, vl);
  if (rows > 1)
    __riscv_vse32_v_f32m4(&pDstY[(i + 1) * O + j], acc1, vl);
  if (rows > 2)
    __riscv_vse32_v_f32m4(&pDstY[(i + 2) * O + j], acc2, vl);
  if (rows > 3)
    __riscv_vse32_v_f32m4(&pDstY[(i + 3) * O + j], acc3, vl);
}

static void gemm_rvv(const float32_t *__restrict__ pSrcA,
                     const float32_t *__restrict__ pSrcB,
                     const float32_t *__restrict__ pSrcC,
                     float32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                     uint32_t O) {
  for (uint32_t i = 0; i < M; i += ROWS) {
    const uint32_t rows = (M - i) < ROWS ? (M - i) : ROWS;
    for (uint32_t j = 0; j < O;) {
      const size_t vl = __riscv_vsetvl_e32m4(O - j);
      gemm_block(pSrcA, pSrcB, pSrcC, pDstY, N, O, i, j, rows, vl);
      j += vl;
    }
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
  /* A transposed A costs only a scalar stride; a transposed B breaks the
   * unit-stride B load this kernel is built on, so both go generic. */
  if (M == 0 || N == 0 || O == 0 || transA || transB) {
    Gemm_fp32_fp32_fp32_fp32_generic(pSrcA, pSrcB, pDstC, pDstY, M, N, O,
                                     transA, transB);
    return;
  }
  gemm_rvv(pSrcA, pSrcB, pDstC, pDstY, M, N, O);
}
