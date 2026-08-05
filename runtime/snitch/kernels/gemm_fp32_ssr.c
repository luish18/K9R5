/* MatMul / GEMM for Snitch, streamed through SSR and issued by FREP.
 *
 * Replaces the Deeploy Generic fp32 kernels of the same name (which stay
 * available as <name>_generic, see pipeline/run.py). The reduction runs in the
 * same order as the generic version, so MatMul is bit-identical to it; GEMM
 * starts the accumulator at C instead of adding C at the end, which can move
 * the last bit.
 *
 * The inner block computes UNROLL columns of one output row at once:
 *
 *   ft0  A[i][k], held for UNROLL consecutive FMAs (SSR repeat)
 *   ft1  B[k][j..j+UNROLL-1], one element per FMA
 *
 * so the two streams issue 1 + UNROLL memory accesses per UNROLL FMAs, and the
 * integer core issues nothing at all inside the loop: `frep.o` hands the eight
 * FMAs to the FPU sequencer, which replays them N times.
 */

#include "DeeployBasicMath.h"

#include "snitch_ssr.h"

/* Columns computed per block. Also the length of the FREP body, which the
 * sequencer's 16-entry buffer bounds, and the number of independent
 * accumulators, which has to cover the 3-cycle FMA latency. */
#define UNROLL 8

/* The Deeploy implementations, renamed by the build (-DMatMul...=..._generic).
 * They cover the shapes the streamed path does not: fewer than UNROLL columns,
 * or an empty problem. */
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

/* Y[M][O] = A[M][N] * B[N][O] (+ C[M][O] when pSrcC is not NULL).
 *
 * transA/transB select A[k][i] / B[j][k] instead; only the stream strides
 * change, the loop nest is the same. */
static void gemm_ssr(const float32_t *__restrict__ pSrcA,
                     const float32_t *__restrict__ pSrcB,
                     const float32_t *__restrict__ pSrcC,
                     float32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                     uint32_t O, int32_t transA, int32_t transB) {
  const uint32_t blocks = O / UNROLL;
  const uint32_t stride = sizeof(float32_t);

  /* A: one element per FMA, repeated across the UNROLL columns of a block,
   * rewound to the start of the row for the next block, one row per i. */
  if (transA) {
    ssr_loop_3d(SSR_DM0, N, blocks, M, M * stride, 0, stride);
  } else {
    ssr_loop_3d(SSR_DM0, N, blocks, M, stride, 0, N * stride);
  }
  ssr_repeat(SSR_DM0, UNROLL);

  /* B: UNROLL columns side by side, walking down k, then the next block of
   * columns, then back to the top for the next row of A. */
  if (transB) {
    ssr_loop_4d(SSR_DM1, UNROLL, N, blocks, M, N * stride, stride,
                UNROLL * N * stride, 0);
  } else {
    ssr_loop_4d(SSR_DM1, UNROLL, N, blocks, M, stride, O * stride,
                UNROLL * stride, 0);
  }

  ssr_read(SSR_DM0, SSR_3D, pSrcA);
  ssr_read(SSR_DM1, SSR_4D, pSrcB);

  for (uint32_t i = 0; i < M; ++i) {
    uint32_t j = 0;

    for (uint32_t b = 0; b < blocks; ++b) {
      /* Start from C so the addition costs nothing beyond the load. */
      float32_t c0, c1, c2, c3, c4, c5, c6, c7;
      if (pSrcC != NULL) {
        const float32_t *c = &pSrcC[i * O + j];
        c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
        c4 = c[4], c5 = c[5], c6 = c[6], c7 = c[7];
      } else {
        c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0.0f;
      }

      __asm__ volatile(SSR_FREP_BEGIN(UNROLL)
                       "fmadd.s %[c0], ft0, ft1, %[c0]\n"
                       "fmadd.s %[c1], ft0, ft1, %[c1]\n"
                       "fmadd.s %[c2], ft0, ft1, %[c2]\n"
                       "fmadd.s %[c3], ft0, ft1, %[c3]\n"
                       "fmadd.s %[c4], ft0, ft1, %[c4]\n"
                       "fmadd.s %[c5], ft0, ft1, %[c5]\n"
                       "fmadd.s %[c6], ft0, ft1, %[c6]\n"
                       "fmadd.s %[c7], ft0, ft1, %[c7]\n" SSR_FREP_END
                       : [c0] "+f"(c0), [c1] "+f"(c1), [c2] "+f"(c2),
                         [c3] "+f"(c3), [c4] "+f"(c4), [c5] "+f"(c5),
                         [c6] "+f"(c6), [c7] "+f"(c7)
                       : [frep_rpt] "r"(N - 1)
                       : "ft0", "ft1", "ft2", "memory");

      float32_t *y = &pDstY[i * O + j];
      y[0] = c0, y[1] = c1, y[2] = c2, y[3] = c3;
      y[4] = c4, y[5] = c5, y[6] = c6, y[7] = c7;
      j += UNROLL;
    }

    /* Columns past the last full block. SSR is disabled here — the streams
     * keep their position and pick up at the next block. */
    for (; j < O; ++j) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; ++k) {
        uint32_t a_idx = transA ? (k * M + i) : (i * N + k);
        uint32_t b_idx = transB ? (j * N + k) : (k * O + j);
        sum += pSrcA[a_idx] * pSrcB[b_idx];
      }
      pDstY[i * O + j] = pSrcC != NULL ? sum + pSrcC[i * O + j] : sum;
    }
  }
}

void MatMul_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                           const float32_t *__restrict__ pSrcB,
                           float32_t *__restrict__ pDstY, uint32_t M,
                           uint32_t N, uint32_t O) {
  if (M == 0 || N == 0 || O < UNROLL) {
    MatMul_fp32_fp32_fp32_generic(pSrcA, pSrcB, pDstY, M, N, O);
    return;
  }
  gemm_ssr(pSrcA, pSrcB, NULL, pDstY, M, N, O, 0, 0);
}

void Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                              const float32_t *__restrict__ pSrcB,
                              const float32_t *__restrict__ pDstC,
                              float32_t *__restrict__ pDstY, uint32_t M,
                              uint32_t N, uint32_t O, int32_t transA,
                              int32_t transB) {
  if (M == 0 || N == 0 || O < UNROLL) {
    Gemm_fp32_fp32_fp32_fp32_generic(pSrcA, pSrcB, pDstC, pDstY, M, N, O,
                                     transA, transB);
    return;
  }
  gemm_ssr(pSrcA, pSrcB, pDstC, pDstY, M, N, O, transA, transB);
}
