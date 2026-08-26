/* The kernels a cluster can be asked to run, and the shape of their argument
 * lists in the job descriptor.
 *
 * The argument order is exactly the Deeploy Generic kernel signature, so the
 * dispatch table calls runtime/snitch/kernels/*.c (Xssr/Xfrep) on the Snitch
 * cluster and the autovectorized Generic kernels on the Spatz pair without
 * either side needing an adapter.
 */
#ifndef HES_JOB_H
#define HES_JOB_H

#include "hes_mailbox.h"

/* MatMul_fp32_fp32_fp32(A, B, Y, M, N, O)
 *   Y[M][O] = A[M][N] * B[N][O] */
enum {
  HES_MM_A = 0, HES_MM_B, HES_MM_Y, HES_MM_M, HES_MM_N, HES_MM_O,
  HES_MM_NARGS
};

/* Gemm_fp32_fp32_fp32_fp32(A, B, C, Y, M, N, O, transA, transB)
 *   Y[M][O] = A[M][N] * B[N][O] + C[M][O] */
enum {
  HES_GEMM_A = 0, HES_GEMM_B, HES_GEMM_C, HES_GEMM_Y, HES_GEMM_M, HES_GEMM_N,
  HES_GEMM_O, HES_GEMM_TRANSA, HES_GEMM_TRANSB,
  HES_GEMM_NARGS
};

/* Conv2d_fp32_fp32_fp32_NCHW(A, C, H, W, B, F, P, Q, SP, SQ, bias, has_bias, Y)
 *   Y[F][H_out][W_out] over input A[C][H][W] with F filters B[F][C][P][Q] */
enum {
  HES_CONV_A = 0, HES_CONV_C, HES_CONV_H, HES_CONV_W, HES_CONV_B, HES_CONV_F,
  HES_CONV_P, HES_CONV_Q, HES_CONV_SP, HES_CONV_SQ, HES_CONV_BIAS,
  HES_CONV_HAS_BIAS, HES_CONV_Y,
  HES_CONV_NARGS
};

/* Set in `flags` to ask the DMA core to stage the operands into TCDM before
 * the kernel runs and copy the result back after. Without it the kernel runs
 * against main memory directly: correct, but every access pays DRAM latency.
 * The cluster clears it if the operands do not fit its scratch. */
#define HES_JOB_STAGE 1u

#endif /* HES_JOB_H */
