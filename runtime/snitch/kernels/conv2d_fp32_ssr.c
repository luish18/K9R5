/* NCHW fp32 convolution for Snitch, streamed through SSR and issued by FREP.
 *
 * Replaces the Deeploy Generic kernel of the same name (which stays available
 * as Conv2d_fp32_fp32_fp32_NCHW_generic, see pipeline/run.py). Taps accumulate
 * in the same order as the generic version, so results are bit-identical.
 *
 * One output pixel is a dot product of length C*P*Q, which is too short a
 * dependency chain to keep the FPU busy on its own. F_UNROLL filters are
 * therefore computed together, sharing the input window:
 *
 *   ft0  the input element at tap k, held for F_UNROLL FMAs (SSR repeat)
 *   ft1  weight k of each of the F_UNROLL filters
 *
 * which is F_UNROLL independent accumulators, 1 + F_UNROLL memory accesses per
 * F_UNROLL FMAs, and one `frep.o` per output pixel: C*P*Q replays of the body
 * with the integer core idle.
 */

#include "DeeployBasicMath.h"

#include "snitch_ssr.h"

/* Filters computed together. Also the FREP body length and the number of
 * independent accumulators, which has to cover the 3-cycle FMA latency. */
#define F_UNROLL 4

/* The Deeploy implementation, renamed by the build. Covers what the streamed
 * path does not: fewer than F_UNROLL filters, or an empty problem. */
void Conv2d_fp32_fp32_fp32_NCHW_generic(
    const float32_t *__restrict__ pSrcA, uint32_t C, uint32_t H_padded,
    uint32_t W_padded, const float32_t *__restrict__ pSrcB, uint32_t F,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC);

void Conv2d_fp32_fp32_fp32_NCHW(const float32_t *__restrict__ pSrcA, uint32_t C,
                                uint32_t H_padded, uint32_t W_padded,
                                const float32_t *__restrict__ pSrcB, uint32_t F,
                                uint32_t P, uint32_t Q, uint32_t SP,
                                uint32_t SQ,
                                const float32_t *__restrict__ pSrcBias,
                                const bool has_bias,
                                float32_t *__restrict__ pDstC) {
  const uint32_t H_out = (H_padded - P) / SP + 1;
  const uint32_t W_out = (W_padded - Q) / SQ + 1;
  const uint32_t taps = C * P * Q;
  const uint32_t groups = F / F_UNROLL;
  const uint32_t stride = sizeof(float32_t);

  if (groups == 0 || taps == 0 || H_out == 0 || W_out == 0) {
    Conv2d_fp32_fp32_fp32_NCHW_generic(pSrcA, C, H_padded, W_padded, pSrcB, F,
                                       P, Q, SP, SQ, pSrcBias, has_bias, pDstC);
    return;
  }

  /* The input window of one output pixel: along the row, down the rows of the
   * window, then to the next input channel. Each element feeds F_UNROLL FMAs. */
  ssr_loop_3d(SSR_DM0, Q, P, C, stride, W_padded * stride,
              H_padded * W_padded * stride);
  ssr_repeat(SSR_DM0, F_UNROLL);

  /* Tap k of F_UNROLL filters back to back, then the next tap. Filters are
   * C*P*Q apart, taps within a filter are contiguous. */
  ssr_loop_2d(SSR_DM1, F_UNROLL, taps, taps * stride, stride);

  for (uint32_t f = 0; f < groups * F_UNROLL; f += F_UNROLL) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        float32_t c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

        /* Both streams restart at every output pixel; writing a pointer also
         * rewinds that stream's counters. */
        ssr_read(SSR_DM0, SSR_3D, &pSrcA[(h * SP) * W_padded + w * SQ]);
        ssr_read(SSR_DM1, SSR_2D, &pSrcB[f * taps]);

        __asm__ volatile(SSR_FREP_BEGIN(F_UNROLL)
                         "fmadd.s %[c0], ft0, ft1, %[c0]\n"
                         "fmadd.s %[c1], ft0, ft1, %[c1]\n"
                         "fmadd.s %[c2], ft0, ft1, %[c2]\n"
                         "fmadd.s %[c3], ft0, ft1, %[c3]\n" SSR_FREP_END
                         : [c0] "+f"(c0), [c1] "+f"(c1), [c2] "+f"(c2),
                           [c3] "+f"(c3)
                         : [frep_rpt] "r"(taps - 1)
                         : "ft0", "ft1", "ft2", "memory");

        float32_t *y = &pDstC[f * H_out * W_out + h * W_out + w];
        const uint32_t plane = H_out * W_out;
        if (has_bias) {
          y[0 * plane] = c0 + pSrcBias[f + 0];
          y[1 * plane] = c1 + pSrcBias[f + 1];
          y[2 * plane] = c2 + pSrcBias[f + 2];
          y[3 * plane] = c3 + pSrcBias[f + 3];
        } else {
          y[0 * plane] = c0;
          y[1 * plane] = c1;
          y[2 * plane] = c2;
          y[3 * plane] = c3;
        }
      }
    }
  }

  /* Filters past the last full group, scalar. */
  for (uint32_t f = groups * F_UNROLL; f < F; ++f) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        float32_t sum = 0.0f;
        for (uint32_t c = 0; c < C; ++c) {
          for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t q = 0; q < Q; ++q) {
              sum += pSrcA[c * H_padded * W_padded + (h * SP + p) * W_padded +
                           (w * SQ + q)] *
                     pSrcB[f * taps + c * P * Q + p * Q + q];
            }
          }
        }
        pDstC[f * H_out * W_out + h * W_out + w] =
            has_bias ? sum + pSrcBias[f] : sum;
      }
    }
  }
}
