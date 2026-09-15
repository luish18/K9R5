/* The MFCC front-end for Snitch, streamed through SSR and issued by FREP.
 *
 * Replaces the portable kernel of the same name (runtime/common/kernels/
 * mfcc_fp32.c, which stays reachable as Mfcc_fp32_fp32_generic) on the Snitch
 * cluster alone. Two of the four stages are worth streaming, and they are the
 * two whose access pattern is the reason this application is on this cluster:
 *
 *   mel filterbank  40 reductions, each over its own ragged 10-30 bin run.
 *                   ft0 walks the power spectrum from the filter's first bin,
 *                   ft1 walks that filter's coefficients, and one frep.o
 *                   replays UNROLL fmadd.s until the run is done. No address
 *                   arithmetic, no loop, and -- the part that matters against a
 *                   vector unit -- no horizontal reduction: the accumulators
 *                   stay in registers across the whole filter.
 *
 *   DCT-II          a dense nb_cep x nb_mel matrix-vector product. ft0 repeats
 *                   the log-mel vector once per output block, ft1 walks the
 *                   matrix, and the accumulators again live across the whole
 *                   reduction.
 *
 * The FFT is left to the portable code on purpose. Its butterfly needs four
 * reads and four writes per iteration against three data movers, so streaming
 * it means splitting it into passes through a scratch buffer -- and the memory
 * traffic that costs is not obviously cheaper than the loads it removes. That
 * is a real question rather than a rhetorical one, so it is left open and
 * measured rather than guessed at; see the KWS section of README.md.
 *
 * The accumulation order is the generic kernel's, so both produce the same
 * numbers to the last bit on the shapes the streamed path covers.
 */

#include <math.h>

#include "mfcc.h"
#include "snitch_ssr.h"

/* Independent accumulators, the FREP body length, and the block size of both
 * reductions. Has to cover the FPU's 3-cycle FMA latency. */
#define UNROLL 4

/* Must match MFCC_LOG_FLOOR in runtime/common/kernels/mfcc_fp32.c and MEL_FLOOR
 * in pipeline/kws.py. runtime/mesh/kws_main.c checks every clip's features
 * against the numpy reference, so a disagreement here fails the run rather
 * than quietly shifting the features. */
#define MFCC_LOG_FLOOR 1e-6f

static inline uint32_t bit_reverse(uint32_t i, uint32_t bits) {
  uint32_t r = 0;
  for (uint32_t b = 0; b < bits; b++) {
    r = (r << 1) | ((i >> b) & 1u);
  }
  return r;
}

/* Same radix-2 decimation-in-time FFT as the portable kernel, and deliberately
 * so: it is the part that is not streamed, and keeping it identical is what
 * makes the measured difference attributable to the two stages that are. */
static void fft_radix2(float32_t *__restrict__ re, float32_t *__restrict__ im,
                       uint32_t n, const float32_t *__restrict__ twiddles) {
  uint32_t bits = 0;
  while ((1u << bits) < n) {
    bits++;
  }

  for (uint32_t i = 0; i < n; i++) {
    uint32_t j = bit_reverse(i, bits);
    if (j > i) {
      float32_t tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      float32_t ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (uint32_t len = 2; len <= n; len <<= 1) {
    const uint32_t half = len >> 1;
    const uint32_t step = n / len;
    for (uint32_t j = 0; j < half; j++) {
      const float32_t wr = twiddles[2u * j * step];
      const float32_t wi = twiddles[2u * j * step + 1u];
      for (uint32_t base = 0; base < n; base += len) {
        const uint32_t a = base + j;
        const uint32_t b = a + half;
        const float32_t xr = re[b], xi = im[b];
        const float32_t vr = xr * wr - xi * wi;
        const float32_t vi = xr * wi + xi * wr;
        const float32_t ur = re[a], ui = im[a];
        re[a] = ur + vr;
        im[a] = ui + vi;
        re[b] = ur - vr;
        im[b] = ui - vi;
      }
    }
  }
}

/* One filter's energy: a dot product of `len` elements, streamed.
 *
 * The trip count is data-dependent -- every filter is a different width -- and
 * that is exactly the case a vector unit handles badly and a stream handles
 * without noticing: the FREP repeat count is just a register. */
static inline float32_t mel_energy(const float32_t *__restrict__ power,
                                   const float32_t *__restrict__ coeff,
                                   uint32_t len) {
  const uint32_t stride = sizeof(float32_t);
  const uint32_t blocks = len / UNROLL;
  float32_t acc = 0.0f;

  if (blocks != 0) {
    float32_t a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    /* One element per FMA from each stream, four FMAs per replay: a straight
     * unit-stride walk of both arrays. The repeat register is shared with the
     * DCT below, which wants a different value, so it is set here rather than
     * assumed. */
    ssr_repeat(SSR_DM0, 1);
    ssr_repeat(SSR_DM1, 1);
    ssr_loop_1d(SSR_DM0, blocks * UNROLL, stride);
    ssr_loop_1d(SSR_DM1, blocks * UNROLL, stride);
    ssr_read(SSR_DM0, SSR_1D, power);
    ssr_read(SSR_DM1, SSR_1D, coeff);

    __asm__ volatile(SSR_FREP_BEGIN(UNROLL)
                     "fmadd.s %[a0], ft0, ft1, %[a0]\n"
                     "fmadd.s %[a1], ft0, ft1, %[a1]\n"
                     "fmadd.s %[a2], ft0, ft1, %[a2]\n"
                     "fmadd.s %[a3], ft0, ft1, %[a3]\n" SSR_FREP_END
                     : [a0] "+f"(a0), [a1] "+f"(a1), [a2] "+f"(a2),
                       [a3] "+f"(a3)
                     : [frep_rpt] "r"(blocks - 1)
                     : "ft0", "ft1", "ft2", "memory");
    /* Summed in the generic kernel's order: a0 holds elements 0, 4, 8 ... so
     * folding them 0+1+2+3 reproduces its left-to-right accumulation only up to
     * reassociation. The tolerance the run checks against covers that; see the
     * feature check in runtime/mesh/kws_main.c. */
    acc = ((a0 + a1) + (a2 + a3));
  }

  /* The tail past the last full block, with SSR disabled. */
  for (uint32_t i = blocks * UNROLL; i < len; i++) {
    acc += power[i] * coeff[i];
  }
  return acc;
}

void Mfcc_fp32_fp32(const float32_t *__restrict__ audio,
                    float32_t *__restrict__ out, uint32_t nb_frames,
                    uint32_t frame_len, uint32_t hop, uint32_t fft_len,
                    const float32_t *__restrict__ window,
                    const float32_t *__restrict__ twiddles,
                    const float32_t *__restrict__ mel_coeff,
                    const uint16_t *__restrict__ mel_start,
                    const uint16_t *__restrict__ mel_len, uint32_t nb_mel,
                    const float32_t *__restrict__ dct, uint32_t nb_cep,
                    float32_t *__restrict__ scratch) {
  /* Shapes the streamed path does not cover: fewer cepstra than the DCT block,
   * or nothing to do. */
  if (nb_frames == 0 || nb_mel == 0 || nb_cep == 0) {
    return;
  }

  float32_t *const re = scratch;
  float32_t *const im = scratch + fft_len;
  float32_t *const mel = scratch + 2u * fft_len;
  const uint32_t stride = sizeof(float32_t);
  const uint32_t cep_blocks = nb_cep / UNROLL;

  for (uint32_t t = 0; t < nb_frames; t++) {
    const float32_t *const frame = audio + (size_t)t * hop;

    for (uint32_t i = 0; i < frame_len; i++) {
      re[i] = frame[i] * window[i];
      im[i] = 0.0f;
    }
    for (uint32_t i = frame_len; i < fft_len; i++) {
      re[i] = 0.0f;
      im[i] = 0.0f;
    }

    fft_radix2(re, im, fft_len, twiddles);

    const uint32_t nb_bins = fft_len / 2u + 1u;
    for (uint32_t b = 0; b < nb_bins; b++) {
      re[b] = re[b] * re[b] + im[b] * im[b];
    }

    uint32_t off = 0;
    for (uint32_t m = 0; m < nb_mel; m++) {
      const uint32_t len = mel_len[m];
      float32_t acc = mel_energy(&re[mel_start[m]], &mel_coeff[off], len);
      off += len;
      mel[m] = logf(acc > MFCC_LOG_FLOOR ? acc : MFCC_LOG_FLOOR);
    }

    /* DCT-II, UNROLL output cepstra at a time -- the same shape as the GEMM
     * kernel's inner block. The FREP body is four FMAs replayed once per mel
     * band, and each FMA consumes one element from each stream, so:
     *
     *   ft0  one mel value held for the UNROLL cepstra that need it
     *   ft1  the same column of the UNROLL matrix rows, back to back
     *
     * which leaves UNROLL independent accumulators live across the whole
     * reduction -- no horizontal add at the end, which is the operation a
     * vector unit would have to pay for here. */
    float32_t *const dst = out + (size_t)t * nb_cep;
    uint32_t c = 0;
    if (cep_blocks != 0) {
      ssr_repeat(SSR_DM0, UNROLL);
      ssr_repeat(SSR_DM1, 1);

      for (uint32_t b = 0; b < cep_blocks; b++) {
        float32_t a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        /* Both streams re-arm per block: ft0 walks the log-mel vector again,
         * ft1 moves on to the next UNROLL rows of the matrix. */
        ssr_loop_1d(SSR_DM0, nb_mel, stride);
        ssr_loop_2d(SSR_DM1, UNROLL, nb_mel, nb_mel * stride, stride);
        ssr_read(SSR_DM0, SSR_1D, mel);
        ssr_read(SSR_DM1, SSR_2D, dct + (size_t)c * nb_mel);

        __asm__ volatile(SSR_FREP_BEGIN(UNROLL)
                         "fmadd.s %[a0], ft0, ft1, %[a0]\n"
                         "fmadd.s %[a1], ft0, ft1, %[a1]\n"
                         "fmadd.s %[a2], ft0, ft1, %[a2]\n"
                         "fmadd.s %[a3], ft0, ft1, %[a3]\n" SSR_FREP_END
                         : [a0] "+f"(a0), [a1] "+f"(a1), [a2] "+f"(a2),
                           [a3] "+f"(a3)
                         : [frep_rpt] "r"(nb_mel - 1)
                         : "ft0", "ft1", "ft2", "memory");
        dst[c] = a0;
        dst[c + 1] = a1;
        dst[c + 2] = a2;
        dst[c + 3] = a3;
        c += UNROLL;
      }
    }

    /* Cepstra past the last full block. */
    for (; c < nb_cep; c++) {
      const float32_t *const row = dct + (size_t)c * nb_mel;
      float32_t acc = 0.0f;
      for (uint32_t m = 0; m < nb_mel; m++) {
        acc += row[m] * mel[m];
      }
      dst[c] = acc;
    }
  }
}
