/* The portable MFCC front-end.
 *
 * This is the reference implementation of runtime/common/mfcc.h: plain C, no
 * vendor extensions, written so the autovectorizer has something to work with
 * on the cores that have a vector unit. On the Snitch cluster the link-time
 * rename in pipeline/build_mesh.py turns it into Mfcc_fp32_fp32_generic and
 * runtime/snitch/kernels/mfcc_fp32_ssr.c takes the name -- the same
 * arrangement Conv2d and GEMM already use.
 *
 * It has to agree numerically with the numpy reference in pipeline/kws.py,
 * which is checked per clip by runtime/mesh/kws_main.c against the features
 * embedded in ops/kws/kws_data.h. The pieces where that agreement is easy to
 * lose are the log floor below and the FFT's accumulation order, so both are
 * spelled out rather than left to a library.
 */

#include <math.h>

#include "mfcc.h"

/* Mel energies are floored before the log, which is what stops a near-silent
 * band turning a tiny absolute error into a large relative one. It must match
 * MEL_FLOOR in pipeline/kws.py; a disagreement shows up immediately as a
 * feature mismatch in the KWS run, not as a subtly worse accuracy. */
#define MFCC_LOG_FLOOR 1e-6f

/* Reverse the low `bits` bits of `i`. Only ever called fft_len times per
 * frame, so a loop is cheaper than the table it would fill. */
static inline uint32_t bit_reverse(uint32_t i, uint32_t bits) {
  uint32_t r = 0;
  for (uint32_t b = 0; b < bits; b++) {
    r = (r << 1) | ((i >> b) & 1u);
  }
  return r;
}

/* In-place radix-2 decimation-in-time FFT over split real/imaginary arrays.
 *
 * Split rather than interleaved so every access in the butterfly is
 * unit-stride within its half -- which is what both a vector unit and a stream
 * semantic register want, and what an interleaved layout would cost a stride
 * of two.
 */
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
      /* The twiddle is fixed for the whole inner loop, so it is loaded once
       * and the loop over blocks is a straight pass over both halves. */
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
  float32_t *const re = scratch;
  float32_t *const im = scratch + fft_len;
  float32_t *const mel = scratch + 2u * fft_len;

  for (uint32_t t = 0; t < nb_frames; t++) {
    const float32_t *const frame = audio + (size_t)t * hop;

    for (uint32_t i = 0; i < frame_len; i++) {
      re[i] = frame[i] * window[i];
      im[i] = 0.0f;
    }
    /* Zero-pad when the FFT is longer than the frame. */
    for (uint32_t i = frame_len; i < fft_len; i++) {
      re[i] = 0.0f;
      im[i] = 0.0f;
    }

    fft_radix2(re, im, fft_len, twiddles);

    /* Power spectrum, over the non-redundant half, written back over `re`:
     * the imaginary part is dead from here on. */
    const uint32_t nb_bins = fft_len / 2u + 1u;
    for (uint32_t b = 0; b < nb_bins; b++) {
      re[b] = re[b] * re[b] + im[b] * im[b];
    }

    /* The mel filterbank. Each filter is a reduction over its own short,
     * ragged run of bins -- the shape this kernel exists to put on a cluster
     * that can stream it. */
    uint32_t off = 0;
    for (uint32_t m = 0; m < nb_mel; m++) {
      const uint32_t start = mel_start[m];
      const uint32_t len = mel_len[m];
      float32_t acc = 0.0f;
      for (uint32_t i = 0; i < len; i++) {
        acc += re[start + i] * mel_coeff[off + i];
      }
      off += len;
      mel[m] = logf(acc > MFCC_LOG_FLOOR ? acc : MFCC_LOG_FLOOR);
    }

    /* DCT-II onto nb_cep cepstra: a small dense matrix-vector product. */
    float32_t *const dst = out + (size_t)t * nb_cep;
    for (uint32_t c = 0; c < nb_cep; c++) {
      const float32_t *const row = dct + (size_t)c * nb_mel;
      float32_t acc = 0.0f;
      for (uint32_t m = 0; m < nb_mel; m++) {
        acc += row[m] * mel[m];
      }
      dst[c] = acc;
    }
  }
}
