/* Correctness test for runtime/snitch/kernels/mfcc_fp32_ssr.c.
 *
 * The KWS run only ever hits one shape -- 40 mel filters, 13 cepstra -- and it
 * checks the result against numpy, which is the strongest check there is but
 * also the least specific: it says the whole front-end agrees, not which of its
 * stages would break first. This runs the shapes the application never reaches
 * and compares against a scalar reference computed here:
 *
 *   nb_cep  exactly UNROLL, one block plus a remainder, and fewer than UNROLL
 *           (so the streamed DCT is skipped entirely and only the tail runs)
 *   filters shorter than UNROLL, exactly UNROLL, and not a multiple of it,
 *           including a single-bin filter -- the ragged widths are the point of
 *           streaming the filterbank, so the ragged edges are what to test
 *
 * Build: this file plus the kernel, with the Deeploy Generic includes on the
 * path; the _generic fallback the kernel calls is defined below with the same
 * scalar code the reference uses.
 */
#include "../common/bench.h"
#include "../common/miniio.h"

#include "DeeployBasicMath.h"

#include <math.h>

#include "mfcc.h"

#define MFCC_LOG_FLOOR 1e-6f

/* --- scalar reference, also the kernel's _generic fallback ---------------- */

static void ref_fft(float32_t *re, float32_t *im, uint32_t n,
                    const float32_t *tw) {
  uint32_t bits = 0;
  while ((1u << bits) < n) {
    bits++;
  }
  for (uint32_t i = 0; i < n; i++) {
    uint32_t r = 0;
    for (uint32_t b = 0; b < bits; b++) {
      r = (r << 1) | ((i >> b) & 1u);
    }
    if (r > i) {
      float32_t t = re[i];
      re[i] = re[r];
      re[r] = t;
      t = im[i];
      im[i] = im[r];
      im[r] = t;
    }
  }
  for (uint32_t len = 2; len <= n; len <<= 1) {
    uint32_t half = len >> 1, step = n / len;
    for (uint32_t j = 0; j < half; j++) {
      float32_t wr = tw[2u * j * step], wi = tw[2u * j * step + 1u];
      for (uint32_t base = 0; base < n; base += len) {
        uint32_t a = base + j, b = a + half;
        float32_t xr = re[b], xi = im[b];
        float32_t vr = xr * wr - xi * wi, vi = xr * wi + xi * wr;
        float32_t ur = re[a], ui = im[a];
        re[a] = ur + vr;
        im[a] = ui + vi;
        re[b] = ur - vr;
        im[b] = ui - vi;
      }
    }
  }
}

void Mfcc_fp32_fp32_generic(const float32_t *__restrict__ audio,
                            float32_t *__restrict__ out, uint32_t nb_frames,
                            uint32_t frame_len, uint32_t hop, uint32_t fft_len,
                            const float32_t *__restrict__ window,
                            const float32_t *__restrict__ twiddles,
                            const float32_t *__restrict__ mel_coeff,
                            const uint16_t *__restrict__ mel_start,
                            const uint16_t *__restrict__ mel_len,
                            uint32_t nb_mel,
                            const float32_t *__restrict__ dct, uint32_t nb_cep,
                            float32_t *__restrict__ scratch) {
  float32_t *re = scratch, *im = scratch + fft_len;
  float32_t *mel = scratch + 2u * fft_len;
  for (uint32_t t = 0; t < nb_frames; t++) {
    const float32_t *frame = audio + (size_t)t * hop;
    for (uint32_t i = 0; i < frame_len; i++) {
      re[i] = frame[i] * window[i];
      im[i] = 0.0f;
    }
    for (uint32_t i = frame_len; i < fft_len; i++) {
      re[i] = 0.0f;
      im[i] = 0.0f;
    }
    ref_fft(re, im, fft_len, twiddles);
    for (uint32_t b = 0; b < fft_len / 2u + 1u; b++) {
      re[b] = re[b] * re[b] + im[b] * im[b];
    }
    uint32_t off = 0;
    for (uint32_t m = 0; m < nb_mel; m++) {
      float32_t acc = 0.0f;
      for (uint32_t i = 0; i < mel_len[m]; i++) {
        acc += re[mel_start[m] + i] * mel_coeff[off + i];
      }
      off += mel_len[m];
      mel[m] = logf(acc > MFCC_LOG_FLOOR ? acc : MFCC_LOG_FLOOR);
    }
    for (uint32_t c = 0; c < nb_cep; c++) {
      float32_t acc = 0.0f;
      for (uint32_t m = 0; m < nb_mel; m++) {
        acc += dct[(size_t)c * nb_mel + m] * mel[m];
      }
      out[(size_t)t * nb_cep + c] = acc;
    }
  }
}

/* --- the fixture --------------------------------------------------------- */

#define FFT_LEN 32
#define FRAME_LEN 32
#define HOP 16
#define NB_FRAMES 3
#define NB_SAMPLES (FRAME_LEN + (NB_FRAMES - 1) * HOP)
#define NB_BINS (FFT_LEN / 2 + 1)
#define MAX_MEL 8
#define MAX_CEP 6

static float32_t audio[NB_SAMPLES];
static float32_t window[FRAME_LEN];
static float32_t twiddles[FFT_LEN];
static float32_t dct[MAX_CEP * MAX_MEL];
static float32_t scratch_a[MFCC_SCRATCH_ELEMS(FFT_LEN, MAX_MEL)];
static float32_t scratch_b[MFCC_SCRATCH_ELEMS(FFT_LEN, MAX_MEL)];
static float32_t got[NB_FRAMES * MAX_CEP];
static float32_t ref[NB_FRAMES * MAX_CEP];

/* Filter widths chosen to straddle UNROLL=4 in every direction: below it, on
 * it, above but not a multiple, and a single bin. */
static const uint16_t mel_start[MAX_MEL] = {0, 1, 3, 5, 6, 9, 12, 14};
static const uint16_t mel_len[MAX_MEL] = {1, 2, 4, 5, 3, 7, 4, 3};
static float32_t mel_coeff[64];

static uint32_t failures;

static void check(const char *what, uint32_t nb_mel, uint32_t nb_cep) {
  const uint32_t n = NB_FRAMES * nb_cep;
  Mfcc_fp32_fp32_generic(audio, ref, NB_FRAMES, FRAME_LEN, HOP, FFT_LEN, window,
                         twiddles, mel_coeff, mel_start, mel_len, nb_mel, dct,
                         nb_cep, scratch_a);
  Mfcc_fp32_fp32(audio, got, NB_FRAMES, FRAME_LEN, HOP, FFT_LEN, window,
                 twiddles, mel_coeff, mel_start, mel_len, nb_mel, dct, nb_cep,
                 scratch_b);

  float32_t worst = 0.0f;
  for (uint32_t i = 0; i < n; i++) {
    float32_t d = fabsf(got[i] - ref[i]);
    if (d > worst) {
      worst = d;
    }
  }
  /* The streamed reductions fold four partial sums at the end instead of
   * accumulating left to right, so this is a reassociation tolerance, not a
   * correctness one: a real error is orders of magnitude larger. */
  const int ok = worst <= 1e-4f;
  if (!ok) {
    failures++;
  }
  print_str(ok ? "  ok   " : "  FAIL ");
  print_str(what);
  print_str("  nb_mel=");
  print_u64(nb_mel);
  print_str(" nb_cep=");
  print_u64(nb_cep);
  print_str(" maxdiff_e6=");
  print_u64((uint64_t)(worst * 1e6f));
  print_str("\n");
}

int main(void) {
  /* A deterministic signal with content in every band. */
  for (uint32_t i = 0; i < NB_SAMPLES; i++) {
    audio[i] = 0.5f * sinf(0.31f * i) + 0.25f * sinf(1.7f * i + 0.4f)
               + 0.125f * sinf(2.9f * i);
  }
  for (uint32_t i = 0; i < FRAME_LEN; i++) {
    window[i] = 0.5f - 0.5f * cosf(2.0f * (float32_t)M_PI * i / FRAME_LEN);
  }
  for (uint32_t k = 0; k < FFT_LEN / 2; k++) {
    float32_t ang = -2.0f * (float32_t)M_PI * k / FFT_LEN;
    twiddles[2 * k] = cosf(ang);
    twiddles[2 * k + 1] = sinf(ang);
  }
  for (uint32_t i = 0; i < 64; i++) {
    mel_coeff[i] = 0.1f + 0.05f * (float32_t)(i % 7);
  }
  for (uint32_t i = 0; i < MAX_CEP * MAX_MEL; i++) {
    dct[i] = cosf(0.21f * i) * 0.4f;
  }

  print_str("[SSR-MFCC] streamed filterbank and DCT vs scalar reference\n");
  /* nb_cep on, over and under the UNROLL=4 block of the streamed DCT. */
  check("cepstra = one whole block ", MAX_MEL, 4);
  check("cepstra = block + remainder", MAX_MEL, 6);
  check("cepstra < one block        ", MAX_MEL, 3);
  /* Fewer filters, so the last ones tested above drop out and the widths the
   * filterbank sees change with them. */
  check("filters, short bank        ", 3, 4);
  check("filters, one only          ", 1, 5);

  print_str("[SSR-MFCC] failures=");
  print_u64(failures);
  print_str("\n");
  return failures ? 1 : 0;
}
