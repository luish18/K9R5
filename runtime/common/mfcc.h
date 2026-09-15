/* The MFCC front-end: the KWS application's first stage.
 *
 * One call turns a run of audio samples into `nb_frames` cepstral vectors:
 *
 *   frame -> window -> radix-2 FFT -> |.|^2 -> mel filterbank -> log -> DCT-II
 *
 * It exists as a kernel rather than as host code because it is the half of the
 * application that suits a cluster with stream semantic registers: the mel
 * filterbank is 40 reductions over ragged 10-30 bin runs, which is short-vector
 * work with a horizontal reduction per filter, and the FFT is an affine walk
 * whose inner extent halves every stage. pipeline/kws.py holds the numpy
 * reference these have to agree with, and emits the constant tables both use.
 *
 * Implementations, selected the same way every other kernel in this tree is --
 * the link-time rename in pipeline/build_mesh.py:
 *
 *   runtime/common/kernels/mfcc_fp32.c      portable C; what the host and the
 *                                           Spatz cluster run (autovectorized
 *                                           there, as Conv2d already is)
 *   runtime/snitch/kernels/mfcc_fp32_ssr.c  the Xssr/Xfrep version, which takes
 *                                           the name on the Snitch cluster and
 *                                           leaves the portable one reachable
 *                                           as Mfcc_fp32_fp32_generic
 *
 * Frames are independent, so a cluster slices this by frame: a core's share is
 * a pointer offset into `audio` and `out` plus a smaller `nb_frames`, and no
 * two cores write the same element.
 */
#ifndef HES_MFCC_H
#define HES_MFCC_H

#include <stdint.h>

#include "DeeployBasicMath.h"

/* Scratch a single frame needs, in floats: the complex spectrum (real and
 * imaginary, split rather than interleaved so every stream is unit-stride)
 * followed by the mel energies. Each core needs its own. */
#define MFCC_SCRATCH_ELEMS(fft_len, nb_mel) (2u * (fft_len) + (nb_mel))

/* out[t][c], t in [0, nb_frames), c in [0, nb_cep): frame-major, so each
 * frame's cepstra are contiguous.
 *
 *   audio       nb_samples >= (nb_frames - 1) * hop + frame_len
 *   window      frame_len
 *   twiddles    fft_len, as cos/sin of -2*pi*k/fft_len interleaved, k < fft_len/2
 *   mel_coeff   the triangular filters, banded: filter m covers bins
 *               [mel_start[m], mel_start[m] + mel_len[m]) and its coefficients
 *               begin at the running offset sum(mel_len[0..m-1])
 *   dct         nb_cep x nb_mel, row-major, orthonormal DCT-II
 *   scratch     MFCC_SCRATCH_ELEMS(fft_len, nb_mel) floats, this core's own
 *
 * fft_len must be a power of two and >= frame_len.
 */
void Mfcc_fp32_fp32(const float32_t *__restrict__ audio,
                    float32_t *__restrict__ out, uint32_t nb_frames,
                    uint32_t frame_len, uint32_t hop, uint32_t fft_len,
                    const float32_t *__restrict__ window,
                    const float32_t *__restrict__ twiddles,
                    const float32_t *__restrict__ mel_coeff,
                    const uint16_t *__restrict__ mel_start,
                    const uint16_t *__restrict__ mel_len, uint32_t nb_mel,
                    const float32_t *__restrict__ dct, uint32_t nb_cep,
                    float32_t *__restrict__ scratch);

/* The portable implementation, always reachable under this name so a streamed
 * version can fall back to it for shapes it does not cover. */
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
                            float32_t *__restrict__ scratch);

#endif /* HES_MFCC_H */
