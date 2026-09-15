/* Keyword spotting on the hetero_soc board, with both clusters running at once.
 *
 * Each clip goes through two stages that want different hardware:
 *
 *   audio --> MFCC front-end --> features --> CNN classifier --> keyword
 *             one cluster                     the other cluster, plus the host
 *
 * and, because clips keep arriving, the two stages of *different* clips are
 * independent. So the front-end for clip n+1 is posted before the classifier
 * for clip n runs, and collected after:
 *
 *   post(front-end, clip n+1)   ->  cluster A starts
 *   RunNetwork(clip n)          ->  cluster B and the host work
 *   wait(front-end)             ->  usually already done
 *
 * RunNetwork() offloads its Conv and Gemm nodes with hes_offload(), which posts
 * and waits on *its* cluster's mailbox -- a different mailbox from the one the
 * front-end is using. So the overlap needs nothing from the generated code: the
 * two clusters simply have jobs in flight at the same time. KWS_PIPELINED=0
 * moves the front-end back after the classifier and computes the same features
 * with no overlap, which is the baseline the pipelined run is measured against.
 *
 * Scoring follows runtime/mesh/mnist_main.c -- prediction against the true
 * label, and against onnxruntime on the same graph, the second being the one
 * that can fail the run -- plus one check MNIST has no equivalent of: the
 * features themselves are compared against the numpy front-end in
 * ops/kws/kws_data.h. Without that a wrong FFT would only ever show up as a
 * misclassification, which is far too late to be useful.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "Network.h"

#include "bench.h"
#include "hes_host.h"
#include "kws_data.h"
#include "miniio.h"

/* Cap the run, so a quick check does not have to do all of them. */
#ifndef HES_SAMPLES
#define HES_SAMPLES KWS_NUM_CLIPS
#endif

#if HES_SAMPLES < KWS_NUM_CLIPS
#define RUN_CLIPS HES_SAMPLES
#else
#define RUN_CLIPS KWS_NUM_CLIPS
#endif

/* Which cluster runs the front-end. The classifier's nodes go wherever the
 * Deeploy cost model put them, which is the other cluster and the host. Making
 * this a build-time choice is what lets the two assignments be measured against
 * each other: the pipeline's period is max(front-end, classifier), so the best
 * placement is not always the one that runs each stage fastest. */
#ifndef KWS_FRONTEND_ENGINE
#define KWS_FRONTEND_ENGINE HES_ENGINE_SNITCH
#endif

/* 1: overlap the two stages. 0: the serial baseline, same work, no overlap. */
#ifndef KWS_PIPELINED
#define KWS_PIPELINED 1
#endif

/* Double-buffered, so the cluster can be filling clip n+1's features while the
 * host is reading clip n's. */
static float features[2][KWS_FEAT_ELEMS];

static uint32_t argmax(const float *p, uint32_t n) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < n; i++) {
    if (p[i] > p[best]) {
      best = i;
    }
  }
  return best;
}

/* Everything about an MFCC job that does not change from clip to clip. */
static hes_mfcc_job_t make_job(void) {
  hes_mfcc_job_t job;
  job.audio = 0;
  job.out = 0;
  job.nb_frames = KWS_NB_FRAMES;
  job.frame_len = KWS_FRAME_LEN;
  job.hop = KWS_HOP;
  job.fft_len = KWS_FFT_LEN;
  job.window = kws_window;
  job.twiddles = kws_twiddles;
  job.mel_coeff = kws_mel_coeff;
  job.mel_start = kws_mel_start;
  job.mel_len = kws_mel_len;
  job.nb_mel = KWS_NB_MEL;
  job.dct = kws_dct;
  job.nb_cep = KWS_NB_CEP;
  job.mel_coeffs = KWS_MEL_COEFFS;
  return job;
}

int main(void) {
  InitNetwork(0, 1);
  hes_engine_init();

  for (uint32_t e = HES_ENGINE_SNITCH; e < HES_NB_ENGINES; e++) {
    if (!hes_engine_ready(e)) {
      print_str("[HES-ERR] cluster ");
      print_str(hes_engine_name(e));
      print_str(" never signed in\n");
    }
  }

  hes_mfcc_job_t job = make_job();
  const uint32_t fe = KWS_FRONTEND_ENGINE;

  uint32_t correct = 0, agree = 0;
  float maxdiff = 0.0f;
  uint64_t fe_wait = 0;

  uint64_t c0 = read_mcycle();

  /* Prime the pipeline: clip 0's features have to exist before its classifier
   * can run, so this one front-end call is never overlapped with anything.
   * It is timed into fe_wait like every other collection, so that `hidden`
   * below stays the difference between what the front-end cost and what the
   * host actually waited for -- and therefore comes out at zero, rather than
   * one job's worth, in the serial build. */
  {
    uint64_t w0 = read_mcycle();
    job.audio = kws_audio[0];
    job.out = features[0];
    hes_offload_mfcc(fe, &job);
    fe_wait += read_mcycle() - w0;
  }

  for (uint32_t n = 0; n < RUN_CLIPS; n++) {
    hes_set_sample(n + 1, RUN_CLIPS);

#if KWS_PIPELINED
    /* Start the next clip's front-end before this clip's classifier, so the
     * two clusters are busy at the same time. */
    if (n + 1 < RUN_CLIPS) {
      job.audio = kws_audio[n + 1];
      job.out = features[(n + 1) & 1];
      hes_post_mfcc(fe, &job);
    }
#endif

    const float *feat = features[n & 1];
    memcpy(DeeployNetwork_inputs[0], feat, DeeployNetwork_inputs_bytes[0]);
    RunNetwork(0, 1);

    /* The front-end is checked on its own, against the numpy reference for the
     * same clip, so a wrong spectrum is a front-end failure rather than a
     * mysterious misclassification. */
    for (uint32_t i = 0; i < KWS_FEAT_ELEMS; i++) {
      float d = fabsf(feat[i] - kws_mfcc_ref[n][i]);
      if (d > maxdiff) {
        maxdiff = d;
      }
    }

    uint32_t got = argmax((const float *)DeeployNetwork_outputs[0],
                          KWS_NUM_CLASSES);
    if (got == kws_labels[n]) {
      correct++;
    }
    if (got == kws_reference[n]) {
      agree++;
    } else {
      print_str("[HES-ERR] clip ");
      print_u64(n);
      print_str(": chip says ");
      print_u64(got);
      print_str(", onnxruntime says ");
      print_u64(kws_reference[n]);
      print_str("\n");
    }

    if (n + 1 < RUN_CLIPS) {
#if KWS_PIPELINED
      /* Whatever is left of the front-end after the classifier finished. The
       * gap between this and the cluster's own busy count is the work the
       * overlap hid. */
      uint64_t w0 = read_mcycle();
      hes_check(fe, "Mfcc", hes_wait(fe));
      fe_wait += read_mcycle() - w0;
#else
      uint64_t w0 = read_mcycle();
      job.audio = kws_audio[n + 1];
      job.out = features[(n + 1) & 1];
      hes_offload_mfcc(fe, &job);
      fe_wait += read_mcycle() - w0;
#endif
    }
  }

  uint64_t total = read_mcycle() - c0;

  const uint32_t snitch_busy = hes_engine_busy(HES_ENGINE_SNITCH);
  const uint32_t spatz_busy = hes_engine_busy(HES_ENGINE_SPATZ);
  const uint32_t fe_busy = hes_engine_busy(fe);
  /* Front-end cycles that cost the host nothing because they happened while
   * the classifier was running. Zero by construction in the serial build. */
  const uint64_t hidden = (uint64_t)fe_busy > fe_wait ? fe_busy - fe_wait : 0;

  print_str("[HES-KWS] clips=");
  print_u64(RUN_CLIPS);
  print_str(" correct=");
  print_u64(correct);
  print_str(" agree_with_onnx=");
  print_u64(agree);
  print_str(" mfcc_maxdiff_e6=");
  print_u64((uint64_t)(maxdiff * 1e6f));
  print_str(" cycles_total=");
  print_u64(total);
  print_str(" cycles_per_clip=");
  print_u64(total / (RUN_CLIPS ? RUN_CLIPS : 1));
  print_str(" frontend_engine=");
  print_u64(fe);
  print_str(" frontend_busy=");
  print_u64(fe_busy);
  print_str(" frontend_wait=");
  print_u64(fe_wait);
  print_str(" hidden=");
  print_u64(hidden);
  print_str(" snitch_busy=");
  print_u64(snitch_busy);
  print_str(" spatz_busy=");
  print_u64(spatz_busy);
  print_str(" pipelined=");
  print_u64(KWS_PIPELINED);
  print_str(" offload_failures=");
  print_u64(hes_failures());
  print_str("\n");

  /* The run fails if the chip disagreed with the reference model, if the
   * features drifted, or if an offload failed -- not if the network got a
   * keyword wrong. */
  return (agree == RUN_CLIPS && hes_failures() == 0 && maxdiff <= 1e-3f) ? 0 : 1;
}
