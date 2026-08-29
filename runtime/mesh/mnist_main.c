/* Classify MNIST digits on the hetero_soc board.
 *
 * The same network runtime/mesh/host_main.c runs, but over many images and
 * scored rather than diffed: for each image the host runs the Deeploy-generated
 * graph -- whose nodes dispatch to whichever engine the mapper chose -- takes
 * the argmax, and checks it two ways.
 *
 *   against mnist_labels     the ground truth, which gives an accuracy
 *   against mnist_reference  what onnxruntime predicts for the same image on
 *                            the same graph
 *
 * The second is the one that can fail: the simulated chip should agree with the
 * reference model on every image. A disagreement means the chip computed
 * something different, and no amount of accuracy makes that acceptable.
 * Accuracy only says whether the network was worth training.
 */
#include <stdint.h>
#include <string.h>

#include "Network.h"

#include "bench.h"
#include "hes_host.h"
#include "miniio.h"
#include "mnist_data.h"

/* Cap the run, so a quick check does not have to do all of them. */
#ifndef HES_SAMPLES
#define HES_SAMPLES MNIST_NUM_IMAGES
#endif

#if HES_SAMPLES < MNIST_NUM_IMAGES
#define RUN_IMAGES HES_SAMPLES
#else
#define RUN_IMAGES MNIST_NUM_IMAGES
#endif

static uint32_t argmax10(const float *p) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < 10; i++) {
    if (p[i] > p[best]) {
      best = i;
    }
  }
  return best;
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

  uint32_t correct = 0, agree = 0;
  uint64_t total_cycles = 0;

  for (uint32_t img = 0; img < RUN_IMAGES; img++) {
    hes_set_sample(img + 1, RUN_IMAGES);

    memcpy(DeeployNetwork_inputs[0], mnist_images[img],
           DeeployNetwork_inputs_bytes[0]);

    uint64_t c0 = read_mcycle();
    RunNetwork(0, 1);
    uint64_t c1 = read_mcycle();
    total_cycles += c1 - c0;

    uint32_t got = argmax10((const float *)DeeployNetwork_outputs[0]);
    if (got == mnist_labels[img]) {
      correct++;
    }
    if (got == mnist_reference[img]) {
      agree++;
    } else {
      print_str("[HES-ERR] image ");
      print_u64(img);
      print_str(": chip says ");
      print_u64(got);
      print_str(", onnxruntime says ");
      print_u64(mnist_reference[img]);
      print_str("\n");
    }
  }

  print_str("[HES-MNIST] images=");
  print_u64(RUN_IMAGES);
  print_str(" correct=");
  print_u64(correct);
  print_str(" agree_with_onnx=");
  print_u64(agree);
  print_str(" cycles_total=");
  print_u64(total_cycles);
  print_str(" cycles_per_image=");
  print_u64(total_cycles / (RUN_IMAGES ? RUN_IMAGES : 1));
  print_str(" offload_failures=");
  print_u64(hes_failures());
  print_str("\n");

  /* The run is a failure if the chip disagreed with the reference model or an
   * offload failed -- not if the network mispredicted a digit. */
  return (agree == RUN_IMAGES && hes_failures() == 0) ? 0 : 1;
}
