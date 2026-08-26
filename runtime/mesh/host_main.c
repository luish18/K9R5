/* Host program for a hetero_soc network run.
 *
 * The same shape as runtime/common/bench_main.c -- run the Deeploy-generated
 * network once, check it against the ONNX reference, print a machine-readable
 * metrics line -- with the differences the SoC brings:
 *
 *   * it waits for both clusters to sign in before dispatching anything;
 *   * the generated network calls hes_offload_* for the nodes the mapper gave
 *     to a cluster, so a failed offload has to be reported rather than left to
 *     show up as a wrong number;
 *   * every node prints a progress beacon as it completes, so the run can be
 *     followed while it happens.
 */
#include <stdint.h>
#include <string.h>

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

#include "bench.h"
#include "hes_host.h"
#include "miniio.h"

#ifndef HES_SAMPLES
#define HES_SAMPLES 1
#endif

/* Tolerance for float outputs: |diff| <= 1e-4 (matches DeeployTest). */
static int float_ok(float diff) { return diff > -1e-4f && diff < 1e-4f; }

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

  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }

  hes_set_sample(1, HES_SAMPLES);

  uint64_t c0 = read_mcycle();
  uint64_t i0 = read_minstret();
  RunNetwork(0, 1);
  uint64_t c1 = read_mcycle();
  uint64_t i1 = read_minstret();

  uint32_t tot = 0, errors = 0;
  float maxdiff = 0.0f;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    uint32_t n = DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);
    tot += n;
    for (uint32_t i = 0; i < n; i++) {
      OUTPUTTYPE expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
      OUTPUTTYPE actual = ((OUTPUTTYPE *)DeeployNetwork_outputs[buf])[i];
      float diff = (float)(expected - actual);
#if ISOUTPUTFLOAT == 1
      if (!float_ok(diff))
        errors++;
#else
      if (diff != 0.0f)
        errors++;
#endif
      float ad = diff < 0 ? -diff : diff;
      if (ad > maxdiff)
        maxdiff = ad;
    }
  }

  print_str("[HES] core=hetero_soc cycles=");
  print_u64(c1 - c0);
  print_str(" instret=");
  print_u64(i1 - i0);
  print_str(" errors=");
  print_u64(errors);
  print_str(" total=");
  print_u64(tot);
  print_str(" maxdiff_e6=");
  print_u64((uint64_t)(maxdiff * 1e6f));
  print_str(" offload_failures=");
  print_u64(hes_failures());
  print_str("\n");

  return (errors == 0 && hes_failures() == 0) ? 0 : 1;
}
