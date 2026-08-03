/* Benchmark harness for Deeploy-generated single-op networks.
 * Runs the op once (cold), then once timed, verifies against the
 * expected outputs, and prints machine-parsable metrics.
 */
#include <stdint.h>
#include <string.h>

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

#include "bench.h"
#include "miniio.h"

#ifndef CORE_NAME
#define CORE_NAME "unknown"
#endif

/* Tolerance for float outputs: |diff| <= 1e-4 (matches DeeployTest). */
static int float_ok(float diff) { return diff > -1e-4f && diff < 1e-4f; }

int main(void) {
  InitNetwork(0, 1);

  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }

  /* Timed run. */
  uint64_t c0 = read_mcycle();
  uint64_t i0 = read_minstret();
  RunNetwork(0, 1);
  uint64_t c1 = read_mcycle();
  uint64_t i1 = read_minstret();

  /* Verification. */
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

  print_str("[HES] core=" CORE_NAME);
  print_str(" cycles=");
  print_u64(c1 - c0);
  print_str(" instret=");
  print_u64(i1 - i0);
  print_str(" errors=");
  print_u64(errors);
  print_str(" total=");
  print_u64(tot);
  print_str(" maxdiff_e6=");
  print_u64((uint64_t)(maxdiff * 1e6f));
  print_str("\n");

  return errors == 0 ? 0 : 1;
}
