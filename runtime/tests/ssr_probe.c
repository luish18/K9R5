/* Probe: does the Snitch model stream through SSR and repeat through FREP?
 *
 * Runs the same dot product three ways — plain scalar, four unrolled scalar
 * accumulators, and SSR+FREP — and prints the result and cycle count of each.
 * A correct run has all three results equal and the SSR one ~1 cycle per FMA.
 */
#include "../common/bench.h"
#include "../common/miniio.h"
#include "../snitch/snitch_ssr.h"

#define N 128

static float a[N], b[N];

static float dot_scalar(const float *x, const float *y, uint32_t n) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < n; i++)
    sum += x[i] * y[i];
  return sum;
}

/* Four independent accumulators: same work, no dependency chain. */
static float dot_unrolled(const float *x, const float *y, uint32_t n) {
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  for (uint32_t i = 0; i < n; i += 4) {
    c0 += x[i + 0] * y[i + 0];
    c1 += x[i + 1] * y[i + 1];
    c2 += x[i + 2] * y[i + 2];
    c3 += x[i + 3] * y[i + 3];
  }
  return (c0 + c1) + (c2 + c3);
}

/* Same four accumulators, but the operands arrive by themselves and the
 * sequencer issues the body: the loop costs zero integer instructions. */
static float dot_ssr(const float *x, const float *y, uint32_t n) {
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

  ssr_loop_1d(SSR_DM0, n, sizeof(float));
  ssr_loop_1d(SSR_DM1, n, sizeof(float));
  ssr_read(SSR_DM0, SSR_1D, x);
  ssr_read(SSR_DM1, SSR_1D, y);

  __asm__ volatile(SSR_FREP_BEGIN(4)
                   "fmadd.s %[c0], ft0, ft1, %[c0]\n"
                   "fmadd.s %[c1], ft0, ft1, %[c1]\n"
                   "fmadd.s %[c2], ft0, ft1, %[c2]\n"
                   "fmadd.s %[c3], ft0, ft1, %[c3]\n" SSR_FREP_END
                   : [c0] "+f"(c0), [c1] "+f"(c1), [c2] "+f"(c2),
                     [c3] "+f"(c3)
                   : [frep_rpt] "r"(n / 4 - 1)
                   : "ft0", "ft1", "ft2", "memory");

  return (c0 + c1) + (c2 + c3);
}

static void report(const char *name, float value, uint64_t cycles) {
  print_str(name);
  print_str(": value*100 = ");
  print_i64((int64_t)(value * 100.0f));
  print_str("  cycles = ");
  print_u64(cycles);
  print_str("\n");
}

int main(void) {
  for (uint32_t i = 0; i < N; i++) {
    a[i] = (float)(i % 7) * 0.5f;
    b[i] = (float)(i % 5) - 2.0f;
  }

  uint64_t t0 = read_mcycle();
  float ref = dot_scalar(a, b, N);
  uint64_t t1 = read_mcycle();
  float unr = dot_unrolled(a, b, N);
  uint64_t t2 = read_mcycle();
  float ssr = dot_ssr(a, b, N);
  uint64_t t3 = read_mcycle();

  report("scalar  ", ref, t1 - t0);
  report("unrolled", unr, t2 - t1);
  report("ssr+frep", ssr, t3 - t2);

  float diff = ssr - ref;
  if (diff < 0)
    diff = -diff;
  int ok = diff < 1e-3f;
  print_str(ok ? "ssr_probe: OK\n" : "ssr_probe: MISMATCH\n");
  return ok ? 0 : 1;
}
