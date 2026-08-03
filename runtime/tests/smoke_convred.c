/* Reproduce conv's strip-mined vfmacc + vfredusum + vfmv.f.s pattern. */
#include "../common/bench.h"
#include "../common/miniio.h"

#define MAXN 40
static float a[MAXN] __attribute__((aligned(8)));
static float b[MAXN] __attribute__((aligned(8)));

static float dotv(const float *x, const float *y, int n) {
  float out;
  __asm__ volatile(
      "vsetvli t0, zero, e32, m1, ta, ma\n\t"
      "vmv.v.i v1, 0\n\t"
      "1:\n\t"
      "vsetvli t0, %[n], e32, m1, tu, ma\n\t"
      "vle32.v v2, (%[x])\n\t"
      "vle32.v v3, (%[y])\n\t"
      "vfmacc.vv v1, v3, v2\n\t"
      "slli t1, t0, 2\n\t"
      "add %[x], %[x], t1\n\t"
      "add %[y], %[y], t1\n\t"
      "sub %[n], %[n], t0\n\t"
      "bnez %[n], 1b\n\t"
      "vsetvli t0, zero, e32, m1, ta, ma\n\t"
      "vmv.s.x v2, zero\n\t"
      "vfredusum.vs v1, v1, v2\n\t"
      "vfmv.f.s %[out], v1\n\t"
      : [out] "=f"(out), [x] "+r"(x), [y] "+r"(y), [n] "+r"(n)
      :
      : "t0", "t1", "memory");
  return out;
}

int main(void) {
  for (int i = 0; i < MAXN; i++) {
    a[i] = (float)(i % 7) - 3.0f;
    b[i] = (float)(i % 5) - 2.0f;
  }

  int sizes[] = {1, 3, 5, 9, 16, 17, 20, 33, 40};
  int errs = 0;
  for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    int n = sizes[s];
    float ref = 0.0f;
    for (int i = 0; i < n; i++)
      ref += a[i] * b[i];
    float got = dotv(a, b, n);
    if (ref != got) {
      errs++;
      print_str("n=");
      print_i64(n);
      print_str(" ref*100=");
      print_i64((int64_t)(ref * 100));
      print_str(" got*100=");
      print_i64((int64_t)(got * 100));
      print_str("\n");
    }
  }
  print_str(errs ? "convred: FAIL\n" : "convred: PASS\n");
  return errs ? 1 : 0;
}
