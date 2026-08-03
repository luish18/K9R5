/* Vector smoke test: elementwise a*b+c, autovectorized to RVV by GCC. */
#include "../common/bench.h"
#include "../common/miniio.h"

#define N 256

static float a[N], b[N], c[N];

__attribute__((noinline, optimize("O3,tree-vectorize"))) void fma_vec(float *restrict x, const float *restrict y,
                                       const float *restrict z, int n) {
  for (int i = 0; i < n; i++)
    x[i] = y[i] * z[i] + x[i];
}

int main(void) {
  print_str("smoke_vec: hello from hart 0\n");

  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = 2.0f;
    c[i] = 1.0f;
  }

  uint64_t c0 = read_mcycle();
  fma_vec(c, a, b, N); /* c[i] = 2*i + 1 */
  uint64_t c1 = read_mcycle();

  int err = 0;
  for (int i = 0; i < N; i++)
    if (c[i] != 2.0f * i + 1.0f)
      err++;

  for (int i = 0; i < 4; i++) {
    print_str("c[i]*10 = ");
    print_i64((int64_t)(c[i] * 10.0f));
    print_str("\n");
  }
  print_str("smoke_vec: errors = ");
  print_u64(err);
  print_str("\nsmoke_vec: cycles = ");
  print_u64(c1 - c0);
  print_str("\n");

  return err == 0 ? 0 : 1;
}
