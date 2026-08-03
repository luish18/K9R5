/* Int8 dot-product smoke test mirroring Deeploy's s8 GEMM inner loop. */
#include "../common/bench.h"
#include "../common/miniio.h"

#define N 64

static int8_t a[N], b[N];

__attribute__((noinline, optimize("O3,tree-vectorize"))) int32_t
dot8(const int8_t *restrict x, const int8_t *restrict y, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; i++)
    sum += (int32_t)x[i] * (int32_t)y[i];
  return sum;
}

int main(void) {
  int32_t ref = 0;
  for (int i = 0; i < N; i++) {
    a[i] = (int8_t)((i * 37 + 11) % 251 - 125);
    b[i] = (int8_t)((i * 53 + 7) % 249 - 124);
    ref += (int32_t)a[i] * (int32_t)b[i];
  }

  int32_t got = dot8(a, b, N);

  print_str("dot8: ref = ");
  print_i64(ref);
  print_str("\ndot8: got = ");
  print_i64(got);
  print_str("\n");

  return ref == got ? 0 : 1;
}
