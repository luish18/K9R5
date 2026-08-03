/* Boot smoke test: prints hello, times a small FP loop, exits cleanly. */
#include "../common/bench.h"
#include "../common/miniio.h"

#define N 256

static float a[N], b[N];

int main(void) {
  print_str("smoke: hello from hart 0\n");

  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = 2.0f;
  }

  uint64_t c0 = read_mcycle();
  uint64_t i0 = read_minstret();

  volatile float acc = 0.0f;
  for (int i = 0; i < N; i++)
    acc += a[i] * b[i];

  uint64_t c1 = read_mcycle();
  uint64_t i1 = read_minstret();

  print_str("smoke: acc*10 = ");
  print_i64((int64_t)(acc * 10.0f)); /* expect 652800 */
  print_str("\nsmoke: cycles = ");
  print_u64(c1 - c0);
  print_str("\nsmoke: instrs = ");
  print_u64(i1 - i0);
  print_str("\n");

  return acc == 65280.0f ? 0 : 1;
}
