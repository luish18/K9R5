/* Deeploy's integer GEMM on the CVA6 + Ara host, vectorized against scalar.
 *
 * ara_probe checks the vector instructions one at a time and in the short
 * sequences GCC strings them into; this checks a whole kernel. The Makefile
 * builds Deeploy's Gemm_s8.c twice -- as Gemm_s8_vec with the flags the
 * pipeline gives the host's kernels (-O3 -ffast-math, `v` in the march) and as
 * Gemm_s8_ref with no vector extension at all -- and this runs both on the
 * shape the Integer GEMM test deploys (32 x 32 x 32, transB = 1), on the other
 * transpose combinations, and on an odd shape with offsets, reporting the first
 * output element that differs.
 */
#include <stdint.h>

#include "../common/bench.h"
#include "../common/miniio.h"

typedef void gemm_s8_fn(int8_t const *A, int8_t const *B, int32_t const *C,
                        int32_t *Y, uint32_t M, uint32_t N, uint32_t P,
                        int32_t alpha, int32_t beta, int32_t transA,
                        int32_t transB, int32_t A_offset, int32_t B_offset,
                        int32_t C_offset, int32_t Y_offset);
gemm_s8_fn Gemm_s8_vec, Gemm_s8_ref;

#define MAX 1024

static int8_t A[MAX], B[MAX];
static int32_t C[MAX], Yv[MAX + 1], Yr[MAX + 1];
static int failures = 0;

static uint32_t lcg = 12345;
static uint32_t next(void) {
  lcg = lcg * 1103515245u + 12345u;
  return lcg >> 8;
}

typedef struct {
  const char *name;
  uint32_t M, N, P;
  int32_t alpha, beta, transA, transB;
  int32_t A_offset, B_offset, C_offset, Y_offset;
} gemm_case_t;

static void run(const gemm_case_t *c) {
  for (uint32_t i = 0; i < MAX; i++) {
    A[i] = (int8_t)next();
    B[i] = (int8_t)next();
    C[i] = (int32_t)(next() % 2001) - 1000;
  }
  /* One element past the output as well, which neither call may touch. */
  for (uint32_t i = 0; i <= MAX; i++)
    Yv[i] = Yr[i] = 0x5a5a5a5a;

  print_str("[ARA-KERNELS] run  case=");
  print_str(c->name);
  print_str("\n");

  uint64_t t0 = read_mcycle();
  Gemm_s8_vec(A, B, C, Yv, c->M, c->N, c->P, c->alpha, c->beta, c->transA,
              c->transB, c->A_offset, c->B_offset, c->C_offset, c->Y_offset);
  uint64_t cycles = read_mcycle() - t0;
  Gemm_s8_ref(A, B, C, Yr, c->M, c->N, c->P, c->alpha, c->beta, c->transA,
              c->transB, c->A_offset, c->B_offset, c->C_offset, c->Y_offset);

  uint32_t n = c->M * c->P, bad = 0, first = 0;
  for (uint32_t i = 0; i <= n; i++) {
    if (Yv[i] != Yr[i]) {
      if (!bad)
        first = i;
      bad++;
    }
  }

  failures += bad != 0;
  if (bad) {
    print_u64(bad);
    print_str(" of ");
    print_u64(n);
    print_str(" elements differ; first at ");
    if (first == n) {
      print_str("the element past the output");
    } else {
      print_str("row ");
      print_u64(first / c->P);
      print_str(" col ");
      print_u64(first % c->P);
    }
    print_str(": got ");
    print_i64(Yv[first]);
    print_str(" want ");
    print_i64(Yr[first]);
    print_str("\n");
  }
  print_str(bad ? "[ARA-KERNELS] FAIL case=" : "[ARA-KERNELS] ok   case=");
  print_str(c->name);
  print_str(" cycles=");
  print_u64(cycles);
  print_str("\n");
}

int main(void) {
  static const gemm_case_t cases[] = {
      /* name                  M   N   P  alpha beta tA tB  offsets A, B, C, Y */
      {"gemm_s8_deployed",    32, 32, 32, 1, 1, 0, 1, 0, 0, 0, 0},
      {"gemm_s8_t00",         32, 32, 32, 1, 1, 0, 0, 0, 0, 0, 0},
      {"gemm_s8_t10",         32, 32, 32, 1, 1, 1, 0, 0, 0, 0, 0},
      {"gemm_s8_t11",         32, 32, 32, 1, 1, 1, 1, 0, 0, 0, 0},
      {"gemm_s8_odd_offsets",  7, 33, 13, 2, 3, 0, 1, 3, -5, 7, 11},
  };
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
    run(&cases[i]);

  print_str("[ARA-KERNELS] failures=");
  print_u64(failures);
  print_str("\n");
  return failures != 0;
}
