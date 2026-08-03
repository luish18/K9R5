/* Probe exact semantics of the vwmul/vmv1r/vwadd.wv sequence GCC emits
 * for int8 dot products, one 16-element strip, dumping intermediates. */
#include "../common/bench.h"
#include "../common/miniio.h"

static int8_t a[16], b[16];
static int32_t acc[16], prod16_as32[16];
static int16_t prod[16];

int main(void) {
  for (int i = 0; i < 16; i++) {
    a[i] = (int8_t)(i + 1);         /* 1..16 */
    b[i] = (int8_t)(-2 * i + 3);    /* 3,1,-1,... */
    acc[i] = 1000 + i;              /* nonzero accumulator */
  }

  __asm__ volatile(
      /* v1 = acc (e32,m1, vl=16) */
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vle32.v v1, (%[acc])\n\t"
      /* loads at e8,mf4 */
      "vsetvli t0, %[n], e8, mf4, ta, ma\n\t"
      "vle8.v v3, (%[a])\n\t"
      "vle8.v v4, (%[b])\n\t"
      /* products: e16 */
      "vwmul.vv v2, v4, v3\n\t"
      /* save products */
      "vsetvli t0, %[n], e16, mf2, tu, ma\n\t"
      "vse16.v v2, (%[prod])\n\t"
      /* accumulator copy + widening add, exactly as GCC emits */
      "vmv1r.v v5, v1\n\t"
      "vwadd.wv v1, v5, v2\n\t"
      /* store result at e32 */
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vse32.v v1, (%[out])\n\t"
      :
      : [n] "r"(16), [a] "r"(a), [b] "r"(b), [acc] "r"(acc),
        [prod] "r"(prod), [out] "r"(prod16_as32)
      : "t0", "memory");

  int errs = 0;
  for (int i = 0; i < 16; i++) {
    int32_t p = (int32_t)a[i] * (int32_t)b[i];
    if (prod[i] != (int16_t)p) {
      errs++;
      print_str("prod mismatch i=");
      print_i64(i);
      print_str(" got=");
      print_i64(prod[i]);
      print_str(" want=");
      print_i64(p);
      print_str("\n");
    }
    int32_t want = 1000 + i + p;
    if (prod16_as32[i] != want) {
      errs++;
      print_str("acc mismatch i=");
      print_i64(i);
      print_str(" got=");
      print_i64(prod16_as32[i]);
      print_str(" want=");
      print_i64(want);
      print_str("\n");
    }
  }
  print_str(errs ? "vec_probe: FAIL\n" : "vec_probe: PASS\n");
  return errs ? 1 : 0;
}
