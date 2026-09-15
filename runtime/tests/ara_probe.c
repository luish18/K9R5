/* Probe: do vector loads and stores on the CVA6 + Ara host complete, and do
 * they land the right bytes?
 *
 * The vector host hung inside a vse8.v with vl=512 -- the whole register at
 * e8 and vlen 4096 -- once a program was long enough for GCC to emit that
 * store. This sweeps the shapes that path can take, one variable at a time,
 * so a hang or a wrong byte points at one of them:
 *
 *   copy_e8     vle8.v + vse8.v at vl 16 / 256 / 511 / 512, destination
 *               aligned to a 64 B cache line, misaligned, and straddling one
 *   copy_e32    vle32.v + vse32.v at vl 16 / 64 / 127 / 128 (128 = VLMAX)
 *   sync        a scalar load, then a scalar store, right behind a vector
 *               store -- the vector unit stalls scalar accesses while vector
 *               ones are pending, and a scalar load must see the new bytes
 *   order       a scalar store into bytes a vector store just wrote: the
 *               later store in program order has to win
 *   memcpy_m1   the loop GCC inlines for memcpy, strip-mined at VLMAX, over
 *   memcpy_m8   an MNIST image, a page, and 64 KiB of cold lines
 *   vtype_load  a load, and a vadd.vv, issued at vl 16 that queue behind a
 *   vtype_compute  cold load while the scalar core runs at vl 4: each must
 *               still process all 16 elements
 *   vid_e32     vid.v at SEW 32, and over the whole register at SEW 8, where
 *   vid_e8      the index wraps (the model had no decoding for vid.v)
 *   gather      vluxei64.v, vlse32.v and vsse32.v: the ways GCC's dense kernels
 *   strided_load   reach an operand that is not one contiguous range (the VLSU
 *   strided_store  used to treat every access as unit stride from the base)
 *   vfredusum   the reduction a vectorized dot product ends in
 *   int_*       what GCC builds the integer GEMM from: an int8 gather through
 *               32-bit offsets, vsext.vf4, vmacc.vv, vmadd.vv, vredsum.vs
 *   vsetvl_keep vsetvli zero, zero keeps vl at an integer and a fractional
 *               LMUL (the model used to reset it to VLMAX)
 *   int_gather_inplace_mf4  the integer GEMM's gather as issued: offsets and
 *               result in one register, SEW 8 at mf4 after a vsetvli zero, zero
 *   int_row_sum the end of each integer GEMM row: vmv.s.x, then vredsum.vs at
 *   int_row_sum_inplace  VLMAX into a separate register and in place
 *   vmv_v_x_scalar  a scalar operand overwritten by the core before the unit
 *   vadd_vx_scalar  runs the instruction must still be the issued value (the
 *               integer GEMM's B offset came out as VLMAX this way)
 *
 * Every case is checked byte for byte against the source, plus a guard byte
 * on each side. A line is printed before each case, so a hang shows which
 * case it was in. The checking code is scalar (the Makefile builds this with
 * the vectorizer off): it must not depend on the thing it is testing.
 */
#include "../common/bench.h"
#include "../common/miniio.h"

#define BIG (64 * 1024)
#define PAD 64
#define FILL 0x5a

static uint8_t src[BIG + 2 * PAD] __attribute__((aligned(64)));
static uint8_t dst[BIG + 2 * PAD] __attribute__((aligned(64)));
static uint32_t src32[256 + 2] __attribute__((aligned(64)));
static uint32_t dst32[256 + 2] __attribute__((aligned(64)));
static uint8_t chk[4];

static int failures = 0;
static uint64_t t_start;

static void fill(uint8_t *p, uint64_t n, uint8_t v) {
  for (uint64_t i = 0; i < n; i++)
    p[i] = v;
}

static void begin(const char *name, uint64_t vl, uint64_t off) {
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=");
  print_u64(vl);
  print_str(" off=");
  print_u64(off);
  print_str("\n");
  fill(dst, sizeof(dst), FILL);
  t_start = read_mcycle();
}

static void end(const char *name, int ok, uint64_t cycles) {
  failures += !ok;
  print_str("[ARA-PROBE] ");
  print_str(ok ? "ok   case=" : "FAIL case=");
  print_str(name);
  print_str(" cycles=");
  print_u64(cycles);
  print_str("\n");
}

/* dst[off, off+n) must equal src[off, off+n), with the bytes on either side
 * untouched. */
static int check_bytes(uint64_t off, uint64_t n) {
  if (dst[off - 1] != FILL || dst[off + n] != FILL) {
    print_str("  guard overwritten\n");
    return 0;
  }
  for (uint64_t i = off; i < off + n; i++) {
    if (dst[i] != src[i]) {
      print_str("  first bad byte at ");
      print_u64(i - off);
      print_str(": got 0x");
      print_hex(dst[i]);
      print_str(" want 0x");
      print_hex(src[i]);
      print_str("\n");
      return 0;
    }
  }
  return 1;
}

static void copy_e8(uint64_t vl, uint64_t off) {
  begin("copy_e8", vl, off);
  __asm__ volatile(
      "vsetvli t0, %[n], e8, m1, ta, ma\n\t"
      "vle8.v v1, (%[s])\n\t"
      "vse8.v v1, (%[d])\n\t"
      :
      : [n] "r"(vl), [s] "r"(&src[off]), [d] "r"(&dst[off])
      : "t0", "v1", "memory");
  uint64_t cycles = read_mcycle() - t_start;
  end("copy_e8", check_bytes(off, vl), cycles);
}

static void copy_e32(uint64_t vl) {
  print_str("[ARA-PROBE] run  case=copy_e32 n=");
  print_u64(vl);
  print_str("\n");
  for (uint64_t i = 0; i < 258; i++)
    dst32[i] = 0x5a5a5a5au;
  uint64_t t0 = read_mcycle();
  __asm__ volatile(
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vle32.v v1, (%[s])\n\t"
      "vse32.v v1, (%[d])\n\t"
      :
      : [n] "r"(vl), [s] "r"(&src32[1]), [d] "r"(&dst32[1])
      : "t0", "v1", "memory");
  uint64_t cycles = read_mcycle() - t0;
  int ok = dst32[0] == 0x5a5a5a5au && dst32[vl + 1] == 0x5a5a5a5au;
  for (uint64_t i = 1; ok && i <= vl; i++)
    ok = dst32[i] == src32[i];
  end("copy_e32", ok, cycles);
}

/* A scalar load and then a scalar store immediately behind a vector store. */
static void sync(uint64_t vl) {
  begin("sync", vl, PAD);
  chk[0] = FILL;
  __asm__ volatile(
      "vsetvli t0, %[n], e8, m1, ta, ma\n\t"
      "vle8.v v1, (%[s])\n\t"
      "vse8.v v1, (%[d])\n\t"
      "lb t1, 0(%[d])\n\t"
      "sb t1, 0(%[c])\n\t"
      :
      : [n] "r"(vl), [s] "r"(&src[PAD]), [d] "r"(&dst[PAD]), [c] "r"(chk)
      : "t0", "t1", "v1", "memory");
  uint64_t cycles = read_mcycle() - t_start;
  int ok = check_bytes(PAD, vl);
  if (chk[0] != src[PAD]) {
    print_str("  scalar load behind the vector store read 0x");
    print_hex(chk[0]);
    print_str(", want 0x");
    print_hex(src[PAD]);
    print_str("\n");
    ok = 0;
  }
  end("sync", ok, cycles);
}

/* A scalar store into a byte a vector store just wrote. */
static void order(uint64_t vl) {
  begin("order", vl, PAD);
  __asm__ volatile(
      "vsetvli t0, %[n], e8, m1, ta, ma\n\t"
      "vle8.v v1, (%[s])\n\t"
      "vse8.v v1, (%[d])\n\t"
      "li t1, 0xab\n\t"
      "sb t1, 5(%[d])\n\t"
      :
      : [n] "r"(vl), [s] "r"(&src[PAD]), [d] "r"(&dst[PAD])
      : "t0", "t1", "v1", "memory");
  uint64_t cycles = read_mcycle() - t_start;
  uint8_t want = src[PAD + 5];
  src[PAD + 5] = 0xab;
  int ok = check_bytes(PAD, vl);
  src[PAD + 5] = want;
  end("order", ok, cycles);
}

/* The strip-mined copy GCC inlines for memcpy. */
static void copy_loop(const char *name, uint64_t n, int lmul8) {
  begin(name, n, PAD);
  if (lmul8) {
    __asm__ volatile(
        "mv t1, %[s]\n\t"
        "mv t2, %[d]\n\t"
        "mv t3, %[n]\n\t"
        "1:\n\t"
        "vsetvli t0, t3, e8, m8, ta, ma\n\t"
        "vle8.v v1, (t1)\n\t"
        "vse8.v v1, (t2)\n\t"
        "add t1, t1, t0\n\t"
        "add t2, t2, t0\n\t"
        "sub t3, t3, t0\n\t"
        "bnez t3, 1b\n\t"
        :
        : [n] "r"(n), [s] "r"(&src[PAD]), [d] "r"(&dst[PAD])
        : "t0", "t1", "t2", "t3", "v1", "memory");
  } else {
    __asm__ volatile(
        "mv t1, %[s]\n\t"
        "mv t2, %[d]\n\t"
        "mv t3, %[n]\n\t"
        "1:\n\t"
        "vsetvli t0, t3, e8, m1, ta, ma\n\t"
        "vle8.v v1, (t1)\n\t"
        "vse8.v v1, (t2)\n\t"
        "add t1, t1, t0\n\t"
        "add t2, t2, t0\n\t"
        "sub t3, t3, t0\n\t"
        "bnez t3, 1b\n\t"
        :
        : [n] "r"(n), [s] "r"(&src[PAD]), [d] "r"(&dst[PAD])
        : "t0", "t1", "t2", "t3", "v1", "memory");
  }
  uint64_t cycles = read_mcycle() - t_start;
  end(name, check_bytes(PAD, n), cycles);
}

/* An instruction issued at vl 16 that the unit cannot start straight away: it
 * queues behind a 512 B load from lines the L1 has never held, and while it
 * waits the scalar core drops vl to 4 for ~500 cycles, then restores it and
 * stores the result. If the unit sizes the work from the live vl when it
 * finally starts, instead of the vl the instruction was issued under, only 4
 * of the 16 elements come out right. v2 is filled first with a value the
 * instruction cannot produce, so a short count cannot hide behind a register
 * that already holds the right data. */
static void vtype_race(const char *name, int compute) {
  static uint8_t cold[512] __attribute__((aligned(64)));
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=16\n");
  for (uint64_t i = 0; i < 258; i++)
    dst32[i] = 0x5a5a5a5au;
  __asm__ volatile(
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vle32.v v2, (%[d])\n\t"
      :
      : [n] "r"(16), [d] "r"(&dst32[1])
      : "t0", "v2", "memory");
  uint64_t t0 = read_mcycle();
  if (compute) {
    __asm__ volatile(
        "vsetvli t0, %[big], e8, m1, ta, ma\n\t"
        "vle8.v v3, (%[c])\n\t"
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vle32.v v1, (%[s])\n\t"
        "vadd.vv v2, v1, v1\n\t"
        "vsetvli t0, %[m], e32, m1, ta, ma\n\t"
        "li t1, 256\n\t"
        "1:\n\t"
        "addi t1, t1, -1\n\t"
        "bnez t1, 1b\n\t"
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vse32.v v2, (%[d])\n\t"
        :
        : [big] "r"(512), [n] "r"(16), [m] "r"(4), [c] "r"(cold),
          [s] "r"(&src32[1]), [d] "r"(&dst32[1])
        : "t0", "t1", "v1", "v2", "v3", "memory");
  } else {
    __asm__ volatile(
        "vsetvli t0, %[big], e8, m1, ta, ma\n\t"
        "vle8.v v3, (%[c])\n\t"
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vle32.v v2, (%[s])\n\t"
        "vsetvli t0, %[m], e32, m1, ta, ma\n\t"
        "li t1, 256\n\t"
        "1:\n\t"
        "addi t1, t1, -1\n\t"
        "bnez t1, 1b\n\t"
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vse32.v v2, (%[d])\n\t"
        :
        : [big] "r"(512), [n] "r"(16), [m] "r"(4), [c] "r"(cold),
          [s] "r"(&src32[1]), [d] "r"(&dst32[1])
        : "t0", "t1", "v2", "v3", "memory");
  }
  uint64_t cycles = read_mcycle() - t0;
  int ok = dst32[0] == 0x5a5a5a5au && dst32[17] == 0x5a5a5a5au;
  for (uint64_t i = 1; i <= 16; i++) {
    uint32_t want = compute ? src32[i] + src32[i] : src32[i];
    if (dst32[i] != want) {
      if (ok) {
        print_str("  first bad element at ");
        print_u64(i - 1);
        print_str(": got 0x");
        print_hex(dst32[i]);
        print_str(" want 0x");
        print_hex(want);
        print_str("\n");
      }
      ok = 0;
    }
  }
  end(name, ok, cycles);
}

/* vid.v: every element takes its own index -- at SEW 32 over a short vector,
 * and at SEW 8 over the whole 512-element register, where the index wraps.
 * GCC emits it for the index vectors of gathers (the autovectorized matmul),
 * and the model had no decoding for it at all. */
static void vid_probe(uint64_t vl, int sew8) {
  const char *name = sew8 ? "vid_e8" : "vid_e32";
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=");
  print_u64(vl);
  print_str("\n");
  fill(dst, sizeof(dst), FILL);
  for (uint64_t i = 0; i < 258; i++)
    dst32[i] = 0x5a5a5a5au;
  uint64_t t0 = read_mcycle();
  if (sew8) {
    __asm__ volatile(
        "vsetvli t0, %[n], e8, m1, ta, ma\n\t"
        "vid.v v2\n\t"
        "vse8.v v2, (%[d])\n\t"
        :
        : [n] "r"(vl), [d] "r"(&dst[PAD])
        : "t0", "v2", "memory");
  } else {
    __asm__ volatile(
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vid.v v2\n\t"
        "vse32.v v2, (%[d])\n\t"
        :
        : [n] "r"(vl), [d] "r"(&dst32[1])
        : "t0", "v2", "memory");
  }
  uint64_t cycles = read_mcycle() - t0;
  int ok = sew8 ? (dst[PAD - 1] == FILL && dst[PAD + vl] == FILL)
                : (dst32[0] == 0x5a5a5a5au && dst32[vl + 1] == 0x5a5a5a5au);
  if (!ok)
    print_str("  guard overwritten\n");
  for (uint64_t i = 0; ok && i < vl; i++) {
    uint64_t got = sew8 ? dst[PAD + i] : dst32[1 + i];
    uint64_t want = sew8 ? (i & 0xff) : i;
    if (got != want) {
      print_str("  first bad element at ");
      print_u64(i);
      print_str(": got ");
      print_u64(got);
      print_str(" want ");
      print_u64(want);
      print_str("\n");
      ok = 0;
    }
  }
  end(name, ok, cycles);
}

/* The memory shapes GCC's dense kernels reach the second operand through:
 * a gather (vluxei64.v with 64-bit indices), a strided load (vlse32.v) and a
 * strided store (vsse32.v). All three name memory the unit cannot treat as one
 * contiguous range starting at the base address. */
static void access_shape(const char *name, int shape) {
  static uint64_t idx[16];
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=16\n");
  for (uint64_t i = 0; i < 258; i++)
    dst32[i] = 0x5a5a5a5au;
  for (uint64_t i = 0; i < 16; i++)
    idx[i] = ((i * 5 + 3) % 16) * 4; /* byte offsets of a scattered permutation */
  uint64_t t0 = read_mcycle();
  if (shape == 0) {
    __asm__ volatile(
        "vsetvli t0, %[n], e64, m1, ta, ma\n\t"
        "vle64.v v4, (%[x])\n\t"
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vluxei64.v v2, (%[s]), v4\n\t"
        "vse32.v v2, (%[d])\n\t"
        :
        : [n] "r"(16), [x] "r"(idx), [s] "r"(&src32[1]), [d] "r"(&dst32[1])
        : "t0", "v2", "v4", "memory");
  } else if (shape == 1) {
    __asm__ volatile(
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vlse32.v v2, (%[s]), %[st]\n\t"
        "vse32.v v2, (%[d])\n\t"
        :
        : [n] "r"(16), [st] "r"(12), [s] "r"(&src32[1]), [d] "r"(&dst32[1])
        : "t0", "v2", "memory");
  } else {
    __asm__ volatile(
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vle32.v v2, (%[s])\n\t"
        "vsse32.v v2, (%[d]), %[st]\n\t"
        :
        : [n] "r"(8), [st] "r"(8), [s] "r"(&src32[1]), [d] "r"(&dst32[1])
        : "t0", "v2", "memory");
  }
  uint64_t cycles = read_mcycle() - t0;
  int ok = 1;
  for (uint64_t i = 0; ok && i < 16; i++) {
    uint32_t got, want;
    if (shape == 0) {
      /* vluxei64 offsets are in bytes; these are all multiples of 4 */
      got = dst32[1 + i], want = src32[1 + idx[i] / 4];
    } else if (shape == 1) {
      got = dst32[1 + i], want = src32[1 + 3 * i];
    } else {
      /* 8 elements at stride 8 B: every other slot written, the rest untouched */
      got = dst32[1 + i], want = (i % 2 == 0 && i / 2 < 8) ? src32[1 + i / 2] : 0x5a5a5a5au;
    }
    if (got != want) {
      print_str("  first bad element at ");
      print_u64(i);
      print_str(": got 0x");
      print_hex(got);
      print_str(" want 0x");
      print_hex(want);
      print_str("\n");
      ok = 0;
    }
  }
  end(name, ok, cycles);
}

/* vfredusum.vs over 16 fp32 elements: the reduction GCC emits for a dot
 * product. The integer-valued data keeps the float sum exact. */
static void reduction(void) {
  static float x[16];
  float got = 0.0f, want = 0.0f;
  print_str("[ARA-PROBE] run  case=vfredusum n=16\n");
  for (uint64_t i = 0; i < 16; i++) {
    x[i] = (float)(i + 1);
    want += x[i];
  }
  uint64_t t0 = read_mcycle();
  __asm__ volatile(
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vle32.v v2, (%[x])\n\t"
      "vmv.s.x v3, zero\n\t"
      "vfredusum.vs v3, v2, v3\n\t"
      "vfmv.f.s ft0, v3\n\t"
      "fsw ft0, 0(%[o])\n\t"
      :
      : [n] "r"(16), [x] "r"(x), [o] "r"(&got)
      : "t0", "ft0", "v2", "v3", "memory");
  uint64_t cycles = read_mcycle() - t0;
  int ok = got == want;
  if (!ok) {
    print_str("  got ");
    print_u64((uint64_t)got);
    print_str(" want ");
    print_u64((uint64_t)want);
    print_str("\n");
  }
  end("vfredusum", ok, cycles);
}

/* The instructions GCC builds the integer GEMM (s8 x s8 -> s32) from, each
 * checked against the scalar result: a gather of int8 elements through 32-bit
 * offsets, sign extension by four, the two integer multiply-adds, and the
 * integer sum reduction. */
static void integer_op(const char *name, int kind) {
  static int8_t a8[16];
  static uint32_t idx32[16];
  static int32_t b32[16], c32[16], out32[16];
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=16\n");
  fill(dst, sizeof(dst), FILL);
  for (int i = 0; i < 16; i++) {
    a8[i] = (int8_t)(i * 17 - 128);
    idx32[i] = (uint32_t)((i * 5 + 3) % 16);
    b32[i] = i * 3 - 20;
    c32[i] = 100 - 7 * i;
    out32[i] = 1000 + i;
  }
  uint64_t t0 = read_mcycle();
  switch (kind) {
  case 0:
    __asm__ volatile(
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vle32.v v4, (%[x])\n\t"
        "vsetvli t0, %[n], e8, m1, ta, ma\n\t"
        "vluxei32.v v2, (%[a]), v4\n\t"
        "vse8.v v2, (%[d])\n\t"
        :
        : [n] "r"(16), [x] "r"(idx32), [a] "r"(a8), [d] "r"(&dst[PAD])
        : "t0", "v2", "v4", "memory");
    break;
  case 1:
    __asm__ volatile(
        "vsetvli t0, %[n], e8, m1, ta, ma\n\t"
        "vle8.v v2, (%[a])\n\t"
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vsext.vf4 v3, v2\n\t"
        "vse32.v v3, (%[o])\n\t"
        :
        : [n] "r"(16), [a] "r"(a8), [o] "r"(out32)
        : "t0", "v2", "v3", "memory");
    break;
  case 2:
  case 3:
    if (kind == 2) {
      __asm__ volatile(
          "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
          "vle32.v v1, (%[b])\n\t"
          "vle32.v v2, (%[c])\n\t"
          "vle32.v v3, (%[o])\n\t"
          "vmacc.vv v3, v1, v2\n\t"
          "vse32.v v3, (%[o])\n\t"
          :
          : [n] "r"(16), [b] "r"(b32), [c] "r"(c32), [o] "r"(out32)
          : "t0", "v1", "v2", "v3", "memory");
    } else {
      __asm__ volatile(
          "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
          "vle32.v v1, (%[b])\n\t"
          "vle32.v v2, (%[c])\n\t"
          "vle32.v v3, (%[o])\n\t"
          "vmadd.vv v3, v1, v2\n\t"
          "vse32.v v3, (%[o])\n\t"
          :
          : [n] "r"(16), [b] "r"(b32), [c] "r"(c32), [o] "r"(out32)
          : "t0", "v1", "v2", "v3", "memory");
    }
    break;
  default:
    __asm__ volatile(
        "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
        "vle32.v v1, (%[c])\n\t"
        "vmv.s.x v2, zero\n\t"
        "vredsum.vs v2, v1, v2\n\t"
        "vmv.x.s t1, v2\n\t"
        "sw t1, 0(%[o])\n\t"
        :
        : [n] "r"(16), [c] "r"(c32), [o] "r"(out32)
        : "t0", "t1", "v1", "v2", "memory");
    break;
  }
  uint64_t cycles = read_mcycle() - t0;

  int ok = 1;
  int64_t sum = 0;
  for (int i = 0; i < 16; i++)
    sum += c32[i];
  for (int i = 0; ok && i < (kind == 4 ? 1 : 16); i++) {
    int64_t got, want;
    switch (kind) {
    case 0: got = (int8_t)dst[PAD + i], want = a8[idx32[i]]; break;
    case 1: got = out32[i], want = a8[i]; break;
    case 2: got = out32[i], want = (1000 + i) + b32[i] * c32[i]; break;
    case 3: got = out32[i], want = b32[i] * (1000 + i) + c32[i]; break;
    default: got = out32[0], want = sum; break;
    }
    if (got != want) {
      print_str("  first bad element at ");
      print_u64((uint64_t)i);
      print_str(": got ");
      print_i64(got);
      print_str(" want ");
      print_i64(want);
      print_str("\n");
      ok = 0;
    }
  }
  end(name, ok, cycles);
}

/* The gather exactly as the integer GEMM issues it: the offsets and the result
 * share one register, and the data is SEW 8 at a fractional LMUL (mf4), so the
 * register's elements narrow from 32 to 8 bits between the offset load and the
 * gather. RVV allows that overlap only because the result is the narrower
 * one, so the unit must not let a written element disturb an offset it has
 * yet to read. */
static void gather_inplace(void) {
  static int8_t a8[64];
  static uint32_t idx32[16];
  print_str("[ARA-PROBE] run  case=int_gather_inplace_mf4 n=16\n");
  fill(dst, sizeof(dst), FILL);
  for (int i = 0; i < 64; i++)
    a8[i] = (int8_t)(i * 37 - 100);
  for (int i = 0; i < 16; i++)
    idx32[i] = (uint32_t)((i * 11 + 7) % 64);
  uint64_t t0 = read_mcycle();
  __asm__ volatile(
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vle32.v v3, (%[x])\n\t"
      "vsetvli zero, zero, e8, mf4, ta, ma\n\t"
      "vluxei32.v v3, (%[a]), v3\n\t"
      "vse8.v v3, (%[d])\n\t"
      :
      : [n] "r"(16), [x] "r"(idx32), [a] "r"(a8), [d] "r"(&dst[PAD])
      : "t0", "v3", "memory");
  uint64_t cycles = read_mcycle() - t0;
  int ok = dst[PAD - 1] == FILL && dst[PAD + 16] == FILL;
  if (!ok)
    print_str("  guard overwritten\n");
  for (int i = 0; ok && i < 16; i++) {
    int8_t got = (int8_t)dst[PAD + i], want = a8[idx32[i]];
    if (got != want) {
      print_str("  first bad element at ");
      print_u64((uint64_t)i);
      print_str(": got ");
      print_i64(got);
      print_str(" want ");
      print_i64(want);
      print_str("\n");
      ok = 0;
    }
  }
  end("int_gather_inplace_mf4", ok, cycles);
}

/* vsetvli with rd and rs1 both zero changes the type and keeps vl -- the form a
 * compiler uses to switch SEW or LMUL mid-loop without touching the length.
 * Checked at an integer and at a fractional LMUL. */
static void vsetvl_keep(void) {
  uint64_t vl_m1 = 0, vl_mf4 = 0;
  print_str("[ARA-PROBE] run  case=vsetvl_keep n=16\n");
  __asm__ volatile(
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vsetvli zero, zero, e8, m1, ta, ma\n\t"
      "csrr t1, vl\n\t"
      "sd t1, 0(%[a])\n\t"
      "vsetvli t0, %[n], e32, m1, ta, ma\n\t"
      "vsetvli zero, zero, e8, mf4, ta, ma\n\t"
      "csrr t1, vl\n\t"
      "sd t1, 0(%[b])\n\t"
      :
      : [n] "r"(16), [a] "r"(&vl_m1), [b] "r"(&vl_mf4)
      : "t0", "t1", "memory");
  int ok = vl_m1 == 16 && vl_mf4 == 16;
  if (!ok) {
    print_str("  vl after e8,m1 = ");
    print_u64(vl_m1);
    print_str(", after e8,mf4 = ");
    print_u64(vl_mf4);
    print_str(" (want 16)\n");
  }
  end("vsetvl_keep", ok, 0);
}

/* The end of each row of the integer GEMM, as GCC issues it: at VLMAX, an
 * accumulator whose first 32 elements hold products and the rest zero; a
 * vmv.s.x zeroing element 0 of a register that holds something else there;
 * and the sum reduced from that register -- in place, into the register being
 * reduced, as the kernel does, and into a separate one -- then read back with
 * vmv.x.s. */
static void row_sum(const char *name, int in_place) {
  static int32_t acc[128], other[128];
  int64_t want = 0;
  uint64_t got = 0;
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=128\n");
  for (int i = 0; i < 128; i++) {
    acc[i] = i < 32 ? i * 1000 - 7000 : 0;
    other[i] = 159 - i;
    want += acc[i];
  }
  uint64_t t0 = read_mcycle();
  if (in_place) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vle32.v v5, (%[a])\n\t"
        "vle32.v v1, (%[x])\n\t"
        "vmv.s.x v1, zero\n\t"
        "vredsum.vs v5, v5, v1\n\t"
        "vmv.x.s t1, v5\n\t"
        "sd t1, 0(%[o])\n\t"
        :
        : [a] "r"(acc), [x] "r"(other), [o] "r"(&got)
        : "t0", "t1", "v1", "v5", "memory");
  } else {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vle32.v v5, (%[a])\n\t"
        "vle32.v v1, (%[x])\n\t"
        "vmv.s.x v1, zero\n\t"
        "vredsum.vs v6, v5, v1\n\t"
        "vmv.x.s t1, v6\n\t"
        "sd t1, 0(%[o])\n\t"
        :
        : [a] "r"(acc), [x] "r"(other), [o] "r"(&got)
        : "t0", "t1", "v1", "v5", "v6", "memory");
  }
  uint64_t cycles = read_mcycle() - t0;
  int ok = (int32_t)got == (int32_t)want;
  if (!ok) {
    print_str("  got ");
    print_i64((int32_t)got);
    print_str(" want ");
    print_i64(want);
    print_str("\n");
  }
  end(name, ok, cycles);
}

/* A vector instruction that takes a scalar operand must use the value the
 * register held when the instruction was issued. The unit runs it later --
 * here it queues behind a 128-element instruction on the same block -- and the
 * core has already put something else in that register by then. This is the
 * integer GEMM's failure: vmv.v.x v9, t1 ran after vsetvli t1, zero and filled
 * the B offset with VLMAX. Checked for vmv.v.x (slide block) and vadd.vx
 * (compute block). */
static void scalar_operand(const char *name, int kind) {
  static int32_t out[128];
  print_str("[ARA-PROBE] run  case=");
  print_str(name);
  print_str(" n=128\n");
  for (int i = 0; i < 128; i++)
    out[i] = 0x5a5a5a5a;
  uint64_t t0 = read_mcycle();
  if (kind == 0) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vmv.v.i v7, 0\n\t"
        "li t1, 12345\n\t"
        "vmv.v.x v8, t1\n\t"
        "li t1, 999\n\t"
        "vse32.v v8, (%[o])\n\t"
        :
        : [o] "r"(out)
        : "t0", "t1", "v7", "v8", "memory");
  } else {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m1, ta, ma\n\t"
        "vmv.v.i v7, 1\n\t"
        "vadd.vv v6, v7, v7\n\t"
        "li t1, 12345\n\t"
        "vadd.vx v8, v6, t1\n\t"
        "li t1, 999\n\t"
        "vse32.v v8, (%[o])\n\t"
        :
        : [o] "r"(out)
        : "t0", "t1", "v6", "v7", "v8", "memory");
  }
  uint64_t cycles = read_mcycle() - t0;
  int32_t want = kind == 0 ? 12345 : 12347;
  int ok = 1;
  for (int i = 0; ok && i < 128; i++) {
    if (out[i] != want) {
      print_str("  first bad element at ");
      print_u64((uint64_t)i);
      print_str(": got ");
      print_i64(out[i]);
      print_str(" want ");
      print_i64(want);
      print_str("\n");
      ok = 0;
    }
  }
  end(name, ok, cycles);
}

int main(void) {
  for (uint64_t i = 0; i < sizeof(src); i++)
    src[i] = (uint8_t)(i * 7 + 1);
  for (uint64_t i = 0; i < 258; i++)
    src32[i] = (uint32_t)(i * 0x01010101u + 0x10203);

  static const uint64_t e8_lengths[] = {16, 256, 511, 512};
  /* Aligned to a cache line, misaligned, and 4 B before a line boundary. */
  static const uint64_t offsets[] = {PAD, PAD + 3, PAD + 60};
  for (unsigned l = 0; l < 4; l++)
    for (unsigned o = 0; o < 3; o++)
      copy_e8(e8_lengths[l], offsets[o]);

  static const uint64_t e32_lengths[] = {16, 64, 127, 128};
  for (unsigned l = 0; l < 4; l++)
    copy_e32(e32_lengths[l]);

  sync(16);
  sync(512);
  order(16);
  order(512);
  vtype_race("vtype_load", 0);
  vtype_race("vtype_compute", 1);
  vid_probe(16, 0);
  vid_probe(512, 1);
  access_shape("gather", 0);
  access_shape("strided_load", 1);
  access_shape("strided_store", 2);
  reduction();
  integer_op("int_gather8", 0);
  integer_op("int_vsext_vf4", 1);
  integer_op("int_vmacc", 2);
  integer_op("int_vmadd", 3);
  integer_op("int_vredsum", 4);
  vsetvl_keep();
  gather_inplace();
  row_sum("int_row_sum", 0);
  row_sum("int_row_sum_inplace", 1);
  scalar_operand("vmv_v_x_scalar", 0);
  scalar_operand("vadd_vx_scalar", 1);

  /* One MNIST image (28 x 28 fp32), a page, and more cold lines than the L1
   * data cache holds. */
  static const uint64_t sizes[] = {28 * 28 * 4, 4096, BIG};
  for (unsigned s = 0; s < 3; s++)
    copy_loop("memcpy_m1", sizes[s], 0);
  for (unsigned s = 0; s < 3; s++)
    copy_loop("memcpy_m8", sizes[s], 1);

  print_str("[ARA-PROBE] failures=");
  print_u64(failures);
  print_str("\n");
  return failures != 0;
}
