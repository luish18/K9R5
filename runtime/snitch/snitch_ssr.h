/* Xssr (stream semantic registers) and Xfrep (FP repetition) for Snitch.
 *
 * These are the two Snitch ISA extensions that make its FP subsystem worth
 * simulating: SSR turns ft0/ft1/ft2 into hardware address generators, so a
 * kernel stops issuing loads, and FREP hands a short instruction sequence to
 * the FPU sequencer, so the integer core stops issuing the loop. Together an
 * FP kernel runs at one FMA per cycle with the scalar core idle.
 *
 * riscv-none-elf-gcc knows neither extension — they are vendor instructions
 * with no upstream binutils support — so every one of them is emitted here as
 * a raw encoding through `.insn`, which needs no assembler support at all:
 *
 *   scfgwi rs1, reg<<5|ssr   ->  .insn r 0x2b, 2, reg, x0, rs1, x<ssr>
 *   scfgri rd,  reg<<5|ssr   ->  .insn r 0x2b, 1, reg, rd, x0, x<ssr>
 *   frep.o rs1, len, 0, 0    ->  .insn i 0x0b, 0, x1, rs1, len-1
 *   frep.i rs1, len, 0, 0    ->  .insn i 0x0b, 0, x0, rs1, len-1
 *
 * The frep rd field carries stagger_mask<<1 | is_outer, so the x1/x0 above is
 * the "outer" bit and not a register: neither instruction touches x0/x1.
 * Register staggering is not used here, hence the two zero immediates.
 *
 * The SSR configuration register map (index, then data mover) and the
 * bound/stride encoding follow the Snitch runtime: bounds hold count-1, and
 * stride[i+1] is the *delta* applied when loop i wraps, i.e. s[i+1] minus the
 * distance loop i already travelled.
 *
 * Usage rule: while SSR is enabled, every FP instruction reading ft0/ft1/ft2
 * consumes a stream element, including any the compiler emits. Keep enabled
 * regions inside a single asm block (SSR_FREP_BEGIN/END do exactly that) and
 * clobber "ft0", "ft1", "ft2" so the register allocator stays away from them.
 */

#ifndef SNITCH_SSR_H
#define SNITCH_SSR_H

#include <stdint.h>

/* Data movers. dm0/dm1/dm2 are wired to ft0/ft1/ft2. */
#define SSR_DM0 0
#define SSR_DM1 1
#define SSR_DM2 2
#define SSR_DM_ALL 31

/* Dimensionality of a stream, as passed to ssr_read()/ssr_write(). */
#define SSR_1D 0
#define SSR_2D 1
#define SSR_3D 2
#define SSR_4D 3

/* Configuration register indices. */
#define SSR_REG_STATUS 0
#define SSR_REG_REPEAT 1
#define SSR_REG_BOUNDS 2  /* 2..5:   one per loop level */
#define SSR_REG_STRIDES 6 /* 6..9:   one per loop level */
#define SSR_REG_RPTR 24   /* 24..27: read pointer, index selects dimension */
#define SSR_REG_WPTR 28   /* 28..31: write pointer, index selects dimension */

/* scfgwi: write `value` to configuration register `reg` of data mover `dm`.
 * Both `reg` and `dm` are encoded in the instruction and must be constants. */
#define ssr_cfg_write(reg, dm, value)                                          \
  __asm__ volatile(".insn r 0x2b, 2, %c[creg], x0, %[cval], x%c[cdm]"          \
                   :                                                           \
                   : [cval] "r"((uint32_t)(value)), [creg] "i"(reg),           \
                     [cdm] "i"(dm))

/* scfgri: read configuration register `reg` of data mover `dm`. */
#define ssr_cfg_read(reg, dm)                                                  \
  ({                                                                           \
    uint32_t _v;                                                               \
    __asm__ volatile(".insn r 0x2b, 1, %c[creg], %[cval], x0, x%c[cdm]"        \
                     : [cval] "=r"(_v)                                         \
                     : [creg] "i"(reg), [cdm] "i"(dm));                        \
    _v;                                                                        \
  })

/* Loop nests. `b*` are iteration counts, `s*` byte strides, outermost last. */
#define ssr_loop_1d(dm, b0, s0)                                                \
  do {                                                                         \
    ssr_cfg_write(SSR_REG_BOUNDS + 0, dm, (uint32_t)(b0) - 1);                 \
    ssr_cfg_write(SSR_REG_STRIDES + 0, dm, (uint32_t)(s0));                    \
  } while (0)

#define ssr_loop_2d(dm, b0, b1, s0, s1)                                        \
  do {                                                                         \
    uint32_t _b0 = (uint32_t)(b0) - 1, _b1 = (uint32_t)(b1) - 1, _a = 0;       \
    ssr_cfg_write(SSR_REG_BOUNDS + 0, dm, _b0);                                \
    ssr_cfg_write(SSR_REG_BOUNDS + 1, dm, _b1);                                \
    ssr_cfg_write(SSR_REG_STRIDES + 0, dm, (uint32_t)(s0) - _a);               \
    _a += (uint32_t)(s0) * _b0;                                                \
    ssr_cfg_write(SSR_REG_STRIDES + 1, dm, (uint32_t)(s1) - _a);               \
    _a += (uint32_t)(s1) * _b1;                                                \
  } while (0)

#define ssr_loop_3d(dm, b0, b1, b2, s0, s1, s2)                                \
  do {                                                                         \
    uint32_t _b0 = (uint32_t)(b0) - 1, _b1 = (uint32_t)(b1) - 1,               \
             _b2 = (uint32_t)(b2) - 1, _a = 0;                                 \
    ssr_cfg_write(SSR_REG_BOUNDS + 0, dm, _b0);                                \
    ssr_cfg_write(SSR_REG_BOUNDS + 1, dm, _b1);                                \
    ssr_cfg_write(SSR_REG_BOUNDS + 2, dm, _b2);                                \
    ssr_cfg_write(SSR_REG_STRIDES + 0, dm, (uint32_t)(s0) - _a);               \
    _a += (uint32_t)(s0) * _b0;                                                \
    ssr_cfg_write(SSR_REG_STRIDES + 1, dm, (uint32_t)(s1) - _a);               \
    _a += (uint32_t)(s1) * _b1;                                                \
    ssr_cfg_write(SSR_REG_STRIDES + 2, dm, (uint32_t)(s2) - _a);               \
    _a += (uint32_t)(s2) * _b2;                                                \
  } while (0)

#define ssr_loop_4d(dm, b0, b1, b2, b3, s0, s1, s2, s3)                        \
  do {                                                                         \
    uint32_t _b0 = (uint32_t)(b0) - 1, _b1 = (uint32_t)(b1) - 1,               \
             _b2 = (uint32_t)(b2) - 1, _b3 = (uint32_t)(b3) - 1, _a = 0;       \
    ssr_cfg_write(SSR_REG_BOUNDS + 0, dm, _b0);                                \
    ssr_cfg_write(SSR_REG_BOUNDS + 1, dm, _b1);                                \
    ssr_cfg_write(SSR_REG_BOUNDS + 2, dm, _b2);                                \
    ssr_cfg_write(SSR_REG_BOUNDS + 3, dm, _b3);                                \
    ssr_cfg_write(SSR_REG_STRIDES + 0, dm, (uint32_t)(s0) - _a);               \
    _a += (uint32_t)(s0) * _b0;                                                \
    ssr_cfg_write(SSR_REG_STRIDES + 1, dm, (uint32_t)(s1) - _a);               \
    _a += (uint32_t)(s1) * _b1;                                                \
    ssr_cfg_write(SSR_REG_STRIDES + 2, dm, (uint32_t)(s2) - _a);               \
    _a += (uint32_t)(s2) * _b2;                                                \
    ssr_cfg_write(SSR_REG_STRIDES + 3, dm, (uint32_t)(s3) - _a);               \
    _a += (uint32_t)(s3) * _b3;                                                \
  } while (0)

/* Deliver every element `count` times before advancing (count-1 in hardware). */
#define ssr_repeat(dm, count)                                                  \
  ssr_cfg_write(SSR_REG_REPEAT, dm, (uint32_t)(count) - 1)

/* Point a stream at memory. Writing the pointer also arms the stream: it
 * selects the loop dimension and resets the address generator's counters. */
#define ssr_read(dm, dim, ptr)                                                 \
  ssr_cfg_write(SSR_REG_RPTR + (dim), dm, (uintptr_t)(ptr))
#define ssr_write(dm, dim, ptr)                                                \
  ssr_cfg_write(SSR_REG_WPTR + (dim), dm, (uintptr_t)(ptr))

/* Enable/disable the ft0-ft2 hijack (CSR 0x7C0).
 *
 * These belong *inside* the asm block that uses the streams — see
 * SSR_FREP_BEGIN — so that no compiler-generated FP instruction can slip into
 * the enabled region and eat a stream element. */
#define SSR_ENABLE_ASM "csrsi 0x7c0, 1\n"
#define SSR_DISABLE_ASM "csrci 0x7c0, 1\n"

/* frep.o: hand the next `len` instructions to the FPU sequencer, which
 * replays them `frep_rpt`+1 times while the integer core runs ahead.
 *
 * `len` is a literal (the sequencer's buffer holds 16 entries) and the
 * repetition count comes from an asm operand that must be named `frep_rpt`.
 * The body must follow immediately, in the same asm block, and must consist
 * of exactly `len` FP instructions. */
#define SSR_STR_(x) #x
#define SSR_STR(x) SSR_STR_(x) /* expand `len` before stringifying it */
#define FREP_O(len) ".insn i 0x0b, 0, x1, %[frep_rpt], (" SSR_STR(len) ") - 1\n"
#define FREP_I(len) ".insn i 0x0b, 0, x0, %[frep_rpt], (" SSR_STR(len) ") - 1\n"

/* One streamed FREP region: enable, repeat `body` (a string of `len` FP
 * instructions) rpt+1 times, disable. Everything the compiler emits stays
 * outside the enabled window. */
#define SSR_FREP_BEGIN(len) SSR_ENABLE_ASM FREP_O(len)
#define SSR_FREP_END SSR_DISABLE_ASM

#endif /* SNITCH_SSR_H */
