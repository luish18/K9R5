/* Minimal RISC-V semihosting layer for GVSoC's ISS.
 * The ISS recognizes the canonical semihosting trap sequence
 * (slli x0,x0,0x1f; ebreak; srai x0,x0,7) with a0 = op, a1 = arg.
 */
#ifndef SEMIHOST_H
#define SEMIHOST_H

#include <stdint.h>

#define SEMIHOST_SYS_WRITEC 0x03
#define SEMIHOST_SYS_WRITE0 0x04
#define SEMIHOST_SYS_EXIT   0x18
#define SEMIHOST_ADP_STOPPED_APPLICATION_EXIT 0x20026

static inline long semihost_call(long op, long arg) {
  register long a0 __asm__("a0") = op;
  register long a1 __asm__("a1") = arg;
  __asm__ volatile(".option push\n\t"
                   ".option norvc\n\t"
                   "slli x0, x0, 0x1f\n\t"
                   "ebreak\n\t"
                   "srai x0, x0, 7\n\t"
                   ".option pop"
                   : "+r"(a0)
                   : "r"(a1)
                   : "memory");
  return a0;
}

static inline void sh_putc(char c) {
  /* SYS_WRITEC takes a pointer to the character. */
  semihost_call(SEMIHOST_SYS_WRITEC, (long)&c);
}

static inline void sh_puts(const char *s) {
  /* SYS_WRITE0 takes a pointer to a NUL-terminated string. */
  semihost_call(SEMIHOST_SYS_WRITE0, (long)s);
}

static inline void __attribute__((noreturn)) sh_exit(int code) {
  /* GVSoC maps a1 == ADP_STOPPED_APPLICATION_EXIT to exit status 0,
   * anything else to 1. */
  semihost_call(SEMIHOST_SYS_EXIT,
                code == 0 ? SEMIHOST_ADP_STOPPED_APPLICATION_EXIT : 0);
  for (;;)
    ;
}

#endif /* SEMIHOST_H */
