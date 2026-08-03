/* Tiny console output helpers on top of semihosting (no libc dependency). */
#ifndef MINIIO_H
#define MINIIO_H

#include "semihost.h"

static inline void print_str(const char *s) { sh_puts(s); }

static inline void print_u64(uint64_t v) {
  char buf[21];
  int i = 20;
  buf[i] = '\0';
  if (v == 0)
    buf[--i] = '0';
  while (v) {
    buf[--i] = '0' + (v % 10);
    v /= 10;
  }
  sh_puts(&buf[i]);
}

static inline void print_i64(int64_t v) {
  if (v < 0) {
    sh_puts("-");
    v = -v;
  }
  print_u64((uint64_t)v);
}

#endif /* MINIIO_H */
