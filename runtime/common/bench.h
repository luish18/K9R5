/* Cycle / instruction counters via RISC-V M-mode CSRs (works on rv32 & rv64). */
#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

static inline uint64_t read_mcycle(void) {
#if __riscv_xlen == 64
  uint64_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
#else
  uint32_t hi, lo, hi2;
  do {
    __asm__ volatile("csrr %0, mcycleh" : "=r"(hi));
    __asm__ volatile("csrr %0, mcycle" : "=r"(lo));
    __asm__ volatile("csrr %0, mcycleh" : "=r"(hi2));
  } while (hi != hi2);
  return ((uint64_t)hi << 32) | lo;
#endif
}

static inline uint64_t read_minstret(void) {
#if __riscv_xlen == 64
  uint64_t c;
  __asm__ volatile("csrr %0, minstret" : "=r"(c));
  return c;
#else
  uint32_t hi, lo, hi2;
  do {
    __asm__ volatile("csrr %0, minstreth" : "=r"(hi));
    __asm__ volatile("csrr %0, minstret" : "=r"(lo));
    __asm__ volatile("csrr %0, minstreth" : "=r"(hi2));
  } while (hi != hi2);
  return ((uint64_t)hi << 32) | lo;
#endif
}

#endif /* BENCH_H */
