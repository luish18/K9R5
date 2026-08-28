/* The cluster iDMA, driven through the Snitch Xdma instructions.
 *
 * GCC knows nothing about Xdma, so each instruction is emitted as a raw
 * encoding with .insn -- the same technique runtime/snitch/snitch_ssr.h uses
 * for Xssr and Xfrep. All of them are opcode 0x2b, funct3 0, with funct7
 * selecting:
 *
 *   funct7  instruction        meaning
 *   ------  -----------------  ------------------------------------------
 *   0       dmsrc  rs1, rs2    source address = (rs2 << 32) | rs1
 *   1       dmdst  rs1, rs2    destination address = (rs2 << 32) | rs1
 *   6       dmstr  rs1, rs2    source stride = rs1, destination stride = rs2
 *   7       dmrep  rs1         repetitions of a 2D transfer
 *   3       dmcpy  rd, rs1,rs2 rd = enqueue(size = rs1, config = rs2)
 *   2       dmcpyi rd, rs1,imm rd = enqueue(size = rs1, config = imm)
 *   5/4     dmstat[i] rd, sel  rd = status(sel)
 *
 * config bit 1 selects a 2D transfer. Status selector 2 reads "a transfer is
 * still outstanding", which is what hes_dma_wait() polls.
 *
 * Only the cluster's DMA core may issue these: GVSoC binds the iDMA's offload
 * port to that core alone (SnitchCluster wires cores[dma_core].o_OFFLOAD), so
 * an Xdma instruction from any other core has nowhere to go.
 */
#ifndef HES_DMA_H
#define HES_DMA_H

#include "hes_cluster.h"

#include <stdint.h>

#define HES_DMA_CFG_1D 0
#define HES_DMA_CFG_2D 2

/* Addresses are 32-bit on the clusters, so the high half is always zero. */
static inline void hes_dma_src(uint32_t addr) {
  __asm__ volatile(".insn r 0x2b, 0, 0, x0, %0, x0" ::"r"(addr) : "memory");
}

static inline void hes_dma_dst(uint32_t addr) {
  __asm__ volatile(".insn r 0x2b, 0, 1, x0, %0, x0" ::"r"(addr) : "memory");
}

static inline void hes_dma_stride(uint32_t src_stride, uint32_t dst_stride) {
  __asm__ volatile(".insn r 0x2b, 0, 6, x0, %0, %1" ::"r"(src_stride),
                   "r"(dst_stride)
                   : "memory");
}

static inline void hes_dma_rep(uint32_t reps) {
  __asm__ volatile(".insn r 0x2b, 0, 7, x0, %0, x0" ::"r"(reps) : "memory");
}

/* Enqueue the transfer configured above. Returns its id. The instruction
 * stalls the core if the DMA cannot take another transfer yet. */
static inline uint32_t hes_dma_start(uint32_t size, uint32_t config) {
  uint32_t id;
  __asm__ volatile(".insn r 0x2b, 0, 3, %0, %1, %2"
                   : "=r"(id)
                   : "r"(size), "r"(config)
                   : "memory");
  return id;
}

/* Non-zero while any enqueued transfer is still in flight. */
static inline uint32_t hes_dma_busy(void) {
  uint32_t status;
  uint32_t sel = 2;
  __asm__ volatile(".insn r 0x2b, 0, 5, %0, x0, %1"
                   : "=r"(status)
                   : "r"(sel)
                   : "memory");
  return status;
}

static inline void hes_dma_wait(void) {
  while (hes_dma_busy()) {
  }
  __asm__ volatile("" ::: "memory");
}

/* One contiguous copy. */
static inline uint32_t hes_dma_copy(uint32_t dst, uint32_t src, uint32_t size) {
  hes_dma_src(src);
  hes_dma_dst(dst);
  return hes_dma_start(size, HES_DMA_CFG_1D);
}

/* `reps` rows of `size` bytes, each row `*_stride` bytes apart. */
static inline uint32_t hes_dma_copy_2d(uint32_t dst, uint32_t src, uint32_t size,
                                       uint32_t dst_stride, uint32_t src_stride,
                                       uint32_t reps) {
  hes_dma_src(src);
  hes_dma_dst(dst);
  hes_dma_stride(src_stride, dst_stride);
  hes_dma_rep(reps);
  return hes_dma_start(size, HES_DMA_CFG_2D);
}

#endif /* HES_DMA_H */
