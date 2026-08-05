/* The mesh address map and its configured latencies, as seen from C.
 *
 * Mirrors targets/hetero/system.py. The probe in runtime/tests/mesh_probe.c
 * checks the model against the latencies declared here, so if the two files
 * disagree the probe fails — which is the point: nothing else in the flow
 * would notice a board built with different numbers than the ones documented.
 */
#ifndef MESH_SYSTEM_H
#define MESH_SYSTEM_H

#include <stdint.h>

/* Address map (system.py: BOOTROM, CLUSTER_BASE, L2_BASE, ...) */
#define MESH_BOOTROM_BASE 0x00001000UL
#define MESH_CLUSTER_BASE 0x10000000UL
#define MESH_CLUSTER_STRIDE 0x00040000UL
#define MESH_L2_BASE 0x70000000UL
#define MESH_L3_BASE 0x80000000UL
#define MESH_HYPERRAM_BASE 0x90000000UL

#define MESH_L1_SIZE 0x00020000UL /* cluster TCDM */
#define MESH_L2_SIZE 0x00400000UL
#define MESH_L3_SIZE 0x04000000UL
#define MESH_HYPERRAM_SIZE 0x02000000UL

/* Configured mapping latencies, in core cycles (system.py). */
#define MESH_L2_LATENCY 20
#define MESH_L3_LATENCY 50
#define MESH_HYPERRAM_LATENCY 150

/* Bytes per cycle each level sustains (system.py). A miss pays the latency
 * above plus the time to move one cache line at this width, which is why the
 * probe cannot compare latencies alone. */
#define MESH_L2_WIDTH 64
#define MESH_L3_WIDTH 64
#define MESH_HYPERRAM_WIDTH 1

/* Host L1 cache line, from memsys.LINE_SIZE — the unit a miss refills. */
#define MESH_LINE_SIZE 64
/* Host L1 data cache, from memsys.DCACHE. The probe's working set has to
 * exceed it or the chase would measure the cache instead of the level. */
#define MESH_DCACHE_SIZE (32 * 1024)

/* Cluster i's TCDM. */
static inline volatile uint64_t *mesh_cluster_tcdm(unsigned int cluster) {
  return (volatile uint64_t *)(MESH_CLUSTER_BASE +
                               (unsigned long)cluster * MESH_CLUSTER_STRIDE);
}

#endif /* MESH_SYSTEM_H */
