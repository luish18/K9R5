/* Startup for every core of a Snitch or Snitch+Spatz cluster.
 *
 * All cores of the cluster enter here at once. Each one takes its own stack
 * at the top of TCDM; the control core alone copies .data out of main memory
 * and clears .bss, and the cluster barrier holds the others until it is done.
 *
 * main() is the job loop and never returns: a cluster core must not call the
 * semihosting exit, which would end the whole simulation.
 */

#include "hes_cluster.h"
#include "hes_mailbox.h"

    .section .text.init, "ax", @progbits
    .globl _start
_start:
    /* Enable the FPU and, where there is one, the vector unit. */
    li      t0, (3 << 13) | (3 << 9)
    csrs    mstatus, t0
    csrw    fcsr, zero

    la      t0, trap_vec
    csrw    mtvec, t0

    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    /* sp = __stack_region + (core_idx + 1) * __stack_size, where
     * core_idx = mhartid - HES_MY_FIRST_HARTID. */
    csrr    t0, mhartid
    li      t1, HES_MY_FIRST_HARTID
    sub     t0, t0, t1          /* t0 = core_idx */
    la      t1, __stack_size
    addi    t2, t0, 1
    mul     t2, t2, t1
    la      t1, __stack_region
    add     sp, t1, t2

    /* Only the control core initializes the shared TCDM image. */
    li      t1, HES_MY_CTRL_CORE
    bne     t0, t1, wait_init

    /* Copy .data from its load address in main memory into TCDM. */
    la      t0, __data_load_start
    la      t1, __data_start
    la      t2, __data_end
    beq     t0, t1, 2f
1:  bgeu    t1, t2, 2f
    lw      t3, 0(t0)
    sw      t3, 0(t1)
    addi    t0, t0, 4
    addi    t1, t1, 4
    j       1b
2:
    /* Zero .bss. */
    la      t0, __bss_start
    la      t1, __bss_end
3:  bgeu    t0, t1, 4f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       3b
4:

wait_init:
    /* Every core joins: the barrier releases only once all of them have, so
     * this is both the wait and the release. */
    csrr    x0, HES_CSR_BARRIER

    call    main

    /* main() is the job loop and should never return. If it does, park
     * without exiting -- the host owns the end of the simulation. */
park:
    wfi
    j       park

    .align 2
trap_vec:
    /* A trap on a cluster core cannot report through semihosting without
     * ending the run, so record mcause where the host can find it and park. */
    csrr    t0, mcause
    li      t1, HES_MY_TCDM + HES_MAILBOX_OFFSET + HES_MBOX_TRAP_OFFSET
    sw      t0, 0(t1)
    j       park
