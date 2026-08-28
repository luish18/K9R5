/* Startup for the CVA6 orchestrator of hetero_soc.
 *
 * Unlike runtime/common/crt0.S this does not park on a non-zero mhartid: the
 * host is the only core that runs this image, and its hartid is picked above
 * every cluster hartid (HES_HOST_HARTID) so traces can tell the cores apart.
 */

    .section .text.init, "ax", @progbits
    .globl _start
_start:
    /* Enable the FPU (mstatus.FS); the host has no vector unit. */
    li      t0, (3 << 13)
    csrs    mstatus, t0
    csrw    fcsr, zero          /* round-to-nearest, clear flags */

    /* Unexpected traps end the simulation with a non-zero status. */
    la      t0, trap_vec
    csrw    mtvec, t0

    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    la      sp, __stack_top

    /* Zero .bss. The heap and the stack are deliberately not program headers
     * (see the generated host.ld), so nothing else needs clearing. */
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:
    call    main

    /* Exit through semihosting; a0 = main()'s return value. This ends the
     * whole simulation, cluster cores included. */
    mv      a1, zero
    bnez    a0, 3f
    li      a1, 0x20026        /* ADP_Stopped_ApplicationExit -> status 0 */
3:  li      a0, 0x18           /* SYS_EXIT */
    .option push
    .option norvc
    slli    x0, x0, 0x1f
    ebreak
    srai    x0, x0, 7
    .option pop

    .align 2
trap_vec:
    /* SYS_EXIT with a non-magic a1 -> simulation exit status 1. */
    li      a0, 0x18
    csrr    a1, mcause
    .option push
    .option norvc
    slli    x0, x0, 0x1f
    ebreak
    srai    x0, x0, 7
    .option pop
1:  j       1b
