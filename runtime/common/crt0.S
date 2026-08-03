/* Bare-metal startup for GVSoC CVA6 / Snitch / Spatz targets.
 * Hart 0 runs main(); all other harts park in a wfi loop.
 */

#if __riscv_xlen == 64
#define STORE sd
#define WORDSIZE 8
#else
#define STORE sw
#define WORDSIZE 4
#endif

    .section .text.init, "ax", @progbits
    .globl _start
_start:
    /* Park secondary harts. */
    csrr    t0, mhartid
    bnez    t0, park

    /* Enable FPU (mstatus.FS) and vector unit (mstatus.VS) if present. */
    li      t0, (3 << 13) | (3 << 9)
    csrs    mstatus, t0
    csrw    fcsr, zero          /* round-to-nearest, clear flags */

    /* Unexpected traps end the simulation with exit code 2. */
    la      t0, trap_vec
    csrw    mtvec, t0

    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop

    la      sp, __stack_top

    /* Copy .data from load address to run address if they differ
     * (used on Spatz, where data runs in TCDM but is loaded in HBM). */
    la      t0, __data_load_start
    la      t1, __data_start
    la      t2, __data_end
    beq     t0, t1, 4f
5:  bgeu    t1, t2, 4f
#if __riscv_xlen == 64
    ld      t3, 0(t0)
#else
    lw      t3, 0(t0)
#endif
    STORE   t3, 0(t1)
    addi    t0, t0, WORDSIZE
    addi    t1, t1, WORDSIZE
    j       5b
4:
    /* Zero .bss */
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    STORE   zero, 0(t0)
    addi    t0, t0, WORDSIZE
    j       1b
2:
    call    main

    /* Exit through semihosting; a0 = main()'s return value. */
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

park:
    wfi
    j       park

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
