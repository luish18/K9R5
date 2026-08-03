/* Shared linker script for the CVA6 / Snitch / Spatz GVSoC boards.
 * All three map main memory at 0x8000_0000 (CVA6 'mem', Snitch/Spatz HBM).
 * CVA6 boots by fetching at 0x8000_0000, so .text.init is placed first.
 * The simulation idealizes this memory to zero latency ("infinite cache"),
 * so code, data, and stack all live here.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000
}

SECTIONS
{
  .text : {
    KEEP(*(.text.init))
    *(.text*)
  } > RAM

  .rodata : { *(.rodata*) *(.srodata*) } > RAM

  . = ALIGN(8);
  .data : { *(.data*) } > RAM

  /* Small-data sections with gp-relative addressing. */
  . = ALIGN(8);
  __global_pointer$ = . + 0x800;
  .sdata : { *(.sdata*) } > RAM

  /* No load-time copy needed: LMA == VMA. crt0 skips the copy loop. */
  __data_start = 0;
  __data_end = 0;
  __data_load_start = 0;

  . = ALIGN(8);
  __bss_start = .;
  .sbss : { *(.sbss*) } > RAM
  .bss  : { *(.bss*) *(COMMON) } > RAM
  . = ALIGN(8);
  __bss_end = .;

  . = ALIGN(16);
  .heap (NOLOAD) : {
    __heap_start = .;
    . += 0x4000000;     /* 64 MiB heap */
    __heap_end = .;
  } > RAM

  . = ALIGN(16);
  .stack (NOLOAD) : {
    . += 0x40000;       /* 256 KiB stack for hart 0 */
    __stack_top = .;
  } > RAM
}
