/* Linker script for the CVA6 manager of the mesh_real board.
 *
 * The manager runs out of L2 on the compute die: it is the core the simulation
 * starts, so its image is what the ELF loader writes and where the boot address
 * points (system.L2_BASE). Cluster memory, L3 and HyperRAM are reached as data,
 * never linked into, so they do not appear here.
 *
 * Sizes track targets/hetero/system.py — L2 is 4 MiB.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  L2 (rwx) : ORIGIN = 0x70000000, LENGTH = 0x00400000
}

SECTIONS
{
  .text : {
    KEEP(*(.text.init))
    *(.text*)
  } > L2

  .rodata : { *(.rodata*) *(.srodata*) } > L2

  . = ALIGN(8);
  .data : { *(.data*) } > L2

  . = ALIGN(8);
  __global_pointer$ = . + 0x800;
  .sdata : { *(.sdata*) } > L2

  /* No load-time copy needed: LMA == VMA. crt0 skips the copy loop. */
  __data_start = 0;
  __data_end = 0;
  __data_load_start = 0;

  . = ALIGN(8);
  __bss_start = .;
  .sbss : { *(.sbss*) } > L2
  .bss  : { *(.bss*) *(COMMON) } > L2
  . = ALIGN(8);
  __bss_end = .;

  . = ALIGN(16);
  .heap (NOLOAD) : {
    __heap_start = .;
    . += 0x00100000;    /* 1 MiB heap */
    __heap_end = .;
  } > L2

  . = ALIGN(16);
  .stack (NOLOAD) : {
    . += 0x00010000;    /* 64 KiB stack for the manager */
    __stack_top = .;
  } > L2
}
