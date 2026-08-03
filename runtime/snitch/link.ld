/* Snitch board linker script.
 * Code in HBM @ 0x8000_0000 (scalar fetch path).
 * All data + stack run in cluster TCDM @ 0x0010_0000 (128 KiB): the Spatz
 * vector LSU ports only reach cluster-local memory, and single-cycle TCDM
 * is exactly the "infinite cache" idealization we want.
 * Data is loaded into HBM (LMA) and copied to TCDM by crt0, because the
 * ELF loader cannot burst-write the banked TCDM.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  ROM  (rx)  : ORIGIN = 0x80000000, LENGTH = 0x10000000
  TCDM (rw)  : ORIGIN = 0x10000000, LENGTH = 0x00020000
}

SECTIONS
{
  .text : {
    KEEP(*(.text.init))
    *(.text*)
  } > ROM

  . = ALIGN(8);
  .data : {
    __data_start = .;
    *(.rodata*) *(.srodata*)
    . = ALIGN(8);
    *(.data*)
    . = ALIGN(8);
    __global_pointer$ = . + 0x800;
    *(.sdata*)
    . = ALIGN(8);
    __data_end = .;
  } > TCDM AT> ROM

  __data_load_start = LOADADDR(.data);

  . = ALIGN(8);
  __bss_start = .;
  .sbss : { *(.sbss*) } > TCDM
  .bss  : { *(.bss*) *(COMMON) } > TCDM
  . = ALIGN(8);
  __bss_end = .;

  /* Heap: everything left in TCDM below a fixed 8 KiB stack. */
  . = ALIGN(16);
  __heap_start = .;
  __heap_end = ORIGIN(TCDM) + LENGTH(TCDM) - 0x2000;
  __stack_top = ORIGIN(TCDM) + LENGTH(TCDM);
  ASSERT(__heap_start <= __heap_end, "TCDM overflow: data+bss too large")
}
