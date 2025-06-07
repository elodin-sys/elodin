MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 2M
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 128K
}

ENTRY(Reset_Handler)

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } >FLASH

    .text : {
        *(.text*)
        *(.rodata*)
        _etext = .;
    } >FLASH

    .data : {
        _sdata = .;
        *(.data*)
        _edata = .;
    } >RAM AT> FLASH
    
    _sidata = LOADADDR(.data);

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } >RAM

    _stack_end = ORIGIN(RAM) + LENGTH(RAM);
}
