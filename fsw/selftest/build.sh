#!/bin/sh -e
# Builds build/selftest.{elf,bin} with a system arm-none-eabi-gcc.
cd "$(dirname "$0")"
mkdir -p build
CFLAGS="-mcpu=cortex-m7 -mthumb -mfloat-abi=soft -Os -g3 -std=c11 -Wall -Wextra -Werror"
CFLAGS="$CFLAGS -ffreestanding -ffunction-sections -fdata-sections -fno-common -Icmsis"
arm-none-eabi-gcc $CFLAGS -nostartfiles -nostdlib -T linker.ld -Wl,--gc-sections \
    -Wl,--print-memory-usage startup.c selftest.c -lgcc -o build/selftest.elf
arm-none-eabi-objcopy -O binary build/selftest.elf build/selftest.bin
arm-none-eabi-size build/selftest.elf
