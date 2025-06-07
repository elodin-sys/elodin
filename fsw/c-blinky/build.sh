#!/bin/sh -e

CFLAGS="-mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16 -Os -g -Wall -Wextra -ffunction-sections -fdata-sections"
LDFLAGS="-T linker.ld -Wl,--gc-sections -Wl,--print-memory-usage"
arm-none-eabi-gcc $CFLAGS $LDFLAGS main.c -o firmware.elf
arm-none-eabi-size firmware.elf
