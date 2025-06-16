# Aleph Blinky Bare Metal C Firmware

Simple LED blink firmware for Aleph written in bare metal C.

## Build + Flash

```bash
# Build firmware.elf (downloads toolchain automatically)
./build.sh

# Use system toolchain instead
./build.sh --system

# Flash using openocd
../openocd/flash.sh firmware.elf
```

## Clock Configuration

- **HSE**: 24MHz external oscillator
- **PLL1**: 24MHz ÷ 3 × 100 ÷ 2 = 400MHz (SYSCLK)
- **Power**: LDO enabled, SMPS disabled

## Attributions

- **SEGGER RTT**: RTT (Real-Time Transfer) protocol structures and concepts are based on SEGGER RTT, Copyright (c) 1995 - 2021 SEGGER Microcontroller GmbH. Used under BSD-style license.
