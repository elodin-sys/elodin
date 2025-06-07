# Aleph Blinky Bare Metal C Firmware

Simple LED blink firmware for Aleph written in bare metal C.

## Toolchain Setup

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi
```

### macOS (Homebrew)
```bash
brew install --cask gcc-arm-embedded
brew install openocd
```

## Build + Flash

```bash
# Build firmware.elf
./build.sh
# Flash using opnocd
../openocd/flash.sh firmware.elf
```

## Clock Configuration

- **HSE**: 24MHz external oscillator
- **PLL1**: 24MHz ÷ 3 × 100 ÷ 2 = 400MHz (SYSCLK)
- **Power**: LDO enabled, SMPS disabled
