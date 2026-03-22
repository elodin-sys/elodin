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

## Aleph Deploy Flow

The Aleph NixOS configuration packages this firmware as `firmware.elf` and
`firmware.bin`, deploys it to the Orin NX, and flashes it with the
`c-blinky-flash` systemd unit during `./deploy.sh`.

The deploy-time flash path uses the STM32 ROM UART bootloader over the
board-to-board link. On the open-source Aleph expansion board, `flash-mcu`
drives `BOOT0` from carrier GPIO9, pulses `NRST` from carrier GPIO11,
configures `/dev/ttyTHS1` for `19200 8E1`, and writes `firmware.bin` with
`stm32flash`. The older custom-board reference used an I2C expander for
`NRST`, but that reset path does not match the open-source hardware.

After flashing, `c-blinky` emits COBS-framed UART log messages on the
board-to-board link. `serial-bridge` forwards those messages into Elodin-DB on
the `aleph.c-blinky.log` stream, which can be viewed in an Editor
`log_stream` pane.

## Clock Configuration

- **HSE**: 24MHz external oscillator
- **PLL1**: 24MHz ÷ 3 × 100 ÷ 2 = 400MHz (SYSCLK)
- **Power**: LDO enabled, SMPS disabled

## Runtime Behavior

- Boots the STM32H7 from the Orin-deployed payload after UART-bootloader flashing
- Toggles the red LED every 500ms
- Emits `c-blinky boot`, `LED: ON`, and `LED: OFF` log lines at `INFO` level
