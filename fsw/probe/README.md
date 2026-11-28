# Debug probe firmware

Bare-metal firmware for U2, the STM32G0B1KBU6N on USB hub port 2, with no RTOS
or vendor HAL. Derived from the Seward probe firmware. `dap` is the probe image;
`blinky` is a first-light image with RTT and LED cadences.

`dap` enumerates as one composite device:

| Interface | Function |
| --- | --- |
| 0 | CMSIS-DAP v2 (bulk, WinUSB), SWD to the STM32H747 |
| 1 | DFU run-time, detaches the probe into its ROM bootloader |
| 2-3 | CDC-ACM `Aleph FC UART3` (probe USART4) |
| 4-5 | CDC-ACM `Aleph FC UART6` (probe USART5) |

Pins are in [bsp.h](bsp.h) and come from `modules/Debugger/Debugger.zen`.

## Build and flash

From the repository root:

```sh
./firmware/probe/build.sh
./firmware/probe/flash.sh dfu dap   # over USB-C; a blank part boots to ROM DFU
./firmware/probe/tests/run.sh       # host-model tests
```

The build writes both images to `firmware/probe/build/`. It uses a system
`arm-none-eabi-gcc` (GCC 13 or later), or downloads the toolchain on
Apple-silicon macOS and x86-64/AArch64 Linux. USB DFU uses `dfu-util`. SWD
(`flash`, `rtt`, `info`, `optbytes`) uses OpenOCD with an external CMSIS-DAP
probe on the bottom pads TP1 (SWCLK) and TP2 (SWDIO) plus GND. The probe's NRST
is not brought out. Run `./firmware/probe/flash.sh help` for every command.

## Using it

- probe-rs, OpenOCD and pyOCD find the probe by its `CMSIS-DAP` interface
  string. Flash the flight controller with, for example,
  `probe-rs run --chip STM32H747XIHx firmware.elf`.
- The consoles open as ordinary serial ports (`/dev/cu.usbmodem*` on macOS,
  `/dev/ttyACM*` on Linux). Baud rate, parity (none, odd, even), 7 or 8 data
  bits and 1, 1.5 or 2 stop bits follow the host's line coding, from 1200 baud
  to 4 Mbaud. DTR, RTS and breaks are accepted and ignored.
- `DAP_Vendor1` (0x81) controls the STM32H747 BOOT0: payload byte 1 forces it
  high, 0 releases it to R3's pull-down. Force it and pulse reset to start the
  STM32 ROM bootloader. `DAP_Vendor0` (0x80) reports the probe's last reset
  cause, crash count and saved fault PC/LR.

## LEDs

| LED | Meaning |
| --- | --- |
| Indicator (PA4) | 1 Hz heartbeat until the host configures the device, then steady on, flickering with DAP or console traffic |
| RGB amber | Not configured by a host |
| RGB green | Configured |
| RGB magenta | BOOT0 forced high |
| RGB red | Running after a counted crash reset |
| RGB blue | Entering the ROM bootloader |

The RGB LED changes only with state. A frame masks interrupts for about 30 us,
and USART4/5 have no FIFO, so per-transfer flicker stays on the GPIO LED.

## Recovery

- `./firmware/probe/flash.sh dfu dap` detaches a running `dap` into the ROM,
  downloads the image and waits for the probe to return. `detach` enters the
  ROM without flashing. The ROM's USB ID is `0483:df11`.
- A blank part boots the ROM through FLASH_ACR.EMPTY, which only a power
  cycle re-evaluates (RM0444 §2.5). Leaving DFU, the ROM jumps straight into
  the image (AN3156 §5.5); `dap` sets VTOR and clears EMPTY, so the first USB
  flash needs no power cycle. `blinky` leaves EMPTY set, so until a power
  cycle a reset of it returns to the ROM.
- Three consecutive counted fault/watchdog resets enter the ROM. A processed
  non-diagnostic DAP request clears the crash count. The watchdog is nominally
  3 s and freezes while a debugger halts the core.
- If USB recovery fails, use SWD on TP1/TP2, or hold TP1 (PA14-BOOT0) high at
  power-up with nBOOT_SEL cleared to force the ROM. `optbytes` restores
  `nBOOT_SEL=1` afterwards. **`blinky` has no USB; replace it over SWD or by
  forcing the ROM.**

## Limits and open items

- SWD defaults to 1 MHz, caps at 2 MHz and rejects rates below 10 kHz. SWDIO is
  wired straight to the STM32H747, with no translator.
- No SWO capture, target voltage sensing or VBUS sensing (PA7 is unused).
- The USB VID:PID is the pid.codes test ID `1209:0001`, shared with Seward and
  provisional until the Aleph Flight Controller gets its own. The UID-derived
  serial separates probes.
- Not yet run on Aleph Flight Controller hardware. Bring-up still needs: enumeration, probe-rs
  against the STM32H747, both consoles at 115200 and 921600 baud, WL-ICLED
  timing on a scope (T0H 0.3 us, T1H 0.9 us, 1.2 us bit), and DFU round trip.
