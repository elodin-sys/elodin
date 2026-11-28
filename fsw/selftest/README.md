# STM32H747 self-test

Bare-metal bring-up test for the flight controller MCU (M7 core). It checks the
MCU, clocks and every onboard part it can reach without external hardware, and
prints one line per check.

## Run it

Flash the debug probe first (`firmware/probe`), then, from the repository root:

```sh
./firmware/selftest/build.sh
./firmware/selftest/flash.sh
```

Open the probe's `Aleph FC UART3` console at 115200 8N1 (for example
`screen /dev/cu.usbmodem*3 115200`) and press `r` to rerun. The blue LED is lit
while the test runs; afterwards green blinks if nothing failed and red blinks
if something did. A fault stops with the red LED on and `[FAULT]` on the
console.

`build.sh` needs `arm-none-eabi-gcc`; `flash.sh` needs `probe-rs`.

## Checks

| Check | Pass condition |
| --- | --- |
| Device ID, flash size | DEV_ID 0x450, 2048 KB; silicon revision reported |
| Readout protection | RDP level 0; BCM7/BCM4 boot options reported |
| Supply | PWR_CR3 LDO on, SMPS off, ACTVOSRDY |
| HSE, PLL1, LSE | HSE ready in 100 ms, PLL1 locks from HSE, LSE ready in 3 s |
| Pulled control lines | SENS1/SENS2/SD/P5V_AUX enables, ELRS BOOT/EN, INS and F-RAM CS read high |
| I2C1..I2C3 | Lines idle high; EEPROM ACK at 0x50; BMI270 0x24, BMM350 0x33, BMP581 0x50 (or BMP384 0x50 at 0x76) |
| I2C3 lines | Rise time, SCL/SDA crosstalk, per-bit readback of both sensor addresses and a full scan, informational |
| I2C4 (J7) | Scan, informational |
| F-RAM (SPI4) | RDID `7F7F7F7F7F7FC22208` (revision field free); last byte written, read back and restored |
| BMI270 (SPI5) | Chip ID 0x24 |
| SENS2 rail (PJ5) | A written ACC_RANGE returns to its reset value after a 100 ms off cycle |
| PC2/PC3 switches | SYSCFG_PMCR PC2SO and PC3SO set, so SPI2 MISO/MOSI are separate from the PC2_C/PC3_C analog pads |
| IMX-6 (SPI2) | One-shot `DID_DEV_INFO` request answered within 3 s: hardware type IMX, running its application; serial number, hardware and firmware versions reported. No reply at all is a warning, since most boards have U26 unfitted |
| IMX-6 IMU | One `DID_PIMU` sample: accelerometer magnitude 900-1100 mg with the board at rest; gyro rate and IMU status reported |
| IMX-6 data ready | `DID_FLASH_CONFIG.ioConfig` read back. If G9 data ready is enabled (`IO_CONFIG_G9_SPI_DRDY`, bits 5:4 = 3), PA6 must be low at idle, high while a reply waits and low once it has been read. A warning if it isn't enabled: the factory default puts data ready on G13, which isn't connected |
| ADC | VDDA from VREFINT 3.1-3.5 V; VIN 4.5-21 V; VBAT 12-16.8 V if a pack is fitted; P5V_AUX current reported |
| Die temperature | Internal sensor from TS_CAL1 (30 C) and TS_CAL2 (130 C), 0-85 C |
| RTC backup cell | BT1 through the internal VBAT/4 channel, 2.0-3.4 V |
| ExpressLRS | Download boot with PG5 low, then normal boot after PG6 reset, both read from the ESP32-C3 ROM's `rst:...,boot:...` line on UART4 |
| CAN1, CAN2 | TXD driven low comes back low on RXD (U13, U12), and the other bus stays recessive |
| UART6 | Sends `Aleph FC UART6 ping` and expects `pong` back from a host on the probe's `Aleph FC UART6` console within 2 s; a warning if nothing answers |
| EEPROM write | Last byte (0xFFFF) written with its complement, read back and restored |
| Connector and microSD pins | The 36 JST GH and microSD signal pins follow the MCU pulls both ways (except the board pulls on CTS, UART8 TX and I2C4), and driving each one moves no other |
| Carrier translators | U8/U10 drive their four STM32-side outputs (SPI1 SCK/MOSI/CS, UART1 RX) with no carrier attached, so P1V8 and OE are up |
| LEDs | Red, green, blue for 0.4 s each, checked by eye |

Not covered: USB, CAN traffic, microSD cards, the P5V_AUX voltage and open
connector pins, which need external hardware or a driver stack. Ethernet has no
STM32 side to test: the RMII pins are not routed off the MCU.

For the UART6 check, keep a terminal on the `Aleph FC UART6` console and type
`pong` when the ping arrives, or script the reply.

## Design

- The supply leaves Run* mode on the LDO before anything else, as ST's
  `ExitRun0Mode` does for `USE_PWR_LDO_SUPPLY`. VDDSMPS is grounded, and
  PWR_CR3 is write-once per power cycle.
- The system clock stays on the 64 MHz HSI. HSE and PLL1 are only started to
  prove they work, so a bad crystal can't stop the console.
- Sensors and memories are bit-banged on GPIOs, so a failure points at the
  part, its rail or its wiring rather than a peripheral setup.
- The SENS2 test holds CS, SCK and MOSI low while the rail is off, as the spec
  requires. PA6 (IMX-6 G9, the module's SPI strap) is only ever an input, and
  the IMX-6 SPI2 lines float outside the INS test, so cycling P3V3_SENS1 can't
  back-power the module.
- The IMX-6 is bit-banged in SPI mode 3 at about 200 kHz with Inertial Sense
  binary (ISB) packets. Requests are one-shot, so the test leaves the module's
  streaming configuration alone. It doesn't send a stop-broadcasts command, and
  it skips packets for other data sets. A module that has just powered up gets
  3 s to boot; press `r` if it needs longer.

Register names come from ST's CMSIS header (`cmsis/`, cmsis-device-h7 v1.10.8
and Arm CMSIS 5.9.0, Apache-2.0). Pin functions come from ST's
STM32_open_pin_data; analog pads and calibration addresses from DS12930 Rev 3
and ST's H7 LL/HAL drivers. Device IDs come from the BMI270 (BST-BMI270-DS000),
BMM350 (BST-BMM350-DS001), BMP581 (BST-BMP581-DS004), BMP384
(BST-BMP384-DS003) and FM25V02A (Cypress 001-90865 Rev *H) datasheets. The IMX-6
SPI timing and framing come from Inertial Sense's SPI protocol page
(docs.inertialsense.com) and the packet, `dev_info_t` and `pimu_t` layouts from
inertial-sense-sdk (`src/ISComm.h`, `src/data_sets.h`); the request bytes match
its `ExampleProjects/SPI_RPi5` examples. The
ESP32-C3 boot line format and 115200 baud are from Espressif's esptool
documentation.
