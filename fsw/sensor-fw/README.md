# Aleph STM32 Firmware

This is the firmware for the STM32H747 MCU on the Aleph FC. It reads sensor data from the IMU, magnetometer, and barometer at a high rate, and stores it on the SD card. It also streams the data over the board-to-board UART to the Orin NX carrier board.

## Pre-requisites

- [`rust` + `cargo`](https://rustup.rs/)
- [`probe-rs`](https://probe.rs/docs/getting-started/installation/) or [`openocd`](https://openocd.org/)
- [`flip-link`](https://github.com/knurling-rs/flip-link#installation)
- [`defmt-print`](https://github.com/knurling-rs/defmt/tree/main/print) for RTT logging: `cargo binstall defmt-print`

## Flashing instructions

1. Connect Aleph FC to your computer using a USB-C cable.
2. Verify that Aleph FC is detected:
```sh
probe-rs info --protocol swd

# expected output:
ARM Chip with debug port Default:
Debug Port: DPv2, Designer: STMicroelectronics, Part: 0x4500, Revision: 0x0, Instance: 0x00
├── 0 MemoryAP
│   └── ROM Table (Class 1), Designer: STMicroelectronics
│       ├── ROM Table (Class 1), Designer: ARM Ltd
│       │   ├── Cortex-M4 SCS   (Generic IP component)
│       │   │   └── CPUID
│       │   │       ├── IMPLEMENTER: ARM Ltd
│       │   │       ├── VARIANT: 1
│       │   │       ├── PARTNO: Cortex-M7
│       │   │       └── REVISION: 1
│       │   ├── Cortex-M3 DWT   (Generic IP component)
│       │   ├── Cortex-M7 FBP   (Generic IP component)
│       │   └── Cortex-M3 ITM   (Generic IP component)
│       ├── Cortex-M7 ETM   (Coresight Component)
│       └── Coresight Component, Part: 0x0906, Devtype: 0x14, Archid: 0x0000, Designer: ARM Ltd
```
3. Build and flash the firmware from the `fsw/sensor-fw` directory:
```sh
# Option 1: Using probe-rs (quick)
cargo rrb fw

# Option 2: Using OpenOCD with RTT logging (includes live logs)
cargo build --release --bin fw
./openocd/flash.sh target/thumbv7em-none-eabihf/release/fw
```


## Convert binary telemetry data to CSV

1. Connect SD card to your computer.
2. Find the path to the binary file on the SD card (should be something like `/Volumes/<sd_card>/DATA.BIN`).
3. Run the `blackbox` CLI from the `fsw/blackbox` directory:
```sh
cargo run --release -- <path_to_binary_file>
```
4. The CSV output will printed to stdout. You can redirect it to a file by running:
```sh
cargo run --release -- <path_to_binary_file> > <output_file>.csv
```

<details>

<summary>Appendix</summary>

## Debugging and recovery

### Flash manually using `probe-rs`

```sh
# build the firmware:
cargo build --release --bin fw
# flash the firmware:
probe-rs run --chip STM32H747IITx target/thumbv7em-none-eabihf/release/fw
```

### Flash + RTT using `openocd`

```sh
# build the firmware:
cargo build --release --bin fw
# flash the firmware + attach to RTT
../openocd/flash.sh target/thumbv7em-none-eabihf/release/fw --defmt
```

### Erase all internal flash memory

```sh
probe-rs erase --chip STM32H747IITx
```

This is useful if portions of the internal flash are being used by the firmware to store data, and it needs to be cleared.
E.g. Betaflight uses the internal flash to store configuration data.

### Soft reset the MCU over SWD

```sh
probe-rs reset --chip STM32H747IITx
```

### Hard reset the MCU (using the nRST pin)

```sh
probe-rs reset --chip STM32H747IITx --connect-under-reset
```

</details>
