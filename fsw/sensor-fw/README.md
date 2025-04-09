# Aleph STM32 Firmware

## Pre-requisites

- [`rust` + `cargo`](https://rustup.rs/)
- [`probe-rs`](https://probe.rs/docs/getting-started/installation/) v0.24 (recommended) or [`openocd`](https://openocd.org/)
- [`flip-link`](https://github.com/knurling-rs/flip-link#installation)

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
cargo rrb fw
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

### Flash manually using `openocd`

```sh
# build the firmware:
cargo build --release --bin fw
# flash the firmware:
openocd -f openocd/aleph.cfg -c "program target/thumbv7em-none-eabihf/release/fw verify reset exit"
```

### Attach to MCU's RTT output

The same ELF file that was flashed to the MCU must be provided to `probe-rs` in order to attach to the RTT output.
This is because the firmware uses [`defmt`](https://github.com/knurling-rs/defmt) for RTT logging, which uses deferred formatting and string compression.

```sh
probe-rs attach --chip STM32H747IITx --log-format "{L} {s} @ {F}:{l}" target/thumbv7em-none-eabihf/release/fw
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
