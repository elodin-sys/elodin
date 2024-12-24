# Aleph MultiCopter Flight Software

## Pre-requisites

Install the following tools:
- [probe-rs](https://probe.rs/)
- [flip-link](https://github.com/knurling-rs/flip-link#installation)

## Flashing instructions

NOTE: Run the following commands from the `paracosm/fsw/multicopter` directory.

1. Connect the Aleph FC to your computer using a USB cable.
2. Verify that the FC is detected by running `probe-rs info` in the terminal.
3. Build and flash the firmware by running `cargo rrb fw` in the terminal.

## Convert binary telemetry data to CSV

1. Connect SD card to your computer.
2. Find the path to the binary file on the SD card (should be something like `/Volumes/<sd_card>/ALEPH/BLACKBOX/TEST/DATA.BIN`).
3. Run the `blackbox` CLI from the `paracosm/fsw/blackbox` directory: `cargo run --release -- <path_to_binary_file>`.
4. The CSV output will printed to stdout. You can redirect it to a file by running `cargo run --release -- <path_to_binary_file> > <output_file>.csv`.
