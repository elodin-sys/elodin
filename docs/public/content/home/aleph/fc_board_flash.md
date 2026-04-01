+++
title = "Flash Expansion Board"
description = "Flash new firmware to the Expansion Board"
draft = false
weight = 105
sort_by = "weight"

[extra]
lead = "Flash and debug firmware on the Expansion Board"
toc = true
top = false
order = 4
icon = ""
+++


The expansion board has a built-in RP2040-based programmer that is pre-flashed with [debugprobe firmware](https://github.com/elodin-sys/debugprobe)
that allows it to function as a CMSIS-DAP probe. This makes it easy to flash new firmware onto the FC board over USB using standard tools such as
[probe-rs](https://probe.rs/docs/getting-started/installation) or [OpenOCD](https://openocd.org/).

### Building the expansion boards default firmware

Clone the `elodin` repository and navigate to the `fsw` directory:

```sh
git clone https://github.com/elodin-sys/elodin.git
cd elodin/fsw
```

The `fsw` directory contains several example projects, navigate into the `sensor-fw` directory & follow the build instructions there. The
last step is the command `cargo rrb fw`, which will build the firmware and flash it to the FC board. You should see the same sensor output as before
when connected to the Base Board, or alternatively can follow the instructions in the `sensor-fw` directory readme to collect sensor data
from the included SD card.

<img src="/assets/aleph-flash-fc.jpg" alt="aleph-flash-fc"/>
<br></br>


### Flashing from the Orin NX via `deploy.sh`

Aleph can also flash the STM32H7 from the Orin NX during a normal NixOS deploy. This is the preferred bring-up path for `fsw/c-blinky` because it packages the firmware into the Aleph closure, deploys it to the target, flashes it over the board-to-board link, and forwards the MCU log output into Elodin-DB.

From the `aleph/` directory:

```sh
nix build --accept-flake-config .#packages.aarch64-linux.toplevel --show-trace
./deploy.sh -h 192.168.4.188 -u root
```

The deploy activates a `c-blinky-flash` one-shot service on Aleph. That service:

1. Uses the packaged `firmware.bin` from the deployed closure
2. Stops `serial-bridge` so `/dev/ttyTHS1` is free for flashing
3. Drives `BOOT0` over carrier GPIO9 and pulses `NRST` over carrier GPIO11
4. Configures `/dev/ttyTHS1` for `19200 8E1`
5. Probes the STM32 ROM UART bootloader with `stm32flash`
6. Writes the application image to `0x08000000`
7. Resets the STM32 back into the flashed application
8. Starts `serial-bridge` again so Linux can ingest the MCU logs

The open-source expansion board uses the direct GPIO11 reset path above. The older custom reference board used an I2C expander for `NRST`, so bring-up notes from that flow need to be translated accordingly.

### Verifying the Flash

On the Aleph host, check the flash service:

```sh
journalctl -u c-blinky-flash -n 50 --no-pager
```

Then confirm the serial bridge is receiving the MCU log frames:

```sh
journalctl -u serial-bridge -n 50 --no-pager
```

If the flash unit reports `Timed out waiting for STM32 bootloader on /dev/ttyTHS1`, the STM32 did not answer the ROM bootloader probe and the board-to-board flashing path still needs hardware-level debugging on that Aleph.

`c-blinky` emits `INFO` log messages such as `c-blinky boot`, `LED: ON`, and `LED: OFF` over the board-to-board UART.

### Viewing Logs in the Editor

`serial-bridge` forwards the MCU lines into Elodin-DB as `LogEntry` messages on the `aleph.stm32.log` stream.

Open the Editor against the Aleph database and add a log panel such as:

```kdl
log_stream "aleph.stm32.log" name="STM32 c-blinky"
```

You should see the `c-blinky boot` message followed by alternating `LED: ON` and `LED: OFF` entries.

### Manual Flashing on Aleph

If you need to bypass deploy-driven flashing and run the tool directly on the Orin NX:

```sh
sudo flash-mcu --elf <elf path>
# or, if you already have a raw binary:
sudo flash-mcu --bin <bin path>
```
