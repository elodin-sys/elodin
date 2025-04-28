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
[probe-rs](https://probe.rs/docs/getting-started/installation) or [OpenOCD](https://openocd.org/). We recommend using probe-rs because it has less of a configuration burden, and
has out-of-the-box support for RTT (Real-Time Transfer) logging.

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


### Flashing from the Orin NX


You can flash the STM32H7 from the Orin NX. This is useful if you are pushing out new firmware updates to the STM32 over the air. In an Orin NX terminal session run the following command. Replace `<elf path>` with the path to your firmware payload in the ELF file format

```
sudo flash-mcu --elf <elf path>
```
