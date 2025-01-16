+++
title = "Aleph Setup"
description = "Unbox and setup your Aleph Flight Computer."
draft = false
weight = 104
sort_by = "weight"

[extra]
lead = "Unbox and setup your Aleph Flight Computer for development"
toc = true
top = false
order = 4
icon = ""
+++

<img src="/assets/aleph.jpg" alt="aleph"/>
<br></br>

Thanks for purchasing an Aleph Flight Computer! This guide will help you get started with your new device. We'll cover:
- Turning on and connecting to Aleph via (USB) serial or SSH
- General development workflow such as flashing custom firmware
- Accessing sensor data from the compute environment

For more detailed information on the hardware (including pinouts, sensor suite, connectors, etc.), please refer to the [ES01](/ES01) datasheet.

## Unboxing & Assembly

<img src="/assets/aleph-unboxing.jpg" alt="aleph-unboxing"/>
<br></br>

Inside the box you'll find (from left to right):
- The Aleph "carrier board", an AI-capable computer including a pre-mounted Nvidia Orin NX SoM
- The Aleph "flight controller (FC) board", an open-source flight controller based on the STM32H747 microcontroller
- An XT60 to Molex Nano-Fit power cable for connecting a 4S LiPo battery

{% alert(kind="warning") %}
Aleph requires 12-18V DC power (minimum 30W) to operate.
Do not connect a battery or any other power source exceeding 18V.
{% end %}

<img src="/assets/aleph-assemble.jpg" alt="aleph-assemble"/>
<br></br>

Let's go ahead and stack the boards. Align the board-to-board connector and gently press the boards together. The boards should be flush with each other.
Use the provided screws to secure the boards together via the pre-mounted standoffs.

## Powering On

Both the carrier and FC boards come with software pre-installed.
During normal development, you'll use 2 USB-C cables: one to power the board, and the other to connect to your laptop:

1. Connect "USB-C Power" to a 30W+ USB-PD power supply.
2. Either:
    - Connect "USB-C w/ Serial" to your laptop for serial console access.
    - **OR** Connect "USB-C w/ Ethernet" to your laptop for an ethernet connection to the carrier board, allowing SSH access

<img src="/assets/aleph-usb-con.jpg" alt="aleph-usb-con"/>
<br></br>

Go ahead and connect your USB-C cables now. Aleph will power on automatically when connected to power. If the fan doesn't spin, it may be in manual mode.
There is a switch on the underside of the carrier board to toggle between manual and automatic power-on. Toggle to automatic and reconnect the power.

{% alert(kind="info") %}
Not all high wattage USB-C power supplies support USB-PD.
For example, some USB-C laptop chargers provide 50W+ but use a proprietary protocol incompatible with USB-PD.
Make sure to use a USB-C power supply that explicitly advertises USB-PD support.
You can also use a lab power supply set to 12-18V DC (minimum 30W).
{% end %}

{% alert(kind="warning") %}
Make sure to connect the power cable first.
Otherwise, the laptop will attempt to power the board over the data cable, but it cannot provide enough wattage
for the Orin module to boot successfully. This will cause the board to power cycle repeatedly.
{% end %}

### Connecting via Serial USB

You can connect to Aleph and observe the boot process by connecting a USB-C cable to the carrier board's USB-C port labeled "Serial". Once connected,
open a terminal and run the following command to observe the boot process:

**Linux/macOS**
```sh
# find the newly connected USB serial device
ls -l /dev/tty*
# connect to the device (for example)
screen /dev/tty.usbserial-DK0FQC0Q 115200
```

**Windows**
```sh
# find the newly connected USB serial device
# open device manager and look for the new COM port
# connect to the device (for example)
putty COM3 115200
```

After booting completes, you should see the login prompt.
You can log in with the default credentials:
- Username: `root`
- Password: `root`

{% alert(kind="info") %}
Aleph ships with root access enabled by default. We recommend disabling root access and enabling passwordless SSH keys for security.
{% end %}

### Accessing Sensor Data

Aleph has a variety of sensors available for use in your applications. You can access the sensor data from the compute environment by
running the following commands in your terminal:

```sh
# terminal 1
cat /sensors/accel
# terminal 2
cat /sensors/gyro
# terminal 3
cat /sensors/mag
```

This data is also written to the provided SD card on the FC board. See [Develop Firmware for FC Board](#develop-firmware-for-fc-board) section below.

### Connecting via SSH / Ethernet USB

<img src="/assets/aleph-power-on.jpg" alt="aleph-power-on"/>
<br></br>

After connecting the USB-C cable to the carrier board's USB-C port labeled "Ethernet", you can connect to the carrier board via SSH.

You can check for a successful connection by running this command in your terminal and looking for the Aleph's IP address:

```sh
ifconfig | grep broadcast
```

{% alert(kind="notice") %}
Wait approximately 30 seconds for the carrier board to boot up before attempting to connect via SSH.
The Aleph connection will be a new local network connection broadcasting as `aleph.local`.
{% end %}

The carrier board runs a custom Linux distribution with a pre-installed SSH server. You can connect to the carrier board via SSH using the following command:

```sh
ssh root@aleph.local
```

{% alert(kind="info") %}
Sometimes the device will connect as an unresolved IP address at a `169.254.x.x` address. If this happens, you can connect to the device by
manually updating the DHCP settings on your computer to use a static IP address of `10.224.0.2` with this device instead. The device server
will be at `10.224.0.1`
{% end %}

## Development Workflows

Software development for Aleph consists of:
- Building and installing an OS image onto the carrier board
- Developing AI/ML workloads for the carrier board
- Developing firmware for the FC board

### Install OS on Carrier Board

*Coming Soon*

### Develop AI/ML Workloads for Carrier Board

*Coming Soon*

### Develop Firmware for FC Board

The FC board has a built-in RP2040-based programmer that is pre-flashed with [debugprobe firmware](https://github.com/elodin-sys/debugprobe)
that allows it to function as a CMSIS-DAP probe. This makes it easy to flash new firmware onto the FC board over USB using standard tools such as
[OpenOCD](https://openocd.org/) or [probe-rs](https://probe.rs/). We recommend using probe-rs because it has less of a configuration burden, and
has out-of-the-box support for RTT (Real-Time Transfer) logging.

{% alert(kind="info") %}
Probe-rs v0.25.0 has a regression that prevents it from working well with the STM32H7 series. We recommend using v0.24.0 until the issue is resolved.
{% end %}

Install probe-rs v0.24.0 using the following install scripts:

**Linux/macOS**
```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/probe-rs/probe-rs/releases/tag/v0.24.0/download/probe-rs-tools-installer.sh | sh
```

**Windows**
```sh
irm https://github.com/probe-rs/probe-rs/releases/tag/v0.24.0/download/probe-rs-tools-installer.ps1 | iex
```

Clone the `elodin` repository and navigate to the `fsw` directory:

```sh
git clone https://github.com/elodin-sys/elodin.git
cd elodin/fsw
```

The `fsw` directory contains several example projects, navigate into the `multicopter` directory & follow the build instructions there. The
last step is the command `cargo rrb fw`, which will build the firmware and flash it to the FC board. You should see the same sensor output as before
when connected to the Base Board, or alternatively can follow the instructions in the `multicopter` directory readme to collect sensor data
from the included SD card.

<img src="/assets/aleph-flash-fc.jpg" alt="aleph-flash-fc"/>
<br></br>

## Reset to Factory Settings

### Carrier Board (AI)
*Coming Soon*

### Flash FC Board's RP2040

This is easiest to perform when separated from the stack to provide access to the bootloader button.

1. While pressing the bootloader button, connect Aleph FC to your computer and confirm that all 3 red power LEDs (labelled “USB”, “5v0”, and “3v3”) are on.

2. Your computer should recognize a new drive labeled "RPI-RP2".

3. Download the debugger UF2 firmware file [here](https://storage.googleapis.com/elodin-releases/debugger/debugprobe.uf2).

4. Drag and drop the UF2 firmware file onto the "RPI-RP2" drive. This flashes the necessary firmware onto the built-in debugger, and resets the board.
Confirm that the “RG0” green LED is now on.

5. Continue to re-flash the firmware as needed using the development process described above.

## Betaflight

Aleph FC is compatible with Betaflight firmware.

### Betaflight Installation

Download the patched Betaflight firmware [here](https://storage.googleapis.com/elodin-releases/betaflight/4.5/betaflight_STM32H743_ALEPH_FC.elf)

{% alert(kind="notice") %}
the Betaflight patches needed to support Aleph are trivial and [open source](https://github.com/betaflight/betaflight/compare/4.5-maintenance...elodin-sys:betaflight:4.5/aleph).
{% end %}

Flash the Betaflight firmware using the following command:

```sh
probe-rs run --chip STM32H747IITx betaflight_STM32H743_ALEPH_FC.elf
```

Keep Aleph FC plugged into your computer, and visit [app.betaflight.com](http://app.betaflight.com).

{% alert(kind="notice") %}
The Betaflight web app requires a Chromium-based browser (Chrome, Chromium, Edge) because it uses Web Serial API, which is not well supported on other browsers.
{% end %}

Go to the `Options` tab and enable the `Show all serial devices` option.

<img src="/assets/betaflight-1.jpg" alt="betaflight-1"/>
<br></br>

Click on `Select your device` in the top-right corner, and choose `I can’t find my USB device`.

<img src="/assets/betaflight-2.jpg" alt="betaflight-2"/>
<br></br>

Choose `Aleph FC Debug Probe` from the list of serial port devices and click on `Connect`.

<img src="/assets/betaflight-3.jpg" alt="betaflight-3"/>
<br></br>

In some cases, you may have to click `Connect` again in the top-right corner to connect to the FC.

<img src="/assets/betaflight-4.jpg" alt="betaflight-4"/>
<br></br>

The betaflight configurator should now be fully connected to the FC.

<img src="/assets/betaflight-5.jpg" alt="betaflight-5"/>
<br></br>

### Reference Quadcopter Design

We've tested Aleph FC with Betaflight on a 7" quadcopter design.

<img src="/assets/aleph-drone.jpg" alt="aleph-drone"/>
<br></br>

If you'd like to replicate our quadcopter design, the shopping list includes:
- 4x V2306 V3.0 KV1750 motors [link](https://www.getfpv.com/t-motor-velox-v2306-v3-motor-1500kv-1750kv-1950kv-2550kv.html)
- Generic 1300 mAh 3S Battery [link](https://www.getfpv.com/lumenier-1300mah-3s-35c-lipo-battery-xt60.html)
- Generic 7" Carbon Fiber Frame [link](https://pyrodrone.com/products/hyperlite-floss-3-0-long-range-7-frame)
- Generic 3-blade propellers [link](https://www.getfpv.com/gemfan-hurricane-5536-3-blade-propeller-set-of-4.html)
- RadioMaster Pocket ELRS controller [link](https://www.getfpv.com/radiomaster-pocket-radio-cc2500-elrs-2-4ghz.html)
- RadioMaster ELRS receiver [link](https://www.amazon.com/RadioMaster-Receiver-ExpressLRS-Antenna-Transmitter/dp/B0BZY2M4BS/)
- XT60 pass-through power module [link](https://holybro.com/collections/power-modules-pdbs/products/pm02-v3-12s-power-module)
- Generic 4-in-1 30x30 ESC [link](https://www.getfpv.com/speedybee-60a-3-6s-blheli-s-4-in-1-esc-30x30.html)

<img src="/assets/aleph-drone-pins.jpg" alt="aleph-drone-pins"/>
<br></br>
