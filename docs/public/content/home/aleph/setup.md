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
order = 3
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
- The Aleph "expansion board", an open-source flight controller based on the STM32H747 microcontroller
- A XT60 to Molex Nano-Fit power cable for connecting a 4S LiPo battery

{% alert(kind="warning") %}
Aleph requires 12-18V DC power (minimum 30W) to operate.
Do not connect a battery or any other power source exceeding 18V.
{% end %}

<img src="/assets/aleph-assemble.jpg" alt="aleph-assemble"/>
<br></br>

Let's go ahead and stack the boards. Align the board-to-board connector and gently press the boards together. The boards should be flush with each other.
Use the provided screws to secure the boards together via the pre-mounted standoffs.

## Powering On

Both the carrier and expansion boards come with software pre-installed.
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

{% alert(kind="info") %}
Sometimes, such as on Ubuntu you may need to grant permissions to the device before connecting with
screen, i.e. `sudo chmod 666 /dev/ttyUSB0`
{% end %}

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

Aleph comes with a variety of sensors pre-installed. The simplest way to access them is through serial connecting to elodin-db using http

```sh
# stream accel data
curl localhost:2248/component/stream/accel/1
#
curl localhost:2248/component/gyro/accel/1
# terminal 3
curl localhost:2248/component/mag/accel/1
```

### Connecting via SSH / Ethernet USB

<img src="/assets/aleph-power-on.jpg" alt="aleph-power-on"/>
<br></br>

After connecting the USB-C cable to the carrier board's USB-C port labeled "Ethernet", you can connect to the carrier board via SSH.

You can check for a successful connection by running this command in your terminal and pinging the well known IP address where Aleph will appear.

```sh
ping fde1:2240:a1ef::1
```

{% alert(kind="notice") %}
Wait approximately 30 seconds for the carrier board to boot up before attempting to connect via SSH.
The Aleph connection will be a new local network connection broadcasting as `aleph.local`.
{% end %}

The carrier board runs a custom Linux distribution with a pre-installed SSH server. You can connect to the carrier board via SSH using the following command:

```sh
ssh root@fde1:2240:a1ef::1
```

### Connect via Elodin Editor

First install Elodin Editor using the instructions in [Quick Start](/home/quickstart#install)

Next launch Elodin. You should be greeted with a startup window. You can connect to Aleph by selecting the "Connect to IP Address" option from the menu and entering `aleph.local:2240` in the IP address field.


<img src="/assets/aleph-connect.png" alt="screenshot of editor connecting to aleph"/>
