+++
title = "Factory Reset"
description = "Reset your Aleph to factory settings."
draft = false
weight = 105
sort_by = "weight"

[extra]
lead = "Reset your Aleph to factory settings"
toc = true
top = false
order = 5
icon = ""
+++


## Carrier Board

If you have broken the OS image beyond repair, or are simply replacing the SSD. These steps will install a fresh OS onto the SSD.

### Requirements
- A flash drive with at least 16GB of storage
- An adapter from USB A to USB C - not necessary if you are using a USB C flash drive

### Flash USB Drive

First download the OS image: [aleph-2025-03-16.img]( https://storage.googleapis.com/elodin-releases/aleph/aleph-2025-03-16.img)

Then you will need to write this image onto your flash drive. There are a number of methods to do this. The two easiest (in ascending order of difficulty) are:

1. Using balenaEtcher - https://etcher.balena.io - This is the recommended way if it works on your machine
2. Using dd - `dd if=<path to img> of=<path to your usb drive> bs=8M status=progress` - Use this with caution you can easily overwrite an entire disk you care about

### HW Setup

Now that you have a flashed USB drive, you will need to plug it into the middle USB port on Aleph. Then you will need to power Aleph using either the DC power port (rated for 12-16V) or through the USB C port. If you have a standard USB C dock, you can plug the USB drive into that dock, and then the power into the dock. That will allow you to power Aleph off of the USB C port while using the USB drive.

It is easiest to connect to Aleph through the serial port for the next steps, though if that is not an option you can do this over the Ethernet or USB C Ethernet gadget as well.

### Boot Device Selection

{% alert(kind="notice") %}
You can only select the boot device over serial port. If you are using ssh, skip ahead to the install commands
{% end %}

While connected to the serial port, hit the reset button on Aleph. After some time, you will see a prompt with the below text:

```
Jetson UEFI firmware (version v35.3.1 built on 2023-01-24T13:18:32+00:00)
ESC   to enter Setup.
F11   to enter Boot Manager Menu.
Enter to continue boot.
**  WARNING: Test Key is used.  **
```

You want to enter the Boot manager by hitting F11. From there if the USB drive is plugged in correctly you should be able to select it.

### OS Install

Aleph should put into a NixOS install. Once it is booted login with `root:root`.

Once logged in you can run `aleph-installer` which will flash the OS image to your internal SSD.

## Expansion Board RP2040 Debugger

This is easiest to perform when separated from the stack to provide access to the bootloader button.

1. While pressing the bootloader button, connect the Expansion Board to your computer and confirm that all 3 red power LEDs (labeled “USB”, “5v0”, and “3v3”) are on.
2. Your computer should recognize a new drive labeled "RPI-RP2".
3. Download the debugger UF2 firmware file [here](https://storage.googleapis.com/elodin-releases/debugger/debugprobe.uf2).
4. Drag and drop the UF2 firmware file onto the "RPI-RP2" drive. This flashes the necessary firmware onto the built-in debugger, and resets the board.
Confirm that the “RG0” green LED is now on.
5. Continue to re-flash the firmware as needed using the development process described above.
