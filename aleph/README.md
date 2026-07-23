# Aleph NixOS Configuration

## Install Nix

- [Determinate Nix installer](https://determinate.systems/nix-installer) (recommended)
- [Upstream Nix installer](https://nix.dev/manual/nix/latest/installation/installing-binary#multi-user-installation)

<details>

<summary>Nix Setup</summary>

### Determinate Nix

Add your username to "trusted-users" in `/etc/nix/nix.custom.conf`:
```
# /etc/nix/nix.custom.conf
trusted-users = root <your_username>
```

Restart the nix-daemon:
```sh
# macOS:
sudo launchctl kickstart -k system/systems.determinate.nix-daemon
# Linux:
sudo systemctl restart nix-daemon.service
```

### Upstream Nix

Enable some nix experimental features and add your username to "trusted-users" in `/etc/nix/nix.conf`:
```
# /etc/nix/nix.conf
experimental-features = nix-command flakes
trusted-users = root <your_username>
```

Restart the nix-daemon:
```sh
# macOS:
sudo launchctl kickstart -k system/org.nixos.nix-daemon
# Linux:
sudo systemctl restart nix-daemon.service
```

</details>

## Initial Setup

When you receive Aleph, it comes with NixOS pre-installed. On first login you will be prompted to connect the device to WiFi and create a new user account.

### Connect Aleph to WiFi
First connect to Aleph with right-most USB-C port. This port has an Ethernet gadget enabled that will allow you to SSH into Aleph.

Run `ssh root@fde1:2240:a1ef::1`. The default password is `root`. Once logged in you will be guided through the setup. If the setup does not start, you can manually start it by running `aleph-setup`.

1. **SSH to Aleph**: Log in to Aleph using SSH. The default root password is `root`.
   ```bash
   ssh root@fde1:2240:a1ef::1
   ```
   This IPv6 address is configured in the [modules/usb-eth.nix](modules/usb-eth.nix) file.

2. **Connect to WiFi**: Use [`iwctl`](https://wiki.archlinux.org/title/Iwd#Connect_to_a_network) to connect to your wireless network. The following command will prompt you for your WiFi password:
   ```bash
   iwctl station wlan0 connect "YourNetworkName"
   ```

After connecting to WiFi, Aleph will store your network credentials and reconnect automatically on subsequent boots. You can verify the connection with `ping google.com` or by checking the assigned IP address with `ip addr show wlan0`.

Once connected to WiFi, you can SSH directly to Aleph over your wireless network using its unique `.local` domain name. Find your device's hostname by running the `hostname` command (e.g., `aleph-99a2`), then connect using: `ssh aleph-99a2.local`. The USB Ethernet connection will remain available as a fallback access method with `fde1:2240:a1ef::1` as the static IPv6 address.

[![asciicast](https://asciinema.org/a/716409.svg)](https://asciinema.org/a/716409)

## System Update

This is the recommended development workflow for iterating on the NixOS configuration. It's significantly faster than the fresh install method (described below) and easier to revert in case of mistakes.

1. Ensure you can SSH into Aleph over WiFi (recommended) or via the USB-C port with the Ethernet gadget enabled (right-most USB-C port).

2. Modify the NixOS configuration in `flake.nix`, `kernel/`, or `modules/`.

3. Run `./deploy.sh` to deploy with default settings, or specify a custom `user@host` target:
   ```bash
   # Deploy using default settings ($USER@fde1:2240:a1ef::1)
   ./deploy.sh
   # Deploy using a custom SSH target (also accepts a ssh alias if one is set)
   ./deploy.sh root@aleph-99a2.local
   # Pass raw SSH options with the -o option, for example a ssh key
   ./deploy.sh -o "-i $HOME/.ssh/key_file -o StrictHostKeyChecking=no" root@aleph-99a2.local
   # The deploy script also accepts ssh options via the NIX_SSHOPTS environment variable
   NIX_SSHOPTS="-i $HOME/.ssh/key_file -o StrictHostKeyChecking=no" ./deploy.sh root@aleph-99a2.local
   # Flash a custom configuration with the --config option (defaults to `sensor-fw` STM firmware):
   ./deploy.sh root@aleph-99a2.local -c "base"        # base config; no STM firmware
   ./deploy.sh root@aleph-99a2.local -c "c-blinky"    # base + c-blinky STM FW

   # Show all available options
   ./deploy.sh --help
   ```
   **Note:** the custom configurations accepted by the deploy script are the variables named in
   `customConfigurations` and `nixosConfigurations` in `flake.nix`.

   The deploy script is also packaged and can be run as follows:
   ```
   nix run .#deploy -- -c "base" aleph-99a2.local
   ```
   **Note:** if calling the deploy script this way, you must pass options after the double-hyphen.

The deploy script will:
- Build the NixOS configuration
- Copy all necessary store paths to Aleph
- Activate the new configuration
- Create a new bootloader entry

The default system configuration for Aleph flashes `sensor-fw` STM32 firmware and ensures 
`serial-bridge` starts afterward so expansion-board sensor data can be streamed into Elodin-DB.
To switch to a more "blink sketch", set `aleph.stm.firmware = "c-blinky"` in your configuration.
All STM configurations start a `[firmware-name]-flash` one-shot service.
The service flashes the packaged STM32 firmware from the deployed closure by driving `BOOT0` on 
carrier GPIO9, pulsing `NRST` on carrier GPIO11, and running `stm32flash` against `/dev/ttyTHS1` 
before `serial-bridge` starts.
Once the STM32 boots back into the application, `serial-bridge` forwards the MCU log lines into 
Elodin-DB on the `aleph.stm32.log` message stream. The older custom expansion board reference 
used an I2C expander for reset; the open-source board uses the direct GPIO11 reset path instead.

Useful verification commands on Aleph:

```bash
journalctl -u c-blinky-flash -n 50 --no-pager
journalctl -u serial-bridge -n 50 --no-pager
```

In the Editor, add a `log_stream "aleph.stm32.log" name="STM32 c-blinky"` pane to watch the bridged MCU logs live.

To revert to the previous configuration, reboot and select the previous bootloader entry from the boot menu.

NOTE: The bootloader can only be accessed via the serial console. So, you'll need to switch the USB-C cable to the debugger port (left-most USB-C port).

## Fresh Install / Recovery (initrd flash over USB-C)

This is the recommended way to bring up a bare Aleph or recover an unbootable unit. One command flashes the UEFI bootloader (QSPI) **and** installs NixOS to the M.2 NVMe over the recovery USB-C port — NVIDIA's supported initrd-flash flow for Orin NX + NVMe.

**Host requirements:** x86_64 Linux, Nix with flakes, and a good USB-C cable to the recovery port.

1. Put Aleph into Force Recovery mode:
   - Power off the module.
   - Hold the recovery button while powering on (or use the recovery jumper).
   - Confirm the device appears as NVIDIA APX:
     ```bash
     lsusb | grep -i nvidia
     # expected: 0955:7xxx NVIDIA Corp. APX
     ```

2. Connect the recovery USB-C port to the host (same port used for `flash-uefi`). Optionally attach a serial console on the debug port (leftmost USB-C, 115200) to watch progress.

3. From `aleph/`, run:
   ```bash
   nix run --accept-flake-config .#flash-initrd
   # or: nix build --accept-flake-config .#packages.x86_64-linux.flash-initrd && sudo ./result/flash-initrd
   ```

4. Wait for the script to report success. The device RCM-boots a flashing initrd, writes QSPI firmware, partitions and images the NVMe (ESP + root), then reboots into NixOS.

5. After reboot, SSH in and continue with [Initial Setup](#initial-setup):
   ```bash
   ssh root@fde1:2240:a1ef::1   # password: root
   ```

Subsequent updates use `./deploy.sh` as usual.

<details>

<summary>Legacy: USB / SD card installer</summary>

The USB sd-image installer remains available if initrd flash is not an option. USB mass-storage boot is unreliable on some Aleph units (host controller enumerates but the stick never appears), which is why initrd flash is preferred.

1. Build and flash a USB stick (≥8GB):
   ```bash
   nix build --accept-flake-config .#packages.aarch64-linux.sdimage
   # Identify the stick with lsblk / diskutil, then:
   sudo dd if=result/sd-image/aleph-os.img of=/dev/sdX bs=4M status=progress oflag=sync
   ```
   ⚠️ `dd` can cause **permanent data loss** — double-check the device name.

2. Insert the stick into the middle USB-C port, power on (debug port or DC), and boot from USB (ESC in the serial boot menu if needed).

3. SSH via the Ethernet gadget (rightmost USB-C) and run the installer:
   ```bash
   ssh root@fde1:2240:a1ef::1   # password: root
   aleph-installer
   ```

4. Remove the stick and reboot, then continue with Initial Setup.

</details>

<details>

<summary>Appendix</summary>

## Manual WiFi Setup

1. **Establish Connection**: Connect to Aleph via the right-most USB-C port (which has the Ethernet gadget enabled). This sets up a local network connection between your computer and Aleph over USB.

2. **SSH to Aleph**: Log in to Aleph using SSH. The default root password is `root`.
   ```bash
   ssh root@fde1:2240:a1ef::1
   ```
   This IPv6 address is configured in the [modules/usb-eth.nix](modules/usb-eth.nix) file.

3. **Connect to WiFi**: Use [`iwctl`](https://wiki.archlinux.org/title/Iwd#Connect_to_a_network) to connect to your wireless network. The following command will prompt you for your WiFi password:
   ```bash
   iwctl station wlan0 connect "YourNetworkName"
   ```

After connecting to WiFi, Aleph will store your network credentials and reconnect automatically on subsequent boots. You can verify the connection with `ping google.com` or by checking the assigned IP address with `ip addr show wlan0`.

Once connected to WiFi, you'll be able to SSH directly to Aleph over your wireless network, which is more convenient for ongoing development. The USB Ethernet connection will remain available as a fallback access method.

</details>
