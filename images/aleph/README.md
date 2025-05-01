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

3. Run `./deploy.sh` to deploy with default settings, or specify a custom host/user:
   ```bash
   # Deploy using default settings ($USER@fde1:2240:a1ef::1)
   ./deploy.sh
   # Deploy using custom host
   ./deploy.sh --host aleph-99a2.local
   # Show all available options
   ./deploy.sh --help
   ```

The deploy script will:
- Build the NixOS configuration
- Copy all necessary store paths to Aleph
- Activate the new configuration
- Create a new bootloader entry

To revert to the previous configuration, reboot and select the previous bootloader entry from the boot menu.

NOTE: The bootloader can only be accessed via the serial console. So, you'll need to switch the USB-C cable to the debugger port (left-most USB-C port).

## Fresh Install / Recovery (via USB / SD card)

This method installs a minimal base NixOS image on Aleph, returning the device to its factory state. It's useful primarily for recovery when the system becomes unbootable or severely corrupted. This method requires a USB drive with at least 8GB of space.

1. Download the latest OS image and decompress it.
   ```bash
   # This convenience script just runs:
   # curl -L https://storage.googleapis.com/elodin-releases/latest/aleph-os.img.zst | zstd -d > aleph-os.img
   ./justfile download-sdimage
   ```

2. Flash the image to a USB drive.

    ⚠️ The `dd` command can cause **PERMANENT DATA LOSS** if used incorrectly. Double-check your device name before proceeding.

    - Identify your USB drive's device name:
        - **Linux:** Run `lsblk` and look for your USB drive (e.g., `/dev/sdb`, `/dev/sdc`).
        - **macOS**: Run `diskutil list` and identify your USB drive (e.g., `/dev/disk2`). For better performance, use the raw device path (e.g., `/dev/rdisk2`).
    - Unmount the USB drive:
        - **Linux:** `sudo umount /dev/sdX*` (replace `/dev/sdX` with your device name)
        - **macOS:** `sudo diskutil unmountDisk /dev/diskX` (replace `/dev/diskX` with your device identifier)
    - Write the image to the USB drive:
      ```bash
      # Replace /dev/sdX with your actual device name
      sudo dd if=aleph-os.img of=/dev/sdX bs=4M status=progress oflag=sync
      ```
    - Safely umount and remove the USB drive.

3. Boot Aleph from the USB drive.
    - Insert the USB drive into the middle USB-C port on Aleph.
    - Power Aleph using the debug USB-C port (leftmost port) or using the DC power connector.
    - The UEFI firmware should automatically boot from USB. If not, access the boot menu by connecting via serial console and pressing ESC during boot.

4. Connect to Aleph and run the installer.
    - Connect to Aleph using the rightmost USB-C port (Ethernet gadget).
    - SSH into Aleph (password: `root`).
      ```bash
      ssh root@fde1:2240:a1ef::1
      ```
    - Run the installer script and follow the prompts.
      ```bash
      aleph-installer
      ```

5. Remove the USB drive and reboot Aleph.

After rebooting, you can re-establish SSH connectivity and proceed with the initial setup as described in the earlier sections.

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
