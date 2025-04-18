# Aleph NixOS Configuration

## Install Nix

- [Determinate Nix installer](https://determinate.systems/nix-installer) (recommended)
- [Upstream Nix installer](https://nix.dev/manual/nix/2.28/installation/installing-binary#multi-user-installation)

NOTE: The rest of the README assumes you're running Determinate Nix on your local machine. If you are running upstream Nix, you will need to modify the steps accordingly.

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

## Initial Setup

When you receive Aleph, it comes with NixOS pre-installed. Setting up network connectivity will make development more convenient, though you can also work directly via USB or Ethernet.

### Connect Aleph to WiFi

1. **Establish Connection**: Connect to Aleph via the right-most USB-C port (which has the Ethernet gadget enabled). This sets up a local network connection between your computer and Aleph over USB.

2. **SSH to Aleph**: Log in to Aleph using SSH. The default root password is `root`.
   ```bash
   ssh root@aleph.local
   ```

   If mDNS resolution fails (common on some networks or operating systems), use the static IPv6 address instead:
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

### User Setup

When you deploy Aleph for the first time, the `deploy.sh` script will automatically prompt you to create your user account if it doesn't already exist on the device.

The script will:
1. Attempt to connect using your current username
2. If that fails, it will offer to create your user account
3. Set up your SSH key for passwordless authentication
4. Add you to the appropriate groups (wheel, video, dialout)
5. Configure sudo access without password

## System Update

This is the recommended development workflow for iterating on the NixOS configuration. It's significantly faster than the fresh install method (described below) and easier to revert in case of mistakes.

1. Ensure you can SSH into Aleph over WiFi (recommended) or via the USB-C port with the Ethernet gadget enabled (right-most USB-C port).

2. Modify the NixOS configuration in `flake.nix`, `kernel/`, or `modules/`.

3. Run `./deploy.sh` to deploy with default settings (current username and "aleph.local"), or specify a custom host/user:
   ```bash
   # Deploy using default settings
   ./deploy.sh
   # Deploy using custom host and username
   ./deploy.sh --host fde1:2240:a1ef::1 --user myuser
   # Don't use Aleph as a remote builder (uses local machine or configured builders instead)
   ./deploy.sh --no-aleph-builder
   # Show all available options
   ./deploy.sh --help
   ```

The deploy script will:
- Check if your user exists on Aleph and create it if needed
- Build the NixOS configuration
- Copy all necessary store paths to Aleph
- Activate the new configuration
- Create a new bootloader entry

To revert to the previous configuration, reboot and select the previous bootloader entry from the boot menu.

NOTE: The bootloader can only be accessed via the serial console. So, you'll need to switch the USB-C cable to the debugger port (left-most USB-C port).

## Fresh Install / Recovery (via USB / SD card)

NOTE: Do not use this method unless you know what you're doing. In most cases, this isn't necessary and you should use the [`System Update`](#system-update) method described above.

This is the recommended way to do a fresh install of NixOS on Aleph. This method is also useful for recovering from a broken system.
This method requires a USB flash drive or SD card with at least 8GB of space. The host system must have Nix installed. In the case of macOS, a remote x86_64-linux (or aarch64-linux) builder is required.

1. Build the sd card image with: `nix build .#sdimage` (run from images/aleph).
2. Flash the sd card image to a flash drive (or SD card if using a nano).
3. Plug the flash drive into the middle USB port on Aleph – you can use a USB hub if you need to power and use the flash drive.
4. Connect to Aleph via serial console and boot from the flash drive. This requires pressing ESC during the boot process, then selecting the flash drive from the boot menu.
5. Log in using `root:root` and run `aleph-installer`.
6. Remove flash drive and reboot Aleph.
