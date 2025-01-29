# Aleph NixOS Configuration

## Fresh Install (via USB / SD card)

This is the recommended way to do a fresh install of NixOS on Aleph. This method is also useful for recovering from a broken system.
This method requires a USB flash drive or SD card with at least 8GB of space. The host system must have Nix installed. In the case of macOS, a remote x86_64-linux (or aarch64-linux) builder is required.

1. Build the sd card image with: `nix build .#sdimage` (run from images/aleph).
2. Flash the sd card image to a flash drive (or SD card if using a nano).
3. Plug the flash drive into the middle USB port on Aleph – you can use a USB hub if you need to power and use the flash drive.
4. Connect to Aleph via serial console and boot from the flash drive. This requires pressing ESC during the boot process, then selecting the flash drive from the boot menu.
5. Log in using root:root and run `aleph-installer`.
6. Remove flash drive and reboot Aleph.

## Live System Update

This is the recommended development workflow for iterating on the NixOS configuration. It's significantly faster than the fresh install method and easier to recover from mistakes.

1. Connect to Aleph via the USB-C port with the Ethernet gadget enabled (right-most USB-C port).
2. Modify the NixOS configuration in `flake.nix`, `kernel/`, or `modules/`.
3. Run `deploy.sh`. This copies all necessary store paths to Aleph, activates the new configuration, and creates a new bootloader entry.

To revert to the previous configuration, reboot and select the previous bootloader entry from the boot menu.

NOTE: The bootloader can only be accessed via the serial console. So, you'll need to switch the USB-C cable to the debugger port (left-most USB-C port).
