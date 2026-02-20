---
name: elodin-aleph
description: Deploy and configure AlephOS on flight computers, write flight software services, and manage NixOS modules for the Aleph platform. Use when working with aleph/, deploying to Jetson Orin hardware, writing NixOS modules, flashing firmware, or composing a flight software stack.
---

# Elodin Aleph

Aleph is an NVIDIA Jetson Orin-based flight computer running NixOS. The `aleph/` directory contains NixOS modules, packages, and deployment tooling for composing a Linux flight software stack.

## Quick Reference

```bash
# SSH into Aleph (USB-C rightmost port)
ssh root@fde1:2240:a1ef::1          # Default password: root

# SSH over WiFi (after setup)
ssh aleph-99a2.local                 # Hostname shown by `hostname` command

# Deploy configuration changes
./deploy.sh                          # Default: $USER@fde1:2240:a1ef::1
./deploy.sh --host aleph-99a2.local  # Custom host

# Build SD image for fresh install
nix build --accept-flake-config .#packages.aarch64-linux.sdimage
```

## Initial Setup

1. Connect USB-C to rightmost port on Aleph
2. `ssh root@fde1:2240:a1ef::1` (password: `root`)
3. Follow `aleph-setup` wizard (or run it manually)
4. Connect to WiFi: `iwctl station wlan0 connect "NetworkName"`
5. Verify: `ping google.com`

After WiFi setup, SSH directly via `aleph-XXXX.local`.

## Development Workflow

The recommended cycle for iterating on the NixOS configuration:

1. Edit files in `aleph/flake.nix`, `aleph/modules/`, or `aleph/kernel/`
2. Run `./deploy.sh` from the `aleph/` directory
3. The script builds, copies store paths, and activates the new config
4. To revert: reboot and select previous bootloader entry (requires serial console on left USB-C port)

## NixOS Module System

Configuration is in `aleph/flake.nix`, composed from modules in `aleph/modules/`:

### Hardware & Base
| Module | Purpose |
|--------|---------|
| `hardware.nix` | Forked kernel, device tree for Orin |
| `fs.nix` | SD-card image compatibility |
| `usb-eth.nix` | USB Ethernet gadget (IPv6 `fde1:2240:a1ef::1`) |
| `wifi.nix` | WiFi via iwd |
| `minimal.nix` | Minimal NixOS profile |
| `aleph-base.nix` | Base configuration |
| `aleph-dev.nix` | Dev packages (CUDA, OpenCV, git) |

### Flight Software Services
| Module | Purpose |
|--------|---------|
| `elodin-db.nix` | Elodin-DB telemetry database service |
| `aleph-serial-bridge.nix` | Sensor data from serial → Elodin-DB |
| `mekf.nix` | MEKF attitude estimation service |
| `msp-osd.nix` | MSP DisplayPort OSD for FPV goggles |
| `udp-component-broadcast.nix` | UDP component broadcasting |
| `udp-component-receive.nix` | UDP component receiving |
| `elodinsink.nix` | GStreamer plugin for video → Elodin-DB |
| `tegrastats-bridge.nix` | Orin SoC telemetry bridge |

### Setup & Status
| Module | Purpose |
|--------|---------|
| `aleph-setup.nix` | First-time setup wizard |
| `installer.nix` | USB installer system |

## Template flake.nix

`aleph/template/flake.nix` is the starting point for user configurations. It imports Aleph modules and configures services:

```nix
# Key structure in template/flake.nix
{
  inputs.elodin.url = "github:elodin-sys/elodin";

  outputs = { self, elodin, nixpkgs }: {
    nixosConfigurations.aleph = nixpkgs.lib.nixosSystem {
      modules = [
        elodin.nixosModules.aleph-base
        elodin.nixosModules.elodin-db
        elodin.nixosModules.mekf
        # ... other modules
      ];
    };
  };
}
```

## Available FSW Applications

These Rust applications in `fsw/` run as NixOS services on Aleph:

| Application | Path | Purpose |
|-------------|------|---------|
| `sensor-fw` | `fsw/sensor-fw/` | STM32H747 firmware: IMU, mag, baro → USB/UART |
| `serial-bridge` | `fsw/serial-bridge/` | Serial port → Elodin-DB |
| `mekf` | `fsw/mekf/` | Sensor fusion for attitude estimation |
| `msp-osd` | `fsw/msp-osd/` | OSD telemetry display for FPV |
| `gstreamer` | `fsw/gstreamer/` | H.264 video → Elodin-DB (`elodinsink`) |
| `video-streamer` | `fsw/video-streamer/` | Video files → AV1 → Elodin-DB |
| `udp_component_broadcast` | `fsw/udp_component_broadcast/` | Real-time UDP component sharing |
| `aleph-status` | `fsw/aleph-status/` | System status display |
| `aleph-setup` | `fsw/aleph-setup/` | Interactive first-time setup |
| `blackbox` | `fsw/blackbox/` | Blackbox data processing |
| `lqr` | `fsw/lqr/` | LQR control system |

## Fresh Install / Recovery

For unbootable or corrupted systems:

1. Build SD image: `nix build --accept-flake-config .#packages.aarch64-linux.sdimage`
2. Flash to USB: `sudo dd if=aleph-os.img of=/dev/sdX bs=4M status=progress oflag=sync`
3. Boot from middle USB-C port
4. SSH via rightmost USB-C: `ssh root@fde1:2240:a1ef::1`
5. Run `aleph-installer`
6. Remove USB and reboot

## MCU Operations

```bash
# Reset the STM32 MCU
aleph/scripts/reset-mcu.sh

# Flash new firmware
aleph/scripts/flash-mcu.sh

# Scan for Aleph devices on network
aleph/scripts/aleph-scan.sh
```

## Key References

- Full setup guide: [aleph/README.md](../../../aleph/README.md)
- Template configuration: [aleph/template/flake.nix](../../../aleph/template/flake.nix)
- Sensor firmware: [fsw/sensor-fw/README.md](../../../fsw/sensor-fw/README.md)
- MSP OSD service: [fsw/msp-osd/README.md](../../../fsw/msp-osd/README.md)
