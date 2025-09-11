# Inscriber

A command-line tool for flashing NixOS images to USB drives and SD cards with automatic decompression and a user-friendly interface.

## Overview

Inscriber was created to simplify the process of flashing Aleph's NixOS images to removable media. The tool addresses several pain points in the traditional `dd` workflow:

- **Automatic Compression Handling**: Supports `.zst` (Zstandard) compressed images with on-the-fly decompression during flashing
- **Safe Drive Selection**: Interactive UI helps prevent accidentally overwriting the wrong disk
- **Cross-Platform**: Works on both macOS and Linux
- **Progress Visualization**: Real-time progress bar with ETA and transfer speed
- **Async I/O**: Uses stellarator's async filesystem operations for optimal performance

## Demo

<video autoplay loop muted playsinline style="width: 100%; height: auto;">
  <source src="inscriber-demo.h264.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Installation

### Building from Source

From the repository root:

```bash
cargo build --release --package inscriber
```

The binary will be available at `target/release/inscriber`.

### Running with Cargo

```bash
cargo run --package inscriber -- <image-path> [--disk <disk-path>]
```

## Usage

### Interactive Mode (Recommended)

Simply provide the path to your image file:

```bash
inscriber path/to/aleph-os.img
# or for compressed images
inscriber path/to/aleph-os.img.zst
```

The tool will:
1. Scan for available external drives
2. Present an interactive selection menu
3. Confirm your selection
4. Unmount the selected drive
5. Flash the image with progress visualization

Example output:
```
Please select an external drive to flash
> disk2 SanDisk Ultra - (32.0 GB)
  disk3 Kingston DataTraveler - (16.0 GB)

▌ Flashing aleph-nixos.img.zst to disk2 

⠹ 42.1% 128MB/s|00:02:15|00:03:05
```

### Direct Mode

If you know the exact disk path, you can skip the interactive selection:

```bash
# macOS
inscriber aleph-os.img --disk /dev/disk2

# Linux
inscriber aleph-os.img --disk /dev/sdb
```

⚠️ **Warning**: Be absolutely certain of your disk path when using direct mode. Writing to the wrong disk can cause permanent data loss.

## Supported Image Formats

- **Raw Images**: `.img`, `.iso`, or any uncompressed disk image
- **Zstandard Compressed**: `.zst` compressed images (automatically decompressed during flashing)

## Platform-Specific Details

### macOS
- Uses `diskutil` for disk enumeration and unmounting
- Identifies external disks via the "external" property
- Recommends using raw disk paths (`/dev/rdiskN`) for better performance

### Linux
- Uses `lsblk` for disk enumeration
- Uses `umount` for unmounting
- Identifies external disks by checking:
  - `/sys/block/<device>/removable` flag
  - USB device path detection
  - Device naming patterns (excludes NVMe drives)

## How It Works

1. **Disk Discovery**: Enumerates all external/removable storage devices using platform-specific tools
2. **User Selection**: Presents available disks with identifiers, names, and sizes
3. **Safety Unmount**: Unmounts all partitions on the selected disk
4. **Streaming Write**: 
   - For raw images: Direct streaming copy with 256KB buffer
   - For `.zst` images: On-the-fly decompression using zstd decoder
5. **Progress Tracking**: Updates progress bar based on bytes written

## Development

### Architecture

The codebase is organized as follows:

- `main.rs`: Core application logic including:
  - CLI argument parsing (using `clap`)
  - Async runtime setup (using `stellarator`)
  - Image flashing logic with compression detection
  - Progress bar implementation (using `kdam`)

- Platform-specific implementations:
  - `list_external_disks()`: Platform-specific disk enumeration
  - `ExternalDisk::unmount()`: Platform-specific unmounting

### Key Dependencies

- **stellarator**: Async I/O runtime and utilities
- **zstd**: Zstandard compression/decompression
- **clap**: Command-line argument parsing
- **promkit**: Interactive terminal UI components
- **kdam**: Progress bar with gradient colors
- **nu-ansi-term**: Terminal color and styling

### Adding New Features

When contributing new features, consider:

1. **Compression Formats**: Add support by implementing a decoder similar to the zstd implementation
2. **Platform Support**: Add platform-specific implementations for disk operations
3. **Safety Features**: Consider adding verification steps (checksums, confirmation prompts)
4. **UI Improvements**: Leverage promkit for additional interactive elements

### Testing

Before submitting changes:

1. Test with both compressed and uncompressed images
2. Verify on available platforms (macOS/Linux)
3. Test the interactive selection with multiple drives connected
4. Ensure proper error handling for edge cases:
   - No external drives available
   - Insufficient permissions
   - Corrupted/incomplete images

## Common Issues

### Permission Denied
- **Linux**: Run with `sudo` or add your user to the `disk` group
- **macOS**: May require administrator privileges for raw disk access

### Drive Not Detected
- Ensure the drive is properly connected
- Some USB hubs may not properly report drives as external
- Try connecting directly to your computer

### Slow Transfer Speed on macOS
- Use raw disk paths (`/dev/rdiskN` instead of `/dev/diskN`)
- The tool should handle this automatically in interactive mode

## Use with Aleph

Inscriber is particularly useful for flashing Aleph NixOS images:

1. Build the Aleph SD image:
   ```bash
   nix build --accept-flake-config .#packages.aarch64-linux.sdimage
   ```

2. Flash to USB drive using Inscriber:
   ```bash
   inscriber result/sd-image/aleph-*.img.zst
   ```

3. Boot Aleph from the USB drive following the setup instructions in `/docs/public/content/home/aleph/setup.md`

## Safety Notes

⚠️ **This tool writes directly to block devices and can cause permanent data loss if used incorrectly.**

- Always double-check the target drive before confirming
- Back up important data before flashing
- Use interactive mode when unsure about disk identifiers
- The tool will unmount drives automatically, but ensure no critical processes are using the drive

## Contributing

We welcome contributions! When submitting PRs:

1. Follow the existing code style (use `rustfmt`)
2. Add appropriate error handling
3. Update this README if adding new features
4. Test on available platforms
5. Consider adding unit tests for new logic

## License

See the repository's LICENSE file for details.