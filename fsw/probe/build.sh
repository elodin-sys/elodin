#!/bin/sh -e

# Builds Aleph Flight Controller probe firmware targets into build/<target>.{elf,bin}.

TOOLCHAIN_VERSION="15.2.rel1"
TARGETS="blinky dap"

CFLAGS="-mcpu=cortex-m0plus -mthumb -mfloat-abi=soft -Os -g3 -flto -Wall -Wextra -fno-builtin -fno-jump-tables"
CFLAGS="$CFLAGS -ffunction-sections -fdata-sections -fno-common -ffreestanding -nostdlib -nostartfiles"
CFLAGS="$CFLAGS $EXTRA_CFLAGS"

# Probe tags only (probe-vX.Y.Z); board release tags would mislabel it.
VERSION=$(git describe --tags --always --dirty --match 'probe-v*' 2>/dev/null || echo unknown)
VERSION=${VERSION#probe-v}
CFLAGS="$CFLAGS -DPROBE_VERSION=\"$VERSION\""
# .ramfunc puts the SWD bit loops in .data, so that segment is RWX by design.
LDFLAGS="-flto -T linker.ld -Wl,--gc-sections -Wl,--print-memory-usage -Wl,--no-warn-rwx-segments"

cd "$(dirname "$0")"

if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    CC=arm-none-eabi-gcc
    OBJCOPY=arm-none-eabi-objcopy
    SIZE=arm-none-eabi-size
    echo "Using system toolchain"
else
    ARCH="$(uname -m)"
    case "$(uname -s):$ARCH" in
        Darwin:arm64)  TOOLCHAIN_NAME="arm-gnu-toolchain-$TOOLCHAIN_VERSION-darwin-arm64-arm-none-eabi" ;;
        Linux:x86_64)  TOOLCHAIN_NAME="arm-gnu-toolchain-$TOOLCHAIN_VERSION-x86_64-arm-none-eabi" ;;
        Linux:aarch64) TOOLCHAIN_NAME="arm-gnu-toolchain-$TOOLCHAIN_VERSION-aarch64-arm-none-eabi" ;;
        Darwin:*)
            echo "Arm GNU $TOOLCHAIN_VERSION has no official Intel macOS archive; install arm-none-eabi-gcc"
            exit 1
            ;;
        *)
            echo "Unsupported host OS/architecture: $(uname -s) $ARCH"
            exit 1
            ;;
    esac

    TOOLCHAIN_PATH="tools/$TOOLCHAIN_NAME"
    if [ ! -d "$TOOLCHAIN_PATH" ]; then
        echo "Downloading Arm GNU toolchain $TOOLCHAIN_VERSION..."
        mkdir -p tools
        curl -fL "https://developer.arm.com/-/media/Files/downloads/gnu/$TOOLCHAIN_VERSION/binrel/$TOOLCHAIN_NAME.tar.xz" |
            tar -xJ -C tools
    fi

    CC="$TOOLCHAIN_PATH/bin/arm-none-eabi-gcc"
    OBJCOPY="$TOOLCHAIN_PATH/bin/arm-none-eabi-objcopy"
    SIZE="$TOOLCHAIN_PATH/bin/arm-none-eabi-size"
    echo "Using local toolchain: $TOOLCHAIN_PATH"
fi

# GCC 13 knows C23 as c2x.
if echo 'int x;' | $CC -std=c23 -x c -fsyntax-only - 2>/dev/null; then STD=c23; else STD=c2x; fi
CFLAGS="$CFLAGS -std=$STD"

echo "Version $VERSION"
mkdir -p build
for TARGET in $TARGETS; do
    echo
    echo "=== $TARGET ==="
    $CC $CFLAGS $LDFLAGS "$TARGET.c" -o "build/$TARGET.elf"
    $OBJCOPY -O binary "build/$TARGET.elf" "build/$TARGET.bin"
    $SIZE "build/$TARGET.elf"
done
