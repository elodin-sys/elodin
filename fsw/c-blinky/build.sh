#!/bin/sh -e

TOOLCHAIN_VERSION="14.2.rel1"
CFLAGS="-mcpu=cortex-m7 -mthumb -mfloat-abi=hard -mfpu=fpv5-d16 -Os -g -Wall -Wextra -ffunction-sections -fdata-sections"
LDFLAGS="-T linker.ld -Wl,--gc-sections -Wl,--print-memory-usage"

# Use system toolchain if requested
if [ "$1" = "--system" ] || [ "$USE_SYSTEM_TOOLCHAIN" = "1" ]; then
    CC="${CC:-arm-none-eabi-gcc}"
    SIZE="${SIZE:-arm-none-eabi-size}"
    echo "Using system toolchain"
else
    # Auto-download toolchain
    ARCH="$(uname -m)"
    case "$(uname -s)" in
        Linux)  TOOLCHAIN_NAME="arm-gnu-toolchain-$TOOLCHAIN_VERSION-$ARCH-arm-none-eabi" ;;
        Darwin) TOOLCHAIN_NAME="arm-gnu-toolchain-$TOOLCHAIN_VERSION-darwin-$ARCH-arm-none-eabi" ;;
    esac

    TOOLCHAIN_PATH="tools/$TOOLCHAIN_NAME"
    # Download if missing
    if [ ! -d "$TOOLCHAIN_PATH" ]; then
        echo "Downloading ARM toolchain..."
        mkdir -p "tools"
        curl -L "https://developer.arm.com/-/media/Files/downloads/gnu/$TOOLCHAIN_VERSION/binrel/$TOOLCHAIN_NAME.tar.xz" | tar -xJ -C "tools"
        echo "Toolchain installed to $TOOLCHAIN_PATH"
    fi
    CC="$TOOLCHAIN_PATH/bin/arm-none-eabi-gcc"
    SIZE="$TOOLCHAIN_PATH/bin/arm-none-eabi-size"
    echo "Using local toolchain: $TOOLCHAIN_PATH"
fi

# Build
$CC $CFLAGS $LDFLAGS main.c -o firmware.elf
$SIZE firmware.elf
