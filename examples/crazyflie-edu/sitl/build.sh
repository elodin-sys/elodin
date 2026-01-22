#!/bin/bash
# Build script for Crazyflie SITL
#
# Usage:
#   ./build.sh          # Build SITL binary
#   ./build.sh clean    # Clean build artifacts
#   ./build.sh run      # Build and run (for testing)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Compiler settings
CC="${CC:-clang}"
CFLAGS="-Wall -Wextra -O2 -g"
LDFLAGS="-lm"

# Source files
SOURCES="sitl_main.c ../user_code.c"
OUTPUT="sitl_main"

case "${1:-build}" in
    build)
        echo "Building Crazyflie SITL..."
        echo "  CC: $CC"
        echo "  Sources: $SOURCES"
        echo "  Output: $OUTPUT"
        $CC $CFLAGS $SOURCES -o $OUTPUT $LDFLAGS
        echo "Build complete: $SCRIPT_DIR/$OUTPUT"
        ;;

    clean)
        echo "Cleaning build artifacts..."
        rm -f "$OUTPUT" *.o
        echo "Clean complete."
        ;;

    run)
        # Build first
        "$0" build
        echo ""
        echo "Running SITL (Ctrl+C to stop)..."
        echo ""
        ./"$OUTPUT"
        ;;

    *)
        echo "Usage: $0 [build|clean|run]"
        exit 1
        ;;
esac

