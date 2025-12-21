#!/bin/bash
# Build script for Betaflight SITL
#
# This script compiles Betaflight firmware for Software-In-The-Loop (SITL)
# simulation, which allows running the flight controller on the host machine
# and communicating with Elodin via UDP.
#
# Usage:
#   ./build.sh          # Build SITL target
#   ./build.sh clean    # Clean build artifacts
#   ./build.sh sdk      # Install ARM SDK (required once, even for SITL on macOS)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BETAFLIGHT_DIR="$SCRIPT_DIR/betaflight"

# Check if betaflight submodule exists
if [ ! -d "$BETAFLIGHT_DIR" ]; then
    echo "Error: Betaflight submodule not found at $BETAFLIGHT_DIR"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

cd "$BETAFLIGHT_DIR"

# macOS compatibility: clang doesn't support -fuse-linker-plugin (GCC-specific LTO flag)
# and some warnings need to be disabled due to compiler differences
MACOS_OPTIMISATION_BASE=""
MACOS_CFLAGS_DISABLED=""
if [[ "$(uname)" == "Darwin" ]]; then
    MACOS_OPTIMISATION_BASE="-flto=auto -ffast-math -fmerge-all-constants"
    # Disable -Werror and other flags that cause issues with clang/macOS
    MACOS_CFLAGS_DISABLED="-Werror -Wunsafe-loop-optimizations -fuse-linker-plugin"
fi

# Helper function to run make with proper flags
run_make() {
    if [[ -n "$MACOS_OPTIMISATION_BASE" ]]; then
        # Use EXTRA_FLAGS to add -Wno-error which disables treating warnings as errors
        make "OPTIMISATION_BASE=$MACOS_OPTIMISATION_BASE" "EXTRA_FLAGS=-Wno-error" "$@"
    else
        make "$@"
    fi
}

case "${1:-build}" in
    sdk)
        echo "Installing ARM SDK (required for build system, even for SITL)..."
        make arm_sdk_install
        echo "ARM SDK installed successfully."
        ;;
    
    clean)
        echo "Cleaning Betaflight SITL build artifacts..."
        run_make TARGET=SITL clean
        echo "Clean complete."
        ;;
    
    build|"")
        echo "Building Betaflight SITL..."
        echo "  Target: SITL"
        echo "  Output: $BETAFLIGHT_DIR/obj/main/betaflight_SITL.elf"
        echo ""
        
        # Check if ARM SDK is installed (required even for SITL on some systems)
        if [ ! -d "tools/gcc-arm-none-eabi"* ] 2>/dev/null; then
            if ! command -v arm-none-eabi-gcc &> /dev/null; then
                echo "Note: ARM SDK not found. Installing (required for build system)..."
                make arm_sdk_install
            fi
        fi
        
        # Enable SIMULATOR_GYROPID_SYNC for lockstep synchronization with Elodin
        # This makes Betaflight block on FDM packets, allowing tight timing control
        TARGET_H="$BETAFLIGHT_DIR/src/platform/SIMULATOR/target/SITL/target.h"
        if grep -q "^//#define SIMULATOR_GYROPID_SYNC" "$TARGET_H"; then
            echo "Enabling SIMULATOR_GYROPID_SYNC for lockstep mode..."
            sed -i.bak 's|^//#define SIMULATOR_GYROPID_SYNC|#define SIMULATOR_GYROPID_SYNC|' "$TARGET_H"
            rm -f "${TARGET_H}.bak"
        elif grep -q "^#define SIMULATOR_GYROPID_SYNC" "$TARGET_H"; then
            echo "SIMULATOR_GYROPID_SYNC already enabled."
        else
            echo "Warning: Could not find SIMULATOR_GYROPID_SYNC in target.h"
        fi
        
        # Build SITL target
        JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        
        # On macOS, we need to create stubs for missing SITL symbols and do a two-pass build
        if [[ "$(uname)" == "Darwin" ]]; then
            STUBS_C="$SCRIPT_DIR/macos_sitl_stubs.c"
            STUBS_O="$BETAFLIGHT_DIR/obj/main/SITL/sitl_stubs.o"
            
            # Create stubs file for macOS (provides missing symbols)
            cat > "$STUBS_C" << 'STUBS_EOF'
/* macOS SITL Stubs - provides missing symbols for Betaflight SITL on macOS */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* GPS stubs */
typedef struct { int32_t lat, lon, alt; uint16_t groundSpeed, groundCourse; uint8_t numSat, fixType; } gpsSolutionData_t;
gpsSolutionData_t gpsSol = {0};
typedef struct { uint8_t gateEnabled; int32_t gateLat, gateLon; uint16_t gateDirection; uint8_t minimumLapTimeSeconds; } gpsLapTimerConfig_t;
gpsLapTimerConfig_t gpsLapTimerConfig_System = {0};
bool gpsHasNewData(void) { return false; }
float getGpsDataFrequencyHz(void) { return 10.0f; }
float getGpsDataIntervalSeconds(void) { return 0.1f; }
void GPS_distance2d(int32_t *lat1, int32_t *lon1, int32_t *lat2, int32_t *lon2, uint32_t *dist) { if (dist) *dist = 0; }
void GPS_distance_cm_bearing(int32_t *lat1, int32_t *lon1, int32_t *lat2, int32_t *lon2, uint32_t *dist, int32_t *bearing) { if (dist) *dist = 0; if (bearing) *bearing = 0; }

/* Audio stubs */
void audioSetupIO(void) {}
void audioSilence(void) {}
void audioPlayTone(uint16_t frequency) { (void)frequency; }
void audioGenerateWhiteNoise(void) {}

/* Clock/timing stubs */
float clockCyclesToMicrosf(uint32_t cycles) { return (float)cycles / 500.0f; }

/* DMA stubs */
typedef struct { void *dummy; } dmaChannelDescriptor_t;
dmaChannelDescriptor_t dmaDescriptors[16] = {{0}};

/* IO stubs */
typedef void* IO_t;
bool IORead(IO_t io) { (void)io; return false; }

/* Gyro stubs */
bool mpuGyroReadRegister(void *dev, uint8_t reg, uint8_t *data, uint8_t length) {
    (void)dev; (void)reg; (void)length;
    if (data && length > 0) for (uint8_t i = 0; i < length; i++) data[i] = 0;
    return true;
}

/* DShot telemetry stub */
bool useDshotTelemetry = false;
STUBS_EOF
            
            # First build - this will fail at link but compile all objects
            echo "Building objects (first pass)..."
            run_make TARGET=SITL -j"$JOBS" || true
            
            # Compile stubs
            echo "Compiling SITL stubs for macOS..."
            mkdir -p "$(dirname "$STUBS_O")"
            gcc -c -O2 -I"$BETAFLIGHT_DIR/src/main" \
                -I"$BETAFLIGHT_DIR/src/platform/SIMULATOR/include" \
                -I"$BETAFLIGHT_DIR/src/platform/SIMULATOR/target/SITL" \
                -o "$STUBS_O" "$STUBS_C"
            
            # Manual link with stubs included
            echo "Linking with stubs..."
            OBJS=$(find "$BETAFLIGHT_DIR/obj/main/SITL" -name "*.o" | tr '\n' ' ')
            # Use system clang for linking to avoid Nix toolchain compatibility issues
            # -Wl,-no_compact_unwind suppresses "could not create compact unwind" warnings
            # which occur because Betaflight's firmware code doesn't use standard stack frames
            /usr/bin/clang -o "$BETAFLIGHT_DIR/obj/main/betaflight_SITL.elf" \
                $OBJS \
                -lm -lpthread \
                -Wl,-no_compact_unwind \
                -Wl,-map,"$BETAFLIGHT_DIR/obj/main/betaflight_SITL.map"
            
            # Clean up temporary stubs file
            rm -f "$STUBS_C"
        else
            run_make TARGET=SITL -j"$JOBS"
        fi
        
        if [ -f "obj/main/betaflight_SITL.elf" ]; then
            echo ""
            echo "Build successful!"
            echo "Binary: $BETAFLIGHT_DIR/obj/main/betaflight_SITL.elf"
            echo ""
            echo "To run SITL manually:"
            echo "  ./betaflight/obj/main/betaflight_SITL.elf"
            echo ""
            echo "To configure via Betaflight Configurator:"
            echo "  1. Open https://app.betaflight.com"
            echo "  2. Click 'Manual selection' -> 'TCP' -> 'localhost:5761'"
            echo "  3. Click 'Connect'"
        else
            echo "Build failed - binary not found"
            exit 1
        fi
        ;;
    
    debug)
        echo "Building Betaflight SITL with debug symbols..."
        # DEBUG=GDB uses simpler LTO flags that work on macOS
        JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        make TARGET=SITL DEBUG=GDB -j"$JOBS"
        echo "Debug build complete."
        ;;
    
    *)
        echo "Usage: $0 [build|clean|sdk|debug]"
        echo ""
        echo "Commands:"
        echo "  build   Build SITL target (default)"
        echo "  clean   Clean build artifacts"
        echo "  sdk     Install ARM SDK (required once)"
        echo "  debug   Build with debug symbols"
        exit 1
        ;;
esac
