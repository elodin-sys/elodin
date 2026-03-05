#!/bin/bash
# Deploy user control code to Crazyflie firmware
#
# Usage (from repo root):
#   ./examples/crazyflie-edu/firmware/deploy.sh ~/crazyflie-firmware
#
# This script:
#   1. Creates an app directory in crazyflie-firmware/examples/
#   2. Copies user_code.c, user_code.h, and app_main.c
#   3. Creates the Kbuild.in configuration file
#   4. Provides build instructions

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <crazyflie-firmware-path>"
    echo ""
    echo "Example:"
    echo "  $0 ~/crazyflie-firmware"
    exit 1
fi

CF_PATH="$1"
APP_NAME="app_user_control"
APP_DIR="$CF_PATH/examples/$APP_NAME"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

# Verify crazyflie-firmware path
if [ ! -d "$CF_PATH/src" ]; then
    echo "Error: $CF_PATH does not appear to be a valid crazyflie-firmware directory"
    exit 1
fi

echo "=========================================="
echo "Deploying User Control to Crazyflie"
echo "=========================================="
echo "Source: $EXAMPLE_DIR"
echo "Target: $APP_DIR"
echo ""

# Create app directory
mkdir -p "$APP_DIR/src"

# Copy source files
echo "Copying source files..."
cp "$EXAMPLE_DIR/user_code.c" "$APP_DIR/src/"
cp "$EXAMPLE_DIR/user_code.h" "$APP_DIR/src/"
cp "$SCRIPT_DIR/app_main.c" "$APP_DIR/src/"

# Create Kbuild.in
echo "Creating Kbuild.in..."
cat > "$APP_DIR/Kbuild.in" << 'EOF'
# Kbuild configuration for User Control App
obj-y += src/app_main.o
obj-y += src/user_code.o
EOF

# Create defconfig
echo "Creating ${APP_NAME}_defconfig..."
cat > "$CF_PATH/configs/${APP_NAME}_defconfig" << EOF
# ${APP_NAME} configuration
# Build with: make ${APP_NAME}_defconfig && make -j
CONFIG_APP_ENABLE=y
CONFIG_APP_PRIORITY=3
CONFIG_APP_STACKSIZE=2000
EOF

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cd $CF_PATH"
echo "  2. make ${APP_NAME}_defconfig"
echo "  3. make -j"
echo "  4. cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M"
echo ""
echo "To test in simulation first (from repo root):"
echo "  ./examples/crazyflie-edu/sitl/build.sh"
echo "  elodin run examples/crazyflie-edu/main.py"

