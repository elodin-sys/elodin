#!/usr/bin/env bash
# Test script for running Avatar OSD locally

set -e

echo "Avatar OSD Local Test Script"
echo "============================"
echo ""
echo "This script demonstrates how to test the Avatar OSD service locally."
echo ""

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust or enter the Nix development shell."
    exit 1
fi

# Build the project
echo "Building Avatar OSD..."
cargo build --release

echo ""
echo "You can now run the Avatar OSD in different modes:"
echo ""
echo "1. Debug Mode (Terminal Display):"
echo "   cargo run -- --mode debug"
echo ""
echo "2. With custom database address:"
echo "   cargo run -- --mode debug --db-addr 127.0.0.1:2240"
echo ""
echo "3. Serial Mode (for actual hardware):"
echo "   cargo run -- --mode serial --serial-port /dev/ttyUSB0"
echo ""
echo "To test with the BDX RC-Jet simulation:"
echo "   1. In one terminal: cd ../../examples/rc-jet && python main.py run"
echo "   2. In another terminal: cargo run -- --mode debug"
echo ""

# Optionally run in debug mode
read -p "Would you like to run Avatar OSD in debug mode now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting Avatar OSD in debug mode..."
    echo "Press Ctrl-C to exit."
    echo ""
    cargo run -- --mode debug
fi
