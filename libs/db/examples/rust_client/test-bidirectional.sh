#!/bin/bash
# Test script for bidirectional control

echo "🚀 Bidirectional Control Test Script"
echo "===================================="
echo ""

# Check if rocket simulation is running
if lsof -i:2240 > /dev/null 2>&1; then
    echo "✅ Found process listening on port 2240"
else
    echo "❌ No process listening on port 2240"
    echo ""
    echo "Please start the rocket simulation first:"
    echo "  cd libs/nox-py"
    echo "  python examples/rocket.py run 0.0.0.0:2240"
    echo ""
    exit 1
fi

# Path to the rust client
RUST_CLIENT="/Users/danieldriscoll/dev/elodin/target/release/rust_client"

if [ ! -f "$RUST_CLIENT" ]; then
    echo "❌ Rust client binary not found at: $RUST_CLIENT"
    echo "Please build it first: cargo build --release -p elodin-db-rust-client"
    exit 1
fi

echo "✅ Rust client binary found"
echo ""
echo "Starting bidirectional control test..."
echo "The client will:"
echo "  1. Connect to the database"
echo "  2. Send sinusoidal trim commands (±2° @ 0.25Hz)"
echo "  3. Receive and display telemetry"
echo ""
echo "Watch for:"
echo "  - 'Trim control oscillating' messages in the logs"
echo "  - fin_control_trim values in telemetry (if supported)"
echo "  - Oscillating fin_deflect values indicating control is working"
echo ""
echo "Press Ctrl+C to stop"
echo "----------------------------------------"
echo ""

# Run the client with info logging
RUST_LOG=info $RUST_CLIENT --host 127.0.0.1 --port 2240
