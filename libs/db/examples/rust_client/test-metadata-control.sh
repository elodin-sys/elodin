#!/bin/bash

echo "==================================================="
echo "Testing Metadata-Based External Control Components"
echo "==================================================="
echo ""
echo "This test demonstrates the proper solution where components"
echo "are marked as 'external_control' via metadata, preventing"
echo "the simulation from writing them back while allowing external"
echo "clients to control them."
echo ""

echo "Step 1: Building components..."
echo "-------------------------------"
cd /Users/danieldriscoll/dev/elodin

echo "Building nox-ecs..."
cargo build --release -p nox-ecs 2>&1 | tail -1

echo "Building nox-py..."
cd libs/nox-py
uvx maturin develop --release --uv 2>&1 | tail -1
cd ../..

echo "Building Rust client..."
cargo build --release -p elodin-db-rust-client 2>&1 | tail -1

echo ""
echo "Step 2: Start the rocket simulation"
echo "------------------------------------"
echo "Run in terminal 1:"
echo "  cd libs/nox-py"
echo "  python examples/rocket.py run 0.0.0.0:2240"
echo ""
echo "Look for: Component registration messages"
echo "Watch for: NO time travel warnings!"
echo ""

echo "Step 3: Run the control client"
echo "-------------------------------"
echo "Run in terminal 2:"
echo "  RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240"
echo ""
echo "Look for: Sinusoidal trim control messages"
echo "Watch for: Smooth oscillation (±10° @ 0.25Hz)"
echo ""

echo "Step 4: Connect Elodin Editor (optional)"
echo "-----------------------------------------"
echo "Run in terminal 3:"
echo "  elodin editor 127.0.0.1:2240"
echo ""
echo "Look for: Rocket rolling/oscillating smoothly"
echo ""

echo "✅ SUCCESS CRITERIA:"
echo "- No 'time travel' warnings in simulation terminal"
echo "- No 'error committing head' errors"
echo "- Control client shows oscillating trim values"
echo "- Rocket responds to control inputs"
echo ""

echo "📝 KEY IMPLEMENTATION POINTS:"
echo "1. Component declared with metadata={\"external_control\": \"true\"}"
echo "2. Component initialized with default value (0.0)"
echo "3. Simulation reads the value from database"
echo "4. Simulation skips writing it back (checks metadata)"
echo "5. Control client has exclusive write access"
echo ""

echo "🎯 This is a scalable solution - any component can be marked"
echo "   as external control simply by adding the metadata flag!"
