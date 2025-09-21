#!/bin/bash

# Script to test with debug logging enabled

echo "=== Testing External Control with Debug Logging ==="
echo
echo "1. Starting rocket simulation with debug logging..."
echo "   Run in terminal 1:"
echo "   cd libs/nox-py"
echo "   RUST_LOG=info .venv/bin/python examples/rocket.py run 0.0.0.0:2240"
echo
echo "2. Wait for simulation to start, then run control client:"
echo "   Run in terminal 2:"
echo "   RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240"
echo
echo "3. Watch for these log messages in terminal 1:"
echo "   - 'Reading external control component from DB: rocket.fin_control_trim'"
echo "   - 'Copying external control rocket.fin_control_trim from DB: value=X.XXX'"
echo
echo "The value should oscillate between -10.0 and 10.0"
echo
echo "If you see 'External control component not found in DB', the client's"
echo "values aren't being written to the database correctly."
