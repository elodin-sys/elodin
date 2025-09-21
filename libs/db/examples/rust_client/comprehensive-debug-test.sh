#!/bin/bash

echo "=== Comprehensive Debug Test for External Control ==="
echo
echo "This test will help identify why fin_control_trim isn't updating"
echo

# Terminal 1
echo "TERMINAL 1: Run the simulation with full debug logging"
echo "----------------------------------------"
echo "cd libs/nox-py"
echo 'RUST_LOG=debug .venv/bin/python examples/rocket.py run 0.0.0.0:2240 2>&1 | grep -E "(fin_control_trim|external|Copying|vtable)" --color=always'
echo

# Terminal 2  
echo "TERMINAL 2: Run the client with logging"
echo "----------------------------------------"
echo "cd libs/db/examples/rust_client"
echo "RUST_LOG=info ../../../../target/release/rust_client --host 127.0.0.1 --port 2240"
echo

# Terminal 3
echo "TERMINAL 3: Monitor the database directly"
echo "----------------------------------------"
echo "elodin editor 127.0.0.1:2240"
echo "- Look at the 'Roll Moment (Cl)' graph"
echo "- Should show 0.5 constant (debug value)"
echo "- Should oscillate ±1.0 if trim is working"
echo

echo "WHAT TO LOOK FOR:"
echo "----------------"
echo "1. In Terminal 1 (simulation):"
echo "   - 'inserting vtable id=[1, 0]' (or similar)"
echo "   - 'Copying external control rocket.fin_control_trim from DB: value=X.XXX'"
echo "   - The value should change from 0.0 when client connects"
echo
echo "2. In Terminal 2 (client):"
echo "   - 'Sent trim: X.XXX° @ t=Y.YY to rocket.fin_control_trim'"
echo "   - Values should oscillate between -10 and +10"
echo
echo "3. In Terminal 3 (editor):"
echo "   - Roll Moment graph should show oscillation if working"
echo "   - Trim Control graph should show oscillation from client"
echo

echo "DIAGNOSIS:"
echo "----------"
echo "If Roll Moment stays at 0.5:"
echo "  → Client data not reaching simulation"
echo "  → Check VTable IDs match"
echo
echo "If 'External control component not found in DB':"
echo "  → Component not properly registered"
echo
echo "If client shows sends but sim shows value=0.000:"
echo "  → VTable mismatch or wrong component ID"
