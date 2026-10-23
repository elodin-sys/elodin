# Lean E1 characterization baseline

The maintained E1 golden test is a small characterization baseline for the
shared mission controller, not an eternal contract for every numerical output.
Normal pytest runs of this module are fully offline: they use deterministic
inputs with the current `controls.py` / `guidance.py` and existing Python,
NumPy, and pytest dependencies. They do not run git commands, read history,
invoke Elodin or Betaflight, use a GPU, or generate expected values.

`test_guidance_golden.py` covers exactly eight representative cases and 78
updates: `full_mission_and_hold_timeout` (15), `braking_yaw_saturation` (11),
`yawed_active_latching_braking_crossing` (5),
`full_attitude_and_missing_truth` (7), `missing_truth_armed` (8),
`gyro_nan_x_armed` (8), `nonmonotonic_True` (7), and
`velocity_and_launch_xy_dwell_interruptions` (17). Their configurations and
inputs are the selected original fixture cases; ordinary behavior remains
covered by `test_guidance.py` and `test_mission_controller.py`.

For each case, the expected SHA-256 is derived from that case's original
expected output rows—not from current controller output—using this canonical
serialization:

```python
json.dumps(
    rows,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("utf-8")
```

Each row retains the old snapshot fields: phase, phase code, failure reason,
braking state, target, and command. Floating-point values remain `float.hex()`
strings, including signed zero. A mismatch reports the scenario, expected and
actual digest, and current output rows. No full trace is stored elsewhere in
the current tree.

## Historical full fixtures

The original full expected records remain in git history at commit
`163c052956a15f8d720a93f2e7059f1c52b8ce71`. The initial E1 exact-equivalence
proof, 39 scenarios / 268 updates, 300-test result, and three truth-hold RMS
results are historical acceptance evidence from that commit; this lean follow-up
is not a new live qualification and does not change E1's original acceptance.

To inspect an old full record, use `git show` against that commit (and optionally
filter one named case with `jq`):

```bash
git show 163c052956a15f8d720a93f2e7059f1c52b8ce71:examples/betaflight-sitl/tests/fixtures/guidance_package_e.json \
  | jq '.cases[] | select(.name == "full_mission_and_hold_timeout")'
git show 163c052956a15f8d720a93f2e7059f1c52b8ce71:examples/betaflight-sitl/tests/fixtures/guidance_package_e_supplement.json \
  | jq '.cases[] | select(.name == "yawed_active_latching_braking_crossing")'
```

The large fixtures and supplemental test module were removed from the current
tree, not rewritten into another trace format.

## Safe original-source digest reproduction

The optional `--capture` mode prints a compact digest map only. It requires the
original `guidance.py` and `controls.py` byte-for-byte hashes from baseline
`806b07721797a15e92b1614c0a206516c036f2e3`; otherwise it refuses to capture.
It never edits `EXPECTED_DIGESTS` or writes a fixture. Reproduce in a new
isolated directory under `$HOME`, not in the working checkout:

```bash
# From the repository root, use Python with NumPy and pytest installed.
base=$(mktemp -d "$HOME/bf-e1-original.XXXXXX")
git archive 806b07721797a15e92b1614c0a206516c036f2e3 \
  examples/betaflight-sitl/controls.py \
  examples/betaflight-sitl/guidance.py | tar -x -C "$base"
mkdir -p "$base/examples/betaflight-sitl/tests"
cp examples/betaflight-sitl/tests/test_guidance_golden.py \
  "$base/examples/betaflight-sitl/tests/"
python3 "$base/examples/betaflight-sitl/tests/test_guidance_golden.py" --capture
```

Compare the printed map to the checked-in map. A future deliberate controller
change may retire or update a relevant characterization through a reviewed
change that explains why; do not blindly rewrite digests just to make a failing
test pass. The initial full equivalence proof remains in history independently
of this ongoing maintenance baseline.
