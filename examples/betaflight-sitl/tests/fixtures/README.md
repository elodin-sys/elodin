# Package E controller golden traces

These fixed fixtures were captured from the **original** Package E controller,
not from the extracted controller. Normal tests replay them and never generate
expected output. Floating-point outputs are hexadecimal strings, including signed
zero, so comparisons are exact (no tolerance).

Baseline: `806b07721797a15e92b1614c0a206516c036f2e3`.
Original `guidance.py` SHA256:
`6783e94e03cd715c5cfdc66a3ca2d09c17fe79be6c1f41c41696af4a729d3b69`.
Original `controls.py` SHA256:
`1bf2504d66ed8666755c648f8c936d81931ff23d4f8ffcbfdefa054cc2b9948f`.

| Fixture | Cases / samples | SHA256 |
| --- | --- | --- |
| `guidance_package_e.json` | 34 / 229 | `95f9257340cd5f0fa3a7a436403018c9e55fa340297d97c5a7de3f630af92bb3` |
| `guidance_package_e_supplement.json` | 5 / 39 | `53fe85c570bb477d1889536fc08256179092acb3fe6ea7e0241e90420c8f348f` |

The initial fixture covers all normal phase transitions, FAULT from each active
phase, all ten fault reasons, before/after-arm invalid time/truth/gyro feedback,
first-reason/target persistence, gate latching despite progress, yaw wrapping,
throttle/yaw/acceleration saturation, stopping-distance and safety-slab braking,
and position/altitude dwell interruptions. Its `yawed_gate_latching_and_crossing`
case crosses early at 5.0 s; it is intentionally retained unchanged as a rotated
**fault** trace, not claimed as healthy yawed flight.

The supplement, requested by independent pre-extraction review, covers healthy
yawed arming/TAKEOFF and braking at 5.8 s before crossing at 5.9 s (with explicit
phase/reason/braking assertions), full non-pure-yaw scalar-last quaternion
normalization/heading and subsequent missing-truth fallback, independently binding
custom tilt and stick caps, and velocity/launch-XY dwell interruptions. The
stationary 8.25 s HOVER sample distinguishes the 8.0 s velocity reset from a
controller that silently omits the HOVER velocity gate. It was
captured in an isolated original-source tree after extraction had started; no
new-controller output was used as expected output. Existing fixture bytes were
regenerated identically and left unchanged.

## Reproduce without touching the extracted controller

Run from the repository root, using Python with numpy and pytest. Use a **new**
scratch directory under `$HOME` (not an existing checkout or the working source):

```bash
base=$(mktemp -d "$HOME/bf-e1-original.XXXXXX")
git archive 806b07721797a15e92b1614c0a206516c036f2e3 \
  examples/betaflight-sitl/controls.py \
  examples/betaflight-sitl/guidance.py | tar -x -C "$base"
mkdir -p "$base/examples/betaflight-sitl/tests"
cp examples/betaflight-sitl/tests/test_guidance_golden*.py \
  "$base/examples/betaflight-sitl/tests/"
python "$base/examples/betaflight-sitl/tests/test_guidance_golden.py" --capture
python "$base/examples/betaflight-sitl/tests/test_guidance_golden_supplement.py" --capture
cmp "$base/examples/betaflight-sitl/tests/fixtures/guidance_package_e.json" \
  examples/betaflight-sitl/tests/fixtures/guidance_package_e.json
cmp "$base/examples/betaflight-sitl/tests/fixtures/guidance_package_e_supplement.json" \
  examples/betaflight-sitl/tests/fixtures/guidance_package_e_supplement.json
python -m pytest "$base/examples/betaflight-sitl/tests" -q
```

Both generators refuse a controller whose hash differs from the original. Never
update the stored digests/expected outputs to accommodate changed controller
behavior. Capture uses no RNG, Elodin, Betaflight, renderer, GPU, or live plant.
