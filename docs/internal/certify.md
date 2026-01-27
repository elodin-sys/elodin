# Release Certification

Manual test scenarios that must pass before certifying a release. Run these after CI is green and before following the steps in [release.md](release.md).

## Prerequisites

- [ ] CI pipeline green on release branch
- [ ] `just install` completed successfully
- [ ] Fresh example directory (delete any existing `.kdl` files in examples)

## Test Scenarios

### Schematic Persistence

#### Monitor creation and persistence

1. Run `elodin examples/ball`
2. Open command palette (Cmd+P)
3. Create a monitor for `ball.world_pos`
4. Drag monitor tab to create a new split
5. Save schematic (Cmd+S or command palette)
6. Close Elodin
7. Run `elodin examples/ball` again
8. **Expected**: Monitor panel is present in the same position

#### Secondary window persistence

1. Run `elodin examples/ball`
2. Create a secondary window via command palette
3. Add a viewport to the secondary window
4. Save schematic
5. Close Elodin
6. Reopen `elodin examples/ball`
7. **Expected**: Secondary window opens with viewport

### Viewport

#### Camera controls

1. Run `elodin examples/drone`
2. Click and drag in viewport to orbit camera
3. Scroll to zoom in/out
4. Right-click drag to pan
5. Press Space to start simulation
6. **Expected**: Drone visible and animating, camera responds to all inputs

#### Entity selection

1. Run `elodin examples/ball`
2. Click on the ball entity in the 3D viewport
3. **Expected**: Ball is highlighted, Inspector panel shows ball components

#### Viewport reset

1. Run `elodin examples/ball`
2. Orbit and zoom the camera to a different position
3. Open command palette (Cmd+P)
4. Run "Reset Cameras"
5. **Expected**: Camera returns to default position/orientation

### Graph/Monitor

#### Live data graphing

1. Run `elodin examples/ball`
2. Create a graph for `ball.world_pos.z` via command palette
3. Press Space to start simulation
4. **Expected**: Graph shows oscillating Z position as ball bounces

#### Multiple monitors

1. Run `elodin examples/cube-sat`
2. Create monitors for `cube_sat.world_pos` and `cube_sat.world_vel`
3. Start simulation
4. **Expected**: Both monitors update with different values

### Timeline

#### Playback speed

1. Run `elodin examples/ball`
2. Start simulation (Space)
3. Open command palette, set playback speed to 0.5x
4. **Expected**: Ball animation runs at half speed

#### Timeline scrubbing

1. Run `elodin examples/ball`
2. Run simulation for a few seconds, then pause
3. Drag timeline slider backward
4. **Expected**: Ball position rewinds to earlier state

### Command Palette

#### Fuzzy search

1. Run `elodin examples/ball`
2. Open command palette (Cmd+P)
3. Type "gra" (partial match for "Create Graph")
4. **Expected**: "Create Graph" appears in filtered results

### Panel Management

#### Panel reorganization

1. Run `elodin examples/ball`
2. Create two graphs
3. Drag one graph tab onto the other to create tabs
4. Drag the split handle to resize
5. Save and reopen
6. **Expected**: Tab layout and sizes preserved

## Example Smoke Tests

Run each example and verify basic functionality:

- [ ] `examples/ball/` - Ball bounces, world_pos.z oscillates
- [ ] `examples/drone/` - Drone model renders, motors respond
- [ ] `examples/rocket/` - Rocket launches and follows trajectory
- [ ] `examples/cube-sat/` - Satellite orbits, attitude visible
- [ ] `examples/three-body/` - All three bodies orbit correctly

## Sign-Off

```
Version: _______________
Date: _______________
Tester: _______________

[ ] All test scenarios passed
[ ] CI pipeline green
[ ] CHANGELOG updated

Notes/Exceptions:
_______________________
```
