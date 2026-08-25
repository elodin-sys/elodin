//! Input handling for gamepad and keyboard
//!
//! Supports:
//! - Gamepad: FrSky X20 R5 (appears as USB HID joystick)
//! - Keyboard: W/S throttle, Q/E or A/D yaw, arrow keys pitch/roll
//!
//! Stick modes:
//! - Mode 2 (US standard): Left stick = Throttle/Yaw, Right stick = Pitch/Roll
//! - Mode 1 (EU/Asia): Left stick = Pitch/Yaw, Right stick = Throttle/Roll

use device_query::{DeviceQuery, DeviceState, Keycode};
use gilrs::{Axis, Gilrs};
use std::f64::consts::PI;

/// Throttle held when no one is on the sticks: the package cruise trim
/// (results/bdx/baseline trim_map cruise row, effective throttle 0.2125).
const IDLE_THROTTLE: f64 = 0.21;

/// Stick travel around centre that reads as neutral.
const DEADZONE: f64 = 0.1;
/// Full elevator/aileron throw, in radians (25°).
const MAX_DEFLECTION_RAD: f64 = 25.0 * PI / 180.0;
/// Full rudder throw, in radians (30°).
const MAX_RUDDER_RAD: f64 = 30.0 * PI / 180.0;
// Stick / key senses as flown: stick up and up-arrow pitch the nose up,
// stick right and right-arrow roll right, Q/A yaw left and E/D yaw right.
// Rudder still uses the plant's TE-left = nose-left convention, so E
// (yaw right) is a negative rudder command.

/// Stick mode configuration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StickMode {
    /// Mode 2 (US): Left = Throttle/Yaw, Right = Pitch/Roll
    #[default]
    Mode2,
    /// Mode 1 (EU): Left = Pitch/Yaw, Right = Throttle/Roll
    Mode1,
}

/// Control outputs from input devices
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlInput {
    /// Elevator command in radians (±0.44 rad = ±25°)
    pub elevator: f64,
    /// Aileron command in radians (±0.44 rad = ±25°)
    pub aileron: f64,
    /// Rudder command in radians (±0.52 rad = ±30°)
    pub rudder: f64,
    /// Throttle command (0.0 to 1.0)
    pub throttle: f64,
}

impl ControlInput {
    /// Convert to f64 array for sending [elevator, aileron, rudder, throttle]
    pub fn as_array(self) -> [f64; 4] {
        [self.elevator, self.aileron, self.rudder, self.throttle]
    }
}

/// Merge the two input sources. Each surface takes whichever device asks for
/// more of it, so the gamepad stays primary wherever it is actually deflected;
/// the throttle is the gamepad's only once its stick has been moved, since a
/// centred stick means "untouched", not "half throttle".
fn combine(gamepad: ControlInput, keyboard: ControlInput, gamepad_throttle: bool) -> ControlInput {
    ControlInput {
        elevator: dominant(gamepad.elevator, keyboard.elevator),
        aileron: dominant(gamepad.aileron, keyboard.aileron),
        rudder: dominant(gamepad.rudder, keyboard.rudder),
        throttle: if gamepad_throttle {
            gamepad.throttle
        } else {
            keyboard.throttle
        },
    }
}

/// The larger of two commands for one surface, gamepad winning a tie.
///
/// Comparing deflections rather than testing the gamepad against a small
/// threshold is what keeps a stick left off centre by trim from swallowing a
/// held key: such an axis is a thousandth of a radian of surface, which used to
/// out-vote an arrow key's half throw.
fn dominant(gamepad: f64, keyboard: f64) -> f64 {
    if gamepad.abs() >= keyboard.abs() {
        gamepad
    } else {
        keyboard
    }
}

/// Input reader combining gamepad and keyboard
pub struct InputReader {
    gilrs: Option<Gilrs>,
    /// Absent on headless hosts: device_query aborts without an X display.
    device_state: Option<DeviceState>,
    stick_mode: StickMode,
    /// Current throttle state (keyboard is incremental)
    keyboard_throttle: f64,
    /// Last control input for smoothing
    last_input: ControlInput,
    /// Whether the gamepad's throttle stick has ever left neutral.
    ///
    /// A gamepad self-centres its sticks and centre maps to mid-throttle, so
    /// an untouched pad would otherwise out-vote the keyboard. A transmitter's
    /// ratcheted throttle sits off centre, so it latches on the first read.
    gamepad_throttle_engaged: bool,
}

impl InputReader {
    pub fn new(stick_mode: StickMode) -> Self {
        // Try to initialize gilrs, but don't fail if no gamepad available
        let gilrs = match Gilrs::new() {
            Ok(g) => {
                // Check for connected gamepads
                let mut has_gamepad = false;
                for (id, gamepad) in g.gamepads() {
                    tracing::info!(
                        "Found gamepad {}: {} ({})",
                        id,
                        gamepad.name(),
                        if gamepad.is_connected() {
                            "connected"
                        } else {
                            "disconnected"
                        }
                    );
                    has_gamepad = true;
                }
                if !has_gamepad {
                    tracing::info!("No gamepads found, using keyboard only");
                }
                Some(g)
            }
            Err(e) => {
                tracing::warn!("Failed to initialize gamepad support: {}", e);
                tracing::info!("Using keyboard only");
                None
            }
        };

        // device_query's Linux backend panics if it cannot open DISPLAY.
        // The nix develop hook sets DISPLAY=:0 even on headless CI, so "is
        // DISPLAY set?" is not enough — probe the X11 socket first.
        let device_state = try_device_state();
        if device_state.is_none() {
            tracing::info!("No X display; keyboard input disabled");
        }

        Self {
            gilrs,
            device_state,
            stick_mode,
            keyboard_throttle: IDLE_THROTTLE,
            last_input: ControlInput {
                throttle: IDLE_THROTTLE,
                ..Default::default()
            },
            gamepad_throttle_engaged: false,
        }
    }

    /// Read current input from gamepad and keyboard
    pub fn read(&mut self) -> ControlInput {
        // Process any pending gamepad events
        if let Some(ref mut gilrs) = self.gilrs {
            while let Some(_event) = gilrs.next_event() {
                // Just consume events to update state
            }
        }

        // Read gamepad input; absent, it contributes nothing and every axis
        // falls through to the keyboard.
        let gamepad_input = self.read_gamepad().unwrap_or_default();

        // Read keyboard input
        let keyboard_input = self.read_keyboard();

        let combined = combine(gamepad_input, keyboard_input, self.gamepad_throttle_engaged);

        // Apply some smoothing
        let smoothing = 0.3;
        let smoothed = ControlInput {
            elevator: self.last_input.elevator * (1.0 - smoothing) + combined.elevator * smoothing,
            aileron: self.last_input.aileron * (1.0 - smoothing) + combined.aileron * smoothing,
            rudder: self.last_input.rudder * (1.0 - smoothing) + combined.rudder * smoothing,
            throttle: self.last_input.throttle * (1.0 - smoothing) + combined.throttle * smoothing,
        };

        self.last_input = smoothed;
        smoothed
    }

    /// Read the connected gamepad, or `None` when there is no pad to read.
    fn read_gamepad(&mut self) -> Option<ControlInput> {
        let axes = self.gilrs.as_ref().and_then(|gilrs| {
            let (_id, gamepad) = gilrs.gamepads().find(|(_, g)| g.is_connected())?;
            Some((
                gamepad.value(Axis::LeftStickX) as f64,
                gamepad.value(Axis::LeftStickY) as f64,
                gamepad.value(Axis::RightStickX) as f64,
                gamepad.value(Axis::RightStickY) as f64,
            ))
        });
        let Some((left_x, left_y, right_x, right_y)) = axes else {
            // The latch goes away with the pad it speaks for, so unplugging
            // one hands the throttle back to the keyboard.
            self.gamepad_throttle_engaged = false;
            return None;
        };

        let left_x = apply_deadzone(left_x);
        let left_y = apply_deadzone(left_y);
        let right_x = apply_deadzone(right_x);
        let right_y = apply_deadzone(right_y);

        // The deadzone reads exactly zero at centre, so any other value means
        // the throttle stick has been moved (or never self-centred).
        let throttle_axis = match self.stick_mode {
            StickMode::Mode2 => left_y,
            StickMode::Mode1 => right_y,
        };
        self.gamepad_throttle_engaged |= throttle_axis != 0.0;

        // Map based on stick mode
        Some(match self.stick_mode {
            StickMode::Mode2 => {
                // Mode 2: Left = Throttle(Y)/Rudder(X), Right = Elevator(Y)/Aileron(X)
                ControlInput {
                    throttle: (left_y + 1.0) / 2.0,          // Convert -1..1 to 0..1
                    rudder: -left_x * MAX_RUDDER_RAD,       // Stick right = yaw right = -rudder
                    elevator: right_y * MAX_DEFLECTION_RAD, // Stick up = nose up
                    aileron: right_x * MAX_DEFLECTION_RAD,
                }
            }
            StickMode::Mode1 => {
                // Mode 1: Left = Elevator(Y)/Rudder(X), Right = Throttle(Y)/Aileron(X)
                ControlInput {
                    elevator: left_y * MAX_DEFLECTION_RAD, // Stick up = nose up
                    rudder: -left_x * MAX_RUDDER_RAD,      // Stick right = yaw right = -rudder
                    throttle: (right_y + 1.0) / 2.0,
                    aileron: right_x * MAX_DEFLECTION_RAD,
                }
            }
        })
    }

    /// Read keyboard input
    fn read_keyboard(&mut self) -> ControlInput {
        let Some(device_state) = &self.device_state else {
            return ControlInput {
                throttle: self.keyboard_throttle,
                ..Default::default()
            };
        };
        let keys = device_state.get_keys();

        // Throttle (W/S) - incremental
        if keys.contains(&Keycode::W) {
            self.keyboard_throttle = (self.keyboard_throttle + 0.01).min(1.0);
        }
        if keys.contains(&Keycode::S) {
            self.keyboard_throttle = (self.keyboard_throttle - 0.01).max(0.0);
        }

        // Rudder (Q/E or A/D): Q/A = yaw left = +rudder (TE-left), E/D = yaw right.
        let rudder = if keys.contains(&Keycode::Q) || keys.contains(&Keycode::A) {
            MAX_RUDDER_RAD * 0.5
        } else if keys.contains(&Keycode::E) || keys.contains(&Keycode::D) {
            -MAX_RUDDER_RAD * 0.5
        } else {
            0.0
        };

        // Elevator (Up/Down arrows): up arrow = nose up.
        let elevator = if keys.contains(&Keycode::Up) {
            MAX_DEFLECTION_RAD * 0.5
        } else if keys.contains(&Keycode::Down) {
            -MAX_DEFLECTION_RAD * 0.5
        } else {
            0.0
        };

        // Aileron (Left/Right arrows)
        let aileron = if keys.contains(&Keycode::Left) {
            -MAX_DEFLECTION_RAD * 0.5
        } else if keys.contains(&Keycode::Right) {
            MAX_DEFLECTION_RAD * 0.5
        } else {
            0.0
        };

        ControlInput {
            elevator,
            aileron,
            rudder,
            throttle: self.keyboard_throttle,
        }
    }
}

fn try_device_state() -> Option<DeviceState> {
    #[cfg(target_os = "linux")]
    {
        if !x11_display_reachable() {
            return None;
        }
    }
    Some(DeviceState::new())
}

#[cfg(target_os = "linux")]
fn x11_display_reachable() -> bool {
    let Ok(display) = std::env::var("DISPLAY") else {
        return false;
    };
    let Some(path) = x11_unix_socket_path(&display) else {
        return false;
    };
    std::os::unix::net::UnixStream::connect(path).is_ok()
}

/// Filesystem socket for a local `DISPLAY` (`:0`, `:0.0`, `unix:0`).
#[cfg(any(target_os = "linux", test))]
fn x11_unix_socket_path(display: &str) -> Option<std::path::PathBuf> {
    let rest = display
        .strip_prefix("unix:")
        .or_else(|| display.strip_prefix(':'))?;
    let n = rest.split('.').next()?;
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(std::path::PathBuf::from(format!("/tmp/.X11-unix/X{n}")))
}

/// Apply deadzone to axis value
fn apply_deadzone(value: f64) -> f64 {
    if value.abs() < DEADZONE {
        0.0
    } else {
        // Rescale to remove deadzone gap
        let sign = value.signum();
        let magnitude = (value.abs() - DEADZONE) / (1.0 - DEADZONE);
        sign * magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An untouched gamepad: sticks centred, which the throttle mapping
    /// `(axis + 1) / 2` turns into mid-throttle.
    const IDLE_GAMEPAD: ControlInput = ControlInput {
        elevator: 0.0,
        aileron: 0.0,
        rudder: 0.0,
        throttle: 0.5,
    };

    /// The keyboard at rest, before any key is pressed.
    const IDLE_KEYBOARD: ControlInput = ControlInput {
        elevator: 0.0,
        aileron: 0.0,
        rudder: 0.0,
        throttle: IDLE_THROTTLE,
    };

    #[test]
    fn untouched_gamepad_keeps_the_idle_throttle() {
        assert_eq!(
            combine(IDLE_GAMEPAD, IDLE_KEYBOARD, false).throttle,
            IDLE_THROTTLE
        );
    }

    #[test]
    fn untouched_gamepad_does_not_out_vote_the_keyboard_throttle() {
        let keyboard = ControlInput {
            throttle: 0.8,
            ..IDLE_KEYBOARD
        };
        assert_eq!(combine(IDLE_GAMEPAD, keyboard, false).throttle, 0.8);
    }

    /// A ratcheted transmitter (FrSky and friends) reports its throttle stick
    /// far off centre the moment it is plugged in, so once latched it rightly
    /// wins the throttle merge.
    #[test]
    fn a_ratcheted_transmitter_wins_the_throttle() {
        let transmitter = ControlInput {
            throttle: 0.75,
            ..IDLE_GAMEPAD
        };
        assert_eq!(combine(transmitter, IDLE_KEYBOARD, true).throttle, 0.75);
    }

    /// A gamepad axis that trim or calibration parks a little past the deadzone
    /// must not silence the pilot who is flying: the keys have to reach the
    /// surfaces, even on the axis the trim sits on.
    #[test]
    fn a_trimmed_gamepad_axis_does_not_swallow_a_held_key() {
        for raw in [0.1021, 0.11, 0.15, -0.15, 0.17] {
            let gamepad = ControlInput {
                elevator: apply_deadzone(raw) * MAX_DEFLECTION_RAD,
                aileron: apply_deadzone(raw) * MAX_DEFLECTION_RAD,
                rudder: apply_deadzone(raw) * MAX_RUDDER_RAD,
                ..IDLE_GAMEPAD
            };
            let keyboard = ControlInput {
                elevator: -MAX_DEFLECTION_RAD * 0.5,
                aileron: MAX_DEFLECTION_RAD * 0.5,
                rudder: -MAX_RUDDER_RAD * 0.5,
                ..IDLE_KEYBOARD
            };
            let combined = combine(gamepad, keyboard, false);
            assert_eq!(combined.elevator, keyboard.elevator, "raw axis {raw}");
            assert_eq!(combined.aileron, keyboard.aileron, "raw axis {raw}");
            assert_eq!(combined.rudder, keyboard.rudder, "raw axis {raw}");
        }
    }

    /// A real stick still beats the keyboard on the same axis, in either
    /// direction: the gamepad is the primary control.
    #[test]
    fn a_deflected_stick_beats_a_held_key() {
        for sign in [1.0, -1.0] {
            let gamepad = ControlInput {
                elevator: sign * apply_deadzone(0.8) * MAX_DEFLECTION_RAD,
                ..IDLE_GAMEPAD
            };
            let keyboard = ControlInput {
                elevator: MAX_DEFLECTION_RAD * 0.5,
                ..IDLE_KEYBOARD
            };
            assert_eq!(combine(gamepad, keyboard, false).elevator, gamepad.elevator);
        }
    }

    /// The smallest input a pilot can actually make reaches the surfaces
    /// untouched: arrow keys and A/D are half throw.
    #[test]
    fn the_smallest_keyboard_input_reaches_the_surfaces() {
        for keyboard in [
            ControlInput {
                elevator: MAX_DEFLECTION_RAD * 0.5,
                ..IDLE_KEYBOARD
            },
            ControlInput {
                aileron: MAX_DEFLECTION_RAD * 0.5,
                ..IDLE_KEYBOARD
            },
            ControlInput {
                rudder: MAX_RUDDER_RAD * 0.5,
                ..IDLE_KEYBOARD
            },
        ] {
            let combined = combine(IDLE_GAMEPAD, keyboard, false);
            assert_eq!(combined.elevator, keyboard.elevator);
            assert_eq!(combined.aileron, keyboard.aileron);
            assert_eq!(combined.rudder, keyboard.rudder);
        }
    }

    #[test]
    fn a_deflected_stick_reaches_the_surfaces() {
        let gamepad = ControlInput {
            aileron: apply_deadzone(0.35) * MAX_DEFLECTION_RAD,
            ..IDLE_GAMEPAD
        };
        assert_eq!(
            combine(gamepad, IDLE_KEYBOARD, false).aileron,
            gamepad.aileron
        );
    }

    /// A pad that goes away contributes a zeroed input with its throttle latch
    /// cleared, so W/S drive the throttle again instead of being out-voted by
    /// a controller that is no longer there.
    #[test]
    fn a_vanished_gamepad_hands_the_throttle_back() {
        let keyboard = ControlInput {
            throttle: 0.9,
            elevator: 0.2,
            ..IDLE_KEYBOARD
        };
        let combined = combine(ControlInput::default(), keyboard, false);
        assert_eq!(combined.throttle, 0.9);
        assert_eq!(combined.elevator, 0.2);
    }

    #[test]
    fn x11_unix_socket_path_parses_local_displays() {
        assert_eq!(
            x11_unix_socket_path(":0").as_deref(),
            Some(std::path::Path::new("/tmp/.X11-unix/X0"))
        );
        assert_eq!(
            x11_unix_socket_path(":0.0").as_deref(),
            Some(std::path::Path::new("/tmp/.X11-unix/X0"))
        );
        assert_eq!(
            x11_unix_socket_path("unix:1").as_deref(),
            Some(std::path::Path::new("/tmp/.X11-unix/X1"))
        );
        assert_eq!(x11_unix_socket_path("localhost:0"), None);
        assert_eq!(x11_unix_socket_path(""), None);
    }
}
