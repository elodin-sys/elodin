//! Input handling for gamepad and keyboard
//!
//! Supports:
//! - Gamepad: FrSky X20 R5 (appears as USB HID joystick)
//! - Keyboard: WASD (left stick) + Arrow keys (right stick)
//!
//! Stick modes:
//! - Mode 2 (US standard): Left stick = Throttle/Yaw, Right stick = Pitch/Roll
//! - Mode 1 (EU/Asia): Left stick = Pitch/Yaw, Right stick = Throttle/Roll

use device_query::{DeviceQuery, DeviceState, Keycode};
use gilrs::{Axis, Gilrs};
use std::f64::consts::PI;

use crate::demo::IdlePilot;

/// Throttle held when no one is on the sticks: enough for level flight.
const IDLE_THROTTLE: f64 = 0.3;

/// Stick travel around centre that reads as neutral.
const DEADZONE: f64 = 0.1;
/// Full elevator/aileron throw, in radians (25°).
const MAX_DEFLECTION_RAD: f64 = 25.0 * PI / 180.0;
/// Full rudder throw, in radians (30°).
const MAX_RUDDER_RAD: f64 = 30.0 * PI / 180.0;

/// Fraction of full throw a surface must move before someone counts as flying.
///
/// A tenth of throw sits between the two cases that matter. Below it: a gamepad
/// axis that trim or a sloppy calibration parks a little past the deadzone,
/// which the `(axis - deadzone) / (1 - deadzone)` rescale then multiplies by the
/// throw — a raw 0.15 is only 1.4° of elevator, and a raw 0.1021 is the 0.057°
/// that a bare `1e-3` rad test would have read as a pilot on the sticks. Above
/// it: every input a pilot can actually make, the smallest being the half-throw
/// of an arrow key.
const PILOT_INPUT_FRACTION: f64 = 0.1;

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

/// Whether a human is on the sticks: any control surface deliberately moved,
/// i.e. past [`PILOT_INPUT_FRACTION`] of its throw. Merely off neutral is not
/// enough — a resting gamepad axis is rarely exactly centred.
///
/// The throttle is deliberately no evidence at all. A ratcheted transmitter
/// reports a throttle stick far off centre from the moment it is plugged in,
/// so reading that as "someone is flying" would retire the idle pilot before
/// it ever flew a bank.
fn is_flying(input: &ControlInput) -> bool {
    input.elevator.abs() > PILOT_INPUT_FRACTION * MAX_DEFLECTION_RAD
        || input.aileron.abs() > PILOT_INPUT_FRACTION * MAX_DEFLECTION_RAD
        || input.rudder.abs() > PILOT_INPUT_FRACTION * MAX_RUDDER_RAD
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
/// out-vote an arrow key's half throw and leave [`is_flying`] reading the trim
/// instead of the pilot.
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
    /// Flies gentle banks until the human takes over
    idle_pilot: IdlePilot,
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
            tracing::info!("No X display; keyboard input disabled, idle pilot flies");
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
            idle_pilot: IdlePilot::default(),
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

        let mut combined = combine(gamepad_input, keyboard_input, self.gamepad_throttle_engaged);

        // Hand the ailerons to the idle pilot until the human flies. Done
        // before smoothing so the handover blends like any other input.
        if let Some(aileron) = self.idle_pilot.aileron(is_flying(&combined)) {
            combined.aileron = aileron;
        }

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
                    throttle: (left_y + 1.0) / 2.0, // Convert -1..1 to 0..1
                    rudder: left_x * MAX_RUDDER_RAD,
                    elevator: -right_y * MAX_DEFLECTION_RAD, // Inverted: stick up = nose up = negative
                    aileron: right_x * MAX_DEFLECTION_RAD,
                }
            }
            StickMode::Mode1 => {
                // Mode 1: Left = Elevator(Y)/Rudder(X), Right = Throttle(Y)/Aileron(X)
                ControlInput {
                    elevator: -left_y * MAX_DEFLECTION_RAD, // Inverted
                    rudder: left_x * MAX_RUDDER_RAD,
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

        // Rudder (A/D)
        let rudder = if keys.contains(&Keycode::A) {
            -MAX_RUDDER_RAD * 0.5 // Half max deflection
        } else if keys.contains(&Keycode::D) {
            MAX_RUDDER_RAD * 0.5
        } else {
            0.0
        };

        // Elevator (Up/Down arrows)
        let elevator = if keys.contains(&Keycode::Up) {
            -MAX_DEFLECTION_RAD * 0.5 // Up arrow = nose up = negative elevator
        } else if keys.contains(&Keycode::Down) {
            MAX_DEFLECTION_RAD * 0.5
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
        let combined = combine(IDLE_GAMEPAD, IDLE_KEYBOARD, false);
        assert_eq!(combined.throttle, IDLE_THROTTLE);
        assert!(!is_flying(&combined));
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
    /// far off centre the moment it is plugged in. It rightly wins the throttle,
    /// but it must not pass for a pilot on the sticks, or the idle banks the
    /// README promises would never fly with the very hardware it highlights.
    #[test]
    fn a_ratcheted_transmitter_still_lets_the_idle_pilot_fly() {
        let transmitter = ControlInput {
            throttle: 0.75,
            ..IDLE_GAMEPAD
        };
        let combined = combine(transmitter, IDLE_KEYBOARD, true);
        assert_eq!(combined.throttle, 0.75);
        assert!(!is_flying(&combined));
        assert!(
            IdlePilot::default().aileron(is_flying(&combined)).is_some(),
            "a connected transmitter must not retire the idle pilot"
        );
    }

    /// A gamepad axis that trim or calibration parks a little past the
    /// deadzone. The rescale makes such an axis map to a fraction of a degree
    /// of surface, which is nobody flying — the idle banks must still run, on
    /// the very FrSky-style hardware the README highlights.
    #[test]
    fn a_trimmed_gamepad_axis_still_lets_the_idle_pilot_fly() {
        for raw in [0.1021, 0.11, 0.15, -0.15, 0.17] {
            let gamepad = ControlInput {
                elevator: apply_deadzone(raw) * MAX_DEFLECTION_RAD,
                aileron: apply_deadzone(raw) * MAX_DEFLECTION_RAD,
                rudder: apply_deadzone(raw) * MAX_RUDDER_RAD,
                ..IDLE_GAMEPAD
            };
            let combined = combine(gamepad, IDLE_KEYBOARD, false);
            assert!(!is_flying(&combined), "raw axis {raw} read as flying");
            assert!(
                IdlePilot::default().aileron(is_flying(&combined)).is_some(),
                "raw axis {raw} retired the idle pilot"
            );
        }
    }

    /// A trimmed axis is nobody flying, but it must not silence the pilot who
    /// is: the keys have to reach the surfaces and hand the aircraft over, even
    /// on the axis the trim sits on.
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
            assert!(is_flying(&combined), "raw axis {raw} hid the pilot");
            assert!(
                IdlePilot::default().aileron(is_flying(&combined)).is_none(),
                "raw axis {raw} kept the idle pilot flying under a held key"
            );
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

    /// The idle pilot's own fingertip aileron is not a pilot input either, or
    /// the demo would retire itself on the tick after it started.
    #[test]
    fn the_idle_pilot_does_not_retire_itself() {
        let aileron = IdlePilot::default()
            .aileron(false)
            .expect("idle pilot flies");
        let combined = combine(
            ControlInput {
                aileron,
                ..IDLE_GAMEPAD
            },
            IDLE_KEYBOARD,
            false,
        );
        assert!(!is_flying(&combined));
    }

    /// The smallest input a pilot can actually make, from either device, has to
    /// hand over: arrow keys and A/D are half throw.
    #[test]
    fn the_smallest_keyboard_input_flies() {
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
            assert!(is_flying(&combine(IDLE_GAMEPAD, keyboard, false)));
        }
    }

    /// A stick moved well clear of the deadzone: a quarter of throw is already
    /// a deliberate input.
    #[test]
    fn a_deflected_stick_flies() {
        let gamepad = ControlInput {
            aileron: apply_deadzone(0.35) * MAX_DEFLECTION_RAD,
            ..IDLE_GAMEPAD
        };
        assert!(is_flying(&combine(gamepad, IDLE_KEYBOARD, false)));
    }

    #[test]
    fn stick_deflection_flies() {
        let gamepad = ControlInput {
            aileron: 0.1,
            ..IDLE_GAMEPAD
        };
        let combined = combine(gamepad, IDLE_KEYBOARD, false);
        assert_eq!(combined.aileron, 0.1);
        assert!(is_flying(&combined));
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

    /// Throttling up with W is not flying either: the demo keeps the wings
    /// working until a surface moves, which is what the ADI is there to show.
    #[test]
    fn throttle_alone_is_not_flying() {
        let keyboard = ControlInput {
            throttle: IDLE_THROTTLE + 0.05,
            ..IDLE_KEYBOARD
        };
        assert!(!is_flying(&combine(IDLE_GAMEPAD, keyboard, false)));
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
