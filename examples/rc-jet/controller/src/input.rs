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

/// Input reader combining gamepad and keyboard
pub struct InputReader {
    gilrs: Option<Gilrs>,
    device_state: DeviceState,
    stick_mode: StickMode,
    deadzone: f64,
    /// Max deflection for elevator/aileron in radians (25°)
    max_deflection_rad: f64,
    /// Max deflection for rudder in radians (30°)
    max_rudder_rad: f64,
    /// Current throttle state (keyboard is incremental)
    keyboard_throttle: f64,
    /// Last control input for smoothing
    last_input: ControlInput,
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

        Self {
            gilrs,
            device_state: DeviceState::new(),
            stick_mode,
            deadzone: 0.1,
            max_deflection_rad: 25.0 * PI / 180.0, // 25° in radians
            max_rudder_rad: 30.0 * PI / 180.0,     // 30° in radians
            keyboard_throttle: 0.3,                // Start with some throttle for level flight
            last_input: ControlInput {
                throttle: 0.3,
                ..Default::default()
            },
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

        // Read gamepad input
        let gamepad_input = self.read_gamepad();

        // Read keyboard input
        let keyboard_input = self.read_keyboard();

        // Combine inputs - gamepad takes priority for each axis if non-zero
        let combined = ControlInput {
            elevator: if gamepad_input.elevator.abs() > 0.001 {
                gamepad_input.elevator
            } else {
                keyboard_input.elevator
            },
            aileron: if gamepad_input.aileron.abs() > 0.001 {
                gamepad_input.aileron
            } else {
                keyboard_input.aileron
            },
            rudder: if gamepad_input.rudder.abs() > 0.001 {
                gamepad_input.rudder
            } else {
                keyboard_input.rudder
            },
            throttle: if (gamepad_input.throttle - 0.3).abs() > 0.01 {
                gamepad_input.throttle
            } else {
                keyboard_input.throttle
            },
        };

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

    /// Read gamepad input
    fn read_gamepad(&self) -> ControlInput {
        let Some(ref gilrs) = self.gilrs else {
            return ControlInput {
                throttle: 0.3,
                ..Default::default()
            };
        };

        // Find first connected gamepad
        let Some((_id, gamepad)) = gilrs.gamepads().find(|(_, g)| g.is_connected()) else {
            return ControlInput {
                throttle: 0.3,
                ..Default::default()
            };
        };

        // Read stick axes
        let left_x = self.apply_deadzone(gamepad.value(Axis::LeftStickX) as f64);
        let left_y = self.apply_deadzone(gamepad.value(Axis::LeftStickY) as f64);
        let right_x = self.apply_deadzone(gamepad.value(Axis::RightStickX) as f64);
        let right_y = self.apply_deadzone(gamepad.value(Axis::RightStickY) as f64);

        // Map based on stick mode
        match self.stick_mode {
            StickMode::Mode2 => {
                // Mode 2: Left = Throttle(Y)/Rudder(X), Right = Elevator(Y)/Aileron(X)
                ControlInput {
                    throttle: (left_y + 1.0) / 2.0, // Convert -1..1 to 0..1
                    rudder: left_x * self.max_rudder_rad,
                    elevator: -right_y * self.max_deflection_rad, // Inverted: stick up = nose up = negative
                    aileron: right_x * self.max_deflection_rad,
                }
            }
            StickMode::Mode1 => {
                // Mode 1: Left = Elevator(Y)/Rudder(X), Right = Throttle(Y)/Aileron(X)
                ControlInput {
                    elevator: -left_y * self.max_deflection_rad, // Inverted
                    rudder: left_x * self.max_rudder_rad,
                    throttle: (right_y + 1.0) / 2.0,
                    aileron: right_x * self.max_deflection_rad,
                }
            }
        }
    }

    /// Read keyboard input
    fn read_keyboard(&mut self) -> ControlInput {
        let keys = self.device_state.get_keys();

        // Throttle (W/S) - incremental
        if keys.contains(&Keycode::W) {
            self.keyboard_throttle = (self.keyboard_throttle + 0.01).min(1.0);
        }
        if keys.contains(&Keycode::S) {
            self.keyboard_throttle = (self.keyboard_throttle - 0.01).max(0.0);
        }

        // Rudder (A/D)
        let rudder = if keys.contains(&Keycode::A) {
            -self.max_rudder_rad * 0.5 // Half max deflection
        } else if keys.contains(&Keycode::D) {
            self.max_rudder_rad * 0.5
        } else {
            0.0
        };

        // Elevator (Up/Down arrows)
        let elevator = if keys.contains(&Keycode::Up) {
            -self.max_deflection_rad * 0.5 // Up arrow = nose up = negative elevator
        } else if keys.contains(&Keycode::Down) {
            self.max_deflection_rad * 0.5
        } else {
            0.0
        };

        // Aileron (Left/Right arrows)
        let aileron = if keys.contains(&Keycode::Left) {
            -self.max_deflection_rad * 0.5
        } else if keys.contains(&Keycode::Right) {
            self.max_deflection_rad * 0.5
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

    /// Apply deadzone to axis value
    fn apply_deadzone(&self, value: f64) -> f64 {
        if value.abs() < self.deadzone {
            0.0
        } else {
            // Rescale to remove deadzone gap
            let sign = value.signum();
            let magnitude = (value.abs() - self.deadzone) / (1.0 - self.deadzone);
            sign * magnitude
        }
    }
}
