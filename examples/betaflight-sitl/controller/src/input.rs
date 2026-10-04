//! Gamepad and keyboard input for manual Betaflight piloting.

use device_query::{DeviceQuery, DeviceState, Keycode};
use gilrs::{Axis, Button, Gilrs};

const DEADZONE: f64 = 0.10;
const KEY_AXIS: f64 = 0.50;
const THROTTLE_STEP: f64 = 0.01;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StickMode {
    #[default]
    Mode2,
    Mode1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlInput {
    /// Positive rolls right (right wing down).
    pub roll: f64,
    /// Positive pitches forward (nose down).
    pub pitch: f64,
    pub throttle: f64,
    /// Positive yaws right (clockwise from above).
    pub yaw: f64,
    pub armed: bool,
    pub angle_mode: bool,
}

impl ControlInput {
    pub fn safe() -> Self {
        Self {
            roll: 0.0,
            pitch: 0.0,
            throttle: 0.0,
            yaw: 0.0,
            armed: false,
            angle_mode: true,
        }
    }

    pub fn as_array(self) -> [f64; 6] {
        [
            self.roll,
            self.pitch,
            self.throttle,
            self.yaw,
            f64::from(self.armed),
            f64::from(self.angle_mode),
        ]
    }
}

impl Default for ControlInput {
    fn default() -> Self {
        Self::safe()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Axes {
    roll: f64,
    pitch: f64,
    throttle: f64,
    yaw: f64,
}

pub struct InputReader {
    gilrs: Option<Gilrs>,
    /// device_query aborts on Linux when no reachable X display exists.
    keyboard: Option<DeviceState>,
    stick_mode: StickMode,
    keyboard_throttle: f64,
    gamepad_throttle_engaged: bool,
    had_gamepad: bool,
    state: ControlInput,
    last_axes: Axes,
    previous_arm_key: bool,
    previous_disarm_key: bool,
    previous_mode_key: bool,
    previous_pad_arm: bool,
    previous_pad_disarm: bool,
    previous_pad_mode: bool,
}

impl InputReader {
    pub fn new(stick_mode: StickMode) -> Self {
        let gilrs = match Gilrs::new() {
            Ok(gilrs) => Some(gilrs),
            Err(error) => {
                tracing::warn!(%error, "gamepad input unavailable");
                None
            }
        };
        let keyboard = try_device_state();
        if keyboard.is_none() {
            tracing::info!("no reachable X display; keyboard input disabled");
        }
        Self {
            gilrs,
            keyboard,
            stick_mode,
            keyboard_throttle: 0.0,
            gamepad_throttle_engaged: false,
            had_gamepad: false,
            state: ControlInput::safe(),
            last_axes: Axes::default(),
            previous_arm_key: false,
            previous_disarm_key: false,
            previous_mode_key: false,
            previous_pad_arm: false,
            previous_pad_disarm: false,
            previous_pad_mode: false,
        }
    }

    pub fn read(&mut self) -> ControlInput {
        if let Some(gilrs) = &mut self.gilrs {
            while gilrs.next_event().is_some() {}
        }

        let gamepad = self.read_gamepad();
        let gamepad_connected = gamepad.is_some();
        if self.had_gamepad && !gamepad_connected {
            // A disappearing physical controller is always an immediate hard
            // disarm. The pilot may deliberately re-arm from the keyboard.
            self.reset_safe();
            self.had_gamepad = false;
            return self.state;
        }
        self.had_gamepad = gamepad_connected;

        let keyboard = self.read_keyboard();
        if gamepad.is_none() && keyboard.is_none() {
            self.reset_safe();
            return self.state;
        }

        let keyboard_axes = keyboard.unwrap_or(Axes {
            throttle: self.keyboard_throttle,
            ..Axes::default()
        });
        let gamepad_axes = gamepad.unwrap_or_default();
        let target = Axes {
            roll: dominant(gamepad_axes.roll, keyboard_axes.roll),
            pitch: dominant(gamepad_axes.pitch, keyboard_axes.pitch),
            yaw: dominant(gamepad_axes.yaw, keyboard_axes.yaw),
            throttle: if self.gamepad_throttle_engaged {
                gamepad_axes.throttle
            } else {
                keyboard_axes.throttle
            },
        };

        // Smooth analog/key axes, but never smooth safety switches.
        const ALPHA: f64 = 0.35;
        self.last_axes = Axes {
            roll: blend(self.last_axes.roll, target.roll, ALPHA),
            pitch: blend(self.last_axes.pitch, target.pitch, ALPHA),
            throttle: blend(self.last_axes.throttle, target.throttle, ALPHA),
            yaw: blend(self.last_axes.yaw, target.yaw, ALPHA),
        };
        self.state.roll = self.last_axes.roll.clamp(-1.0, 1.0);
        self.state.pitch = self.last_axes.pitch.clamp(-1.0, 1.0);
        self.state.throttle = self.last_axes.throttle.clamp(0.0, 1.0);
        self.state.yaw = self.last_axes.yaw.clamp(-1.0, 1.0);
        self.state
    }

    fn reset_safe(&mut self) {
        self.state = ControlInput::safe();
        self.last_axes = Axes::default();
        self.keyboard_throttle = 0.0;
        self.gamepad_throttle_engaged = false;
    }

    fn read_gamepad(&mut self) -> Option<Axes> {
        let (_id, pad) = self
            .gilrs
            .as_ref()?
            .gamepads()
            .find(|(_, gamepad)| gamepad.is_connected())?;
        let left_x = apply_deadzone(pad.value(Axis::LeftStickX) as f64);
        let left_y = apply_deadzone(pad.value(Axis::LeftStickY) as f64);
        let right_x = apply_deadzone(pad.value(Axis::RightStickX) as f64);
        let right_y = apply_deadzone(pad.value(Axis::RightStickY) as f64);

        let throttle_axis = match self.stick_mode {
            StickMode::Mode2 => left_y,
            StickMode::Mode1 => right_y,
        };
        self.gamepad_throttle_engaged |= throttle_axis != 0.0;

        let arm = pad.is_pressed(Button::South);
        let disarm = pad.is_pressed(Button::East);
        let mode = pad.is_pressed(Button::North);
        if arm && !self.previous_pad_arm && self.state.throttle <= 0.05 {
            self.state.armed = true;
        }
        if disarm && !self.previous_pad_disarm {
            self.state.armed = false;
            self.keyboard_throttle = 0.0;
            self.gamepad_throttle_engaged = false;
        }
        if mode && !self.previous_pad_mode {
            self.state.angle_mode = !self.state.angle_mode;
        }
        self.previous_pad_arm = arm;
        self.previous_pad_disarm = disarm;
        self.previous_pad_mode = mode;

        Some(map_gamepad_axes(
            self.stick_mode,
            left_x,
            left_y,
            right_x,
            right_y,
        ))
    }

    fn read_keyboard(&mut self) -> Option<Axes> {
        let keys = self.keyboard.as_ref()?.get_keys();
        if keys.contains(&Keycode::W) {
            self.keyboard_throttle = (self.keyboard_throttle + THROTTLE_STEP).min(1.0);
        }
        if keys.contains(&Keycode::S) {
            self.keyboard_throttle = (self.keyboard_throttle - THROTTLE_STEP).max(0.0);
        }

        let arm = keys.contains(&Keycode::R)
            && (keys.contains(&Keycode::LShift) || keys.contains(&Keycode::RShift));
        let disarm = keys.contains(&Keycode::F);
        let mode = keys.contains(&Keycode::M);
        if arm && !self.previous_arm_key && self.state.throttle <= 0.05 {
            self.state.armed = true;
        }
        if disarm && !self.previous_disarm_key {
            self.state.armed = false;
            self.keyboard_throttle = 0.0;
            self.gamepad_throttle_engaged = false;
        }
        if mode && !self.previous_mode_key {
            self.state.angle_mode = !self.state.angle_mode;
        }
        self.previous_arm_key = arm;
        self.previous_disarm_key = disarm;
        self.previous_mode_key = mode;

        Some(Axes {
            throttle: self.keyboard_throttle,
            yaw: if keys.contains(&Keycode::Q) || keys.contains(&Keycode::A) {
                -KEY_AXIS
            } else if keys.contains(&Keycode::E) || keys.contains(&Keycode::D) {
                KEY_AXIS
            } else {
                0.0
            },
            pitch: if keys.contains(&Keycode::Up) {
                KEY_AXIS
            } else if keys.contains(&Keycode::Down) {
                -KEY_AXIS
            } else {
                0.0
            },
            roll: if keys.contains(&Keycode::Left) {
                -KEY_AXIS
            } else if keys.contains(&Keycode::Right) {
                KEY_AXIS
            } else {
                0.0
            },
        })
    }
}

fn map_gamepad_axes(
    stick_mode: StickMode,
    left_x: f64,
    left_y: f64,
    right_x: f64,
    right_y: f64,
) -> Axes {
    match stick_mode {
        StickMode::Mode2 => Axes {
            throttle: (left_y + 1.0) / 2.0,
            yaw: left_x,
            pitch: right_y,
            roll: right_x,
        },
        StickMode::Mode1 => Axes {
            pitch: left_y,
            yaw: left_x,
            throttle: (right_y + 1.0) / 2.0,
            roll: right_x,
        },
    }
}

fn blend(previous: f64, target: f64, alpha: f64) -> f64 {
    previous * (1.0 - alpha) + target * alpha
}

fn dominant(gamepad: f64, keyboard: f64) -> f64 {
    if gamepad.abs() >= keyboard.abs() {
        gamepad
    } else {
        keyboard
    }
}

fn apply_deadzone(value: f64) -> f64 {
    if value.abs() < DEADZONE {
        0.0
    } else {
        value.signum() * (value.abs() - DEADZONE) / (1.0 - DEADZONE)
    }
}

fn try_device_state() -> Option<DeviceState> {
    #[cfg(target_os = "linux")]
    if !x11_display_reachable() {
        return None;
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

#[cfg(any(target_os = "linux", test))]
fn x11_unix_socket_path(display: &str) -> Option<std::path::PathBuf> {
    let rest = display
        .strip_prefix("unix:")
        .or_else(|| display.strip_prefix(':'))?;
    let number = rest.split('.').next()?;
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(std::path::PathBuf::from(format!(
        "/tmp/.X11-unix/X{number}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_is_continuous_and_bounded() {
        assert_eq!(apply_deadzone(0.09), 0.0);
        assert_eq!(apply_deadzone(-0.10), 0.0);
        assert_eq!(apply_deadzone(1.0), 1.0);
        assert_eq!(apply_deadzone(-1.0), -1.0);
    }

    #[test]
    fn keyboard_can_override_an_idle_trimmed_axis() {
        assert_eq!(dominant(0.02, -0.5), -0.5);
        assert_eq!(dominant(-0.8, 0.5), -0.8);
    }

    #[test]
    fn mode_two_and_mode_one_assign_sticks_as_documented() {
        let mode2 = map_gamepad_axes(StickMode::Mode2, 0.1, -1.0, 0.3, 0.4);
        assert_eq!(mode2.throttle, 0.0);
        assert_eq!(mode2.yaw, 0.1);
        assert_eq!(mode2.roll, 0.3);
        assert_eq!(mode2.pitch, 0.4);

        let mode1 = map_gamepad_axes(StickMode::Mode1, 0.1, 0.2, 0.3, -1.0);
        assert_eq!(mode1.throttle, 0.0);
        assert_eq!(mode1.yaw, 0.1);
        assert_eq!(mode1.roll, 0.3);
        assert_eq!(mode1.pitch, 0.2);
    }

    #[test]
    fn safe_input_is_disarmed_at_minimum_throttle_in_angle_mode() {
        assert_eq!(
            ControlInput::safe().as_array(),
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn x11_paths_only_accept_local_displays() {
        assert_eq!(
            x11_unix_socket_path(":0.0").as_deref(),
            Some(std::path::Path::new("/tmp/.X11-unix/X0"))
        );
        assert_eq!(x11_unix_socket_path("localhost:0"), None);
    }
}
