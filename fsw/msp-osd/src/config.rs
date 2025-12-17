use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Coordinate frame convention for heading interpretation
///
/// - **ENU** (East-North-Up): Used by Elodin simulations. 0°=East, 90°=North.
/// - **NED** (North-East-Down): Aviation convention. 0°=North, 90°=East.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CoordinateFrame {
    /// East-North-Up: 0°=East, 90°=North (Elodin simulation default)
    #[default]
    Enu,
    /// North-East-Down: 0°=North, 90°=East (Aviation convention)
    Ned,
}

impl CoordinateFrame {
    /// Convert a heading from ENU convention to aviation/NED convention
    /// ENU: 0°=East, 90°=North
    /// NED: 0°=North, 90°=East
    pub fn to_aviation_heading(self, heading_deg: f32) -> f32 {
        match self {
            CoordinateFrame::Enu => {
                // Convert ENU to NED: heading_ned = (90 - heading_enu) mod 360
                let ned = 90.0 - heading_deg;
                if ned < 0.0 {
                    ned + 360.0
                } else if ned >= 360.0 {
                    ned - 360.0
                } else {
                    ned
                }
            }
            CoordinateFrame::Ned => {
                // Already in aviation convention
                heading_deg
            }
        }
    }

    /// Convert Z position to display altitude
    /// ENU: Z is up, positive = altitude above reference
    /// NED: Z is down, positive = depth below reference (must negate)
    pub fn to_display_altitude(self, z_position: f32) -> f32 {
        match self {
            CoordinateFrame::Enu => z_position,
            CoordinateFrame::Ned => -z_position,
        }
    }

    /// Convert Z velocity to display climb rate
    /// ENU: positive vz = climbing
    /// NED: positive vz = descending (must negate)
    pub fn to_display_climb_rate(self, z_velocity: f32) -> f32 {
        match self {
            CoordinateFrame::Enu => z_velocity,
            CoordinateFrame::Ned => -z_velocity,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub db: DbConfig,
    pub osd: OsdConfig,
    pub serial: SerialConfig,
    pub inputs: InputMappings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsdConfig {
    pub rows: u8,
    pub cols: u8,
    pub refresh_rate_hz: f32,
    /// Coordinate frame convention for heading interpretation
    /// Defaults to ENU for Elodin simulations, use NED for real aviation hardware
    #[serde(default)]
    pub coordinate_frame: CoordinateFrame,
    /// Character aspect ratio (height/width) for horizon line rendering.
    /// HD OSD systems like Walksnail Avatar use ~12x18 pixel characters (ratio 1.5).
    /// This compensates for non-square characters so the horizon tilt angle
    /// matches the actual aircraft roll angle.
    #[serde(default = "default_char_aspect_ratio")]
    pub char_aspect_ratio: f32,
    /// Pitch scale in degrees per row for the artificial horizon.
    /// Lower values = more sensitive pitch response (horizon moves more per degree).
    /// Should be calibrated to match camera vertical FOV for accurate overlay.
    /// Formula: pitch_scale ≈ camera_vertical_fov / osd_rows
    /// Example: 90° VFOV / 18 rows ≈ 5° per row
    #[serde(default = "default_pitch_scale")]
    pub pitch_scale: f32,
}

fn default_char_aspect_ratio() -> f32 {
    1.5
}

fn default_pitch_scale() -> f32 {
    // Default to 5 degrees per row, suitable for ~90° VFOV cameras
    // on an 18-row OSD grid
    5.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialConfig {
    pub port: String,
    pub baud: u32,
}

/// Input mappings declare how to extract OSD telemetry from Elodin-DB components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMappings {
    pub position: Vec3Mapping,
    pub orientation: QuatMapping,
    pub velocity: Vec3Mapping,
    /// Optional target position for OSD target tracking
    #[serde(default)]
    pub target: Option<Vec3Mapping>,
}

/// Mapping for a 3D vector (x, y, z) from a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vec3Mapping {
    /// The Elodin-DB component name (e.g., "bdx.world_pos")
    pub component: String,
    /// Array index for x value
    pub x: usize,
    /// Array index for y value
    pub y: usize,
    /// Array index for z value
    pub z: usize,
}

/// Mapping for a quaternion from a component
/// Elodin stores quaternions as [x, y, z, w] (scalar w is last)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuatMapping {
    /// The Elodin-DB component name (e.g., "bdx.world_pos")
    pub component: String,
    /// Array index for x (i) component
    pub qx: usize,
    /// Array index for y (j) component
    pub qy: usize,
    /// Array index for z (k) component
    pub qz: usize,
    /// Array index for w (scalar) component
    pub qw: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db: DbConfig {
                host: "127.0.0.1".to_string(),
                port: 2240,
            },
            osd: OsdConfig {
                rows: 18,
                cols: 50,
                refresh_rate_hz: 20.0,
                coordinate_frame: CoordinateFrame::Enu,
                char_aspect_ratio: 1.5,
                pitch_scale: 5.0,
            },
            serial: SerialConfig {
                port: "/dev/ttyTHS7".to_string(),
                baud: 115200,
            },
            inputs: InputMappings {
                // Default to rc jet example mappings
                position: Vec3Mapping {
                    component: "bdx.world_pos".to_string(),
                    x: 4,
                    y: 5,
                    z: 6,
                },
                orientation: QuatMapping {
                    component: "bdx.world_pos".to_string(),
                    qx: 0,
                    qy: 1,
                    qz: 2,
                    qw: 3,
                },
                velocity: Vec3Mapping {
                    component: "bdx.world_vel".to_string(),
                    x: 3,
                    y: 4,
                    z: 5,
                },
                target: None,
            },
        }
    }
}

impl Config {
    pub fn from_file_or_default(path: &str) -> Result<Self> {
        if Path::new(path).exists() {
            let contents = fs::read_to_string(path)?;
            Ok(toml::from_str(&contents)?)
        } else {
            tracing::info!("Config file not found at {}, using defaults", path);
            Ok(Self::default())
        }
    }

    /// Get unique list of component names needed from DB
    #[allow(dead_code)]
    pub fn required_components(&self) -> Vec<String> {
        let mut components = vec![
            self.inputs.position.component.clone(),
            self.inputs.orientation.component.clone(),
            self.inputs.velocity.component.clone(),
        ];
        if let Some(target) = &self.inputs.target {
            components.push(target.component.clone());
        }
        components.sort();
        components.dedup();
        components
    }

    #[allow(dead_code)]
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let contents = toml::to_string_pretty(self)?;
        fs::write(path, contents)?;
        Ok(())
    }
}
