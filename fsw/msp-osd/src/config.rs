use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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
