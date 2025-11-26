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
    /// The Elodin-DB component name (e.g., "drone.world_pos")
    pub component: String,
    /// Array index for x value
    pub x: usize,
    /// Array index for y value
    pub y: usize,
    /// Array index for z value
    pub z: usize,
}

/// Mapping for a quaternion (q0, q1, q2, q3) from a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuatMapping {
    /// The Elodin-DB component name (e.g., "drone.world_pos")
    pub component: String,
    /// Array index for q0 (w) value
    pub q0: usize,
    /// Array index for q1 (x) value
    pub q1: usize,
    /// Array index for q2 (y) value
    pub q2: usize,
    /// Array index for q3 (z) value
    pub q3: usize,
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
                // Default to drone example mappings
                position: Vec3Mapping {
                    component: "drone.world_pos".to_string(),
                    x: 4,
                    y: 5,
                    z: 6,
                },
                orientation: QuatMapping {
                    component: "drone.world_pos".to_string(),
                    q0: 0,
                    q1: 1,
                    q2: 2,
                    q3: 3,
                },
                velocity: Vec3Mapping {
                    component: "drone.world_vel".to_string(),
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
