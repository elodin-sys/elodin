use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub db: DbConfig,
    pub osd: OsdConfig,
    pub serial: SerialConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    pub host: String,
    pub port: u16,
    pub components: Vec<String>,
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

impl Default for Config {
    fn default() -> Self {
        Self {
            db: DbConfig {
                host: "127.0.0.1".to_string(),
                port: 2240,
                components: vec![
                    "drone.gyro".to_string(),
                    "drone.accel".to_string(),
                    "drone.magnetometer".to_string(),
                    "drone.attitude_target".to_string(),
                    "drone.body_ang_vel".to_string(),
                    "drone.world_pos".to_string(),
                    "drone.world_vel".to_string(),
                ],
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

    #[allow(dead_code)]
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let contents = toml::to_string_pretty(self)?;
        fs::write(path, contents)?;
        Ok(())
    }
}
