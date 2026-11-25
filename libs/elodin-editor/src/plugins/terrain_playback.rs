//! Per-editor terrain policy. Never persisted back to the recording.
use bevy::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainMode {
    Offline,
    Hybrid,
}

#[derive(Debug, thiserror::Error)]
pub enum TerrainOptionError {
    #[error("terrain mode must be offline or hybrid")]
    Mode,
}

impl std::str::FromStr for TerrainMode {
    type Err = TerrainOptionError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "offline" => Ok(Self::Offline),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(TerrainOptionError::Mode),
        }
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct TerrainPlayback {
    pub mode: Option<TerrainMode>,
}

impl TerrainPlayback {
    pub fn parse(mode: Option<&str>) -> Result<Self, TerrainOptionError> {
        Ok(Self {
            mode: mode.map(str::parse).transpose()?,
        })
    }
}

pub(super) fn install(app: &mut App) {
    app.init_resource::<TerrainPlayback>();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_reject_mode_typos() {
        assert_eq!(
            TerrainPlayback::parse(Some("offline")).unwrap().mode,
            Some(TerrainMode::Offline)
        );
        assert!(TerrainPlayback::parse(Some("ofline")).is_err());
        assert_eq!(
            TerrainPlayback::parse(None).unwrap(),
            TerrainPlayback::default()
        );
    }
}
