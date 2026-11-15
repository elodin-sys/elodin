//! Per-editor terrain policy. Never persisted back to the recording.
use bevy::prelude::*;
use bevy_geo_frames::{GeoFrame, GeoPosition, GeoRotation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainMode {
    Offline,
    Hybrid,
}

#[derive(Debug, thiserror::Error)]
pub enum TerrainOptionError {
    #[error("terrain mode must be offline or hybrid")]
    Mode,
    #[error("terrain basemap must be a relative DB asset path ending in .glb")]
    Basemap,
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
    pub basemap: Option<String>,
}

impl TerrainPlayback {
    pub fn parse(mode: Option<&str>, basemap: Option<&str>) -> Result<Self, TerrainOptionError> {
        let mode = mode.map(str::parse).transpose()?;
        let basemap = basemap.map(|s| s.strip_prefix("db:").unwrap_or(s));
        if basemap.is_some_and(|s| {
            !s.ends_with(".glb")
                || s.split('/')
                    .any(|p| p.is_empty() || matches!(p, "." | ".."))
                || s.chars().any(|c| c.is_control() || ":\\?#%".contains(c))
        }) {
            return Err(TerrainOptionError::Basemap);
        }
        Ok(Self {
            mode,
            basemap: basemap.map(str::to_owned),
        })
    }
}

#[derive(Component)]
struct TerrainBasemap;

pub(super) fn install(app: &mut App) {
    app.init_resource::<TerrainPlayback>()
        .add_systems(Update, sync_basemap)
        .add_observer(basemap_ready);
}

#[expect(clippy::too_many_arguments, reason = "Bevy system parameters")]
fn sync_basemap(
    mut commands: Commands,
    options: Res<TerrainPlayback>,
    schematic: Option<Res<crate::ui::schematic::CurrentSchematic>>,
    connection: Option<Res<impeller_bevy::ConnectionAddr>>,
    assets: Res<AssetServer>,
    big_space: Res<crate::spatial::BigSpaceRootEntity>,
    roots: Query<Entity, With<TerrainBasemap>>,
    mut previous: Local<Option<String>>,
) {
    let path = options.basemap.clone().or_else(|| {
        let impeller_wkt::TerrainConfig::Asset { path } = schematic.as_ref()?.terrain.as_ref()?
        else {
            return None;
        };
        let path = path.strip_prefix("db:").unwrap_or(path);
        let (directory, _) = path.rsplit_once('/')?;
        Some(format!("{directory}/earth-lowres.glb"))
    });
    let url = path.and_then(|path| {
        Some(
            connection
                .as_ref()?
                .endpoint()
                .assets_url(&path)
                .to_string(),
        )
    });
    if *previous == url {
        return;
    }
    for root in &roots {
        commands.entity(root).despawn();
    }
    *previous = url.clone();
    if let Some(url) = url {
        commands.spawn((
            TerrainBasemap,
            Name::new("Terrain global background"),
            WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(url))),
            ChildOf(big_space.0),
            Transform::default(),
            crate::spatial::GridCell::default(),
            GeoPosition(GeoFrame::ECEF, bevy::math::DVec3::ZERO),
            GeoRotation::absolute(GeoFrame::ECEF, bevy::math::DQuat::IDENTITY),
        ));
    }
}

fn basemap_ready(
    event: On<bevy::world_serialization::WorldInstanceReady>,
    roots: Query<(), With<TerrainBasemap>>,
    children: Query<&Children>,
    mut commands: Commands,
) {
    if roots.get(event.entity).is_err() {
        return;
    }
    for child in children.iter_descendants(event.entity) {
        commands.entity(child).insert((
            bevy::camera::visibility::RenderLayers::default()
                .with(super::render_layer_alloc::CINEMATIC_EARTH_RENDER_LAYER),
            bevy::light::NotShadowCaster,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_reject_typos_and_external_or_escaping_assets() {
        assert_eq!(
            TerrainPlayback::parse(Some("offline"), Some("db:terrain/earth.glb"))
                .unwrap()
                .mode,
            Some(TerrainMode::Offline)
        );
        assert!(TerrainPlayback::parse(Some("ofline"), None).is_err());
        for path in [
            "https://example/a.glb",
            "../a.glb",
            "/a.glb",
            "a%2f.glb",
            "a/../b.glb",
            "earth.png",
        ] {
            assert!(TerrainPlayback::parse(None, Some(path)).is_err());
        }
        assert_eq!(
            TerrainPlayback::parse(None, None).unwrap(),
            TerrainPlayback::default()
        );
    }
}
