//! Startup asset preference for the optional nearby terrain experiment.
use bevy::{
    asset::{DependencyLoadState, LoadState, RecursiveDependencyLoadState, UntypedAssetId},
    prelude::*,
};
use bevy_cesium::CesiumViews;

const ASSET_PRIORITY_SECONDS: f64 = 10.;
const ASSET_QUIET_SECONDS: f64 = 0.25;

#[derive(Resource, Default)]
pub struct TerrainPriority {
    pub(crate) enabled: bool,
    pub(crate) camera_ready: bool,
    origin: Option<[f64; 3]>,
    started: Option<f64>,
    last_busy: f64,
    idle_frames: u8,
    released: bool,
}

impl TerrainPriority {
    pub fn nearby() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    pub(super) fn reset(&mut self) {
        *self = Self {
            enabled: self.enabled,
            ..Default::default()
        };
    }

    fn observe_camera(&mut self, position: Option<[f64; 3]>) {
        if self.camera_ready && self.origin.is_none() {
            self.origin =
                position.filter(|position| position.iter().all(|value| value.is_finite()));
        }
    }

    fn allow_terrain(&mut self, now: f64, pending: bool) -> bool {
        if !self.enabled || self.released {
            return true;
        }
        if !self.camera_ready {
            return false;
        }
        let started = *self.started.get_or_insert_with(|| {
            self.last_busy = now;
            now
        });
        if pending {
            self.last_busy = now;
            self.idle_frames = 0;
        } else {
            self.idle_frames = self.idle_frames.saturating_add(1);
        }
        self.released = now - started >= ASSET_PRIORITY_SECONDS
            || (self.idle_frames >= 2 && now - self.last_busy >= ASSET_QUIET_SECONDS);
        self.released
    }
}

fn load_pending(
    states: Option<(LoadState, DependencyLoadState, RecursiveDependencyLoadState)>,
) -> bool {
    let Some((root, direct, recursive)) = states else {
        return false;
    };
    if root.is_failed() || direct.is_failed() || recursive.is_failed() {
        return false;
    }
    !root.is_loaded() || !direct.is_loaded() || !recursive.is_loaded()
}

pub(super) fn asset_loading(server: &AssetServer, id: impl Into<UntypedAssetId>) -> bool {
    load_pending(server.get_load_states(id))
}

pub(super) fn update(world: &mut World) {
    if !world.resource::<TerrainPriority>().enabled {
        return;
    }
    let position = world
        .resource::<CesiumViews>()
        .0
        .first()
        .map(|view| view.position);
    world
        .resource_mut::<TerrainPriority>()
        .observe_camera(position);
    if world.resource::<TerrainPriority>().released {
        return;
    }
    let mut roots = world.query::<&WorldAssetRoot>();
    let assets = world.resource::<AssetServer>();
    let pending = roots
        .iter(world)
        .any(|root| asset_loading(assets, root.0.id()))
        || super::thruster_particles::startup_assets_pending(world);
    let now = world.resource::<Time<Real>>().elapsed_secs_f64();
    if !world
        .resource_mut::<TerrainPriority>()
        .allow_terrain(now, pending)
    {
        world.resource_mut::<CesiumViews>().0.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_origin_freezes_before_models_finish_loading() {
        let mut gate = TerrainPriority::nearby();
        gate.observe_camera(Some([0., 0., 0.]));
        assert!(
            gate.origin.is_none(),
            "a temporary camera must not establish the origin"
        );
        gate.camera_ready = true;
        gate.observe_camera(Some([f64::NAN, 0., 0.]));
        assert!(gate.origin.is_none());
        gate.observe_camera(Some([1., 2., 3.]));
        assert!(!gate.allow_terrain(1., true));
        gate.observe_camera(Some([100., 200., 300.]));
        assert!(!gate.allow_terrain(2., true));
        assert_eq!(gate.origin, Some([1., 2., 3.]));
        gate.reset();
        assert!(gate.origin.is_none());
    }

    #[test]
    fn assets_get_a_bounded_head_start_after_a_real_camera_exists() {
        let mut gate = TerrainPriority::nearby();
        assert!(!gate.allow_terrain(100., false));
        assert!(gate.started.is_none());
        gate.camera_ready = true;
        assert!(!gate.allow_terrain(101., true));
        assert!(!gate.allow_terrain(110.9, true));
        assert!(
            gate.allow_terrain(111., true),
            "a hung model must not block terrain forever"
        );
        assert!(gate.allow_terrain(112., true));
        gate.reset();
        assert!(!gate.allow_terrain(113., false));
    }

    #[test]
    fn completed_or_failed_assets_release_after_a_quiet_interval() {
        let mut gate = TerrainPriority::nearby();
        gate.camera_ready = true;
        assert!(!gate.allow_terrain(1., true));
        assert!(!gate.allow_terrain(1.1, false));
        assert!(gate.allow_terrain(1.3, false));
        assert!(TerrainPriority::default().allow_terrain(1., true));
    }

    #[test]
    fn recursive_dependencies_count_but_failures_do_not_hold_the_gate() {
        let error = std::sync::Arc::new(bevy::asset::AssetLoadError::AssetMetaReadError);
        assert!(load_pending(Some((
            LoadState::Loaded,
            DependencyLoadState::Loaded,
            RecursiveDependencyLoadState::Loading
        ))));
        assert!(load_pending(Some((
            LoadState::Loading,
            DependencyLoadState::NotLoaded,
            RecursiveDependencyLoadState::NotLoaded
        ))));
        assert!(!load_pending(Some((
            LoadState::Failed(error.clone()),
            DependencyLoadState::Loading,
            RecursiveDependencyLoadState::Loading
        ))));
        assert!(!load_pending(Some((
            LoadState::Loaded,
            DependencyLoadState::Loaded,
            RecursiveDependencyLoadState::Failed(error)
        ))));
        assert!(!load_pending(Some((
            LoadState::Loaded,
            DependencyLoadState::Loaded,
            RecursiveDependencyLoadState::Loaded
        ))));
        assert!(!load_pending(None));
    }
}
