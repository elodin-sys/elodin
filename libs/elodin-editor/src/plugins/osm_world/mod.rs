use bevy::prelude::Message;

#[derive(Message, Clone, Copy, Debug, Default)]
pub struct OsmWorldRedrawRequest;

#[cfg(not(target_family = "wasm"))]
mod native {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::path::PathBuf;
    use std::str::FromStr;
    use std::thread;
    use std::time::{Duration, Instant};

    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    use bevy::prelude::*;
    use bevy::window::PrimaryWindow;
    use bevy_egui::{EguiContext, egui};
    use bevy_picking::prelude::{Pickable, Pointer};
    use impeller2::types::ComponentId;
    use impeller2_wkt::WorldPos;
    use prost::Message;
    use serde::Deserialize;

    use super::OsmWorldRedrawRequest;
    use crate::WorldPosExt;

    const DEFAULT_ORIGIN_LAT: f64 = 34.054085661510506;
    const DEFAULT_ORIGIN_LON: f64 = -118.42558289792434;
    const DEFAULT_OVERPASS_URLS: [&str; 2] = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ];
    const TILE_QUERY_MARGIN_M: f64 = 32.0;
    const TELEPORT_RESET_TILE_DELTA: i32 = 5;

    pub struct OsmWorldPlugin;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GeoTileBackend {
        Overpass,
        Martin,
    }

    #[derive(Clone, Resource)]
    struct OsmWorldConfig {
        enabled: bool,
        backend: GeoTileBackend,
        origin_lat: f64,
        origin_lon: f64,
        retry_interval: Duration,
        tile_size_m: f64,
        tile_radius: i32,
        prefetch_tile_radius: i32,
        max_inflight_fetches: usize,
        max_buildings_per_tile: usize,
        max_roads_per_tile: usize,
        max_areas_per_tile: usize,
        tracked_world_pos_component: String,
        overpass_urls: Vec<String>,
        martin_url: String,
        martin_source: String,
        martin_zoom: u8,
        use_cache: bool,
        prefer_cache: bool,
        cache_dir: Option<PathBuf>,
        spawn_per_frame: usize,
    }

    impl OsmWorldConfig {
        fn from_env() -> Self {
            let enabled = env_flag(
                "ELODIN_OSM_ENABLED",
                env_flag("ELODIN_OSM_BUILDINGS", false),
            );
            let backend = std::env::var("ELODIN_GEO_BACKEND")
                .ok()
                .or_else(|| std::env::var("ELODIN_OSM_BACKEND").ok())
                .map(|value| value.to_ascii_lowercase())
                .as_deref()
                .map(|value| {
                    if value == "martin" {
                        GeoTileBackend::Martin
                    } else {
                        GeoTileBackend::Overpass
                    }
                })
                .unwrap_or(GeoTileBackend::Overpass);
            let origin_lat = env_parse("ELODIN_OSM_ORIGIN_LAT").unwrap_or(DEFAULT_ORIGIN_LAT);
            let origin_lon = env_parse("ELODIN_OSM_ORIGIN_LON").unwrap_or(DEFAULT_ORIGIN_LON);
            let retry_interval_s: f64 = env_parse("ELODIN_OSM_REFRESH_S").unwrap_or(8.0);
            let tile_size_m: f64 = env_parse("ELODIN_OSM_TILE_SIZE_M").unwrap_or(260.0);
            let tile_radius: i32 = env_parse("ELODIN_OSM_TILE_RADIUS").unwrap_or(1);
            let prefetch_default = env_parse("ELODIN_OSM_PREFETCH_TILE_RADIUS").unwrap_or(3);
            let prefetch_tile_radius: i32 = prefetch_default.max(tile_radius.max(0));
            let max_inflight_fetches =
                env_parse("ELODIN_OSM_MAX_INFLIGHT_FETCHES").unwrap_or(6usize);
            let max_buildings_per_tile = env_parse("ELODIN_OSM_MAX_BUILDINGS_PER_TILE")
                .or_else(|| env_parse("ELODIN_OSM_MAX_BUILDINGS"))
                .unwrap_or(100usize);
            let max_roads_per_tile = env_parse("ELODIN_OSM_MAX_ROADS_PER_TILE").unwrap_or(150usize);
            let max_areas_per_tile = env_parse("ELODIN_OSM_MAX_AREAS_PER_TILE").unwrap_or(80usize);
            let spawn_per_frame = env_parse("ELODIN_OSM_SPAWN_PER_FRAME").unwrap_or(64usize);
            let tracked_world_pos_component = std::env::var("ELODIN_OSM_TRACK_COMPONENT")
                .or_else(|_| std::env::var("ELODIN_OSM_DRONE_COMPONENT"))
                .unwrap_or_default();
            let overpass_urls = std::env::var("ELODIN_OSM_OVERPASS_URLS")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .filter(|urls| !urls.is_empty())
                .unwrap_or_else(|| {
                    DEFAULT_OVERPASS_URLS
                        .iter()
                        .map(ToString::to_string)
                        .collect()
                });
            let martin_url = std::env::var("ELODIN_MARTIN_URL")
                .or_else(|_| std::env::var("ELODIN_OSM_MARTIN_URL"))
                .unwrap_or_else(|_| "http://127.0.0.1:3000".to_string());
            let martin_source = std::env::var("ELODIN_MARTIN_SOURCE")
                .or_else(|_| std::env::var("ELODIN_OSM_MARTIN_SOURCE"))
                .unwrap_or_else(|_| "osm".to_string());
            let martin_zoom = env_parse("ELODIN_MARTIN_ZOOM")
                .or_else(|| env_parse("ELODIN_OSM_MARTIN_ZOOM"))
                .unwrap_or(15u8)
                .clamp(8, 18);
            let use_cache = !env_flag("ELODIN_OSM_DISABLE_CACHE", false);
            let prefer_cache = env_flag("ELODIN_OSM_PREFER_CACHE", true);
            let cache_dir = if use_cache {
                let dir = std::env::var("ELODIN_OSM_CACHE_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| default_cache_dir());
                Some(dir)
            } else {
                None
            };
            Self {
                enabled,
                backend,
                origin_lat,
                origin_lon,
                retry_interval: Duration::from_secs_f64(retry_interval_s.max(0.25)),
                tile_size_m: tile_size_m.max(120.0),
                tile_radius: tile_radius.max(0),
                prefetch_tile_radius: prefetch_tile_radius.max(tile_radius.max(0)),
                max_inflight_fetches: max_inflight_fetches.max(1),
                max_buildings_per_tile: max_buildings_per_tile.max(20),
                max_roads_per_tile: max_roads_per_tile.max(20),
                max_areas_per_tile: max_areas_per_tile.max(20),
                tracked_world_pos_component,
                overpass_urls,
                martin_url,
                martin_source,
                martin_zoom,
                use_cache,
                prefer_cache,
                cache_dir,
                spawn_per_frame: spawn_per_frame.max(1),
            }
        }

        fn origin(&self) -> GeoPoint {
            GeoPoint {
                lat: self.origin_lat,
                lon: self.origin_lon,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct GeoPoint {
        lat: f64,
        lon: f64,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TrackMode {
        RcJet,
        Drone,
        Target,
        Unknown,
    }

    #[derive(Clone, Copy, Debug)]
    struct TrackSample {
        geo: GeoPoint,
        mode: TrackMode,
    }

    #[derive(Component, Clone, Debug)]
    struct OsmBuildingMeta {
        id: i64,
        name: String,
        height_m: f32,
        levels: Option<f32>,
    }

    #[derive(Component, Clone, Debug)]
    struct OsmRoadMeta {
        id: i64,
        name: String,
        kind: String,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct TileKey {
        x: i32,
        y: i32,
    }

    #[derive(Debug)]
    struct PendingRenderable {
        tile: TileKey,
        item: RenderableItem,
    }

    #[derive(Debug, Clone)]
    enum RenderableItem {
        Building(BuildingGeom),
        Road(RoadGeom),
        Area(LandAreaGeom),
    }

    #[derive(Resource, Default, Clone, Debug)]
    struct HoveredBuilding(pub Option<OsmBuildingMeta>);

    #[derive(Resource, Default, Clone, Debug)]
    struct HoveredRoad(pub Option<OsmRoadMeta>);

    #[derive(Resource, Default, Clone)]
    struct OsmWorldStatus {
        message: String,
        is_error: bool,
    }

    #[derive(Resource)]
    struct OsmWorldState {
        fetch_tx: flume::Sender<TileFetchResult>,
        fetch_rx: flume::Receiver<TileFetchResult>,
        inflight_tiles: HashSet<TileKey>,
        pending_tiles: VecDeque<TileKey>,
        loaded_tiles: HashMap<TileKey, Vec<Entity>>,
        active_tiles: HashSet<TileKey>,
        prefetched_tiles: HashMap<TileKey, TileSceneData>,
        pending_to_despawn: VecDeque<Entity>,
        pending_renderables: VecDeque<PendingRenderable>,
        pending_total: usize,
        failed_attempts: HashMap<TileKey, u32>,
        next_retry_at: HashMap<TileKey, Instant>,
        initial_burst_done: bool,
        last_prefetch_center_local: Option<Vec2>,
        last_prefetch_tile: Option<TileKey>,
        last_reported_center_tile: Option<TileKey>,
    }

    impl Default for OsmWorldState {
        fn default() -> Self {
            let (fetch_tx, fetch_rx) = flume::unbounded();
            Self {
                fetch_tx,
                fetch_rx,
                inflight_tiles: HashSet::new(),
                pending_tiles: VecDeque::new(),
                loaded_tiles: HashMap::new(),
                active_tiles: HashSet::new(),
                prefetched_tiles: HashMap::new(),
                pending_to_despawn: VecDeque::new(),
                pending_renderables: VecDeque::new(),
                pending_total: 0,
                failed_attempts: HashMap::new(),
                next_retry_at: HashMap::new(),
                initial_burst_done: false,
                last_prefetch_center_local: None,
                last_prefetch_tile: None,
                last_reported_center_tile: None,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct BuildingGeom {
        id: i64,
        name: String,
        tint: BuildingTint,
        height_m: f32,
        levels: Option<f32>,
        points: Vec<Vec2>,
        area_m2: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum BuildingTint {
        Default,
        School,
        Pier,
        Hospital,
        FireStation,
    }

    #[derive(Debug, Clone)]
    struct RoadGeom {
        id: i64,
        kind: String,
        name: String,
        fast: bool,
        points: Vec<Vec2>,
        width_m: f32,
        length_m: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum LandTint {
        Forest,
        Park,
        Water,
        Beach,
        School,
        Pier,
    }

    #[derive(Debug, Clone)]
    struct LandAreaGeom {
        id: i64,
        kind: String,
        tint: LandTint,
        points: Vec<Vec2>,
        area_m2: f32,
    }

    #[derive(Debug, Clone, Default)]
    struct TileSceneData {
        buildings: Vec<BuildingGeom>,
        roads: Vec<RoadGeom>,
        areas: Vec<LandAreaGeom>,
    }

    #[derive(Debug)]
    struct TileFetchResult {
        tile: TileKey,
        scene: TileSceneData,
        error: Option<String>,
        from_cache: bool,
    }

    #[derive(Deserialize)]
    struct OverpassResponse {
        elements: Vec<OverpassElement>,
    }

    #[derive(Deserialize)]
    struct OverpassElement {
        #[serde(rename = "type")]
        element_type: String,
        id: i64,
        lat: Option<f64>,
        lon: Option<f64>,
        nodes: Option<Vec<i64>>,
        tags: Option<HashMap<String, String>>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct VectorTile {
        #[prost(message, repeated, tag = "3")]
        layers: Vec<VectorTileLayer>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct VectorTileLayer {
        #[prost(string, required, tag = "1")]
        name: String,
        #[prost(message, repeated, tag = "2")]
        features: Vec<VectorTileFeature>,
        #[prost(string, repeated, tag = "3")]
        keys: Vec<String>,
        #[prost(message, repeated, tag = "4")]
        values: Vec<VectorTileValue>,
        #[prost(uint32, optional, tag = "5")]
        extent: Option<u32>,
        #[prost(uint32, required, tag = "15")]
        version: u32,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct VectorTileFeature {
        #[prost(uint64, optional, tag = "1")]
        id: Option<u64>,
        #[prost(uint32, repeated, tag = "2")]
        tags: Vec<u32>,
        #[prost(enumeration = "VectorTileGeomType", optional, tag = "3")]
        geometry_type: Option<i32>,
        #[prost(uint32, repeated, tag = "4")]
        geometry: Vec<u32>,
    }

    #[derive(Clone, PartialEq, prost::Message)]
    struct VectorTileValue {
        #[prost(string, optional, tag = "1")]
        string_value: Option<String>,
        #[prost(float, optional, tag = "2")]
        float_value: Option<f32>,
        #[prost(double, optional, tag = "3")]
        double_value: Option<f64>,
        #[prost(int64, optional, tag = "4")]
        int_value: Option<i64>,
        #[prost(uint64, optional, tag = "5")]
        uint_value: Option<u64>,
        #[prost(sint64, optional, tag = "6")]
        sint_value: Option<i64>,
        #[prost(bool, optional, tag = "7")]
        bool_value: Option<bool>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
    #[repr(i32)]
    enum VectorTileGeomType {
        Unknown = 0,
        Point = 1,
        LineString = 2,
        Polygon = 3,
    }

    impl Plugin for OsmWorldPlugin {
        fn build(&self, app: &mut App) {
            let config = OsmWorldConfig::from_env();
            app.add_message::<OsmWorldRedrawRequest>();
            app.insert_resource(config.clone());
            if !config.enabled {
                info!("OSM world disabled. Set ELODIN_OSM_ENABLED=1 to enable it.");
                return;
            }
            let track = if config.tracked_world_pos_component.is_empty() {
                "auto".to_string()
            } else {
                config.tracked_world_pos_component.clone()
            };
            info!(
                "OSM world enabled (backend: {:?}, origin: {}, {}, track: {}, tile_size: {}m, radius: {}, prefetch: {}, inflight: {}, cache_first: {}, max: b{} r{} a{}, martin: {}/{}/z{})",
                config.backend,
                config.origin_lat,
                config.origin_lon,
                track,
                config.tile_size_m,
                config.tile_radius,
                config.prefetch_tile_radius,
                config.max_inflight_fetches,
                config.prefer_cache,
                config.max_buildings_per_tile,
                config.max_roads_per_tile,
                config.max_areas_per_tile,
                config.martin_url,
                config.martin_source,
                config.martin_zoom
            );
            app.init_resource::<HoveredBuilding>()
                .init_resource::<HoveredRoad>()
                .init_resource::<OsmWorldStatus>()
                .init_resource::<OsmWorldState>()
                .add_systems(Update, update_osm_world)
                .add_systems(Update, render_hover_overlay)
                .add_systems(Update, render_road_hover_overlay)
                .add_systems(Update, render_status_overlay);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_osm_world(
        config: Res<OsmWorldConfig>,
        mut state: ResMut<OsmWorldState>,
        mut hovered: ResMut<HoveredBuilding>,
        mut hovered_road: ResMut<HoveredRoad>,
        mut status: ResMut<OsmWorldStatus>,
        mut redraw_requests: MessageReader<OsmWorldRedrawRequest>,
        world_pos_query: Query<(&ComponentId, &WorldPos)>,
        existing_entities: Query<Entity>,
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        // On the very first frame, seed all tiles around the origin so fetching
        // starts immediately - before the drone world_pos is even available.
        if !state.initial_burst_done {
            state.initial_burst_done = true;
            let origin_tile = geo_to_tile(config.origin(), &config);
            let burst_radius = config.prefetch_tile_radius;
            let burst_tiles = desired_tiles_around(origin_tile, burst_radius);
            info!(
                "OSM initial burst: seeding {} tiles around origin ({}, {})",
                burst_tiles.len(),
                origin_tile.x,
                origin_tile.y
            );
            for tile in burst_tiles {
                if !state.prefetched_tiles.contains_key(&tile)
                    && !state.inflight_tiles.contains(&tile)
                    && !state.pending_tiles.contains(&tile)
                {
                    state.pending_tiles.push_back(tile);
                }
            }
        }

        let track = current_center(&config, &world_pos_query);
        let center = track.geo;
        let center_tile = geo_to_tile(center, &config);
        let previous_center_tile = state.last_reported_center_tile;
        if state.last_reported_center_tile != Some(center_tile) {
            info!(
                "OSM center tile: ({}, {}) [{:?}]",
                center_tile.x, center_tile.y, track.mode
            );
            state.last_reported_center_tile = Some(center_tile);
        }
        let (active_radius, prefetch_radius, request_distance_m) = match track.mode {
            TrackMode::RcJet => {
                let active = config.tile_radius.max(2);
                let prefetch = config.prefetch_tile_radius.max(active + 3).min(active + 6);
                (active, prefetch, (config.tile_size_m * 0.22) as f32)
            }
            TrackMode::Drone => {
                let active = config.tile_radius.max(1);
                let prefetch = config.prefetch_tile_radius.max(active + 1).min(active + 3);
                (active, prefetch, (config.tile_size_m * 0.90) as f32)
            }
            TrackMode::Target | TrackMode::Unknown => {
                let active = config.tile_radius.max(1);
                let prefetch = config.prefetch_tile_radius.max(active + 2).min(active + 4);
                (active, prefetch, (config.tile_size_m * 0.60) as f32)
            }
        };
        let active_tiles_vec = desired_tiles_around(center_tile, active_radius);
        let prefetch_tiles_vec = desired_tiles_around(center_tile, prefetch_radius);
        let active_tiles_set: HashSet<TileKey> = active_tiles_vec.iter().copied().collect();
        let prefetch_tiles_set: HashSet<TileKey> = prefetch_tiles_vec.iter().copied().collect();
        state.active_tiles = active_tiles_set.clone();

        // If another system reset/reloaded scene entities, stale handles can stay in
        // `loaded_tiles` and block respawn forever. Prune dead entity IDs eagerly.
        let removed_stale_handles = prune_missing_loaded_entities(&mut state, &existing_entities);
        if removed_stale_handles > 0 {
            hovered.0 = None;
            hovered_road.0 = None;
            info!(
                "OSM recovered from scene reset: pruned {} stale entity handles",
                removed_stale_handles
            );
        }

        if redraw_requests.read().next().is_some() {
            let refreshed_tiles = force_redraw_active_tiles(
                &mut state,
                &mut hovered,
                &mut hovered_road,
                &active_tiles_vec,
            );
            status.message = format!(
                "OSM redraw requested: {} active tiles around ({}, {})",
                refreshed_tiles, center_tile.x, center_tile.y
            );
            status.is_error = false;
            info!(
                "OSM redraw requested around center tile ({}, {}), refreshed {} tiles",
                center_tile.x, center_tile.y, refreshed_tiles
            );
        }

        if let Some(previous_tile) = previous_center_tile {
            let jump_tiles = tile_jump_chebyshev(previous_tile, center_tile);
            if jump_tiles >= TELEPORT_RESET_TILE_DELTA {
                force_recenter_streaming_state(
                    &mut state,
                    &mut hovered,
                    &mut hovered_road,
                    center_tile,
                    &prefetch_tiles_set,
                );
                status.message = format!(
                    "OSM recenter: jump {} tiles -> refreshing around ({}, {})",
                    jump_tiles, center_tile.x, center_tile.y
                );
                status.is_error = false;
                info!(
                    "OSM recenter triggered: jump {} tiles ({}, {}) -> ({}, {})",
                    jump_tiles, previous_tile.x, previous_tile.y, center_tile.x, center_tile.y
                );
            }
        }

        loop {
            let result = match state.fetch_rx.try_recv() {
                Ok(result) => result,
                Err(flume::TryRecvError::Empty) => break,
                Err(flume::TryRecvError::Disconnected) => {
                    status.message = "OSM tiles: fetch channel disconnected".to_string();
                    status.is_error = true;
                    break;
                }
            };

            let tile = result.tile;
            state.inflight_tiles.remove(&tile);
            if let Some(err) = result.error {
                let failures = state.failed_attempts.entry(tile).or_insert(0);
                *failures = failures.saturating_add(1);
                let retry_after = retry_backoff(config.retry_interval, *failures);
                state
                    .next_retry_at
                    .insert(tile, Instant::now() + retry_after);
                status.message = format!(
                    "OSM tiles: error tile ({}, {}) - retry in {:.1}s",
                    tile.x,
                    tile.y,
                    retry_after.as_secs_f32()
                );
                status.is_error = true;
                warn!(
                    "OSM tile fetch failed for ({}, {}): {}",
                    tile.x, tile.y, err
                );
                continue;
            }

            state.failed_attempts.remove(&tile);
            state.next_retry_at.remove(&tile);
            let counts = (
                result.scene.buildings.len(),
                result.scene.roads.len(),
                result.scene.areas.len(),
            );
            state.prefetched_tiles.insert(tile, result.scene.clone());

            if !active_tiles_set.contains(&tile) {
                continue;
            }

            if let Some(old_entities) = state.loaded_tiles.remove(&tile) {
                state.pending_to_despawn.extend(old_entities);
                hovered.0 = None;
                hovered_road.0 = None;
            }
            state
                .pending_renderables
                .retain(|pending| pending.tile != tile);
            enqueue_tile_scene(tile, &result.scene, &mut state.pending_renderables);
            state.pending_total = state.pending_renderables.len();
            status.message = format!(
                "OSM tile ({}, {}): b{} r{} a{}{}",
                tile.x,
                tile.y,
                counts.0,
                counts.1,
                counts.2,
                if result.from_cache { " (cache)" } else { "" }
            );
            status.is_error = false;
            info!(
                "OSM tile updated: ({}, {}) -> b{} r{} a{}{}",
                tile.x,
                tile.y,
                counts.0,
                counts.1,
                counts.2,
                if result.from_cache { " (cache)" } else { "" }
            );
        }

        let stale_tiles: Vec<TileKey> = state
            .loaded_tiles
            .keys()
            .copied()
            .filter(|tile| !active_tiles_set.contains(tile))
            .collect();
        if !stale_tiles.is_empty() {
            hovered.0 = None;
            hovered_road.0 = None;
        }
        for tile in stale_tiles {
            if let Some(entities) = state.loaded_tiles.remove(&tile) {
                state.pending_to_despawn.extend(entities);
            }
        }

        let active_tiles_snapshot = active_tiles_set.clone();
        let prefetch_tiles_snapshot = prefetch_tiles_set.clone();
        state
            .pending_tiles
            .retain(|tile| prefetch_tiles_snapshot.contains(tile));
        state
            .pending_renderables
            .retain(|pending| active_tiles_snapshot.contains(&pending.tile));
        state
            .prefetched_tiles
            .retain(|tile, _| prefetch_tiles_snapshot.contains(tile));
        state.pending_total = state.pending_renderables.len();

        for tile in &active_tiles_vec {
            if state.loaded_tiles.contains_key(tile) {
                continue;
            }
            if state
                .pending_renderables
                .iter()
                .any(|pending| pending.tile == *tile)
            {
                continue;
            }
            let Some(cached_scene) = state.prefetched_tiles.get(tile).cloned() else {
                continue;
            };
            enqueue_tile_scene(*tile, &cached_scene, &mut state.pending_renderables);
        }
        state.pending_total = state.pending_renderables.len();

        let center_local = geo_to_local(center, config.origin());
        let moved_since_last_prefetch = state
            .last_prefetch_center_local
            .map(|last| last.distance(center_local))
            .unwrap_or(f32::MAX);
        let tile_changed_since_last_prefetch = state.last_prefetch_tile != Some(center_tile);
        let should_enqueue_prefetch =
            tile_changed_since_last_prefetch || moved_since_last_prefetch >= request_distance_m;
        if should_enqueue_prefetch {
            let now = Instant::now();
            for tile in &prefetch_tiles_vec {
                if state.prefetched_tiles.contains_key(tile)
                    || state.inflight_tiles.contains(tile)
                    || state.pending_tiles.contains(tile)
                {
                    continue;
                }
                if let Some(retry_at) = state.next_retry_at.get(tile)
                    && now < *retry_at
                {
                    continue;
                }
                state.pending_tiles.push_back(*tile);
            }
            state.last_prefetch_center_local = Some(center_local);
            state.last_prefetch_tile = Some(center_tile);
        }
        reprioritize_pending_tiles(&mut state.pending_tiles, center_tile, &active_tiles_set);

        apply_pending_renderables(
            &config,
            &mut state,
            &mut status,
            &mut commands,
            &mut meshes,
            &mut materials,
        );

        while state.inflight_tiles.len() < config.max_inflight_fetches {
            let Some(tile) = state.pending_tiles.pop_front() else {
                break;
            };
            if state.prefetched_tiles.contains_key(&tile) {
                continue;
            }
            let tx = state.fetch_tx.clone();
            let cfg = config.clone();
            state.inflight_tiles.insert(tile);
            thread::spawn(move || {
                let result = fetch_tile_buildings(tile, &cfg);
                let _ = tx.send(result);
            });
        }

        if state.pending_tiles.is_empty()
            && state.inflight_tiles.is_empty()
            && state.pending_renderables.is_empty()
            && state.pending_to_despawn.is_empty()
            && !status.is_error
        {
            let entity_count: usize = state.loaded_tiles.values().map(Vec::len).sum();
            status.message = format!(
                "OSM tiles: center ({}, {}), {} loaded, {} cached, {} entities",
                center_tile.x,
                center_tile.y,
                state.loaded_tiles.len(),
                state.prefetched_tiles.len(),
                entity_count
            );
        } else if !status.is_error && state.pending_renderables.is_empty() {
            status.message = format!(
                "OSM tiles: center ({}, {}), prefetch {} pending, {} inflight, {} cached",
                center_tile.x,
                center_tile.y,
                state.pending_tiles.len(),
                state.inflight_tiles.len(),
                state.prefetched_tiles.len()
            );
        }
    }

    fn apply_pending_renderables(
        config: &OsmWorldConfig,
        state: &mut OsmWorldState,
        status: &mut OsmWorldStatus,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
    ) {
        let mut despawn_budget = config.spawn_per_frame.saturating_mul(2).max(8);
        while despawn_budget > 0 {
            let Some(entity) = state.pending_to_despawn.pop_front() else {
                break;
            };
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
            despawn_budget -= 1;
        }

        let mut process_budget = config.spawn_per_frame.max(1);
        while process_budget > 0 {
            let Some(pending) = state.pending_renderables.pop_front() else {
                break;
            };
            process_budget -= 1;

            if !state.active_tiles.contains(&pending.tile) {
                continue;
            }

            let entity = match pending.item {
                RenderableItem::Building(building) => {
                    let wall_color =
                        wall_color_for_building(building.height_m, building.id, building.tint);
                    let roof_color = roof_color_for_wall(wall_color);
                    let Some(mesh) = extrude_building_mesh(
                        &building.points,
                        building.height_m,
                        wall_color,
                        roof_color,
                    ) else {
                        continue;
                    };
                    let material = materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        unlit: false,
                        double_sided: true,
                        cull_mode: None,
                        perceptual_roughness: 0.82,
                        metallic: 0.03,
                        reflectance: 0.14,
                        ..Default::default()
                    });
                    commands
                        .spawn((
                            Name::new(format!("osm_building_{}", building.id)),
                            Mesh3d(meshes.add(mesh)),
                            MeshMaterial3d(material),
                            Pickable::default(),
                            OsmBuildingMeta {
                                id: building.id,
                                name: building.name,
                                height_m: building.height_m,
                                levels: building.levels,
                            },
                        ))
                        .observe(on_building_hover_over)
                        .observe(on_building_hover_out)
                        .id()
                }
                RenderableItem::Road(road) => {
                    let color = road_color_for_kind(&road.kind);
                    let Some(mesh) = road_strip_mesh(&road.points, road.width_m, 0.05, color)
                    else {
                        continue;
                    };
                    let material = materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        unlit: false,
                        cull_mode: None,
                        perceptual_roughness: 0.95,
                        metallic: 0.01,
                        reflectance: 0.06,
                        emissive: LinearRgba::BLACK,
                        ..Default::default()
                    });
                    let mut entity = commands.spawn((
                        Name::new(format!("osm_road_{}", road.id)),
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material),
                    ));
                    if road.fast {
                        entity
                            .insert((
                                Pickable::default(),
                                OsmRoadMeta {
                                    id: road.id,
                                    name: road.name.clone(),
                                    kind: road.kind.clone(),
                                },
                            ))
                            .observe(on_road_hover_over)
                            .observe(on_road_hover_out);
                    }
                    entity.id()
                }
                RenderableItem::Area(area) => {
                    let color = area_color(area.tint);
                    let Some(mesh) = ground_polygon_mesh(&area.points, 0.03, color) else {
                        continue;
                    };
                    let material = materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        unlit: false,
                        cull_mode: None,
                        perceptual_roughness: 0.96,
                        metallic: 0.0,
                        reflectance: 0.04,
                        ..Default::default()
                    });
                    commands
                        .spawn((
                            Name::new(format!("osm_area_{}_{}", area.kind, area.id)),
                            Mesh3d(meshes.add(mesh)),
                            MeshMaterial3d(material),
                        ))
                        .id()
                }
            };
            state
                .loaded_tiles
                .entry(pending.tile)
                .or_default()
                .push(entity);
        }

        state.pending_total = state.pending_renderables.len();
        if state.pending_total > 0 || !state.pending_to_despawn.is_empty() {
            status.message = format!("OSM tiles: applying {} remaining", state.pending_total);
            status.is_error = false;
        }
    }

    fn on_building_hover_over(
        event: On<Pointer<bevy_picking::prelude::Over>>,
        meta_query: Query<&OsmBuildingMeta>,
        mut hovered: ResMut<HoveredBuilding>,
    ) {
        let event_target = event.event().event_target();
        if let Ok(meta) = meta_query.get(event_target) {
            hovered.0 = Some(meta.clone());
        }
    }

    fn on_building_hover_out(
        event: On<Pointer<bevy_picking::prelude::Out>>,
        meta_query: Query<&OsmBuildingMeta>,
        mut hovered: ResMut<HoveredBuilding>,
    ) {
        let event_target = event.event().event_target();
        if let Ok(meta) = meta_query.get(event_target) {
            let hovered_id = hovered.0.as_ref().map(|m| m.id);
            if hovered_id == Some(meta.id) {
                hovered.0 = None;
            }
        }
    }

    fn on_road_hover_over(
        event: On<Pointer<bevy_picking::prelude::Over>>,
        road_query: Query<(&OsmRoadMeta, &MeshMaterial3d<StandardMaterial>)>,
        mut hovered: ResMut<HoveredRoad>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        let event_target = event.event().event_target();
        let Ok((meta, material_handle)) = road_query.get(event_target) else {
            return;
        };

        hovered.0 = Some(meta.clone());
        if let Some(material) = materials.get_mut(&material_handle.0) {
            let [r, g, b, _] = road_highlight_color_for_kind(&meta.kind)
                .to_linear()
                .to_f32_array();
            material.emissive = LinearRgba::new(r * 0.45, g * 0.45, b * 0.45, 1.0);
        }
    }

    fn on_road_hover_out(
        event: On<Pointer<bevy_picking::prelude::Out>>,
        road_query: Query<(&OsmRoadMeta, &MeshMaterial3d<StandardMaterial>)>,
        mut hovered: ResMut<HoveredRoad>,
        mut materials: ResMut<Assets<StandardMaterial>>,
    ) {
        let event_target = event.event().event_target();
        let Ok((meta, material_handle)) = road_query.get(event_target) else {
            return;
        };

        if let Some(material) = materials.get_mut(&material_handle.0) {
            material.emissive = LinearRgba::BLACK;
        }
        let hovered_id = hovered.0.as_ref().map(|m| m.id);
        if hovered_id == Some(meta.id) {
            hovered.0 = None;
        }
    }

    fn render_hover_overlay(
        hovered: Res<HoveredBuilding>,
        windows: Query<&Window, With<PrimaryWindow>>,
        mut contexts: Query<&mut EguiContext, With<PrimaryWindow>>,
    ) {
        let Some(meta) = hovered.0.as_ref() else {
            return;
        };
        let Ok(window) = windows.single() else {
            return;
        };
        let Some(cursor_pos) = window.cursor_position() else {
            return;
        };
        let Ok(mut context) = contexts.single_mut() else {
            return;
        };
        let ctx = context.get_mut();
        ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        let pos = egui::pos2(cursor_pos.x + 14.0, cursor_pos.y + 14.0);
        egui::Area::new(egui::Id::new("osm_building_hover_overlay"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(&meta.name).strong());
                    ui.label(format!("id: {}", meta.id));
                    ui.label(format!("height: {:.1} m", meta.height_m));
                    if let Some(levels) = meta.levels {
                        ui.label(format!("levels: {:.1}", levels));
                    }
                    ui.label("source: OpenStreetMap / Overpass");
                });
            });
    }

    fn render_road_hover_overlay(
        hovered_road: Res<HoveredRoad>,
        hovered_building: Res<HoveredBuilding>,
        windows: Query<&Window, With<PrimaryWindow>>,
        mut contexts: Query<&mut EguiContext, With<PrimaryWindow>>,
    ) {
        if hovered_building.0.is_some() {
            return;
        }
        let Some(meta) = hovered_road.0.as_ref() else {
            return;
        };
        let Ok(window) = windows.single() else {
            return;
        };
        let Some(cursor_pos) = window.cursor_position() else {
            return;
        };
        let Ok(mut context) = contexts.single_mut() else {
            return;
        };
        let ctx = context.get_mut();
        ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
        let pos = egui::pos2(cursor_pos.x + 14.0, cursor_pos.y + 14.0);
        egui::Area::new(egui::Id::new("osm_road_hover_overlay"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(egui::RichText::new(&meta.name).strong());
                    ui.label(format!("type: {}", meta.kind));
                    ui.label("highlight: fast road");
                });
            });
    }

    fn render_status_overlay(
        status: Res<OsmWorldStatus>,
        mut contexts: Query<&mut EguiContext, With<PrimaryWindow>>,
    ) {
        // Keep the editor tabs/viewport layout clean: only show this overlay for real errors.
        if status.message.is_empty() || !status.is_error {
            return;
        }
        let Ok(mut context) = contexts.single_mut() else {
            return;
        };
        let ctx = context.get_mut();
        let pos = egui::pos2(12.0, 52.0);
        egui::Area::new(egui::Id::new("osm_building_status_overlay"))
            .order(egui::Order::Foreground)
            .fixed_pos(pos)
            .show(ctx, |ui| {
                let text = egui::RichText::new(&status.message).color(egui::Color32::LIGHT_RED);
                egui::Frame::NONE.show(ui, |ui| {
                    ui.label(text);
                });
            });
    }

    fn fetch_tile_buildings(tile: TileKey, cfg: &OsmWorldConfig) -> TileFetchResult {
        match fetch_tile_buildings_inner(tile, cfg) {
            Ok((mut scene, from_cache)) => {
                let tile_center = tile_center_local(tile, cfg);
                scene.buildings.sort_by(|a, b| {
                    let da = centroid(&a.points).distance_squared(tile_center);
                    let db = centroid(&b.points).distance_squared(tile_center);
                    da.total_cmp(&db)
                        .then_with(|| b.area_m2.total_cmp(&a.area_m2))
                });
                scene.buildings.truncate(cfg.max_buildings_per_tile);

                scene.roads.sort_by(|a, b| {
                    let da = centroid(&a.points).distance_squared(tile_center);
                    let db = centroid(&b.points).distance_squared(tile_center);
                    da.total_cmp(&db)
                        .then_with(|| b.length_m.total_cmp(&a.length_m))
                });
                scene.roads.truncate(cfg.max_roads_per_tile);

                scene.areas.sort_by(|a, b| {
                    let da = centroid(&a.points).distance_squared(tile_center);
                    let db = centroid(&b.points).distance_squared(tile_center);
                    area_tint_priority(a.tint)
                        .cmp(&area_tint_priority(b.tint))
                        .then_with(|| da.total_cmp(&db))
                        .then_with(|| b.area_m2.total_cmp(&a.area_m2))
                });
                scene.areas.truncate(cfg.max_areas_per_tile);
                TileFetchResult {
                    tile,
                    scene,
                    error: None,
                    from_cache,
                }
            }
            Err(err) => TileFetchResult {
                tile,
                scene: TileSceneData::default(),
                error: Some(err),
                from_cache: false,
            },
        }
    }

    fn fetch_tile_buildings_inner(
        tile: TileKey,
        cfg: &OsmWorldConfig,
    ) -> Result<(TileSceneData, bool), String> {
        match cfg.backend {
            GeoTileBackend::Overpass => fetch_tile_overpass_inner(tile, cfg),
            GeoTileBackend::Martin => fetch_tile_martin_inner(tile, cfg),
        }
    }

    fn fetch_tile_overpass_inner(
        tile: TileKey,
        cfg: &OsmWorldConfig,
    ) -> Result<(TileSceneData, bool), String> {
        let cache_key = cache_key_for_tile(tile, cfg);
        if cfg.use_cache
            && cfg.prefer_cache
            && let Some(text) = load_cache(cfg, &cache_key)
            && let Ok(parsed) = parse_overpass(&text, cfg)
        {
            return Ok((parsed, true));
        }

        let (south, west, north, east) = tile_bounds_geo(tile, cfg);
        let query = format!(
            "[out:json][timeout:25];\
            (\
                way[\"building\"]({south},{west},{north},{east});\
                way[\"highway\"][\"highway\"!~\"footway|path|cycleway|steps|pedestrian|track|service|corridor|bridleway\"]({south},{west},{north},{east});\
                way[\"man_made\"=\"pier\"]({south},{west},{north},{east});\
                way[\"tourism\"=\"pier\"]({south},{west},{north},{east});\
                way[\"amenity\"~\"school|college|university|kindergarten\"]({south},{west},{north},{east});\
                way[\"landuse\"~\"forest|grass|meadow|recreation_ground|village_green|reservoir|basin\"]({south},{west},{north},{east});\
                way[\"natural\"~\"wood|scrub|grassland|water|wetland|beach|sand|coastline\"]({south},{west},{north},{east});\
                way[\"leisure\"~\"park|golf_course|nature_reserve\"]({south},{west},{north},{east});\
                way[\"waterway\"~\"riverbank|dock|canal\"]({south},{west},{north},{east});\
            );\
            out body;\
            >;\
            out skel qt;"
        );

        let response_text = match request_overpass(query, cfg) {
            Ok(text) => {
                if cfg.use_cache {
                    save_cache(cfg, &cache_key, &text);
                }
                text
            }
            Err(err) => {
                if cfg.use_cache
                    && let Some(text) = load_cache(cfg, &cache_key)
                {
                    return parse_overpass(&text, cfg).map(|b| (b, true));
                }
                return Err(err);
            }
        };

        parse_overpass(&response_text, cfg).map(|b| (b, false))
    }

    fn fetch_tile_martin_inner(
        tile: TileKey,
        cfg: &OsmWorldConfig,
    ) -> Result<(TileSceneData, bool), String> {
        let geo_center = tile_center_geo_from_local_tile(tile, cfg);
        let martin_tile = geo_to_slippy_tile(geo_center, cfg.martin_zoom);
        let cache_key = format!(
            "martin_{}_z{}_{}_{}",
            cfg.martin_source, cfg.martin_zoom, martin_tile.x, martin_tile.y
        );

        if cfg.use_cache
            && cfg.prefer_cache
            && let Some(bytes) = load_cache_bytes(cfg, &cache_key)
            && let Ok(parsed) = parse_martin_tile(&bytes, martin_tile, cfg, tile)
        {
            return Ok((parsed, true));
        }

        let tile_bytes = match request_martin_tile(martin_tile, cfg) {
            Ok(bytes) => {
                if cfg.use_cache {
                    save_cache_bytes(cfg, &cache_key, &bytes);
                }
                bytes
            }
            Err(err) => {
                if cfg.use_cache
                    && let Some(bytes) = load_cache_bytes(cfg, &cache_key)
                {
                    return parse_martin_tile(&bytes, martin_tile, cfg, tile)
                        .map(|scene| (scene, true));
                }
                return Err(err);
            }
        };

        parse_martin_tile(&tile_bytes, martin_tile, cfg, tile).map(|scene| (scene, false))
    }

    fn parse_overpass(response_text: &str, cfg: &OsmWorldConfig) -> Result<TileSceneData, String> {
        let parsed: OverpassResponse = serde_json::from_str(response_text)
            .map_err(|e| format!("invalid overpass JSON: {e}"))?;

        let mut node_map: HashMap<i64, (f64, f64)> = HashMap::new();
        let mut ways: Vec<(i64, Vec<i64>, HashMap<String, String>)> = Vec::new();
        for element in parsed.elements {
            match element.element_type.as_str() {
                "node" => {
                    if let (Some(lat), Some(lon)) = (element.lat, element.lon) {
                        node_map.insert(element.id, (lat, lon));
                    }
                }
                "way" => {
                    let Some(nodes) = element.nodes else {
                        continue;
                    };
                    let tags = element.tags.unwrap_or_default();
                    ways.push((element.id, nodes, tags));
                }
                _ => {}
            }
        }

        let origin = cfg.origin();
        let lat_scale = meters_per_degree_lat(origin.lat);
        let lon_scale = meters_per_degree_lon(origin.lat);

        let mut buildings = Vec::new();
        let mut roads = Vec::new();
        let mut areas = Vec::new();
        for (way_id, node_ids, tags) in ways {
            let mut points = Vec::new();
            for node_id in node_ids {
                let Some((lat, lon)) = node_map.get(&node_id).copied() else {
                    continue;
                };
                let east = (lon - origin.lon) * lon_scale;
                let north = (lat - origin.lat) * lat_scale;
                points.push(Vec2::new(east as f32, -north as f32));
            }

            if tags.contains_key("building") {
                dedupe_ring_points(&mut points);
                if points.len() < 3 {
                    continue;
                }

                let area = polygon_area_abs(&points);
                if area < 8.0 {
                    continue;
                }

                let levels = tags
                    .get("building:levels")
                    .and_then(|s| parse_prefixed_f32(s));
                let mut height_m = tags
                    .get("height")
                    .and_then(|s| parse_prefixed_f32(s))
                    .or_else(|| levels.map(|x| x * 3.2))
                    .unwrap_or(12.0);
                height_m = height_m.clamp(3.0, 450.0);

                let name = tags
                    .get("name")
                    .cloned()
                    .or_else(|| {
                        tags.get("addr:housenumber")
                            .zip(tags.get("addr:street"))
                            .map(|(n, s)| format!("{n} {s}"))
                    })
                    .unwrap_or_else(|| format!("building {way_id}"));
                let tint = classify_building_tint(&tags, &name);

                buildings.push(BuildingGeom {
                    id: way_id,
                    name,
                    tint,
                    height_m,
                    levels,
                    points,
                    area_m2: area,
                });
                continue;
            }

            if is_pier_way(&tags) {
                let mut ring_points = points.clone();
                dedupe_ring_points(&mut ring_points);
                if ring_points.len() >= 3 {
                    let area = polygon_area_abs(&ring_points);
                    if area >= 80.0 {
                        areas.push(LandAreaGeom {
                            id: way_id,
                            kind: "man_made:pier".to_string(),
                            tint: LandTint::Pier,
                            points: ring_points,
                            area_m2: area,
                        });
                        continue;
                    }
                }

                dedupe_polyline_points(&mut points);
                if points.len() >= 2 {
                    let length_m = polyline_length(&points);
                    if length_m >= 12.0 {
                        roads.push(RoadGeom {
                            id: way_id,
                            kind: "pier".to_string(),
                            name: road_display_name(&tags, "pier", way_id),
                            fast: false,
                            width_m: road_width_for_kind("pier"),
                            points,
                            length_m,
                        });
                    }
                }
                continue;
            }

            if is_coastline_way(&tags) {
                dedupe_polyline_points(&mut points);
                if points.len() < 2 {
                    continue;
                }
                let length_m = polyline_length(&points);
                if length_m < 30.0 {
                    continue;
                }
                roads.push(RoadGeom {
                    id: way_id,
                    kind: "coastline".to_string(),
                    name: road_display_name(&tags, "coastline", way_id),
                    fast: false,
                    width_m: road_width_for_kind("coastline"),
                    points,
                    length_m,
                });
                continue;
            }

            if let Some(kind) = tags.get("highway") {
                if !include_highway(kind) {
                    continue;
                }
                dedupe_polyline_points(&mut points);
                if points.len() < 2 {
                    continue;
                }
                let length_m = polyline_length(&points);
                if length_m < 20.0 {
                    continue;
                }
                roads.push(RoadGeom {
                    id: way_id,
                    kind: kind.clone(),
                    name: road_display_name(&tags, kind, way_id),
                    fast: is_fast_road_kind(kind),
                    width_m: road_width_for_kind(kind),
                    points,
                    length_m,
                });
                continue;
            }

            if let Some((tint, kind)) = classify_area_tint(&tags) {
                dedupe_ring_points(&mut points);
                if points.len() < 3 {
                    continue;
                }
                let area = polygon_area_abs(&points);
                if area < 120.0 {
                    continue;
                }
                areas.push(LandAreaGeom {
                    id: way_id,
                    kind,
                    tint,
                    points,
                    area_m2: area,
                });
            }
        }

        Ok(TileSceneData {
            buildings,
            roads,
            areas,
        })
    }

    fn request_martin_tile(tile: TileKey, cfg: &OsmWorldConfig) -> Result<Vec<u8>, String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("tokio runtime init failed: {e}"))?;

        let base = cfg.martin_url.trim_end_matches('/');
        let source = cfg.martin_source.trim_matches('/');
        let candidates = [
            format!("{base}/{source}/{}/{}/{}", cfg.martin_zoom, tile.x, tile.y),
            format!(
                "{base}/{source}/{}/{}/{}.mvt",
                cfg.martin_zoom, tile.x, tile.y
            ),
        ];

        rt.block_on(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(20))
                .build()
                .map_err(|e| format!("http client init failed: {e}"))?;

            let mut errors = Vec::new();
            for url in candidates {
                let response = match client
                    .get(&url)
                    .header("Accept", "application/x-protobuf,application/octet-stream")
                    .send()
                    .await
                {
                    Ok(response) => response,
                    Err(e) => {
                        errors.push(format!("{url}: request error: {e}"));
                        continue;
                    }
                };

                if !response.status().is_success() {
                    errors.push(format!("{url}: HTTP {}", response.status()));
                    continue;
                }

                match response.bytes().await {
                    Ok(bytes) => return Ok(bytes.to_vec()),
                    Err(e) => errors.push(format!("{url}: invalid bytes: {e}")),
                }
            }
            Err(format!(
                "martin tile request failed: {}",
                errors.join(" | ")
            ))
        })
    }

    fn parse_martin_tile(
        tile_bytes: &[u8],
        martin_tile: TileKey,
        cfg: &OsmWorldConfig,
        local_tile: TileKey,
    ) -> Result<TileSceneData, String> {
        let vector_tile =
            VectorTile::decode(tile_bytes).map_err(|e| format!("invalid MVT tile: {e}"))?;

        let mut buildings = Vec::new();
        let mut roads = Vec::new();
        let mut areas = Vec::new();
        let (min_x, max_x, min_y, max_y) = local_tile_bounds_rect(local_tile, cfg);
        let mut synthetic_id = -1_i64;

        for layer in vector_tile.layers {
            let extent = layer.extent.unwrap_or(4096).max(1);

            for feature in &layer.features {
                let geometry_type =
                    match feature.geometry_type.and_then(VectorTileGeomType::from_i32) {
                        Some(kind) => kind,
                        None => continue,
                    };

                let tags = decode_mvt_tags(&layer, feature);
                let paths = decode_mvt_paths(&feature.geometry);
                if paths.is_empty() {
                    continue;
                }

                let make_id = |path_idx: usize, synthetic_id: &mut i64| -> i64 {
                    if let Some(base) = feature.id.filter(|id| *id > 0) {
                        (base as i64)
                            .saturating_mul(16)
                            .saturating_add(path_idx as i64)
                    } else {
                        let value = *synthetic_id;
                        *synthetic_id -= 1;
                        value
                    }
                };

                match geometry_type {
                    VectorTileGeomType::Polygon => {
                        for (path_idx, path) in paths.iter().enumerate() {
                            let mut points = path
                                .iter()
                                .map(|(u, v)| {
                                    geo_to_local(
                                        mvt_point_to_geo(
                                            *u,
                                            *v,
                                            martin_tile,
                                            cfg.martin_zoom,
                                            extent,
                                        ),
                                        cfg.origin(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            dedupe_ring_points(&mut points);
                            if points.len() < 3 {
                                continue;
                            }
                            let center = centroid(&points);
                            if !point_in_rect(center, min_x, max_x, min_y, max_y) {
                                continue;
                            }

                            if tags.contains_key("building") {
                                let area = polygon_area_abs(&points);
                                if area < 8.0 {
                                    continue;
                                }

                                let levels = tags
                                    .get("building:levels")
                                    .and_then(|s| parse_prefixed_f32(s));
                                let mut height_m = tags
                                    .get("height")
                                    .and_then(|s| parse_prefixed_f32(s))
                                    .or_else(|| levels.map(|x| x * 3.2))
                                    .unwrap_or(12.0);
                                height_m = height_m.clamp(3.0, 450.0);
                                let name = tags
                                    .get("name")
                                    .cloned()
                                    .or_else(|| tags.get("addr:housename").cloned())
                                    .unwrap_or_else(|| {
                                        format!("Building #{}", feature.id.unwrap_or(0))
                                    });
                                let id = make_id(path_idx, &mut synthetic_id);
                                let tint = classify_building_tint(&tags, &name);
                                buildings.push(BuildingGeom {
                                    id,
                                    name,
                                    tint,
                                    height_m,
                                    levels,
                                    points,
                                    area_m2: area,
                                });
                                continue;
                            }

                            if let Some((tint, kind)) = classify_area_tint(&tags) {
                                let area = polygon_area_abs(&points);
                                if area < 120.0 {
                                    continue;
                                }
                                areas.push(LandAreaGeom {
                                    id: make_id(path_idx, &mut synthetic_id),
                                    kind,
                                    tint,
                                    points,
                                    area_m2: area,
                                });
                            }
                        }
                    }
                    VectorTileGeomType::LineString => {
                        for (path_idx, path) in paths.iter().enumerate() {
                            let mut points = path
                                .iter()
                                .map(|(u, v)| {
                                    geo_to_local(
                                        mvt_point_to_geo(
                                            *u,
                                            *v,
                                            martin_tile,
                                            cfg.martin_zoom,
                                            extent,
                                        ),
                                        cfg.origin(),
                                    )
                                })
                                .collect::<Vec<_>>();
                            dedupe_polyline_points(&mut points);
                            if points.len() < 2 {
                                continue;
                            }
                            let center = centroid(&points);
                            if !point_in_rect(center, min_x, max_x, min_y, max_y) {
                                continue;
                            }

                            if is_pier_way(&tags) {
                                let length_m = polyline_length(&points);
                                if length_m >= 20.0 {
                                    roads.push(RoadGeom {
                                        id: make_id(path_idx, &mut synthetic_id),
                                        kind: "pier".to_string(),
                                        name: road_display_name(
                                            &tags,
                                            "pier",
                                            feature.id.unwrap_or(0) as i64,
                                        ),
                                        fast: false,
                                        width_m: road_width_for_kind("pier"),
                                        points,
                                        length_m,
                                    });
                                }
                                continue;
                            }

                            if is_coastline_way(&tags) {
                                let length_m = polyline_length(&points);
                                if length_m >= 30.0 {
                                    roads.push(RoadGeom {
                                        id: make_id(path_idx, &mut synthetic_id),
                                        kind: "coastline".to_string(),
                                        name: road_display_name(
                                            &tags,
                                            "coastline",
                                            feature.id.unwrap_or(0) as i64,
                                        ),
                                        fast: false,
                                        width_m: road_width_for_kind("coastline"),
                                        points,
                                        length_m,
                                    });
                                }
                                continue;
                            }

                            if let Some(kind) = tags.get("highway") {
                                if !include_highway(kind) {
                                    continue;
                                }
                                let length_m = polyline_length(&points);
                                if length_m < 20.0 {
                                    continue;
                                }
                                roads.push(RoadGeom {
                                    id: make_id(path_idx, &mut synthetic_id),
                                    kind: kind.clone(),
                                    name: road_display_name(
                                        &tags,
                                        kind,
                                        feature.id.unwrap_or(0) as i64,
                                    ),
                                    fast: is_fast_road_kind(kind),
                                    width_m: road_width_for_kind(kind),
                                    points,
                                    length_m,
                                });
                            }
                        }
                    }
                    VectorTileGeomType::Point | VectorTileGeomType::Unknown => {}
                }
            }
        }

        Ok(TileSceneData {
            buildings,
            roads,
            areas,
        })
    }

    fn request_overpass(query: String, cfg: &OsmWorldConfig) -> Result<String, String> {
        let urls = cfg.overpass_urls.clone();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("tokio runtime init failed: {e}"))?;

        rt.block_on(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(25))
                .build()
                .map_err(|e| format!("http client init failed: {e}"))?;

            let mut errors = Vec::new();
            for url in urls {
                let response = match client
                    .post(&url)
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .form(&[("data", query.clone())])
                    .send()
                    .await
                {
                    Ok(response) => response,
                    Err(e) => {
                        errors.push(format!("{url}: request failed: {e}"));
                        continue;
                    }
                };

                let status = response.status();
                let body = match response.text().await {
                    Ok(body) => body,
                    Err(e) => {
                        errors.push(format!("{url}: response read failed: {e}"));
                        continue;
                    }
                };

                if status.is_success() {
                    return Ok(body);
                }

                let snippet = if body.len() > 256 {
                    format!("{}...", &body[..256])
                } else {
                    body
                };
                errors.push(format!("{url}: HTTP {status}: {snippet}"));
            }

            Err(format!(
                "all overpass endpoints failed: {}",
                errors.join(" | ")
            ))
        })
    }

    fn extrude_building_mesh(
        points: &[Vec2],
        height_m: f32,
        wall_color: Color,
        roof_color: Color,
    ) -> Option<Mesh> {
        if points.len() < 3 || height_m <= 0.0 {
            return None;
        }

        let n = points.len();
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let roof_rgba = color_to_linear_rgba(roof_color);
        let wall_rgba = color_to_linear_rgba(wall_color);
        let wall_base_rgba = scale_rgb(wall_rgba, 0.72);

        let roof_start = positions.len() as u32;
        for p in points {
            positions.push([p.x, height_m, p.y]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([p.x * 0.001, p.y * 0.001]);
            colors.push(roof_rgba);
        }

        let clockwise = polygon_area_signed(points) < 0.0;
        for i in 1..(n - 1) {
            if clockwise {
                indices.push(roof_start);
                indices.push(roof_start + i as u32);
                indices.push(roof_start + i as u32 + 1);
            } else {
                indices.push(roof_start);
                indices.push(roof_start + i as u32 + 1);
                indices.push(roof_start + i as u32);
            }
        }

        for i in 0..n {
            let j = (i + 1) % n;
            let p0 = points[i];
            let p1 = points[j];
            let edge = p1 - p0;
            let len = edge.length();
            if len < 0.01 {
                continue;
            }
            let mut nx = edge.y / len;
            let mut nz = -edge.x / len;
            if clockwise {
                nx = -nx;
                nz = -nz;
            }

            let base = positions.len() as u32;
            positions.push([p0.x, 0.0, p0.y]);
            positions.push([p1.x, 0.0, p1.y]);
            positions.push([p1.x, height_m, p1.y]);
            positions.push([p0.x, height_m, p0.y]);
            for _ in 0..4 {
                normals.push([nx, 0.0, nz]);
            }
            uvs.push([0.0, 0.0]);
            uvs.push([1.0, 0.0]);
            uvs.push([1.0, 1.0]);
            uvs.push([0.0, 1.0]);
            colors.push(wall_base_rgba);
            colors.push(wall_base_rgba);
            colors.push(wall_rgba);
            colors.push(wall_rgba);

            if clockwise {
                indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
            } else {
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh.insert_indices(Indices::U32(indices));
        Some(mesh)
    }

    fn enqueue_tile_scene(
        tile: TileKey,
        scene: &TileSceneData,
        queue: &mut VecDeque<PendingRenderable>,
    ) {
        for area in &scene.areas {
            queue.push_back(PendingRenderable {
                tile,
                item: RenderableItem::Area(area.clone()),
            });
        }
        for road in &scene.roads {
            queue.push_back(PendingRenderable {
                tile,
                item: RenderableItem::Road(road.clone()),
            });
        }
        for building in &scene.buildings {
            queue.push_back(PendingRenderable {
                tile,
                item: RenderableItem::Building(building.clone()),
            });
        }
    }

    fn include_highway(kind: &str) -> bool {
        matches!(
            kind,
            "motorway"
                | "motorway_link"
                | "trunk"
                | "trunk_link"
                | "primary"
                | "primary_link"
                | "secondary"
                | "secondary_link"
                | "tertiary"
                | "tertiary_link"
                | "residential"
                | "unclassified"
                | "living_street"
                | "road"
        )
    }

    fn is_fast_road_kind(kind: &str) -> bool {
        matches!(
            kind,
            "motorway" | "motorway_link" | "trunk" | "trunk_link" | "primary" | "primary_link"
        )
    }

    fn road_display_name(tags: &HashMap<String, String>, kind: &str, way_id: i64) -> String {
        tags.get("name")
            .or_else(|| tags.get("official_name"))
            .or_else(|| tags.get("ref"))
            .cloned()
            .unwrap_or_else(|| format!("{kind} #{way_id}"))
    }

    fn road_width_for_kind(kind: &str) -> f32 {
        match kind {
            "coastline" => 20.0,
            "motorway" | "motorway_link" => 10.0,
            "trunk" | "trunk_link" => 8.0,
            "primary" | "primary_link" => 6.6,
            "secondary" | "secondary_link" => 5.6,
            "tertiary" | "tertiary_link" => 4.8,
            "residential" | "living_street" => 4.0,
            "pier" => 15.0,
            _ => 3.6,
        }
    }

    fn road_color_for_kind(kind: &str) -> Color {
        match kind {
            "coastline" => Color::srgb(0.84, 0.78, 0.56),
            "pier" => Color::srgb(0.90, 0.48, 0.16),
            "motorway" | "motorway_link" => Color::srgb(0.28, 0.28, 0.30),
            "trunk" | "trunk_link" => Color::srgb(0.30, 0.30, 0.32),
            "primary" | "primary_link" => Color::srgb(0.33, 0.33, 0.34),
            "secondary" | "secondary_link" => Color::srgb(0.36, 0.36, 0.37),
            _ => Color::srgb(0.40, 0.40, 0.41),
        }
    }

    fn road_highlight_color_for_kind(kind: &str) -> Color {
        match kind {
            "motorway" | "motorway_link" => Color::srgb(0.95, 0.85, 0.20),
            "trunk" | "trunk_link" => Color::srgb(0.94, 0.78, 0.24),
            "primary" | "primary_link" => Color::srgb(0.92, 0.72, 0.28),
            _ => Color::srgb(0.90, 0.76, 0.26),
        }
    }

    fn is_school_value(value: &str) -> bool {
        matches!(value, "school" | "college" | "university" | "kindergarten")
    }

    fn is_school_way(tags: &HashMap<String, String>) -> bool {
        tags.get("amenity")
            .map(|value| is_school_value(value))
            .unwrap_or(false)
            || tags
                .get("building")
                .map(|value| is_school_value(value))
                .unwrap_or(false)
    }

    fn is_hospital_way(tags: &HashMap<String, String>) -> bool {
        tags.get("amenity")
            .map(|value| value == "hospital")
            .unwrap_or(false)
            || tags
                .get("building")
                .map(|value| value == "hospital")
                .unwrap_or(false)
            || tags
                .get("healthcare")
                .map(|value| value == "hospital")
                .unwrap_or(false)
    }

    fn is_fire_station_way(tags: &HashMap<String, String>) -> bool {
        tags.get("amenity")
            .map(|value| value == "fire_station")
            .unwrap_or(false)
            || tags
                .get("emergency")
                .map(|value| value == "fire_station")
                .unwrap_or(false)
            || tags
                .get("building")
                .map(|value| value == "fire_station")
                .unwrap_or(false)
    }

    fn is_pier_way(tags: &HashMap<String, String>) -> bool {
        tags.get("man_made")
            .map(|value| value == "pier")
            .unwrap_or(false)
            || tags
                .get("tourism")
                .map(|value| value == "pier")
                .unwrap_or(false)
            || tags
                .get("leisure")
                .map(|value| value == "pier")
                .unwrap_or(false)
            || tags
                .get("building")
                .map(|value| value == "pier")
                .unwrap_or(false)
    }

    fn is_coastline_way(tags: &HashMap<String, String>) -> bool {
        tags.get("natural")
            .map(|value| value == "coastline")
            .unwrap_or(false)
    }

    fn classify_building_tint(tags: &HashMap<String, String>, name: &str) -> BuildingTint {
        if is_pier_way(tags) || name.to_ascii_lowercase().contains("pier") {
            BuildingTint::Pier
        } else if is_hospital_way(tags) || name.to_ascii_lowercase().contains("hospital") {
            BuildingTint::Hospital
        } else if is_fire_station_way(tags) || name.to_ascii_lowercase().contains("fire station") {
            BuildingTint::FireStation
        } else if is_school_way(tags) {
            BuildingTint::School
        } else {
            BuildingTint::Default
        }
    }

    fn classify_area_tint(tags: &HashMap<String, String>) -> Option<(LandTint, String)> {
        if is_pier_way(tags) {
            return Some((LandTint::Pier, "man_made:pier".to_string()));
        }

        if let Some(value) = tags.get("amenity")
            && is_school_value(value)
        {
            return Some((LandTint::School, format!("amenity:{value}")));
        }

        if let Some(value) = tags.get("waterway")
            && matches!(value.as_str(), "riverbank" | "dock" | "canal")
        {
            return Some((LandTint::Water, format!("waterway:{value}")));
        }

        if let Some(value) = tags.get("natural") {
            match value.as_str() {
                "water" | "wetland" | "bay" => {
                    return Some((LandTint::Water, format!("natural:{value}")));
                }
                "beach" | "sand" => {
                    return Some((LandTint::Beach, format!("natural:{value}")));
                }
                "wood" | "scrub" => return Some((LandTint::Forest, format!("natural:{value}"))),
                "grassland" => return Some((LandTint::Park, format!("natural:{value}"))),
                _ => {}
            }
        }

        if let Some(value) = tags.get("landuse") {
            match value.as_str() {
                "reservoir" | "basin" => {
                    return Some((LandTint::Water, format!("landuse:{value}")));
                }
                "forest" => return Some((LandTint::Forest, format!("landuse:{value}"))),
                "grass" | "meadow" | "recreation_ground" | "village_green" => {
                    return Some((LandTint::Park, format!("landuse:{value}")));
                }
                _ => {}
            }
        }

        if let Some(value) = tags.get("leisure")
            && matches!(value.as_str(), "park" | "golf_course" | "nature_reserve")
        {
            return Some((LandTint::Park, format!("leisure:{value}")));
        }

        None
    }

    fn area_color(tint: LandTint) -> Color {
        match tint {
            LandTint::Forest => Color::srgb(0.16, 0.28, 0.16),
            LandTint::Park => Color::srgb(0.24, 0.36, 0.21),
            LandTint::Water => Color::srgb(0.12, 0.27, 0.40),
            LandTint::Beach => Color::srgb(0.88, 0.82, 0.55),
            LandTint::School => Color::srgb(0.72, 0.20, 0.20),
            LandTint::Pier => Color::srgb(0.90, 0.48, 0.16),
        }
    }

    fn area_tint_priority(tint: LandTint) -> u8 {
        match tint {
            LandTint::Water => 0,
            LandTint::Beach => 1,
            LandTint::Pier => 2,
            LandTint::School => 3,
            LandTint::Forest => 4,
            LandTint::Park => 5,
        }
    }

    fn polyline_length(points: &[Vec2]) -> f32 {
        if points.len() < 2 {
            return 0.0;
        }
        let mut length = 0.0;
        for i in 0..(points.len() - 1) {
            length += points[i].distance(points[i + 1]);
        }
        length
    }

    fn dedupe_polyline_points(points: &mut Vec<Vec2>) {
        if points.len() < 2 {
            return;
        }
        let mut deduped = Vec::with_capacity(points.len());
        for p in points.iter().copied() {
            if deduped
                .last()
                .map(|q: &Vec2| q.distance(p) < 0.01)
                .unwrap_or(false)
            {
                continue;
            }
            deduped.push(p);
        }
        *points = deduped;
    }

    fn road_strip_mesh(points: &[Vec2], width_m: f32, y: f32, color: Color) -> Option<Mesh> {
        if points.len() < 2 || width_m <= 0.1 {
            return None;
        }
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let rgba = color_to_linear_rgba(color);
        let half = width_m * 0.5;
        let mut along = 0.0_f32;

        for i in 0..(points.len() - 1) {
            let p0 = points[i];
            let p1 = points[i + 1];
            let edge = p1 - p0;
            let len = edge.length();
            if len < 0.1 {
                continue;
            }
            let normal = Vec2::new(-edge.y, edge.x).normalize_or_zero();
            if normal.length_squared() < 0.0001 {
                continue;
            }
            let offset = normal * half;
            let l0 = p0 + offset;
            let r0 = p0 - offset;
            let l1 = p1 + offset;
            let r1 = p1 - offset;

            let base = positions.len() as u32;
            positions.push([l0.x, y, l0.y]);
            positions.push([r0.x, y, r0.y]);
            positions.push([r1.x, y, r1.y]);
            positions.push([l1.x, y, l1.y]);
            for _ in 0..4 {
                normals.push([0.0, 1.0, 0.0]);
                colors.push(rgba);
            }
            let v0 = along * 0.02;
            along += len;
            let v1 = along * 0.02;
            uvs.push([0.0, v0]);
            uvs.push([1.0, v0]);
            uvs.push([1.0, v1]);
            uvs.push([0.0, v1]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        if indices.is_empty() {
            return None;
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh.insert_indices(Indices::U32(indices));
        Some(mesh)
    }

    fn ground_polygon_mesh(points: &[Vec2], y: f32, color: Color) -> Option<Mesh> {
        if points.len() < 3 {
            return None;
        }
        let n = points.len();
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n);
        let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n);
        let mut indices: Vec<u32> = Vec::new();
        let rgba = color_to_linear_rgba(color);

        for p in points {
            positions.push([p.x, y, p.y]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([p.x * 0.001, p.y * 0.001]);
            colors.push(rgba);
        }

        let clockwise = polygon_area_signed(points) < 0.0;
        for i in 1..(n - 1) {
            if clockwise {
                indices.push(0);
                indices.push(i as u32);
                indices.push(i as u32 + 1);
            } else {
                indices.push(0);
                indices.push(i as u32 + 1);
                indices.push(i as u32);
            }
        }

        if indices.is_empty() {
            return None;
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh.insert_indices(Indices::U32(indices));
        Some(mesh)
    }

    fn wall_color_for_building(height_m: f32, building_id: i64, tint: BuildingTint) -> Color {
        let jitter = palette_jitter(building_id);
        match tint {
            BuildingTint::Pier => Color::srgb(
                (0.90 * jitter).clamp(0.0, 1.0),
                (0.48 * jitter).clamp(0.0, 1.0),
                (0.16 * jitter).clamp(0.0, 1.0),
            ),
            BuildingTint::School => Color::srgb(
                (0.72 * jitter).clamp(0.0, 1.0),
                (0.20 * jitter).clamp(0.0, 1.0),
                (0.20 * jitter).clamp(0.0, 1.0),
            ),
            BuildingTint::Hospital => Color::srgb(
                (0.22 * jitter).clamp(0.0, 1.0),
                (0.62 * jitter).clamp(0.0, 1.0),
                (0.29 * jitter).clamp(0.0, 1.0),
            ),
            BuildingTint::FireStation => Color::srgb(
                (0.90 * jitter).clamp(0.0, 1.0),
                (0.78 * jitter).clamp(0.0, 1.0),
                (0.18 * jitter).clamp(0.0, 1.0),
            ),
            BuildingTint::Default => {
                let t = (height_m / 250.0).clamp(0.0, 1.0);
                let r = (0.45 + 0.22 * t) * jitter;
                let g = (0.46 + 0.20 * t) * jitter;
                let b = (0.44 + 0.14 * (1.0 - t)) * jitter;
                Color::srgb(r, g, b)
            }
        }
    }

    fn roof_color_for_wall(wall: Color) -> Color {
        let [r, g, b, _] = wall.to_linear().to_f32_array();
        Color::linear_rgba(
            (r * 1.10).clamp(0.0, 1.0),
            (g * 1.10).clamp(0.0, 1.0),
            (b * 1.05).clamp(0.0, 1.0),
            1.0,
        )
    }

    fn palette_jitter(building_id: i64) -> f32 {
        let mut x = building_id as u64;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        let unit = (x & 0xffff) as f32 / 65535.0;
        0.88 + 0.24 * unit
    }

    fn color_to_linear_rgba(color: Color) -> [f32; 4] {
        color.to_linear().to_f32_array()
    }

    fn scale_rgb(color: [f32; 4], factor: f32) -> [f32; 4] {
        [
            (color[0] * factor).clamp(0.0, 1.0),
            (color[1] * factor).clamp(0.0, 1.0),
            (color[2] * factor).clamp(0.0, 1.0),
            color[3],
        ]
    }

    fn cache_key_for_tile(tile: TileKey, cfg: &OsmWorldConfig) -> String {
        format!(
            "tile_v6_x{}_y{}_size{:.0}_o{:.5}_{:.5}",
            tile.x, tile.y, cfg.tile_size_m, cfg.origin_lat, cfg.origin_lon
        )
    }

    fn cache_path(cfg: &OsmWorldConfig, key: &str) -> Option<PathBuf> {
        cfg.cache_dir
            .as_ref()
            .map(|dir| dir.join(format!("{key}.json")))
    }

    fn save_cache(cfg: &OsmWorldConfig, key: &str, data: &str) {
        let Some(path) = cache_path(cfg, key) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, data);
    }

    fn load_cache(cfg: &OsmWorldConfig, key: &str) -> Option<String> {
        let path = cache_path(cfg, key)?;
        std::fs::read_to_string(path).ok()
    }

    fn save_cache_bytes(cfg: &OsmWorldConfig, key: &str, data: &[u8]) {
        let Some(path) = cache_path(cfg, key) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, data);
    }

    fn load_cache_bytes(cfg: &OsmWorldConfig, key: &str) -> Option<Vec<u8>> {
        let path = cache_path(cfg, key)?;
        std::fs::read(path).ok()
    }

    fn tile_bounds_geo(tile: TileKey, cfg: &OsmWorldConfig) -> (f64, f64, f64, f64) {
        let origin = cfg.origin();
        let west_m = f64::from(tile.x) * cfg.tile_size_m - TILE_QUERY_MARGIN_M;
        let east_m = (f64::from(tile.x) + 1.0) * cfg.tile_size_m + TILE_QUERY_MARGIN_M;
        let south_m = f64::from(tile.y) * cfg.tile_size_m - TILE_QUERY_MARGIN_M;
        let north_m = (f64::from(tile.y) + 1.0) * cfg.tile_size_m + TILE_QUERY_MARGIN_M;

        let sw = local_m_to_geo(west_m, south_m, origin);
        let ne = local_m_to_geo(east_m, north_m, origin);

        let south = sw.lat.min(ne.lat);
        let north = sw.lat.max(ne.lat);
        let west = sw.lon.min(ne.lon);
        let east = sw.lon.max(ne.lon);
        (south, west, north, east)
    }

    fn tile_center_local(tile: TileKey, cfg: &OsmWorldConfig) -> Vec2 {
        let east = (f64::from(tile.x) + 0.5) * cfg.tile_size_m;
        let north = (f64::from(tile.y) + 0.5) * cfg.tile_size_m;
        Vec2::new(east as f32, -(north as f32))
    }

    fn geo_to_tile(point: GeoPoint, cfg: &OsmWorldConfig) -> TileKey {
        let local = geo_to_local(point, cfg.origin());
        let east = local.x as f64;
        let north = -(local.y as f64);
        TileKey {
            x: (east / cfg.tile_size_m).floor() as i32,
            y: (north / cfg.tile_size_m).floor() as i32,
        }
    }

    fn desired_tiles_around(center: TileKey, radius: i32) -> Vec<TileKey> {
        let mut out = Vec::new();
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                out.push(TileKey {
                    x: center.x + dx,
                    y: center.y + dy,
                });
            }
        }

        out.sort_by(|a, b| {
            let adx = i64::from(a.x - center.x);
            let ady = i64::from(a.y - center.y);
            let bdx = i64::from(b.x - center.x);
            let bdy = i64::from(b.y - center.y);
            let da = adx * adx + ady * ady;
            let db = bdx * bdx + bdy * bdy;
            da.cmp(&db)
        });
        out
    }

    fn tile_distance_sq(a: TileKey, b: TileKey) -> i64 {
        let dx = i64::from(a.x - b.x);
        let dy = i64::from(a.y - b.y);
        dx * dx + dy * dy
    }

    fn tile_jump_chebyshev(a: TileKey, b: TileKey) -> i32 {
        (a.x - b.x).abs().max((a.y - b.y).abs())
    }

    fn force_recenter_streaming_state(
        state: &mut OsmWorldState,
        hovered: &mut HoveredBuilding,
        hovered_road: &mut HoveredRoad,
        center_tile: TileKey,
        prefetch_tiles: &HashSet<TileKey>,
    ) {
        hovered.0 = None;
        hovered_road.0 = None;

        for (_, entities) in state.loaded_tiles.drain() {
            state.pending_to_despawn.extend(entities);
        }
        state.pending_renderables.clear();
        state.pending_total = 0;
        state.pending_tiles.clear();

        // In-flight worker threads cannot be cancelled; drop accounting so new center
        // can schedule fetches immediately.
        state.inflight_tiles.clear();
        state.failed_attempts.clear();
        state.next_retry_at.clear();

        state
            .prefetched_tiles
            .retain(|tile, _| prefetch_tiles.contains(tile));
        state.last_prefetch_center_local = None;
        state.last_prefetch_tile = None;
        state.last_reported_center_tile = Some(center_tile);
    }

    fn force_redraw_active_tiles(
        state: &mut OsmWorldState,
        hovered: &mut HoveredBuilding,
        hovered_road: &mut HoveredRoad,
        active_tiles: &[TileKey],
    ) -> usize {
        hovered.0 = None;
        hovered_road.0 = None;

        let mut refreshed_tiles = 0usize;
        for tile in active_tiles {
            if let Some(entities) = state.loaded_tiles.remove(tile) {
                state.pending_to_despawn.extend(entities);
            }
            state
                .pending_renderables
                .retain(|pending| pending.tile != *tile);
            if let Some(scene) = state.prefetched_tiles.get(tile).cloned() {
                enqueue_tile_scene(*tile, &scene, &mut state.pending_renderables);
                refreshed_tiles += 1;
            }
        }

        state.pending_total = state.pending_renderables.len();
        refreshed_tiles
    }

    fn prune_missing_loaded_entities(
        state: &mut OsmWorldState,
        existing_entities: &Query<Entity>,
    ) -> usize {
        let mut removed = 0usize;
        state.loaded_tiles.retain(|_, entities| {
            let before = entities.len();
            entities.retain(|entity| existing_entities.contains(*entity));
            removed += before.saturating_sub(entities.len());
            !entities.is_empty()
        });
        removed
    }

    fn reprioritize_pending_tiles(
        pending_tiles: &mut VecDeque<TileKey>,
        center: TileKey,
        active_tiles: &HashSet<TileKey>,
    ) {
        let mut seen = HashSet::new();
        let mut pending: Vec<TileKey> = pending_tiles
            .drain(..)
            .filter(|tile| seen.insert(*tile))
            .collect();

        pending.sort_by(|a, b| {
            let a_priority = if active_tiles.contains(a) { 0_u8 } else { 1_u8 };
            let b_priority = if active_tiles.contains(b) { 0_u8 } else { 1_u8 };
            a_priority
                .cmp(&b_priority)
                .then_with(|| tile_distance_sq(*a, center).cmp(&tile_distance_sq(*b, center)))
        });

        *pending_tiles = pending.into_iter().collect();
    }

    fn geo_to_local(p: GeoPoint, origin: GeoPoint) -> Vec2 {
        let east = (p.lon - origin.lon) * meters_per_degree_lon(origin.lat);
        let north = (p.lat - origin.lat) * meters_per_degree_lat(origin.lat);
        Vec2::new(east as f32, -north as f32)
    }

    fn local_m_to_geo(east_m: f64, north_m: f64, origin: GeoPoint) -> GeoPoint {
        let lat = origin.lat + north_m / meters_per_degree_lat(origin.lat);
        let lon = origin.lon + east_m / meters_per_degree_lon(origin.lat);
        GeoPoint { lat, lon }
    }

    fn tile_center_geo_from_local_tile(tile: TileKey, cfg: &OsmWorldConfig) -> GeoPoint {
        let east_m = (f64::from(tile.x) + 0.5) * cfg.tile_size_m;
        let north_m = (f64::from(tile.y) + 0.5) * cfg.tile_size_m;
        local_m_to_geo(east_m, north_m, cfg.origin())
    }

    fn geo_to_slippy_tile(point: GeoPoint, zoom: u8) -> TileKey {
        let zoom_scale = 2_f64.powi(i32::from(zoom));
        let x = (((point.lon + 180.0) / 360.0) * zoom_scale).floor() as i32;
        let lat_rad = point.lat.to_radians();
        let y = ((1.0 - ((lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI))
            * 0.5
            * zoom_scale)
            .floor() as i32;
        TileKey { x, y }
    }

    fn mvt_point_to_geo(u: i32, v: i32, tile: TileKey, zoom: u8, extent: u32) -> GeoPoint {
        let extent = f64::from(extent.max(1));
        let zoom_scale = 2_f64.powi(i32::from(zoom));
        let x = f64::from(tile.x) + f64::from(u) / extent;
        let y = f64::from(tile.y) + f64::from(v) / extent;
        let lon = x / zoom_scale * 360.0 - 180.0;
        let n = std::f64::consts::PI * (1.0 - 2.0 * y / zoom_scale);
        let lat = n.sinh().atan().to_degrees();
        GeoPoint { lat, lon }
    }

    fn local_tile_bounds_rect(tile: TileKey, cfg: &OsmWorldConfig) -> (f32, f32, f32, f32) {
        let west_m = f64::from(tile.x) * cfg.tile_size_m - TILE_QUERY_MARGIN_M;
        let east_m = (f64::from(tile.x) + 1.0) * cfg.tile_size_m + TILE_QUERY_MARGIN_M;
        let south_m = f64::from(tile.y) * cfg.tile_size_m - TILE_QUERY_MARGIN_M;
        let north_m = (f64::from(tile.y) + 1.0) * cfg.tile_size_m + TILE_QUERY_MARGIN_M;
        let min_x = west_m as f32;
        let max_x = east_m as f32;
        let min_y = -(north_m as f32);
        let max_y = -(south_m as f32);
        (min_x, max_x, min_y, max_y)
    }

    fn point_in_rect(point: Vec2, min_x: f32, max_x: f32, min_y: f32, max_y: f32) -> bool {
        point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
    }

    fn decode_mvt_tags(
        layer: &VectorTileLayer,
        feature: &VectorTileFeature,
    ) -> HashMap<String, String> {
        let mut tags = HashMap::new();
        for pair in feature.tags.chunks_exact(2) {
            let key_idx = pair[0] as usize;
            let value_idx = pair[1] as usize;
            let Some(key) = layer.keys.get(key_idx) else {
                continue;
            };
            let Some(value) = layer.values.get(value_idx) else {
                continue;
            };
            tags.insert(key.clone(), vector_tile_value_to_string(value));
        }
        tags
    }

    fn vector_tile_value_to_string(value: &VectorTileValue) -> String {
        if let Some(v) = &value.string_value {
            return v.clone();
        }
        if let Some(v) = value.bool_value {
            return if v { "true" } else { "false" }.to_string();
        }
        if let Some(v) = value.int_value {
            return v.to_string();
        }
        if let Some(v) = value.uint_value {
            return v.to_string();
        }
        if let Some(v) = value.sint_value {
            return v.to_string();
        }
        if let Some(v) = value.double_value {
            return format!("{v}");
        }
        if let Some(v) = value.float_value {
            return format!("{v}");
        }
        String::new()
    }

    fn decode_mvt_paths(commands: &[u32]) -> Vec<Vec<(i32, i32)>> {
        let mut paths: Vec<Vec<(i32, i32)>> = Vec::new();
        let mut path: Vec<(i32, i32)> = Vec::new();
        let mut cursor = 0usize;
        let mut x = 0i32;
        let mut y = 0i32;

        while cursor < commands.len() {
            let command = commands[cursor];
            cursor += 1;
            let id = command & 0x7;
            let count = command >> 3;
            match id {
                1 => {
                    for _ in 0..count {
                        if cursor + 1 >= commands.len() {
                            break;
                        }
                        x += decode_zigzag_i32(commands[cursor]);
                        y += decode_zigzag_i32(commands[cursor + 1]);
                        cursor += 2;
                        if !path.is_empty() {
                            paths.push(path);
                            path = Vec::new();
                        }
                        path.push((x, y));
                    }
                }
                2 => {
                    for _ in 0..count {
                        if cursor + 1 >= commands.len() {
                            break;
                        }
                        x += decode_zigzag_i32(commands[cursor]);
                        y += decode_zigzag_i32(commands[cursor + 1]);
                        cursor += 2;
                        path.push((x, y));
                    }
                }
                7 => {
                    if !path.is_empty() {
                        if let Some(first) = path.first().copied() {
                            path.push(first);
                        }
                    }
                }
                _ => break,
            }
        }
        if !path.is_empty() {
            paths.push(path);
        }
        paths
    }

    fn decode_zigzag_i32(value: u32) -> i32 {
        ((value >> 1) as i32) ^ -((value & 1) as i32)
    }

    fn centroid(points: &[Vec2]) -> Vec2 {
        if points.is_empty() {
            return Vec2::ZERO;
        }
        let sum = points
            .iter()
            .fold(Vec2::ZERO, |acc, p| Vec2::new(acc.x + p.x, acc.y + p.y));
        sum / points.len() as f32
    }

    fn retry_backoff(base: Duration, failures: u32) -> Duration {
        let base_s = base.as_secs_f64().max(0.25);
        let exponent = failures.saturating_sub(1).min(8) as i32;
        let secs = (base_s * 2.0_f64.powi(exponent)).min(300.0);
        Duration::from_secs_f64(secs)
    }

    fn current_center(
        config: &OsmWorldConfig,
        world_pos_query: &Query<(&ComponentId, &WorldPos)>,
    ) -> TrackSample {
        let origin = config.origin();
        if !config.tracked_world_pos_component.is_empty() {
            let configured = ComponentId::new(&config.tracked_world_pos_component);
            for (component_id, world_pos) in world_pos_query.iter() {
                if *component_id == configured {
                    return TrackSample {
                        geo: world_pos_to_geo(world_pos, origin),
                        mode: track_mode_for_component(*component_id),
                    };
                }
            }
        }

        // Prefer the RC-jet when present so fast trajectories are prefetched early.
        let fallback_ids = [
            ComponentId::new("bdx.world_pos"),
            ComponentId::new("drone.world_pos"),
            ComponentId::new("target.world_pos"),
        ];
        for wanted in fallback_ids {
            for (component_id, world_pos) in world_pos_query.iter() {
                if *component_id == wanted {
                    return TrackSample {
                        geo: world_pos_to_geo(world_pos, origin),
                        mode: track_mode_for_component(*component_id),
                    };
                }
            }
        }

        world_pos_query
            .iter()
            .next()
            .map(|(id, wp)| TrackSample {
                geo: world_pos_to_geo(wp, origin),
                mode: track_mode_for_component(*id),
            })
            .unwrap_or(TrackSample {
                geo: origin,
                mode: TrackMode::Unknown,
            })
    }

    fn track_mode_for_component(component_id: ComponentId) -> TrackMode {
        if component_id == ComponentId::new("bdx.world_pos") {
            TrackMode::RcJet
        } else if component_id == ComponentId::new("drone.world_pos") {
            TrackMode::Drone
        } else if component_id == ComponentId::new("target.world_pos") {
            TrackMode::Target
        } else {
            TrackMode::Unknown
        }
    }

    fn world_pos_to_geo(world_pos: &WorldPos, origin: GeoPoint) -> GeoPoint {
        let bevy = world_pos.bevy_pos();
        let east_m = bevy.x;
        let north_m = -bevy.z;
        let lat = origin.lat + north_m / meters_per_degree_lat(origin.lat);
        let lon = origin.lon + east_m / meters_per_degree_lon(origin.lat);
        GeoPoint { lat, lon }
    }

    fn meters_per_degree_lat(lat_deg: f64) -> f64 {
        let lat = lat_deg.to_radians();
        111_132.92 - 559.82 * (2.0 * lat).cos() + 1.175 * (4.0 * lat).cos()
    }

    fn meters_per_degree_lon(lat_deg: f64) -> f64 {
        let lat = lat_deg.to_radians();
        111_412.84 * lat.cos() - 93.5 * (3.0 * lat).cos()
    }

    fn dedupe_ring_points(points: &mut Vec<Vec2>) {
        if points.len() < 3 {
            return;
        }
        let mut deduped = Vec::with_capacity(points.len());
        for p in points.iter().copied() {
            if deduped
                .last()
                .map(|q: &Vec2| q.distance(p) < 0.01)
                .unwrap_or(false)
            {
                continue;
            }
            deduped.push(p);
        }
        if deduped.len() >= 2
            && deduped
                .first()
                .zip(deduped.last())
                .map(|(a, b)| a.distance(*b) < 0.01)
                .unwrap_or(false)
        {
            let _ = deduped.pop();
        }
        *points = deduped;
    }

    fn polygon_area_signed(points: &[Vec2]) -> f32 {
        let mut area = 0.0;
        for i in 0..points.len() {
            let j = (i + 1) % points.len();
            area += points[i].x * points[j].y - points[j].x * points[i].y;
        }
        area * 0.5
    }

    fn polygon_area_abs(points: &[Vec2]) -> f32 {
        polygon_area_signed(points).abs()
    }

    fn parse_prefixed_f32(value: &str) -> Option<f32> {
        let mut out = String::new();
        let mut seen_digit = false;
        for c in value.chars() {
            if c.is_ascii_digit() || c == '.' || c == '-' {
                out.push(c);
                if c.is_ascii_digit() {
                    seen_digit = true;
                }
            } else if seen_digit {
                break;
            }
        }
        if !seen_digit {
            return None;
        }
        f32::from_str(out.trim()).ok()
    }

    fn env_flag(name: &str, default: bool) -> bool {
        std::env::var(name)
            .ok()
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(default)
    }

    fn env_parse<T: FromStr>(name: &str) -> Option<T> {
        std::env::var(name).ok().and_then(|v| v.parse::<T>().ok())
    }

    fn default_cache_dir() -> PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".cache")
                .join("elodin")
                .join("osm_tiles");
        }
        std::env::current_dir()
            .unwrap_or_else(|_| std::env::temp_dir())
            .join(".cache")
            .join("osm_tiles")
    }
}

#[cfg(not(target_family = "wasm"))]
pub use native::*;

#[cfg(target_family = "wasm")]
pub struct OsmWorldPlugin;

#[cfg(target_family = "wasm")]
impl bevy::app::Plugin for OsmWorldPlugin {
    fn build(&self, _app: &mut bevy::app::App) {}
}
