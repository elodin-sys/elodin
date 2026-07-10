use crate::icon_rasterizer::IconTextureCache;
use crate::object_3d::CompileError;
use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    tasks::{IoTaskPool, Task, futures_lite::future},
    window::PrimaryWindow,
};
#[cfg(target_os = "macos")]
use bevy_defer::AsyncCommandsExtension;
use bevy_egui::egui::{Color32, Id};
use bevy_geo_frames::prelude::*;
use bevy_mat3_material::Mat3Material;
use egui_tiles::{Container, Tile, TileId};
use impeller2_bevy::{ComponentPath, ComponentSchemaRegistry, ConnectionAddr};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{
    Graph, Line3d, Object3D, Panel, Schematic, VectorArrow3d, Viewport, WindowSchematic,
};
use miette::{Diagnostic, miette};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[cfg(not(target_os = "macos"))]
use crate::tiles::WindowRelayout;
#[cfg(target_os = "macos")]
use crate::ui::window::placement::apply_physical_screen_rect;
use crate::{
    EqlContext, MainCamera, TimeRangeBehavior,
    plugins::{
        kdl_document::{
            CurrentDocument, DocumentCleared, DocumentCommandFailed, DocumentLoadFailed,
            DocumentLoaded, DocumentReloaded, LastActiveSchematicContent, SchematicDocumentAsset,
            SchematicWindow,
        },
        render_layer_alloc::RenderLayerAllocator,
    },
    ui::{
        DEFAULT_SECONDARY_RECT, HdrEnabled,
        colors::{self, EColor},
        data_overview::DataOverviewPane,
        modal::ModalDialog,
        monitor::MonitorPane,
        plot::GraphBundle,
        query_plot::QueryPlotData,
        schematic::{CurrentSchematic, EqlExt},
        tiles::{
            GraphPane, Pane, TileState, TreePane, ViewportPane, WindowDescriptor, WindowId,
            WindowState,
        },
        timeline::{TelemetryMode, TimelineSettings},
    },
    vector_arrow::{VectorArrowState, ViewportArrow},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PanelContext {
    Main,
    Window(WindowId),
}

/// Unwrap singleton Tabs whose only child is a Graph pane so telemetry mode
/// can reclaim tab-header chrome without affecting multi-tab viewport groups.
fn strip_singleton_graph_tabs(tile_state: &mut TileState) {
    let candidates: Vec<(TileId, TileId)> = tile_state
        .tree
        .tiles
        .iter()
        .filter_map(|(id, tile)| {
            let Tile::Container(Container::Tabs(tabs)) = tile else {
                return None;
            };
            if tabs.children.len() != 1 {
                return None;
            }
            let child = tabs.children[0];
            match tile_state.tree.tiles.get(child) {
                Some(Tile::Pane(Pane::Graph(_))) => Some((*id, child)),
                _ => None,
            }
        })
        .collect();

    for (tabs_id, graph_id) in candidates {
        if tile_state.tree.root == Some(tabs_id) {
            tile_state.tree.root = Some(graph_id);
        } else if let Some(parent_id) = tile_state.tree.tiles.parent_of(tabs_id) {
            match tile_state.tree.tiles.get_mut(parent_id) {
                Some(Tile::Container(Container::Linear(linear))) => {
                    if let Some(i) = linear.children.iter().position(|c| *c == tabs_id) {
                        linear.children[i] = graph_id;
                    }
                    linear.shares.replace_with(tabs_id, graph_id);
                }
                Some(Tile::Container(Container::Tabs(tabs))) => {
                    if let Some(i) = tabs.children.iter().position(|c| *c == tabs_id) {
                        tabs.children[i] = graph_id;
                    }
                    if tabs.active == Some(tabs_id) {
                        tabs.active = Some(graph_id);
                    }
                }
                // Grid uses hole-aware indexing; leave singleton graph tabs nested there.
                _ => {}
            }
        }
        tile_state.tree.tiles.remove(tabs_id);
        tile_state.container_titles.remove(&tabs_id);
    }
}

fn tabs_parent_for_panels(tile_state: &TileState, panel_count: usize) -> Option<TileId> {
    if panel_count > 0 {
        tile_state.tree.root()
    } else {
        None
    }
}

fn apply_fallback_frame_to_panel(
    panel: &Panel,
    fallback_frame: Option<bevy_geo_frames::GeoFrame>,
) -> Panel {
    match panel {
        Panel::Viewport(viewport) => {
            let mut v = viewport.clone();
            if v.frame.is_none() {
                v.frame = fallback_frame;
            }
            Panel::Viewport(v)
        }
        Panel::Tabs(panels) => Panel::Tabs(
            panels
                .iter()
                .map(|p| apply_fallback_frame_to_panel(p, fallback_frame))
                .collect(),
        ),
        Panel::HSplit(split) => {
            let mut s = split.clone();
            s.panels = s
                .panels
                .iter()
                .map(|p| apply_fallback_frame_to_panel(p, fallback_frame))
                .collect();
            Panel::HSplit(s)
        }
        Panel::VSplit(split) => {
            let mut s = split.clone();
            s.panels = s
                .panels
                .iter()
                .map(|p| apply_fallback_frame_to_panel(p, fallback_frame))
                .collect();
            Panel::VSplit(s)
        }
        other => other.clone(),
    }
}

#[derive(Component)]
pub struct SyncedViewport;

#[derive(Component)]
pub struct SchematicSpawned;

/// Number of fetch/parse attempts for one window sub-schematic before it is
/// dropped with a warning. A window's asset can 404 or parse torn bytes
/// transiently — a follower still mirroring it, or a save's `PUT`s mid-flight —
/// and a single silent failure would make the window vanish for the session.
const MAX_WINDOW_FETCH_ATTEMPTS: u32 = 10;

/// Delay between window sub-schematic fetch attempts.
const WINDOW_FETCH_RETRY_DELAY: Duration = Duration::from_millis(400);

/// A window sub-schematic whose `db:`/HTTP KDL is being fetched off the main
/// thread. Spawned by `load_schematic` and drained by
/// `apply_pending_window_schematics` once the fetch lands, so a slow or
/// unreachable DB never blocks the load for each window's request timeout.
struct PendingWindowLoad {
    descriptor: WindowDescriptor,
    theme_mode: Option<String>,
    theme_scheme: String,
    path: PathBuf,
    /// `None` while waiting for a scheduled retry slot.
    task: Option<Task<Result<String, String>>>,
    /// Failed attempts so far; transient errors retry bounded instead of
    /// silently dropping the window.
    attempts: u32,
    retry_at: Option<Instant>,
}

/// Queue of in-flight remote window sub-schematic fetches (RFD #724). Cleared at
/// the start of every `load_schematic` so a superseded load's windows never
/// spawn into the new document.
#[derive(Resource, Default)]
pub struct PendingWindowSchematics {
    loads: Vec<PendingWindowLoad>,
}

#[derive(Component)]
pub struct MonitorsRoot;

pub(crate) fn plugin(app: &mut App) {
    app.add_systems(Startup, |mut commands: Commands| {
        commands.spawn((MonitorsRoot, Name::new("monitors")));
    });
}

#[derive(SystemParam)]
pub struct LoadSchematicParams<'w, 's> {
    pub commands: Commands<'w, 's>,
    primary_window: Single<'w, 's, Entity, With<PrimaryWindow>>,
    pub asset_server: Res<'w, AssetServer>,
    pub current_document: ResMut<'w, CurrentDocument>,
    pub document_assets: Res<'w, Assets<SchematicDocumentAsset>>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<StandardMaterial>>,
    pub world_mesh_materials: ResMut<'w, Assets<bevy_world_mesh::prelude::WorldMeshMaterial>>,
    pub mat3_materials: ResMut<'w, Assets<Mat3Material>>,
    pub images: ResMut<'w, Assets<Image>>,
    pub icon_cache: ResMut<'w, IconTextureCache>,
    pub render_layer_alloc: ResMut<'w, RenderLayerAllocator>,
    pub hdr_enabled: ResMut<'w, HdrEnabled>,
    pub timeline_settings: ResMut<'w, TimelineSettings>,
    pub time_range_behavior: ResMut<'w, TimeRangeBehavior>,
    pub telemetry_mode: ResMut<'w, TelemetryMode>,
    pub schema_reg: Res<'w, ComponentSchemaRegistry>,
    pub eql: Res<'w, EqlContext>,
    connection_addr: Option<Res<'w, ConnectionAddr>>,
    pub geo_context: ResMut<'w, GeoContext>,
    pub sensor_camera_configs: Res<'w, crate::sensor_camera::SensorCameraConfigs>,
    pub coordinate: ResMut<'w, crate::Coordinate>,
    cameras: Query<'w, 's, &'static mut Camera>,
    schematic_spawned: Query<'w, 's, Entity, With<SchematicSpawned>>,
    window_states: Query<'w, 's, (Entity, &'static WindowId, &'static mut WindowState)>,
    pub schematic_bindings: ResMut<'w, super::SchematicBindings>,
    pub current_schematic: ResMut<'w, CurrentSchematic>,
    pending_windows: ResMut<'w, PendingWindowSchematics>,
    /// Records each spawned `db:` window's stored KDL so a revision-gated
    /// refetch can tell a window-only remote save apart from an unrelated
    /// asset write (RFD #724, Bug 2). Optional: absent in minimal test apps.
    last_content: Option<ResMut<'w, LastActiveSchematicContent>>,
    #[cfg(feature = "big_space")]
    big_space_root: Option<Res<'w, crate::spatial::BigSpaceRootEntity>>,
    monitor_root: Single<'w, 's, Entity, With<MonitorsRoot>>,
}

fn apply_theme(theme: Option<&impeller2_wkt::ThemeConfig>) -> colors::SchemeSelection {
    let current = colors::current_selection();
    let scheme = theme
        .and_then(|t| t.scheme.as_deref())
        .unwrap_or(&current.scheme);
    let mode = theme
        .and_then(|t| t.mode.as_deref())
        .unwrap_or(&current.mode);
    colors::set_active_scheme(scheme, mode)
}

struct WindowDescriptors {
    main: Option<WindowDescriptor>,
    windows: Vec<WindowDescriptor>,
}

/// True for window sub-schematics served by the DB Asset Server (`db:`) or a
/// raw HTTP(S) URL, as opposed to a local filesystem path (RFD #724).
fn is_remote_asset_path(path: &str) -> bool {
    path.starts_with("db:") || path.starts_with("http://") || path.starts_with("https://")
}

/// Read a window sub-schematic's KDL. `db:`/HTTP references are fetched from the
/// DB Asset Server (RFD #724); everything else is read from the local filesystem
/// (offline `--kdl` dev). The fetch is bounded so an unreachable DB cannot hang
/// the load.
fn read_window_schematic_kdl(
    path: &Path,
    connection_addr: Option<std::net::SocketAddr>,
) -> Result<String, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| "non-utf8 window path".to_string())?;
    if !is_remote_asset_path(path_str) {
        return std::fs::read_to_string(path).map_err(|err| err.to_string());
    }

    let url = crate::object_3d::resolve_db_asset_url(path_str, connection_addr);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|err| format!("{url}: {err}"))?;
    let response = client
        .get(&url)
        .send()
        .map_err(|err| format!("{url}: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    response.text().map_err(|err| format!("{url}: {err}"))
}

fn resolve_window_descriptor(
    window: &WindowSchematic,
    base_dir: Option<&Path>,
    theme_mode: Option<&str>,
) -> Option<WindowDescriptor> {
    let path_str = window.path.as_ref()?;
    // Keep `db:`/HTTP references verbatim; only local paths are anchored to the
    // schematic's directory. The remote ones are fetched over HTTP at spawn time.
    let resolved = if is_remote_asset_path(path_str) {
        PathBuf::from(path_str)
    } else {
        let mut resolved = PathBuf::from(path_str);
        if resolved.is_relative() {
            if let Some(base) = base_dir {
                resolved = base.join(resolved);
            } else if let Ok(cwd) = std::env::current_dir() {
                resolved = cwd.join(resolved);
            }
        }
        resolved
    };

    Some(WindowDescriptor {
        path: Some(resolved),
        title: window.title.clone(),
        screen: window.screen.map(|idx| idx as usize),
        mode: theme_mode.map(|m| m.to_string()),
        screen_rect: window.screen_rect.or(Some(DEFAULT_SECONDARY_RECT)),
    })
}

fn collect_window_descriptors(
    schematic: &Schematic,
    base_dir: Option<&Path>,
    theme_mode: Option<&str>,
) -> WindowDescriptors {
    let mut main = None;
    let mut windows = Vec::new();

    for elem in &schematic.elems {
        let impeller2_wkt::SchematicElem::Window(window) = elem else {
            continue;
        };

        if let Some(descriptor) = resolve_window_descriptor(window, base_dir, theme_mode) {
            windows.push(descriptor);
        } else {
            main = Some(WindowDescriptor {
                screen: window.screen.map(|idx| idx as usize),
                screen_rect: window.screen_rect,
                ..default()
            });
        }
    }

    WindowDescriptors { main, windows }
}

fn apply_loaded_document(
    params: &mut LoadSchematicParams,
    save_path: Option<&Path>,
    document: &SchematicDocumentAsset,
) {
    let base_dir = save_path.and_then(Path::parent);
    params.load_schematic(&document.root, base_dir, Some(&document.windows));
}

pub fn render_diag(diagnostic: &dyn Diagnostic) -> String {
    let mut buf = String::new();
    miette::GraphicalReportHandler::new_themed(miette::GraphicalTheme::unicode_nocolor())
        .render_report(&mut buf, diagnostic)
        .expect("Failed to render diagnostic");
    buf
}

impl LoadSchematicParams<'_, '_> {
    pub fn load_schematic(
        &mut self,
        schematic: &Schematic,
        base_dir: Option<&Path>,
        window_assets: Option<&[SchematicWindow]>,
    ) {
        // Set global coordinate frame from schematic
        self.coordinate.0 = schematic.frame;

        // Set the GeoContext origin from the schematic (degrees -> radians),
        // resetting to the default when absent so reloads are deterministic.
        // Only write on change: mutating GeoContext re-touches every
        // GeoPosition/GeoRotation in the world.
        let origin = schematic
            .origin
            .map(|o| {
                bevy_geo_frames::GeoOrigin::new_from_degrees(o.latitude, o.longitude, o.altitude)
            })
            .unwrap_or_default();
        let current = &self.geo_context.origin;
        if (current.latitude, current.longitude, current.altitude)
            != (origin.latitude, origin.longitude, origin.altitude)
        {
            self.geo_context.origin = origin;
        }

        // Drop any remote window fetches still in flight from a prior load so
        // their windows can't spawn into this freshly-cleared document.
        self.pending_windows.loads.clear();
        for (id, window_id, mut window_state) in &mut self.window_states {
            if window_id.is_primary() {
                continue;
            }
            window_state.tile_state.clear(&mut self.commands);
            self.commands.entity(id).despawn();
        }

        let primary_window = *self.primary_window;
        let mut main_state = {
            let mut window_state = self
                .window_states
                .get_mut(primary_window)
                .expect("no primary window")
                .2;
            std::mem::take(&mut window_state.tile_state)
        };
        main_state.clear(&mut self.commands);
        self.hdr_enabled.0 = false;
        for entity in self.schematic_spawned.iter() {
            self.commands.entity(entity).despawn();
        }
        let theme_selection = apply_theme(schematic.theme.as_ref());
        let theme_mode = Some(theme_selection.mode.clone());
        let theme_mode_str = theme_mode.as_deref();
        let mut descriptors = collect_window_descriptors(schematic, base_dir, theme_mode_str);
        let timeline = schematic.timeline.clone().unwrap_or_default();
        *self.timeline_settings = timeline.clone().into();
        *self.time_range_behavior = timeline
            .range
            .as_deref()
            .and_then(TimeRangeBehavior::from_schematic_range)
            .unwrap_or_default();
        *self.telemetry_mode = TelemetryMode(schematic.telemetry_mode);

        let panel_count = schematic
            .elems
            .iter()
            .filter(|elem| matches!(elem, impeller2_wkt::SchematicElem::Panel(_)))
            .count();
        let tabs_parent = tabs_parent_for_panels(&main_state, panel_count);
        let mut main_ui_state = crate::ui::WindowUiState::default();

        let fallback_frame = self.coordinate.0;
        let force_graph_lock = schematic.telemetry_mode;
        for elem in &schematic.elems {
            match elem {
                impeller2_wkt::SchematicElem::Panel(p) => {
                    let p = apply_fallback_frame_to_panel(p, fallback_frame);
                    self.spawn_panel(
                        &mut main_state,
                        &mut main_ui_state,
                        &p,
                        tabs_parent,
                        PanelContext::Main,
                        force_graph_lock,
                    );
                }
                impeller2_wkt::SchematicElem::Object3d(object_3d) => {
                    let mut obj = object_3d.clone();
                    if obj.frame.is_none() {
                        obj.frame = fallback_frame;
                    }
                    self.spawn_object_3d(obj);
                }
                impeller2_wkt::SchematicElem::Line3d(line_3d) => {
                    let mut line = line_3d.clone();
                    if line.frame.is_none() {
                        line.frame = fallback_frame;
                    }
                    self.spawn_line_3d(line);
                }
                impeller2_wkt::SchematicElem::VectorArrow(vector_arrow) => {
                    let mut arrow = vector_arrow.clone();
                    if arrow.frame.is_none() {
                        arrow.frame = fallback_frame;
                    }
                    self.spawn_vector_arrow(arrow, None);
                }
                impeller2_wkt::SchematicElem::WorldMesh(world_mesh) => {
                    let mut wm = world_mesh.clone();
                    if wm.frame.is_none() {
                        wm.frame = fallback_frame;
                    }
                    self.spawn_world_mesh(wm);
                }
                impeller2_wkt::SchematicElem::Window(_) => {}
                impeller2_wkt::SchematicElem::Theme(_) => {}
                impeller2_wkt::SchematicElem::Timeline(_) => {}
                impeller2_wkt::SchematicElem::Coordinate(_) => {}
            }
        }

        {
            let mut window_state = self
                .window_states
                .get_mut(*self.primary_window)
                .expect("no primary window")
                .2;
            match descriptors.main.take() {
                Some(d) => {
                    window_state.descriptor.screen = d.screen;
                    window_state.descriptor.screen_rect = d.screen_rect;
                }
                None => {
                    // No primary window entry in KDL: clear any stale placement to avoid
                    // reapplying an old rect.
                    window_state.descriptor.screen = None;
                    window_state.descriptor.screen_rect = None;
                }
            }
            let _ = std::mem::replace(&mut window_state.tile_state, main_state);
            // Apply sidebar visibility from loaded schematic.
            window_state.ui_state.left_sidebar_visible = main_ui_state.left_sidebar_visible;
            window_state.ui_state.right_sidebar_visible = main_ui_state.right_sidebar_visible;
            if schematic.telemetry_mode {
                strip_singleton_graph_tabs(&mut window_state.tile_state);
            }
            // Resize if necessary.
            //
            #[cfg(not(target_os = "macos"))]
            {
                if let Some(screen) = window_state.descriptor.screen.as_ref() {
                    self.commands.write_message(WindowRelayout::Screen {
                        window: primary_window,
                        screen: *screen,
                    });
                }

                if let Some(rect) = window_state.descriptor.screen_rect.as_ref() {
                    self.commands.write_message(WindowRelayout::Rect {
                        window: primary_window,
                        rect: *rect,
                    });
                }
            }
            #[cfg(target_os = "macos")]
            {
                if let (Some(screen_idx), Some(rect)) = (
                    window_state.descriptor.screen,
                    window_state.descriptor.screen_rect,
                ) {
                    self.commands.spawn_task(move || async move {
                        apply_physical_screen_rect(primary_window, screen_idx, rect)
                            .await
                            .ok();
                        Ok(())
                    });
                } else {
                    info!("mac primary: no screen/rect in KDL, skipping placement");
                }
            }
        }

        if let Some(mode) = theme_mode.as_ref() {
            for (_entity, _window_id, mut window_state) in self.window_states.iter_mut() {
                window_state.descriptor.mode = Some(mode.clone());
            }
        }

        let mut loaded_windows = window_assets.unwrap_or(&[]).iter();
        for descriptor in descriptors.windows {
            if let Some(window) = loaded_windows.next()
                && let Some(root) = self
                    .document_assets
                    .get(&window.handle)
                    .map(|d| d.root.clone())
            {
                let asset_path = window.asset_path.path().to_path_buf();
                self.spawn_window(
                    &root,
                    descriptor,
                    theme_mode.as_deref(),
                    &theme_selection.scheme,
                    Some(&asset_path),
                );
                continue;
            }

            if let Some(path) = descriptor.path.clone() {
                let connection_addr = self.connection_addr.as_ref().map(|addr| addr.0);
                if path.to_str().is_some_and(is_remote_asset_path) {
                    // Fetch `db:`/HTTP windows off the main thread: a blocking
                    // request per window could otherwise stall the UI for the
                    // full timeout each (RFD #724). The window spawns from
                    // `apply_pending_window_schematics` once the fetch lands.
                    let fetch_path = path.clone();
                    let task = IoTaskPool::get().spawn(async move {
                        read_window_schematic_kdl(&fetch_path, connection_addr)
                    });
                    self.pending_windows.loads.push(PendingWindowLoad {
                        descriptor,
                        theme_mode: theme_mode.clone(),
                        theme_scheme: theme_selection.scheme.clone(),
                        path,
                        task: Some(task),
                        attempts: 0,
                        retry_at: None,
                    });
                    continue;
                }
                match read_window_schematic_kdl(&path, connection_addr) {
                    Ok(kdl) => match impeller2_wkt::Schematic::from_kdl(&kdl) {
                        Ok(window_schematic) => {
                            self.spawn_window(
                                &window_schematic,
                                descriptor,
                                theme_mode.as_deref(),
                                &theme_selection.scheme,
                                Some(path.as_path()),
                            );
                        }
                        Err(err) => {
                            let diag = render_diag(&err);
                            let report = miette!(err.clone());
                            warn!(
                                ?report,
                                path = ?descriptor.path,
                                "Failed to parse window schematic: \n{diag}"
                            );
                        }
                    },
                    Err(err) => {
                        warn!(
                            %err,
                            path = ?descriptor.path,
                            "Failed to read window schematic"
                        );
                    }
                }
            }
        }

        self.current_schematic.0.skybox = schematic.skybox.clone();
    }

    /// Spawns windows whose remote sub-schematic fetch (queued by
    /// `load_schematic`) has completed, draining the in-flight queue. Cheap when
    /// nothing is pending; called every frame by `apply_pending_window_schematics`.
    ///
    /// Fetch and parse failures are retried bounded (Bug 4): a window's asset
    /// can 404 or hold torn bytes transiently — a follower still mirroring it,
    /// or a save's `PUT`s mid-flight — and a single silent failure would make
    /// the window vanish for the rest of the session.
    fn poll_pending_window_schematics(&mut self) {
        if self.pending_windows.loads.is_empty() {
            return;
        }
        let connection_addr = self.connection_addr.as_ref().map(|addr| addr.0);
        let now = Instant::now();
        let mut ready = Vec::new();
        self.pending_windows.loads.retain_mut(|load| {
            if load.task.is_none() {
                if load.retry_at.is_some_and(|at| now < at) {
                    return true;
                }
                let fetch_path = load.path.clone();
                load.retry_at = None;
                load.task =
                    Some(IoTaskPool::get().spawn(async move {
                        read_window_schematic_kdl(&fetch_path, connection_addr)
                    }));
            }
            let task = load.task.as_mut().expect("window fetch task just spawned");
            let Some(result) = future::block_on(future::poll_once(task)) else {
                return true;
            };
            load.task = None;
            // Parse here so torn bytes share the retry path with fetch errors.
            let parsed = result.and_then(|kdl| {
                impeller2_wkt::Schematic::from_kdl(&kdl)
                    .map(|schematic| (kdl, schematic))
                    .map_err(|err| render_diag(&err))
            });
            match parsed {
                Ok((kdl, schematic)) => {
                    ready.push((
                        std::mem::take(&mut load.descriptor),
                        load.theme_mode.take(),
                        std::mem::take(&mut load.theme_scheme),
                        std::mem::take(&mut load.path),
                        kdl,
                        schematic,
                    ));
                    false
                }
                Err(err) => {
                    load.attempts += 1;
                    if load.attempts >= MAX_WINDOW_FETCH_ATTEMPTS {
                        warn!(
                            %err,
                            path = ?load.path,
                            "Failed to load window schematic; giving up"
                        );
                        false
                    } else {
                        debug!(
                            %err,
                            path = ?load.path,
                            attempts = load.attempts,
                            "Window schematic fetch failed; retrying"
                        );
                        load.retry_at = Some(now + WINDOW_FETCH_RETRY_DELAY);
                        true
                    }
                }
            }
        });
        for (descriptor, theme_mode, theme_scheme, path, kdl, window_schematic) in ready {
            // Record the stored content so a revision-gated refetch can detect
            // a later remote save that only touched this window (Bug 2).
            if let (Some(key), Some(last_content)) = (
                path.to_str().and_then(|p| p.strip_prefix("db:")),
                self.last_content.as_deref_mut(),
            ) {
                last_content.record_window(key, &kdl);
            }
            self.spawn_window(
                &window_schematic,
                descriptor,
                theme_mode.as_deref(),
                &theme_scheme,
                Some(path.as_path()),
            );
        }
    }

    fn spawn_window(
        &mut self,
        window_schematic: &Schematic,
        descriptor: WindowDescriptor,
        fallback_theme_mode: Option<&str>,
        theme_scheme: &str,
        log_path: Option<&Path>,
    ) {
        let id = WindowId::default();
        let state = self.build_window_state(
            id,
            window_schematic,
            descriptor,
            fallback_theme_mode,
            theme_scheme,
        );
        info!(path = ?log_path, "Loaded window schematic");
        self.commands.spawn((id, state));
    }

    fn build_window_state(
        &mut self,
        id: WindowId,
        sec_schematic: &Schematic,
        descriptor: WindowDescriptor,
        fallback_theme_mode: Option<&str>,
        theme_scheme: &str,
    ) -> WindowState {
        let theme_mode = sec_schematic
            .theme
            .as_ref()
            .and_then(|t| t.mode.as_deref())
            .or(fallback_theme_mode);
        let resolved_mode = theme_mode.map(|mode| {
            if colors::scheme_supports_mode(theme_scheme, mode) {
                mode.to_string()
            } else {
                "dark".to_string()
            }
        });
        let mut tile_state = TileState::new(Id::new(("window_tab_tree", id.0)));
        let panel_count = sec_schematic
            .elems
            .iter()
            .filter(|elem| matches!(elem, impeller2_wkt::SchematicElem::Panel(_)))
            .count();
        let tabs_parent = tabs_parent_for_panels(&tile_state, panel_count);
        let mut ui_state = crate::ui::WindowUiState::default();

        for elem in &sec_schematic.elems {
            if let impeller2_wkt::SchematicElem::Panel(panel) = elem {
                self.spawn_panel(
                    &mut tile_state,
                    &mut ui_state,
                    panel,
                    tabs_parent,
                    PanelContext::Window(id),
                    false,
                );
            }
        }

        let graph_entities = tile_state.collect_graph_entities();
        for &graph in &graph_entities {
            if let Ok(mut camera) = self.cameras.get_mut(graph) {
                // Window graph cameras are activated after their WindowRef is assigned.
                camera.is_active = false;
            }
        }

        WindowState {
            descriptor: WindowDescriptor {
                mode: resolved_mode,
                ..descriptor
            },
            tile_state,
            graph_entities,
            ui_state,
        }
    }

    fn reload_changed_windows(
        &mut self,
        document: &SchematicDocumentAsset,
        base_dir: Option<&Path>,
        changed_window_indices: &[usize],
    ) -> bool {
        if changed_window_indices.is_empty() {
            return true;
        }

        let current_theme = colors::current_selection();
        let fallback_theme_mode = document
            .root
            .theme
            .as_ref()
            .and_then(|theme| theme.mode.as_deref())
            .unwrap_or(&current_theme.mode);
        let descriptors =
            collect_window_descriptors(&document.root, base_dir, Some(fallback_theme_mode)).windows;
        if descriptors.len() != document.windows.len() {
            return false;
        }

        let mut windows_by_path = HashMap::new();
        for (entity, window_id, state) in &mut self.window_states {
            if window_id.is_primary() {
                continue;
            }
            let Some(path) = state.descriptor.path.clone() else {
                return false;
            };
            if windows_by_path.insert(path, entity).is_some() {
                return false;
            }
        }

        for &index in changed_window_indices {
            let Some(descriptor) = descriptors.get(index).cloned() else {
                return false;
            };
            let Some(path) = descriptor.path.clone() else {
                return false;
            };
            let Some(entity) = windows_by_path.get(&path).copied() else {
                return false;
            };
            let Some(window) = document.windows.get(index) else {
                return false;
            };
            let Some(root) = self
                .document_assets
                .get(&window.handle)
                .map(|d| d.root.clone())
            else {
                return false;
            };
            let window_id = {
                let Ok((_, window_id, mut window_state)) = self.window_states.get_mut(entity)
                else {
                    return false;
                };
                let window_id = *window_id;
                window_state.tile_state.clear(&mut self.commands);
                window_state.graph_entities.clear();
                window_id
            };
            let new_state = self.build_window_state(
                window_id,
                &root,
                descriptor,
                Some(fallback_theme_mode),
                &current_theme.scheme,
            );
            let Ok((_, _, mut window_state)) = self.window_states.get_mut(entity) else {
                return false;
            };
            *window_state = new_state;
        }

        true
    }

    pub fn spawn_object_3d(&mut self, object_3d: Object3D) {
        let Ok(expr) = self.eql.0.parse_str(&object_3d.eql) else {
            return;
        };
        let icon = object_3d.icon.clone();
        let mesh_vr = object_3d.mesh_visibility_range.clone();
        let connection_addr = self.connection_addr.as_ref().map(|addr| addr.0);
        let result = crate::object_3d::create_object_3d_entity(
            &mut self.commands,
            object_3d.clone(),
            expr,
            &self.eql.0,
            &mut self.materials,
            &mut self.meshes,
            &mut self.mat3_materials,
            &self.asset_server,
            &self.geo_context,
            connection_addr,
        );
        match result {
            Ok(entity) => {
                {
                    let mut e = self.commands.entity(entity);
                    e.insert(SchematicSpawned);
                    #[cfg(feature = "big_space")]
                    crate::spatial::parent_under_big_space(&mut e, self.big_space_root.as_deref());
                }
                if let Some(icon) = &icon {
                    crate::object_3d::spawn_billboard_icon(
                        &mut self.commands,
                        entity,
                        icon,
                        mesh_vr.as_ref(),
                        &mut self.materials,
                        &mut self.meshes,
                        &mut self.images,
                        &self.asset_server,
                        &mut self.icon_cache,
                        connection_addr,
                    );
                }
            }
            Err(err) => {
                warn!("Unable to spawn object 3d due to eql compile error: {err}");
            }
        }
    }

    pub fn spawn_line_3d(&mut self, line_3d: Line3d) {
        let frame = line_3d.frame.or_default().unwrap_or_default();
        let mut spawn = self.commands.spawn(line_3d);
        spawn.insert((
            Name::new("line_3d"),
            Transform::default(),
            GlobalTransform::default(),
            // Absolute: vertex data is frame-relative; GeoRotation carries
            // the frame → Bevy basis. GeoPosition tracks the LineTree's first
            // sample (visible-window anchor; see sync_line_3d_anchor).
            bevy_geo_frames::GeoPosition(frame, bevy::math::DVec3::ZERO),
            bevy_geo_frames::GeoRotation::absolute(frame, bevy::math::DQuat::IDENTITY),
            #[cfg(feature = "big_space")]
            crate::spatial::GridCell::default(),
        ));
        #[cfg(feature = "big_space")]
        crate::spatial::parent_under_big_space(&mut spawn, self.big_space_root.as_deref());
        spawn.insert(SchematicSpawned);
    }

    pub fn spawn_vector_arrow(
        &mut self,
        vector_arrow: VectorArrow3d,
        viewport_camera: Option<Entity>,
    ) {
        use crate::object_3d::compile_eql_expr;

        let vector_expr = self
            .eql
            .0
            .parse_str(&vector_arrow.vector)
            .map_err(CompileError::Parse)
            .and_then(compile_eql_expr)
            .ok();

        let origin_expr = vector_arrow.origin.as_ref().and_then(|origin| {
            self.eql
                .0
                .parse_str(origin)
                .map_err(CompileError::Parse)
                .and_then(compile_eql_expr)
                .ok()
        });

        // The arrow's frame is carried by VectorArrow3d and applied to its
        // endpoint entities in `evaluate_vector_arrows`.
        let mut spawn = self.commands.spawn((
            vector_arrow,
            VectorArrowState {
                vector_expr,
                origin_expr,
                visuals: HashMap::new(),
                label: None,
                ..default()
            },
            SchematicSpawned,
        ));

        if let Some(camera) = viewport_camera {
            spawn.insert(ViewportArrow { camera });
        }
    }

    pub fn spawn_world_mesh(&mut self, world_mesh: impeller2_wkt::WorldMesh) {
        let entity = crate::plugins::world_mesh::spawn_world_mesh_terrain(
            &mut self.commands,
            &mut self.meshes,
            &mut self.materials,
            &mut self.world_mesh_materials,
            &world_mesh,
        );
        let mut e = self.commands.entity(entity);
        e.insert((SchematicSpawned, world_mesh));
        #[cfg(feature = "big_space")]
        crate::spatial::parent_under_big_space(&mut e, self.big_space_root.as_deref());
    }

    fn spawn_panel(
        &mut self,
        tile_state: &mut TileState,
        ui_state: &mut crate::ui::WindowUiState,
        panel: &Panel,
        parent_id: Option<TileId>,
        context: PanelContext,
        force_graph_lock: bool,
    ) -> Option<TileId> {
        match panel {
            Panel::Viewport(viewport) => {
                let label = viewport_label(viewport);
                let pane = ViewportPane::spawn(
                    &mut self.commands,
                    &self.asset_server,
                    &mut self.meshes,
                    &mut self.materials,
                    &mut self.render_layer_alloc,
                    &self.eql.0,
                    viewport,
                    label,
                );
                self.hdr_enabled.0 |= viewport.hdr;
                if let Some(camera) = pane.camera {
                    for arrow in viewport.local_arrows.clone() {
                        self.spawn_vector_arrow(arrow, Some(camera));
                    }
                }
                if let Some(parent) = pane.parent {
                    let mut e = self.commands.entity(parent);
                    e.insert(SchematicSpawned);
                    #[cfg(feature = "big_space")]
                    crate::spatial::parent_under_big_space(&mut e, self.big_space_root.as_deref());
                }
                if let Some(grid) = pane.grid {
                    self.commands.entity(grid).insert(SchematicSpawned);
                }
                tile_state.insert_tile(Tile::Pane(Pane::Viewport(pane)), parent_id, viewport.active)
            }
            Panel::HSplit(split) | Panel::VSplit(split) => {
                let linear = egui_tiles::Linear::new(
                    match panel {
                        Panel::HSplit(_) => egui_tiles::LinearDir::Horizontal,
                        Panel::VSplit(_) => egui_tiles::LinearDir::Vertical,
                        _ => unreachable!(),
                    },
                    vec![],
                );
                let tile_id = tile_state.insert_tile(
                    Tile::Container(Container::Linear(linear)),
                    parent_id,
                    false,
                );
                if let (Some(tile_id), Some(name)) = (tile_id, split.name.clone()) {
                    tile_state.container_titles.insert(tile_id, name);
                }
                for (i, panel) in split.panels.iter().enumerate() {
                    let child_id = self.spawn_panel(
                        tile_state,
                        ui_state,
                        panel,
                        tile_id,
                        context,
                        force_graph_lock,
                    );
                    let Some(tile_id) = tile_id else {
                        continue;
                    };

                    let Some(child_id) = child_id else {
                        continue;
                    };
                    let Some(share) = split.shares.get(&i) else {
                        continue;
                    };
                    let Some(Tile::Container(Container::Linear(linear))) =
                        tile_state.tree.tiles.get_mut(tile_id)
                    else {
                        continue;
                    };
                    linear.shares.set_share(child_id, *share);
                }
                tile_id
            }
            Panel::Tabs(tabs) => {
                let mut tile_id = None;
                let mut parent_for_children = parent_id;
                let mut reuse_parent = false;

                if matches!(context, PanelContext::Window(_))
                    && let Some(parent_id) = parent_id
                    && tile_state.tree.root() == Some(parent_id)
                    && let Some(Tile::Container(Container::Tabs(root_tabs))) =
                        tile_state.tree.tiles.get(parent_id)
                {
                    // Reuse parent if root tabs are empty
                    if root_tabs.children.is_empty() {
                        reuse_parent = true;
                        tile_id = Some(parent_id);
                        parent_for_children = Some(parent_id);
                    }
                }

                if !reuse_parent {
                    tile_id = tile_state.insert_tile(
                        Tile::Container(Container::new_tabs(vec![])),
                        parent_id,
                        false,
                    );
                    parent_for_children = tile_id;
                }

                for panel in tabs.iter() {
                    self.spawn_panel(
                        tile_state,
                        ui_state,
                        panel,
                        parent_for_children,
                        context,
                        force_graph_lock,
                    );
                }
                tile_id
            }
            Panel::Graph(graph) => {
                let eql = self
                    .eql
                    .0
                    .parse_str(&graph.eql)
                    .inspect_err(|err| {
                        let (ctx, path) = match context {
                            PanelContext::Main => ("main".to_string(), None),
                            PanelContext::Window(target_id) => {
                                let path = self
                                    .window_states
                                    .iter()
                                    .find(|(_entity, id, _state)| **id == target_id)
                                    .and_then(|(_, _, s)| {
                                        s.descriptor.path.as_ref().map(|p| p.display().to_string())
                                    });
                                (format!("window({})", target_id.0), path)
                            }
                        };
                        if let Some(p) = path {
                            warn!(
                                ?err,
                                eql = %graph.eql,
                                name = ?graph.name,
                                context = %ctx,
                                path = %p,
                                "error parsing graph eql"
                            );
                        } else {
                            warn!(
                                ?err,
                                eql = %graph.eql,
                                name = ?graph.name,
                                context = %ctx,
                                "error parsing graph eql"
                            );
                        }
                    })
                    .ok()?;
                let mut component_vec = eql.to_graph_components();
                component_vec.sort();
                let mut components_tree: BTreeMap<ComponentPath, Vec<(bool, Color32)>> =
                    BTreeMap::new();
                for (j, (component, i)) in component_vec.iter().enumerate() {
                    let line_color = graph
                        .colors
                        .get(j)
                        .copied()
                        .map(EColor::into_color32)
                        .unwrap_or_else(|| colors::get_color_by_index_all(j));
                    if let Some(elements) = components_tree.get_mut(component) {
                        elements[*i] = (true, line_color);
                    } else {
                        let Some(schema) = self.schema_reg.0.get(&component.id) else {
                            continue;
                        };
                        let len: usize = schema.shape().iter().copied().product();
                        let mut elements: Vec<(bool, Color32)> =
                            (0..len).map(|_| (false, line_color)).collect();
                        elements[*i] = (true, line_color);
                        components_tree.insert(component.clone(), elements);
                    }
                }

                let graph_label = graph_label(graph);

                let mut bundle = GraphBundle::new(
                    &mut self.render_layer_alloc,
                    components_tree,
                    graph_label.clone(),
                );
                bundle.graph_state.locked = force_graph_lock || graph.locked;
                if matches!(context, PanelContext::Window(_)) {
                    bundle.camera.is_active = false;
                }
                bundle.graph_state.auto_y_range = graph.auto_y_range;
                bundle.graph_state.y_range = graph.y_range.clone();
                bundle.graph_state.graph_type = graph.graph_type;
                let graph_id = self.commands.spawn(bundle).id();
                if matches!(context, PanelContext::Window(_)) {
                    self.commands.entity(graph_id).remove::<MainCamera>();
                }
                let graph = GraphPane::new(graph_id, graph_label);
                tile_state.insert_tile(Tile::Pane(Pane::Graph(graph)), parent_id, false)
            }
            Panel::ComponentMonitor(monitor) => {
                let label = monitor
                    .name
                    .clone()
                    .unwrap_or_else(|| monitor.component_name.clone());
                let entity = self
                    .commands
                    .spawn((
                        super::monitor::MonitorData {
                            component_name: monitor.component_name.clone(),
                        },
                        Name::new(label.clone()),
                        ChildOf(*self.monitor_root),
                    ))
                    .id();
                let pane = MonitorPane::new(entity, label);
                tile_state.insert_tile(Tile::Pane(Pane::Monitor(pane)), parent_id, false)
            }
            Panel::QueryTable(data) => {
                let has_query = !data.query.trim().is_empty();
                let entity = self
                    .commands
                    .spawn(super::query_table::QueryTableData {
                        data: data.clone(),
                        pending_execution: has_query,
                        ..Default::default()
                    })
                    .id();
                let label = data
                    .name
                    .clone()
                    .filter(|name| !name.trim().is_empty())
                    .unwrap_or_else(|| "Query Table".to_string());
                let pane = super::query_table::QueryTablePane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::QueryTable(pane)), parent_id, false)
            }
            Panel::ActionPane(action) => {
                let entity = self
                    .commands
                    .spawn(super::actions::ActionTile {
                        button_name: action.name.clone(),
                        lua: action.lua.clone(),
                        status: Default::default(),
                    })
                    .id();
                let pane = super::tiles::ActionTilePane {
                    entity,
                    name: action.name.clone(),
                };
                tile_state.insert_tile(Tile::Pane(Pane::ActionTile(pane)), parent_id, false)
            }
            Panel::VideoStream(video_stream) => {
                let msg_id = impeller2::types::msg_id(&video_stream.msg_name);
                let label = video_stream
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Video Stream {}", video_stream.msg_name));

                let entity = self
                    .commands
                    .spawn((
                        crate::ui::video_stream::VideoStream {
                            msg_id,
                            msg_name: video_stream.msg_name.clone(),
                            ..Default::default()
                        },
                        bevy::ui::Node {
                            position_type: bevy::ui::PositionType::Absolute,
                            ..Default::default()
                        },
                        bevy::prelude::ImageNode {
                            image_mode: bevy::ui::widget::NodeImageMode::Stretch,
                            ..Default::default()
                        },
                        crate::ui::video_stream::VideoDecoderHandle::default(),
                        crate::ui::video_stream::VideoFrameCache::default(),
                    ))
                    .id();

                let pane = crate::ui::video_stream::VideoStreamPane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::VideoStream(pane)), parent_id, false)
            }
            Panel::SensorView(sensor_view) => {
                let msg_id = impeller2::types::msg_id(&sensor_view.msg_name);
                let label = sensor_view
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Sensor View {}", sensor_view.msg_name));

                let raw_rgba_dims = self
                    .sensor_camera_configs
                    .0
                    .iter()
                    .find(|c| c.camera_name == sensor_view.msg_name)
                    .map(|c| (c.width, c.height));

                let entity = self
                    .commands
                    .spawn((
                        crate::ui::video_stream::VideoStream {
                            msg_id,
                            msg_name: sensor_view.msg_name.clone(),
                            raw_rgba_dims,
                            ..Default::default()
                        },
                        bevy::ui::Node {
                            position_type: bevy::ui::PositionType::Absolute,
                            ..Default::default()
                        },
                        bevy::prelude::ImageNode {
                            image_mode: bevy::ui::widget::NodeImageMode::Stretch,
                            ..Default::default()
                        },
                        crate::ui::video_stream::VideoDecoderHandle::default(),
                        crate::ui::video_stream::VideoFrameCache::for_raw_rgba(),
                    ))
                    .id();

                let pane = crate::ui::video_stream::VideoStreamPane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::SensorView(pane)), parent_id, false)
            }
            Panel::LogStream(log_stream) => {
                let msg_id = impeller2::types::msg_id(&log_stream.msg_name);
                let label = log_stream
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Log Stream {}", log_stream.msg_name));

                let entity = self
                    .commands
                    .spawn((
                        crate::ui::log_stream::LogStreamState {
                            msg_id,
                            msg_name: log_stream.msg_name.clone(),
                            ..Default::default()
                        },
                        crate::ui::log_stream::LogCache::default(),
                    ))
                    .id();

                let pane = crate::ui::log_stream::LogStreamPane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::LogStream(pane)), parent_id, false)
            }
            // Inspector and Hierarchy are now fixed sidebars, not tile panels.
            // Set the corresponding sidebar visibility flags so they appear.
            Panel::Inspector => {
                ui_state.right_sidebar_visible = true;
                None
            }
            Panel::Hierarchy => {
                ui_state.left_sidebar_visible = true;
                None
            }
            Panel::SchematicTree(name) => {
                let entity = self.commands.spawn(super::TreeWidgetState::default()).id();
                let label = name.clone().unwrap_or_else(|| "Tree".to_string());
                let pane = TreePane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::SchematicTree(pane)), parent_id, false)
            }
            Panel::DataOverview(name) => {
                let mut pane = DataOverviewPane::default();
                if let Some(name) = name.clone() {
                    pane.name = name;
                }
                let pane = Pane::DataOverview(pane);
                tile_state.insert_tile(Tile::Pane(pane), parent_id, false)
            }
            Panel::QueryPlot(plot) => {
                let graph_bundle = GraphBundle::new(
                    &mut self.render_layer_alloc,
                    BTreeMap::default(),
                    plot.name.clone(),
                );
                let auto_color = plot.color.into_color32() == colors::get_scheme().highlight;
                let entity = self
                    .commands
                    .spawn(QueryPlotData {
                        data: plot.clone(),
                        auto_color,
                        last_refresh: None,
                        ..Default::default()
                    })
                    .insert(graph_bundle)
                    .id();
                let pane = Pane::QueryPlot(super::query_plot::QueryPlotPane {
                    entity,
                    rect: None,
                    scrub_icon: None,
                });
                tile_state.insert_tile(Tile::Pane(pane), parent_id, false)
            }
        }
    }

    /// Create a default Data Overview panel when no schematic is found.
    /// This provides immediate visibility into the database contents.
    pub fn load_default_data_overview(&mut self) -> bool {
        if self.eql.0.component_parts.is_empty() {
            return false;
        }

        let primary_window = *self.primary_window;
        let mut window_state = self
            .window_states
            .get_mut(primary_window)
            .expect("no primary window")
            .2;

        // Only add if the tile tree is empty (no non-sidebar content)
        if !window_state.tile_state.is_empty() {
            return true;
        }

        let target_id = {
            let tree = &window_state.tile_state.tree;
            tree.root()
                .and_then(|root_id| match tree.tiles.get(root_id) {
                    Some(Tile::Container(Container::Linear(linear))) => {
                        let center_idx = linear.children.len() / 2;
                        linear.children.get(center_idx).copied()
                    }
                    _ => Some(root_id),
                })
        };

        let mut central_tabs_id = None;
        if let Some(target_id) = target_id {
            match window_state.tile_state.tree.tiles.get(target_id) {
                Some(Tile::Container(Container::Tabs(_))) => central_tabs_id = Some(target_id),
                Some(Tile::Container(_)) => {
                    let tabs_container = Tile::Container(Container::new_tabs(vec![]));
                    central_tabs_id =
                        window_state
                            .tile_state
                            .insert_tile(tabs_container, Some(target_id), false);
                }
                _ => {}
            }
        }

        if central_tabs_id.is_none() {
            let tabs_container = Tile::Container(Container::new_tabs(vec![]));
            central_tabs_id = window_state
                .tile_state
                .insert_tile(tabs_container, None, false);
        }

        // The central tabs container is effectively invisible; nest a new Tabs container
        // so the tab bar (+) is visible, then add Data Overview inside it.
        if let Some(central_tabs_id) = central_tabs_id {
            let tabs_container = Tile::Container(Container::new_tabs(vec![]));
            if let Some(tabs_id) =
                window_state
                    .tile_state
                    .insert_tile(tabs_container, Some(central_tabs_id), false)
            {
                let pane = Pane::DataOverview(DataOverviewPane::default());
                if let Some(tile_id) =
                    window_state
                        .tile_state
                        .insert_tile(Tile::Pane(pane), Some(tabs_id), true)
                {
                    window_state
                        .tile_state
                        .tree
                        .make_active(|id, _| id == tile_id);
                }
            }
        }
        true
    }
}

pub fn viewport_label(viewport: &Viewport) -> String {
    viewport
        .name
        .clone()
        .unwrap_or_else(|| "Viewport".to_string())
}

/// Prefer the explicit `name` when set (and not the generic "Graph").
/// Otherwise, derive a readable label from the first EQL term.
pub fn graph_label(graph: &Graph) -> String {
    if let Some(name) = graph.name.as_ref() {
        let trimmed = name.trim();
        if !trimmed.is_empty() && trimmed != "Graph" {
            return trimmed.to_string();
        }
    }
    graph
        .eql
        .split(',')
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Graph".to_string())
}

pub fn apply_document_reloaded(
    mut events: MessageReader<DocumentReloaded>,
    mut params: LoadSchematicParams,
) {
    let Some(event) = events.read().last().cloned() else {
        return;
    };
    let save_path = event.save_path;
    let document = event.document;

    if !event.changed_window_indices.is_empty()
        && params.reload_changed_windows(
            &document,
            save_path.as_deref().and_then(Path::parent),
            &event.changed_window_indices,
        )
    {
        return;
    }

    apply_loaded_document(&mut params, save_path.as_deref(), &document);
}

pub fn apply_document_loaded(
    mut events: MessageReader<DocumentLoaded>,
    mut params: LoadSchematicParams,
    mut pending: ResMut<PendingDataOverview>,
) {
    let Some(event) = events.read().last().cloned() else {
        return;
    };
    pending.0 = false;
    apply_loaded_document(&mut params, event.save_path.as_deref(), &event.document);
}

/// Spawns remote window sub-schematics once their off-thread fetch completes
/// (RFD #724). Runs each frame and early-returns when nothing is pending.
pub fn apply_pending_window_schematics(mut params: LoadSchematicParams) {
    params.poll_pending_window_schematics();
}

/// Pending Data Overview load after DocumentCleared raced ahead of EQL metadata.
#[derive(Resource, Default)]
pub struct PendingDataOverview(pub bool);

pub fn apply_document_cleared(
    mut events: MessageReader<DocumentCleared>,
    mut params: LoadSchematicParams,
    mut pending: ResMut<PendingDataOverview>,
) {
    if events.read().next().is_none() {
        return;
    }
    if !params.load_default_data_overview() {
        pending.0 = true;
    }
}

/// Finish a pending Data Overview once component metadata is available.
pub fn retry_pending_data_overview(
    mut pending: ResMut<PendingDataOverview>,
    mut params: LoadSchematicParams,
) {
    if !pending.0 {
        return;
    }
    if params.load_default_data_overview() {
        pending.0 = false;
    }
}

pub fn show_document_command_failures(
    mut errors: MessageReader<DocumentCommandFailed>,
    mut modal: ModalDialog,
) {
    for error in errors.read() {
        modal.dialog_error(error.title.clone(), &error.message);
    }
}

pub fn show_document_load_failures(
    mut events: MessageReader<DocumentLoadFailed>,
    mut modal: ModalDialog,
) {
    for event in events.read() {
        modal.dialog_error(
            format!("Invalid Schematic in {}", event.path),
            &event.message,
        );
        error!(path = %event.path, error = %event.message, "Failed to load schematic document");
    }
}

#[cfg(test)]
mod tests {
    use super::LoadSchematicParams;
    use crate::ui::widgets::SystemStateExt;
    use crate::{
        Coordinate, EqlContext,
        icon_rasterizer::IconTextureCache,
        plugins::render_layer_alloc,
        sensor_camera::SensorCameraConfigs,
        ui::{
            HdrEnabled,
            schematic::{
                CurrentDocument, CurrentSchematic, SchematicBindings, SchematicDocumentAsset,
            },
            tiles,
            timeline::TimelineSettings,
        },
    };
    use bevy::{
        asset::{AssetApp, AssetPlugin, UnapprovedPathMode},
        ecs::system::SystemState,
        math::DVec3,
        prelude::*,
        window::{PrimaryWindow, Window},
    };
    use bevy_geo_frames::{GeoFrame, GeoFramePlugin, GeoPosition, GeoRotation, RotationKind};
    use bevy_mat3_material::Mat3Material;
    use impeller2_bevy::ComponentSchemaRegistry;
    use impeller2_kdl::FromKdl;
    use impeller2_wkt::Schematic;
    use test_case::test_case;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin {
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..Default::default()
        });
        app.init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Image>()
            .init_asset::<Mat3Material>()
            .init_asset::<bevy_world_mesh::prelude::WorldMeshMaterial>()
            .init_asset::<SchematicDocumentAsset>();
        app.add_plugins(GeoFramePlugin {
            apply_transforms: false,
            ..default()
        });
        render_layer_alloc::plugin(&mut app);
        tiles::plugin(&mut app);
        app.init_resource::<CurrentDocument>()
            .init_resource::<IconTextureCache>()
            .init_resource::<HdrEnabled>()
            .init_resource::<TimelineSettings>()
            .init_resource::<crate::TimeRangeBehavior>()
            .init_resource::<crate::ui::timeline::TelemetryMode>()
            .init_resource::<ComponentSchemaRegistry>()
            .init_resource::<EqlContext>()
            .init_resource::<SensorCameraConfigs>()
            .init_resource::<Coordinate>()
            .init_resource::<SchematicBindings>()
            .init_resource::<super::PendingWindowSchematics>()
            .insert_resource(CurrentSchematic(Default::default()));

        app.world_mut().spawn((Window::default(), PrimaryWindow));
        settle(&mut app);
        app
    }

    fn settle(app: &mut App) {
        for _ in 0..4 {
            app.update();
        }
    }

    fn entity_count(app: &mut App) -> usize {
        let world = app.world_mut();
        let mut query = world.query::<Entity>();
        query.iter(world).count()
    }

    fn primary_window_state(app: &mut App) -> crate::ui::tiles::WindowState {
        let primary_window = {
            let world = app.world_mut();
            let mut query = world.query_filtered::<Entity, With<PrimaryWindow>>();
            query.iter(world).next().expect("primary window")
        };
        app.world()
            .get::<crate::ui::tiles::WindowState>(primary_window)
            .expect("window state")
            .clone()
    }

    fn load_schematic(app: &mut App, schematic: &Schematic) {
        let mut system_state: SystemState<LoadSchematicParams> = SystemState::new(app.world_mut());
        let mut params = system_state.params_mut(app.world_mut());
        params.load_schematic(schematic, None, None);
        system_state.apply(app.world_mut());
        settle(app);
    }

    #[test_case("line_3d \"(0,0,0)\""; "line_3d")]
    #[test_case("vector_arrow \"(0,0,1)\"" ; "vector_arrow")]
    #[test_case("object_3d \"(0,0,0,1, 0,0,0)\" { sphere radius=1.0 { color 0 0 0 } }" ; "object_3d")]
    #[test_case("world_mesh \"death_valley\"" ; "world_mesh")]
    #[test_case("world_mesh \"globe\"" ; "world_mesh_globe")]
    #[test_case("world_mesh \"no_such_region\"" ; "world_mesh_unknown_region")]
    fn scene_roots_clear_cleanly(content: &str) {
        let mut app = test_app();
        let baseline = entity_count(&mut app);

        let schematic_text = format!(
            r#"
            {}
            "#,
            content
        );

        eprintln!("{}", schematic_text);

        let schematic = Schematic::from_kdl(&schematic_text).expect("parse test schematic");

        load_schematic(&mut app, &schematic);
        let loaded_count = entity_count(&mut app);
        assert!(
            loaded_count > baseline,
            "loading the schematic should increase the entity count"
        );

        load_schematic(&mut app, &Schematic::default());
        let cleared_count = entity_count(&mut app);
        assert_eq!(
            cleared_count, baseline,
            "clearing the schematic should restore the entity count to baseline"
        );
        let cleared_state = primary_window_state(&mut app);
        assert!(
            cleared_state.tile_state.is_empty(),
            "clearing the schematic should empty the primary window tile state"
        );
        assert!(
            !cleared_state.ui_state.left_sidebar_visible,
            "clearing the schematic should hide the hierarchy sidebar"
        );
        assert!(
            !cleared_state.ui_state.right_sidebar_visible,
            "clearing the schematic should hide the inspector sidebar"
        );
    }

    #[test_case("viewport show_view_cube=#false" => (true, false, false) ; "viewport")]
    #[test_case("graph \"1.0\" name=\"Constant\"" => (true, false, false) ; "graph")]
    #[test_case("component_monitor component_name=\"a.world_pos\"" => (true, false, false) ; "component_monitor")]
    #[test_case("action_pane name=\"Run\" lua=\"return true\"" => (true, false, false) ; "action_pane")]
    #[test_case("query_table \"from telemetry\"" => (true, false, false) ; "query_table")]
    #[test_case("query_plot name=\"Telemetry\" query=\"a.world_pos\"" => (true, false, false) ; "query_plot")]
    #[test_case("schematic_tree" => (true, false, false) ; "schematic_tree")]
    #[test_case("data_overview" => (true, false, false) ; "data_overview")]
    #[test_case("inspector" => (false, false, true) ; "inspector")]
    #[test_case("hierarchy" => (false, true, false) ; "hierarchy")]
    #[test_case("tabs { data_overview }" => (true, false, false) ; "tabs")]
    #[test_case("hsplit { data_overview share=0.5 data_overview share=0.5 }" => (true, false, false) ; "hsplit")]
    #[test_case("vsplit { data_overview share=0.5 data_overview share=0.5 }" => (true, false, false) ; "vsplit")]
    #[test_case("tabs { video_stream \"obs-camera\" }" => (true, false, false) ; "video_stream")]
    #[test_case("tabs { sensor_view \"drone.scene_cam\" }" => (true, false, false) ; "sensor_view")]
    #[test_case("tabs { log_stream \"fsw.log\" }" => (true, false, false) ; "log_stream")]
    fn panel_roots_clear_cleanly(content: &str) -> (bool, bool, bool) {
        let mut app = test_app();
        let baseline = entity_count(&mut app);

        let schematic = Schematic::from_kdl(content).expect("parse test schematic");
        load_schematic(&mut app, &schematic);

        let loaded_state = primary_window_state(&mut app);
        let loaded = (
            !loaded_state.tile_state.is_empty(),
            loaded_state.ui_state.left_sidebar_visible,
            loaded_state.ui_state.right_sidebar_visible,
        );

        load_schematic(&mut app, &Schematic::default());

        let cleared_state = primary_window_state(&mut app);
        let cleared_count = entity_count(&mut app);
        assert!(
            cleared_state.tile_state.is_empty(),
            "clearing the schematic should empty the primary window tile state"
        );
        assert_eq!(
            cleared_count, baseline,
            "clearing the schematic should restore the entity count to baseline"
        );
        assert!(
            !cleared_state.ui_state.left_sidebar_visible,
            "clearing the schematic should hide the hierarchy sidebar"
        );
        assert!(
            !cleared_state.ui_state.right_sidebar_visible,
            "clearing the schematic should hide the inspector sidebar"
        );

        loaded
    }

    #[test]
    fn remote_asset_paths_are_detected() {
        use super::is_remote_asset_path;
        assert!(is_remote_asset_path("db:schematics/window.kdl"));
        assert!(is_remote_asset_path(
            "http://127.0.0.1:2241/schematics/w.kdl"
        ));
        assert!(is_remote_asset_path("https://example.com/w.kdl"));
        assert!(!is_remote_asset_path("schematics/window.kdl"));
        assert!(!is_remote_asset_path("/abs/path/window.kdl"));
    }

    #[test]
    fn read_window_schematic_kdl_reads_local_file() {
        use super::read_window_schematic_kdl;
        let dir = std::env::temp_dir().join(format!(
            "elodin-window-kdl-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("window.kdl");
        std::fs::write(&path, "viewport name=\"W\"\n").unwrap();

        let kdl = read_window_schematic_kdl(&path, None).expect("read local window kdl");
        assert!(kdl.contains("viewport"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn coordinate_is_reset_on_clear() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl("coordinate frame=NED").expect("parse test schematic");

        load_schematic(&mut app, &schematic);
        assert_eq!(
            app.world().resource::<crate::Coordinate>().0,
            Some(GeoFrame::NED)
        );

        load_schematic(&mut app, &Schematic::default());
        assert_eq!(app.world().resource::<crate::Coordinate>().0, None);
    }

    #[test]
    fn line_3d_spawns_with_frame_and_transform_components() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl(
            r#"
            line_3d frame="ECEF" "sat.world_pos"
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);

        let mut query = app.world_mut().query::<(
            &impeller2_wkt::Line3d,
            &GeoPosition,
            &GeoRotation,
            &Transform,
            &GlobalTransform,
        )>();
        let (line, geo_pos, geo_rot, _, _) = query
            .iter(app.world())
            .next()
            .expect("line_3d should spawn");

        assert_eq!(line.frame, Some(GeoFrame::ECEF));
        assert_eq!(geo_pos.0, GeoFrame::ECEF);
        assert_eq!(geo_rot.0, GeoFrame::ECEF);
    }

    #[test]
    fn object_3d_spawns_with_absolute_orientation() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl(
            r#"
            object_3d frame="NED" orientation=absolute "(0,0,0,1, 0,0,0)" {
                sphere radius=0.2 {
                    color 0 0 0
                }
            }
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);

        let mut query = app
            .world_mut()
            .query::<(&crate::object_3d::Object3DState, &GeoRotation)>();
        let (object_3d, geo_rot) = query
            .iter(app.world())
            .next()
            .expect("object_3d should spawn");

        assert_eq!(object_3d.data.frame, Some(GeoFrame::NED));
        assert_eq!(object_3d.data.orientation, RotationKind::Absolute);
        assert_eq!(geo_rot.0, GeoFrame::NED);
        assert_eq!(geo_rot.2, RotationKind::Absolute);
    }

    #[test]
    fn object_3d_spawns_with_separate_frame_orientation() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl(
            r#"
            object_3d frame="ECEF" frame_orientation="NED" orientation=absolute "(0,0,0,1, 0,0,0)" {
                sphere radius=0.2 {
                    color 0 0 0
                }
            }
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);

        let mut query =
            app.world_mut()
                .query::<(&crate::object_3d::Object3DState, &GeoPosition, &GeoRotation)>();
        let (object_3d, geo_pos, geo_rot) = query
            .iter(app.world())
            .next()
            .expect("object_3d should spawn");

        assert_eq!(object_3d.data.frame, Some(GeoFrame::ECEF));
        assert_eq!(object_3d.data.frame_orientation, Some(GeoFrame::NED));
        assert_eq!(geo_pos.0, GeoFrame::ECEF);
        assert_eq!(geo_rot.0, GeoFrame::NED);
        assert_eq!(geo_rot.2, RotationKind::Absolute);
    }

    #[test]
    fn geo_origin_is_applied_and_reset_on_clear() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl("coordinate frame=NED lat=34.72 lon=-86.64 alt=180.0")
            .expect("parse test schematic");

        load_schematic(&mut app, &schematic);
        let origin = app.world().resource::<bevy_geo_frames::GeoContext>().origin;
        assert_eq!(origin.latitude, 34.72f64.to_radians());
        assert_eq!(origin.longitude, (-86.64f64).to_radians());
        assert_eq!(origin.altitude, 180.0);

        load_schematic(&mut app, &Schematic::default());
        let origin = app.world().resource::<bevy_geo_frames::GeoContext>().origin;
        let default_origin = bevy_geo_frames::GeoOrigin::default();
        assert_eq!(origin.latitude, default_origin.latitude);
        assert_eq!(origin.longitude, default_origin.longitude);
        assert_eq!(origin.altitude, default_origin.altitude);
    }

    #[test]
    fn world_mesh_inherits_coordinate_frame_and_translate() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl(
            r#"
            coordinate frame="NED"
            world_mesh "no_such_region" translate="(1, 2, 3)"
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);

        let mut query = app.world_mut().query_filtered::<
            (&GeoPosition, &GeoRotation),
            With<crate::plugins::world_mesh::WorldMeshTerrain>,
        >();
        let (geo_pos, geo_rot) = query.iter(app.world()).next().expect("world_mesh terrain");

        assert_eq!(geo_pos.0, GeoFrame::NED);
        assert_eq!(geo_pos.1, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(geo_rot.0, GeoFrame::NED);
    }

    #[test]
    fn timeline_is_reset_on_clear() {
        let mut app = test_app();
        let schematic =
            Schematic::from_kdl("timeline follow_latest=#true").expect("parse test schematic");

        load_schematic(&mut app, &schematic);
        let loaded = app.world().resource::<TimelineSettings>();
        assert!(loaded.follow_latest, "timeline settings should be applied");

        load_schematic(&mut app, &Schematic::default());
        let cleared = app.world().resource::<TimelineSettings>();
        assert!(
            !cleared.follow_latest,
            "clearing the schematic should restore default timeline settings"
        );
    }

    #[test]
    fn timeline_range_and_telemetry_mode_apply_and_clear() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl(
            r#"
            timeline follow_latest=#true range="last_5s"
            telemetry_mode #true
            graph "1.0" name="Constant"
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);

        let behavior = *app.world().resource::<crate::TimeRangeBehavior>();
        assert_eq!(
            behavior,
            crate::TimeRangeBehavior::from_schematic_range("last_5s").unwrap(),
            "timeline range should map to LAST_5S"
        );
        assert!(
            app.world()
                .resource::<crate::ui::timeline::TelemetryMode>()
                .0,
            "telemetry mode should be enabled"
        );

        let mut query = app.world_mut().query::<&crate::ui::plot::GraphState>();
        let graphs: Vec<_> = query.iter(app.world()).collect();
        assert!(!graphs.is_empty(), "expected a loaded graph");
        assert!(
            graphs.iter().all(|g| g.locked),
            "telemetry mode should force graph locks"
        );

        load_schematic(&mut app, &Schematic::default());
        assert_eq!(
            *app.world().resource::<crate::TimeRangeBehavior>(),
            crate::TimeRangeBehavior::default(),
            "clearing should restore default time range"
        );
        assert!(
            !app.world()
                .resource::<crate::ui::timeline::TelemetryMode>()
                .0,
            "clearing should disable telemetry mode"
        );
    }

    #[test]
    fn from_schematic_range_last_5s() {
        let last_5s = crate::TimeRangeBehavior::from_schematic_range("last_5s").unwrap();
        assert_eq!(
            crate::TimeRangeBehavior::from_schematic_range("5s"),
            Some(last_5s)
        );
        assert_eq!(last_5s.to_schematic_range().as_deref(), Some("last_5s"));
    }

    #[test]
    fn show_frustums_kdl_does_not_enable_ellipsoid_overlays_by_default() {
        let mut app = test_app();
        let schematic = Schematic::from_kdl(
            r#"
            viewport name="Frustum View" show_view_cube=#false show_frustums=#true
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);

        let mut query = app.world_mut().query::<&tiles::ViewportConfig>();
        let config = query
            .iter(app.world())
            .find(|config| config.show_frustums)
            .expect("viewport with show_frustums");

        assert!(config.show_frustums);
        assert!(
            !config.show_coverage_in_viewport,
            "show_frustums from KDL must not auto-enable coverage overlays",
        );
        assert!(
            !config.show_projection_2d,
            "show_frustums from KDL must not auto-enable projection overlays",
        );
    }

    #[test]
    fn mixed_schematic_clears_cleanly() {
        let mut app = test_app();
        let baseline = entity_count(&mut app);

        let schematic = Schematic::from_kdl(
            r#"
            viewport name="V1" show_view_cube=#false
            viewport name="V2" show_view_cube=#false
            viewport name="V3" show_view_cube=#false
            line_3d "(0,0,0)"
            line_3d "(1,1,1)"
            "#,
        )
        .expect("parse test schematic");

        load_schematic(&mut app, &schematic);
        let loaded_count = entity_count(&mut app);
        assert!(
            loaded_count > baseline,
            "loading the schematic should increase the entity count"
        );

        load_schematic(&mut app, &Schematic::default());
        let cleared_count = entity_count(&mut app);
        assert_eq!(
            cleared_count, baseline,
            "clearing the schematic should restore the entity count to baseline"
        );
    }
}
