use std::{collections::BTreeMap, net::SocketAddr, str::FromStr};

use crate::plugins::kdl_document::{
    ACTIVE_SCHEMATIC_KEY, CurrentDocument, DocumentCommandFailed, LastSyncedSchematicContent,
    PendingActiveSchematic, SchematicDocumentAsset, fetch_schematic_index, plan_db_save,
    schematic_save_key_from_name, upload_db_save_plan,
};
use crate::skybox_generation::{LocallyPushedSkyboxActive, SkyboxDocumentSyncMut};
use bevy::{
    app::Update,
    asset::{AssetServer, Assets},
    camera::visibility::Visibility,
    ecs::{
        change_detection::DetectChanges,
        query::With,
        system::{Commands, InRef, IntoSystem, Query, Res, ResMut, System},
        world::World,
    },
    log::error,
    pbr::{StandardMaterial, wireframe::WireframeConfig},
    prelude::{Entity, In, MessageWriter, Mut, Resource, Transform},
    tasks::{IoTaskPool, Task, futures_lite::future},
    window::PrimaryWindow,
};
use bevy_ai_skybox::prelude::{
    GenerateSkybox, SetActiveSkybox, SkyboxCache, SkyboxGenerationSettings, SkyboxGenerationUi,
    SkyboxResolution, SkyboxStyle,
};
use bevy_editor_cam::controller::{component::EditorCam, motion::CurrentMotion};
use bevy_geo_frames::GeoContext;
use bevy_infinite_grid::InfiniteGrid;
use egui_tiles::{Tile, TileId};
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2::types::Timestamp;
use impeller2_bevy::{
    ComponentMetadataRegistry, ComponentPathRegistry, ConnectionAddr, EntityMap, PacketTx,
};
use impeller2_kdl::ToKdl;
use impeller2_wkt::SkyboxConfig;
use impeller2_wkt::{
    ComponentPath, ComponentValue, CurrentTimestamp, DbConfig, EarliestTimestamp, IsRecording,
    LastUpdated, Material, Mesh, Object3D, SetDbConfig, SimulationTimeStep,
};
use nox::ArrayBuf;

use crate::{
    EqlContext, MainCamera, Offset, SelectedTimeRange, TimeRangeBehavior, TimeRangeError,
    plugins::render_layer_alloc::RenderLayerAllocator,
    ui::{
        FocusedWindow, HdrEnabled, Paused, colors,
        command_palette::CommandPaletteState,
        plot::{GraphBundle, graph_lines_from_component},
        schematic::{
            CurrentSchematic, CurrentWindowSchematics, LoadSchematicParams, WindowDocumentSave,
        },
        tiles::{self, set_mode_all},
        timeline::{
            AutoFollowLatestState, LatestFollow, PlaybackSpeed, StreamTickOrigin,
            timeline_slider::UITick,
        },
    },
};

pub(crate) fn plugin(app: &mut bevy::app::App) {
    app.init_resource::<PendingSchematicSaveKey>()
        .init_resource::<SchematicSaveInFlight>()
        .init_resource::<SchematicIndexCache>()
        .add_systems(Update, (poll_schematic_save, refresh_schematic_index));
}

/// Carries the target asset key from a "Save Schematic As..." prompt into the
/// shared DB-save system. `None` means a plain "Save" that overwrites whatever
/// schematic is currently active.
#[derive(Resource, Default)]
pub(crate) struct PendingSchematicSaveKey(pub Option<String>);

/// In-flight DB-native schematic save (RFD #724). The HTTP `PUT`s run off the
/// main thread on the IO task pool; `poll_schematic_save` repoints
/// `schematic.active` once they all land, so a slow DB never freezes the UI.
#[derive(Resource, Default)]
pub(crate) struct SchematicSaveInFlight {
    task: Option<Task<Result<(), String>>>,
    active_key: Option<String>,
    active_content: Option<String>,
    /// The "Save As" name consumed for this save, restored to
    /// `PendingSchematicSaveKey` if the upload fails so a retry still targets
    /// the chosen name rather than the previous active schematic.
    pending_key: Option<String>,
}

/// Cached listing of the DB's stored schematics (RFD #724). Refreshed off the
/// main thread on connect and whenever `DbConfig` changes, so "Open Schematic…"
/// presents the list without a blocking HTTP request.
#[derive(Resource, Default)]
pub(crate) struct SchematicIndexCache {
    keys: Vec<String>,
    error: Option<String>,
    loaded_once: bool,
    /// Connection the cached listing came from, so reconnecting or switching to
    /// a different DB re-lists instead of showing a previous session's assets.
    addr: Option<SocketAddr>,
    task: Option<Task<Result<Vec<String>, String>>>,
}

pub struct PalettePage {
    items: Vec<PaletteItem>,
    pub label: Option<String>,
    initialized: bool,
    pub prompt: Option<String>,
    /// Filter text active on this page when it was left for a sub-page, so it
    /// can be restored when the user navigates back (e.g. a skybox prompt).
    pub remembered_filter: Option<String>,
}

impl PalettePage {
    pub fn new(items: Vec<PaletteItem>) -> Self {
        Self {
            items,
            initialized: false,
            label: None,
            prompt: None,
            remembered_filter: None,
        }
    }

    pub fn label(mut self, label: impl ToString) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn initialize(&mut self, world: &mut World) {
        if !self.initialized {
            for item in &mut self.items {
                item.system.initialize(world);
                item.label.init(world);
            }
        }
    }

    pub fn filter(&mut self, filter: &str) -> Vec<MatchedPaletteItem<'_>> {
        let matcher = SkimMatcherV2::default();
        let mut items: Vec<_> = self
            .items
            .iter_mut()
            .filter_map(|item| {
                let label = match &item.label {
                    LabelSource::Label(l) => l.as_str(),
                    LabelSource::System(_) => "",
                };
                let Some((mut score, match_indices)) = matcher.fuzzy_indices(label, filter) else {
                    return if item.default {
                        Some(MatchedPaletteItem {
                            item,
                            score: 0,
                            match_indices: Vec::new(),
                        })
                    } else {
                        None
                    };
                };
                if match_indices.len() == label.len() {
                    score *= 16
                }
                Some(MatchedPaletteItem {
                    item,
                    score,
                    match_indices,
                })
            })
            .collect();
        items.sort_by_key(|item| std::cmp::Reverse(item.score));
        items
    }

    pub fn prompt(mut self, prompt: impl ToString) -> Self {
        self.prompt = Some(prompt.to_string());
        self
    }

    pub fn into_event(self) -> PaletteEvent {
        self.into()
    }
}

pub struct PaletteItem {
    pub label: LabelSource,
    pub header: String,
    pub icon: PaletteIcon,
    pub system: Box<dyn System<In = In<String>, Out = PaletteEvent>>,
    pub default: bool,
}

pub enum LabelSource {
    Label(String),
    System(Box<dyn System<In = InRef<'static, str>, Out = String>>),
}

impl LabelSource {
    pub fn placeholder(placeholder: impl ToString) -> Self {
        let placeholder = placeholder.to_string();
        LabelSource::system(move |label: InRef<'_, str>| {
            if label.is_empty() {
                placeholder.clone()
            } else {
                label.0.to_string()
            }
        })
    }

    pub fn system<M, I: IntoSystem<InRef<'static, str>, String, M>>(system: I) -> Self {
        LabelSource::System(Box::new(I::into_system(system)))
    }

    pub fn init(&mut self, world: &mut World) {
        if let LabelSource::System(system) = self {
            system.initialize(world);
        }
    }

    pub fn get(&mut self, world: &mut World, filter: &str) -> String {
        match self {
            LabelSource::Label(l) => l.clone(),
            LabelSource::System(system) => system.run(filter, world).expect("Missing label"),
        }
    }
}

impl From<&str> for LabelSource {
    fn from(value: &str) -> Self {
        LabelSource::Label(value.into())
    }
}

impl From<String> for LabelSource {
    fn from(val: String) -> Self {
        LabelSource::Label(val)
    }
}

pub enum PaletteIcon {
    None,
    Link,
}

impl PaletteItem {
    pub fn new<M, I: IntoSystem<In<String>, PaletteEvent, M>>(
        label: impl Into<LabelSource>,
        header: impl ToString,
        system: I,
    ) -> Self {
        Self {
            label: label.into(),
            header: header.to_string(),
            system: Box::new(I::into_system(system)),
            icon: PaletteIcon::None,
            default: false,
        }
    }

    pub fn icon(mut self, icon: PaletteIcon) -> Self {
        self.icon = icon;
        self
    }

    pub fn default(mut self) -> Self {
        self.default = true;
        self
    }
}

pub enum PaletteEvent {
    NextPage {
        prev_page_label: Option<String>,
        next_page: PalettePage,
    },
    Exit,
    Error(String),
}

impl From<PalettePage> for PaletteEvent {
    fn from(v: PalettePage) -> Self {
        Self::NextPage {
            prev_page_label: None,
            next_page: v,
        }
    }
}

pub struct MatchedPaletteItem<'a> {
    pub item: &'a mut PaletteItem,
    pub score: i64,
    pub match_indices: Vec<usize>,
}

const VIEWPORT_LABEL: &str = "Viewport";
const TILES_LABEL: &str = "Tiles";
const SIMULATION_LABEL: &str = "Simulation";
const TIME_LABEL: &str = "Time";
const HELP_LABEL: &str = "Help";
const PRESETS_LABEL: &str = "Presets";
const SKYBOX_LABEL: &str = "Skybox";

struct ViewportEntry {
    label: String,
    camera: Entity,
}

fn gather_viewport_entries(
    window_states: &Query<(&tiles::WindowState, &tiles::WindowId)>,
) -> Vec<ViewportEntry> {
    let mut entries = Vec::new();
    for (window_state, window_id) in window_states.iter() {
        let window_label = window_state
            .descriptor
            .title
            .clone()
            .unwrap_or_else(|| format!("Window {}", window_id.0));
        if let Some(root) = window_state.tile_state.tree.root() {
            collect_viewport_entries(
                &window_state.tile_state.tree,
                root,
                &window_label,
                &mut entries,
            );
        }
    }
    entries
}

fn collect_viewport_entries(
    tree: &egui_tiles::Tree<tiles::Pane>,
    tile_id: TileId,
    window_label: &str,
    entries: &mut Vec<ViewportEntry>,
) {
    let Some(tile) = tree.tiles.get(tile_id) else {
        return;
    };
    match tile {
        Tile::Pane(tiles::Pane::Viewport(viewport)) => {
            if let Some(camera) = viewport.camera {
                entries.push(ViewportEntry {
                    label: viewport_display_label(&viewport.name, window_label),
                    camera,
                });
            }
        }
        Tile::Container(container) => {
            for child in container.children() {
                collect_viewport_entries(tree, *child, window_label, entries);
            }
        }
        _ => {}
    }
}

fn viewport_display_label(label: &str, window_label: &str) -> String {
    let base = if label.is_empty() { "Viewport" } else { label };
    if window_label.is_empty() || window_label == base {
        base.to_string()
    } else {
        format!("{base} ({window_label})")
    }
}

fn reset_editor_cam(transform: &mut Transform, editor_cam: &mut EditorCam) {
    *transform = Transform::IDENTITY;
    editor_cam.current_motion = CurrentMotion::Stationary;
    editor_cam.last_anchor_depth = -2.0;
}

#[derive(bevy::ecs::system::SystemParam)]
pub struct TileParam<'w, 's> {
    windows_state: Query<'w, 's, &'static mut tiles::WindowState>,
    primary_window: Query<'w, 's, Entity, With<PrimaryWindow>>,
    focused_window: Res<'w, FocusedWindow>,
}

impl<'w, 's> TileParam<'w, 's> {
    pub fn target_state(&mut self, target: Option<Entity>) -> Option<Mut<'_, tiles::WindowState>> {
        let target_id = target
            .or(self.focused_window.0)
            .or_else(|| self.primary_window.iter().next());
        target_id.and_then(|target_id| self.windows_state.get_mut(target_id).ok())
    }

    pub fn target(&mut self, target: Option<Entity>) -> Option<Mut<'_, tiles::TileState>> {
        self.target_state(target)
            .map(|s| s.map_unchanged(|s| &mut s.tile_state))
    }
}

pub fn create_action(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new("Create Action", TILES_LABEL, move |_: In<String>| {
        PalettePage::new(vec![
                    PaletteItem::new(
                        LabelSource::placeholder("Enter a label for the button"),
                        "",
                        move |In(label): In<String>| {
                            let msg_label = label.clone();
                            PalettePage::new(vec![
                                PaletteItem::new("send_msg", "Presets", move |_: In<String>| {
                                let msg_label = msg_label.clone();
                                    PalettePage::new(vec![PaletteItem::new(
                                        LabelSource::placeholder("Enter the name of the msg to send"),
                                        "Enter the name of the msg to send",
                                        move |In(name): In<String>| {
                                            let msg_label = msg_label.clone();
                                            PalettePage::new(vec![PaletteItem::new(
                                                LabelSource::placeholder("Msg"),
                                                "Contents of the msg as a lua table - {foo = \"bar\"}",
                                                move |In(msg): In<String>,
                                                mut tile_param: TileParam,
                                                      palette_state: Res<CommandPaletteState>| {
                                                          let Some(mut tile_state) = tile_param.target(palette_state.target_window)
                                                     else {
                                                        return PaletteEvent::Error(
                                                            "Secondary window unavailable"
                                                                .to_string(),
                                                        );
                                                    };
                                                    tile_state.create_action_tile(
                                                        msg_label.clone(),
                                                        format!("client:send_msg({name:?}, {msg})"),
                                                        tile_id,
                                                    );
                                                    PaletteEvent::Exit
                                                },
                                            ).default()])
                                            .into_event()
                                        },
                                    ).default()])
                                    .into_event()
                                }),
                                PaletteItem::new(
                                    LabelSource::placeholder("Enter a lua command (i.e client:send_table)"),
                                    "Enter a custom lua command",
                                    move |lua: In<String>,
                                    mut tile_param: TileParam,
                                          palette_state: Res<CommandPaletteState>| {
                                              let Some(mut tile_state) = tile_param.target(palette_state.target_window)
                                         else {
                                            return PaletteEvent::Error(
                                                "Secondary window unavailable".to_string(),
                                            );
                                        };
                                        tile_state.create_action_tile(label.clone(), lua.0, tile_id);
                                        PaletteEvent::Exit
                                    },
                                )
                                .default(),
                            ]).prompt("Enter a lua command to send")
                            .into_event()
                        },
                    )
                    .default(),
                ])
                .prompt("Enter a label for the action button")
                .into_event()
    })
}

pub fn create_graph(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Graph",
        TILES_LABEL,
        move |_: In<String>, context: Res<EqlContext>| {
            PalettePage::new(graph_parts(&context.0.component_parts, tile_id))
                .prompt("Select a component to graph")
                .into_event()
        },
    )
}

fn graph_parts(
    parts: &BTreeMap<String, eql::ComponentPart>,
    tile_id: Option<TileId>,
) -> Vec<PaletteItem> {
    parts
        .iter()
        .map(|(name, part)| {
            let part = part.clone();
            PaletteItem::new(
                name.clone(),
                "Component",
                move |_: In<String>,
                      mut render_layer_alloc: ResMut<RenderLayerAllocator>,
                      mut tile_param: TileParam,
                      path_reg: Res<ComponentPathRegistry>,
                      schema_reg: Res<impeller2_bevy::ComponentSchemaRegistry>,
                      metadata_reg: Res<ComponentMetadataRegistry>,
                      palette_state: Res<CommandPaletteState>| {
                    let Some(mut tile_state) = tile_param.target(palette_state.target_window)
                    else {
                        return PaletteEvent::Error("Secondary window unavailable".to_string());
                    };
                    if let Some(component) = &part.component {
                        let component_id = component.id;
                        let Some(schema) = schema_reg.0.get(&component_id) else {
                            return PaletteEvent::Error(format!(
                                "No schema registered for component {}",
                                component.name
                            ));
                        };
                        if metadata_reg.get_metadata(&component_id).is_none() {
                            return PaletteEvent::Error(format!(
                                "No metadata registered for component {}",
                                component.name
                            ));
                        }

                        let component_path = path_reg
                            .get(&component_id)
                            .cloned()
                            .unwrap_or_else(|| ComponentPath::from_name(&component.name));

                        let values = graph_lines_from_component(&component_path, schema);
                        let graph_label = component.name.clone();
                        let components =
                            BTreeMap::from_iter(std::iter::once((component_path, values)));
                        let Some(bundle) =
                            GraphBundle::try_new(&mut render_layer_alloc, components, graph_label)
                        else {
                            return PaletteEvent::Error(
                                "Unable to create graph: render layer budget exhausted".to_string(),
                            );
                        };
                        tile_state.create_graph_tile(tile_id, bundle);
                        PaletteEvent::Exit
                    } else {
                        PalettePage::new(graph_parts(&part.children, tile_id)).into()
                    }
                },
            )
        })
        .collect()
}

pub fn create_monitor(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Monitor",
        TILES_LABEL,
        move |_: In<String>, eql: Res<EqlContext>| {
            PalettePage::new(monitor_parts(&eql.0.component_parts, tile_id))
                .prompt("Select a component to monitor")
                .into_event()
        },
    )
}

fn monitor_parts(
    parts: &BTreeMap<String, eql::ComponentPart>,
    tile_id: Option<TileId>,
) -> Vec<PaletteItem> {
    parts
        .iter()
        .map(|(name, part)| {
            let part = part.clone();
            PaletteItem::new(
                name.clone(),
                "Component",
                move |_: In<String>,
                      mut tile_param: TileParam,
                      palette_state: Res<CommandPaletteState>| {
                    let Some(mut tile_state) = tile_param.target(palette_state.target_window)
                    else {
                        return PaletteEvent::Error("Secondary window unavailable".to_string());
                    };
                    if let Some(component) = &part.component {
                        tile_state.create_monitor_tile(component.name.clone(), tile_id);
                        PaletteEvent::Exit
                    } else {
                        PalettePage::new(monitor_parts(&part.children, tile_id)).into()
                    }
                },
            )
        })
        .collect()
}

fn reset_cameras() -> PaletteItem {
    PaletteItem::new(
        "Reset Cameras",
        VIEWPORT_LABEL,
        |_: In<String>, window_states: Query<(&tiles::WindowState, &tiles::WindowId)>| {
            let entries = gather_viewport_entries(&window_states);
            if entries.is_empty() {
                return PalettePage::new(vec![PaletteItem::new(
                    "No viewports available",
                    VIEWPORT_LABEL,
                    |_: In<String>| PaletteEvent::Exit,
                )])
                .into_event();
            }

            let all_cameras: Vec<_> = entries.iter().map(|entry| entry.camera).collect();
            let mut items = Vec::with_capacity(entries.len() + 1);
            items.push(PaletteItem::new(
                "Reset all viewports",
                VIEWPORT_LABEL,
                move |_: In<String>,
                      mut query: Query<(&mut Transform, &mut EditorCam), With<MainCamera>>| {
                    for camera in &all_cameras {
                        if let Ok((mut transform, mut editor_cam)) = query.get_mut(*camera) {
                            reset_editor_cam(&mut transform, &mut editor_cam);
                        }
                    }
                    PaletteEvent::Exit
                },
            ));

            for entry in entries {
                let label = format!("Reset {}", entry.label);
                let camera = entry.camera;
                items.push(
                    PaletteItem::new(
                        label,
                        VIEWPORT_LABEL,
                        move |_: In<String>,
                              mut query: Query<
                            (&mut Transform, &mut EditorCam),
                            With<MainCamera>,
                        >| {
                            if let Ok((mut transform, mut editor_cam)) = query.get_mut(camera) {
                                reset_editor_cam(&mut transform, &mut editor_cam);
                            }
                            PaletteEvent::Exit
                        },
                    ),
                );
            }

            PalettePage::new(items)
                .prompt("Select viewport to reset")
                .into()
        },
    )
}

pub fn create_viewport(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Viewport",
        TILES_LABEL,
        move |_: In<String>, mut tile_param: TileParam, palette_state: Res<CommandPaletteState>| {
            let Some(mut tile_state) = tile_param.target(palette_state.target_window) else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_viewport_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_window() -> PaletteItem {
    PaletteItem::new("Create Window", TILES_LABEL, move |_: In<String>| {
        PalettePage::new(vec![
            PaletteItem::new(
                LabelSource::placeholder("Enter window title"),
                "Leave blank for a default title",
                move |In(title): In<String>,
                      mut commands: Commands,

                      mut palette_state: ResMut<CommandPaletteState>| {
                    let title_opt = if title.trim().is_empty() {
                        None
                    } else {
                        Some(title.trim().to_string())
                    };
                    let (state, id) = tiles::create_secondary_window(title_opt);
                    let entity = commands.spawn((id, state)).id();
                    palette_state.target_window = Some(entity);
                    PaletteEvent::Exit
                },
            )
            .default(),
        ])
        .prompt("Enter a title for the new window")
        .into_event()
    })
}

pub fn create_query_table(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Query Table",
        TILES_LABEL,
        move |_: In<String>, mut tile_param: TileParam, palette_state: Res<CommandPaletteState>| {
            let Some(mut tile_state) = tile_param.target(palette_state.target_window) else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_query_table_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_query_plot(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Query Plot",
        TILES_LABEL,
        move |_: In<String>, mut tile_param: TileParam, palette_state: Res<CommandPaletteState>| {
            let Some(mut tile_state) = tile_param.target(palette_state.target_window) else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_query_plot_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_schematic_tree(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Schematic Tree",
        TILES_LABEL,
        move |_: In<String>, mut tile_param: TileParam, palette_state: Res<CommandPaletteState>| {
            let Some(mut tile_state) = tile_param.target(palette_state.target_window) else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_tree_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_data_overview(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Data Overview",
        TILES_LABEL,
        move |_: In<String>, mut tile_param: TileParam, palette_state: Res<CommandPaletteState>| {
            let Some(mut tile_state) = tile_param.target(palette_state.target_window) else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_data_overview_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_video_stream(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Video Stream",
        TILES_LABEL,
        move |_: In<String>| -> PaletteEvent {
            PalettePage::new(vec![
                PaletteItem::new(
                    LabelSource::placeholder(
                        "Enter the name of the msg containing the video frames",
                    ),
                    "",
                    move |In(msg_name): In<String>,
                          mut tile_param: TileParam,
                          palette_state: Res<CommandPaletteState>| {
                        let Some(mut tile_state) = tile_param.target(palette_state.target_window)
                        else {
                            return PaletteEvent::Error("Secondary window unavailable".to_string());
                        };
                        let msg_name = msg_name.trim().to_string();
                        let label = format!("Video Stream {}", msg_name);
                        tile_state.create_video_stream_tile(msg_name, label, tile_id);
                        PaletteEvent::Exit
                    },
                )
                .default(),
            ])
            .into_event()
        },
    )
}

pub fn create_log_stream(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Log Stream",
        TILES_LABEL,
        move |_: In<String>| -> PaletteEvent {
            PalettePage::new(vec![
                PaletteItem::new(
                    LabelSource::placeholder("Enter the name of the message log stream"),
                    "",
                    move |In(msg_name): In<String>,
                          mut tile_param: TileParam,
                          palette_state: Res<CommandPaletteState>| {
                        let Some(mut tile_state) = tile_param.target(palette_state.target_window)
                        else {
                            return PaletteEvent::Error("Secondary window unavailable".to_string());
                        };
                        let msg_name = msg_name.trim().to_string();
                        let label = format!("Log Stream {}", msg_name);
                        tile_state.create_log_stream_tile(msg_name, label, tile_id);
                        PaletteEvent::Exit
                    },
                )
                .default(),
            ])
            .into_event()
        },
    )
}

fn set_playback_speed() -> PaletteItem {
    PaletteItem::new("Set Playback Speed", TIME_LABEL, |_: In<String>| {
        let speeds = [
            0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0,
        ];
        let next_page = PalettePage::new(
            speeds
                .into_iter()
                .map(|speed| {
                    PaletteItem::new(
                        speed.to_string(),
                        "SPEED".to_string(),
                        move |_: In<String>, mut playback_speed: ResMut<PlaybackSpeed>| {
                            playback_speed.0 = speed;
                            PaletteEvent::Exit
                        },
                    )
                })
                .collect(),
        );

        PaletteEvent::NextPage {
            prev_page_label: None,
            next_page,
        }
    })
}

fn fix_current_time_range() -> PaletteItem {
    PaletteItem::new(
        "Fix Current Time Range",
        TIME_LABEL,
        |_: In<String>,
         selected_range: Res<SelectedTimeRange>,
         mut behavior: ResMut<TimeRangeBehavior>| {
            behavior.start = Offset::Fixed(selected_range.0.start);
            behavior.end = Offset::Fixed(selected_range.0.end);
            PaletteEvent::Exit
        },
    )
}

fn set_time_range_behavior() -> PaletteItem {
    PaletteItem::new("Set Time Range", TIME_LABEL, |_: In<String>| {
        PalettePage::new(vec![
            PaletteItem::new(
                LabelSource::placeholder(
                    "Enter start offset (e.g., '+5m', '-10s', '=2023-01-01T00:00:00Z')",
                ),
                "Start Offset",
                move |start_str: In<String>| {
                    let Ok(start_offset) = Offset::from_str(&start_str.0) else {
                        return PaletteEvent::Error(format!(
                            "Invalid start offset format: {}",
                            start_str.0
                        ));
                    };

                    PalettePage::new(vec![
                        PaletteItem::new(
                            LabelSource::placeholder(
                                "Enter end offset (e.g., '+5m', '-10s', '=2023-01-01T00:00:00Z')",
                            ),
                            "End Offset",
                            move |end_str: In<String>,
                                  mut behavior: ResMut<TimeRangeBehavior>,
                                  earliest: Res<EarliestTimestamp>,
                                  latest: Res<LastUpdated>| {
                                let Ok(end_offset) = Offset::from_str(&end_str.0) else {
                                    return PaletteEvent::Error(format!(
                                        "Invalid end offset format: {}",
                                        end_str.0
                                    ));
                                };

                                let preview = TimeRangeBehavior {
                                    start: start_offset,
                                    end: end_offset,
                                };

                                match preview.calculate_selected_range(earliest.0, latest.0) {
                                    Ok(_) | Err(TimeRangeError::NoData) => {
                                        *behavior = preview;
                                        PaletteEvent::Exit
                                    }
                                    Err(TimeRangeError::InvalidRange { .. }) => PaletteEvent::Error(
                                        format!(
                                            "Invalid time range: `{}` must resolve before `{}` for the current data",
                                            start_offset, end_offset
                                        ),
                                    ),
                                }
                            },
                        )
                        .default(),
                    ])
                    .prompt("Enter the end offset")
                    .into_event()
                },
            )
            .default(),
        ])
        .prompt("Enter the start offset")
        .into_event()
    })
}

fn goto_tick() -> PaletteItem {
    PaletteItem::new("Goto Tick...", TIME_LABEL, |_: In<String>| {
        PalettePage::new(vec![
            PaletteItem::new(
                LabelSource::placeholder("Enter tick number"),
                "",
                move |In(tick_str): In<String>,
                      mut current_tick: ResMut<CurrentTimestamp>,
                      mut ui_tick: ResMut<UITick>,
                      mut paused: ResMut<Paused>,
                      mut latest_follow: ResMut<LatestFollow>,
                      mut auto_follow_latest_state: ResMut<AutoFollowLatestState>,
                      earliest_timestamp: Res<EarliestTimestamp>,
                      mut tick_origin: ResMut<StreamTickOrigin>,
                      tick_time: Res<SimulationTimeStep>| {
                    let trimmed = tick_str.trim();
                    if trimmed.is_empty() {
                        return PaletteEvent::Error(
                            "Tick value cannot be empty. Please enter a non-negative integer."
                                .into(),
                        );
                    }

                    let Ok(parsed_tick) = trimmed.parse::<u64>() else {
                        return PaletteEvent::Error(format!(
                            "Invalid tick value: {trimmed}. Please enter a non-negative integer."
                        ));
                    };

                    if parsed_tick >= i64::MAX as u64 {
                        return PaletteEvent::Error("Tick value is too large".to_string());
                    }

                    let tick_duration_us = (hifitime::Duration::from_seconds(tick_time.0)
                        .total_nanoseconds()
                        / 1000) as i64;

                    if tick_duration_us <= 0 {
                        return PaletteEvent::Error(
                            "Simulation tick duration must be positive to compute timestamp"
                                .to_string(),
                        );
                    }

                    let parsed_tick = parsed_tick as i64;
                    let Some(offset_us) = tick_duration_us.checked_mul(parsed_tick) else {
                        return PaletteEvent::Error(
                            "Tick value is too large to convert to a timestamp".to_string(),
                        );
                    };

                    let base_timestamp = tick_origin.origin(earliest_timestamp.0).0;
                    let Some(target_us) = base_timestamp.checked_add(offset_us) else {
                        return PaletteEvent::Error(
                            "Computed timestamp is out of range".to_string(),
                        );
                    };

                    let timestamp = Timestamp(target_us);
                    auto_follow_latest_state.cancel();
                    paused.0 = true;
                    latest_follow.0 = false;
                    current_tick.0 = timestamp;
                    ui_tick.0 = timestamp.0;
                    if parsed_tick == 0 {
                        tick_origin.request_rebase();
                    }

                    PaletteEvent::Exit
                },
            )
            .default(),
        ])
        .prompt("Enter the tick to jump to")
        .into_event()
    })
}

pub fn save_schematic() -> PaletteItem {
    PaletteItem::new(
        "Save Schematic",
        PRESETS_LABEL,
        |_name: In<String>, mut commands: Commands| {
            // Capture descriptors/screens and rebuild the schematic from the live
            // UI, then write it back to the DB in a dedicated system (RFD #724).
            commands.run_system_cached(crate::ui::capture_window_screens_oneoff);
            commands.run_system_cached(crate::ui::schematic::tiles_to_schematic);
            commands.run_system_cached(queue_save_schematic_db_now);
            PaletteEvent::Exit
        },
    )
}

/// Save the current schematic under a new name (RFD #724 Phase 2). Prompts for
/// a name, stores it as `schematics/<name>.kdl`, and points `schematic.active`
/// at it — the named counterpart to "Save Schematic", and what populates the
/// "Open Schematic..." picker.
pub fn save_schematic_as() -> PaletteItem {
    PaletteItem::new("Save Schematic As...", PRESETS_LABEL, |_: In<String>| {
        PalettePage::new(vec![save_schematic_as_prompt()])
            .label("Save Schematic As")
            .prompt("Enter a schematic name...")
            .into_event()
    })
}

fn save_schematic_as_prompt() -> PaletteItem {
    PaletteItem::new(
        LabelSource::placeholder("Enter a schematic name..."),
        PRESETS_LABEL,
        |In(name): In<String>,
         mut pending_key: ResMut<PendingSchematicSaveKey>,
         mut commands: Commands| {
            let key = match schematic_save_key_from_name(&name) {
                Ok(key) => key,
                Err(err) => return PaletteEvent::Error(err),
            };
            pending_key.0 = Some(key);
            commands.run_system_cached(crate::ui::capture_window_screens_oneoff);
            commands.run_system_cached(crate::ui::schematic::tiles_to_schematic);
            commands.run_system_cached(queue_save_schematic_db_now);
            PaletteEvent::Exit
        },
    )
}

/// DB-native save (RFD #724 Phase 2): upload the active schematic, its window
/// sub-schematics and any newly added local assets to the DB Asset Server over
/// HTTP `PUT`, then point `schematic.active` at the uploaded schematic. The
/// `PUT`s are acknowledged and complete before `SetDbConfig` is sent, so the
/// pointer never claims a save that did not land.
#[allow(clippy::too_many_arguments)]
fn queue_save_schematic_db_now(
    schematic: Res<CurrentSchematic>,
    window_schematics: Res<CurrentWindowSchematics>,
    skybox_cache: Option<Res<SkyboxCache>>,
    connection_addr: Option<Res<ConnectionAddr>>,
    config: Res<DbConfig>,
    mut pending_key: ResMut<PendingSchematicSaveKey>,
    mut save_in_flight: ResMut<SchematicSaveInFlight>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    // A "Save As" supplies an explicit key; a plain "Save" overwrites whatever
    // schematic is currently active (falling back to the default main key on a
    // fresh DB). Resolve the target without consuming the pending name yet: a
    // rejected or disconnected save must keep it so a retry still targets the
    // chosen name rather than silently writing to the previous active schematic.
    let pending = pending_key.0.clone();
    let active_key = pending.clone().unwrap_or_else(|| {
        config
            .schematic_active()
            .map(str::to_string)
            .unwrap_or_else(|| ACTIVE_SCHEMATIC_KEY.to_string())
    });

    if save_in_flight.task.is_some() {
        failed.write(DocumentCommandFailed {
            title: "Failed to Save Schematic".to_string(),
            message: "A schematic save is already in progress.".to_string(),
        });
        return;
    }

    let Some(addr) = connection_addr.as_ref().map(|c| c.0) else {
        failed.write(DocumentCommandFailed {
            title: "Failed to Save Schematic".to_string(),
            message: "Not connected to a database.".to_string(),
        });
        return;
    };

    // Committed to uploading: consume the pending name now. `poll_schematic_save`
    // restores it if the upload itself fails.
    pending_key.0 = None;

    let root = root_schematic_for_save(&schematic, skybox_cache.as_deref());
    let windows = window_document_saves(&window_schematics);
    let plan = plan_db_save(&root, &windows, &active_key);

    // The bytes the DB will mirror into `schematic.content`, recorded once the
    // upload lands so config sync doesn't mistake the DB's echo of our own save
    // for an external change (which would reload over HTTP on the saving client).
    save_in_flight.active_content = plan.active_schematic_content();
    save_in_flight.active_key = Some(active_key);
    save_in_flight.pending_key = pending;
    // Upload off the main thread; `poll_schematic_save` repoints `schematic.active`
    // only after every `PUT` is acknowledged, so the pointer never claims a save
    // that did not land.
    save_in_flight.task =
        Some(IoTaskPool::get().spawn(async move { upload_db_save_plan(&plan, Some(addr)) }));
}

/// Applies the outcome of an in-flight DB-native save: on success, record the
/// stored content and repoint `schematic.active`; on failure, surface it.
fn poll_schematic_save(
    mut save_in_flight: ResMut<SchematicSaveInFlight>,
    tx: Option<Res<PacketTx>>,
    mut pending_key: ResMut<PendingSchematicSaveKey>,
    mut pending_active: ResMut<PendingActiveSchematic>,
    mut last_synced: ResMut<LastSyncedSchematicContent>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    let Some(task) = save_in_flight.task.as_mut() else {
        return;
    };
    let Some(result) = future::block_on(future::poll_once(task)) else {
        return;
    };
    save_in_flight.task = None;
    let active_key = save_in_flight.active_key.take();
    let active_content = save_in_flight.active_content.take();
    let pending = save_in_flight.pending_key.take();

    match result {
        Ok(()) => {
            if let Some(content) = active_content {
                last_synced.0 = Some(content);
            }
            if let Some(active_key) = active_key {
                last_synced.1 = Some(active_key.clone());
                // Pin the key we just saved so config sync ignores the DB's still
                // -stale active pointer until it echoes this repoint, avoiding a
                // brief revert to the previously active schematic.
                pending_active.0 = Some(active_key.clone());
                if let Some(tx) = tx {
                    tx.send_msg(SetDbConfig {
                        metadata: [("schematic.active".to_string(), active_key)]
                            .into_iter()
                            .collect(),
                        ..Default::default()
                    });
                }
            }
        }
        Err(err) => {
            // Restore the "Save As" name so the next save retries the chosen
            // target instead of overwriting the previous active schematic.
            if pending.is_some() {
                pending_key.0 = pending;
            }
            failed.write(DocumentCommandFailed {
                title: "Failed to Save Schematic".to_string(),
                message: err,
            });
        }
    }
}

/// Refreshes [`SchematicIndexCache`] off the main thread: warms it on connect
/// and re-lists whenever `DbConfig` changes (e.g. after a save adds a schematic).
fn refresh_schematic_index(
    config: Option<Res<DbConfig>>,
    connection_addr: Option<Res<ConnectionAddr>>,
    mut cache: ResMut<SchematicIndexCache>,
) {
    if let Some(task) = cache.task.as_mut() {
        if let Some(result) = future::block_on(future::poll_once(task)) {
            cache.task = None;
            cache.loaded_once = true;
            match result {
                Ok(keys) => {
                    cache.keys = keys;
                    cache.error = None;
                }
                // Drop the now-unverified list so "Open Schematic…" surfaces the
                // error instead of offering stale names that may no longer exist.
                Err(err) => {
                    cache.keys.clear();
                    cache.error = Some(err);
                }
            }
        }
        return;
    }

    let Some(addr) = connection_addr.as_ref().map(|c| c.0) else {
        // Disconnected: drop the listing so a later connection re-lists and we
        // never offer assets from a previous session.
        if cache.loaded_once || !cache.keys.is_empty() || cache.error.is_some() {
            *cache = SchematicIndexCache::default();
        }
        return;
    };
    let config_changed = config.as_ref().is_some_and(|c| c.is_changed());
    let connection_changed =
        cache.addr != Some(addr) || connection_addr.as_ref().is_some_and(|c| c.is_changed());
    if cache.loaded_once && !config_changed && !connection_changed {
        return;
    }
    cache.addr = Some(addr);
    cache.task = Some(IoTaskPool::get().spawn(async move { fetch_schematic_index(Some(addr)) }));
}

pub fn clear_schematic() -> PaletteItem {
    PaletteItem::new(
        "Clear Schematic",
        PRESETS_LABEL,
        |_: In<String>, mut params: LoadSchematicParams| {
            params.current_document.clear();
            params.load_schematic(&impeller2_wkt::Schematic::default(), None, None);
            PaletteEvent::Exit
        },
    )
}

/// DB-native load (RFD #724 Phase 2): present the schematics the DB Asset Server
/// holds — listed off the main thread into [`SchematicIndexCache`] — and open
/// the chosen one over HTTP.
pub fn open_schematic() -> PaletteItem {
    PaletteItem::new(
        "Open Schematic...",
        PRESETS_LABEL,
        |_: In<String>,
         index: Res<SchematicIndexCache>,
         connection_addr: Option<Res<ConnectionAddr>>| {
            if connection_addr.is_none() {
                return PaletteEvent::Error("Not connected to a database".into());
            }
            if index.keys.is_empty() {
                if let Some(err) = &index.error {
                    return PaletteEvent::Error(format!("Failed to list schematics: {err}"));
                }
                if !index.loaded_once {
                    return PaletteEvent::Error("Loading schematics from the database…".into());
                }
                return PaletteEvent::Error("No schematics found in the database".into());
            }
            let items = index
                .keys
                .iter()
                .cloned()
                .map(open_schematic_item)
                .collect();
            PalettePage::new(items)
                .label("Open Schematic")
                .prompt("Select a schematic to open...")
                .into_event()
        },
    )
}

fn open_schematic_item(key: String) -> PaletteItem {
    let label = key
        .strip_prefix("schematics/")
        .unwrap_or(&key)
        .strip_suffix(".kdl")
        .map(str::to_string)
        .unwrap_or_else(|| key.clone());
    PaletteItem::new(
        label,
        PRESETS_LABEL,
        move |_: In<String>,
              tx: Res<PacketTx>,
              mut pending_active: ResMut<PendingActiveSchematic>| {
            // Repoint the DB's active schematic rather than loading locally:
            // config sync then loads it over HTTP, and `schematic.active` stays
            // authoritative so later DbConfig updates and "Save Schematic" both
            // target the schematic the user just opened.
            //
            // Pin the requested key so config sync ignores the DB's still-stale
            // pointer until it echoes this repoint, instead of momentarily
            // reloading the schematic we're switching away from.
            pending_active.0 = Some(key.clone());
            tx.send_msg(SetDbConfig {
                metadata: [("schematic.active".to_string(), key.clone())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            });
            PaletteEvent::Exit
        },
    )
}

fn root_schematic_for_save(
    schematic: &CurrentSchematic,
    skybox_cache: Option<&SkyboxCache>,
) -> impeller2_wkt::Schematic {
    let mut root = schematic.0.clone();
    if let Some(cache) = skybox_cache {
        root.skybox = cache.active.as_ref().map(|active| SkyboxConfig {
            name: active.clone(),
        });
    }
    root
}

fn window_document_saves(window_schematics: &CurrentWindowSchematics) -> Vec<WindowDocumentSave> {
    window_schematics
        .0
        .iter()
        .map(|entry| WindowDocumentSave {
            window_id: entry.window_id,
            file_name: entry.file_name.clone(),
            kdl: entry.schematic.to_kdl(),
        })
        .collect()
}

pub fn set_color_scheme() -> PaletteItem {
    PaletteItem::new("Set Color Scheme", PRESETS_LABEL, |_: In<String>| {
        let presets = colors::available_presets();
        let mut items = vec![];
        for preset in presets {
            let name = preset.name.to_string();
            let label = preset.label.to_string();
            items.push(PaletteItem::new(
                label,
                "",
                move |_: In<String>, mut windows_state: Query<&mut tiles::WindowState>| {
                    let current = colors::current_selection();
                    let desired_mode = if colors::scheme_supports_mode(&name, &current.mode) {
                        current.mode
                    } else {
                        "dark".to_string()
                    };
                    let selection = colors::apply_scheme_and_mode(&name, &desired_mode);
                    set_mode_all(&selection.mode, &mut windows_state);
                    PaletteEvent::Exit
                },
            ));
        }
        PalettePage::new(items).into_event()
    })
}

pub fn set_color_scheme_mode() -> PaletteItem {
    PaletteItem::new("Set Color Scheme Mode", PRESETS_LABEL, |_: In<String>| {
        let current = colors::current_selection();
        let scheme_name = current.scheme.clone();
        let options = [("Dark", "dark"), ("Light", "light")];
        let mut items = vec![];
        for (label, mode) in options {
            let available = colors::scheme_supports_mode(&scheme_name, mode);
            let display_label = if available {
                label.to_string()
            } else {
                format!("{label} (unavailable)")
            };
            let scheme_name = scheme_name.clone();
            items.push(PaletteItem::new(
                display_label,
                "",
                move |_: In<String>, mut windows_state: Query<&mut tiles::WindowState>| {
                    if !colors::scheme_supports_mode(&scheme_name, mode) {
                        return PaletteEvent::Error(
                            "This scheme does not provide that variant".to_string(),
                        );
                    }
                    let selection = colors::apply_scheme_and_mode(&scheme_name, mode);
                    set_mode_all(&selection.mode, &mut windows_state);
                    PaletteEvent::Exit
                },
            ));
        }
        PalettePage::new(items).into_event()
    })
}

fn clear_skybox() -> PaletteItem {
    PaletteItem::new(
        "Clear Skybox",
        SKYBOX_LABEL,
        |_: In<String>,
         mut cache: ResMut<SkyboxCache>,
         mut skyboxes: MessageWriter<SetActiveSkybox>,
         mut schematic: ResMut<CurrentSchematic>,
         mut current_document: ResMut<CurrentDocument>,
         mut document_assets: ResMut<Assets<SchematicDocumentAsset>>,
         mut last_synced_content: ResMut<LastSyncedSchematicContent>,
         mut locally_pushed: ResMut<LocallyPushedSkyboxActive>,
         config: Res<DbConfig>,
         tx: Res<PacketTx>| {
            if cache.active.is_none() && schematic.skybox.is_none() {
                return PaletteEvent::Error("No skybox is active".into());
            }
            skyboxes.write(SetActiveSkybox::Clear);
            SkyboxDocumentSyncMut {
                schematic: &mut schematic,
                current_document: &mut current_document,
                document_assets: &mut document_assets,
                last_synced_content: &mut last_synced_content,
                locally_pushed: &mut locally_pushed,
                cache: &mut cache,
                tx: &tx,
                active_key: config.schematic_active().unwrap_or(ACTIVE_SCHEMATIC_KEY),
            }
            .sync_skybox_to_document_and_db(None);
            PaletteEvent::Exit
        },
    )
}

fn activate_skybox_item(label: String, name: String) -> PaletteItem {
    PaletteItem::new(
        label,
        SKYBOX_LABEL,
        move |_: In<String>,
              mut cache: ResMut<SkyboxCache>,
              mut skyboxes: MessageWriter<SetActiveSkybox>,
              mut schematic: ResMut<CurrentSchematic>,
              mut current_document: ResMut<CurrentDocument>,
              mut document_assets: ResMut<Assets<SchematicDocumentAsset>>,
              mut last_synced_content: ResMut<LastSyncedSchematicContent>,
              mut locally_pushed: ResMut<LocallyPushedSkyboxActive>,
              config: Res<DbConfig>,
              tx: Res<PacketTx>| {
            skyboxes.write(SetActiveSkybox::ByName(name.clone()));
            SkyboxDocumentSyncMut {
                schematic: &mut schematic,
                current_document: &mut current_document,
                document_assets: &mut document_assets,
                last_synced_content: &mut last_synced_content,
                locally_pushed: &mut locally_pushed,
                cache: &mut cache,
                tx: &tx,
                active_key: config.schematic_active().unwrap_or(ACTIVE_SCHEMATIC_KEY),
            }
            .sync_skybox_to_document_and_db(Some(SkyboxConfig { name: name.clone() }));
            PaletteEvent::Exit
        },
    )
}

fn revert_previous_skybox_item(name: String) -> PaletteItem {
    PaletteItem::new(
        format!("Revert to {name}"),
        SKYBOX_LABEL,
        |_: In<String>, mut commands: Commands| {
            commands.run_system_cached(crate::skybox_generation::revert_previous_skybox);
            PaletteEvent::Exit
        },
    )
}

fn skybox_menu() -> PaletteItem {
    PaletteItem::new(
        "Skybox...",
        SKYBOX_LABEL,
        |_: In<String>, cache: Res<SkyboxCache>, skybox_ui: Res<SkyboxGenerationUi>| {
            let active = cache.active.as_deref();
            let mut items = Vec::with_capacity(cache.manifest.entries.len() + 4);
            items.push(PaletteItem::new(
                "Generate Skybox...",
                SKYBOX_LABEL,
                |_: In<String>| {
                    PalettePage::new(vec![generate_skybox_from_prompt()])
                        .label("Generate Skybox")
                        .prompt("Describe a new skybox...")
                        .into_event()
                },
            ));
            if active.is_some() {
                items.push(clear_skybox());
            }
            if let Some(revert) = skybox_ui.revert_name.clone()
                && active != Some(revert.as_str())
            {
                items.push(revert_previous_skybox_item(revert));
            }
            for entry in &cache.manifest.entries {
                let name = entry.name.clone();
                let label = if active == Some(name.as_str()) {
                    format!("{name} (active)")
                } else {
                    name.clone()
                };
                items.push(activate_skybox_item(label, name));
            }

            PalettePage::new(items)
                .label("Skybox")
                .prompt("Generate, clear, or select a skybox...")
                .into_event()
        },
    )
}

fn generate_skybox_from_prompt() -> PaletteItem {
    PaletteItem::new(
        LabelSource::placeholder("Describe a new skybox..."),
        SKYBOX_LABEL,
        |In(prompt): In<String>| {
            let prompt = prompt.trim();
            if prompt.is_empty() {
                error!(
                    target: "bevy_ai_skybox",
                    "skybox generation rejected from command palette: prompt cannot be empty"
                );
                return PaletteEvent::Error("Prompt cannot be empty".into());
            }

            PalettePage::new(skybox_style_items(prompt.to_string()))
                .label("Skybox Style")
                .prompt("Select skybox style...")
                .into_event()
        },
    )
    .default()
}

fn skybox_style_items(prompt: String) -> Vec<PaletteItem> {
    [
        ("M3 Photoreal (default)", SkyboxStyle::M3Photoreal),
        ("M3 UHD Render", SkyboxStyle::M3UhdRender),
        ("M3 Advanced", SkyboxStyle::M3Advanced),
    ]
    .into_iter()
    .map(|(label, style)| {
        let prompt = prompt.clone();
        PaletteItem::new(label, SKYBOX_LABEL, move |_: In<String>| {
            PalettePage::new(skybox_resolution_items(prompt.clone(), style))
                .label("Skybox Resolution")
                .prompt("Select skybox resolution...")
                .into_event()
        })
    })
    .collect()
}

fn skybox_resolution_items(prompt: String, style: SkyboxStyle) -> Vec<PaletteItem> {
    // Default (4K) first so confirming with Enter without moving the selection
    // picks the documented default, matching the style page convention.
    [
        ("4K (default, 1024 px faces)", SkyboxResolution::FourK),
        ("1K (256 px faces)", SkyboxResolution::OneK),
        ("2K (512 px faces)", SkyboxResolution::TwoK),
        ("8K (2048 px faces)", SkyboxResolution::EightK),
        ("16K (4096 px faces)", SkyboxResolution::SixteenK),
    ]
    .into_iter()
    .map(|(label, resolution)| {
        let prompt = prompt.clone();
        PaletteItem::new(
            label,
            SKYBOX_LABEL,
            move |_: In<String>,
                  settings: Option<Res<SkyboxGenerationSettings>>,
                  skybox_ui: Res<SkyboxGenerationUi>,
                  mut skyboxes: MessageWriter<GenerateSkybox>| {
            if skybox_ui.is_busy() {
                error!(
                    target: "bevy_ai_skybox",
                    prompt = %prompt,
                    "skybox generation rejected from command palette: generation already in progress"
                );
                return PaletteEvent::Error(
                    "A skybox is already generating — check the status bar".into(),
                );
            }
            let Some(settings) = settings else {
                error!(
                    target: "bevy_ai_skybox",
                    prompt = %prompt,
                    "skybox generation rejected from command palette: plugin is not installed"
                );
                return PaletteEvent::Error("Skybox generation plugin is not installed".into());
            };
            if settings.resolved_api_key().is_none() {
                error!(
                    target: "bevy_ai_skybox",
                    prompt = %prompt,
                    "skybox generation rejected from command palette: missing BLOCKADE_API_KEY"
                );
                return PaletteEvent::Error(
                    "Set BLOCKADE_API_KEY in the environment, then restart the editor \
                     (e.g. BLOCKADE_API_KEY=… elodin editor …)"
                        .into(),
                );
            }

            skyboxes.write(GenerateSkybox {
                prompt: prompt.clone(),
                style: Some(style),
                resolution: Some(resolution),
                ..Default::default()
            });
            PaletteEvent::Exit
        },
        )
    })
    .collect()
}

fn create_object_3d_with_color(eql: String, expr: eql::Expr, mesh: Mesh) -> PaletteEvent {
    PalettePage::new(vec![
        PaletteItem::new(
            LabelSource::placeholder("Enter color (hex or name, default: #cccccc)"),
            "Enter color as rgb",
            move |In(color_str): In<String>,
                  mut commands: Commands,
                  eql_ctx: Res<EqlContext>,
                  entity_map: Res<EntityMap>,
                  component_value_maps: Query<&'static ComponentValue>,
                  mut material_assets: ResMut<Assets<StandardMaterial>>,
                  mut mesh_assets: ResMut<Assets<bevy::prelude::Mesh>>,
                  mut mat3_material_assets: ResMut<Assets<bevy_mat3_material::Mat3Material>>,
                  assets: Res<AssetServer>,
                  geo_context: Res<GeoContext>,
                  connection_addr: Option<Res<ConnectionAddr>>| {
                let color_str = color_str.trim();
                let (r, g, b) =
                    parse_color(color_str, &eql_ctx.0, &entity_map, component_value_maps)
                        .unwrap_or((0.8, 0.8, 0.8));
                let connection_addr = connection_addr.as_ref().map(|addr| addr.0);

                let mesh_source = impeller2_wkt::Object3DMesh::Mesh {
                    mesh: mesh.clone(),
                    material: Material::color(r, g, b),
                };

                let _ = crate::object_3d::create_object_3d_entity(
                    &mut commands,
                    Object3D {
                        eql: eql.clone(),
                        mesh: mesh_source,
                        icon: None,
                        thrusters: Vec::new(),
                        mesh_visibility_range: None,
                        frame: None,
                        node_id: Default::default(),
                    },
                    expr.clone(),
                    &eql_ctx.0,
                    &mut material_assets,
                    &mut mesh_assets,
                    &mut mat3_material_assets,
                    &assets,
                    &geo_context,
                    connection_addr,
                );

                PaletteEvent::Exit
            },
        )
        .default(),
    ])
    .prompt("Enter color for mesh")
    .into_event()
}

fn parse_color(
    expr: &str,
    ctx: &eql::Context,
    entity_map: &EntityMap,
    component_value_maps: Query<&'static ComponentValue>,
) -> Option<(f32, f32, f32)> {
    let expr = ctx.parse_str(expr).ok()?;
    let expr = crate::object_3d::compile_eql_expr(expr).ok()?;
    let val = expr.execute(entity_map, &component_value_maps).ok()?;

    let ComponentValue::F64(array) = val else {
        return None;
    };
    let buf = array.buf.as_buf();
    match buf {
        [r, g, b, ..] => Some((*r as f32, *g as f32, *b as f32)),
        _ => None,
    }
}

pub fn create_3d_object() -> PaletteItem {
    PaletteItem::new("Create 3D Object", TILES_LABEL, move |_: In<String>| {
        PalettePage::new(vec![
            PaletteItem::new(
                LabelSource::placeholder("Enter EQL expression (e.g., 'entity.position')"),
                "Enter an EQL expression that resolves to a 7-component array [qx, qy, qz, qw, px, py, pz]",
                move |In(eql): In<String>, eql_context: Res<EqlContext>| {
                    let expr = match eql_context.0.parse_str(&eql) {
                        Ok(expr) => expr,
                        Err(err) => {
                            return PaletteEvent::Error(err.to_string())
                        }
                    };
                    PalettePage::new(vec![
                        PaletteItem::new(
                            "GLTF",
                            "",
                            {
                                let eql = eql.clone();
                                let expr = expr.clone();
                                move |_: In<String>| {
                                    let eql = eql.clone();
                                    let expr = expr.clone();
                                    PalettePage::new(vec![
                                        PaletteItem::new(
                                            LabelSource::placeholder("Enter GLTF path"),
                                            "Enter path to GLTF file for the 3D object visualization",
                                            move |In(gltf_path): In<String>,
                                                  mut commands: Commands,
                                                  eql_ctx: Res<EqlContext>,
                                                  mut material_assets: ResMut<Assets<StandardMaterial>>,
                                                  mut mesh_assets: ResMut<Assets<bevy::prelude::Mesh>>,
                                                  mut mat3_material_assets: ResMut<Assets<bevy_mat3_material::Mat3Material>>,
                                                  assets: Res<AssetServer>,
                                                  geo_context: Res<GeoContext>,
                                                  connection_addr: Option<Res<ConnectionAddr>>
                                                | {
                                                let obj = impeller2_wkt::Object3DMesh::glb(gltf_path.trim());
                                                let connection_addr = connection_addr.as_ref().map(|addr| addr.0);

                                                let _ = crate::object_3d::create_object_3d_entity(
                                                    &mut commands,
                                                    Object3D { eql: eql.clone(), mesh: obj, icon: None, thrusters: Vec::new(), mesh_visibility_range: None, frame: None, node_id: Default::default() },
                                                    expr.clone(),
                                                    &eql_ctx.0,
                                                    &mut material_assets,
                                                    &mut mesh_assets,
                    &mut mat3_material_assets,
                    &assets,
                    &geo_context,
                    connection_addr,
                );

                                                PaletteEvent::Exit
                                            },
                                        ).default()
                                    ])
                                    .prompt("Enter GLTF path")
                                    .into_event()
                                }
                            },
                        ),
                        PaletteItem::new(
                            "Sphere",
                            "",
                            {
                                let eql = eql.clone();
                                let expr = expr.clone();
                                move |_: In<String>| {
                                    let eql = eql.clone();
                                    let expr = expr.clone();
                                    PalettePage::new(vec![
                                        PaletteItem::new(
                                            LabelSource::placeholder("Enter radius (default: 1.0)"),
                                            "Enter the radius for the sphere",
                                            move |In(radius_str): In<String>| {
                                                let radius = radius_str.trim().parse::<f32>().unwrap_or(1.0);
                                                create_object_3d_with_color(
                                                    eql.clone(),
                                                    expr.clone(),
                                                    Mesh::sphere(radius)
                                                )
                                            },
                                        ).default()
                                    ])
                                    .prompt("Enter sphere radius")
                                    .into_event()
                                }
                            },
                        ),
                        PaletteItem::new(
                            "Cylinder",
                            "",
                            {
                                let eql = eql.clone();
                                let expr = expr.clone();
                                move |_: In<String>| {
                                    let eql = eql.clone();
                                    let expr = expr.clone();
                                    PalettePage::new(vec![
                                        PaletteItem::new(
                                            LabelSource::placeholder("Enter radius and height (default: 1.0 2.0)"),
                                            "Enter the radius and height for the cylinder",
                                            move |In(dimensions_str): In<String>| {
                                                let parts: Vec<f32> = dimensions_str
                                                    .split_whitespace()
                                                    .filter_map(|s| s.parse().ok())
                                                    .collect();

                                                let (radius, height) = match parts.as_slice() {
                                                    [r, h] => (*r, *h),
                                                    [r] => (*r, *r * 2.0),
                                                    _ => (1.0, 2.0),
                                                };

                                                create_object_3d_with_color(
                                                    eql.clone(),
                                                    expr.clone(),
                                                    Mesh::Cylinder { radius, height }
                                                )
                                            },
                                        ).default()
                                    ])
                                    .prompt("Enter cylinder dimensions")
                                    .into_event()
                                }
                            },
                        ),
                        PaletteItem::new(
                            "Cuboid",
                            "",
                            {
                                let eql = eql.clone();
                                let expr = expr.clone();
                                move |_: In<String>| {
                                    let eql = eql.clone();
                                    let expr = expr.clone();
                                    PalettePage::new(vec![
                                        PaletteItem::new(
                                            LabelSource::placeholder("Enter dimensions (x y z, default: 1 1 1)"),
                                            "Enter the dimensions for the cuboid (width height depth)",
                                            move |In(dimensions): In<String>| {
                                                let parts: Vec<f32> = dimensions
                                                    .split_whitespace()
                                                    .filter_map(|s| s.parse().ok())
                                                    .collect();

                                                let (x, y, z) = match parts.as_slice() {
                                                    [x, y, z] => (*x, *y, *z),
                                                    [x] => (*x, *x, *x),
                                                    _ => (1.0, 1.0, 1.0),
                                                };

                                                create_object_3d_with_color(
                                                    eql.clone(),
                                                    expr.clone(),
                                                    Mesh::cuboid(x, y, z)
                                                )
                                            },
                                        ).default()
                                    ])
                                    .prompt("Enter cuboid dimensions")
                                    .into_event()
                                }
                            },
                        ),
                        PaletteItem::new(
                            "Plane",
                            "",
                            {
                                let eql = eql.clone();
                                let expr = expr.clone();
                                move |_: In<String>| {
                                    let eql = eql.clone();
                                    let expr = expr.clone();
                                    PalettePage::new(vec![
                                        PaletteItem::new(
                                            LabelSource::placeholder(
                                                "Enter width and depth (default: 10 10)",
                                            ),
                                            "Enter the width and depth for the plane",
                                            move |In(dimensions): In<String>| {
                                                let parts: Vec<f32> = dimensions
                                                    .split_whitespace()
                                                    .filter_map(|s| s.parse().ok())
                                                    .collect();

                                                let (width, depth) = match parts.as_slice() {
                                                    [w, d] => (*w, *d),
                                                    [w] => (*w, *w),
                                                    _ => (10.0, 10.0),
                                                };

                                                create_object_3d_with_color(
                                                    eql.clone(),
                                                    expr.clone(),
                                                    Mesh::plane(width, depth)
                                                )
                                            },
                                        )
                                        .default()
                                    ])
                                    .prompt("Enter plane size")
                                    .into_event()
                                }
                            },
                        ),
                    ])
                    .prompt("Choose 3D object visualization type")
                    .into_event()
                },
            ).default()
        ])
        .prompt("Enter EQL expression for 3D object positioning")
        .into_event()
    })
}

pub fn create_tiles(tile_id: TileId) -> PalettePage {
    PalettePage::new(vec![
        create_graph(Some(tile_id)),
        create_action(Some(tile_id)),
        create_monitor(Some(tile_id)),
        create_viewport(Some(tile_id)),
        create_query_table(Some(tile_id)),
        create_query_plot(Some(tile_id)),
        create_video_stream(Some(tile_id)),
        create_log_stream(Some(tile_id)),
        create_schematic_tree(Some(tile_id)),
        create_data_overview(Some(tile_id)),
    ])
}

impl Default for PalettePage {
    fn default() -> PalettePage {
        PalettePage::new(vec![
            PaletteItem::new(
                "Toggle Wireframe",
                VIEWPORT_LABEL,
                |_: In<String>, mut wireframe: ResMut<WireframeConfig>| {
                    wireframe.global = !wireframe.global;
                    PaletteEvent::Exit
                },
            ),
            PaletteItem::new(
                "Toggle HDR",
                VIEWPORT_LABEL,
                |_: In<String>, mut hdr: ResMut<HdrEnabled>| {
                    hdr.0 = !hdr.0;
                    PaletteEvent::Exit
                },
            ),
            PaletteItem::new(
                "Toggle Grid",
                VIEWPORT_LABEL,
                |_: In<String>, mut grid_visibility: Query<&mut Visibility, With<InfiniteGrid>>| {
                    let all_hidden = grid_visibility
                        .iter()
                        .all(|grid_visibility| grid_visibility == Visibility::Hidden);

                    for mut grid_visibility in grid_visibility.iter_mut() {
                        *grid_visibility = if all_hidden {
                            Visibility::Visible
                        } else {
                            Visibility::Hidden
                        };
                    }

                    PaletteEvent::Exit
                },
            ),
            reset_cameras(),
            PaletteItem::new(
                "Toggle Recording",
                SIMULATION_LABEL,
                |_: In<String>, packet_tx: Res<PacketTx>, mut simulating: ResMut<IsRecording>| {
                    simulating.0 = !simulating.0;
                    packet_tx.send_msg(SetDbConfig {
                        recording: Some(simulating.0),
                        ..Default::default()
                    });
                    PaletteEvent::Exit
                },
            ),
            set_playback_speed(),
            goto_tick(),
            fix_current_time_range(),
            set_time_range_behavior(),
            create_window(),
            create_graph(None),
            create_action(None),
            create_monitor(None),
            create_viewport(None),
            create_query_table(None),
            create_query_plot(None),
            create_video_stream(None),
            create_log_stream(None),
            create_schematic_tree(None),
            create_data_overview(None),
            create_3d_object(),
            save_schematic(),
            save_schematic_as(),
            open_schematic(),
            clear_schematic(),
            skybox_menu(),
            set_color_scheme_mode(),
            set_color_scheme(),
            PaletteItem::new("Documentation", HELP_LABEL, |_: In<String>| {
                let _ = opener::open("https://docs.elodin.systems");
                PaletteEvent::Exit
            })
            .icon(PaletteIcon::Link),
            PaletteItem::new("Release Notes", HELP_LABEL, |_: In<String>| {
                let _ = opener::open("https://docs.elodin.systems/updates/changelog");
                PaletteEvent::Exit
            })
            .icon(PaletteIcon::Link),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2_kdl::FromKdl;
    use impeller2_wkt::Schematic;
    use std::path::PathBuf;

    fn parse_saved_schematic(kdl: &str) -> Schematic {
        Schematic::from_kdl(kdl).expect("saved KDL should parse")
    }

    #[test]
    fn save_kdl_uses_active_skybox_from_cache() {
        let schematic = CurrentSchematic(Schematic::default());
        let mut cache = SkyboxCache::empty(PathBuf::from("manifest.ron"));
        cache.active = Some("mont_blanc_france".to_string());

        let kdl = root_schematic_for_save(&schematic, Some(&cache)).to_kdl();
        let parsed = parse_saved_schematic(&kdl);

        assert_eq!(
            parsed.skybox.as_ref().map(|skybox| skybox.name.as_str()),
            Some("mont_blanc_france")
        );
    }

    #[test]
    fn save_kdl_active_skybox_replaces_stale_schematic_skybox() {
        let schematic = CurrentSchematic(Schematic {
            skybox: Some(SkyboxConfig {
                name: "desert_night".to_string(),
            }),
            ..Default::default()
        });
        let mut cache = SkyboxCache::empty(PathBuf::from("manifest.ron"));
        cache.active = Some("mont_blanc_france".to_string());

        let kdl = root_schematic_for_save(&schematic, Some(&cache)).to_kdl();
        let parsed = parse_saved_schematic(&kdl);

        assert_eq!(
            parsed.skybox.as_ref().map(|skybox| skybox.name.as_str()),
            Some("mont_blanc_france")
        );
    }

    #[test]
    fn save_kdl_clears_stale_schematic_skybox_when_cache_has_no_active_skybox() {
        let schematic = CurrentSchematic(Schematic {
            skybox: Some(SkyboxConfig {
                name: "grand_canyon".to_string(),
            }),
            ..Default::default()
        });
        let cache = SkyboxCache::empty(PathBuf::from("manifest.ron"));

        let kdl = root_schematic_for_save(&schematic, Some(&cache)).to_kdl();
        let parsed = parse_saved_schematic(&kdl);

        assert!(parsed.skybox.is_none());
    }

    #[test]
    fn save_kdl_preserves_schematic_skybox_when_cache_is_unavailable() {
        let schematic = CurrentSchematic(Schematic {
            skybox: Some(SkyboxConfig {
                name: "grand_canyon".to_string(),
            }),
            ..Default::default()
        });

        let kdl = root_schematic_for_save(&schematic, None).to_kdl();
        let parsed = parse_saved_schematic(&kdl);

        assert_eq!(
            parsed.skybox.as_ref().map(|skybox| skybox.name.as_str()),
            Some("grand_canyon")
        );
    }
}
