use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use bevy::{
    asset::{AssetServer, Assets},
    ecs::{
        query::With,
        system::{Commands, InRef, IntoSystem, Query, Res, ResMut, System},
        world::World,
    },
    log::{error, info, warn},
    pbr::{StandardMaterial, wireframe::WireframeConfig},
    prelude::{Deref, DerefMut, In, Resource},
    camera::visibility::Visibility,
};
use bevy_infinite_grid::InfiniteGrid;
use egui_tiles::TileId;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2::types::{Timestamp, msg_id};
use impeller2_bevy::{CommandsExt, ComponentPathRegistry, CurrentStreamId, EntityMap, PacketTx};
use impeller2_kdl::{
    ToKdl,
    env::{schematic_dir_or_cwd, schematic_file},
};
use impeller2_wkt::{
    ArchiveFormat, ArchiveSaved, ComponentPath, ComponentValue, CurrentTimestamp, DbConfig,
    EarliestTimestamp, ErrorResponse, IsRecording, LastUpdated, Material, Mesh, Object3D,
    SaveArchive, SetDbConfig, SetStreamState, SimulationTimeStep,
};
use miette::IntoDiagnostic;
use nox::ArrayBuf;

use crate::{
    EqlContext, Offset, SelectedTimeRange, TimeRangeBehavior, TimeRangeError,
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        HdrEnabled, Paused, colors,
        command_palette::CommandPaletteState,
        plot::{GraphBundle, default_component_values},
        schematic::{
            CurrentSchematic, CurrentSecondarySchematics, LoadSchematicParams,
            SchematicLiveReloadRx, load_schematic_file,
        },
        tiles,
        timeline::{StreamTickOrigin, timeline_slider::UITick},
    },
};

/// Stores a path to the last directory used in the schematic load dialog.
///
/// I would have preferred for this to be a `Local<Option<PathBuf>>`, but the
/// `BoxedSystem` that PaletteItem stores is regenerated every time.
#[derive(Debug, Default, Resource, Deref, DerefMut)]
struct DialogLastPath(Option<PathBuf>);

pub(crate) fn plugin(app: &mut bevy::app::App) {
    app.init_resource::<DialogLastPath>();
}

pub struct PalettePage {
    items: Vec<PaletteItem>,
    pub label: Option<String>,
    initialized: bool,
    pub prompt: Option<String>,
}

impl PalettePage {
    pub fn new(items: Vec<PaletteItem>) -> Self {
        Self {
            items,
            initialized: false,
            label: None,
            prompt: None,
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
        items.sort_by(|a, b| b.score.cmp(&a.score));
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

fn target_tile_state_mut(
    windows: &mut tiles::WindowManager,
    target: Option<tiles::SecondaryWindowId>,
) -> Option<&mut tiles::TileState> {
    match target {
        Some(id) => windows
            .get_secondary_mut(id)
            .map(|state| &mut state.tile_state),
        None => Some(windows.main_mut()),
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
                                                      mut windows: ResMut<tiles::WindowManager>,
                                                      palette_state: Res<CommandPaletteState>| {
                                                    let Some(tile_state) = target_tile_state_mut(
                                                        &mut windows,
                                                        palette_state.target_window,
                                                    ) else {
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
                                          mut windows: ResMut<tiles::WindowManager>,
                                          palette_state: Res<CommandPaletteState>| {
                                        let Some(tile_state) = target_tile_state_mut(
                                            &mut windows,
                                            palette_state.target_window,
                                        ) else {
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
                      query: Query<&ComponentValue>,
                      entity_map: Res<EntityMap>,
                      mut render_layer_alloc: ResMut<RenderLayerAlloc>,
                      mut windows: ResMut<tiles::WindowManager>,
                      path_reg: Res<ComponentPathRegistry>,
                      palette_state: Res<CommandPaletteState>| {
                    let Some(tile_state) =
                        target_tile_state_mut(&mut windows, palette_state.target_window)
                    else {
                        return PaletteEvent::Error("Secondary window unavailable".to_string());
                    };
                    if let Some(component) = &part.component {
                        let component_id = component.id;
                        let Some(entity) = entity_map.get(&component_id) else {
                            return PaletteEvent::Exit;
                        };
                        let Ok(value) = query.get(*entity) else {
                            return PaletteEvent::Exit;
                        };

                        let values = default_component_values(&component_id, value);

                        let component_path = path_reg
                            .get(&component_id)
                            .cloned()
                            .unwrap_or_else(|| ComponentPath::from_name(&component.name));

                        let components =
                            BTreeMap::from_iter(std::iter::once((component_path, values.clone())));
                        let bundle = GraphBundle::new(
                            &mut render_layer_alloc,
                            components,
                            "Graph".to_string(),
                        );
                        tile_state.create_graph_tile(tile_id, bundle);
                        PaletteEvent::Exit
                    } else {
                        PalettePage::new(graph_parts(&part.children, tile_id)).into_event()
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
                      mut windows: ResMut<tiles::WindowManager>,
                      palette_state: Res<CommandPaletteState>| {
                    let Some(tile_state) =
                        target_tile_state_mut(&mut windows, palette_state.target_window)
                    else {
                        return PaletteEvent::Error("Secondary window unavailable".to_string());
                    };
                    if let Some(component) = &part.component {
                        tile_state.create_monitor_tile(component.name.clone(), tile_id);
                        PaletteEvent::Exit
                    } else {
                        PalettePage::new(monitor_parts(&part.children, tile_id)).into_event()
                    }
                },
            )
        })
        .collect()
}

fn toggle_body_axes() -> PaletteItem {
    PaletteItem::new("Toggle Body Axes", VIEWPORT_LABEL, |_: In<String>| {
        // TODO: This functionality needs to be updated once BodyAxes is migrated from EntityId to ComponentId
        // For now, return an empty page
        PalettePage::new(vec![])
            .prompt("Body axes functionality is temporarily disabled during refactor")
            .into_event()
    })
}

pub fn create_viewport(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Viewport",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
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
                      mut windows: ResMut<tiles::WindowManager>,
                      mut palette_state: ResMut<CommandPaletteState>| {
                    let title_opt = if title.trim().is_empty() {
                        None
                    } else {
                        Some(title.trim().to_string())
                    };
                    let id = windows.create_secondary_window(title_opt);
                    palette_state.target_window = Some(id);
                    PaletteEvent::Exit
                },
            )
            .default(),
        ])
        .prompt("Enter a title for the new window")
        .into()
    })
}

pub fn create_query_table(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Query Table",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
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
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_query_plot_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_hierarchy(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Hierarchy",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_hierarchy_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_inspector(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Inspector",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_inspector_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_schematic_tree(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Schematic Tree",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_tree_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_dashboard(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Dashboard",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_dashboard_tile(Default::default(), "Dashboard".to_string(), tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_sidebars() -> PaletteItem {
    PaletteItem::new(
        "Create Sidebars",
        TILES_LABEL,
        move |_: In<String>,
              mut windows: ResMut<tiles::WindowManager>,
              palette_state: Res<CommandPaletteState>| {
            let Some(tile_state) = target_tile_state_mut(&mut windows, palette_state.target_window)
            else {
                return PaletteEvent::Error("Secondary window unavailable".to_string());
            };
            tile_state.create_sidebars_layout();
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
                          mut windows: ResMut<tiles::WindowManager>,
                          palette_state: Res<CommandPaletteState>| {
                        let Some(tile_state) =
                            target_tile_state_mut(&mut windows, palette_state.target_window)
                        else {
                            return PaletteEvent::Error("Secondary window unavailable".to_string());
                        };
                        let msg_name = msg_name.trim();
                        let label = format!("Video Stream {}", msg_name);
                        tile_state.create_video_stream_tile(msg_id(msg_name), label, tile_id);
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
            0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0,
            50.0, 100.0,
        ];
        let next_page = PalettePage::new(
            speeds
                .into_iter()
                .map(|speed| {
                    PaletteItem::new(
                        speed.to_string(),
                        "SPEED".to_string(),
                        move |_: In<String>,
                              packet_tx: Res<PacketTx>,
                              stream_id: Res<CurrentStreamId>| {
                            packet_tx.send_msg(SetStreamState {
                                id: stream_id.0,
                                playing: None,
                                timestamp: None,
                                time_step: Some(Duration::from_secs_f64(speed / 60.0)),
                                frequency: None,
                            });
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
                      stream_id: Res<CurrentStreamId>,
                      packet_tx: Res<PacketTx>,
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
                    paused.0 = true;
                    current_tick.0 = timestamp;
                    ui_tick.0 = timestamp.0;
                    if parsed_tick == 0 {
                        tick_origin.request_rebase();
                    }
                    packet_tx.send_msg(SetStreamState::rewind(**stream_id, timestamp));

                    PaletteEvent::Exit
                },
            )
            .default(),
        ])
        .prompt("Enter the tick to jump to")
        .into_event()
    })
}

pub fn save_schematic_as() -> PaletteItem {
    PaletteItem::new(
        "Save Schematic As...",
        PRESETS_LABEL,
        |_name: In<String>| PalettePage::new(vec![save_schematic_inner()]).into_event(),
    )
}

pub fn save_schematic() -> PaletteItem {
    PaletteItem::new(
        "Save Schematic",
        PRESETS_LABEL,
        |_name: In<String>,
         db_config: Res<DbConfig>,
         schematic: Res<CurrentSchematic>,
         secondary: Res<CurrentSecondarySchematics>,
         mut live_reload: ResMut<SchematicLiveReloadRx>| {
            match db_config.schematic_path() {
                Some(path) => {
                    live_reload.guard_for(Duration::from_millis(2000));
                    let kdl = schematic.0.to_kdl();
                    let path = Path::new(path).with_extension("kdl");
                    let dest = schematic_file(&path);
                    if let Err(e) = std::fs::write(&dest, kdl) {
                        error!(?e, "saving schematic to {:?}", dest.display());
                    } else {
                        info!("saved schematic to {:?}", dest.display());
                        live_reload.ignore_path(dest.clone());
                        live_reload.record_loaded(&dest);
                        let base_dir = dest.parent().unwrap_or_else(|| Path::new("."));
                        write_secondary_schematics(base_dir, &secondary, &mut live_reload);
                    }
                    PaletteEvent::Exit
                }
                None => PalettePage::new(vec![save_schematic_inner()]).into_event(),
            }
        },
    )
}

pub fn save_schematic_db() -> PaletteItem {
    PaletteItem::new(
        "Save Schematic To DB",
        PRESETS_LABEL,
        |_name: In<String>, tx: Res<PacketTx>, schematic: Res<CurrentSchematic>| {
            let kdl = schematic.0.to_kdl();
            tx.send_msg(SetDbConfig {
                metadata: [("schematic.content".to_string(), kdl)]
                    .into_iter()
                    .collect(),
                ..Default::default()
            });
            PaletteEvent::Exit
        },
    )
}

fn save_db_native_prompt_item() -> PaletteItem {
    PaletteItem::new(
        LabelSource::placeholder("Enter a name for the Save DB directory"),
        "",
        |In(input): In<String>, mut commands: Commands| {
            let trimmed = input.trim();
            if trimmed.is_empty() {
                return PaletteEvent::Error("Directory name cannot be empty".to_string());
            }
            let raw_path = PathBuf::from(trimmed);
            let mut path_is_absolute =
                raw_path.is_absolute() || trimmed.starts_with('/') || trimmed.starts_with('\\');
            let mut last_component: Option<&std::ffi::OsStr> = None;
            for component in raw_path.components() {
                match component {
                    Component::Normal(part) => {
                        if part.is_empty() {
                            return PaletteEvent::Error(
                                "Path contains an empty segment".to_string(),
                            );
                        }
                        last_component = Some(part);
                    }
                    Component::CurDir => {
                        return PaletteEvent::Error(
                            "`.` segments are not allowed in the path".to_string(),
                        );
                    }
                    Component::ParentDir => {
                        return PaletteEvent::Error(
                            "Path may not traverse outside the workspace".to_string(),
                        );
                    }
                    Component::Prefix(_) | Component::RootDir => {
                        path_is_absolute = true;
                    }
                }
            }
            let Some(_name) = last_component.and_then(|c| c.to_str()) else {
                return PaletteEvent::Error("Invalid directory name".to_string());
            };
            let request_path = if path_is_absolute {
                raw_path.clone()
            } else {
                let cwd = match std::env::current_dir() {
                    Ok(cwd) => cwd,
                    Err(err) => {
                        error!(?err, "Failed to resolve workspace directory");
                        return PaletteEvent::Error(
                            "Failed to resolve workspace directory".to_string(),
                        );
                    }
                };
                let target = cwd.join(&raw_path);
                if target.exists() {
                    return PaletteEvent::Error("Directory already exists".to_string());
                }
                if let Err(err) = target.strip_prefix(&cwd) {
                    error!(?err, "Save path escaped workspace");
                    return PaletteEvent::Error(
                        "Path must stay within the workspace directory".to_string(),
                    );
                }
                raw_path.clone()
            };
            commands.send_req_reply(
                SaveArchive {
                    path: request_path,
                    format: ArchiveFormat::Native,
                },
                |In(res): In<Result<ArchiveSaved, ErrorResponse>>,
                 mut palette_state: ResMut<CommandPaletteState>| {
                    match res {
                        Ok(saved) => {
                            let display_path = std::env::current_dir()
                                .ok()
                                .and_then(|cwd| {
                                    saved
                                        .path
                                        .strip_prefix(cwd)
                                        .ok()
                                        .map(|p| format!("{}", p.display()))
                                })
                                .unwrap_or_else(|| saved.path.display().to_string());
                            info!(path = %display_path, "Saved DB snapshot");
                    }
                    Err(err) => {
                        warn!(?err, "Failed to save DB snapshot");
                        let message = if err.description.contains("Serde Deserialization Error")
                        {
                            "Connected database does not support native DB snapshots. Please update elodin-db and try again.".to_string()
                        } else {
                            err.description.clone()
                        };
                        palette_state.show = true;
                        palette_state.filter.clear();
                        palette_state.input_focus = true;
                        palette_state.selected_index = 0;
                        palette_state.page_stack.clear();
                        palette_state.page_stack.push(save_db_native_prompt_page());
                        palette_state.auto_open_item = None;
                        palette_state.error = Some(message);
                    }
                }
                true
            },
        );
            PaletteEvent::Exit
        },
    )
    .default()
}

fn save_db_native_prompt_page() -> PalettePage {
    PalettePage::new(vec![save_db_native_prompt_item()]).prompt(
        "Enter a directory name within the workspace or an absolute path on the database host",
    )
}

pub fn save_db_native() -> PaletteItem {
    PaletteItem::new("Save DB", PRESETS_LABEL, |_name: In<String>| {
        save_db_native_prompt_page().into_event()
    })
}

pub fn clear_schematic() -> PaletteItem {
    PaletteItem::new(
        "Clear Schematic",
        PRESETS_LABEL,
        |_: In<String>, mut params: LoadSchematicParams, mut rx: ResMut<SchematicLiveReloadRx>| {
            params.load_schematic(&impeller2_wkt::Schematic::default(), None);
            rx.clear();
            PaletteEvent::Exit
        },
    )
}

pub fn save_schematic_inner() -> PaletteItem {
    PaletteItem::new(
        LabelSource::placeholder("Enter a name for the schematic"),
        "",
        move |In(name): In<String>,
              schematic: Res<CurrentSchematic>,
              secondary: Res<CurrentSecondarySchematics>,
              mut live_reload: ResMut<SchematicLiveReloadRx>| {
            let kdl = schematic.0.to_kdl();
            let path = PathBuf::from(name).with_extension("kdl");
            let dest = schematic_file(&path);
            live_reload.guard_for(Duration::from_millis(2000));
            if let Err(e) = std::fs::write(&dest, kdl) {
                error!(?e, "saving schematic");
            } else {
                info!("saved schematic to {:?}", dest.display());
                live_reload.ignore_path(dest.clone());
                let base_dir = dest.parent().unwrap_or_else(|| Path::new("."));
                write_secondary_schematics(base_dir, &secondary, &mut live_reload);
            }
            PaletteEvent::Exit
        },
    )
    .default()
}

fn write_secondary_schematics(
    base_dir: &Path,
    secondary: &CurrentSecondarySchematics,
    live_reload: &mut SchematicLiveReloadRx,
) {
    for entry in &secondary.0 {
        let dest = base_dir.join(&entry.file_name);
        if let Some(parent) = dest.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            error!(
                ?e,
                path = %dest.display(),
                "creating directory for secondary schematic"
            );
            continue;
        }

        let kdl = entry.schematic.to_kdl();
        if let Err(e) = std::fs::write(&dest, kdl) {
            error!(?e, path = %dest.display(), "saving secondary schematic");
        } else {
            info!(path = %dest.display(), "saved secondary schematic");
            live_reload.ignore_path(dest.clone());
            live_reload.record_loaded(&dest);
        }
    }
}

pub fn load_schematic() -> PaletteItem {
    PaletteItem::new("Load Schematic", PRESETS_LABEL, |_: In<String>| {
        let Ok(dir) = schematic_dir_or_cwd().inspect_err(|e| error!(?e, "getting schematic dir"))
        else {
            return PaletteEvent::Exit;
        };
        let elems = match std::fs::read_dir(&dir) {
            Ok(x) => x,
            Err(e) => {
                error!(?e, "reading schematic dir {:?}", dir.display());
                return PaletteEvent::Exit;
            }
        };

        let mut items = vec![load_schematic_picker()];
        let mut file = dir;
        for elem in elems {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if path.extension().and_then(|e| e.to_str()) != Some("kdl") {
                continue;
            }
            file.push(file_name);
            if let Some(item) = load_schematic_inner(&file) {
                items.push(item);
            }
            file.pop();
        }
        PalettePage::new(items).into_event()
    })
}

pub fn load_schematic_picker() -> PaletteItem {
    PaletteItem::new(
        "Use File Dialog",
        "",
        |_: In<String>,
         mut params: LoadSchematicParams,
         rx: ResMut<SchematicLiveReloadRx>,
         mut last_dir: ResMut<DialogLastPath>| {
            let mut dialog = rfd::FileDialog::new().add_filter("kdl", &["kdl"]);
            if let Some(dir) = last_dir.take().or_else(|| schematic_dir_or_cwd().ok()) {
                dialog = dialog.set_directory(dir);
            }
            if let Some(path) = dialog.pick_file() {
                **last_dir = path.parent().map(PathBuf::from);
                info!("PATH {:?}", path.display());
                if let Err(err) =
                    load_schematic_file(dbg!(&path), &mut params, rx).into_diagnostic()
                {
                    return PaletteEvent::Error(err.to_string());
                }
            }
            PaletteEvent::Exit
        },
    )
}

fn load_schematic_inner(path: &Path) -> Option<PaletteItem> {
    let name = path
        .file_name()
        .map(|name_os| name_os.to_string_lossy().into_owned());
    let path = PathBuf::from(path);
    name.map(|name| {
        PaletteItem::new(
            name,
            "",
            move |_: In<String>,
                  mut params: LoadSchematicParams,
                  rx: ResMut<SchematicLiveReloadRx>| {
                if let Err(err) = load_schematic_file(&path, &mut params, rx) {
                    PaletteEvent::Error(err.to_string())
                } else {
                    PaletteEvent::Exit
                }
            },
        )
    })
}

pub fn set_color_scheme() -> PaletteItem {
    PaletteItem::new("Set Color Scheme", PRESETS_LABEL, |_: In<String>| {
        let schemes = [
            ("DARK", &colors::DARK),
            ("LIGHT", &colors::LIGHT),
            ("CATPPUCINI LATTE", &colors::CATPPUCINI_LATTE),
            ("CATPPUCINI MOCHA", &colors::CATPPUCINI_MOCHA),
            ("CATPPUCINI MACCHIATO", &colors::CATPPUCINI_MACCHIATO),
        ];
        let mut items = vec![];
        for (name, schema) in schemes {
            items.push(PaletteItem::new(name, "", move |_: In<String>| {
                colors::set_schema(schema);
                PaletteEvent::Exit
            }));
        }
        PalettePage::new(items).into_event()
    })
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
                  assets: Res<AssetServer>| {
                let color_str = color_str.trim();
                let (r, g, b) =
                    parse_color(color_str, &eql_ctx.0, &entity_map, component_value_maps)
                        .unwrap_or((0.8, 0.8, 0.8));

                let mesh_source = impeller2_wkt::Object3DMesh::Mesh {
                    mesh: mesh.clone(),
                    material: Material::color(r, g, b),
                };

                crate::object_3d::create_object_3d_entity(
                    &mut commands,
                    Object3D {
                        eql: eql.clone(),
                        mesh: mesh_source,
                        aux: (),
                    },
                    expr.clone(),
                    &eql_ctx.0,
                    &mut material_assets,
                    &mut mesh_assets,
                    &assets,
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
    let expr = crate::object_3d::compile_eql_expr(expr);
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
                                                  assets: Res<AssetServer>| {
                                                let obj = impeller2_wkt::Object3DMesh::Glb(gltf_path.trim().to_string());

                                                crate::object_3d::create_object_3d_entity(
                                                    &mut commands,
                                                    Object3D { eql: eql.clone(), mesh: obj, aux: () },
                                                    expr.clone(),
                                                    &eql_ctx.0,
                                                    &mut material_assets,
                                                    &mut mesh_assets,
                                                    &assets,
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
        create_hierarchy(Some(tile_id)),
        create_schematic_tree(Some(tile_id)),
        create_dashboard(Some(tile_id)),
        create_inspector(Some(tile_id)),
        create_sidebars(),
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
            toggle_body_axes(),
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
            create_hierarchy(None),
            create_inspector(None),
            create_schematic_tree(None),
            create_dashboard(None),
            create_sidebars(),
            create_3d_object(),
            save_db_native(),
            save_schematic(),
            save_schematic_as(),
            save_schematic_db(),
            load_schematic(),
            clear_schematic(),
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
