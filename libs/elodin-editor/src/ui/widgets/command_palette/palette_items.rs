use std::{collections::BTreeMap, str::FromStr, time::Duration};

use bevy::{
    ecs::{
        entity::Entity,
        query::With,
        system::{Commands, InRef, IntoSystem, Query, Res, ResMut, System},
        world::World,
    },
    pbr::wireframe::WireframeConfig,
    prelude::In,
    render::view::Visibility,
};
use bevy_infinite_grid::InfiniteGrid;
use egui_tiles::TileId;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2::types::{ComponentId, msg_id};
use impeller2_bevy::{ComponentMetadataRegistry, CurrentStreamId, PacketTx};
use impeller2_wkt::{BodyAxes, EntityMetadata, IsRecording, SetDbSettings, SetStreamState};

use crate::{
    Offset, SelectedTimeRange, TimeRangeBehavior,
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        self, EntityData, HdrEnabled, colors,
        tiles::{self, SyncViewportParams},
        widgets::plot::{GraphBundle, default_component_values},
    },
};

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
            LabelSource::System(system) => system.run(filter, world),
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
                                                move |In(msg): In<String>, mut tile_state: ResMut<tiles::TileState>| {
                                                    tile_state.create_action_tile(
                                                        msg_label.clone(),
                                                        format!("client:send_msg({name:?}, {msg})"),
                                                        tile_id,
                                                    );
                                                    PaletteEvent::Exit
                                                },
                                            ).default()])
                                            .into()
                                        },
                                    ).default()])
                                    .into()
                                }),
                                PaletteItem::new(
                                    LabelSource::placeholder("Enter a lua command (i.e client:send_table)"),
                                    "Enter a custom lua command",
                                    move |lua: In<String>, mut tile_state: ResMut<tiles::TileState>| {
                                        tile_state.create_action_tile(label.clone(), lua.0, tile_id);
                                        PaletteEvent::Exit
                                    },
                                )
                                .default(),
                            ]).prompt("Enter a lua command to send")
                            .into()
                        },
                    )
                    .default(),
                ])
                .prompt("Enter a label for the action button")
                .into()
    })
}

pub fn create_graph(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Graph",
        TILES_LABEL,
        move |_: In<String>, entities: Query<EntityData>| {
            PalettePage::new(
                entities
                    .iter()
                    .map(|(_, entity, _, metadata)| graph_entity_item(metadata, entity, tile_id))
                    .collect(),
            )
            .prompt("Select an entity to graph")
            .into()
        },
    )
}

fn graph_entity_item(
    entity_metadata: &EntityMetadata,
    entity: Entity,
    tile_id: Option<TileId>,
) -> PaletteItem {
    PaletteItem::new(
        entity_metadata.name.clone(),
        "Entities",
        move |_: In<String>,
              entities: Query<EntityData>,
              metadata: Res<ComponentMetadataRegistry>| {
            let (_, entity, components, entity_metadata) = entities.get(entity).unwrap();
            let items = components
                .0
                .keys()
                .filter_map(|id| metadata.get_metadata(id).cloned())
                .map(move |item| {
                    PaletteItem::new(
                        item.name.clone(),
                        "Components",
                        move |_: In<String>,
                              entities: Query<EntityData>,
                              mut render_layer_alloc: ResMut<RenderLayerAlloc>,
                              mut tile_state: ResMut<tiles::TileState>| {
                            let component_id = item.component_id;
                            let Ok((entity_id, _, component_value_map, _)) = entities.get(entity)
                            else {
                                return PaletteEvent::Exit;
                            };

                            let component_value = component_value_map.0.get(&component_id).unwrap();
                            let values =
                                default_component_values(entity_id, &component_id, component_value);
                            let entities = BTreeMap::from_iter(std::iter::once((
                                entity_id.to_owned(),
                                BTreeMap::from_iter(std::iter::once((
                                    component_id,
                                    values.clone(),
                                ))),
                            )));
                            let bundle = GraphBundle::new(
                                &mut render_layer_alloc,
                                entities,
                                "Graph".to_string(),
                            );
                            tile_state.create_graph_tile(tile_id, bundle);
                            PaletteEvent::Exit
                        },
                    )
                })
                .collect();
            PaletteEvent::NextPage {
                prev_page_label: Some(entity_metadata.name.clone()),
                next_page: PalettePage::new(items).prompt("Select a component to graph"),
            }
        },
    )
}

pub fn create_monitor(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Monitor",
        TILES_LABEL,
        move |_: In<String>, entities: Query<EntityData>| {
            PalettePage::new(
                entities
                    .iter()
                    .map(|(_, entity, _, metadata)| monitor_entity_item(metadata, entity, tile_id))
                    .collect(),
            )
            .prompt("Select an entity to monitor")
            .into()
        },
    )
}

fn monitor_entity_item(
    entity_metadata: &EntityMetadata,
    entity: Entity,
    tile_id: Option<TileId>,
) -> PaletteItem {
    PaletteItem::new(
        entity_metadata.name.clone(),
        "Entities",
        move |_: In<String>,
              entities: Query<EntityData>,
              metadata: Res<ComponentMetadataRegistry>| {
            let (_, entity, components, entity_metadata) = entities.get(entity).unwrap();
            let items = components
                .0
                .keys()
                .filter_map(|id| metadata.get_metadata(id).cloned())
                .map(move |item| {
                    PaletteItem::new(
                        item.name.clone(),
                        "Components",
                        move |_: In<String>,
                              entities: Query<EntityData>,
                              mut tile_state: ResMut<tiles::TileState>| {
                            let component_id = item.component_id;
                            let Ok((entity_id, _, _, _)) = entities.get(entity) else {
                                return PaletteEvent::Exit;
                            };
                            tile_state.create_monitor_tile(*entity_id, component_id, tile_id);
                            PaletteEvent::Exit
                        },
                    )
                })
                .collect();
            PaletteEvent::NextPage {
                prev_page_label: Some(entity_metadata.name.clone()),
                next_page: PalettePage::new(items).prompt("Select a component to monitor"),
            }
        },
    )
}

fn toggle_body_axes() -> PaletteItem {
    PaletteItem::new(
        "Toggle Body Axes",
        VIEWPORT_LABEL,
        |_: In<String>, entities: Query<EntityData>| {
            PalettePage::new(
                entities
                    .iter()
                    .filter_map(|(&entity_id, _, value_map, metadata)| {
                        if value_map.0.contains_key(&ComponentId::new("world_pos")) {
                            Some(PaletteItem::new(
                                metadata.name.clone(),
                                "Entities",
                                move |_: In<String>, mut commands: Commands, query: Query<(Entity, &BodyAxes)>| {
                                    let mut found = false;
                                    for (e, axes) in query.iter() {
                                        if axes.entity_id == entity_id {
                                            found = true;
                                            commands.entity(e).remove::<BodyAxes>();
                                        }
                                    }
                                    if !found {
                                        commands.spawn(BodyAxes {
                                            entity_id,
                                            scale: 1.0,
                                        });
                                    }
                                    PaletteEvent::Exit
                                },
                            ))
                        } else {
                            None
                        }
                    })
                    .collect(),
            )
            .into()
        },
    )
}

pub fn create_viewport(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Viewport",
        TILES_LABEL,
        move |_: In<String>, entities: Query<EntityData>| {
            PalettePage::new(
                std::iter::once(PaletteItem::new(
                    "None",
                    "Entities",
                    move |_: In<String>, mut tile_state: ResMut<tiles::TileState>| {
                        tile_state.create_viewport_tile(None, tile_id);
                        PaletteEvent::Exit
                    },
                ))
                .chain(
                    entities
                        .iter()
                        .filter_map(|(&entity_id, _, value_map, metadata)| {
                            if value_map.0.contains_key(&ComponentId::new("world_pos")) {
                                Some(PaletteItem::new(
                                    metadata.name.clone(),
                                    "Entities",
                                    move |_: In<String>, mut tile_state: ResMut<tiles::TileState>| {
                                        tile_state.create_viewport_tile(Some(entity_id), tile_id);
                                        PaletteEvent::Exit
                                    },
                                ))
                            } else {
                                None
                            }
                        }),
                )
                .collect(),
            )
            .prompt("Select the entity the viewport will track")
            .into()
        },
    )
}

pub fn create_sql(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create SQL Table",
        TILES_LABEL,
        move |_: In<String>, mut tile_state: ResMut<tiles::TileState>| {
            tile_state.create_sql_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_hierarchy(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Hierarchy",
        TILES_LABEL,
        move |_: In<String>, mut tile_state: ResMut<tiles::TileState>| {
            tile_state.create_hierarchy_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_inspector(tile_id: Option<TileId>) -> PaletteItem {
    PaletteItem::new(
        "Create Inspector",
        TILES_LABEL,
        move |_: In<String>, mut tile_state: ResMut<tiles::TileState>| {
            tile_state.create_inspector_tile(tile_id);
            PaletteEvent::Exit
        },
    )
}

pub fn create_sidebars() -> PaletteItem {
    PaletteItem::new(
        "Create Sidebars",
        TILES_LABEL,
        move |_: In<String>, mut tile_state: ResMut<tiles::TileState>| {
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
                    move |In(msg_name): In<String>, mut tile_state: ResMut<tiles::TileState>| {
                        let msg_name = msg_name.trim();
                        let label = format!("Video Stream {}", msg_name);
                        tile_state.create_video_stream_tile(msg_id(msg_name), label, tile_id);
                        PaletteEvent::Exit
                    },
                )
                .default(),
            ])
            .into()
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
                            move |end_str: In<String>, mut behavior: ResMut<TimeRangeBehavior>| {
                                let Ok(end_offset) = Offset::from_str(&end_str.0) else {
                                    return PaletteEvent::Error(format!(
                                        "Invalid end offset format: {}",
                                        end_str.0
                                    ));
                                };

                                behavior.start = start_offset;
                                behavior.end = end_offset;
                                PaletteEvent::Exit
                            },
                        )
                        .default(),
                    ])
                    .prompt("Enter the end offset")
                    .into()
                },
            )
            .default(),
        ])
        .prompt("Enter the start offset")
        .into()
    })
}

pub fn save_preset() -> PaletteItem {
    PaletteItem::new("Save Preset", PRESETS_LABEL, |_name: In<String>| {
        PalettePage::new(vec![save_preset_inner()]).into()
    })
}

pub fn save_preset_inner() -> PaletteItem {
    PaletteItem::new(
        LabelSource::placeholder("Enter a name for the preset"),
        "",
        move |name: In<String>,
              sql_tables: Query<&ui::sql_table::SqlTable>,
              cameras: Query<ui::CameraQuery>,
              entity_id: Query<&impeller2::types::EntityId>,
              action_tiles: Query<&ui::actions::ActionTile>,
              graph_states: Query<&ui::widgets::plot::GraphState>,
              ui_state: Res<tiles::TileState>| {
            if let Some(tile_id) = ui_state.tree.root() {
                let panel = crate::ui::preset::tile_to_panel(
                    &sql_tables,
                    &cameras,
                    &entity_id,
                    &action_tiles,
                    &graph_states,
                    tile_id,
                    &ui_state,
                );
                let dirs = crate::dirs();
                let dir = dirs.data_dir().join("presets");
                let _ = std::fs::create_dir(&dir);
                let _ = std::fs::write(
                    dir.join(name.clone()),
                    serde_json::to_string(&panel).unwrap(),
                );
            }
            PaletteEvent::Exit
        },
    )
    .default()
}

pub fn load_preset() -> PaletteItem {
    PaletteItem::new("Load Preset", PRESETS_LABEL, |_: In<String>| {
        let dirs = crate::dirs();
        let dir = dirs.data_dir().join("presets");
        let Ok(elems) = std::fs::read_dir(dir) else {
            return PaletteEvent::Exit;
        };

        let mut items = vec![];
        for elem in elems {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            items.push(load_preset_inner(file_name.to_string()))
        }
        PalettePage::new(items).into()
    })
}

pub fn load_preset_inner(name: String) -> PaletteItem {
    PaletteItem::new(
        name.clone(),
        "",
        move |_: In<String>,
              params: SyncViewportParams,
              mut selected_object: ResMut<ui::SelectedObject>| {
            let dirs = crate::dirs();
            let path = dirs.data_dir().join("presets").join(name.clone());
            if let Ok(json) = std::fs::read_to_string(path) {
                if let Ok(panel) = serde_json::from_str::<impeller2_wkt::Panel>(&json) {
                    let SyncViewportParams {
                        mut commands,
                        mut tile_state,
                        asset_server,
                        mut meshes,
                        mut materials,
                        mut render_layer_alloc,
                        entity_map,
                        entity_metadata,
                        grid_cell,
                        metadata_store,
                        mut hdr_enabled,
                        schema_reg,
                        ..
                    } = params;
                    render_layer_alloc.free_all();
                    tile_state.clear(&mut commands, &mut selected_object);
                    tiles::spawn_panel(
                        &panel,
                        None,
                        &asset_server,
                        &mut tile_state,
                        &mut meshes,
                        &mut materials,
                        &mut render_layer_alloc,
                        &mut commands,
                        &entity_map,
                        &entity_metadata,
                        &grid_cell,
                        &metadata_store,
                        &mut hdr_enabled,
                        &schema_reg,
                    );
                }
            }
            PaletteEvent::Exit
        },
    )
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
        PalettePage::new(items).into()
    })
}

pub fn create_tiles(tile_id: TileId) -> PalettePage {
    PalettePage::new(vec![
        create_graph(Some(tile_id)),
        create_action(Some(tile_id)),
        create_monitor(Some(tile_id)),
        create_viewport(Some(tile_id)),
        create_sql(Some(tile_id)),
        create_video_stream(Some(tile_id)),
        create_hierarchy(Some(tile_id)),
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
                    packet_tx.send_msg(SetDbSettings {
                        recording: Some(simulating.0),
                        ..Default::default()
                    });
                    PaletteEvent::Exit
                },
            ),
            set_playback_speed(),
            fix_current_time_range(),
            set_time_range_behavior(),
            create_graph(None),
            create_action(None),
            create_monitor(None),
            create_viewport(None),
            create_sql(None),
            create_video_stream(None),
            create_hierarchy(None),
            create_inspector(None),
            create_sidebars(),
            save_preset(),
            load_preset(),
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
