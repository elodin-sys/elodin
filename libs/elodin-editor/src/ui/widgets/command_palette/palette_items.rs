use std::collections::BTreeMap;

use bevy::{
    ecs::{
        entity::Entity,
        event::EventWriter,
        query::With,
        system::{Commands, IntoSystem, Query, Res, ResMut, System},
        world::World,
    },
    pbr::wireframe::WireframeConfig,
    render::view::Visibility,
};
use bevy_infinite_grid::InfiniteGrid;
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use impeller::{
    bevy::Simulating,
    query::MetadataStore,
    well_known::{BodyAxes, EntityMetadata},
    ComponentId, ControlMsg,
};

use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        tiles,
        widgets::plot::{default_component_values, GraphBundle},
        EntityData, HdrEnabled,
    },
};

pub struct PalettePage {
    items: Vec<PaletteItem>,
    pub label: Option<String>,
    initialized: bool,
}

impl PalettePage {
    pub fn new(items: Vec<PaletteItem>) -> Self {
        Self {
            items,
            initialized: false,
            label: None,
        }
    }

    pub fn label(mut self, label: impl ToString) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn initialize(&mut self, world: &mut World) {
        if !self.initialized {
            for item in &mut self.items {
                item.system.initialize(world)
            }
        }
    }

    pub fn filter(&mut self, filter: &str) -> Vec<MatchedPaletteItem<'_>> {
        let matcher = SkimMatcherV2::default();
        let mut items: Vec<_> = self
            .items
            .iter_mut()
            .filter_map(|item| {
                let (score, match_indices) = matcher.fuzzy_indices(&item.label, filter)?;
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
}

pub struct PaletteItem {
    pub label: String,
    pub header: String,
    pub icon: PaletteIcon,
    pub system: Box<dyn System<In = (), Out = PaletteEvent>>,
}

pub enum PaletteIcon {
    None,
    Link,
}

impl PaletteItem {
    pub fn new<M, I: IntoSystem<(), PaletteEvent, M>>(
        label: impl ToString,
        header: impl ToString,
        system: I,
    ) -> Self {
        Self {
            label: label.to_string(),
            header: header.to_string(),
            system: Box::new(I::into_system(system)),
            icon: PaletteIcon::None,
        }
    }

    pub fn icon(mut self, icon: PaletteIcon) -> Self {
        self.icon = icon;
        self
    }
}

pub enum PaletteEvent {
    NextPage {
        prev_page_label: Option<String>,
        next_page: PalettePage,
    },
    Exit,
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

const VIEWPORT_LABEL: &str = "VIEWPORT";
const SIMULATION_LABEL: &str = "SIMULATION";
const HELP_LABEL: &str = "HELP";

fn create_graph() -> PaletteItem {
    PaletteItem::new(
        "Create Graph",
        VIEWPORT_LABEL,
        |entities: Query<EntityData>| {
            PalettePage::new(
                entities
                    .iter()
                    .map(|(_, entity, _, metadata)| graph_entity_item(metadata, entity))
                    .collect(),
            )
            .into()
        },
    )
}

fn graph_entity_item(entity_metadata: &EntityMetadata, entity: Entity) -> PaletteItem {
    PaletteItem::new(
        entity_metadata.name.clone(),
        "Entities",
        move |entities: Query<EntityData>, metadata: Res<MetadataStore>| {
            let (_, entity, components, entity_metadata) = entities.get(entity).unwrap();
            let items = components
                .0
                .keys()
                .filter_map(|id| metadata.get_metadata(id).cloned())
                .map(move |item| {
                    PaletteItem::new(
                        item.name.clone(),
                        "Components",
                        move |entities: Query<EntityData>,
                              mut render_layer_alloc: ResMut<RenderLayerAlloc>,
                              mut tile_state: ResMut<tiles::TileState>| {
                            let component_id = item.component_id();
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
                            let bundle = GraphBundle::new(&mut render_layer_alloc, entities, None);
                            tile_state.create_graph_tile(None, bundle);
                            PaletteEvent::Exit
                        },
                    )
                })
                .collect();
            PaletteEvent::NextPage {
                prev_page_label: Some(entity_metadata.name.clone()),
                next_page: PalettePage::new(items),
            }
        },
    )
}

fn toggle_body_axes() -> PaletteItem {
    PaletteItem::new(
        "Toggle Body Axes",
        VIEWPORT_LABEL,
        |entities: Query<EntityData>| {
            PalettePage::new(
                entities
                    .iter()
                    .filter_map(|(&entity_id, _, value_map, metadata)| {
                        if value_map.0.contains_key(&ComponentId::new("world_pos")) {
                            Some(PaletteItem::new(
                                metadata.name.clone(),
                                "Entities",
                                move |mut commands: Commands, query: Query<(Entity, &BodyAxes)>| {
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

fn create_viewport() -> PaletteItem {
    PaletteItem::new(
        "Create Viewport",
        VIEWPORT_LABEL,
        |entities: Query<EntityData>| {
            PalettePage::new(
                std::iter::once(PaletteItem::new(
                    "None",
                    "Entities",
                    move |mut tile_state: ResMut<tiles::TileState>| {
                        tile_state.create_viewport_tile(None);
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
                                    move |mut tile_state: ResMut<tiles::TileState>| {
                                        tile_state.create_viewport_tile(Some(entity_id));
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
            .into()
        },
    )
}

impl Default for PalettePage {
    fn default() -> PalettePage {
        PalettePage::new(vec![
            PaletteItem::new(
                "Toggle Wireframe",
                VIEWPORT_LABEL,
                |mut wireframe: ResMut<WireframeConfig>| {
                    wireframe.global = !wireframe.global;
                    PaletteEvent::Exit
                },
            ),
            PaletteItem::new(
                "Toggle HDR",
                VIEWPORT_LABEL,
                |mut hdr: ResMut<HdrEnabled>| {
                    hdr.0 = !hdr.0;
                    PaletteEvent::Exit
                },
            ),
            PaletteItem::new(
                "Toggle Grid",
                VIEWPORT_LABEL,
                |mut grid_visibility: Query<&mut Visibility, With<InfiniteGrid>>| {
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
            PaletteItem::new(
                "Toggle Simulating",
                VIEWPORT_LABEL,
                |mut event: EventWriter<ControlMsg>, simulating: Res<Simulating>| {
                    event.send(ControlMsg::SetSimulating(!simulating.0));
                    PaletteEvent::Exit
                },
            ),
            create_graph(),
            create_viewport(),
            toggle_body_axes(),
            PaletteItem::new(
                "Save Replay",
                SIMULATION_LABEL,
                |mut event: EventWriter<ControlMsg>| {
                    event.send(ControlMsg::SaveReplay);
                    PaletteEvent::Exit
                },
            ),
            PaletteItem::new("Documentation", HELP_LABEL, || {
                let _ = opener::open("https://docs.elodin.systems");
                PaletteEvent::Exit
            })
            .icon(PaletteIcon::Link),
            PaletteItem::new("Release Notes", HELP_LABEL, || {
                let _ = opener::open("https://docs.elodin.systems/updates/changelog");
                PaletteEvent::Exit
            })
            .icon(PaletteIcon::Link),
        ])
    }
}
