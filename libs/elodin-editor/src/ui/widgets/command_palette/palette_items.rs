use bevy::{
    ecs::{
        query::With,
        system::{Query, ResMut, SystemParam, SystemState},
        world::World,
    },
    prelude::*,
    render::view::Visibility,
};
use bevy_infinite_grid::InfiniteGrid;
use conduit::ControlMsg;
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};

use crate::ui::{
    colors,
    tiles::{self, NewTileState},
    widgets::{WidgetSystem, WidgetSystemExt},
    HdrEnabled,
};

use super::{CommandPaletteIcons, CommandPaletteState, PaletteItem};

type CommandPaletteOptionWidget = Box<
    dyn Fn(&mut egui::Ui, &mut World, (bool, bool), Vec<usize>, &CommandPaletteIcons, egui::Margin),
>;

pub struct PaletteItemWrapper {
    pub label: String,
    pub group_label: bool,
    pub match_indices: Vec<usize>,
    pub widget: CommandPaletteOptionWidget,
}

pub fn show_palette_item(
    palette_item: &PaletteItemWrapper,
    matcher: &SkimMatcherV2,
    filter: &str,
) -> Option<Vec<usize>> {
    if palette_item.group_label {
        return Some(vec![]);
    }

    if let Some((_, indices)) = matcher.fuzzy_indices(&palette_item.label, filter) {
        return Some(indices);
    }

    None
}

pub fn filter_palette_items(
    palette_items: Vec<PaletteItemWrapper>,
    filter: &str,
) -> Vec<PaletteItemWrapper> {
    let matcher = SkimMatcherV2::default();
    let mut palette_items_filtered = vec![];

    for palette_item in palette_items {
        if let Some(indices) = show_palette_item(&palette_item, &matcher, filter) {
            palette_items_filtered.push(PaletteItemWrapper {
                match_indices: indices,
                ..palette_item
            });
        }
    }

    if palette_items_filtered.len() > 1 {
        palette_items_filtered
    } else {
        vec![]
    }
}

pub fn palette_viewport_items(filter: &str) -> Vec<PaletteItemWrapper> {
    let palette_items = vec![
        PaletteItemWrapper {
            label: String::from("Viewport"),
            group_label: true,
            match_indices: vec![],
            widget: Box::new(|ui, _, _, _, _, row_margin| {
                egui::Frame::none().inner_margin(row_margin).show(ui, |ui| {
                    ui.label(egui::RichText::new("Viewport").color(colors::PRIMARY_CREAME_6));
                });
            }),
        },
        PaletteItemWrapper {
            label: String::from("Toggle HDR"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, _, row_margin| {
                    ui.add_widget_with::<PaletteItemViewportToggleHdr>(
                        world,
                        "viewport_toggle_hdr",
                        (
                            (request_focus, use_item),
                            row_margin,
                            String::from("Toggle HDR"),
                            matched_char_indices,
                        ),
                    );
                },
            ),
        },
        PaletteItemWrapper {
            label: String::from("Toggle Grid"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, _, row_margin| {
                    ui.add_widget_with::<PaletteItemViewportToggleGrid>(
                        world,
                        "viewport_toggle_grid",
                        (
                            (request_focus, use_item),
                            row_margin,
                            String::from("Toggle Grid"),
                            matched_char_indices,
                        ),
                    );
                },
            ),
        },
        PaletteItemWrapper {
            label: String::from("Create Viewport"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, _, row_margin| {
                    ui.add_widget_with::<PaletteItemViewportCreateTile>(
                        world,
                        "viewport_create_graph",
                        (
                            (request_focus, use_item),
                            PaletteItemCreateTileType::Viewport,
                            row_margin,
                            String::from("Create Viewport"),
                            matched_char_indices,
                        ),
                    );
                },
            ),
        },
        PaletteItemWrapper {
            label: String::from("Create Graph"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, _, row_margin| {
                    ui.add_widget_with::<PaletteItemViewportCreateTile>(
                        world,
                        "viewport_create_graph",
                        (
                            (request_focus, use_item),
                            PaletteItemCreateTileType::Graph,
                            row_margin,
                            String::from("Create Graph"),
                            matched_char_indices,
                        ),
                    );
                },
            ),
        },
    ];

    filter_palette_items(palette_items, filter)
}

pub fn palette_sim_items(filter: &str) -> Vec<PaletteItemWrapper> {
    let palette_items = vec![
        PaletteItemWrapper {
            label: String::from("Simulation"),
            group_label: true,
            match_indices: vec![],
            widget: Box::new(|ui, _, _, _, _, row_margin| {
                egui::Frame::none().inner_margin(row_margin).show(ui, |ui| {
                    ui.label(egui::RichText::new("Simulation").color(colors::PRIMARY_CREAME_6));
                });
            }),
        },
        PaletteItemWrapper {
            label: String::from("Save Replay"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, _, row_margin| {
                    ui.add_widget_with::<PaletteItemSaveReplay>(
                        world,
                        "save_replay",
                        (
                            (request_focus, use_item),
                            row_margin,
                            String::from("Save Replay"),
                            matched_char_indices,
                        ),
                    );
                },
            ),
        },
    ];

    filter_palette_items(palette_items, filter)
}

pub fn palette_help_items(filter: &str) -> Vec<PaletteItemWrapper> {
    let palette_items = vec![
        PaletteItemWrapper {
            label: String::from("Help"),
            group_label: true,
            match_indices: vec![],
            widget: Box::new(|ui, _, _, _, _, row_margin| {
                egui::Frame::none().inner_margin(row_margin).show(ui, |ui| {
                    ui.label(egui::RichText::new("Help").color(colors::PRIMARY_CREAME_6));
                });
            }),
        },
        PaletteItemWrapper {
            label: String::from("Documentation"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, icons, row_margin| {
                    ui.add_widget_with::<PaletteItemOpenLink>(
                        world,
                        "help_documentation",
                        (
                            (request_focus, use_item),
                            icons.link,
                            row_margin,
                            String::from("Documentation"),
                            matched_char_indices,
                            String::from("https://docs.elodin.systems"),
                        ),
                    );
                },
            ),
        },
        PaletteItemWrapper {
            label: String::from("Release Notes"),
            group_label: false,
            match_indices: vec![],
            widget: Box::new(
                |ui, world, (request_focus, use_item), matched_char_indices, icons, row_margin| {
                    ui.add_widget_with::<PaletteItemOpenLink>(
                        world,
                        "help_release_notes",
                        (
                            (request_focus, use_item),
                            icons.link,
                            row_margin,
                            String::from("Release Notes"),
                            matched_char_indices,
                            String::from("https://docs.elodin.systems/updates/changelog"),
                        ),
                    );
                },
            ),
        },
    ];

    filter_palette_items(palette_items, filter)
}

#[derive(SystemParam)]
pub struct PaletteItemOpenLink<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
}

impl WidgetSystem for PaletteItemOpenLink<'_> {
    type Args = (
        (bool, bool),
        egui::TextureId,
        egui::Margin,
        String,
        Vec<usize>,
        String,
    );
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;

        let (
            (request_focus, use_item),
            icon,
            row_margin,
            item_label,
            matched_char_indices,
            item_url,
        ) = args;

        let btn = ui.add(
            PaletteItem::new(item_label, matched_char_indices)
                .icon(icon)
                .margin(row_margin),
        );

        if request_focus {
            btn.request_focus();
        }

        if btn.clicked() || use_item {
            ui.ctx().open_url(egui::OpenUrl {
                url: item_url,
                new_tab: true,
            });

            command_palette_state.show = false;
        }
    }
}

#[derive(SystemParam)]
pub struct PaletteItemViewportToggleHdr<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    hdr_enabled: ResMut<'w, HdrEnabled>,
}

impl WidgetSystem for PaletteItemViewportToggleHdr<'_> {
    type Args = ((bool, bool), egui::Margin, String, Vec<usize>);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;
        let mut hdr_enabled = state_mut.hdr_enabled;

        let ((request_focus, use_item), row_margin, item_label, matched_char_indices) = args;

        let btn = ui.add(PaletteItem::new(item_label, matched_char_indices).margin(row_margin));

        if request_focus {
            btn.request_focus();
        }

        if btn.clicked() || use_item {
            hdr_enabled.0 = !hdr_enabled.0;
            command_palette_state.show = false;
        }
    }
}

#[derive(SystemParam)]
pub struct PaletteItemViewportToggleGrid<'w, 's> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    grid_visibility: Query<'w, 's, &'static mut Visibility, With<InfiniteGrid>>,
}

impl WidgetSystem for PaletteItemViewportToggleGrid<'_, '_> {
    type Args = ((bool, bool), egui::Margin, String, Vec<usize>);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;
        let mut grid_visibility = state_mut.grid_visibility;

        let ((request_focus, use_item), row_margin, item_label, matched_char_indices) = args;

        let btn = ui.add(PaletteItem::new(item_label, matched_char_indices).margin(row_margin));

        if request_focus {
            btn.request_focus();
        }

        if btn.clicked() || use_item {
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

            command_palette_state.show = false;
        }
    }
}

pub enum PaletteItemCreateTileType {
    Viewport,
    Graph,
}

#[derive(SystemParam)]
pub struct PaletteItemViewportCreateTile<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    new_tile_state: ResMut<'w, tiles::NewTileState>,
}

impl WidgetSystem for PaletteItemViewportCreateTile<'_> {
    type Args = (
        (bool, bool),
        PaletteItemCreateTileType,
        egui::Margin,
        String,
        Vec<usize>,
    );
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;
        let mut new_tile_state = state_mut.new_tile_state;

        let ((request_focus, use_item), tile_type, row_margin, item_label, matched_char_indices) =
            args;

        let btn = ui.add(PaletteItem::new(item_label, matched_char_indices).margin(row_margin));

        if request_focus {
            btn.request_focus();
        }

        if btn.clicked() || use_item {
            match tile_type {
                PaletteItemCreateTileType::Viewport => {
                    *new_tile_state = NewTileState::Viewport(None, None)
                }
                PaletteItemCreateTileType::Graph => {
                    *new_tile_state = NewTileState::Graph(None, None, None)
                }
            }

            command_palette_state.show = false;
        }
    }
}

#[derive(SystemParam)]
pub struct PaletteItemSaveReplay<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    event: EventWriter<'w, ControlMsg>,
}

impl WidgetSystem for PaletteItemSaveReplay<'_> {
    type Args = ((bool, bool), egui::Margin, String, Vec<usize>);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;
        let mut event = state_mut.event;

        let ((request_focus, use_item), row_margin, item_label, matched_char_indices) = args;

        let btn = ui.add(PaletteItem::new(item_label, matched_char_indices).margin(row_margin));

        if request_focus {
            btn.request_focus();
        }

        if btn.clicked() || use_item {
            event.send(ControlMsg::SaveReplay);
            command_palette_state.show = false;
        }
    }
}
