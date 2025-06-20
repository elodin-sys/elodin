use std::{collections::BTreeMap, fmt::Display};

use bevy::ecs::{
    system::{Query, Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::{Children, Resource};
use bevy_egui::egui::{self, Align, Color32, Layout, RichText, emath};
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2::types::ComponentId;
use impeller2_bevy::{
    ComponentMetadataRegistry, ComponentPath, ComponentPathRegistry, ComponentValue,
    ComponentValueExt, ElementValueMut,
};
use impeller2_wkt::{ComponentMetadata, Glb, MetadataExt};
use smallvec::SmallVec;

use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        EntityPair,
        colors::get_scheme,
        theme::configure_input_with_border,
        tiles::TreeAction,
        utils::{MarginSides, format_num},
        widgets::{
            WidgetSystem, label,
            plot::{GraphBundle, default_component_values},
            query_plot::QueryType,
        },
    },
};

use super::{
    InspectorIcons, empty_inspector,
    graph::{eql_autocomplete, inspector_text_field, query},
};

#[derive(SystemParam)]
pub struct InspectorEntity<'w, 's> {
    children: Query<'w, 's, &'static Children>,
    component_ids: Query<'w, 's, &'static ComponentId>,
    values: Query<'w, 's, &'static mut ComponentValue>,
    metadata_query: Query<'w, 's, &'static mut ComponentMetadata>,
    object_3d: Query<'w, 's, &'static mut crate::object_3d::Object3D>,
    glb: Query<'w, 's, &'static mut Glb>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    path_reg: Res<'w, ComponentPathRegistry>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    filter: ResMut<'w, ComponentFilter>,
    eql_context: ResMut<'w, crate::EqlContext>,
}

impl WidgetSystem for InspectorEntity<'_, '_> {
    type Args = (InspectorIcons, EntityPair);
    type Output = SmallVec<[TreeAction; 4]>;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output {
        let width = ui.available_width();
        let mut tree_actions = SmallVec::new();
        let InspectorEntity {
            children,
            component_ids,
            metadata_query,
            mut values,
            mut object_3d,
            mut glb,
            metadata_store,
            path_reg,
            mut render_layer_alloc,
            mut filter,
            eql_context,
        } = state.get_mut(world);

        let (icons, pair) = args;

        let entity_id = pair.impeller;
        let Ok(metadata) = metadata_query.get(pair.bevy) else {
            ui.add(empty_inspector());
            return tree_actions;
        };

        let icon_chart = icons.chart;

        let mono_font = egui::TextStyle::Monospace.resolve(ui.style_mut());
        egui::Frame::NONE
            .inner_margin(egui::Margin::ZERO.top(8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        label::ELabel::new(&metadata.name)
                            .padding(egui::Margin::same(0).bottom(24.))
                            .bottom_stroke(egui::Stroke {
                                width: 1.0,
                                color: get_scheme().border_primary,
                            })
                            .margin(egui::Margin::same(0).bottom(8.)),
                    );

                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.label(
                            RichText::new(entity_id.0.to_string())
                                .color(get_scheme().text_primary)
                                .font(mono_font.clone()),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("ENTITY ID")
                                .color(get_scheme().text_secondary)
                                .font(mono_font.clone()),
                        );
                    });
                });
            });

        search(ui, &mut filter.0, icons.search);
        ui.add_space(4.0);

        egui::Frame::NONE.show(ui, |ui| {
            if let Ok(mut object_3d) = object_3d.get_mut(pair.bevy) {
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("EQL").color(get_scheme().text_secondary));
                ui.add_space(8.0);
                configure_input_with_border(ui.style_mut());
                let query_res = query(ui, &mut object_3d.eql, QueryType::EQL);
                eql_autocomplete(ui, &eql_context.0, &query_res, &mut object_3d.eql);
                if query_res.changed() {
                    match eql_context.0.parse_str(&object_3d.eql) {
                        Ok(expr) => {
                            object_3d.compiled_expr =
                                Some(crate::object_3d::compile_eql_expr(expr));
                        }
                        Err(err) => {
                            ui.colored_label(get_scheme().error, err.to_string());
                        }
                    }
                }
                ui.add_space(8.0);
            }
            if let Ok(mut glb) = glb.get_mut(pair.bevy) {
                ui.separator();
                ui.add_space(8.0);
                ui.label(egui::RichText::new("GLB").color(get_scheme().text_secondary));
                ui.add_space(8.0);
                if inspector_text_field(ui, &mut glb.0, "Enter a path to a glb").changed() {
                    // commands
                    //     .entity(pair.bevy)
                    //     .remove::<crate::SyncedGlb>()
                    //     .insert(impeller2_bevy::AssetHandle::<Glb>::new(fastrand::u64(..)));
                }
                ui.add_space(8.0);
            }
        });

        let matcher = SkimMatcherV2::default().smart_case().use_cache(true);
        let Ok(children) = children.get(pair.bevy) else {
            return tree_actions;
        };

        // let mut components = component_value_map
        //     .0
        //     .keys()
        //     .filter_map(|id| {
        //         let metadata = metadata_store.get_metadata(id)?;
        //         let priority = metadata.priority();
        //         Some((*id, priority, metadata))
        //     })
        let mut components = children
            .iter()
            .filter_map(|&child| {
                let id = component_ids.get(child).ok()?;
                let metadata = metadata_store.get_metadata(id)?;
                let priority = metadata.priority();
                Some((id, child, priority, metadata))
            })
            .filter_map(|(id, child, priority, metadata)| {
                if filter.0.is_empty() {
                    Some((id, child, priority, metadata))
                } else {
                    matcher
                        .fuzzy_match(&metadata.name, &filter.0)
                        .map(|score| (id, child, score, metadata))
                }
            })
            .filter(|(_, _, _, metadata)| !metadata.asset)
            .filter(|(_, _, priority, _)| *priority >= 0)
            .collect::<Vec<_>>();
        components.sort_by_key(|(_, _, priority, metadata)| (*priority, &metadata.name));

        for (component_id, child, _, metadata) in components.into_iter().rev() {
            let Ok(mut component_value) = values.get_mut(child) else {
                continue;
            };
            let label = metadata.name.clone();
            let element_names = metadata.element_names();

            ui.add(egui::Separator::default().spacing(16.0));

            let mut create_graph = false;

            inspector_item_multi(
                ui,
                &label,
                element_names,
                &mut component_value,
                icon_chart,
                &mut create_graph,
                width,
            );

            if create_graph {
                let values = default_component_values(component_id, &component_value);
                let component_path = path_reg
                    .get(component_id)
                    .cloned()
                    .unwrap_or_else(|| ComponentPath::from_name(&metadata.name));
                let components =
                    BTreeMap::from_iter(std::iter::once((component_path, values.clone())));
                let bundle =
                    GraphBundle::new(&mut render_layer_alloc, components, metadata.name.clone());
                tree_actions.push(TreeAction::AddGraph(None, Some(bundle)));
            }
        }
        tree_actions
    }
}

#[derive(Resource, Default)]
pub struct ComponentFilter(pub String);

pub fn search(
    ui: &mut egui::Ui,
    filter: &mut String,
    search_icon: egui::TextureId,
) -> egui::Response {
    ui.vertical(|ui| {
        egui::Frame::NONE
            .corner_radius(egui::CornerRadius::same(3))
            .inner_margin(egui::Margin::same(4))
            .fill(get_scheme().bg_secondary)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::widgets::Image::new(egui::load::SizedTexture::new(
                            search_icon,
                            [ui.spacing().interact_size.y, ui.spacing().interact_size.y],
                        ))
                        .tint(get_scheme().text_secondary),
                    );

                    let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                    font_id.size = 12.0;
                    ui.add(
                        egui::TextEdit::singleline(filter)
                            .desired_width(ui.available_width())
                            .frame(false)
                            .font(font_id),
                    );
                });
            });
    })
    .response
}

fn inspector_item_value_ui(
    ui: &mut egui::Ui,
    label: &str,
    value: ElementValueMut<'_>,
    size: egui::Vec2,
) -> egui::Response {
    let label_color = get_scheme().text_secondary;
    ui.allocate_ui_with_layout(
        size,
        Layout::left_to_right(Align::Center).with_main_justify(true),
        |ui| {
            ui.with_layout(Layout::top_down_justified(Align::LEFT), |ui| {
                ui.vertical(|ui| {
                    ui.add_space(3.0);
                    ui.style_mut().override_font_id =
                        Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
                    ui.label(RichText::new(label).color(label_color))
                })
            });

            match value {
                ElementValueMut::U8(n) => comp_drag_value(ui, n),
                ElementValueMut::U16(n) => comp_drag_value(ui, n),
                ElementValueMut::U32(n) => comp_drag_value(ui, n),
                ElementValueMut::U64(n) => comp_drag_value(ui, n),
                ElementValueMut::I8(n) => comp_drag_value(ui, n),
                ElementValueMut::I16(n) => comp_drag_value(ui, n),
                ElementValueMut::I32(n) => comp_drag_value(ui, n),
                ElementValueMut::I64(n) => comp_drag_value(ui, n),
                ElementValueMut::F64(n) => comp_drag_value(ui, n),
                ElementValueMut::F32(n) => comp_drag_value(ui, n),
                ElementValueMut::Bool(b) => ui.checkbox(b, ""),
            }
        },
    )
    .inner
}

fn comp_drag_value<Num: emath::Numeric>(ui: &mut egui::Ui, value: &mut Num) -> egui::Response {
    ui.with_layout(Layout::top_down_justified(Align::RIGHT), |ui| {
        let schema = get_scheme();
        ui.style_mut().visuals.widgets.active.weak_bg_fill = schema.bg_secondary;
        ui.style_mut().visuals.widgets.open.weak_bg_fill = schema.bg_secondary;
        ui.style_mut().visuals.widgets.hovered.weak_bg_fill = schema.bg_secondary;
        ui.style_mut().visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
        ui.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
        ui.style_mut().override_font_id = Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
        ui.add(egui::DragValue::new(value).custom_formatter(|v, _| format_num(v)))
    })
    .inner
}

pub fn inspector_item_value<'a>(
    label: &'a str,
    value: ElementValueMut<'a>,
    size: egui::Vec2,
) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| inspector_item_value_ui(ui, label, value, size)
}

fn inspector_item_multi(
    ui: &mut egui::Ui,
    label: &str,
    element_names: &str,
    values: &mut ComponentValue,
    icon_chart: egui::TextureId,
    create_graph: &mut bool,
    line_width: f32,
) -> egui::Response {
    let element_names = element_names
        .split(',')
        .filter(|s| !s.is_empty())
        .map(Option::Some)
        .chain(std::iter::repeat(None));
    let resp = ui.vertical(|ui| {
        ui.set_width(line_width);
        let [graph_clicked] = label::label_with_buttons(
            ui,
            [icon_chart],
            label,
            get_scheme().text_primary,
            egui::Margin::symmetric(0, 4).bottom(12.0),
        );
        *create_graph = graph_clicked;

        let item_spacing = egui::vec2(8.0, 8.0);

        let line_height = ui.spacing().interact_size.y * 1.4;

        let item_width_min = ui.spacing().interact_size.x * 2.4;
        let items_per_line = (line_width / item_width_min).floor();

        let necessary_spacing = (items_per_line - 1.0) * item_spacing.x;
        let item_width = (line_width - necessary_spacing) / items_per_line;

        let desired_size = egui::vec2(item_width - 1.0, line_height);

        ui.horizontal_wrapped(|ui| {
            ui.style_mut().spacing.item_spacing = item_spacing;
            values.indexed_iter_mut().zip(element_names).fold(
                None,
                |res: Option<egui::Response>, ((dim_i, value), element_name)| {
                    let label = element_name
                        .map(|name| name.to_string())
                        .unwrap_or_else(|| format!("{dim_i:?}"));

                    let new_res = ui.add(inspector_item_value(&label, value, desired_size));
                    if let Some(res) = res {
                        Some(res | new_res)
                    } else {
                        Some(new_res)
                    }
                },
            )
        })
    });
    if let Some(inner_resp) = resp.inner.inner {
        resp.response | inner_resp
    } else {
        resp.response
    }
}

pub struct DimIndexFormat(SmallVec<[usize; 4]>);
impl Display for DimIndexFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x) in self.0.iter().enumerate() {
            write!(f, "{x}")?;
            if i + 1 < self.0.len() {
                write!(f, ".")?;
            }
        }
        Ok(())
    }
}
