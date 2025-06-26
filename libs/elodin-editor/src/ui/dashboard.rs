use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::text::{TextColor, TextFont};
use bevy::ui::{
    AlignContent, AlignItems, AlignSelf, BoxSizing, Display, FlexDirection, FlexWrap,
    JustifyContent, JustifyItems, JustifySelf, Node, Overflow, OverflowAxis, OverflowClipBox,
    OverflowClipMargin, PositionType,
};
use bevy::window::PrimaryWindow;
use eql::FmtExpr;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, DashboardNode};
use nox::ArrayBuf;
use smallvec::{SmallVec, smallvec};

use crate::object_3d::{CompiledExpr, compile_eql_expr};
use crate::ui::colors::{self, ColorExt};
use crate::ui::widgets::WidgetSystem;

#[derive(SystemParam)]
pub struct DashboardWidget<'w, 's> {
    query: Query<'w, 's, &'static mut Node>,
    window: Query<'w, 's, &'static bevy_egui::EguiContextSettings, With<PrimaryWindow>>,
}

impl WidgetSystem for DashboardWidget<'_, '_> {
    type Args = Entity;

    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Ok(mut ui_node) = state.query.get_mut(entity) else {
            return;
        };

        let max_rect = ui.max_rect();

        let Some(egui_settings) = state.window.iter().next() else {
            return;
        };

        let scale_factor = egui_settings.scale_factor;
        let viewport_pos = max_rect.left_top().to_vec2() * scale_factor;
        let viewport_size = max_rect.size() * scale_factor;

        ui_node.position_type = PositionType::Absolute;
        ui_node.left = Val::Px(viewport_pos.x);
        ui_node.top = Val::Px(viewport_pos.y);
        ui_node.width = Val::Px(viewport_size.x);
        ui_node.height = Val::Px(viewport_size.y);
        ui_node.max_width = Val::Px(viewport_size.x);
        ui_node.max_height = Val::Px(viewport_size.y);
        ui_node.min_width = Val::Px(viewport_size.x);
        ui_node.min_height = Val::Px(viewport_size.y);
    }
}

#[derive(SystemParam)]
pub struct NodeUpdaterParams<'w, 's> {
    entity_map: Res<'w, EntityMap>,
    values: Query<'w, 's, &'static ComponentValue>,
}

type NodeFn = dyn for<'a, 'b> Fn(
        &'a NodeUpdaterParams<'b, 'b>,
    ) -> Result<(Node, Option<Text>, Option<TextFont>, Option<TextColor>), String>
    + Send
    + Sync;

#[derive(Component)]
pub struct NodeUpdater(Box<NodeFn>);

pub type NodeUpdaterArgs = (
    &'static NodeUpdater,
    &'static mut Node,
    Option<&'static mut Text>,
    Option<&'static mut TextFont>,
    Option<&'static mut TextColor>,
);

pub fn update_nodes(mut query: Query<NodeUpdaterArgs>, params: NodeUpdaterParams) {
    for (node_updater, mut node, text, text_font, text_color) in query.iter_mut() {
        let Ok((new_node, new_text, new_text_font, new_text_color)) = (node_updater.0)(&params)
            .inspect_err(|err| {
                warn!(?err, "Failed to update node");
            })
        else {
            continue;
        };
        *node = new_node;
        if let (Some(mut text), Some(new_text)) = (text, new_text) {
            *text = new_text;
        }
        if let (Some(mut text_font), Some(new_text_font)) = (text_font, new_text_font) {
            *text_font = new_text_font;
        }
        if let (Some(mut text_color), Some(new_text_color)) = (text_color, new_text_color) {
            *text_color = new_text_color;
        }
    }
}

pub enum CompiledVal {
    Auto,
    Px(CompiledExpr),
    Percent(CompiledExpr),
    Vw(CompiledExpr),
    Vh(CompiledExpr),
    VMin(CompiledExpr),
    VMax(CompiledExpr),
}

impl CompiledVal {
    pub fn execute(
        &self,
        entity_map: &EntityMap,
        values: &Query<&'static ComponentValue>,
    ) -> Result<Val, String> {
        let val = match self {
            CompiledVal::Auto => Val::Auto,
            CompiledVal::Px(expr) => {
                let val = expr.execute(entity_map, values)?;
                Val::Px(val.as_f32().ok_or("invalid value")?)
            }
            CompiledVal::Percent(expr) => {
                let val = expr.execute(entity_map, values)?;
                Val::Percent(val.as_f32().ok_or("invalid value")?)
            }
            CompiledVal::Vw(expr) => {
                let val = expr.execute(entity_map, values)?;
                Val::Vw(val.as_f32().ok_or("invalid value")?)
            }
            CompiledVal::Vh(expr) => {
                let val = expr.execute(entity_map, values)?;
                Val::Vh(val.as_f32().ok_or("invalid value")?)
            }
            CompiledVal::VMin(expr) => {
                let val = expr.execute(entity_map, values)?;
                Val::VMin(val.as_f32().ok_or("invalid value")?)
            }
            CompiledVal::VMax(expr) => {
                let val = expr.execute(entity_map, values)?;
                Val::VMax(val.as_f32().ok_or("invalid value")?)
            }
        };
        Ok(val)
    }
}

pub fn compile_val(ctx: &eql::Context, val: &impeller2_wkt::Val) -> CompiledVal {
    try_compile_val(ctx, val).unwrap_or(CompiledVal::Auto)
}

pub fn try_compile_val(
    ctx: &eql::Context,
    val: &impeller2_wkt::Val,
) -> Result<CompiledVal, eql::Error> {
    let val = match val {
        impeller2_wkt::Val::Auto => CompiledVal::Auto,
        impeller2_wkt::Val::Px(px) => CompiledVal::Px(compile_eql_expr(ctx.parse_str(px)?)),
        impeller2_wkt::Val::Percent(percent) => {
            CompiledVal::Percent(compile_eql_expr(ctx.parse_str(percent)?))
        }
        impeller2_wkt::Val::Vw(vw) => CompiledVal::Vw(compile_eql_expr(ctx.parse_str(vw)?)),
        impeller2_wkt::Val::Vh(vh) => CompiledVal::Vh(compile_eql_expr(ctx.parse_str(vh)?)),
        impeller2_wkt::Val::VMin(vmin) => CompiledVal::VMin(compile_eql_expr(ctx.parse_str(vmin)?)),
        impeller2_wkt::Val::VMax(vmax) => CompiledVal::VMax(compile_eql_expr(ctx.parse_str(vmax)?)),
    };
    Ok(val)
}

pub fn spawn_dashboard(
    source: &impeller2_wkt::Dashboard,
    eql: &eql::Context,
    commands: &mut Commands,
    params: &NodeUpdaterParams,
) -> Result<Entity, eql::Error> {
    let mut parent = commands.spawn((
        Node {
            overflow: Overflow {
                x: OverflowAxis::Scroll,
                y: OverflowAxis::Scroll,
            },
            ..Default::default()
        },
        BackgroundColor(colors::SURFACE_PRIMARY.into_bevy()),
    ));
    let parent_id = parent.id();
    let node = spawn_node(
        &source.root,
        eql,
        parent.with_child(()),
        parent_id,
        smallvec![],
        params,
    )?;
    parent.insert(impeller2_wkt::Dashboard {
        root: node,
        aux: parent_id,
    });
    parent.insert(DashboardNodePath {
        root: parent_id,
        path: smallvec![],
    });
    Ok(parent_id)
}

#[derive(Component, Clone)]
pub struct DashboardNodePath {
    pub root: Entity,
    pub path: SmallVec<[usize; 4]>,
}

pub fn spawn_node<T>(
    source: &impeller2_wkt::DashboardNode<T>,
    eql: &eql::Context,
    commands: &mut EntityCommands,
    root: Entity,
    path: SmallVec<[usize; 4]>,
    params: &NodeUpdaterParams,
) -> Result<DashboardNode<Entity>, eql::Error> {
    let left = compile_val(eql, &source.left);
    let right = compile_val(eql, &source.right);
    let top = compile_val(eql, &source.top);
    let bottom = compile_val(eql, &source.bottom);
    let width = compile_val(eql, &source.width);
    let height = compile_val(eql, &source.height);
    let min_width = compile_val(eql, &source.min_width);
    let min_height = compile_val(eql, &source.min_height);
    let max_width = compile_val(eql, &source.max_width);
    let max_height = compile_val(eql, &source.max_height);
    let margin_left = compile_val(eql, &source.margin.left);
    let margin_right = compile_val(eql, &source.margin.right);
    let margin_top = compile_val(eql, &source.margin.top);
    let margin_bottom = compile_val(eql, &source.margin.bottom);
    let padding_left = compile_val(eql, &source.padding.left);
    let padding_right = compile_val(eql, &source.padding.right);
    let padding_top = compile_val(eql, &source.padding.top);
    let padding_bottom = compile_val(eql, &source.padding.bottom);
    let border_left = compile_val(eql, &source.border.left);
    let border_right = compile_val(eql, &source.border.right);
    let border_top = compile_val(eql, &source.border.top);
    let border_bottom = compile_val(eql, &source.border.bottom);
    let flex_basis = compile_val(eql, &source.flex_basis);
    let row_gap = compile_val(eql, &source.row_gap);
    let column_gap = compile_val(eql, &source.column_gap);
    let text = source.text.as_ref().and_then(|src| {
        let expr = eql.parse_fmt_string(src).ok()?;
        Some(compile_fmt_string(expr))
    });
    let node = Node {
        display: match source.display {
            impeller2_wkt::Display::None => Display::None,
            impeller2_wkt::Display::Block => Display::Block,
            impeller2_wkt::Display::Flex => Display::Flex,
            impeller2_wkt::Display::Grid => Display::Grid,
        },
        box_sizing: match source.box_sizing {
            impeller2_wkt::BoxSizing::BorderBox => BoxSizing::BorderBox,
            impeller2_wkt::BoxSizing::ContentBox => BoxSizing::ContentBox,
        },
        position_type: match source.position_type {
            impeller2_wkt::PositionType::Relative => PositionType::Relative,
            impeller2_wkt::PositionType::Absolute => PositionType::Absolute,
        },
        overflow: Overflow {
            x: match source.overflow.x {
                impeller2_wkt::OverflowAxis::Visible => OverflowAxis::Visible,
                impeller2_wkt::OverflowAxis::Clip => OverflowAxis::Clip,
                impeller2_wkt::OverflowAxis::Hidden => OverflowAxis::Hidden,
                impeller2_wkt::OverflowAxis::Scroll => OverflowAxis::Scroll,
            },
            y: match source.overflow.y {
                impeller2_wkt::OverflowAxis::Visible => OverflowAxis::Visible,
                impeller2_wkt::OverflowAxis::Clip => OverflowAxis::Clip,
                impeller2_wkt::OverflowAxis::Hidden => OverflowAxis::Hidden,
                impeller2_wkt::OverflowAxis::Scroll => OverflowAxis::Scroll,
            },
        },
        overflow_clip_margin: OverflowClipMargin {
            visual_box: match source.overflow_clip_margin.visual_box {
                impeller2_wkt::OverflowClipBox::ContentBox => OverflowClipBox::ContentBox,
                impeller2_wkt::OverflowClipBox::PaddingBox => OverflowClipBox::PaddingBox,
                impeller2_wkt::OverflowClipBox::BorderBox => OverflowClipBox::BorderBox,
            },
            margin: 0.0,
        },
        align_items: match source.align_items {
            impeller2_wkt::AlignItems::Default => AlignItems::Default,
            impeller2_wkt::AlignItems::Start => AlignItems::Start,
            impeller2_wkt::AlignItems::End => AlignItems::End,
            impeller2_wkt::AlignItems::FlexStart => AlignItems::FlexStart,
            impeller2_wkt::AlignItems::FlexEnd => AlignItems::FlexEnd,
            impeller2_wkt::AlignItems::Center => AlignItems::Center,
            impeller2_wkt::AlignItems::Baseline => AlignItems::Baseline,
            impeller2_wkt::AlignItems::Stretch => AlignItems::Stretch,
        },
        justify_items: match source.justify_items {
            impeller2_wkt::JustifyItems::Default => JustifyItems::Default,
            impeller2_wkt::JustifyItems::Start => JustifyItems::Start,
            impeller2_wkt::JustifyItems::End => JustifyItems::End,
            impeller2_wkt::JustifyItems::Center => JustifyItems::Center,
            impeller2_wkt::JustifyItems::Baseline => JustifyItems::Baseline,
            impeller2_wkt::JustifyItems::Stretch => JustifyItems::Stretch,
        },
        align_self: match source.align_self {
            impeller2_wkt::AlignSelf::Auto => AlignSelf::Auto,
            impeller2_wkt::AlignSelf::Start => AlignSelf::Start,
            impeller2_wkt::AlignSelf::End => AlignSelf::End,
            impeller2_wkt::AlignSelf::FlexStart => AlignSelf::FlexStart,
            impeller2_wkt::AlignSelf::FlexEnd => AlignSelf::FlexEnd,
            impeller2_wkt::AlignSelf::Center => AlignSelf::Center,
            impeller2_wkt::AlignSelf::Baseline => AlignSelf::Baseline,
            impeller2_wkt::AlignSelf::Stretch => AlignSelf::Stretch,
        },
        justify_self: match source.justify_self {
            impeller2_wkt::JustifySelf::Auto => JustifySelf::Auto,
            impeller2_wkt::JustifySelf::Start => JustifySelf::Start,
            impeller2_wkt::JustifySelf::End => JustifySelf::End,
            impeller2_wkt::JustifySelf::Center => JustifySelf::Center,
            impeller2_wkt::JustifySelf::Baseline => JustifySelf::Baseline,
            impeller2_wkt::JustifySelf::Stretch => JustifySelf::Stretch,
        },
        align_content: match source.align_content {
            impeller2_wkt::AlignContent::Default => AlignContent::Default,
            impeller2_wkt::AlignContent::Start => AlignContent::Start,
            impeller2_wkt::AlignContent::End => AlignContent::End,
            impeller2_wkt::AlignContent::FlexStart => AlignContent::FlexStart,
            impeller2_wkt::AlignContent::FlexEnd => AlignContent::FlexEnd,
            impeller2_wkt::AlignContent::Center => AlignContent::Center,
            impeller2_wkt::AlignContent::Stretch => AlignContent::Stretch,
            impeller2_wkt::AlignContent::SpaceBetween => AlignContent::SpaceBetween,
            impeller2_wkt::AlignContent::SpaceEvenly => AlignContent::SpaceEvenly,
            impeller2_wkt::AlignContent::SpaceAround => AlignContent::SpaceAround,
        },
        justify_content: match source.justify_content {
            impeller2_wkt::JustifyContent::Default => JustifyContent::Default,
            impeller2_wkt::JustifyContent::Start => JustifyContent::Start,
            impeller2_wkt::JustifyContent::End => JustifyContent::End,
            impeller2_wkt::JustifyContent::FlexStart => JustifyContent::FlexStart,
            impeller2_wkt::JustifyContent::FlexEnd => JustifyContent::FlexEnd,
            impeller2_wkt::JustifyContent::Center => JustifyContent::Center,
            impeller2_wkt::JustifyContent::Stretch => JustifyContent::Stretch,
            impeller2_wkt::JustifyContent::SpaceBetween => JustifyContent::SpaceBetween,
            impeller2_wkt::JustifyContent::SpaceEvenly => JustifyContent::SpaceEvenly,
            impeller2_wkt::JustifyContent::SpaceAround => JustifyContent::SpaceAround,
        },
        flex_direction: match source.flex_direction {
            impeller2_wkt::FlexDirection::Row => FlexDirection::Row,
            impeller2_wkt::FlexDirection::Column => FlexDirection::Column,
            impeller2_wkt::FlexDirection::RowReverse => FlexDirection::RowReverse,
            impeller2_wkt::FlexDirection::ColumnReverse => FlexDirection::ColumnReverse,
        },
        flex_wrap: match source.flex_wrap {
            impeller2_wkt::FlexWrap::NoWrap => FlexWrap::NoWrap,
            impeller2_wkt::FlexWrap::Wrap => FlexWrap::Wrap,
            impeller2_wkt::FlexWrap::WrapReverse => FlexWrap::WrapReverse,
        },
        flex_grow: source.flex_grow,
        flex_shrink: source.flex_shrink,
        ..Default::default()
    };
    let updater_node = node.clone();
    let font_size = source.font_size;
    let text_color = source.text_color;
    let node_updater = NodeUpdater(Box::new(move |params| {
        let NodeUpdaterParams {
            entity_map: e,
            values: q,
        } = params;
        let mut node = updater_node.clone();
        node.left = left.execute(e, q)?;
        node.right = right.execute(e, q)?;
        node.top = top.execute(e, q)?;
        node.bottom = bottom.execute(e, q)?;

        node.width = width.execute(e, q)?;
        node.height = height.execute(e, q)?;

        node.min_width = min_width.execute(e, q)?;
        node.min_height = min_height.execute(e, q)?;
        node.max_width = max_width.execute(e, q)?;
        node.max_height = max_height.execute(e, q)?;
        node.margin.left = margin_left.execute(e, q)?;
        node.margin.right = margin_right.execute(e, q)?;
        node.margin.top = margin_top.execute(e, q)?;
        node.margin.bottom = margin_bottom.execute(e, q)?;
        node.padding.left = padding_left.execute(e, q)?;
        node.padding.right = padding_right.execute(e, q)?;
        node.padding.top = padding_top.execute(e, q)?;
        node.padding.bottom = padding_bottom.execute(e, q)?;
        node.border.left = border_left.execute(e, q)?;
        node.border.right = border_right.execute(e, q)?;
        node.border.top = border_top.execute(e, q)?;
        node.border.bottom = border_bottom.execute(e, q)?;
        node.flex_basis = flex_basis.execute(e, q)?;
        node.row_gap = row_gap.execute(e, q)?;
        node.column_gap = column_gap.execute(e, q)?;
        let text = text
            .as_ref()
            .and_then(|text| text.execute(e, q).ok())
            .map(Text);
        let text_font = text.as_ref().map(|_| TextFont {
            font_size,
            ..Default::default()
        });
        let text_color_component = text.as_ref().map(|_| {
            TextColor(Color::srgba(
                text_color.r,
                text_color.g,
                text_color.b,
                text_color.a,
            ))
        });
        Ok((node, text, text_font, text_color_component))
    }));

    let (node, text, text_font, text_color) =
        ((node_updater.0)(params)).unwrap_or((node, None, None, None));
    let node = commands.insert((
        node,
        node_updater,
        BackgroundColor(Color::srgba(
            source.color.r,
            source.color.g,
            source.color.b,
            source.color.a,
        )),
        DashboardNodePath {
            root,
            path: path.clone(),
        },
    ));
    if let Some(text) = text {
        node.insert(text);
    }
    if let Some(text_font) = text_font {
        node.insert(text_font);
    }
    if let Some(text_color) = text_color {
        node.insert(text_color);
    }
    let node = node.id();
    let node = DashboardNode {
        label: source.label.clone(),
        display: source.display,
        box_sizing: source.box_sizing,
        position_type: source.position_type,
        overflow: source.overflow,
        overflow_clip_margin: source.overflow_clip_margin.clone(),
        left: source.left.clone(),
        right: source.right.clone(),
        top: source.top.clone(),
        bottom: source.bottom.clone(),
        width: source.width.clone(),
        height: source.height.clone(),
        min_width: source.min_width.clone(),
        min_height: source.min_height.clone(),
        max_width: source.max_width.clone(),
        max_height: source.max_height.clone(),
        aspect_ratio: source.aspect_ratio,
        align_items: source.align_items,
        justify_items: source.justify_items,
        align_self: source.align_self,
        justify_self: source.justify_self,
        align_content: source.align_content,
        justify_content: source.justify_content,
        margin: source.margin.clone(),
        padding: source.padding.clone(),
        border: source.border.clone(),
        flex_direction: source.flex_direction,
        flex_wrap: source.flex_wrap,
        flex_grow: source.flex_grow,
        flex_shrink: source.flex_shrink,
        flex_basis: source.flex_basis.clone(),
        row_gap: source.row_gap.clone(),
        column_gap: source.column_gap.clone(),
        children: source
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                let mut path = path.clone();
                path.push(index);
                let parent_id = commands.id();
                let mut commands = commands.commands();
                let mut commands = commands.spawn(ChildOf(parent_id));
                spawn_node(child, eql, &mut commands, root, path, params)
            })
            .collect::<Result<Vec<_>, _>>()?,
        color: source.color,
        text: source.text.clone(),
        font_size: source.font_size,
        text_color: source.text_color,
        aux: node,
    };
    Ok(node)
}

type FmtExprFn = dyn for<'a, 'b> Fn(
        &'a EntityMap,
        &'a Query<'b, 'b, &'static ComponentValue>,
    ) -> Result<String, String>
    + Send
    + Sync;

pub enum CompiledFmtExpr {
    String(String),
    Closure(Box<FmtExprFn>),
}

impl CompiledFmtExpr {
    pub fn execute<'a, 'b>(
        &'a self,
        entity_map: &'a EntityMap,
        values: &'a Query<'b, 'b, &'static ComponentValue>,
    ) -> Result<String, String> {
        match self {
            CompiledFmtExpr::String(str) => Ok(str.clone()),
            CompiledFmtExpr::Closure(c) => (c)(entity_map, values),
        }
    }
}

fn compile_fmt_expr(expr: FmtExpr) -> CompiledFmtExpr {
    match expr {
        FmtExpr::String(str) => CompiledFmtExpr::String(str),
        FmtExpr::Expr(expr) => {
            let expr = compile_eql_expr(expr);
            CompiledFmtExpr::Closure(Box::new(move |e, q| {
                expr.execute(e, q).map(|v| v.to_string())
            }))
        }
    }
}

pub fn compile_fmt_string(expr: Vec<FmtExpr>) -> CompiledFmtExpr {
    let exprs: Vec<_> = expr.into_iter().map(compile_fmt_expr).collect();
    CompiledFmtExpr::Closure(Box::new(move |e, q| {
        exprs.iter().try_fold(String::new(), |mut acc, expr| {
            acc.push_str(&expr.execute(e, q)?);
            Ok(acc)
        })
    }))
}

pub trait ComponentValueExt {
    fn as_f32(&self) -> Option<f32>;
}

impl ComponentValueExt for ComponentValue {
    fn as_f32(&self) -> Option<f32> {
        match self {
            ComponentValue::U8(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::U16(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::U32(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::U64(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::I8(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::I16(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::I32(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::I64(a) => a.buf.as_buf().first().map(|&v| v as f32),
            ComponentValue::Bool(a) => a.buf.as_buf().first().map(|&v| if v { 1.0 } else { 0.0 }),
            ComponentValue::F32(array) => array.buf.as_buf().first().copied(),
            ComponentValue::F64(array) => array.buf.as_buf().first().map(|&v| v as f32),
        }
    }
}
