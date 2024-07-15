use std::collections::BTreeMap;

use bevy::{
    app::{Plugin, Update},
    asset::{Assets, Handle},
    ecs::{
        entity::Entity,
        event::EventWriter,
        query::{With, Without},
        schedule::IntoSystemConfigs,
        system::{Commands, Query, Res, ResMut},
    },
    math::{Mat4, Vec4},
    render::view::RenderLayers,
};
use big_space::GridCell;
use conduit::{
    bevy::Tick,
    query::MetadataStore,
    well_known::{EntityMetadata, Line3d},
    ControlMsg,
};

use self::{
    data::{CollectedGraphData, LineData, PlotDataComponent, PlotDataEntity},
    gpu::{LineBundle, LineConfig, LineUniform},
};

use super::entity_data::collect_entity_data;

pub mod data;
pub mod gpu;

pub fn sync_line_plot_3d(
    mut collected_graph_data: ResMut<CollectedGraphData>,
    line_plot_3d_query: Query<(Entity, &EntityMetadata, &Line3d), Without<Handle<LineData>>>,
    metadata_store: Res<MetadataStore>,
    mut uniforms: Query<(&Line3d, &mut LineUniform), With<Handle<LineData>>>,
    mut line_3ds: ResMut<Assets<LineData>>,
    mut commands: Commands,
) {
    for (entity, metadata, line_plot) in line_plot_3d_query.iter() {
        let line_3d = collected_graph_data
            .entities
            .entry(line_plot.entity)
            .or_insert_with(|| PlotDataEntity {
                label: metadata.name.clone(),
                components: BTreeMap::new(),
            });
        let Some(metadata) = metadata_store.get_metadata(&line_plot.component_id) else {
            continue;
        };

        let data_component = line_3d
            .components
            .entry(line_plot.component_id)
            .or_insert_with(|| {
                PlotDataComponent::new(metadata.name.clone(), Some(line_plot.index))
            });
        let line = if let Some(line) = &data_component.line {
            line.clone()
        } else {
            let line = line_3ds.add(LineData::default());
            data_component.line = Some(line.clone());
            line
        };
        commands.entity(entity).insert(LineBundle {
            line,
            uniform: LineUniform {
                line_width: line_plot.line_width,
                color: Vec4::new(line_plot.color.r, line_plot.color.g, line_plot.color.b, 1.0),
                depth_bias: 0.0,
                model: Mat4::IDENTITY,
                perspective: if line_plot.perspective { 1 } else { 0 },
            },
            config: LineConfig {
                render_layers: RenderLayers::default(),
            },
            global_transform: Default::default(),
            transform: Default::default(),
            grid_cell: GridCell::default(),
        });
    }
    for (line_plot, mut uniform) in uniforms.iter_mut() {
        uniform.color = Vec4::new(line_plot.color.r, line_plot.color.g, line_plot.color.b, 1.0);
        uniform.line_width = line_plot.line_width;
        uniform.perspective = if line_plot.perspective { 1 } else { 0 };
    }
}

pub fn set_line_plot_3d_range(
    mut query: Query<(&Handle<LineData>, &Line3d)>,
    mut line_3ds: ResMut<Assets<LineData>>,
    max_tick: Res<Tick>,
    mut control_msg: EventWriter<ControlMsg>,
) {
    for (handle, meta) in &mut query {
        let Some(line) = line_3ds.get_mut(handle) else {
            continue;
        };
        let new_range = 0..max_tick.0 as usize;
        let old_range = std::mem::replace(&mut line.range, new_range.clone());
        if old_range != (0..(max_tick.0.saturating_sub(1) as usize)) {
            line.mark_unfetched();
        }

        for chunk in line.data.chunks_range(line.range.clone()) {
            if !chunk.unfetched.is_empty() {
                let start = chunk.unfetched.min().expect("unexpected empty chunk") as u64;
                let mut end = chunk.unfetched.max().expect("unexpected empty chunk") as u64;
                if start == end {
                    end += 1;
                }
                let end = end + 1;
                let time_range = start..end;

                if end.saturating_sub(start) > 0 {
                    chunk.unfetched.remove_range(start as u32..end as u32);
                    control_msg.send(ControlMsg::Query {
                        time_range,
                        query: conduit::Query {
                            component_id: meta.component_id,
                            with_component_ids: vec![],
                            entity_ids: vec![meta.entity],
                        },
                    });
                }
            }
        }
    }
}

pub struct LinePlot3dPlugin;

impl Plugin for LinePlot3dPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<CollectedGraphData>()
            .add_plugins(gpu::Plot3dGpuPlugin)
            .add_systems(Update, sync_line_plot_3d)
            .add_systems(Update, set_line_plot_3d_range.after(collect_entity_data));
    }
}
