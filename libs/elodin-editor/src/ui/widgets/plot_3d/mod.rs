use std::collections::BTreeMap;

use bevy::{
    app::{Startup, Update},
    asset::Assets,
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    math::{Mat4, Vec4},
    render::view::RenderLayers,
};
use big_space::GridCell;
use gpu::LineDataHandle;
use impeller2_bevy::{CommandsExt, ComponentMetadataRegistry};
use impeller2_wkt::MaxTick;
use impeller2_wkt::{EntityMetadata, GetTimeSeries, Line3d};

use self::data::collect_entity_data;
use self::{
    data::{CollectedGraphData, LineData, PlotDataComponent, PlotDataEntity},
    gpu::{LineBundle, LineConfig, LineUniform},
};

pub mod data;
pub mod gpu;

pub fn sync_line_plot_3d(
    mut collected_graph_data: ResMut<CollectedGraphData>,
    line_plot_3d_query: Query<(Entity, &EntityMetadata, &Line3d), Without<LineDataHandle>>,
    metadata_store: Res<ComponentMetadataRegistry>,
    mut uniforms: Query<(&Line3d, &mut LineUniform), With<LineDataHandle>>,
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
            line: LineDataHandle(line),
            uniform: LineUniform {
                line_width: line_plot.line_width,
                color: Vec4::new(line_plot.color.r, line_plot.color.g, line_plot.color.b, 1.0),
                depth_bias: 0.0,
                model: Mat4::IDENTITY,
                perspective: if line_plot.perspective { 1 } else { 0 },
                #[cfg(target_arch = "wasm32")]
                _padding: Default::default(),
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
    mut query: Query<(&LineDataHandle, &Line3d)>,
    mut line_3ds: ResMut<Assets<LineData>>,
    max_tick: Res<MaxTick>,
    mut commands: Commands,
) {
    for (handle, meta) in &mut query {
        let Some(line) = line_3ds.get_mut(&handle.0) else {
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

                    let packet_id = fastrand::u64(..).to_le_bytes()[..3]
                        .try_into()
                        .expect("id wrong size");

                    commands.send_req_with_handler(
                        GetTimeSeries {
                            id: packet_id,
                            range: time_range.clone(),
                            entity_id: meta.entity,
                            component_id: meta.component_id,
                        },
                        packet_id,
                        data::time_series_handler(meta.entity, meta.component_id, time_range),
                    );
                }
            }
        }
    }
}

pub struct LinePlot3dPlugin;

impl bevy::app::Plugin for LinePlot3dPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<CollectedGraphData>()
            .add_plugins(gpu::Plot3dGpuPlugin)
            .add_systems(Update, sync_line_plot_3d)
            .add_systems(Startup, data::setup_pkt_handler)
            .add_systems(Update, collect_entity_data)
            .add_systems(Update, set_line_plot_3d_range);
    }
}
