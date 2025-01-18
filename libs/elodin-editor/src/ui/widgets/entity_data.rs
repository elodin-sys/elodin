use bevy::asset::Handle;
use bevy::ecs::entity::Entity;
use bevy::transform::components::{GlobalTransform, Transform};
use big_space::GridCell;

use super::plot_3d::data::LineData;

pub type Plot3dData = (
    Entity,
    &'static Handle<LineData>,
    &'static mut GridCell<i128>,
    &'static Transform,
    &'static mut GlobalTransform,
);

// #[allow(clippy::too_many_arguments)]
// pub fn collect_entity_data(
//     mut collected_graph_data: ResMut<CollectedGraphData>,
//     mut graphs: Query<&mut GraphState>,
//     reader: Res<ImpellerMsgReceiver>,
//     mut lines: ResMut<Assets<Line>>,
//     mut collected_graph_data_3d: ResMut<plot_3d::data::CollectedGraphData>,
//     mut plots_3d: Query<Plot3dData, Without<FloatingOrigin>>,
//     mut lines_3d: ResMut<Assets<LineData>>,
//     floating_origin: Res<FloatingOriginSettings>,
//     origin: Query<(&GridCell<i128>, &FloatingOrigin)>,
//     mut commands: Commands,
// ) {
//     while let Ok(msg) = reader.try_recv() {
//         plot_3d::data::collect_entity_data(
//             &msg,
//             &mut collected_graph_data_3d,
//             &mut plots_3d,
//             &mut lines_3d,
//             &floating_origin,
//             &origin,
//             &mut commands,
//         );
//         plot::collect_entity_data(
//             &msg,
//             &mut collected_graph_data,
//             &mut graphs,
//             &mut lines,
//             &mut commands,
//         );
//     }
// }
