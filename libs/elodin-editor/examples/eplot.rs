use std::collections::BTreeMap;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use conduit::{ComponentId, EntityId};
use elodin_editor::ui::widgets::eplot::{
    EPlot, EPlotData, EPlotDataComponent, EPlotDataEntity, EPlotDataLine,
};

#[derive(Resource)]
pub struct PlotData(pub EPlotData);

enum Modifier {
    Sin,
    Cos,
    Sum,
}

fn generate_line_values(ticks: u64, tps: f64, offset: f64, modifier: Modifier) -> Vec<f64> {
    let max_x = ticks as f64 / tps;
    (1..=ticks)
        .collect::<Vec<u64>>()
        .iter()
        .map(|n| {
            let time = egui::remap_clamp(*n as f64, 1.0..=(ticks as f64), 0.0..=max_x);

            match modifier {
                Modifier::Sin => (time + offset).sin(),
                Modifier::Cos => (time + offset).cos(),
                Modifier::Sum => (time + offset).sin() + (time + offset).cos(),
            }
        })
        .collect::<Vec<f64>>()
}

fn generate_baseline(ticks: u64, tps: f64) -> Vec<f64> {
    (1..=ticks)
        .collect::<Vec<u64>>()
        .iter()
        .map(|n| *n as f64 * (1.0 / tps))
        .collect::<Vec<f64>>()
}

fn main() {
    let color_violet_500 = egui::Color32::from_rgb(0x7F, 0x70, 0xFF);
    let color_yellow_400 = egui::Color32::from_rgb(0xFE, 0xC5, 0x04);
    let color_green_300 = egui::Color32::from_rgb(0x88, 0xDE, 0x9F);

    let ticks: u64 = 1000;
    let tps: f64 = 60.0;

    let mut lines: BTreeMap<usize, EPlotDataLine> = BTreeMap::new();
    lines.insert(
        1,
        EPlotDataLine {
            label: "Z".to_string(),
            values: generate_line_values(ticks, tps, 0.0, Modifier::Sin),
            color: color_violet_500,
        },
    );
    lines.insert(
        2,
        EPlotDataLine {
            label: "Y".to_string(),
            values: generate_line_values(ticks, tps, 0.0, Modifier::Cos),
            color: color_yellow_400,
        },
    );
    lines.insert(
        3,
        EPlotDataLine {
            label: "X".to_string(),
            values: generate_line_values(ticks, tps, 0.5, Modifier::Sum),
            color: color_green_300,
        },
    );

    let mut components: BTreeMap<ComponentId, EPlotDataComponent> = BTreeMap::new();
    components.insert(
        ComponentId(1),
        EPlotDataComponent {
            label: "World Position".to_string(),
            lines,
        },
    );

    let mut entities: BTreeMap<EntityId, EPlotDataEntity> = BTreeMap::new();
    entities.insert(
        EntityId(1),
        EPlotDataEntity {
            label: "Earth".to_string(),
            components,
        },
    );

    App::new()
        .insert_resource(PlotData(EPlotData {
            baseline: generate_baseline(ticks, tps),
            entities,
        }))
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Update, ui_example_system)
        .run();
}

fn ui_example_system(mut contexts: EguiContexts, plot_data: Res<PlotData>) {
    egui::CentralPanel::default().show(contexts.ctx_mut(), |ui| {
        EPlot::new(plot_data.0.clone()).steps(6, 4).render(ui);
    });
}
