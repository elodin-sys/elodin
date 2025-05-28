use std::fmt::Display;
use std::str::FromStr;

use egui::{Response, Ui};
use hifitime::prelude::*;

use crate::ui::colors::get_scheme;

pub fn time_label(time: Epoch) -> impl for<'a> FnOnce(&'a mut Ui) -> Response {
    move |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.style_mut().override_text_valign = Some(egui::Align::BOTTOM);

            let fmt = Format::from_str("%Y-%m-%dT%H:%M:%S").unwrap();
            let formatter = Formatter::new(time, fmt);
            let time_value =
                egui::RichText::new(formatter.to_string()).color(get_scheme().text_primary);
            let fmt = Format::from_str("%f").unwrap();
            let formatter = Formatter::new(time, fmt);
            let subsecond = egui::RichText::new(format!(".{}", formatter))
                .size(10.0)
                .color(get_scheme().text_secondary);

            let mut elements = [time_value, subsecond];
            if ui.layout().main_dir == egui::Direction::RightToLeft {
                elements.reverse();
            }
            for elem in elements.into_iter() {
                ui.add(
                    egui::Label::new(elem)
                        .halign(egui::Align::BOTTOM)
                        .selectable(false),
                );
            }
        })
        .response
    }
}

#[derive(Clone, Copy)]
pub struct PrettyDuration(pub hifitime::Duration);

impl Display for PrettyDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nanos = self.0.total_nanoseconds();
        if nanos == 0 {
            write!(f, "0")
        } else {
            write!(f, "{}", self.0)
        }
    }
}
