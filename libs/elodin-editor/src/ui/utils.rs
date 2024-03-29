use bevy_egui::egui;
use conduit::{query::MetadataStore, ComponentId, ComponentValue, TagValue};

pub fn get_galley_layout_job(
    text: impl ToString,
    wrap_width: f32,
    font_id: egui::FontId,
    text_color: egui::Color32,
) -> egui::text::LayoutJob {
    let rich_text = egui::RichText::new(text.to_string()).color(text_color);
    let mut layout_job = egui::text::LayoutJob::single_section(
        rich_text.text().to_string(),
        egui::TextFormat::simple(font_id, text_color),
    );
    layout_job.wrap.max_width = wrap_width;
    layout_job.wrap.max_rows = 1;
    layout_job.wrap.break_anywhere = true;

    layout_job
}

pub fn time_label(time_in_seconds: usize, with_hours: bool) -> String {
    let seconds = time_in_seconds % 60;
    let minutes = (time_in_seconds / 60) % 60;
    if with_hours {
        let hours = (time_in_seconds / (60 * 60)) % 60;
        format!("{hours:0>2}:{minutes:0>2}:{seconds:0>2}")
    } else {
        format!("{minutes:0>2}:{seconds:0>2}")
    }
}

pub fn time_label_ms(time: f64) -> String {
    let time_in_seconds = time.floor() as usize;
    let milliseconds = (time - time_in_seconds as f64) * 1000.0;
    let seconds = time_in_seconds % 60;
    let minutes = (time_in_seconds / 60) % 60;

    format!("{minutes:0>2}:{seconds:0>2}.{milliseconds:0>3.0}")
}

pub fn component_value_to_vec(component_value: &ComponentValue) -> Vec<f64> {
    match component_value {
        ComponentValue::F64(arr) => arr.iter().cloned().collect::<Vec<f64>>(),
        _ => vec![],
    }
}

pub fn get_component_label(metadata_store: &MetadataStore, component_id: &ComponentId) -> String {
    if let Some(name) = metadata_store
        .get_metadata(component_id)
        .and_then(|m| m.tags.get("name"))
        .and_then(TagValue::as_str)
    {
        name.to_string().to_uppercase()
    } else {
        format!("ID[{}]", component_id.0)
    }
}

pub fn get_rects_from_relative_width(
    rect: egui::Rect,
    relative_width: f32,
    height: f32,
) -> (egui::Rect, egui::Rect) {
    let first_rect_width = rect.width() * relative_width;

    let first_rect = egui::Rect::from_min_size(rect.min, egui::vec2(first_rect_width, height));

    let second_rect = egui::Rect::from_min_size(
        rect.translate(egui::vec2(first_rect_width, 0.0)).min,
        egui::vec2(rect.width() - first_rect_width, height),
    );

    (first_rect, second_rect)
}

pub trait Shrink4 {
    fn shrink4(self, margin: egui::Margin) -> egui::Rect;
}

impl Shrink4 for egui::Rect {
    fn shrink4(self, margin: egui::Margin) -> Self {
        egui::Rect::from_min_max(
            egui::pos2(self.min.x + margin.left, self.min.y + margin.top),
            egui::pos2(self.max.x - margin.right, self.max.y - margin.bottom),
        )
    }
}

pub trait MarginSides {
    fn left(self, left: f32) -> egui::Margin;
    fn right(self, right: f32) -> egui::Margin;
    fn top(self, top: f32) -> egui::Margin;
    fn bottom(self, bottom: f32) -> egui::Margin;
}

impl MarginSides for egui::Margin {
    fn left(mut self, left: f32) -> Self {
        self.left = left;
        self
    }

    fn right(mut self, right: f32) -> Self {
        self.right = right;
        self
    }

    fn top(mut self, top: f32) -> Self {
        self.top = top;
        self
    }

    fn bottom(mut self, bottom: f32) -> Self {
        self.bottom = bottom;
        self
    }
}
