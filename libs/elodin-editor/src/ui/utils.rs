use bevy_egui::egui;

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
