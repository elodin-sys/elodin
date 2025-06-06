use bevy_egui::egui;
use hifitime::Epoch;

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

pub fn time_label_ms(time: f64) -> String {
    let time_in_seconds = time.floor() as usize;
    let milliseconds = (time - time_in_seconds as f64) * 1000.0;
    let seconds = time_in_seconds % 60;
    let minutes = (time_in_seconds / 60) % 60;

    format!("{minutes:0>2}:{seconds:0>2}.{milliseconds:0>3.0}")
}

pub struct FriendlyEpoch(pub Epoch);

impl std::fmt::Display for FriendlyEpoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (y, month, day, hour, min, sec, ns) = self.0.to_gregorian_utc();
        let sec = (sec as f64) + (ns as f64) * 1e-9;
        write!(f, "{y}-{month}-{day}T{hour}:{min}:{sec:0.3}")
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
            egui::pos2(
                self.min.x + margin.left as f32,
                self.min.y + margin.top as f32,
            ),
            egui::pos2(
                self.max.x - margin.right as f32,
                self.max.y - margin.bottom as f32,
            ),
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
        self.left = left as i8;
        self
    }

    fn right(mut self, right: f32) -> Self {
        self.right = right as i8;
        self
    }

    fn top(mut self, top: f32) -> Self {
        self.top = top as i8;
        self
    }

    fn bottom(mut self, bottom: f32) -> Self {
        self.bottom = bottom as i8;
        self
    }
}

pub fn format_num(mut num: f64) -> String {
    if !num.is_finite() {
        return format!("{}", num);
    }
    let width: usize = 8;
    // need 2 characters for the sign and the decimal point
    let digit_width = width - 2;
    let digits = (num.abs().log10().ceil() as usize).saturating_add(1);
    let precision = digit_width.saturating_sub(digits).clamp(1, 3);
    // round to the nearest multiple of 10^(-precision)
    num = (num * 10.0_f64.powi(precision as i32)).round() / 10.0_f64.powi(precision as i32);
    // -0.0 is weird, just make it 0.0
    if num == -0.0 {
        num = 0.0;
    }
    let use_scientific = num.abs() >= 1e5;
    if use_scientific {
        format!("{:.2e}", num)
    } else {
        format!("{:.*}", precision, num)
    }
}

#[cfg(test)]
mod tests {
    use super::format_num;

    #[test]
    fn test_format_num() {
        assert_eq!(format_num(0.0), "0.000");
        assert_eq!(format_num(1.0), "1.000");
        assert_eq!(format_num(9.999), "9.999");
        assert_eq!(format_num(9.9999), "10.000");
        assert_eq!(format_num(99.999), "99.999");
        assert_eq!(format_num(999.99), "999.99");
        assert_eq!(format_num(9999.9), "9999.9");
        assert_eq!(format_num(99999.9), "99999.9");
        assert_eq!(format_num(99999.99), "1.00e5");
        assert_eq!(format_num(100000.0), "1.00e5");

        // test negatives:
        assert_eq!(format_num(-0.0), "0.000");
        assert_eq!(format_num(-1.0), "-1.000");
        assert_eq!(format_num(-9.999), "-9.999");
        assert_eq!(format_num(-9.9999), "-10.000");
        assert_eq!(format_num(-99.999), "-99.999");
        assert_eq!(format_num(-999.99), "-999.99");
        assert_eq!(format_num(-9999.9), "-9999.9");
        assert_eq!(format_num(-99999.9), "-99999.9");
        assert_eq!(format_num(-99999.99), "-1.00e5");
        assert_eq!(format_num(-100000.0), "-1.00e5");
    }
}
