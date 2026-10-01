use std::fmt::Display;
use std::str::FromStr;

use egui::{Response, Ui};
use hifitime::prelude::*;
use impeller::types::Timestamp;

use crate::ui::colors::get_scheme;

/// Unix microseconds of 2000-01-01T00:00:00Z. Without clock metadata,
/// treat small timestamps as elapsed simulation time instead of a 1970 date.
const WALL_CLOCK_UNIX_MICROS: i64 = 946_684_800 * 1_000_000;

pub fn is_wall_clock(micros: i64) -> bool {
    micros >= WALL_CLOCK_UNIX_MICROS
}

/// Seconds since the epoch, trailing zeros stripped. `1_500_000` is `1.5`.
pub fn format_seconds(micros: i64) -> String {
    let negative = micros < 0;
    let abs = micros.unsigned_abs();
    let secs = abs / 1_000_000;
    let frac = abs % 1_000_000;
    let mut body = if frac == 0 {
        secs.to_string()
    } else {
        let mut frac_s = format!("{frac:06}");
        while frac_s.ends_with('0') {
            frac_s.pop();
        }
        format!("{secs}.{frac_s}")
    };
    if negative {
        body.insert(0, '-');
    }
    body
}

/// An explicitly UTC timestamp, preserving all six microsecond digits.
pub fn format_wall_clock(micros: i64) -> String {
    let epoch =
        Epoch::from_unix_duration(Duration::from_total_nanoseconds(i128::from(micros) * 1000));
    let fmt = Format::from_str("%Y-%m-%dT%H:%M:%S").unwrap();
    format!(
        "{}.{:06}Z",
        Formatter::new(epoch, fmt),
        micros.rem_euclid(1_000_000)
    )
}

pub fn format_time_input(micros: i64) -> String {
    if is_wall_clock(micros) {
        format_wall_clock(micros)
    } else {
        format!("{} s", format_seconds(micros))
    }
}

/// Accept a microsecond integer, `12.5` / `12.5s` seconds, or a UTC timestamp.
pub fn parse_time_input(text: &str) -> Option<i64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Some(micros) = text.strip_suffix("µs").or_else(|| text.strip_suffix("us")) {
        return micros.trim().parse().ok();
    }
    if let Some(seconds) = text.strip_suffix('s').or_else(|| text.strip_suffix('S')) {
        return parse_seconds(seconds.trim());
    }
    if text.contains('T') || (text.contains('-') && text.contains(':')) {
        return parse_wall_clock(text);
    }
    if text.contains('.') {
        return parse_seconds(text);
    }
    text.parse::<i64>().ok()
}

fn parse_seconds(text: &str) -> Option<i64> {
    let negative = text.starts_with('-');
    let magnitude = text.strip_prefix(['-', '+']).unwrap_or(text);
    let (whole, fraction) = magnitude.split_once('.').unwrap_or((magnitude, ""));
    if whole.is_empty() && fraction.is_empty()
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    // Avoid f64 rounding of UTC-sized values and saturating integer casts.
    // Extra zero digits are harmless; sub-microsecond inputs are rejected.
    let fraction = fraction.trim_end_matches('0');
    if fraction.len() > 6 {
        return None;
    }
    let whole = if whole.is_empty() {
        0
    } else {
        whole.parse::<i128>().ok()?
    };
    let fraction = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<i128>().ok()? * 10_i128.pow(6 - fraction.len() as u32)
    };
    let micros = whole.checked_mul(1_000_000)?.checked_add(fraction)?;
    i64::try_from(if negative { -micros } else { micros }).ok()
}

fn parse_wall_clock(text: &str) -> Option<i64> {
    let text = text.strip_suffix('Z').unwrap_or(text);
    let epoch = Epoch::from_format_str(text, "%Y-%m-%dT%H:%M:%S.%f")
        .or_else(|_| Epoch::from_format_str(text, "%Y-%m-%dT%H:%M:%S"))
        .or_else(|_| Epoch::from_format_str(text, "%Y-%m-%d %H:%M:%S.%f"))
        .or_else(|_| Epoch::from_format_str(text, "%Y-%m-%d %H:%M:%S"))
        .ok()?;
    let nanos =
        (epoch.to_utc_duration() - hifitime::UNIX_REF_EPOCH.to_utc_duration()).total_nanoseconds();
    if nanos % 1000 != 0 {
        return None;
    }
    i64::try_from(nanos / 1000).ok()
}

pub fn time_label(time: Epoch) -> impl for<'a> FnOnce(&'a mut Ui) -> Response {
    move |ui| {
        let micros = Timestamp::from(time).0;
        if !is_wall_clock(micros) {
            let text = format!("{} s", format_seconds(micros));
            return ui.add(
                egui::Label::new(egui::RichText::new(text).color(get_scheme().text_primary))
                    .halign(egui::Align::BOTTOM)
                    .selectable(false),
            );
        }

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

#[cfg(test)]
mod tests {
    use super::{format_seconds, format_time_input, is_wall_clock, parse_time_input};

    #[test]
    fn seconds_strip_trailing_zeros() {
        assert_eq!(format_seconds(0), "0");
        assert_eq!(format_seconds(1_500_000), "1.5");
        assert_eq!(format_seconds(1), "0.000001");
        assert_eq!(format_seconds(-2_500_000), "-2.5");
    }

    #[test]
    fn epoch_times_are_seconds_and_later_times_are_dates() {
        assert!(!is_wall_clock(12_000_000));
        assert!(is_wall_clock(1_700_000_000_000_000));
        assert_eq!(format_time_input(12_000_000), "12 s");
        assert!(format_time_input(1_700_000_000_000_000).starts_with("2023-11-14T22:13:20"));
    }

    #[test]
    fn parse_micros_seconds_and_utc() {
        assert_eq!(parse_time_input("1500000"), Some(1_500_000));
        assert_eq!(parse_time_input("1.5"), Some(1_500_000));
        assert_eq!(parse_time_input("1.5s"), Some(1_500_000));
        assert_eq!(parse_time_input("1500000 µs"), Some(1_500_000));
        assert_eq!(parse_time_input("  -2.5 s"), Some(-2_500_000));
        assert_eq!(
            parse_time_input("2023-11-14T22:13:20"),
            Some(1_700_000_000_000_000)
        );
        assert_eq!(parse_time_input("not a time"), None);
    }

    #[test]
    fn copied_display_values_round_trip_without_changing_units_or_precision() {
        for micros in [0, 1, -1, 12_000_000, -2_500_000, 1_700_000_000_123_457] {
            assert_eq!(parse_time_input(&format_time_input(micros)), Some(micros));
            assert_eq!(parse_time_input(&micros.to_string()), Some(micros));
        }
        assert_eq!(
            parse_time_input("2023-11-14 22:13:20.123457Z"),
            Some(1_700_000_000_123_457)
        );
        assert_eq!(
            parse_time_input("1700000000.123457s"),
            Some(1_700_000_000_123_457)
        );
    }

    #[test]
    fn seconds_accept_exact_limits_and_reject_invalid_or_overflowing_values() {
        assert_eq!(parse_time_input("9223372036854.775807s"), Some(i64::MAX));
        assert_eq!(parse_time_input("-9223372036854.775808s"), Some(i64::MIN));
        for value in [
            "9223372036854.775808s",
            "-9223372036854.775809s",
            "NaNs",
            "infs",
            "1.0000001s",
            ".s",
            "s",
            "1..2s",
        ] {
            assert_eq!(parse_time_input(value), None, "{value}");
        }
    }
}

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
