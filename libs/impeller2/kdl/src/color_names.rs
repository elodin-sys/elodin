use impeller2_wkt::Color;

pub const NAMED_COLORS: [(&str, Color); 19] = [
    ("black", Color::BLACK),
    ("white", Color::WHITE),
    ("blue", Color::BLUE),
    ("red", Color::RED),
    ("orange", Color::ORANGE),
    ("yellow", Color::YELLOW),
    ("yalk", Color::YALK),
    ("pink", Color::PINK),
    ("cyan", Color::CYAN),
    ("gray", Color::GRAY),
    ("green", Color::GREEN),
    ("mint", Color::MINT),
    ("turquoise", Color::TURQUOISE),
    ("slate", Color::SLATE),
    ("pumpkin", Color::PUMPKIN),
    ("yolk", Color::YOLK),
    ("peach", Color::PEACH),
    ("reddish", Color::REDDISH),
    ("hyperblue", Color::HYPERBLUE),
];

pub fn color_from_name(name: &str) -> Option<Color> {
    NAMED_COLORS
        .iter()
        .find_map(|(key, value)| if key == &name { Some(*value) } else { None })
}

pub fn color_to_ints(color: &Color) -> (i128, i128, i128, i128) {
    (
        float_color_component_to_int(color.r),
        float_color_component_to_int(color.g),
        float_color_component_to_int(color.b),
        float_color_component_to_int(color.a),
    )
}

pub fn name_from_color(color: &Color) -> Option<&'static str> {
    let (r, g, b, a) = color_to_ints(color);
    name_from_components(r, g, b, a)
}

pub fn name_from_components(r: i128, g: i128, b: i128, a: i128) -> Option<&'static str> {
    if a != 255 {
        return None;
    }
    for (name, color) in NAMED_COLORS {
        let (nr, ng, nb, na) = color_to_ints(&color);
        if nr == r && ng == g && nb == b && na == a {
            return Some(name);
        }
    }
    None
}

pub fn float_color_component_to_int(component: f32) -> i128 {
    let clamped = component.clamp(0.0, 1.0);
    (clamped * 255.0).round() as i128
}
