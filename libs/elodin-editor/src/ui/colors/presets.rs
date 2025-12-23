use super::{Color32, ColorScheme, ColorSchemePreset, MINT_DEFAULT, PresetSource, REDDISH_DEFAULT};

pub const DARK: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0x1F, 0x1F, 0x1F),
    bg_secondary: Color32::from_rgb(0x16, 0x16, 0x16),

    text_primary: Color32::from_rgb(0xFF, 0xFB, 0xF0),
    text_secondary: Color32::from_rgb(0x6D, 0x6D, 0x6D),
    text_tertiary: Color32::from_rgb(0x6B, 0x6B, 0x6B),

    icon_primary: Color32::from_rgb(0xFF, 0xFB, 0xF0),
    icon_secondary: Color32::from_rgb(0x62, 0x62, 0x62),

    border_primary: Color32::from_rgb(0x2E, 0x2D, 0x2C),

    highlight: Color32::from_rgb(0x14, 0x5F, 0xCF),
    blue: Color32::from_rgb(0x14, 0x5F, 0xCF),
    error: REDDISH_DEFAULT,
    success: MINT_DEFAULT,

    shadow: Color32::BLACK,
};

pub const LIGHT: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0xFF, 0xFB, 0xF0),
    bg_secondary: Color32::from_rgb(0xE6, 0xE2, 0xD8),

    text_primary: Color32::from_rgb(0x17, 0x16, 0x15),
    text_secondary: Color32::from_rgb(0x2E, 0x2D, 0x2C),
    text_tertiary: Color32::from_rgb(0x45, 0x45, 0x44),

    icon_primary: Color32::from_rgb(0x17, 0x16, 0x15),
    icon_secondary: Color32::from_rgb(0x2E, 0x2D, 0x2C),

    border_primary: Color32::from_rgb(0xCD, 0xC3, 0xB0),

    highlight: Color32::from_rgb(0x14, 0x5F, 0xCF),
    blue: Color32::from_rgb(0x14, 0x5F, 0xCF),
    error: REDDISH_DEFAULT,
    success: MINT_DEFAULT,

    shadow: Color32::BLACK,
};

pub const CATPPUCINI_LATTE: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0xEF, 0xF1, 0xF5),
    bg_secondary: Color32::from_rgb(0xDC, 0xE0, 0xE8),

    text_primary: Color32::from_rgb(0x4C, 0x4F, 0x69),
    text_secondary: Color32::from_rgb(0x5C, 0x5F, 0x77),
    text_tertiary: Color32::from_rgb(0x6C, 0x6F, 0x85),

    icon_primary: Color32::from_rgb(0x40, 0x40, 0x40),
    icon_secondary: Color32::from_rgb(0x80, 0x80, 0x80),

    border_primary: Color32::from_rgb(0xCC, 0xD0, 0xDA),

    highlight: Color32::from_rgb(0x7C, 0x7F, 0x93),
    blue: Color32::from_rgb(0x1E, 0x66, 0xF5),
    error: Color32::from_rgb(0xE6, 0x45, 0x53),
    success: Color32::from_rgb(0x40, 0xA0, 0x2B),

    shadow: Color32::BLACK,
};

pub const CATPPUCINI_MOCHA: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0x1E, 0x1E, 0x2E),
    bg_secondary: Color32::from_rgb(0x11, 0x11, 0x1B),

    text_primary: Color32::from_rgb(0xCD, 0xD6, 0xF4),
    text_secondary: Color32::from_rgb(0xBA, 0xC2, 0xDE),
    text_tertiary: Color32::from_rgb(0xA6, 0xAD, 0xC8),

    icon_primary: Color32::from_rgb(0xCD, 0xD6, 0xF4),
    icon_secondary: Color32::from_rgb(0xBA, 0xC2, 0xDE),

    border_primary: Color32::from_rgb(0x31, 0x32, 0x44),

    highlight: Color32::from_rgb(0x93, 0x99, 0xB2),
    blue: Color32::from_rgb(0x89, 0xB4, 0xFA),
    error: Color32::from_rgb(0xF3, 0x8B, 0xA8),
    success: Color32::from_rgb(0xA6, 0xE3, 0xA1),

    shadow: Color32::BLACK,
};

pub const CATPPUCINI_MACCHIATO: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0x24, 0x27, 0x3A),
    bg_secondary: Color32::from_rgb(0x1E, 0x20, 0x30),

    text_primary: Color32::from_rgb(0xCA, 0xD3, 0xF5),
    text_secondary: Color32::from_rgb(0xA5, 0xAD, 0xCB),
    text_tertiary: Color32::from_rgb(0xB8, 0xC0, 0xE0),

    icon_primary: Color32::from_rgb(0xCA, 0xD3, 0xF5),
    icon_secondary: Color32::from_rgb(0xA5, 0xAD, 0xCB),

    border_primary: Color32::from_rgb(0x45, 0x47, 0x5A),

    highlight: Color32::from_rgb(0x6E, 0x73, 0x8D),
    blue: Color32::from_rgb(0x8A, 0xAD, 0xF4),
    error: Color32::from_rgb(0xED, 0x87, 0x96),
    success: Color32::from_rgb(0xA6, 0xDA, 0x95),

    shadow: Color32::BLACK,
};

pub const CATPPUCINI_MACCHIATO_LIGHT: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0xEF, 0xF1, 0xF5),
    bg_secondary: Color32::from_rgb(0xDC, 0xE0, 0xE8),

    text_primary: Color32::from_rgb(0x4C, 0x4F, 0x69),
    text_secondary: Color32::from_rgb(0x5C, 0x5F, 0x77),
    text_tertiary: Color32::from_rgb(0x6C, 0x6F, 0x85),

    icon_primary: Color32::from_rgb(0x40, 0x40, 0x40),
    icon_secondary: Color32::from_rgb(0x80, 0x80, 0x80),

    border_primary: Color32::from_rgb(0xCC, 0xD0, 0xDA),

    highlight: Color32::from_rgb(0x7C, 0x7F, 0x93),
    blue: Color32::from_rgb(0x1E, 0x66, 0xF5),
    error: Color32::from_rgb(0xE6, 0x45, 0x53),
    success: Color32::from_rgb(0x40, 0xA0, 0x2B),

    shadow: Color32::BLACK,
};

pub const EGGPLANT_DARK: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0x1B, 0x13, 0x25),
    bg_secondary: Color32::from_rgb(0x14, 0x0C, 0x1C),
    text_primary: Color32::from_rgb(0xF8, 0xEE, 0xFE),
    text_secondary: Color32::from_rgb(0xCB, 0xB8, 0xE0),
    text_tertiary: Color32::from_rgb(0xA6, 0x8D, 0xC6),
    icon_primary: Color32::from_rgb(0xF8, 0xEE, 0xFE),
    icon_secondary: Color32::from_rgb(0xCB, 0xB8, 0xE0),
    border_primary: Color32::from_rgb(0x2E, 0x1B, 0x36),
    highlight: Color32::from_rgb(0xB2, 0x6F, 0xD4),
    blue: Color32::from_rgb(0xB2, 0x6F, 0xD4),
    error: REDDISH_DEFAULT,
    success: MINT_DEFAULT,
    shadow: Color32::BLACK,
};

pub const EGGPLANT_LIGHT: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0xF3, 0xED, 0xFA),
    bg_secondary: Color32::from_rgb(0xE4, 0xD9, 0xF2),
    text_primary: Color32::from_rgb(0x2A, 0x18, 0x37),
    text_secondary: Color32::from_rgb(0x4A, 0x31, 0x5C),
    text_tertiary: Color32::from_rgb(0x6B, 0x4C, 0x7D),
    icon_primary: Color32::from_rgb(0x2A, 0x18, 0x37),
    icon_secondary: Color32::from_rgb(0x4A, 0x31, 0x5C),
    border_primary: Color32::from_rgb(0xC8, 0xB6, 0xD8),
    highlight: Color32::from_rgb(0x8A, 0x3B, 0xB2),
    blue: Color32::from_rgb(0x8A, 0x3B, 0xB2),
    error: REDDISH_DEFAULT,
    success: MINT_DEFAULT,
    shadow: Color32::BLACK,
};

pub const MATRIX_DARK: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0x0B, 0x0F, 0x0C),
    bg_secondary: Color32::from_rgb(0x08, 0x0B, 0x09),
    text_primary: Color32::from_rgb(0xE0, 0xFF, 0xE1),
    text_secondary: Color32::from_rgb(0x6F, 0xDC, 0x6F),
    text_tertiary: Color32::from_rgb(0x4A, 0xA4, 0x4A),
    icon_primary: Color32::from_rgb(0xE0, 0xFF, 0xE1),
    icon_secondary: Color32::from_rgb(0x6F, 0xDC, 0x6F),
    border_primary: Color32::from_rgb(0x1A, 0x3A, 0x1A),
    highlight: Color32::from_rgb(0x4A, 0xE3, 0x4A),
    blue: Color32::from_rgb(0x4A, 0xE3, 0x4A),
    error: REDDISH_DEFAULT,
    success: MINT_DEFAULT,
    shadow: Color32::BLACK,
};

pub const MATRIX_LIGHT: ColorScheme = ColorScheme {
    bg_primary: Color32::from_rgb(0xE6, 0xF5, 0xE7),
    bg_secondary: Color32::from_rgb(0xD2, 0xE8, 0xD4),
    text_primary: Color32::from_rgb(0x0F, 0x1F, 0x12),
    text_secondary: Color32::from_rgb(0x27, 0x44, 0x2B),
    text_tertiary: Color32::from_rgb(0x3C, 0x5C, 0x40),
    icon_primary: Color32::from_rgb(0x0F, 0x1F, 0x12),
    icon_secondary: Color32::from_rgb(0x27, 0x44, 0x2B),
    border_primary: Color32::from_rgb(0x9E, 0xC9, 0xA1),
    highlight: Color32::from_rgb(0x2F, 0x9E, 0x34),
    blue: Color32::from_rgb(0x2F, 0x9E, 0x34),
    error: REDDISH_DEFAULT,
    success: MINT_DEFAULT,
    shadow: Color32::BLACK,
};

pub fn builtin_presets() -> Vec<ColorSchemePreset> {
    vec![
        ColorSchemePreset::new(
            "default",
            "Default",
            PresetSource::Builtin,
            &DARK,
            Some(&LIGHT),
        ),
        ColorSchemePreset::new(
            "eggplant",
            "Eggplant",
            PresetSource::Builtin,
            &EGGPLANT_DARK,
            Some(&EGGPLANT_LIGHT),
        ),
        ColorSchemePreset::new(
            "catppuccini-macchiato",
            "Catppuccini Macchiato",
            PresetSource::Builtin,
            &CATPPUCINI_MACCHIATO,
            Some(&CATPPUCINI_MACCHIATO_LIGHT),
        ),
        ColorSchemePreset::new(
            "catppuccini-mocha",
            "Catppuccini Mocha",
            PresetSource::Builtin,
            &CATPPUCINI_MOCHA,
            None,
        ),
        ColorSchemePreset::new(
            "catppuccini-latte",
            "Catppuccini Latte",
            PresetSource::Builtin,
            &CATPPUCINI_MOCHA,
            Some(&CATPPUCINI_LATTE),
        ),
        ColorSchemePreset::new(
            "matrix",
            "Matrix",
            PresetSource::Builtin,
            &MATRIX_DARK,
            Some(&MATRIX_LIGHT),
        ),
    ]
}
