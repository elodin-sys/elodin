use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Mutex, OnceLock,
        atomic::{self, AtomicPtr},
    },
};

use egui::Color32;
use impeller2_wkt::Color;
use serde::{Deserialize, Serialize};

use crate::dirs;

mod presets;
pub use presets::*;

pub const WHITE: Color32 = Color32::WHITE;
pub const TRANSPARENT: Color32 = Color32::TRANSPARENT;

pub const TURQUOISE_DEFAULT: Color32 = Color32::from_rgb(0x69, 0xB3, 0xBF);
pub const SLATE_DEFAULT: Color32 = Color32::from_rgb(0x7F, 0x70, 0xFF);
pub const PUMPKIN_DEFAULT: Color32 = Color32::from_rgb(0xFF, 0x6F, 0x1E);
pub const YOLK_DEFAULT: Color32 = Color32::from_rgb(0xFE, 0xC5, 0x04);
pub const PEACH_DEFAULT: Color32 = Color32::from_rgb(0xFF, 0xD7, 0xB3);
pub const REDDISH_DEFAULT: Color32 = Color32::from_rgb(0xE9, 0x4B, 0x14);
pub const HYPERBLUE_DEFAULT: Color32 = Color32::from_rgb(0x14, 0x5F, 0xCF);
pub const MINT_DEFAULT: Color32 = Color32::from_rgb(0x88, 0xDE, 0x9F);
pub const BONE_DEFAULT: Color32 = Color32::from_rgb(0xE4, 0xD9, 0xC3);

pub const TURQUOISE_40: Color32 = Color32::from_rgb(0x38, 0x55, 0x59);
pub const SLATE_40: Color32 = Color32::from_rgb(0x41, 0x3A, 0x73);
pub const PUMPKIN_40: Color32 = Color32::from_rgb(0x74, 0x3A, 0x1A);
pub const YOLK_40: Color32 = Color32::from_rgb(0x73, 0x5C, 0x0D);
pub const PEACH_40: Color32 = Color32::from_rgb(0x74, 0x63, 0x54);
pub const REDDISH_40: Color32 = Color32::from_rgb(0x6B, 0x2B, 0x15);
pub const HYPERBLUE_40: Color32 = Color32::from_rgb(0x16, 0x33, 0x60);
pub const MINT_40: Color32 = Color32::from_rgb(0x43, 0x66, 0x4C);

pub const SURFACE_PRIMARY: Color32 = Color32::from_rgb(0x1F, 0x1F, 0x1F);
pub const SURFACE_SECONDARY: Color32 = Color32::from_rgb(0x16, 0x16, 0x16);

pub fn get_color_by_index_solid(index: usize) -> Color32 {
    let colors = [
        TURQUOISE_DEFAULT,
        SLATE_DEFAULT,
        PUMPKIN_DEFAULT,
        YOLK_DEFAULT,
        PEACH_DEFAULT,
        REDDISH_DEFAULT,
        HYPERBLUE_DEFAULT,
        MINT_DEFAULT,
    ];
    colors[index % colors.len()]
}

pub const ALL_COLORS_DARK: &[Color32] = &[
    Color32::from_rgb(0x6A, 0x9B, 0xA5),
    Color32::from_rgb(0x72, 0x57, 0xB3),
    Color32::from_rgb(0xC6, 0x6B, 0x42),
    Color32::from_rgb(0xD6, 0xA4, 0x36),
    Color32::from_rgb(0xB6, 0x52, 0x2D),
    Color32::from_rgb(0x42, 0x69, 0xA8),
    Color32::from_rgb(0x7E, 0xBF, 0x7F),
    Color32::from_rgb(0x5A, 0x82, 0x90),
    Color32::from_rgb(0x64, 0x48, 0xA8),
    Color32::from_rgb(0xB0, 0x62, 0x3E),
    Color32::from_rgb(0xC9, 0x93, 0x36),
    Color32::from_rgb(0x9D, 0x4B, 0x2A),
    Color32::from_rgb(0x34, 0x5C, 0x91),
    Color32::from_rgb(0x6A, 0x9C, 0x6A),
    Color32::from_rgb(0x50, 0x6A, 0x7A),
    Color32::from_rgb(0x5C, 0x3C, 0xA5),
    Color32::from_rgb(0x9C, 0x58, 0x3A),
    Color32::from_rgb(0xB6, 0x88, 0x30),
    Color32::from_rgb(0x82, 0x36, 0x21),
    Color32::from_rgb(0x2C, 0x4B, 0x7B),
    Color32::from_rgb(0x5F, 0x8A, 0x5F),
    Color32::from_rgb(0x43, 0x5B, 0x70),
    Color32::from_rgb(0x4A, 0x31, 0x92),
    Color32::from_rgb(0x85, 0x48, 0x2F),
    Color32::from_rgb(0x9B, 0x71, 0x30),
    Color32::from_rgb(0x6B, 0x27, 0x17),
    Color32::from_rgb(0x24, 0x3C, 0x66),
    Color32::from_rgb(0x48, 0x6D, 0x48),
    Color32::from_rgb(0x37, 0x48, 0x5A),
    Color32::from_rgb(0x3A, 0x25, 0x86),
    Color32::from_rgb(0x6C, 0x3C, 0x28),
    Color32::from_rgb(0x84, 0x5D, 0x28),
    Color32::from_rgb(0x52, 0x1F, 0x14),
    Color32::from_rgb(0x1C, 0x2A, 0x58),
    Color32::from_rgb(0x3D, 0x68, 0x3D),
    Color32::from_rgb(0x2C, 0x3A, 0x48),
    Color32::from_rgb(0x33, 0x23, 0x7A),
    Color32::from_rgb(0x53, 0x2D, 0x1C),
    Color32::from_rgb(0x63, 0x46, 0x22),
    Color32::from_rgb(0x39, 0x19, 0x0E),
    Color32::from_rgb(0x14, 0x1A, 0x3A),
    Color32::from_rgb(0x2A, 0x4B, 0x2A),
    Color32::from_rgb(0x1E, 0x29, 0x36),
    Color32::from_rgb(0x25, 0x18, 0x64),
    Color32::from_rgb(0x45, 0x24, 0x1A),
    Color32::from_rgb(0x57, 0x35, 0x1E),
    Color32::from_rgb(0x2C, 0x11, 0x0A),
    Color32::from_rgb(0x10, 0x16, 0x2E),
    Color32::from_rgb(0x22, 0x3C, 0x22),
    Color32::from_rgb(0x16, 0x20, 0x2C),
];

static ALL_COLORS_LIGHT: OnceLock<Vec<Color32>> = OnceLock::new();

fn scale_color(color: Color32, scale: f32) -> Color32 {
    let [r, g, b, a] = color.to_srgba_unmultiplied();
    let scale = scale.clamp(0.0, 1.0);
    Color32::from_rgba_unmultiplied(
        (r as f32 * scale) as u8,
        (g as f32 * scale) as u8,
        (b as f32 * scale) as u8,
        a,
    )
}

pub fn all_colors() -> &'static [Color32] {
    let mode = current_selection().mode;
    if mode.eq_ignore_ascii_case("light") {
        ALL_COLORS_LIGHT.get_or_init(|| {
            ALL_COLORS_DARK
                .iter()
                .copied()
                .map(|color| scale_color(color, 0.85))
                .collect()
        })
    } else {
        ALL_COLORS_DARK
    }
}

pub fn get_color_by_index_all(index: usize) -> Color32 {
    let colors = all_colors();
    colors[index % colors.len()]
}

pub fn with_opacity(color: Color32, opacity: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (255.0 * opacity) as u8)
}

pub trait EColor {
    fn into_color32(self) -> Color32;
    fn from_color32(color: egui::Color32) -> Self;
}

impl EColor for Color {
    fn into_color32(self) -> Color32 {
        Color32::from_rgba_unmultiplied(
            (255.0 * self.r) as u8,
            (255.0 * self.g) as u8,
            (255.0 * self.b) as u8,
            (255.0 * self.a) as u8,
        )
    }

    fn from_color32(color: egui::Color32) -> Self {
        let [r, g, b, a] = color.to_srgba_unmultiplied();
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }
}

pub trait ColorExt {
    fn into_bevy(self) -> ::bevy::prelude::Color;
    fn opacity(self, opacity: f32) -> Self;
    fn from_bevy(c: ::bevy::prelude::Color) -> Self;
}

impl ColorExt for Color32 {
    fn into_bevy(self) -> ::bevy::prelude::Color {
        let [r, g, b, a] = self.to_srgba_unmultiplied().map(|c| c as f32 / 255.0);
        ::bevy::prelude::Color::srgba(r, g, b, a)
    }

    fn opacity(self, opacity: f32) -> Self {
        with_opacity(self, opacity)
    }

    fn from_bevy(c: ::bevy::prelude::Color) -> Self {
        use ::bevy::color::ColorToPacked;
        let [r, g, b, a] = c.to_srgba().to_u8_array();
        Color32::from_rgba_unmultiplied(r, g, b, a)
    }
}

pub mod bevy {
    use bevy::prelude::Color;
    pub const RED: Color = Color::srgb(0.91, 0.29, 0.08);
    pub const GREEN: Color = Color::srgb(0.53, 0.87, 0.62);
    pub const BLUE: Color = Color::srgb(0.08, 0.38, 0.82);
    pub const GREY_900: Color = Color::srgb(0.2, 0.2, 0.2);
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ColorScheme {
    pub bg_primary: Color32,
    pub bg_secondary: Color32,

    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_tertiary: Color32,

    pub icon_primary: Color32,
    pub icon_secondary: Color32,

    pub border_primary: Color32,

    pub highlight: Color32,
    pub blue: Color32,
    pub error: Color32,
    pub success: Color32,

    pub shadow: Color32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetSource {
    Builtin,
    User,
}

#[derive(Clone)]
pub struct ColorSchemePreset {
    pub name: String,
    pub label: String,
    pub source: PresetSource,
    pub dark: &'static ColorScheme,
    pub light: Option<&'static ColorScheme>,
}

#[derive(Clone, Debug)]
pub struct SchemeSelection {
    pub scheme: String,
    pub mode: String,
}

struct PresetRegistry {
    presets: Vec<ColorSchemePreset>,
    index: HashMap<String, usize>,
}

static PRESET_REGISTRY: OnceLock<PresetRegistry> = OnceLock::new();
static COLOR_SCHEME: AtomicPtr<ColorScheme> = AtomicPtr::new(std::ptr::null_mut());
static SELECTION: OnceLock<Mutex<SchemeSelection>> = OnceLock::new();

impl ColorSchemePreset {
    fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        source: PresetSource,
        dark: &'static ColorScheme,
        light: Option<&'static ColorScheme>,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            source,
            dark,
            light,
        }
    }

    fn user(
        name: String,
        label: String,
        dark: &'static ColorScheme,
        light: Option<&'static ColorScheme>,
    ) -> Self {
        Self::new(name, label, PresetSource::User, dark, light)
    }
}

fn selection_store() -> &'static Mutex<SchemeSelection> {
    SELECTION.get_or_init(|| {
        Mutex::new(SchemeSelection {
            scheme: "default".to_string(),
            mode: "dark".to_string(),
        })
    })
}

fn normalize_mode(mode: &str) -> String {
    match mode.to_ascii_lowercase().as_str() {
        "light" => "light".to_string(),
        _ => "dark".to_string(),
    }
}

fn preset_key(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn prettify_label(name: &str) -> String {
    let parts: Vec<String> = name
        .split(['-', '_'])
        .filter(|s| !s.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => format!(
                    "{}{}",
                    first.to_ascii_uppercase(),
                    chars.as_str().to_ascii_lowercase()
                ),
                None => String::new(),
            }
        })
        .collect();
    if parts.is_empty() {
        name.to_string()
    } else {
        parts.join(" ")
    }
}

fn color_scheme_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os("ELODIN_ASSETS_DIR") {
        roots.push(PathBuf::from(dir));
    } else if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join("assets"));
    }
    roots.push(dirs().data_dir().to_path_buf());
    roots
        .into_iter()
        .map(|root| root.join("color_schemes"))
        .filter(|path| path.exists() && path.is_dir())
        .collect()
}

fn parse_scheme_target(path: &Path, parent_name: Option<&str>) -> Option<(String, String)> {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        != Some(true)
    {
        return None;
    }
    let stem = path.file_stem()?.to_string_lossy().to_string();
    let lower_stem = stem.to_ascii_lowercase();
    if let Some(parent) = parent_name
        && (lower_stem == "dark" || lower_stem == "light")
    {
        return Some((parent.to_string(), lower_stem));
    }
    if let Some(stripped) = lower_stem.strip_suffix("_dark")
        && !stripped.is_empty()
    {
        return Some((stripped.to_string(), "dark".to_string()));
    }
    if let Some(stripped) = lower_stem.strip_suffix("_light")
        && !stripped.is_empty()
    {
        return Some((stripped.to_string(), "light".to_string()));
    }
    None
}

fn read_scheme_file(path: &Path) -> Option<ColorScheme> {
    let contents = fs::read_to_string(path).ok()?;
    match serde_json::from_str(&contents) {
        Ok(colors) => Some(colors),
        Err(err) => {
            eprintln!("Failed to parse color scheme at {}: {err}", path.display());
            None
        }
    }
}

fn load_user_presets() -> Vec<ColorSchemePreset> {
    struct Variants {
        name: String,
        label: String,
        dark: Option<&'static ColorScheme>,
        light: Option<&'static ColorScheme>,
    }

    let mut presets: Vec<Variants> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();

    for dir in color_scheme_dirs() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let Some(parent) = path.file_name().map(|n| n.to_string_lossy().to_string()) else {
                    continue;
                };
                let Ok(child_entries) = fs::read_dir(&path) else {
                    continue;
                };
                for child in child_entries.flatten() {
                    let child_path = child.path();
                    let Some((name, mode)) = parse_scheme_target(&child_path, Some(&parent)) else {
                        continue;
                    };
                    let Some(colors) = read_scheme_file(&child_path) else {
                        continue;
                    };
                    let key = preset_key(&name);
                    let idx = *index.entry(key.clone()).or_insert_with(|| {
                        presets.push(Variants {
                            name: name.clone(),
                            label: prettify_label(&name),
                            dark: None,
                            light: None,
                        });
                        presets.len() - 1
                    });
                    let variant = &mut presets[idx];
                    variant.name = name.clone();
                    variant.label = prettify_label(&name);
                    match mode.as_str() {
                        "light" => variant.light = Some(Box::leak(Box::new(colors))),
                        _ => variant.dark = Some(Box::leak(Box::new(colors))),
                    }
                }
            } else if let Some((name, mode)) = parse_scheme_target(&path, None) {
                let Some(colors) = read_scheme_file(&path) else {
                    continue;
                };
                let key = preset_key(&name);
                let idx = *index.entry(key.clone()).or_insert_with(|| {
                    presets.push(Variants {
                        name: name.clone(),
                        label: prettify_label(&name),
                        dark: None,
                        light: None,
                    });
                    presets.len() - 1
                });
                let variant = &mut presets[idx];
                variant.name = name.clone();
                variant.label = prettify_label(&name);
                match mode.as_str() {
                    "light" => variant.light = Some(Box::leak(Box::new(colors))),
                    _ => variant.dark = Some(Box::leak(Box::new(colors))),
                }
            }
        }
    }

    presets
        .into_iter()
        .filter(|variant| variant.dark.is_some())
        .map(|variant| {
            ColorSchemePreset::user(
                variant.name,
                variant.label,
                variant.dark.expect("dark variant should exist"),
                variant.light,
            )
        })
        .collect()
}

fn upsert_preset(
    presets: &mut Vec<ColorSchemePreset>,
    index: &mut HashMap<String, usize>,
    preset: ColorSchemePreset,
) {
    let key = preset_key(&preset.name);
    if let Some(idx) = index.get(&key) {
        presets[*idx] = preset;
    } else {
        index.insert(key, presets.len());
        presets.push(preset);
    }
}

fn build_preset_registry() -> PresetRegistry {
    let mut presets = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for preset in presets::builtin_presets() {
        upsert_preset(&mut presets, &mut index, preset);
    }
    for preset in load_user_presets() {
        upsert_preset(&mut presets, &mut index, preset);
    }
    PresetRegistry { presets, index }
}

fn preset_registry() -> &'static PresetRegistry {
    PRESET_REGISTRY.get_or_init(build_preset_registry)
}

fn find_preset(name: &str) -> Option<&'static ColorSchemePreset> {
    let key = preset_key(name);
    let registry = preset_registry();
    registry
        .index
        .get(&key)
        .and_then(|idx| registry.presets.get(*idx))
}

fn resolve_variant<'a>(
    preset: &'a ColorSchemePreset,
    requested_mode: &str,
) -> (&'a ColorScheme, String) {
    let mode = normalize_mode(requested_mode);
    match mode.as_str() {
        "light" if preset.light.is_some() => (preset.light.unwrap(), "light".to_string()),
        _ => (preset.dark, "dark".to_string()),
    }
}

fn apply_selection(scheme: String, mode: String, colors: &'static ColorScheme) -> SchemeSelection {
    COLOR_SCHEME.store(colors as *const _ as *mut _, atomic::Ordering::Relaxed);
    {
        let mut guard = selection_store().lock().unwrap_or_else(|e| e.into_inner());
        guard.scheme = scheme.clone();
        guard.mode = mode.clone();
    }
    persist_selection(&scheme, &mode, colors);
    SchemeSelection { scheme, mode }
}

fn set_to_preset(name: &str, mode: &str) -> SchemeSelection {
    if let Some(preset) = find_preset(name) {
        let (colors, resolved_mode) = resolve_variant(preset, mode);
        apply_selection(preset.name.clone(), resolved_mode, colors)
    } else {
        let registry = preset_registry();
        let fallback = registry
            .presets
            .first()
            .expect("at least one preset should exist");
        let (colors, resolved_mode) = resolve_variant(fallback, mode);
        apply_selection(fallback.name.clone(), resolved_mode, colors)
    }
}

fn set_custom_scheme(name: String, mode: String, colors: ColorScheme) -> SchemeSelection {
    let leaked = Box::leak(Box::new(colors));
    apply_selection(name, normalize_mode(&mode), leaked)
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum StoredColorScheme {
    Named {
        scheme: String,
        mode: Option<String>,
        colors: ColorScheme,
    },
    Legacy(ColorScheme),
}

fn load_color_scheme() -> Option<StoredColorScheme> {
    let color_scheme_path = dirs().data_dir().join("color_scheme.json");
    let json = fs::read_to_string(color_scheme_path).ok()?;
    serde_json::from_str(&json).ok()
}

fn persist_selection(scheme: &str, mode: &str, colors: &ColorScheme) {
    let payload = StoredColorScheme::Named {
        scheme: scheme.to_string(),
        mode: Some(mode.to_string()),
        colors: colors.clone(),
    };
    let color_scheme_path = dirs().data_dir().join("color_scheme.json");
    if let Ok(json) = serde_json::to_string(&payload) {
        if let Some(parent) = color_scheme_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(color_scheme_path, json);
    }
}

fn ensure_initialized() -> &'static ColorScheme {
    let _ = preset_registry();
    let ptr = COLOR_SCHEME.load(atomic::Ordering::Relaxed);
    if !ptr.is_null() {
        // SAFETY: pointer is either null or points to a ColorScheme we set.
        return unsafe { &*ptr };
    }

    match load_color_scheme() {
        Some(StoredColorScheme::Named {
            scheme,
            mode,
            colors,
        }) => {
            if find_preset(&scheme).is_some() {
                set_to_preset(&scheme, mode.as_deref().unwrap_or("dark"))
            } else {
                set_custom_scheme(scheme, mode.unwrap_or_else(|| "dark".to_string()), colors)
            }
        }
        Some(StoredColorScheme::Legacy(colors)) => {
            set_custom_scheme("custom".to_string(), "dark".to_string(), colors)
        }
        None => set_to_preset("default", "dark"),
    };
    unsafe { &*COLOR_SCHEME.load(atomic::Ordering::Relaxed) }
}

pub fn get_scheme() -> &'static ColorScheme {
    ensure_initialized()
}

pub fn current_selection() -> SchemeSelection {
    ensure_initialized();
    selection_store()
        .lock()
        .map(|g| SchemeSelection {
            scheme: g.scheme.clone(),
            mode: g.mode.clone(),
        })
        .unwrap_or_else(|e| {
            let guard = e.into_inner();
            SchemeSelection {
                scheme: guard.scheme.clone(),
                mode: guard.mode.clone(),
            }
        })
}

pub fn set_active_scheme(scheme: &str, mode: &str) -> SchemeSelection {
    set_to_preset(scheme, mode)
}

pub fn set_active_mode(mode: &str) -> SchemeSelection {
    let current = current_selection();
    set_to_preset(&current.scheme, mode)
}

pub fn scheme_supports_mode(scheme: &str, mode: &str) -> bool {
    find_preset(scheme).is_some_and(|preset| match normalize_mode(mode).as_str() {
        "light" => preset.light.is_some(),
        _ => true,
    })
}

pub fn available_presets() -> &'static [ColorSchemePreset] {
    &preset_registry().presets
}

/// Apply a scheme/mode request and return the resolved selection.
pub fn apply_scheme_and_mode(scheme: &str, mode: &str) -> SchemeSelection {
    set_to_preset(scheme, mode)
}

/// Get the active palette and selection.
pub fn current_colors() -> (&'static ColorScheme, SchemeSelection) {
    let colors = get_scheme();
    (colors, current_selection())
}

/// Backwards-compatible entry point: sets a custom scheme while keeping the current mode.
pub fn set_schema(schema: &'static ColorScheme) {
    let current = current_selection();
    apply_selection("custom".to_string(), current.mode, schema);
}
