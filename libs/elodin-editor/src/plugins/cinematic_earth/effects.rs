//! Code-built Hanabi effects for the built-in cinematic Earth: star fields,
//! Milky Way band, city lights, and airglow shells.
//!
//! Ported from pyrotechnique's `effects/builders.rs` (same bevy_hanabi rev),
//! constructed directly as [`EffectAsset`]s instead of `.effect` RON so no
//! reflection registry or asset files are involved. Contracts:
//! - Once-burst spawners at full capacity. Day/night dimming is the
//!   `intensity` / `sun_dir` properties. Density is authored at build time
//!   (rebuild the asset); do not scale a live `count`.
//! - `SimulationSpace::Local`: particles ride their parent entity (globe or
//!   sky root), which keeps them stable under big_space floating-origin
//!   rebases.
//! - Sizes are live `Attribute::SIZE` properties. Stars/airglow are world
//!   metres; city lights are screen pixels (`ScreenSpaceSizeModifier`).

use bevy::asset::RenderAssetUsages;
use bevy::math::{Vec3, Vec4};
use bevy::prelude::Image;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy_hanabi::graph::expr::{ExprHandle, PropertyHandle, WriterExpr};
use bevy_hanabi::prelude::*;
use impeller2_wkt::{EarthCityLightsConfig, EarthConfig};

use super::curves::{CITY_NIGHT_FULL, CITY_NIGHT_START};
use super::modifiers::{CityTileCdfModifier, SphereMapColorModifier};
use super::{
    HEIGHT_PROPERTY, INTENSITY_PROPERTY, SIZE_PROPERTY, SUN_DIR_PROPERTY, VIEW_POS_PROPERTY,
};

const STAR_DIM_COUNT: u32 = 800_000;
const STAR_BRIGHT_COUNT: u32 = 40_000;
const MILKY_WAY_COUNT: u32 = 400_000;
const CITY_COUNT: u32 = 1_500_000;
const AIRGLOW_GREEN_COUNT: u32 = 520_000;
const AIRGLOW_RED_COUNT: u32 = 340_000;

const STAR_DIM_HDR: Vec4 = Vec4::new(14.0, 14.0, 18.0, 1.0);
const STAR_BRIGHT_HDR: Vec4 = Vec4::new(48.0, 40.0, 62.0, 1.0);
const MILKY_WAY_HDR: Vec4 = Vec4::new(11.0, 8.5, 6.0, 0.85);
const CITY_HDR: Vec4 = Vec4::new(16.0, 10.0, 3.5, 0.75);
const AIRGLOW_GREEN_HDR: Vec4 = Vec4::new(0.12, 2.8, 0.65, 0.042);
const AIRGLOW_RED_HDR: Vec4 = Vec4::new(0.32, 0.09, 0.03, 0.01);

/// Globe radius the shells wrap (earth_v5.glb equatorial radius, metres).
pub const EARTH_R: f32 = 6_378_140.0;
/// 1e11 m (~0.67 AU) so solar-system cameras stay inside the shell.
pub const STAR_RADIUS: f32 = 100_000_000_000.0;
const VACUUM_LIFETIME: f32 = 1.0e9;

/// 128×64 luma×cos(lat) tile CDF baked from the NASA Black Marble map.
const CITY_TILE_CDF: &[u8] = include_bytes!("../../assets/earth/city_tile_cdf.bin");

fn power_law_mag(writer: &ExprWriter) -> WriterExpr {
    let u = writer.rand(ScalarType::Float);
    u.clone() * u.clone() * u.clone() * u.sqrt()
}

fn init_age_lifetime(
    writer: &ExprWriter,
    lifetime: WriterExpr,
) -> (SetAttributeModifier, SetAttributeModifier) {
    let init_age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.0).expr());
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, lifetime.expr());
    (init_age, init_lifetime)
}

fn init_zero_velocity(writer: &ExprWriter) -> SetAttributeModifier {
    SetAttributeModifier::new(Attribute::VELOCITY, writer.lit(Vec3::ZERO).expr())
}

fn packed_scale(writer: &ExprWriter, scale: WriterExpr) -> ExprHandle {
    scale
        .clone()
        .vec3(scale.clone(), scale)
        .vec4_xyz_w(writer.lit(1.0))
        .pack4x8unorm()
        .expr()
}

fn star_sphere(writer: &ExprWriter) -> SetPositionSphereModifier {
    SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(STAR_RADIUS).expr(),
        dimension: ShapeDimension::Surface,
    }
}

fn earth_shell(writer: &ExprWriter, radius: f32) -> SetPositionSphereModifier {
    SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(radius).expr(),
        dimension: ShapeDimension::Surface,
    }
}

fn thicken_shell(writer: &ExprWriter, radius: f32, thickness: f32) -> SetAttributeModifier {
    let n = writer.attr(Attribute::POSITION).normalized();
    let r = writer
        .lit(radius - thickness * 0.5)
        .uniform(writer.lit(radius + thickness * 0.5));
    SetAttributeModifier::new(Attribute::POSITION, (n * r).expr())
}

/// Night-side × limb-band visibility for airglow shells. Frame-agnostic dot
/// products; `sun_dir` / `view_pos` properties arrive in the emitter's local
/// (globe model) frame.
fn night_and_limb(
    writer: &ExprWriter,
    sun_dir: PropertyHandle,
    view_pos: PropertyHandle,
    limb_mu: f32,
    disc_sharp: f32,
    space_sharp: f32,
) -> WriterExpr {
    let n = writer.attr(Attribute::POSITION).normalized();
    let sun = writer.prop(sun_dir).normalized();
    let night = (writer.lit(0.08) - n.clone().dot(sun)).saturate();
    // Peak at the geometric limb. Kill the Earth disc hard; fade softly into space.
    let mu = n.dot(writer.prop(view_pos).normalized());
    let d = mu - writer.lit(limb_mu);
    let toward_disc = d.clone().max(writer.lit(0.0));
    let toward_space = (writer.lit(0.0) - d).max(writer.lit(0.0));
    let limb = (writer.lit(1.0)
        - toward_disc * writer.lit(disc_sharp)
        - toward_space * writer.lit(space_sharp))
    .saturate();
    night * limb
}

/// World size that subtends `pixels` at [`STAR_RADIUS`] (50° / 900 px).
fn star_world_size(pixels: f32) -> f32 {
    STAR_RADIUS * 0.000969 * pixels
}

fn scaled_count(base: u32, density: f32) -> u32 {
    let density = density.clamp(EarthConfig::DENSITY_MIN, EarthConfig::DENSITY_MAX);
    ((base as f32) * density).round().max(1.0) as u32
}

fn scale_hdr(hdr: Vec4, brightness: f32) -> Vec4 {
    Vec4::new(
        hdr.x * brightness,
        hdr.y * brightness,
        hdr.z * brightness,
        hdr.w,
    )
}

fn star_field(
    name: &str,
    capacity: u32,
    pixel_size: f32,
    hdr: Vec4,
    color_vary: bool,
) -> EffectAsset {
    let writer = ExprWriter::new();
    let intensity = writer.add_property(INTENSITY_PROPERTY, 1.0f32.into());
    let size_scale = writer.add_property(SIZE_PROPERTY, 1.0f32.into());

    let init_pos = star_sphere(&writer);
    let init_vel = init_zero_velocity(&writer);
    let mag = power_law_mag(&writer);
    let init_mag = SetAttributeModifier::new(Attribute::F32_0, mag.expr());
    let tint = writer.rand(ScalarType::Float);
    let init_tint = SetAttributeModifier::new(Attribute::F32_1, tint.expr());
    let (init_age, init_lifetime) = init_age_lifetime(&writer, writer.lit(VACUUM_LIFETIME));

    let scale = writer.attr(Attribute::F32_0) * writer.prop(intensity);
    let update_color = if color_vary {
        let t = writer.attr(Attribute::F32_1);
        let r =
            (writer.lit(0.72) + writer.lit(0.4) * (writer.lit(1.0) - t.clone())) * scale.clone();
        let g = writer.lit(0.86) * scale.clone();
        let b = (writer.lit(0.7) + writer.lit(0.45) * t) * scale;
        SetAttributeModifier::new(
            Attribute::COLOR,
            r.vec3(g, b)
                .vec4_xyz_w(writer.lit(1.0))
                .pack4x8unorm()
                .expr(),
        )
    } else {
        SetAttributeModifier::new(Attribute::COLOR, packed_scale(&writer, scale))
    };

    let world_size = star_world_size(pixel_size);
    let size = writer.lit(world_size) * writer.prop(size_scale);
    let update_size = SetAttributeModifier::new(Attribute::SIZE, size.expr());

    let mut color = Gradient::new();
    color.add_key(0.0, hdr);
    color.add_key(1.0, hdr);

    let mask_slot = writer.lit(0u32).expr();
    let mut module = writer.finish();
    module.add_texture_slot("mask");

    EffectAsset::new(
        capacity,
        SpawnerSettings::once((capacity as f32).into()),
        module,
    )
    .with_name(name)
    .with_simulation_space(SimulationSpace::Local)
    .with_simulation_condition(SimulationCondition::Always)
    .with_alpha_mode(bevy_hanabi::AlphaMode::Add)
    .init(init_pos)
    .init(init_vel)
    .init(init_mag)
    .init(init_tint)
    .init(init_age)
    .init(init_lifetime)
    .update(update_color)
    .update(update_size)
    .render(OrientModifier::new(OrientMode::FaceCameraPosition))
    .render(ParticleTextureModifier {
        texture_slot: mask_slot,
        sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
    })
    .render(ColorOverLifetimeModifier {
        gradient: color,
        blend: ColorBlendMode::Modulate,
        mask: ColorBlendMask::RGBA,
    })
}

pub fn stars_dim(earth: &EarthConfig) -> EffectAsset {
    let earth = earth.clamp();
    star_field(
        "stars_dim",
        scaled_count(STAR_DIM_COUNT, earth.stars.density),
        0.55,
        scale_hdr(STAR_DIM_HDR, earth.stars.brightness),
        false,
    )
}

pub fn stars_bright(earth: &EarthConfig) -> EffectAsset {
    let earth = earth.clamp();
    star_field(
        "stars_bright",
        scaled_count(STAR_BRIGHT_COUNT, earth.stars.density),
        1.5,
        scale_hdr(STAR_BRIGHT_HDR, earth.stars.brightness),
        true,
    )
}

pub fn milky_way(earth: &EarthConfig) -> EffectAsset {
    let earth = earth.clamp();
    let capacity = scaled_count(MILKY_WAY_COUNT, earth.stars.density);
    let hdr = scale_hdr(MILKY_WAY_HDR, earth.stars.brightness);
    let writer = ExprWriter::new();
    let intensity = writer.add_property(INTENSITY_PROPERTY, 1.0f32.into());
    let size_scale = writer.add_property(SIZE_PROPERTY, 1.0f32.into());

    let init_pos = star_sphere(&writer);
    let init_vel = init_zero_velocity(&writer);
    let mag = power_law_mag(&writer);
    let init_mag = SetAttributeModifier::new(Attribute::F32_0, mag.expr());

    let n = writer.attr(Attribute::POSITION).normalized();
    // Sky-local +Y is galactic north; the sky root maps it onto the IAU pole in ECEF.
    let pole = writer.lit(Vec3::Y);
    let lat = n.dot(pole).abs();
    // Gaussian in galactic latitude so density feathers instead of a hard strip.
    let keep = (lat.clone() * lat * writer.lit(-10.3)).exp();
    let (init_age, init_lifetime) = init_age_lifetime(
        &writer,
        writer.lit(VACUUM_LIFETIME) * keep.step(writer.rand(ScalarType::Float)),
    );

    let scale = writer.attr(Attribute::F32_0) * writer.prop(intensity);
    let update_color = SetAttributeModifier::new(Attribute::COLOR, packed_scale(&writer, scale));
    let mw_size = star_world_size(0.95);
    let size = writer.lit(mw_size) * writer.prop(size_scale);
    let update_size = SetAttributeModifier::new(Attribute::SIZE, size.expr());

    let mut color = Gradient::new();
    color.add_key(0.0, hdr);
    color.add_key(1.0, hdr);

    let mask_slot = writer.lit(0u32).expr();
    let mut module = writer.finish();
    module.add_texture_slot("mask");

    EffectAsset::new(
        capacity,
        SpawnerSettings::once((capacity as f32).into()),
        module,
    )
    .with_name("milky_way")
    .with_simulation_space(SimulationSpace::Local)
    .with_simulation_condition(SimulationCondition::Always)
    .with_alpha_mode(bevy_hanabi::AlphaMode::Add)
    .init(init_pos)
    .init(init_vel)
    .init(init_mag)
    .init(init_age)
    .init(init_lifetime)
    .update(update_color)
    .update(update_size)
    .render(OrientModifier::new(OrientMode::FaceCameraPosition))
    .render(ParticleTextureModifier {
        texture_slot: mask_slot,
        sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
    })
    .render(ColorOverLifetimeModifier {
        gradient: color,
        blend: ColorBlendMode::Modulate,
        mask: ColorBlendMask::RGBA,
    })
}

pub fn city_lights(earth: &EarthConfig) -> EffectAsset {
    let earth = earth.clamp();
    let capacity = scaled_count(CITY_COUNT, earth.city_lights.density);
    let hdr = scale_hdr(CITY_HDR, earth.city_lights.brightness);
    let height = earth.city_lights.height;
    let writer = ExprWriter::new();
    let sun_dir = writer.add_property(SUN_DIR_PROPERTY, Vec3::Y.into());
    let _view_pos = writer.add_property(VIEW_POS_PROPERTY, Vec3::new(0.0, 6_778_140.0, 0.0).into());
    let intensity = writer.add_property(INTENSITY_PROPERTY, 0.0f32.into());
    let size_px = writer.add_property(SIZE_PROPERTY, EarthCityLightsConfig::default_size().into());
    let height_prop = writer.add_property(HEIGHT_PROPERTY, height.into());

    let init_pos = CityTileCdfModifier::from_bytes(CITY_TILE_CDF, EARTH_R + height);
    let init_vel = init_zero_velocity(&writer);
    // Geography is the Black Marble sample. Tiny mag jitter only — wide mag
    // packed into 8-bit COLOR was the limb sparkle.
    let mag = writer.lit(0.96) + writer.lit(0.04) * writer.rand(ScalarType::Float);
    let init_mag = SetAttributeModifier::new(Attribute::F32_0, mag.expr());
    let (init_age, init_lifetime) = init_age_lifetime(&writer, writer.lit(VACUUM_LIFETIME));

    let n = writer.attr(Attribute::POSITION).normalized();
    let sun = writer.prop(sun_dir).normalized();
    let night = ((writer.lit(CITY_NIGHT_START) - n.clone().dot(sun))
        / writer.lit(CITY_NIGHT_START - CITY_NIGHT_FULL))
    .saturate();
    let scale = writer.attr(Attribute::F32_0) * writer.prop(intensity) * night;
    let update_color = SetAttributeModifier::new(Attribute::COLOR, packed_scale(&writer, scale));
    let radius = writer.lit(EARTH_R) + writer.prop(height_prop);
    let update_pos = SetAttributeModifier::new(Attribute::POSITION, (n * radius).expr());
    let update_size = SetAttributeModifier::new(Attribute::SIZE, writer.prop(size_px).expr());

    let mut color = Gradient::new();
    color.add_key(0.0, hdr);
    color.add_key(1.0, hdr);

    let veil_slot = writer.lit(0u32).expr();
    let mut module = writer.finish();
    module.add_texture_slot("veil");
    module.add_texture_slot("night");

    EffectAsset::new(
        capacity,
        SpawnerSettings::once((capacity as f32).into()),
        module,
    )
    .with_name("city_lights")
    .with_simulation_space(SimulationSpace::Local)
    .with_simulation_condition(SimulationCondition::Always)
    .with_alpha_mode(bevy_hanabi::AlphaMode::Add)
    .init(init_pos)
    .init(init_vel)
    .init(init_mag)
    .init(init_age)
    .init(init_lifetime)
    .update(update_pos)
    .update(update_color)
    .update(update_size)
    // Screen-aligned so lights stay round at the limb.
    .render(OrientModifier::new(OrientMode::ParallelCameraDepthPlane))
    .render(ParticleTextureModifier {
        texture_slot: veil_slot,
        sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
    })
    .render(SphereMapColorModifier {
        texture_slot: 1,
        hdr_boost: 1.0,
        luma_kill: 0.06,
    })
    .render(ColorOverLifetimeModifier {
        gradient: color,
        blend: ColorBlendMode::Modulate,
        mask: ColorBlendMask::RGBA,
    })
    .render(ScreenSpaceSizeModifier)
}

#[allow(clippy::too_many_arguments)]
fn airglow_shell(
    name: &str,
    altitude_m: f32,
    thickness_m: f32,
    capacity: u32,
    size_m: f32,
    hdr: Vec4,
    limb_mu: f32,
    disc_sharp: f32,
    space_sharp: f32,
) -> EffectAsset {
    let writer = ExprWriter::new();
    let sun_dir = writer.add_property(SUN_DIR_PROPERTY, Vec3::Y.into());
    let view_pos = writer.add_property(VIEW_POS_PROPERTY, Vec3::new(0.0, 6_778_140.0, 0.0).into());
    let intensity = writer.add_property(INTENSITY_PROPERTY, 0.0f32.into());
    let size_scale = writer.add_property(SIZE_PROPERTY, 1.0f32.into());

    let init_pos = earth_shell(&writer, EARTH_R + altitude_m);
    let init_thick = thicken_shell(&writer, EARTH_R + altitude_m, thickness_m);
    let init_vel = init_zero_velocity(&writer);
    let mag = writer.lit(0.96) + writer.lit(0.04) * writer.rand(ScalarType::Float);
    let init_mag = SetAttributeModifier::new(Attribute::F32_0, mag.expr());
    let (init_age, init_lifetime) = init_age_lifetime(&writer, writer.lit(VACUUM_LIFETIME));

    let vis = night_and_limb(&writer, sun_dir, view_pos, limb_mu, disc_sharp, space_sharp);
    let scale = writer.attr(Attribute::F32_0) * writer.prop(intensity) * vis;
    let update_color = SetAttributeModifier::new(Attribute::COLOR, packed_scale(&writer, scale));
    let size = writer.lit(size_m) * writer.prop(size_scale);
    let update_size = SetAttributeModifier::new(Attribute::SIZE, size.expr());

    let mut color = Gradient::new();
    color.add_key(0.0, hdr);
    color.add_key(1.0, hdr);

    let veil_slot = writer.lit(0u32).expr();
    let mut module = writer.finish();
    module.add_texture_slot("veil");

    EffectAsset::new(
        capacity,
        SpawnerSettings::once((capacity as f32).into()),
        module,
    )
    .with_name(name)
    .with_simulation_space(SimulationSpace::Local)
    .with_simulation_condition(SimulationCondition::Always)
    .with_alpha_mode(bevy_hanabi::AlphaMode::Add)
    .init(init_pos)
    .init(init_thick)
    .init(init_vel)
    .init(init_mag)
    .init(init_age)
    .init(init_lifetime)
    .update(update_color)
    .update(update_size)
    .render(OrientModifier::new(OrientMode::FaceCameraPosition))
    .render(ParticleTextureModifier {
        texture_slot: veil_slot,
        sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
    })
    .render(ColorOverLifetimeModifier {
        gradient: color,
        blend: ColorBlendMode::Modulate,
        mask: ColorBlendMask::RGBA,
    })
}

pub fn airglow_green(earth: &EarthConfig) -> EffectAsset {
    let earth = earth.clamp();
    airglow_shell(
        "airglow_green",
        95_000.0,
        10_000.0,
        scaled_count(AIRGLOW_GREEN_COUNT, earth.airglow.density),
        32_000.0,
        scale_hdr(AIRGLOW_GREEN_HDR, earth.airglow.brightness),
        0.955,
        55.0,
        14.0,
    )
}

pub fn airglow_red(earth: &EarthConfig) -> EffectAsset {
    let earth = earth.clamp();
    airglow_shell(
        "airglow_red",
        150_000.0,
        12_000.0,
        scaled_count(AIRGLOW_RED_COUNT, earth.airglow.density),
        40_000.0,
        scale_hdr(AIRGLOW_RED_HDR, earth.airglow.brightness),
        0.963,
        50.0,
        10.0,
    )
}

/// Radial falloff sprite: opaque center feathering to transparent rim.
/// Sampled via `ImageSampleMapping::ModulateOpacityFromR`.
pub fn build_soft_circle_image() -> Image {
    const SIZE: u32 = 128;
    let mut data = vec![0u8; (SIZE * SIZE) as usize];
    let center = (SIZE as f32 - 1.0) * 0.5;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = (x as f32 - center) / center;
            let dy = (y as f32 - center) / center;
            let d = (dx * dx + dy * dy).sqrt().min(1.0);
            let falloff = (1.0 - d).powf(1.7);
            data[(y * SIZE + x) as usize] = (falloff * 255.0) as u8;
        }
    }
    gray_image(SIZE, data)
}

/// Wide Gaussian with live corners — no hard disc edge, no hot core after
/// overlap. Used by city lights and airglow.
pub fn build_glow_veil_image() -> Image {
    const SIZE: u32 = 256;
    let mut data = vec![0u8; (SIZE * SIZE) as usize];
    let center = (SIZE as f32 - 1.0) * 0.5;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = (x as f32 - center) / center;
            let dy = (y as f32 - center) / center;
            let r2 = dx * dx + dy * dy;
            let falloff = (-1.9 * r2).exp();
            data[(y * SIZE + x) as usize] = (falloff * 255.0) as u8;
        }
    }
    gray_image(SIZE, data)
}

fn gray_image(size: u32, data: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}
