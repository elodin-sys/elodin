//! Custom Hanabi modifiers for the built-in cinematic Earth, ported from
//! pyrotechnique (`sphere_map.rs`, `city_tile_cdf.rs`).
//!
//! Both use the **earth_v5.glb model frame** so particles parented to the
//! globe root line up with its textures: north pole = model `-Z`, longitude 0
//! (Greenwich) = model `+X`, 90°E = model `-Y`. (Pyrotechnique's originals
//! used a +Y-north convention; the WGSL here is re-derived against the GLB's
//! measured UV mapping.)

use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy_hanabi::graph::ExprError;
use bevy_hanabi::prelude::*;

/// Multiply particle color by an equirectangular map sampled at the particle's
/// spherical direction (`normalize(position)` in the globe's model frame).
///
/// Used for city lights (NASA Black Marble) so geography lives in a texture
/// while placement is a Hanabi shell. Sampling happens in the fragment shader
/// (`textureSampleLevel`); init/update compute cannot bind material images.
/// `textureSample` is avoided: UV is constant across a billboard, so
/// screen-space derivatives are zero.
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
pub struct SphereMapColorModifier {
    /// Index into the effect's texture layout (`0` = first slot).
    pub texture_slot: u32,
    /// HDR multiplier on the sampled RGB.
    pub hdr_boost: f32,
    /// Discard samples dimmer than this luma (keeps oceans from drawing).
    pub luma_kill: f32,
}

impl Modifier for SphereMapColorModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Render
    }

    fn as_render(&self) -> Option<&dyn RenderModifier> {
        Some(self)
    }

    fn as_render_mut(&mut self) -> Option<&mut dyn RenderModifier> {
        Some(self)
    }

    fn into_boxed_render(self: Box<Self>) -> Option<Box<dyn RenderModifier>> {
        Some(self)
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(*self)
    }

    fn apply(&self, _module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        Err(ExprError::InvalidModifierContext(
            context.modifier_context(),
            ModifierContext::Render,
        ))
    }
}

impl RenderModifier for SphereMapColorModifier {
    fn apply_render(
        &self,
        _module: &mut Module,
        context: &mut RenderContext,
    ) -> Result<(), ExprError> {
        context.set_needs_particle_fragment();
        let slot = self.texture_slot;
        let boost = self.hdr_boost;
        let kill = self.luma_kill;
        // Model frame: lon = atan2(-y, x), lat = asin(-z).
        context.fragment_code += &format!(
            "    {{
    let sm_n = normalize(particle.position);
    let sm_u = atan2(-sm_n.y, sm_n.x) * 0.15915494309 + 0.5;
    let sm_v = 0.5 + asin(clamp(sm_n.z, -1.0, 1.0)) * 0.31830988618;
    let sm_tex = textureSampleLevel(material_texture_{slot}, material_sampler_{slot}, vec2<f32>(sm_u, sm_v), 0.0);
    let sm_luma = dot(sm_tex.rgb, vec3<f32>(0.3, 0.6, 0.1));
    color = vec4<f32>(color.rgb * sm_tex.rgb * {boost:.4}, color.a * step({kill:.4}, sm_luma));
    }}
"
        );
        Ok(())
    }

    fn boxed_render_clone(&self) -> Box<dyn RenderModifier> {
        Box::new(*self)
    }

    fn as_modifier(&self) -> &dyn Modifier {
        self
    }
}

pub const TILES_U: u32 = 128;
pub const TILES_V: u32 = 64;
pub const TILE_COUNT: usize = (TILES_U * TILES_V) as usize;

/// Init modifier: place a particle on the Earth shell using a baked
/// luma×cos(lat) tile CDF (inverse-CDF pick a tile, jitter UV inside it, then
/// convert with the same equirect convention as [`SphereMapColorModifier`]).
///
/// Hanabi init compute cannot bind material images, so the 128×64 city-light
/// CDF is embedded in the generated WGSL.
#[derive(Debug, Clone, PartialEq, Reflect)]
pub struct CityTileCdfModifier {
    /// Shell radius in metres (`EARTH_R + height`).
    pub radius: f32,
    /// Inclusive prefix sums, length [`TILE_COUNT`], last value ≈ 1.0.
    pub cdf: Vec<f32>,
}

impl CityTileCdfModifier {
    /// Parse the little-endian f32 CDF blob embedded in the editor binary.
    pub fn from_bytes(bytes: &[u8], radius: f32) -> Self {
        assert_eq!(
            bytes.len(),
            TILE_COUNT * 4,
            "city tile CDF expected {} bytes, got {}",
            TILE_COUNT * 4,
            bytes.len()
        );
        let cdf = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self { radius, cdf }
    }

    fn extra_wgsl(&self) -> String {
        let mut out = String::with_capacity(TILE_COUNT * 12 + 2048);
        for row in 0..TILES_V {
            let start = (row * TILES_U) as usize;
            let vals = self.cdf[start..start + TILES_U as usize]
                .iter()
                .map(|value| format!("{value:.8}"))
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!(
                "const CITY_CDF_R{row}: array<f32, {TILES_U}> = array<f32, {TILES_U}>({vals});\n"
            ));
        }
        out.push_str(
            r#"
fn city_cdf_at(i: u32) -> f32 {
    let row = i / 128u;
    let col = i % 128u;
    switch row {
"#,
        );
        for row in 0..TILES_V {
            out.push_str(&format!(
                "        case {row}u: {{ return CITY_CDF_R{row}[col]; }}\n"
            ));
        }
        out.push_str(
            r#"        default: { return 1.0; }
    }
}

fn city_tile_index(u: f32) -> u32 {
    var lo = 0u;
    var hi = 8192u;
    for (var step = 0u; step < 14u; step++) {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) >> 1u;
        if (city_cdf_at(mid) < u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return min(lo, 8191u);
}
"#,
        );
        out
    }
}

impl Modifier for CityTileCdfModifier {
    fn context(&self) -> ModifierContext {
        ModifierContext::Init
    }

    fn attributes(&self) -> &[Attribute] {
        &[Attribute::POSITION]
    }

    fn boxed_clone(&self) -> BoxedModifier {
        Box::new(self.clone())
    }

    fn apply(&self, _module: &mut Module, context: &mut ShaderWriter) -> Result<(), ExprError> {
        if self.cdf.len() != TILE_COUNT {
            return Err(ExprError::GraphEvalError(format!(
                "CityTileCdfModifier cdf length {} != {TILE_COUNT}",
                self.cdf.len()
            )));
        }
        context.extra_code += &self.extra_wgsl();
        let radius = self.radius;
        // Model frame: X = lon 0, -Y = 90E, -Z = north.
        context.main_code += &format!(
            r#"    {{
    let ctc_idx = city_tile_index(frand());
    let ctc_tx = ctc_idx % 128u;
    let ctc_ty = ctc_idx / 128u;
    let ctc_u = (f32(ctc_tx) + frand()) * 0.0078125;
    let ctc_v = (f32(ctc_ty) + frand()) * 0.015625;
    let ctc_lon = (ctc_u - 0.5) * 6.28318530718;
    let ctc_lat = (0.5 - ctc_v) * 3.14159265359;
    let ctc_cl = cos(ctc_lat);
    let ctc_n = vec3<f32>(ctc_cl * cos(ctc_lon), -ctc_cl * sin(ctc_lon), -sin(ctc_lat));
    particle.position = ctc_n * {radius:.4};
    }}
"#
        );
        Ok(())
    }
}
