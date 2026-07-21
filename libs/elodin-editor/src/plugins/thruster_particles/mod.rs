//! GPU thruster exhaust driven by KDL `thruster` nodes on `object_3d`.
//!
//! Each node carries a scalar or 3-vector `intensity` EQL expression plus the
//! emitter geometry (position, direction, scale, rate, cutoff). Apollo's RCS and
//! DPS plumes are authored this way in the lander schematic; this plugin only
//! evaluates the nodes and drives the particle effects.
//!
//! Particle integration uses hanabi `Time<EffectSimulation>`, synced each
//! frame to the Impeller playhead (`CurrentTimestamp` / `Paused`): pause freezes
//! trails, playback speed scales spawn/integration, and backward seeks rebuild
//! thruster rigs so particles do not integrate in reverse. Live 1x viewing does
//! not require paced sim ticks for trail correctness under timeline replay.
//!
//! Effects come from two sources:
//! - built-in Rust presets (`plume`, `cold_gas`), or
//! - hanabi `.effect` RON files (`effect="db:effects/<project>/<name>.effect"`),
//!   authored/tuned externally (pyrotechnique) and served by the DB Asset
//!   Server like GLBs. File effects load asynchronously; the jet stays hidden
//!   and `ParticleEffect` + `EffectMaterial` are inserted *together* once the
//!   asset is ready, so hanabi never sees a compiled effect whose texture
//!   slots have no bound images (that mismatch asserts in
//!   `prepare_bind_groups`). Sprite textures bind by slot-name convention:
//!   `mask` -> the built-in procedural soft circle, `smoke` ->
//!   `db:textures/smoke_puff.png`, anything else ->
//!   `db:textures/soft_circle.png`.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::math::{DQuat, DVec3, Quat, Vec4};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::transform::TransformSystems;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use bevy_hanabi::{
    AlphaMode, Attribute, CpuValue, EffectAsset, EffectMaterial, EffectProperties, EffectSpawner,
    EffectSimulation, EffectSimulationTime, Gradient, HanabiPlugin, Module, ParticleEffect,
    SimulationSpace, SpawnerSettings,
    modifier::{
        ShapeDimension,
        attr::SetAttributeModifier,
        force::LinearDragModifier,
        output::{
            ColorBlendMask, ColorBlendMode, ColorOverLifetimeModifier, ImageSampleMapping,
            OrientMode, OrientModifier, ParticleTextureModifier, SizeOverLifetimeModifier,
        },
        position::SetPositionCone3dModifier,
        velocity::SetVelocitySphereModifier,
    },
};
use impeller2_bevy::{ConnectionAddr, EntityMap};
use impeller2_wkt::{ComponentValue as WktComponentValue, CurrentTimestamp, Thruster, WorldPos};

use crate::EqlContext;
use crate::WorldPosExt;
use crate::object_3d::{CompiledExpr, Object3DState, compile_eql_expr, resolve_db_asset_url};
use crate::plugins::render_layer_alloc::THRUSTER_PARTICLES_RENDER_LAYER;
use crate::ui::Paused;
use crate::vector_arrow::component_value_tail_to_vec3;

/// Reference exhaust axis used to build each emitter's local orientation (Bevy Y-up).
const DPS_EXHAUST_BODY: Vec3 = Vec3::NEG_Y;
const MIN_THRUST_VECTOR_LENGTH_SQUARED: f32 = 1e-12;

/// Property names of the anchored-trail contract, shared with pyrotechnique
/// (see its `builders::exhaust_smoke`): a `.effect` declaring these vec3
/// properties runs `SimulationSpace::Local` on a **world-fixed anchor entity**
/// and receives the live nozzle pose (in the anchor's frame) through them
/// every frame. Particles hang in world space — a persistent smoke trail —
/// while surviving big_space floating-origin rebases, which
/// `SimulationSpace::Global` cannot.
pub const SPAWN_ORIGIN_PROPERTY: &str = "spawn_origin";
pub const SPAWN_AXIS_PROPERTY: &str = "spawn_axis";

/// Optional throttle property (shared convention with pyrotechnique): effects
/// declaring `intensity` receive the live 0..1 signal as a shader uniform
/// every frame, next to the spawner-rate scaling, so throttle can drive plume
/// length/brightness instead of only particle density.
pub const INTENSITY_PROPERTY: &str = "intensity";

#[derive(Resource)]
struct ThrusterEffectAssets {
    plume: Handle<EffectAsset>,
    cold_gas: Handle<EffectAsset>,
    /// Soft radial mask so billboards render as round puffs, not hard squares.
    mask: Handle<Image>,
}

impl ThrusterEffectAssets {
    /// Selects a built-in preset by name, falling back to the plume.
    fn by_name(&self, effect: &str) -> Handle<EffectAsset> {
        match effect {
            "cold_gas" => self.cold_gas.clone(),
            _ => self.plume.clone(),
        }
    }
}

#[derive(Component)]
struct KdlThrusterRig {
    jets: Vec<Entity>,
    /// The thruster configs this rig was built from. A schematic live-reload
    /// that changes them tears the rig down so it is rebuilt from the new
    /// configs (stale emitter positions were invisible otherwise).
    configs: Vec<Thruster>,
}

/// Back-reference from a jet entity to the `object_3d` it belongs to. Jets are
/// parented under that object (`ChildOf`) so they inherit its `GridCell` /
/// floating-origin pose; this handle is what lets orphaned jets be swept when
/// their object despawns.
#[derive(Component)]
struct KdlThrusterJetOf(Entity);

/// Light child of a jet entity (KDL `thruster { light ... }`). Luminous power
/// follows the jet's live intensity; the local transform holds the authored
/// down-exhaust offset (and, for spots, the -Z -> -Y exhaust aim).
#[derive(Component)]
struct KdlThrusterLight {
    /// Peak luminous power (lumens) at intensity = 1.
    peak_lm: f32,
}

/// Jet re-homed onto a world-fixed anchor because its `.effect` declares the
/// anchored-trail properties. The jet rides the anchor with an identity
/// transform; `sync_kdl_thruster_transforms` writes the nozzle pose into the
/// effect properties instead of moving the jet.
#[derive(Component)]
struct TrailAnchoredJet {
    anchor: Entity,
}

/// World-fixed anchor entity of an anchored-trail jet (back-reference for
/// cleanup when the jet despawns).
#[derive(Component)]
struct KdlTrailAnchorOf(Entity);

#[derive(Component)]
struct KdlThrusterJet {
    body_offset: Vec3,
    fixed_exhaust: Vec3,
    vector_intensity: bool,
    body_frame: bool,
    frame: Option<GeoFrame>,
    intensity: Option<CompiledExpr>,
    scale: f32,
    /// `Some(rate)`: spawn `intensity × rate`/s (presets always; file effects
    /// when the KDL sets `emission_rate` as an override).
    /// `None`: scale the rate authored inside the `.effect` file
    /// (`authored_settings`) by `intensity`.
    base_rate: Option<f32>,
    /// `.effect` file handle awaiting load; `ParticleEffect` + `EffectMaterial`
    /// are inserted together once the asset is ready, then this is cleared.
    pending_effect: Option<Handle<EffectAsset>>,
    /// Spawner settings authored in the loaded `.effect` (file effects only).
    authored_settings: Option<SpawnerSettings>,
    /// The loaded `.effect` declares the `intensity` throttle property.
    has_intensity_property: bool,
    cutoff: f32,
}

struct KdlThrusterEval {
    exhaust: Vec3,
    intensity: f32,
}

pub struct ThrusterParticlesPlugin;

impl Plugin for ThrusterParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(HanabiPlugin)
            .init_resource::<EffectPlayheadClock>()
            .add_systems(Startup, setup_thruster_effects)
            // Drive hanabi's EffectSimulation clock from the Impeller playhead
            // before TimeSystems advances it from Virtual.
            .add_systems(
                First,
                sync_effect_simulation_clock.before(bevy::time::TimeSystems),
            )
            .add_systems(
                PostUpdate,
                (
                    refresh_kdl_thrusters,
                    sweep_trail_anchors,
                    ensure_kdl_thrusters,
                    bind_file_effect_assets,
                    sync_kdl_thruster_transforms,
                    sync_kdl_thruster_particles,
                )
                    .chain()
                    .after(TransformSystems::Propagate),
            );
    }
}

/// Tracks the last Impeller playhead sample so we can derive sim-time Δt and
/// detect backward seeks (which reset thruster instances).
#[derive(Resource, Default)]
struct EffectPlayheadClock {
    last_playhead_us: Option<i64>,
    last_wall: Option<std::time::Instant>,
}

/// Max sim-time advance applied to particles in one render frame (avoids
/// spawn bursts / capacity clamps at high playback speeds).
const MAX_EFFECT_DT_S: f64 = 0.25;

/// Sync `Time<EffectSimulation>` to the editor playhead: pause when the
/// timeline is paused or the playhead stalls; set `relative_speed` so particle
/// integration tracks sim time; on backward seek, tear down thruster rigs so
/// they rebuild empty (no reverse integration).
fn sync_effect_simulation_clock(
    mut effect_time: ResMut<Time<EffectSimulation>>,
    current: Res<CurrentTimestamp>,
    paused: Res<Paused>,
    mut clock: ResMut<EffectPlayheadClock>,
    mut commands: Commands,
    rigs: Query<(Entity, &KdlThrusterRig)>,
) {
    let now = std::time::Instant::now();
    let playhead = current.0.0;
    let (Some(last_ph), Some(last_wall)) = (clock.last_playhead_us, clock.last_wall) else {
        clock.last_playhead_us = Some(playhead);
        clock.last_wall = Some(now);
        effect_time.pause();
        return;
    };

    let wall_dt = now.saturating_duration_since(last_wall).as_secs_f64().max(1e-6);
    let sim_dt = (playhead - last_ph) as f64 * 1e-6;

    if sim_dt < -0.05 {
        for (object, rig) in &rigs {
            for &jet in &rig.jets {
                commands.entity(jet).despawn();
            }
            commands.entity(object).remove::<KdlThrusterRig>();
        }
        effect_time.pause();
        clock.last_playhead_us = Some(playhead);
        clock.last_wall = Some(now);
        return;
    }

    if paused.0 || sim_dt <= 1e-9 {
        effect_time.pause();
    } else {
        effect_time.unpause();
        let speed = (sim_dt / wall_dt).clamp(0.0, 64.0);
        let max_speed = MAX_EFFECT_DT_S / wall_dt;
        effect_time.set_relative_speed_f64(speed.min(max_speed));
    }

    clock.last_playhead_us = Some(playhead);
    clock.last_wall = Some(now);
}

fn setup_thruster_effects(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut images: ResMut<Assets<Image>>,
) {
    let plume = effects.add(build_dps_exhaust());
    let cold_gas = effects.add(build_rcs_jet());
    let mask = images.add(build_soft_particle_image());
    commands.insert_resource(ThrusterEffectAssets {
        plume,
        cold_gas,
        mask,
    });
}

/// Single-channel radial falloff (opacity 1 at center, 0 at the rim), sampled
/// via `ImageSampleMapping::ModulateOpacityFromR` so square billboard quads
/// render as soft round puffs even when zoomed in.
fn build_soft_particle_image() -> Image {
    const SIZE: u32 = 64;
    let mut data = vec![0u8; (SIZE * SIZE) as usize];
    let center = (SIZE as f32 - 1.0) * 0.5;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = (x as f32 - center) / center;
            let dy = (y as f32 - center) / center;
            let dist = (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0);
            let falloff = (1.0 - dist) * (1.0 - dist);
            data[(y * SIZE + x) as usize] = (falloff * 255.0) as u8;
        }
    }
    Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

fn build_dps_exhaust() -> EffectAsset {
    let mut module = Module::default();
    let init_pos = SetPositionCone3dModifier {
        height: module.lit(0.18),
        base_radius: module.lit(0.08),
        top_radius: module.lit(0.22),
        dimension: ShapeDimension::Volume,
    };
    let init_vel =
        SetAttributeModifier::new(Attribute::VELOCITY, module.lit(DPS_EXHAUST_BODY * 11.0));
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(1.05));
    let size = SetAttributeModifier::new(Attribute::SIZE3, module.lit(Vec3::new(0.55, 0.24, 0.24)));
    let drag = LinearDragModifier::new(module.lit(1.6));

    let mut color = Gradient::<Vec4>::new();
    color.add_key(0.0, Vec4::new(1.0, 0.96, 0.86, 0.38));
    color.add_key(0.1, Vec4::new(1.0, 0.82, 0.46, 0.3));
    color.add_key(0.35, Vec4::new(0.82, 0.58, 0.34, 0.17));
    color.add_key(0.7, Vec4::new(0.46, 0.38, 0.3, 0.06));
    color.add_key(1.0, Vec4::ZERO);

    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::new(0.55, 0.24, 0.24));
    size_over_life.add_key(0.15, Vec3::new(1.05, 0.46, 0.46));
    size_over_life.add_key(0.45, Vec3::new(1.45, 0.62, 0.62));
    size_over_life.add_key(0.75, Vec3::new(1.18, 0.52, 0.52));
    size_over_life.add_key(1.0, Vec3::new(0.42, 0.2, 0.2));

    let mask_slot = module.lit(0u32);
    module.add_texture_slot("mask");

    EffectAsset::new(16384, SpawnerSettings::rate(220.0.into()), module)
        .with_name("dps_exhaust")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Blend)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
        .update(drag)
        .render(OrientModifier::new(OrientMode::AlongVelocity))
        // Soft round mask so the quads are not visible flat squares up close.
        .render(ParticleTextureModifier {
            texture_slot: mask_slot,
            sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
        })
        .render(SizeOverLifetimeModifier {
            gradient: size_over_life,
            screen_space_size: false,
        })
        .render(ColorOverLifetimeModifier {
            gradient: color,
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
}

fn build_rcs_jet() -> EffectAsset {
    let mut module = Module::default();

    // Miniature DPS-style plume: anchored at the nozzle mouth, expanding back
    // into the emitter volume while velocity carries particles outward.
    let init_pos = SetPositionCone3dModifier {
        height: module.lit(0.08),
        base_radius: module.lit(0.02),
        top_radius: module.lit(0.04),
        dimension: ShapeDimension::Volume,
    };

    // Tighter, faster cone (center farther behind the nozzle = smaller
    // divergence) so the jet reads as a punchy collimated gas thruster that
    // shoots out, not a slow diffuse vent puff.
    let init_vel = SetVelocitySphereModifier {
        center: module.lit(Vec3::new(0.0, 0.42, 0.0)),
        speed: module.lit(11.0),
    };

    // Slightly longer reach with low drag so particles keep momentum and the
    // stream stays continuous instead of popping out in discrete puffs.
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(0.5));
    let size = SetAttributeModifier::new(Attribute::SIZE3, module.lit(Vec3::splat(0.09)));
    let drag = LinearDragModifier::new(module.lit(1.2));

    // Blue-white cold gas. Alpha-blended (not additive) and slightly toned down
    // so a dense puff stays readable instead of blooming into a dazzling glow.
    let mut gradient = Gradient::<Vec4>::new();
    gradient.add_key(0.0, Vec4::new(0.85, 0.95, 1.2, 0.6));
    gradient.add_key(0.1, Vec4::new(0.58, 0.78, 1.1, 0.5));
    gradient.add_key(0.35, Vec4::new(0.34, 0.56, 0.95, 0.34));
    gradient.add_key(0.7, Vec4::new(0.18, 0.34, 0.72, 0.16));
    gradient.add_key(1.0, Vec4::ZERO);

    // Round puffs that grow modestly downstream: a tight jet near the nozzle
    // widening slightly, kept dense by the emission rate rather than by size.
    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::splat(0.07));
    size_over_life.add_key(0.15, Vec3::splat(0.16));
    size_over_life.add_key(0.45, Vec3::splat(0.22));
    size_over_life.add_key(0.75, Vec3::splat(0.17));
    size_over_life.add_key(1.0, Vec3::splat(0.07));

    let mask_slot = module.lit(0u32);
    module.add_texture_slot("mask");

    EffectAsset::new(16384, SpawnerSettings::rate(1100.0.into()), module)
        .with_name("rcs_jet")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Blend)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
        .update(drag)
        // Camera-facing billboards keep the jet volumetric from any angle.
        .render(OrientModifier::new(OrientMode::FaceCameraPosition))
        // Soft round mask so the quads are not visible flat squares up close.
        .render(ParticleTextureModifier {
            texture_slot: mask_slot,
            sample_mapping: ImageSampleMapping::ModulateOpacityFromR,
        })
        .render(SizeOverLifetimeModifier {
            gradient: size_over_life,
            screen_space_size: false,
        })
        .render(ColorOverLifetimeModifier {
            gradient,
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
}

/// Tears down thruster rigs that no longer match their schematic data and
/// sweeps jets orphaned by their object despawning.
///
/// Both happen during schematic live-reload: a full reload despawns every
/// `object_3d` (leaking the free-standing jet entities, which would keep
/// emitting at their last world transform), and in-place edits change
/// `Object3DState.data.thrusters` under an existing rig (which would keep
/// stale emitter positions/rates until restart). After teardown,
/// `ensure_kdl_thrusters` rebuilds from the current configs.
fn refresh_kdl_thrusters(
    mut commands: Commands,
    rigs: Query<(Entity, &KdlThrusterRig, &Object3DState)>,
    jets: Query<(Entity, &KdlThrusterJetOf)>,
    objects: Query<(), With<Object3DState>>,
) {
    for (object, rig, state) in &rigs {
        if rig.configs != state.data.thrusters {
            for &jet in &rig.jets {
                commands.entity(jet).despawn();
            }
            commands.entity(object).remove::<KdlThrusterRig>();
        }
    }
    for (jet, owner) in &jets {
        if objects.get(owner.0).is_err() {
            commands.entity(jet).despawn();
        }
    }
}

/// Sweeps trail anchors whose jet has despawned (rig teardown, live reload).
/// Runs separately from `refresh_kdl_thrusters` so the jet despawn commands
/// from the previous frame have applied.
fn sweep_trail_anchors(
    mut commands: Commands,
    anchors: Query<(Entity, &KdlTrailAnchorOf)>,
    jets: Query<(), With<KdlThrusterJet>>,
) {
    for (anchor, of) in &anchors {
        if jets.get(of.0).is_err() {
            commands.entity(anchor).despawn();
        }
    }
}

fn ensure_kdl_thrusters(
    mut commands: Commands,
    objects: Query<(Entity, &Object3DState), Without<KdlThrusterRig>>,
    assets: Res<ThrusterEffectAssets>,
    asset_server: Res<AssetServer>,
    connection_addr: Option<Res<ConnectionAddr>>,
    eql: Res<EqlContext>,
) {
    let connection_addr = connection_addr.as_ref().map(|addr| addr.0);
    for (object, state) in &objects {
        if state.data.thrusters.is_empty() {
            continue;
        }

        let mut jets = Vec::with_capacity(state.data.thrusters.len());
        for (idx, config) in state.data.thrusters.iter().enumerate() {
            // One jet entity per effect layer, all sharing the emitter config
            // (position/direction/intensity). Layers exist so a single KDL
            // node can stack e.g. a camera-facing volume halo over a
            // velocity-stretched core instead of declaring duplicate emitters.
            for (layer_idx, effect) in config.effect_layers().enumerate() {
                let intensity = eql
                    .0
                    .parse_str(&config.intensity)
                    .map_err(crate::object_3d::CompileError::Parse)
                    .and_then(compile_eql_expr)
                    .inspect_err(|err| {
                        if layer_idx == 0 {
                            warn!(
                                "unable to compile thruster intensity '{}' on {}: {err}",
                                config.intensity, state.data.eql
                            );
                        }
                    })
                    .ok();
                let mut jet = KdlThrusterJet::from_config(config, state.data.frame, intensity);
                jet.base_rate = layer_base_rate(config, layer_idx);
                let effect_is_file = Thruster::effect_path_is_file(effect);
                if effect_is_file {
                    // Resolved like GLBs: `db:` keys hit the DB Asset Server;
                    // bare paths fall back to the Bevy asset root (offline
                    // --kdl dev).
                    let url = resolve_db_asset_url(effect, connection_addr);
                    jet.pending_effect = Some(asset_server.load::<EffectAsset>(url));
                }
                let base_name = config
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("thruster_{idx}"));
                let name = if layer_idx == 0 {
                    base_name
                } else {
                    format!("{base_name}_layer{layer_idx}")
                };
                let mut entity = commands.spawn((
                    jet,
                    KdlThrusterJetOf(object),
                    ChildOf(object),
                    Transform::default(),
                    GlobalTransform::default(),
                    Visibility::Hidden,
                    // Explicit layer (not inherited): particles must stay on the
                    // thruster layer even though the parent mesh is on layer 0.
                    RenderLayers::layer(THRUSTER_PARTICLES_RENDER_LAYER),
                    Name::new(name),
                ));
                if !effect_is_file {
                    // Presets are ready immediately; file effects get their
                    // ParticleEffect + EffectMaterial in bind_file_effect_assets.
                    entity.insert((
                        ParticleEffect::new(assets.by_name(effect)),
                        EffectMaterial {
                            images: vec![assets.mask.clone()],
                        },
                    ));
                }
                let jet_entity = entity.id();
                // The light belongs to the nozzle, not to each layer.
                if layer_idx == 0
                    && let Some(light) = &config.light
                {
                    spawn_thruster_light(&mut commands, jet_entity, light);
                }
                jets.push(jet_entity);
            }
        }

        commands.entity(object).insert(KdlThrusterRig {
            jets,
            configs: state.data.thrusters.clone(),
        });
    }
}

/// Spawns the Bevy light child for a thruster (KDL `light` node): a point
/// light, or a spot aimed down the exhaust when `spot_angle` is set (Bevy
/// spots shine along local -Z; the jet's exhaust is local -Y). Starts at zero
/// intensity; `sync_kdl_thruster_particles` drives it with the live signal.
fn spawn_thruster_light(
    commands: &mut Commands,
    jet: Entity,
    light: &impeller2_wkt::ThrusterLight,
) {
    let color = Color::srgb(light.color.0, light.color.1, light.color.2);
    let transform = Transform {
        translation: Vec3::new(0.0, -light.offset, 0.0),
        rotation: Quat::from_rotation_arc(Vec3::NEG_Z, Vec3::NEG_Y),
        scale: Vec3::ONE,
    };
    let mut entity = commands.spawn((
        KdlThrusterLight {
            peak_lm: light.intensity.max(0.0),
        },
        transform,
        GlobalTransform::default(),
        Visibility::default(),
        ChildOf(jet),
        Name::new("thruster_light"),
    ));
    match light.spot_angle {
        Some(angle) => {
            entity.insert(SpotLight {
                color,
                intensity: 0.0,
                range: light.range,
                shadow_maps_enabled: light.shadows,
                outer_angle: (angle.to_radians() * 0.5).clamp(0.0, std::f32::consts::FRAC_PI_2),
                inner_angle: 0.0,
                ..Default::default()
            });
        }
        None => {
            entity.insert(PointLight {
                color,
                intensity: 0.0,
                range: light.range,
                shadow_maps_enabled: light.shadows,
                ..Default::default()
            });
        }
    }
}

/// Sprite for a `.effect` texture slot, by name convention (shared with
/// pyrotechnique): `mask` uses the built-in procedural soft circle, `smoke`
/// the shared smoke sprite from the DB, anything else the DB soft circle.
fn slot_image(
    slot: &str,
    assets: &ThrusterEffectAssets,
    asset_server: &AssetServer,
    connection_addr: Option<std::net::SocketAddr>,
) -> Handle<Image> {
    match slot {
        "mask" => assets.mask.clone(),
        "smoke" => asset_server.load(resolve_db_asset_url(
            "db:textures/smoke_puff.png",
            connection_addr,
        )),
        _ => asset_server.load(resolve_db_asset_url(
            "db:textures/soft_circle.png",
            connection_addr,
        )),
    }
}

/// True when a loaded `.effect` declares the anchored-trail properties.
fn is_anchored_trail(asset: &EffectAsset) -> bool {
    asset
        .properties()
        .iter()
        .any(|p| p.name() == SPAWN_ORIGIN_PROPERTY)
}

/// "Up" in a geo frame's own coordinates, for orienting trail anchors so the
/// effect's authored +Y (buoyancy, kill planes) points away from the ground.
fn frame_up(frame: GeoFrame, position_in_frame: DVec3) -> DVec3 {
    match frame {
        GeoFrame::ENU => DVec3::Z,
        GeoFrame::NED => DVec3::NEG_Z,
        // Geocentric up: within 0.2 deg of geodetic up, invisible at trail scale.
        GeoFrame::ECEF => position_in_frame.try_normalize().unwrap_or(DVec3::Z),
    }
}

/// Spawns the world-fixed anchor for an anchored-trail jet: a regular
/// high-precision world entity (GeoPosition + grid cell) frozen at the owning
/// object's position at bind time, oriented so anchor-local +Y is up.
fn spawn_trail_anchor(
    commands: &mut Commands,
    jet_entity: Entity,
    frame: Option<GeoFrame>,
    world_pos: Option<&WorldPos>,
) -> Entity {
    let frame = frame.unwrap_or_default();
    let position = world_pos.map(|wp| wp.pos()).unwrap_or_default();
    let up = frame_up(frame, position);
    let rotation = DQuat::from_rotation_arc(DVec3::Y, up);
    commands
        .spawn((
            KdlTrailAnchorOf(jet_entity),
            Name::new("thruster_trail_anchor"),
            Transform::default(),
            GlobalTransform::default(),
            Visibility::default(),
            // The `GridCell` add hook parents this under the big_space root.
            #[cfg(feature = "big_space")]
            crate::spatial::GridCell::default(),
            GeoPosition(frame, position),
            GeoRotation::absolute(frame, rotation),
        ))
        .id()
}

/// Completes file-effect jets whose `.effect` asset has finished loading:
/// captures the authored spawner settings and inserts `ParticleEffect` +
/// `EffectMaterial` in one command so hanabi compiles the effect with its
/// texture slots already bound. Effects declaring the anchored-trail
/// properties are additionally re-homed from the vehicle onto a world-fixed
/// anchor entity.
fn bind_file_effect_assets(
    mut commands: Commands,
    mut jets: Query<(Entity, &mut KdlThrusterJet, &KdlThrusterJetOf)>,
    objects: Query<&WorldPos, With<Object3DState>>,
    effects: Res<Assets<EffectAsset>>,
    assets: Res<ThrusterEffectAssets>,
    asset_server: Res<AssetServer>,
    connection_addr: Option<Res<ConnectionAddr>>,
) {
    let connection_addr = connection_addr.as_ref().map(|addr| addr.0);
    for (entity, mut jet, owner) in &mut jets {
        let Some(handle) = jet.pending_effect.clone() else {
            continue;
        };
        let Some(asset) = effects.get(&handle) else {
            continue;
        };
        jet.authored_settings = Some(asset.spawner);
        let images: Vec<Handle<Image>> = asset
            .texture_layout()
            .layout
            .iter()
            .map(|slot| slot_image(&slot.name, &assets, &asset_server, connection_addr))
            .collect();
        jet.has_intensity_property = asset
            .properties()
            .iter()
            .any(|p| p.name() == INTENSITY_PROPERTY);
        let anchored = is_anchored_trail(asset);
        if anchored {
            let anchor =
                spawn_trail_anchor(&mut commands, entity, jet.frame, objects.get(owner.0).ok());
            commands.entity(entity).insert((
                TrailAnchoredJet { anchor },
                EffectProperties::default(),
                // The trail spans kilometers away from the anchor entity;
                // entity-AABB culling would freeze/hide it whenever the anchor
                // leaves the frustum.
                NoFrustumCulling,
                ChildOf(anchor),
                Transform::IDENTITY,
            ));
        } else if jet.has_intensity_property {
            commands.entity(entity).insert(EffectProperties::default());
        }
        let mut entity = commands.entity(entity);
        if images.is_empty() {
            entity.insert(ParticleEffect::new(handle));
        } else {
            entity.insert((ParticleEffect::new(handle), EffectMaterial { images }));
        }
        jet.pending_effect = None;
    }
}

impl KdlThrusterJet {
    fn from_config(
        config: &Thruster,
        frame: Option<GeoFrame>,
        intensity: Option<CompiledExpr>,
    ) -> Self {
        let position = Vec3::new(config.position.0, config.position.1, config.position.2);
        let vector_intensity = config.vector_intensity();
        let fixed_exhaust = config
            .direction
            .map(|direction| {
                let exhaust = Vec3::new(direction.0, direction.1, direction.2).normalize_or_zero();
                if exhaust == Vec3::ZERO {
                    DPS_EXHAUST_BODY
                } else {
                    exhaust
                }
            })
            .unwrap_or(DPS_EXHAUST_BODY);
        let base_rate = layer_base_rate(config, 0);
        Self {
            body_offset: position,
            fixed_exhaust,
            vector_intensity,
            body_frame: config.body_frame,
            frame,
            intensity,
            scale: config.scale.max(0.0),
            base_rate,
            pending_effect: None,
            authored_settings: None,
            has_intensity_property: false,
            cutoff: config.cutoff.max(0.0),
        }
    }
}

/// Spawner base rate for one effect layer of a thruster.
///
/// Presets always have a fixed base rate (KDL value or the default). File
/// effects use the rate authored in the `.effect` unless the KDL sets an
/// explicit `emission_rate` override — and that override applies to the
/// **primary layer only**: stacked layers (e.g. the volume halo) are tuned
/// against their own authored rates.
fn layer_base_rate(config: &Thruster, layer_idx: usize) -> Option<f32> {
    let effect = if layer_idx == 0 {
        config.effect.as_str()
    } else {
        config
            .extra_effects
            .get(layer_idx - 1)
            .map(String::as_str)
            .unwrap_or_default()
    };
    if Thruster::effect_path_is_file(effect) {
        if layer_idx == 0 {
            config.emission_rate.map(|rate| rate.max(0.0))
        } else {
            None
        }
    } else {
        let rate = if layer_idx == 0 {
            config
                .emission_rate
                .unwrap_or_else(Thruster::default_emission_rate)
        } else {
            Thruster::default_emission_rate()
        };
        Some(rate.max(0.0))
    }
}

fn body_rotation(world_pos: &WorldPos, frame: Option<GeoFrame>, geo_context: &GeoContext) -> Quat {
    if let Some(frame) = frame {
        GeoRotation::relative(frame, world_pos.att())
            .to_bevy(geo_context)
            .as_quat()
    } else {
        world_pos.bevy_att().as_quat()
    }
}

fn vector_to_bevy(
    vector: Vec3,
    body_frame: bool,
    frame: Option<GeoFrame>,
    body_att: Quat,
    body_transform: Option<&GlobalTransform>,
    geo_context: &GeoContext,
) -> Vec3 {
    if body_frame {
        body_transform
            .map(|transform| transform.compute_transform().rotation * vector)
            .unwrap_or(body_att * vector)
    } else if let Some(frame) = frame {
        (GeoFrame::bevy_R_(&frame, geo_context) * DVec3::from(vector)).as_vec3()
    } else {
        vector
    }
}

fn evaluate_kdl_thruster(
    jet: &KdlThrusterJet,
    world_pos: &WorldPos,
    geo_context: &GeoContext,
    entity_map: &EntityMap,
    component_values: &Query<'_, '_, &'static WktComponentValue>,
    body_transform: Option<&GlobalTransform>,
) -> Option<KdlThrusterEval> {
    let value = jet
        .intensity
        .as_ref()?
        .execute(entity_map, component_values)
        .ok()?;
    let body_att = body_rotation(world_pos, jet.frame, geo_context);

    if jet.vector_intensity {
        let thrust = component_value_tail_to_vec3(&value)?;
        // Vector thrusters carry their visual direction in telemetry, so keep
        // the original world-pos attitude path. Fixed-direction scalar jets use
        // the rendered object transform below to line up with GLB nozzle meshes.
        let thrust = jet.scale
            * vector_to_bevy(
                thrust.as_vec3(),
                jet.body_frame,
                jet.frame,
                body_att,
                None,
                geo_context,
            );
        Some(evaluate_vector_thruster(thrust, jet.cutoff))
    } else {
        let intensity = component_value_scalar(&value)?.clamp(0.0, 1.0);
        let exhaust = vector_to_bevy(
            jet.fixed_exhaust,
            jet.body_frame,
            jet.frame,
            body_att,
            body_transform,
            geo_context,
        );
        Some(KdlThrusterEval { exhaust, intensity })
    }
}

fn evaluate_vector_thruster(thrust: Vec3, cutoff: f32) -> KdlThrusterEval {
    let magnitude = thrust.length();
    let intensity = magnitude.clamp(0.0, 1.0);
    if intensity <= cutoff || magnitude * magnitude <= MIN_THRUST_VECTOR_LENGTH_SQUARED {
        return KdlThrusterEval {
            exhaust: Vec3::ZERO,
            intensity: 0.0,
        };
    }
    KdlThrusterEval {
        exhaust: (-thrust).normalize(),
        intensity,
    }
}

/// Rig-object query for the transform sync: `Without<KdlThrusterLight>` keeps
/// it provably disjoint from `LightTransformQuery`'s mutable `GlobalTransform`.
type RigObjectQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static KdlThrusterRig,
        &'static WorldPos,
        &'static GlobalTransform,
    ),
    (Without<KdlThrusterJet>, Without<KdlThrusterLight>),
>;

/// Light children get their `GlobalTransform` written manually, in lockstep
/// with the jets (which bypass transform propagation).
type LightTransformQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Transform, &'static mut GlobalTransform),
    (With<KdlThrusterLight>, Without<KdlThrusterJet>),
>;

/// Anchor transforms are read-only here; disjoint from the jets' mutable
/// `GlobalTransform` access via the marker filters.
type TrailAnchorQuery<'w, 's> = Query<
    'w,
    's,
    &'static GlobalTransform,
    (
        With<KdlTrailAnchorOf>,
        Without<KdlThrusterJet>,
        Without<KdlThrusterLight>,
    ),
>;

/// Jet mutation set for the transform sync (kept as a `type` for clippy).
type JetTransformQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static KdlThrusterJet,
        &'static mut Transform,
        &'static mut GlobalTransform,
        Option<&'static Children>,
        Option<&'static TrailAnchoredJet>,
        Option<&'static mut EffectProperties>,
    ),
>;

fn sync_kdl_thruster_transforms(
    objects: RigObjectQuery,
    mut jets: JetTransformQuery,
    anchors: TrailAnchorQuery,
    mut lights: LightTransformQuery,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
    geo_context: Res<GeoContext>,
) {
    for (rig, world_pos, object_global_transform) in &objects {
        for &entity in &rig.jets {
            let Ok((jet, mut transform, mut global_transform, children, anchored, properties)) =
                jets.get_mut(entity)
            else {
                continue;
            };
            let body_att = body_rotation(world_pos, jet.frame, &geo_context);
            let Some(eval) = evaluate_kdl_thruster(
                jet,
                world_pos,
                &geo_context,
                &entity_map,
                &component_values,
                Some(object_global_transform),
            ) else {
                continue;
            };
            let exhaust = if eval.exhaust == Vec3::ZERO {
                if jet.vector_intensity {
                    DPS_EXHAUST_BODY
                } else {
                    vector_to_bevy(
                        jet.fixed_exhaust,
                        jet.body_frame,
                        jet.frame,
                        body_att,
                        Some(object_global_transform),
                        &geo_context,
                    )
                }
            } else {
                eval.exhaust
            };
            // Jets are `ChildOf` the object_3d: write LOCAL pose so big_space /
            // GridCell propagation stays consistent with the mesh. Body-frame
            // scalar jets compose mesh-relative -Y → authored direction (for
            // direction=(0,-1,0) that is identity). World-space exhaust (vector
            // thrusters / non-body-frame) is converted into the parent's frame.
            let parent_rot = object_global_transform.to_scale_rotation_translation().1;
            let local_rotation = if jet.body_frame && !jet.vector_intensity {
                let local_dir = jet.fixed_exhaust.normalize_or_zero();
                if local_dir.length_squared() < 1e-12 {
                    Quat::IDENTITY
                } else {
                    Quat::from_rotation_arc(DPS_EXHAUST_BODY, local_dir)
                }
            } else {
                let dir = exhaust.normalize_or_zero();
                if dir.length_squared() < 1e-12 {
                    Quat::IDENTITY
                } else {
                    // World exhaust → parent-local, then align effect -Y.
                    let local_dir = parent_rot.inverse() * dir;
                    Quat::from_rotation_arc(DPS_EXHAUST_BODY, local_dir.normalize_or_zero())
                }
            };
            let nozzle_local = Transform {
                translation: jet.body_offset,
                rotation: local_rotation,
                scale: Vec3::ONE,
            };
            let nozzle_global = *object_global_transform * nozzle_local;

            if let (Some(anchored), Some(mut properties)) = (anchored, properties) {
                // Anchored-trail jet: the jet entity stays put on its anchor
                // (identity transform, normal propagation); the moving nozzle
                // pose flows through the effect properties in anchor-local
                // coordinates. Both globals live in the same render space, so
                // the relative pose is invariant under floating-origin
                // rebases; f32 is exact to ~2 mm at the trail's 20 km reach.
                let Ok(anchor_global) = anchors.get(anchored.anchor) else {
                    continue;
                };
                let relative = anchor_global.affine().inverse() * nozzle_global.affine();
                let origin = Vec3::from(relative.translation);
                let axis = (relative.matrix3 * DPS_EXHAUST_BODY).normalize_or(DPS_EXHAUST_BODY);
                properties.set(SPAWN_ORIGIN_PROPERTY, origin.into());
                properties.set(SPAWN_AXIS_PROPERTY, axis.into());
                continue;
            }

            *transform = nozzle_local;
            // Sync runs after Propagate, so write globals manually (same for
            // light children) or they lag one frame — meters at descent speeds.
            *global_transform = nozzle_global;
            if let Some(children) = children {
                for &child in children {
                    if let Ok((light_local, mut light_global)) = lights.get_mut(child) {
                        *light_global = nozzle_global * *light_local;
                    }
                }
            }
        }
    }
}

/// Jet mutation set for the spawner/visibility sync (kept as a `type` for
/// clippy).
type JetSpawnerQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static KdlThrusterJet,
        &'static mut EffectSpawner,
        &'static mut Visibility,
        Option<&'static Children>,
        Option<&'static mut EffectProperties>,
    ),
>;

fn sync_kdl_thruster_particles(
    rig_objects: Query<(&KdlThrusterRig, &WorldPos)>,
    mut jets: JetSpawnerQuery,
    mut lights: Query<(
        &KdlThrusterLight,
        Option<&mut PointLight>,
        Option<&mut SpotLight>,
    )>,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
    geo_context: Res<GeoContext>,
) {
    for (rig, world_pos) in &rig_objects {
        for &entity in &rig.jets {
            let Ok((jet, mut spawner, mut visibility, children, properties)) = jets.get_mut(entity)
            else {
                continue;
            };
            let intensity = evaluate_kdl_thruster(
                jet,
                world_pos,
                &geo_context,
                &entity_map,
                &component_values,
                None,
            )
            .map(|eval| eval.intensity)
            .unwrap_or(0.0);
            apply_kdl_spawner(&mut spawner, &mut visibility, intensity, jet);
            if jet.has_intensity_property
                && let Some(mut properties) = properties
            {
                properties.set(INTENSITY_PROPERTY, intensity.clamp(0.0, 1.0).into());
            }
            // Light luminous power tracks the same signal (0 below cutoff).
            if let Some(children) = children {
                let lm = if intensity <= jet.cutoff {
                    0.0
                } else {
                    intensity
                };
                for &child in children {
                    let Ok((light, point, spot)) = lights.get_mut(child) else {
                        continue;
                    };
                    if let Some(mut point) = point {
                        point.intensity = light.peak_lm * lm;
                    }
                    if let Some(mut spot) = spot {
                        spot.intensity = light.peak_lm * lm;
                    }
                }
            }
        }
    }
}

/// Scales a spawner count by `factor`, preserving `Uniform` ranges (the same
/// formula pyrotechnique uses, so intensity semantics match the authoring
/// tool).
fn scale_cpu_value(value: &CpuValue<f32>, factor: f32) -> CpuValue<f32> {
    match value {
        CpuValue::Single(v) => CpuValue::Single(v * factor),
        CpuValue::Uniform((lo, hi)) => CpuValue::Uniform((lo * factor, hi * factor)),
        other => *other,
    }
}

fn apply_kdl_spawner(
    spawner: &mut EffectSpawner,
    visibility: &mut Visibility,
    intensity: f32,
    jet: &KdlThrusterJet,
) {
    if intensity <= jet.cutoff {
        spawner.active = false;
        *visibility = Visibility::Hidden;
        return;
    }
    let settings = match (jet.base_rate, jet.authored_settings) {
        // Fixed rate: presets, or file effects with an emission_rate override.
        (Some(base_rate), _) => SpawnerSettings::rate((intensity * base_rate).into()),
        // File effect: authored settings from the asset, count scaled.
        (None, Some(authored)) => {
            let mut settings = authored;
            settings.set_count(scale_cpu_value(&settings.count(), intensity));
            settings
        }
        // File effect still loading; keep hidden until bound.
        (None, None) => {
            spawner.active = false;
            *visibility = Visibility::Hidden;
            return;
        }
    };
    *visibility = Visibility::Visible;
    spawner.active = true;
    spawner.settings = settings;
}

fn component_value_f64_array(value: &WktComponentValue) -> Option<Vec<f64>> {
    use nox::ArrayBuf;
    match value {
        WktComponentValue::F32(array) => {
            Some(array.buf.as_buf().iter().map(|v| f64::from(*v)).collect())
        }
        WktComponentValue::F64(array) => Some(array.buf.as_buf().to_vec()),
        _ => None,
    }
}

fn component_value_scalar(value: &WktComponentValue) -> Option<f32> {
    component_value_f64_array(value).and_then(|values| values.first().copied().map(|v| v as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_near(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 1e-6,
            "expected {expected:?}, got {actual:?}",
        );
    }

    #[test]
    fn vector_thruster_eval_reports_direction_sense_and_intensity() {
        let eval = evaluate_vector_thruster(Vec3::new(0.0, 0.0, 0.25), 0.0);

        assert_vec3_near(eval.exhaust, Vec3::NEG_Z);
        assert!((eval.intensity - 0.25).abs() < 1e-6);

        let saturated = evaluate_vector_thruster(Vec3::new(0.0, 0.0, -2.0), 0.0);

        assert_vec3_near(saturated.exhaust, Vec3::Z);
        assert!((saturated.intensity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vector_thruster_eval_hides_below_cutoff() {
        let eval = evaluate_vector_thruster(Vec3::new(0.0, 0.0, 0.004), 0.006);

        assert_vec3_near(eval.exhaust, Vec3::ZERO);
        assert_eq!(eval.intensity, 0.0);
    }

    #[test]
    fn vector_body_frame_conversion_can_use_attitude_without_render_transform() {
        let attitude = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let render_transform = GlobalTransform::from(Transform::from_rotation(
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        ));

        let via_attitude =
            vector_to_bevy(Vec3::X, true, None, attitude, None, &GeoContext::default());
        let via_render_transform = vector_to_bevy(
            Vec3::X,
            true,
            None,
            attitude,
            Some(&render_transform),
            &GeoContext::default(),
        );

        assert_vec3_near(via_attitude, Vec3::Y);
        assert!((via_render_transform - via_attitude).length() > 0.5);
    }

    #[test]
    fn sync_transform_queries_are_disjoint() {
        let mut app = App::new();
        app.init_resource::<EntityMap>()
            .insert_resource(GeoContext::default())
            .add_systems(Update, sync_kdl_thruster_transforms);

        app.update();
    }

    fn test_thruster(position: (f32, f32, f32)) -> Thruster {
        Thruster {
            name: Some("DPS".to_string()),
            body_frame: true,
            position,
            direction: Some((0.0, -1.0, 0.0)),
            intensity: "lander.main_thrust_viz[2]".to_string(),
            effect: Thruster::default_effect(),
            extra_effects: Vec::new(),
            emission_rate: None,
            cutoff: 0.0,
            scale: 1.0,
            light: None,
        }
    }

    fn test_object_state(thrusters: Vec<Thruster>) -> Object3DState {
        Object3DState {
            compiled_expr: None,
            scale_expr: None,
            scale_error: None,
            error_covariance_cholesky_expr: None,
            joint_animations: Vec::new(),
            data: impeller2_wkt::Object3D {
                eql: "lander.world_pos".to_string(),
                mesh: impeller2_wkt::Object3DMesh::glb("lander.glb"),
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                icon: None,
                thrusters,
                mesh_visibility_range: None,
                node_id: Default::default(),
            },
        }
    }

    #[test]
    fn refresh_tears_down_rig_when_configs_change() {
        let mut app = App::new();
        app.add_systems(Update, refresh_kdl_thrusters);

        let object = app
            .world_mut()
            .spawn(test_object_state(vec![test_thruster((0.0, -1.9, 0.0))]))
            .id();
        let jet = app.world_mut().spawn(KdlThrusterJetOf(object)).id();
        app.world_mut().entity_mut(object).insert(KdlThrusterRig {
            jets: vec![jet],
            configs: vec![test_thruster((0.0, -0.12, 0.0))],
        });

        app.update();

        assert!(
            app.world().get::<KdlThrusterRig>(object).is_none(),
            "changed configs must remove the rig so it rebuilds"
        );
        assert!(
            app.world().get_entity(jet).is_err(),
            "stale jets must despawn"
        );
    }

    #[test]
    fn refresh_keeps_rig_when_configs_match() {
        let mut app = App::new();
        app.add_systems(Update, refresh_kdl_thrusters);

        let configs = vec![test_thruster((0.0, -1.9, 0.0))];
        let object = app
            .world_mut()
            .spawn(test_object_state(configs.clone()))
            .id();
        let jet = app.world_mut().spawn(KdlThrusterJetOf(object)).id();
        app.world_mut().entity_mut(object).insert(KdlThrusterRig {
            jets: vec![jet],
            configs,
        });

        app.update();

        assert!(app.world().get::<KdlThrusterRig>(object).is_some());
        assert!(app.world().get_entity(jet).is_ok());
    }

    #[test]
    fn layer_base_rate_override_applies_to_primary_only() {
        let mut config = test_thruster((0.0, -1.9, 0.0));
        config.effect = "effects/apollo-lander/descent_plume.effect".to_string();
        config.extra_effects = vec!["effects/apollo-lander/descent_glow.effect".to_string()];

        // No override: both layers use their authored rates.
        assert_eq!(layer_base_rate(&config, 0), None);
        assert_eq!(layer_base_rate(&config, 1), None);

        // KDL emission_rate override pins the primary; the halo layer keeps
        // its authored rate.
        config.emission_rate = Some(1234.0);
        assert_eq!(layer_base_rate(&config, 0), Some(1234.0));
        assert_eq!(layer_base_rate(&config, 1), None);

        // Presets always have a fixed rate.
        config.effect = "plume".to_string();
        assert_eq!(layer_base_rate(&config, 0), Some(1234.0));
        config.emission_rate = None;
        assert_eq!(
            layer_base_rate(&config, 0),
            Some(Thruster::default_emission_rate())
        );
    }

    #[test]
    fn refresh_sweeps_jets_of_despawned_objects() {
        let mut app = App::new();
        app.add_systems(Update, refresh_kdl_thrusters);

        let object = app
            .world_mut()
            .spawn(test_object_state(vec![test_thruster((0.0, -1.9, 0.0))]))
            .id();
        let jet = app.world_mut().spawn(KdlThrusterJetOf(object)).id();
        app.world_mut().entity_mut(object).despawn();

        app.update();

        assert!(
            app.world().get_entity(jet).is_err(),
            "jets orphaned by schematic reload must despawn"
        );
    }
}
