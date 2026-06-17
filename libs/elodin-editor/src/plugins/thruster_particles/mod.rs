//! GPU thruster exhaust for simulations that expose thrust / RCS viz components.
//!
//! - KDL `thruster` nodes on `object_3d`: scalar or 3-vector `intensity` EQL
//! - Apollo RCS: 16 cold-gas jets from `lander.rcs_thruster_viz`; geometry in this plugin

use bevy::math::{DVec3, Quat, Vec4};
use bevy::prelude::*;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation};
use bevy_hanabi::{
    AlphaMode, Attribute, EffectAsset, EffectSpawner, Gradient, HanabiPlugin, Module,
    ParticleEffect, SimulationSpace, SpawnerSettings,
    modifier::{
        ShapeDimension,
        attr::SetAttributeModifier,
        force::LinearDragModifier,
        output::{
            ColorBlendMask, ColorBlendMode, ColorOverLifetimeModifier, OrientMode, OrientModifier,
            SizeOverLifetimeModifier,
        },
        position::SetPositionCone3dModifier,
        velocity::SetVelocitySphereModifier,
    },
};
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue as WktComponentValue, Thruster, WorldPos};

use crate::EqlContext;
use crate::WorldPosExt;
use crate::object_3d::{CompiledExpr, Object3DState, compile_eql_expr};
use crate::vector_arrow::component_value_tail_to_vec3;

/// Reference exhaust axis used to build each emitter's local orientation (Bevy Y-up).
const DPS_EXHAUST_BODY: Vec3 = Vec3::NEG_Y;
const MIN_THRUST_VECTOR_LENGTH_SQUARED: f32 = 1e-12;

#[derive(Resource)]
struct ThrusterEffectAssets {
    plume: Handle<EffectAsset>,
    cold_gas: Handle<EffectAsset>,
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
}

#[derive(Component)]
struct KdlThrusterJet {
    body_offset: Vec3,
    fixed_exhaust: Vec3,
    vector_intensity: bool,
    body_frame: bool,
    frame: Option<GeoFrame>,
    intensity: Option<CompiledExpr>,
    scale: f32,
    base_rate: f32,
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
            .add_systems(Startup, setup_thruster_effects)
            .add_systems(
                PostUpdate,
                (
                    ensure_kdl_thrusters,
                    sync_kdl_thruster_transforms,
                    sync_kdl_thruster_particles,
                )
                    .chain(),
            );
    }
}

fn setup_thruster_effects(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    let plume = effects.add(build_dps_exhaust());
    let cold_gas = effects.add(build_rcs_jet());
    commands.insert_resource(ThrusterEffectAssets { plume, cold_gas });
}

fn build_dps_exhaust() -> EffectAsset {
    let mut module = Module::default();
    let init_pos = SetPositionCone3dModifier {
        height: module.lit(0.22),
        base_radius: module.lit(0.12),
        top_radius: module.lit(0.32),
        dimension: ShapeDimension::Volume,
    };
    let init_vel =
        SetAttributeModifier::new(Attribute::VELOCITY, module.lit(DPS_EXHAUST_BODY * 15.0));
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(1.65));
    let size = SetAttributeModifier::new(Attribute::SIZE3, module.lit(Vec3::new(0.9, 0.38, 0.38)));
    let drag = LinearDragModifier::new(module.lit(1.25));

    let mut color = Gradient::<Vec4>::new();
    color.add_key(0.0, Vec4::new(1.0, 0.94, 0.82, 0.75));
    color.add_key(0.1, Vec4::new(1.0, 0.68, 0.16, 0.7));
    color.add_key(0.35, Vec4::new(1.0, 0.4, 0.05, 0.55));
    color.add_key(0.7, Vec4::new(0.82, 0.24, 0.03, 0.22));
    color.add_key(1.0, Vec4::ZERO);

    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::new(0.9, 0.42, 0.42));
    size_over_life.add_key(0.15, Vec3::new(1.8, 0.82, 0.82));
    size_over_life.add_key(0.45, Vec3::new(2.5, 1.15, 1.15));
    size_over_life.add_key(0.75, Vec3::new(2.1, 1.0, 1.0));
    size_over_life.add_key(1.0, Vec3::new(0.7, 0.35, 0.35));

    EffectAsset::new(32768, SpawnerSettings::rate(340.0.into()), module)
        .with_name("dps_exhaust")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Add)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
        .update(drag)
        .render(OrientModifier::new(OrientMode::AlongVelocity))
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

    // Tight nozzle mouth: keep RCS puffs compact so they read as attitude jets,
    // not large square sprites detached from the vehicle.
    let init_pos = SetPositionCone3dModifier {
        height: module.lit(0.025),
        base_radius: module.lit(0.01),
        top_radius: module.lit(0.028),
        dimension: ShapeDimension::Volume,
    };

    // Diverging exhaust from a virtual throat just behind the mouth. Lower
    // speed and drag keep the jet short instead of streaking across the frame.
    let throat = module.lit(-DPS_EXHAUST_BODY * 0.10);
    let speed = module.lit(6.5);
    let init_vel = SetVelocitySphereModifier {
        center: throat,
        speed,
    };

    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(0.26));
    let size =
        SetAttributeModifier::new(Attribute::SIZE3, module.lit(Vec3::new(0.20, 0.055, 0.055)));
    let drag = LinearDragModifier::new(module.lit(4.2));

    // Compact cold-gas puff: visible enough to debug RCS firing without
    // returning to the old opaque square sprites.
    let mut gradient = Gradient::<Vec4>::new();
    gradient.add_key(0.0, Vec4::new(0.98, 1.05, 1.16, 0.72));
    gradient.add_key(0.22, Vec4::new(0.72, 0.84, 1.0, 0.46));
    gradient.add_key(0.58, Vec4::new(0.42, 0.54, 0.70, 0.18));
    gradient.add_key(1.0, Vec4::ZERO);

    // Elongated, oriented particles read as short puffs instead of square
    // billboards. Keep the cross-section narrow throughout the lifetime.
    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::new(0.12, 0.035, 0.035));
    size_over_life.add_key(0.2, Vec3::new(0.34, 0.10, 0.10));
    size_over_life.add_key(1.0, Vec3::new(0.10, 0.025, 0.025));

    EffectAsset::new(8192, SpawnerSettings::rate(130.0.into()), module)
        .with_name("rcs_jet")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Blend)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
        .update(drag)
        .render(OrientModifier::new(OrientMode::AlongVelocity))
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

fn ensure_kdl_thrusters(
    mut commands: Commands,
    objects: Query<(Entity, &Object3DState), Without<KdlThrusterRig>>,
    assets: Res<ThrusterEffectAssets>,
    eql: Res<EqlContext>,
) {
    for (object, state) in &objects {
        if state.data.thrusters.is_empty() {
            continue;
        }

        let mut jets = Vec::with_capacity(state.data.thrusters.len());
        for (idx, config) in state.data.thrusters.iter().enumerate() {
            let intensity = eql
                .0
                .parse_str(&config.intensity)
                .map_err(crate::object_3d::CompileError::Parse)
                .and_then(compile_eql_expr)
                .inspect_err(|err| {
                    warn!(
                        "unable to compile thruster intensity '{}' on {}: {err}",
                        config.intensity, state.data.eql
                    );
                })
                .ok();
            let jet = commands
                .spawn((
                    KdlThrusterJet::from_config(config, state.data.frame, intensity),
                    ParticleEffect::new(assets.by_name(&config.effect)),
                    Transform::default(),
                    GlobalTransform::default(),
                    Visibility::Hidden,
                    Name::new(
                        config
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("thruster_{idx}")),
                    ),
                ))
                .id();
            jets.push(jet);
        }

        commands.entity(object).insert(KdlThrusterRig { jets });
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
        Self {
            body_offset: position,
            fixed_exhaust,
            vector_intensity,
            body_frame: config.body_frame,
            frame,
            intensity,
            scale: config.scale.max(0.0),
            base_rate: config.emission_rate.max(0.0),
            cutoff: config.cutoff.max(0.0),
        }
    }
}

fn body_rotation(world_pos: &WorldPos, frame: Option<GeoFrame>, geo_context: &GeoContext) -> Quat {
    if let Some(frame) = frame {
        GeoRotation(frame, world_pos.att())
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
    geo_context: &GeoContext,
) -> Vec3 {
    if body_frame {
        body_att * vector
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
) -> Option<KdlThrusterEval> {
    let value = jet
        .intensity
        .as_ref()?
        .execute(entity_map, component_values)
        .ok()?;
    let body_att = body_rotation(world_pos, jet.frame, geo_context);

    if jet.vector_intensity {
        let thrust = component_value_tail_to_vec3(&value)?;
        let thrust = jet.scale
            * vector_to_bevy(
                thrust.as_vec3(),
                jet.body_frame,
                jet.frame,
                body_att,
                geo_context,
            );
        let magnitude = thrust.length();
        let intensity = magnitude.clamp(0.0, 1.0);
        if intensity <= jet.cutoff || magnitude * magnitude <= MIN_THRUST_VECTOR_LENGTH_SQUARED {
            return Some(KdlThrusterEval {
                exhaust: Vec3::ZERO,
                intensity: 0.0,
            });
        }
        Some(KdlThrusterEval {
            exhaust: (-thrust).normalize(),
            intensity,
        })
    } else {
        let intensity = component_value_scalar(&value)?.clamp(0.0, 1.0);
        let exhaust = vector_to_bevy(
            jet.fixed_exhaust,
            jet.body_frame,
            jet.frame,
            body_att,
            geo_context,
        );
        Some(KdlThrusterEval { exhaust, intensity })
    }
}

fn sync_kdl_thruster_transforms(
    objects: Query<(&KdlThrusterRig, &WorldPos)>,
    mut jets: Query<(&KdlThrusterJet, &mut Transform, &mut GlobalTransform)>,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
    geo_context: Res<GeoContext>,
) {
    for (rig, world_pos) in &objects {
        let body_origin = world_pos.bevy_pos().as_vec3();

        for &entity in &rig.jets {
            let Ok((jet, mut transform, mut global_transform)) = jets.get_mut(entity) else {
                continue;
            };
            let body_att = body_rotation(world_pos, jet.frame, &geo_context);
            let Some(eval) =
                evaluate_kdl_thruster(jet, world_pos, &geo_context, &entity_map, &component_values)
            else {
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
                        &geo_context,
                    )
                }
            } else {
                eval.exhaust
            };
            *transform = Transform {
                translation: body_origin + body_att * jet.body_offset,
                rotation: Quat::from_rotation_arc(DPS_EXHAUST_BODY, exhaust.normalize_or_zero()),
                scale: Vec3::ONE,
            };
            *global_transform = GlobalTransform::from(*transform);
        }
    }
}

fn sync_kdl_thruster_particles(
    rig_objects: Query<(&KdlThrusterRig, &WorldPos)>,
    mut jets: Query<(&KdlThrusterJet, &mut EffectSpawner, &mut Visibility)>,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
    geo_context: Res<GeoContext>,
) {
    for (rig, world_pos) in &rig_objects {
        for &entity in &rig.jets {
            let Ok((jet, mut spawner, mut visibility)) = jets.get_mut(entity) else {
                continue;
            };
            let intensity =
                evaluate_kdl_thruster(jet, world_pos, &geo_context, &entity_map, &component_values)
                    .map(|eval| eval.intensity)
                    .unwrap_or(0.0);
            apply_kdl_spawner(
                &mut spawner,
                &mut visibility,
                intensity,
                jet.cutoff,
                jet.base_rate,
            );
        }
    }
}

fn apply_kdl_spawner(
    spawner: &mut EffectSpawner,
    visibility: &mut Visibility,
    intensity: f32,
    cutoff: f32,
    base_rate: f32,
) {
    if intensity <= cutoff {
        spawner.active = false;
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Visible;
    spawner.active = true;
    spawner.settings = SpawnerSettings::rate((intensity * base_rate).into());
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
