//! GPU thruster exhaust for simulations that expose thrust / RCS viz components.
//!
//! - KDL `thruster` nodes on `object_3d`: scalar or 3-vector `intensity` EQL
//! - Apollo RCS: 16 cold-gas jets from `lander.rcs_thruster_viz`; geometry in this plugin

use bevy::math::{DVec3, Quat, Vec4};
use bevy::prelude::*;
use bevy::transform::TransformSystems;
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
                    .chain()
                    .after(TransformSystems::Propagate),
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
    color.add_key(0.0, Vec4::new(1.0, 0.94, 0.82, 0.42));
    color.add_key(0.1, Vec4::new(1.0, 0.68, 0.16, 0.34));
    color.add_key(0.35, Vec4::new(1.0, 0.4, 0.05, 0.22));
    color.add_key(0.7, Vec4::new(0.82, 0.24, 0.03, 0.08));
    color.add_key(1.0, Vec4::ZERO);

    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::new(0.55, 0.24, 0.24));
    size_over_life.add_key(0.15, Vec3::new(1.05, 0.46, 0.46));
    size_over_life.add_key(0.45, Vec3::new(1.45, 0.62, 0.62));
    size_over_life.add_key(0.75, Vec3::new(1.18, 0.52, 0.52));
    size_over_life.add_key(1.0, Vec3::new(0.42, 0.2, 0.2));

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
        height: module.lit(0.12),
        base_radius: module.lit(0.025),
        top_radius: module.lit(0.09),
        dimension: ShapeDimension::Volume,
    };

    let init_vel =
        SetAttributeModifier::new(Attribute::VELOCITY, module.lit(DPS_EXHAUST_BODY * 8.0));

    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(0.42));
    let size = SetAttributeModifier::new(Attribute::SIZE3, module.lit(Vec3::new(0.26, 0.08, 0.08)));
    let drag = LinearDragModifier::new(module.lit(2.0));

    // Blue-white cold gas, brighter than the earlier transparent haze but
    // still slimmer than the DPS exhaust.
    let mut gradient = Gradient::<Vec4>::new();
    gradient.add_key(0.0, Vec4::new(0.85, 0.95, 1.35, 0.9));
    gradient.add_key(0.1, Vec4::new(0.58, 0.78, 1.25, 0.8));
    gradient.add_key(0.35, Vec4::new(0.34, 0.56, 1.0, 0.58));
    gradient.add_key(0.7, Vec4::new(0.18, 0.34, 0.78, 0.28));
    gradient.add_key(1.0, Vec4::ZERO);

    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::new(0.22, 0.07, 0.07));
    size_over_life.add_key(0.15, Vec3::new(0.46, 0.16, 0.16));
    size_over_life.add_key(0.45, Vec3::new(0.64, 0.22, 0.22));
    size_over_life.add_key(0.75, Vec3::new(0.48, 0.17, 0.17));
    size_over_life.add_key(1.0, Vec3::new(0.18, 0.06, 0.06));

    EffectAsset::new(8192, SpawnerSettings::rate(360.0.into()), module)
        .with_name("rcs_jet")
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
            body_transform,
            geo_context,
        );
        Some(KdlThrusterEval { exhaust, intensity })
    }
}

fn sync_kdl_thruster_transforms(
    objects: Query<(&KdlThrusterRig, &WorldPos, &GlobalTransform), Without<KdlThrusterJet>>,
    mut jets: Query<(&KdlThrusterJet, &mut Transform, &mut GlobalTransform)>,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
    geo_context: Res<GeoContext>,
) {
    for (rig, world_pos, object_global_transform) in &objects {
        for &entity in &rig.jets {
            let Ok((jet, mut transform, mut global_transform)) = jets.get_mut(entity) else {
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
            *transform = Transform {
                translation: object_global_transform.transform_point(jet.body_offset),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_transform_queries_are_disjoint() {
        let mut app = App::new();
        app.init_resource::<EntityMap>()
            .insert_resource(GeoContext::default())
            .add_systems(Update, sync_kdl_thruster_transforms);

        app.update();
    }
}
