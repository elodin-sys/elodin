//! GPU thruster exhaust for simulations that expose thrust / RCS viz components.
//!
//! Spike: auto-attaches to the schematic `lander.world_pos` object.
//! - DPS: `main_thrust_viz` (future: KDL `thruster` node — see apollo-lander docs)
//! - Cold gas RCS: `rcs_thruster_viz[16]`; nozzle geometry stays in this plugin, not KDL.

use bevy::math::{Quat, Vec4};
use bevy::prelude::*;
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
use impeller2::types::ComponentId;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue as WktComponentValue, Thruster, WorldPos};

use crate::EqlContext;
use crate::WorldPosExt;
use crate::object_3d::{CompiledExpr, Object3DState, compile_eql_expr};
use crate::vector_arrow::component_value_tail_to_vec3;

const LANDER_EQL: &str = "lander.world_pos";
const THRUST_VIZ: &str = "lander.main_thrust_viz";
const RCS_THRUSTERS_VIZ: &str = "lander.rcs_thruster_viz";
/// DPS deck under the gray ascent stage (Bevy body Y-up).
const DPS_NOZZLE_BODY: Vec3 = Vec3::new(0.0, -0.55, 0.0);
/// Body +Z thrust in ENU maps to Bevy +Y; exhaust is opposite.
const DPS_EXHAUST_BODY: Vec3 = Vec3::NEG_Y;

#[derive(Resource)]
struct ThrusterEffectAssets {
    dps: Handle<EffectAsset>,
    rcs: Handle<EffectAsset>,
}

#[derive(Component)]
struct LanderThrusterRig {
    dps: Entity,
    rcs: [Entity; RCS_JET_COUNT],
}

#[derive(Component)]
struct KdlThrusterRig {
    jets: Vec<Entity>,
}

#[derive(Component)]
struct KdlThrusterJet {
    body_offset: Vec3,
    exhaust: Vec3,
    body_frame: bool,
    intensity: Option<CompiledExpr>,
    base_rate: f32,
    cutoff: f32,
}

#[derive(Component)]
struct ThrusterJet {
    kind: JetKind,
    body_offset: Vec3,
    body_exhaust: Vec3,
}

#[derive(Clone, Copy)]
enum JetKind {
    Dps,
    Rcs { index: u8 },
}

const RCS_JET_COUNT: usize = 16;

struct RcsJetMount {
    body_pos: Vec3,
    body_exhaust: Vec3,
}

/// Sixteen cold-gas jets on the ascent-stage corners (4 clusters × 4 nozzles).
/// Index order must match `RCS_THRUSTER_AXIS` / `RCS_THRUSTER_SIGN` in apollo-lander/sim.py.
const RCS_JETS: [RcsJetMount; RCS_JET_COUNT] = [
    RcsJetMount {
        body_pos: Vec3::new(2.15, 0.85, 1.45),
        body_exhaust: Vec3::NEG_X,
    },
    RcsJetMount {
        body_pos: Vec3::new(2.15, 0.85, -1.45),
        body_exhaust: Vec3::NEG_X,
    },
    RcsJetMount {
        body_pos: Vec3::new(2.15, 1.35, 0.0),
        body_exhaust: Vec3::NEG_Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(2.15, 0.35, 0.0),
        body_exhaust: Vec3::Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(-2.15, 0.85, 1.45),
        body_exhaust: Vec3::X,
    },
    RcsJetMount {
        body_pos: Vec3::new(-2.15, 0.85, -1.45),
        body_exhaust: Vec3::X,
    },
    RcsJetMount {
        body_pos: Vec3::new(-2.15, 1.35, 0.0),
        body_exhaust: Vec3::NEG_Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(-2.15, 0.35, 0.0),
        body_exhaust: Vec3::Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(1.45, 0.85, 2.15),
        body_exhaust: Vec3::NEG_Z,
    },
    RcsJetMount {
        body_pos: Vec3::new(-1.45, 0.85, 2.15),
        body_exhaust: Vec3::NEG_Z,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.0, 1.35, 2.15),
        body_exhaust: Vec3::NEG_Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.0, 0.35, 2.15),
        body_exhaust: Vec3::Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(1.45, 0.85, -2.15),
        body_exhaust: Vec3::Z,
    },
    RcsJetMount {
        body_pos: Vec3::new(-1.45, 0.85, -2.15),
        body_exhaust: Vec3::Z,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.0, 1.35, -2.15),
        body_exhaust: Vec3::NEG_Y,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.0, 0.35, -2.15),
        body_exhaust: Vec3::Y,
    },
];

pub struct ThrusterParticlesPlugin;

impl Plugin for ThrusterParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(HanabiPlugin)
            .add_systems(Startup, setup_thruster_effects)
            .add_systems(
                PostUpdate,
                (
                    ensure_lander_thrusters,
                    ensure_kdl_thrusters,
                    sync_thruster_transforms,
                    sync_kdl_thruster_transforms,
                    sync_thruster_particles,
                    sync_kdl_thruster_particles,
                )
                    .chain(),
            );
    }
}

fn setup_thruster_effects(mut commands: Commands, mut effects: ResMut<Assets<EffectAsset>>) {
    let dps = effects.add(build_dps_exhaust());
    let rcs = effects.add(build_rcs_jet());
    commands.insert_resource(ThrusterEffectAssets { dps, rcs });
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
    let init_pos = SetPositionCone3dModifier {
        height: module.lit(0.06),
        base_radius: module.lit(0.04),
        top_radius: module.lit(0.07),
        dimension: ShapeDimension::Volume,
    };
    let init_vel =
        SetAttributeModifier::new(Attribute::VELOCITY, module.lit(DPS_EXHAUST_BODY * 10.0));
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(0.32));
    let size = SetAttributeModifier::new(Attribute::SIZE, module.lit(0.14));
    let drag = LinearDragModifier::new(module.lit(2.5));

    let mut gradient = Gradient::<Vec4>::new();
    gradient.add_key(0.0, Vec4::new(0.92, 0.96, 1.0, 0.85));
    gradient.add_key(0.35, Vec4::new(0.78, 0.86, 0.98, 0.45));
    gradient.add_key(1.0, Vec4::ZERO);

    let mut size_over_life = Gradient::<Vec3>::new();
    size_over_life.add_key(0.0, Vec3::splat(0.35));
    size_over_life.add_key(0.2, Vec3::splat(0.9));
    size_over_life.add_key(1.0, Vec3::splat(0.25));

    EffectAsset::new(8192, SpawnerSettings::rate(90.0.into()), module)
        .with_name("rcs_jet")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Add)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
        .update(drag)
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

fn ensure_lander_thrusters(
    mut commands: Commands,
    landers: Query<(Entity, &Object3DState), Without<LanderThrusterRig>>,
    assets: Res<ThrusterEffectAssets>,
) {
    for (lander, state) in &landers {
        if state.data.eql != LANDER_EQL {
            continue;
        }

        let dps = commands
            .spawn((
                ThrusterJet {
                    kind: JetKind::Dps,
                    body_offset: DPS_NOZZLE_BODY,
                    body_exhaust: DPS_EXHAUST_BODY,
                },
                ParticleEffect::new(assets.dps.clone()),
                Transform::default(),
                GlobalTransform::default(),
                Visibility::Hidden,
                Name::new("dps_exhaust"),
            ))
            .id();

        let mut rcs = [Entity::PLACEHOLDER; RCS_JET_COUNT];
        for (idx, mount) in RCS_JETS.iter().enumerate() {
            let jet = commands
                .spawn((
                    ThrusterJet {
                        kind: JetKind::Rcs { index: idx as u8 },
                        body_offset: mount.body_pos,
                        body_exhaust: mount.body_exhaust,
                    },
                    ParticleEffect::new(assets.rcs.clone()),
                    Transform::default(),
                    GlobalTransform::default(),
                    Visibility::Hidden,
                    Name::new(format!("rcs_jet_{idx}")),
                ))
                .id();
            rcs[idx] = jet;
        }

        commands
            .entity(lander)
            .insert(LanderThrusterRig { dps, rcs });
        info!("attached thruster particle rig to Apollo lander");
    }
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
                    KdlThrusterJet::from_config(config, intensity),
                    ParticleEffect::new(assets.dps.clone()),
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
    fn from_config(config: &Thruster, intensity: Option<CompiledExpr>) -> Self {
        let position = Vec3::new(config.position.0, config.position.1, config.position.2);
        let exhaust = Vec3::new(config.direction.0, config.direction.1, config.direction.2)
            .normalize_or_zero();
        Self {
            body_offset: position,
            exhaust: if exhaust == Vec3::ZERO {
                DPS_EXHAUST_BODY
            } else {
                exhaust
            },
            body_frame: config.body_frame,
            intensity,
            base_rate: config.emission_rate.max(0.0),
            cutoff: config.cutoff.max(0.0),
        }
    }
}

fn sync_thruster_transforms(
    landers: Query<(&LanderThrusterRig, &WorldPos)>,
    mut jets: Query<(&ThrusterJet, &mut Transform, &mut GlobalTransform)>,
) {
    for (rig, world_pos) in &landers {
        let body_att = world_pos.bevy_att().as_quat();
        let body_origin = world_pos.bevy_pos().as_vec3();

        for entity in core::iter::once(rig.dps).chain(rig.rcs) {
            let Ok((jet, mut transform, mut global_transform)) = jets.get_mut(entity) else {
                continue;
            };
            let local_rotation =
                Quat::from_rotation_arc(DPS_EXHAUST_BODY, jet.body_exhaust.normalize_or_zero());
            *transform = Transform {
                translation: body_origin + body_att * jet.body_offset,
                rotation: body_att * local_rotation,
                scale: Vec3::ONE,
            };
            *global_transform = GlobalTransform::from(*transform);
        }
    }
}

fn sync_kdl_thruster_transforms(
    objects: Query<(&KdlThrusterRig, &WorldPos)>,
    mut jets: Query<(&KdlThrusterJet, &mut Transform, &mut GlobalTransform)>,
) {
    for (rig, world_pos) in &objects {
        let body_att = world_pos.bevy_att().as_quat();
        let body_origin = world_pos.bevy_pos().as_vec3();

        for &entity in &rig.jets {
            let Ok((jet, mut transform, mut global_transform)) = jets.get_mut(entity) else {
                continue;
            };
            let exhaust = if jet.body_frame {
                body_att * jet.exhaust
            } else {
                jet.exhaust
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

fn sync_thruster_particles(
    rigs: Query<&LanderThrusterRig>,
    mut jets: Query<(&ThrusterJet, &mut EffectSpawner, &mut Visibility)>,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
) {
    let thrust = read_thrust_level(&entity_map, &component_values);
    let rcs = read_rcs_thruster_levels(&entity_map, &component_values);

    for rig in &rigs {
        for entity in core::iter::once(rig.dps).chain(rig.rcs) {
            let Ok((jet, mut spawner, mut visibility)) = jets.get_mut(entity) else {
                continue;
            };
            let intensity = match jet.kind {
                JetKind::Dps => thrust,
                JetKind::Rcs { index } => rcs.get(index as usize).copied().unwrap_or(0.0),
            };
            apply_spawner(&mut spawner, &mut visibility, intensity, jet.kind);
        }
    }
}

fn sync_kdl_thruster_particles(
    rigs: Query<&KdlThrusterRig>,
    mut jets: Query<(&KdlThrusterJet, &mut EffectSpawner, &mut Visibility)>,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
) {
    for rig in &rigs {
        for &entity in &rig.jets {
            let Ok((jet, mut spawner, mut visibility)) = jets.get_mut(entity) else {
                continue;
            };
            let intensity = jet
                .intensity
                .as_ref()
                .and_then(|expr| expr.execute(&entity_map, &component_values).ok())
                .and_then(|value| component_value_scalar(&value))
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
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

fn apply_spawner(
    spawner: &mut EffectSpawner,
    visibility: &mut Visibility,
    intensity: f32,
    kind: JetKind,
) {
    let (cutoff, base) = match kind {
        JetKind::Dps => (0.02, 400.0),
        JetKind::Rcs { .. } => (0.006, 140.0),
    };
    if intensity <= cutoff {
        spawner.active = false;
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Visible;
    spawner.active = true;
    let rate = match kind {
        JetKind::Rcs { .. } => intensity.sqrt() * base,
        JetKind::Dps => intensity * base,
    };
    spawner.settings = SpawnerSettings::rate(rate.into());
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

fn read_thrust_level(
    entity_map: &EntityMap,
    component_values: &Query<&'static WktComponentValue>,
) -> f32 {
    let id = ComponentId::new(THRUST_VIZ);
    let Some(entity) = entity_map.get(&id) else {
        return 0.0;
    };
    let Ok(value) = component_values.get(*entity) else {
        return 0.0;
    };
    component_value_tail_to_vec3(value)
        .map(|v| v.z.max(0.0) as f32)
        .unwrap_or(0.0)
}

fn read_rcs_thruster_levels(
    entity_map: &EntityMap,
    component_values: &Query<&'static WktComponentValue>,
) -> [f32; RCS_JET_COUNT] {
    let id = ComponentId::new(RCS_THRUSTERS_VIZ);
    let Some(entity) = entity_map.get(&id) else {
        return [0.0; RCS_JET_COUNT];
    };
    let Ok(value) = component_values.get(*entity) else {
        return [0.0; RCS_JET_COUNT];
    };
    component_value_f64_array(value)
        .map(|values| {
            let mut levels = [0.0f32; RCS_JET_COUNT];
            for (idx, level) in levels.iter_mut().enumerate() {
                *level = values.get(idx).copied().unwrap_or(0.0) as f32;
            }
            levels
        })
        .unwrap_or([0.0; RCS_JET_COUNT])
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
