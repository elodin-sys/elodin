//! GPU thruster exhaust for simulations that expose `main_thrust_viz` / `rcs_torque_viz`.
//!
//! Spike: auto-attaches to the schematic `lander.world_pos` object and drives particle
//! spawn rates from live telemetry (no KDL node yet).

use bevy::math::{Quat, Vec4};
use bevy::prelude::*;
use bevy_hanabi::{
    AlphaMode, Attribute, EffectAsset, EffectSpawner, Gradient, HanabiPlugin, Module,
    ParticleEffect, SimulationSpace, SpawnerSettings,
    modifier::{
        ShapeDimension,
        attr::SetAttributeModifier,
        output::{ColorBlendMask, ColorBlendMode, ColorOverLifetimeModifier},
        position::SetPositionCircleModifier,
    },
};
use impeller2::types::ComponentId;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue as WktComponentValue, WorldPos};

use crate::WorldPosExt;
use crate::object_3d::Object3DState;
use crate::vector_arrow::component_value_tail_to_vec3;

const LANDER_EQL: &str = "lander.world_pos";
const THRUST_VIZ: &str = "lander.main_thrust_viz";
const RCS_VIZ: &str = "lander.rcs_torque_viz";
/// Matches `apollo-lunar-module.glb translate=(0,-2.5,0)` in Bevy body space.
const DPS_NOZZLE_BODY: Vec3 = Vec3::new(0.0, -2.5, 0.0);
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
struct ThrusterJet {
    kind: JetKind,
    body_offset: Vec3,
    body_exhaust: Vec3,
}

#[derive(Clone, Copy)]
enum JetKind {
    Dps,
    Rcs { axis: u8, sign: f32 },
}

const RCS_JET_COUNT: usize = 12;

struct RcsJetMount {
    body_pos: Vec3,
    body_exhaust: Vec3,
    axis: u8,
    sign: f32,
}

const RCS_JETS: [RcsJetMount; RCS_JET_COUNT] = [
    RcsJetMount {
        body_pos: Vec3::new(2.2, 0.9, 1.2),
        body_exhaust: Vec3::NEG_X,
        axis: 0,
        sign: 1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(-2.2, 0.9, 1.2),
        body_exhaust: Vec3::X,
        axis: 0,
        sign: -1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(2.2, -0.9, 1.2),
        body_exhaust: Vec3::NEG_X,
        axis: 0,
        sign: 1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(-2.2, -0.9, 1.2),
        body_exhaust: Vec3::X,
        axis: 0,
        sign: -1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.9, 2.2, 1.2),
        body_exhaust: Vec3::NEG_Z,
        axis: 1,
        sign: 1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(-0.9, 2.2, 1.2),
        body_exhaust: Vec3::Z,
        axis: 1,
        sign: -1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.9, -2.2, 1.2),
        body_exhaust: Vec3::NEG_Z,
        axis: 1,
        sign: 1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(-0.9, -2.2, 1.2),
        body_exhaust: Vec3::Z,
        axis: 1,
        sign: -1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(1.4, 0.0, 2.0),
        body_exhaust: Vec3::NEG_Y,
        axis: 2,
        sign: 1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(-1.4, 0.0, 2.0),
        body_exhaust: Vec3::Y,
        axis: 2,
        sign: -1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.0, 1.4, 2.0),
        body_exhaust: Vec3::NEG_Y,
        axis: 2,
        sign: 1.0,
    },
    RcsJetMount {
        body_pos: Vec3::new(0.0, -1.4, 2.0),
        body_exhaust: Vec3::Y,
        axis: 2,
        sign: -1.0,
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
                    sync_thruster_transforms,
                    sync_thruster_particles,
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
    let init_pos = SetPositionCircleModifier {
        center: module.lit(Vec3::ZERO),
        axis: module.lit(Vec3::Y),
        radius: module.lit(0.45),
        dimension: ShapeDimension::Surface,
    };
    let init_vel =
        SetAttributeModifier::new(Attribute::VELOCITY, module.lit(DPS_EXHAUST_BODY * 10.0));
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(0.9));
    let size = SetAttributeModifier::new(Attribute::SIZE, module.lit(0.35));

    let mut gradient = Gradient::<Vec4>::new();
    gradient.add_key(0.0, Vec4::new(1.0, 0.55, 0.15, 1.0));
    gradient.add_key(0.35, Vec4::new(1.0, 0.35, 0.05, 0.85));
    gradient.add_key(1.0, Vec4::ZERO);

    EffectAsset::new(8192, SpawnerSettings::rate(120.0.into()), module)
        .with_name("dps_exhaust")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Add)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
        .render(ColorOverLifetimeModifier {
            gradient,
            blend: ColorBlendMode::Overwrite,
            mask: ColorBlendMask::RGBA,
        })
}

fn build_rcs_jet() -> EffectAsset {
    let mut module = Module::default();
    let init_pos = SetPositionCircleModifier {
        center: module.lit(Vec3::ZERO),
        axis: module.lit(Vec3::Y),
        radius: module.lit(0.08),
        dimension: ShapeDimension::Surface,
    };
    let init_vel =
        SetAttributeModifier::new(Attribute::VELOCITY, module.lit(DPS_EXHAUST_BODY * 4.0));
    let lifetime = SetAttributeModifier::new(Attribute::LIFETIME, module.lit(0.18));
    let size = SetAttributeModifier::new(Attribute::SIZE, module.lit(0.08));

    let mut gradient = Gradient::<Vec4>::new();
    gradient.add_key(0.0, Vec4::new(0.95, 0.97, 1.0, 1.0));
    gradient.add_key(1.0, Vec4::ZERO);

    EffectAsset::new(4096, SpawnerSettings::rate(40.0.into()), module)
        .with_name("rcs_jet")
        .with_simulation_space(SimulationSpace::Local)
        .with_alpha_mode(AlphaMode::Add)
        .init(init_pos)
        .init(init_vel)
        .init(lifetime)
        .init(size)
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
                        kind: JetKind::Rcs {
                            axis: mount.axis,
                            sign: mount.sign,
                        },
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

fn sync_thruster_particles(
    rigs: Query<&LanderThrusterRig>,
    mut jets: Query<(&ThrusterJet, &mut EffectSpawner, &mut Visibility)>,
    entity_map: Res<EntityMap>,
    component_values: Query<&WktComponentValue>,
) {
    let thrust = read_thrust_level(&entity_map, &component_values);
    let torque = read_torque(&entity_map, &component_values);

    for rig in &rigs {
        for entity in core::iter::once(rig.dps).chain(rig.rcs) {
            let Ok((jet, mut spawner, mut visibility)) = jets.get_mut(entity) else {
                continue;
            };
            let intensity = match jet.kind {
                JetKind::Dps => thrust,
                JetKind::Rcs { axis, sign } => {
                    let component = match axis {
                        0 => torque.x,
                        1 => torque.y,
                        _ => torque.z,
                    };
                    (component * sign).max(0.0)
                }
            };
            apply_spawner(&mut spawner, &mut visibility, intensity, jet.kind);
        }
    }
}

fn apply_spawner(
    spawner: &mut EffectSpawner,
    visibility: &mut Visibility,
    intensity: f32,
    kind: JetKind,
) {
    if intensity <= 0.02 {
        spawner.active = false;
        *visibility = Visibility::Hidden;
        return;
    }
    *visibility = Visibility::Visible;
    spawner.active = true;
    let base = match kind {
        JetKind::Dps => 180.0,
        JetKind::Rcs { .. } => 55.0,
    };
    spawner.settings = SpawnerSettings::rate((intensity * base).into());
}

fn read_thrust_level(entity_map: &EntityMap, component_values: &Query<&WktComponentValue>) -> f32 {
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

fn read_torque(entity_map: &EntityMap, component_values: &Query<&WktComponentValue>) -> Vec3 {
    let id = ComponentId::new(RCS_VIZ);
    let Some(entity) = entity_map.get(&id) else {
        return Vec3::ZERO;
    };
    let Ok(value) = component_values.get(*entity) else {
        return Vec3::ZERO;
    };
    component_value_tail_to_vec3(value)
        .map(|v| Vec3::new(v.x as f32, v.y as f32, v.z as f32))
        .unwrap_or(Vec3::ZERO)
}
