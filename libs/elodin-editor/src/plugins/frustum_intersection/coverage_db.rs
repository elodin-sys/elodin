//! Persist frustum coverage ratios to impeller2/Elodin DB.
//!
//! Creates or updates `{ellipsoid_name}.frustum_coverage` components with the volume ratio.
//! Stale coverage entities (ellipsoids no longer in view) are reset to 0.0.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use impeller2::types::ComponentId;
use impeller2_bevy::{ComponentMetadataRegistry, ComponentValue, EntityMap};
use impeller2_wkt::ComponentMetadata;
use std::collections::{HashMap, HashSet};

use super::IntersectionRatios;

#[derive(SystemParam)]
pub(super) struct CoverageDbParams<'w, 's> {
    pub ratios: Res<'w, IntersectionRatios>,
    pub entity_map: ResMut<'w, EntityMap>,
    pub metadata_reg: ResMut<'w, ComponentMetadataRegistry>,
    pub schema_reg: ResMut<'w, impeller2_bevy::ComponentSchemaRegistry>,
    pub path_reg: ResMut<'w, impeller2_bevy::ComponentPathRegistry>,
    pub values: Query<'w, 's, &'static mut ComponentValue>,
    pub names: Query<'w, 's, &'static Name>,
}

/// Write a coverage ratio to an entity's ComponentValue. Creates an F32 array if missing.
fn set_coverage_value(
    entity: Entity,
    ratio: f32,
    values: &mut Query<'_, '_, &'static mut ComponentValue>,
    commands: &mut Commands,
) {
    if let Ok(mut value) = values.get_mut(entity)
        && let ComponentValue::F32(arr) = &mut *value
    {
        let buf = nox::ArrayBuf::as_mut_buf(&mut arr.buf);
        if !buf.is_empty() {
            buf[0] = ratio;
            return;
        }
    }

    let mut arr = nox::Array::<f32, nox::Dyn>::zeroed(&[1]);
    nox::ArrayBuf::as_mut_buf(&mut arr.buf)[0] = ratio;
    commands.entity(entity).insert(ComponentValue::F32(arr));
}

/// Persist coverage ratios to impeller2 components. Spawns new entities for unseen ellipsoids,
/// updates existing ones, and resets coverage to 0.0 for ellipsoids no longer in any frustum.
pub(super) fn write_coverage_to_db(mut params: CoverageDbParams<'_, '_>, mut commands: Commands) {
    let mut ratios_by_ellipsoid: HashMap<Entity, f32> = HashMap::new();
    for ratio in params.ratios.0.iter() {
        let entry = ratios_by_ellipsoid.entry(ratio.ellipsoid).or_insert(0.0);
        *entry = (*entry).max(ratio.ratio);
    }

    let mut touched_cids = HashSet::new();
    for (&ellipsoid, &ratio) in &ratios_by_ellipsoid {
        let ellipsoid_name = params
            .names
            .get(ellipsoid)
            .map(|n| n.as_str())
            .unwrap_or("ellipsoid");
        let full_name = format!("{ellipsoid_name}.frustum_coverage");
        let cid = ComponentId::new(&full_name);
        touched_cids.insert(cid);

        let entity = if let Some(&e) = params.entity_map.get(&cid) {
            e
        } else {
            let metadata = params
                .metadata_reg
                .entry(cid)
                .or_insert_with(|| ComponentMetadata {
                    component_id: cid,
                    name: full_name.clone(),
                    metadata: Default::default(),
                })
                .clone();

            params.schema_reg.0.entry(cid).or_insert_with(|| {
                use impeller2::component::Component;
                impeller2_wkt::FrustumCoverage::schema()
            });

            params
                .path_reg
                .0
                .entry(cid)
                .or_insert_with(|| impeller2_bevy::ComponentPath::from_name(&full_name));

            let e = commands
                .spawn((cid, impeller2_bevy::ComponentValueMap::default(), metadata))
                .id();
            params.entity_map.insert(cid, e);
            e
        };

        set_coverage_value(entity, ratio, &mut params.values, &mut commands);
    }

    // Ellipsoids that had coverage last frame but are not in this frame's ratios get 0.0.
    let stale_coverage_entities = params
        .entity_map
        .iter()
        .filter_map(|(cid, entity)| {
            if touched_cids.contains(cid) {
                return None;
            }
            let is_coverage = params
                .metadata_reg
                .get(cid)
                .is_some_and(|metadata| metadata.name.ends_with(".frustum_coverage"));
            if is_coverage { Some(*entity) } else { None }
        })
        .collect::<Vec<_>>();

    for entity in stale_coverage_entities {
        set_coverage_value(entity, 0.0, &mut params.values, &mut commands);
    }
}
