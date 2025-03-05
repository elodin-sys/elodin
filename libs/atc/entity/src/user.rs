use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use enumflags2::{BitFlags, bitflags};
use sea_orm::{FromJsonQueryResult, entity::prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, Deserialize, Serialize)]
#[sea_orm(table_name = "users")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub auth0_id: String,
    pub permissions: Permissions,
    pub avatar: String,
    pub license_type: LicenseType,
    pub monte_carlo_active: bool,
    pub onboarding_data: Option<Json>,
    pub billing_account_id: Option<Uuid>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::sandbox::Entity")]
    Sandbox,
}

impl ActiveModelBehavior for ActiveModel {}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize, FromJsonQueryResult, Default)]
pub struct Permissions(pub BTreeMap<Uuid, Permission>);

impl Deref for Permissions {
    type Target = BTreeMap<Uuid, Permission>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Permissions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize, FromJsonQueryResult)]
pub struct Permission {
    pub entity_type: EntityType,
    pub verb: enumflags2::BitFlags<Verb>,
}

impl Permission {
    pub fn new(entity_type: EntityType, verb: enumflags2::BitFlags<Verb>) -> Self {
        Self { entity_type, verb }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum EntityType {
    Sandbox,
}

#[bitflags]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Verb {
    Write,
    Read,
    Delete,
}

impl Permissions {
    pub fn has_perm(&self, id: &Uuid, entity_type: EntityType, verb: BitFlags<Verb>) -> bool {
        let Permissions(perms) = self;
        let Some(perm) = perms.get(id) else {
            return false;
        };
        perm.entity_type == entity_type && perm.verb.contains(verb)
    }

    pub fn resources(
        &self,
        entity_type: EntityType,
        verb: BitFlags<Verb>,
    ) -> impl Iterator<Item = Uuid> + '_ {
        let Permissions(perms) = self;
        perms
            .iter()
            .filter(move |(_, p)| p.entity_type == entity_type && p.verb.contains(verb))
            .map(|(id, _)| *id)
    }
}

#[derive(EnumIter, DeriveActiveEnum, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, Copy)]
#[sea_orm(rs_type = "i32", db_type = "Integer")]
pub enum LicenseType {
    #[sea_orm(num_value = 0)]
    None = 0,
    #[sea_orm(num_value = 1)]
    NonCommercial = 1,
    #[sea_orm(num_value = 2)]
    Commercial = 2,
    #[sea_orm(num_value = 3)]
    GodTier = 3,
}

impl From<LicenseType> for elodin_types::api::LicenseType {
    fn from(val: LicenseType) -> Self {
        match val {
            LicenseType::None => elodin_types::api::LicenseType::None,
            LicenseType::NonCommercial => elodin_types::api::LicenseType::NonCommercial,
            LicenseType::Commercial => elodin_types::api::LicenseType::Commercial,
            LicenseType::GodTier => elodin_types::api::LicenseType::GodTier,
        }
    }
}

impl From<elodin_types::api::LicenseType> for LicenseType {
    fn from(val: elodin_types::api::LicenseType) -> Self {
        match val {
            elodin_types::api::LicenseType::None => LicenseType::None,
            elodin_types::api::LicenseType::NonCommercial => LicenseType::NonCommercial,
            elodin_types::api::LicenseType::Commercial => LicenseType::Commercial,
            elodin_types::api::LicenseType::GodTier => LicenseType::GodTier,
        }
    }
}
