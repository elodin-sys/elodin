use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, Deserialize, Serialize)]
#[sea_orm(table_name = "usage_events")]
pub struct Model {
    #[sea_orm(primary_key)]
    id: Uuid,
    event_type: EventType,
    count: i32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

#[derive(EnumIter, DeriveActiveEnum, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, Copy)]
#[sea_orm(rs_type = "i32", db_type = "Integer")]
pub enum EventType {
    #[sea_orm(num_value = 0)]
    MonteCarloBatch,
}

impl ActiveModelBehavior for ActiveModel {}
