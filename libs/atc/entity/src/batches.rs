use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Deserialize, Serialize)]
#[sea_orm(table_name = "batches")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub run_id: Uuid,
    #[sea_orm(primary_key)]
    pub batch_number: i32,
    pub samples: i32,
    pub failures: Vec<u8>,
    pub finished: Option<ChronoDateTimeUtc>,
    pub status: Status,
    pub runtime: i32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::mc_run::Entity",
        from = "Column::RunId",
        to = "super::mc_run::Column::Id"
    )]
    Run,
}

impl Related<super::mc_run::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Run.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}

#[derive(EnumIter, DeriveActiveEnum, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, Copy)]
#[sea_orm(rs_type = "i32", db_type = "Integer")]
pub enum Status {
    #[sea_orm(num_value = 0)]
    Pending,
    #[sea_orm(num_value = 1)]
    Running,
    #[sea_orm(num_value = 2)]
    Done,
}
