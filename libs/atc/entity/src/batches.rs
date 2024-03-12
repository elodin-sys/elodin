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

impl From<Model> for elodin_types::api::MonteCarloBatch {
    fn from(batch: Model) -> Self {
        Self {
            run_id: batch.run_id.as_bytes().to_vec(),
            batch_number: batch.batch_number as u32,
            samples: batch.samples as u32,
            failures: batch.failures,
            finished_time: batch.finished.map(|t| t.timestamp() as u64),
            status: elodin_types::api::monte_carlo_batch::Status::from(batch.status).into(),
        }
    }
}
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

impl From<Status> for elodin_types::api::monte_carlo_batch::Status {
    fn from(val: Status) -> Self {
        use elodin_types::api::monte_carlo_batch;
        match val {
            Status::Pending => monte_carlo_batch::Status::Pending,
            Status::Running => monte_carlo_batch::Status::Running,
            Status::Done => monte_carlo_batch::Status::Done,
        }
    }
}
