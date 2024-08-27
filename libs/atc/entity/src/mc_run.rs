use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Serialize, Deserialize)]
#[sea_orm(table_name = "monte_carlo_runs")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub user_id: Uuid,
    pub samples: i32,
    pub name: String,
    pub metadata: Json,
    pub status: Status,
    pub max_duration: i64,
    pub started: Option<ChronoDateTimeUtc>,
    pub billed: bool,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::user::Entity",
        from = "Column::UserId",
        to = "super::user::Column::Id"
    )]
    User,
}

impl Related<super::user::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::User.def()
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

impl From<Status> for elodin_types::api::monte_carlo_run::Status {
    fn from(val: Status) -> Self {
        use elodin_types::api::monte_carlo_run;
        match val {
            Status::Pending => monte_carlo_run::Status::Pending,
            Status::Running => monte_carlo_run::Status::Running,
            Status::Done => monte_carlo_run::Status::Done,
        }
    }
}

impl From<Model> for elodin_types::api::MonteCarloRun {
    fn from(run: Model) -> Self {
        use elodin_types::api::MonteCarloRun;
        MonteCarloRun {
            id: run.id.as_bytes().to_vec(),
            samples: run.samples as u32,
            name: run.name,
            metadata: run.metadata.to_string(),
            status: elodin_types::api::monte_carlo_run::Status::from(run.status).into(),
            max_duration: run.max_duration as u64,
            started: run.started.map(|t| t.timestamp() as u64),
            batches: vec![],
        }
    }
}
