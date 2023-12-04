use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, Deserialize, Serialize)]
#[sea_orm(table_name = "sandboxes")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,

    pub user_id: Uuid,

    pub name: String,
    pub code: String,

    pub status: Status,
    pub vm_id: Option<Uuid>,

    pub last_used: ChronoDateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::user::Entity",
        from = "Column::UserId",
        to = "super::user::Column::Id"
    )]
    User,
    #[sea_orm(
        has_one = "super::vm::Entity",
        from = "Column::VmId",
        to = "super::vm::Column::Id"
    )]
    Vm,
}

impl Related<super::user::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::User.def()
    }
}

impl Related<super::vm::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Vm.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}

#[derive(EnumIter, DeriveActiveEnum, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, Copy)]
#[sea_orm(rs_type = "i32", db_type = "Integer")]
pub enum Status {
    #[sea_orm(num_value = 0)]
    Off,
    #[sea_orm(num_value = 1)]
    VmBooting,
    #[sea_orm(num_value = 2)]
    Error,
    #[sea_orm(num_value = 3)]
    Running,
}

impl From<Status> for paracosm_types::api::sandbox::Status {
    fn from(val: Status) -> Self {
        use paracosm_types::api::sandbox;
        match val {
            Status::Off => sandbox::Status::Off,
            Status::VmBooting => sandbox::Status::VmBooting,
            Status::Error => sandbox::Status::Error,
            Status::Running => sandbox::Status::Running,
        }
    }
}

impl From<Model> for paracosm_types::api::Sandbox {
    fn from(sandbox: Model) -> Self {
        paracosm_types::api::Sandbox {
            id: sandbox.id.as_bytes().to_vec(),
            name: sandbox.name,
            code: sandbox.code,
            status: paracosm_types::api::sandbox::Status::from(sandbox.status).into(),
        }
    }
}
