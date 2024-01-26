use sea_orm::{entity::prelude::*, FromJsonQueryResult};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq)]
#[sea_orm(table_name = "monte_carlo_run")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub user_id: Uuid,
    pub samples: i32,
    #[sea_orm(column_type = "JsonBinary")]
    pub manifest: Manifest,
    #[sea_orm(column_type = "JsonBinary", nullable)]
    pub results: Option<Results>,
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

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize, FromJsonQueryResult, Default)]
pub struct Manifest {
    pub name: String,
    pub artifact_dir_uri: String,
    pub metadata: Json,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize, FromJsonQueryResult, Default)]
pub struct Results {
    pub summary_uri: String,
    pub replay_dir_uri: String,
}
