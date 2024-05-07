use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, Deserialize, Serialize)]
#[sea_orm(table_name = "billing_accounts")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,
    pub name: String,
    pub owner_user_id: Uuid,
    pub customer_id: String,
    pub seat_subscription_id: Option<String>,
    pub usage_subscription_id: Option<String>,
    pub monte_carlo_active: bool,
    pub seat_count: i32,
    pub seat_license_type: crate::user::LicenseType,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
