pub use sea_orm_migration::prelude::*;

mod m20231121_012747_init;
mod m20240114_071737_add_profile_pic;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20231121_012747_init::Migration),
            Box::new(m20240114_071737_add_profile_pic::Migration),
        ]
    }
}
