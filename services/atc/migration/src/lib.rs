pub use sea_orm_migration::prelude::*;

mod m20231121_012747_init;
mod m20240114_071737_add_profile_pic;
mod m20240115_003703_add_draft_code;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20231121_012747_init::Migration),
            Box::new(m20240114_071737_add_profile_pic::Migration),
            Box::new(m20240115_003703_add_draft_code::Migration),
        ]
    }
}
