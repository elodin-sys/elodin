pub use sea_orm_migration::prelude::*;

mod m20231121_012747_init;
mod m20240114_071737_add_profile_pic;
mod m20240115_003703_add_draft_code;
mod m20240115_010451_add_public_sandbox;
mod m20240125_172746_add_monte_carlo_run;
mod m20240212_165856_add_max_duration;
mod m20240301_223247_add_anon_sandbox;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20231121_012747_init::Migration),
            Box::new(m20240114_071737_add_profile_pic::Migration),
            Box::new(m20240115_003703_add_draft_code::Migration),
            Box::new(m20240115_010451_add_public_sandbox::Migration),
            Box::new(m20240125_172746_add_monte_carlo_run::Migration),
            Box::new(m20240212_165856_add_max_duration::Migration),
            Box::new(m20240301_223247_add_anon_sandbox::Migration),
        ]
    }
}
