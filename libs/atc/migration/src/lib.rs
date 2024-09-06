pub use sea_orm_migration::prelude::*;

mod m20231121_012747_init;
mod m20240114_071737_add_profile_pic;
mod m20240115_003703_add_draft_code;
mod m20240115_010451_add_public_sandbox;
mod m20240125_172746_add_monte_carlo_run;
mod m20240212_165856_add_max_duration;
mod m20240301_223247_add_anon_sandbox;
mod m20240307_235204_add_run_start;
mod m20240308_002847_add_batches;
mod m20240311_190106_add_batch_state;
mod m20240402_101606_change_vms_sandbox_id_fk_on_delete;
mod m20240430_043601_add_usage_events;
mod m20240430_044602_add_billing_account;
mod m20240430_044605_add_license_state;
mod m20240502_161407_add_billing_account_seat_count;
mod m20240502_233542_add_onboarding_data;
mod m20240821_134416_add_monte_carlo_billed_runtime;
mod m20240829_133724_remove_email_constraint;

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
            Box::new(m20240307_235204_add_run_start::Migration),
            Box::new(m20240308_002847_add_batches::Migration),
            Box::new(m20240311_190106_add_batch_state::Migration),
            Box::new(m20240402_101606_change_vms_sandbox_id_fk_on_delete::Migration),
            Box::new(m20240430_043601_add_usage_events::Migration),
            Box::new(m20240430_044602_add_billing_account::Migration),
            Box::new(m20240430_044605_add_license_state::Migration),
            Box::new(m20240502_161407_add_billing_account_seat_count::Migration),
            Box::new(m20240502_233542_add_onboarding_data::Migration),
            Box::new(m20240821_134416_add_monte_carlo_billed_runtime::Migration),
            Box::new(m20240829_133724_remove_email_constraint::Migration),
        ]
    }
}
