use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(MonteCarloRuns::Table)
                    .add_column(
                        ColumnDef::new(MonteCarloRuns::MaxDuration)
                            .big_integer()
                            .not_null()
                            .default(30),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Replace the sample below with your own migration scripts
        manager
            .alter_table(
                Table::alter()
                    .table(MonteCarloRuns::Table)
                    .drop_column(MonteCarloRuns::MaxDuration)
                    .to_owned(),
            )
            .await?;
        Ok(())
    }
}

#[derive(DeriveIden)]
enum MonteCarloRuns {
    Table,
    MaxDuration,
}
