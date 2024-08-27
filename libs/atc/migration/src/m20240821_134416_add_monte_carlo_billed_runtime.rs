use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Batches::Table)
                    .add_column(
                        ColumnDef::new(Batches::Runtime)
                            .integer()
                            .not_null()
                            .default(0),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .alter_table(
                Table::alter()
                    .table(MonteCarloRuns::Table)
                    .add_column(
                        ColumnDef::new(MonteCarloRuns::Billed)
                            .boolean()
                            .default(false),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Batches::Table)
                    .drop_column(Batches::Runtime)
                    .to_owned(),
            )
            .await?;

        manager
            .alter_table(
                Table::alter()
                    .table(MonteCarloRuns::Table)
                    .drop_column(MonteCarloRuns::Billed)
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum Batches {
    Table,
    Runtime,
}

#[derive(DeriveIden)]
enum MonteCarloRuns {
    Table,
    Billed,
}
