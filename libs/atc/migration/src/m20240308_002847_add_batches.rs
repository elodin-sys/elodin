use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Batches::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Batches::RunId).uuid().not_null())
                    .col(ColumnDef::new(Batches::BatchNumber).integer().not_null())
                    .col(ColumnDef::new(Batches::Samples).integer().not_null())
                    .col(ColumnDef::new(Batches::Failures).binary().not_null())
                    .col(
                        ColumnDef::new(Batches::Finished)
                            .timestamp_with_time_zone()
                            .not_null(),
                    )
                    .primary_key(
                        Index::create()
                            .col(Batches::RunId)
                            .col(Batches::BatchNumber),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .from(Batches::Table, Batches::RunId)
                            .to(MonteCarloRuns::Table, MonteCarloRuns::Id),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Batches::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Batches {
    Table,
    RunId,
    BatchNumber,
    Samples,
    Failures,
    Finished,
}

#[derive(DeriveIden)]
enum MonteCarloRuns {
    Table,
    Id,
}
