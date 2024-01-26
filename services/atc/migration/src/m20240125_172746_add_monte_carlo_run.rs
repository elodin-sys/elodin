use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(MonteCarloRuns::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(MonteCarloRuns::Id)
                            .uuid()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(MonteCarloRuns::UserId).uuid().not_null())
                    .col(ColumnDef::new(MonteCarloRuns::Samples).integer().not_null())
                    .col(
                        ColumnDef::new(MonteCarloRuns::Manifest)
                            .json_binary()
                            .not_null(),
                    )
                    .col(ColumnDef::new(MonteCarloRuns::Results).json_binary())
                    .foreign_key(
                        ForeignKey::create()
                            .from(MonteCarloRuns::Table, MonteCarloRuns::UserId)
                            .to(Users::Table, Users::Id),
                    )
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(MonteCarloRuns::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Users {
    Table,
    Id,
}

#[derive(DeriveIden)]
enum MonteCarloRuns {
    Table,
    Id,
    UserId,
    Samples,
    Manifest,
    Results,
}
