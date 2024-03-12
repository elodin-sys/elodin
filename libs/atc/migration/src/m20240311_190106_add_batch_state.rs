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
                        ColumnDef::new(Batches::Status)
                            .integer()
                            .not_null()
                            .default(0),
                    )
                    .modify_column(ColumnDef::new(Batches::Finished).null())
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(Batches::Table)
                    .drop_column(Batches::Status)
                    .modify_column(ColumnDef::new(Batches::Finished).not_null())
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum Batches {
    Table,
    Status,
    Finished,
}
