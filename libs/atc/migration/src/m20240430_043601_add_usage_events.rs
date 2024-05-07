use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(UsageEvents::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(UsageEvents::Id)
                            .uuid()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(UsageEvents::Type).integer().not_null())
                    .col(ColumnDef::new(UsageEvents::Count).big_integer().not_null())
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(UsageEvents::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum UsageEvents {
    Table,
    Id,
    Type,
    Count,
}
