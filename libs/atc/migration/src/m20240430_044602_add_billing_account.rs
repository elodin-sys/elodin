use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(BillingAccounts::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(BillingAccounts::Id)
                            .uuid()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(BillingAccounts::Name).string().not_null())
                    .col(
                        ColumnDef::new(BillingAccounts::CustomerId)
                            .string()
                            .not_null(),
                    )
                    .col(ColumnDef::new(BillingAccounts::SeatSubscriptionId).string())
                    .col(ColumnDef::new(BillingAccounts::UsageSubscriptionId).string())
                    .col(
                        ColumnDef::new(BillingAccounts::OwnerUserId)
                            .uuid()
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .alter_table(
                Table::alter()
                    .table(Users::Table)
                    .add_column(ColumnDef::new(Users::BillingAccountId).uuid())
                    .to_owned(),
            )
            .await?;

        let mut foreign_key = ForeignKey::create();
        foreign_key
            .from(Users::Table, Users::BillingAccountId)
            .to(BillingAccounts::Table, BillingAccounts::Id)
            .on_delete(ForeignKeyAction::SetNull);

        manager.create_foreign_key(foreign_key).await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(BillingAccounts::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum BillingAccounts {
    Table,
    Id,
    Name,
    CustomerId,
    SeatSubscriptionId,
    UsageSubscriptionId,
    OwnerUserId,
}

#[derive(DeriveIden)]
enum Users {
    Table,
    BillingAccountId,
}
