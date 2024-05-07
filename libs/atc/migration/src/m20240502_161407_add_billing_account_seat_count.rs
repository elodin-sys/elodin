use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .alter_table(
                Table::alter()
                    .table(BillingAccounts::Table)
                    .add_column(
                        ColumnDef::new(BillingAccounts::MonteCarloActive)
                            .boolean()
                            .default(false),
                    )
                    .add_column(
                        ColumnDef::new(BillingAccounts::SeatCount)
                            .integer()
                            .default(0)
                            .not_null(),
                    )
                    .add_column(
                        ColumnDef::new(BillingAccounts::SeatLicenseType)
                            .integer()
                            .default(0)
                            .not_null(),
                    )
                    .to_owned(),
            )
            .await?;
        manager
            .alter_table(
                Table::alter()
                    .table(Users::Table)
                    .add_column(
                        ColumnDef::new(Users::MonteCarloActive)
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
                    .table(BillingAccounts::Table)
                    .drop_column(BillingAccounts::MonteCarloActive)
                    .drop_column(BillingAccounts::SeatCount)
                    .drop_column(BillingAccounts::SeatLicenseType)
                    .to_owned(),
            )
            .await?;
        manager
            .alter_table(
                Table::alter()
                    .table(Users::Table)
                    .drop_column(Users::MonteCarloActive)
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum BillingAccounts {
    Table,
    MonteCarloActive,
    SeatCount,
    SeatLicenseType,
}

#[derive(DeriveIden)]
enum Users {
    Table,
    MonteCarloActive,
}
