use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_table(
                Table::create()
                    .table(Users::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Users::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(Users::Name).string().not_null())
                    .col(
                        ColumnDef::new(Users::Email)
                            .string()
                            .not_null()
                            .unique_key(),
                    )
                    .col(
                        ColumnDef::new(Users::Auth0Id)
                            .string()
                            .not_null()
                            .unique_key(),
                    )
                    .col(ColumnDef::new(Users::Permissions).json_binary().not_null())
                    .to_owned(),
            )
            .await?;
        manager
            .create_table(
                Table::create()
                    .table(Sandboxes::Table)
                    .if_not_exists()
                    .col(
                        ColumnDef::new(Sandboxes::Id)
                            .uuid()
                            .not_null()
                            .primary_key(),
                    )
                    .col(ColumnDef::new(Sandboxes::UserId).uuid().not_null())
                    .col(ColumnDef::new(Sandboxes::Name).string().not_null())
                    .col(ColumnDef::new(Sandboxes::Code).string().not_null())
                    .col(ColumnDef::new(Sandboxes::Status).integer().not_null())
                    .col(ColumnDef::new(Sandboxes::VmId).uuid())
                    .foreign_key(
                        ForeignKey::create()
                            .from(Sandboxes::Table, Sandboxes::UserId)
                            .to(Users::Table, Users::Id),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Vms::Table)
                    .if_not_exists()
                    .col(ColumnDef::new(Vms::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(Vms::PodName).string().not_null())
                    .col(ColumnDef::new(Vms::Status).integer().not_null())
                    .col(ColumnDef::new(Vms::PodIp).string())
                    .col(ColumnDef::new(Vms::SandboxId).uuid())
                    .to_owned(),
            )
            .await?;
        // manager.foreign_key(
        //     ForeignKey::create()
        //         .from(Sandboxs::Table, Sandboxs::VmId)
        //         .to(Vms::Table, Vms::Id),
        // );

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Users::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Vms::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Sandboxes::Table).to_owned())
            .await
    }
}

#[derive(DeriveIden)]
enum Users {
    Table,
    Id,
    Email,
    Name,
    Auth0Id,
    Permissions,
}

#[derive(DeriveIden)]
enum Sandboxes {
    Table,
    Id,
    UserId,
    Name,
    Code,
    Status,
    VmId,
}

#[derive(DeriveIden)]
enum Vms {
    Table,
    Id,
    PodName,
    Status,
    PodIp,
    SandboxId,
}
