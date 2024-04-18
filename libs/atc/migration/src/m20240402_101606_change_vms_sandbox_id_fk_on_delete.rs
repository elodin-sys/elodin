use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_foreign_key(
                ForeignKey::drop()
                    .name("vms_sandbox_id_fkey")
                    .table(Vms::Table)
                    .to_owned(),
            )
            .await?;

        manager
            .create_foreign_key(
                ForeignKey::create()
                    .from(Vms::Table, Vms::SandboxId)
                    .to(Sandboxes::Table, Sandboxes::Id)
                    .on_delete(ForeignKeyAction::Cascade)
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_foreign_key(
                ForeignKey::drop()
                    .name("vms_sandbox_id_fkey")
                    .table(Vms::Table)
                    .to_owned(),
            )
            .await?;

        manager
            .create_foreign_key(
                ForeignKey::create()
                    .from(Vms::Table, Vms::SandboxId)
                    .to(Sandboxes::Table, Sandboxes::Id)
                    .on_delete(ForeignKeyAction::SetNull)
                    .to_owned(),
            )
            .await
    }
}

#[derive(DeriveIden)]
enum Sandboxes {
    Table,
    Id,
}

#[derive(DeriveIden)]
enum Vms {
    Table,
    SandboxId,
}
