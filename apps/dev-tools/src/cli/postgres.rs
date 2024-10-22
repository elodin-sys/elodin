use clap::Subcommand;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

use crate::cli::{export_csv, make_csv};

#[derive(clap::Args, Clone)]
pub struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Get list of users and export it to csv
    GetUsers(GetUsersArgs),
    /// Delete user with `user_id`
    DeleteUser(DeleteUserArgs),
}

#[derive(clap::Args, Clone)]
struct GetUsersArgs {
    /// Filter users by email
    #[arg(short, long)]
    email: Option<String>,
    /// Export basic user data to csv
    #[arg(short, long)]
    csv: Option<String>,
}

#[derive(clap::Args, Clone)]
struct DeleteUserArgs {
    /// Filter users by email
    #[arg(short, long)]
    user_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct PostgresUser {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub auth0_id: String,
    pub billing_account_id: Option<Uuid>,
    pub stripe_customer_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PostgresDB {
    url: String,
}

impl PostgresDB {
    pub fn new(url: impl ToString) -> Self {
        Self {
            url: url.to_string(),
        }
    }

    pub async fn run(self, args: Args) -> anyhow::Result<()> {
        match args.command {
            Commands::GetUsers(cmd_args) => {
                self.get_users(cmd_args.email, cmd_args.csv).await?;
            }
            Commands::DeleteUser(cmd_args) => {
                self.delete_user(cmd_args.user_id).await?;
            }
        }

        Ok(())
    }

    pub async fn get_users(
        &self,
        email: Option<String>,
        csv_filename: Option<String>,
    ) -> anyhow::Result<Vec<PostgresUser>> {
        let pg_pool = PgPoolOptions::new().connect(&self.url).await?;
        let response: Vec<PostgresUser> = sqlx::query_as(
            r#"
            SELECT
                u.id as id,
                u.email as email,
                u.name as name,
                u.auth0_id as auth0_id,
                u.billing_account_id as billing_account_id,
                ba.customer_id as stripe_customer_id
            FROM users u
            FULL OUTER JOIN billing_accounts ba ON ba.id = u.billing_account_id
            WHERE u.email ILIKE $1
        "#,
        )
        .bind(email.unwrap_or_default())
        .fetch_all(&pg_pool)
        .await?;

        println!("received {} user(s)", response.len());

        if let Some(csv_filename) = csv_filename {
            let data = make_csv(&response)?;
            export_csv(&csv_filename, &data)?;
        } else {
            for user in &response {
                println!("{user:?}");
            }
        }

        Ok(response)
    }

    pub async fn get_user_by_id(
        &self,
        user_id: Uuid,
        csv_filename: Option<String>,
    ) -> anyhow::Result<Option<PostgresUser>> {
        let pg_pool = PgPoolOptions::new().connect(&self.url).await?;
        let response: Vec<PostgresUser> = sqlx::query_as(
            r#"
            SELECT
                u.id as id,
                u.email as email,
                u.name as name,
                u.auth0_id as auth0_id,
                u.billing_account_id as billing_account_id,
                ba.customer_id as stripe_customer_id
            FROM users u
            FULL OUTER JOIN billing_accounts ba ON ba.id = u.billing_account_id
            WHERE u.id = $1
        "#,
        )
        .bind(user_id)
        .fetch_all(&pg_pool)
        .await?;

        println!("received {} user(s)", response.len());

        if let Some(csv_filename) = csv_filename {
            let data = make_csv(&response)?;
            export_csv(&csv_filename, &data)?;
        } else {
            for user in &response {
                println!("{user:?}");
            }
        }

        Ok(response.first().cloned())
    }

    pub async fn delete_user(&self, user_id: Uuid) -> anyhow::Result<()> {
        let pg_pool = PgPoolOptions::new().connect(&self.url).await?;
        let response = sqlx::query("DELETE FROM users WHERE id = $1")
            .bind(user_id)
            .execute(&pg_pool)
            .await?;

        println!("deleted {} user(s)", response.rows_affected());

        Ok(())
    }
}
