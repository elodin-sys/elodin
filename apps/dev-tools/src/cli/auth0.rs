use crate::cli::{export_csv, make_csv};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

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
    DeleteUserById(DeleteUserByIdArgs),
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
struct DeleteUserByIdArgs {
    /// Filter users by `user_id`
    #[arg(short, long)]
    user_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Auth0User {
    user_id: String,
    email: String,
    name: String,
}

#[derive(Debug, Clone)]
pub struct Auth0 {
    token: String,
    base_url: String,
    client: reqwest::Client,
}

impl Auth0 {
    pub fn new(token: String, prod: bool) -> Self {
        let client = reqwest::Client::new();
        let base_url = if prod {
            "https://elodin.us.auth0.com"
        } else {
            "https://dev-i2ytsp68gngieek3.us.auth0.com"
        };

        Self {
            token,
            base_url: base_url.to_string(),
            client,
        }
    }

    pub async fn run(self, args: Args) -> anyhow::Result<()> {
        match args.command {
            Commands::GetUsers(cmd_args) => {
                self.get_users(cmd_args.email, cmd_args.csv).await?;
            }
            Commands::DeleteUserById(cmd_args) => {
                self.delete_user_by_id(&cmd_args.user_id).await?;
            }
        }

        Ok(())
    }

    pub async fn get_users(
        &self,
        email: Option<String>,
        csv_filename: Option<String>,
    ) -> anyhow::Result<Vec<Auth0User>> {
        let query_email = email
            .map(|email| format!("&q=email:{email}"))
            .unwrap_or_default();
        let query = format!("fields=user_id,email,name{query_email}");

        let mut users = vec![];
        let mut page: u32 = 0;

        loop {
            let mut response = self
                .client
                .get(format!(
                    "{}/api/v2/users?{query}&per_page=100&page={page}",
                    self.base_url
                ))
                .bearer_auth(self.token.clone())
                .send()
                .await?
                .json::<Vec<Auth0User>>()
                .await?;

            println!("received {} user(s)", response.len());

            if !response.is_empty() {
                page += 1;
                users.append(&mut response);
            } else {
                break;
            }
        }

        println!("total - received {} user(s)", users.len());

        if let Some(csv_filename) = csv_filename {
            let data = make_csv(&users)?;
            export_csv(&csv_filename, &data)?;
        } else {
            for user in &users {
                println!("{user:?}");
            }
        }

        Ok(users)
    }

    pub async fn delete_user_by_id(&self, user_id: &str) -> anyhow::Result<()> {
        let response = self
            .client
            .delete(format!("{}/api/v2/users/{user_id}", self.base_url))
            .bearer_auth(self.token.clone())
            .send()
            .await?;

        if response.status().is_success() {
            println!("{user_id} user was removed");
        } else {
            panic!(
                "delete_user_by_id - unexpected response status : {:?}",
                response.status()
            );
        }

        Ok(())
    }
}
