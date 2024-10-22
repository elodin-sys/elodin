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
    /// Use substring filter
    #[arg(short, long)]
    approximate: bool,
}

#[derive(clap::Args, Clone)]
struct DeleteUserArgs {
    /// Filter users by `user_id`
    #[arg(short, long)]
    user_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct StripeCustomerMetadata {
    billing_account_id: String,
    user_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StripeCustomer {
    id: String,
    email: String,
    name: String,
    metadata: StripeCustomerMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StripeCustomerFlat {
    id: String,
    email: String,
    name: String,
    billing_account_id: String,
    user_id: String,
}

impl StripeCustomerFlat {
    pub fn new(sc: StripeCustomer) -> Self {
        Self {
            id: sc.id,
            email: sc.email,
            name: sc.name,
            billing_account_id: sc.metadata.billing_account_id,
            user_id: sc.metadata.user_id,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct StripeCustomerResponse {
    data: Vec<StripeCustomer>,
    next_page: Option<String>,
}

pub struct Stripe {
    token: String,
    base_url: String,
    client: reqwest::Client,
}

impl Stripe {
    pub fn new(token: String, base_url: impl ToString) -> Self {
        let client = reqwest::Client::new();

        Self {
            token,
            base_url: base_url.to_string(),
            client,
        }
    }

    pub async fn run(self, args: Args) -> anyhow::Result<()> {
        match args.command {
            Commands::GetUsers(cmd_args) => {
                self.get_users(cmd_args.email, cmd_args.csv, cmd_args.approximate)
                    .await?;
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
        approximate: bool,
    ) -> anyhow::Result<Vec<StripeCustomerFlat>> {
        let search_char = if approximate { "~" } else { ":" };
        let query = email
            .map(|email| format!("&query=email{search_char}\"{email}\""))
            .unwrap_or_default();

        let mut customers: Vec<StripeCustomerFlat> = vec![];
        let mut next_page: Option<String> = None;

        loop {
            let query_page = next_page
                .map(|page| format!("&page={page}"))
                .unwrap_or_default();

            let response = self
                .client
                .get(format!(
                    "{}/v1/customers/search?limit=100{query_page}{query}",
                    self.base_url
                ))
                .basic_auth(self.token.clone(), Some(""))
                .send()
                .await?
                .json::<StripeCustomerResponse>()
                .await?;

            println!("received {} user(s)", response.data.len());

            let mut new_customers = response
                .data
                .into_iter()
                .map(StripeCustomerFlat::new)
                .collect::<Vec<StripeCustomerFlat>>();

            customers.append(&mut new_customers);

            if response.next_page.is_some() {
                next_page = response.next_page;
            } else {
                break;
            }
        }

        println!("total - received {} user(s)", customers.len());

        if let Some(csv_filename) = csv_filename {
            let data = make_csv(&customers)?;
            export_csv(&csv_filename, &data)?;
        } else {
            for user in &customers {
                println!("{user:?}");
            }
        }

        Ok(customers)
    }

    pub async fn delete_user(&self, user_id: String) -> anyhow::Result<()> {
        let response = self
            .client
            .delete(format!("{}/v1/customers/{user_id}", self.base_url))
            .basic_auth(self.token.clone(), Some(""))
            .send()
            .await?
            .error_for_status()?;

        if response.status().is_success() {
            println!("{user_id} user was removed");
        }

        Ok(())
    }
}
