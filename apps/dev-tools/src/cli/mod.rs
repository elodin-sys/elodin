use auth0::Auth0;
use clap::{Parser, Subcommand};
use postgres::PostgresDB;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use stripe::Stripe;
use uuid::Uuid;

use crate::config::Config;

mod auth0;
mod postgres;
mod stripe;

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    /// Run commands against production environment
    #[arg(long)]
    prod: bool,
    #[arg(long, hide = true)]
    markdown_help: bool,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Manage users in Auth0
    Auth0(auth0::Args),
    /// Manage users in Stripe
    Stripe(stripe::Args),
    /// Manage users in Postgres database located in out cluster
    PostgresDB(postgres::Args),
    /// Display all users from all services filtered by email
    GetUsers(GetUsersArgs),
    /// Delete user by id
    DeleteUser(DeleteUserArgs),
}

#[derive(clap::Args, Clone)]
struct GetUsersArgs {
    /// Filter users by email
    #[arg(short, long)]
    email: String,
}

#[derive(clap::Args, Clone)]
struct DeleteUserArgs {
    /// User ID in the Postgres database
    #[arg(short, long)]
    user_id: Uuid,
}

impl Cli {
    pub fn from_os_args() -> Self {
        Self::parse()
    }

    pub fn run(self) -> anyhow::Result<()> {
        let config = Config::new()?;

        if self.markdown_help {
            clap_markdown::print_help_markdown::<Cli>();
            std::process::exit(0);
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime failed to start");

        match self.command {
            Some(Commands::Auth0(args)) => {
                if let Some(auth0_token) = config.auth0_token {
                    rt.block_on(Auth0::new(auth0_token, self.prod).run(args))?;
                } else {
                    println!("ELODIN_AUTH0_TOKEN is missing from env");
                }
            }
            Some(Commands::Stripe(args)) => {
                let Some(stripe_token) = config.stripe_token else {
                    println!("ELODIN_STRIPE_TOKEN is missing from env");
                    return Ok(());
                };

                if self.prod && stripe_token.starts_with("sk_test") {
                    println!(
                        "ERROR: You're trying to use Stripe test token with a production flag"
                    );
                    return Ok(());
                }

                rt.block_on(Stripe::new(stripe_token, "https://api.stripe.com").run(args))?;
            }
            Some(Commands::PostgresDB(args)) => {
                rt.block_on(
                    PostgresDB::new("postgres://postgres:secret@127.0.0.1:5432/postgres").run(args),
                )?;
            }
            Some(Commands::GetUsers(args)) => {
                let Some(stripe_token) = config.stripe_token else {
                    println!("ELODIN_STRIPE_TOKEN is missing from env");
                    return Ok(());
                };

                let Some(auth0_token) = config.auth0_token else {
                    println!("ELODIN_AUTH0_TOKEN is missing from env");
                    return Ok(());
                };

                if self.prod && stripe_token.starts_with("sk_test") {
                    println!(
                        "ERROR: You're trying to use Stripe test token with a production flag"
                    );
                    return Ok(());
                }

                rt.block_on(get_users(
                    Stripe::new(stripe_token, "https://api.stripe.com"),
                    Auth0::new(auth0_token, self.prod),
                    PostgresDB::new("postgres://postgres:secret@127.0.0.1:5432/postgres"),
                    args.email,
                ))?;
            }
            Some(Commands::DeleteUser(args)) => {
                let Some(stripe_token) = config.stripe_token else {
                    println!("ELODIN_STRIPE_TOKEN is missing from env");
                    return Ok(());
                };

                let Some(auth0_token) = config.auth0_token else {
                    println!("ELODIN_AUTH0_TOKEN is missing from env");
                    return Ok(());
                };

                if self.prod && stripe_token.starts_with("sk_test") {
                    println!(
                        "ERROR: You're trying to use Stripe test token with a production flag"
                    );
                    return Ok(());
                }

                rt.block_on(delete_user(
                    Stripe::new(stripe_token, "https://api.stripe.com"),
                    Auth0::new(auth0_token, self.prod),
                    PostgresDB::new("postgres://postgres:secret@127.0.0.1:5432/postgres"),
                    args.user_id,
                ))?;
            }
            None => {
                println!("empty");
            }
        }

        Ok(())
    }
}

pub fn export_csv(filename: &str, content: &str) -> std::io::Result<()> {
    let mut output = File::create(format!("{filename}.csv"))?;
    write!(output, "{}", content)
}

pub fn make_csv<T: Serialize>(records: &Vec<T>) -> anyhow::Result<String> {
    let mut wrt = csv::Writer::from_writer(vec![]);

    for record in records {
        wrt.serialize(record)?;
    }

    let csv_str = String::from_utf8(wrt.into_inner()?)?;

    Ok(csv_str)
}

pub async fn get_users(
    stripe: Stripe,
    auth0: Auth0,
    pg: PostgresDB,
    email: String,
) -> anyhow::Result<()> {
    println!("---- get user data from postgres ----");
    pg.get_users(Some(format!("%{email}%")), None).await?;
    println!("---- get user data from stripe ----");
    stripe.get_users(Some(email.clone()), None, true).await?;
    println!("---- get user data from auth0 ----");
    auth0.get_users(Some(format!("*{email}*")), None).await?;

    Ok(())
}

pub async fn delete_user(
    stripe: Stripe,
    auth0: Auth0,
    pg: PostgresDB,
    user_id: Uuid,
) -> anyhow::Result<()> {
    println!("---- find user by id in postgres ----");
    let pg_user = pg.get_user_by_id(user_id, None).await?;
    if let Some(pg_user) = pg_user {
        println!("found user: {:?}", pg_user);

        println!("---- delete {} user from auth0 ----", pg_user.auth0_id);
        if let Err(err) = auth0.delete_user_by_id(&pg_user.auth0_id).await {
            println!("{err:?}");
            println!("ERROR: couldn't delete user from auth0, if this is unexpected - please investigate");
        }

        if let Some(stripe_customer_id) = pg_user.stripe_customer_id {
            println!(
                "---- delete {} customer from stripe ----",
                stripe_customer_id
            );
            if let Err(err) = stripe.delete_user(stripe_customer_id).await {
                println!("{err:?}");
                println!("ERROR: couldn't delete customer from stripe, if this is unexpected - please investigate");
            }
        } else {
            println!("---- no billing information associated with this user, skipping stripe ----");
        }

        println!("---- delete {user_id} customer from postgres ----");
        pg.delete_user(user_id).await?;
    } else {
        println!("---- couldn't find this user_id in postgres ----");
    }

    Ok(())
}
