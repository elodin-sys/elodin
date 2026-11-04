//! Elodin Cloud authentication, projects, and recording uploads.

use std::path::PathBuf;

use miette::{Context, IntoDiagnostic};

use crate::cli::Cli;

pub(super) mod auth;
pub(super) mod projects;
pub(super) mod upload;

/// Resolved context shared by the cloud subcommands.
pub(crate) struct AuthCtx {
    pub client: reqwest::Client,
    pub issuer: String,
    pub api_url: String,
    pub creds_path: PathBuf,
}

impl AuthCtx {
    pub async fn discover(&self) -> miette::Result<auth::OidcConfig> {
        auth::discover_at(&self.client, &self.issuer).await
    }
}

impl Cli {
    fn auth_ctx(&self) -> miette::Result<AuthCtx> {
        let config_dir = match &self.config_dir {
            Some(path) => path.clone(),
            None => self.dirs().into_diagnostic()?.config_dir().to_path_buf(),
        };
        std::fs::create_dir_all(&config_dir)
            .into_diagnostic()
            .wrap_err("failed to create config directory")?;
        let creds_path = config_dir.join("credentials.json");
        let client = reqwest::Client::builder()
            .user_agent(concat!("elodin-cli/", env!("CARGO_PKG_VERSION")))
            .build()
            .into_diagnostic()
            .wrap_err("failed to build HTTP client")?;
        Ok(AuthCtx {
            client,
            issuer: self.issuer.trim_end_matches('/').to_string(),
            api_url: self.api_url.trim_end_matches('/').to_string(),
            creds_path,
        })
    }
}
