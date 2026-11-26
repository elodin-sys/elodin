//! Elodin Cloud authentication, projects, and recording uploads.

use serde::Deserialize;
use std::io::IsTerminal;
use std::path::PathBuf;

use miette::{Context, IntoDiagnostic};

use crate::cli::Cli;

pub(super) mod auth;
pub(super) mod projects;
pub(super) mod runs;
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

#[derive(Deserialize)]
struct Project {
    id: uuid::Uuid,
    name: String,
}

#[derive(Deserialize)]
struct Projects {
    projects: Vec<Project>,
}

async fn resolve_project(
    ctx: &AuthCtx,
    creds: &auth::tokens::Credentials,
    requested: Option<&str>,
) -> miette::Result<uuid::Uuid> {
    let projects: Projects = response(
        ctx.client
            .get(format!(
                "{}/v1/projects",
                creds.api_url.trim_end_matches('/')
            ))
            .bearer_auth(&creds.access_token)
            .send()
            .await
            .into_diagnostic()?,
    )
    .await?;
    select_project(&projects.projects, requested)
}

fn select_project(projects: &[Project], requested: Option<&str>) -> miette::Result<uuid::Uuid> {
    if let Some(value) = requested {
        // UUIDs are checked by the run endpoint, including grants and tenant scope.
        if let Ok(id) = value.parse() {
            return Ok(id);
        }
        let matches: Vec<_> = projects.iter().filter(|p| p.name == value).collect();
        return match matches.as_slice() {
            [project] => Ok(project.id),
            _ => Err(miette::miette!(
                "project '{value}' not found or ambiguous; use its UUID"
            )),
        };
    }
    match projects {
        [] => Err(miette::miette!(
            "no projects available; run `elodin projects create <name>` first"
        )),
        [project] => Ok(project.id),
        _ if !std::io::stdin().is_terminal() => Err(miette::miette!(
            "multiple projects available; specify --project <name or UUID> or ELODIN_PROJECT"
        )),
        _ => {
            for (index, project) in projects.iter().enumerate() {
                eprintln!("  {}. {} ({})", index + 1, project.name, project.id);
            }
            eprintln!("Select a project by number:");
            let mut choice = String::new();
            std::io::stdin().read_line(&mut choice).into_diagnostic()?;
            choice
                .trim()
                .parse::<usize>()
                .ok()
                .and_then(|n| n.checked_sub(1))
                .and_then(|n| projects.get(n))
                .map(|p| p.id)
                .ok_or_else(|| miette::miette!("invalid project selection"))
        }
    }
}

async fn response<T: serde::de::DeserializeOwned>(resp: reqwest::Response) -> miette::Result<T> {
    let status = resp.status();
    if !status.is_success() {
        let detail: serde_json::Value = resp.json().await.unwrap_or_default();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(miette::miette!(
                "the API rejected the token (401); run `elodin login` again"
            ));
        }
        return Err(miette::miette!("API error ({status}): {detail}"));
    }
    resp.json()
        .await
        .into_diagnostic()
        .wrap_err("invalid API response")
}
