//! Browser-driven OIDC authentication for Elodin Cloud.
//!
//! `elodin login` runs the OAuth 2.0 Authorization Code flow with PKCE and a
//! transient loopback redirect (with a `--device` fallback for headless shells),
//! caches the resulting tokens, and `elodin whoami` calls an auth-protected API
//! route with the access token.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use miette::{Context, IntoDiagnostic};
use serde::Deserialize;
use tokio::runtime::Runtime;

use super::Cli;

mod login;
mod logout;
mod projects;
mod signup;
mod tokens;
mod whoami;

/// Public OIDC client id for the CLI (matches the Keycloak realm import).
const CLIENT_ID: &str = "elodin-cli";
/// Scopes requested by the CLI. `offline_access` yields a refresh token.
const SCOPE: &str = "openid profile email offline_access";

#[derive(Args, Clone, Default)]
pub struct LoginArgs {
    /// Use the device authorization flow (print a URL + code) for headless/SSH shells.
    #[arg(long)]
    pub device: bool,
    /// Print the authorization URL instead of opening a browser.
    #[arg(long)]
    pub no_browser: bool,
}

#[derive(Args, Clone, Default)]
pub struct WhoamiArgs {
    /// Print the raw identity JSON returned by the API.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Clone, Default)]
pub struct SignupArgs {
    /// Email to register (prompted if omitted).
    #[arg(long)]
    pub email: Option<String>,
    /// Initial password (prompted if omitted).
    #[arg(long)]
    pub password: Option<String>,
    /// Organization name (defaults to the email's local part).
    #[arg(long)]
    pub org_name: Option<String>,
}

#[derive(Args, Clone)]
pub struct ProjectsArgs {
    #[command(subcommand)]
    pub command: ProjectsCommand,
}

#[derive(Subcommand, Clone)]
pub enum ProjectsCommand {
    /// Create a project in your organization
    Create {
        /// Project name (unique within your org)
        name: String,
        /// Print the raw JSON returned by the API.
        #[arg(long)]
        json: bool,
    },
    /// List your organization's projects
    List {
        /// Print the raw JSON returned by the API.
        #[arg(long)]
        json: bool,
    },
}

/// Resolved context shared by the auth subcommands.
pub(crate) struct AuthCtx {
    pub client: reqwest::Client,
    pub issuer: String,
    pub api_url: String,
    pub creds_path: PathBuf,
}

impl AuthCtx {
    pub async fn discover(&self) -> miette::Result<OidcConfig> {
        discover_at(&self.client, &self.issuer).await
    }
}

/// The subset of the OIDC discovery document the CLI needs.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct OidcConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(default)]
    pub device_authorization_endpoint: Option<String>,
    #[serde(default)]
    pub end_session_endpoint: Option<String>,
}

pub(crate) async fn discover_at(
    client: &reqwest::Client,
    issuer: &str,
) -> miette::Result<OidcConfig> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer.trim_end_matches('/')
    );
    client
        .get(&url)
        .send()
        .await
        .into_diagnostic()
        .wrap_err("failed to reach the OIDC issuer")?
        .error_for_status()
        .into_diagnostic()
        .wrap_err("OIDC discovery request failed")?
        .json::<OidcConfig>()
        .await
        .into_diagnostic()
        .wrap_err("failed to parse the OIDC discovery document")
}

#[derive(Debug, Deserialize)]
pub(crate) struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expires_in: Option<u64>,
}

/// PKCE verifier + S256 challenge (RFC 7636).
pub(crate) fn pkce() -> (String, String) {
    use base64::Engine;
    use ring::digest;
    use ring::rand::{SecureRandom, SystemRandom};

    let mut bytes = [0u8; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .expect("system rng unavailable");
    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let digest = digest::digest(&digest::SHA256, verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest.as_ref());
    (verifier, challenge)
}

/// Random URL-safe token, used for the OAuth `state` parameter.
pub(crate) fn random_token() -> String {
    use base64::Engine;
    use ring::rand::{SecureRandom, SystemRandom};

    let mut bytes = [0u8; 24];
    SystemRandom::new()
        .fill(&mut bytes)
        .expect("system rng unavailable");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Percent-encode a value for use in a URL query (RFC 3986 unreserved set).
pub(crate) fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

impl Cli {
    fn auth_ctx(&self) -> miette::Result<AuthCtx> {
        let dirs = self.dirs().into_diagnostic()?;
        std::fs::create_dir_all(dirs.config_dir())
            .into_diagnostic()
            .wrap_err("failed to create config directory")?;
        let creds_path = dirs.config_dir().join("credentials.json");
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

    pub fn login(self, args: LoginArgs, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(login::run(&ctx, &args))
    }

    pub fn logout(self, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(logout::run(&ctx))
    }

    pub fn whoami(self, args: WhoamiArgs, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(whoami::run(&ctx, &args))
    }

    pub fn signup(self, args: SignupArgs, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(signup::run(&ctx, &args))
    }

    pub fn projects(self, args: ProjectsArgs, rt: Runtime) -> miette::Result<()> {
        let ctx = self.auth_ctx()?;
        rt.block_on(projects::run(&ctx, &args))
    }
}
