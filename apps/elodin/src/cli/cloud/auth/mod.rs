//! Browser-driven OIDC authentication for Elodin Cloud.
//!
//! `elodin login` runs the OAuth 2.0 Authorization Code flow with PKCE and a
//! transient loopback redirect (with a `--device` fallback for headless shells),
//! caches the resulting tokens, and `elodin whoami` calls an auth-protected API
//! route with the access token.

use clap::Args;
use miette::{Context, IntoDiagnostic};
use serde::Deserialize;
use tokio::runtime::Runtime;

use crate::cli::Cli;

use super::AuthCtx;

mod login;
mod logout;
pub(super) mod tokens;
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
    /// Print the authorization URL instead of opening a browser (also with --device).
    #[arg(long)]
    pub no_browser: bool,
}

#[derive(Args, Clone, Default)]
pub struct WhoamiArgs {
    /// Print the raw identity JSON returned by the API.
    #[arg(long)]
    pub json: bool,
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
}
