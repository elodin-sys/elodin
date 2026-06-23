//! Credential storage (`credentials.json`, mode 600) and refresh handling.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use miette::{Context, IntoDiagnostic};
use serde::{Deserialize, Serialize};

use super::{AuthCtx, CLIENT_ID, OidcConfig, TokenResponse, discover_at};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    /// Unix seconds at which `access_token` expires.
    pub expires_at: u64,
    pub issuer: String,
    pub api_url: String,
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Credentials {
    pub fn from_token(
        token: TokenResponse,
        issuer: &str,
        api_url: &str,
        prev_refresh: Option<String>,
    ) -> Self {
        Credentials {
            expires_at: now() + token.expires_in.unwrap_or(300),
            access_token: token.access_token,
            refresh_token: token.refresh_token.or(prev_refresh),
            issuer: issuer.to_string(),
            api_url: api_url.to_string(),
        }
    }

    /// True if the access token is expired (with a small skew).
    pub fn is_expired(&self) -> bool {
        now() + 30 >= self.expires_at
    }

    pub fn load(path: &Path) -> miette::Result<Option<Credentials>> {
        match std::fs::read(path) {
            Ok(bytes) => Ok(Some(
                serde_json::from_slice(&bytes)
                    .into_diagnostic()
                    .wrap_err("failed to parse credentials.json")?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e)
                .into_diagnostic()
                .wrap_err("failed to read credentials.json"),
        }
    }

    pub fn save(&self, path: &Path) -> miette::Result<()> {
        let json = serde_json::to_vec_pretty(self).into_diagnostic()?;
        write_private(path, &json).wrap_err("failed to write credentials.json")
    }
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> miette::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .into_diagnostic()?;
    file.write_all(bytes).into_diagnostic()?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> miette::Result<()> {
    std::fs::write(path, bytes).into_diagnostic()
}

/// Exchange the stored refresh token for a fresh access token.
pub async fn refresh(
    ctx: &AuthCtx,
    oidc: &OidcConfig,
    creds: &mut Credentials,
) -> miette::Result<()> {
    let refresh_token = creds
        .refresh_token
        .clone()
        .ok_or_else(|| miette::miette!("no refresh token stored; run `elodin login` again"))?;
    let resp = ctx
        .client
        .post(&oidc.token_endpoint)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", refresh_token.as_str()),
        ])
        .send()
        .await
        .into_diagnostic()
        .wrap_err("token refresh request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(miette::miette!("token refresh failed ({status}): {body}"));
    }
    let token: TokenResponse = resp
        .json()
        .await
        .into_diagnostic()
        .wrap_err("failed to parse refreshed token")?;
    *creds = Credentials::from_token(token, &creds.issuer, &creds.api_url, Some(refresh_token));
    Ok(())
}

/// Load credentials, transparently refreshing the access token if expired.
pub async fn ensure_valid(ctx: &AuthCtx) -> miette::Result<Credentials> {
    let mut creds = Credentials::load(&ctx.creds_path)?
        .ok_or_else(|| miette::miette!("not logged in; run `elodin login`"))?;
    if creds.is_expired() {
        let oidc = discover_at(&ctx.client, &creds.issuer).await?;
        refresh(ctx, &oidc, &mut creds)
            .await
            .wrap_err("failed to refresh session; run `elodin login` again")?;
        creds.save(&ctx.creds_path)?;
    }
    Ok(creds)
}
