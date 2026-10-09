//! Credential storage (`credentials.json`, mode 600) and refresh handling. The file keeps one
//! session per issuer, so logging in to one Elodin Cloud environment leaves the others signed in.

use std::collections::BTreeMap;
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

/// Sessions keyed by issuer.
#[derive(Default, Serialize, Deserialize)]
struct Store {
    sessions: BTreeMap<String, Credentials>,
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

    /// The session for `issuer`, if any.
    pub fn load(path: &Path, issuer: &str) -> miette::Result<Option<Credentials>> {
        Ok(read_store(path)?.sessions.remove(issuer))
    }

    pub fn save(&self, path: &Path) -> miette::Result<()> {
        let mut store = read_store(path)?;
        store.sessions.insert(self.issuer.clone(), self.clone());
        write_store(path, &store)
    }

    /// Forget the session for `issuer`; false if there was none.
    pub fn remove(path: &Path, issuer: &str) -> miette::Result<bool> {
        let mut store = read_store(path)?;
        if store.sessions.remove(issuer).is_none() {
            return Ok(false);
        }
        write_store(path, &store)?;
        Ok(true)
    }
}

fn read_store(path: &Path) -> miette::Result<Store> {
    match std::fs::read(path) {
        Ok(bytes) => {
            if let Ok(store) = serde_json::from_slice::<Store>(&bytes) {
                return Ok(store);
            }
            // Before per-issuer sessions the file held a single session.
            let single: Credentials = serde_json::from_slice(&bytes)
                .into_diagnostic()
                .wrap_err("failed to parse credentials.json")?;
            Ok(Store {
                sessions: BTreeMap::from([(single.issuer.clone(), single)]),
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Store::default()),
        Err(e) => Err(e)
            .into_diagnostic()
            .wrap_err("failed to read credentials.json"),
    }
}

/// Write a temporary file and rename it over the old one, so a crash never leaves a truncated
/// file and the result is always mode 600.
fn write_store(path: &Path, store: &Store) -> miette::Result<()> {
    let json = serde_json::to_vec_pretty(store).into_diagnostic()?;
    let tmp = path.with_extension("json.tmp");
    let _ = std::fs::remove_file(&tmp);
    write_private(&tmp, &json).wrap_err("failed to write credentials.json")?;
    std::fs::rename(&tmp, path)
        .into_diagnostic()
        .wrap_err("failed to replace credentials.json")
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> miette::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .into_diagnostic()?;
    file.write_all(bytes).into_diagnostic()?;
    file.sync_all().into_diagnostic()?;
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

/// Load this issuer's credentials, transparently refreshing the access token if expired.
pub async fn ensure_valid(ctx: &AuthCtx) -> miette::Result<Credentials> {
    let mut creds = Credentials::load(&ctx.creds_path, &ctx.issuer)?
        .ok_or_else(|| miette::miette!("not logged in to {}; run `elodin login`", ctx.issuer))?;
    if creds.is_expired() {
        let oidc = discover_at(&ctx.client, &creds.issuer).await?;
        refresh(ctx, &oidc, &mut creds)
            .await
            .wrap_err("failed to refresh session; run `elodin login` again")?;
        creds.save(&ctx.creds_path)?;
    }
    Ok(creds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(issuer: &str) -> Credentials {
        Credentials {
            access_token: format!("token-{issuer}"),
            refresh_token: None,
            expires_at: now() + 300,
            issuer: issuer.to_string(),
            api_url: "https://api.example.test".to_string(),
        }
    }

    #[test]
    fn sessions_are_kept_per_issuer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        session("https://a.test/realms/elodin").save(&path).unwrap();
        session("https://b.test/realms/elodin").save(&path).unwrap();
        let a = Credentials::load(&path, "https://a.test/realms/elodin").unwrap();
        assert_eq!(
            a.unwrap().access_token,
            "token-https://a.test/realms/elodin"
        );
        assert!(Credentials::remove(&path, "https://b.test/realms/elodin").unwrap());
        assert!(
            Credentials::load(&path, "https://b.test/realms/elodin")
                .unwrap()
                .is_none()
        );
        assert!(
            Credentials::load(&path, "https://a.test/realms/elodin")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn reads_the_single_session_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        let old = session("https://old.test/realms/elodin");
        std::fs::write(&path, serde_json::to_vec(&old).unwrap()).unwrap();
        let loaded = Credentials::load(&path, "https://old.test/realms/elodin").unwrap();
        assert_eq!(loaded.unwrap().access_token, old.access_token);
    }

    #[cfg(unix)]
    #[test]
    fn rewrites_with_private_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credentials.json");
        std::fs::write(&path, b"{\"sessions\":{}}").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        session("https://a.test/realms/elodin").save(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
