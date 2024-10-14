use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{prelude::*, TimeDelta};
use miette::Diagnostic;
use miette::{diagnostic, Context, IntoDiagnostic};
use thiserror::Error;
use tonic::{metadata, service};

use super::Cli;

const DEV_CLIENT_ID: &str = "N4CQBSuOoiBsQKTBPWwMBYGZ5AQaj7MG";
const DEV_AUTH_URL: &str = "https://auth.elodin.dev";

const PROD_CLIENT_ID: &str = "wSUtuPb5wzXsyFmunbZT2QEzgoK9Ia0p";
const PROD_AUTH_URL: &str = "https://auth.elodin.systems";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: String,
    expires_in: u32,
    interval: u32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ErrorResponse {
    error: String,
    error_description: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    id_token: String,
    token_type: String,
    expires_in: u32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct AccessTokenPayload {
    iss: String,
    sub: String,
    aud: Vec<String>,
    #[serde(with = "chrono::serde::ts_seconds")]
    iat: DateTime<Utc>,
    #[serde(with = "chrono::serde::ts_seconds")]
    exp: DateTime<Utc>,
    azp: String,
    scope: String,
}

pub struct AuthInterceptor {
    access_token: metadata::MetadataValue<metadata::Ascii>,
}

impl service::Interceptor for AuthInterceptor {
    fn call(
        &mut self,
        mut request: tonic::Request<()>,
    ) -> Result<tonic::Request<()>, tonic::Status> {
        request
            .metadata_mut()
            .insert("authorization", self.access_token.clone());
        Ok(request)
    }
}

#[derive(Error, Diagnostic, Debug)]
#[diagnostic(help("please run `elodin login`"))]
pub enum AuthError {
    #[error(transparent)]
    #[diagnostic(code(elodin::io_error))]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    #[diagnostic(code(elodin::json_parse_error))]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    #[diagnostic(code(elodin::base64_decode_error))]
    Base64(#[from] base64::DecodeError),
    #[error("access token has expired")]
    #[diagnostic(code(elodin::access_token_expired))]
    AccessTokenExpired,
    #[error("invalid access token")]
    #[diagnostic(code(elodin::invalid_access_token))]
    InvalidAccessToken,
    #[error("access token missing")]
    #[diagnostic(code(elodin::no_access_token))]
    NoAccessToken,
}

impl Cli {
    pub fn auth_interceptor(&self) -> miette::Result<AuthInterceptor> {
        let access_token = self.access_token()?;
        let access_token = format!("Bearer {access_token}").parse().into_diagnostic()?;
        Ok(AuthInterceptor { access_token })
    }

    fn access_token(&self) -> Result<String, AuthError> {
        let dirs = self.dirs()?;
        let data_dir = dirs.data_dir();
        let access_token_path = data_dir.join("access_token");
        if !access_token_path.exists() {
            return Err(AuthError::NoAccessToken);
        }
        let access_token = std::fs::read_to_string(access_token_path)?;
        let payload = access_token
            .split('.')
            .nth(1)
            .ok_or(AuthError::InvalidAccessToken)?;
        let payload = URL_SAFE_NO_PAD.decode(payload)?;
        let payload = serde_json::from_slice::<AccessTokenPayload>(&payload)?;
        if Utc::now() > payload.exp {
            return Err(AuthError::AccessTokenExpired);
        }
        Ok(access_token)
    }

    pub async fn login(&self) -> miette::Result<()> {
        let dev = self.is_dev();
        let client_id = if dev { DEV_CLIENT_ID } else { PROD_CLIENT_ID };
        let auth_url = if dev { DEV_AUTH_URL } else { PROD_AUTH_URL };
        let audience = format!("{auth_url}/atc");

        let timeout = std::time::Duration::from_secs(5);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .https_only(true)
            .build()
            .into_diagnostic()?;

        // request device activation
        let res = client
            .post(format!("{auth_url}/oauth/device/code"))
            .form(&[
                ("client_id", client_id),
                ("scope", "openid profile email"),
                ("audience", &audience),
            ])
            .send()
            .await
            .into_diagnostic()?
            .error_for_status()
            .into_diagnostic()?
            .json::<DeviceCodeResponse>()
            .await
            .into_diagnostic()?;
        let DeviceCodeResponse {
            device_code,
            user_code,
            verification_uri_complete,
            expires_in,
            mut interval,
            ..
        } = res;

        let expires_in_mins = expires_in / 60;
        println!("Login via the browser: {verification_uri_complete}.");
        println!("You should see the following code: {user_code}, which expires in {expires_in_mins} minutes.");
        // Open browser if possible
        let _ = opener::open_browser(verification_uri_complete);

        // request access token
        loop {
            let res = client
                .post(format!("{auth_url}/oauth/token"))
                .form(&[
                    ("client_id", client_id),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", &device_code),
                ])
                .send()
                .await
                .into_diagnostic()?;

            if res.status().is_success() {
                let token_res = res.json::<TokenResponse>().await.into_diagnostic()?;
                let expires_in = TimeDelta::try_seconds(token_res.expires_in as i64).unwrap();
                let expires_at = Utc::now() + expires_in;
                let dirs = self.dirs().into_diagnostic()?;
                let data_dir = dirs.data_dir();
                std::fs::create_dir_all(data_dir)
                    .into_diagnostic()
                    .context("failed to create data directory")?;
                let id_token_path = data_dir.join("id_token");
                let access_token_path = data_dir.join("access_token");
                let expires_at_path = data_dir.join("expires_at");
                std::fs::write(id_token_path, token_res.id_token).into_diagnostic()?;
                std::fs::write(access_token_path, token_res.access_token).into_diagnostic()?;
                std::fs::write(expires_at_path, expires_at.to_rfc3339()).into_diagnostic()?;
                println!("Logged in successfully.");
                return Ok(());
            } else if res.status().is_client_error() {
                let err_res = res.json::<ErrorResponse>().await.into_diagnostic()?;
                match err_res.error.as_str() {
                    "authorization_pending" => {}
                    "slow_down" => interval += 1,
                    "expired_token" => miette::bail!("Authentication flow has expired"),
                    err => miette::bail!("Failed to authenticate: {err}"),
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(interval as u64)).await;
        }
    }
}
