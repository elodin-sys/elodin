use anyhow::Context;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{prelude::*, TimeDelta};
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
struct AcessTokenPayload {
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

impl Cli {
    pub fn auth_interceptor(&self) -> anyhow::Result<AuthInterceptor> {
        let access_token = self.access_token()?;
        let access_token = format!("Bearer {access_token}").parse()?;
        Ok(AuthInterceptor { access_token })
    }

    fn access_token(&self) -> anyhow::Result<String> {
        (|| {
            let access_token_path = self
                .xdg_dirs()
                .find_data_file("access_token")
                .ok_or(anyhow::anyhow!("Missing access token"))?;
            println!("reading from {}", access_token_path.display());
            let access_token = std::fs::read_to_string(access_token_path)?;
            let payload = access_token
                .split('.')
                .nth(1)
                .ok_or(anyhow::anyhow!("Invalid access token"))?;
            let payload = URL_SAFE_NO_PAD.decode(payload)?;
            let payload = serde_json::from_slice::<AcessTokenPayload>(&payload)?;
            if Utc::now() > payload.exp {
                anyhow::bail!("Access token has expired");
            }
            Ok(access_token)
        })()
        .context("Please run `elodin login`")
    }

    pub async fn login(&self) -> anyhow::Result<()> {
        let dev = self.is_dev();
        let client_id = if dev { DEV_CLIENT_ID } else { PROD_CLIENT_ID };
        let auth_url = if dev { DEV_AUTH_URL } else { PROD_AUTH_URL };
        let audience = format!("{auth_url}/atc");

        let timeout = std::time::Duration::from_secs(5);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .https_only(true)
            .build()?;

        // request device activation
        let res = client
            .post(format!("{auth_url}/oauth/device/code"))
            .form(&[
                ("client_id", client_id),
                ("scope", "openid profile email"),
                ("audience", &audience),
            ])
            .send()
            .await?
            .error_for_status()?
            .json::<DeviceCodeResponse>()
            .await?;
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
        opener::open_browser(verification_uri_complete)?;

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
                .await?;

            if res.status().is_success() {
                let token_res = res.json::<TokenResponse>().await?;

                let expires_in = TimeDelta::try_seconds(token_res.expires_in as i64).unwrap();
                let expires_at = Utc::now() + expires_in;
                let xdg_dirs = self.xdg_dirs();
                let id_token_path = xdg_dirs.place_data_file("id_token")?;
                let access_token_path = xdg_dirs.place_data_file("access_token")?;
                let expires_at_path = xdg_dirs.place_data_file("expires_at")?;
                std::fs::write(id_token_path, token_res.id_token)?;
                std::fs::write(access_token_path, token_res.access_token)?;
                std::fs::write(expires_at_path, expires_at.to_rfc3339())?;
                println!("Logged in successfully.");
                return Ok(());
            } else if res.status().is_client_error() {
                let err_res = res.json::<ErrorResponse>().await?;
                match err_res.error.as_str() {
                    "authorization_pending" => {}
                    "slow_down" => interval += 1,
                    "expired_token" => anyhow::bail!("Authentication flow has expired"),
                    err => anyhow::bail!("Failed to authenticate: {err}"),
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(interval as u64)).await;
        }
    }
}
