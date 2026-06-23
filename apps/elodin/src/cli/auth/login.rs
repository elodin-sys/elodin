//! `elodin login`: browser Authorization Code + PKCE (default) and a device-code fallback.

use std::time::Duration;

use miette::{Context, IntoDiagnostic};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::tokens::{self, Credentials};
use super::{AuthCtx, LoginArgs, OidcConfig, TokenResponse, pkce, random_token, urlencode};
use super::{CLIENT_ID, SCOPE};

pub async fn run(ctx: &AuthCtx, args: &LoginArgs) -> miette::Result<()> {
    let oidc = ctx.discover().await.wrap_err("OIDC discovery failed")?;
    let creds = if args.device {
        device_flow(ctx, &oidc).await?
    } else {
        browser_flow(ctx, &oidc, args).await?
    };
    creds.save(&ctx.creds_path)?;
    println!(
        "\nLogged in to {}. Credentials saved to {}.",
        ctx.issuer,
        ctx.creds_path.display()
    );
    Ok(())
}

async fn browser_flow(
    ctx: &AuthCtx,
    oidc: &OidcConfig,
    args: &LoginArgs,
) -> miette::Result<Credentials> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .into_diagnostic()
        .wrap_err("failed to bind the loopback redirect listener")?;
    let port = listener.local_addr().into_diagnostic()?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let (verifier, challenge) = pkce();
    let state = random_token();
    let auth_url = format!(
        "{endpoint}?response_type=code&client_id={client}&redirect_uri={redirect}&scope={scope}\
         &code_challenge={challenge}&code_challenge_method=S256&state={state}",
        endpoint = oidc.authorization_endpoint,
        client = urlencode(CLIENT_ID),
        redirect = urlencode(&redirect_uri),
        scope = urlencode(SCOPE),
        challenge = urlencode(&challenge),
        state = urlencode(&state),
    );

    if args.no_browser {
        println!("Open this URL to log in:\n  {auth_url}");
    } else {
        println!("Opening your browser to log in...");
        if opener::open(&auth_url).is_err() {
            println!("Could not open a browser automatically. Open this URL:\n  {auth_url}");
        }
    }

    let (code, returned_state) = wait_for_callback(listener).await?;
    if returned_state != state {
        return Err(miette::miette!("OAuth state mismatch; aborting login"));
    }

    let token = exchange_code(ctx, oidc, &code, &redirect_uri, &verifier).await?;
    Ok(Credentials::from_token(
        token,
        &ctx.issuer,
        &ctx.api_url,
        None,
    ))
}

/// Accept the single loopback redirect and extract `code`/`state`.
async fn wait_for_callback(listener: TcpListener) -> miette::Result<(String, String)> {
    loop {
        let (mut stream, _) = listener.accept().await.into_diagnostic()?;
        let mut buf = vec![0u8; 8192];
        let n = stream.read(&mut buf).await.into_diagnostic()?;
        let request = String::from_utf8_lossy(&buf[..n]);
        let target = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("");

        if !target.starts_with("/callback") {
            respond(&mut stream, "Not found").await;
            continue;
        }

        let query = target.split_once('?').map(|(_, q)| q).unwrap_or("");
        let (mut code, mut state, mut error) = (None, None, None);
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            let value = urldecode(value);
            match key {
                "code" => code = Some(value),
                "state" => state = Some(value),
                "error" => error = Some(value),
                _ => {}
            }
        }

        respond(
            &mut stream,
            "<html><body style=\"font-family:sans-serif\"><h2>Elodin login complete</h2>\
             <p>You can close this tab and return to your terminal.</p></body></html>",
        )
        .await;

        if let Some(error) = error {
            return Err(miette::miette!("authorization failed: {error}"));
        }
        let code = code.ok_or_else(|| miette::miette!("no authorization code in callback"))?;
        return Ok((code, state.unwrap_or_default()));
    }
}

async fn respond(stream: &mut tokio::net::TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.flush().await;
}

async fn exchange_code(
    ctx: &AuthCtx,
    oidc: &OidcConfig,
    code: &str,
    redirect_uri: &str,
    verifier: &str,
) -> miette::Result<TokenResponse> {
    let resp = ctx
        .client
        .post(&oidc.token_endpoint)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", CLIENT_ID),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", verifier),
        ])
        .send()
        .await
        .into_diagnostic()
        .wrap_err("token exchange request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(miette::miette!("token exchange failed ({status}): {body}"));
    }
    resp.json()
        .await
        .into_diagnostic()
        .wrap_err("failed to parse token response")
}

#[derive(Deserialize)]
struct DeviceAuth {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    #[serde(default)]
    interval: Option<u64>,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Deserialize)]
struct DeviceError {
    error: String,
}

async fn device_flow(ctx: &AuthCtx, oidc: &OidcConfig) -> miette::Result<Credentials> {
    let endpoint = oidc
        .device_authorization_endpoint
        .clone()
        .ok_or_else(|| miette::miette!("the provider does not advertise a device flow"))?;
    let (verifier, challenge) = pkce();

    let resp = ctx
        .client
        .post(&endpoint)
        .form(&[
            ("client_id", CLIENT_ID),
            ("scope", SCOPE),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
        ])
        .send()
        .await
        .into_diagnostic()?
        .error_for_status()
        .into_diagnostic()
        .wrap_err("device authorization request failed")?;
    let device: DeviceAuth = resp
        .json()
        .await
        .into_diagnostic()
        .wrap_err("failed to parse device authorization response")?;

    println!(
        "To log in, open:\n  {}\nand enter the code:  {}",
        device.verification_uri, device.user_code
    );
    if let Some(complete) = &device.verification_uri_complete {
        let _ = opener::open(complete);
    }
    println!("Waiting for authorization...");

    let interval = device.interval.unwrap_or(5).max(1);
    let deadline = tokens::now() + device.expires_in.unwrap_or(600);
    loop {
        tokio::time::sleep(Duration::from_secs(interval)).await;
        if tokens::now() > deadline {
            return Err(miette::miette!("device login timed out"));
        }
        let resp = ctx
            .client
            .post(&oidc.token_endpoint)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("client_id", CLIENT_ID),
                ("device_code", device.device_code.as_str()),
                ("code_verifier", verifier.as_str()),
            ])
            .send()
            .await
            .into_diagnostic()?;
        if resp.status().is_success() {
            let token: TokenResponse = resp
                .json()
                .await
                .into_diagnostic()
                .wrap_err("failed to parse token response")?;
            return Ok(Credentials::from_token(
                token,
                &ctx.issuer,
                &ctx.api_url,
                None,
            ));
        }
        let err: DeviceError = resp.json().await.unwrap_or(DeviceError {
            error: "unknown_error".to_string(),
        });
        match err.error.as_str() {
            "authorization_pending" => continue,
            "slow_down" => tokio::time::sleep(Duration::from_secs(interval)).await,
            other => return Err(miette::miette!("device login failed: {other}")),
        }
    }
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 3 <= bytes.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(byte) => {
                    out.push(byte);
                    i += 3;
                }
                Err(_) => {
                    out.push(bytes[i]);
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
