//! `elodin whoami`: call the auth-protected `/v1/me` route with the stored token.

use miette::{Context, IntoDiagnostic};

use super::{AuthCtx, WhoamiArgs, tokens};

pub async fn run(ctx: &AuthCtx, args: &WhoamiArgs) -> miette::Result<()> {
    let creds = tokens::ensure_valid(ctx).await?;
    let url = format!("{}/v1/me", creds.api_url.trim_end_matches('/'));
    let resp = ctx
        .client
        .get(&url)
        .bearer_auth(&creds.access_token)
        .send()
        .await
        .into_diagnostic()
        .wrap_err("request to the Elodin Cloud API failed")?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err(miette::miette!(
            "the API rejected the token (401); run `elodin login` again"
        ));
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(miette::miette!("API error ({status}): {body}"));
    }

    let value: serde_json::Value = resp
        .json()
        .await
        .into_diagnostic()
        .wrap_err("failed to parse the API response")?;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value).into_diagnostic()?
        );
        return Ok(());
    }

    let field = |key: &str| {
        value
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("-")
            .to_string()
    };
    println!("Logged in as {}", field("preferred_username"));
    println!("  subject:  {}", field("sub"));
    println!("  email:    {}", field("email"));
    println!("  org:      {}", field("org_id"));
    println!("  org role: {}", field("org_role"));
    println!("  issuer:   {}", field("issuer"));
    println!("  api:      {}", creds.api_url);
    Ok(())
}
