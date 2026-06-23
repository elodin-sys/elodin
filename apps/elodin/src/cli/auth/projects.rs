//! `elodin projects`: create/list org-scoped projects via the authenticated API.

use miette::{Context, IntoDiagnostic};

use super::tokens::Credentials;
use super::{AuthCtx, ProjectsArgs, ProjectsCommand, tokens};

pub async fn run(ctx: &AuthCtx, args: &ProjectsArgs) -> miette::Result<()> {
    let creds = tokens::ensure_valid(ctx).await?;
    match &args.command {
        ProjectsCommand::Create { name, json } => create(ctx, &creds, name, *json).await,
        ProjectsCommand::List { json } => list(ctx, &creds, *json).await,
    }
}

async fn create(ctx: &AuthCtx, creds: &Credentials, name: &str, json: bool) -> miette::Result<()> {
    let url = format!("{}/v1/projects", creds.api_url.trim_end_matches('/'));
    let resp = ctx
        .client
        .post(&url)
        .bearer_auth(&creds.access_token)
        .json(&serde_json::json!({ "name": name }))
        .send()
        .await
        .into_diagnostic()
        .wrap_err("request to the Elodin Cloud API failed")?;
    let value = parse(resp).await?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value).into_diagnostic()?
        );
    } else {
        let id = value.get("id").and_then(|v| v.as_str()).unwrap_or("-");
        println!("Created project '{name}' ({id})");
    }
    Ok(())
}

async fn list(ctx: &AuthCtx, creds: &Credentials, json: bool) -> miette::Result<()> {
    let url = format!("{}/v1/projects", creds.api_url.trim_end_matches('/'));
    let resp = ctx
        .client
        .get(&url)
        .bearer_auth(&creds.access_token)
        .send()
        .await
        .into_diagnostic()
        .wrap_err("request to the Elodin Cloud API failed")?;
    let value = parse(resp).await?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&value).into_diagnostic()?
        );
        return Ok(());
    }
    let projects = value
        .get("projects")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if projects.is_empty() {
        println!("No projects yet. Create one with `elodin projects create <name>`.");
        return Ok(());
    }
    println!("Projects:");
    for p in projects {
        let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("-");
        let id = p.get("id").and_then(|v| v.as_str()).unwrap_or("-");
        println!("  {name}  ({id})");
    }
    Ok(())
}

/// Map non-2xx responses to friendly errors; otherwise parse the JSON body.
async fn parse(resp: reqwest::Response) -> miette::Result<serde_json::Value> {
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(miette::miette!(
            "the API rejected the token (401); run `elodin login` again"
        ));
    }
    if status == reqwest::StatusCode::FORBIDDEN {
        return Err(miette::miette!(
            "your account has no organization scope; run `elodin signup` first"
        ));
    }
    let value: serde_json::Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        let detail = value
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("request failed");
        return Err(miette::miette!("API error ({status}): {detail}"));
    }
    Ok(value)
}
