//! `elodin signup`: self-serve account + org creation via the public API.

use std::io::Write;

use miette::{Context, IntoDiagnostic};

use super::{AuthCtx, SignupArgs};

pub async fn run(ctx: &AuthCtx, args: &SignupArgs) -> miette::Result<()> {
    let email = match &args.email {
        Some(e) => e.clone(),
        None => prompt("Email: ")?,
    };
    let password = match &args.password {
        Some(p) => p.clone(),
        None => prompt("Password (min 8 chars): ")?,
    };

    let url = format!("{}/v1/signup", ctx.api_url.trim_end_matches('/'));
    let mut body = serde_json::json!({ "email": email, "password": password });
    if let Some(org) = &args.org_name {
        body["org_name"] = serde_json::Value::String(org.clone());
    }

    let resp = ctx
        .client
        .post(&url)
        .json(&body)
        .send()
        .await
        .into_diagnostic()
        .wrap_err("signup request to the Elodin Cloud API failed")?;

    let status = resp.status();
    let value: serde_json::Value = resp.json().await.unwrap_or_default();
    if !status.is_success() {
        let detail = value
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("signup failed");
        return Err(miette::miette!("signup failed ({status}): {detail}"));
    }

    println!("Account created for {email}.");
    if let Some(org) = value.get("org_name").and_then(|v| v.as_str()) {
        println!("  organization: {org}");
    }
    println!("\nNext: run `elodin login` to authenticate the CLI.");
    Ok(())
}

fn prompt(label: &str) -> miette::Result<String> {
    print!("{label}");
    std::io::stdout().flush().into_diagnostic()?;
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .into_diagnostic()
        .wrap_err("failed to read input")?;
    Ok(input.trim().to_string())
}
