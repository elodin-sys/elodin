//! `elodin logout`: end the Keycloak session and delete the local credentials.

use miette::{Context, IntoDiagnostic};

use super::tokens::Credentials;
use super::{AuthCtx, CLIENT_ID, discover_at};

pub async fn run(ctx: &AuthCtx) -> miette::Result<()> {
    // Best-effort session revocation at the issuer the token came from.
    if let Some(creds) = Credentials::load(&ctx.creds_path)?
        && let Some(refresh_token) = creds.refresh_token.clone()
        && let Ok(oidc) = discover_at(&ctx.client, &creds.issuer).await
        && let Some(end_session) = oidc.end_session_endpoint
    {
        let _ = ctx
            .client
            .post(&end_session)
            .form(&[
                ("client_id", CLIENT_ID),
                ("refresh_token", refresh_token.as_str()),
            ])
            .send()
            .await;
    }

    match std::fs::remove_file(&ctx.creds_path) {
        Ok(()) => println!("Logged out; removed {}", ctx.creds_path.display()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => println!("Not logged in."),
        Err(e) => {
            return Err(e)
                .into_diagnostic()
                .wrap_err("failed to remove credentials");
        }
    }
    Ok(())
}
