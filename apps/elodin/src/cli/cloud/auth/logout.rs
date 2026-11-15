//! `elodin logout`: end the Keycloak session and forget this issuer's local credentials.

use super::tokens::Credentials;
use super::{AuthCtx, CLIENT_ID, discover_at};

pub async fn run(ctx: &AuthCtx) -> miette::Result<()> {
    // Best-effort session revocation at the issuer the token came from.
    if let Some(creds) = Credentials::load(&ctx.creds_path, &ctx.issuer)?
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

    if Credentials::remove(&ctx.creds_path, &ctx.issuer)? {
        println!("Logged out of {}.", ctx.issuer);
    } else {
        println!("Not logged in to {}.", ctx.issuer);
    }
    Ok(())
}
