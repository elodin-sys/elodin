//! `elodin signup`: create an account on the Elodin Cloud registration page, then log in.

use super::{AuthCtx, LoginArgs, SignupArgs, login};

pub async fn run(ctx: &AuthCtx, args: &SignupArgs) -> miette::Result<()> {
    println!(
        "Create your account in the browser. Once you verify your email and choose a password, \
         the CLI finishes logging in."
    );
    let login_args = LoginArgs {
        device: false,
        no_browser: args.no_browser,
    };
    login::sign_in(ctx, &login_args, Some("create")).await
}
