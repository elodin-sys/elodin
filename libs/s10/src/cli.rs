use crate::recipe::Recipe;
use clap::Parser;
use miette::miette;
use std::collections::HashMap;
use stellarator::util::CancelToken;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    // The recipe to build
    recipe: Option<String>,
    #[arg(long)]
    release: bool,
    #[arg(long)]
    watch: bool,
}

impl Args {
    pub async fn run(&self, mut recipes: HashMap<String, Recipe>) -> miette::Result<()> {
        let recipe_name = self.recipe.clone().unwrap_or_else(|| "default".to_string());
        let recipe = recipes
            .remove(&recipe_name)
            .ok_or_else(|| miette!("{} recipe not found", recipe_name))?;
        run_recipe(recipe_name, recipe, self.watch, self.release).await
    }
}

/// Run a recipe, creating an internal CancelToken that is cancelled when the recipe completes.
pub async fn run_recipe(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
) -> miette::Result<()> {
    let cancel_token = CancelToken::new();
    let result =
        run_recipe_with_token(recipe_name, recipe, watch, release, cancel_token.clone()).await;
    cancel_token.cancel();
    result
}

/// Run a recipe with an externally provided CancelToken.
///
/// This allows the caller to control when the recipe is cancelled,
/// enabling graceful termination from outside the recipe's execution context.
///
/// Ctrl+C handling is still set up, so the recipe can be cancelled interactively.
pub async fn run_recipe_with_token(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
    cancel_token: CancelToken,
) -> miette::Result<()> {
    // Set up Ctrl+C handling so the recipe can be cancelled interactively
    let ctrl_c_cancel_token = cancel_token.clone();
    tokio::spawn(async move {
        let _drop = ctrl_c_cancel_token.drop_guard();
        tokio::signal::ctrl_c().await
    });

    let res = if watch {
        recipe.watch(recipe_name, release, cancel_token).await
    } else {
        recipe.run(recipe_name, release, cancel_token).await
    };
    Ok(res?)
}
