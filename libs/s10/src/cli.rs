use crate::recipe::Recipe;
use clap::Parser;
use miette::miette;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

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

pub async fn run_recipe(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
) -> miette::Result<()> {
    let cancel_token = CancellationToken::new();
    let ctrl_c_cancel_token = cancel_token.clone();
    tokio::spawn(async move {
        let _drop = ctrl_c_cancel_token.drop_guard(); // binding needs to be named to ensure drop is called at end of scope
        tokio::signal::ctrl_c().await
    });
    let res = if watch {
        recipe
            .watch(recipe_name, release, cancel_token.clone())
            .await
    } else {
        recipe.run(recipe_name, release, cancel_token.clone()).await
    };
    cancel_token.cancel();
    Ok(res?)
}
