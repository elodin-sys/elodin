use clap::Parser;
use miette::miette;
use s10::{
    error::Error,
    recipe::{GroupItem, Recipe},
};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

#[tokio::main(flavor = "current_thread")]
async fn main() -> miette::Result<()> {
    let args = Args::parse();
    let mut config = Config::parse(args.config)?;
    config.resolve_recipes();
    let recipe_name = args.recipe.unwrap_or_else(|| "default".to_string());
    let recipe = config
        .recipes
        .remove(&recipe_name)
        .ok_or_else(|| miette!("{} recipe not found", recipe_name))?;
    let cancel_token = CancellationToken::new();
    let start = async {
        if args.watch {
            recipe
                .watch(recipe_name, args.release, cancel_token.clone())
                .await
        } else {
            recipe
                .run(recipe_name, args.release, cancel_token.clone())
                .await
        }
    };
    tokio::select! {
        res = start => {
            res?;
            println!("recipes has ended; now exiting");
        }
        _ = tokio::signal::ctrl_c() => {
            println!("killing processes");
            cancel_token.cancel();
        }
    }

    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // The recipe to build
    recipe: Option<String>,
    // Path to the config file
    #[arg(short, long)]
    config: Option<String>,
    #[arg(long)]
    release: bool,
    #[arg(long)]
    watch: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct Config {
    #[serde(flatten)]
    recipes: HashMap<String, s10::recipe::Recipe>,
}

impl Config {
    pub fn resolve_recipes(&mut self) {
        let mut resolved_recipes = self.recipes.clone();
        for recipe in &mut resolved_recipes.values_mut() {
            let Recipe::Group(g) = recipe else { continue };
            for item in &mut g.recipes {
                let GroupItem::Ref(name) = item else { continue };
                let Some(recipe) = self.recipes.get(name).cloned() else {
                    continue;
                };
                let name = name.to_string();
                *item = GroupItem::Recipe { recipe, name }
            }
        }
        self.recipes = resolved_recipes;
    }

    pub fn parse(path: Option<String>) -> Result<Self, Error> {
        let config_paths = [
            std::env::var("S10_CONFIG").unwrap_or_else(|_| "/etc/elodin/s10.toml".to_string()),
            "./config.toml".to_string(),
        ];
        for path in path.into_iter().chain(config_paths) {
            let Ok(config) = std::fs::read_to_string(path) else {
                continue;
            };
            return Ok(toml::from_str(&config)?);
        }
        Err(Error::ConfigNotFound)
    }
}
