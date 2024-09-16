use clap::Parser;
use s10::{error::Error, recipe::Recipe};
use std::collections::HashMap;

#[tokio::main(flavor = "current_thread")]
async fn main() -> miette::Result<()> {
    let _ = tracing_subscriber::fmt::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive("info".parse().expect("invalid filter"))
                .from_env_lossy(),
        )
        .try_init();

    let args = Cli::parse();
    let mut config = Config::parse(args.config)?;
    config.resolve_recipes();
    args.inner.run(config.recipes).await?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    // Path to the config file
    #[arg(short, long)]
    config: Option<String>,
    #[clap(flatten)]
    inner: s10::cli::Args,
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
            for name in &g.refs {
                let Some(recipe) = self.recipes.get(name).cloned() else {
                    continue;
                };
                let name = name.to_string();
                g.recipes.insert(name, recipe);
            }
        }
        self.recipes = resolved_recipes;
    }

    pub fn parse(path: Option<String>) -> Result<Self, Error> {
        let config_paths = [
            std::env::var("S10_CONFIG").unwrap_or_else(|_| "/etc/elodin/s10.toml".to_string()),
            "./s10.toml".to_string(),
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
