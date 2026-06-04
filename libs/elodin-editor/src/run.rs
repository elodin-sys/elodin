#[cfg(not(target_os = "windows"))]
use bevy::prelude::*;
#[cfg(not(target_os = "windows"))]
use miette::{Context, IntoDiagnostic, miette};
#[cfg(not(target_os = "windows"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_os = "windows"))]
use stellarator::util::CancelToken;

#[cfg(not(target_os = "windows"))]
pub async fn run_recipe(
    cache_dir: PathBuf,
    path: PathBuf,
    cancel_token: CancelToken,
) -> miette::Result<()> {
    let mut path = if path.is_dir() {
        let toml = path.join("s10.toml");
        let py = path.join("main.py");
        if path.join("s10.toml").exists() {
            toml
        } else if py.exists() {
            py
        } else {
            return Err(miette!(
                "couldn't find a elodin config, please add either a main.py or s10.toml file to the directory"
            ));
        }
    } else {
        path.clone()
    };

    // If python file, generate s10.toml
    if path.extension().and_then(|ext| ext.to_str()) == Some("py") {
        let date_time = hifitime::Epoch::now().unwrap().to_string();
        let out_dir = cache_dir.join(date_time);
        std::fs::create_dir_all(&out_dir).into_diagnostic()?;
        let output = s10::python_command()?
            .arg(path.clone())
            .arg("plan")
            .arg(&out_dir)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .output()
            .into_diagnostic()?;
        if output.status.code() != Some(0) {
            return Err(miette!("error generating s10 plan from python file"));
        }
        path = out_dir.join("s10.toml");
        debug!("Generated s10 plan: {}", path.display());
    }

    // If not a s10 plan file, bail out
    if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
        return Err(miette!(
            "invalid file type, must be a s10 plan or python file"
        ));
    }

    let contents = std::fs::read_to_string(&path).into_diagnostic()?;
    let mut recipe: s10::Recipe = toml::from_str(&contents)
        .into_diagnostic()
        .with_context(|| format!("failed to parse s10 recipe from file: {}", path.display()))?;
    if let Ok(exe) = std::env::current_exe() {
        pin_render_server_recipe(&mut recipe, &exe);
    }

    recipe
        .watch("sim".to_string(), false, cancel_token.clone())
        .await?;
    cancel_token.cancel();
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn pin_render_server_recipe(recipe: &mut s10::Recipe, exe: &Path) {
    let exe = exe.to_string_lossy().into_owned();
    pin_render_server_recipe_inner(recipe, &exe);
}

#[cfg(not(target_os = "windows"))]
fn pin_render_server_recipe_inner(recipe: &mut s10::Recipe, exe: &str) {
    let s10::Recipe::Group(group) = recipe else {
        return;
    };

    if let Some(s10::Recipe::Process(process)) = group.recipes.get_mut("render-server")
        && process
            .process_args
            .args
            .first()
            .is_some_and(|arg| arg == "render-server")
    {
        process.cmd = exe.to_string();
    }

    for recipe in group.recipes.values_mut() {
        pin_render_server_recipe_inner(recipe, exe);
    }
}

#[cfg(all(test, not(target_os = "windows")))]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use s10::{GroupRecipe, ProcessArgs, ProcessRecipe, Recipe, RestartPolicy};

    use super::pin_render_server_recipe;

    #[test]
    fn pins_render_server_recipe_to_current_binary() {
        let mut recipe = Recipe::Group(GroupRecipe {
            refs: vec!["sim".to_string(), "render-server".to_string()],
            recipes: HashMap::from([
                (
                    "sim".to_string(),
                    Recipe::Process(ProcessRecipe {
                        cmd: "python".to_string(),
                        process_args: empty_process_args(),
                        no_watch: false,
                    }),
                ),
                (
                    "render-server".to_string(),
                    Recipe::Process(ProcessRecipe {
                        cmd: "elodin".to_string(),
                        process_args: process_args(["render-server", "--addr", "[::]:2240"]),
                        no_watch: true,
                    }),
                ),
            ]),
        });

        pin_render_server_recipe(&mut recipe, Path::new("/tmp/current-elodin"));

        let Recipe::Group(group) = recipe else {
            panic!("expected group recipe");
        };
        let Some(Recipe::Process(render_server)) = group.recipes.get("render-server") else {
            panic!("expected render-server process");
        };
        assert_eq!(render_server.cmd, "/tmp/current-elodin");
    }

    #[test]
    fn does_not_pin_unrelated_render_server_recipe() {
        let mut recipe = Recipe::Group(GroupRecipe {
            refs: vec!["render-server".to_string()],
            recipes: HashMap::from([(
                "render-server".to_string(),
                Recipe::Process(ProcessRecipe {
                    cmd: "custom-render-server".to_string(),
                    process_args: process_args(["serve"]),
                    no_watch: true,
                }),
            )]),
        });

        pin_render_server_recipe(&mut recipe, Path::new("/tmp/current-elodin"));

        let Recipe::Group(group) = recipe else {
            panic!("expected group recipe");
        };
        let Some(Recipe::Process(render_server)) = group.recipes.get("render-server") else {
            panic!("expected render-server process");
        };
        assert_eq!(render_server.cmd, "custom-render-server");
    }

    fn empty_process_args() -> ProcessArgs {
        process_args([])
    }

    fn process_args(args: impl IntoIterator<Item = &'static str>) -> ProcessArgs {
        ProcessArgs {
            args: args.into_iter().map(str::to_string).collect(),
            cwd: None,
            env: HashMap::new(),
            restart_policy: RestartPolicy::Never,
            fail_on_error: false,
            log_path: None,
        }
    }
}
