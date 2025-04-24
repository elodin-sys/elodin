use bevy::prelude::*;
use miette::{Context, IntoDiagnostic, miette};
use std::path::PathBuf;
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
    let recipe: s10::Recipe = toml::from_str(&contents)
        .into_diagnostic()
        .with_context(|| format!("failed to parse s10 recipe from file: {}", path.display()))?;

    recipe
        .watch("sim".to_string(), false, cancel_token.clone())
        .await?;
    cancel_token.cancel();
    Ok(())
}
