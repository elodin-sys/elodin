use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use miette::{IntoDiagnostic, miette};

#[derive(clap::Args, Clone)]
pub struct Args {
    #[command(subcommand)]
    command: TerrainCommand,
}

#[derive(clap::Subcommand, Clone)]
enum TerrainCommand {
    /// Prepare terrain declared in a Python simulation, without starting it.
    Prepare(PrepareArgs),
}

#[derive(clap::Args, Clone)]
struct PrepareArgs {
    /// Python simulation declaring the terrain regions to prepare.
    python: PathBuf,
}

pub fn run(args: Args) -> miette::Result<()> {
    let TerrainCommand::Prepare(args) = args.command;
    if !args.python.is_file() || args.python.extension().and_then(|ext| ext.to_str()) != Some("py")
    {
        return Err(miette!(
            "Expected a Python simulation file (.py) declaring terrain regions"
        ));
    }
    prepare_script(&args.python)
}

fn prepare_script(path: &Path) -> miette::Result<()> {
    let python = super::editor::python_bin()?;
    let work = tempfile::tempdir().into_diagnostic()?;
    let request_path = work.path().join("terrain.json");
    let status = Command::new(&python)
        .arg(path)
        .arg("terrain-prepare")
        .arg(&request_path)
        .env("ELODIN_TERRAIN_PREPARE", "1")
        .status()
        .into_diagnostic()?;
    if !status.success() {
        return Err(miette!(
            "Could not read the simulation's terrain configuration"
        ));
    }
    let mut request: serde_json::Value =
        serde_json::from_slice(&fs::read(&request_path).into_diagnostic()?).into_diagnostic()?;
    let basemap_path = work.path().join("earth-lowres.glb");
    fs::write(&basemap_path, basemap()?).into_diagnostic()?;
    request["basemap"] = serde_json::json!(basemap_path);
    fs::write(
        &request_path,
        serde_json::to_vec(&request).into_diagnostic()?,
    )
    .into_diagnostic()?;
    let status = Command::new(&python)
        .args(["-m", "elodin.terrain_prepare"])
        .arg(&request_path)
        .env(
            "ELODIN_CESIUM_TOKEN",
            elodin_editor::plugins::cesium::ion_access_token(),
        )
        .status()
        .into_diagnostic()?;
    if !status.success() {
        return Err(miette!(
            "Terrain preparation failed; the simulation was not started"
        ));
    }
    Ok(())
}

fn basemap() -> miette::Result<Vec<u8>> {
    let source = include_bytes!("../../../../assets/earth.glb");
    if source.get(..4) != Some(b"glTF") {
        return Err(miette!(
            "Earth asset is missing; fetch Git LFS assets before building"
        ));
    }
    let json_len = u32::from_le_bytes(source[12..16].try_into().into_diagnostic()?) as usize;
    let mut document: serde_json::Value =
        serde_json::from_slice(&source[20..20 + json_len]).into_diagnostic()?;
    // The bundled Earth is ECEF-sized; leave a 12 km gap below detailed terrain.
    // Exclude its unrelated small origin cube from the background scene.
    document["scenes"][0]["nodes"] = serde_json::json!([1]);
    for value in document["nodes"][1]["scale"]
        .as_array_mut()
        .ok_or_else(|| miette!("Unexpected Earth asset layout"))?
    {
        *value = serde_json::json!(
            value
                .as_f64()
                .ok_or_else(|| miette!("Invalid Earth scale"))?
                * 0.998
        );
    }
    let mut json = serde_json::to_vec(&document).into_diagnostic()?;
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let rest = &source[20 + json_len..];
    let mut output = Vec::with_capacity(20 + json.len() + rest.len());
    output.extend_from_slice(b"glTF");
    output.extend_from_slice(&2_u32.to_le_bytes());
    output.extend_from_slice(&((20 + json.len() + rest.len()) as u32).to_le_bytes());
    output.extend_from_slice(&(json.len() as u32).to_le_bytes());
    output.extend_from_slice(&0x4e4f534a_u32.to_le_bytes());
    output.extend_from_slice(&json);
    output.extend_from_slice(rest);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basemap_is_self_contained_and_below_the_reference_ellipsoid() {
        let bytes = basemap().unwrap();
        let size = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let document: serde_json::Value = serde_json::from_slice(&bytes[20..20 + size]).unwrap();
        assert_eq!(document["scenes"][0]["nodes"], serde_json::json!([1]));
        assert!(document["nodes"][1]["scale"][0].as_f64().unwrap() * 500.0 < 6_370_000.0);
        assert!(
            document["images"]
                .as_array()
                .unwrap()
                .iter()
                .all(|i| i.get("uri").is_none())
        );
    }
}
