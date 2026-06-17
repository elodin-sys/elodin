use std::{
    env,
    path::{Path, PathBuf},
};

use bevy_ai_skybox::{
    SkyboxManifest,
    cubemap_convert::{self, BUNDLED_CUBEMAP_FACE_SIZE},
};
use image::ImageReader;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let skybox_dir = repo_root.join("assets/skyboxes");
    let manifest = SkyboxManifest::read_or_create(&skybox_dir.join("manifest.ron"))
        .map_err(|error| error.to_string())?;
    let toktx = cubemap_convert::resolve_toktx_executable();

    let names = parse_names(&manifest)?;
    if names.is_empty() {
        return Err(
            "no skybox entries with equirect sources found in manifest; \
             pass skybox names on the command line (e.g. pack_skybox_assets desert_night)"
                .into(),
        );
    }
    for name in names {
        pack_one(&skybox_dir, &toktx, &manifest, &name)?;
        println!("packed {name}");
    }
    Ok(())
}

fn parse_names(manifest: &SkyboxManifest) -> Result<Vec<String>, String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() {
        return Ok(manifest
            .entries
            .iter()
            .filter(|entry| entry.equirect_file.is_some())
            .map(|entry| entry.name.clone())
            .collect());
    }
    Ok(args)
}

fn pack_one(
    skybox_dir: &Path,
    toktx: &Path,
    manifest: &SkyboxManifest,
    name: &str,
) -> Result<(), String> {
    let equirect_file = manifest
        .get(name)
        .and_then(|entry| entry.equirect_file.as_deref())
        .map(str::to_string)
        .unwrap_or_else(|| cubemap_convert::equirect_manifest_filename(name));
    let equirect_path = skybox_dir.join(&equirect_file);
    if !equirect_path.is_file() {
        return Err(format!(
            "missing equirect source for `{name}`: {}",
            equirect_path.display()
        ));
    }

    let equirect = ImageReader::open(&equirect_path)
        .map_err(|error| error.to_string())?
        .decode()
        .map_err(|error| error.to_string())?
        .to_rgba8();

    let output = skybox_dir.join(cubemap_convert::cubemap_ktx2_filename(name));
    cubemap_convert::write_cubemap_ktx2(&equirect, BUNDLED_CUBEMAP_FACE_SIZE, &output, toktx)
}
