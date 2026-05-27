use std::{
    env,
    path::{Path, PathBuf},
};

use bevy_ai_skybox::cubemap_convert::{self, BUNDLED_CUBEMAP_FACE_SIZE};
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
    let toktx = cubemap_convert::resolve_toktx_executable();

    let names = parse_names()?;
    for name in names {
        pack_one(&skybox_dir, &toktx, &name)?;
        println!("packed {name}");
    }
    Ok(())
}

fn parse_names() -> Result<Vec<String>, String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() {
        return Ok(vec![
            "seaport".into(),
            "coastal_beach".into(),
            "grand_canyon".into(),
        ]);
    }
    Ok(args)
}

fn pack_one(skybox_dir: &Path, toktx: &Path, name: &str) -> Result<(), String> {
    let equirect_path = skybox_dir.join(cubemap_convert::equirect_manifest_filename(name));
    if !equirect_path.is_file() {
        return Err(format!(
            "missing equirect source: {}",
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
