use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
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
    let toktx = env::var("TOKTX").unwrap_or_else(|_| "toktx".into());

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

fn pack_one(skybox_dir: &Path, toktx: &str, name: &str) -> Result<(), String> {
    let equirect_path = skybox_dir.join(format!("{name}.equirect.png"));
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
    let stacked =
        cubemap_convert::equirect_to_stacked_cubemap(&equirect, BUNDLED_CUBEMAP_FACE_SIZE);

    let temp = env::temp_dir().join(format!("elodin-skybox-pack-{name}"));
    if temp.exists() {
        fs::remove_dir_all(&temp).map_err(|error| error.to_string())?;
    }
    fs::create_dir_all(&temp).map_err(|error| error.to_string())?;

    let mut face_paths = Vec::with_capacity(6);
    for face in 0..6 {
        let face_path = temp.join(format!("face{face}.png"));
        cubemap_convert::stacked_face(&stacked, face, BUNDLED_CUBEMAP_FACE_SIZE)
            .save(&face_path)
            .map_err(|error| error.to_string())?;
        face_paths.push(face_path);
    }

    let output = skybox_dir.join(format!("{name}.cubemap.ktx2"));
    let mut command = Command::new(toktx);
    command
        .arg("--t2")
        .arg("--zcmp")
        .arg("--genmipmap")
        .arg("--cubemap")
        .arg("--assign_oetf")
        .arg("srgb")
        .arg(&output);
    for face_path in &face_paths {
        command.arg(face_path);
    }
    let status = command
        .status()
        .map_err(|error| format!("failed to run `{toktx}`: {error}"))?;
    let _ = fs::remove_dir_all(&temp);
    if !status.success() {
        return Err(format!("toktx failed for `{name}` (exit {status})"));
    }
    Ok(())
}
