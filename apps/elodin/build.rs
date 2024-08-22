use std::{io::Write, path::PathBuf};

use ignore::Walk;

fn main() -> anyhow::Result<()> {
    let out_dir = std::env::var("OUT_DIR")?;
    let out_dir = PathBuf::from(out_dir);

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let workspace_root = PathBuf::from(manifest_dir).join("../..");
    let examples_dir = workspace_root.join("examples");

    let ball_example_dir = examples_dir.join("ball");
    let dest_path = out_dir.join("ball.tar.zst");
    bundle_example(ball_example_dir, dest_path)?;

    Ok(())
}

fn bundle_example(example_dir: PathBuf, dest_path: PathBuf) -> anyhow::Result<()> {
    let dest_file = std::fs::File::create(&dest_path)?;
    let zstd_compressor = zstd::stream::Encoder::new(dest_file, 0)?;
    let mut tar_builder = tar::Builder::new(zstd_compressor);

    for entry in Walk::new(&example_dir) {
        let path = entry?.into_path();
        if !path.is_file() {
            continue;
        }

        let relative_path = path.strip_prefix(&example_dir)?;
        tar_builder.append_path_with_name(&path, relative_path)?;
        println!("cargo:rerun-if-changed={}", path.display());
    }
    tar_builder.into_inner()?.finish()?.flush()?;
    Ok(())
}
