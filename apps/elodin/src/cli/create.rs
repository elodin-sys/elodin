use super::Cli;
use std::{io, path::Path};

const BALL_EXAMPLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ball.tar.zst"));
const DRONE_EXAMPLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/drone.tar.zst"));
const CUBE_SAT_EXAMPLE: &[u8] = include_bytes!("../../../../libs/nox-py/examples/cube-sat.py");
const ROCKET_EXAMPLE: &[u8] = include_bytes!("../../../../libs/nox-py/examples/rocket.py");
const THREE_BODY_EXAMPLE: &[u8] = include_bytes!("../../../../libs/nox-py/examples/three-body.py");

#[derive(clap::ValueEnum, Clone, Default)]
enum TemplateType {
    #[default]
    Rocket,
    Drone,
    CubeSat,
    ThreeBody,
    Ball,
}

#[derive(clap::Args, Clone, Default)]
pub struct Args {
    /// Name of the template
    #[arg(short, long)]
    template: TemplateType,
    /// Path where the result will be located
    #[arg(short, long, default_value_t = String::from("."))]
    path: String,
}

impl Cli {
    pub fn create_template(&self, args: &Args) -> io::Result<()> {
        let path = Path::new(&args.path);
        std::fs::create_dir_all(path)?;

        match args.template {
            TemplateType::CubeSat => std::fs::write(path.join("cube-sat.py"), CUBE_SAT_EXAMPLE)?,
            TemplateType::Rocket => std::fs::write(path.join("rocket.py"), ROCKET_EXAMPLE)?,
            TemplateType::ThreeBody => {
                std::fs::write(path.join("three-body.py"), THREE_BODY_EXAMPLE)?
            }
            TemplateType::Ball => Self::write_dir(&path.join("ball"), BALL_EXAMPLE)?,
            TemplateType::Drone => Self::write_dir(&path.join("drone"), DRONE_EXAMPLE)?,
        }

        Ok(())
    }

    fn write_dir(path: &Path, data: &[u8]) -> io::Result<()> {
        let tar = zstd::stream::Decoder::new(data)?;
        let mut archive = tar::Archive::new(tar);
        archive.unpack(path)?;
        Ok(())
    }
}
