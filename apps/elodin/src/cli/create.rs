use super::Cli;
use std::path::Path;

#[derive(clap::ValueEnum, Clone, Default)]
enum TemplateType {
    #[default]
    Rocket,
    CubeSat,
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
    pub fn create_template(&self, args: &Args) -> anyhow::Result<()> {
        let (template_filename, template_code) = match args.template {
            TemplateType::CubeSat => (
                "cube-sat.py",
                include_str!("../../../../libs/nox-py/examples/cube-sat.py"),
            ),
            TemplateType::Rocket => (
                "rocket.py",
                include_str!("../../../../libs/nox-py/examples/rocket.py"),
            ),
        };

        let path = Path::new(&args.path);
        std::fs::create_dir_all(path)?;
        std::fs::write(path.join(template_filename), template_code.as_bytes())?;

        Ok(())
    }
}
