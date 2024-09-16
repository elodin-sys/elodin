use miette::Diagnostic;
use thiserror::Error;

use crate::ProcessError;
#[derive(Error, Debug, Diagnostic)]
pub enum Error {
    #[error("io error:  {0}")]
    Io(#[from] std::io::Error),
    #[error("config not found")]
    ConfigNotFound,
    #[error("toml parsing failed")]
    Toml(#[from] toml::de::Error),
    #[error("error building {0}")]
    PackageBuild(String),
    #[error("cargo metadata error {0}")]
    CargoMetadata(#[from] cargo_metadata::Error),
    #[error("a cargo workspace was specified with no package selected")]
    #[diagnostic(help = "please specify either a `package` or a `bin` in the recipe")]
    MustSelectPackage,
    #[error("package metadata not found {0}")]
    PackageMetadataNotFound(String),
    #[error("unreseolved recipe {0}")]
    UnresolvedRecipe(String),
    #[error("failed to build sim {0:?}")]
    SimBuildFailed(Option<i32>),
    #[error("nox ecs {0}")]
    NoxEcs(#[from] nox_ecs::Error),
    #[error("join error")]
    JoinError,
    #[error(transparent)]
    #[diagnostic(transparent)]
    Process(#[from] ProcessError),
    #[error(transparent)]
    Ignore(#[from] ignore::Error),
}
