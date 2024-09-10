use miette::Diagnostic;
use thiserror::Error;
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
    MustSelectPackage,
    #[error("package metadata not found {0}")]
    PackageMetadataNotFound(String),
    #[error("unreseolved recipe {0}")]
    UnresolvedRecipe(String),
}
