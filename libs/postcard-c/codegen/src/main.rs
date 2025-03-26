use std::path::PathBuf;

use clap::Parser;
use miette::IntoDiagnostic;
use postcard_schema::schema::owned::OwnedNamedType;

#[derive(Parser)]
struct Args {
    path: PathBuf,
}
fn main() -> miette::Result<()> {
    let args = Args::parse();
    let contents = std::fs::read_to_string(args.path).into_diagnostic()?;
    let named_ty: OwnedNamedType = ron::from_str(&contents).into_diagnostic()?;
    println!("{}", postcard_c_codegen::generate_cpp(&named_ty)?);
    Ok(())
}
