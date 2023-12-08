use clap::Parser;
use starlark::environment::{Globals, Module};
use starlark::eval::Evaluator;
use starlark::syntax::{AstModule, Dialect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let content = std::fs::read_to_string(&args.path)?;
    let ast = AstModule::parse(args.path.to_str().unwrap(), content, &Dialect::Extended)?;
    let globals = Globals::standard();
    let module = Module::new();
    let mut eval = Evaluator::new(&module);
    let quad = eval.eval_module(ast, &globals)?;
    println!("{}", quad.to_json().unwrap());
    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    path: std::path::PathBuf,
}
