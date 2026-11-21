//! Compile one StableHLO module to a relocatable object on the current target.
//!
//!     cargo run -p cranelift-mlir --example compile-object -- \
//!       INPUT.mlir OUTPUT.o PREFIX WORKERS native|generic

use std::env;
use std::fs;
use std::path::Path;

use cranelift_mlir::const_fold::fold_module;
use cranelift_mlir::lower::{compile_module_to_object, object_target};
use cranelift_mlir::parser::parse_module;

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let [_, input, output, prefix, workers, cpu] = args.as_slice() else {
        return Err(
            "usage: compile-object INPUT.mlir OUTPUT.o PREFIX WORKERS native|generic".into(),
        );
    };
    let workers = workers
        .parse::<usize>()
        .map_err(|error| format!("invalid worker count: {error}"))?;
    let mlir = fs::read_to_string(input).map_err(|error| format!("read {input}: {error}"))?;
    let mut module = parse_module(&mlir).map_err(|error| format!("parse {input}: {error}"))?;
    fold_module(&mut module);
    let object = compile_module_to_object(&module, prefix, workers, cpu)?;
    fs::write(Path::new(output), object).map_err(|error| format!("write {output}: {error}"))?;
    let (triple, features) = object_target(cpu)?;
    let target = serde_json::json!({
        "triple": triple,
        "cpu": cpu,
        "features": features.into_iter().collect::<std::collections::BTreeMap<_, _>>(),
    });
    let target_path = format!("{output}.target.json");
    fs::write(
        &target_path,
        serde_json::to_vec_pretty(&target).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {target_path}: {error}"))
}
