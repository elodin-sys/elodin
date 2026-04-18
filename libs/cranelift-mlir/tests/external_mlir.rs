/// Integration test for validating customer-provided StableHLO MLIR dumps.
///
/// Reads `$ELODIN_CRANELIFT_DEBUG_DIR/stablehlo.mlir` — the same
/// file a live run writes there. Run with:
///   ELODIN_CRANELIFT_DEBUG_DIR=/path/to/dump \
///     cargo test -p cranelift-mlir --test external_mlir --release -- --ignored --nocapture
#[test]
#[ignore]
fn test_parse_external_mlir() {
    let path = debug_dir_mlir_path();
    let mlir = std::fs::read_to_string(&path).expect("read MLIR");
    let module = cranelift_mlir::parser::parse_module(&mlir).expect("parse");
    assert!(
        !module.functions.is_empty(),
        "parsed module has no functions"
    );
    let report = cranelift_mlir::const_fold::measure(&module);
    eprintln!(
        "[external_mlir] parsed {} functions ({} instr) from {}",
        module.functions.len(),
        report.total_instructions_before,
        path
    );
}

#[test]
#[ignore]
fn test_compile_external_mlir() {
    let path = debug_dir_mlir_path();
    let mlir = std::fs::read_to_string(&path).expect("read MLIR");
    let mut module = cranelift_mlir::parser::parse_module(&mlir).expect("parse");
    assert!(
        !module.functions.is_empty(),
        "parsed module has no functions"
    );
    cranelift_mlir::const_fold::fold_module(&mut module);
    let compiled = cranelift_mlir::lower::compile_module(&module).expect("compile");
    eprintln!(
        "[external_mlir] compiled {} functions from {}",
        module.functions.len(),
        path
    );
    drop(compiled);
}

fn debug_dir_mlir_path() -> String {
    let dir = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
        .expect("set ELODIN_CRANELIFT_DEBUG_DIR to a directory containing stablehlo.mlir");
    format!("{dir}/stablehlo.mlir")
}
