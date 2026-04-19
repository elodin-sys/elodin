use cranelift_mlir::lower::compile_module;
use cranelift_mlir::parser::parse_module;

fn smoke_parse_compile(mlir: &str) {
    let module = parse_module(mlir).expect("failed to parse");
    assert!(module.main_func().is_some(), "no main function found");
    assert!(module.functions.len() > 1);
    let compiled = compile_module(&module).expect("failed to compile");
    assert!(!compiled.get_main_fn().is_null());
}

#[test]
fn ball_e2e() {
    let mlir = include_str!("../testdata/ball.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    let main = module.main_func().unwrap();
    assert!(main.is_public);
    assert_eq!(main.params.len(), 9);
    assert_eq!(main.result_types.len(), 9);
    assert!(!main.body.is_empty());
    let compiled = compile_module(&module).expect("failed to compile");
    assert!(!compiled.get_main_fn().is_null());
}

#[test]
fn drone_e2e() {
    smoke_parse_compile(include_str!("../testdata/drone.stablehlo.mlir"));
}

#[test]
fn rocket_e2e() {
    smoke_parse_compile(include_str!("../testdata/rocket.stablehlo.mlir"));
}

#[test]
fn linalg_e2e() {
    let mlir = include_str!("../testdata/linalg.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse");
    let main = module.main_func().unwrap();
    assert!(main.is_public);
    assert!(!main.body.is_empty());
    let compiled = compile_module(&module).expect("failed to compile");
    assert!(!compiled.get_main_fn().is_null());
}
