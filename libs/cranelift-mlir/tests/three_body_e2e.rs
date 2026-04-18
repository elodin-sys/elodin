use cranelift_mlir::parser::parse_module;

#[test]
fn parse_three_body_mlir() {
    let mlir = include_str!("../testdata/three-body.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse three-body MLIR");

    assert!(
        module.main_func().is_some(),
        "module should have a main function"
    );

    let main = module.main_func().unwrap();
    assert!(main.is_public);
    assert_eq!(main.params.len(), 7, "three-body main has 7 inputs");
    assert_eq!(main.result_types.len(), 7, "three-body main has 7 outputs");
    assert!(!main.body.is_empty());

    assert_eq!(
        module.functions.len(),
        4,
        "main + inner + closed_call + norm"
    );

    for f in &module.functions {
        for ir in &f.body {
            if let cranelift_mlir::ir::Instruction::While {
                loop_body,
                cond_body,
                ..
            } = &ir.instr
            {
                let body_dyn_slice = loop_body.iter().any(|ir| {
                    matches!(
                        ir.instr,
                        cranelift_mlir::ir::Instruction::DynamicSlice { .. }
                    )
                });
                let body_dyn_update = loop_body.iter().any(|ir| {
                    matches!(
                        ir.instr,
                        cranelift_mlir::ir::Instruction::DynamicUpdateSlice { .. }
                    )
                });
                let body_call = loop_body
                    .iter()
                    .any(|ir| matches!(ir.instr, cranelift_mlir::ir::Instruction::Call { .. }));
                eprintln!(
                    "  {}: while loop body has {} instrs (cond={} instrs), dyn_slice={body_dyn_slice}, dyn_update={body_dyn_update}, call={body_call}",
                    f.name,
                    loop_body.len(),
                    cond_body.len(),
                );
            }
        }
    }
}

#[test]
fn compile_three_body_mlir() {
    let mlir = include_str!("../testdata/three-body.stablehlo.mlir");
    let module = parse_module(mlir).expect("failed to parse three-body MLIR");
    let compiled =
        cranelift_mlir::lower::compile_module(&module).expect("failed to compile three-body MLIR");
    let fn_ptr = compiled.get_main_fn();
    assert!(!fn_ptr.is_null());
}
