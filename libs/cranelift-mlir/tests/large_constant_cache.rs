mod common;

use common::{read_f64s, run_mlir_mem};
use cranelift_mlir::ir::{ConstantValue, Instruction};
use cranelift_mlir::parser::parse_module;

const N: usize = 131_072;
const INDEX: usize = 12_345;

fn large_constant_mlir() -> (String, f64) {
    let mut hex = String::with_capacity(N * 16);
    for i in 0..N {
        let value = i as f64 + 0.25;
        for byte in value.to_le_bytes() {
            hex.push_str(&format!("{byte:02x}"));
        }
    }
    let expected = INDEX as f64 + 0.25;
    let mlir = format!(
        r#"
module @module {{
  func.func public @main() -> tensor<1xf64> {{
    %0 = stablehlo.constant dense<"0x{hex}"> : tensor<{N}xf64>
    %1 = stablehlo.slice %0 [{INDEX}:{limit}] : (tensor<{N}xf64>) -> tensor<1xf64>
    return %1 : tensor<1xf64>
  }}
}}
"#,
        limit = INDEX + 1
    );
    (mlir, expected)
}

#[test]
fn large_hex_constant_parses_as_external_cache_entry() {
    let (mlir, _) = large_constant_mlir();
    let module = parse_module(&mlir).expect("parse");
    let main = module.main_func().expect("main");
    let constant = main
        .body
        .iter()
        .find_map(|ir| match &ir.instr {
            Instruction::Constant { value } => Some(value),
            _ => None,
        })
        .expect("constant");
    match constant {
        ConstantValue::DenseExternal { byte_len, data, .. } => {
            assert_eq!(*byte_len, N * std::mem::size_of::<f64>());
            assert_eq!(data.byte_len, *byte_len);
            assert!(data.path.exists());
        }
        other => panic!("expected DenseExternal, got {other:?}"),
    }
}

#[test]
fn large_external_constant_slice_matches_expected_value() {
    let (mlir, expected) = large_constant_mlir();
    let out = run_mlir_mem(&mlir, &[], &[std::mem::size_of::<f64>()]);
    let value = read_f64s(&out[0])[0];
    assert_eq!(value.to_bits(), expected.to_bits());
}
