mod common;

use common::{f64_buf, run_mlir_workers};
use cranelift_mlir::parser::parse_module;

/// Four row chunks through a scalar-ABI callee and two whole-tensor calls through a pointer-ABI
/// one, none reading another's result: the shape `jax.vmap` per entity chunk lowers to.
const CHUNKED: &str = r#"
module @module {
  func.func public @main(%arg0: tensor<8x16xf64>) -> (tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>) {
    %0 = stablehlo.slice %arg0 [0:2, 0:16] : (tensor<8x16xf64>) -> tensor<2x16xf64>
    %1 = call @chunk(%0) : (tensor<2x16xf64>) -> tensor<2x16xf64>
    %2 = stablehlo.slice %arg0 [2:4, 0:16] : (tensor<8x16xf64>) -> tensor<2x16xf64>
    %3 = call @chunk(%2) : (tensor<2x16xf64>) -> tensor<2x16xf64>
    %4 = stablehlo.slice %arg0 [4:6, 0:16] : (tensor<8x16xf64>) -> tensor<2x16xf64>
    %5 = call @chunk(%4) : (tensor<2x16xf64>) -> tensor<2x16xf64>
    %6 = stablehlo.slice %arg0 [6:8, 0:16] : (tensor<8x16xf64>) -> tensor<2x16xf64>
    %7 = call @chunk(%6) : (tensor<2x16xf64>) -> tensor<2x16xf64>
    %8 = stablehlo.concatenate %1, %3, %5, %7, dim = 0 : (tensor<2x16xf64>, tensor<2x16xf64>, tensor<2x16xf64>, tensor<2x16xf64>) -> tensor<8x16xf64>
    %9 = call @whole(%arg0) : (tensor<8x16xf64>) -> tensor<8x16xf64>
    %10 = stablehlo.negate %arg0 : tensor<8x16xf64>
    %11 = call @whole(%10) : (tensor<8x16xf64>) -> tensor<8x16xf64>
    return %8, %9, %11 : tensor<8x16xf64>, tensor<8x16xf64>, tensor<8x16xf64>
  }
  func.func private @chunk(%arg0: tensor<2x16xf64>) -> tensor<2x16xf64> {
    %0 = stablehlo.sine %arg0 : tensor<2x16xf64>
    %1 = stablehlo.multiply %0, %arg0 : tensor<2x16xf64>
    return %1 : tensor<2x16xf64>
  }
  func.func private @whole(%arg0: tensor<8x16xf64>) -> tensor<8x16xf64> {
    %0 = stablehlo.cosine %arg0 : tensor<8x16xf64>
    %1 = stablehlo.add %0, %arg0 : tensor<8x16xf64>
    return %1 : tensor<8x16xf64>
  }
}
"#;

#[test]
fn independent_calls_form_one_group() {
    let mut module = parse_module(CHUNKED).unwrap();
    let groups = cranelift_mlir::parallel::group_module_calls(&mut module, |f| f.name == "main");
    assert_eq!(groups["main"].len(), 1);
    assert_eq!(groups["main"][0].len(), 6);
}

#[test]
fn workers_reproduce_the_serial_bits() {
    let input = f64_buf(&(0..128).map(|i| i as f64 * 0.37 - 20.0).collect::<Vec<_>>());
    let sizes = [128 * 8; 3];
    let serial = run_mlir_workers(CHUNKED, &[&input], &sizes, 1);
    for _ in 0..20 {
        assert_eq!(run_mlir_workers(CHUNKED, &[&input], &sizes, 4), serial);
    }
}
