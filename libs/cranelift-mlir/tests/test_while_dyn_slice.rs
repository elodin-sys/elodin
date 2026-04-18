mod common;
use common::*;

#[test]
fn test_while_with_dynamic_slice_accumulate() {
    // Simulates the three-body edge_fold pattern:
    // A 2x3 table, while loop iterates 2 times,
    // each iteration slices a row and adds it to an accumulator.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<2x3xf64>) -> tensor<3xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %init = stablehlo.constant dense<[0.0, 0.0, 0.0]> : tensor<3xf64>
    %1:3 = stablehlo.while(%iterArg = %c0, %iterArg_1 = %arg0, %iterArg_2 = %init) : tensor<i64>, tensor<2x3xf64>, tensor<3xf64>
    cond {
      %c2 = stablehlo.constant dense<2> : tensor<i64>
      %cmp = stablehlo.compare  LT, %iterArg, %c2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cmp : tensor<i1>
    } do {
      %c_zero = stablehlo.constant dense<0> : tensor<i64>
      %row = stablehlo.dynamic_slice %iterArg_1, %iterArg, %c_zero, sizes = [1, 3] : (tensor<2x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
      %flat = stablehlo.reshape %row : (tensor<1x3xf64>) -> tensor<3xf64>
      %sum = stablehlo.add %iterArg_2, %flat : tensor<3xf64>
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %next_i = stablehlo.add %iterArg, %c1 : tensor<i64>
      stablehlo.return %next_i, %iterArg_1, %sum : tensor<i64>, tensor<2x3xf64>, tensor<3xf64>
    }
    return %1#2 : tensor<3xf64>
  }
}
"#;
    // Table: [[1,2,3],[4,5,6]]
    let table = f64_buf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = run_mlir(mlir, &[&table], &[24]);
    let result = read_f64s(&out[0]);
    // Should sum both rows: [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_eq!(result, &[5.0, 7.0, 9.0]);
}
