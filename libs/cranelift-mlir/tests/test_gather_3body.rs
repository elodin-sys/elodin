mod common;
use common::*;

#[test]
fn test_gather_1x1_index_from_constant() {
    // Exact three-body pattern: broadcast a scalar constant to 1x1xui32,
    // then gather one row from a 3x7 table.
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x7xf64>) -> tensor<1x7xf64> {
    %c = stablehlo.constant dense<1> : tensor<1xui32>
    %idx = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %0 = "stablehlo.gather"(%arg0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    return %0 : tensor<1x7xf64>
  }
}
"#;
    // 3x7 table: row0=[0..7), row1=[7..14), row2=[14..21)
    let table: Vec<f64> = (0..21).map(|i| i as f64).collect();
    let in0 = f64_buf(&table);
    let out = run_mlir(mlir, &[&in0], &[56]);
    let result = read_f64s(&out[0]);
    // Should select row 1 = [7,8,9,10,11,12,13]
    let expected: Vec<f64> = (7..14).map(|i| i as f64).collect();
    assert_eq!(result, expected, "gather should select row 1");
}

#[test]
fn test_gather_2x1_index_from_constant() {
    // Two-row gather from the three-body pattern
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x7xf64>) -> tensor<2x7xf64> {
    %c = stablehlo.constant dense<[2, 0]> : tensor<2xui32>
    %idx = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<2xui32>) -> tensor<2x1xui32>
    %0 = "stablehlo.gather"(%arg0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<2x1xui32>) -> tensor<2x7xf64>
    return %0 : tensor<2x7xf64>
  }
}
"#;
    let table: Vec<f64> = (0..21).map(|i| i as f64).collect();
    let in0 = f64_buf(&table);
    let out = run_mlir(mlir, &[&in0], &[112]);
    let result = read_f64s(&out[0]);
    // Should select row 2 = [14..21), then row 0 = [0..7)
    let mut expected: Vec<f64> = (14..21).map(|i| i as f64).collect();
    expected.extend((0..7).map(|i| i as f64));
    assert_eq!(result, expected);
}

#[test]
fn test_three_body_inner_fragment() {
    // Test the pattern used in @inner: gather a row, reshape to 7, reshape back to 1x7
    let mlir = r#"
module @module {
  func.func public @main(%arg0: tensor<3x7xf64>) -> tensor<7xf64> {
    %c = stablehlo.constant dense<2> : tensor<1xui32>
    %idx = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<1xui32>) -> tensor<1x1xui32>
    %0 = "stablehlo.gather"(%arg0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 7>}> : (tensor<3x7xf64>, tensor<1x1xui32>) -> tensor<1x7xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x7xf64>) -> tensor<7xf64>
    return %1 : tensor<7xf64>
  }
}
"#;
    let table: Vec<f64> = (0..21).map(|i| (i + 1) as f64).collect();
    let in0 = f64_buf(&table);
    let out = run_mlir(mlir, &[&in0], &[56]);
    let result = read_f64s(&out[0]);
    // Row 2 = [15,16,17,18,19,20,21]
    let expected: Vec<f64> = (15..22).map(|i| i as f64).collect();
    assert_eq!(result, expected);
}
