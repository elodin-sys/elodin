// Reproducer for iree_linalg_ext.scatter verification failure.
//
// JAX's .at[row:row+3, col:col+3].set(block) on an 18x18 matrix produces
// stablehlo.scatter with tensor<2xi32> indices and 2D update_window_dims.
// IREE's ConvertStableHloToLinalgExt lowers this to iree_linalg_ext.scatter
// whose verifier rejects the indices rank.
//
// Our BlockScatterToDynamicUpdateSlice rewrite (patch 5) should rewrite
// this to stablehlo.dynamic_update_slice, but the pattern is not matching.
//
// Extracted from customer sim_FT19: @inner_323 (EKF navigation Jacobian).
module @module {
  func.func public @main(%arg0: tensor<18x18xf64>, %update: tensor<3x3xf64>) -> tensor<18x18xf64> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c9 = stablehlo.constant dense<9> : tensor<i32>
    %idx0_a = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %idx1_a = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %indices_a = stablehlo.concatenate %idx0_a, %idx1_a, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

    // First scatter: .at[0:3, 0:3].set(update)
    %s1 = "stablehlo.scatter"(%arg0, %indices_a, %update) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1]>,
      unique_indices = true
    }> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      stablehlo.return %arg2 : tensor<f64>
    }) : (tensor<18x18xf64>, tensor<2xi32>, tensor<3x3xf64>) -> tensor<18x18xf64>

    // Second scatter (chained): .at[0:3, 9:12].set(update)
    %idx0_b = stablehlo.broadcast_in_dim %c0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %idx1_b = stablehlo.broadcast_in_dim %c9, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %indices_b = stablehlo.concatenate %idx0_b, %idx1_b, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>

    %s2 = "stablehlo.scatter"(%s1, %indices_b, %update) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [0, 1],
        scatter_dims_to_operand_dims = [0, 1]>,
      unique_indices = true
    }> ({
    ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
      stablehlo.return %arg4 : tensor<f64>
    }) : (tensor<18x18xf64>, tensor<2xi32>, tensor<3x3xf64>) -> tensor<18x18xf64>

    return %s2 : tensor<18x18xf64>
  }
}
