// Returns two outputs: element-wise sum and product.
// Tests that pop_output() can be called multiple times per invocation.
// Compile with: iree-compile --iree-hal-target-backends=llvm-cpu multi_output.mlir -o <arch>/multi_output.vmfb
// Omitting target triple compiles for the host architecture; see compile_fixtures.sh.
func.func @multi_output(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %sum = arith.addf %a, %b : tensor<4xf32>
  %prod = arith.mulf %a, %b : tensor<4xf32>
  return %sum, %prod : tensor<4xf32>, tensor<4xf32>
}
