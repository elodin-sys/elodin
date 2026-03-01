// 2D matrix multiply: [3x4] * [4x2] -> [3x2]
// Tests multi-dimensional tensor shapes through the runtime.
// Compile with: iree-compile --iree-hal-target-backends=llvm-cpu matmul.mlir -o matmul.vmfb
// Do NOT add --iree-llvmcpu-target-triple; omit it for cross-platform VMFBs.
func.func @matmul(%a: tensor<3x4xf32>, %b: tensor<4x2xf32>) -> tensor<3x2xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.splat %zero : tensor<3x2xf32>
  %c = linalg.matmul ins(%a, %b : tensor<3x4xf32>, tensor<4x2xf32>) outs(%init : tensor<3x2xf32>) -> tensor<3x2xf32>
  return %c : tensor<3x2xf32>
}
