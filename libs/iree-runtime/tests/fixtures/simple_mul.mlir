// Element-wise multiplication of two f32 tensors.
// Compile with: iree-compile --iree-hal-target-backends=llvm-cpu simple_mul.mlir -o <arch>/simple_mul.vmfb
// Omitting target triple compiles for the host architecture; see compile_fixtures.sh.
func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
