// Identity function for i64 tensors (pass-through).
// Tests integer type round-trip through the runtime.
// Compile with: iree-compile --iree-hal-target-backends=llvm-cpu identity_i64.mlir -o identity_i64.vmfb
// Do NOT add --iree-llvmcpu-target-triple; omit it for cross-platform VMFBs.
func.func @identity_i64(%a: tensor<4xi64>) -> tensor<4xi64> {
  return %a : tensor<4xi64>
}
