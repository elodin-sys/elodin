//! Safe typed executor for UI display kernels compiled from StableHLO.

use crate::const_fold;
use crate::ir::Module;
use crate::parser;

#[cfg(feature = "jit")]
use crate::lower::{CompiledModule, compile_module};

#[cfg(feature = "jit")]
type TickFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

enum KernelBackend {
    #[cfg(feature = "jit")]
    Jit {
        _compiled: Box<CompiledModule>,
        tick_fn: TickFn,
        input_ptrs: Vec<*const u8>,
        output_ptrs: Vec<*mut u8>,
    },
    Interp(Module),
}

/// Compiled StableHLO module with a checked pointer ABI.
pub struct DisplayKernelExec {
    backend: KernelBackend,
    input_bytes: Vec<usize>,
    output_bytes: Vec<usize>,
}

// Safety: the JIT module is immutable after `compile_module` returns, and
// pointer scratch is only rewritten through `&mut self` invoke. The
// interpreter only holds owned IR.
unsafe impl Send for DisplayKernelExec {}
unsafe impl Sync for DisplayKernelExec {}

impl DisplayKernelExec {
    pub fn compile(
        mlir: &str,
        input_bytes: &[usize],
        output_bytes: &[usize],
    ) -> Result<Self, String> {
        compile_with_backend(mlir, input_bytes, output_bytes, cfg!(feature = "jit"))
    }

    /// Parse, fold, and interpret without emitting native code.
    pub fn compile_interp(
        mlir: &str,
        input_bytes: &[usize],
        output_bytes: &[usize],
    ) -> Result<Self, String> {
        compile_with_backend(mlir, input_bytes, output_bytes, false)
    }

    pub fn validate(mlir: &str) -> Result<(), String> {
        let mut ir_module =
            parser::parse_module(mlir).map_err(|err| format!("MLIR parse failed: {err}"))?;
        const_fold::fold_module(&mut ir_module);
        #[cfg(feature = "jit")]
        compile_module(&ir_module).map_err(|err| format!("Cranelift compile failed: {err}"))?;
        #[cfg(not(feature = "jit"))]
        {
            let _ = ir_module.main_func().ok_or_else(|| {
                "compiled module is missing a callable main entrypoint".to_string()
            })?;
        }
        Ok(())
    }

    pub fn invoke(&mut self, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<(), String> {
        check_buffers(&self.input_bytes, &self.output_bytes, inputs, outputs)?;
        match &mut self.backend {
            #[cfg(feature = "jit")]
            KernelBackend::Jit {
                tick_fn,
                input_ptrs,
                output_ptrs,
                ..
            } => {
                for (slot, buf) in input_ptrs.iter_mut().zip(inputs) {
                    *slot = buf.as_ptr();
                }
                for (slot, buf) in output_ptrs.iter_mut().zip(outputs.iter_mut()) {
                    *slot = buf.as_mut_ptr();
                }
                unsafe {
                    (*tick_fn)(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr());
                }
                Ok(())
            }
            KernelBackend::Interp(module) => crate::interp::interpret_main(module, inputs, outputs),
        }
    }

    pub fn input_bytes(&self) -> &[usize] {
        &self.input_bytes
    }

    pub fn output_bytes(&self) -> &[usize] {
        &self.output_bytes
    }
}

fn compile_with_backend(
    mlir: &str,
    input_bytes: &[usize],
    output_bytes: &[usize],
    use_jit: bool,
) -> Result<DisplayKernelExec, String> {
    if input_bytes.contains(&0) || output_bytes.contains(&0) {
        return Err("display kernel tensor sizes must be non-zero".into());
    }
    let mut ir_module =
        parser::parse_module(mlir).map_err(|err| format!("MLIR parse failed: {err}"))?;
    const_fold::fold_module(&mut ir_module);
    let backend = if use_jit {
        #[cfg(feature = "jit")]
        {
            jit_backend(&ir_module, input_bytes.len(), output_bytes.len())?
        }
        #[cfg(not(feature = "jit"))]
        {
            let _ = use_jit;
            KernelBackend::Interp(ir_module)
        }
    } else {
        KernelBackend::Interp(ir_module)
    };
    Ok(DisplayKernelExec {
        backend,
        input_bytes: input_bytes.to_vec(),
        output_bytes: output_bytes.to_vec(),
    })
}

#[cfg(feature = "jit")]
fn jit_backend(
    ir_module: &Module,
    n_inputs: usize,
    n_outputs: usize,
) -> Result<KernelBackend, String> {
    let compiled =
        compile_module(ir_module).map_err(|err| format!("Cranelift compile failed: {err}"))?;
    let fn_ptr = compiled.get_main_fn();
    if fn_ptr.is_null() {
        return Err("compiled module is missing a callable main entrypoint".into());
    }
    let tick_fn = unsafe { std::mem::transmute::<*const u8, TickFn>(fn_ptr) };
    Ok(KernelBackend::Jit {
        _compiled: Box::new(compiled),
        tick_fn,
        input_ptrs: vec![std::ptr::null(); n_inputs],
        output_ptrs: vec![std::ptr::null_mut(); n_outputs],
    })
}

fn check_buffers(
    input_bytes: &[usize],
    output_bytes: &[usize],
    inputs: &[&[u8]],
    outputs: &mut [&mut [u8]],
) -> Result<(), String> {
    if inputs.len() != input_bytes.len() {
        return Err(format!(
            "expected {} inputs, got {}",
            input_bytes.len(),
            inputs.len()
        ));
    }
    if outputs.len() != output_bytes.len() {
        return Err(format!(
            "expected {} outputs, got {}",
            output_bytes.len(),
            outputs.len()
        ));
    }
    for (index, (buf, expected)) in inputs.iter().zip(input_bytes).enumerate() {
        if buf.len() != *expected {
            return Err(format!(
                "input {index} is {} bytes, expected {expected}",
                buf.len()
            ));
        }
    }
    for (index, (buf, expected)) in outputs.iter().zip(output_bytes).enumerate() {
        if buf.len() != *expected {
            return Err(format!(
                "output {index} is {} bytes, expected {expected}",
                buf.len()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sqrt_mlir() -> &'static str {
        r#"
module @module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.sqrt %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
"#
    }

    fn eval_sqrt(mut exec: DisplayKernelExec) {
        let input = 16.0_f64.to_le_bytes();
        let mut output = [0u8; 8];
        exec.invoke(&[&input], &mut [&mut output]).unwrap();
        let value = f64::from_le_bytes(output);
        assert!((value - 4.0).abs() < 1e-12);
    }

    #[test]
    fn compiles_and_evaluates_scalar_sqrt() {
        eval_sqrt(DisplayKernelExec::compile(sqrt_mlir(), &[8], &[8]).unwrap());
    }

    #[test]
    fn interpreter_evaluates_scalar_sqrt() {
        eval_sqrt(DisplayKernelExec::compile_interp(sqrt_mlir(), &[8], &[8]).unwrap());
    }

    #[test]
    fn interpreter_clamps_with_scalar_bounds() {
        let mlir = r#"
module @module {
  func.func @main(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %lo = stablehlo.constant dense<-1.0> : tensor<f64>
    %hi = stablehlo.constant dense<1.0> : tensor<f64>
    %0 = stablehlo.clamp %lo, %arg0, %hi : (tensor<f64>, tensor<3xf64>, tensor<f64>) -> tensor<3xf64>
    return %0 : tensor<3xf64>
  }
}
"#;
        let input = f64_bytes(&[-5.0, 0.5, 3.0]);
        let out = eval_fixture(mlir, &[24], &[24], &input);
        assert_eq!(read_f64s(&out), vec![-1.0, 0.5, 1.0]);
        #[cfg(feature = "jit")]
        assert_interp_matches_jit(mlir, &[24], &[24], &input);
    }

    #[test]
    fn interpreter_compares_against_scalar() {
        let mlir = r#"
module @module {
  func.func @main(%arg0: tensor<3xf64>) -> tensor<3xi1> {
    %c = stablehlo.constant dense<0.0> : tensor<f64>
    %0 = stablehlo.compare GT, %arg0, %c, FLOAT : (tensor<3xf64>, tensor<f64>) -> tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}
"#;
        let input = f64_bytes(&[-1.0, 0.0, 2.0]);
        let interp = invoke_bytes(
            DisplayKernelExec::compile_interp(mlir, &[24], &[3]).unwrap(),
            &[&input],
            &[3],
        );
        assert_eq!(interp[0], vec![0, 0, 1]);
    }

    #[test]
    fn rejects_buffer_size_mismatch() {
        let mut exec = DisplayKernelExec::compile(sqrt_mlir(), &[8], &[8]).unwrap();
        let input = [0u8; 4];
        let mut output = [0u8; 8];
        assert!(exec.invoke(&[&input], &mut [&mut output]).is_err());
    }

    fn packed_cov() -> [f64; 6] {
        // SPD [[4, 1, 0], [1, 3, 0.5], [0, 0.5, 2]] packed as [p00,p10,p20,p11,p21,p22].
        [4.0, 1.0, 0.0, 3.0, 0.5, 2.0]
    }

    fn f64_bytes(vals: &[f64]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn read_f64s(buf: &[u8]) -> Vec<f64> {
        buf.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn invoke_bytes(
        mut exec: DisplayKernelExec,
        inputs: &[&[u8]],
        out_sizes: &[usize],
    ) -> Vec<Vec<u8>> {
        let mut outs: Vec<Vec<u8>> = out_sizes.iter().map(|&n| vec![0u8; n]).collect();
        let mut refs: Vec<&mut [u8]> = outs.iter_mut().map(|b| b.as_mut_slice()).collect();
        exec.invoke(inputs, &mut refs).unwrap();
        outs
    }

    fn assert_f64_close(lhs: &[u8], rhs: &[u8]) {
        let a = read_f64s(lhs);
        let b = read_f64s(rhs);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert!(
                (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())),
                "{x} vs {y}"
            );
        }
    }

    fn expected_stats(cov: &[f64; 6]) -> [f64; 2] {
        let [p00, p10, p20, p11, p21, p22] = *cov;
        let det = p00 * (p11 * p22 - p21 * p21) - p10 * (p10 * p22 - p20 * p21)
            + p20 * (p10 * p21 - p11 * p20);
        [det, p00 + p11 + p22]
    }

    fn expected_chol(cov: &[f64; 6]) -> [f64; 9] {
        let [p00, p10, p20, p11, p21, p22] = *cov;
        let l00 = p00.sqrt();
        let l10 = p10 / l00;
        let l20 = p20 / l00;
        let l11 = (p11 - l10 * l10).sqrt();
        let l21 = (p21 - l20 * l10) / l11;
        let l22 = (p22 - l20 * l20 - l21 * l21).sqrt();
        [l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22]
    }

    #[derive(serde::Deserialize)]
    struct KernelFixture {
        batch_size: u32,
        scalar_mlir: String,
        batched_mlir: String,
        inputs: Vec<KernelTensor>,
        outputs: Vec<KernelTensor>,
    }

    #[derive(serde::Deserialize)]
    struct KernelTensor {
        shape: Vec<i64>,
        dtype: String,
    }

    fn nbytes(tensors: &[KernelTensor], batch: Option<u32>) -> Vec<usize> {
        tensors
            .iter()
            .map(|t| {
                let width = match t.dtype.as_str() {
                    "f32" | "i32" | "u32" => 4,
                    "bool" => 1,
                    _ => 8,
                };
                let elems = t.shape.iter().fold(1i64, |acc, d| acc * d).max(1) as usize;
                let rows = batch.map(|n| n as usize).unwrap_or(1);
                rows * elems * width
            })
            .collect()
    }

    fn load_fixture(json: &str) -> KernelFixture {
        serde_json::from_str(json).expect("kernel fixture")
    }

    fn eval_fixture(mlir: &str, in_sizes: &[usize], out_sizes: &[usize], input: &[u8]) -> Vec<u8> {
        let outs = invoke_bytes(
            DisplayKernelExec::compile_interp(mlir, in_sizes, out_sizes).unwrap(),
            &[input],
            out_sizes,
        );
        outs.into_iter().next().unwrap()
    }

    #[test]
    fn interpreter_evaluates_covariance_stats_fixture() {
        let fixture = load_fixture(include_str!(
            "../testdata/display-kernel-covariance-stats.json"
        ));
        let in_sizes = nbytes(&fixture.inputs, None);
        let out_sizes = nbytes(&fixture.outputs, None);
        let input = f64_bytes(&packed_cov());
        let out = eval_fixture(&fixture.scalar_mlir, &in_sizes, &out_sizes, &input);
        let got = read_f64s(&out);
        let expect = expected_stats(&packed_cov());
        assert_eq!(got.len(), 2);
        assert!((got[0] - expect[0]).abs() < 1e-12);
        assert!((got[1] - expect[1]).abs() < 1e-12);
    }

    #[test]
    fn interpreter_evaluates_covariance_cholesky_fixture() {
        let fixture = load_fixture(include_str!(
            "../testdata/display-kernel-covariance-cholesky.json"
        ));
        let in_sizes = nbytes(&fixture.inputs, None);
        let out_sizes = nbytes(&fixture.outputs, None);
        let input = f64_bytes(&packed_cov());
        let out = eval_fixture(&fixture.scalar_mlir, &in_sizes, &out_sizes, &input);
        let got = read_f64s(&out);
        let expect = expected_chol(&packed_cov());
        assert_eq!(got.len(), 9);
        for (x, y) in got.iter().zip(&expect) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[cfg(feature = "jit")]
    fn assert_interp_matches_jit(
        mlir: &str,
        in_sizes: &[usize],
        out_sizes: &[usize],
        input: &[u8],
    ) {
        let jit = invoke_bytes(
            DisplayKernelExec::compile(mlir, in_sizes, out_sizes).unwrap(),
            &[input],
            out_sizes,
        );
        let interp = invoke_bytes(
            DisplayKernelExec::compile_interp(mlir, in_sizes, out_sizes).unwrap(),
            &[input],
            out_sizes,
        );
        assert_eq!(jit.len(), interp.len());
        for (a, b) in jit.iter().zip(&interp) {
            assert_f64_close(a, b);
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn interpreter_matches_jit_sqrt() {
        let input = 16.0_f64.to_le_bytes();
        assert_interp_matches_jit(sqrt_mlir(), &[8], &[8], &input);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn interpreter_matches_jit_on_display_kernels() {
        for json in [
            include_str!("../testdata/display-kernel-covariance-stats.json"),
            include_str!("../testdata/display-kernel-covariance-cholesky.json"),
        ] {
            let fixture = load_fixture(json);
            let cov = packed_cov();
            let scalar_in = nbytes(&fixture.inputs, None);
            let scalar_out = nbytes(&fixture.outputs, None);
            let input = f64_bytes(&cov);
            assert_interp_matches_jit(&fixture.scalar_mlir, &scalar_in, &scalar_out, &input);

            let batched_in = nbytes(&fixture.inputs, Some(fixture.batch_size));
            let batched_out = nbytes(&fixture.outputs, Some(fixture.batch_size));
            let mut batched = vec![0u8; batched_in[0]];
            let row = f64_bytes(&cov);
            for chunk in batched.chunks_exact_mut(row.len()) {
                chunk.copy_from_slice(&row);
            }
            assert_interp_matches_jit(&fixture.batched_mlir, &batched_in, &batched_out, &batched);
        }
    }
}
