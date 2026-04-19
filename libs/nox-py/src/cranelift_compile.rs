use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashSet;
use std::time::Instant;

use crate::cranelift_exec::CraneliftExec;
use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::system::{CompiledSystem, noxpr_to_callable};
use crate::utils::PrimTypeExt;
use crate::world::World;

pub fn compile_cranelift_module(
    py: Python<'_>,
    compiled_system: &CompiledSystem,
    world: &World,
) -> Result<CraneliftExec, Error> {
    let func = noxpr_to_callable(compiled_system.computation.func.clone());

    let mut input_arrays = vec![];
    let mut visited_ids = HashSet::new();

    for slot in &compiled_system.input_slots {
        if !visited_ids.insert(slot.component_id) {
            continue;
        }
        let col = world
            .column_by_id(slot.component_id)
            .ok_or(Error::ComponentNotFound)?;
        let elem_ty = col.schema.prim_type;
        let dtype = nox::jax::dtype(&elem_ty.to_element_type())?;
        let shape_vec: Vec<_> = slot.shape.iter().map(|&dim| dim as u64).collect();
        let jnp = py.import("jax.numpy")?;
        let arr = jnp.getattr("zeros")?.call((shape_vec, dtype), None)?;
        input_arrays.push(arr.unbind());
    }

    let py_code = r#"
import jax
import os
import re
import json
import numpy as np
os.environ["JAX_ENABLE_X64"] = "1"
jax.config.update("jax_enable_x64", True)

def lower_to_stablehlo(func, input_arrays):
    jit_fn = jax.jit(func, keep_unused=True)
    lowered = jit_fn.lower(*input_arrays)
    stablehlo_module = lowered.compiler_ir(dialect="stablehlo")
    stablehlo_mlir = str(stablehlo_module)
    stablehlo_mlir = re.sub(r'module @\S+', 'module @module', stablehlo_mlir, count=1)

    debug_dir = os.environ.get('ELODIN_CRANELIFT_DEBUG_DIR')
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'stablehlo.mlir'), 'w') as f:
            f.write(stablehlo_mlir)
        input_summaries = [
            {'shape': [int(d) for d in arr.shape], 'dtype': str(arr.dtype)}
            for arr in input_arrays
        ]
        with open(os.path.join(debug_dir, 'compile_context.json'), 'w') as f:
            json.dump({'inputs': input_summaries}, f, indent=2)
        import sys
        print(f'[elodin-cranelift] dumped StableHLO to {debug_dir}', file=sys.stderr)

    return stablehlo_mlir

def run_xla_reference(func, real_input_arrays, debug_dir):
    """Run the function with XLA and save reference outputs."""
    import sys
    try:
        os.makedirs(debug_dir, exist_ok=True)
        jit_fn = jax.jit(func, keep_unused=True)
        results = jit_fn(*real_input_arrays)
        if not isinstance(results, (list, tuple)):
            results = (results,)
        for i, r in enumerate(results):
            arr = np.asarray(r)
            path = os.path.join(debug_dir, f'xla_output_{i}.bin')
            arr.tofile(path)
        print(f'[elodin-cranelift] checkpoint: saved {len(results)} XLA reference outputs to {debug_dir}', file=sys.stderr)
    except Exception as e:
        print(f'[elodin-cranelift] checkpoint: XLA reference failed: {e}', file=sys.stderr)
"#;

    let module = PyModule::new(py, "cranelift_compile")?;
    let globals = module.dict();
    let code_cstr = std::ffi::CString::new(py_code).expect("Python code C string");
    py.run(code_cstr.as_ref(), Some(&globals), None)?;
    let lower_fn: Py<PyAny> = module.getattr("lower_to_stablehlo")?.into();

    let py_input_arrays = pyo3::types::PyList::new(py, input_arrays.iter().map(|a| a.bind(py)))?;
    let func_for_xla = func.clone_ref(py);

    let lower_start = Instant::now();
    let result = lower_fn.call1(py, (func, py_input_arrays))?;
    let stablehlo_mlir: String = result.extract(py)?;
    let lower_ms = lower_start.elapsed().as_secs_f64() * 1000.0;

    let parse_start = Instant::now();
    let mut ir_module = cranelift_mlir::parser::parse_module(&stablehlo_mlir)
        .map_err(|e| Error::CraneliftBackend(format!("MLIR parse failed: {e}")))?;
    let parse_ms = parse_start.elapsed().as_secs_f64() * 1000.0;

    let fold_start = Instant::now();
    cranelift_mlir::const_fold::fold_module(&mut ir_module);
    let fold_ms = fold_start.elapsed().as_secs_f64() * 1000.0;

    let compile_start = Instant::now();
    let compiled = cranelift_mlir::lower::compile_module(&ir_module)
        .map_err(|e| Error::CraneliftBackend(format!("Cranelift compile failed: {e}")))?;
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    let timings = compiled.timings;
    let total_ms = lower_ms + parse_ms + fold_ms + compile_ms;
    eprintln!(
        "[elodin-cranelift] lower={lower_ms:.1}ms parse={parse_ms:.1}ms \
         fold={fold_ms:.1}ms ir_build={ir_build_ms:.1}ms codegen={codegen_ms:.1}ms \
         link={link_ms:.1}ms total={total_ms:.1}ms",
        ir_build_ms = timings.ir_build_ms,
        codegen_ms = timings.codegen_ms,
        link_ms = timings.link_ms,
    );

    if let Ok(debug_dir) = std::env::var("ELODIN_CRANELIFT_DEBUG_DIR") {
        let xla_ref_fn: Py<PyAny> = module.getattr("run_xla_reference")?.into();
        let np = py.import("numpy")?;
        let jnp = py.import("jax.numpy")?;
        let mut real_inputs = vec![];
        let mut visited = HashSet::new();
        for slot in &compiled_system.input_slots {
            if !visited.insert(slot.component_id) {
                continue;
            }
            if let Some(col) = world.column_by_id(slot.component_id) {
                let dtype = nox::jax::dtype(&col.schema.prim_type.to_element_type())?;
                let arr = np
                    .getattr("frombuffer")?
                    .call1((pyo3::types::PyBytes::new(py, col.column), dtype))?;
                let shape_vec: Vec<i64> = slot.shape.to_vec();
                let reshaped = arr.call_method1("reshape", (shape_vec,))?;
                let jax_arr = jnp.getattr("array")?.call1((reshaped,))?;
                real_inputs.push(jax_arr.unbind());
            }
        }
        let real_list = pyo3::types::PyList::new(py, real_inputs.iter().map(|a| a.bind(py)))?;
        let _ = xla_ref_fn.call1(py, (func_for_xla, real_list, &debug_dir));
        let mlir_path = format!("{debug_dir}/stablehlo.mlir");
        let _ = std::fs::write(&mlir_path, &stablehlo_mlir);
        eprintln!("[elodin-cranelift] checkpoint: saved MLIR to {mlir_path}");
    }

    let metadata = ExecMetadata {
        arg_ids: compiled_system.inputs.clone(),
        ret_ids: compiled_system.outputs.clone(),
        arg_slots: compiled_system.input_slots.clone(),
        ret_slots: compiled_system.output_slots.clone(),
        has_singleton_lowering: compiled_system.has_singleton_lowering,
        promoted_constants: vec![],
    };

    CraneliftExec::new(metadata, compiled, world)
}
