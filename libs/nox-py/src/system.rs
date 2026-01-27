use crate::*;

use nox_ecs::{
    CompiledSystem, ExecMetadata, SystemParam,
    graph::{EdgeComponent, GraphQuery, TotalEdge},
    nox::{NoxprComp, NoxprFn, NoxprNode, NoxprTy, jax::JaxNoxprFn},
};
use nox_ecs::{ComponentSchema, World};
use pyo3::IntoPyObjectExt;
use pyo3::types::IntoPyDict;
use std::{collections::{HashMap, HashSet}, sync::Arc};

use crate::iree_exec::IREEExec;

#[pyclass]
#[derive(Debug)]
pub struct PyFnSystem {
    sys: Py<PyAny>,
    input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    edge_ids: Vec<ComponentId>,
    #[allow(dead_code)]
    name: String,
}

impl Clone for PyFnSystem {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            sys: self.sys.clone_ref(py),
            input_ids: self.input_ids.clone(),
            output_ids: self.output_ids.clone(),
            edge_ids: self.edge_ids.clone(),
            name: self.name.clone(),
        })
    }
}

#[pymethods]
impl PyFnSystem {
    #[new]
    fn new(
        sys: PyObject,
        input_ids: Vec<String>,
        output_ids: Vec<String>,
        edge_ids: Vec<String>,
        name: String,
    ) -> Self {
        Self {
            sys,
            input_ids: input_ids.iter().map(|x| ComponentId::new(x)).collect(),
            output_ids: output_ids.iter().map(|x| ComponentId::new(x)).collect(),
            edge_ids: edge_ids.iter().map(|x| ComponentId::new(x)).collect(),
            name,
        }
    }

    fn system(&self) -> System {
        System::new(self.clone())
    }
}

impl nox_ecs::System for PyFnSystem {
    type Arg = ();

    type Ret = ();

    fn init(&self, builder: &mut nox_ecs::SystemBuilder) -> Result<(), nox_ecs::Error> {
        for &id in &self.input_ids {
            builder.init_with_column(id)?;
        }
        for &id in &self.output_ids {
            builder.init_with_column(id)?;
        }

        Ok(())
    }

    fn compile(&self, world: &World) -> Result<nox_ecs::CompiledSystem, nox_ecs::Error> {
        let sys = Python::with_gil(|py| self.sys.clone_ref(py));
        let mut input_ids = self.input_ids.clone();
        let output_ids = self.output_ids.clone();
        let builder = nox_ecs::SystemBuilder::new(world);
        let mut py_builder = SystemBuilder {
            total_edges: GraphQuery::<TotalEdge>::param(&builder)?.edges,
            ..Default::default()
        };
        for id in output_ids.iter() {
            if !input_ids.contains(id) {
                input_ids.push(*id);
            }
        }
        for (index, &id) in input_ids.iter().enumerate() {
            if py_builder.arg_map.contains_key(&id) {
                continue;
            }
            let col = builder
                .world
                .column_by_id(id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let arg_metadata = ArgMetadata {
                entity_map: col.entity_map(),
                len: col.len(),
                schema: col.schema.clone(),
                component: Component {
                    name: col.metadata.name.to_string(),
                    ty: Some(col.schema.clone().into()),
                    metadata: col.metadata.metadata.clone(),
                },
            };
            py_builder.arg_map.insert(id, (arg_metadata, index));
        }
        for &id in &self.edge_ids {
            let col = builder.world.column_by_id(id).unwrap();
            let edges = col
                .iter()
                .map(move |(_, value)| nox_ecs::graph::Edge::from_value(value).unwrap())
                .collect();
            py_builder.edge_map.insert(id, edges);
        }
        let mut tys = self
            .output_ids
            .iter()
            .map(|id| NoxprTy::ArrayTy(builder.world.column_by_id(*id).unwrap().buffer_ty()));

        let ty = if self.output_ids.len() == 1 {
            tys.next().unwrap()
        } else {
            NoxprTy::Tuple(tys.collect())
        };
        let func = Python::with_gil(|py| {
            let func = sys.call1(py, (py_builder,))?;
            let jax = py.import("jax").unwrap();
            // Clean up unused variable warning
            let _jit_args = [("keep_unused", true)].into_py_dict(py);
            // Get the jit function and evaluate it directly
            let jit_fn = jax.getattr("jit")?;
            Ok::<_, pyo3::PyErr>(jit_fn.call1((func,))?.into())
        })?;
        let func = NoxprFn::new(vec![], Noxpr::jax(func));
        Ok(CompiledSystem {
            computation: NoxprComp::new(func, ty),
            inputs: input_ids,
            outputs: output_ids,
        })
    }
}

pub trait CompiledSystemExt {
    fn arg_arrays(&self, py: Python<'_>, world: &World) -> Result<Vec<Py<PyAny>>, Error>;
    fn compile_iree_module(&self, py: Python<'_>, world: &World) -> Result<IREEExec, Error>;
    fn compile_jax_module(&self, py: Python<'_>) -> Result<Py<PyAny>, Error>;
}

impl CompiledSystemExt for CompiledSystem {
    fn arg_arrays(&self, py: Python<'_>, world: &World) -> Result<Vec<Py<PyAny>>, Error> {
        // Enable 64-bit floating point support in JAX before creating arrays
        // This MUST happen before any JAX operations
        let jax = py.import("jax")?;
        jax.getattr("config")?.call_method1("update", ("jax_enable_x64", true))?;
        
        let mut res = vec![];
        let mut visited_ids = HashSet::new();
        
        for id in self.inputs.iter().chain(self.outputs.iter()) {
            // Use the value (*id) not the reference for deduplication
            if visited_ids.contains(id) {
                continue;
            }
            let jnp = py.import("jax.numpy")?;
            let col = world
                .column_by_id(*id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let elem_ty = col.schema.prim_type;
            let dtype = prim_type_to_dtype(elem_ty)?;
            
            // Build tuple shape manually instead of using PyTuple
            let shape_vec: Vec<_> = std::iter::once(col.len() as u64)
                .chain(col.schema.shape().iter().copied())
                .collect();
            
            // Use direct call pattern that takes standard args
            let arr = jnp.getattr("zeros")?.call((shape_vec, dtype), None)?;
            
            
            visited_ids.insert(*id);  // Insert the value, not the reference
            res.push(arr.into());
        }
        Ok(res)
    }

    fn compile_iree_module(&self, py: Python<'_>, world: &World) -> Result<IREEExec, Error> {
        let func = noxpr_to_callable(self.computation.func.clone());
        let input_arrays = self.arg_arrays(py, world)?;
        
        // Build deduplicated arg_ids matching what arg_arrays() produces
        // Note: We insert the value (*id), not the reference, for proper deduplication
        let mut deduplicated_arg_ids = Vec::new();
        let mut visited_ids = HashSet::new();
        for &id in self.inputs.iter().chain(self.outputs.iter()) {
            if visited_ids.insert(id) {
                deduplicated_arg_ids.push(id);
            }
        }
        
        
        // Python code to export JAX function to StableHLO and compile with IREE
        let py_code = r#"
from jax import export
import jax

def export_and_compile(func, args, backend="llvm-cpu"):
    # Enable 64-bit floating point support in JAX
    jax.config.update("jax_enable_x64", True)
    
    # Wrap the function in jax.jit first (required by jax.export)
    jit_fn = jax.jit(func)
    
    # Get input shapes for export
    input_shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
    
    # Export to StableHLO
    exported = export.export(jit_fn)(*input_shapes)
    stablehlo_mlir = str(exported.mlir_module())
    
    
    # Compile with IREE
    # Important: Use extra_args to enable f64 support and prevent demotion to f32
    from iree.compiler import compile_str
    from iree import runtime as rt
    vmfb = compile_str(
        stablehlo_mlir, 
        target_backends=[backend],
        extra_args=[
            "--iree-vm-target-extension-f64",
            "--iree-input-demote-f64-to-f32=false"
        ]
    )
    
    # Load the module temporarily to inspect the function signature
    # IREE may optimize away unused inputs, so we need to check the actual signature
    config = rt.Config("local-task")
    ctx = rt.SystemContext(config=config)
    vm_module = rt.VmModule.copy_buffer(ctx.instance, vmfb)
    vm_func = vm_module.lookup_function('main')
    
    # Get the actual number of inputs from the function reflection
    reflection = vm_func.reflection
    abi_decl = reflection.get('iree.abi.declaration', '')
    
    # Parse the expected input and output shapes from the ABI declaration
    # Format: %input0: tensor<1xi64>, %input1: tensor<3x7xf64>, ... -> (%output0: tensor<...>, ...)
    import re
    
    # Parse input shapes
    input_shapes = []
    for match in re.finditer(r'%input\d+:\s*tensor<([^>]+)>', abi_decl):
        shape_dtype = match.group(1)
        # Parse shape like "3x7xf64" -> ([3, 7], "f64")
        parts = shape_dtype.split('x')
        dtype = parts[-1]
        shape = tuple(int(p) for p in parts[:-1])
        input_shapes.append((shape, dtype))
    
    # Parse output shapes
    output_shapes = []
    for match in re.finditer(r'%output\d+:\s*tensor<([^>]+)>', abi_decl):
        shape_dtype = match.group(1)
        parts = shape_dtype.split('x')
        dtype = parts[-1]
        shape = tuple(int(p) for p in parts[:-1])
        output_shapes.append((shape, dtype))
    
    # Return vmfb, input shapes, and output shapes
    return vmfb, input_shapes, output_shapes
"#;
        
        // Create a module and execute the code
        let module = PyModule::new(py, "iree_compile")?;
        let globals = module.dict();
        let code_cstr = std::ffi::CString::new(py_code).expect("Python code C string");
        py.run(code_cstr.as_ref(), Some(&globals), None)?;
        
        let compile_fn: Py<PyAny> = module.getattr("export_and_compile")?.into();
        
        // Call the compile function - returns (vmfb_bytes, input_shapes, output_shapes)
        let result = compile_fn.call1(py, (func, input_arrays))?;
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
        let vmfb_bytes: Vec<u8> = tuple.get_item(0)?.extract()?;
        let input_shapes: Vec<(Vec<i64>, String)> = tuple.get_item(1)?.extract()?;
        let output_shapes: Vec<(Vec<i64>, String)> = tuple.get_item(2)?.extract()?;
        
        // Build shape-indexed lists for inputs (preserving original order within each shape)
        // Key: (shape, dtype) -> Vec of (original_index, ComponentId)
        let mut shape_to_input_ids: HashMap<(Vec<u64>, String), Vec<(usize, ComponentId)>> = HashMap::new();
        for (orig_idx, &id) in deduplicated_arg_ids.iter().enumerate() {
            let col = world.column_by_id(id).ok_or(nox_ecs::Error::ComponentNotFound)?;
            let shape: Vec<u64> = std::iter::once(col.len() as u64)
                .chain(col.schema.shape().iter().copied())
                .collect();
            let dtype = prim_type_to_dtype(col.schema.prim_type)?.to_string();
            shape_to_input_ids.entry((shape, dtype)).or_default().push((orig_idx, id));
        }
        
        // Match IREE expected input shapes to our arg_ids (take FIRST match, not last)
        let mut actual_arg_ids = Vec::with_capacity(input_shapes.len());
        let mut used_input_indices = HashSet::new();
        
        for (expected_shape, expected_dtype) in &input_shapes {
            let expected_shape: Vec<u64> = expected_shape.iter().map(|&x| x as u64).collect();
            let normalized_dtype = match expected_dtype.as_str() {
                "i64" => "uint64".to_string(),
                other => other.to_string(),
            };
            
            // Try exact match first
            let key = (expected_shape.clone(), normalized_dtype.clone());
            let mut found = false;
            
            if let Some(ids) = shape_to_input_ids.get(&key) {
                // Find first unused match
                for (orig_idx, id) in ids {
                    if !used_input_indices.contains(orig_idx) {
                        actual_arg_ids.push(*id);
                        used_input_indices.insert(*orig_idx);
                        found = true;
                        break;
                    }
                }
            }
            
            // Try f64 variant if not found
            if !found {
                let f64_key = (expected_shape.clone(), "float64".to_string());
                if let Some(ids) = shape_to_input_ids.get(&f64_key) {
                    for (orig_idx, id) in ids {
                        if !used_input_indices.contains(orig_idx) {
                            actual_arg_ids.push(*id);
                            used_input_indices.insert(*orig_idx);
                            found = true;
                            break;
                        }
                    }
                }
            }
            
            if !found {
                tracing::error!("No arg_id found for IREE input shape {:?}/{}", expected_shape, expected_dtype);
                return Err(Error::from(nox_ecs::Error::ComponentNotFound));
            }
        }
        
        // Same approach for outputs
        let mut shape_to_output_ids: HashMap<(Vec<u64>, String), Vec<(usize, ComponentId)>> = HashMap::new();
        for (orig_idx, &id) in self.outputs.iter().enumerate() {
            let col = world.column_by_id(id).ok_or(nox_ecs::Error::ComponentNotFound)?;
            let shape: Vec<u64> = std::iter::once(col.len() as u64)
                .chain(col.schema.shape().iter().copied())
                .collect();
            let dtype = prim_type_to_dtype(col.schema.prim_type)?.to_string();
            shape_to_output_ids.entry((shape, dtype)).or_default().push((orig_idx, id));
        }
        
        let mut actual_ret_ids = Vec::with_capacity(output_shapes.len());
        let mut used_output_indices = HashSet::new();
        
        for (expected_shape, expected_dtype) in &output_shapes {
            let expected_shape: Vec<u64> = expected_shape.iter().map(|&x| x as u64).collect();
            let normalized_dtype = match expected_dtype.as_str() {
                "i64" => "uint64".to_string(),
                other => other.to_string(),
            };
            
            let key = (expected_shape.clone(), normalized_dtype.clone());
            let mut found = false;
            
            if let Some(ids) = shape_to_output_ids.get(&key) {
                for (orig_idx, id) in ids {
                    if !used_output_indices.contains(orig_idx) {
                        actual_ret_ids.push(*id);
                        used_output_indices.insert(*orig_idx);
                        found = true;
                        break;
                    }
                }
            }
            
            if !found {
                let f64_key = (expected_shape.clone(), "float64".to_string());
                if let Some(ids) = shape_to_output_ids.get(&f64_key) {
                    for (orig_idx, id) in ids {
                        if !used_output_indices.contains(orig_idx) {
                            actual_ret_ids.push(*id);
                            used_output_indices.insert(*orig_idx);
                            found = true;
                            break;
                        }
                    }
                }
            }
            
            if !found {
                tracing::error!("No ret_id found for IREE output shape {:?}/{}", expected_shape, expected_dtype);
                return Err(Error::from(nox_ecs::Error::ComponentNotFound));
            }
        }
        
        tracing::debug!("IREE compiled: {} inputs, {} outputs", actual_arg_ids.len(), actual_ret_ids.len());
        
        Ok(IREEExec::new(
            ExecMetadata {
                arg_ids: actual_arg_ids,
                ret_ids: actual_ret_ids,
            },
            vmfb_bytes,
        ))
    }

    fn compile_jax_module(&self, py: Python<'_>) -> Result<Py<PyAny>, Error> {
        let func = noxpr_to_callable(self.computation.func.clone());

        let py_code = "
import jax
def build_expr(func):
    res = jax.jit(func)
    return res";

        // We need to create a module directly and exec code in it
        let module = PyModule::new(py, "build_expr")?;
        let globals = module.dict();
        // Use std::ffi::CString for the string arguments to run
        let code_cstr = std::ffi::CString::new(py_code).unwrap();
        py.run(code_cstr.as_ref(), Some(&globals), None)?;
        let fun: Py<PyAny> = module.getattr("build_expr")?.into();

        let comp = fun.call1(py, (func,))?;

        Ok(comp)
    }
}

/// Convert PrimType to JAX dtype string
fn prim_type_to_dtype(prim_type: impeller2::types::PrimType) -> Result<&'static str, Error> {
    use impeller2::types::PrimType;
    match prim_type {
        PrimType::I8 => Ok("int8"),
        PrimType::I16 => Ok("int16"),
        PrimType::I32 => Ok("int32"),
        PrimType::I64 => Ok("int64"),
        PrimType::U8 => Ok("uint8"),
        PrimType::U16 => Ok("uint16"),
        PrimType::U32 => Ok("uint32"),
        PrimType::U64 => Ok("uint64"),
        PrimType::F32 => Ok("float32"),
        PrimType::F64 => Ok("float64"),
        PrimType::Bool => Ok("bool"),
    }
}

#[pyclass]
#[derive(Clone, Default)]
pub struct SystemBuilder {
    arg_map: HashMap<ComponentId, (ArgMetadata, usize)>,
    pub edge_map: HashMap<ComponentId, Vec<nox_ecs::graph::Edge>>,
    pub total_edges: Vec<nox_ecs::graph::Edge>,
}

#[derive(Clone)]
pub struct ArgMetadata {
    pub entity_map: BTreeMap<impeller2::types::EntityId, usize>,
    pub len: usize,
    pub schema: ComponentSchema,
    pub component: Component,
}

impl SystemBuilder {
    pub fn get_var(&self, id: ComponentId) -> Option<(ArgMetadata, usize)> {
        self.arg_map.get(&id).cloned()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct System {
    pub inner: Arc<dyn nox_ecs::System<Arg = (), Ret = ()> + Send + Sync>,
}

impl System {
    pub fn new(sys: impl nox_ecs::System + Send + Sync + 'static) -> Self {
        let inner = Arc::new(ErasedSystem::new(sys));
        Self { inner }
    }
}

#[pymethods]
impl System {
    pub fn pipe(&self, other: System) -> System {
        let pipe = nox_ecs::Pipe::new(self.clone(), other);
        System::new(pipe)
    }

    pub fn __or__(&self, other: Option<System>) -> System {
        match other {
            Some(other_sys) => self.pipe(other_sys),
            None => self.clone(),
        }
    }

    pub fn __ror__(&self, _other: PyObject) -> System {
        // Handle the case where the left operand is None
        // Return self unchanged (None is effectively skipped)
        self.clone()
    }
}

impl nox_ecs::System for System {
    type Arg = ();

    type Ret = ();

    fn init(&self, builder: &mut nox_ecs::SystemBuilder) -> Result<(), nox_ecs::Error> {
        self.inner.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, nox_ecs::Error> {
        self.inner.compile(world)
    }
}

fn noxpr_to_callable(func: Arc<NoxprFn>) -> Py<PyAny> {
    if let NoxprNode::Jax(j) = &*func.inner.node {
        return Python::with_gil(|py| j.clone_ref(py));
    }
    let func = JaxNoxprFn {
        tracer: Default::default(),
        inner: func,
    };

    Python::with_gil(|py| func.into_py_any(py).unwrap())
}
