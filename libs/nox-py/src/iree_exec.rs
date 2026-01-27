//! IREE execution wrapper for compiled JAX/StableHLO computations.
//!
//! This module provides the `IREEExec` struct which wraps compiled IREE VMFB
//! bytecode and provides execution capabilities via the IREE Python runtime.

use crate::Error;
use nox_ecs::ExecMetadata;
use pyo3::{prelude::*, types::PyBytes, IntoPyObjectExt};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Compiled IREE executable wrapper.
/// 
/// Contains the VMFB bytecode and metadata needed to execute the computation.
pub struct IREEExec {
    pub metadata: ExecMetadata,
    vmfb: Vec<u8>,
    /// Cached runtime context (lazily initialized on first run)
    runtime_cache: Option<IREERuntimeCache>,
}

impl Clone for IREEExec {
    fn clone(&self) -> Self {
        // Clone the bytecode and metadata, but not the runtime cache
        // (it will be re-initialized lazily when needed)
        Self {
            metadata: self.metadata.clone(),
            vmfb: self.vmfb.clone(),
            runtime_cache: None, // Cache is re-created lazily
        }
    }
}

/// Cached IREE runtime objects to avoid re-initialization on each run.
struct IREERuntimeCache {
    /// The SystemContext must be kept alive as it owns the VM modules.
    /// The function and device reference objects owned by this context.
    #[allow(dead_code)]
    context: PyObject,
    function: PyObject,
    device: PyObject,
}

impl IREEExec {
    /// Create a new IREEExec from metadata and compiled VMFB bytecode.
    pub fn new(metadata: ExecMetadata, vmfb: Vec<u8>) -> Self {
        Self {
            metadata,
            vmfb,
            runtime_cache: None,
        }
    }

    /// Get the VMFB bytecode.
    pub fn vmfb(&self) -> &[u8] {
        &self.vmfb
    }

    /// Get the execution metadata.
    pub fn metadata(&self) -> &ExecMetadata {
        &self.metadata
    }

    /// Initialize the IREE runtime and cache it for subsequent runs.
    fn init_runtime(&mut self, py: Python<'_>, device: &str) -> Result<(), Error> {
        if self.runtime_cache.is_some() {
            return Ok(());
        }

        let py_code = r#"
from iree import runtime as rt

def init_iree_runtime(vmfb_bytes, device_name):
    config = rt.Config(device_name)
    ctx = rt.SystemContext(config=config)
    vm_module = rt.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)
    
    # Get the main function from the vm_module directly
    vm_func = vm_module.lookup_function('main')
    
    # Create the function invoker
    invoker = rt.FunctionInvoker(ctx.vm_context, config.device, vm_func)
    
    return ctx, invoker, config.device
"#;

        let module = pyo3::types::PyModule::new(py, "iree_init")?;
        let globals = module.dict();
        let code_cstr = std::ffi::CString::new(py_code).expect("Python code C string");
        py.run(code_cstr.as_ref(), Some(&globals), None)?;

        let init_fn: Py<PyAny> = module.getattr("init_iree_runtime")?.into();
        
        let vmfb_bytes = PyBytes::new(py, &self.vmfb);
        let result = init_fn.call1(py, (vmfb_bytes, device))?;
        
        let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
        let context = tuple.get_item(0)?.into();
        let function = tuple.get_item(1)?.into();
        let device = tuple.get_item(2)?.into();

        self.runtime_cache = Some(IREERuntimeCache {
            context,
            function,
            device,
        });

        Ok(())
    }

    /// Run the compiled computation with the given inputs.
    /// 
    /// # Arguments
    /// * `py` - Python GIL token
    /// * `inputs` - Input arrays as PyObjects (numpy arrays)
    /// * `device` - IREE device string (e.g., "local-task", "cuda", "vulkan")
    /// 
    /// # Returns
    /// Vector of output arrays as PyObjects
    pub fn run(
        &mut self,
        py: Python<'_>,
        inputs: &[PyObject],
        device: &str,
    ) -> Result<Vec<PyObject>, Error> {
        // Initialize runtime if not already done
        self.init_runtime(py, device)?;

        let cache = self.runtime_cache.as_ref().unwrap();

        let py_code = r#"
from iree import runtime as rt
import numpy as np

def run_iree(func, device, inputs):
    # Convert inputs to device arrays
    device_inputs = [rt.asdevicearray(device, inp) for inp in inputs]
    
    # Execute
    results = func(*device_inputs)
    
    # Convert back to numpy
    if isinstance(results, (list, tuple)):
        output = []
        for r in results:
            if hasattr(r, 'to_host'):
                output.append(np.asarray(r.to_host()))
            else:
                output.append(np.asarray(r))
        return output
    else:
        return [np.asarray(results.to_host())]
"#;

        let module = pyo3::types::PyModule::new(py, "iree_run")?;
        let globals = module.dict();
        let code_cstr = std::ffi::CString::new(py_code).expect("Python code C string");
        py.run(code_cstr.as_ref(), Some(&globals), None)?;

        let run_fn: Py<PyAny> = module.getattr("run_iree")?.into();
        
        let inputs_list = pyo3::types::PyList::new(py, inputs)?;
        let result = run_fn.call1(py, (&cache.function, &cache.device, inputs_list))?;
        
        let result_list = result.downcast_bound::<pyo3::types::PyList>(py)?;
        let outputs: Vec<PyObject> = result_list
            .iter()
            .map(|item| item.into())
            .collect();

        Ok(outputs)
    }

    /// Save the VMFB to a directory for later use.
    pub fn save_to_dir(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        
        // Save metadata
        let metadata_file = File::create(path.join("metadata.json"))?;
        serde_json::to_writer(metadata_file, &self.metadata)?;
        
        // Save VMFB
        std::fs::write(path.join("module.vmfb"), &self.vmfb)?;
        
        Ok(())
    }

    /// Load an IREEExec from a directory.
    pub fn load_from_dir(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        
        // Load metadata
        let metadata_file = File::open(path.join("metadata.json"))?;
        let metadata: ExecMetadata = serde_json::from_reader(metadata_file)?;
        
        // Load VMFB
        let vmfb = std::fs::read(path.join("module.vmfb"))?;
        
        Ok(Self::new(metadata, vmfb))
    }
}

/// IREE World execution wrapper - manages world state and IREE execution.
pub struct IREEWorldExec {
    pub world: nox_ecs::World,
    pub tick_exec: IREEExec,
    pub startup_exec: Option<IREEExec>,
    pub profiler: nox_ecs::profile::Profiler,
    /// Device string for IREE runtime
    device: String,
}

impl IREEWorldExec {
    /// Create a new IREEWorldExec.
    pub fn new(
        world: nox_ecs::World,
        tick_exec: IREEExec,
        startup_exec: Option<IREEExec>,
        device: &str,
    ) -> Self {
        Self {
            world,
            tick_exec,
            startup_exec,
            profiler: Default::default(),
            device: device.to_string(),
        }
    }

    /// Get the current simulation tick.
    pub fn tick(&self) -> u64 {
        self.world.tick()
    }

    /// Fork this execution context (creates a copy with fresh state).
    pub fn fork(&self) -> Self {
        Self {
            world: self.world.clone(),
            tick_exec: self.tick_exec.clone(),
            startup_exec: self.startup_exec.clone(),
            profiler: self.profiler.clone(),
            device: self.device.clone(),
        }
    }

    /// Run a single tick of the simulation.
    pub fn run(&mut self, py: Python<'_>) -> Result<(), Error> {
        use std::time::Instant;
        
        let start = &mut Instant::now();
        
        // Collect inputs from world
        let inputs = self.collect_inputs(py)?;
        self.profiler.copy_to_client.observe(start);
        
        // Run startup exec if present (only on first tick)
        if let Some(ref mut startup_exec) = self.startup_exec.take() {
            let outputs = startup_exec.run(py, &inputs, &self.device)?;
            self.apply_outputs(py, &outputs)?;
        }
        
        // Run tick exec
        let outputs = self.tick_exec.run(py, &inputs, &self.device)?;
        self.profiler.execute_buffers.observe(start);
        
        // Apply outputs back to world
        self.apply_outputs(py, &outputs)?;
        self.profiler.copy_to_host.observe(start);
        
        // Advance tick
        self.world.advance_tick();
        self.profiler.add_to_history.observe(start);
        
        Ok(())
    }

    /// Collect input arrays from the world state.
    fn collect_inputs(&self, py: Python<'_>) -> Result<Vec<PyObject>, Error> {
        use numpy::PyArrayMethods;
        use zerocopy::FromBytes;
        
        let mut inputs = Vec::new();
        for id in self.tick_exec.metadata.arg_ids.iter() {
            let col = self.world.column_by_id(*id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            
            let schema = col.schema;
            let data = col.column;
            let mut dim: Vec<usize> = schema.dim.iter().map(|&x| x as usize).collect();
            dim.insert(0, col.entities.len() / 8);
            
            // Convert to numpy array based on type
            let arr = match schema.prim_type {
                impeller2::types::PrimType::F32 => {
                    let slice = <[f32]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::F64 => {
                    let slice = <[f64]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::I32 => {
                    let slice = <[i32]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::I64 => {
                    let slice = <[i64]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::U32 => {
                    let slice = <[u32]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::U64 => {
                    let slice = <[u64]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::I8 => {
                    let slice = <[i8]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::I16 => {
                    let slice = <[i16]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::U8 => {
                    let slice = <[u8]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::U16 => {
                    let slice = <[u16]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
                impeller2::types::PrimType::Bool => {
                    let slice = <[u8]>::ref_from_bytes(data).unwrap();
                    let py_array = numpy::PyArray::from_slice(py, slice);
                    py_array.reshape(dim)?.into_py_any(py)?
                }
            };
            
            
            inputs.push(arr);
        }
        
        Ok(inputs)
    }

    /// Apply output arrays back to the world state.
    fn apply_outputs(&mut self, py: Python<'_>, outputs: &[PyObject]) -> Result<(), Error> {
        for (idx, id) in self.tick_exec.metadata.ret_ids.iter().enumerate() {
            if idx >= outputs.len() {
                tracing::warn!("apply_outputs: idx {} >= outputs.len() {}", idx, outputs.len());
                break;
            }
            
            let output = &outputs[idx];
            let host_buf = self.world.host.get_mut(id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            
            // Get numpy array and copy to host buffer using tobytes() method
            let arr_bound = output.bind(py);
            
            
            // Call the Python tobytes() method to get raw bytes
            let bytes_obj = arr_bound.call_method0("tobytes")?;
            let py_bytes = bytes_obj.downcast::<pyo3::types::PyBytes>()?;
            let bytes = py_bytes.as_bytes();
            
            host_buf.buffer.clear();
            host_buf.buffer.extend_from_slice(bytes);
        }
        
        Ok(())
    }

    /// Get profiling information.
    pub fn profile(&self) -> HashMap<&'static str, f64> {
        self.profiler.profile(self.world.sim_time_step().0)
    }
}
