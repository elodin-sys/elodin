use std::collections::HashSet;
use std::time::Instant;

use impeller2::types::ComponentId;
use pyo3::prelude::*;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::profile::Profiler;
use crate::system::CompiledSystem;
use crate::utils::SchemaExt;
use crate::world::World;

pub struct JaxExec {
    pub metadata: ExecMetadata,
    jax_fn: PyObject,
}

impl JaxExec {
    pub fn new(py: Python<'_>, compiled_system: &CompiledSystem) -> Result<Self, Error> {
        use crate::system::CompiledSystemExt;
        let jax_fn = compiled_system.compile_jax_module(py)?;
        Ok(Self {
            metadata: ExecMetadata {
                arg_ids: compiled_system.inputs.clone(),
                ret_ids: compiled_system.outputs.clone(),
            },
            jax_fn,
        })
    }

    pub fn invoke(&self, inputs: &[JaxInput<'_>]) -> Result<Vec<Vec<u8>>, Error> {
        Python::with_gil(|py| {
            let jnp = py.import("jax.numpy")?;
            let np = py.import("numpy")?;

            let input_arrays = inputs
                .iter()
                .map(|input| -> Result<_, Error> {
                    let np_dtype = numpy_dtype_str(input.element_type)?;
                    let np_arr = np.call_method1(
                        "frombuffer",
                        (pyo3::types::PyBytes::new(py, input.data), np_dtype),
                    )?;
                    let shape_tuple: Vec<i64> = input.shape.clone();
                    let np_arr = np_arr.call_method1("reshape", (shape_tuple,))?;
                    Ok(jnp.call_method1("array", (np_arr,))?)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let py_args = pyo3::types::PyTuple::new(py, input_arrays)?;

            let result = self.jax_fn.call(py, &py_args, None)?;

            let num_outputs = self.metadata.ret_ids.iter().collect::<HashSet<_>>().len();
            let mut outputs = Vec::with_capacity(num_outputs);

            if num_outputs == 1 {
                let np_arr = np.call_method1("asarray", (result.bind(py),))?;
                let bytes_obj = np_arr.call_method1("tobytes", ())?;
                let bytes: &[u8] = bytes_obj.extract()?;
                outputs.push(bytes.to_vec());
            } else {
                for i in 0..num_outputs {
                    let item = result.call_method1(py, "__getitem__", (i,))?;
                    let np_arr = np.call_method1("asarray", (item.bind(py),))?;
                    let bytes_obj = np_arr.call_method1("tobytes", ())?;
                    let bytes: &[u8] = bytes_obj.extract()?;
                    outputs.push(bytes.to_vec());
                }
            }

            Ok(outputs)
        })
    }
}

fn numpy_dtype_str(et: nox::ElementType) -> Result<&'static str, Error> {
    match et {
        nox::ElementType::F64 => Ok("float64"),
        nox::ElementType::F32 => Ok("float32"),
        nox::ElementType::S32 => Ok("int32"),
        nox::ElementType::S64 => Ok("int64"),
        nox::ElementType::U32 => Ok("uint32"),
        nox::ElementType::U64 => Ok("uint64"),
        nox::ElementType::U8 => Ok("uint8"),
        nox::ElementType::U16 => Ok("uint16"),
        nox::ElementType::S8 => Ok("int8"),
        nox::ElementType::S16 => Ok("int16"),
        nox::ElementType::Pred => Ok("bool"),
        other => Err(Error::UnsupportedDtype(format!("{other:?}"))),
    }
}

pub struct JaxInput<'a> {
    pub data: &'a [u8],
    pub shape: Vec<i64>,
    pub element_type: nox::ElementType,
}

pub struct JaxWorldExec {
    pub world: World,
    pub tick_exec: JaxExec,
    pub startup_exec: Option<JaxExec>,
    pub profiler: Profiler,
}

impl JaxWorldExec {
    pub fn new(world: World, tick_exec: JaxExec, startup_exec: Option<JaxExec>) -> Self {
        Self {
            world,
            tick_exec,
            startup_exec,
            profiler: Default::default(),
        }
    }

    pub fn tick(&self) -> u64 {
        self.world.tick()
    }

    pub fn run(&mut self) -> Result<(), Error> {
        let start = &mut Instant::now();

        if let Some(startup_exec) = self.startup_exec.take() {
            let arg_ids = startup_exec.metadata.arg_ids.clone();
            let ret_ids = startup_exec.metadata.ret_ids.clone();
            let inputs = self.collect_inputs(&arg_ids)?;
            let outputs = startup_exec.invoke(&inputs)?;
            self.apply_outputs(&ret_ids, outputs)?;
        }

        let arg_ids = self.tick_exec.metadata.arg_ids.clone();
        let ret_ids = self.tick_exec.metadata.ret_ids.clone();
        let inputs = self.collect_inputs(&arg_ids)?;
        let outputs = self.tick_exec.invoke(&inputs)?;
        self.apply_outputs(&ret_ids, outputs)?;
        self.profiler.execute_buffers.observe(start);

        self.world.advance_tick();
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    fn collect_inputs(&self, arg_ids: &[ComponentId]) -> Result<Vec<JaxInput<'_>>, Error> {
        let mut visited = HashSet::new();
        let mut inputs = Vec::new();

        for id in arg_ids {
            if !visited.insert(id) {
                continue;
            }
            let col = self
                .world
                .column_by_id(*id)
                .ok_or(Error::ComponentNotFound)?;
            let nox_elem = col.schema.element_type();
            let shape: Vec<i64> = std::iter::once(col.len() as i64)
                .chain(col.schema.shape().iter().map(|&x| x as i64))
                .collect();
            inputs.push(JaxInput {
                data: col.column.as_ref(),
                shape,
                element_type: nox_elem,
            });
        }

        Ok(inputs)
    }

    fn apply_outputs(
        &mut self,
        ret_ids: &[ComponentId],
        outputs: Vec<Vec<u8>>,
    ) -> Result<(), Error> {
        let mut visited = HashSet::new();
        let mut output_idx = 0;

        for id in ret_ids {
            if !visited.insert(id) {
                continue;
            }
            if output_idx >= outputs.len() {
                return Err(Error::UnexpectedInput);
            }
            let output_bytes = &outputs[output_idx];
            output_idx += 1;

            let host_buf = self
                .world
                .host
                .get_mut(id)
                .ok_or(Error::ComponentNotFound)?;
            if output_bytes.len() != host_buf.buffer.len() {
                return Err(Error::ValueSizeMismatch);
            }
            host_buf.buffer.copy_from_slice(output_bytes);
        }

        Ok(())
    }

    pub fn profile(&self) -> std::collections::HashMap<&'static str, f64> {
        self.profiler.profile(self.world.sim_time_step().0)
    }
}
