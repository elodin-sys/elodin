use std::collections::HashSet;
use std::time::{Duration, Instant};

use impeller2::types::ComponentId;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::profile::{Profiler, TickTimings};
use crate::system::CompiledSystem;
use crate::utils::SchemaExt;
use crate::world::World;

struct InputSlot {
    component_id: ComponentId,
    shape: Vec<i64>,
    np_dtype: &'static str,
}

pub struct JaxExec {
    pub metadata: ExecMetadata,
    cpu_backend: bool,
    jax_fn: PyObject,
    np: PyObject,
    jnp: PyObject,
    jax: PyObject,
    inputs: Vec<InputSlot>,
    output_ids: Vec<ComponentId>,
    mutable_overlap: Vec<(usize, usize)>,
}

impl JaxExec {
    pub fn new(
        py: Python<'_>,
        compiled_system: &CompiledSystem,
        world: &World,
        gpu: bool,
    ) -> Result<Self, Error> {
        use crate::system::CompiledSystemExt;

        let np = py.import("numpy")?;
        let jnp = py.import("jax.numpy")?;
        let jax = py.import("jax")?;
        if gpu {
            let gpu_devices = jax.call_method1("devices", ("gpu",))?;
            let count: usize = gpu_devices.call_method0("__len__")?.extract()?;
            if count == 0 {
                return Err(Error::UnknownCommand(
                    "backend='jax-gpu' requested but no GPU device is available".to_string(),
                ));
            }
        }

        let metadata = ExecMetadata {
            arg_ids: compiled_system.inputs.clone(),
            ret_ids: compiled_system.outputs.clone(),
        };
        let mut output_ids = Vec::new();
        let mut seen_outputs = HashSet::new();
        for id in &metadata.ret_ids {
            if seen_outputs.insert(*id) {
                output_ids.push(*id);
            }
        }

        let mut inputs = Vec::new();
        let mut seen_inputs = HashSet::new();

        for id in &metadata.arg_ids {
            if !seen_inputs.insert(*id) {
                continue;
            }

            let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
            let np_dtype = numpy_dtype_str(col.schema.element_type())?;
            let shape: Vec<i64> = std::iter::once(col.len() as i64)
                .chain(col.schema.shape().iter().map(|&x| x as i64))
                .collect();
            inputs.push(InputSlot {
                component_id: *id,
                shape,
                np_dtype,
            });
        }

        let mut output_slot_by_id = std::collections::HashMap::new();
        for (slot, id) in output_ids.iter().enumerate() {
            output_slot_by_id.insert(*id, slot);
        }
        let mutable_overlap = inputs
            .iter()
            .enumerate()
            .filter_map(|(input_slot, slot)| {
                output_slot_by_id
                    .get(&slot.component_id)
                    .copied()
                    .map(|output_slot| (input_slot, output_slot))
            })
            .collect();
        let donate_argnums: Vec<usize> = Vec::new();
        let target_backend = if gpu { Some("gpu") } else { Some("cpu") };
        let jax_fn = compiled_system.compile_jax_module(py, &donate_argnums, target_backend)?;

        Ok(Self {
            metadata,
            cpu_backend: !gpu,
            jax_fn,
            np: np.unbind().into(),
            jnp: jnp.unbind().into(),
            jax: jax.unbind().into(),
            inputs,
            output_ids,
            mutable_overlap,
        })
    }

    pub fn invoke_in_place(
        &mut self,
        world: &mut World,
        detailed: bool,
    ) -> Result<TickTimings, Error> {
        self.invoke_batch(world, 1, detailed)
    }

    pub fn invoke_batch(
        &mut self,
        world: &mut World,
        n: u64,
        detailed: bool,
    ) -> Result<TickTimings, Error> {
        let batch_ticks = n.max(1) as usize;
        Python::with_gil(|py| {
            let np = self.np.bind(py);
            let jnp = self.jnp.bind(py);
            let jax = self.jax.bind(py);

            let h2d_start = detailed.then(Instant::now);
            let mut device_inputs = Vec::with_capacity(self.inputs.len());
            for input in &self.inputs {
                let col = world
                    .column_by_id(input.component_id)
                    .ok_or(Error::ComponentNotFound)?;
                let src = np.call_method1(
                    "frombuffer",
                    (PyBytes::new(py, col.column.as_slice()), input.np_dtype),
                )?;
                let src = src.call_method1("reshape", (input.shape.clone(),))?;
                if self.cpu_backend {
                    device_inputs.push(src.unbind());
                } else {
                    let dev = jnp.call_method1("array", (src,))?;
                    device_inputs.push(dev.unbind());
                }
            }
            let h2d_upload_ms = h2d_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default();

            let kernel_start = detailed.then(Instant::now);
            let mut final_outputs: Vec<PyObject> = Vec::new();
            for batch_idx in 0..batch_ticks {
                let py_args = PyTuple::new(py, device_inputs.iter().map(|arg| arg.clone_ref(py)))?;
                let result = self.jax_fn.call(py, &py_args, None)?;
                jax.call_method1("block_until_ready", (result.bind(py),))?;
                let outputs = extract_outputs(result.bind(py), self.output_ids.len())?;
                if batch_idx + 1 < batch_ticks {
                    for (input_slot, output_slot) in &self.mutable_overlap {
                        device_inputs[*input_slot] = outputs[*output_slot].clone_ref(py);
                    }
                } else {
                    final_outputs = outputs;
                }
            }
            let kernel_invoke_ms = kernel_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default();

            let d2h_start = detailed.then(Instant::now);
            for (idx, id) in self.output_ids.iter().enumerate() {
                copy_output_to_world(py, np, world, *id, final_outputs[idx].bind(py))?;
            }
            let d2h_download_ms = d2h_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or_default();

            Ok(TickTimings {
                h2d_upload_ms,
                call_setup_ms: 0.0,
                kernel_invoke_ms,
                d2h_download_ms,
            })
        })
    }
}

fn extract_outputs(result: &Bound<'_, PyAny>, output_count: usize) -> Result<Vec<PyObject>, Error> {
    if output_count == 0 {
        return Ok(Vec::new());
    }
    if output_count == 1 {
        return Ok(vec![result.clone().unbind().into()]);
    }
    let mut outputs = Vec::with_capacity(output_count);
    for idx in 0..output_count {
        outputs.push(result.call_method1("__getitem__", (idx,))?.unbind().into());
    }
    Ok(outputs)
}
fn copy_output_to_world(
    _py: Python<'_>,
    np: &Bound<'_, PyAny>,
    world: &mut World,
    id: ComponentId,
    value: &Bound<'_, PyAny>,
) -> Result<(), Error> {
    let np_arr = np.call_method1("asarray", (value,))?;
    let bytes_obj = np_arr.call_method1("tobytes", ())?;
    let bytes: &[u8] = bytes_obj.extract()?;
    let host_buf = world.host.get_mut(&id).ok_or(Error::ComponentNotFound)?;
    if bytes.len() != host_buf.buffer.len() {
        return Err(Error::ValueSizeMismatch);
    }
    host_buf.buffer.copy_from_slice(bytes);
    Ok(())
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
        let ticks_per_telemetry = self.world.ticks_per_telemetry();

        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.invoke_in_place(&mut self.world, self.profiler.detailed_timing)?;
        }

        let tick_start = Instant::now();
        let timings = self.tick_exec.invoke_batch(
            &mut self.world,
            ticks_per_telemetry,
            self.profiler.detailed_timing,
        )?;
        let tick_elapsed = tick_start.elapsed();
        if self.profiler.detailed_timing {
            let h2d = Duration::from_secs_f64(timings.h2d_upload_ms / 1000.0);
            let call_setup = Duration::from_secs_f64(timings.call_setup_ms / 1000.0);
            let kernel = Duration::from_secs_f64(timings.kernel_invoke_ms / 1000.0);
            let d2h = Duration::from_secs_f64(timings.d2h_download_ms / 1000.0);
            self.profiler.copy_to_client.observe_duration(h2d);
            self.profiler.execute_buffers.observe_duration(kernel);
            self.profiler.copy_to_host.observe_duration(d2h);
            self.profiler.h2d_upload.observe_duration(h2d);
            self.profiler.call_setup.observe_duration(call_setup);
            self.profiler.kernel_invoke.observe_duration(kernel);
            self.profiler.d2h_download.observe_duration(d2h);
        } else {
            self.profiler.execute_buffers.observe_duration(tick_elapsed);
        }

        *start = Instant::now();
        for _ in 0..ticks_per_telemetry {
            self.world.advance_tick();
        }
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    pub fn profile(&self) -> std::collections::HashMap<&'static str, f64> {
        self.profiler.profile(
            self.world.sim_time_step().0,
            self.world.ticks_per_telemetry(),
        )
    }
}
