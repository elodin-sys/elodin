use std::time::Instant;

use impeller2::types::ComponentId;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::profile::Profiler;
use crate::utils::SchemaExt;
use crate::world::World;

fn nox_to_iree_element_type(ty: nox::ElementType) -> Result<iree_runtime::ElementType, Error> {
    match ty {
        nox::ElementType::Pred => Ok(iree_runtime::ElementType::Bool),
        nox::ElementType::S8 => Ok(iree_runtime::ElementType::Int8),
        nox::ElementType::S16 => Ok(iree_runtime::ElementType::Int16),
        nox::ElementType::S32 => Ok(iree_runtime::ElementType::Int32),
        nox::ElementType::S64 => Ok(iree_runtime::ElementType::Int64),
        nox::ElementType::U8 => Ok(iree_runtime::ElementType::Uint8),
        nox::ElementType::U16 => Ok(iree_runtime::ElementType::Uint16),
        nox::ElementType::U32 => Ok(iree_runtime::ElementType::Uint32),
        nox::ElementType::U64 => Ok(iree_runtime::ElementType::Uint64),
        nox::ElementType::F16 => Ok(iree_runtime::ElementType::Float16),
        nox::ElementType::F32 => Ok(iree_runtime::ElementType::Float32),
        nox::ElementType::Bf16 => Ok(iree_runtime::ElementType::BFloat16),
        nox::ElementType::F64 => Ok(iree_runtime::ElementType::Float64),
        nox::ElementType::C64 | nox::ElementType::C128 => Err(Error::IreeCompilationFailed(
            "complex types not supported by IREE".to_string(),
        )),
    }
}

pub struct IREEExec {
    pub metadata: ExecMetadata,
    vmfb: Vec<u8>,
    session: iree_runtime::Session,
    #[allow(dead_code)]
    instance: iree_runtime::Instance,
}

unsafe impl Send for IREEExec {}
unsafe impl Sync for IREEExec {}

impl IREEExec {
    pub fn new(vmfb: &[u8], metadata: ExecMetadata) -> Result<Self, Error> {
        let instance =
            iree_runtime::Instance::new().map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        let device = instance
            .create_device("local-task")
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        let session = iree_runtime::Session::new(&instance, &device)
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        session
            .load_vmfb(vmfb)
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        Ok(Self {
            metadata,
            vmfb: vmfb.to_vec(),
            session,
            instance,
        })
    }

    pub fn invoke(
        &self,
        inputs: &[InputBuffer<'_>],
        num_outputs: usize,
    ) -> Result<Vec<Vec<u8>>, Error> {
        let mut call = self
            .session
            .call("module.main")
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

        for input in inputs {
            let buf = iree_runtime::BufferView::from_bytes(
                &self.session,
                input.data,
                &input.shape,
                input.element_type,
            )
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            call.push_input(&buf)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        }

        call.invoke()
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

        let mut outputs = Vec::with_capacity(num_outputs);
        for _ in 0..num_outputs {
            let output = call
                .pop_output()
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            outputs.push(
                output
                    .to_bytes()
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            );
        }

        Ok(outputs)
    }
}

pub struct InputBuffer<'a> {
    pub data: &'a [u8],
    pub shape: Vec<i64>,
    pub element_type: iree_runtime::ElementType,
}

pub struct IREEWorldExec {
    pub world: World,
    pub tick_exec: IREEExec,
    pub startup_exec: Option<IREEExec>,
    pub profiler: Profiler,
}

impl IREEWorldExec {
    pub fn new(world: World, tick_exec: IREEExec, startup_exec: Option<IREEExec>) -> Self {
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
            let outputs = startup_exec.invoke(&inputs, ret_ids.len())?;
            self.apply_outputs(&ret_ids, outputs)?;
        }

        let arg_ids = self.tick_exec.metadata.arg_ids.clone();
        let ret_ids = self.tick_exec.metadata.ret_ids.clone();
        let inputs = self.collect_inputs(&arg_ids)?;
        let outputs = self.tick_exec.invoke(&inputs, ret_ids.len())?;
        self.apply_outputs(&ret_ids, outputs)?;
        self.profiler.execute_buffers.observe(start);

        self.world.advance_tick();
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    fn collect_inputs(&self, arg_ids: &[ComponentId]) -> Result<Vec<InputBuffer<'_>>, Error> {
        let mut visited = std::collections::HashSet::new();
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
            let iree_elem = nox_to_iree_element_type(nox_elem)?;
            let shape: Vec<i64> = std::iter::once(col.len() as i64)
                .chain(col.schema.shape().iter().map(|&x| x as i64))
                .collect();
            inputs.push(InputBuffer {
                data: col.column.as_ref(),
                shape,
                element_type: iree_elem,
            });
        }

        Ok(inputs)
    }

    fn apply_outputs(
        &mut self,
        ret_ids: &[ComponentId],
        outputs: Vec<Vec<u8>>,
    ) -> Result<(), Error> {
        let mut visited = std::collections::HashSet::new();
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

    pub fn fork(&self) -> Self {
        let tick_exec = IREEExec::new(&self.tick_exec.vmfb, self.tick_exec.metadata.clone())
            .expect("failed to fork IREE exec");
        Self {
            world: self.world.clone(),
            tick_exec,
            startup_exec: None,
            profiler: self.profiler.clone(),
        }
    }

    pub fn profile(&self) -> std::collections::HashMap<&'static str, f64> {
        self.profiler.profile(self.world.sim_time_step().0)
    }
}
