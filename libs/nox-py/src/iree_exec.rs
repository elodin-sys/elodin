use std::collections::HashSet;
use std::time::Instant;

use impeller2::types::ComponentId;

use crate::error::Error;
use crate::exec::ExecMetadata;
use crate::iree_compile::IreeCompileStats;
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

#[derive(Clone, Copy)]
enum InputArenaKind {
    Constant,
    Mutable,
}

#[derive(Clone, Copy)]
struct InputBinding {
    arena_kind: InputArenaKind,
    slot: usize,
}

pub struct IREEExec {
    pub metadata: ExecMetadata,
    pub compile_stats: Option<IreeCompileStats>,
    vmfb: Vec<u8>,
    device_uri: String,
    mutable_input_ids: Vec<ComponentId>,
    output_ids: Vec<ComponentId>,
    input_bindings: Vec<InputBinding>,
    constant_input_arena: Option<iree_runtime::DeviceArena>,
    mutable_input_arena: Option<iree_runtime::DeviceArena>,
    output_arena: Option<iree_runtime::DeviceArena>,
    session: iree_runtime::Session,
    #[allow(dead_code)]
    instance: iree_runtime::Instance,
}

unsafe impl Send for IREEExec {}
unsafe impl Sync for IREEExec {}

impl IREEExec {
    pub fn new(
        vmfb: &[u8],
        metadata: ExecMetadata,
        compile_stats: Option<IreeCompileStats>,
        device_uri: &str,
        world: &World,
    ) -> Result<Self, Error> {
        let instance =
            iree_runtime::Instance::new().map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        let device = match instance.create_device(device_uri) {
            Ok(device) => device,
            Err(primary_err) if device_uri != "local-task" => instance
                .create_device("local-task")
                .map_err(|fallback_err| {
                    Error::IreeRuntimeError(format!(
                        "failed to create requested device '{device_uri}': {primary_err}; \
                         local-task fallback also failed: {fallback_err}"
                    ))
                })?,
            Err(primary_err) => return Err(Error::IreeRuntimeError(primary_err.to_string())),
        };
        let session = iree_runtime::Session::new(&instance, &device)
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        session
            .load_vmfb(vmfb)
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

        let mut output_ids = Vec::new();
        let mut seen_outputs = HashSet::new();
        for id in &metadata.ret_ids {
            if seen_outputs.insert(*id) {
                output_ids.push(*id);
            }
        }
        let output_set: HashSet<ComponentId> = output_ids.iter().copied().collect();

        let mut input_bindings = Vec::new();
        let mut mutable_input_ids = Vec::new();
        let mut const_specs = Vec::new();
        let mut mutable_specs = Vec::new();
        let mut const_ids = Vec::new();
        let mut seen_inputs = HashSet::new();
        for id in &metadata.arg_ids {
            if !seen_inputs.insert(*id) {
                continue;
            }
            let spec = build_buffer_spec(world, *id)?;
            if output_set.contains(id) {
                let slot = mutable_specs.len();
                mutable_specs.push(spec);
                mutable_input_ids.push(*id);
                input_bindings.push(InputBinding {
                    arena_kind: InputArenaKind::Mutable,
                    slot,
                });
            } else {
                let slot = const_specs.len();
                const_specs.push(spec);
                const_ids.push(*id);
                input_bindings.push(InputBinding {
                    arena_kind: InputArenaKind::Constant,
                    slot,
                });
            }
        }

        let mut constant_input_arena = if const_specs.is_empty() {
            None
        } else {
            Some(
                iree_runtime::DeviceArena::new(&session, &const_specs)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            )
        };
        if let Some(arena) = &mut constant_input_arena {
            let mut slices = Vec::with_capacity(const_ids.len());
            for id in &const_ids {
                let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
                slices.push(col.column.as_slice());
            }
            arena
                .upload_all(&session, &slices)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        }

        let mutable_input_arena = if mutable_specs.is_empty() {
            None
        } else {
            Some(
                iree_runtime::DeviceArena::new(&session, &mutable_specs)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            )
        };

        let output_arena = if output_ids.is_empty() {
            None
        } else {
            let mut specs = Vec::with_capacity(output_ids.len());
            for id in &output_ids {
                specs.push(build_buffer_spec(world, *id)?);
            }
            Some(
                iree_runtime::DeviceArena::new(&session, &specs)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?,
            )
        };

        Ok(Self {
            metadata,
            compile_stats,
            vmfb: vmfb.to_vec(),
            device_uri: device_uri.to_string(),
            mutable_input_ids,
            output_ids,
            input_bindings,
            constant_input_arena,
            mutable_input_arena,
            output_arena,
            session,
            instance,
        })
    }

    pub fn invoke_in_place(&mut self, world: &mut World) -> Result<(), Error> {
        if let Some(arena) = &mut self.mutable_input_arena {
            let mut slices = Vec::with_capacity(self.mutable_input_ids.len());
            for id in &self.mutable_input_ids {
                let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
                slices.push(col.column.as_slice());
            }
            arena
                .upload_all(&self.session, &slices)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        }

        let mut call = self
            .session
            .call("module.main")
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        for binding in &self.input_bindings {
            let view = match binding.arena_kind {
                InputArenaKind::Constant => self
                    .constant_input_arena
                    .as_ref()
                    .ok_or_else(|| Error::IreeRuntimeError("missing constant arena".into()))?
                    .view(binding.slot),
                InputArenaKind::Mutable => self
                    .mutable_input_arena
                    .as_ref()
                    .ok_or_else(|| Error::IreeRuntimeError("missing mutable arena".into()))?
                    .view(binding.slot),
            };
            call.push_input(view)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
        }
        call.invoke()
            .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;

        if let Some(arena) = &self.output_arena {
            for slot in 0..arena.len() {
                let output = call
                    .pop_output()
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
                arena
                    .copy_slot_from_view(&self.session, slot, &output)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            }
        } else {
            for id in &self.output_ids {
                let output = call
                    .pop_output()
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
                let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
                output
                    .download_into(&self.session, &mut host.buffer)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            }
            return Ok(());
        }

        if let Some(arena) = &mut self.output_arena {
            arena
                .download_all(&self.session)
                .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            for (slot, id) in self.output_ids.iter().enumerate() {
                let host = world.host.get_mut(id).ok_or(Error::ComponentNotFound)?;
                arena
                    .copy_slot_to_host(slot, &mut host.buffer)
                    .map_err(|e| Error::IreeRuntimeError(e.to_string()))?;
            }
        }
        Ok(())
    }
}

fn build_buffer_spec(world: &World, id: ComponentId) -> Result<iree_runtime::BufferSpec, Error> {
    let col = world.column_by_id(id).ok_or(Error::ComponentNotFound)?;
    let element_type = nox_to_iree_element_type(col.schema.element_type())?;
    let shape: Vec<i64> = std::iter::once(col.len() as i64)
        .chain(col.schema.shape().iter().map(|&x| x as i64))
        .collect();
    Ok(iree_runtime::BufferSpec {
        byte_len: col.column.len(),
        shape,
        element_type,
    })
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

        if let Some(mut startup_exec) = self.startup_exec.take() {
            startup_exec.invoke_in_place(&mut self.world)?;
        }

        self.tick_exec.invoke_in_place(&mut self.world)?;
        self.profiler.execute_buffers.observe(start);

        self.world.advance_tick();
        self.profiler.add_to_history.observe(start);
        Ok(())
    }

    pub fn fork(&self) -> Self {
        let tick_exec = IREEExec::new(
            &self.tick_exec.vmfb,
            self.tick_exec.metadata.clone(),
            self.tick_exec.compile_stats.clone(),
            &self.tick_exec.device_uri,
            &self.world,
        )
        .expect("failed to fork IREE exec");
        Self {
            world: self.world.clone(),
            tick_exec,
            startup_exec: None,
            profiler: self.profiler.clone(),
        }
    }

    pub fn profile(&self) -> std::collections::HashMap<&'static str, f64> {
        let mut profile = self.profiler.profile(self.world.sim_time_step().0);
        if let Some(stats) = &self.tick_exec.compile_stats {
            profile.insert(
                "compile",
                stats.lower_ms + stats.stablehlo_emit_ms + stats.iree_compile_ms,
            );
            profile.insert("iree_lower", stats.lower_ms);
            profile.insert("iree_stablehlo_emit", stats.stablehlo_emit_ms);
            profile.insert("iree_compile", stats.iree_compile_ms);
            profile.insert("iree_vmfb_size_bytes", stats.vmfb_size_bytes as f64);
        }
        profile
    }
}
