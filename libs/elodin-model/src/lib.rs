//! A simulation exported from Elodin with `JaxSim.export_model`, behind the C interface of
//! `include/elodin_model.h`. `build.rs` embeds the model and picks how its code is compiled.
// Without a model only the tests load programs.
#![cfg_attr(not(any(elodin_model = "aot", elodin_model = "jit")), allow(dead_code))]

mod hook_adapter;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use cranelift_mlir::display_kernel::DisplayKernelExec;
use serde::Deserialize;

/// One tick: a buffer per component in, the next state's buffers out, never overlapping.
type Step = Box<dyn Fn(&[&[u8]], &mut [&mut [u8]]) -> Result<(), String>>;

const ELODIN_MODEL_OK: c_int = 0;
const ELODIN_MODEL_ERR_ARGUMENT: c_int = 1;
const ELODIN_MODEL_ERR_STATE: c_int = 2;
const ELODIN_MODEL_ERR_STEP: c_int = 3;

const ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP: u32 = 1 << 0;
const ELODIN_MODEL_COMPONENT_EXTERNAL_CONTROL: u32 = 1 << 1;
const ELODIN_MODEL_COMPONENT_TRANSIENT: u32 = 1 << 2;
const ELODIN_MODEL_COMPONENT_PRIVATE: u32 = 1 << 3;
const ELODIN_MODEL_COMPONENT_TICK: u32 = 1 << 4;
const ELODIN_MODEL_COMPONENT_TIME_STEP: u32 = 1 << 5;

#[derive(Deserialize)]
struct Schema {
    #[serde(default = "model_format")]
    format: String,
    #[serde(default = "abi_v1")]
    abi_version: u32,
    #[serde(default = "little_endian")]
    endianness: String,
    /// The worker count the step's parallel groups were compiled for; only `step.o` has groups.
    #[serde(default = "one")]
    #[cfg_attr(not(elodin_model = "aot"), allow(dead_code))]
    workers: usize,
    #[serde(default)]
    simulation: SimulationSchema,
    components: Vec<ComponentSchema>,
    #[serde(default)]
    interfaces: Vec<InterfaceSchema>,
}

fn one() -> usize {
    1
}

fn abi_v1() -> u32 {
    1
}

fn model_format() -> String {
    "elodin.model".into()
}

fn little_endian() -> String {
    "little".into()
}

#[derive(Default, Deserialize)]
struct SimulationSchema {
    #[serde(default)]
    rate_hz: f64,
    #[serde(default)]
    time_step_s: f64,
    #[serde(default)]
    tick_component: Option<String>,
    #[serde(default)]
    time_step_component: Option<String>,
    #[serde(default)]
    tick_advanced_by_step: bool,
}

#[derive(Deserialize)]
struct ComponentSchema {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    #[serde(default)]
    element_shape: Vec<usize>,
    #[serde(default)]
    entity_axis: String,
    /// The entity of each row.
    #[serde(default)]
    entities: Vec<String>,
    /// Absent from models exported before entity ids were.
    #[serde(default)]
    entity_ids: Option<Vec<u64>>,
    #[serde(default)]
    rows: usize,
    #[serde(default)]
    row_bytes: usize,
    #[serde(default)]
    size: usize,
    #[serde(default)]
    written_by_step: bool,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Deserialize)]
struct InterfaceSchema {
    name: String,
    #[serde(default)]
    rows: usize,
    fields: Vec<InterfaceFieldSchema>,
}

#[derive(Deserialize)]
struct InterfaceFieldSchema {
    name: String,
    component: String,
    access: String,
    entity_ids: Vec<u64>,
    row_bytes: usize,
}

struct Component {
    name: CString,
    dtype: CString,
    size: usize,
    shape: Vec<usize>,
    element_shape: Vec<usize>,
    row_bytes: usize,
    flags: u32,
    entity_axis: u32,
    /// Each row's entity id and name.
    entities: Vec<(u64, CString)>,
}

struct InterfaceField {
    name: CString,
    component: usize,
    rows: Vec<usize>,
    row_bytes: usize,
    writable: bool,
}

struct Interface {
    name: CString,
    rows: usize,
    fields: Vec<InterfaceField>,
}

struct Layout {
    format: CString,
    abi_version: u32,
    endianness: CString,
    simulation: SimulationSchema,
    workers: usize,
    components: Vec<Component>,
    interfaces: Vec<Interface>,
    initial: Vec<Vec<u8>>,
}

/// A model's code, its components and their initial values.
pub struct Program {
    format: CString,
    abi_version: u32,
    endianness: CString,
    simulation: SimulationSchema,
    workers: usize,
    components: Vec<Component>,
    interfaces: Vec<Interface>,
    initial: Vec<Vec<u8>>,
    step: Step,
}

/// The components `schema` describes, each with its initial value from `state`.
fn layout(schema: &str, state: &[u8]) -> Result<Layout, String> {
    let schema: Schema =
        serde_json::from_str(schema).map_err(|e| format!("invalid schema.json: {e}"))?;
    if schema.endianness != "little" {
        return Err(format!(
            "schema.json endianness must be little, not {:?}",
            schema.endianness
        ));
    }
    let mut components = Vec::new();
    let mut initial = Vec::new();
    let mut rest = state;
    for component in schema.components {
        let calculated_size =
            dtype_size(&component.dtype)? * component.shape.iter().product::<usize>();
        let size = if component.size == 0 {
            calculated_size
        } else if component.size == calculated_size {
            component.size
        } else {
            return Err(format!(
                "{} says it is {} bytes but its dtype and shape describe {calculated_size}",
                component.name, component.size
            ));
        };
        let (value, tail) = rest
            .split_at_checked(size)
            .ok_or("state.bin is shorter than schema.json describes")?;
        initial.push(value.to_vec());
        rest = tail;
        let ids = component
            .entity_ids
            .unwrap_or_else(|| vec![u64::MAX; component.entities.len()]);
        if component.entities.len() != ids.len() {
            return Err(format!(
                "{} has {} entity names and {} entity ids",
                component.name,
                component.entities.len(),
                ids.len()
            ));
        }
        let entities = ids.into_iter().zip(component.entities);
        let entities = entities
            .map(|(id, name)| Ok((id, CString::new(name).map_err(|e| e.to_string())?)))
            .collect::<Result<Vec<_>, String>>()?;
        let rows = entities.len();
        if component.rows != 0 && component.rows != rows {
            return Err(format!(
                "{} says it has {} rows but lists {rows} entities",
                component.name, component.rows
            ));
        }
        let row_bytes = if component.row_bytes == 0 {
            size.checked_div(rows)
                .filter(|row| row * rows == size)
                .ok_or_else(|| format!("{} cannot be divided into {rows} rows", component.name))?
        } else {
            component.row_bytes
        };
        if row_bytes * rows != size {
            return Err(format!(
                "{} has {rows} rows of {row_bytes} bytes but is {size} bytes",
                component.name
            ));
        }
        let element_shape = if component.element_shape.is_empty() {
            if rows > 1 {
                component.shape.get(1..).unwrap_or_default().to_vec()
            } else {
                component.shape.clone()
            }
        } else {
            component.element_shape
        };
        let entity_axis = match component.entity_axis.as_str() {
            "" => u32::from(rows > 1),
            "none" => 0,
            "leading" => 1,
            other => {
                return Err(format!(
                    "{} entity_axis must be none or leading, not {other:?}",
                    component.name
                ));
            }
        };
        let mut flags = 0;
        if component.written_by_step {
            flags |= ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP;
        }
        if component
            .metadata
            .get("external_control")
            .is_some_and(|value| value == "true")
        {
            flags |= ELODIN_MODEL_COMPONENT_EXTERNAL_CONTROL;
        }
        if component
            .metadata
            .get("transient")
            .is_some_and(|value| value == "true")
        {
            flags |= ELODIN_MODEL_COMPONENT_TRANSIENT;
        }
        if component
            .metadata
            .get("private")
            .is_some_and(|value| value == "true")
        {
            flags |= ELODIN_MODEL_COMPONENT_PRIVATE;
        }
        if component.name == "tick" {
            flags |= ELODIN_MODEL_COMPONENT_TICK;
        }
        if component.name == "simulation_time_step" {
            flags |= ELODIN_MODEL_COMPONENT_TIME_STEP;
        }
        components.push(Component {
            name: CString::new(component.name).map_err(|e| e.to_string())?,
            dtype: CString::new(component.dtype).map_err(|e| e.to_string())?,
            size,
            shape: component.shape,
            element_shape,
            row_bytes,
            flags,
            entity_axis,
            entities,
        });
    }
    if !rest.is_empty() {
        return Err("state.bin is longer than schema.json describes".into());
    }
    let component_index: HashMap<String, usize> = components
        .iter()
        .enumerate()
        .map(|(index, component)| (component.name.to_string_lossy().into_owned(), index))
        .collect();
    let mut interfaces = Vec::new();
    for interface in schema.interfaces {
        let mut fields = Vec::new();
        for field in interface.fields {
            let &component = component_index.get(&field.component).ok_or_else(|| {
                format!(
                    "interface {:?} has no component {:?}",
                    interface.name, field.component
                )
            })?;
            let column = &components[component];
            if field.row_bytes != column.row_bytes {
                return Err(format!(
                    "interface {:?} field {:?} row is {} bytes, not {}",
                    interface.name, field.name, field.row_bytes, column.row_bytes
                ));
            }
            let row_of: HashMap<u64, usize> = column
                .entities
                .iter()
                .enumerate()
                .map(|(row, (id, _))| (*id, row))
                .collect();
            let rows = field
                .entity_ids
                .iter()
                .map(|id| {
                    row_of.get(id).copied().ok_or_else(|| {
                        format!(
                            "interface {:?} field {:?}: entity {id} has no {:?}",
                            interface.name, field.name, field.component
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let writable = match field.access.as_str() {
                "read" => false,
                "write" => true,
                other => {
                    return Err(format!(
                        "interface {:?} field {:?} access must be read or write, not {other:?}",
                        interface.name, field.name
                    ));
                }
            };
            fields.push(InterfaceField {
                name: CString::new(field.name).map_err(|e| e.to_string())?,
                component,
                rows,
                row_bytes: field.row_bytes,
                writable,
            });
        }
        let rows = fields.first().map_or(0, |field| field.rows.len());
        if interface.rows != rows || fields.iter().any(|field| field.rows.len() != rows) {
            return Err(format!(
                "interface {:?} fields do not all have {} rows",
                interface.name, interface.rows
            ));
        }
        interfaces.push(Interface {
            name: CString::new(interface.name).map_err(|e| e.to_string())?,
            rows,
            fields,
        });
    }
    Ok(Layout {
        format: CString::new(schema.format).map_err(|e| e.to_string())?,
        abi_version: schema.abi_version,
        endianness: CString::new(schema.endianness).map_err(|e| e.to_string())?,
        simulation: schema.simulation,
        workers: schema.workers.max(1),
        components,
        interfaces,
        initial,
    })
}

fn dtype_size(dtype: &str) -> Result<usize, String> {
    Ok(match dtype {
        "float64" | "int64" | "uint64" => 8,
        "float32" | "int32" | "uint32" => 4,
        "int16" | "uint16" => 2,
        "int8" | "uint8" | "bool" => 1,
        other => return Err(format!("unsupported dtype {other}")),
    })
}

impl Program {
    /// A program whose `step` handles the components `schema` describes.
    fn new(schema: &str, state: &[u8], step: Step) -> Result<Self, String> {
        let layout = layout(schema, state)?;
        Ok(Self {
            format: layout.format,
            abi_version: layout.abi_version,
            endianness: layout.endianness,
            simulation: layout.simulation,
            workers: layout.workers,
            components: layout.components,
            interfaces: layout.interfaces,
            initial: layout.initial,
            step,
        })
    }

    /// Compiles `mlir`, whose `main` must map the state `schema` describes to its next value.
    #[cfg_attr(elodin_model = "aot", allow(dead_code))]
    fn jit(mlir: &str, schema: &str, state: &[u8]) -> Result<Self, String> {
        let layout = layout(schema, state)?;
        let module = cranelift_mlir::parser::parse_module(mlir)
            .map_err(|e| format!("StableHLO parse failed: {e}"))?;
        let main = module.main_func().ok_or("the StableHLO has no main")?;
        let sizes: Vec<usize> = layout.components.iter().map(|c| c.size).collect();
        let params: Vec<usize> = main.params.iter().map(|(_, ty)| ty.byte_size()).collect();
        let results: Vec<usize> = main.result_types.iter().map(|ty| ty.byte_size()).collect();
        if params != sizes || results != sizes {
            return Err(
                "main does not map the state schema.json describes to its next value".into(),
            );
        }
        let exec = Mutex::new(DisplayKernelExec::compile(mlir, &sizes, &sizes)?);
        let step = move |inputs: &[&[u8]], outputs: &mut [&mut [u8]]| {
            let mut exec = exec.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            exec.invoke(inputs, outputs).map_err(|e| e.to_string())
        };
        Self::new(schema, state, Box::new(step))
    }

    fn index(&self, name: &CStr) -> Option<usize> {
        self.components
            .iter()
            .position(|c| c.name.as_c_str() == name)
    }
}

/// One run of a program.
pub struct Instance<'a> {
    program: &'a Program,
    state: Vec<Vec<u8>>,
    /// Where the step writes, since its outputs must not overlap its inputs.
    next: Vec<Vec<u8>>,
    generation: u64,
}

impl<'a> Instance<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            program,
            state: program.initial.clone(),
            next: program.initial.clone(),
            generation: 0,
        }
    }

    fn reset(&mut self) {
        self.state.clone_from(&self.program.initial);
        self.generation = self.generation.wrapping_add(1);
    }

    fn step(&mut self) -> Result<(), String> {
        let inputs: Vec<&[u8]> = self.state.iter().map(Vec::as_slice).collect();
        let mut outputs: Vec<&mut [u8]> = self.next.iter_mut().map(Vec::as_mut_slice).collect();
        (self.program.step)(&inputs, &mut outputs)?;
        std::mem::swap(&mut self.state, &mut self.next);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    fn component(&self, component: usize, size: usize) -> Result<usize, String> {
        let expected = self
            .program
            .components
            .get(component)
            .ok_or_else(|| format!("no component {component}"))?
            .size;
        if size != expected {
            return Err(format!(
                "component {component} is {expected} bytes, not {size}"
            ));
        }
        Ok(component)
    }

    fn read(&self, component: usize, dst: &mut [u8]) -> Result<(), String> {
        let component = self.component(component, dst.len())?;
        dst.copy_from_slice(&self.state[component]);
        Ok(())
    }

    fn write(&mut self, component: usize, src: &[u8]) -> Result<(), String> {
        let component = self.component(component, src.len())?;
        self.state[component].copy_from_slice(src);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    fn read_rows(
        &self,
        component: usize,
        first: usize,
        count: usize,
        dst: &mut [u8],
    ) -> Result<(), String> {
        let column = self
            .program
            .components
            .get(component)
            .ok_or_else(|| format!("no component {component}"))?;
        let stop = first
            .checked_add(count)
            .filter(|stop| *stop <= column.entities.len())
            .ok_or_else(|| {
                format!(
                    "component {component} has no rows {first}..{}",
                    first + count
                )
            })?;
        let bytes = count
            .checked_mul(column.row_bytes)
            .ok_or("row byte count overflows")?;
        if dst.len() != bytes {
            return Err(format!("rows are {bytes} bytes, not {}", dst.len()));
        }
        let start = first * column.row_bytes;
        let end = stop * column.row_bytes;
        dst.copy_from_slice(&self.state[component][start..end]);
        Ok(())
    }

    fn write_rows(
        &mut self,
        component: usize,
        first: usize,
        count: usize,
        src: &[u8],
    ) -> Result<(), String> {
        let column = self
            .program
            .components
            .get(component)
            .ok_or_else(|| format!("no component {component}"))?;
        let stop = first
            .checked_add(count)
            .filter(|stop| *stop <= column.entities.len())
            .ok_or_else(|| {
                format!(
                    "component {component} has no rows {first}..{}",
                    first + count
                )
            })?;
        let bytes = count
            .checked_mul(column.row_bytes)
            .ok_or("row byte count overflows")?;
        if src.len() != bytes {
            return Err(format!("rows are {bytes} bytes, not {}", src.len()));
        }
        let start = first * column.row_bytes;
        let end = stop * column.row_bytes;
        self.state[component][start..end].copy_from_slice(src);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    fn interface_read(&self, interface: usize, field: usize, dst: &mut [u8]) -> Result<(), String> {
        let field = self
            .program
            .interfaces
            .get(interface)
            .ok_or_else(|| format!("no interface {interface}"))?
            .fields
            .get(field)
            .ok_or_else(|| format!("interface {interface} has no field {field}"))?;
        let bytes = field
            .rows
            .len()
            .checked_mul(field.row_bytes)
            .ok_or("interface byte count overflows")?;
        if dst.len() != bytes {
            return Err(format!(
                "interface field is {bytes} bytes, not {}",
                dst.len()
            ));
        }
        if field.row_bytes == 0 {
            return Ok(());
        }
        for (out, &row) in dst.chunks_exact_mut(field.row_bytes).zip(&field.rows) {
            let start = row * field.row_bytes;
            out.copy_from_slice(&self.state[field.component][start..start + field.row_bytes]);
        }
        Ok(())
    }

    fn interface_write(
        &mut self,
        interface: usize,
        field_index: usize,
        src: &[u8],
    ) -> Result<(), String> {
        let field = self
            .program
            .interfaces
            .get(interface)
            .ok_or_else(|| format!("no interface {interface}"))?
            .fields
            .get(field_index)
            .ok_or_else(|| format!("interface {interface} has no field {field_index}"))?;
        if !field.writable {
            return Err(format!(
                "interface {interface} field {field_index} is read-only"
            ));
        }
        let bytes = field
            .rows
            .len()
            .checked_mul(field.row_bytes)
            .ok_or("interface byte count overflows")?;
        if src.len() != bytes {
            return Err(format!(
                "interface field is {bytes} bytes, not {}",
                src.len()
            ));
        }
        if field.row_bytes == 0 {
            return Ok(());
        }
        for (input, &row) in src.chunks_exact(field.row_bytes).zip(&field.rows) {
            let start = row * field.row_bytes;
            self.state[field.component][start..start + field.row_bytes].copy_from_slice(input);
        }
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }
}

#[repr(C)]
pub struct ElodinModelComponentType {
    /// Caller sets this to `sizeof(ElodinModelComponentType)`.
    pub struct_size: usize,
    pub dtype: *const c_char,
    pub shape: *const usize,
    pub rank: usize,
    pub element_shape: *const usize,
    pub element_rank: usize,
    pub element_bytes: usize,
    pub rows: usize,
    pub row_bytes: usize,
    pub total_bytes: usize,
    pub flags: u32,
    /// 0: no entity axis, 1: leading entity axis.
    pub entity_axis: u32,
}

#[repr(C)]
pub struct ElodinModelInterfaceField {
    /// Caller sets this to `sizeof(ElodinModelInterfaceField)`.
    pub struct_size: usize,
    pub name: *const c_char,
    pub component: usize,
    /// 0: read-only to the caller, 1: caller may write.
    pub writable: u32,
    pub rows: usize,
    pub row_bytes: usize,
}

#[cfg(any(elodin_model = "aot", elodin_model = "jit"))]
const SCHEMA: &str = include_str!(concat!(env!("ELODIN_MODEL_DIR"), "/schema.json"));
#[cfg(any(elodin_model = "aot", elodin_model = "jit"))]
const STATE: &[u8] = include_bytes!(concat!(env!("ELODIN_MODEL_DIR"), "/state.bin"));
#[cfg(any(elodin_model = "aot", elodin_model = "jit"))]
const BUILD_INFO: &str = include_str!(env!("ELODIN_MODEL_BUILD_INFO"));
#[cfg(not(any(elodin_model = "aot", elodin_model = "jit")))]
const BUILD_INFO: &str = "{}";

/// Worker threads `elodin_model_set_workers` asked for; 0 for the exported count.
static WORKERS: AtomicUsize = AtomicUsize::new(0);
static OPEN_PROGRAMS: AtomicUsize = AtomicUsize::new(0);

#[cfg(elodin_model = "aot")]
fn open() -> Result<Program, String> {
    let exported = serde_json::from_str::<Schema>(SCHEMA)
        .map_err(|e| format!("invalid schema.json: {e}"))?
        .workers;
    let workers = match WORKERS.load(Ordering::Relaxed) {
        0 => exported,
        requested => requested,
    };
    let existing = cranelift_mlir::pool::worker_count() + 1;
    if existing > 1 && workers < existing {
        return Err(format!(
            "the process worker pool already has {existing} workers and cannot shrink to {workers}"
        ));
    }
    // The calling thread works alongside the pool, and one worker runs every group itself.
    cranelift_mlir::pool::ensure_workers(workers.max(1) - 1);
    let mut program = Program::new(SCHEMA, STATE, aot::step()?)?;
    program.workers = workers.max(1);
    Ok(program)
}

#[cfg(elodin_model = "jit")]
fn open() -> Result<Program, String> {
    let mlir = include_str!(concat!(env!("ELODIN_MODEL_DIR"), "/step.stablehlo.mlir"));
    let mut program = Program::jit(mlir, SCHEMA, STATE)?;
    program.workers = 1;
    Ok(program)
}

#[cfg(not(any(elodin_model = "aot", elodin_model = "jit")))]
fn open() -> Result<Program, String> {
    Err("this library holds no model; build it with ELODIN_MODEL_DIR set".into())
}

/// The step `step.o` defines. Its calls into the Cranelift runtime go through a table this
/// library fills in, so the object needs nothing from the linker but libc.
#[cfg(elodin_model = "aot")]
mod aot {
    use std::ffi::{CStr, c_char};
    use std::sync::OnceLock;

    use super::Step;

    unsafe extern "C" {
        fn elodin_model_kernel_step(inputs: *const *const u8, outputs: *mut *mut u8);
        static elodin_model_kernel_import_count: u64;
        static elodin_model_kernel_import_names: c_char;
        static mut elodin_model_kernel_imports: *const u8;
    }

    pub fn step() -> Result<Step, String> {
        static BOUND: OnceLock<Result<(), String>> = OnceLock::new();
        // Safety: the table and names are the object's, and `OnceLock` writes the table once,
        // before any step can read it.
        BOUND.get_or_init(|| unsafe { bind() }).clone()?;
        Ok(Box::new(|inputs: &[&[u8]], outputs: &mut [&mut [u8]]| {
            let inputs: Vec<*const u8> = inputs.iter().map(|b| b.as_ptr()).collect();
            let mut outputs: Vec<*mut u8> = outputs.iter_mut().map(|b| b.as_mut_ptr()).collect();
            cranelift_mlir::pool::wake();
            // Safety: `Program` passes a buffer per component of the schema `step.o` was
            // exported with, sized by it, and outputs separate from the inputs.
            unsafe { elodin_model_kernel_step(inputs.as_ptr(), outputs.as_mut_ptr()) };
            Ok(())
        }))
    }

    /// # Safety
    ///
    /// Call once, before the first step.
    unsafe fn bind() -> Result<(), String> {
        // Safety: the object holds `count` NUL-terminated names back to back, and as many slots.
        unsafe {
            let count = elodin_model_kernel_import_count as usize;
            let mut name = &raw const elodin_model_kernel_import_names;
            let table = &raw mut elodin_model_kernel_imports;
            for slot in 0..count {
                let symbol = CStr::from_ptr(name);
                let symbol_name = symbol.to_str().map_err(|e| e.to_string())?;
                let function = cranelift_mlir::lower::runtime_symbol(symbol_name)
                    .ok_or_else(|| format!("this library has no runtime function {symbol_name}"))?;
                table.add(slot).write(function);
                name = name.add(symbol.to_bytes().len() + 1);
            }
        }
        Ok(())
    }
}

thread_local! {
    static ERROR: RefCell<CString> = RefCell::default();
}

fn fail(message: impl Into<Vec<u8>>) {
    let message = CString::new(message).unwrap_or_default();
    ERROR.with(|error| *error.borrow_mut() = message);
}

fn clear_error() {
    ERROR.with(|error| *error.borrow_mut() = CString::default());
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "Rust panic crossed the model boundary".into()
    }
}

fn ffi_status(error_code: c_int, operation: impl FnOnce() -> Result<(), String>) -> c_int {
    clear_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => ELODIN_MODEL_OK,
        Ok(Err(error)) => {
            fail(error);
            error_code
        }
        Err(payload) => {
            fail(panic_message(payload));
            ELODIN_MODEL_ERR_STEP
        }
    }
}

unsafe fn output_bytes<'a>(data: *mut c_void, size: usize) -> Result<&'a mut [u8], String> {
    if size == 0 {
        return Ok(&mut []);
    }
    if data.is_null() {
        return Err("destination is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(data.cast::<u8>(), size) })
}

unsafe fn input_bytes<'a>(data: *const c_void, size: usize) -> Result<&'a [u8], String> {
    if size == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err("source is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size) })
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_error() -> *const c_char {
    ERROR.with(|error| error.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_abi_version() -> u32 {
    2
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_backend() -> *const c_char {
    if cfg!(elodin_model = "aot") {
        c"aot".as_ptr()
    } else if cfg!(elodin_model = "jit") {
        c"jit".as_ptr()
    } else {
        c"none".as_ptr()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_build_info() -> *const c_char {
    #[cfg(any(elodin_model = "aot", elodin_model = "jit"))]
    {
        static INFO: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
        INFO.get_or_init(|| CString::new(BUILD_INFO).unwrap_or_default())
            .as_ptr()
    }
    #[cfg(not(any(elodin_model = "aot", elodin_model = "jit")))]
    {
        c"{}".as_ptr()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_configure_workers(workers: usize) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_STATE, || {
        if OPEN_PROGRAMS.load(Ordering::Acquire) != 0 {
            return Err("configure workers before opening a model".into());
        }
        if workers == 0 {
            WORKERS.store(0, Ordering::Release);
            return Ok(());
        }
        let existing = cranelift_mlir::pool::worker_count() + 1;
        if existing > 1 && workers < existing {
            return Err(format!(
                "the process worker pool already has {existing} workers and cannot shrink to {workers}"
            ));
        }
        WORKERS.store(workers, Ordering::Release);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_set_workers(workers: usize) {
    let _ = elodin_model_configure_workers(workers);
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_open() -> *mut Program {
    clear_error();
    match catch_unwind(AssertUnwindSafe(open)) {
        Ok(Ok(program)) => {
            OPEN_PROGRAMS.fetch_add(1, Ordering::AcqRel);
            Box::into_raw(Box::new(program))
        }
        Ok(Err(error)) => {
            fail(error);
            ptr::null_mut()
        }
        Err(payload) => {
            fail(panic_message(payload));
            ptr::null_mut()
        }
    }
}

/// # Safety
///
/// `program` is null or from `elodin_model_open`, and none of its instances remain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_close(program: *mut Program) {
    if !program.is_null() {
        drop(unsafe { Box::from_raw(program) });
        OPEN_PROGRAMS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_count(program: *const Program) -> usize {
    unsafe { program.as_ref() }.map_or(0, |program| program.components.len())
}

/// Model artifact schema ABI; 1 for directories exported before versioning.
///
/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_schema_abi(program: *const Program) -> u32 {
    unsafe { program.as_ref() }.map_or(0, |program| program.abi_version)
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_format(program: *const Program) -> *const c_char {
    unsafe { program.as_ref() }.map_or(ptr::null(), |program| program.format.as_ptr())
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_endianness(program: *const Program) -> *const c_char {
    unsafe { program.as_ref() }.map_or(ptr::null(), |program| program.endianness.as_ptr())
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_effective_workers(program: *const Program) -> usize {
    unsafe { program.as_ref() }.map_or(0, |program| program.workers)
}

/// Whether one model step advances its tick component.
///
/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_tick_advanced_by_step(program: *const Program) -> u32 {
    unsafe { program.as_ref() }.map_or(0, |program| {
        u32::from(program.simulation.tick_advanced_by_step)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_recommended_stack() -> usize {
    256 * 1024 * 1024
}

/// # Safety
/// `program` is null or live; `seconds` is null or points to writable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_nominal_time_step(
    program: *const Program,
    seconds: *mut f64,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let program = unsafe { program.as_ref() }.ok_or("program is null")?;
        let seconds = unsafe { seconds.as_mut() }.ok_or("seconds is null")?;
        let value = if program.simulation.time_step_s > 0.0 {
            program.simulation.time_step_s
        } else if program.simulation.rate_hz > 0.0 {
            1.0 / program.simulation.rate_hz
        } else {
            let index = program
                .index(c"simulation_time_step")
                .ok_or("model has no nominal time-step metadata")?;
            let bytes: [u8; 8] = program.initial[index]
                .as_slice()
                .try_into()
                .map_err(|_| "simulation_time_step is not one f64")?;
            f64::from_le_bytes(bytes)
        };
        *seconds = value;
        Ok(())
    })
}

/// # Safety
/// `program` is null or live; `output` is null or points to its declared `struct_size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_type(
    program: *const Program,
    component: usize,
    output: *mut ElodinModelComponentType,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let program = unsafe { program.as_ref() }.ok_or("program is null")?;
        let output = unsafe { output.as_mut() }.ok_or("component type is null")?;
        if output.struct_size < size_of::<ElodinModelComponentType>() {
            return Err(format!(
                "component type struct is {} bytes, need {}",
                output.struct_size,
                size_of::<ElodinModelComponentType>()
            ));
        }
        let component = program
            .components
            .get(component)
            .ok_or_else(|| format!("no component {component}"))?;
        let dtype = component.dtype.to_str().map_err(|e| e.to_string())?;
        *output = ElodinModelComponentType {
            struct_size: size_of::<ElodinModelComponentType>(),
            dtype: component.dtype.as_ptr(),
            shape: component.shape.as_ptr(),
            rank: component.shape.len(),
            element_shape: component.element_shape.as_ptr(),
            element_rank: component.element_shape.len(),
            element_bytes: dtype_size(dtype)?,
            rows: component.entities.len(),
            row_bytes: component.row_bytes,
            total_bytes: component.size,
            flags: component.flags,
            entity_axis: component.entity_axis,
        };
        Ok(())
    })
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_name(
    program: *const Program,
    component: usize,
) -> *const c_char {
    unsafe { program.as_ref() }
        .and_then(|program| program.components.get(component).map(|c| c.name.as_ptr()))
        .unwrap_or(ptr::null())
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_size(
    program: *const Program,
    component: usize,
) -> usize {
    unsafe { program.as_ref() }
        .and_then(|program| program.components.get(component).map(|c| c.size))
        .unwrap_or(0)
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_rows(
    program: *const Program,
    component: usize,
) -> usize {
    unsafe { program.as_ref() }
        .and_then(|program| program.components.get(component).map(|c| c.entities.len()))
        .unwrap_or(0)
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_entity_name(
    program: *const Program,
    component: usize,
    row: usize,
) -> *const c_char {
    unsafe { program.as_ref() }
        .and_then(|program| program.components.get(component))
        .and_then(|c| c.entities.get(row))
        .map_or(ptr::null(), |(_, name)| name.as_ptr())
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_entity_id(
    program: *const Program,
    component: usize,
    row: usize,
) -> u64 {
    unsafe { program.as_ref() }
        .and_then(|program| program.components.get(component))
        .and_then(|c| c.entities.get(row))
        .map_or(u64::MAX, |(id, _)| *id)
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open, and `name` is NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_index(
    program: *const Program,
    name: *const c_char,
) -> i64 {
    let Some(program) = (unsafe { program.as_ref() }) else {
        fail("program is null");
        return -1;
    };
    if name.is_null() {
        fail("component name is null");
        return -1;
    }
    let name = unsafe { CStr::from_ptr(name) };
    program.index(name).map_or(-1, |index| index as i64)
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_count(program: *const Program) -> usize {
    unsafe { program.as_ref() }.map_or(0, |program| program.interfaces.len())
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_name(
    program: *const Program,
    interface: usize,
) -> *const c_char {
    unsafe { program.as_ref() }
        .and_then(|program| program.interfaces.get(interface))
        .map_or(ptr::null(), |interface| interface.name.as_ptr())
}

/// # Safety
/// `program` is null or live; `name` is null or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_index(
    program: *const Program,
    name: *const c_char,
) -> i64 {
    let Some(program) = (unsafe { program.as_ref() }) else {
        fail("program is null");
        return -1;
    };
    if name.is_null() {
        fail("interface name is null");
        return -1;
    }
    let name = unsafe { CStr::from_ptr(name) };
    program
        .interfaces
        .iter()
        .position(|interface| interface.name.as_c_str() == name)
        .map_or(-1, |index| index as i64)
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_rows(
    program: *const Program,
    interface: usize,
) -> usize {
    unsafe { program.as_ref() }
        .and_then(|program| program.interfaces.get(interface))
        .map_or(0, |interface| interface.rows)
}

/// # Safety
/// `program` is null or from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_field_count(
    program: *const Program,
    interface: usize,
) -> usize {
    unsafe { program.as_ref() }
        .and_then(|program| program.interfaces.get(interface))
        .map_or(0, |interface| interface.fields.len())
}

/// # Safety
/// `program` is null or live; `output` is null or points to its declared `struct_size`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_field(
    program: *const Program,
    interface: usize,
    field: usize,
    output: *mut ElodinModelInterfaceField,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let program = unsafe { program.as_ref() }.ok_or("program is null")?;
        let output = unsafe { output.as_mut() }.ok_or("interface field is null")?;
        if output.struct_size < size_of::<ElodinModelInterfaceField>() {
            return Err(format!(
                "interface field struct is {} bytes, need {}",
                output.struct_size,
                size_of::<ElodinModelInterfaceField>()
            ));
        }
        let field = program
            .interfaces
            .get(interface)
            .ok_or_else(|| format!("no interface {interface}"))?
            .fields
            .get(field)
            .ok_or_else(|| format!("interface {interface} has no field {field}"))?;
        *output = ElodinModelInterfaceField {
            struct_size: size_of::<ElodinModelInterfaceField>(),
            name: field.name.as_ptr(),
            component: field.component,
            writable: u32::from(field.writable),
            rows: field.rows.len(),
            row_bytes: field.row_bytes,
        };
        Ok(())
    })
}

/// # Safety
///
/// `program` is from `elodin_model_open`, and stays open while the instance lives.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_create(program: *const Program) -> *mut Instance<'static> {
    clear_error();
    match catch_unwind(AssertUnwindSafe(|| {
        unsafe { program.as_ref() }
            .map(Instance::new)
            .ok_or("program is null")
    })) {
        Ok(Ok(instance)) => Box::into_raw(Box::new(instance)),
        Ok(Err(error)) => {
            fail(error);
            ptr::null_mut()
        }
        Err(payload) => {
            fail(panic_message(payload));
            ptr::null_mut()
        }
    }
}

/// # Safety
///
/// `instance` is null or from `elodin_model_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_destroy(instance: *mut Instance<'static>) {
    if !instance.is_null() {
        drop(unsafe { Box::from_raw(instance) });
    }
}

/// # Safety
///
/// `instance` is from `elodin_model_create`, and no other thread uses it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_reset(instance: *mut Instance<'static>) {
    let _ = unsafe { elodin_model_reset_checked(instance) };
}

/// # Safety
///
/// `instance` is from `elodin_model_create`, and no other thread uses it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_reset_checked(instance: *mut Instance<'static>) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        unsafe { instance.as_mut() }
            .ok_or("instance is null")?
            .reset();
        Ok(())
    })
}

/// # Safety
///
/// `instance` is from `elodin_model_create`, and no other thread uses it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_step(instance: *mut Instance<'static>) -> c_int {
    if instance.is_null() {
        fail("instance is null");
        return ELODIN_MODEL_ERR_ARGUMENT;
    }
    ffi_status(ELODIN_MODEL_ERR_STEP, || {
        // Safety: checked above; the caller owns exclusive use for this call.
        unsafe { &mut *instance }.step()
    })
}

/// Monotonic state-buffer generation. Step, reset, and successful writes increment it.
///
/// # Safety
/// `instance` is null or from `elodin_model_create` and still live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_generation(instance: *const Instance<'static>) -> u64 {
    unsafe { instance.as_ref() }.map_or(0, |instance| instance.generation)
}

/// # Safety
/// `instance` is null or live; `tick` is null or points to writable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_current_tick(
    instance: *const Instance<'static>,
    tick: *mut u64,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_ref() }.ok_or("instance is null")?;
        let tick = unsafe { tick.as_mut() }.ok_or("tick is null")?;
        let name = instance
            .program
            .simulation
            .tick_component
            .as_deref()
            .unwrap_or("tick");
        let component = instance
            .program
            .index(CString::new(name).map_err(|e| e.to_string())?.as_c_str())
            .ok_or("model has no tick component")?;
        let bytes: [u8; 8] = instance.state[component]
            .as_slice()
            .try_into()
            .map_err(|_| "tick component is not one u64")?;
        *tick = u64::from_le_bytes(bytes);
        Ok(())
    })
}

/// # Safety
/// `instance` is null or live and is used by no other thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_set_time_step(
    instance: *mut Instance<'static>,
    seconds: f64,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err("time step must be finite and positive".into());
        }
        let instance = unsafe { instance.as_mut() }.ok_or("instance is null")?;
        let name = instance
            .program
            .simulation
            .time_step_component
            .as_deref()
            .unwrap_or("simulation_time_step");
        let component = instance
            .program
            .index(CString::new(name).map_err(|e| e.to_string())?.as_c_str())
            .ok_or("model has no simulation_time_step component")?;
        instance.write(component, &seconds.to_le_bytes())
    })
}

/// # Safety
///
/// `instance` is from `elodin_model_create`, and `dst` holds `size` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_read(
    instance: *const Instance<'static>,
    component: usize,
    dst: *mut c_void,
    size: usize,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_ref() }.ok_or("instance is null")?;
        let dst = unsafe { output_bytes(dst, size) }?;
        instance.read(component, dst)
    })
}

/// # Safety
///
/// `instance` is from `elodin_model_create`, no other thread uses it, and `src` holds `size`
/// readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_write(
    instance: *mut Instance<'static>,
    component: usize,
    src: *const c_void,
    size: usize,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_mut() }.ok_or("instance is null")?;
        let src = unsafe { input_bytes(src, size) }?;
        instance.write(component, src)
    })
}

/// # Safety
/// `instance` is null or live; `dst` holds `size` writable bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_read_rows(
    instance: *const Instance<'static>,
    component: usize,
    first: usize,
    count: usize,
    dst: *mut c_void,
    size: usize,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_ref() }.ok_or("instance is null")?;
        let dst = unsafe { output_bytes(dst, size) }?;
        instance.read_rows(component, first, count, dst)
    })
}

/// # Safety
/// `instance` is null or live; `src` holds `size` readable bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_write_rows(
    instance: *mut Instance<'static>,
    component: usize,
    first: usize,
    count: usize,
    src: *const c_void,
    size: usize,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_mut() }.ok_or("instance is null")?;
        let src = unsafe { input_bytes(src, size) }?;
        instance.write_rows(component, first, count, src)
    })
}

/// # Safety
/// `instance` is null or live; `dst` holds `size` writable bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_read(
    instance: *const Instance<'static>,
    interface: usize,
    field: usize,
    dst: *mut c_void,
    size: usize,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_ref() }.ok_or("instance is null")?;
        let dst = unsafe { output_bytes(dst, size) }?;
        instance.interface_read(interface, field, dst)
    })
}

/// # Safety
/// `instance` is null or live; `src` holds `size` readable bytes when non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_interface_write(
    instance: *mut Instance<'static>,
    interface: usize,
    field: usize,
    src: *const c_void,
    size: usize,
) -> c_int {
    ffi_status(ELODIN_MODEL_ERR_ARGUMENT, || {
        let instance = unsafe { instance.as_mut() }.ok_or("instance is null")?;
        let src = unsafe { input_bytes(src, size) }?;
        instance.interface_write(interface, field, src)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each step moves each room a quarter of the way to its ambient temperature.
    const ROOMS: &str = r#"
module @module {
  func.func public @main(%arg0: tensor<ui64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>, %arg3: tensor<f64>) -> (tensor<ui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>) {
    %c = stablehlo.constant dense<1> : tensor<ui64>
    %0 = stablehlo.add %arg0, %c : tensor<ui64>
    %1 = stablehlo.subtract %arg2, %arg1 : tensor<2xf64>
    %dt = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f64>) -> tensor<2xf64>
    %2 = stablehlo.multiply %1, %dt : tensor<2xf64>
    %3 = stablehlo.add %arg1, %2 : tensor<2xf64>
    return %0, %3, %arg2, %arg3 : tensor<ui64>, tensor<2xf64>, tensor<2xf64>, tensor<f64>
  }
}
"#;

    fn schema(rooms: usize) -> String {
        let names: Vec<String> = (0..rooms).map(|i| format!(r#""room_{i}""#)).collect();
        let ids: Vec<String> = (1..=rooms).map(|i| i.to_string()).collect();
        let room = format!(
            r#""dtype": "float64", "shape": [{rooms}], "entities": [{}], "entity_ids": [{}]"#,
            names.join(", "),
            ids.join(", ")
        );
        format!(
            r#"{{"components": [
                {{"name": "tick", "dtype": "uint64", "shape": [], "entities": ["house"], "entity_ids": [0]}},
                {{"name": "temperature", {room}}},
                {{"name": "ambient", {room}}},
                {{"name": "simulation_time_step", "dtype": "float64", "shape": [], "entities": ["house"], "entity_ids": [0]}}
            ]}}"#
        )
    }

    fn schema_v2() -> String {
        r#"{
            "format": "elodin.model",
            "abi_version": 2,
            "endianness": "little",
            "workers": 3,
            "simulation": {
                "rate_hz": 4.0,
                "time_step_s": 0.25,
                "tick_component": "tick",
                "time_step_component": "simulation_time_step",
                "tick_advanced_by_step": true
            },
            "components": [
                {
                    "name": "tick", "dtype": "uint64", "shape": [],
                    "element_shape": [], "entity_axis": "none",
                    "entities": ["house"], "entity_ids": [0], "rows": 1,
                    "row_bytes": 8, "size": 8, "written_by_step": true,
                    "metadata": {}
                },
                {
                    "name": "temperature", "dtype": "float64", "shape": [2],
                    "element_shape": [], "entity_axis": "leading",
                    "entities": ["room_0", "room_1"], "entity_ids": [1, 2], "rows": 2,
                    "row_bytes": 8, "size": 16, "written_by_step": true,
                    "metadata": {"element_names": "temperature"}
                },
                {
                    "name": "ambient", "dtype": "float64", "shape": [2],
                    "element_shape": [], "entity_axis": "leading",
                    "entities": ["room_0", "room_1"], "entity_ids": [1, 2], "rows": 2,
                    "row_bytes": 8, "size": 16, "written_by_step": false,
                    "metadata": {"external_control": "true"}
                },
                {
                    "name": "simulation_time_step", "dtype": "float64", "shape": [],
                    "element_shape": [], "entity_axis": "none",
                    "entities": ["house"], "entity_ids": [0], "rows": 1,
                    "row_bytes": 8, "size": 8, "written_by_step": false,
                    "metadata": {}
                }
            ],
            "interfaces": [{
                "name": "rooms.exchange",
                "rows": 2,
                "fields": [
                    {
                        "name": "measured", "component": "temperature", "access": "read",
                        "entity_ids": [2, 1], "row_bytes": 8
                    },
                    {
                        "name": "command", "component": "ambient", "access": "write",
                        "entity_ids": [1, 2], "row_bytes": 8
                    }
                ]
            }]
        }"#
        .into()
    }

    fn state() -> Vec<u8> {
        let mut state = 0u64.to_le_bytes().to_vec();
        for value in [20.0f64, 28.0, 28.0, 20.0] {
            state.extend(value.to_le_bytes());
        }
        state.extend(0.25f64.to_le_bytes());
        state
    }

    fn rooms() -> Program {
        Program::jit(ROOMS, &schema(2), &state()).unwrap()
    }

    fn f64s(instance: &Instance, component: usize) -> Vec<f64> {
        let mut bytes = vec![0; 16];
        instance.read(component, &mut bytes).unwrap();
        bytes
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn tick(instance: &Instance) -> u64 {
        let mut bytes = [0; 8];
        instance.read(0, &mut bytes).unwrap();
        u64::from_le_bytes(bytes)
    }

    #[test]
    fn steps_carry_state_and_inputs() {
        let program = rooms();
        let mut a = Instance::new(&program);
        a.step().unwrap();
        assert_eq!(f64s(&a, 1), [22.0, 26.0]);
        a.step().unwrap();
        assert_eq!(f64s(&a, 1), [23.5, 24.5]);
        assert_eq!(f64s(&a, 2), [28.0, 20.0]);

        let ambient: Vec<u8> = [30.0f64, 22.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        a.write(2, &ambient).unwrap();
        a.step().unwrap();
        assert_eq!(f64s(&a, 1), [25.125, 23.875]);
        assert_eq!(tick(&a), 3);
    }

    #[test]
    fn instances_are_independent_and_reset() {
        let program = rooms();
        let mut a = Instance::new(&program);
        a.step().unwrap();
        a.step().unwrap();
        let mut b = Instance::new(&program);
        b.step().unwrap();
        assert_eq!(f64s(&b, 1), [22.0, 26.0]);
        assert_eq!(tick(&a), 2);

        a.reset();
        assert_eq!(tick(&a), 0);
        a.step().unwrap();
        assert_eq!(f64s(&a, 1), [22.0, 26.0]);
    }

    #[test]
    fn names_index_components_and_rows() {
        let program = rooms();
        assert_eq!(program.index(c"ambient"), Some(2));
        assert_eq!(program.index(c"humidity"), None);
        let program = &raw const program;
        // Safety: `program` is a live program.
        unsafe {
            assert_eq!(elodin_model_component_rows(program, 2), 2);
            assert_eq!(elodin_model_entity_id(program, 2, 1), 2);
            assert_eq!(
                CStr::from_ptr(elodin_model_entity_name(program, 2, 1)),
                c"room_1"
            );
            assert_eq!(elodin_model_entity_id(program, 0, 0), 0);
            assert!(elodin_model_entity_name(program, 2, 2).is_null());
            assert_eq!(elodin_model_entity_id(program, 4, 0), u64::MAX);
        }
    }

    #[test]
    fn versioned_schema_exposes_types_interfaces_and_caller_time() {
        let program = Program::jit(ROOMS, &schema_v2(), &state()).unwrap();
        let program_ptr = &raw const program;
        let instance_ptr = unsafe { elodin_model_create(program_ptr) };
        assert!(!instance_ptr.is_null());
        // Safety: the program and instance stay live for every call.
        unsafe {
            assert_eq!(elodin_model_schema_abi(program_ptr), 2);
            assert_eq!(
                CStr::from_ptr(elodin_model_format(program_ptr)),
                c"elodin.model"
            );
            assert_eq!(elodin_model_effective_workers(program_ptr), 3);
            let mut dt = 0.0;
            assert_eq!(
                elodin_model_nominal_time_step(program_ptr, &raw mut dt),
                ELODIN_MODEL_OK
            );
            assert_eq!(dt, 0.25);

            let mut ty = ElodinModelComponentType {
                struct_size: size_of::<ElodinModelComponentType>(),
                dtype: ptr::null(),
                shape: ptr::null(),
                rank: 0,
                element_shape: ptr::null(),
                element_rank: 0,
                element_bytes: 0,
                rows: 0,
                row_bytes: 0,
                total_bytes: 0,
                flags: 0,
                entity_axis: 0,
            };
            assert_eq!(
                elodin_model_component_type(program_ptr, 1, &raw mut ty),
                ELODIN_MODEL_OK
            );
            assert_eq!(CStr::from_ptr(ty.dtype), c"float64");
            assert_eq!(ty.rows, 2);
            assert_eq!(ty.row_bytes, 8);
            assert_eq!(ty.flags & ELODIN_MODEL_COMPONENT_WRITTEN_BY_STEP, 1);

            assert_eq!(elodin_model_interface_count(program_ptr), 1);
            assert_eq!(
                elodin_model_interface_index(program_ptr, c"rooms.exchange".as_ptr()),
                0
            );
            let mut field = ElodinModelInterfaceField {
                struct_size: size_of::<ElodinModelInterfaceField>(),
                name: ptr::null(),
                component: 0,
                writable: 0,
                rows: 0,
                row_bytes: 0,
            };
            assert_eq!(
                elodin_model_interface_field(program_ptr, 0, 1, &raw mut field),
                ELODIN_MODEL_OK
            );
            assert_eq!(CStr::from_ptr(field.name), c"command");
            assert_eq!(field.writable, 1);

            let mut measured = [0u8; 16];
            assert_eq!(
                elodin_model_interface_read(
                    instance_ptr,
                    0,
                    0,
                    measured.as_mut_ptr().cast(),
                    measured.len()
                ),
                ELODIN_MODEL_OK
            );
            let measured: Vec<f64> = measured
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| f64::from_le_bytes(*bytes))
                .collect();
            assert_eq!(measured, [28.0, 20.0]);

            let command: Vec<u8> = [30.0f64, 22.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            assert_eq!(
                elodin_model_interface_write(
                    instance_ptr,
                    0,
                    1,
                    command.as_ptr().cast(),
                    command.len()
                ),
                ELODIN_MODEL_OK
            );
            assert_eq!(
                elodin_model_set_time_step(instance_ptr, 0.5),
                ELODIN_MODEL_OK
            );
            assert_eq!(elodin_model_step(instance_ptr), ELODIN_MODEL_OK);
            assert_eq!(f64s(&*instance_ptr, 1), [25.0, 25.0]);
            let mut tick = 0;
            assert_eq!(
                elodin_model_current_tick(instance_ptr, &raw mut tick),
                ELODIN_MODEL_OK
            );
            assert_eq!(tick, 1);
            assert_eq!(elodin_model_generation(instance_ptr), 3);
            elodin_model_destroy(instance_ptr);
        }
    }

    #[test]
    fn reset_repeats_cases_in_one_process() {
        let program = Program::jit(ROOMS, &schema_v2(), &state()).unwrap();
        let mut instance = Instance::new(&program);
        let expected = (0..100)
            .map(|_| {
                instance.reset();
                instance
                    .write(2, &[30.0f64.to_le_bytes(), 22.0f64.to_le_bytes()].concat())
                    .unwrap();
                instance.step().unwrap();
                (tick(&instance), f64s(&instance, 1))
            })
            .next()
            .unwrap();
        for _ in 0..100 {
            instance.reset();
            instance
                .write(2, &[30.0f64.to_le_bytes(), 22.0f64.to_le_bytes()].concat())
                .unwrap();
            instance.step().unwrap();
            assert_eq!((tick(&instance), f64s(&instance, 1)), expected);
        }
    }

    #[test]
    fn null_arguments_return_errors_instead_of_panicking() {
        unsafe {
            assert_eq!(
                elodin_model_step(ptr::null_mut()),
                ELODIN_MODEL_ERR_ARGUMENT
            );
            assert_eq!(
                elodin_model_reset_checked(ptr::null_mut()),
                ELODIN_MODEL_ERR_ARGUMENT
            );
            assert_eq!(
                elodin_model_read(ptr::null(), 0, ptr::null_mut(), 0),
                ELODIN_MODEL_ERR_ARGUMENT
            );
        }
    }

    #[test]
    fn models_exported_without_entity_ids_still_open() {
        let schema = schema(2).replace(r#", "entity_ids": [1, 2]"#, "");
        let program = Program::jit(ROOMS, &schema, &state()).unwrap();
        let program = &raw const program;
        // Safety: `program` is a live program.
        unsafe {
            assert_eq!(elodin_model_component_rows(program, 1), 2);
            assert_eq!(elodin_model_entity_id(program, 1, 0), u64::MAX);
            assert_eq!(
                CStr::from_ptr(elodin_model_entity_name(program, 1, 0)),
                c"room_0"
            );
        }
    }

    #[test]
    fn mismatched_buffers_are_rejected() {
        let program = rooms();
        let mut a = Instance::new(&program);
        assert!(a.write(1, &[0; 8]).is_err());
        assert!(a.read(4, &mut [0; 8]).is_err());
        assert!(Program::jit(ROOMS, &schema(2), &state()[..32]).is_err());
        let three_rooms = [state(), vec![0; 16]].concat();
        assert!(Program::jit(ROOMS, &schema(3), &three_rooms).is_err());
    }

    #[test]
    #[cfg(not(any(elodin_model = "aot", elodin_model = "jit")))]
    fn a_library_without_a_model_says_so() {
        assert!(elodin_model_open().is_null());
        let error = unsafe { CStr::from_ptr(elodin_model_error()) };
        assert!(error.to_str().unwrap().contains("ELODIN_MODEL_DIR"));
    }
}
