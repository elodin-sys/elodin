//! A simulation exported from Elodin with `JaxSim.export_model`, behind the C interface of
//! `include/elodin_model.h`. `build.rs` embeds the model and picks how its code is compiled.
// Without a model only the tests load programs.
#![cfg_attr(not(any(elodin_model = "aot", elodin_model = "jit")), allow(dead_code))]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;

use cranelift_mlir::lower::CompiledModule;
use serde::Deserialize;

/// The exported step: one buffer per component in, the next state's buffers out.
type StepFn = unsafe extern "C" fn(*const *const u8, *mut *mut u8);

#[derive(Deserialize)]
struct Schema {
    components: Vec<ComponentSchema>,
}

#[derive(Deserialize)]
struct ComponentSchema {
    name: String,
    dtype: String,
    shape: Vec<usize>,
}

struct Component {
    name: CString,
    size: usize,
}

/// A model's code, its components and their initial values.
pub struct Program {
    components: Vec<Component>,
    initial: Vec<Vec<u8>>,
    step: StepFn,
    /// Owns the JIT code `step` points into.
    _jit: Option<CompiledModule>,
}

/// The components `schema` describes, each with its initial value from `state`.
fn layout(schema: &str, state: &[u8]) -> Result<(Vec<Component>, Vec<Vec<u8>>), String> {
    let schema: Schema =
        serde_json::from_str(schema).map_err(|e| format!("invalid schema.json: {e}"))?;
    let mut components = Vec::new();
    let mut initial = Vec::new();
    let mut rest = state;
    for component in schema.components {
        let size = dtype_size(&component.dtype)? * component.shape.iter().product::<usize>();
        let (value, tail) = rest
            .split_at_checked(size)
            .ok_or("state.bin is shorter than schema.json describes")?;
        initial.push(value.to_vec());
        rest = tail;
        components.push(Component {
            name: CString::new(component.name).map_err(|e| e.to_string())?,
            size,
        });
    }
    if !rest.is_empty() {
        return Err("state.bin is longer than schema.json describes".into());
    }
    Ok((components, initial))
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
    fn new(
        schema: &str,
        state: &[u8],
        step: StepFn,
        jit: Option<CompiledModule>,
    ) -> Result<Self, String> {
        let (components, initial) = layout(schema, state)?;
        Ok(Self {
            components,
            initial,
            step,
            _jit: jit,
        })
    }

    /// Compiles `mlir`, whose `main` must map the state `schema` describes to its next value.
    fn jit(mlir: &str, schema: &str, state: &[u8]) -> Result<Self, String> {
        let (components, _) = layout(schema, state)?;
        let mut module = cranelift_mlir::parser::parse_module(mlir)
            .map_err(|e| format!("StableHLO parse failed: {e}"))?;
        let main = module.main_func().ok_or("the StableHLO has no main")?;
        let sizes: Vec<usize> = components.iter().map(|c| c.size).collect();
        let params: Vec<usize> = main.params.iter().map(|(_, ty)| ty.byte_size()).collect();
        let results: Vec<usize> = main.result_types.iter().map(|ty| ty.byte_size()).collect();
        if params != sizes || results != sizes {
            return Err(
                "main does not map the state schema.json describes to its next value".into(),
            );
        }
        cranelift_mlir::const_fold::fold_module(&mut module);
        let compiled = cranelift_mlir::lower::compile_module(&module)
            .map_err(|e| format!("Cranelift compile failed: {e}"))?;
        // Safety: Cranelift compiles `main` to `StepFn`, and `_jit` keeps its code alive.
        let step = unsafe { std::mem::transmute::<*const u8, StepFn>(compiled.get_main_fn()) };
        Self::new(schema, state, step, Some(compiled))
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
    inputs: Vec<*const u8>,
    outputs: Vec<*mut u8>,
}

impl<'a> Instance<'a> {
    fn new(program: &'a Program) -> Self {
        let count = program.components.len();
        Self {
            program,
            state: program.initial.clone(),
            next: program.initial.clone(),
            inputs: vec![ptr::null(); count],
            outputs: vec![ptr::null_mut(); count],
        }
    }

    fn reset(&mut self) {
        self.state.clone_from(&self.program.initial);
    }

    fn step(&mut self) {
        for (input, buffer) in self.inputs.iter_mut().zip(&self.state) {
            *input = buffer.as_ptr();
        }
        for (output, buffer) in self.outputs.iter_mut().zip(&mut self.next) {
            *output = buffer.as_mut_ptr();
        }
        // Safety: the step reads one buffer per component, of the sizes the program's schema
        // gives, and writes the next state into separate buffers of the same sizes.
        unsafe { (self.program.step)(self.inputs.as_ptr(), self.outputs.as_mut_ptr()) };
        std::mem::swap(&mut self.state, &mut self.next);
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
        Ok(())
    }
}

#[cfg(any(elodin_model = "aot", elodin_model = "jit"))]
const SCHEMA: &str = include_str!(concat!(env!("ELODIN_MODEL_DIR"), "/schema.json"));
#[cfg(any(elodin_model = "aot", elodin_model = "jit"))]
const STATE: &[u8] = include_bytes!(concat!(env!("ELODIN_MODEL_DIR"), "/state.bin"));

#[cfg(elodin_model = "aot")]
fn open() -> Result<Program, String> {
    Program::new(SCHEMA, STATE, aot::step()?, None)
}

#[cfg(elodin_model = "jit")]
fn open() -> Result<Program, String> {
    let mlir = include_str!(concat!(env!("ELODIN_MODEL_DIR"), "/step.stablehlo.mlir"));
    Program::jit(mlir, SCHEMA, STATE)
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

    use super::StepFn;

    unsafe extern "C" {
        fn elodin_model_kernel_step(inputs: *const *const u8, outputs: *mut *mut u8);
        static elodin_model_kernel_import_count: u64;
        static elodin_model_kernel_import_names: c_char;
        static mut elodin_model_kernel_imports: *const u8;
    }

    pub fn step() -> Result<StepFn, String> {
        static BOUND: OnceLock<Result<(), String>> = OnceLock::new();
        // Safety: the table and names are the object's, and `OnceLock` writes the table once,
        // before any step can read it.
        BOUND.get_or_init(|| unsafe { bind() }).clone()?;
        Ok(elodin_model_kernel_step)
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

#[unsafe(no_mangle)]
pub extern "C" fn elodin_model_error() -> *const c_char {
    ERROR.with(|error| error.borrow().as_ptr())
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
pub extern "C" fn elodin_model_open() -> *mut Program {
    match open() {
        Ok(program) => Box::into_raw(Box::new(program)),
        Err(error) => {
            fail(error);
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
    }
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_count(program: *const Program) -> usize {
    unsafe { &*program }.components.len()
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_name(
    program: *const Program,
    component: usize,
) -> *const c_char {
    unsafe { &*program }
        .components
        .get(component)
        .map_or(ptr::null(), |c| c.name.as_ptr())
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_size(
    program: *const Program,
    component: usize,
) -> usize {
    unsafe { &*program }
        .components
        .get(component)
        .map_or(0, |c| c.size)
}

/// # Safety
///
/// `program` is from `elodin_model_open` and still open, and `name` is NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_component_index(
    program: *const Program,
    name: *const c_char,
) -> i64 {
    let name = unsafe { CStr::from_ptr(name) };
    unsafe { &*program }
        .index(name)
        .map_or(-1, |index| index as i64)
}

/// # Safety
///
/// `program` is from `elodin_model_open`, and stays open while the instance lives.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_create(program: *const Program) -> *mut Instance<'static> {
    match unsafe { program.as_ref() } {
        Some(program) => Box::into_raw(Box::new(Instance::new(program))),
        None => {
            fail("program is null");
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
    unsafe { &mut *instance }.reset();
}

/// # Safety
///
/// `instance` is from `elodin_model_create`, and no other thread uses it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn elodin_model_step(instance: *mut Instance<'static>) {
    unsafe { &mut *instance }.step();
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
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.cast::<u8>(), size) };
    status(unsafe { &*instance }.read(component, dst))
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
    let src = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), size) };
    status(unsafe { &mut *instance }.write(component, src))
}

fn status(result: Result<(), String>) -> c_int {
    match result {
        Ok(()) => 0,
        Err(error) => {
            fail(error);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each step moves each room a quarter of the way to its ambient temperature.
    const ROOMS: &str = r#"
module @module {
  func.func public @main(%arg0: tensor<ui64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>) -> (tensor<ui64>, tensor<2xf64>, tensor<2xf64>) {
    %c = stablehlo.constant dense<1> : tensor<ui64>
    %0 = stablehlo.add %arg0, %c : tensor<ui64>
    %1 = stablehlo.subtract %arg2, %arg1 : tensor<2xf64>
    %cst = stablehlo.constant dense<2.500000e-01> : tensor<2xf64>
    %2 = stablehlo.multiply %1, %cst : tensor<2xf64>
    %3 = stablehlo.add %arg1, %2 : tensor<2xf64>
    return %0, %3, %arg2 : tensor<ui64>, tensor<2xf64>, tensor<2xf64>
  }
}
"#;

    fn schema(rooms: usize) -> String {
        let room = format!(r#""dtype": "float64", "shape": [{rooms}]"#);
        format!(
            r#"{{"components": [
                {{"name": "tick", "dtype": "uint64", "shape": []}},
                {{"name": "temperature", {room}}},
                {{"name": "ambient", {room}}}
            ]}}"#
        )
    }

    fn state() -> Vec<u8> {
        let mut state = 0u64.to_le_bytes().to_vec();
        for value in [20.0f64, 28.0, 28.0, 20.0] {
            state.extend(value.to_le_bytes());
        }
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
        a.step();
        assert_eq!(f64s(&a, 1), [22.0, 26.0]);
        a.step();
        assert_eq!(f64s(&a, 1), [23.5, 24.5]);
        assert_eq!(f64s(&a, 2), [28.0, 20.0]);

        let ambient: Vec<u8> = [30.0f64, 22.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        a.write(2, &ambient).unwrap();
        a.step();
        assert_eq!(f64s(&a, 1), [25.125, 23.875]);
        assert_eq!(tick(&a), 3);
    }

    #[test]
    fn instances_are_independent_and_reset() {
        let program = rooms();
        let mut a = Instance::new(&program);
        a.step();
        a.step();
        let mut b = Instance::new(&program);
        b.step();
        assert_eq!(f64s(&b, 1), [22.0, 26.0]);
        assert_eq!(tick(&a), 2);

        a.reset();
        assert_eq!(tick(&a), 0);
        a.step();
        assert_eq!(f64s(&a, 1), [22.0, 26.0]);
    }

    #[test]
    fn names_index_components() {
        let program = rooms();
        assert_eq!(program.index(c"ambient"), Some(2));
        assert_eq!(program.index(c"humidity"), None);
    }

    #[test]
    fn mismatched_buffers_are_rejected() {
        let program = rooms();
        let mut a = Instance::new(&program);
        assert!(a.write(1, &[0; 8]).is_err());
        assert!(a.read(3, &mut [0; 8]).is_err());
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
