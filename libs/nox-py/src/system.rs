use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};

use crate::World;
use crate::utils::SchemaExt;
use impeller2::types::ComponentId;
use nox::{ArrayTy, Noxpr, NoxprComp, NoxprFn, NoxprId, NoxprTy};

use crate::{ComponentArray, Error};

pub struct SystemBuilder<'a> {
    pub vars: BTreeMap<ComponentId, ComponentArray<()>>,
    pub inputs: Vec<(ComponentId, Noxpr)>,
    pub world: &'a World,
}

impl<'a> SystemBuilder<'a> {
    pub fn new(world: &'a World) -> Self {
        Self {
            vars: BTreeMap::new(),
            inputs: Vec::new(),
            world,
        }
    }

    pub fn to_compiled_system(&self) -> Result<CompiledSystem, Error> {
        let inputs = self.inputs.iter().map(|(_, p)| p).cloned().collect();
        let mut output = self.vars.values().map(|v| v.buffer.clone());
        let output = if self.vars.len() == 1 {
            output.next().expect("iterator empty")
        } else {
            Noxpr::tuple(output.collect())
        };

        let batch1 = self.world.batch1;
        let mut tys = self.vars.keys().map(|id| {
            NoxprTy::ArrayTy(
                self.world
                    .column_by_id(*id)
                    .unwrap()
                    .buffer_ty_batch1(batch1),
            )
        });

        let ty = if self.vars.len() == 1 {
            tys.next().expect("iterator empty")
        } else {
            NoxprTy::Tuple(tys.collect())
        };
        let func = NoxprFn::new(inputs, output);
        Ok(CompiledSystem {
            computation: NoxprComp {
                func: Arc::new(func),
                id: NoxprId::default(),
                ty,
            },
            inputs: self.inputs.iter().map(|(k, _)| k).copied().collect(),
            outputs: self.vars.keys().copied().collect(),
            system_names: vec!["<compiled>".to_string()],
            diagnostic_subsystems: vec![],
        })
    }

    pub fn init_with_column(&mut self, id: ComponentId) -> Result<(), Error> {
        if self.vars.contains_key(&id) {
            return Ok(());
        }
        let column = self
            .world
            .column_by_id(id)
            .ok_or(Error::ComponentNotFound)
            .unwrap();
        let len = column.len();
        let batch1 = self.world.batch1;
        let mut ty: ArrayTy = column.schema.clone().to_array_ty();
        if !(batch1 && len <= 1) {
            ty.shape.insert(0, len as i64);
        }
        let op = Noxpr::parameter(
            self.inputs.len() as i64,
            nox::NoxprTy::ArrayTy(ty),
            format!("{}::{:?}", self.inputs.len(), id),
        );
        self.inputs.push((id, op.clone()));

        let arr = ComponentArray {
            buffer: op,
            phantom_data: PhantomData,
            len,
            entity_map: column.entity_map(),
            component_id: id,
            batch1,
        };
        self.vars.insert(id, arr);
        Ok(())
    }

    pub fn get_or_init_var(&mut self, id: ComponentId) -> Result<ComponentArray<()>, Error> {
        self.init_with_column(id)?;
        if let Some(var) = self.vars.get(&id) {
            Ok(var.clone())
        } else {
            Err(Error::ComponentNotFound)
        }
    }
}

#[derive(Clone)]
pub struct CompiledSystem {
    pub computation: NoxprComp,
    pub inputs: Vec<ComponentId>,
    pub outputs: Vec<ComponentId>,
    pub system_names: Vec<String>,
    pub diagnostic_subsystems: Vec<CompiledSystem>,
}

impl CompiledSystem {
    pub fn insert_into_builder(self, builder: &mut SystemBuilder) -> Result<(), Error> {
        let mut args = vec![];
        let world = &mut builder.world;
        let batch1 = world.batch1;
        let vars = &mut builder.vars;
        let inputs = &mut builder.inputs;
        for id in self.inputs {
            if let Some(op) = vars.get(&id) {
                args.push(op.clone());
            } else {
                let col = world
                    .column_by_id(id)
                    .ok_or(Error::ComponentNotFound)
                    .unwrap();
                let mut ty: ArrayTy = col.schema.to_array_ty();
                let len = col.len();
                if !(batch1 && len <= 1) {
                    ty.shape.insert(0, len as i64);
                }
                let ty = nox::NoxprTy::ArrayTy(ty);
                let var = Noxpr::parameter(vars.len() as i64, ty, vars.len().to_string());
                inputs.push((id, var.clone()));
                let arr = ComponentArray {
                    buffer: var,
                    phantom_data: PhantomData,
                    len,
                    entity_map: col.entity_map(),
                    component_id: id,
                    batch1,
                };

                vars.insert(id, arr.clone());
                args.push(arr);
            }
        }
        let args = args.into_iter().map(|a| a.buffer).collect::<Vec<_>>();
        let out = Noxpr::call(self.computation, args);
        if self.outputs.len() == 1 {
            let id = self.outputs[0];
            let col = world
                .column_by_id(id)
                .ok_or(Error::ComponentNotFound)
                .unwrap();
            let len = col.len();
            let arr = ComponentArray {
                buffer: out,
                phantom_data: PhantomData,
                len,
                entity_map: col.entity_map(),
                component_id: id,
                batch1,
            };

            vars.insert(self.outputs[0], arr);
        } else {
            for (i, id) in self.outputs.iter().enumerate() {
                let col = world
                    .column_by_id(*id)
                    .ok_or(Error::ComponentNotFound)
                    .unwrap();
                let len = col.len();
                let out = out.get_tuple_element(i);
                let arr = ComponentArray {
                    buffer: out,
                    phantom_data: PhantomData,
                    len,
                    entity_map: col.entity_map(),
                    component_id: *id,
                    batch1,
                };

                vars.insert(*id, arr);
            }
        }
        Ok(())
    }
}

pub struct SystemOutput {
    pub vars: BTreeMap<ComponentId, ComponentArray<()>>,
}

pub trait System {
    type Arg;
    type Ret;
    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error>;
    fn compile(&self, world: &World) -> Result<CompiledSystem, Error>;
}

pub trait SystemParam {
    type Item;

    fn init(builder: &mut SystemBuilder) -> Result<(), Error>;

    fn param(builder: &SystemBuilder) -> Result<Self::Item, Error>;

    fn component_ids() -> impl Iterator<Item = ComponentId>;

    fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, Error>;
}

pub struct SystemFn<M, F> {
    func: F,
    phantom_data: PhantomData<M>,
}

impl<Ret, F> System for SystemFn<(Ret,), F>
where
    F: Fn() -> Ret,
    Ret: SystemParam,
{
    type Arg = ();
    type Ret = Ret;
    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        Ret::init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        let mut builder = SystemBuilder::new(world);
        self.init(&mut builder)?;
        let output = (self.func)();
        let component_ids = Ret::component_ids().collect::<Vec<_>>();
        let noxpr = output.output(&mut builder)?;
        let ty = noxpr.ty().unwrap();
        let func = NoxprFn::new(vec![], noxpr);
        let computation = NoxprComp {
            func: Arc::new(func),
            id: NoxprId::default(),
            ty,
        };
        Ok(CompiledSystem {
            computation,
            inputs: vec![],
            outputs: component_ids,
            system_names: vec!["<system>".to_string()],
            diagnostic_subsystems: vec![],
        })
    }
}

pub trait IntoSystem<Marker, Arg, Ret> {
    type System: System<Arg = Arg, Ret = Ret>;
    fn into_system(self) -> Self::System;

    fn pipe<M2, A2, R2, OtherSys: IntoSystem<M2, A2, R2>>(
        self,
        other: OtherSys,
    ) -> Pipe<Self::System, OtherSys::System>
    where
        Self: Sized,
    {
        Pipe {
            a: self.into_system(),
            b: other.into_system(),
        }
    }
}

macro_rules! impl_system_param {
      ($($ty:tt),+) =>{
          #[allow(non_snake_case)]
          impl<$($ty,)* > SystemParam for ($($ty,)*)
            where $($ty: SystemParam,)*
          {
            type Item = ($($ty::Item,)*);

            fn init(builder: &mut SystemBuilder) -> Result<(), Error> {
                $(
                    $ty::init(builder)?;
                )*
                Ok(())
            }

            fn param(builder: &SystemBuilder) -> Result<Self::Item, Error> {
                Ok(($(
                    $ty::param(builder)?,
                )*))
            }

            fn component_ids() -> impl Iterator<Item = ComponentId> {
                std::iter::empty()
                $(
                    .chain($ty::component_ids())
                )*
            }

            fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, Error> {
                let ($($ty,)*) = self;
                let items = vec![
                    $(
                        $ty.output(builder)?,
                    )*
                ];
                Ok(Noxpr::tuple(items))
            }
          }


             impl<$($ty,)* Ret, F> IntoSystem<F, ($($ty,)*), Ret> for F
             where
                 F: Fn($($ty,)*) -> Ret,
                 F: for<'a> Fn($($ty::Item, )*) -> Ret,
                 $($ty: SystemParam,)*
                 Ret: SystemParam,
             {
                 type System = SystemFn<($($ty,)* Ret,), F>;
                 fn into_system(self) -> Self::System {
                     SystemFn {
                         func: self,
                         phantom_data: PhantomData,
                     }
                 }
             }


             impl<$($ty,)* Ret, F> System for SystemFn<($($ty,)* Ret,), F>
             where
                 F: Fn($($ty,)*) -> Ret,
                 F: for<'a> Fn($($ty::Item, )*) -> Ret,
                 $($ty: SystemParam,)*
                 Ret: SystemParam,
             {

                    type Arg = ($($ty,)*);
                    type Ret = Ret;

                    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
                        $($ty::init(builder)?;)*
                        Ret::init(builder)
                    }

                    #[allow(non_snake_case)]
                    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
                        let mut builder = SystemBuilder::new(world);
                        let builder = &mut builder;
                        self.init(builder)?;
                        $(
                            let $ty = $ty::param(builder)?;
                        )*
                        let output = (self.func)($($ty,)*);
                        let outputs = Ret::component_ids().collect::<Vec<_>>();
                        let noxpr = output.output(builder)?;
                        let args = builder.inputs.iter().map(|(_, v)| v).cloned().collect::<Vec<_>>();
                        let batch1 = builder.world.batch1;
                        let mut tys = outputs
                            .iter()
                            .map(|id| NoxprTy::ArrayTy(builder.world.column_by_id(*id).unwrap().buffer_ty_batch1(batch1)));

                        let ty = if outputs.len() == 1 {
                            tys.next().unwrap()
                        } else {
                            NoxprTy::Tuple(tys.collect())
                        };

                        let func = NoxprFn::new(args, noxpr);
                        let computation = NoxprComp {
                            func: Arc::new(func),
                            id: NoxprId::default(),
                            ty,
                        };
                        let inputs = builder.inputs.iter().map(|(k, _)| k).copied().collect::<Vec<_>>();
                        Ok(CompiledSystem {
                            computation,
                            inputs,
                            outputs,
                            system_names: vec!["<system>".to_string()],
                            diagnostic_subsystems: vec![],
                        })
                    }

             }

      }
 }

impl_system_param!(T1);
impl_system_param!(T1, T2);
impl_system_param!(T1, T2, T3);
impl_system_param!(T1, T2, T3, T4);
impl_system_param!(T1, T2, T3, T4, T5);
impl_system_param!(T1, T2, T3, T4, T5, T6);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15);
impl_system_param!(
    T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16
);
impl_system_param!(
    T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17
);
impl_system_param!(
    T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18
);

struct FnMarker;

impl<Ret, F> IntoSystem<FnMarker, (), Ret> for F
where
    F: Fn() -> Ret,
    Ret: SystemParam,
{
    type System = SystemFn<(Ret,), F>;

    fn into_system(self) -> Self::System {
        SystemFn {
            func: self,
            phantom_data: PhantomData,
        }
    }
}

pub struct SysMarker<S>(S);

impl<Arg, Ret, Sys> IntoSystem<SysMarker<Sys>, Arg, Ret> for Sys
where
    Sys: System<Arg = Arg, Ret = Ret>,
{
    type System = Sys;

    fn into_system(self) -> Self::System {
        self
    }
}

pub fn merge_compiled_systems(
    systems: impl IntoIterator<Item = CompiledSystem>,
    pipeline: &mut SystemBuilder,
) -> Result<CompiledSystem, Error> {
    for system in systems {
        system.insert_into_builder(pipeline)?;
    }
    pipeline.to_compiled_system()
}

pub struct Pipe<A: System, B: System> {
    a: A,
    b: B,
}

impl<A: System, B: System> Pipe<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

pub struct Schedule<A: System> {
    system: A,
}

impl<A: System> System for Schedule<A> {
    type Arg = A::Arg;
    type Ret = A::Ret;
    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.system.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        self.system.compile(world)
    }
}

impl<A: System, B: System> System for Pipe<A, B> {
    type Arg = (A::Arg, B::Arg);
    type Ret = (A::Ret, B::Ret);
    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.a.init(builder)?;
        self.b.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        let a = self.a.compile(world)?;
        let b = self.b.compile(world)?;
        let mut inner_builder = SystemBuilder::new(world);
        self.init(&mut inner_builder)?;
        let mut merged = merge_compiled_systems([a.clone(), b.clone()], &mut inner_builder)?;
        merged.system_names = a
            .system_names
            .iter()
            .chain(b.system_names.iter())
            .cloned()
            .collect();
        let mut diagnostic_subsystems = Vec::new();
        if a.diagnostic_subsystems.is_empty() {
            diagnostic_subsystems.push(a);
        } else {
            diagnostic_subsystems.extend(a.diagnostic_subsystems.clone());
        }
        if b.diagnostic_subsystems.is_empty() {
            diagnostic_subsystems.push(b);
        } else {
            diagnostic_subsystems.extend(b.diagnostic_subsystems.clone());
        }
        merged.diagnostic_subsystems = diagnostic_subsystems;
        Ok(merged)
    }
}

impl System for () {
    type Arg = ();
    type Ret = ();
    fn init(&self, _builder: &mut SystemBuilder) -> Result<(), Error> {
        Ok(())
    }

    fn compile(&self, _builder: &World) -> Result<CompiledSystem, Error> {
        let func = NoxprFn {
            args: vec![],
            inner: Noxpr::tuple(vec![]),
        };
        Ok(CompiledSystem {
            computation: NoxprComp::new(func, NoxprTy::Tuple(vec![])),
            inputs: vec![],
            outputs: vec![],
            system_names: vec!["<empty>".to_string()],
            diagnostic_subsystems: vec![],
        })
    }
}

impl<A, R> System for Arc<dyn System<Arg = A, Ret = R> + Send + Sync> {
    type Arg = A;

    type Ret = R;

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.as_ref().init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        self.as_ref().compile(world)
    }
}

impl<S: System> System for &S {
    type Arg = S::Arg;

    type Ret = S::Ret;

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        (*self).init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        (*self).compile(world)
    }
}

// --- ErasedSystem from mod.rs ---

pub struct ErasedSystem<Sys, Arg, Ret> {
    system: Sys,
    phantom: PhantomData<fn(Arg, Ret) -> ()>,
}

impl<Sys, Arg, Ret> ErasedSystem<Sys, Arg, Ret> {
    pub fn new(system: Sys) -> Self {
        Self {
            system,
            phantom: PhantomData,
        }
    }
}

impl<Sys, Arg, Ret> System for ErasedSystem<Sys, Arg, Ret>
where
    Sys: System<Arg = Arg, Ret = Ret>,
{
    type Arg = ();
    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.system.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        self.system.compile(world)
    }
}

// --- Python wrappers ---

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::collections::HashMap;

use crate::graph::{EdgeComponent, GraphQuery, TotalEdge};
use crate::{ComponentSchema, PyComponent};
use nox::{NoxprNode, jax::JaxNoxprFn};

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

    fn system(&self) -> PySystem {
        PySystem::new(self.clone())
    }
}

impl System for PyFnSystem {
    type Arg = ();

    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        for &id in &self.input_ids {
            builder.init_with_column(id)?;
        }
        for &id in &self.output_ids {
            builder.init_with_column(id)?;
        }

        Ok(())
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        let sys = Python::with_gil(|py| self.sys.clone_ref(py));
        let mut input_ids = self.input_ids.clone();
        let output_ids = self.output_ids.clone();
        let builder = SystemBuilder::new(world);
        let mut py_builder = PySystemBuilder {
            total_edges: GraphQuery::<TotalEdge>::param(&builder)?.edges,
            batch1_flag: world.batch1,
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
                .ok_or(Error::ComponentNotFound)?;
            let arg_metadata = ArgMetadata {
                entity_map: col.entity_map(),
                len: col.len(),
                schema: col.schema.clone(),
                component: PyComponent {
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
                .map(move |(_, value)| crate::graph::Edge::from_value(value).unwrap())
                .collect();
            py_builder.edge_map.insert(id, edges);
        }
        let batch1 = builder.world.batch1;
        let mut tys = self.output_ids.iter().map(|id| {
            NoxprTy::ArrayTy(
                builder
                    .world
                    .column_by_id(*id)
                    .unwrap()
                    .buffer_ty_batch1(batch1),
            )
        });

        let ty = if self.output_ids.len() == 1 {
            tys.next().unwrap()
        } else {
            NoxprTy::Tuple(tys.collect())
        };
        let func = Python::with_gil(|py| {
            let func = sys.call1(py, (py_builder,))?;
            let jax = py.import("jax").unwrap();
            let _jit_args = [("keep_unused", true)].into_py_dict(py);
            let jit_fn = jax.getattr("jit")?;
            Ok::<_, pyo3::PyErr>(jit_fn.call1((func,))?.into())
        })?;
        let func = NoxprFn::new(vec![], Noxpr::jax(func));
        Ok(CompiledSystem {
            computation: NoxprComp::new(func, ty),
            inputs: input_ids,
            outputs: output_ids,
            system_names: vec![self.name.clone()],
            diagnostic_subsystems: vec![],
        })
    }
}

pub trait CompiledSystemExt {
    fn compile_iree_module(
        &self,
        py: Python<'_>,
        world: &World,
        iree_device: &str,
        extra_iree_flags: &[String],
    ) -> Result<crate::iree_compile::IreeCompileResult, Error>;
    fn compile_jax_module(
        &self,
        py: Python<'_>,
        donate_argnums: &[usize],
        backend: Option<&str>,
    ) -> Result<Py<PyAny>, Error>;
}

impl CompiledSystemExt for CompiledSystem {
    fn compile_iree_module(
        &self,
        py: Python<'_>,
        world: &World,
        iree_device: &str,
        extra_iree_flags: &[String],
    ) -> Result<crate::iree_compile::IreeCompileResult, Error> {
        crate::iree_compile::compile_iree_module(py, self, world, iree_device, extra_iree_flags)
    }

    fn compile_jax_module(
        &self,
        py: Python<'_>,
        donate_argnums: &[usize],
        backend: Option<&str>,
    ) -> Result<Py<PyAny>, Error> {
        let func = noxpr_to_callable(self.computation.func.clone());

        let py_code = "
import os
os.environ.setdefault(
    'XLA_FLAGS',
    '--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true',
)
import jax
def build_expr(func, donate_argnums, backend):
    kwargs = {'keep_unused': True, 'donate_argnums': tuple(donate_argnums)}
    if backend:
        kwargs['backend'] = backend
    res = jax.jit(func, **kwargs)
    return res";

        let module = PyModule::new(py, "build_expr")?;
        let globals = module.dict();
        let code_cstr = std::ffi::CString::new(py_code).unwrap();
        py.run(code_cstr.as_ref(), Some(&globals), None)?;
        let fun: Py<PyAny> = module.getattr("build_expr")?.into();

        let comp = fun.call1(
            py,
            (func, donate_argnums.to_vec(), backend.map(str::to_string)),
        )?;

        Ok(comp)
    }
}

#[pyclass(name = "SystemBuilder")]
#[derive(Clone, Default)]
pub struct PySystemBuilder {
    arg_map: HashMap<ComponentId, (ArgMetadata, usize)>,
    pub edge_map: HashMap<ComponentId, Vec<crate::graph::Edge>>,
    pub total_edges: Vec<crate::graph::Edge>,
    pub batch1_flag: bool,
}

#[derive(Clone)]
pub struct ArgMetadata {
    pub entity_map: BTreeMap<impeller2::types::EntityId, usize>,
    pub len: usize,
    pub schema: ComponentSchema,
    pub component: PyComponent,
}

impl PySystemBuilder {
    pub fn get_var(&self, id: ComponentId) -> Option<(ArgMetadata, usize)> {
        self.arg_map.get(&id).cloned()
    }

    pub fn batch1(&self) -> bool {
        self.batch1_flag
    }
}

#[pyclass(name = "System")]
#[derive(Clone)]
pub struct PySystem {
    pub inner: Arc<dyn System<Arg = (), Ret = ()> + Send + Sync>,
}

impl PySystem {
    pub fn new(sys: impl System + Send + Sync + 'static) -> Self {
        let inner = Arc::new(ErasedSystem::new(sys));
        Self { inner }
    }
}

#[pymethods]
impl PySystem {
    pub fn pipe(&self, other: PySystem) -> PySystem {
        let pipe = Pipe::new(self.clone(), other);
        PySystem::new(pipe)
    }

    pub fn __or__(&self, other: Option<PySystem>) -> PySystem {
        match other {
            Some(other_sys) => self.pipe(other_sys),
            None => self.clone(),
        }
    }

    pub fn __ror__(&self, _other: PyObject) -> PySystem {
        self.clone()
    }
}

impl System for PySystem {
    type Arg = ();

    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.inner.init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        self.inner.compile(world)
    }
}

pub fn noxpr_to_callable(func: Arc<NoxprFn>) -> Py<PyAny> {
    if let NoxprNode::Jax(j) = &*func.inner.node {
        return Python::with_gil(|py| j.clone_ref(py));
    }
    let func = JaxNoxprFn {
        tracer: Default::default(),
        inner: func,
    };

    Python::with_gil(|py| func.into_py_any(py).unwrap())
}
