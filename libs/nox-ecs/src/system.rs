use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};

use impeller::{ComponentId, World};
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

        let mut tys = self
            .vars
            .keys()
            .map(|id| NoxprTy::ArrayTy(self.world.column_by_id(*id).unwrap().buffer_ty()));

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
        })
    }

    pub fn init_with_column(&mut self, id: ComponentId) -> Result<(), Error> {
        if self.vars.contains_key(&id) {
            return Ok(());
        }
        let column = self
            .world
            .column_by_id(id)
            .ok_or(Error::ComponentNotFound)?;
        let len = column.len();
        let mut ty: ArrayTy = column.metadata.component_type.clone().into();
        ty.shape.insert(0, len as i64);
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
}

impl CompiledSystem {
    pub fn insert_into_builder(self, builder: &mut SystemBuilder) -> Result<(), Error> {
        let mut args = vec![];
        let world = &mut builder.world;
        let vars = &mut builder.vars;
        let inputs = &mut builder.inputs;
        for id in self.inputs {
            if let Some(op) = vars.get(&id) {
                args.push(op.clone());
            } else {
                let col = world.column_by_id(id).ok_or(Error::ComponentNotFound)?;
                let mut ty: ArrayTy = col.metadata.component_type.clone().into();
                ty.shape.insert(0, col.len() as i64);
                let ty = nox::NoxprTy::ArrayTy(ty);
                let var = Noxpr::parameter(vars.len() as i64, ty, vars.len().to_string());
                inputs.push((id, var.clone()));
                let len = col.len();
                let arr = ComponentArray {
                    buffer: var,
                    phantom_data: PhantomData,
                    len,
                    entity_map: col.entity_map(),
                    component_id: id,
                };

                vars.insert(id, arr.clone());
                args.push(arr);
            }
        }
        let args = args.into_iter().map(|a| a.buffer).collect::<Vec<_>>();
        let out = Noxpr::call(self.computation, args);
        if self.outputs.len() == 1 {
            let id = self.outputs[0];
            let col = world.column_by_id(id).ok_or(Error::ComponentNotFound)?;
            let len = col.len();
            let arr = ComponentArray {
                buffer: out,
                phantom_data: PhantomData,
                len,
                entity_map: col.entity_map(),
                component_id: id,
            };

            vars.insert(self.outputs[0], arr);
        } else {
            for (i, id) in self.outputs.iter().enumerate() {
                let col = world.column_by_id(*id).ok_or(Error::ComponentNotFound)?;
                let len = col.len();
                let out = out.get_tuple_element(i);
                let arr = ComponentArray {
                    buffer: out,
                    phantom_data: PhantomData,
                    len,
                    entity_map: col.entity_map(),
                    component_id: *id,
                };

                vars.insert(*id, arr);
            }
        }
        Ok(())
    }

    // pub fn insert_into_builder(self, builder: &mut SystemBuilder) -> Result<(), Error> {
    //     let out = self.call(&mut builder.vars, &builder.world)?;
    //     builder.vars.extend(out.vars);
    //     Ok(())
    // }
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
                        let mut tys = outputs
                            .iter()
                            .map(|id| NoxprTy::ArrayTy(builder.world.column_by_id(*id).unwrap().buffer_ty()));

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
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17);
impl_system_param!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18);

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

        merge_compiled_systems([a, b], &mut inner_builder)
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
