use alloc::sync::Arc;
use core::marker::PhantomData;

pub trait Builder {
    type Error;
}

pub trait SystemParam<B: Builder> {
    type Item;

    fn init(builder: &mut B) -> Result<(), B::Error>;
    fn from_builder(builder: &B) -> Self::Item;
    fn insert_into_builder(self, builder: &mut B);
}

pub trait System<B: Builder> {
    type Arg;
    type Ret;

    fn init_builder(&self, builder: &mut B) -> Result<(), B::Error>;
    fn add_to_builder(&self, builder: &mut B) -> Result<(), B::Error>;
}

impl<B: Builder, Sys: System<B>> System<B> for Arc<Sys> {
    type Arg = Sys::Arg;
    type Ret = Sys::Arg;

    fn add_to_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        self.as_ref().add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        self.as_ref().init_builder(builder)
    }
}

impl<B: Builder> System<B> for Arc<dyn System<B, Arg = (), Ret = ()> + Send + Sync> {
    type Arg = ();
    type Ret = ();

    fn add_to_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        self.as_ref().add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        self.as_ref().init_builder(builder)
    }
}

pub trait IntoSystem<B: Builder, Marker, Arg, Ret> {
    type System: System<B, Arg = Arg, Ret = Ret>;
    fn into_system(self) -> Self::System;
    fn pipe<M2, A2, R2, OtherSys: IntoSystem<B, M2, A2, R2>>(
        self,
        other: OtherSys,
    ) -> Pipe<B, Self::System, OtherSys::System>
    where
        Self: Sized,
    {
        Pipe {
            a: self.into_system(),
            b: other.into_system(),
            _phantom_data: PhantomData,
        }
    }
}

pub struct SystemFn<B, M, F> {
    func: F,
    phantom_data: PhantomData<(fn() -> B, M)>,
}

macro_rules! impl_system_param {
      ($($ty:tt),+) => {
          #[allow(non_snake_case)]
          impl<B: Builder, $($ty,)* > SystemParam<B> for ($($ty,)*)
            where $($ty: SystemParam<B>,)*
          {
            type Item = ($($ty::Item,)*);

            fn init(builder: &mut B) -> Result<(), B::Error> {
                $(
                    $ty::init(builder)?;
                )*
                Ok(())
            }

            fn from_builder(builder: &B) -> Self::Item {
                ($(
                    $ty::from_builder(builder),
                )*)
            }

            fn insert_into_builder(self, builder: &mut B) {
                let ($($ty,)*) = self;
                $(
                    $ty.insert_into_builder(builder);
                )*
            }
          }


            impl<B: Builder, $($ty,)* Ret, F> IntoSystem<B, F, ($($ty,)*), Ret> for F
            where
                F: Fn($($ty,)*) -> Ret,
                F: for<'a> Fn($($ty::Item, )*) -> Ret,
                $($ty: SystemParam<B>,)*
                Ret: SystemParam<B>,
            {
                type System = SystemFn<B, ($($ty,)* Ret,), F>;
                fn into_system(self) -> Self::System {
                    SystemFn {
                        func: self,
                        phantom_data: PhantomData,
                    }
                }
            }


            impl<B: Builder, $($ty,)* Ret, F> System<B> for SystemFn<B, ($($ty,)* Ret,), F>
            where
                F: Fn($($ty,)*) -> Ret,
                F: for<'a> Fn($($ty::Item, )*) -> Ret,
                $($ty: SystemParam<B>,)*
                Ret: SystemParam<B>,
            {
                type Arg = ($($ty,)*);
                type Ret = Ret;
                fn init_builder(&self, builder: &mut B) -> Result<(), B::Error> {
                    $(
                        $ty::init(builder)?;
                    )*
                    Ok(())
                }
                fn add_to_builder(&self, builder: &mut B) -> Result<(), B::Error> {
                    let ret = (self.func)(
                        $(
                            $ty::from_builder(builder),
                        )*
                    );
                    ret.insert_into_builder(builder);
                    Ok(())
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

impl<B: Builder, Ret, F> System<B> for SystemFn<B, (Ret,), F>
where
    F: Fn() -> Ret,
    Ret: SystemParam<B>,
{
    type Arg = ();
    type Ret = Ret;

    fn init_builder(&self, _: &mut B) -> Result<(), B::Error> {
        Ok(())
    }

    fn add_to_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        let ret = (self.func)();
        ret.insert_into_builder(builder);
        Ok(())
    }
}

struct FnMarker;

impl<B: Builder, Ret, F> IntoSystem<B, FnMarker, (), Ret> for F
where
    F: Fn() -> Ret,
    Ret: SystemParam<B>,
{
    type System = SystemFn<B, (Ret,), F>;

    fn into_system(self) -> Self::System {
        SystemFn {
            func: self,
            phantom_data: PhantomData,
        }
    }
}

pub struct SysMarker<S>(S);

impl<B: Builder, Arg, Ret, Sys> IntoSystem<B, SysMarker<Sys>, Arg, Ret> for Sys
where
    Sys: System<B, Arg = Arg, Ret = Ret>,
{
    type System = Sys;

    fn into_system(self) -> Self::System {
        self
    }
}

pub struct Pipe<Builda: Builder, A: System<Builda>, B: System<Builda>> {
    a: A,
    b: B,
    _phantom_data: PhantomData<fn() -> Builda>,
}

impl<Builda: Builder, A: System<Builda>, B: System<Builda>> Pipe<Builda, A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _phantom_data: PhantomData,
        }
    }
}

impl<Builda: Builder, A: System<Builda>, B: System<Builda>> System<Builda> for Pipe<Builda, A, B> {
    type Arg = (A::Arg, B::Arg);
    type Ret = (A::Ret, B::Ret);

    fn add_to_builder(&self, builder: &mut Builda) -> Result<(), Builda::Error> {
        self.a.add_to_builder(builder)?;
        self.b.add_to_builder(builder)
    }

    fn init_builder(&self, builder: &mut Builda) -> Result<(), Builda::Error> {
        self.a.init_builder(builder)?;
        self.b.init_builder(builder)
    }
}

pub struct JoinSystem<B> {
    systems: Vec<Box<dyn System<B, Arg = (), Ret = ()>>>,
}

impl<B: Builder> System<B> for JoinSystem<B> {
    type Arg = ();
    type Ret = ();
    fn add_to_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        for system in &self.systems {
            system.add_to_builder(builder)?;
        }
        Ok(())
    }

    fn init_builder(&self, builder: &mut B) -> Result<(), B::Error> {
        for system in &self.systems {
            system.init_builder(builder)?;
        }
        Ok(())
    }
}

impl<B: Builder> System<B> for () {
    type Arg = ();
    type Ret = ();
    fn add_to_builder(&self, _builder: &mut B) -> Result<(), B::Error> {
        Ok(())
    }

    fn init_builder(&self, _builder: &mut B) -> Result<(), B::Error> {
        Ok(())
    }
}
