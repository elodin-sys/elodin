use elodin_conduit::{Component, ComponentId, ComponentValue};
use nox::{FromBuilder, IntoOp, Noxpr};
use std::marker::PhantomData;

pub struct Handle<T> {
    id: u64,
    _phantom: PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<T: elodin_conduit::Component> IntoOp for Handle<T> {
    fn into_op(self) -> Noxpr {
        use nox::NoxprScalarExt;
        self.id.constant()
    }
}

impl<T: elodin_conduit::Component> FromBuilder for Handle<T> {
    type Item<'a> = Handle<T>;

    fn from_builder(_builder: &nox::Builder) -> Self::Item<'_> {
        todo!()
    }
}

impl<T: elodin_conduit::Component> crate::Component for Handle<T> {
    type Inner = u64;

    type HostTy = Handle<T>;

    fn host(val: Self::HostTy) -> Self {
        val
    }

    fn component_id() -> ComponentId {
        T::component_id()
    }

    fn component_type() -> elodin_conduit::ComponentType {
        elodin_conduit::ComponentType::U64
    }
}

#[derive(Default)]
pub struct AssetStore {
    data: Vec<Asset>,
}

struct Asset {
    generation: usize,
    inner: Box<dyn ErasedComponent>,
}

impl AssetStore {
    pub fn insert<C: Component + Send + Sync + 'static>(&mut self, val: C) -> Handle<C> {
        let id = self.data.len() as u64;
        let inner = Box::new(val);
        self.data.push(Asset {
            generation: 1,
            inner,
        });
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn value<C>(&self, handle: Handle<C>) -> Option<ComponentValue<'_>> {
        let val = self.data.get(handle.id as usize)?;
        Some(val.inner.component_value())
    }

    pub fn gen<C>(&self, handle: Handle<C>) -> Option<usize> {
        let val = self.data.get(handle.id as usize)?;
        Some(val.generation)
    }
}

pub trait ErasedComponent: Send + Sync {
    fn component_id(&self) -> ComponentId;
    fn component_value(&self) -> ComponentValue<'_>;
}

impl<T: elodin_conduit::Component + Send + Sync> ErasedComponent for T {
    fn component_id(&self) -> ComponentId {
        T::component_id()
    }

    fn component_value(&self) -> ComponentValue<'_> {
        elodin_conduit::Component::component_value(self)
    }
}
