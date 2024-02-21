use bytes::Bytes;
use elodin_conduit::{Asset, AssetId, ComponentId, ComponentValue};
use nox::{FromBuilder, IntoOp, Noxpr};

use std::{collections::HashMap, marker::PhantomData};

#[derive(Debug)]
pub struct Handle<T> {
    pub id: AssetId,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    pub fn new(id: AssetId) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<T> IntoOp for Handle<T> {
    fn into_op(self) -> Noxpr {
        use nox::NoxprScalarExt;
        self.id.0.constant()
    }
}

impl<T> FromBuilder for Handle<T> {
    type Item<'a> = Handle<T>;

    fn from_builder(_builder: &nox::Builder) -> Self::Item<'_> {
        todo!()
    }
}

impl<T: Asset> crate::Component for Handle<T> {
    type Inner = u64;

    type HostTy = Handle<T>;

    fn host(val: Self::HostTy) -> Self {
        val
    }

    fn component_id() -> ComponentId {
        ComponentId(T::ASSET_ID.0)
    }

    fn component_type() -> elodin_conduit::ComponentType {
        elodin_conduit::ComponentType::u64()
    }

    fn is_asset() -> bool {
        true
    }
}

#[derive(Default, Clone)]
pub struct AssetStore {
    map: HashMap<AssetId, usize>,
    data: Vec<AssetItem>,
}

#[derive(Clone)]
pub struct AssetItem {
    pub generation: usize,
    pub inner: Bytes,
    pub asset_id: AssetId,
}

impl AssetStore {
    pub fn insert<A: Asset + Send + Sync + 'static>(&mut self, val: A) -> Handle<A> {
        let asset_id = val.asset_id();
        let Handle { id, .. } = self.insert_bytes(asset_id, postcard::to_allocvec(&val).unwrap());
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn insert_bytes(&mut self, asset_id: AssetId, bytes: impl Into<Bytes>) -> Handle<()> {
        let inner = bytes.into();
        let id = self.data.len();
        self.map.insert(asset_id, id);
        self.data.push(AssetItem {
            generation: 1,
            inner,
            asset_id,
        });
        Handle {
            id: asset_id,
            _phantom: PhantomData,
        }
    }

    pub fn value<C>(&self, handle: Handle<C>) -> Option<&AssetItem> {
        let id = self.map.get(&handle.id)?;
        let val = self.data.get(*id)?;
        Some(val)
    }

    pub fn gen<C>(&self, handle: Handle<C>) -> Option<usize> {
        let id = self.map.get(&handle.id)?;
        let val = self.data.get(*id)?;
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
