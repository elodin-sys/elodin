use crate::{concat_str, Asset, ComponentType};

use alloc::vec::Vec;
use bytes::Bytes;
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

use crate::Component;

#[derive(Debug)]
pub struct Handle<T> {
    pub id: u64,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<T: Asset> Component for Handle<T> {
    const NAME: &'static str = concat_str!("asset_handle_", T::ASSET_NAME);
    const ASSET: bool = true;

    fn component_type() -> ComponentType {
        ComponentType::u64()
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct AssetStore {
    data: Vec<AssetItem>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct AssetItem {
    pub generation: usize,
    pub inner: Bytes,
}

impl AssetStore {
    pub fn insert<A: Asset + Send + Sync + 'static>(&mut self, val: A) -> Handle<A> {
        let Handle { id, .. } = self.insert_bytes(postcard::to_allocvec(&val).unwrap());
        Handle {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn insert_bytes(&mut self, bytes: impl Into<Bytes>) -> Handle<()> {
        let inner = bytes.into();
        let id = self.data.len();
        self.data.push(AssetItem {
            generation: 1,
            inner,
        });
        Handle {
            id: id as u64,
            _phantom: PhantomData,
        }
    }

    pub fn value<C>(&self, handle: Handle<C>) -> Option<&AssetItem> {
        let val = self.data.get(handle.id as usize)?;
        Some(val)
    }

    pub fn gen<C>(&self, handle: Handle<C>) -> Option<usize> {
        let val = self.data.get(handle.id as usize)?;
        Some(val.generation)
    }
}

#[cfg(feature = "xla")]
mod nox_impl {
    use super::*;
    use crate::{types::ComponentExt, Archetype, ArchetypeName};
    use nox::FromBuilder;

    impl<T: Asset + 'static> Archetype for Handle<T> {
        fn name() -> crate::ArchetypeName {
            ArchetypeName::from(Self::NAME)
        }

        fn components() -> Vec<crate::Metadata> {
            vec![Self::metadata()]
        }

        fn insert_into_world(self, world: &mut crate::World) {
            let mut col = world.column_mut::<Self>().unwrap();
            col.push_raw(&self.id.to_le_bytes());
        }
    }

    impl<T> FromBuilder for Handle<T> {
        type Item<'a> = Handle<T>;

        fn from_builder(_builder: &nox::Builder) -> Self::Item<'_> {
            todo!()
        }
    }
}
