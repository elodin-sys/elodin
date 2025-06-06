use impeller2::{component::Asset, concat_str, schema::Schema};

use bytes::Bytes;
use core::marker::PhantomData;
use nox::FromBuilder;
use serde::{Deserialize, Serialize};

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

impl<T: Asset> impeller2::component::Component for Handle<T> {
    const NAME: &'static str = concat_str!("asset_handle_", T::NAME);
    const ASSET: bool = true;

    fn schema() -> Schema<Vec<u64>> {
        Schema::new(impeller2::types::PrimType::U64, [0usize; 0]).unwrap()
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
        let buf = postcard::to_allocvec(&val).unwrap();
        let Handle { id, .. } = self.insert_bytes(buf);
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

    pub fn generation<C>(&self, handle: Handle<C>) -> Option<usize> {
        let val = self.data.get(handle.id as usize)?;
        Some(val.generation)
    }

    pub fn iter(&self) -> impl Iterator<Item = &'_ AssetItem> {
        self.data.iter()
    }
}

impl<T> FromBuilder for Handle<T> {
    type Item<'a> = Handle<T>;

    fn from_builder(_builder: &nox::Builder) -> Self::Item<'_> {
        todo!()
    }
}
