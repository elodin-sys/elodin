use impeller2::{
    com_de::AsComponentView,
    types::{Msg, PacketId},
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

pub type AssetId = u64;

#[derive(Serialize, Deserialize, Debug)]
pub struct Asset<'a> {
    pub id: AssetId,
    pub buf: Cow<'a, [u8]>,
}

impl Msg for Asset<'_> {
    const ID: PacketId = [224, 14];
}

#[derive(Debug, zerocopy::IntoBytes, zerocopy::Immutable)]
#[repr(transparent)]
pub struct AssetHandle<T> {
    pub id: u64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> AssetHandle<T> {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: impeller2::component::Asset> impeller2::component::Component for AssetHandle<T> {
    const NAME: &'static str = impeller2::concat_str!("asset_handle_", T::NAME);
    const ASSET: bool = true;

    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        impeller2::schema::Schema::new(impeller2::types::PrimType::U64, [0usize; 0]).unwrap()
    }
}

impl<T> AsComponentView for AssetHandle<T> {
    fn as_component_view(&self) -> impeller2::types::ComponentView<'_> {
        self.id.as_component_view()
    }
}
