use bytes::{buf::UninitSlice, BufMut};
use conduit::{
    ser_de::Frozen, Component, ComponentId, ComponentValue, ComponentValueDim, ConstComponent,
    EntityId, ValueRepr,
};
use heapless::FnvIndexMap;
use tracing::warn;

use crate::{Componentize, Decomponentize};

pub struct Column<V: ValueRepr + Component + ConstComponent, const N: usize> {
    map: FnvIndexMap<EntityId, V, N>,
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Column<V, N> {
    pub fn new() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Default for Column<V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Decomponentize for Column<V, N> {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentValue<'_, D>,
    ) {
        if component_id != ComponentId::new(V::NAME) {
            return;
        }
        if let Some(val) = V::from_component_value(value) {
            if self.map.insert(entity_id, val).is_err() {
                warn!("column map full");
            }
        }
    }
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Componentize for Column<V, N> {
    fn sink_columns(&self, output: &mut impl Decomponentize) {
        self.map.iter().for_each(|(id, value)| {
            let value = value.fixed_dim_component_value();
            output.apply_value(ComponentId::new(V::NAME), *id, value);
        });
    }
}

#[derive(Default)]
pub struct FixedBuffer<const N: usize>(pub heapless::Vec<u8, N>);

impl<const N: usize> AsRef<[u8]> for FixedBuffer<N> {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl<const N: usize> Frozen for FixedBuffer<N> {
    type Freezable = FixedBuffer<N>;

    fn freeze(val: Self::Freezable) -> Self {
        val
    }
}

unsafe impl<const N: usize> BufMut for FixedBuffer<N> {
    fn remaining_mut(&self) -> usize {
        N - self.0.len()
    }

    unsafe fn advance_mut(&mut self, cnt: usize) {
        let pos = self.0.len() + cnt;
        if pos >= N {
            panic!("Advance out of range");
        }
        self.0.set_len(pos);
    }

    fn chunk_mut(&mut self) -> &mut bytes::buf::UninitSlice {
        let len = self.0.len();

        let ptr = self.0.as_mut_ptr();
        unsafe { &mut UninitSlice::from_raw_parts_mut(ptr, N)[len..] }
    }
}
