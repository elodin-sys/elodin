use heapless::FnvIndexMap;
use impeller2::com_de::{AsComponentView, FromComponentView};
use impeller2::component::Component;
use impeller2::types::{ComponentId, ComponentView, EntityId};
use tracing::warn;

use crate::{Componentize, Decomponentize};

pub struct Column<V: Component, const N: usize> {
    map: FnvIndexMap<EntityId, V, N>,
}

impl<V: Component, const N: usize> Column<V, N> {
    pub fn new() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

impl<V: Component, const N: usize> Default for Column<V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Component + FromComponentView, const N: usize> Decomponentize for Column<V, N> {
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentView<'_>,
    ) {
        if component_id != ComponentId::new(V::NAME) {
            return;
        }
        if let Ok(val) = V::from_component_view(value) {
            if self.map.insert(entity_id, val).is_err() {
                warn!("column map full");
            }
        }
    }
}

impl<V: Component + AsComponentView, const N: usize> Componentize for Column<V, N> {
    fn sink_columns(&self, output: &mut impl Decomponentize) {
        self.map.iter().for_each(|(id, value)| {
            output.apply_value(ComponentId::new(V::NAME), *id, value.as_component_view());
        });
    }
}
