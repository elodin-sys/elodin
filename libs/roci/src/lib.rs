use conduit::{
    ser_de::Frozen, ColumnPayload, Component, ComponentId, ConstComponent, EntityId, Metadata,
    ValueRepr,
};
use heapless::FnvIndexMap;

pub use conduit;

pub use roci_macros::{Componentize, Decomponentize};
use tracing::warn;

pub mod flume;
pub mod tokio;

pub trait Decomponentize {
    fn apply_column<B: AsRef<[u8]>>(&mut self, metadata: &Metadata, payload: &ColumnPayload<B>);
}

pub trait Componentize {
    fn sink_columns<Buf: Frozen>(&self, output: &mut impl ColumnSink<Buf>);
    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata>;
}

pub trait ColumnSink<Buf: Frozen> {
    fn sink_column(&mut self, component_id: ComponentId, payload: ColumnPayload<Buf>);
}

pub trait Handler {
    type World: Default + Decomponentize + Componentize;
    fn tick(&mut self, world: &mut Self::World);
}

pub struct Column<V: ValueRepr + Component + ConstComponent, const N: usize> {
    map: FnvIndexMap<EntityId, V, N>,
    metadata: Metadata,
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Column<V, N> {
    pub fn new() -> Self {
        let metadata = Metadata {
            name: V::NAME.into(),
            component_type: V::TY,
            tags: Default::default(),
            asset: V::ASSET,
        };
        Self {
            map: Default::default(),
            metadata,
        }
    }
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Default for Column<V, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Decomponentize for Column<V, N> {
    fn apply_column<B: AsRef<[u8]>>(&mut self, metadata: &Metadata, payload: &ColumnPayload<B>) {
        if metadata.name != V::NAME {
            return;
        }
        let payload = payload.as_ref();
        let mut iter = payload.into_iter(metadata.component_type.clone());
        while let Some(Ok(conduit::ser_de::ColumnValue { entity_id, value })) = iter.next() {
            if let Some(val) = V::from_component_value(value) {
                if self.map.insert(entity_id, val).is_err() {
                    warn!("column map full");
                }
            }
        }
    }
}

impl<V: ValueRepr + Component + ConstComponent, const N: usize> Componentize for Column<V, N> {
    fn sink_columns<Buf: Frozen>(&self, output: &mut impl ColumnSink<Buf>) {
        output.sink_column(
            self.metadata.component_id(),
            conduit::ColumnPayload::try_from_value_iter(
                0,
                self.map
                    .iter()
                    .map(|(id, value)| conduit::ser_de::ColumnValue {
                        entity_id: *id,
                        value: value.component_value(),
                    }),
            )
            .unwrap(),
        );

        // let mut payload = Vec::new();
        // for (entity_id, value) in self.map.iter() {
        //     payload.push(conduit::ser_de::ColumnValue {
        //         entity_id: *entity_id,
        //         value: value.to_component_value(),
        //     });
        // }
        // output.sink_column(ComponentId::new(V::NAME), ColumnPayload::new(payload));
    }

    fn get_metadata(&self, component_id: ComponentId) -> Option<&Metadata> {
        if component_id == ComponentId::new(V::NAME) {
            Some(&self.metadata)
        } else {
            None
        }
    }
}
