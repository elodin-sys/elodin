use conduit::{ComponentId, ComponentValue, ComponentValueDim, EntityId};

pub trait Decomponentize {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentValue<'_, D>,
    );
}

impl Decomponentize for () {
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        _component_id: ComponentId,
        _entity_id: EntityId,
        _value: ComponentValue<'_, D>,
    ) {
    }
}

impl<T1, T2> Decomponentize for (T1, T2)
where
    T1: Decomponentize,
    T2: Decomponentize,
{
    fn apply_value<D: ComponentValueDim>(
        &mut self,
        component_id: ComponentId,
        entity_id: EntityId,
        value: ComponentValue<'_, D>,
    ) {
        self.0.apply_value(component_id, entity_id, value.clone());
        self.1.apply_value(component_id, entity_id, value);
    }
}
