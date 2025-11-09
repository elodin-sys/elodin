use crate::ecs::Component;
use crate::ecs::query::intersect_ids;
use crate::ecs::system::{SystemBuilder, SystemParam};
use crate::ecs::utils::SchemaExt;
use impeller2::types::{ComponentId, EntityId};
use nox::{ArrayTy, CompFn, Noxpr};
use smallvec::{SmallVec, smallvec};
use std::collections::BTreeMap;
use std::iter::once;
use std::marker::PhantomData;

pub struct ComponentArray<T> {
    pub buffer: Noxpr,
    pub len: usize,
    pub entity_map: BTreeMap<EntityId, usize>,
    pub phantom_data: PhantomData<T>,
    pub component_id: ComponentId,
}

impl<T> Clone for ComponentArray<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            len: self.len,
            entity_map: self.entity_map.clone(),
            phantom_data: PhantomData,
            component_id: self.component_id,
        }
    }
}

impl<T> ComponentArray<T> {
    pub(crate) fn cast<D>(self) -> ComponentArray<D> {
        ComponentArray {
            buffer: self.buffer,
            phantom_data: PhantomData,
            entity_map: self.entity_map,
            len: self.len,
            component_id: self.component_id,
        }
    }

    pub fn buffer(&self) -> &Noxpr {
        &self.buffer
    }
}

impl<T: Component> ComponentArray<T> {
    pub fn get(&self, offset: i64) -> T {
        let ty: ArrayTy = T::schema().to_array_ty();
        let shape = ty.shape;

        let start_indices: SmallVec<_> = once(offset).chain(shape.iter().map(|_| 0)).collect();
        let stop_indices: SmallVec<_> = once(offset + 1).chain(shape.clone()).collect();
        let strides: SmallVec<_> = smallvec![1; shape.len() + 1];

        let op = self
            .buffer
            .clone()
            .slice(start_indices, stop_indices, strides)
            .reshape(shape);
        T::from_inner(op)
    }
}

impl<T: Component + 'static> SystemParam for ComponentArray<T> {
    type Item = Self;

    fn init(builder: &mut SystemBuilder) -> Result<(), crate::Error> {
        let id = T::COMPONENT_ID;
        builder.init_with_column(id)?;
        Ok(())
    }

    fn param(builder: &SystemBuilder) -> Result<Self::Item, crate::Error> {
        let id = T::COMPONENT_ID;
        if let Some(var) = builder.vars.get(&id) {
            Ok(var.clone().cast())
        } else {
            Err(crate::Error::ComponentNotFound)
        }
    }

    fn component_ids() -> impl Iterator<Item = ComponentId> {
        std::iter::once(T::COMPONENT_ID)
    }

    fn output(&self, builder: &mut SystemBuilder) -> Result<Noxpr, crate::Error> {
        if let Some(var) = builder.vars.get_mut(&T::COMPONENT_ID)
            && var.entity_map != self.entity_map
        {
            return Ok(update_var(
                &var.entity_map,
                &self.entity_map,
                &var.buffer,
                &self.buffer,
            ));
        }
        Ok(self.buffer.clone())
    }
}

pub fn update_var(
    old_entity_map: &BTreeMap<EntityId, usize>,
    update_entity_map: &BTreeMap<EntityId, usize>,
    old_buffer: &Noxpr,
    update_buffer: &Noxpr,
) -> Noxpr {
    use nox::NoxprScalarExt;
    let (old, new, _) = intersect_ids(old_entity_map, update_entity_map);
    let shape = update_buffer.shape().unwrap();
    old.iter().zip(new.iter()).fold(
        old_buffer.clone(),
        |buffer, (existing_index, update_index)| {
            let mut start = shape.clone();
            start[0] = *update_index as i64;
            for x in start.iter_mut().skip(1) {
                *x = 0;
            }
            let mut stop = shape.clone();
            stop[0] = *update_index as i64 + 1;
            let start = std::iter::once(*update_index as i64)
                .chain(std::iter::repeat_n(0, shape.len() - 1))
                .collect();
            let existing_index = std::iter::once((*existing_index as i64).constant())
                .chain(std::iter::repeat_n(0i64.constant(), shape.len() - 1))
                .collect();
            buffer.dynamic_update_slice(
                existing_index,
                update_buffer
                    .clone()
                    .slice(start, stop, shape.iter().map(|_| 1).collect()),
            )
        },
    )
}

impl<C: Component> ComponentArray<C> {
    pub fn map<O: Component>(
        &self,
        func: impl CompFn<(C,), O>,
    ) -> Result<ComponentArray<O>, crate::Error> {
        let func = func.build_expr()?;
        let buffer = Noxpr::vmap_with_axis(func, &[0], std::slice::from_ref(&self.buffer))?;
        Ok(ComponentArray {
            buffer,
            len: self.len,
            phantom_data: PhantomData,
            entity_map: self.entity_map.clone(),
            component_id: O::COMPONENT_ID,
        })
    }
}

