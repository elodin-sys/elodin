use std::{collections::BTreeMap, ops::Deref};

use bytemuck::Pod;
use conduit::{ComponentValue, EntityId};
use nox::{
    xla::{ArrayElement, PjRtBuffer},
    Client, NoxprNode,
};
use polars::series::Series;

use crate::{polars::to_series, ColumnRef, Component, Error};

impl<'a, B: 'a + AsRef<[u8]>> ColumnRef<'a, B> {
    pub fn typed_buf<T: ArrayElement + Pod>(&self) -> Option<&[T]> {
        if self.metadata.component_type.primitive_ty.element_type() != T::TY {
            return None;
        }
        bytemuck::try_cast_slice(self.column.as_ref()).ok()
    }

    pub fn entity_ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        bytemuck::cast_slice::<_, u64>(self.entities.as_ref())
            .iter()
            .copied()
            .map(EntityId)
    }

    pub fn entity_map(&self) -> BTreeMap<EntityId, usize> {
        self.entity_ids()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect()
    }

    pub fn values_iter(&self) -> impl Iterator<Item = ComponentValue<'_>> + '_ {
        let mut buf_offset = 0;
        std::iter::from_fn(move || {
            let buf = self.column.as_ref().get(buf_offset..)?;
            let (offset, value) = self.metadata.component_type.parse_value(buf).ok()?;
            buf_offset += offset;
            Some(value)
        })
    }

    pub fn copy_to_client(&self, client: &Client) -> Result<PjRtBuffer, Error> {
        let mut dims = self.metadata.component_type.shape.clone();
        dims.insert(0, self.len() as i64);
        client
            .copy_raw_host_buffer(
                self.metadata.component_type.primitive_ty.element_type(),
                self.column.as_ref(),
                &dims[..],
            )
            .map_err(Error::from)
    }

    pub fn series(&self) -> Result<Series, Error> {
        to_series(self.column.as_ref(), self.metadata)
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, ComponentValue<'_>)> {
        self.entity_ids().zip(self.values_iter())
    }

    pub fn typed_iter<T: conduit::Component>(&self) -> impl Iterator<Item = (EntityId, T)> + '_ {
        assert_eq!(self.metadata.component_type, T::component_type());
        self.entity_ids().zip(
            self.values_iter()
                .filter_map(|v| T::from_component_value(v)),
        )
    }

    pub fn len(&self) -> usize {
        self.column.as_ref().len() / self.metadata.component_type.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> ColumnRef<'a, &'a mut Vec<u8>> {
    pub fn typed_buf_mut<T: ArrayElement + Pod>(&mut self) -> Option<&mut [T]> {
        if self.metadata.component_type.primitive_ty.element_type() != T::TY {
            return None;
        }
        bytemuck::try_cast_slice_mut(self.column.as_mut_slice()).ok()
    }

    pub fn update(&mut self, offset: usize, value: ComponentValue<'_>) -> Result<(), Error> {
        let size = self.metadata.component_type.size();
        let offset = offset * size;
        let out = &mut self.column[offset..offset + size];
        if let Some(bytes) = value.bytes() {
            if bytes.len() != out.len() {
                return Err(Error::ValueSizeMismatch);
            }
            out.copy_from_slice(bytes);
        }
        Ok(())
    }

    pub fn push<T: Component + 'static>(&mut self, val: T) {
        assert_eq!(self.metadata.component_type, T::component_type());
        let op = val.into_op();
        let NoxprNode::Constant(c) = op.deref() else {
            panic!("push into host column must be constant expr");
        };
        self.push_raw(c.data.raw_buf());
    }

    pub fn push_raw(&mut self, raw: &[u8]) {
        self.column.extend_from_slice(raw);
    }
}
