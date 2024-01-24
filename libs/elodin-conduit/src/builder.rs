use std::{borrow::Borrow, iter, marker::PhantomData};

use crate::{
    parser::varint_max, Component, ComponentData, ComponentId, ComponentType, ComponentValue,
    EntityId, Error,
};

pub struct Builder<B> {
    buf: B,
}

impl Builder<Vec<u8>> {
    pub fn with_time(time: u64) -> Result<Self, Error> {
        Builder::new(vec![], time)
    }
}

impl<B: Extend> Builder<B> {
    pub fn new(mut buf: B, time: u64) -> Result<Self, Error> {
        buf.extend_from_slice(&time.to_le_bytes())?;
        Ok(Builder { buf })
    }

    pub fn append_data(&mut self, data: ComponentData<'_>) -> Result<&mut Self, Error> {
        let Some((_, comp)) = data.storage.first() else {
            return Ok(self);
        };
        let ty = comp.ty();
        let id = data.component_id;
        let entity_iter = data.storage.iter().map(|(id, _)| id);
        let value_iter = data.storage.iter().map(|(_, val)| val);
        let entity_len = entity_iter.len();
        if entity_len != value_iter.len() {
            return Err(Error::EntityComponentLengthMismatch);
        }
        ComponentBuilder::new(
            (id, ty),
            EntityIdEncoder(entity_iter),
            ComponentValueEncoder(value_iter),
        )
        .encode(&mut self.buf)?;
        Ok(self)
    }

    pub fn append_builder<C: Encodable, E: Encodable, V: Encodable>(
        &mut self,
        builder: ComponentBuilder<C, E, V>,
    ) -> Result<&mut Self, Error> {
        builder.encode(&mut self.buf)?;
        Ok(self)
    }

    pub fn append_iter<C: Component>(
        &mut self,
        entity_iter: impl Iterator<Item = EntityId> + ExactSizeIterator,
        value_iter: impl Iterator<Item = C> + ExactSizeIterator,
    ) -> Result<&mut Self, Error> {
        self.append_builder(ComponentBuilder::new(
            ComponentIdEncoder::<C>::default(),
            EntityIdEncoder(entity_iter),
            ComponentEncoder::new(value_iter),
        ))
    }

    pub fn append_component(
        &mut self,
        entity_id: impl Into<EntityId>,
        value: impl Component,
    ) -> Result<&mut Self, Error> {
        self.append_iter(iter::once(entity_id.into()), iter::once(value))
    }

    pub fn buf(&self) -> &B {
        &self.buf
    }

    pub fn into_buf(self) -> B {
        self.buf
    }
}

impl<'l> ComponentValue<'l> {
    pub fn encode(&self, out: &mut impl Extend) -> Result<(), Error> {
        match self {
            ComponentValue::String(b) => {
                let mut arr = [0; varint_max::<usize>()];
                out.extend_from_slice(encode_varint_usize(b.as_bytes().len(), &mut arr))?;
            }
            ComponentValue::Bytes(b) => {
                let mut arr = [0; varint_max::<usize>()];
                out.extend_from_slice(encode_varint_usize(b.len(), &mut arr))?;
            }
            _ => {}
        };
        self.with_bytes(|bytes| out.extend_from_slice(bytes))
    }
}

pub struct ComponentBuilder<C, E, V> {
    component_encoder: C,
    entity_id_encoder: E,
    value_encoder: V,
}

impl<C: Encodable, E: Encodable, V: Encodable> ComponentBuilder<C, E, V> {
    pub fn new(component_encoder: C, entity_id_encoder: E, value_encoder: V) -> Self {
        Self {
            component_encoder,
            entity_id_encoder,
            value_encoder,
        }
    }

    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        self.component_encoder.encode(out)?;
        self.entity_id_encoder.encode(out)?;
        self.value_encoder.encode(out)?;
        Ok(())
    }
}

pub trait Encodable {
    fn encode(self, out: &mut impl Extend) -> Result<(), Error>;
}

impl Encodable for (ComponentId, ComponentType) {
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        let (id, ty) = self;
        out.extend_from_slice(&id.0.to_le_bytes())?;
        out.extend_from_slice(&[ty.into()])?;
        Ok(())
    }
}

pub struct ComponentIdEncoder<C>(PhantomData<C>);
impl<C: Component> Default for ComponentIdEncoder<C> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<C: Component> Encodable for ComponentIdEncoder<C> {
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        let id = C::component_id();
        let ty = C::component_type();
        (id, ty).encode(out)
    }
}

pub struct ComponentValueEncoder<I>(I);

impl<'a, V: Borrow<ComponentValue<'a>>, E: Iterator<Item = V>> Encodable
    for ComponentValueEncoder<E>
{
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        for value in self.0 {
            value.borrow().encode(out)?;
        }
        Ok(())
    }
}

pub struct EntityIdEncoder<I>(I);

impl<I: Borrow<EntityId>, E: Iterator<Item = I> + ExactSizeIterator> Encodable
    for EntityIdEncoder<E>
{
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        let entity_len = self.0.len();
        let mut arr = [0; varint_max::<usize>()];
        out.extend_from_slice(encode_varint_usize(entity_len, &mut arr))?;

        for e in self.0 {
            let e = e.borrow();
            out.extend_from_slice(&e.0.to_le_bytes())?;
        }

        Ok(())
    }
}

pub struct ComponentEncoder<I, C> {
    iter: I,
    phantom_data: PhantomData<C>,
}

impl<I, C> ComponentEncoder<I, C> {
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            phantom_data: PhantomData,
        }
    }
}

impl<C: Component, V: Iterator<Item = C> + ExactSizeIterator> Encodable for ComponentEncoder<V, C> {
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        for value in self.iter {
            let value = value.component_value();
            value.encode(out)?;
        }
        Ok(())
    }
}

impl Encodable for &'_ [u8] {
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        out.extend_from_slice(self)
    }
}

impl Encodable for (usize, &'_ [u8]) {
    fn encode(self, out: &mut impl Extend) -> Result<(), Error> {
        let (len, bytes) = self;
        let mut arr = [0; varint_max::<usize>()];
        out.extend_from_slice(encode_varint_usize(len, &mut arr))?;
        out.extend_from_slice(bytes)
    }
}

// source: https://github.com/jamesmunns/postcard/blob/a095b49935f7bd1bab04e6c5914ab11c3bbd5fee/src/varint.rs#L74
#[inline]
pub fn encode_varint_usize(n: usize, out: &mut [u8; varint_max::<usize>()]) -> &mut [u8] {
    let mut value = n;
    for i in 0..varint_max::<usize>() {
        out[i] = value.to_le_bytes()[0];
        if value < 128 {
            return &mut out[..=i];
        }

        out[i] |= 0x80;
        value >>= 7;
    }
    debug_assert_eq!(value, 0);
    &mut out[..]
}

pub trait Extend {
    fn extend_from_slice(&mut self, slice: &[u8]) -> Result<(), Error>;
}

impl Extend for Vec<u8> {
    fn extend_from_slice(&mut self, slice: &[u8]) -> Result<(), Error> {
        self.extend_from_slice(slice);
        Ok(())
    }
}

impl<'a> Extend for &'a mut Vec<u8> {
    fn extend_from_slice(&mut self, slice: &[u8]) -> Result<(), Error> {
        Vec::extend_from_slice(self, slice);
        Ok(())
    }
}

impl<'a> Extend for &'a mut [u8] {
    fn extend_from_slice(&mut self, slice: &[u8]) -> Result<(), Error> {
        self.get_mut(..slice.len())
            .ok_or(Error::BufferOverflow)?
            .copy_from_slice(slice);
        let (_, b) = std::mem::take(self).split_at_mut(slice.len());
        *self = b;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Quaternion, Vector3};

    use crate::{
        cid,
        parser::{ComponentPair, Parser},
        ComponentType,
    };

    use super::*;

    #[test]
    fn test_single_component() {
        let mut builder = Builder::with_time(1234).unwrap();
        let a = builder.append_component(1337u64, 101.05f64).unwrap().buf();

        let mut b = vec![];
        b.extend_from_slice(&1234u64.to_le_bytes()); // Time
        b.extend_from_slice(&cid!(31:0:10).0.to_le_bytes()); // Component Id
        b.extend_from_slice(&[ComponentType::F64.into()]); // Component Type
        b.extend_from_slice(&[1]); // Entity Length
        b.extend_from_slice(&1337u64.to_le_bytes()); // Entity Id
        b.extend_from_slice(&101.05f64.to_le_bytes()); // Component

        assert_eq!(a, &b)
    }

    #[test]
    fn test_write_parse() {
        let mut builder = Builder::with_time(1234).unwrap();
        let a = builder
            .append_iter(
                [EntityId(1), EntityId(2)].into_iter(),
                [1u32, 2u32].into_iter(),
            )
            .unwrap()
            .buf();
        let mut parser = Parser::new(a).unwrap();
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();
        assert_eq!(EntityId(1), entity_id);
        assert_eq!(u32::component_id(), component_id);
        assert_eq!(ComponentValue::U32(1), value);
    }

    #[test]
    fn test_bytes() {
        let mut builder = Builder::with_time(1234).unwrap();
        let a = builder
            .append_iter([EntityId(1)].into_iter(), [vec![0xBA; 512]].into_iter())
            .unwrap()
            .buf();
        let mut parser = Parser::new(a).unwrap();
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();
        assert_eq!(EntityId(1), entity_id);
        assert_eq!(Vec::<u8>::component_id(), component_id);
        assert_eq!(ComponentValue::Bytes(vec![0xBA; 512].into()), value);
    }

    #[test]
    fn test_spatial_pos() {
        let mut builder = Builder::with_time(1234).unwrap();
        let a = builder
            .append_data(ComponentData {
                component_id: cid!(0:1),
                storage: vec![(
                    EntityId(1),
                    ComponentValue::SpatialPosF64((
                        Quaternion::new(1.0, 2.0, 3.0, 4.0),
                        Vector3::new(5.0, 6.0, 7.0),
                    )),
                )],
            })
            .unwrap()
            .buf();
        let mut parser = Parser::new(a).unwrap();
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();
        assert_eq!(EntityId(1), entity_id);
        assert_eq!(cid!(0:1), component_id);
        assert_eq!(
            ComponentValue::SpatialPosF64((
                Quaternion::new(1.0, 2.0, 3.0, 4.0),
                Vector3::new(5.0, 6.0, 7.0),
            )),
            value
        );
    }
}
