//! [`Parser`] provides a mechanism for parsing conduit messages
//!
//! Conduit utilizes a custom data format optimized for space
//! The data format looks roughly like this:
//!
//! ```text
//!  +--------------------------+
//!  |        Time (u64)        |
//!  +--------------------------+
//!  |   Components (Repeated)  |
//!  +--------------------------+
//!    |  Component Id (u64)  |
//!    +----------------------+
//!    |    Value Type (u8)   |
//!    +----------------------+
//!    |  Entity Len (varint) |
//!    +----------------------+
//!    |     Entities (u64)   |
//!    +----------------------+
//!    |        Values        |
//!    +----------------------+
//! ```
//!
//! Each packet always contains a time header, then it contains some number of repeated "component" fields. A component field contains, an id, the type of the values, a list of entites, and a list of data values.
//! The goal is to allow the list of entries to be directly loaded into memory in some situations. For instance, if you are transmitting a series of `Vector3<f32>`, since they are all stored contingously, you could
//! load them all into memory as a single `Tensor<3, Dyn>`.
//!
//! You'll notice that this packet does not include a length for the components. This is because the entire packet needs to be length-delimited, and components are of variable length.
//! Parsing is done by walking through the entities and components until all are found, or the end of the buffer occurs.
//!
//! This packet format is also optimized for space over speed of deserialization. Values are typically unaligned, varints are used for length. This is so the parser can exist on
//!
use crate::{
    ComponentBatch, ComponentData, ComponentFilter, ComponentId, ComponentType, ComponentValue,
    EntityId,
};
use nalgebra::{Matrix3, Quaternion, Vector3, Vector4};
use std::{borrow::Cow, mem::size_of};

pub struct Parser<B> {
    time: u64,
    buf: B,
    component_buf_pos: usize,
    parser: ComponentParser,
}

impl<B: AsRef<[u8]>> Parser<B> {
    pub fn new(buf: B) -> Option<Self> {
        let time = buf.as_ref().get_u64()?;
        let component_buf = buf.as_ref().get(8..)?;
        let parser = ComponentParser::new(component_buf)?;
        Some(Parser {
            time,
            buf,
            component_buf_pos: 8,
            parser,
        })
    }

    pub fn parse_next(&mut self) -> Option<ComponentPair<'_>> {
        let component_buf = self.buf.as_ref().get(self.component_buf_pos..)?;
        if let Some(value) = self.parser.next(component_buf) {
            Some(value)
        } else {
            self.component_buf_pos += self.parser.component_buf_pos;
            let component_buf = self.buf.as_ref().get(self.component_buf_pos..)?;
            self.parser = ComponentParser::new(component_buf)?;
            self.parser.next(component_buf)
        }
    }

    pub fn time(&self) -> u64 {
        self.time
    }

    pub fn parse_data_msg(&mut self) -> Option<ComponentBatch<'static>> {
        let mut data_msg = ComponentBatch {
            time: self.time,
            components: vec![],
        };
        let mut current_component_id = None;
        for pair in self {
            if Some(pair.component_id) == current_component_id {
                let Some(component) = data_msg.components.last_mut() else {
                    unreachable!();
                };
                component
                    .storage
                    .push((pair.entity_id, pair.value.into_owned()));
            } else {
                current_component_id = Some(pair.component_id);
                data_msg.components.push(ComponentData {
                    component_id: pair.component_id,
                    storage: vec![(pair.entity_id, pair.value.into_owned())],
                });
            }
        }
        Some(data_msg)
    }
}

impl<B: AsRef<[u8]>> Iterator for Parser<B> {
    type Item = ComponentPair<'static>;

    fn next(&mut self) -> Option<Self::Item> {
        self.parse_next().map(|p| ComponentPair {
            component_id: p.component_id,
            entity_id: p.entity_id,
            value: p.value.into_owned(),
        })
    }
}

pub struct ComponentPair<'l> {
    pub component_id: ComponentId,
    pub entity_id: EntityId,
    pub value: ComponentValue<'l>,
}

struct ComponentParser {
    id: u64,
    ty: ComponentType,
    len: usize,
    pos: usize,
    entity_buf_pos: usize,
    component_buf_pos: usize,
}

impl ComponentParser {
    pub fn new(buf: &[u8]) -> Option<Self> {
        let orig_len = buf.len();
        let parse_buf = &mut &buf[..];
        let id = parse_buf.pop_u64()?;
        let ty = parse_buf.pop_u8()?;
        let ty = ComponentType::try_from(ty).ok()?;
        let len = parse_buf.pop_varint()?;
        let entity_buf_pos = orig_len - parse_buf.len();
        let component_buf_pos = entity_buf_pos + size_of::<u64>() * len;
        if component_buf_pos >= buf.len() {
            return None;
        }
        Some(Self {
            id,
            ty,
            len,
            pos: 0,
            entity_buf_pos,
            component_buf_pos,
        })
    }
    fn next<'a>(&mut self, buf: &'a [u8]) -> Option<ComponentPair<'a>> {
        if self.pos >= self.len {
            return None;
        }
        let entity_id = (&buf[self.entity_buf_pos..]).get_u64()?;
        self.entity_buf_pos += 8;
        let Some(component_buf) = buf.get(self.component_buf_pos..) else {
            return None;
        };
        let Some((offset, value)) = self.ty.parse(component_buf) else {
            return None;
        };
        self.component_buf_pos += offset;
        self.pos += 1;
        Some(ComponentPair {
            component_id: ComponentId(self.id),
            entity_id: EntityId(entity_id),
            value,
        })
    }
}

impl ComponentType {
    pub fn parse<'a>(&self, mut buf: &'a [u8]) -> Option<(usize, ComponentValue<'a>)> {
        match self {
            ComponentType::U8 => buf
                .get(..size_of::<u8>())
                .and_then(|u| u.try_into().ok())
                .map(u8::from_le_bytes)
                .map(|b| (size_of::<u8>(), ComponentValue::U8(b))),
            ComponentType::U16 => buf
                .get(..size_of::<u16>())
                .and_then(|u| u.try_into().ok())
                .map(u16::from_le_bytes)
                .map(|b| (size_of::<u16>(), ComponentValue::U16(b))),
            ComponentType::U32 => buf
                .get(..size_of::<u32>())
                .and_then(|u| u.try_into().ok())
                .map(u32::from_le_bytes)
                .map(|b| (size_of::<u32>(), ComponentValue::U32(b))),
            ComponentType::U64 => buf
                .get(..size_of::<u64>())
                .and_then(|u| u.try_into().ok())
                .map(u64::from_le_bytes)
                .map(|b| (size_of::<u64>(), ComponentValue::U64(b))),
            ComponentType::I8 => buf
                .get(..size_of::<i8>())
                .and_then(|u| u.try_into().ok())
                .map(i8::from_le_bytes)
                .map(|b| (size_of::<i8>(), ComponentValue::I8(b))),
            ComponentType::I16 => buf
                .get(..size_of::<i16>())
                .and_then(|u| u.try_into().ok())
                .map(i16::from_le_bytes)
                .map(|b| (size_of::<i16>(), ComponentValue::I16(b))),
            ComponentType::I32 => buf
                .get(..size_of::<i32>())
                .and_then(|u| u.try_into().ok())
                .map(i32::from_le_bytes)
                .map(|b| (size_of::<i32>(), ComponentValue::I32(b))),
            ComponentType::I64 => buf
                .get(..size_of::<i64>())
                .and_then(|u| u.try_into().ok())
                .map(i64::from_le_bytes)
                .map(|b| (size_of::<i64>(), ComponentValue::I64(b))),

            ComponentType::Bool => buf
                .get(..1)
                .map(|b| b[0] == 1)
                .map(|b| (1, ComponentValue::Bool(b))),
            ComponentType::F32 => buf
                .get(..size_of::<f32>())
                .and_then(|u| u.try_into().ok())
                .map(f32::from_le_bytes)
                .map(|b| (size_of::<f32>(), ComponentValue::F32(b))),
            ComponentType::F64 => buf
                .get(..size_of::<f64>())
                .and_then(|u| u.try_into().ok())
                .map(f64::from_le_bytes)
                .map(|b| (size_of::<f64>(), ComponentValue::F64(b))),
            ComponentType::String => {
                let orig_len = buf.len();
                let len_buf = &mut buf;
                let len = len_buf.pop_varint()?;
                let total_len = orig_len - len_buf.len() + len;
                let str = std::str::from_utf8(buf.get(..len)?).ok()?;
                Some((total_len, ComponentValue::String(Cow::from(str))))
            }
            ComponentType::Bytes => {
                let orig_len = buf.len();
                let len_buf = &mut buf;
                let len = len_buf.pop_varint()?;
                let total_len = orig_len - len_buf.len() + len;
                let bytes = len_buf.get(..len)?;
                Some((total_len, ComponentValue::Bytes(Cow::from(bytes))))
            }
            ComponentType::Vector3F32 => {
                if buf.len() < size_of::<f32>() * 3 {
                    return None;
                }
                let vec = Vector3::from_iterator(
                    buf.chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                );
                Some((size_of::<f32>() * 3, ComponentValue::Vector3F32(vec)))
            }
            ComponentType::Vector3F64 => {
                if buf.len() < size_of::<f64>() * 3 {
                    return None;
                }
                let vec = Vector3::from_iterator(
                    buf.chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                );
                Some((size_of::<f64>() * 3, ComponentValue::Vector3F64(vec)))
            }
            ComponentType::Matrix3x3F32 => {
                if buf.len() < size_of::<f32>() * 9 {
                    return None;
                }
                let vec = Matrix3::from_iterator(
                    buf.chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                );
                Some((size_of::<f32>() * 9, ComponentValue::Matrix3x3F32(vec)))
            }
            ComponentType::Matrix3x3F64 => {
                if buf.len() < size_of::<f64>() * 9 {
                    return None;
                }
                let vec = Matrix3::from_iterator(
                    buf.chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                );
                Some((size_of::<f64>() * 9, ComponentValue::Matrix3x3F64(vec)))
            }
            ComponentType::QuaternionF32 => {
                if buf.len() < size_of::<f32>() * 4 {
                    return None;
                }
                let vec = Quaternion::from_vector(Vector4::from_iterator(
                    buf.chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                ));
                Some((size_of::<f32>() * 4, ComponentValue::QuaternionF32(vec)))
            }
            ComponentType::QuaternionF64 => {
                if buf.len() < size_of::<f64>() * 4 {
                    return None;
                }
                let vec = Quaternion::from_vector(Vector4::from_iterator(
                    buf.chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                ));
                Some((size_of::<f64>() * 4, ComponentValue::QuaternionF64(vec)))
            }
            ComponentType::SpatialPosF32 => {
                if buf.len() < size_of::<f32>() * 7 {
                    return None;
                }
                let quat = Quaternion::from_vector(Vector4::from_iterator(
                    buf.chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                ));
                let vec = Vector3::from_iterator(
                    buf[{ 4 * size_of::<f32>() }..]
                        .chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                );
                Some((
                    size_of::<f32>() * 7,
                    ComponentValue::SpatialPosF32((quat, vec)),
                ))
            }
            ComponentType::SpatialPosF64 => {
                if buf.len() < size_of::<f64>() * 7 {
                    return None;
                }
                let quat = Quaternion::from_vector(Vector4::from_iterator(
                    buf.chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                ));
                let vec = Vector3::from_iterator(
                    buf[{ 4 * size_of::<f64>() }..]
                        .chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                );
                Some((
                    size_of::<f64>() * 7,
                    ComponentValue::SpatialPosF64((quat, vec)),
                ))
            }
            ComponentType::SpatialMotionF32 => {
                if buf.len() < size_of::<f32>() * 6 {
                    return None;
                }
                let pos = Vector3::from_iterator(
                    buf.chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                );
                let dot = Vector3::from_iterator(
                    buf[{ 3 * size_of::<f32>() }..]
                        .chunks_exact(size_of::<f32>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f32::from_le_bytes),
                );
                Some((
                    size_of::<f32>() * 6,
                    ComponentValue::SpatialMotionF32((pos, dot)),
                ))
            }
            ComponentType::SpatialMotionF64 => {
                if buf.len() < size_of::<f64>() * 6 {
                    return None;
                }
                let pos = Vector3::from_iterator(
                    buf.chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                );
                let dot = Vector3::from_iterator(
                    buf[{ 3 * size_of::<f64>() }..]
                        .chunks_exact(size_of::<f64>())
                        .filter_map(|u| u.try_into().ok())
                        .map(f64::from_le_bytes),
                );
                Some((
                    size_of::<f64>() * 7,
                    ComponentValue::SpatialMotionF64((pos, dot)),
                ))
            }
            ComponentType::Filter => {
                let id = buf
                    .get(..size_of::<u64>())
                    .and_then(|u| u.try_into().ok())
                    .map(u64::from_le_bytes)?;
                let mask_len = buf
                    .get(..size_of::<u8>())
                    .and_then(|u| u.try_into().ok())
                    .map(u8::from_le_bytes)?;
                let filter = ComponentFilter { id, mask_len };
                Some((size_of::<u64>() * 2, ComponentValue::Filter(filter)))
            }
        }
    }
}

pub trait SliceExt {
    fn get_u8(&self) -> Option<u8>;
    fn get_u32(&self) -> Option<u32>;
    fn get_u64(&self) -> Option<u64>;

    fn pop_u8(&mut self) -> Option<u8>;
    fn pop_u32(&mut self) -> Option<u32>;
    fn pop_u64(&mut self) -> Option<u64>;
    fn pop_varint(&mut self) -> Option<usize>;
}

impl<'a> SliceExt for &'a [u8] {
    fn get_u8(&self) -> Option<u8> {
        self.first().copied()
    }

    fn get_u32(&self) -> Option<u32> {
        self.get(..size_of::<u32>())
            .and_then(|u| u.try_into().ok())
            .map(u32::from_le_bytes)
    }

    fn get_u64(&self) -> Option<u64> {
        self.get(..size_of::<u64>())
            .and_then(|u| u.try_into().ok())
            .map(u64::from_le_bytes)
    }

    fn pop_u8(&mut self) -> Option<u8> {
        let v = self.get_u8()?;
        *self = &self[1..];
        Some(v)
    }

    fn pop_u32(&mut self) -> Option<u32> {
        let v = self.get_u32()?;
        *self = &self[4..];
        Some(v)
    }

    fn pop_u64(&mut self) -> Option<u64> {
        let v = self.get_u64()?;
        *self = &self[8..];
        Some(v)
    }

    fn pop_varint(&mut self) -> Option<usize> {
        // source: https://github.com/jamesmunns/postcard/blob/a095b49935f7bd1bab04e6c5914ab11c3bbd5fee/src/de/deserializer.rs#L110
        let mut out = 0;
        for i in 0..varint_max::<usize>() {
            let val = self.pop_u8()?;
            let carry = (val & 0x7F) as usize;
            out |= carry << (7 * i);

            if (val & 0x80) == 0 {
                if i == varint_max::<usize>() - 1 && val > max_of_last_byte::<usize>() {
                    return None;
                } else {
                    return Some(out);
                }
            }
        }
        None
    }
}

// source: https://github.com/jamesmunns/postcard/blob/a095b49935f7bd1bab04e6c5914ab11c3bbd5fee/src/varint.rs#L1C1-L24C1
/// Returns the maximum number of bytes required to encode T.
pub const fn varint_max<T: Sized>() -> usize {
    const BITS_PER_BYTE: usize = 8;
    const BITS_PER_VARINT_BYTE: usize = 7;

    // How many data bits do we need for this type?
    let bits = core::mem::size_of::<T>() * BITS_PER_BYTE;

    // We add (BITS_PER_VARINT_BYTE - 1), to ensure any integer divisions
    // with a remainder will always add exactly one full byte, but
    // an evenly divided number of bits will be the same
    let roundup_bits = bits + (BITS_PER_VARINT_BYTE - 1);

    // Apply division, using normal "round down" integer division
    roundup_bits / BITS_PER_VARINT_BYTE
}

/// Returns the maximum value stored in the last encoded byte.
const fn max_of_last_byte<T: Sized>() -> u8 {
    let max_bits = core::mem::size_of::<T>() * 8;
    let extra_bits = max_bits % 7;
    (1 << extra_bits) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_f64() {
        let mut buf = vec![];
        buf.extend_from_slice(&1702059810518727000u64.to_le_bytes()); // Time
        buf.extend_from_slice(&1234u64.to_le_bytes()); // Component Id
        buf.extend_from_slice(&[ComponentType::F64.into()]); // Component Type
        buf.extend_from_slice(&[1]); // Entity Length
        buf.extend_from_slice(&1337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&101.05f64.to_le_bytes()); // Component
        let mut parser = Parser::new(buf).unwrap();

        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();
        assert_eq!(entity_id, EntityId(1337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::F64(101.05));
    }

    #[test]
    fn test_two_f64() {
        let mut buf = vec![];
        buf.extend_from_slice(&1702059810518727000u64.to_le_bytes()); // Time
        buf.extend_from_slice(&1234u64.to_le_bytes()); // Component Id
        buf.extend_from_slice(&[ComponentType::F64.into()]); // Component Type
        buf.extend_from_slice(&[2]); // Entity Length
        buf.extend_from_slice(&1337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&2337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&101.05f64.to_le_bytes()); // Component
        buf.extend_from_slice(&1.05f64.to_le_bytes()); // Component
        let mut parser = Parser::new(buf).unwrap();
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(entity_id, EntityId(1337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::F64(101.05));
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(entity_id, EntityId(2337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::F64(1.05));
    }

    #[test]
    fn test_two_u32() {
        let mut buf = vec![];
        buf.extend_from_slice(&1702059810518727000u64.to_le_bytes()); // Time
        buf.extend_from_slice(&1234u64.to_le_bytes()); // Component Id
        buf.extend_from_slice(&[ComponentType::U32.into()]); // Component Type
        buf.extend_from_slice(&[2]); // Entity Length
        buf.extend_from_slice(&1337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&2337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&154u32.to_le_bytes()); // Component
        buf.extend_from_slice(&254u32.to_le_bytes()); // Component
        let mut parser = Parser::new(buf).unwrap();
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(entity_id, EntityId(1337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::U32(154));
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(entity_id, EntityId(2337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::U32(254));
    }

    #[test]
    fn test_two_components() {
        let mut buf = vec![];
        buf.extend_from_slice(&1702059810518727000u64.to_le_bytes()); // Time
        buf.extend_from_slice(&1234u64.to_le_bytes()); // Component Id
        buf.extend_from_slice(&[ComponentType::U32.into()]); // Component Type
        buf.extend_from_slice(&[2]); // Entity Length
        buf.extend_from_slice(&1337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&2337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&154u32.to_le_bytes()); // Component
        buf.extend_from_slice(&254u32.to_le_bytes()); // Component

        buf.extend_from_slice(&2234u64.to_le_bytes()); // Component Id
        buf.extend_from_slice(&[ComponentType::U8.into()]); // Component Type
        buf.extend_from_slice(&[2]); // Entity Length
        buf.extend_from_slice(&1337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&2337u64.to_le_bytes()); // Entity Id
        buf.extend_from_slice(&154u8.to_le_bytes()); // Component
        buf.extend_from_slice(&254u8.to_le_bytes()); // Component

        let mut parser = Parser::new(buf).unwrap();
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();
        assert_eq!(entity_id, EntityId(1337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::U32(154));
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(entity_id, EntityId(2337));
        assert_eq!(component_id, ComponentId(1234));
        assert_eq!(value, ComponentValue::U32(254));
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(component_id, ComponentId(2234));
        assert_eq!(entity_id, EntityId(1337));
        assert_eq!(value, ComponentValue::U8(154));
        let ComponentPair {
            component_id,
            entity_id,
            value,
        } = parser.next().unwrap();

        assert_eq!(component_id, ComponentId(2234));
        assert_eq!(entity_id, EntityId(2337));
        assert_eq!(value, ComponentValue::U8(254));
    }

    #[test]
    fn test_pop_u8() {
        let mut buf = &[1, 2, 3, 4, 5, 6][..];
        let buf = &mut buf;
        assert_eq!(buf.pop_u8(), Some(1));
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_pop_varint() {
        let mut buf = &[128, 4][..];
        let buf = &mut buf;
        assert_eq!(buf.pop_varint(), Some(512));
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_pop_u64() {
        let mut buf = &[1, 0, 0, 0, 0, 0, 0, 0][..];
        let buf = &mut buf;
        assert_eq!(buf.pop_u64(), Some(1));
        assert_eq!(buf.len(), 0);
    }
}
