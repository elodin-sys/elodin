use crate::{
    ColumnPayload, ComponentType, ComponentValue, ControlMsg, EntityId, Error, Packet, Payload,
    PrimitiveTy, StreamId,
};
use bytemuck::CheckedBitPattern;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::mem::size_of;
use try_buf::TryBuf;

impl<B: Buf> Packet<B> {
    /// Parse a raw packet from a buffer, just parsing the stream id, and leaving the payload
    /// unparsed.
    pub fn parse_raw(mut buf: B) -> Result<Self, Error> {
        let stream_id = StreamId(buf.try_get_u32()?);
        Ok(Packet {
            stream_id,
            payload: buf,
        })
    }
}

impl<B: Buf + Slice> Packet<Payload<B>> {
    /// Parse a packet from a buffer
    pub fn parse(buf: B) -> Result<Self, Error> {
        let Packet { stream_id, payload } = Packet::parse_raw(buf)?;

        let payload = if stream_id == StreamId::CONTROL {
            Payload::ControlMsg(ControlMsg::parse(payload)?)
        } else {
            Payload::Column(ColumnPayload::parse(payload)?)
        };

        Ok(Packet { stream_id, payload })
    }

    pub fn write(&self, mut buf: impl BufMut) -> Result<(), Error> {
        buf.put_u32(self.stream_id.0);
        match &self.payload {
            Payload::ControlMsg(msg) => msg.write(buf)?,
            Payload::Column(column) => column.write(buf),
        };
        Ok(())
    }
}

impl<B: Buf + Slice> ColumnPayload<B> {
    /// Parse a column payload from a buffer
    pub fn parse(mut buf: B) -> Result<Self, Error> {
        let time = buf.try_get_u64()?;
        let len = buf.try_get_u32()?;
        let entity_len = len as usize * size_of::<u64>();
        let entity_buf = buf.try_get_slice(entity_len).ok_or(Error::EOF)?;
        let value_buf = buf;
        Ok(Self {
            time,
            len,
            entity_buf,
            value_buf,
        })
    }

    pub fn as_ref(&self) -> ColumnPayload<&[u8]> {
        ColumnPayload {
            time: self.time,
            len: self.len,
            entity_buf: self.entity_buf.chunk(),
            value_buf: self.value_buf.chunk(),
        }
    }

    pub fn write(&self, mut buf: impl BufMut) {
        buf.put_u64(self.time);
        buf.put_u32(self.len);
        buf.put_slice(self.entity_buf.chunk());
        buf.put_slice(self.value_buf.chunk());
    }
}

impl ColumnPayload<Bytes> {
    pub fn try_from_value_iter<'a>(
        time: u64,
        iter: impl Iterator<Item = ColumnValue<'a>> + 'a,
    ) -> Result<Self, Error> {
        let mut entity_buf = BytesMut::default();
        let mut value_buf = BytesMut::default();
        let mut len = 0;
        for ColumnValue { entity_id, value } in iter {
            entity_buf.put_u64_le(entity_id.0);
            value_buf.put(value.bytes().ok_or(Error::InvalidAlignment)?);
            len += 1;
        }
        Ok(ColumnPayload {
            time,
            len,
            entity_buf: entity_buf.freeze(),
            value_buf: value_buf.freeze(),
        })
    }
}

impl<'a> ColumnPayload<&'a [u8]> {
    pub fn into_iter(
        self,
        component_type: ComponentType,
    ) -> impl Iterator<Item = Result<ColumnValue<'a>, Error>> + 'a {
        let mut entity_buf = self.entity_buf;
        let mut value_buf = self.value_buf;
        (0..self.len).map(move |_| {
            let entity_id = EntityId(entity_buf.try_get_u64_le()?);
            let (size, value) = component_type.parse_value(value_buf)?;
            value_buf.advance(size);
            Ok(ColumnValue { entity_id, value })
        })
    }
}

#[derive(Debug)]
pub struct ColumnValue<'a> {
    pub entity_id: EntityId,
    pub value: ComponentValue<'a>,
}

impl ControlMsg {
    pub fn parse(buf: impl Buf) -> Result<Self, Error> {
        postcard::from_bytes(buf.chunk()).map_err(Error::from)
    }

    pub fn write(&self, buf: impl BufMut) -> Result<(), Error> {
        let writer = buf.writer();
        postcard::to_io(self, writer)?;
        Ok(())
    }
}

pub trait Slice {
    fn try_get_slice(&mut self, len: usize) -> Option<Self>
    where
        Self: Sized;
}

impl<'a> Slice for &'a [u8] {
    fn try_get_slice(&mut self, len: usize) -> Option<Self> {
        let out = self.get(..len)?;
        *self = &self[len..];
        Some(out)
    }
}

impl Slice for Bytes {
    fn try_get_slice(&mut self, len: usize) -> Option<Self>
    where
        Self: Sized,
    {
        (self.remaining() > len).then(|| self.split_to(len))
    }
}

impl ComponentType {
    pub fn parse(mut buf: impl Buf) -> Result<ComponentType, Error> {
        let prim_type = buf.try_get_u8()?;
        let primitive_ty =
            PrimitiveTy::try_from(prim_type).map_err(|_| Error::UnknownPrimitiveTy)?;
        let dim = buf.try_get_u8()?;
        let shape = (0..dim)
            .map(|_| buf.try_get_i64())
            .collect::<Result<_, try_buf::ErrorKind>>()?;
        Ok(ComponentType {
            primitive_ty,
            shape,
        })
    }

    pub fn parse_value<'a>(&self, buf: &'a [u8]) -> Result<(usize, ComponentValue<'a>), Error> {
        let size = self.size();
        let buf = buf.get(..size).ok_or(Error::EOF)?;
        let comp_shape = self.shape.iter().map(|n| *n as _).collect::<Vec<_>>();
        let shape = ndarray::IxDyn(&comp_shape);
        fn cow_array<T: CheckedBitPattern>(
            buf: &[u8],
            shape: ndarray::IxDyn,
        ) -> Result<ndarray::CowArray<'_, T, ndarray::IxDyn>, Error> {
            if let Ok(buf) = bytemuck::checked::try_cast_slice(buf) {
                ndarray::ArrayView::from_shape(shape, buf)
                    .map_err(Error::from)
                    .map(ndarray::CowArray::from)
            } else {
                let array: ndarray::Array<T, ndarray::Ix1> = buf
                    .chunks_exact(size_of::<T>())
                    .map(|chunk| {
                        bytemuck::checked::try_pod_read_unaligned::<T>(chunk)
                            .map_err(|_| Error::CheckedCast)
                    })
                    .collect::<Result<_, Error>>()?;
                let array = array.into_dyn();
                let array = array.into_shape(shape)?;

                Ok(ndarray::CowArray::from(array))
            }
        }
        let value = match self.primitive_ty {
            PrimitiveTy::U8 => ndarray::ArrayView::from_shape(shape, buf)
                .map_err(Error::from)
                .map(ndarray::CowArray::from)
                .map(ComponentValue::U8),
            PrimitiveTy::U16 => cow_array(buf, shape).map(ComponentValue::U16),
            PrimitiveTy::U32 => cow_array(buf, shape).map(ComponentValue::U32),
            PrimitiveTy::U64 => cow_array(buf, shape).map(ComponentValue::U64),
            PrimitiveTy::I8 => cow_array(buf, shape).map(ComponentValue::I8),
            PrimitiveTy::I16 => cow_array(buf, shape).map(ComponentValue::I16),
            PrimitiveTy::I32 => cow_array(buf, shape).map(ComponentValue::I32),
            PrimitiveTy::I64 => cow_array(buf, shape).map(ComponentValue::I64),
            PrimitiveTy::Bool => cow_array(buf, shape).map(ComponentValue::Bool),
            PrimitiveTy::F32 => cow_array(buf, shape).map(ComponentValue::F32),
            PrimitiveTy::F64 => cow_array(buf, shape).map(ComponentValue::F64),
        }?;
        Ok((size, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_control() {
        let packet: Packet<Payload<Bytes>> = Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::Exit),
        };
        let mut buf = BytesMut::default();
        packet.write(&mut buf).unwrap();
        let packet2 = Packet::parse(buf.freeze()).unwrap();
        assert_eq!(packet, packet2);
    }

    #[test]
    fn test_serialize_deserialize_col() {
        let packet: Packet<Payload<Bytes>> = Packet {
            stream_id: StreamId(224),
            payload: Payload::Column(
                ColumnPayload::try_from_value_iter(
                    0,
                    [
                        ColumnValue {
                            entity_id: EntityId(0),
                            value: ComponentValue::U32(ndarray::arr1(&[1, 2, 3]).into_dyn().into()),
                        },
                        ColumnValue {
                            entity_id: EntityId(1),
                            value: ComponentValue::U32(ndarray::arr1(&[1, 0, 1]).into_dyn().into()),
                        },
                    ]
                    .into_iter(),
                )
                .unwrap(),
            ),
        };
        let mut buf = BytesMut::default();
        packet.write(&mut buf).unwrap();
        let packet2 = Packet::parse(buf.freeze()).unwrap();
        assert_eq!(packet, packet2);
    }
}
