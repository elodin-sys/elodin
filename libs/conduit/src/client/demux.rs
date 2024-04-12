use crate::{
    ser_de::{ColumnValue, Slice},
    ColumnPayload, ControlMsg, Error, Metadata, Packet, Payload, StreamId,
};
use alloc::sync::Arc;
use bytes::{Buf, Bytes};
use hashbrown::HashMap;

#[derive(Clone, Default)]
pub struct Demux {
    streams: HashMap<StreamId, Arc<Metadata>>,
}

impl Demux {
    pub fn handle<B: Buf + Slice>(&mut self, packet: Packet<Payload<B>>) -> Result<Msg<B>, Error> {
        match packet.payload {
            Payload::ControlMsg(ControlMsg::Metadata {
                stream_id,
                metadata,
            }) => {
                self.streams.insert(stream_id, Arc::new(metadata.clone()));
                Ok(Msg::Control(ControlMsg::Metadata {
                    stream_id,
                    metadata,
                }))
            }
            Payload::ControlMsg(m) => Ok(Msg::Control(m)),
            Payload::Column(payload) => {
                let metadata = self
                    .streams
                    .get(&packet.stream_id)
                    .ok_or(Error::StreamNotFound(packet.stream_id))?;
                Ok(Msg::Column(ColumnMsg {
                    metadata: metadata.clone(),
                    payload,
                }))
            }
        }
    }
}

#[derive(Debug)]
pub enum Msg<B = Bytes> {
    Control(ControlMsg),
    Column(ColumnMsg<B>),
}

#[derive(Debug)]
pub struct ColumnMsg<B> {
    pub metadata: Arc<Metadata>,
    pub payload: ColumnPayload<B>,
}

impl ColumnMsg<Bytes> {
    pub fn iter(&self) -> impl Iterator<Item = Result<ColumnValue<'_>, Error>> + '_ {
        self.payload
            .as_ref()
            .into_iter(self.metadata.component_type.clone())
    }
}
