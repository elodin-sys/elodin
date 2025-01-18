use bytes::Bytes;
use impeller2::{client::AsyncClient, ColumnPayload, Packet, Payload, StreamId};
use pyo3::prelude::*;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;

use crate::{Archetype, EntityId, Error, Metadata, PyUntypedArrayExt};

#[pyclass]
pub struct Impeller2 {
    inner: Arc<Mutex<Impeller2Inner>>,
    rt: tokio::runtime::Runtime,
}

pub struct Impeller2Inner {
    addr: SocketAddr,
    client: Option<impeller2::client::TcpClient>,
}

#[pymethods]
impl Impeller2 {
    #[staticmethod]
    pub fn tcp(addr: String) -> Result<Self, Error> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let addr: SocketAddr = addr.parse().unwrap();
        let inner = Arc::new(Mutex::new(Impeller2Inner::new(addr)));
        Ok(Impeller2 { inner, rt })
    }

    pub fn send(
        &self,
        entity_id: EntityId,
        archetype: Archetype<'_>,
        time: Option<u64>,
    ) -> Result<(), Error> {
        let time = time.unwrap_or_default();
        let Archetype {
            component_data,
            arrays,
            ..
        } = archetype;
        let inner = self.inner.clone();
        let arrays = arrays
            .iter()
            .zip(component_data.iter())
            .map(|(arr, data)| {
                let ty: impeller2::ComponentType = data.component_type.clone();
                let elem_size = ty.primitive_ty.element_type().element_size_in_bytes();
                unsafe { Bytes::copy_from_slice(arr.buf(elem_size)) }
            })
            .collect();
        self.rt.block_on(async move {
            let mut inner = inner.lock().await;
            inner.send(entity_id, component_data, arrays, time).await
        })
    }
}

impl Impeller2Inner {
    fn new(addr: SocketAddr) -> Self {
        Self { addr, client: None }
    }

    async fn client(&mut self) -> Result<&mut impeller2::client::TcpClient, Error> {
        if let Some(ref mut client) = self.client {
            Ok(client)
        } else {
            let stream = tokio::net::TcpStream::connect(self.addr).await?;
            let client = AsyncClient::from_stream(stream);
            Ok(self.client.insert(client))
        }
    }

    async fn send(
        &mut self,
        entity_id: EntityId,
        component_data: Vec<Metadata>,
        arrays: Vec<Bytes>,
        time: u64,
    ) -> Result<(), Error> {
        let client = self.client().await?;
        for _ in 0..2 {
            match send_inner(client, entity_id, &component_data, &arrays, time).await {
                Ok(_) => break,
                Err(impeller2::Error::Io(_)) => {
                    continue;
                }
                Err(err) => return Err(err.into()),
            }
        }
        Ok(())
    }
}

async fn send_inner(
    client: &mut impeller2::client::TcpClient,
    entity_id: EntityId,
    component_data: &[Metadata],
    arrays: &[Bytes],
    time: u64,
) -> Result<(), impeller2::Error> {
    let stream_id = StreamId::rand();
    for (data, value_buf) in component_data.iter().zip(arrays.iter()) {
        let packet: Packet<Payload<Bytes>> = Packet::start_stream(stream_id, data.inner.clone());
        client.send(packet).await?;

        client
            .send(Packet {
                stream_id,
                payload: Payload::Column(ColumnPayload {
                    time,
                    len: 1,
                    entity_buf: Bytes::copy_from_slice(&entity_id.inner.0.to_le_bytes()),
                    value_buf: value_buf.clone(),
                }),
            })
            .await?;
    }
    Ok(())
}
