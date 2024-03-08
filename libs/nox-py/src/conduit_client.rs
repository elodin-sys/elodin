use bytes::Bytes;
use conduit::{client::AsyncClient, ColumnPayload, Metadata, Packet, Payload, StreamId};
use pyo3::prelude::*;
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;

use crate::{Archetype, ComponentData, EntityId, Error, PyUntypedArrayExt};

#[pyclass]
pub struct Conduit {
    inner: Arc<Mutex<ConduitInner>>,
    rt: tokio::runtime::Runtime,
}

pub struct ConduitInner {
    addr: SocketAddr,
    client: Option<conduit::client::TcpClient>,
}

#[pymethods]
impl Conduit {
    #[staticmethod]
    pub fn tcp(addr: String) -> Result<Self, Error> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let addr: SocketAddr = addr.parse().unwrap();
        let inner = Arc::new(Mutex::new(ConduitInner::new(addr)));
        Ok(Conduit { inner, rt })
    }

    pub fn send(
        &self,
        entity_id: EntityId,
        archetype: Archetype<'_>,
        time: Option<u64>,
    ) -> Result<(), Error> {
        let time = time.unwrap_or_default();
        let Archetype {
            component_datas,
            arrays,
            ..
        } = archetype;
        let inner = self.inner.clone();
        let arrays = arrays
            .iter()
            .zip(component_datas.iter())
            .map(|(arr, data)| {
                let ty: conduit::ComponentType = data.ty.clone().into();
                let elem_size = ty.primitive_ty.element_type().element_size_in_bytes();
                unsafe { Bytes::copy_from_slice(arr.buf(elem_size)) }
            })
            .collect();
        self.rt.block_on(async move {
            let mut inner = inner.lock().await;
            inner.send(entity_id, component_datas, arrays, time).await
        })
    }
}

impl ConduitInner {
    fn new(addr: SocketAddr) -> Self {
        Self { addr, client: None }
    }

    async fn client(&mut self) -> Result<&mut conduit::client::TcpClient, Error> {
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
        component_datas: Vec<ComponentData>,
        arrays: Vec<Bytes>,
        time: u64,
    ) -> Result<(), Error> {
        let client = self.client().await?;
        for _ in 0..2 {
            match send_inner(client, entity_id.clone(), &component_datas, &arrays, time).await {
                Ok(_) => break,
                Err(conduit::Error::Io(_)) => {
                    continue;
                }
                Err(err) => return Err(err.into()),
            }
        }
        Ok(())
    }
}

async fn send_inner(
    client: &mut conduit::client::TcpClient,
    entity_id: EntityId,
    component_datas: &[ComponentData],
    arrays: &[Bytes],
    time: u64,
) -> Result<(), conduit::Error> {
    let stream_id = StreamId::rand();
    for (data, value_buf) in component_datas.iter().zip(arrays.iter()) {
        let ty: conduit::ComponentType = data.ty.clone().into();
        let packet: Packet<Payload<Bytes>> = Packet::metadata(
            stream_id,
            Metadata {
                component_id: data.id.inner,
                component_type: ty.clone(),
                tags: HashMap::new(),
            },
        );
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
