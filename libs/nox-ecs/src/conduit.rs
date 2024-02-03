use crate::{assets::Handle, Error, Exec};
use bytes::{Bytes, BytesMut};
use elodin_conduit::{
    parser::{ComponentPair, Parser},
    ComponentId, ComponentType,
};
use nox::{Client, FromBuilder, IntoOp};
use std::{collections::BTreeMap, future::Future};

pub struct ConduitExec<Tx, Rx> {
    subscriptions: BTreeMap<ComponentId, Subscription<Tx>>,
    rx: Rx,
    exec: Exec,
}

struct Subscription<Tx> {
    tx: Tx,
    sent_generation: usize,
}

impl<Tx: ClientTx, Rx> ConduitExec<Tx, Rx> {
    pub async fn send(&mut self) -> Result<(), Error> {
        for (comp_id, sub) in &mut self.subscriptions {
            let _ = self.exec.column(*comp_id)?;
            let col = self.exec.cached_column(*comp_id)?;
            if col.column.buffer.asset {
                let Some(buf) = col.column.buffer.typed_buf::<u64>() else {
                    // TODO: warn
                    continue;
                };
                let mut out = vec![];
                let mut changed = false;
                for id in buf.iter() {
                    let gen = self
                        .exec
                        .host_world
                        .assets
                        .gen(Handle::<()>::new(*id))
                        .ok_or(Error::AssetNotFound)?;
                    if gen > sub.sent_generation {
                        changed = true;
                        sub.sent_generation = gen;
                    }
                }
                if !changed {
                    continue;
                }
                for id in buf.iter() {
                    let Some(value) = self.exec.host_world.assets.value(Handle::<()>::new(*id))
                    else {
                        todo!("gracefully handle")
                    };
                    value.with_bytes(|bytes| out.extend_from_slice(bytes));
                }
                sub.tx
                    .send(
                        *comp_id,
                        col.column.buffer.component_type,
                        col.column.buffer.len,
                        &col.entities.buf,
                        &out,
                    )
                    .await?;
            } else {
                sub.tx
                    .send(
                        *comp_id,
                        col.column.buffer.component_type,
                        col.column.buffer.len,
                        &col.entities.buf,
                        &col.column.buffer.buf,
                    )
                    .await?;
            }
        }
        Ok(())
    }
}

impl<Tx, Rx: ClientRx> ConduitExec<Tx, Rx> {
    pub async fn recv(&mut self) -> Result<(), Error> {
        let Some(parser) = self.rx.recv().await? else {
            return Ok(());
        };
        for ComponentPair {
            component_id,
            entity_id,
            value,
        } in parser
        {
            let mut col = self.exec.column_mut(component_id)?;
            let Some(out) = col.entity_buf(entity_id) else {
                continue;
            };
            value.with_bytes(|bytes| {
                if bytes.len() != out.len() {
                    return Err(Error::ValueSizeMismatch);
                }
                out.copy_from_slice(bytes);
                Ok(())
            })?;
        }
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        self.exec.clear_cache();
    }
}

impl<Tx: ClientTx, Rx: ClientRx> ConduitExec<Tx, Rx> {
    pub async fn run(&mut self, client: &Client) -> Result<(), Error> {
        self.exec.run(client)?;
        self.send().await?;
        self.recv().await?;
        Ok(())
    }
}

pub trait ClientTx {
    fn send(
        &mut self,
        component_id: ComponentId,
        component_ty: ComponentType,
        len: usize,
        entities: &[u8],
        values: &[u8],
    ) -> impl Future<Output = Result<(), Error>>;
}

pub trait ClientRx {
    fn recv(&mut self) -> impl Future<Output = Result<Option<Parser<Bytes>>, Error>>;
}

#[cfg(feature = "tokio")]
impl<T> ClientTx for elodin_conduit::tokio::Client<T>
where
    T: futures::Sink<Bytes, Error = std::io::Error> + Unpin,
{
    async fn send(
        &mut self,
        component_id: ComponentId,
        component_ty: ComponentType,
        len: usize,
        entities: &[u8],
        values: &[u8],
    ) -> Result<(), Error> {
        use elodin_conduit::builder::{Builder, ComponentBuilder};
        let capacity = 26 + entities.len() + values.len();
        let mut builder = Builder::new(bytes::BytesMut::with_capacity(capacity), 0).unwrap();
        builder.append_builder(ComponentBuilder::new(
            (component_id, component_ty),
            (len, entities),
            values,
        ))?;
        self.send_builder(builder).await.map_err(Error::from)
    }
}

#[cfg(feature = "tokio")]
impl<T> ClientRx for elodin_conduit::tokio::Client<T>
where
    T: futures::Stream<Item = Result<BytesMut, std::io::Error>> + Unpin,
{
    async fn recv(&mut self) -> Result<Option<Parser<Bytes>>, Error> {
        self.recv_parser().await.map_err(Error::from)
    }
}

pub struct WorldPos(nox::SpatialTransform<f64>);
impl FromBuilder for WorldPos {
    type Item<'a> = Self;

    fn from_builder(builder: &nox::Builder) -> Self::Item<'_> {
        WorldPos(nox::SpatialTransform::from_builder(builder))
    }
}

impl IntoOp for WorldPos {
    fn into_op(self) -> nox::Noxpr {
        self.0.into_op()
    }
}

impl crate::Component for WorldPos {
    type Inner = nox::SpatialTransform<f64>;

    type HostTy = Self;

    fn host(val: Self::HostTy) -> Self {
        val
    }

    fn component_id() -> ComponentId {
        ComponentId::new("world_pos")
    }

    fn component_type() -> ComponentType {
        ComponentType::SpatialPosF64
    }
}
