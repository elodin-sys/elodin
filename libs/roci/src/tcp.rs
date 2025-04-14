use impeller2::buf::{Slice, deref};
use impeller2::types::{Msg, PacketId};
use impeller2_stellar::{Client, Error, SubStream};
use impeller2_wkt::{MsgMetadata, SetMsgMetadata, StreamReply, VTableMsg, VTableStream};
use std::future::Future;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use stellarator::io::AsyncWrite;
use zerocopy::{Immutable, KnownLayout};

use crate::{AsVTable, Metadatatize};

pub trait SinkExt {
    fn send_vtable<V: AsVTable>(&self, id: PacketId) -> impl Future<Output = Result<(), Error>>;
    fn send_metadata<V: Metadatatize>(&self) -> impl Future<Output = Result<(), Error>>;
    fn init_world<V: AsVTable + Metadatatize>(
        &self,
        vtable_id: PacketId,
    ) -> impl Future<Output = Result<(), Error>>;
    fn init_msg<M: postcard_schema::Schema + Msg>(&self)
    -> impl Future<Output = Result<(), Error>>;
}

impl<W: AsyncWrite> SinkExt for impeller2_stellar::PacketSink<W> {
    async fn send_vtable<V: AsVTable>(&self, id: PacketId) -> Result<(), Error> {
        let vtable = V::as_vtable();
        self.send(&(VTableMsg { id, vtable })).await.0?;
        Ok(())
    }

    async fn send_metadata<V: Metadatatize>(&self) -> Result<(), Error> {
        for metadata in V::metadata() {
            self.send(&impeller2_wkt::SetComponentMetadata(metadata))
                .await
                .0?;
        }
        Ok(())
    }

    async fn init_world<V: AsVTable + Metadatatize>(
        &self,
        vtable_id: PacketId,
    ) -> Result<(), Error> {
        self.send_vtable::<V>(vtable_id).await?;
        self.send_metadata::<V>().await?;
        Ok(())
    }

    async fn init_msg<M: postcard_schema::Schema + Msg>(&self) -> Result<(), Error> {
        let schema = M::SCHEMA;
        let name = std::any::type_name::<M>();
        let metadata = MsgMetadata {
            name: name.to_string(),
            schema: schema.into(),
            metadata: Default::default(),
        };
        self.send(&SetMsgMetadata {
            id: M::ID,
            metadata,
        })
        .await
        .0?;
        Ok(())
    }
}

impl SinkExt for Client {
    fn send_vtable<V: AsVTable>(&self, id: PacketId) -> impl Future<Output = Result<(), Error>> {
        self.tx.send_vtable::<V>(id)
    }

    fn send_metadata<V: Metadatatize>(&self) -> impl Future<Output = Result<(), Error>> {
        self.tx.send_metadata::<V>()
    }

    fn init_world<V: AsVTable + Metadatatize>(
        &self,
        vtable_id: PacketId,
    ) -> impl Future<Output = Result<(), Error>> {
        self.tx.init_world::<V>(vtable_id)
    }

    fn init_msg<M: postcard_schema::Schema + Msg>(
        &self,
    ) -> impl Future<Output = Result<(), Error>> {
        self.tx.init_msg::<M>()
    }
}

pub trait StreamExt {
    fn subscribe<T>(&mut self) -> impl Future<Output = Result<Subscription<'_, T>, Error>>
    where
        T: AsVTable + zerocopy::TryFromBytes + Immutable + KnownLayout + Clone;
}

impl StreamExt for Client {
    async fn subscribe<T: AsVTable + zerocopy::TryFromBytes + Immutable + KnownLayout + Clone>(
        &mut self,
    ) -> Result<Subscription<'_, T>, Error> {
        let vtable = T::as_vtable();
        let msg = VTableStream {
            id: fastrand::u16(..).to_le_bytes(),
            vtable,
        };
        let sub = self.stream(&msg).await?;
        Ok(Subscription {
            sub,
            _phantom: PhantomData,
        })
    }
}

pub struct Subscription<'a, T: AsVTable + zerocopy::TryFromBytes> {
    sub: SubStream<'a, StreamReply<Slice<Vec<u8>>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: AsVTable + zerocopy::TryFromBytes + Immutable + KnownLayout + Clone> Subscription<'_, T> {
    pub async fn next(&mut self) -> Result<T, Error> {
        loop {
            if let StreamReply::Table(table) = self.sub.next().await? {
                let t = <T>::try_ref_from_bytes(deref(&table.buf))
                    .map_err(impeller2::error::Error::from)?;
                return Ok(t.clone());
            }
        }
    }
}

impl<T: AsVTable + zerocopy::TryFromBytes + Immutable + KnownLayout + Clone> Deref
    for Subscription<'_, T>
{
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        self.sub.deref()
    }
}

impl<T: AsVTable + zerocopy::TryFromBytes + Immutable + KnownLayout + Clone> DerefMut
    for Subscription<'_, T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.sub.deref_mut()
    }
}
